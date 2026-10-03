# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL grouped GEMM for dist_moe.

Persistent, warp-specialized SM100 kernel for FPROP/DGRAD/WGRAD. The
implementation preserves the scheduling and deterministic numerical contract
validated by internal runtime parity.

Several @cute.kernel / @cute.jit bodies trip C901 (cyclomatic
complexity). They are tightly coupled per-tile state machines
(per-group accumulators, per-tile fences, cluster handshakes);
decomposing them forces re-threading dozens of @cute.jit arguments
and breaks JIT signature binding. We mark the affected defs with a
per-function ``# noqa: C901`` because flake8 (the lint backend used
in this repo) does not honor file-level per-rule disables.
"""

import threading

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

from ._environment import num_sms_per_device
from ._kernel_name_prefix import scoped_kernel_name_prefixes
from .config import (  # noqa: F401
    _resolve_grouped_gemm_config,
    auto_grouped_gemm_config,
    auto_grouped_gemm_mega_config,
    DEFAULT_GROUPED_GEMM_WGRAD_CONFIG,
    derive_auto_grouped_gemm_mega_config,
    grouped_gemm_configs_for_dtype,
    registered_grouped_gemm_config_name,
    resolve_grouped_gemm_config,
)
from .grouped_gemm_kernel import (
    GroupedGemmKernel,
)
from .tile_scheduler import _DGRAD, _FPROP, _WGRAD

# ---------------------------------------------------------------------------
# Helper utilities (index / coordinate math).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Kernel class. Host-side __call__ builds SharedStorage + TMA atoms and
# launches; device-side `kernel` runs the warp-specialized loop. Layout:
#   - 4 warp groups (epilog / MMA / TMA / idle), reqntid=256
#   - cluster shape (2, 1, 1) for NUM_CTAS=2, else (1, 1, 1)
#   - dynamic atomic-counter scheduler by default; static scheduler optional
#   - mbarriers initialized in the prologue, followed by
#     fence_view_async_shared + cluster_arrive_relaxed / cluster_wait.
# ---------------------------------------------------------------------------


# ===========================================================================
# Host-side launchers for the grouped-GEMM kernels.
# They validate inputs, allocate the persistent workspace, and dispatch the
# cached CuTe kernel.
# ===========================================================================


def _validate_inputs(
    x: torch.Tensor, w: torch.Tensor, split_sizes: torch.Tensor
) -> torch.Tensor:
    if x.dtype != w.dtype:
        raise TypeError(f"x.dtype != w.dtype: {x.dtype} vs {w.dtype}")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported input dtype {x.dtype}; expected fp16/bf16/fp32")
    if not (x.is_cuda and w.is_cuda and split_sizes.is_cuda):
        raise ValueError("inputs must reside on CUDA device")
    # Cross-device launches feed raw pointers from one device into a kernel
    # launched on another, which is an illegal-access class bug. Require all
    # inputs to share x.device.
    if w.device != x.device or split_sizes.device != x.device:
        raise ValueError(
            f"all inputs must share the same CUDA device; got "
            f"x={x.device}, w={w.device}, split_sizes={split_sizes.device}"
        )
    if split_sizes.dtype != torch.int32:
        # Kernel-side split-size loads use int32 values.
        split_sizes = split_sizes.to(torch.int32)
    return split_sizes


_DUMMY_OFFSET_CACHE: dict = {}


def _dummy_offset_cute(device: torch.device):
    """Returns a cute.Tensor over a 1-element int64 zero tensor — used as
    placeholder when `use_activation_buffer=False` so the kernel signature
    is uniform. Cached per device to avoid per-launch allocations."""
    key = str(device)
    if key not in _DUMMY_OFFSET_CACHE:
        t = torch.zeros(1, dtype=torch.int64, device=device)
        _DUMMY_OFFSET_CACHE[key] = (t, from_dlpack(t, assumed_align=8))
    return _DUMMY_OFFSET_CACHE[key][1]


# Reusing the GroupedGemmKernel instance is necessary but NOT sufficient
# for performance — the `@cute.jit __call__` re-traces the host-side body
# on every call (build TMA atoms, SMEM layouts, partition tensors, ...),
# which costs ~500 ms per call even on small shapes. We use `cute.compile`
# to JIT once and cache the resulting launchable, then call it directly.
# Cache the compiled launchable so repeated calls avoid retracing the host-side
# setup path.
_KERNEL_CACHE: dict = {}
_COMPILED_CACHE: dict = {}
_COMPILE_LOCK = threading.Lock()


def _get_kernel(
    config_name: str,
    problem_type: int,
    force_n_major: bool,
    num_n_clusters: int,
    world_size: int,
    static_scheduler: bool,
    swap_ab: bool,
    *,
    dtype: torch.dtype,
) -> "GroupedGemmKernel":
    key = (
        config_name,
        problem_type,
        force_n_major,
        num_n_clusters,
        world_size,
        static_scheduler,
        dtype,
        swap_ab,
    )
    inst = _KERNEL_CACHE.get(key)
    if inst is None:
        inst = GroupedGemmKernel(
            config=grouped_gemm_configs_for_dtype(dtype)[config_name],
            problem_type=problem_type,
            force_n_major=force_n_major,
            num_n_clusters=num_n_clusters,
            world_size=world_size,
            static_scheduler=static_scheduler,
            swap_ab=swap_ab,
        )
        _KERNEL_CACHE[key] = inst
    return inst


def _torch_layout_signature(t: torch.Tensor):
    """Hashable summary of a torch tensor's compile-relevant layout:
    dtype + per-axis is-stride-one. Two tensors with the same dtype and
    majorness pattern share a compiled artifact regardless of size."""
    return (str(t.dtype), tuple(int(s == 1) for s in t.stride()))


def _compile_or_get(
    cache_key: tuple,
    kernel: "GroupedGemmKernel",
    *args,
    name_prefix: str | None = None,
    prefix_target=None,
    **kwargs,
):
    """Lookup or compile-and-cache a launchable for this (kernel, args) shape."""
    compiled = _COMPILED_CACHE.get(cache_key)
    if compiled is not None:
        return compiled
    with _COMPILE_LOCK:
        compiled = _COMPILED_CACHE.get(cache_key)
        if compiled is not None:
            return compiled
        if name_prefix is not None:
            assert prefix_target is not None
            prefixes = ((prefix_target, name_prefix),)
        else:
            prefixes = ()
        with scoped_kernel_name_prefixes(prefixes):
            compiled = cute.compile(kernel, *args, **kwargs)
        _COMPILED_CACHE[cache_key] = compiled
    return compiled


def _allocate_workspace(
    num_clusters: int,
    NUM_CTAS: int,
    device: torch.device,
    *,
    counter: torch.Tensor | None = None,
):
    """Workspace tensors, freshly allocated per call so each launch is
    independent and safe under cudagraph capture and concurrent streams:
    * counter: int32[1] zero-init persistent tile counter.
    * tensormaps: int64[grid_size, 3, 16] TMA descriptors, written by the
      kernel each launch (no zero-init required)."""
    grid_size = num_clusters * NUM_CTAS
    if counter is None:
        counter = torch.zeros(1, dtype=torch.int32, device=device)
    elif (
        counter.dtype != torch.int32
        or counter.device != device
        or not counter.is_contiguous()
        or counter.numel() < 1
    ):
        raise ValueError(
            "counter must be a contiguous int32 tensor on the workspace device"
        )
    else:
        counter = counter[:1]
    tensormaps = torch.empty((grid_size, 3, 16), dtype=torch.int64, device=device)
    return counter, tensormaps


def grouped_gemm_fprop(
    x: torch.Tensor,  # [GM, K]
    w: torch.Tensor,  # [G, N, K]
    split_sizes: torch.Tensor,  # [G]
    *,
    y: torch.Tensor | None = None,  # [GM, N]
    num_sms: int | None = None,
    config: str | None = None,
    force_n_major: bool = False,
    num_n_clusters: int = 1,
    local_rank: int = 0,
    world_size: int = 1,
    static_scheduler: bool = False,
    swap_ab: bool = False,
) -> torch.Tensor:
    """Compute `y[g] = x[g] @ w[g].T` for each group."""
    split_sizes = _validate_inputs(x, w, split_sizes)
    GM, K = x.shape
    G, N, K_w = w.shape
    if K != K_w:
        raise ValueError(f"K mismatch: x has {K}, w has {K_w}")
    if y is None:
        y = torch.empty((GM, N), dtype=x.dtype, device=x.device)
    elif y.shape != (GM, N) or y.dtype != x.dtype or y.device != x.device:
        raise ValueError(
            f"y has wrong shape/dtype/device: {y.shape}/{y.dtype}/{y.device}"
        )

    if num_sms is None:
        num_sms = num_sms_per_device()

    config = _resolve_grouped_gemm_config(
        dtype=x.dtype,
        config=config,
        GM=GM,
        G=G,
        problem_type=_FPROP,
        N=N,
        num_sms=num_sms,
    )
    cfg = grouped_gemm_configs_for_dtype(x.dtype)[config]
    NUM_CTAS = cfg["NUM_CTAS"]
    num_clusters = max(1, num_sms // NUM_CTAS)

    # Workspace only needs the persistent counter and TMA descriptors.
    counter, tensormaps = _allocate_workspace(num_clusters, NUM_CTAS, x.device)

    kernel = _get_kernel(
        config_name=config,
        dtype=x.dtype,
        problem_type=_FPROP,
        force_n_major=force_n_major,
        num_n_clusters=num_n_clusters,
        world_size=world_size,
        static_scheduler=static_scheduler,
        swap_ab=swap_ab,
    )

    # Placeholders capture dtype and layout for compile caching.
    a_placeholder = x.detach().unsqueeze(-1)
    b_placeholder = w.detach().permute(1, 2, 0)
    c_placeholder = y.detach().unsqueeze(-1)
    a_cute = from_dlpack(a_placeholder, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    b_cute = from_dlpack(b_placeholder, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    c_cute = from_dlpack(c_placeholder, assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    split_cute = from_dlpack(split_sizes.detach(), assumed_align=4)
    counter_cute = from_dlpack(counter, assumed_align=4)
    tensormaps_cute = from_dlpack(tensormaps, assumed_align=8)
    # Unused placeholder keeps the launch signature uniform.
    dummy_offset_cute = _dummy_offset_cute(x.device)

    # FPROP descriptor strides:
    #   A = x as (m, k) with strides (K, 1).
    #   B = w[g] as (n, k) with strides (K, 1).
    #   C = y as (m, n) with strides (N, 1).
    elem = x.element_size()
    stream = cutlass_torch.current_stream()
    # `cute.compile` bakes constexpr arguments into the compiled artifact.
    # The runtime launch passes only the non-constexpr subset from args_spec.
    compile_args = (
        a_cute,
        b_cute,
        c_cute,
        split_cute,
        counter_cute,
        tensormaps_cute,
        x.data_ptr(),
        w.data_ptr(),
        y.data_ptr(),
        x.stride(0),
        x.stride(1),
        w.stride(1),
        w.stride(2),
        y.stride(0),
        y.stride(1),
        elem,
        elem,
        y.element_size(),  # constexpr (compile only)
        False,  # constexpr use_activation_buffer
        cutlass.Int64(0),
        dummy_offset_cute,
        dummy_offset_cute,
        False,  # constexpr output_accum
        G,  # constexpr G
        GM,
        N,
        K,
        local_rank,
        num_clusters,
        stream,
    )
    runtime_args = (
        a_cute,
        b_cute,
        c_cute,
        split_cute,
        counter_cute,
        tensormaps_cute,
        x.data_ptr(),
        w.data_ptr(),
        y.data_ptr(),
        x.stride(0),
        x.stride(1),
        w.stride(1),
        w.stride(2),
        y.stride(0),
        y.stride(1),
        cutlass.Int64(0),
        dummy_offset_cute,
        dummy_offset_cute,
        GM,
        N,
        K,
        local_rank,
        num_clusters,
        stream,
    )
    cache_key = (
        config,
        _FPROP,
        force_n_major,
        num_n_clusters,
        world_size,
        static_scheduler,
        swap_ab,
        _torch_layout_signature(a_placeholder),
        _torch_layout_signature(b_placeholder),
        _torch_layout_signature(c_placeholder),
        G,
        elem,
        elem,
        y.element_size(),
        False,
        False,  # constexprs
    )
    compiled = _compile_or_get(
        cache_key,
        kernel,
        *compile_args,
        name_prefix="_cute_grouped_gemm_fprop",
        prefix_target=GroupedGemmKernel.kernel,
    )
    compiled(*runtime_args)
    return y


def grouped_gemm_dgrad(
    grad_y: torch.Tensor,  # (GM, N)
    w: torch.Tensor,  # (G, N, K_in)
    split_sizes: torch.Tensor,  # (G,)
    *,
    grad_x: torch.Tensor | None = None,  # (GM, K_in)
    num_sms: int | None = None,
    config: str | None = None,
    force_n_major: bool = False,
    num_n_clusters: int = 1,
    local_rank: int = 0,
    world_size: int = 1,
    static_scheduler: bool = False,
) -> torch.Tensor:
    """Compute `grad_x[g] = grad_y[g] @ w[g]` for each group."""
    split_sizes = _validate_inputs(grad_y, w, split_sizes)
    GM, N = grad_y.shape
    G, N_w, K_in = w.shape
    if N != N_w:
        raise ValueError(f"N mismatch: grad_y has {N}, w has {N_w}")
    if grad_x is None:
        grad_x = torch.empty((GM, K_in), dtype=grad_y.dtype, device=grad_y.device)
    elif (
        grad_x.shape != (GM, K_in)
        or grad_x.dtype != grad_y.dtype
        or grad_x.device != grad_y.device
    ):
        raise ValueError(
            f"grad_x has wrong shape/dtype/device: "
            f"{grad_x.shape}/{grad_x.dtype}/{grad_x.device}"
        )

    if num_sms is None:
        num_sms = num_sms_per_device()

    config = _resolve_grouped_gemm_config(
        dtype=grad_y.dtype,
        config=config,
        GM=GM,
        G=G,
        problem_type=_DGRAD,
        N=K_in,
        num_sms=num_sms,
    )
    cfg = grouped_gemm_configs_for_dtype(grad_y.dtype)[config]
    NUM_CTAS = cfg["NUM_CTAS"]
    num_clusters = max(1, num_sms // NUM_CTAS)

    counter, tensormaps = _allocate_workspace(num_clusters, NUM_CTAS, grad_y.device)

    kernel = _get_kernel(
        config_name=config,
        dtype=grad_y.dtype,
        problem_type=_DGRAD,
        force_n_major=force_n_major,
        num_n_clusters=num_n_clusters,
        world_size=world_size,
        static_scheduler=static_scheduler,
        swap_ab=False,
    )

    # Placeholders capture dtype and layout for compile caching.
    a_placeholder = grad_y.detach().unsqueeze(-1)
    b_placeholder = w.detach().permute(2, 1, 0)
    c_placeholder = grad_x.detach().unsqueeze(-1)
    a_cute = from_dlpack(a_placeholder, assumed_align=128).mark_layout_dynamic(
        leading_dim=1
    )
    b_cute = from_dlpack(b_placeholder, assumed_align=128).mark_layout_dynamic(
        leading_dim=0
    )
    c_cute = from_dlpack(c_placeholder, assumed_align=128).mark_layout_dynamic(
        leading_dim=1
    )
    split_cute = from_dlpack(split_sizes.detach(), assumed_align=4)
    counter_cute = from_dlpack(counter, assumed_align=4)
    tensormaps_cute = from_dlpack(tensormaps, assumed_align=8)

    # DGRAD descriptor strides:
    #   A = grad_y as (m, k=N) with strides (N, 1).
    #   B = w[g] as (n=K_in, k=N) with strides (1, K_in).
    #   C = grad_x as (m, n=K_in) with strides (K_in, 1).
    elem = grad_y.element_size()
    dummy_offset_cute = _dummy_offset_cute(grad_y.device)
    stream = cutlass_torch.current_stream()
    # B = w[g].t() — view of w[g] (N, K_in) as (K_in, N). For a contiguous
    # `w` this gives `(1, K_in)`; for non-contiguous `w` we derive from
    # the actual tensor strides (swapped by the transpose).
    b_s0 = w.stride(2)
    b_s1 = w.stride(1)
    compile_args = (
        a_cute,
        b_cute,
        c_cute,
        split_cute,
        counter_cute,
        tensormaps_cute,
        grad_y.data_ptr(),
        w.data_ptr(),
        grad_x.data_ptr(),
        grad_y.stride(0),
        grad_y.stride(1),
        b_s0,
        b_s1,
        grad_x.stride(0),
        grad_x.stride(1),
        elem,
        elem,
        grad_x.element_size(),
        False,
        cutlass.Int64(0),
        dummy_offset_cute,
        dummy_offset_cute,
        False,
        G,
        GM,
        K_in,
        N,
        local_rank,
        num_clusters,
        stream,
    )
    runtime_args = (
        a_cute,
        b_cute,
        c_cute,
        split_cute,
        counter_cute,
        tensormaps_cute,
        grad_y.data_ptr(),
        w.data_ptr(),
        grad_x.data_ptr(),
        grad_y.stride(0),
        grad_y.stride(1),
        b_s0,
        b_s1,
        grad_x.stride(0),
        grad_x.stride(1),
        cutlass.Int64(0),
        dummy_offset_cute,
        dummy_offset_cute,
        GM,
        K_in,
        N,
        local_rank,
        num_clusters,
        stream,
    )
    cache_key = (
        config,
        _DGRAD,
        force_n_major,
        num_n_clusters,
        world_size,
        static_scheduler,
        _torch_layout_signature(a_placeholder),
        _torch_layout_signature(b_placeholder),
        _torch_layout_signature(c_placeholder),
        G,
        elem,
        elem,
        grad_x.element_size(),
        False,
        False,
    )
    compiled = _compile_or_get(
        cache_key,
        kernel,
        *compile_args,
        name_prefix="_cute_grouped_gemm_dgrad",
        prefix_target=GroupedGemmKernel.kernel,
    )
    compiled(*runtime_args)
    return grad_x


def _validate_activation_buffer_args(
    activation_buffer: torch.Tensor,
    dy_offset: torch.Tensor,
    x_offset: torch.Tensor,
) -> None:
    """Sanity-check the activation-buffer wgrad fast path inputs.

    The kernel reads ``[0]`` from each offset tensor with no host
    fallback, so each rule below is load-bearing for safety."""
    if dy_offset.dtype != torch.int64 or x_offset.dtype != torch.int64:
        raise TypeError(
            "`dy` and `x` must be int64 offset tensors in activation-buffer mode"
        )
    if not activation_buffer.is_cuda:
        raise ValueError("activation_buffer must reside on CUDA device")
    expected_device = activation_buffer.device
    if dy_offset.device != expected_device or x_offset.device != expected_device:
        raise ValueError(
            f"activation_buffer + offsets must share device={expected_device}; got "
            f"buffer={activation_buffer.device}, dy={dy_offset.device}, x={x_offset.device}"
        )
    if dy_offset.numel() == 0 or x_offset.numel() == 0:
        raise ValueError(
            "`dy` and `x` offset tensors must be non-empty (kernel reads index 0)"
        )


def grouped_gemm_wgrad(
    dy: torch.Tensor,  # (GM, N) data tensor, OR int64[1] offset tensor when activation_buffer is set
    x: torch.Tensor,  # (GM, K_in) data tensor, OR int64[1] offset tensor when activation_buffer is set
    split_sizes: torch.Tensor,  # (G,)
    *,
    wgrad: torch.Tensor | None = None,  # (G, N, K_in)
    output_accum: bool = False,
    num_sms: int | None = None,
    config: str = DEFAULT_GROUPED_GEMM_WGRAD_CONFIG,
    activation_buffer: torch.Tensor | None = None,
    hidden_dim_dy: int | None = None,
    hidden_dim_x: int | None = None,
    dtype: torch.dtype | None = None,
    force_n_major: bool = False,
    num_n_clusters: int = 1,
    local_rank: int = 0,
    world_size: int = 1,
    static_scheduler: bool = False,
) -> torch.Tensor:
    """Compute ``wgrad[g] = dy[g].T @ x[g]`` for each group.

    Two input modes are supported:
      * Direct (``activation_buffer is None``): ``dy`` and ``x`` are real
        ``(GM, N)`` / ``(GM, K_in)`` tensors.
      * Activation-buffer fast path: ``dy`` and ``x`` are int64[1] device
        tensors holding byte offsets into ``activation_buffer``;
        ``hidden_dim_dy`` / ``hidden_dim_x`` / ``dtype`` describe the
        underlying bf16/fp16 view. The kernel reads bytes from
        ``activation_buffer + offset[0]``.
    """
    use_ab = activation_buffer is not None
    if use_ab:
        if hidden_dim_dy is None or hidden_dim_x is None or dtype is None:
            raise ValueError(
                "activation_buffer requires hidden_dim_dy, hidden_dim_x, and dtype"
            )
        _validate_activation_buffer_args(activation_buffer, dy, x)
        # `dy` / `x` are offset tensors; build dummy real tensors of the
        # described dtype + hidden dim so the rest of the launcher can
        # derive TMA layouts and strides uniformly. The kernel only reads
        # bytes from `activation_buffer + offset[0]`; the dummy contents
        # are never accessed. The M dimension of the placeholder is
        # `mark_layout_dynamic` below and patched per-group at runtime via
        # `update_tma_descriptor`, so the placeholder's M extent is unused
        # by the TMA atom and is set to 1 to avoid allocating GB-sized
        # dummies on every wgrad call. Allocating large per-call dummies
        # of differing shapes (w13 vs w2 have different `hidden_dim_x`)
        # fragments the caching allocator and OOMs FSDP reduce-scatter
        # tens of GiB later.
        dy_offset_t = dy.detach()
        x_offset_t = x.detach()
        dy = torch.empty(
            (1, hidden_dim_dy), dtype=dtype, device=activation_buffer.device
        )
        x = torch.empty((1, hidden_dim_x), dtype=dtype, device=activation_buffer.device)
        a_base = activation_buffer.data_ptr()
        b_base = activation_buffer.data_ptr()
    else:
        if hidden_dim_dy is not None or hidden_dim_x is not None or dtype is not None:
            raise ValueError(
                "hidden_dim_dy / hidden_dim_x / dtype are only valid with activation_buffer"
            )
        a_base = dy.data_ptr()
        b_base = x.data_ptr()

    split_sizes = _validate_inputs(x, dy, split_sizes)
    GM, K_in = x.shape
    GM_y, N = dy.shape
    if GM != GM_y:
        raise ValueError(f"M mismatch: x has {GM}, dy has {GM_y}")
    G = split_sizes.shape[0]
    if wgrad is None:
        # Overwrite mode does not need zero-init; accumulation mode expects the
        # caller to pass an initialized output tensor.
        wgrad = torch.empty((G, N, K_in), dtype=x.dtype, device=x.device)
    elif wgrad.shape != (G, N, K_in) or wgrad.device != x.device:
        raise ValueError(
            f"wgrad has wrong shape/device: {wgrad.shape}/{wgrad.device} "
            f"(expected {(G, N, K_in)} on {x.device})"
        )

    if num_sms is None:
        num_sms = num_sms_per_device()

    cfg = grouped_gemm_configs_for_dtype(dy.dtype)[config]
    NUM_CTAS = cfg["NUM_CTAS"]
    num_clusters = max(1, num_sms // NUM_CTAS)

    counter, tensormaps = _allocate_workspace(num_clusters, NUM_CTAS, x.device)

    # WGRAD descriptor strides (per-group, used by update_tma_descriptor):
    #   A = dy.T     with strides (1, dy.stride(0)).
    #   B = x.T      with strides (1, x.stride(0)).
    #   C = wgrad[g] with strides (K_in, 1).
    # The M-axis (contraction) stride must come from dy/x.stride(0) — not
    # the hardcoded N / K_in — to handle non-contiguous inputs (e.g. tensors
    # sliced from a larger allocation, where stride(0) > inner extent).

    kernel = _get_kernel(
        config_name=config,
        dtype=dy.dtype,
        problem_type=_WGRAD,
        force_n_major=force_n_major,
        num_n_clusters=num_n_clusters,
        world_size=world_size,
        static_scheduler=static_scheduler,
        swap_ab=False,
    )

    # Placeholders capture dtype and layout for compile caching. CuTeDSL
    # rejects 0-sized shapes in tensors used to build TMA atoms, so when
    # GM == 0 we substitute a 1-row dummy along the contraction axis. The
    # kernel still receives GM as a runtime arg and skips the k-iter loop
    # (num_k_iters=0); the epilog writes zeros via its `if num_k_tiles == 0`
    # branch, which defines the empty-input result.
    if GM == 0:
        a_dummy = torch.empty((N, 1), dtype=dy.dtype, device=dy.device)
        b_dummy = torch.empty((K_in, 1), dtype=x.dtype, device=x.device)
        a_placeholder = a_dummy.unsqueeze(-1)
        b_placeholder = b_dummy.unsqueeze(-1)
    else:
        a_placeholder = dy.t().detach().unsqueeze(-1)
        b_placeholder = x.t().detach().unsqueeze(-1)
    c_placeholder = wgrad[0].detach().unsqueeze(-1)
    # Wgrad's contraction axis (dim 1) varies per group via update_tensormap.
    # Mirror the cute/gemm reference: mark A/B with
    # leading_dim=0 (innermost is the M-dim of dy.T / x.T, stride-1) and C
    # with leading_dim=1 (innermost of wgrad[g] is K_in, stride-1). This
    # makes the OTHER strides + extents dynamic so update_tma_descriptor's
    # per-group patches actually take effect at TMA-atom granularity.
    a_cute = from_dlpack(a_placeholder, assumed_align=128).mark_layout_dynamic(
        leading_dim=0
    )
    b_cute = from_dlpack(b_placeholder, assumed_align=128).mark_layout_dynamic(
        leading_dim=0
    )
    c_cute = from_dlpack(c_placeholder, assumed_align=128).mark_layout_dynamic(
        leading_dim=1
    )
    split_cute = from_dlpack(split_sizes.detach(), assumed_align=4)
    counter_cute = from_dlpack(counter, assumed_align=4)
    tensormaps_cute = from_dlpack(tensormaps, assumed_align=8)

    elem = x.element_size()
    if use_ab:
        a_offset_cute = from_dlpack(dy_offset_t, assumed_align=8)
        b_offset_cute = from_dlpack(x_offset_t, assumed_align=8)
        activation_buffer_size_bytes = activation_buffer.nbytes
    else:
        a_offset_cute = _dummy_offset_cute(x.device)
        b_offset_cute = _dummy_offset_cute(x.device)
        activation_buffer_size_bytes = 0
    # Strides:
    #   A = dy.t()    — view of dy (GM, N) as (N, GM). After
    #     transpose, .stride(0) = dy.stride(1) and
    #     .stride(1) = dy.stride(0).
    #   B = x.t()     — same pattern on x (GM, K_in).
    #   C = wgrad[g]  — slice of wgrad (G, N, K_in) as (N, K_in) with
    #     strides (wgrad.stride(1), wgrad.stride(2)).
    # GM==0 dummy path: contiguous (N,1)/(K_in,1) inputs → stride(1)=1.
    # The kernel skips the k-iter loop in that case so the values of
    # `a_s1` / `b_s1` (= original M-axis stride) don't affect numerics.
    if GM == 0:
        a_s0 = cutlass.Int32(1)
        a_s1 = cutlass.Int32(1)
        b_s0 = cutlass.Int32(1)
        b_s1 = cutlass.Int32(1)
    else:
        a_s0 = dy.stride(1)
        a_s1 = dy.stride(0)
        b_s0 = x.stride(1)
        b_s1 = x.stride(0)
    c_s0 = wgrad.stride(1)
    c_s1 = wgrad.stride(2)
    stream = cutlass_torch.current_stream()
    compile_args = (
        a_cute,
        b_cute,
        c_cute,
        split_cute,
        counter_cute,
        tensormaps_cute,
        a_base,
        b_base,
        wgrad.data_ptr(),
        a_s0,
        a_s1,
        b_s0,
        b_s1,
        c_s0,
        c_s1,
        elem,
        elem,
        wgrad.element_size(),
        use_ab,
        cutlass.Int64(activation_buffer_size_bytes),
        a_offset_cute,
        b_offset_cute,
        output_accum,
        G,
        N,
        K_in,
        GM,  # MMA's m,n,k
        local_rank,
        num_clusters,
        stream,
    )
    runtime_args = (
        a_cute,
        b_cute,
        c_cute,
        split_cute,
        counter_cute,
        tensormaps_cute,
        a_base,
        b_base,
        wgrad.data_ptr(),
        a_s0,
        a_s1,
        b_s0,
        b_s1,
        c_s0,
        c_s1,
        cutlass.Int64(activation_buffer_size_bytes),
        a_offset_cute,
        b_offset_cute,
        N,
        K_in,
        GM,
        local_rank,
        num_clusters,
        stream,
    )
    cache_key = (
        config,
        _WGRAD,
        force_n_major,
        num_n_clusters,
        world_size,
        static_scheduler,
        _torch_layout_signature(a_placeholder),
        _torch_layout_signature(b_placeholder),
        _torch_layout_signature(c_placeholder),
        G,
        elem,
        elem,
        wgrad.element_size(),
        use_ab,
        output_accum,
    )
    compiled = _compile_or_get(
        cache_key,
        kernel,
        *compile_args,
        name_prefix="_cute_grouped_gemm_wgrad",
        prefix_target=GroupedGemmKernel.kernel,
    )
    compiled(*runtime_args)
    return wgrad
