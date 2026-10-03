# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL fused RMSNorm + weighted TOPK reduction.

Public API:
    fused_rmsnorm_combine(x, weights, eps) -> Tensor
        x:       [B, TOPK, DIM] bf16 contiguous
        weights: [B, TOPK]      f32  contiguous
        returns: [B, DIM]       bf16 (with autograd support)

Advantages over an unfused Triton formulation:
  - Optional Kahan-compensated sum-of-squares for near-FP64 accuracy in FP32.
  - Efficient vectorized global→shared→register transfers via CuTe cp.async.
"""

import math

import cutlass.cute as cute
import torch
from cutlass import Float32
from torch import Tensor

from .. import _dsl_compat as _cute_extern  # noqa: F401
from .._cuda_context import ensure_cuda_driver_context
from .._jit_cache import jit_cache
from .._quack.cute_dsl_utils import torch2cute_dtype_map
from .._quack_compile_utils import make_fake_stream, make_fake_tensor as fake_tensor
from .._tensor_views import _same_view, _views_overlap
from ..triton.broadcast_n_reduction import (
    broadcast_and_scale,
    scale_and_sum,
    SCALE_AND_SUM_TILE_D_MAX_TOKENS,
)
from ._fused_rmsnorm_combine_kernel import (
    _exact_bwd_vecsize,
    FusedRMSNormCombineBwd,
    FusedRMSNormCombineBwdExact,
    FusedRMSNormCombineFwd,
)
from ._rmsnorm_common import (
    _aligned,
    _fake_div,
    _normalize_for_alignment,
)
from .rmsnorm import cute_rmsnorm_bwd, cute_rmsnorm_fwd

# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------


@jit_cache
def _compile_fwd(
    dtype,
    norm_dtype,
    output_dtype,
    weight_dtype,
    N,
    TOPK,
    use_kahan,
    require_bitwise,
    tile_d,
    copy_input,
    has_weight,
):
    batch_sym = cute.sym_int()
    batch_topk_sym = cute.sym_int()
    div = math.gcd(N, *(128 // dt.width for dt in [dtype, norm_dtype, output_dtype]))
    x_cute = fake_tensor(dtype, (batch_topk_sym, N), div)
    w_cute = fake_tensor(weight_dtype, (batch_topk_sym,))
    y_cute = fake_tensor(output_dtype, (batch_sym, N), div)
    rstd_cute = fake_tensor(Float32, (batch_topk_sym,))
    x_copy_cute = fake_tensor(dtype, (batch_topk_sym, N), div)
    g_cute = fake_tensor(dtype, (1, N), div) if has_weight else None
    kernel = FusedRMSNormCombineFwd(
        dtype,
        norm_dtype,
        N,
        TOPK,
        use_kahan=use_kahan,
        require_bitwise=require_bitwise,
        tile_d=tile_d,
        copy_input=copy_input,
        has_weight=has_weight,
    )
    if not kernel.can_implement(
        x_cute,
        w_cute,
        y_cute,
        rstd_cute,
        x_copy_cute,
        g_cute,
    ):
        raise ValueError(
            "FusedRMSNormCombineFwd requires 16-byte aligned "
            f"row shapes; got N={N}, input_bits={dtype.width}, "
            f"output_bits={output_dtype.width}"
        )
    return cute.compile(
        kernel,
        x_cute,
        w_cute,
        y_cute,
        rstd_cute,
        x_copy_cute,
        g_cute,
        Float32(0),  # eps
        Float32(0),  # gain_center
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _bwd_divisibility(
    dtype,
    grad_y_dtype,
    N,
    use_exact_kernel,
) -> int:
    """Return the element divisibility required by backward vector copies.

    The exact kernel pins its reduction grouping; the optimized kernel shares
    `_fake_div` with its emitted vector width.
    """
    return (
        _exact_bwd_vecsize(N)
        if use_exact_kernel
        # `grad_x` is required to match `x`, so repeat `dtype` to mirror the
        # kernel's `(x, grad_y, grad_x)` vector-width inputs exactly.
        else _fake_div(N, (dtype, grad_y_dtype, dtype))
    )


def _prepare_bwd_grad_y(
    grad_y: Tensor,
    norm_output_dtype: torch.dtype,
    *,
    require_bitwise: bool,
) -> Tensor:
    """Prepare the backward gradient, optionally preserving the low-HBM BF16 path."""
    if grad_y.dtype == torch.bfloat16 and not require_bitwise:
        return grad_y
    return grad_y.to(
        dtype=norm_output_dtype,
        memory_format=torch.contiguous_format,
    ).contiguous()


@jit_cache
def _compile_bwd(
    dtype,
    grad_y_dtype,
    N,
    TOPK,
    use_exact_kernel,
):
    batch_sym = cute.sym_int()
    batch_topk_sym = cute.sym_int()
    div = _bwd_divisibility(
        dtype,
        grad_y_dtype,
        N,
        use_exact_kernel,
    )
    grad_y_cute = fake_tensor(grad_y_dtype, (batch_sym, N), div)
    x_cute = fake_tensor(dtype, (batch_topk_sym, N), div)
    w_cute = fake_tensor(Float32, (batch_topk_sym,))
    rstd_cute = fake_tensor(Float32, (batch_topk_sym,))
    grad_x_cute = fake_tensor(dtype, (batch_topk_sym, N), div)
    grad_w_cute = fake_tensor(Float32, (batch_topk_sym,))
    kernel_cls = (
        FusedRMSNormCombineBwdExact if use_exact_kernel else FusedRMSNormCombineBwd
    )
    kernel = kernel_cls(dtype, grad_y_dtype, N, TOPK)
    if not kernel.can_implement(
        grad_y_cute,
        x_cute,
        w_cute,
        rstd_cute,
        grad_x_cute,
        grad_w_cute,
    ):
        raise ValueError(
            "FusedRMSNormCombineBwd cannot implement the requested dtype widths "
            "or 16-byte row alignment; "
            f"got N={N}, x_bits={dtype.width}, "
            f"grad_y_bits={grad_y_dtype.width}, grad_x_bits={dtype.width}"
        )
    return cute.compile(
        kernel,
        grad_y_cute,
        x_cute,
        w_cute,
        rstd_cute,
        grad_x_cute,
        grad_w_cute,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

# The optimized backward stages one grad_y row in shared memory. Cap it at 12288
# so that row uses at most 48 KiB for FP32 input; the kernel accounts separately
# for its other shared allocations. The fused bitwise kernel preserves both
# standalone reduction topologies through 8192; wider bitwise dimensions use the
# standalone backward below.
_MAX_OPTIMIZED_FUSED_BWD_DIM = 12288
_MAX_BITWISE_FUSED_BWD_DIM = 8192
_MAX_FWD_DIM = 16384


def _is_supported_fwd_dim(dim: int) -> bool:
    return 256 <= dim <= _MAX_FWD_DIM and dim % 128 == 0


def supports_accelerated_fwd_contract(
    *,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    norm_output_dtype: torch.dtype,
    output_dtype: torch.dtype,
    topk: int,
    dim: int,
) -> bool:
    return (
        input_dtype == torch.bfloat16
        and weight_dtype == torch.float32
        and norm_output_dtype == torch.float32
        and output_dtype in (torch.bfloat16, torch.float32)
        and topk in (2, 8)
        and _is_supported_fwd_dim(dim)
    )


def _supports_accelerated_fwd(
    x: Tensor,
    weights: Tensor,
    norm_output_dtype: torch.dtype,
    output_dtype: torch.dtype,
    use_kahan: bool,
) -> bool:
    _, topk, dim = x.shape
    return (
        x.is_cuda
        and x.is_contiguous()
        and weights.is_contiguous()
        and supports_accelerated_fwd_contract(
            input_dtype=x.dtype,
            weight_dtype=weights.dtype,
            norm_output_dtype=norm_output_dtype,
            output_dtype=output_dtype,
            topk=topk,
            dim=dim,
        )
        and not use_kahan
    )


def _fused_rmsnorm_combine_bwd_kind(
    *, capability: int, dim: int, require_bitwise: bool
) -> str:
    max_fused_dim = (
        _MAX_BITWISE_FUSED_BWD_DIM
        if require_bitwise or capability < 10
        else _MAX_OPTIMIZED_FUSED_BWD_DIM
    )
    if (require_bitwise and capability == 9 and dim == 256) or dim > max_fused_dim:
        return "standalone"
    return "fused_exact" if require_bitwise or capability < 10 else "fused_optimized"


def _rmsnorm_weighted_topk_reduction_unfused_fwd(
    x: Tensor,
    weights: Tensor,
    eps: float,
    *,
    norm_output_dtype: torch.dtype,
    output_dtype: torch.dtype,
    weight: Tensor | None = None,
    gain_center: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    B, TOPK, DIM = x.shape
    x_2d = x.reshape(B * TOPK, DIM)
    x_norm, rstd = cute_rmsnorm_fwd(
        x=x_2d,
        w=weight,
        eps=eps,
        output_dtype=norm_output_dtype,
        gain_center=gain_center,
        input_scale=weight is not None,
    )
    x_norm_3d = x_norm.reshape(B, TOPK, DIM)
    y, _ = scale_and_sum(
        x=x_norm_3d,
        scale=weights,
        return_copy=False,
        output_dtype=output_dtype,
    )
    return y, rstd.reshape(B, TOPK), x_norm_3d


def _fused_rmsnorm_combine_fwd(
    x: Tensor,
    weights: Tensor,
    eps: float = 1e-5,
    use_kahan: bool = False,
    *,
    norm_output_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    require_bitwise: bool = True,
    x_copy: Tensor | None = None,
    weight: Tensor | None = None,
    gain_center: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Launch the fused forward kernel.

    Args:
        x: [B, TOPK, DIM] bf16 contiguous tensor.
        weights: [B, TOPK] f32 contiguous.
        eps: Epsilon for floating-point stability.
        use_kahan: Use Kahan-compensated sum-of-squares.
        require_bitwise: Preserve the standalone operator's exact reduction
            order.
        x_copy: Optional preallocated tensor receiving an exact copy of `x`.
        weight: Optional contiguous [DIM] input-scale gamma matching x's dtype.
            The kernel normalizes ``(weight + gain_center) * x`` in the single
            fused launch — no separate pre-scaling kernel — with the same
            per-element rounding as ``cute_rmsnorm_fwd(input_scale=True)``.
            Forward-only: no backward kernel consumes it.
        gain_center: Additive center folded into ``weight`` in-kernel.

    Returns:
        (y, rstd) where y is [B, DIM] bf16 and rstd is [B, TOPK] f32.
    """
    if norm_output_dtype is None:
        norm_output_dtype = x.dtype
    if output_dtype is None:
        output_dtype = x.dtype
    if x_copy is not None and (
        x_copy.shape != x.shape
        or x_copy.dtype != x.dtype
        or x_copy.device != x.device
        or not x_copy.is_contiguous()
    ):
        raise ValueError(
            "x_copy must be a contiguous tensor matching x's shape, dtype, and device"
        )
    if weight is not None:
        _validate_input_scale_weight(x, weight)
        if use_kahan:
            raise NotImplementedError(
                "input-scale weight is not supported with Kahan summation"
            )
        if x_copy is not None:
            raise NotImplementedError(
                "input-scale weight is forward-only; the training input copy "
                "is not supported with it"
            )
    if not use_kahan and not _supports_accelerated_fwd(
        x=x,
        weights=weights,
        norm_output_dtype=norm_output_dtype,
        output_dtype=output_dtype,
        use_kahan=use_kahan,
    ):
        y, rstd, _ = _rmsnorm_weighted_topk_reduction_unfused_fwd(
            x=x,
            weights=weights,
            eps=eps,
            norm_output_dtype=norm_output_dtype,
            output_dtype=output_dtype,
            weight=weight,
            gain_center=gain_center,
        )
        if x_copy is not None:
            x_copy.copy_(x)
        return y, rstd

    B, TOPK, DIM = x.shape
    x_2d = x.reshape(B * TOPK, DIM)
    x_copy_2d = x_2d if x_copy is None else x_copy.reshape(B * TOPK, DIM)
    w_flat = weights.reshape(B * TOPK)
    g_2d = None if weight is None else weight.reshape(1, DIM)

    y = torch.empty(B, DIM, dtype=output_dtype, device=x.device)
    rstd = torch.empty(B * TOPK, dtype=torch.float32, device=x.device)

    dtype = torch2cute_dtype_map[x.dtype]
    norm_dtype = torch2cute_dtype_map[norm_output_dtype]
    out_dtype = torch2cute_dtype_map[output_dtype]
    weight_dtype = torch2cute_dtype_map[weights.dtype]
    with ensure_cuda_driver_context():
        _compile_fwd(
            dtype,
            norm_dtype,
            out_dtype,
            weight_dtype,
            DIM,
            TOPK,
            use_kahan,
            require_bitwise,
            B <= SCALE_AND_SUM_TILE_D_MAX_TOKENS,
            x_copy is not None,
            weight is not None,
        )(x_2d, w_flat, y, rstd, x_copy_2d, g_2d, eps, gain_center)

    return y, rstd.reshape(B, TOPK)


def _fused_rmsnorm_combine_bwd(
    grad_y: Tensor,
    x: Tensor,
    weights: Tensor,
    rstd: Tensor | None,
    *,
    eps: float = 1e-5,
    require_bitwise: bool = True,
    use_kahan: bool = False,
    grad_x: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Launch the fused backward kernel.

    The fused paths normalize read-only inputs to the alignment required by their
    vectorized copies. A caller-provided `grad_x` must be contiguous. If its base
    address is insufficiently aligned, the kernel writes an aligned temporary and
    copies the result into the requested buffer.

    Args:
        grad_y: [B, DIM] bf16/fp32 tensor; arbitrary strides are supported.
        x: [B, TOPK, DIM] bf16 tensor with unit inner stride. Compatible row-strided
            views bind directly; other layouts incur a full-size aligned clone.
        weights: [B, TOPK] f32 contiguous.
        rstd: Optional [B, TOPK] f32 contiguous tensor from forward. When it is
            absent, the fused path recomputes it and the fallback path obtains
            it while recomputing the normalized input.
        use_kahan: Must match the forward's. Every path that recomputes the
            sum-of-squares needs it, or the gradients belong to a different
            normalization than the one the forward applied.
        grad_x: Optional preallocated [B, TOPK, DIM] bf16 output.

    Returns:
        (grad_x, grad_weights) where grad_x is [B, TOPK, DIM] bf16
        and grad_weights is [B, TOPK] f32.
    """
    B, TOPK, DIM = x.shape
    if grad_x is not None and (
        grad_x.shape != x.shape
        or grad_x.dtype != x.dtype
        or grad_x.device != x.device
        or not grad_x.is_contiguous()
    ):
        raise ValueError(
            "grad_x must be a contiguous tensor matching x's shape, dtype, and device"
        )

    capability = torch.cuda.get_device_capability(x.device)[0]
    bwd_kind = _fused_rmsnorm_combine_bwd_kind(
        capability=capability,
        dim=DIM,
        require_bitwise=require_bitwise,
    )
    if bwd_kind == "standalone":
        # Without an explicit output dtype, `broadcast_and_scale` derives
        # `grad_x_norm`'s dtype from `dy`. Preserve the historical FP32 input so
        # standalone RMSNorm backward continues to receive FP32 `grad_x_norm`.
        fallback_grad_y = grad_y.float().contiguous()
        x_norm, unfused_rstd = cute_rmsnorm_fwd(
            x.reshape(B * TOPK, DIM),
            None,
            eps=eps,
            output_dtype=torch.float32,
            use_kahan_summation=use_kahan,
        )
        grad_x_norm, grad_weights = broadcast_and_scale(
            dy=fallback_grad_y,
            scale=weights,
            x=x_norm.reshape(B, TOPK, DIM),
        )
        grad_x_out, _ = cute_rmsnorm_bwd(
            dy=grad_x_norm.reshape(B * TOPK, DIM),
            x=x.reshape(B * TOPK, DIM),
            w=None,
            rstd=unfused_rstd,
            dx=None if grad_x is None else grad_x.reshape(B * TOPK, DIM),
        )
        return grad_x_out.reshape(B, TOPK, DIM), grad_weights

    if rstd is None:
        _, rstd = _fused_rmsnorm_combine_fwd(
            x,
            weights,
            eps=eps,
            norm_output_dtype=torch.float32,
            output_dtype=x.dtype,
            require_bitwise=require_bitwise,
            use_kahan=use_kahan,
        )

    dtype = torch2cute_dtype_map[x.dtype]
    grad_y_dtype = torch2cute_dtype_map[grad_y.dtype]
    use_exact_kernel = bwd_kind == "fused_exact"
    div = _bwd_divisibility(
        dtype,
        grad_y_dtype,
        DIM,
        use_exact_kernel,
    )

    grad_y = _normalize_for_alignment(grad_y, div)
    x_2d = _normalize_for_alignment(x.reshape(B * TOPK, DIM), div)
    requested_grad_x = None if grad_x is None else grad_x.reshape(B * TOPK, DIM)
    grad_x_out = (
        requested_grad_x
        if requested_grad_x is not None and _aligned(requested_grad_x, div)
        else torch.empty(B * TOPK, DIM, dtype=x.dtype, device=x.device)
    )
    if (
        grad_x_out is requested_grad_x
        and _views_overlap(x_2d, grad_x_out)
        and not _same_view(x_2d, grad_x_out)
    ):
        grad_x_out = torch.empty(B * TOPK, DIM, dtype=x.dtype, device=x.device)
    if grad_x_out is requested_grad_x and _views_overlap(grad_y, grad_x_out):
        grad_y = grad_y.clone(memory_format=torch.contiguous_format)
    w_flat = weights.reshape(B * TOPK)
    rstd_flat = rstd.reshape(B * TOPK)

    grad_w = torch.empty(B * TOPK, dtype=torch.float32, device=x.device)

    with ensure_cuda_driver_context():
        # The optimized backward is GB10x-only. On older GPUs, preserve the
        # fused kernel and reduction topology used before this optimization.
        _compile_bwd(
            dtype,
            grad_y_dtype,
            DIM,
            TOPK,
            use_exact_kernel,
        )(
            grad_y,
            x_2d,
            w_flat,
            rstd_flat,
            grad_x_out,
            grad_w,
        )

    if requested_grad_x is not None and grad_x_out is not requested_grad_x:
        requested_grad_x.copy_(grad_x_out)
        grad_x_out = requested_grad_x
    return grad_x_out.reshape(B, TOPK, DIM), grad_w.reshape(B, TOPK)


# ---------------------------------------------------------------------------
# Input-scale (gamma) validation
# ---------------------------------------------------------------------------


def _validate_input_scale_weight(x: Tensor, weight: Tensor) -> None:
    if (
        weight.shape != (x.shape[-1],)
        or weight.dtype != x.dtype
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "fused_rmsnorm_combine weight must be a contiguous [DIM] tensor "
            f"matching x's dtype and device; got weight shape "
            f"{tuple(weight.shape)} dtype {weight.dtype} for x shape "
            f"{tuple(x.shape)} dtype {x.dtype}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class _FusedRMSNormCombine(torch.autograd.Function):
    """Fused RMSNorm followed by weighted top-k reduction.

    Scalarless by default; an optional ``weight`` selects the input-scaled
    form ``rmsnorm((weight + gain_center) * x)`` in the same single kernel.
    The weighted form is inference/forward-only: no backward supports it.
    Supported forward calls use the accelerated CuTe kernel. Unsupported
    calls raise instead of silently falling back.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weights: Tensor,
        eps: float = 1e-5,
        norm_output_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
        require_bitwise: bool = True,
        use_kahan: bool = False,
        weight: Tensor | None = None,
        gain_center: float = 0.0,
    ) -> Tensor:
        if norm_output_dtype is None:
            norm_output_dtype = x.dtype
        if output_dtype is None:
            output_dtype = x.dtype

        if not _supports_accelerated_fwd(
            x=x,
            weights=weights,
            norm_output_dtype=norm_output_dtype,
            output_dtype=output_dtype,
            use_kahan=False,
        ):
            raise ValueError(
                "fused_rmsnorm_combine only supports "
                "128-aligned MoE dimensions with bf16 inputs, fp32 weights, "
                "topk=2/8, fp32 RMSNorm output, and bf16/fp32 reduction output"
            )
        y, rstd = _fused_rmsnorm_combine_fwd(
            x=x,
            weights=weights,
            eps=eps,
            norm_output_dtype=norm_output_dtype,
            output_dtype=output_dtype,
            require_bitwise=require_bitwise,
            use_kahan=use_kahan,
            weight=weight,
            gain_center=gain_center,
        )

        ctx.save_for_backward(x, weights, rstd)
        ctx.norm_output_dtype = norm_output_dtype
        ctx.eps = eps
        ctx.require_bitwise = require_bitwise
        ctx.use_kahan = use_kahan
        ctx.had_weight = weight is not None
        return y

    @staticmethod
    def backward(
        ctx,
        grad_y: Tensor,
    ) -> tuple[Tensor | None, ...]:
        # The public wrapper rejects grad-requiring inputs with a weight; this
        # backstops direct .apply() misuse — the scalarless backward below
        # would silently produce gradients for the wrong forward.
        assert not ctx.had_weight, (
            "fused_rmsnorm_combine input-scale weight is inference/forward-only"
        )
        x, weights, rstd = ctx.saved_tensors
        # Preserve the public autograd path's historical FP32 reduction topology;
        # only DistMoE opts into the low-HBM BF16 preparation helper.
        grad_y = grad_y.to(
            dtype=ctx.norm_output_dtype,
            memory_format=torch.contiguous_format,
        ).contiguous()
        grad_x, grad_weights = _fused_rmsnorm_combine_bwd(
            grad_y,
            x,
            weights,
            rstd,
            eps=ctx.eps,
            require_bitwise=ctx.require_bitwise,
            use_kahan=ctx.use_kahan,
        )
        return (grad_x, grad_weights) + (None,) * 7


def fused_rmsnorm_combine(
    x: Tensor,
    weights: Tensor,
    eps: float = 1e-5,
    *,
    norm_output_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    require_bitwise: bool = True,
    use_kahan: bool = False,
    weight: Tensor | None = None,
    gain_center: float = 0.0,
) -> Tensor:
    """Fused RMSNorm + weighted top-k reduction for MoE.

    Args:
        x: [B, TOPK, DIM] contiguous tensor.
        weights: [B, TOPK] contiguous routing weights.
        eps: RMSNorm epsilon.
        norm_output_dtype: dtype materialized by the standalone RMSNorm kernel.
        output_dtype: dtype materialized by the standalone top-k reduction.
        require_bitwise: Preserve bitwise equivalence with the standalone operators.
            Set to ``False`` to enable the HBM-bound fused backward and permit
            DIM clustering when batch parallelism is limited.
        use_kahan: Use Kahan-compensated FP32 sum-of-squares instead of naive
            FP32.
        weight: Optional contiguous [DIM] gamma matching ``x``'s dtype. Selects
            the input-scaled norm ``rmsnorm((weight + gain_center) * x)`` — the
            ``rmsnorm_with_input_scale`` form — applied inside the same single
            fused kernel. Inference/forward-only: the call raises when any
            input requires grad.
        gain_center: Additive center for ``weight`` (e.g. 1.0 for
            zero-centered gamma), folded in-kernel. Only meaningful with
            ``weight``.

    Returns:
        [B, DIM] output matching ``cute_rmsnorm_fwd`` (with
        ``input_scale=True`` when ``weight`` is given) followed by
        ``scale_and_sum`` bit-for-bit when ``require_bitwise=True``.

    Raises:
        ValueError: If the input is outside the fused contract.
    """
    if (
        not require_bitwise
        and x.shape[-1] > _MAX_OPTIMIZED_FUSED_BWD_DIM
        and torch.is_grad_enabled()
        and (x.requires_grad or weights.requires_grad)
    ):
        raise ValueError(
            "non-bitwise fused RMSNorm dimensions above "
            f"{_MAX_OPTIMIZED_FUSED_BWD_DIM} are inference-only"
        )
    if (
        weight is not None
        and torch.is_grad_enabled()
        and (x.requires_grad or weights.requires_grad or weight.requires_grad)
    ):
        raise NotImplementedError(
            "the input-scale weight of fused_rmsnorm_combine is "
            "inference/forward-only; no backward supports it"
        )
    return _FusedRMSNormCombine.apply(
        x,
        weights,
        eps,
        norm_output_dtype,
        output_dtype,
        require_bitwise,
        use_kahan,
        weight,
        gain_center,
    )
