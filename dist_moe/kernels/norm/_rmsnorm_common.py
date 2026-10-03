# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the CuTe DSL RMSNorm kernels.

FMA-chain dot products used by the forward, backward, and small-D kernels'
sum-of-squares / mean reductions, plus the small-D kernels' shared host-side
infrastructure: the torch-input decode (:func:`_resolve_input` /
:class:`_NormLaunch`), the divisibility contract (:func:`_fake_div`), the
load-walk selection (:func:`_head_walk` / :func:`_walk_row_coords`), and the
warp-pipeline smem/grid planning (:func:`_plan_grid`).
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr, Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

# Row bytes each pipeline lane moves per vector access; sized for
# LDGSTS.128 / LDS.128 / STG.128.
_LANE_BYTES = 16
# Upper bound on a per-warp smem ring: stages * chunk_rows * row_bytes.
_MAX_WARP_SMEM_BYTES = 32 * 1024
# Conservative per-CTA smem ceiling (sm_100a allows ~227KB per block).
_MAX_CTA_SMEM_BYTES = 224 * 1024


@dsl_user_op
def _fma_rn_f32(
    a: float | Float32,
    b: float | Float32,
    c: float | Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _binary_rn_f32(a, b, op: str, *, loc=None, ip=None) -> Float32:
    assert op in ("add", "sub", "mul")
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            f"{op}.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _zero_minus_f32(value: Float32) -> Float32:
    """The zero-sign ABI requires subtraction from positive zero, not fneg."""
    return _binary_rn_f32(0.0, value, "sub")


@cute.jit
def _fma_dot_f32(x: cute.TensorSSA, y: cute.TensorSSA) -> Float32:
    acc = Float32(0.0)
    for i in cutlass.range(cute.size(x.shape), unroll_full=True):
        acc = _fma_rn_f32(x[i], y[i], acc)
    return acc


@cute.jit
def _walk_row_coords(
    row,
    head_group: cutlass.Constexpr[int],
    tokens: cutlass.Constexpr[int],
    t_minor: cutlass.Constexpr[bool],
):
    """(t, h) of walk-row ``row``: the t-minor walk runs tokens fastest, the
    h-minor walk runs heads fastest (= flat output order). Shared by the
    small-D forward and backward — the walk order determines the output-row
    mapping, so the two must stay identical. Both divisors are compile-time
    constants, so the div/mod folds to multiply-shift sequences."""
    if const_expr(t_minor):
        h = row // tokens
        t = row - h * tokens
    else:
        t = row // head_group
        h = row - t * head_group
    return t, h


class _NormXForm:
    """Input forms a torch x can bind to (see :meth:`_NormLaunch._bind_x`)."""

    VIEW_2D = 0  # flattens to a (M, D) view; rows may carry a uniform stride
    HEAD_3D = 1  # head-sliced (T, H, D) view, read in place
    COPY = 2  # contiguous-copy fallback


class _NormLaunch:
    """A norm-kernel variant bound to one torch input form.

    Built by ``for_input``: carries the compile parameters (including the
    head-group selection derived from the input's strides) and the resolved
    input form. ``launch`` (and the ``compile()`` launcher it backs) takes x
    in its original layout and rebinds it to the view (or contiguous copy)
    the compiled variant expects, so call sites read identically for
    contiguous, row-strided, and head-sliced inputs.
    """

    __slots__ = ("_kernel_cls", "_compile_args", "_form", "head_group")

    def __init__(
        self,
        kernel_cls,
        compile_args: tuple,
        form: int,
        head_group: Optional[int],
    ):
        self._kernel_cls = kernel_cls
        self._compile_args = compile_args
        self._form = form
        self.head_group = head_group

    def _bind_x(self, x: torch.Tensor) -> torch.Tensor:
        D = x.shape[-1]
        if self._form == _NormXForm.VIEW_2D:
            return x.view(-1, D)
        if self._form == _NormXForm.HEAD_3D:
            return x.view(-1, self.head_group, D)
        return x.reshape(-1, D).contiguous()

    def launch(self, x: torch.Tensor, *args):
        """Compile (cached) and launch with x in its original layout."""
        return self._kernel_cls.compile(*self._compile_args)(self._bind_x(x), *args)

    def compile(self):
        """Resolve the cached compiled variant; returns the launcher."""
        fn = self._kernel_cls.compile(*self._compile_args)

        def launch(x: torch.Tensor, *args):
            return fn(self._bind_x(x), *args)

        return launch


def _fake_div_widths(N: int, widths) -> int:
    """:func:`_fake_div` over raw operand widths in bits."""
    return math.gcd(N, *(128 // width for width in widths))


def _fake_div(N: int, dtypes: tuple) -> int:
    """Compile-time divisibility (in elements) the kernel assumes on every
    non-contiguous stride and on the base address. Single source for
    ``compile``'s fake tensors and ``for_input``'s alignment gate.
    """
    return _fake_div_widths(N, (dt.width for dt in dtypes if dt is not None))


def _aligned(t: torch.Tensor, div: int) -> bool:
    """Base address and every non-inner stride honor the kernels'
    compile-time divisibility (``div``, in elements)."""
    return t.data_ptr() % (div * t.element_size()) == 0 and all(
        s % div == 0 for s in t.stride()[:-1]
    )


def _normalize_for_alignment(tensor: torch.Tensor, div: int) -> torch.Tensor:
    """Return storage satisfying the vectorized norm kernels' alignment contract.

    Compatible row-strided views bind directly. Other layouts are cloned rather
    than made merely contiguous because `.contiguous()` can preserve an already
    contiguous view whose base address is misaligned.
    """
    return (
        tensor
        if tensor.stride(-1) == 1 and _aligned(tensor, div)
        else tensor.clone(memory_format=torch.contiguous_format)
    )


def _resolve_input(x: torch.Tensor, div: int) -> tuple[int, Optional[int]]:
    """Resolve x into a kernel input form without copying when possible.

    Returns ``(form, head_group)``:
      - ``(VIEW_2D, None)`` when x flattens to a 2D view (rows may carry a
        uniform stride);
      - ``(HEAD_3D, H)`` when x is a head-sliced view of a wider row (e.g.
        the Q/K slices of a fused qkv projection), read in place;
      - ``(COPY, None)`` otherwise (misaligned base/strides, non-unit inner
        stride).

    ``div`` is the compile-time divisibility the kernel assumes on every
    non-contiguous stride and on the base address (in elements). Callers
    gate on their own compute geometry before resolving — any head count
    binds once the (N, widths) geometry does.
    """
    D = x.shape[-1]
    if x.stride(-1) == 1:
        try:
            x_2d = x.view(-1, D)
        except RuntimeError:
            x_2d = None
        if x_2d is not None and _aligned(x_2d, div):
            return _NormXForm.VIEW_2D, None
        if x.ndim >= 3:
            try:
                x_3d = x.view(-1, x.shape[-2], D)
            except RuntimeError:
                x_3d = None
            if x_3d is not None and x_3d.shape[0] > 0 and _aligned(x_3d, div):
                return _NormXForm.HEAD_3D, x_3d.shape[1]
    return _NormXForm.COPY, None


def _head_walk(x: torch.Tensor) -> tuple[bool, int]:
    """Load-walk order for a head-sliced x: ``(t_minor, tokens)``.

    Walk whichever of the H / T row modes is memory-minor so a warp's
    consecutive rows are gmem-adjacent regardless of how the slice is laid
    out. h-minor (the fused qkv slice case) walks rows in output order;
    t-minor (transposed layouts) walks tokens fastest and needs the token
    count baked for the store-time remap.
    """
    x_3d = x.view(-1, x.shape[-2], x.shape[-1])
    if x_3d.stride(0) < x_3d.stride(1):
        return True, x_3d.shape[0]
    return False, 0


def _plan_grid(
    N: int,
    chunk_bytes: int,
    stages: int,
    warps_per_cta: int,
    chunk_rows: int,
    T_hint: int,
    warp_ring_cap: int,
) -> int:
    """Validate the warp-pipeline smem budget and size the persistent grid.

    Returns the CTA count: enough warps to cover T_hint's chunks once,
    capped at the smem-residency limit (CTAs beyond it would only serialize
    on the smem ring anyway). ``warp_ring_cap`` is the per-warp ring budget
    (the backward rings two streams, so it doubles the forward's).
    """
    if stages * chunk_bytes > warp_ring_cap:
        raise ValueError(
            f"per-warp smem ring {stages}x{chunk_bytes}B exceeds "
            f"{warp_ring_cap}B for N={N}"
        )
    if warps_per_cta * stages * chunk_bytes > _MAX_CTA_SMEM_BYTES:
        raise ValueError(
            f"CTA smem {warps_per_cta}x{stages}x{chunk_bytes}B exceeds "
            f"{_MAX_CTA_SMEM_BYTES}B for N={N}"
        )
    if not torch.cuda.is_available():
        raise ValueError("the small-D RMSNorm kernels require CUDA")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm_smem = getattr(props, "shared_memory_per_multiprocessor", 228 * 1024)
    cta_smem = warps_per_cta * stages * chunk_bytes
    ctas_per_sm = max(1, min(sm_smem // cta_smem, 2048 // (warps_per_cta * 32)))
    cap = props.multi_processor_count * ctas_per_sm
    if T_hint > 0:
        chunks_hint = -(-T_hint // chunk_rows)
        ctas_hint = -(-chunks_hint // warps_per_cta)
    else:
        ctas_hint = cap
    return max(1, min(ctas_hint, cap))
