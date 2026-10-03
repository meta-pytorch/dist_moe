# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL RMSNorm kernel wrapper with FP64 reduction and fused gain_center.

This module provides a PyTorch-facing API for RMSNorm using the Quack CuTe DSL
kernel infrastructure. It extends the upstream Quack RMSNorm with:

1. **Fused gain_center** — the weight offset (w + gain_center) is applied inside
    the kernel in both forward and backward passes, avoiding a separate
    elementwise add. This is a dynamic runtime argument that does not trigger
    recompilation.

2. **Fused input_scale** — the effective weight scales the input before the
    RMS reduction without materializing a temporary tensor.

3. **FP64 internal reduction support**

4. **GB300 guard** — errors out if FP64 reduction is requested on GB300.

"""

import functools
import math
from typing import Optional, Tuple

import torch
from cutlass import Float8E4M3FN
from torch import Tensor

from ...formats import (
    block_scaled_format_constants,
    BlockScaledFormat,
    ScaleFactorLayout,
)
from .. import _dsl_compat as _cute_extern  # noqa: F401
from .._cuda_context import ensure_cuda_driver_context
from .._quack.cute_dsl_utils import torch2cute_dtype_map
from ._fused_dw_reduce_and_quant_kernel import FusedDwReduce
from ._rmsnorm_bwd_kernel import _MX_SF_VEC_SIZE, FusedRMSNormBwd
from ._rmsnorm_common import (
    _fake_div,
    _fake_div_widths,
    _normalize_for_alignment,
)
from ._rmsnorm_fwd_kernel import FusedRMSNormFwd, RMSNormKahan
from ._rmsnorm_small_d_bwd_kernel import RMSNormSmallDBwd
from ._rmsnorm_small_d_fwd_kernel import RMSNormSmallDFwd

# Minimum cp_async copy size in bits.
_MIN_CP_ASYNC_BITS = 32


def _check_dimension_alignment(D: int, *dtypes: torch.dtype) -> None:
    """Validate D is compatible with CuTe cp_async (min 32-bit copy size)."""
    max_width = max(
        torch2cute_dtype_map[dt].width for dt in dtypes if dt in torch2cute_dtype_map
    )
    vecsize = math.gcd(D, 128 // max_width)
    cp_size = vecsize * max_width
    if cp_size < _MIN_CP_ASYNC_BITS:
        raise ValueError(
            f"CuTe RMSNorm requires D={D} to satisfy cp_async alignment. "
            f"For {max_width}-bit dtypes, D must be divisible by "
            f"{_MIN_CP_ASYNC_BITS // max_width}."
        )


def _small_d_bwd_selected(
    x: Tensor,
    w: Optional[Tensor],
    dy: Tensor,
    D: int,
    quant_dx: bool,
    resolved_dx_dtype: torch.dtype,
    use_fused_norm_reductions: bool,
) -> bool:
    """Static kernel selection for the backward: the small-D kernel serves
    every input it is capable of — scaleless (it has no dw accumulation),
    non-quant, equal-width x/dy, dx in {x.dtype, fp32}, and a supported
    (D, widths) geometry regardless of layout. On Blackwell the bf16
    compute tree matches the general kernel's, so dx
    is bitwise identical to a general-kernel launch of the same rows (the
    parity tests enforce this); elsewhere (fp32 everywhere, bf16 on Hopper
    whose general launch table differs) the trees diverge at round-off
    level: equally accurate, not bitwise. Dispatch therefore requires the
    ``use_fused_norm_reductions`` opt-in, like the rest of the
    numerics-changing kernel bundle. TODO: drop the opt-in requirement once
    a numerics ladder derisks the bundle and it becomes the default."""
    if (
        not use_fused_norm_reductions
        or w is not None
        or quant_dx
        or dy.dtype != x.dtype
        or resolved_dx_dtype not in (x.dtype, torch.float32)
    ):
        return False
    return RMSNormSmallDBwd.can_implement(
        torch2cute_dtype_map[x.dtype],
        torch2cute_dtype_map[dy.dtype],
        torch2cute_dtype_map[resolved_dx_dtype],
        D,
    )


def _small_d_bwd(
    x: Tensor,
    dy_2d: Tensor,
    rstd_1d: Tensor,
    D: int,
    resolved_dx_dtype: torch.dtype,
    T_hint: int,
) -> Tensor:
    """Launch the small-D backward (selection already made by
    :func:`_small_d_bwd_selected`); x binds in its original layout."""
    plan = RMSNormSmallDBwd.for_input(
        x,
        torch2cute_dtype_map[x.dtype],
        torch2cute_dtype_map[dy_2d.dtype],
        torch2cute_dtype_map[resolved_dx_dtype],
        D,
        T_hint=T_hint,
    )
    dx = torch.empty(x.numel() // D, D, device=x.device, dtype=resolved_dx_dtype)
    with ensure_cuda_driver_context():
        plan.launch(x, dy_2d, rstd_1d, dx)
    return dx


def _reshape_for_large_d(x: Tensor, div: int) -> Tensor:
    """Flatten x for the general kernels, materializing when the flattened
    view violates their compile-time contract (unit inner stride, div-aligned
    base and row stride) — e.g. a permuted layout whose reshape merges into
    a non-unit-inner-stride view."""
    return _normalize_for_alignment(x.reshape(-1, x.shape[-1]), div)


# ---------------------------------------------------------------------------
# Compilation and launch helpers for the fused forward kernel
# ---------------------------------------------------------------------------


def _fused_rmsnorm_fwd(
    x: Tensor,
    w: Optional[Tensor],
    *,
    eps: float,
    output_dtype: torch.dtype,
    gain_center: float = 0.0,
    use_fp64_reduction: bool = False,
    input_scale: bool = False,
    use_fused_norm_reductions: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Launch the fused forward kernel with gain_center.

    Static dispatch on (D, dtypes), behind the ``use_fused_norm_reductions``
    opt-in: shapes the small-D load-once kernel supports (D <= 256 with 16B
    rows) run on it in every layout — the strided qkv Q/K slices are read in
    place, contiguous inputs bind as flat rows. Its bf16 reduction tree
    matches the general kernel's, so those outputs are bitwise identical to
    a general-kernel launch of the same rows (fp32 diverges at round-off
    level, hence the opt-in; the dispatch also changes the memory profile by
    eliding the strided-slice copies). Wide-D shapes, fp64-reduction
    requests, and flag-off callers take the general kernel exactly as
    before.
    """
    D = x.shape[-1]
    M = x.numel() // D
    dtype = torch2cute_dtype_map[x.dtype]
    out_dt = torch2cute_dtype_map[output_dtype]
    w_dt = torch2cute_dtype_map[w.dtype] if w is not None else None

    # TODO: drop the flag condition once a numerics ladder derisks the
    # bundle and it becomes the default
    if (
        use_fused_norm_reductions
        and not input_scale
        and not use_fp64_reduction
        and RMSNormSmallDFwd.can_implement(dtype, out_dt, w_dt, D)
    ):
        # Bucket M to a power of two so varying token counts do not recompile
        # the JIT; T_hint only sizes the persistent grid and chunk defaults,
        # never the per-row numerics.
        T_hint = 1 << max(0, int(M) - 1).bit_length()
        plan = RMSNormSmallDFwd.for_input(
            x, dtype, out_dt, w_dt, D, True, T_hint=T_hint
        )
        out = torch.empty(M, D, device=x.device, dtype=output_dtype)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)
        with ensure_cuda_driver_context():
            plan.launch(x, w, out, rstd, eps, gain_center)
        return out.reshape(x.shape), rstd.reshape(x.shape[:-1])

    x_2d = _reshape_for_large_d(x, _fake_div(D, (dtype, out_dt, w_dt)))
    out = torch.empty_like(x_2d, dtype=output_dtype)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    with ensure_cuda_driver_context():
        FusedRMSNormFwd.compile(
            dtype,
            out_dt,
            w_dt,
            D,
            True,
            use_fp64_reduction,
            input_scale,
        )(x_2d, w, out, rstd, eps, gain_center)

    return out.reshape(x.shape), rstd.reshape(x.shape[:-1])


# ---------------------------------------------------------------------------
# Compilation and launch helpers for the fused backward kernel
# ---------------------------------------------------------------------------


def _fused_rmsnorm_bwd(
    x: Tensor,
    w: Optional[Tensor],
    dy: Tensor,
    rstd: Tensor,
    *,
    gain_center: float = 0.0,
    dx_dtype: Optional[torch.dtype] = None,
    sf_dtype: Optional[torch.dtype] = None,
    sf_vec_size: Optional[int] = None,
    input_scale: bool = False,
    use_fused_norm_reductions: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Launch the fused backward kernel with gain_center.

    Returns ``(dx, dx_scales, dw)``. ``dx_scales`` is None unless
    ``dx_dtype=torch.float8_e4m3fn``, which selects the fused MXFP8 epilogue:
    dx is E4M3 data and dx_scales holds the NATURAL-layout
    ``[M, D // sf_vec_size]`` scale bytes in ``sf_dtype`` (both resolved by
    the caller from the block-scaled format table).

    ``use_fused_norm_reductions`` opts the plain round-to-nearest path into
    the numerics-changing optimization bundle: the FusedDwReduce dw finalize
    (instead of the caller-side ``dw_partial.sum``) and the retuned
    pure-16-bit launch bands. Quantized variants always use the fused finalize.
    """
    D = x.shape[-1]
    M = x.numel() // D
    device = x.device
    involved_widths = [
        torch2cute_dtype_map[x.dtype].width,
        torch2cute_dtype_map[dy.dtype].width,
        torch2cute_dtype_map[(dx_dtype or x.dtype)].width,
    ]
    if w is not None:
        involved_widths.append(torch2cute_dtype_map[w.dtype].width)
    bwd_div = _fake_div_widths(D, involved_widths)
    dy_2d = _reshape_for_large_d(dy, bwd_div)
    rstd_1d = rstd.reshape(-1).to(torch.float32)
    # Bucket M to a power of two so varying token counts do not recompile the JIT.
    T_hint = 1 << max(0, int(M) - 1).bit_length()

    quant_dx = dx_dtype == torch.float8_e4m3fn
    resolved_dx_dtype = dx_dtype or x.dtype
    # TODO: remove the legacy finalize once a numerics ladder derisks
    # use_fused_norm_reductions and it becomes the default
    use_legacy_rn_dw_finalize = not quant_dx and not use_fused_norm_reductions

    # Static dispatch on (D, dtypes): every backward the small-D kernel is
    # capable of: usually the qk-norm slices, but generally any small red dim.
    if _small_d_bwd_selected(
        x, w, dy, D, quant_dx, resolved_dx_dtype, use_fused_norm_reductions
    ):
        dx = _small_d_bwd(
            x,
            dy_2d,
            rstd_1d,
            D,
            resolved_dx_dtype,
            T_hint,
        )
        return dx, None, None

    x_2d = _reshape_for_large_d(x, bwd_div)

    if quant_dx:
        # Mirror the kernel's MXFP8 epilogue constraint before allocation.
        max_width = max(involved_widths)
        vecsize = FusedRMSNormBwd.dx_epilogue_vecsize(D, max_width)
        if vecsize not in (4, 8):
            raise NotImplementedError(
                "MXFP8 dx epilogues require a 4- or 8-element vector "
                f"size, got {vecsize} for D={D} (widest operand: "
                f"{max_width}-bit)"
            )

    dx_sf = None
    if quant_dx:
        assert sf_dtype is not None and sf_vec_size == _MX_SF_VEC_SIZE
        dx = torch.empty(M, D, device=device, dtype=torch.float8_e4m3fn)
        dx_sf = torch.empty(M, D // sf_vec_size, device=device, dtype=sf_dtype)
        dx_arg = dx.view(torch.uint8)
        dx_dt = Float8E4M3FN
    else:
        dx = torch.empty_like(x_2d, dtype=resolved_dx_dtype)
        dx_arg = dx
        dx_dt = torch2cute_dtype_map[dx.dtype]
    sm_count = FusedRMSNormBwd.resolve_sm_count(
        D,
        x.element_size() * 8,
        dy.element_size() * 8,
        dx.element_size() * 8,
        T_hint,
        device,
        use_fused_norm_reductions=use_fused_norm_reductions,
    )
    if w is not None:
        dw_partial = torch.empty(sm_count, D, device=device, dtype=torch.float32)
        dw = (
            None
            if use_legacy_rn_dw_finalize
            else torch.empty(D, device=device, dtype=w.dtype)
        )
    else:
        dw_partial = None
        dw = None

    dtype = torch2cute_dtype_map[x.dtype]
    dout_dt = torch2cute_dtype_map[dy.dtype]
    w_dt = torch2cute_dtype_map[w.dtype] if w is not None else None

    with ensure_cuda_driver_context():
        FusedRMSNormBwd.compile(
            D,
            dtype,
            dout_dt,
            dx_dt,
            w_dt,
            False,  # has_db_partial
            None,  # dres_dtype
            None,  # dres_out_dtype
            dw_partial is not None,
            T_hint=T_hint,
            use_fused_norm_reductions=use_fused_norm_reductions,
            input_scale=input_scale,
        )(
            x_2d,
            w,
            dy_2d,
            None,  # dresidual_out
            rstd_1d,
            dx_arg,
            dx_sf.view(torch.uint8) if dx_sf is not None else None,
            dw_partial,
            None,  # dresidual
            None,  # db_partial
            gain_center,
        )
        if w is not None and not use_legacy_rn_dw_finalize:
            # One fused kernel column-reduces the fp32 partials and downcasts.
            assert dw is not None
            FusedDwReduce.compile(D, torch2cute_dtype_map[w.dtype])(dw_partial, dw)
    if w is not None and use_legacy_rn_dw_finalize:
        assert dw_partial is not None
        dw = dw_partial.sum(dim=0).to(w.dtype)
    return dx, dx_sf, dw


# ---------------------------------------------------------------------------
# Compilation and launch helpers for the Kahan-compensated forward kernel
# ---------------------------------------------------------------------------


def _kahan_rmsnorm_fwd(
    x: Tensor,
    w: Tensor | None,
    *,
    eps: float,
    output_dtype: torch.dtype | None = None,
    gain_center: float = 0.0,
    input_scale: bool = False,
) -> tuple[Tensor, Tensor]:
    """Launch the Kahan-compensated forward kernel.

    Uses Kahan compensated summation for the per-thread sum-of-squares,
    giving nearly FP64-equivalent accuracy while staying in FP32.

    Args:
        x: Input tensor of shape (..., D).
        w: Optional weight tensor of shape (D,).
        eps: Epsilon for numerical stability.
        output_dtype: Output dtype (default: x.dtype).
        gain_center: Additive center for weight. Default 0.0.
        input_scale: If True, weight is applied to input before normalization:
            y = rms_norm(w * x). Requires w to be not None. Default False.

    Returns:
        Tuple of (output, rstd).
    """
    if output_dtype is None:
        output_dtype = x.dtype

    orig_shape = x.shape
    D = x.shape[-1]
    kahan_widths = [
        torch2cute_dtype_map[x.dtype].width,
        torch2cute_dtype_map[output_dtype].width,
    ]
    if w is not None:
        kahan_widths.append(torch2cute_dtype_map[w.dtype].width)
    x_2d = _reshape_for_large_d(x, _fake_div_widths(D, kahan_widths))
    M = x_2d.shape[0]

    if M == 0:
        raise ValueError("Kahan RMSNorm does not support empty tensors (M=0).")

    out = torch.empty_like(x_2d, dtype=output_dtype)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    dtype = torch2cute_dtype_map[x.dtype]
    out_dtype_cute = torch2cute_dtype_map[output_dtype]
    weight_dtype = torch2cute_dtype_map[w.dtype] if w is not None else None

    with ensure_cuda_driver_context():
        RMSNormKahan.compile(
            dtype,
            out_dtype_cute,
            weight_dtype,
            D,
            True,
            input_scale,
        )(x_2d, w, out, rstd, eps, gain_center)

    return out.reshape(orig_shape), rstd.reshape(orig_shape[:-1])


def _validate_reduction_options(
    device: torch.device,
    use_kahan_summation: bool,
    use_fp64_reduction: Optional[bool],
) -> None:
    if use_fp64_reduction and use_kahan_summation:
        raise ValueError(
            "use_fp64_reduction=True cannot be combined with use_kahan_summation=True."
        )

    if use_fp64_reduction:
        device_name = torch.cuda.get_device_properties(device).name
        if "GB300" in device_name:
            raise RuntimeError(
                "use_fp64_reduction=True is not supported on GB300 "
                "(FP64 throughput is 1/64 of FP32). "
                "Use use_fp64_reduction=False or None instead."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cute_rmsnorm_fwd(
    x: torch.Tensor,
    w: torch.Tensor | None,
    *,
    eps: float,
    y: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    gain_center: float = 0.0,
    use_kahan_summation: bool = False,
    use_fp64_reduction: Optional[bool] = None,
    input_scale: bool = False,
    use_fused_norm_reductions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for RMSNorm using CuTe DSL.

    Selects the kernel variant based on use_kahan_summation/use_fp64_reduction:
      - use_kahan_summation=True: Kahan compensated FP32 reduction
      - use_kahan_summation=False: FP32 reduction unless use_fp64_reduction=True
      - use_fp64_reduction=True: FP64 sum-of-squares reduction (errors on GB300)
      - use_fp64_reduction=False/None: FP32 reduction (default)

    Args:
        x: Input tensor of shape (..., D).
        w: Optional weight tensor of shape (D,).
        eps: Numerical stability constant.
        y: Optional pre-allocated output tensor.
        output_dtype: Output dtype (default: x.dtype).
        gain_center: Additive center for weight (fused). Default 0.0.
        use_kahan_summation: Whether to use Kahan summation for numerical stability.
            Default False.
        use_fp64_reduction: Use FP64 reduction for sum-of-squares. None = False.
            Raises RuntimeError on GB300 where FP64 throughput is 1/64 of FP32.
        input_scale: If True, weight is applied to input before normalization:
            y = rms_norm(w * x). Requires w to be not None. Default False.
        use_fused_norm_reductions: Opt into the numerics-changing kernel
            optimization bundle; for the forward this enables the small-D
            load-once dispatch (see :func:`_fused_rmsnorm_fwd`). Default
            False.

    Returns:
        Tuple of (output, rstd).
    """
    if input_scale:
        assert w is not None, "input_scale requires weight"
    if output_dtype is None:
        output_dtype = torch.promote_types(x.dtype, w.dtype) if input_scale else x.dtype

    if x.dtype == torch.float64 or output_dtype == torch.float64:
        raise NotImplementedError("CuTe RMSNorm does not support float64 input/output.")

    D = x.shape[-1]
    M = x.numel() // D
    involved_dtypes = [x.dtype, output_dtype]
    if w is not None:
        involved_dtypes.append(w.dtype)
    _check_dimension_alignment(D, *involved_dtypes)
    if y is not None:
        assert y.shape == x.shape, f"{y.shape=} != {x.shape=}"
        assert y.dtype == output_dtype, f"{y.dtype=} != {output_dtype=}"
        if y.stride(-1) != 1:
            raise NotImplementedError("RMSNorm kernel requires contiguous features")
        # view, not reshape: a buffer that cannot alias a flattened 2-D layout
        # raises here instead of receiving the result through a silent copy.
        y = y.view(-1, D)

    if M == 0:
        # Empty expert batches can occur during MoE prefill warmup when no
        # tokens are routed to a particular expert. Return empty output.
        out = torch.empty_like(x, dtype=output_dtype) if y is None else y.view(x.shape)
        return out, torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)

    _validate_reduction_options(x.device, use_kahan_summation, use_fp64_reduction)

    if use_kahan_summation:
        y_out, rstd = _kahan_rmsnorm_fwd(
            x,
            w,
            eps=eps,
            output_dtype=output_dtype,
            gain_center=gain_center,
            input_scale=input_scale,
        )
    else:
        y_out, rstd = _fused_rmsnorm_fwd(
            x,
            w,
            eps=eps,
            output_dtype=output_dtype,
            gain_center=gain_center,
            use_fp64_reduction=use_fp64_reduction or False,
            input_scale=input_scale,
            use_fused_norm_reductions=use_fused_norm_reductions,
        )

    if y is not None:
        y.copy_(y_out.reshape(-1, D))
        return y.view(x.shape), rstd

    return y_out, rstd


@functools.lru_cache(maxsize=None)
def _is_sm100(device: torch.device) -> bool:
    """Cached per-device SM100 gate: the answer is immutable per device and
    the backward validates on every call, so skip the repeated
    device-property queries."""
    return (
        torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 10
    )


def _validate_input_scale_grad_buffers(
    dx: torch.Tensor | None,
    dw: torch.Tensor | None,
    x: torch.Tensor,
    w: torch.Tensor,
) -> None:
    if dx is not None:
        if dx.shape != x.shape:
            raise ValueError(
                f"Pre-allocated dx shape {dx.shape} must match input shape {x.shape}"
            )
        if dx.stride(-1) != 1:
            raise ValueError(
                "Pre-allocated dx must be contiguous along the feature dimension"
            )
        if dx.dtype != x.dtype:
            raise ValueError(
                f"Pre-allocated dx dtype {dx.dtype} must match input dtype {x.dtype} "
                "with input_scale"
            )
        if dx.device != x.device:
            raise ValueError(
                f"Pre-allocated dx device {dx.device} must match input device "
                f"{x.device} with input_scale"
            )
    if dw is not None:
        if dw.shape != w.shape:
            raise ValueError(
                f"Pre-allocated dw shape {dw.shape} must match weight shape {w.shape}"
            )
        if dw.stride(-1) != 1:
            raise ValueError(
                "Pre-allocated dw must be contiguous along the feature dimension"
            )
        if dw.dtype != w.dtype:
            raise ValueError(
                f"Pre-allocated dw dtype {dw.dtype} must match weight dtype {w.dtype} "
                "with input_scale"
            )
        if dw.device != w.device:
            raise ValueError(
                f"Pre-allocated dw device {dw.device} must match weight device "
                f"{w.device} with input_scale"
            )


def cute_rmsnorm_bwd(  # noqa: C901
    dy: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor | None,
    rstd: torch.Tensor,
    *,
    dx: torch.Tensor | None = None,
    dw: torch.Tensor | None = None,
    gain_center: float = 0.0,
    input_scale: bool = False,
    dx_dtype: torch.dtype | None = None,
    use_fused_norm_reductions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Backward pass for RMSNorm using CuTe DSL kernel with fused gain_center.

    The gain_center offset is fused directly into the backward kernel, computing
    wdy = dout * (w + gain_center) without a separate elementwise add.
    dw = sum(dout * x_hat) is independent of the gain_center constant.

    ``use_fused_norm_reductions`` opts into the numerics-changing optimization
    bundle (fused dw finalize + retuned 16-bit launch bands); see
    :func:`_fused_rmsnorm_bwd`. Off by default: the plain RN path then
    reduces dw with the caller-side ``dw_partial.sum`` and keeps the
    bitwise-stable launch geometry.

    Args:
        dy: Gradient of loss w.r.t. output, shape (..., D).
        x: Original input tensor from forward pass.
        w: Optional weight tensor from forward pass.
        rstd: Reciprocal standard deviation from forward pass.
        dx: Optional pre-allocated tensor for input gradient. With
            ``input_scale=True``, it must match ``x`` in shape, dtype, and
            device and be contiguous along the feature dimension; the final
            chain-rule gradient is written into it in-place.
        dw: Optional pre-allocated tensor for weight gradient. With
            ``input_scale=True``, it must match ``w`` in shape, dtype, and
            device and be contiguous along the feature dimension; the final
            chain-rule gradient is written into it in-place.
        gain_center: Additive center for weight (must match forward). Default 0.0.
        input_scale: If True, backward pass accounts for weight applied to input
            before normalization. Requires w to be not None. Default False.
        dx_dtype: Dtype of the returned dx. Defaults to ``x.dtype``. The kernel
            computes dx in FP32 and downcasts once on the final store.

    Returns:
        Tuple of (grad_input, grad_weight) where grad_weight is None if w is None.
    """
    if input_scale:
        if dx_dtype is not None:
            raise NotImplementedError("dx_dtype is not supported with input_scale")
        if w is None:
            raise ValueError("input_scale requires weight")
        _validate_input_scale_grad_buffers(dx, dw, x, w)
    if x.dtype == torch.float64 or dy.dtype == torch.float64:
        raise NotImplementedError("CuTe RMSNorm backward does not support float64.")

    if dx is not None and dx_dtype is not None and dx.dtype != dx_dtype:
        raise ValueError(
            f"Pre-allocated dx dtype {dx.dtype} does not match dx_dtype {dx_dtype}"
        )
    if dx_dtype is None and dx is not None:
        dx_dtype = dx.dtype
    resolved_dx_dtype = dx_dtype or x.dtype
    if resolved_dx_dtype == torch.float8_e4m3fn:
        raise NotImplementedError(
            "MXFP8 dx needs the scale outputs; use cute_rmsnorm_bwd_quant"
        )
    D = x.shape[-1]
    M = x.numel() // D
    involved_dtypes = [x.dtype, dy.dtype, resolved_dx_dtype]
    if w is not None:
        involved_dtypes.append(w.dtype)
    _check_dimension_alignment(D, *involved_dtypes)

    if M == 0:
        # Honor pre-allocated buffers and match the M>0 path's dw dtype
        # (w.dtype) so empty expert batches don't flip dtypes or leave a
        # caller's persistent dw holding the previous step's gradients.
        dx_out = dx if dx is not None else torch.empty_like(x, dtype=resolved_dx_dtype)
        if w is None:
            dw_out = None
        elif dw is None:
            dw_out = torch.zeros_like(w)
        else:
            dw.zero_()
            dw_out = dw
        return dx_out, dw_out

    orig_shape = x.shape

    dx_out, _, dw_out = _fused_rmsnorm_bwd(
        x,
        w,
        dy,
        rstd,
        gain_center=gain_center,
        dx_dtype=resolved_dx_dtype,
        input_scale=input_scale,
        use_fused_norm_reductions=use_fused_norm_reductions,
    )

    dx_out = dx_out.reshape(orig_shape)

    if dx is not None:
        dx.copy_(dx_out)
        dx_out = dx

    if dw is not None and dw_out is not None:
        dw.copy_(dw_out)
        dw_out = dw

    return dx_out, dw_out


def cute_rmsnorm_bwd_quant(
    dy: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor | None,
    rstd: torch.Tensor,
    *,
    gain_center: float = 0.0,
    format: BlockScaledFormat = BlockScaledFormat.MXFP8_E4M3,
    layout: ScaleFactorLayout = ScaleFactorLayout.NATURAL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """RMSNorm backward with dx fused-quantized to MXFP8 (E4M3 + E8M0 scales).

    Computes the same dx as :func:`cute_rmsnorm_bwd` in FP32 internally, then
    quantizes it in-register per 32-column block: rceil E8M0 scale (identical
    numerics to the retained block-scaled quantization rule) and an E4M3 data
    round-to-nearest-even cast. Intended for the post-expert-norm
    backward, whose dx feeds the MoE combine-backward all-to-all in MXFP8.

    Args:
        dy: Gradient of loss w.r.t. output, shape (..., D).
        x: Original input tensor from forward pass.
        w: Optional weight tensor from forward pass.
        rstd: Reciprocal standard deviation from forward pass.
        gain_center: Additive center for weight (must match forward). Default 0.0.
        format: Target block-scaled format. Only MXFP8_E4M3 is supported.
        layout: Scale-factor layout. Only NATURAL (``[*, D // 32]`` row-major)
            is supported.

    Returns:
        Tuple of (dx_q, dx_scales, dw): ``dx_q`` is float8_e4m3fn of x's shape,
        ``dx_scales`` is float8_e8m0fnu of shape ``(*x.shape[:-1], D // 32)``,
        and ``dw`` is None if w is None.
    """
    if format != BlockScaledFormat.MXFP8_E4M3:
        raise NotImplementedError(
            f"Fused RMSNorm bwd quantization only supports MXFP8_E4M3, got {format}"
        )
    if layout != ScaleFactorLayout.NATURAL:
        raise NotImplementedError(
            f"Fused RMSNorm bwd quantization only supports NATURAL scales, got {layout}"
        )
    if x.dtype == torch.float64 or dy.dtype == torch.float64:
        raise NotImplementedError("CuTe RMSNorm backward does not support float64.")
    if not _is_sm100(x.device):
        raise NotImplementedError(
            "Fused MXFP8 dx quantization requires SM100+ (Blackwell)"
        )
    qdata_dtype, sf_dtype, sf_vec_size = block_scaled_format_constants(format)
    assert sf_vec_size == _MX_SF_VEC_SIZE
    D = x.shape[-1]
    M = x.numel() // D
    if D % sf_vec_size != 0:
        raise ValueError(f"D={D} must be divisible by sf_vec_size={sf_vec_size}")
    involved_dtypes = [x.dtype, dy.dtype, qdata_dtype]
    if w is not None:
        involved_dtypes.append(w.dtype)
    _check_dimension_alignment(D, *involved_dtypes)

    orig_shape = x.shape
    sf_shape = (*orig_shape[:-1], D // sf_vec_size)
    if M == 0:
        dx_q = torch.empty(orig_shape, dtype=qdata_dtype, device=x.device)
        dx_sf = torch.empty(sf_shape, dtype=sf_dtype, device=x.device)
        # w.dtype matches the M>0 path (FusedDwReduce downcasts the fp32
        # partials straight to w.dtype) and the registered fake.
        dw_out = torch.zeros_like(w) if w is not None else None
        return dx_q, dx_sf, dw_out

    dx_q, dx_sf, dw_out = _fused_rmsnorm_bwd(
        x,
        w,
        dy,
        rstd,
        gain_center=gain_center,
        dx_dtype=qdata_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
    )
    return dx_q.reshape(orig_shape), dx_sf.reshape(sf_shape), dw_out
