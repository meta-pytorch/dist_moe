# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
from cutlass import (
    BFloat16,
    Float16,
    Float32,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import dsl_user_op, T

from ..formats import (
    BlockScaledProducer,
    CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE,
    CUBLAS_BLOCKED_ROW_LANES,
    CUBLAS_BLOCKED_ROWS_PER_ATOM,
    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
    E4M3_MIN_SUBNORMAL,
    E8M0_EXP_BIAS,
    FP4_E2M1_HALF_RANGE_ROUND_BOUNDARY,
    FP4_E2M1_MAX,
    FP8_E4M3_HALF_RANGE_ROUND_BOUNDARY,
    FP8_E4M3_MAX,
    FP8_E5M2_HALF_RANGE_ROUND_BOUNDARY,
    FP8_E5M2_MAX,
    NVFP4_MAX,
    NVFP4_NO_CLIP_TARGET_MAX,
    NVFP4_TOKEN_AMAX_FLOOR,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)
from . import _dsl_compat as _cute_extern  # noqa: F401
from ._dsl_compat import fmax, fmin
from ._quack_utils import make_vector
from ._quant_conversion import (
    _abs_f32,
    _cvt_f16x2_to_f32x2,
    _cvt_f32_to_qdata_byte_rn,
    _cvt_f32x2_to_fp8_e8m0x2_rp,
    _cvt_f32x2_to_qdata_byte_rn,
    _cvt_fp8_byte_to_f32,
    _FP32_ONE,
    _FP32_ZERO,
    _pack_b16x2,
    _pack_bits,
    _pack_f32x2_to_u64,
    _unpack_b16x2,
    _unpack_b64_to_i32x2,
)
from ._quant_packing import (
    _cvt_qdata_for_format_id,
    _pack_fp4_e2m1_xn_to_u32_rn,
    _pack_fp8_xn_to_u32,
    _pack_fp8_xn_to_u32_rn,
    _pack_packed_f32x2x2_to_fp8x4_u32_rn,
    BLOCK_SCALED_FORMAT_MXFP4,
    BLOCK_SCALED_FORMAT_MXFP8_E4M3,
    BLOCK_SCALED_FORMAT_MXFP8_E5M2,
    BLOCK_SCALED_FORMAT_NVFP4,
)

BLOCK_SCALED_PRODUCER_IDENTITY = BlockScaledProducer.IDENTITY.value
BLOCK_SCALED_PRODUCER_SWIGLU_FWD = BlockScaledProducer.SWIGLU_FWD.value

_AXES_KERNEL_MODE_SCALAR = 0
_AXES_KERNEL_MODE_TILE2D_FP8_VECTOR = 1
_AXES_KERNEL_MODE_BOTH1D_VECTOR = 2
_AXES_KERNEL_MODE_TILE2D_FP4_VECTOR = 3
_AXES_KERNEL_MODE_SWIGLU_BWD_DXY_F32X2_VECTOR = 4

_FP32_TWO: cutlass.Constexpr[float] = 2.0
_E8M0_RECIP_EXP_BIAS: cutlass.Constexpr[int] = 2 * E8M0_EXP_BIAS
_FP32_EXP_BITS_SHIFT: cutlass.Constexpr[int] = 23
_E8M0_ZERO_BYTE: cutlass.Constexpr[int] = 0
# Per-block scale floor before the E4M3 cast: the smallest E4M3 subnormal.
# A smaller floor (e.g. 2^-16) rounds to zero in the E4M3 cast and turns
# all-zero blocks into inf total scales.
_NVFP4_DYNAMIC_SCALE_FLOOR: cutlass.Constexpr[float] = E4M3_MIN_SUBNORMAL
_NVFP4_TOKEN_AMAX_FLOOR: cutlass.Constexpr[float] = NVFP4_TOKEN_AMAX_FLOOR
# Numerator of the per-token global scale: ``numerator * recip(amax)`` — the
# fleet gs convention (authority: `nvfp4_weight_global_scale`).
_NVFP4_TOKEN_SCALE_NUMERATOR: cutlass.Constexpr[float] = NVFP4_MAX


# -----------------------------------------------------------------------------
# Scale and amax helpers.
# -----------------------------------------------------------------------------


@dsl_user_op
def _compute_fp8_e8m0_recip_from_scale_byte(
    scale_byte: Uint8,
    *,
    loc=None,
    ip=None,
) -> Float32:
    recip_exp = Int32(_E8M0_RECIP_EXP_BIAS) - Int32(scale_byte)
    recip_bits = recip_exp << Int32(_FP32_EXP_BITS_SHIFT)
    return Float32(
        llvm.bitcast(T.f32(), recip_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@cute.jit
def _compute_fp8_e8m0_scale_rceil(max_abs: Float32, target_max_recip: Float32):
    scale_input = max_abs * target_max_recip
    scale, zero_scale = _cvt_f32x2_to_fp8_e8m0x2_rp(scale_input, Float32(0.0))
    scale_byte = Uint8(Uint16(_pack_bits(Int16, scale, zero_scale)) & Uint16(0xFF))
    recip_scale = _compute_fp8_e8m0_recip_from_scale_byte(scale_byte)
    # not-less-than-inf catches a NaN amax too (callers using max.NaN chains
    # deliver NaN here instead of substituting inf per element); both must
    # poison the data bytes the same way.
    recip_scale = (
        recip_scale if scale_input < Float32(float("inf")) else Float32(float("nan"))
    )
    return scale_byte, recip_scale


@cute.jit
def _compute_fp8_e8m0_scale_rceil_x2(
    max_abs0: Float32,
    max_abs1: Float32,
    target_max_recip: Float32,
):
    scale_input0, scale_input1 = cute.arch.mul_packed_f32x2(
        (max_abs0, max_abs1),
        (target_max_recip, target_max_recip),
        rnd="rn",
    )
    scale0, scale1 = _cvt_f32x2_to_fp8_e8m0x2_rp(scale_input0, scale_input1)
    scale_u16 = Uint16(_pack_bits(Int16, scale0, scale1))
    scale0_byte = Uint8(scale_u16 & Uint16(0xFF))
    scale1_byte = Uint8(scale_u16 >> Uint16(8))
    recip0 = _compute_fp8_e8m0_recip_from_scale_byte(scale0_byte)
    recip1 = _compute_fp8_e8m0_recip_from_scale_byte(scale1_byte)
    recip0 = Float32(float("nan")) if scale_input0 == Float32(float("inf")) else recip0
    recip1 = Float32(float("nan")) if scale_input1 == Float32(float("inf")) else recip1
    return scale0_byte, recip0, scale1_byte, recip1


@cute.jit
def _compute_fp8_e4m3_scale_byte_rtne(
    max_abs: Float32, target_max_recip: Float32
) -> Uint8:
    block_amax = Float32(float("nan")) if max_abs == Float32(float("inf")) else max_abs
    block_scale = block_amax * target_max_recip
    block_scale = (
        Float32(E4M3_MIN_SUBNORMAL)
        if block_scale < Float32(E4M3_MIN_SUBNORMAL)
        else block_scale
    )
    block_scale = (
        Float32(FP8_E4M3_MAX) if block_scale > Float32(FP8_E4M3_MAX) else block_scale
    )
    return _cvt_f32_to_qdata_byte_rn(
        block_scale,
        dst_kind=nvvm.CVTPackFloatKind.E4M3x2,
    )


@cute.jit
def _compute_recip_from_fp8_e4m3_scale_byte(scale_byte: Uint8) -> Float32:
    return Float32(_FP32_ONE) / _cvt_fp8_byte_to_f32(
        scale_byte,
        src_kind=nvvm.CVTPackFloatKind.E4M3x2,
    )


@cute.jit
def _compute_nvfp4_scale_byte(
    max_abs: Float32,
    use_nvfp4_no_clip_scale: cutlass.Constexpr[bool],
) -> Uint8:
    target_max: cutlass.Constexpr[float] = FP4_E2M1_MAX
    if cutlass.const_expr(use_nvfp4_no_clip_scale):
        target_max = NVFP4_NO_CLIP_TARGET_MAX
    scale_byte = _compute_fp8_e4m3_scale_byte_rtne(
        max_abs,
        Float32(_FP32_ONE / target_max),
    )
    if cutlass.const_expr(use_nvfp4_no_clip_scale):
        # E4M3 subnormals have fixed absolute spacing, so 17/16 headroom can
        # still round below the ideal scale. One code step is sufficient.
        scale = _cvt_fp8_byte_to_f32(
            scale_byte,
            src_kind=nvvm.CVTPackFloatKind.E4M3x2,
        )
        should_bump = scale_byte < Uint8(0x7E)
        should_bump = should_bump and scale * Float32(FP4_E2M1_MAX) < max_abs
        scale_byte = Uint8(scale_byte + Uint8(1)) if should_bump else scale_byte
    return scale_byte


@cute.jit
def _compute_nvfp4_per_token_scale_and_multiplier(
    block_amax: Float32,
    row_scale: Float32,
    mNvfp4RecipLut: cute.Tensor,
    use_nvfp4_no_clip_scale: cutlass.Constexpr[bool],
):
    target_max: cutlass.Constexpr[float] = FP4_E2M1_MAX
    if cutlass.const_expr(use_nvfp4_no_clip_scale):
        target_max = NVFP4_NO_CLIP_TARGET_MAX
    # Order is load-bearing: fleet convention is ``(amax * gs) * fl(1/6)`` —
    # see the Triton `_nvfp4_scale_and_recip` for why (E4M3 RTNE ties).
    raw_scale = (block_amax * row_scale) * Float32(1.0 / target_max)
    raw_scale = (
        Float32(_NVFP4_DYNAMIC_SCALE_FLOOR)
        if raw_scale < Float32(_NVFP4_DYNAMIC_SCALE_FLOOR)
        else raw_scale
    )
    raw_scale = (
        Float32(FP8_E4M3_MAX) if raw_scale > Float32(FP8_E4M3_MAX) else raw_scale
    )
    scale_byte = _cvt_f32_to_qdata_byte_rn(
        raw_scale,
        dst_kind=nvvm.CVTPackFloatKind.E4M3x2,
    )
    if cutlass.const_expr(use_nvfp4_no_clip_scale):
        # Same one-code bump as _compute_nvfp4_scale_byte: E4M3 subnormals
        # have fixed absolute spacing, so 17/16 headroom can still round the
        # scale below block_amax * row_scale / 6, and the satfinite FP4 store
        # would then pin the block max deterministically (D117854242). The
        # comparison is in the row-scaled domain the multiplier normalizes.
        scale_probe = _cvt_fp8_byte_to_f32(
            scale_byte,
            src_kind=nvvm.CVTPackFloatKind.E4M3x2,
        )
        should_bump = scale_byte < Uint8(0x7E)
        should_bump = (
            should_bump and scale_probe * Float32(FP4_E2M1_MAX) < block_amax * row_scale
        )
        scale_byte = Uint8(scale_byte + Uint8(1)) if should_bump else scale_byte
    scale_f32 = _cvt_fp8_byte_to_f32(
        scale_byte,
        src_kind=nvvm.CVTPackFloatKind.E4M3x2,
    )
    scale_recip = Float32(mNvfp4RecipLut[Int32(scale_byte)])
    # The zero-byte branch is unreachable by construction: the scale floors at
    # _NVFP4_DYNAMIC_SCALE_FLOOR (E4M3 min subnormal, byte 0x01) before the
    # cast, and a NaN amax casts to 0x7F. Kept as defense so that a future
    # layout/packing bug producing byte 0x00 zero-flushes instead of reading a
    # garbage reciprocal from the LUT.
    multiplier = (
        _div_rn_f32_via_rcp_mul(row_scale, scale_f32, scale_recip)
        if scale_byte != Uint8(0)
        else Float32(_FP32_ZERO)
    )
    return scale_byte, multiplier


@cute.jit
def _target_max_for_format_id(format_id: cutlass.Constexpr[int]) -> Float32:
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2):
        return Float32(FP8_E5M2_MAX)
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E4M3):
        return Float32(FP8_E4M3_MAX)
    return Float32(FP4_E2M1_MAX)


@cute.jit
def _target_max_recip_for_format_id(format_id: cutlass.Constexpr[int]) -> Float32:
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2):
        return Float32(_FP32_ONE / FP8_E5M2_MAX)
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E4M3):
        return Float32(_FP32_ONE / FP8_E4M3_MAX)
    return Float32(_FP32_ONE / FP4_E2M1_MAX)


@cute.jit
def _canonicalize_mx_half_range(
    max_abs: Float32,
    scale_byte: Uint8,
    recip_scale: Float32,
    format_id: cutlass.Constexpr[int],
):
    scaled_max = max_abs * recip_scale
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E4M3):
        fits_half_range = scaled_max <= Float32(FP8_E4M3_HALF_RANGE_ROUND_BOUNDARY)
    elif cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2):
        fits_half_range = scaled_max < Float32(FP8_E5M2_HALF_RANGE_ROUND_BOUNDARY)
    else:
        fits_half_range = scaled_max < Float32(FP4_E2M1_HALF_RANGE_ROUND_BOUNDARY)
    should_shift = max_abs > Float32(_FP32_ZERO)
    should_shift = should_shift and scale_byte > Uint8(_E8M0_ZERO_BYTE)
    should_shift = should_shift and fits_half_range
    canonical_scale = Uint8(scale_byte - Uint8(1)) if should_shift else scale_byte
    canonical_recip = recip_scale * Float32(_FP32_TWO) if should_shift else recip_scale
    return canonical_scale, canonical_recip


@cute.jit
def _compute_scale_and_recip_from_amax(
    max_abs: Float32,
    format_id: cutlass.Constexpr[int],
    half_range_scale: cutlass.Constexpr[bool],
    use_nvfp4_no_clip_scale: cutlass.Constexpr[bool] = False,
):
    if cutlass.const_expr(
        format_id == BLOCK_SCALED_FORMAT_MXFP8_E4M3
        or format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2
    ):
        scale_byte, recip_scale = _compute_fp8_e8m0_scale_rceil(
            max_abs,
            _target_max_recip_for_format_id(format_id),
        )
        if cutlass.const_expr(half_range_scale):
            scale_byte, recip_scale = _canonicalize_mx_half_range(
                max_abs,
                scale_byte,
                recip_scale,
                format_id,
            )
        return scale_byte, recip_scale
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP4):
        scale_byte, recip_scale = _compute_fp8_e8m0_scale_rceil(
            max_abs,
            Float32(_FP32_ONE / FP4_E2M1_MAX),
        )
        if cutlass.const_expr(half_range_scale):
            scale_byte, recip_scale = _canonicalize_mx_half_range(
                max_abs,
                scale_byte,
                recip_scale,
                format_id,
            )
        return scale_byte, recip_scale
    scale_byte = _compute_nvfp4_scale_byte(
        max_abs,
        use_nvfp4_no_clip_scale,
    )
    return scale_byte, _compute_recip_from_fp8_e4m3_scale_byte(scale_byte)


@cute.jit
def _compute_scale_and_recip_from_amax_x2(
    max_abs0: Float32,
    max_abs1: Float32,
    format_id: cutlass.Constexpr[int],
    half_range_scale: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(
        not half_range_scale
        and (
            format_id == BLOCK_SCALED_FORMAT_MXFP8_E4M3
            or format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2
            or format_id == BLOCK_SCALED_FORMAT_MXFP4
        )
    ):
        scale0_byte, recip0, scale1_byte, recip1 = _compute_fp8_e8m0_scale_rceil_x2(
            max_abs0,
            max_abs1,
            _target_max_recip_for_format_id(format_id),
        )
        return scale0_byte, recip0, scale1_byte, recip1

    scale0_byte, recip0 = _compute_scale_and_recip_from_amax(
        max_abs0,
        format_id,
        half_range_scale,
    )
    scale1_byte, recip1 = _compute_scale_and_recip_from_amax(
        max_abs1,
        format_id,
        half_range_scale,
    )
    return scale0_byte, recip0, scale1_byte, recip1


@cute.jit
def _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
    max_abs: Float32,
    format_id: cutlass.Constexpr[int],
    half_range_scale: cutlass.Constexpr[bool],
    mNvfp4RecipLut: cute.Tensor,
    use_nvfp4_no_clip_scale: cutlass.Constexpr[bool] = False,
):
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_NVFP4):
        scale_byte = _compute_nvfp4_scale_byte(
            max_abs,
            use_nvfp4_no_clip_scale,
        )
        return scale_byte, Float32(mNvfp4RecipLut[Int32(scale_byte)])
    return _compute_scale_and_recip_from_amax(
        max_abs,
        format_id,
        half_range_scale,
        use_nvfp4_no_clip_scale,
    )


@cute.jit
def _clamp_to_target_max(
    x: Float32,
    recip_scale: Float32,
    target_max: Float32,
) -> Float32:
    clamped = cute.arch.fmax(x * recip_scale, -target_max)
    return target_max if clamped > target_max else clamped


@cute.jit
def _compute_block_amax_nonfinite(x: Float32) -> Float32:
    abs_v = _abs_f32(x)
    return Float32(float("inf")) if abs_v != abs_v else abs_v


@dsl_user_op
def _fmax_nan(
    a: Float32, b: Float32, c: Optional[Float32] = None, *, loc=None, ip=None
) -> Float32:
    """NaN-propagating max (PTX ``max.NaN.f32``, 2- or 3-input on SM100): NaN
    in any input yields NaN, unlike plain ``max.f32`` which drops it. Lets
    amax chains carry a NaN straight to the E8M0 cvt (NaN -> 0xFF scale byte)
    without the per-element ``NaN -> inf`` substitution of
    :func:`_compute_block_amax_nonfinite`."""
    return Float32(
        fmax(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            nan=True,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _reduce_amax_f32(
    amax: Float32,
    lane_stride: cutlass.Constexpr[int],
    reduce_stages: cutlass.Constexpr[int],
    nan_propagate: cutlass.Constexpr[bool] = False,
) -> Float32:
    op = _fmax_nan if cutlass.const_expr(nan_propagate) else cute.arch.fmax
    if cutlass.const_expr(lane_stride == 1):
        return cute.arch.warp_reduction(
            amax,
            op,
            threads_in_group=1 << reduce_stages,
        )

    for stage in cutlass.range_constexpr(reduce_stages):
        peer_amax = cute.arch.shuffle_sync_bfly(
            amax,
            offset=lane_stride * (1 << stage),
        )
        amax = op(amax, peer_amax)
    return amax


@cute.jit
def _warp_reduce_amax_f32(amax: Float32) -> Float32:
    reduced = llvm.inline_asm(
        Float32.mlir_type,
        [amax.ir_value()],
        "redux.sync.max.NaN.f32 $0, $1, 0xffffffff;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(reduced)


@cute.jit
def _frag_amax_nonfinite(
    vals: cute.Tensor,
    base: cutlass.Constexpr[int],
    count: cutlass.Constexpr[int],
) -> Float32:
    """Amax of ``|vals[base : base + count]|`` with the NaN -> inf convention.

    SM100 3-input ``max.NaN`` chains carry a NaN through the tree, and one
    :func:`_compute_block_amax_nonfinite` on the reduced value restores the
    per-element convention (any non-finite element -> +inf), so the result is
    bitwise identical to the per-element substitution at roughly one |x| plus
    half a max per element instead of abs + NaN-test + select + max."""
    return _compute_block_amax_nonfinite(_frag_amax_nan(vals, base, count))


@cute.jit
def _frag_amax_nan(
    vals: cute.Tensor,
    base: cutlass.Constexpr[int],
    count: cutlass.Constexpr[int],
) -> Float32:
    """NaN-carrying amax of ``|vals[base : base + count]|`` (3-input
    ``max.NaN`` tree; a NaN element survives to the caller)."""
    amax = _abs_f32(vals[base])
    for p in cutlass.range_constexpr((count - 1) // 2):
        amax = _fmax_nan(
            amax,
            _abs_f32(vals[base + 1 + 2 * p]),
            _abs_f32(vals[base + 2 + 2 * p]),
        )
    if cutlass.const_expr(count % 2 == 0):
        amax = _fmax_nan(amax, _abs_f32(vals[base + count - 1]))
    return amax


# -----------------------------------------------------------------------------
# Packed lane math helpers.
# -----------------------------------------------------------------------------


@dsl_user_op
def _minmax_b16x2(
    lhs: Int32,
    rhs: Int32,
    source_dtype: type[cutlass.Numeric],
    op_name: str,
    *,
    loc=None,
    ip=None,
) -> Int32:
    assert source_dtype in [BFloat16, Float16], "packed b16x2 op requires BF16 or FP16"
    assert op_name in ("max", "min"), "packed b16x2 op must be max or min"
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(lhs).ir_value(loc=loc, ip=ip),
                Int32(rhs).ir_value(loc=loc, ip=ip),
            ],
            f"{op_name}.{'bf16x2' if source_dtype is BFloat16 else 'f16x2'} $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _max_b16x2(
    lhs: Int32,
    rhs: Int32,
    source_dtype: type[cutlass.Numeric],
    *,
    loc=None,
    ip=None,
) -> Int32:
    return _minmax_b16x2(lhs, rhs, source_dtype, "max", loc=loc, ip=ip)


@dsl_user_op
def _min_b16x2(
    lhs: Int32,
    rhs: Int32,
    source_dtype: type[cutlass.Numeric],
    *,
    loc=None,
    ip=None,
) -> Int32:
    return _minmax_b16x2(lhs, rhs, source_dtype, "min", loc=loc, ip=ip)


@cute.jit
def _compute_amax_nonfinite_b16x2(
    packed: Int32,
    source_dtype: type[cutlass.Numeric],
) -> Int32:
    if cutlass.const_expr(source_dtype is BFloat16):
        inf_packed = Int32(0x7F807F80)
    elif cutlass.const_expr(source_dtype is Float16):
        inf_packed = Int32(0x7C007C00)
    else:
        raise TypeError("packed b16x2 amax requires BF16 or FP16")
    abs_packed = packed & Int32(0x7FFF7FFF)
    return _min_b16x2(abs_packed, inf_packed, source_dtype)


@cute.jit
def _compute_row_amax_b16x2_x16(
    vals_packed,
    rep: cutlass.Constexpr[int],
    source_dtype: type[cutlass.Numeric],
):
    row_amax_packed = Int32(0)
    for pair in cutlass.range_constexpr(8):
        row_amax_packed = _max_b16x2(
            row_amax_packed,
            _compute_amax_nonfinite_b16x2(vals_packed[rep, pair], source_dtype),
            source_dtype,
        )
    row_amax0, row_amax1 = _unpack_b16x2(row_amax_packed, source_dtype)
    return cute.arch.fmax(Float32(row_amax0), Float32(row_amax1))


@cute.jit
def _reduce_nvfp4_block_amaxes(
    block_amaxes,
    blocks_per_lane: cutlass.Constexpr[int],
) -> Float32:
    row_amax = Float32(_FP32_ZERO)
    for block_pair in cutlass.range_constexpr(blocks_per_lane // 2):
        row_amax = _fmax_nan(
            row_amax,
            block_amaxes[2 * block_pair],
            block_amaxes[2 * block_pair + 1],
        )
    if cutlass.const_expr(blocks_per_lane % 2 != 0):
        row_amax = cute.arch.fmax(
            row_amax,
            block_amaxes[blocks_per_lane - 1],
        )
    return row_amax


@cute.jit
def _reduce_amax_b16x2(
    packed_amax: Int32,
    lane_stride: cutlass.Constexpr[int],
    reduce_stages: cutlass.Constexpr[int],
    source_dtype: type[cutlass.Numeric],
):
    if cutlass.const_expr(source_dtype != BFloat16 and source_dtype != Float16):
        raise TypeError("packed b16x2 reduction requires BF16 or FP16")
    if cutlass.const_expr(lane_stride == 1):
        packed_amax = cute.arch.warp_reduction(
            packed_amax,
            lambda a, b: _max_b16x2(a, b, source_dtype),
            threads_in_group=1 << reduce_stages,
        )
    else:
        for stage in cutlass.range_constexpr(reduce_stages):
            peer_packed_amax = cute.arch.shuffle_sync_bfly(
                packed_amax,
                offset=lane_stride * (1 << stage),
            )
            packed_amax = _max_b16x2(
                packed_amax,
                peer_packed_amax,
                source_dtype,
            )
    return _unpack_b16x2(
        packed_amax,
        source_dtype,
    )


@dsl_user_op
def _rcp_rn_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.rn.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@cute.jit
def _compute_nvfp4_token_scales(row_amax: Float32):
    # gs convention: numerator * recip(amax) — bitwise-matching the fake-quant
    # synthesis and the Triton transport kernels; the wire token_scale_inv is
    # the correctly-rounded inverse of the scale. Returns (row_scale,
    # token_scale_inv), the same order as the Triton
    # `nvfp4_token_scale_and_recip` — the two outputs have reciprocal
    # magnitudes, so a mismatched unpack would run silently.
    row_scale = Float32(_NVFP4_TOKEN_SCALE_NUMERATOR) * _rcp_rn_f32(
        cute.arch.fmax(row_amax, Float32(_NVFP4_TOKEN_AMAX_FLOOR))
    )
    return row_scale, _rcp_rn_f32(row_scale)


@cute.jit
def _scale_nvfp4_per_token_block(
    packed,
    rep: cutlass.Constexpr[int],
    block_amax: Float32,
    row_scale: Float32,
    mNvfp4RecipLut: cute.Tensor,
    source_dtype: type[cutlass.Numeric],
    values: cute.Tensor,
    use_nvfp4_no_clip_scale: cutlass.Constexpr[bool],
) -> Uint8:
    scale_byte, multiplier = _compute_nvfp4_per_token_scale_and_multiplier(
        block_amax,
        row_scale,
        mNvfp4RecipLut,
        use_nvfp4_no_clip_scale,
    )
    for pair in cutlass.range_constexpr(8):
        value0, value1 = _unpack_b16x2(packed[rep, pair], source_dtype)
        values[2 * pair], values[2 * pair + 1] = cute.arch.mul_packed_f32x2(
            (Float32(value0), Float32(value1)),
            (multiplier, multiplier),
            rnd="rn",
        )
    return scale_byte


@dsl_user_op
def _fma_rn_f32(
    a: Float32,
    b: Float32,
    c: Float32,
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


@cute.jit
def _div_rn_f32_via_rcp_mul(
    numerator: Float32,
    denominator: Float32,
    denominator_recip: Float32,
) -> Float32:
    """Match ``div.rn.f32`` with reciprocal-multiply plus FMA correction."""
    quotient = numerator * denominator_recip
    remainder = _fma_rn_f32(-quotient, denominator, numerator)
    return _fma_rn_f32(remainder, denominator_recip, quotient)


@cute.jit
def _sigmoid_f32(
    x: Float32,
    fast_math: cutlass.Constexpr[bool] = False,
) -> Float32:
    if cutlass.const_expr(fast_math):
        return cute.arch.rcp_approx(
            Float32(_FP32_ONE) + cute_math.exp(-x, fastmath=True)
        )
    return _rcp_rn_f32(Float32(_FP32_ONE) + cute_math.exp(-x, fastmath=False))


@cute.jit
def _mul_f32x2(a, b):
    return cute.arch.mul_packed_f32x2(
        a,
        b,
        rnd="rn",
    )


@cute.jit
def _add_f32x2(a, b):
    return cute.arch.add_packed_f32x2(
        a,
        b,
        rnd="rn",
    )


@cute.jit
def _fma_f32x2(a, b, c):
    return cute.arch.fma_packed_f32x2(
        a,
        b,
        c,
        rnd="rn",
    )


# -----------------------------------------------------------------------------
# Producer math helpers.
# -----------------------------------------------------------------------------


@dsl_user_op
def _fmin_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`min.f32`: NaN-ignoring, the same PTX `tl.minimum` lowers to."""
    return Float32(
        fmin(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            nan=False,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _swiglu_clamped_preacts_f32(
    x: Float32,
    y: Float32,
    limit: cutlass.Constexpr[float],
):
    g = _fmin_f32(x, Float32(limit))
    u = _fmin_f32(cute.arch.fmax(y, Float32(-limit)), Float32(limit))
    return g, u


@cute.jit
def _swiglu_clamped_fwd_f32(
    x: Float32,
    y: Float32,
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    fast_math: cutlass.Constexpr[bool],
) -> Float32:
    """GPT-OSS clamped SwiGLU, op for op the Triton `swiglu_fwd` kernel."""
    g, u = _swiglu_clamped_preacts_f32(x, y, limit)
    s = _sigmoid_f32(Float32(alpha) * g, fast_math)
    return g * s * (u + Float32(_FP32_ONE))


@cute.jit
def _swiglu_clamped_bwd_f32(
    dz: Float32,
    x: Float32,
    y: Float32,
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    fast_math: cutlass.Constexpr[bool],
):
    """Clamped SwiGLU backward; returns `(dx, dy, h2)`.

    Op for op the Triton `swiglu_bwd` kernel. The grad is zeroed only strictly
    outside the clamp, and `s + alpha*g*s*(1-s)` uses the FMA Triton emits, so
    the two backends stay bitwise equal.
    """
    g, u = _swiglu_clamped_preacts_f32(x, y, limit)
    s = _sigmoid_f32(Float32(alpha) * g, fast_math)
    silu = g * s
    u_plus_one = u + Float32(_FP32_ONE)
    d_silu = _fma_rn_f32(Float32(alpha) * g * s, Float32(_FP32_ONE) - s, s)
    dx = dz * u_plus_one * d_silu
    dx = Float32(_FP32_ZERO) if x > Float32(limit) else dx
    dy = dz * silu
    dy = Float32(_FP32_ZERO) if y > Float32(limit) else dy
    dy = Float32(_FP32_ZERO) if y < Float32(-limit) else dy
    return dx, dy, silu * u_plus_one


@cute.jit
def _apply_block_scaled_quant_producer(
    primary: Float32,
    producer_b: Float32,
    producer_id: cutlass.Constexpr[int],
    fast_math: cutlass.Constexpr[bool] = False,
    clamped: cutlass.Constexpr[bool] = False,
    alpha: cutlass.Constexpr[float] = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: cutlass.Constexpr[float] = SWIGLU_CLAMP_LIMIT_DEFAULT,
) -> Float32:
    if cutlass.const_expr(producer_id == BLOCK_SCALED_PRODUCER_IDENTITY):
        return primary
    if cutlass.const_expr(producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD):
        if cutlass.const_expr(clamped):
            return _swiglu_clamped_fwd_f32(primary, producer_b, alpha, limit, fast_math)
        return primary * _sigmoid_f32(primary, fast_math) * producer_b
    return primary


@cute.jit
def _apply_block_scaled_quant_dxy_values(
    dz: Float32,
    x: Float32,
    y: Float32,
    fast_math: cutlass.Constexpr[bool] = False,
):
    sigmoid_x = _sigmoid_f32(x, fast_math)
    x_sigmoid = x * sigmoid_x
    dx = dz * y * sigmoid_x * (Float32(_FP32_ONE) + x - x_sigmoid)
    dy = dz * x * sigmoid_x
    return dx, dy, x_sigmoid


@cute.jit
def _apply_block_scaled_quant_dxy_producer(
    dz: Float32,
    x: Float32,
    y: Float32,
    fast_math: cutlass.Constexpr[bool] = False,
    clamped: cutlass.Constexpr[bool] = False,
    alpha: cutlass.Constexpr[float] = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: cutlass.Constexpr[float] = SWIGLU_CLAMP_LIMIT_DEFAULT,
):
    if cutlass.const_expr(clamped):
        dx, dy, _ = _swiglu_clamped_bwd_f32(dz, x, y, alpha, limit, fast_math)
        return dx, dy
    dx, dy, _ = _apply_block_scaled_quant_dxy_values(dz, x, y, fast_math)
    return dx, dy


@cute.jit
def _apply_block_scaled_quant_dxy_fwd_producer(
    dz: Float32,
    x: Float32,
    y: Float32,
    fast_math: cutlass.Constexpr[bool] = False,
    clamped: cutlass.Constexpr[bool] = False,
    alpha: cutlass.Constexpr[float] = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: cutlass.Constexpr[float] = SWIGLU_CLAMP_LIMIT_DEFAULT,
):
    if cutlass.const_expr(clamped):
        return _swiglu_clamped_bwd_f32(dz, x, y, alpha, limit, fast_math)
    dx, dy, x_sigmoid = _apply_block_scaled_quant_dxy_values(dz, x, y, fast_math)
    return dx, dy, x_sigmoid * y


@cute.jit
def _swiglu_clamped_bwd_f32x2(
    dz_f32x2,
    x_f32x2,
    y_f32x2,
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    fast_math: cutlass.Constexpr[bool],
):
    """Packed-lane twin of `_swiglu_clamped_bwd_f32`, same op order."""
    g0, u0 = _swiglu_clamped_preacts_f32(x_f32x2[0], y_f32x2[0], limit)
    g1, u1 = _swiglu_clamped_preacts_f32(x_f32x2[1], y_f32x2[1], limit)
    g_f32x2 = (g0, g1)
    alpha_g_f32x2 = _mul_f32x2((Float32(alpha), Float32(alpha)), g_f32x2)
    s_f32x2 = (
        _sigmoid_f32(alpha_g_f32x2[0], fast_math),
        _sigmoid_f32(alpha_g_f32x2[1], fast_math),
    )
    silu_f32x2 = _mul_f32x2(g_f32x2, s_f32x2)
    u_plus_one_f32x2 = _add_f32x2((u0, u1), (Float32(_FP32_ONE), Float32(_FP32_ONE)))
    one_minus_s_f32x2 = (
        Float32(_FP32_ONE) - s_f32x2[0],
        Float32(_FP32_ONE) - s_f32x2[1],
    )
    d_silu_f32x2 = _fma_f32x2(
        _mul_f32x2(alpha_g_f32x2, s_f32x2),
        one_minus_s_f32x2,
        s_f32x2,
    )
    dx_f32x2 = _mul_f32x2(_mul_f32x2(dz_f32x2, u_plus_one_f32x2), d_silu_f32x2)
    dy_f32x2 = _mul_f32x2(dz_f32x2, silu_f32x2)
    h2_f32x2 = _mul_f32x2(silu_f32x2, u_plus_one_f32x2)
    dx0 = Float32(_FP32_ZERO) if x_f32x2[0] > Float32(limit) else dx_f32x2[0]
    dx1 = Float32(_FP32_ZERO) if x_f32x2[1] > Float32(limit) else dx_f32x2[1]
    dy0 = Float32(_FP32_ZERO) if y_f32x2[0] > Float32(limit) else dy_f32x2[0]
    dy0 = Float32(_FP32_ZERO) if y_f32x2[0] < Float32(-limit) else dy0
    dy1 = Float32(_FP32_ZERO) if y_f32x2[1] > Float32(limit) else dy_f32x2[1]
    dy1 = Float32(_FP32_ZERO) if y_f32x2[1] < Float32(-limit) else dy1
    return (dx0, dx1), (dy0, dy1), h2_f32x2


@cute.jit
def _apply_block_scaled_quant_dxy_values_f32x2(
    dz_f32x2,
    x_f32x2,
    y_f32x2,
    fast_math: cutlass.Constexpr[bool] = False,
):
    sigmoid_f32x2 = (
        _sigmoid_f32(x_f32x2[0], fast_math),
        _sigmoid_f32(x_f32x2[1], fast_math),
    )
    x_sigmoid_f32x2 = _mul_f32x2(x_f32x2, sigmoid_f32x2)
    one_plus_x_f32x2 = _add_f32x2(
        (Float32(_FP32_ONE), Float32(_FP32_ONE)),
        x_f32x2,
    )
    dx_factor_f32x2 = _fma_f32x2(
        (-x_f32x2[0], -x_f32x2[1]),
        sigmoid_f32x2,
        one_plus_x_f32x2,
    )
    dz_y_f32x2 = _mul_f32x2(
        dz_f32x2,
        y_f32x2,
    )
    dx_f32x2 = _mul_f32x2(
        _mul_f32x2(
            dz_y_f32x2,
            sigmoid_f32x2,
        ),
        dx_factor_f32x2,
    )
    dz_x_f32x2 = _mul_f32x2(
        dz_f32x2,
        x_f32x2,
    )
    dy_f32x2 = _mul_f32x2(
        dz_x_f32x2,
        sigmoid_f32x2,
    )
    return dx_f32x2, dy_f32x2, x_sigmoid_f32x2


@cute.jit
def _apply_block_scaled_quant_dxy_fwd_producer_f32x2(
    dz_f32x2,
    x_f32x2,
    y_f32x2,
    fast_math: cutlass.Constexpr[bool] = False,
    clamped: cutlass.Constexpr[bool] = False,
    alpha: cutlass.Constexpr[float] = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: cutlass.Constexpr[float] = SWIGLU_CLAMP_LIMIT_DEFAULT,
):
    if cutlass.const_expr(clamped):
        return _swiglu_clamped_bwd_f32x2(
            dz_f32x2, x_f32x2, y_f32x2, alpha, limit, fast_math
        )
    dx_f32x2, dy_f32x2, x_sigmoid_f32x2 = _apply_block_scaled_quant_dxy_values_f32x2(
        dz_f32x2,
        x_f32x2,
        y_f32x2,
        fast_math,
    )
    h2_f32x2 = _mul_f32x2(x_sigmoid_f32x2, y_f32x2)
    return dx_f32x2, dy_f32x2, h2_f32x2


@cute.jit
def _apply_block_scaled_quant_dxy_producer_f32x2(
    dz_f32x2,
    x_f32x2,
    y_f32x2,
    fast_math: cutlass.Constexpr[bool] = False,
    clamped: cutlass.Constexpr[bool] = False,
    alpha: cutlass.Constexpr[float] = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: cutlass.Constexpr[float] = SWIGLU_CLAMP_LIMIT_DEFAULT,
):
    if cutlass.const_expr(clamped):
        dx_f32x2, dy_f32x2, _ = _swiglu_clamped_bwd_f32x2(
            dz_f32x2, x_f32x2, y_f32x2, alpha, limit, fast_math
        )
        return dx_f32x2, dy_f32x2
    dx_f32x2, dy_f32x2, _ = _apply_block_scaled_quant_dxy_values_f32x2(
        dz_f32x2,
        x_f32x2,
        y_f32x2,
        fast_math,
    )
    return dx_f32x2, dy_f32x2


# -----------------------------------------------------------------------------
# Layout helpers.
# -----------------------------------------------------------------------------


@cute.jit
def _cublas_blockscaled_qscale_offset(
    scale_row,
    scale_col,
    n_col_blocks,
):
    row_atom = scale_row // CUBLAS_BLOCKED_ROWS_PER_ATOM
    row_in_atom = scale_row - row_atom * CUBLAS_BLOCKED_ROWS_PER_ATOM
    col_atom = scale_col // CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    col_in_atom = scale_col - col_atom * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM

    atom_idx = row_atom * n_col_blocks + col_atom
    atom_stride = CUBLAS_BLOCKED_ROW_LANES * CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE
    atom_base = atom_idx * atom_stride

    row_group = row_in_atom // CUBLAS_BLOCKED_ROW_LANES
    row_in_group = row_in_atom - row_group * CUBLAS_BLOCKED_ROW_LANES
    row_group_offset = row_group * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    row_offset = row_in_group * CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE

    return atom_base + row_offset + row_group_offset + col_in_atom


# -----------------------------------------------------------------------------
# Load helpers.
# -----------------------------------------------------------------------------
@dsl_user_op
def _copy_load_f16xn_as_f32(
    mXWords,
    row,
    word_col,
    src_dtype: cutlass.Constexpr,
    num_elems: cutlass.Constexpr[int],
    *,
    loc=None,
    ip=None,
):
    if cutlass.const_expr(num_elems != 8 and num_elems != 16):
        raise ValueError("f16 load expects 8 or 16 elements")
    src_words = cute.tiled_divide(mXWords[row, None], (4,))
    dst = cute.make_rmem_tensor(4, Uint32, loc=loc, ip=ip)
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        Uint32,
        num_bits_per_copy=128,
        loc=loc,
        ip=ip,
    )
    cute.copy(copy_atom, src_words[None, word_col // 4], dst, loc=loc, ip=ip)
    x0, x1 = _cvt_f16x2_to_f32x2(dst[0], src_dtype, loc=loc, ip=ip)
    x2, x3 = _cvt_f16x2_to_f32x2(dst[1], src_dtype, loc=loc, ip=ip)
    x4, x5 = _cvt_f16x2_to_f32x2(dst[2], src_dtype, loc=loc, ip=ip)
    x6, x7 = _cvt_f16x2_to_f32x2(dst[3], src_dtype, loc=loc, ip=ip)
    if cutlass.const_expr(num_elems == 8):
        return x0, x1, x2, x3, x4, x5, x6, x7

    cute.copy(
        copy_atom,
        src_words[None, (word_col + 4) // 4],
        dst,
        loc=loc,
        ip=ip,
    )
    x8, x9 = _cvt_f16x2_to_f32x2(dst[0], src_dtype, loc=loc, ip=ip)
    x10, x11 = _cvt_f16x2_to_f32x2(dst[1], src_dtype, loc=loc, ip=ip)
    x12, x13 = _cvt_f16x2_to_f32x2(dst[2], src_dtype, loc=loc, ip=ip)
    x14, x15 = _cvt_f16x2_to_f32x2(dst[3], src_dtype, loc=loc, ip=ip)
    return x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15


@cute.jit
def _load_tensor_row_as_b16x2(
    mX: cute.Tensor,
    row: Int32,
    k_base: Int32,
    values: cute.Tensor,
    num_elems: cutlass.Constexpr[int],
    source_dtype: type[cutlass.Numeric],
) -> None:
    if cutlass.const_expr(source_dtype != BFloat16 and source_dtype != Float16):
        raise TypeError("packed tensor-row source loads require BF16 or FP16")
    if cutlass.const_expr(mX.element_type != source_dtype):
        raise TypeError("packed tensor-row source load dtype mismatch")
    if cutlass.const_expr(values.element_type != Int32):
        raise TypeError("packed tensor-row source load destination mismatch")
    if cutlass.const_expr(num_elems % 2 != 0):
        raise ValueError("packed b16x2 source loads require even num_elems")
    if cutlass.const_expr(
        num_elems != 4 and num_elems != 8 and num_elems != 16 and num_elems != 32
    ):
        raise ValueError("packed tensor-row source load expects 4, 8, 16, or 32 elems")

    row_i64 = Int64(row)
    if cutlass.const_expr(num_elems == 4):
        # Dynamic SwiGLU row starts guarantee 64-bit, not 128-bit, alignment;
        # CopyUniversal rejects a wider atom when the row stride is symbolic.
        ptr = cute.recast_ptr(
            mX.iterator + cute.crd2idx((row_i64, k_base), mX.layout),
            dtype=Uint64,
        )
        packed = cute.arch.load(
            ptr,
            Uint64,
            cop="cs",
        )
        values[0], values[1] = _unpack_b64_to_i32x2(packed)
        return

    chunk_elems: cutlass.Constexpr[int] = 128 // source_dtype.width
    if cutlass.const_expr(num_elems % chunk_elems != 0):
        raise ValueError("packed tensor-row lane elems must divide 128-bit loads")
    src_tiles = cute.tiled_divide(mX[row_i64, None], (chunk_elems,))
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        source_dtype,
        num_bits_per_copy=128,
    )
    for chunk in cutlass.range_constexpr(num_elems // chunk_elems):
        loaded = cute.make_rmem_tensor(chunk_elems, source_dtype)
        cute.copy(
            copy_atom,
            src_tiles[None, k_base // Int32(chunk_elems) + chunk],
            loaded,
        )
        for pair in cutlass.range_constexpr(chunk_elems // 2):
            values[chunk * (chunk_elems // 2) + pair] = _pack_b16x2(
                loaded[2 * pair],
                loaded[2 * pair + 1],
                source_dtype,
            )


@cute.jit
def _load_gather_row_as_b16x2(
    row_addr: Int64,
    source_cols,
    k_base,
    values: cute.Tensor,
    num_elems: cutlass.Constexpr[int],
    source_dtype: type[cutlass.Numeric],
) -> None:
    if cutlass.const_expr(source_dtype != BFloat16 and source_dtype != Float16):
        raise TypeError("packed gather-row source loads require BF16 or FP16")
    if cutlass.const_expr(values.element_type != Int32):
        raise TypeError("packed gather-row source load destination mismatch")
    if cutlass.const_expr(num_elems % 2 != 0):
        raise ValueError("packed gather-row source loads require even num_elems")

    pair_elems: cutlass.Constexpr[int] = num_elems // 2
    chunk_pairs: cutlass.Constexpr[int] = 128 // Int32.width
    if cutlass.const_expr(pair_elems == 2):
        chunk_pairs = 2
    if cutlass.const_expr(pair_elems % chunk_pairs != 0):
        raise ValueError("packed lane pairs must divide vector load width")

    values.fill(Int32(0))
    if row_addr != Int64(0):
        # Gather rows are aligned to packed b16x2 pairs; full tiles use 128-bit
        # copies and 4-element subtiles use one 64-bit copy.
        row_ptr = cute.make_ptr(
            Int32,
            row_addr,
            mem_space=cute.AddressSpace.gmem,
            assumed_align=128,
        )
        row_tensor = cute.make_tensor(
            row_ptr,
            cute.make_ordered_layout((source_cols // Int32(2),), order=(0,)),
        )
        row_chunks = cute.tiled_divide(row_tensor, (chunk_pairs,))
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            Int32,
            num_bits_per_copy=Int32.width * chunk_pairs,
        )
        for chunk in cutlass.range_constexpr(pair_elems // chunk_pairs):
            loaded = cute.make_rmem_tensor(chunk_pairs, Int32)
            cute.copy(
                copy_atom,
                row_chunks[None, k_base // Int32(2 * chunk_pairs) + chunk],
                loaded,
            )
            for i in cutlass.range_constexpr(chunk_pairs):
                values[chunk * chunk_pairs + i] = loaded[i]


# -----------------------------------------------------------------------------
# Store helpers.
# -----------------------------------------------------------------------------
@cute.jit
def _copy_store_u32_words(
    mQWords,
    row,
    q_word_col,
    values,
    num_words: cutlass.Constexpr[int],
    value_offset: cutlass.Constexpr[int] = 0,
):
    # Callers must align q_word_col to num_words; assumed_align encodes this contract.
    dst_ptr = cute.make_ptr(
        dtype=Uint32,
        value=(
            mQWords.iterator + cute.crd2idx((row, q_word_col), mQWords.layout)
        ).toint(),
        mem_space=mQWords.iterator.memspace,
        assumed_align=4 * num_words,
    )
    dst_words = cute.make_tensor(dst_ptr, cute.make_layout((num_words,)))
    src = cute.make_rmem_tensor(num_words, Uint32)
    for i in cutlass.range_constexpr(num_words):
        src[i] = values[value_offset + i]
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        Uint32,
        num_bits_per_copy=32 * num_words,
    )
    cute.copy(copy_atom, src, dst_words)


@cute.jit
def _cvt_fp4_pair(
    q_lo: Float32,
    q_hi: Float32,
) -> Uint8:
    """Pack an FP4 (E2M1) element pair with round-to-nearest conversion."""
    return _cvt_f32x2_to_qdata_byte_rn(
        q_lo,
        q_hi,
        dst_kind=nvvm.CVTPackFloatKind.E2M1x2,
    )


@cute.jit
def _store_qdata_row(
    mQ: cute.Tensor,
    vals,
    recip: Float32,
    row,
    col_start,
    sf_vec_size: cutlass.Constexpr[int],
    is_fp4: cutlass.Constexpr[bool],
    format_id: cutlass.Constexpr[int],
):
    if cutlass.const_expr(is_fp4):
        for k in cutlass.range_constexpr(sf_vec_size // 2):
            q_lo = vals[2 * k] * recip
            q_hi = vals[2 * k + 1] * recip
            q_byte_col = (col_start + 2 * k) // 2
            q_byte = _cvt_fp4_pair(q_lo, q_hi)
            mQ[row, q_byte_col] = q_byte
    else:
        for k in cutlass.range_constexpr(sf_vec_size):
            q = vals[k] * recip
            q_col = col_start + k
            mQ[row, q_col] = _cvt_qdata_for_format_id(q, format_id)


@cute.jit
def _store_qdata_col_major_fp4(
    mQ: cute.Tensor,
    q_codes,
    tidx,
    row_start,
    col_start,
    sf_vec_size: cutlass.Constexpr[int],
):
    row_pair = (row_start // 2) + (tidx // 2)
    is_even = (tidx % Int32(2)) == Int32(0)
    for k in cutlass.range_constexpr(sf_vec_size):
        q_col = col_start + k
        partner_code = cute.arch.shuffle_sync_bfly(q_codes[k], offset=1)
        if is_even:
            q_lo = q_codes[k]
            q_hi = partner_code
            byte = _cvt_fp4_pair(q_lo, q_hi)
            mQ[q_col, row_pair] = byte


@cute.jit
def _store_qdata_col_major_fp8(
    mQ: cute.Tensor,
    q_codes,
    tidx,
    row_start,
    col_start,
    sf_vec_size: cutlass.Constexpr[int],
    format_id: cutlass.Constexpr[int],
):
    row = row_start + tidx
    for k in cutlass.range_constexpr(sf_vec_size):
        q_col = col_start + k
        mQ[q_col, row] = _cvt_qdata_for_format_id(q_codes[k], format_id)


@cute.jit
def _copy_store_u32_words_chunked(
    mQWords: cute.Tensor,
    row,
    q_word_col,
    values,
    num_words: cutlass.Constexpr[int],
) -> None:
    if cutlass.const_expr(num_words != 1 and num_words != 2 and num_words % 4 != 0):
        raise ValueError("unsupported packed u32 store width")
    if cutlass.const_expr(num_words == 1):
        _copy_store_u32_words(mQWords, row, q_word_col, values, 1)
    elif cutlass.const_expr(num_words == 2):
        _copy_store_u32_words(mQWords, row, q_word_col, values, 2)
    else:
        for word_quad in cutlass.range_constexpr(num_words // 4):
            _copy_store_u32_words(
                mQWords,
                row,
                q_word_col + Int32(4 * word_quad),
                values,
                4,
                4 * word_quad,
            )


@cute.jit
def _store_blockscaled_xn_words(
    mQWords: cute.Tensor,
    row,
    q_word_col,
    vals,
    *,
    recip: Float32,
    num_elems: cutlass.Constexpr[int],
    is_fp4: cutlass.Constexpr[bool],
    format_id: cutlass.Constexpr[int],
    scale_values: cutlass.Constexpr[bool],
) -> None:
    if cutlass.const_expr(is_fp4):
        num_words: cutlass.Constexpr[int] = num_elems // 8
        packed = cute.make_rmem_tensor(num_words, Uint32)
        _pack_fp4_e2m1_xn_to_u32_rn(
            vals,
            packed,
            recip=recip,
            base=0,
            num_elems=num_elems,
            scale_values=scale_values,
        )
        _copy_store_u32_words_chunked(mQWords, row, q_word_col, packed, num_words)
    else:
        num_words: cutlass.Constexpr[int] = num_elems // 4
        packed = cute.make_rmem_tensor(num_words, Uint32)
        _pack_fp8_xn_to_u32(
            vals,
            packed,
            recip=recip,
            base=0,
            num_elems=num_elems,
            scale_values=scale_values,
            format_id=format_id,
        )
        _copy_store_u32_words_chunked(mQWords, row, q_word_col, packed, num_words)


@cute.jit
def _store_fp8_qwords(  # noqa: C901
    mQWords: cute.Tensor,
    q_vals: cute.Tensor,
    row: Int32,
    q_word_col: Int32,
    num_elems: cutlass.Constexpr[int],
    format_id: cutlass.Constexpr[int],
    qdata_elems_per_word: cutlass.Constexpr[int] = 4,
    cache_modifier: cutlass.Constexpr = None,
    packed_f32x2: cutlass.Constexpr[bool] = False,
) -> None:
    if cutlass.const_expr(mQWords.element_type not in (Uint32, Int32)):
        raise TypeError("qword stores require a 32-bit integer tensor")
    if cutlass.const_expr(
        format_id != BLOCK_SCALED_FORMAT_MXFP8_E4M3
        and format_id != BLOCK_SCALED_FORMAT_MXFP8_E5M2
    ):
        raise TypeError("qword stores require an FP8 format")
    if cutlass.const_expr(packed_f32x2 and q_vals.element_type != Uint64):
        raise TypeError("packed f32x2 qword stores require Uint64 values")
    if cutlass.const_expr(not packed_f32x2 and q_vals.element_type != Float32):
        raise TypeError("qword stores require pre-scaled FP32 values")
    if cutlass.const_expr(num_elems % qdata_elems_per_word != 0):
        raise ValueError("qword lane elems must divide qdata word width")
    if cutlass.const_expr(packed_f32x2 and num_elems != qdata_elems_per_word):
        raise ValueError("packed f32x2 qword stores expect 4 elems")
    num_q_words: cutlass.Constexpr[int] = num_elems // qdata_elems_per_word
    if cutlass.const_expr(
        num_q_words != 1 and num_q_words != 2 and num_q_words % 4 != 0
    ):
        raise ValueError("unsupported qword store width")
    q_words = cute.make_rmem_tensor(num_q_words, Uint32)

    if cutlass.const_expr(packed_f32x2):
        q_words[0] = _pack_packed_f32x2x2_to_fp8x4_u32_rn(
            q_vals[0],
            q_vals[1],
            format_id=format_id,
        )
    else:
        for word in cutlass.range_constexpr(num_q_words):
            if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2):
                _pack_fp8_xn_to_u32_rn(
                    q_vals,
                    q_words,
                    base=word * qdata_elems_per_word,
                    packed_base=word,
                    num_elems=qdata_elems_per_word,
                    recip=Float32(1.0),
                    scale_values=False,
                    dst_kind=nvvm.CVTPackFloatKind.E5M2x2,
                )
            else:
                _pack_fp8_xn_to_u32_rn(
                    q_vals,
                    q_words,
                    base=word * qdata_elems_per_word,
                    packed_base=word,
                    num_elems=qdata_elems_per_word,
                    recip=Float32(1.0),
                    scale_values=False,
                    dst_kind=nvvm.CVTPackFloatKind.E4M3x2,
                )

    if cutlass.const_expr(num_q_words == 1):
        if cutlass.const_expr(
            cache_modifier is None and mQWords.element_type == Uint32
        ):
            dst_words = cute.tiled_divide(mQWords[row, None], (1,))
            src = cute.make_rmem_tensor(1, Uint32)
            src[0] = q_words[0]
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Uint32,
                num_bits_per_copy=32,
            )
            cute.copy(copy_atom, src, dst_words[None, q_word_col])
        else:
            ptr = cute.recast_ptr(
                mQWords.iterator + cute.crd2idx((row, q_word_col), mQWords.layout),
                dtype=Uint32,
            )
            if cutlass.const_expr(cache_modifier is None):
                cute.arch.store(ptr, q_words[0])
            else:
                cute.arch.store(ptr, q_words[0], cop=cache_modifier)
    elif cutlass.const_expr(num_q_words == 2):
        if cutlass.const_expr(
            cache_modifier is None and mQWords.element_type == Uint32
        ):
            _copy_store_u32_words(mQWords, row, q_word_col, q_words, 2)
        else:
            ptr = cute.recast_ptr(
                mQWords.iterator + cute.crd2idx((row, q_word_col), mQWords.layout),
                dtype=Uint32,
            )
            value = make_vector(Uint32, q_words[0], q_words[1])
            if cutlass.const_expr(cache_modifier is None):
                cute.arch.store(ptr, value)
            else:
                cute.arch.store(ptr, value, cop=cache_modifier)
    else:
        for word_quad in cutlass.range_constexpr(num_q_words // 4):
            q_word_offset = q_word_col + Int32(4 * word_quad)
            if cutlass.const_expr(
                cache_modifier is None and mQWords.element_type == Uint32
            ):
                _copy_store_u32_words(
                    mQWords,
                    row,
                    q_word_offset,
                    q_words,
                    4,
                    4 * word_quad,
                )
            else:
                ptr = cute.recast_ptr(
                    mQWords.iterator
                    + cute.crd2idx((row, q_word_offset), mQWords.layout),
                    dtype=Uint32,
                )
                value = make_vector(
                    Uint32,
                    q_words[4 * word_quad],
                    q_words[4 * word_quad + 1],
                    q_words[4 * word_quad + 2],
                    q_words[4 * word_quad + 3],
                )
                if cutlass.const_expr(cache_modifier is None):
                    cute.arch.store(ptr, value)
                else:
                    cute.arch.store(ptr, value, cop=cache_modifier)


@cute.jit
def _store_qdata_axis0_words(
    mQWords: cute.Tensor,
    vals,
    recip: Float32,
    col,
    row_start,
    sf_vec_size: cutlass.Constexpr[int],
    is_fp4: cutlass.Constexpr[bool],
    format_id: cutlass.Constexpr[int],
):
    flat_col = col
    if cutlass.const_expr(is_fp4):
        q_word_row = row_start // 8
        _store_blockscaled_xn_words(
            mQWords,
            flat_col,
            q_word_row,
            vals,
            recip=recip,
            num_elems=sf_vec_size,
            is_fp4=is_fp4,
            format_id=format_id,
            scale_values=True,
        )
    else:
        q_word_row = row_start // 4
        _store_blockscaled_xn_words(
            mQWords,
            flat_col,
            q_word_row,
            vals,
            recip=recip,
            num_elems=sf_vec_size,
            is_fp4=is_fp4,
            format_id=format_id,
            scale_values=True,
        )


@cute.jit
def _store_qdata_row_words(  # noqa: C901
    mQWords: cute.Tensor,
    vals,
    recip: Float32,
    flat_row,
    scale_col,
    lane,
    axes_kernel_mode: cutlass.Constexpr[int],
    is_fp4: cutlass.Constexpr[bool],
    sf_vec_size: cutlass.Constexpr[int],
    num_elems: cutlass.Constexpr[int],
    format_id: cutlass.Constexpr[int],
):
    if cutlass.const_expr(
        axes_kernel_mode == _AXES_KERNEL_MODE_TILE2D_FP8_VECTOR
        or (axes_kernel_mode == _AXES_KERNEL_MODE_BOTH1D_VECTOR and not is_fp4)
        or (
            axes_kernel_mode == _AXES_KERNEL_MODE_SWIGLU_BWD_DXY_F32X2_VECTOR
            and not is_fp4
        )
    ):
        q_word_col = scale_col * (sf_vec_size // 4)
        _store_blockscaled_xn_words(
            mQWords,
            flat_row,
            q_word_col,
            vals,
            recip=recip,
            num_elems=sf_vec_size,
            is_fp4=is_fp4,
            format_id=format_id,
            scale_values=True,
        )
    elif cutlass.const_expr(
        (
            axes_kernel_mode == _AXES_KERNEL_MODE_BOTH1D_VECTOR
            or axes_kernel_mode == _AXES_KERNEL_MODE_SWIGLU_BWD_DXY_F32X2_VECTOR
            or axes_kernel_mode == _AXES_KERNEL_MODE_TILE2D_FP4_VECTOR
        )
        and is_fp4
    ):
        q_word_col = scale_col * (sf_vec_size // 8)
        _store_blockscaled_xn_words(
            mQWords,
            flat_row,
            q_word_col,
            vals,
            recip=recip,
            num_elems=sf_vec_size,
            is_fp4=is_fp4,
            format_id=format_id,
            scale_values=True,
        )
    elif cutlass.const_expr(is_fp4):
        q_word_col = scale_col * (sf_vec_size // 8) + lane * (num_elems // 8)
        _store_blockscaled_xn_words(
            mQWords,
            flat_row,
            q_word_col,
            vals,
            recip=recip,
            num_elems=16,
            is_fp4=is_fp4,
            format_id=format_id,
            scale_values=True,
        )
    else:
        q_word_col = scale_col * (sf_vec_size // 4) + lane * (num_elems // 4)
        _store_blockscaled_xn_words(
            mQWords,
            flat_row,
            q_word_col,
            vals,
            recip=recip,
            num_elems=16,
            is_fp4=is_fp4,
            format_id=format_id,
            scale_values=True,
        )


# -----------------------------------------------------------------------------
# Generic tile quantization helpers.
# -----------------------------------------------------------------------------


@cute.jit
def _tile_scale_recip(
    scale_recips,
    rep: cutlass.Constexpr[int],
    elem_idx: cutlass.Constexpr[int],
    col_quant: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(col_quant):
        return scale_recips[elem_idx]
    return scale_recips[rep]


@cute.jit
def _store_fp4_tile_qdata(
    mQWords: cute.Tensor,
    q_vals: cute.Tensor,
    row: Int32,
    row_lane: Int32,
    q_word: Int32,
    num_elems: cutlass.Constexpr[int],
    reduce_lane_stride: cutlass.Constexpr[int],
    qdata_elems_per_word: cutlass.Constexpr[int],
    format_id: cutlass.Constexpr[int],
    col_quant: cutlass.Constexpr[bool],
) -> None:
    if cutlass.const_expr(col_quant):
        for i in cutlass.range_constexpr(num_elems):
            partner = cute.arch.shuffle_sync_bfly(q_vals[i], offset=reduce_lane_stride)
            if row_lane % Int32(2) == Int32(0):
                q_byte = _cvt_f32x2_to_qdata_byte_rn(
                    q_vals[i],
                    partner,
                    dst_kind=nvvm.CVTPackFloatKind.E2M1x2,
                )
                q_col = q_word * Int32(qdata_elems_per_word) + Int32(i)
                if cutlass.const_expr(num_elems < qdata_elems_per_word):
                    q_col += (
                        cute.arch.lane_idx() % Int32(qdata_elems_per_word // num_elems)
                    ) * Int32(num_elems)
                q_byte_row = row // Int32(2)
                q_word_row = q_byte_row // Int32(4)
                q_byte_in_word = q_byte_row % Int32(4)
                q_byte_ptr = cute.recast_ptr(
                    mQWords.iterator
                    + cute.crd2idx((q_col, q_word_row), mQWords.layout),
                    dtype=Uint8,
                )
                cute.arch.store(q_byte_ptr + q_byte_in_word, q_byte, cop="cs")
    else:
        if cutlass.const_expr(num_elems == 4):
            if cutlass.const_expr(reduce_lane_stride != 1):
                raise ValueError(
                    "row-quant FP4 lane packing requires reduce_lane_stride == 1"
                )
            packed_vals = cute.make_rmem_tensor(8, Float32)
            for i in cutlass.range_constexpr(4):
                packed_vals[i] = q_vals[i]
                packed_vals[4 + i] = cute.arch.shuffle_sync_bfly(q_vals[i], offset=1)
            if cute.arch.lane_idx() % Int32(2) == Int32(0):
                _store_blockscaled_xn_words(
                    mQWords,
                    row,
                    q_word,
                    packed_vals,
                    recip=Float32(1.0),
                    num_elems=8,
                    is_fp4=True,
                    format_id=format_id,
                    scale_values=False,
                )
        else:
            _store_blockscaled_xn_words(
                mQWords,
                row,
                q_word,
                q_vals,
                recip=Float32(1.0),
                num_elems=num_elems,
                is_fp4=True,
                format_id=format_id,
                scale_values=False,
            )


@cute.jit
def _fp8_tile_qword_values(
    vals: cute.Tensor,
    scale_recips: cute.Tensor,
    rep: cutlass.Constexpr[int],
    elem_idx: cutlass.Constexpr[int],
    col_quant: cutlass.Constexpr[bool],
    *,
    scale_values: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(not scale_values):
        return (
            vals[rep, elem_idx],
            vals[rep, elem_idx + 1],
            vals[rep, elem_idx + 2],
            vals[rep, elem_idx + 3],
        )
    return (
        vals[rep, elem_idx] * _tile_scale_recip(scale_recips, rep, elem_idx, col_quant),
        vals[rep, elem_idx + 1]
        * _tile_scale_recip(scale_recips, rep, elem_idx + 1, col_quant),
        vals[rep, elem_idx + 2]
        * _tile_scale_recip(scale_recips, rep, elem_idx + 2, col_quant),
        vals[rep, elem_idx + 3]
        * _tile_scale_recip(scale_recips, rep, elem_idx + 3, col_quant),
    )


@cute.jit
def _quantize_fp8_tile_values(  # noqa: C901
    vals: cute.Tensor,
    mQWords: cute.Tensor,
    mScale: cute.Tensor,
    scale_recips: cute.Tensor,
    q_vals: cute.Tensor,
    q_f32x2_packed: cute.Tensor,
    row_start: Int32,
    row_lane: Int32,
    q_word: Int32,
    scale_offset_base: Int32,
    scale_store_lane,
    row_lanes_cfg: cutlass.Constexpr[int],
    row_reps_cfg: cutlass.Constexpr[int],
    num_elems: cutlass.Constexpr[int],
    lane_pairs: cutlass.Constexpr[int],
    reduce_lane_stride: cutlass.Constexpr[int],
    reduce_stages: cutlass.Constexpr[int],
    scale_store_stride,
    col_quant: cutlass.Constexpr[bool],
    packed_f32x2: cutlass.Constexpr[bool],
    precomputed_amax: cutlass.Constexpr[bool],
    paired_scale_compute: cutlass.Constexpr[bool],
    format_id: cutlass.Constexpr[int],
    half_range_scale: cutlass.Constexpr[bool] = False,
    qdata_elems_per_word: cutlass.Constexpr[int] = 4,
    cache_modifier: cutlass.Constexpr = None,
) -> None:
    if cutlass.const_expr(col_quant and row_reps_cfg < num_elems):
        raise ValueError("col quant needs recip scratch space")

    is_fp4: cutlass.Constexpr[bool] = (
        format_id == BLOCK_SCALED_FORMAT_NVFP4 or format_id == BLOCK_SCALED_FORMAT_MXFP4
    )
    scale_count: cutlass.Constexpr[int] = num_elems if col_quant else row_reps_cfg
    if cutlass.const_expr(not precomputed_amax):
        # NaN-carrying max.NaN accumulation with one NaN -> inf substitution
        # per accumulator (instead of per element); bitwise identical, see
        # _frag_amax_nonfinite.
        scale_recips.fill(Float32(_FP32_ZERO))
        for rep in cutlass.range_constexpr(row_reps_cfg):
            for i in cutlass.range_constexpr(num_elems):
                if cutlass.const_expr(col_quant):
                    scale_recips[i] = _fmax_nan(scale_recips[i], _abs_f32(vals[rep, i]))
                else:
                    scale_recips[rep] = _fmax_nan(
                        scale_recips[rep], _abs_f32(vals[rep, i])
                    )
        for k in cutlass.range_constexpr(scale_count):
            scale_recips[k] = _compute_block_amax_nonfinite(scale_recips[k])

    scale_stride = Int32(scale_store_stride)
    if cutlass.const_expr(paired_scale_compute and scale_count % 2 == 0):
        for scale_pair in cutlass.range_constexpr(scale_count // 2):
            scale_idx0: cutlass.Constexpr[int] = scale_pair * 2
            scale_idx1: cutlass.Constexpr[int] = scale_idx0 + 1
            block_amax0 = _reduce_amax_f32(
                scale_recips[scale_idx0],
                reduce_lane_stride,
                reduce_stages,
            )
            block_amax1 = _reduce_amax_f32(
                scale_recips[scale_idx1],
                reduce_lane_stride,
                reduce_stages,
            )
            scale0_byte, recip0, scale1_byte, recip1 = (
                _compute_scale_and_recip_from_amax_x2(
                    block_amax0,
                    block_amax1,
                    format_id,
                    half_range_scale,
                )
            )
            scale_recips[scale_idx0] = recip0
            scale_recips[scale_idx1] = recip1
            if scale_store_lane:
                scale_offset0 = Int32(scale_idx0) * scale_stride
                scale_offset1 = Int32(scale_idx1) * scale_stride
                store_cache_modifier = "cs" if col_quant else cache_modifier
                if cutlass.const_expr(store_cache_modifier is not None):
                    scale_ptr = cute.recast_ptr(
                        mScale.iterator + scale_offset_base,
                        dtype=Uint8,
                    )
                    cute.arch.store(
                        scale_ptr + scale_offset0,
                        scale0_byte,
                        cop=store_cache_modifier,
                    )
                    cute.arch.store(
                        scale_ptr + scale_offset1,
                        scale1_byte,
                        cop=store_cache_modifier,
                    )
                else:
                    mScale[scale_offset_base + scale_offset0] = scale0_byte
                    mScale[scale_offset_base + scale_offset1] = scale1_byte
    else:
        for scale_idx in cutlass.range_constexpr(scale_count):
            block_amax = _reduce_amax_f32(
                scale_recips[scale_idx],
                reduce_lane_stride,
                reduce_stages,
            )
            scale_byte, scale_recip = _compute_scale_and_recip_from_amax(
                block_amax,
                format_id,
                half_range_scale,
            )
            scale_recips[scale_idx] = scale_recip
            if scale_store_lane:
                scale_offset = Int32(scale_idx) * scale_stride
                store_cache_modifier = "cs" if col_quant else cache_modifier
                if cutlass.const_expr(
                    store_cache_modifier is not None or mScale.element_type == Int8
                ):
                    scale_ptr = cute.recast_ptr(
                        mScale.iterator + scale_offset_base + scale_offset,
                        dtype=Uint8,
                    )
                    if cutlass.const_expr(store_cache_modifier is None):
                        cute.arch.store(scale_ptr, scale_byte)
                    else:
                        cute.arch.store(scale_ptr, scale_byte, cop=store_cache_modifier)
                else:
                    mScale[scale_offset_base + scale_offset] = scale_byte

    for rep in cutlass.range_constexpr(row_reps_cfg):
        row_rep_offset = Int32(rep * row_lanes_cfg)
        row = row_start + row_lane + row_rep_offset
        if cutlass.const_expr(packed_f32x2):
            for pair in cutlass.range_constexpr(lane_pairs):
                scale0 = _tile_scale_recip(scale_recips, rep, 2 * pair, col_quant)
                scale1 = _tile_scale_recip(scale_recips, rep, 2 * pair + 1, col_quant)
                q0, q1 = cute.arch.mul_packed_f32x2(
                    (vals[rep, 2 * pair], vals[rep, 2 * pair + 1]),
                    (scale0, scale1),
                    rnd="rn",
                )
                if cutlass.const_expr(is_fp4):
                    q_vals[2 * pair] = q0
                    q_vals[2 * pair + 1] = q1
                else:
                    q_f32x2_packed[pair] = _pack_f32x2_to_u64(q0, q1)
            if cutlass.const_expr(is_fp4):
                _store_fp4_tile_qdata(
                    mQWords,
                    q_vals,
                    row,
                    row_lane,
                    q_word,
                    num_elems,
                    reduce_lane_stride,
                    qdata_elems_per_word,
                    format_id,
                    col_quant,
                )
            else:
                store_cache_modifier = "cs" if col_quant else cache_modifier
                _store_fp8_qwords(
                    mQWords,
                    q_f32x2_packed,
                    row,
                    q_word,
                    num_elems,
                    format_id,
                    qdata_elems_per_word,
                    cache_modifier=store_cache_modifier,
                    packed_f32x2=True,
                )
        else:
            for i in cutlass.range_constexpr(num_elems):
                q_vals[i] = vals[rep, i] * _tile_scale_recip(
                    scale_recips, rep, i, col_quant
                )
            store_cache_modifier = "cs" if col_quant else cache_modifier
            if cutlass.const_expr(is_fp4):
                _store_fp4_tile_qdata(
                    mQWords,
                    q_vals,
                    row,
                    row_lane,
                    q_word,
                    num_elems,
                    reduce_lane_stride,
                    qdata_elems_per_word,
                    format_id,
                    col_quant,
                )
            else:
                _store_fp8_qwords(
                    mQWords,
                    q_vals,
                    row,
                    q_word,
                    num_elems,
                    format_id,
                    qdata_elems_per_word,
                    cache_modifier=store_cache_modifier,
                )
