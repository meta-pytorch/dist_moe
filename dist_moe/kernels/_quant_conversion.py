# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scalar numeric conversion and register-level pack/unpack primitives.

Bottom layer of the CuTe quant helper stack; format dispatch and multi-value
packing live in ``_quant_packing.py``.
"""

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import (
    BFloat16,
    Float8E8M0FNU,
    Float16,
    Float32,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import (
    arith as _arith,
    llvm,
    nvvm,
    vector,
)
from cutlass.cutlass_dsl import dsl_user_op, T

from ..formats import E8M0_EXP_BIAS

# Installs the sys.path shim that makes ``quack`` importable; kept here (the
# bottom module of the quant helper stack) so every transitive importer keeps
# the guarantee the monolithic ``common`` module used to provide.
from . import _dsl_compat as _cute_extern  # noqa: F401
from ._dsl_compat import (
    absf,
    cvt_packfloat,
    cvt_packfloat_f32,
    RoundingMode,
)

_FP32_ZERO: cutlass.Constexpr[float] = 0.0
_FP32_ONE: cutlass.Constexpr[float] = 1.0
_E8M0_NAN_BYTE: cutlass.Constexpr[int] = 255


# -----------------------------------------------------------------------------
# Scalar utilities.
# -----------------------------------------------------------------------------


@cute.jit
def _ceil_div_i32(numerator: Int32, denominator: Int32) -> Int32:
    return (numerator + denominator - Int32(1)) // denominator


@cute.jit
def _abs_f32(x: Float32) -> Float32:
    return absf(x)


# -----------------------------------------------------------------------------
# Pack and unpack helpers.
# -----------------------------------------------------------------------------
@dsl_user_op
def _pack_bits(dtype, *vs, loc=None, ip=None):
    """Pack scalar values into a wider scalar dtype by preserving lane bits.

    This is the inverse of ``_unpack_bits``: bitcast each lane to a same-width integer,
    build an integer vector, then bitcast the vector to the packed dtype. Avoid
    vector<...xfp8>; see https://github.com/NVIDIA/cutlass/issues/3342.
    """
    assert len(vs) > 0
    lane_type = type(vs[0])
    assert all(type(v) is lane_type for v in vs)
    assert len(vs) * lane_type.width == dtype.width
    lane_int_type = T.i(lane_type.width)
    return dtype(
        llvm.bitcast(
            dtype.mlir_type,
            vector.from_elements(
                T.vector(len(vs), lane_int_type),
                [
                    lane_type(v).ir_value(loc=loc, ip=ip)
                    if ir.IntegerType.isinstance(lane_type.mlir_type)
                    # arith (not llvm) bitcast: 4.6's LLVM path rejects i8<->fp8.
                    else _arith.bitcast(
                        lane_int_type,
                        lane_type(v).ir_value(loc=loc, ip=ip),
                        loc=loc,
                        ip=ip,
                    )
                    for v in vs
                ],
                loc=loc,
                ip=ip,
            ),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _unpack_bits(dtype, v, *, loc=None, ip=None):
    """Unpack a cutlass integral carrier into scalar dtype values.

    Keep the vector step in the integer domain and scalar-bitcast each lane at
    the end. CuTe bitcasts packed scalar registers directly to bf16/f16 vectors:
    https://github.com/NVIDIA/cutlass/blob/v4.5.2/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py#L1489-L1490
    We avoid vector<...xfp8> because it currently crashes this compiler path:
    https://github.com/NVIDIA/cutlass/issues/3342
    """
    carrier_type = type(v)
    assert ir.IntegerType.isinstance(carrier_type.mlir_type)
    assert carrier_type.width % dtype.width == 0
    vec = llvm.bitcast(
        T.vector(carrier_type.width // dtype.width, T.i(dtype.width)),
        carrier_type(v).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return tuple(
        dtype(
            # arith (not llvm) bitcast: 4.6's LLVM path rejects i8<->fp8.
            _arith.bitcast(
                dtype.mlir_type,
                vector.extract(
                    vec, dynamic_position=[], static_position=[i], loc=loc, ip=ip
                ),
                loc=loc,
                ip=ip,
            )
        )
        for i in range(carrier_type.width // dtype.width)
    )


@cute.jit
def _pack_b16x2(
    lo,
    hi,
    source_dtype: type[cutlass.Numeric],
) -> Int32:
    if cutlass.const_expr(source_dtype != BFloat16 and source_dtype != Float16):
        raise TypeError("packed b16x2 shuffle requires BF16 or FP16")
    pair = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), source_dtype)
    pair[0] = lo
    pair[1] = hi
    return cute.recast_tensor(pair, Int32)[0]


@cute.jit
def _unpack_b16x2(
    packed: Int32,
    source_dtype: type[cutlass.Numeric],
):
    if cutlass.const_expr(source_dtype != BFloat16 and source_dtype != Float16):
        raise TypeError("packed b16x2 shuffle requires BF16 or FP16")
    pair_bits = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), Int32)
    pair_bits[0] = packed
    pair = cute.recast_tensor(pair_bits, source_dtype)
    return pair[0], pair[1]


@cute.jit
def _pack_f32x2_to_u64(lo: Float32, hi: Float32) -> Uint64:
    vals = cute.make_rmem_tensor(
        cute.make_layout((2,), stride=(1,)),
        Float32,
    )
    vals[0] = lo
    vals[1] = hi
    return cute.recast_tensor(vals, Uint64)[0]


@dsl_user_op
def _unpack_u64_to_f32x2(
    packed: Uint64,
    *,
    loc=None,
    ip=None,
) -> tuple[Float32, Float32]:
    struct_ty = ir.Type.parse("!llvm.struct<(f32, f32)>")
    result = llvm.inline_asm(
        struct_ty,
        [Uint64(packed).ir_value(loc=loc, ip=ip)],
        "mov.b64 {$0, $1}, $2;",
        "=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
    )
    lo = Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    hi = Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    return lo, hi


# Use explicit register moves for subword packing. The generic ``pack`` helper
# and manual bit assembly both lower to extra PRMT/LOP3 in row1D quant.
@dsl_user_op
def _pack_u16x2_to_u32_mov(
    lo: Uint16,
    hi: Uint16,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint16(lo).ir_value(loc=loc, ip=ip),
                Uint16(hi).ir_value(loc=loc, ip=ip),
            ],
            "mov.b32 $0, {$1, $2};",
            "=r,h,h",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def _pack_low_byte_u16x4_to_u32_mov(
    b0: Uint16,
    b1: Uint16,
    b2: Uint16,
    b3: Uint16,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint16(b0).ir_value(loc=loc, ip=ip),
                Uint16(b1).ir_value(loc=loc, ip=ip),
                Uint16(b2).ir_value(loc=loc, ip=ip),
                Uint16(b3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 byte0;
                .reg .b8 byte1;
                .reg .b8 byte2;
                .reg .b8 byte3;
                .reg .b8 tmp;
                mov.b16 {byte0, tmp}, $1;
                mov.b16 {byte1, tmp}, $2;
                mov.b16 {byte2, tmp}, $3;
                mov.b16 {byte3, tmp}, $4;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            "=r,h,h,h,h",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@cute.jit
def _unpack_b64_to_i32x2(x: Uint64) -> tuple:
    packed_bits = cute.make_rmem_tensor(
        cute.make_layout((1,), stride=(1,)),
        Uint64,
    )
    packed_bits[0] = x
    words = cute.recast_tensor(packed_bits, Int32)
    return words[0], words[1]


# -----------------------------------------------------------------------------
# Numeric conversion helpers.
# -----------------------------------------------------------------------------
@dsl_user_op
def _cvt_i64_bits_to_u64(x: Int64, *, loc=None, ip=None) -> Uint64:
    return Uint64(Int64(x).ir_value(loc=loc, ip=ip))


@dsl_user_op
def _cvt_bf16x2_to_f32x2(
    word: Uint32,
    *,
    loc=None,
    ip=None,
) -> tuple[Float32, Float32]:
    # Generic BF16 unpack inserts PRMTs after the 128-bit row load in row1D quant.
    lo_bits = Int32((word & Uint32(0xFFFF)) << Uint32(16))
    hi_bits = Int32(word & Uint32(0xFFFF0000))
    lo = Float32(
        llvm.bitcast(T.f32(), lo_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )
    hi = Float32(
        llvm.bitcast(T.f32(), hi_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )
    return lo, hi


@dsl_user_op
def _cvt_fp16x2_to_f32x2(
    word: Uint32,
    *,
    loc=None,
    ip=None,
) -> tuple[Float32, Float32]:
    lo, hi = _unpack_bits(Float16, Uint32(word), loc=loc, ip=ip)
    return lo.to(Float32), hi.to(Float32)


@dsl_user_op
def _cvt_f16x2_to_f32x2(
    word: Uint32,
    src_dtype: cutlass.Constexpr,
    *,
    loc=None,
    ip=None,
) -> tuple[Float32, Float32]:
    if cutlass.const_expr(src_dtype == BFloat16):
        return _cvt_bf16x2_to_f32x2(word, loc=loc, ip=ip)
    return _cvt_fp16x2_to_f32x2(word, loc=loc, ip=ip)


@dsl_user_op
def _cvt_f32x2_to_f16x2_rn(
    x0: Float32,
    x1: Float32,
    dst_dtype: cutlass.Constexpr,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    if dst_dtype not in (BFloat16, Float16):
        raise TypeError("destination must be bfloat16 or float16")
    return Uint32(
        cvt_packfloat_f32(
            T.i32(),
            Float32(x1).ir_value(loc=loc, ip=ip),
            Float32(x0).ir_value(loc=loc, ip=ip),
            Int32(0).ir_value(loc=loc, ip=ip),
            (
                nvvm.CVTPackFloatKind.BF16x2
                if dst_dtype == BFloat16
                else nvvm.CVTPackFloatKind.F16x2
            ),
            rnd=RoundingMode.RN,
            sat=nvvm.SaturationModeKind.NONE,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _cvt_fp4_e2m1_code_to_f32(code: Uint8) -> Float32:
    mag_code = code & Uint8(0x7)
    mag = Float32(0.0)
    if mag_code == Uint8(1):
        mag = Float32(0.5)
    elif mag_code == Uint8(2):
        mag = Float32(1.0)
    elif mag_code == Uint8(3):
        mag = Float32(1.5)
    elif mag_code == Uint8(4):
        mag = Float32(2.0)
    elif mag_code == Uint8(5):
        mag = Float32(3.0)
    elif mag_code == Uint8(6):
        mag = Float32(4.0)
    elif mag_code == Uint8(7):
        mag = Float32(6.0)
    if (code & Uint8(0x8)) != Uint8(0):
        mag = -mag
        if mag_code == Uint8(0):
            mag = Float32(-0.0)
    return mag


@dsl_user_op
def _cvt_fp8_byte_to_f32(
    x: Uint8,
    *,
    src_kind,
    loc=None,
    ip=None,
) -> Float32:
    x_u32 = llvm.zext(T.i32(), Uint8(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    f16x2_bits = Int32(
        cvt_packfloat(
            T.i32(),
            x_u32,
            Int32(0).ir_value(loc=loc, ip=ip),
            src_kind,
            nvvm.CVTPackFloatKind.F16x2,
            rnd=RoundingMode.RN,
            sat=nvvm.SaturationModeKind.NONE,
            loc=loc,
            ip=ip,
        )
    )
    lo, _ = _unpack_bits(Float16, f16x2_bits, loc=loc, ip=ip)
    return lo.to(Float32)


@cute.jit
def _cvt_fp8_e8m0_byte_to_f32(scale_byte: Uint8) -> Float32:
    scale = cute.arch.exp2(Float32(scale_byte) - Float32(E8M0_EXP_BIAS))
    # Byte 0 encodes 2^-127, subnormal in fp32; ex2.approx flushes it to zero.
    scale = Float32(2.0**-127) if scale_byte == Uint8(0) else scale
    return Float32(float("nan")) if scale_byte == Uint8(_E8M0_NAN_BYTE) else scale


@dsl_user_op
def _cvt_fp8_e8m0x2_to_bf16x2(
    x: Int16, *, loc=None, ip=None
) -> Tuple[BFloat16, BFloat16]:
    """Convert two UE8M0 bytes to two bf16 values with the hardware vector cvt."""
    x_u32 = llvm.zext(T.i32(), Int16(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    bf16x2_bits = Int32(
        cvt_packfloat(
            T.i32(),
            x_u32,
            Int32(0).ir_value(loc=loc, ip=ip),
            nvvm.CVTPackFloatKind.UE8M0x2,
            nvvm.CVTPackFloatKind.BF16x2,
            rnd=RoundingMode.RN,
            sat=nvvm.SaturationModeKind.NONE,
            loc=loc,
            ip=ip,
        )
    )
    return _unpack_bits(BFloat16, bf16x2_bits)


@cute.jit
def _cvt_fp8_e8m0x2_to_f32x2(x: Int16) -> Tuple[Float32, Float32]:
    lo, hi = _cvt_fp8_e8m0x2_to_bf16x2(x)
    return lo.to(Float32), hi.to(Float32)


@dsl_user_op
def _cvt_f32x2_to_packed_u16_rn(
    x0: Float32,
    x1: Float32,
    *,
    dst_kind,
    loc=None,
    ip=None,
) -> Uint16:
    packed = cvt_packfloat_f32(
        T.i32(),
        Float32(x1).ir_value(loc=loc, ip=ip),
        Float32(x0).ir_value(loc=loc, ip=ip),
        Int32(0).ir_value(loc=loc, ip=ip),
        dst_kind,
        rnd=RoundingMode.RN,
        sat=nvvm.SaturationModeKind.SATFINITE,
        loc=loc,
        ip=ip,
    )
    return Uint16(
        llvm.trunc(T.i16(), packed, llvm.IntegerOverflowFlags(0), loc=loc, ip=ip)
    )


@dsl_user_op
def _cvt_f32x2_to_packed_u32_rn(
    x0: Float32,
    x1: Float32,
    *,
    dst_kind,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        cvt_packfloat_f32(
            T.i32(),
            Float32(x1).ir_value(loc=loc, ip=ip),
            Float32(x0).ir_value(loc=loc, ip=ip),
            Int32(0).ir_value(loc=loc, ip=ip),
            dst_kind,
            rnd=RoundingMode.RN,
            sat=nvvm.SaturationModeKind.SATFINITE,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _cvt_f32_to_qdata_byte_rn(
    x: Float32,
    *,
    dst_kind: cutlass.Constexpr,
) -> Uint8:
    return _cvt_f32x2_to_qdata_byte_rn(
        x,
        Float32(_FP32_ZERO),
        dst_kind=dst_kind,
    )


@cute.jit
def _cvt_f32x2_to_qdata_byte_rn(
    x0: Float32,
    x1: Float32,
    *,
    dst_kind: cutlass.Constexpr,
) -> Uint8:
    return Uint8(
        _cvt_f32x2_to_packed_u32_rn(
            x0,
            x1,
            dst_kind=dst_kind,
        )
        & Uint32(0xFF)
    )


@dsl_user_op
def _cvt_f32x2_to_fp8_e8m0x2_rp(
    x0: Float32, x1: Float32, *, loc=None, ip=None
) -> Tuple[Float8E8M0FNU, Float8E8M0FNU]:
    # Plain ue8m0x2 (no .satfinite) routes inf -> 0xFF. With .satfinite, inf
    # would clamp to byte 254 and silently drop the non-finite signal that
    # MXFP8 carries via the E8M0 scale byte rather than the FP8 data byte.
    # Normal inputs cannot overflow E8M0 (FLT_MAX / 448 fits comfortably), so
    # inf only arises from a caller-injected NaN -> inf propagation upstream.
    packed = cvt_packfloat_f32(
        T.i32(),
        Float32(x1).ir_value(loc=loc, ip=ip),
        Float32(x0).ir_value(loc=loc, ip=ip),
        Int32(0).ir_value(loc=loc, ip=ip),
        nvvm.CVTPackFloatKind.UE8M0x2,
        rnd=RoundingMode.RP,
        sat=nvvm.SaturationModeKind.NONE,
        loc=loc,
        ip=ip,
    )
    packed16 = Int16(
        llvm.trunc(T.i16(), packed, llvm.IntegerOverflowFlags(0), loc=loc, ip=ip)
    )
    return _unpack_bits(Float8E8M0FNU, packed16)


@cute.jit
def _cvt_f32x4_to_fp8x4_u32_rn(
    x0: Float32,
    x1: Float32,
    x2: Float32,
    x3: Float32,
    *,
    dst_kind: cutlass.Constexpr,
) -> Uint32:
    lo = _cvt_f32x2_to_packed_u16_rn(
        x0,
        x1,
        dst_kind=dst_kind,
    )
    hi = _cvt_f32x2_to_packed_u16_rn(
        x2,
        x3,
        dst_kind=dst_kind,
    )
    return _pack_u16x2_to_u32_mov(lo, hi)


@cute.jit
def _cvt_packed_f32x2x2_to_fp8x4_u32_rn(
    x01: Uint64,
    x23: Uint64,
    *,
    dst_kind: cutlass.Constexpr,
) -> Uint32:
    x0, x1 = _unpack_u64_to_f32x2(x01)
    x2, x3 = _unpack_u64_to_f32x2(x23)
    return _cvt_f32x4_to_fp8x4_u32_rn(
        x0,
        x1,
        x2,
        x3,
        dst_kind=dst_kind,
    )


@cute.jit
def _cvt_f32x8_to_fp4_e2m1x8_u32_rn(
    x0: Float32,
    x1: Float32,
    x2: Float32,
    x3: Float32,
    x4: Float32,
    x5: Float32,
    x6: Float32,
    x7: Float32,
) -> Uint32:
    byte0 = _cvt_f32x2_to_packed_u16_rn(
        x0,
        x1,
        dst_kind=nvvm.CVTPackFloatKind.E2M1x2,
    )
    byte1 = _cvt_f32x2_to_packed_u16_rn(
        x2,
        x3,
        dst_kind=nvvm.CVTPackFloatKind.E2M1x2,
    )
    byte2 = _cvt_f32x2_to_packed_u16_rn(
        x4,
        x5,
        dst_kind=nvvm.CVTPackFloatKind.E2M1x2,
    )
    byte3 = _cvt_f32x2_to_packed_u16_rn(
        x6,
        x7,
        dst_kind=nvvm.CVTPackFloatKind.E2M1x2,
    )
    return _pack_low_byte_u16x4_to_u32_mov(
        byte0,
        byte1,
        byte2,
        byte3,
    )
