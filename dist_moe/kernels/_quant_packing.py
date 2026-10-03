# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-element deterministic FP8/FP4 qdata packing."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint8, Uint32, Uint64
from cutlass._mlir.dialects import nvvm

from ..formats import BlockScaledFormatId
from . import _dsl_compat as _cute_extern  # noqa: F401
from ._quant_conversion import (
    _cvt_f32_to_qdata_byte_rn,
    _cvt_f32x2_to_qdata_byte_rn,
    _cvt_f32x4_to_fp8x4_u32_rn,
    _cvt_f32x8_to_fp4_e2m1x8_u32_rn,
    _cvt_packed_f32x2x2_to_fp8x4_u32_rn,
    _FP32_ZERO,
)

BLOCK_SCALED_FORMAT_MXFP8_E4M3 = BlockScaledFormatId.MXFP8_E4M3.value
BLOCK_SCALED_FORMAT_MXFP8_E5M2 = BlockScaledFormatId.MXFP8_E5M2.value
BLOCK_SCALED_FORMAT_NVFP4 = BlockScaledFormatId.NVFP4.value
BLOCK_SCALED_FORMAT_MXFP4 = BlockScaledFormatId.MXFP4.value


@cute.jit
def _cvt_qdata_for_format_id(
    q: Float32,
    format_id: cutlass.Constexpr[int],
) -> Uint8:
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E4M3):
        return _cvt_f32_to_qdata_byte_rn(
            q,
            dst_kind=nvvm.CVTPackFloatKind.E4M3x2,
        )
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2):
        return _cvt_f32_to_qdata_byte_rn(
            q,
            dst_kind=nvvm.CVTPackFloatKind.E5M2x2,
        )
    return _cvt_f32x2_to_qdata_byte_rn(
        q,
        Float32(_FP32_ZERO),
        dst_kind=nvvm.CVTPackFloatKind.E2M1x2,
    )


@cute.jit
def _pack_fp4_e2m1_xn_to_u32_rn(
    q_values,
    packed,
    *,
    base: cutlass.Constexpr[int] = 0,
    packed_base: cutlass.Constexpr[int] = 0,
    num_elems: cutlass.Constexpr[int],
    recip: Float32,
    scale_values: cutlass.Constexpr[bool],
) -> None:
    if cutlass.const_expr(num_elems != 8 and num_elems != 16 and num_elems != 32):
        raise ValueError("FP4 pack expects 8, 16, or 32 elements")
    for word in cutlass.range_constexpr(num_elems // 8):
        elem: cutlass.Constexpr[int] = base + word * 8
        if cutlass.const_expr(scale_values):
            v0 = q_values[elem + 0] * recip
            v1 = q_values[elem + 1] * recip
            v2 = q_values[elem + 2] * recip
            v3 = q_values[elem + 3] * recip
            v4 = q_values[elem + 4] * recip
            v5 = q_values[elem + 5] * recip
            v6 = q_values[elem + 6] * recip
            v7 = q_values[elem + 7] * recip
        else:
            v0 = q_values[elem + 0]
            v1 = q_values[elem + 1]
            v2 = q_values[elem + 2]
            v3 = q_values[elem + 3]
            v4 = q_values[elem + 4]
            v5 = q_values[elem + 5]
            v6 = q_values[elem + 6]
            v7 = q_values[elem + 7]
        packed[packed_base + word] = _cvt_f32x8_to_fp4_e2m1x8_u32_rn(
            v0,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
        )


@cute.jit
def _pack_packed_f32x2x2_to_fp8x4_u32_rn(
    q01: Uint64,
    q23: Uint64,
    *,
    format_id: cutlass.Constexpr[int],
) -> Uint32:
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2):
        return _cvt_packed_f32x2x2_to_fp8x4_u32_rn(
            q01,
            q23,
            dst_kind=nvvm.CVTPackFloatKind.E5M2x2,
        )
    return _cvt_packed_f32x2x2_to_fp8x4_u32_rn(
        q01,
        q23,
        dst_kind=nvvm.CVTPackFloatKind.E4M3x2,
    )


@cute.jit
def _pack_fp8_xn_to_u32_rn(
    q_values,
    packed,
    *,
    base: cutlass.Constexpr[int] = 0,
    packed_base: cutlass.Constexpr[int] = 0,
    num_elems: cutlass.Constexpr[int],
    recip: Float32,
    scale_values: cutlass.Constexpr[bool],
    dst_kind: cutlass.Constexpr,
) -> None:
    if cutlass.const_expr(
        num_elems != 4 and num_elems != 8 and num_elems != 16 and num_elems != 32
    ):
        raise ValueError("FP8 pack expects 4, 8, 16, or 32 elements")
    for word in cutlass.range_constexpr(num_elems // 4):
        elem: cutlass.Constexpr[int] = base + word * 4
        if cutlass.const_expr(scale_values):
            v0 = q_values[elem + 0] * recip
            v1 = q_values[elem + 1] * recip
            v2 = q_values[elem + 2] * recip
            v3 = q_values[elem + 3] * recip
        else:
            v0 = q_values[elem + 0]
            v1 = q_values[elem + 1]
            v2 = q_values[elem + 2]
            v3 = q_values[elem + 3]
        packed[packed_base + word] = _cvt_f32x4_to_fp8x4_u32_rn(
            v0,
            v1,
            v2,
            v3,
            dst_kind=dst_kind,
        )


@cute.jit
def _pack_fp8_xn_to_u32(
    q_values,
    packed,
    *,
    base: cutlass.Constexpr[int] = 0,
    packed_base: cutlass.Constexpr[int] = 0,
    format_id: cutlass.Constexpr[int],
    num_elems: cutlass.Constexpr[int],
    recip: Float32,
    scale_values: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2):
        _pack_fp8_xn_to_u32_rn(
            q_values,
            packed,
            base=base,
            packed_base=packed_base,
            num_elems=num_elems,
            recip=recip,
            scale_values=scale_values,
            dst_kind=nvvm.CVTPackFloatKind.E5M2x2,
        )
        return
    _pack_fp8_xn_to_u32_rn(
        q_values,
        packed,
        base=base,
        packed_base=packed_base,
        num_elems=num_elems,
        recip=recip,
        scale_values=scale_values,
        dst_kind=nvvm.CVTPackFloatKind.E4M3x2,
    )
