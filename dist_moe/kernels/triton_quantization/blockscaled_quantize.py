# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic Triton quantization kernels for block-scaled formats.

Supported formats: MXFP8 (E4M3 / E5M2), MXFP4, and NVFP4.

Design: numerics live in shared helpers, kernels only orchestrate
=================================================================
Every floating-point decision -- the per-block scale rule, the FP8 / FP4 cast,
NaN propagation through the block reduction, and half-range canonicalization --
is concentrated in the ``@triton.jit`` helpers at the top of this file. The
launched kernels add only tile / layout / address math on top; they contain no
freestanding numerics. To change quantization behavior, edit a helper, not a
kernel.

Launched kernels
================
- ``_triton_quantize_blockscaled_row1d_tile`` -- row-1D quantization, generic layout.
- ``_triton_quantize_blockscaled_row1d_cublas_blocked`` -- row-1D, cuBLAS-blocked SF fast path.
- ``_triton_quantize_blockscaled_tile2d`` -- axis=0 / axis=1 / 2D-tile reduction.

Per-block numerics pipeline
===========================
Each block flows through the shared helpers in this order (bracketed steps are
optional):

    amax -> scale + reciprocal ->[half-range]-> cast / clamp -> pack

1. Reduction
   - ``_compute_block_amax_nonfinite`` -- block-wise amax that maps ``NaN -> inf``
     so ``tl.max`` (IEEE maxNum, which drops NaN) still carries the non-finite
     signal forward to the scale path.

2. Scale + reciprocal
   - ``_compute_e8m0_scale_rceil`` -- MXFP8 / MXFP4 round-up E8M0 scale byte +
     reciprocal. Owns the 0xFF-byte override that poisons every element of an
     inf / NaN block.
   - ``_compute_nvfp4_scale_rtne`` -- NVFP4 FP8 E4M3 scale (inf -> NaN
     restore, clamp / EPS rule) + its fp32 reciprocal, where NaN-byte 0x7F
     naturally produces fp32 NaN. The E4M3 analog of the E8M0 helper above.

3. Half-range canonicalization -- optional, ``half_range_scale=True``
   - ``_canonicalize_mx_half_range`` -- E8M0 exponent shift for bitwise-stable
     quantize -> dequantize -> quantize across MXFP8 (E4M3/E5M2) and MXFP4,
     selecting the per-format half-range midpoint via ``FORMAT_ID``.

4. Cast / clamp
   - ``_clamp_to_target_max`` -- clamp ``x * recip`` into the target dtype's range
     with ``propagate_nan=ALL`` so NaN survives the clamp.

"""

from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .formats import (
    AxisMask,
    BlockScaledFormatId,
    BlockScaledProducer,
    build_block_tile_tag as _build_block_tile_tag,
    canonical_swiglu_clamp,
    CUBLAS_BLOCKED_ENTRIES_PER_ATOM,
    CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE,
    CUBLAS_BLOCKED_ROW_LANES,
    CUBLAS_BLOCKED_ROWS_PER_ATOM,
    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
    E4M3_MIN_SUBNORMAL,
    E8M0_EXP_BIAS,
    E8M0_MAX_UNBIASED_EXP,
    E8M0_MIN_UNBIASED_EXP,
    FORMAT_SF_VEC as _FORMAT_SF_VEC,
    FORMAT_TAG as _FORMAT_TAG,
    FP4_E2M1_HALF_RANGE_ROUND_BOUNDARY,
    FP4_E2M1_MAX,
    FP4_E2M1_MAX_RECIP,
    FP4_FORMAT_IDS as _FP4_FORMAT_IDS,
    FP8_E4M3_HALF_RANGE_ROUND_BOUNDARY,
    FP8_E4M3_MAX,
    FP8_E5M2_HALF_RANGE_ROUND_BOUNDARY,
    FP8_E5M2_MAX,
    LAYOUT_TAG as _LAYOUT_TAG,
    MXFP6_FORMAT_IDS as _MXFP6_FORMAT_IDS,
    MXFP8_FORMAT_IDS as _MXFP8_FORMAT_IDS,
    NVFP4_VARIANT_FORMAT_IDS,
    PRODUCER_TAG as _PRODUCER_TAG,
    ScaleFactorLayoutId,
    ScaleReduction,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
    UE5M3_EXP_BIAS,
    UE5M3_MAX,
    UE5M3_MAX_FINITE_CODE,
    UE5M3_MIN_SUBNORMAL,
    UE5M3_NAN_BYTE,
)
from .fp_math import sigmoid
from .global_scale import (
    GlobalScaleRank,
    NVFP4_GLOBAL_SCALE_EPS as _GLOBAL_SCALE_EPS_VALUE,
)
from .repr_utils import make_dtype_repr
from .tma_utils import supports_tma


def _build_quant_output_kind_tag(
    axis_mask: int | None,
    scale_reduction: int | None,
) -> str:
    has_m = axis_mask is not None and bool(axis_mask & AxisMask.M.value)
    has_k = axis_mask is not None and bool(axis_mask & AxisMask.K.value)
    if axis_mask is None or (has_k and not has_m):
        return "row1d"
    if has_m and not has_k:
        return "col1d"
    if scale_reduction == ScaleReduction.TWO_D.value:
        return "tile2d"
    return "both1d"


def _build_quantize_spec_name(
    op_tag: str,
    layout_constexpr: str | None,
    sf_vec_constexpr: str,
    axis_mask_constexpr: str | None = None,
    scale_reduction_constexpr: str | None = None,
    producer_id_constexpr: str | None = None,
    producer_tag_override: str | None = None,
) -> Callable[..., str]:
    """Build the ``@triton.jit(repr=...)`` callable that names a quantize
    specialization
    ``_triton_quantize_<producer>_<in_dtype>_<fmt>_<layout>_<output_kind>_<RxC>``.

    Each field is read from the kernel's signature/constexprs at compile time:
    input dtype from ``x_ptr`` (or ``SRC_DTYPE`` for the int64 ZEROCOPY_GATHER
    pointer table), layout from ``layout_constexpr`` (or ``op_tag`` when it is
    ``None``), output kind and ``RxC`` SF-tile shape from
    ``(axis_mask, scale_reduction)``, and the producer prefix (``id``,
    ``swiglu_fwd``, ``swiglu_bwd``, or ``zerocopy_gather``) from
    ``producer_id_constexpr``.
    """

    def _unwrap(value):
        return value.value if hasattr(value, "value") else value

    def _build_base(specialization) -> str:
        constants = specialization.constants
        in_dtype = specialization.signature.get("x_ptr", "")
        if in_dtype.startswith("*"):
            in_dtype = in_dtype[1:]
        if in_dtype in ("", "i64") and "SRC_DTYPE" in constants:
            in_dtype = constants["SRC_DTYPE"].name
        if layout_constexpr is not None and layout_constexpr in constants:
            layout_id = _unwrap(constants[layout_constexpr])
            layout = _LAYOUT_TAG.get(layout_id, f"layout{layout_id}")
        else:
            layout = op_tag  # 'natural' or 'blocked' from kernel def
        fmt_id = _unwrap(constants.get("FORMAT_ID"))
        fmt = _FORMAT_TAG.get(fmt_id, f"fmt{fmt_id}")
        sf_vec = (
            _unwrap(constants[sf_vec_constexpr])
            if sf_vec_constexpr in constants
            else _FORMAT_SF_VEC.get(fmt_id, 0)
        )
        axis_mask = (
            _unwrap(constants[axis_mask_constexpr])
            if axis_mask_constexpr and axis_mask_constexpr in constants
            else None
        )
        scale_reduction = (
            _unwrap(constants[scale_reduction_constexpr])
            if scale_reduction_constexpr and scale_reduction_constexpr in constants
            else None
        )
        output_kind = _build_quant_output_kind_tag(axis_mask, scale_reduction)
        block_tile = _build_block_tile_tag(axis_mask, scale_reduction, sf_vec)
        if producer_tag_override is not None:
            producer_tag = producer_tag_override + "_"
        elif producer_id_constexpr and producer_id_constexpr in constants:
            producer_id = _unwrap(constants[producer_id_constexpr])
            producer_tag = _PRODUCER_TAG.get(producer_id, f"prod{producer_id}") + "_"
        else:
            producer_tag = ""
        return (
            f"_triton_quantize_{producer_tag}{in_dtype}_{fmt}_{layout}_"
            f"{output_kind}_{block_tile}"
        )

    return _build_base


# =============================================================================
# Constants (JIT-visible constexpr views + layout sizes)
# =============================================================================


# Triton JIT can only reference module globals that are instantiated as
# ``tl.constexpr``. The public ABI enums / constants in
# ``interfaces.quant.formats`` remain the source of truth; ``_wrap`` builds
# JIT-visible views of the same values (reducing IntEnum members to ``.value``).
def _wrap(value):
    return tl.constexpr(value.value if hasattr(value, "value") else value)


_TL_BLOCK_SCALED_FORMAT_MXFP8_E4M3 = _wrap(BlockScaledFormatId.MXFP8_E4M3)
_TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2 = _wrap(BlockScaledFormatId.MXFP8_E5M2)
_TL_BLOCK_SCALED_FORMAT_NVFP4 = _wrap(BlockScaledFormatId.NVFP4)
_TL_BLOCK_SCALED_FORMAT_MXFP4 = _wrap(BlockScaledFormatId.MXFP4)
_TL_BLOCK_SCALED_FORMAT_NVFP4_UE5M3 = _wrap(BlockScaledFormatId.NVFP4_UE5M3)
_TL_SCALE_FACTOR_LAYOUT_NATURAL = _wrap(ScaleFactorLayoutId.NATURAL)
_TL_SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED = _wrap(ScaleFactorLayoutId.CUBLAS_BLOCKED)
_TL_BLOCK_SCALED_PRODUCER_IDENTITY = _wrap(BlockScaledProducer.IDENTITY)
_TL_BLOCK_SCALED_PRODUCER_SWIGLU_FWD = _wrap(BlockScaledProducer.SWIGLU_FWD)
_TL_BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY = _wrap(BlockScaledProducer.SWIGLU_BWD_DXY)
_TL_BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER = _wrap(BlockScaledProducer.ZEROCOPY_GATHER)
_TL_AXIS_MASK_M = tl.constexpr(AxisMask.M.value)
_TL_AXIS_MASK_K = tl.constexpr(AxisMask.K.value)
_TL_SCALE_REDUCTION_ONE_D = tl.constexpr(ScaleReduction.ONE_D.value)
_TL_SCALE_REDUCTION_TWO_D = tl.constexpr(ScaleReduction.TWO_D.value)

_TL_FP32_ZERO = _wrap(0.0)
_TL_FP32_ONE = _wrap(1.0)
_TL_FP32_TWO = _wrap(2.0)
_TL_FP8_E4M3_MAX = _wrap(FP8_E4M3_MAX)
_TL_FP8_E4M3_MAX_RECIP = _wrap(1.0 / FP8_E4M3_MAX)
_TL_FP8_E5M2_MAX = _wrap(FP8_E5M2_MAX)
_TL_FP8_E5M2_MAX_RECIP = _wrap(1.0 / FP8_E5M2_MAX)
_TL_FP8_E4M3_HALF_RANGE_ROUND_BOUNDARY = _wrap(FP8_E4M3_HALF_RANGE_ROUND_BOUNDARY)
_TL_FP8_E5M2_HALF_RANGE_ROUND_BOUNDARY = _wrap(FP8_E5M2_HALF_RANGE_ROUND_BOUNDARY)
_TL_FP4_E2M1_MAX = _wrap(FP4_E2M1_MAX)
_TL_FP4_E2M1_MAX_RECIP = _wrap(FP4_E2M1_MAX_RECIP)
_TL_FP4_E2M1_HALF_RANGE_ROUND_BOUNDARY = _wrap(FP4_E2M1_HALF_RANGE_ROUND_BOUNDARY)
_TL_FP8_E4M3_MIN_SUBNORMAL = _wrap(E4M3_MIN_SUBNORMAL)
# UE5M3 scale constants (PTX ISA 9.4 ue5m3; see formats.py for the derivation).
_TL_UE5M3_MAX = _wrap(UE5M3_MAX)
_TL_UE5M3_MIN_SUBNORMAL = _wrap(UE5M3_MIN_SUBNORMAL)
# fp32 bit pattern of the smallest UE5M3 normal, 2^(1 - bias) = 2^-14.
_TL_UE5M3_F32_MIN_NORMAL_BITS = _wrap((127 + 1 - UE5M3_EXP_BIAS) << 23)
# Packed-code rebias: fp32 (bits >> 20) carries exponent bias 127; the UE5M3
# code carries bias 15 (E5M2's), so subtract (127 - 15) << 3.
_TL_UE5M3_F32_CODE_REBIAS = _wrap((127 - UE5M3_EXP_BIAS) << 3)
# 1 / min subnormal: maps a subnormal-range value onto its integer code 0..8.
_TL_UE5M3_SUBNORMAL_QUANTUM_RECIP = _wrap(1.0 / UE5M3_MIN_SUBNORMAL)
# No-clip subnormal threshold: max(amax, T) / 96 must reach half the UE5M3
# subnormal spacing (2^-18), so T = 96 * 2^-18 = 3 * 2^-13 (E4M3 analog: 3/32).
_TL_UE5M3_NO_CLIP_SUB_THRESHOLD = _wrap(3.0 * 2.0**-13)
# 0xFE: only the NaN byte 0xFF is non-numeric (ue4m3/ue8m0 convention);
# see the formats.py derivation.
_TL_UE5M3_MAX_FINITE_CODE = _wrap(UE5M3_MAX_FINITE_CODE)
_TL_UE5M3_NAN_BYTE = _wrap(UE5M3_NAN_BYTE)
_TL_E8M0_EXP_BIAS = _wrap(E8M0_EXP_BIAS)
_TL_E8M0_MIN_UNBIASED_EXP = _wrap(E8M0_MIN_UNBIASED_EXP)
_TL_E8M0_MAX_UNBIASED_EXP = _wrap(E8M0_MAX_UNBIASED_EXP)
_TL_E8M0_NAN_BYTE = _wrap(255.0)
_TL_E8M0_NAN_BYTE_INT = _wrap(255)
_TL_E8M0_MIN_NORMAL_BYTE = _wrap(1)
_TL_E8M0_RECIP_EXP_BIAS = _wrap(254)
_TL_E8M0_BYTE0_RECIP = _wrap(2.0**127)
_TL_GLOBAL_SCALE_EPS = _wrap(_GLOBAL_SCALE_EPS_VALUE)
_TL_GLOBAL_SCALE_RANK_2D = _wrap(GlobalScaleRank.TWO_D)

_TL_CUBLAS_BLOCKED_ROWS_PER_ATOM = _wrap(CUBLAS_BLOCKED_ROWS_PER_ATOM)
_TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM = _wrap(CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM)
_TL_CUBLAS_BLOCKED_ROW_LANES = _wrap(CUBLAS_BLOCKED_ROW_LANES)
_TL_CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE = _wrap(CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE)
_TL_CUBLAS_BLOCKED_ENTRIES_PER_ATOM = _wrap(CUBLAS_BLOCKED_ENTRIES_PER_ATOM)

_NVFP4_RECIP_LUT_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _build_nvfp4_recip_lut(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    lut = _NVFP4_RECIP_LUT_CACHE.get(key)
    if lut is None:
        scale_bytes = torch.arange(256, dtype=torch.uint8, device=device)
        lut = 1.0 / scale_bytes.view(torch.float8_e4m3fn).to(torch.float32)
        _NVFP4_RECIP_LUT_CACHE[key] = lut
    return lut


# =============================================================================
# Shared numerics helpers (@triton.jit FP math)
# =============================================================================


@triton.jit
def _is_blackwell():
    return tl.target_info.cuda_capability_geq(10, 0)


@triton.jit
def _recip_from_e8m0_byte(scale_u8):
    """fp32 reciprocal ``2^(127 - byte)`` from a supplied E8M0 scale byte.

    Exact via the fp32 exponent field (integer-only, no transcendental). The NaN
    byte ``0xFF`` would otherwise give a garbage reciprocal (byte 255 shifts to -inf),
    so it is overridden to NaN to poison the block. Compares on ``int32`` (not
    ``uint8``) to sidestep the uint8-predicate Triton codegen miscompile noted in
    ``_canonicalize_mx_half_range``.
    """
    scale_i32 = scale_u8.to(tl.int32)
    recip_exp = _TL_E8M0_RECIP_EXP_BIAS - scale_i32
    recip_scale = (recip_exp << 23).to(tl.float32, bitcast=True)
    return tl.where(scale_i32 == _TL_E8M0_NAN_BYTE_INT, float("nan"), recip_scale)


@triton.jit
def _compute_e8m0_scale_rceil(
    max_abs,
    TARGET_MAX_RECIP: tl.constexpr,
):
    """Round-up E8M0 block scale: returns (scale_u8, recip_scale).

    Inf in ``max_abs`` maps to the E8M0 NaN byte 0xFF, not the max finite
    exponent -- MXFP8 can't signal NaN in the data byte, so it rides the scale
    byte. Finite fp32 can't overflow E8M0, so inf only comes from a caller's
    NaN -> inf substitution upstream.
    """
    tl.static_assert(max_abs.dtype == tl.float32, "max_abs must be fp32")
    scale_input = max_abs * TARGET_MAX_RECIP
    if _is_blackwell():
        # Hardware round-up cvt (E8M0 has no native Triton cast). Plain (not
        # .satfinite) maps inf -> 0xFF; real value is b=$1. pack=1 + dummy 0.0
        # lane handles any count; the =r output avoids an =h ue8m0x2 miscompile.
        scale_u8 = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b16 packed;
                cvt.rp.ue8m0x2.f32 packed, 0.0, $1;
                cvt.u32.u16 $0, packed;
            }
            """,
            constraints="=r,r",
            args=[scale_input],
            dtype=tl.uint8,
            is_pure=True,
            pack=1,
        )
    else:
        # Pure-Triton equivalent for pre-Blackwell: rceil(log2) biased into
        # E8M0, non-finite -> NaN byte 0xFF.
        is_non_finite = (scale_input != scale_input) | (
            tl.abs(scale_input) == float("inf")
        )
        is_zero = max_abs == _TL_FP32_ZERO
        log2_scale_input = tl.where(
            is_zero, _TL_E8M0_MIN_UNBIASED_EXP, tl.log2(scale_input)
        )
        ceil_log2 = tl.math.ceil(log2_scale_input)
        clamped_exp = tl.clamp(
            ceil_log2, _TL_E8M0_MIN_UNBIASED_EXP, _TL_E8M0_MAX_UNBIASED_EXP
        )
        scale_u8 = tl.where(
            is_non_finite,
            _TL_E8M0_NAN_BYTE,
            clamped_exp + _TL_E8M0_EXP_BIAS,
        ).to(tl.uint8)

    recip_scale = _recip_from_e8m0_byte(scale_u8)
    return scale_u8, recip_scale


@triton.jit
def _compute_nvfp4_scale_rtne(max_abs, USE_NVFP4_NO_CLIP_SCALE: tl.constexpr):
    """Round-to-nearest-even NVFP4 block scale and its fp32 reciprocal.

    Inf/NaN in ``max_abs`` maps to the E4M3 NaN byte 0x7F -- NVFP4 can't signal
    NaN in the E2M1 data byte, so it rides the scale byte (and 1/NaN then poisons
    the block). The scale is returned as ``float8e4nv`` (not bitcast) so its byte
    never co-lives with the reciprocal across a scale ``tl.store`` (NVFP4 Triton
    miscompile).
    """
    tl.static_assert(max_abs.dtype == tl.float32, "max_abs must be fp32")
    block_amax = tl.where(max_abs == float("inf"), float("nan"), max_abs)
    if USE_NVFP4_NO_CLIP_SCALE:
        # Normal E4M3 bins need 17/16 relative headroom. Subnormal bins have
        # fixed 2^-9 spacing, so add half that spacing before RTNE instead:
        # amax / 6 + max(amax / 96, 2^-10).
        scale_input = tl.fma(
            block_amax,
            _TL_FP4_E2M1_MAX_RECIP,
            tl.maximum(block_amax, 3.0 / 32.0) * (1.0 / 96.0),
        )
    else:
        scale_input = block_amax * _TL_FP4_E2M1_MAX_RECIP
    # ``satfinite`` covers the top (>448 -> 448) + NaN -> 0x7F; the clamp's lower
    # bound is the EPS rule keeping scale off zero so 1/scale stays finite.
    scale_input = tl.clamp(
        scale_input,
        _TL_FP8_E4M3_MIN_SUBNORMAL,
        _TL_FP8_E4M3_MAX,
        propagate_nan=tl.PropagateNan.ALL,
    )
    # Explicit cvt.rn.satfinite.e4m3x2.f32 (round-to-nearest-even, NaN -> 0x7F);
    # real value is b=$1. pack=1 + dummy 0.0 lane handles any count, incl. the
    # 1x1 cuBLAS-blocked tile2d atom that pack=2 rejects. Same =r-output shape as
    # _compute_e8m0_scale_rceil.
    scale_u8 = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b16 packed;
            cvt.rn.satfinite.e4m3x2.f32 packed, 0.0, $1;
            cvt.u32.u16 $0, packed;
        }
        """,
        constraints="=r,r",
        args=[scale_input],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )
    scale_fp8 = scale_u8.to(tl.float8e4nv, bitcast=True)
    recip_scale = _TL_FP32_ONE / scale_fp8.to(tl.float32)
    return scale_fp8, recip_scale


@triton.jit
def _global_scale_row_offsets(
    rows,
    global_scale_stride0,
    global_scale_stride1,
    GLOBAL_SCALE_RANK: tl.constexpr,
    GLOBAL_SCALE_ROWS_PER_GROUP: tl.constexpr,
):
    if GLOBAL_SCALE_RANK == _TL_GLOBAL_SCALE_RANK_2D:
        group = rows // GLOBAL_SCALE_ROWS_PER_GROUP
        row = rows - group * GLOBAL_SCALE_ROWS_PER_GROUP
        return group * global_scale_stride0 + row * global_scale_stride1
    return rows * global_scale_stride0


@triton.jit
def _safe_global_scale(global_scale):
    return tl.where(
        global_scale > _TL_GLOBAL_SCALE_EPS,
        global_scale,
        _TL_GLOBAL_SCALE_EPS,
    )


@triton.jit
def _load_global_scale_rows(
    global_scale_ptr,
    global_scale_stride0,
    global_scale_stride1,
    rows,
    row_mask,
    HAS_GLOBAL_SCALE: tl.constexpr,
    GLOBAL_SCALE_RANK: tl.constexpr,
    GLOBAL_SCALE_ROWS_PER_GROUP: tl.constexpr,
):
    """Per-row fp32 global-scale tensor shaped like ``rows``.

    Returns all-ones when ``HAS_GLOBAL_SCALE`` is off; otherwise gathers one
    value per row via ``_global_scale_row_offsets`` (masked rows read 1.0).
    """
    global_scale = tl.full(rows.shape, _TL_FP32_ONE, tl.float32)
    if HAS_GLOBAL_SCALE:
        global_scale_offsets = _global_scale_row_offsets(
            rows,
            global_scale_stride0,
            global_scale_stride1,
            GLOBAL_SCALE_RANK,
            GLOBAL_SCALE_ROWS_PER_GROUP,
        )
        global_scale = tl.load(
            global_scale_ptr + global_scale_offsets,
            mask=row_mask,
            other=_TL_FP32_ONE,
        ).to(tl.float32)
    return global_scale


@triton.jit
def _nvfp4_scale_and_recip(
    max_abs,
    global_scale,
    HAS_GLOBAL_SCALE: tl.constexpr,
    USE_NVFP4_NO_CLIP_SCALE: tl.constexpr = False,
):
    if HAS_GLOBAL_SCALE:
        global_scale_safe = _safe_global_scale(global_scale)
        # Raw-scale order is the fleet convention: ``(max_abs * gs) * fl(1/6)``
        # (`_compute_nvfp4_scale_rtne` applies the fl(1/6) multiply last).
        # Every NVFP4 quantizer computes this order. Order
        # matters: an amax-first evaluation rounds to the opposite side of an
        # E4M3 RTNE tie for some inputs (e.g. amax=3.125, gs=768: exactly
        # 400.0 -> scale 384 here vs 400.00003 -> scale 416 with an early
        # fl(1/6) multiply), flipping the scale byte and with it the whole
        # 16-element block. When ``max_abs * gs`` is exactly representable
        # this is also the closer evaluation of ``amax * gs / 6``.
        block_scale_fp8 = _compute_nvfp4_scale_rtne(
            max_abs * global_scale_safe, USE_NVFP4_NO_CLIP_SCALE
        )[0]
        # Single correctly-rounded division: reciprocal-then-multiply double
        # rounds and can land 1 ulp off, flipping FP4 codes for values that
        # scale to within 1 ulp of an E2M1 rounding boundary.
        recip_scale = tl.div_rn(global_scale_safe, block_scale_fp8.to(tl.float32))
    else:
        block_scale_fp8, recip_scale = _compute_nvfp4_scale_rtne(
            max_abs, USE_NVFP4_NO_CLIP_SCALE
        )
    return block_scale_fp8, recip_scale


@triton.jit
def _ue5m3_code_from_clamped_f32(v):
    """UE5M3 code byte (round-to-nearest-even, like the E4M3 scale path)
    for a positive finite fp32 ``v`` already clamped to
    ``[UE5M3_MIN_SUBNORMAL, UE5M3_MAX]``.

    Rounded in software: no ``cvt.*.ue5m3x2`` exists below sm_107f, and
    emulation must run on any arch. Normals: for positive fp32,
    ``bits >> 20`` is ``(biased_exp << 3) | top3_mantissa``, so rounding
    the dropped 20 bits (carry flows into the exponent) and rebasing
    127 -> 15 gives the code directly. Subnormals (below the 2^-14 min
    normal): round ``v / 2^-17`` to an integer code in [1, 8]; 8 is the
    first normal, so the two paths meet without a gap.
    """
    tl.static_assert(v.dtype == tl.float32, "v must be fp32")
    bits = v.to(tl.int32, bitcast=True)
    base = bits >> 20
    rem = bits & 0xFFFFF
    round_up = (rem > 0x80000) | ((rem == 0x80000) & ((base & 1) == 1))
    normal_code = base + round_up.to(tl.int32) - _TL_UE5M3_F32_CODE_REBIAS

    x = v * _TL_UE5M3_SUBNORMAL_QUANTUM_RECIP
    k = tl.floor(x)
    frac = x - k
    ki = k.to(tl.int32)
    sub_up = (frac > 0.5) | ((frac == 0.5) & ((ki & 1) == 1))
    sub_code = ki + sub_up.to(tl.int32)

    code = tl.where(bits >= _TL_UE5M3_F32_MIN_NORMAL_BITS, normal_code, sub_code)
    return tl.minimum(code, _TL_UE5M3_MAX_FINITE_CODE)


@triton.jit
def _ue5m3_code_to_f32(code):
    """Exact fp32 value of a finite UE5M3 code (int32; NaN byte is the
    caller's job -- 255 here would decode as a large normal)."""
    normal_bits = (code + _TL_UE5M3_F32_CODE_REBIAS) << 20
    normal = normal_bits.to(tl.float32, bitcast=True)
    sub = (code & 7).to(tl.float32) * _TL_UE5M3_MIN_SUBNORMAL
    return tl.where((code >> 3) > 0, normal, sub)


@triton.jit
def _compute_nvfp4_ue5m3_scale(
    max_abs,
    USE_NVFP4_NO_CLIP_SCALE: tl.constexpr,
):
    """UE5M3 block scale: ``(code_i32, scale_fp32, recip_fp32)``.

    Same contract as ``_compute_nvfp4_scale_rtne``: an inf/NaN amax sets the
    scale byte to 0xFF and poisons the block with a NaN scale; the lower
    clamp keeps 1/scale finite.
    """
    tl.static_assert(max_abs.dtype == tl.float32, "max_abs must be fp32")
    block_amax = tl.where(max_abs == float("inf"), float("nan"), max_abs)
    if USE_NVFP4_NO_CLIP_SCALE:
        # Headroom so the rounded (RTNE) scale stays >= amax/6 (no clipping):
        # rounding can go down by half an ulp (1/16 relative), so add
        # amax/96; below T = 3 * 2^-13 the fixed subnormal half-spacing
        # (2^-18) dominates, supplied by T/96. Same rule as E4M3.
        headroom = tl.maximum(block_amax, _TL_UE5M3_NO_CLIP_SUB_THRESHOLD) * (
            1.0 / 96.0
        )
        scale_input = tl.fma(block_amax, _TL_FP4_E2M1_MAX_RECIP, headroom)
    else:
        scale_input = block_amax * _TL_FP4_E2M1_MAX_RECIP
    is_nan = scale_input != scale_input
    scale_input = tl.clamp(
        scale_input,
        _TL_UE5M3_MIN_SUBNORMAL,
        _TL_UE5M3_MAX,
        propagate_nan=tl.PropagateNan.ALL,
    )
    safe_input = tl.where(is_nan, _TL_UE5M3_MIN_SUBNORMAL, scale_input)
    code = _ue5m3_code_from_clamped_f32(safe_input)
    # The fake path only uses scale_f32; the code byte (with 0xFF for NaN)
    # is kept for a future sm_107f real-quantize kernel that stores it.
    code = tl.where(is_nan, _TL_UE5M3_NAN_BYTE, code)
    scale_f32 = tl.where(is_nan, float("nan"), _ue5m3_code_to_f32(code))
    recip_scale = _TL_FP32_ONE / scale_f32
    return code, scale_f32, recip_scale


@triton.jit
def _nvfp4_ue5m3_scale_and_recip(
    max_abs,
    global_scale,
    HAS_GLOBAL_SCALE: tl.constexpr,
    USE_NVFP4_NO_CLIP_SCALE: tl.constexpr = False,
):
    """Two-level UE5M3 composition; same raw-scale order and single
    correctly-rounded recip division as ``_nvfp4_scale_and_recip``."""
    if HAS_GLOBAL_SCALE:
        global_scale_safe = _safe_global_scale(global_scale)
        code, scale_f32, _ = _compute_nvfp4_ue5m3_scale(
            max_abs * global_scale_safe, USE_NVFP4_NO_CLIP_SCALE
        )
        recip_scale = tl.div_rn(global_scale_safe, scale_f32)
    else:
        code, scale_f32, recip_scale = _compute_nvfp4_ue5m3_scale(
            max_abs, USE_NVFP4_NO_CLIP_SCALE
        )
    return code, scale_f32, recip_scale


@triton.jit
def _compute_block_amax_nonfinite(x, AXIS: tl.constexpr):
    """Block-wise abs-max along ``AXIS`` with ``NaN -> inf`` substitution.

    ``tl.max`` follows IEEE maxNum and drops NaN, which would silently lose the
    NaN-block signal needed for the per-format NaN scale byte. Mapping NaN to inf
    before the reduction preserves the non-finite signal: MXFP8/MXFP4 see
    ``max_abs == inf -> E8M0 byte 0xFF`` and NVFP4 sees it become FP8 E4M3 byte
    0x7F via bitcast. The reduction runs in the caller's tile dtype and returns
    fp32 for the downstream scale math.
    """
    x_abs = tl.abs(x)
    x_abs = tl.minimum(x_abs, float("inf"))
    return tl.max(x_abs, axis=AXIS).to(tl.float32)


@triton.jit
def _compute_mxfp8_e4m3_block_scale(x, axis):
    """Per-block MXFP8 E4M3 scale from an abs'd tile: returns (recip_scale, scale_u8).

    Reduces ``x`` to its per-block amax along ``axis`` then computes the E8M0
    block-scale pair targeting the E4M3 element max, for on-the-fly grouped GEMM
    quantization.
    """
    scale_u8, recip_scale = _compute_e8m0_scale_rceil(
        _compute_block_amax_nonfinite(x, axis),
        _TL_FP8_E4M3_MAX_RECIP,
    )
    return recip_scale, scale_u8


@triton.jit
def _canonicalize_mx_half_range(
    max_abs,
    scale_u8,
    recip_scale,
    FORMAT_ID: tl.constexpr,
):
    """Opt-in half-range E8M0 exponent shift for MX round-trip stability.

    After ``_compute_e8m0_scale_rceil`` the rounded value lands in
    ``(fmt_max/2, fmt_max]``. When it also fits the format's half-range we
    decrement the E8M0 exponent by one to pick the finest scale whose quantized
    values still fit, making ``quantize -> dequantize -> quantize`` bitwise
    stable. The per-format midpoint sets the comparison: E4M3 rounds 232 -> 224
    (inclusive), while E5M2 rounds 30720 up and E2M1 rounds 3.5 up (exclusive).
    Off by default to match the pinned reference rules.
    """
    tl.static_assert(max_abs.dtype == tl.float32, "max_abs must be fp32")
    tl.static_assert(scale_u8.dtype == tl.uint8, "scale_u8 must be uint8")
    tl.static_assert(recip_scale.dtype == tl.float32, "recip_scale must be fp32")
    scaled_max = max_abs * recip_scale
    if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E4M3:
        fits_half_range = scaled_max <= _TL_FP8_E4M3_HALF_RANGE_ROUND_BOUNDARY
    elif FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2:
        fits_half_range = scaled_max < _TL_FP8_E5M2_HALF_RANGE_ROUND_BOUNDARY
    else:
        tl.static_assert(
            FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP4,
            "half-range canonicalization supports MXFP8/MXFP4 only",
        )
        fits_half_range = scaled_max < _TL_FP4_E2M1_HALF_RANGE_ROUND_BOUNDARY
    # Byte-0 guard via recip_scale (recip == 2^127 iff scale_u8 == 0). Equivalent
    # to scale_u8 > 0 since recip = 2^(127 - scale_u8), but avoids a uint8
    # predicate that triggers Triton codegen issues.
    should_shift = (
        (max_abs > _TL_FP32_ZERO)
        & fits_half_range
        & (recip_scale < _TL_E8M0_BYTE0_RECIP)
    )
    scale_u8 = tl.where(should_shift, scale_u8 - 1, scale_u8)
    recip_scale = tl.where(should_shift, recip_scale * _TL_FP32_TWO, recip_scale)
    return scale_u8, recip_scale


@triton.jit
def _compute_scale_and_recip_from_amax(
    amax,  # fp32 [N_BLOCKS] — per-block amax (NaN already mapped to inf).
    FORMAT_ID: tl.constexpr,
    USE_HALF_RANGE_SCALE: tl.constexpr,
    USE_NVFP4_NO_CLIP_SCALE: tl.constexpr = False,
):
    """Per-block scale byte + fp32 reciprocal, shared by all quant kernels.

    Returns ``(scale_u8, recip_scale)`` with a uniform ``(uint8, fp32)`` type
    across formats so it can be one shared ``@triton.jit`` helper: the E8M0
    byte is returned directly for MX formats, and the FP8 E4M3 NVFP4 scale is
    returned bitcast to ``uint8`` (callers store the scale buffer through a
    ``uint8`` view). The quantize + pack step stays in each kernel because its
    output dtype is format-specific (``float8e4nv`` / ``float8e5`` / packed
    ``uint32``) and so cannot share a single return type.
    """
    if (
        FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E4M3
        or FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2
    ):
        target_max_recip: tl.constexpr = _TL_FP8_E4M3_MAX_RECIP
        if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2:
            target_max_recip = _TL_FP8_E5M2_MAX_RECIP
        scale_u8, recip_scale = _compute_e8m0_scale_rceil(amax, target_max_recip)
        if USE_HALF_RANGE_SCALE:
            scale_u8, recip_scale = _canonicalize_mx_half_range(
                amax, scale_u8, recip_scale, FORMAT_ID=FORMAT_ID
            )
        return scale_u8, recip_scale
    if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP4:
        scale_u8, recip_scale = _compute_e8m0_scale_rceil(
            amax,
            _TL_FP4_E2M1_MAX_RECIP,
        )
        if USE_HALF_RANGE_SCALE:
            scale_u8, recip_scale = _canonicalize_mx_half_range(
                amax, scale_u8, recip_scale, FORMAT_ID=FORMAT_ID
            )
        return scale_u8, recip_scale
    scale_fp8, recip_scale = _compute_nvfp4_scale_rtne(amax, USE_NVFP4_NO_CLIP_SCALE)
    scale_u8 = scale_fp8.to(tl.uint8, bitcast=True)
    return scale_u8, recip_scale


@triton.jit
def _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
    amax,
    nvfp4_recip_lut_ptr,
    FORMAT_ID: tl.constexpr,
    USE_HALF_RANGE_SCALE: tl.constexpr,
    USE_NVFP4_RECIP_LUT: tl.constexpr,
    USE_NVFP4_NO_CLIP_SCALE: tl.constexpr = False,
):
    """Compute scale + reciprocal from amax, optionally using the NVFP4 recip LUT."""
    scale_u8, recip = _compute_scale_and_recip_from_amax(
        amax,
        FORMAT_ID=FORMAT_ID,
        USE_HALF_RANGE_SCALE=USE_HALF_RANGE_SCALE,
        USE_NVFP4_NO_CLIP_SCALE=USE_NVFP4_NO_CLIP_SCALE,
    )
    if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_NVFP4 and USE_NVFP4_RECIP_LUT:
        # Optional NVFP4 fast path: read the reciprocal from a precomputed LUT
        # indexed by the FP8 scale byte instead of computing 1 / scale.
        recip = tl.load(nvfp4_recip_lut_ptr + scale_u8.to(tl.int32))
    return scale_u8, recip


@triton.jit
def _clamp_to_target_max(x, recip_scale, target_max):
    """Scale ``x`` by ``recip_scale`` and clamp to ``[-target_max, target_max]``.

    Format-agnostic: ``target_max`` is the element max of the destination dtype
    (FP8 E4M3/E5M2 for MXFP8, FP4 E2M1 for MXFP4/NVFP4). ``propagate_nan=ALL``
    keeps NaN through the clamp, so a block whose ``max_abs`` reached inf or NaN
    ends up with NaN in every quantized element on top of the per-format NaN
    scale byte. Default ``tl.clamp`` NaN behavior is unspecified.
    """
    tl.static_assert(x.dtype == tl.float32, "x must be fp32 before the quantized cast")
    tl.static_assert(recip_scale.dtype == tl.float32, "recip_scale must be fp32")
    return tl.clamp(
        x * recip_scale,
        -target_max,
        target_max,
        propagate_nan=tl.PropagateNan.ALL,
    )


@triton.jit
def _scale_for_fp8_cast(x, recip_scale, target_max):
    if _is_blackwell():
        # Blackwell FP8 conversion is satfinite, so the explicit clamp is
        # redundant on this hot quantization path.
        return x * recip_scale
    return _clamp_to_target_max(x, recip_scale, target_max)


@triton.jit
def _cast_fp8_qdata_for_format_id(
    q_values,
    FORMAT_ID: tl.constexpr,
):
    """Cast already scaled/clamped qdata to the selected FP8 dtype."""
    if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E4M3:
        return q_values.to(tl.float8e4nv)

    else:
        tl.static_assert(
            FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2,
            "FP8 qdata helper supports MXFP8_E4M3 or MXFP8_E5M2 only",
        )
        return q_values.to(tl.float8e5)


@triton.jit
def _apply_block_scaled_quant_producer(
    primary,
    producer_b,
    PRODUCER_ID: tl.constexpr,
    FAST_MATH: tl.constexpr = False,
    CLAMPED: tl.constexpr = False,
    ALPHA: tl.constexpr = 1.702,
    LIMIT: tl.constexpr = 7.0,
):
    """Apply the fused producer (identity or SwiGLU fwd) in fp32.

    Producer math runs in fp32 immediately after the HBM load; the returned
    tensor is the exact value whose block amax and quantized data are emitted.
    ``CLAMPED`` selects the GPT-OSS clamped SwiGLU; its op order mirrors
    the standalone SwiGLU quantization path so the two stay bitwise equal.
    """
    primary = primary.to(tl.float32)
    if PRODUCER_ID == _TL_BLOCK_SCALED_PRODUCER_IDENTITY:
        return primary

    tl.static_assert(
        PRODUCER_ID == _TL_BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
        "simple block-scaled producer supports identity or SwiGLU fwd only",
    )
    producer_b = producer_b.to(tl.float32)
    if CLAMPED:
        g = tl.minimum(primary, LIMIT)
        u = tl.minimum(tl.maximum(producer_b, -LIMIT), LIMIT)
        return g * sigmoid(ALPHA * g, FAST_MATH=FAST_MATH) * (u + 1.0)
    return primary * sigmoid(primary, FAST_MATH=FAST_MATH) * producer_b


@triton.jit
def _apply_block_scaled_quant_dxy_concat_producer(
    dz,
    x,
    y,
    is_dy_tile,
    FAST_MATH: tl.constexpr = False,
    CLAMPED: tl.constexpr = False,
    ALPHA: tl.constexpr = 1.702,
    LIMIT: tl.constexpr = 7.0,
):
    """Apply the concatenated SwiGLU backward producer for a dx or dy tile.

    The concatenated backward emits logical ``[dx, dy]`` without materializing
    it. CTAs never cross the dx/dy boundary.

    Compute both ``dx`` and ``dy`` and select with ``tl.where`` so this producer
    preserves the established deterministic operation order and remains
    bitwise-compatible with the CuTe path.
    """
    dx, dy, _ = _apply_block_scaled_quant_dxy_fwd_producer(
        dz,
        x,
        y,
        FAST_MATH=FAST_MATH,
        CLAMPED=CLAMPED,
        ALPHA=ALPHA,
        LIMIT=LIMIT,
    )
    return tl.where(is_dy_tile, dy, dx)


@triton.jit
def _apply_block_scaled_quant_dxy_fwd_producer(
    dz,
    x,
    y,
    FAST_MATH: tl.constexpr = False,
    CLAMPED: tl.constexpr = False,
    ALPHA: tl.constexpr = 1.702,
    LIMIT: tl.constexpr = 7.0,
):
    dz = dz.to(tl.float32)
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    if CLAMPED:
        # Op for op the bf16 swiglu_bwd kernel, incl. its strict clamp masks.
        g = tl.minimum(x, LIMIT)
        u = tl.minimum(tl.maximum(y, -LIMIT), LIMIT)
        s = sigmoid(ALPHA * g, FAST_MATH=FAST_MATH)
        silu = g * s
        d_silu = s + ALPHA * g * s * (1.0 - s)
        dx = dz * (u + 1.0) * d_silu
        dx = tl.where(x > LIMIT, 0.0, dx)
        dy = dz * silu
        dy = tl.where((y > LIMIT) | (y < -LIMIT), 0.0, dy)
        return dx, dy, silu * (u + 1.0)
    sigmoid_x = sigmoid(x, FAST_MATH=FAST_MATH)
    x_sigmoid = x * sigmoid_x
    dx = dz * y * sigmoid_x * (_TL_FP32_ONE + x - x_sigmoid)
    dy = dz * x * sigmoid_x
    return dx, dy, x_sigmoid * y


@triton.jit
def _load_row1d_tile_with_simple_producer(
    x_ptr,
    producer_b_ptr,
    x_offsets,
    producer_b_offsets,
    load_mask,
    PRODUCER_ID: tl.constexpr,
    FAST_MATH: tl.constexpr = False,
    CLAMPED: tl.constexpr = False,
    ALPHA: tl.constexpr = 1.702,
    LIMIT: tl.constexpr = 7.0,
):
    x = tl.load(x_ptr + x_offsets, mask=load_mask, other=_TL_FP32_ZERO)
    if PRODUCER_ID != _TL_BLOCK_SCALED_PRODUCER_IDENTITY:
        producer_b = tl.load(
            producer_b_ptr + producer_b_offsets,
            mask=load_mask,
            other=_TL_FP32_ZERO,
        )
        return _apply_block_scaled_quant_producer(
            x,
            producer_b,
            PRODUCER_ID=PRODUCER_ID,
            FAST_MATH=FAST_MATH,
            CLAMPED=CLAMPED,
            ALPHA=ALPHA,
            LIMIT=LIMIT,
        )
    return x.to(tl.float32)


# -----------------------------------------------------------------------------
# Packing helpers.
# -----------------------------------------------------------------------------


@triton.jit
def _pack_fp4_e2m1x8_to_u32(x0, x1, x2, x3, x4, x5, x6, x7):
    """Pack eight scaled fp32 values into one uint32 of FP4 E2M1 nibbles.

    Uses the Blackwell ``cvt.rn.satfinite.e2m1x2.f32`` instruction (two lanes per
    byte) and assembles the four result bytes into the output word.
    """
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 byte0;
            .reg .b8 byte1;
            .reg .b8 byte2;
            .reg .b8 byte3;
            cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;
            cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;
            cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;
            cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints="=r,f,f,f,f,f,f,f,f",
        args=[x0, x1, x2, x3, x4, x5, x6, x7],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pack_fp4_e2m1_blocks_to_u32(
    x_scaled,
    N_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Pack scaled E2M1 values into uint32 words.

    Each uint32 contains eight fp4 elements.  The split/interleave sequence
    feeds PTX in the operand order expected by cvt.rn.satfinite.e2m1x2.f32.
    """
    x0, x1 = tl.split(tl.reshape(x_scaled, (N_BLOCKS, BLOCK_SIZE // 2, 2)))
    x_pairs = tl.reshape(
        tl.interleave(x0, x1),
        (N_BLOCKS, BLOCK_SIZE // 8, 2, 2, 2),
    )
    even, odd = tl.split(x_pairs)
    even_lo, even_hi = tl.split(even)
    odd_lo, odd_hi = tl.split(odd)
    v0, v4 = tl.split(even_lo)
    v2, v6 = tl.split(even_hi)
    v1, v5 = tl.split(odd_lo)
    v3, v7 = tl.split(odd_hi)
    return _pack_fp4_e2m1x8_to_u32(v0, v1, v2, v3, v4, v5, v6, v7)


@triton.jit
def _pack_u4x8_row_lanes_to_u32(
    values,
    ROW_GROUPS: tl.constexpr,
    COL_WORDS: tl.constexpr,
):
    even, odd = tl.split(tl.reshape(values, (ROW_GROUPS, COL_WORDS, 4, 2)))
    even_lo, even_hi = tl.split(tl.reshape(even, (ROW_GROUPS, COL_WORDS, 2, 2)))
    odd_lo, odd_hi = tl.split(tl.reshape(odd, (ROW_GROUPS, COL_WORDS, 2, 2)))
    v0, v4 = tl.split(even_lo)
    v2, v6 = tl.split(even_hi)
    v1, v5 = tl.split(odd_lo)
    v3, v7 = tl.split(odd_hi)
    return (
        v0.to(tl.uint32)
        | (v1.to(tl.uint32) << 4)
        | (v2.to(tl.uint32) << 8)
        | (v3.to(tl.uint32) << 12)
        | (v4.to(tl.uint32) << 16)
        | (v5.to(tl.uint32) << 20)
        | (v6.to(tl.uint32) << 24)
        | (v7.to(tl.uint32) << 28)
    )


@triton.jit
def _pack_u8x4_to_u32(
    values,
    N_ROWS: tl.constexpr,
    N_WORDS: tl.constexpr,
):
    """Pack four adjacent scale bytes into one little-endian uint32 word.

    The reshape to ``(..., 2, 2)`` maps ``values[2k + j]`` to ``pairs[..., k,
    j]``, so the first split (last axis ``j``) puts even-index values in ``lo``
    and odd-index in ``hi``; ``v_i`` then holds ``values[i]`` so the shifts place
    byte ``i`` at memory offset ``i``.
    """
    pairs = tl.reshape(values, (N_ROWS, N_WORDS, 2, 2))
    lo, hi = tl.split(pairs)
    v0, v2 = tl.split(lo)
    v1, v3 = tl.split(hi)
    return (
        v0.to(tl.uint32)
        | (v1.to(tl.uint32) << 8)
        | (v2.to(tl.uint32) << 16)
        | (v3.to(tl.uint32) << 24)
    )


@triton.jit
def _pack_fp4_e2m1_blocks_to_u32_rn(
    x_scaled,
    N_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    return _pack_fp4_e2m1_blocks_to_u32(
        x_scaled,
        N_BLOCKS=N_BLOCKS,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# -----------------------------------------------------------------------------
# Store and offset helpers.
# -----------------------------------------------------------------------------


@triton.jit
def _compute_cublas_blocked_scale_offset(
    row,
    col,
    N_COL_BLOCKS,  # runtime-or-constexpr: only used in offset arithmetic
):
    """Linear element offset of a scale factor in the cuBLAS blocked SF layout.

    The layout is atom-major over 128-row x 4-scale-column atoms; within an atom
    rows are grouped by ``row % 32``, each lane owning four 32-row bands x four
    scale columns stored as 16 contiguous entries.
    """
    row_atom = row // _TL_CUBLAS_BLOCKED_ROWS_PER_ATOM
    row_in_atom = row - row_atom * _TL_CUBLAS_BLOCKED_ROWS_PER_ATOM
    col_atom = col // _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    col_in_atom = col - col_atom * _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM

    atom_id = row_atom * N_COL_BLOCKS + col_atom
    atom_base = atom_id * _TL_CUBLAS_BLOCKED_ENTRIES_PER_ATOM

    # Within one atom, rows are first grouped by row % 32.  Each row lane owns
    # four 32-row bands x four scale columns, stored as 16 contiguous entries.
    row_lane = row_in_atom % _TL_CUBLAS_BLOCKED_ROW_LANES
    row_band = row_in_atom // _TL_CUBLAS_BLOCKED_ROW_LANES
    lane_base = row_lane * _TL_CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE

    return (
        atom_base
        + lane_base
        + row_band * _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
        + col_in_atom
    )


@triton.jit
def _compute_row1d_scale_store_offsets(
    row,
    scale_col,
    rows_per_group,
    SCALE_COLS: tl.constexpr,
    N_COL_BLOCKS: tl.constexpr,
    group_output_elems,
    SCALE_FACTOR_LAYOUT: tl.constexpr,
    HAS_GROUPS: tl.constexpr,
):
    """Output offset for a scale byte under the natural or cuBLAS-blocked layout.

    Natural layout is plain row-major. The blocked layout routes through
    ``_compute_cublas_blocked_scale_offset`` per group, offsetting by
    ``group_output_elems`` when ``HAS_GROUPS``.
    """
    if SCALE_FACTOR_LAYOUT == _TL_SCALE_FACTOR_LAYOUT_NATURAL:
        return row * SCALE_COLS + scale_col

    group = tl.full((), 0, tl.int32)
    row_in_group = row
    if HAS_GROUPS:
        group = row // rows_per_group
        row_in_group = row - group * rows_per_group

    group_base = group * group_output_elems
    blocked_offset = _compute_cublas_blocked_scale_offset(
        row_in_group,
        scale_col,
        N_COL_BLOCKS=N_COL_BLOCKS,
    )
    return group_base + blocked_offset


@triton.jit
def _compute_tile2d_scale_store_offsets(
    rows,
    cols,
    row_stride,
    N_COL_BLOCKS,  # runtime-or-constexpr: only used in offset arithmetic
    SCALE_FACTOR_LAYOUT: tl.constexpr,
):
    if SCALE_FACTOR_LAYOUT == _TL_SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED:
        return _compute_cublas_blocked_scale_offset(rows, cols, N_COL_BLOCKS).to(
            tl.int64
        )
    return rows.to(tl.int64) * row_stride + cols


@triton.jit
def _store_row1d_fp4_qdata_packed_u32_row_major(
    q_ptr,
    packed,
    rows,
    q_word_col,
    valid_blocks,
    q_cols: tl.constexpr,
):
    """Store packed uint32 FP4 words row-major, as words or as individual bytes.

    Uses a uint32 pointer and word columns when ``q_cols`` is a multiple of 4;
    otherwise falls back to extracting and storing the four bytes individually.
    """
    if q_cols % 4 == 0:
        q_words = q_ptr.to(tl.pointer_type(tl.uint32))
        q_word_cols: tl.constexpr = q_cols // 4
        q_offsets = rows[:, None] * q_word_cols + q_word_col
        tl.store(
            q_words + q_offsets,
            packed,
            mask=valid_blocks[:, None] & (q_word_col < q_word_cols),
            cache_modifier=".cs",
        )
    else:
        byte_lanes = tl.arange(0, 4)
        q_byte_col = q_word_col[:, :, None] * 4 + byte_lanes[None, None, :]
        q_offsets = rows[:, None, None] * q_cols + q_byte_col
        q_bytes = ((packed[:, :, None] >> (byte_lanes[None, None, :] * 8)) & 0xFF).to(
            tl.uint8
        )
        tl.store(
            q_ptr + q_offsets,
            q_bytes,
            mask=valid_blocks[:, None, None] & (q_byte_col < q_cols),
            cache_modifier=".cs",
        )


@triton.jit
def _store_tile2d_scales_cublas_blocked(
    # Input.
    tile_scale_u8,
    rows,
    s0_block_rows,
    col_start,
    n_rows,
    # Output.
    s0_ptr,
    s1_ptr,
    s0_row_stride,
    s1_row_stride,
    # Shape & addressing.
    N_COLS: tl.constexpr,
    S0_N_COL_BLOCKS,  # runtime: M-derived (ragged rows) — keeps one compile across shapes
    S1_N_COL_BLOCKS: tl.constexpr,
    # Tiling.
    M_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
    SF_VEC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(K_TILE == 4, "packed tile-scale store expects four K blocks")

    # s0 (axis-0 BLOCKED) is emitted in MMA-A frame: atom-MN spans K
    # positions (broadcast V K-elements per tile scale), atom-K spans M/V
    # scale positions. The atom-K direction has at most M_TILE scales per
    # tile (typically 1 in fused producer paths), so u32-packing 4 atom-K
    # cols isn't possible — fall back to byte stores. Broadcast each
    # (m_v, k_v) tile scale across SF_VEC K-elements so the store writes
    # one byte per atom-MN row.
    s0_cols = col_start + tl.arange(0, BLOCK_K)
    s0_scale_broadcast = tl.reshape(
        tl.reshape(tile_scale_u8, (M_TILE, K_TILE, 1))
        + tl.zeros((M_TILE, K_TILE, SF_VEC), tl.uint8),
        (M_TILE, BLOCK_K),
    )
    s0_offsets = _compute_tile2d_scale_store_offsets(
        s0_cols[None, :],
        s0_block_rows[:, None],
        s0_row_stride,
        S0_N_COL_BLOCKS,
        _TL_SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED,
    )
    tl.store(
        s0_ptr + s0_offsets,
        s0_scale_broadcast,
        mask=(s0_block_rows[:, None] * SF_VEC < n_rows) & (s0_cols[None, :] < N_COLS),
        cache_modifier=".cs",
    )

    # s1 broadcasts the same tile scales across SF_VEC rows. K_TILE is four,
    # so the four adjacent scale columns again fit one packed u32 per row.
    s1_scale_u8x4 = tl.reshape(
        tl.reshape(tile_scale_u8, (M_TILE, 1, K_TILE))
        + tl.zeros((M_TILE, SF_VEC, K_TILE), tl.uint8),
        (BLOCK_M, 1, K_TILE),
    )
    s1_packed = _pack_u8x4_to_u32(
        s1_scale_u8x4,
        N_ROWS=BLOCK_M,
        N_WORDS=1,
    )
    s1_col4 = col_start // SF_VEC + tl.arange(0, 1) * 4
    s1_offsets4 = _compute_tile2d_scale_store_offsets(
        rows[:, None],
        s1_col4[None, :],
        s1_row_stride,
        S1_N_COL_BLOCKS,
        _TL_SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED,
    )
    tl.store(
        s1_ptr.to(tl.pointer_type(tl.uint32)) + s1_offsets4 // 4,
        s1_packed,
        mask=(rows[:, None] < n_rows) & (s1_col4[None, :] < N_COLS // SF_VEC),
        cache_modifier=".cs",
    )


@triton.jit
def _store_tile2d_qdata_col_major_fp8(
    # Input.
    q_values,
    rows,
    cols,
    row_mask,
    col_mask,
    # Output.
    q_ptr,
    q_row_stride,
    # Quant / format config.
    FORMAT_ID: tl.constexpr,
):
    rows = rows.to(tl.int64)
    cols = cols.to(tl.int64)
    q_offsets = cols[:, None] * q_row_stride + rows[None, :]
    q_fp8 = _cast_fp8_qdata_for_format_id(
        q_values,
        FORMAT_ID=FORMAT_ID,
    )
    tl.store(
        q_ptr + q_offsets,
        tl.permute(q_fp8, (1, 0)),
        mask=col_mask[:, None] & row_mask[None, :],
        cache_modifier=".cs",
    )


@triton.jit
def _store_tile2d_qdata_fp4_both_layouts(
    # Input.
    q_values,
    rows,
    row_mask,
    col_mask,
    row_start,
    col_start,
    # Output.
    q0_ptr,
    q1_ptr,
    q0_row_stride,
    q1_row_stride,
    # Shape & addressing.
    N_COLS: tl.constexpr,
    # Tiling.
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D FP4 uses one scale for both qdata layouts. Convert to FP4 once,
    # store q1 row-major, then repack nibbles into q0's col-major layout.
    packed = _pack_fp4_e2m1_blocks_to_u32(
        q_values,
        N_BLOCKS=BLOCK_M,
        BLOCK_SIZE=BLOCK_K,
    )
    q1_words = q1_ptr.to(tl.pointer_type(tl.uint32))
    col_word_offsets = col_start // 8 + tl.arange(0, BLOCK_K // 8)
    col_mask_words = col_mask.reshape(BLOCK_K // 8, 8)
    valid_col_word = tl.max(col_mask_words.to(tl.int32), axis=1) != 0
    rows = rows.to(tl.int64)
    q1_offsets = rows[:, None] * (q1_row_stride // 4) + col_word_offsets[None, :]
    tl.store(
        q1_words + q1_offsets,
        packed,
        mask=row_mask[:, None] & valid_col_word[None, :],
        cache_modifier=".cs",
    )

    q0_words = q0_ptr.to(tl.pointer_type(tl.uint32))
    row_groups = row_start // 8 + tl.arange(0, BLOCK_M // 8)
    col_words = tl.arange(0, BLOCK_K // 8)
    row_mask_words = tl.max(row_mask.reshape(BLOCK_M // 8, 8).to(tl.int32), axis=1) != 0
    for nibble_id in tl.static_range(0, 8):
        # Extract one K-lane nibble from each row-major word, then pack eight
        # row lanes into the corresponding col-major output word.
        nibble = ((packed >> (nibble_id * 4)) & 0xF).to(tl.uint8)
        lanes = tl.permute(
            tl.reshape(nibble, (BLOCK_M // 8, 8, BLOCK_K // 8)),
            (0, 2, 1),
        )
        transposed = _pack_u4x8_row_lanes_to_u32(
            lanes,
            ROW_GROUPS=BLOCK_M // 8,
            COL_WORDS=BLOCK_K // 8,
        )
        out_cols = col_start + col_words * 8 + nibble_id
        out_cols_mask = out_cols[None, :] < N_COLS
        out_cols = out_cols.to(tl.int64)
        q0_offsets = out_cols[None, :] * (q0_row_stride // 4) + row_groups[:, None]
        tl.store(
            q0_words + q0_offsets,
            transposed,
            mask=row_mask_words[:, None] & out_cols_mask,
            cache_modifier=".cs",
        )


@triton.jit
def _store_tile2d_qdata_col_major_fp4(
    # Input.
    q_values,
    cols,
    row_mask,
    col_mask,
    row_start,
    # Output.
    q_ptr,
    q_row_stride,
    N_COLS: tl.constexpr,
    # Shape & addressing.
    # Tiling.
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # FP4 col-major: pack two adjacent N (row) codes per byte. Buffer shape
    # is (cols, rows // 2). Pack eight row codes into one u32 store per
    # column to match the fast axis=1 FP4 path's store width.
    q_t = tl.permute(q_values, (1, 0))
    packed = _pack_fp4_e2m1_blocks_to_u32_rn(
        q_t,
        N_BLOCKS=BLOCK_K,
        BLOCK_SIZE=BLOCK_M,
    )
    q_words = q_ptr.to(tl.pointer_type(tl.uint32))
    word_offsets = row_start // 8 + tl.arange(0, BLOCK_M // 8)
    row_mask_words = tl.max(row_mask.reshape(BLOCK_M // 8, 8).to(tl.int32), axis=1) != 0
    cols = cols.to(tl.int64)
    word_offsets = cols[:, None] * (q_row_stride // 4) + word_offsets[None, :]
    tl.store(
        q_words + word_offsets,
        packed,
        mask=col_mask[:, None] & row_mask_words[None, :],
        cache_modifier=".cs",
    )


@triton.jit
def _store_tile2d_qdata_row_major(
    # Input.
    q_values,
    rows,
    cols,
    row_mask,
    col_mask,
    col_start,
    # Output.
    q_ptr,
    q_row_stride,
    # Quant / format config.
    FORMAT_ID: tl.constexpr,
    # Shape & addressing.
    # Tiling.
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # FP8 paths cast and store the full BLOCK_M x BLOCK_K tile. FP4 packs two
    # E2M1 codes per byte along K, halving the K-axis output stride.
    rows = rows.to(tl.int64)
    if (
        FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E4M3
        or FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2
    ):
        q_offsets = rows[:, None] * q_row_stride + cols[None, :]
        q_fp8 = _cast_fp8_qdata_for_format_id(
            q_values,
            FORMAT_ID=FORMAT_ID,
        )
        tl.store(
            q_ptr + q_offsets,
            q_fp8,
            mask=row_mask[:, None] & col_mask[None, :],
            cache_modifier=".cs",
        )
    else:
        # FP4: pack eight adjacent K-axis codes into one u32 output word.
        # The byte layout is still two codes per byte; the wider store removes
        # the generic path's previous byte-store bottleneck.
        packed = _pack_fp4_e2m1_blocks_to_u32_rn(
            q_values,
            N_BLOCKS=BLOCK_M,
            BLOCK_SIZE=BLOCK_K,
        )
        q_words = q_ptr.to(tl.pointer_type(tl.uint32))
        word_offsets = col_start // 8 + tl.arange(0, BLOCK_K // 8)
        col_mask_words = col_mask.reshape(BLOCK_K // 8, 8)
        valid_col_word = tl.max(col_mask_words.to(tl.int32), axis=1) != 0
        q_offsets = rows[:, None] * (q_row_stride // 4) + word_offsets[None, :]
        tl.store(
            q_words + q_offsets,
            packed,
            mask=row_mask[:, None] & valid_col_word[None, :],
            cache_modifier=".cs",
        )


@triton.jit
def _store_row1d_qdata_row_major(
    x_values,
    recip_scale,
    rows,
    scale_col,
    elem_mask,
    valid_blocks,
    q_ptr,
    FORMAT_ID: tl.constexpr,
    Q_COLS: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if (
        FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E4M3
        or FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2
    ):
        target_max: tl.constexpr = _TL_FP8_E4M3_MAX
        if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2:
            target_max = _TL_FP8_E5M2_MAX
        q = _scale_for_fp8_cast(x_values, recip_scale[:, None], target_max)
        elem_offsets = tl.arange(0, BLOCK_SIZE)
        q_offsets = rows[:, None] * Q_COLS + scale_col[:, None] * BLOCK_SIZE
        q_fp8 = _cast_fp8_qdata_for_format_id(
            q,
            FORMAT_ID=FORMAT_ID,
        )
        tl.store(
            q_ptr + q_offsets + elem_offsets[None, :],
            q_fp8,
            mask=elem_mask,
            cache_modifier=".cs",
        )
    else:
        x_scaled = x_values * recip_scale[:, None]
        packed = _pack_fp4_e2m1_blocks_to_u32_rn(
            x_scaled,
            N_BLOCKS=N_BLOCKS,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        word_offsets = tl.arange(0, BLOCK_SIZE // 8)
        q_word_col = scale_col[:, None] * (BLOCK_SIZE // 8) + word_offsets[None, :]
        _store_row1d_fp4_qdata_packed_u32_row_major(
            q_ptr,
            packed,
            rows,
            q_word_col,
            valid_blocks,
            Q_COLS,
        )


@triton.jit
def _store_row1d_cublas_blocked_qdata_row_major_fp8(
    x_blocks,
    recip_scale,
    row_offsets,
    col_offsets,
    mask,
    q_ptr,
    FORMAT_ID: tl.constexpr,
    Q_COLS: tl.constexpr,
    M_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
):
    target_max: tl.constexpr = _TL_FP8_E4M3_MAX
    if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2:
        target_max = _TL_FP8_E5M2_MAX
    q = _scale_for_fp8_cast(x_blocks, recip_scale[:, :, None], target_max)
    q = q.reshape(M_TILE, K_TILE)
    q_offsets = row_offsets * Q_COLS + col_offsets
    q_fp8 = _cast_fp8_qdata_for_format_id(
        q,
        FORMAT_ID=FORMAT_ID,
    )
    tl.store(
        q_ptr + q_offsets,
        q_fp8,
        mask=mask,
        cache_modifier=".cs",
    )


# =============================================================================
# Shared host utilities (host-side; none run on the GPU).
# =============================================================================


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


_INT32_OFFSET_MAX = 2**31 - 1


def _compute_max_strided_offset(
    rows: int, cols: int, row_stride: int, col_stride: int
) -> int:
    if rows <= 0 or cols <= 0:
        return 0
    return (rows - 1) * abs(row_stride) + (cols - 1) * abs(col_stride)


def _needs_64bit_offsets(*offsets: int) -> bool:
    return any(offset > _INT32_OFFSET_MAX for offset in offsets)


@dataclass(frozen=True)
class _GlobalScaleLaunchInfo:
    tensor: torch.Tensor
    rank: GlobalScaleRank
    stride0: int
    stride1: int
    rows_per_group: int
    max_offset: int


def _global_scale_launch_info(
    global_scale: torch.Tensor | None,
    *,
    rows: int,
    device: torch.device,
    require_block_constant: bool = False,
) -> _GlobalScaleLaunchInfo | None:
    if global_scale is None:
        return None
    if global_scale.device != device:
        raise ValueError("global_scale must be on the input device")

    scale = global_scale.to(torch.float32)
    if scale.numel() == 1:
        return _GlobalScaleLaunchInfo(
            tensor=scale.reshape(1),
            rank=GlobalScaleRank.SCALAR,
            stride0=0,
            stride1=0,
            rows_per_group=1,
            max_offset=0,
        )
    if (
        require_block_constant
        and scale.numel() == rows
        and (scale.ndim == 1 or (scale.ndim == 2 and scale.stride(1) != 0))
    ):
        raise NotImplementedError(
            "global_scale must be per-tensor or per-expert for blocks spanning rows"
        )
    if scale.ndim == 1 and scale.numel() == rows:
        return _GlobalScaleLaunchInfo(
            tensor=scale,
            rank=GlobalScaleRank.ONE_D,
            stride0=scale.stride(0),
            stride1=0,
            rows_per_group=rows,
            max_offset=(rows - 1) * abs(scale.stride(0)) if rows > 0 else 0,
        )
    if scale.ndim == 2 and scale.numel() == rows:
        return _GlobalScaleLaunchInfo(
            tensor=scale,
            rank=GlobalScaleRank.TWO_D,
            stride0=scale.stride(0),
            stride1=scale.stride(1),
            rows_per_group=scale.shape[1],
            max_offset=_compute_max_strided_offset(
                scale.shape[0],
                scale.shape[1],
                scale.stride(0),
                scale.stride(1),
            ),
        )
    raise ValueError(
        "global_scale must be scalar, flattened per-row, or row-shaped with "
        f"{rows} elements; got shape={tuple(global_scale.shape)}"
    )


def _global_scale_kernel_kwargs(
    global_scale_info: _GlobalScaleLaunchInfo | None,
    *,
    fallback_ptr: torch.Tensor,
) -> dict[str, object]:
    """Kernel kwargs for the global-scale argument block, with a null-object
    default when no global scale is given.

    ``fallback_ptr`` only keeps the pointer argument well-typed in the
    disabled case; it is never dereferenced (``HAS_GLOBAL_SCALE=False``
    compiles the loads out).
    """
    if global_scale_info is None:
        return {
            "global_scale_ptr": fallback_ptr,
            "global_scale_stride0": 0,
            "global_scale_stride1": 0,
            "HAS_GLOBAL_SCALE": False,
            "GLOBAL_SCALE_RANK": 0,
            "GLOBAL_SCALE_ROWS_PER_GROUP": 1,
        }
    return {
        "global_scale_ptr": global_scale_info.tensor,
        "global_scale_stride0": global_scale_info.stride0,
        "global_scale_stride1": global_scale_info.stride1,
        "HAS_GLOBAL_SCALE": True,
        "GLOBAL_SCALE_RANK": int(global_scale_info.rank),
        "GLOBAL_SCALE_ROWS_PER_GROUP": global_scale_info.rows_per_group,
    }


def _padded_cublas_blocked_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    return (
        ceil_div(rows, CUBLAS_BLOCKED_ROWS_PER_ATOM) * CUBLAS_BLOCKED_ROWS_PER_ATOM,
        ceil_div(cols, CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM)
        * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
    )


@dataclass(frozen=True)
class _Row1dTileKernelConfig:
    row_tile: int
    scale_col_tile: int
    num_warps: int


@dataclass(frozen=True)
class _Row1dCublasBlockedKernelConfig:
    m_tile: int
    block_size: int
    num_warps: int


@dataclass(frozen=True)
class _Tile2dKernelConfig:
    max_m_tile: int
    max_k_tile: int
    num_warps: int


def _is_mxfp8_format_id(format_id: int) -> bool:
    return format_id in _MXFP8_FORMAT_IDS


def _is_fp4_format_id(format_id: int) -> bool:
    return format_id in _FP4_FORMAT_IDS


def _get_sf_vec_size_for_format_id(format_id: int) -> int:
    if format_id in NVFP4_VARIANT_FORMAT_IDS:
        return 16
    if format_id in (
        BlockScaledFormatId.MXFP4.value,
        *_MXFP8_FORMAT_IDS,
        *_MXFP6_FORMAT_IDS,
    ):
        return 32
    raise ValueError(f"Unsupported format_id={format_id}")


def _get_qdata_dtype_for_format_id(format_id: int) -> torch.dtype:
    if format_id == BlockScaledFormatId.MXFP8_E4M3.value:
        return torch.float8_e4m3fn
    if format_id == BlockScaledFormatId.MXFP8_E5M2.value:
        return torch.float8_e5m2
    if _is_fp4_format_id(format_id):
        return torch.float4_e2m1fn_x2
    raise ValueError(f"Unsupported format_id={format_id}")


def _get_scale_dtype_for_format_id(format_id: int) -> torch.dtype:
    if format_id == BlockScaledFormatId.NVFP4.value:
        return torch.float8_e4m3fn
    if format_id in (*_MXFP8_FORMAT_IDS, BlockScaledFormatId.MXFP4.value):
        return torch.float8_e8m0fnu
    raise ValueError(f"Unsupported format_id={format_id}")


def _convert_torch_dtype_to_tl_dtype(dtype: torch.dtype):
    """Map a torch dtype to the corresponding Triton constexpr dtype.

    Used by the ``ZEROCOPY_GATHER`` producer path to cast the int64 remote
    address into a typed pointer for the gather load.
    """
    match dtype:
        case torch.bfloat16:
            return tl.bfloat16
        case torch.float16:
            return tl.float16
        case torch.float32:
            return tl.float32
        case _:
            raise NotImplementedError(f"No tl-dtype mapping for {dtype}")


def _empty_quantized_qdata(
    shape: tuple[int, ...] | torch.Size,
    *,
    format_id: int,
    device: torch.device,
) -> torch.Tensor:
    shape = tuple(shape)
    if _is_fp4_format_id(format_id):
        q_u8 = torch.empty(
            (*shape[:-1], shape[-1] // 2), dtype=torch.uint8, device=device
        )
        return q_u8.view(torch.float4_e2m1fn_x2)
    return torch.empty(
        shape,
        dtype=_get_qdata_dtype_for_format_id(format_id),
        device=device,
    )


def _get_row1d_tile_kernel_config(format_id: int) -> _Row1dTileKernelConfig:
    if format_id == BlockScaledFormatId.MXFP4.value:
        return _Row1dTileKernelConfig(row_tile=1, scale_col_tile=32, num_warps=4)
    if format_id == BlockScaledFormatId.NVFP4.value:
        return _Row1dTileKernelConfig(row_tile=1, scale_col_tile=64, num_warps=4)
    if _is_mxfp8_format_id(format_id):
        return _Row1dTileKernelConfig(row_tile=4, scale_col_tile=32, num_warps=4)
    raise ValueError(f"Unsupported format_id={format_id}")


def _get_row1d_cublas_blocked_kernel_config(
    format_id: int,
) -> _Row1dCublasBlockedKernelConfig | None:
    if format_id == BlockScaledFormatId.NVFP4.value:
        # m_tile=32, num_warps=1 was the sweep winner after the inf/NaN
        # correctness fixes and the reference-compatible refactor. The kernel became
        # compute-bound from the per-element NaN check, so smaller CTAs with
        # fewer warps maximise SM-level parallelism.
        return _Row1dCublasBlockedKernelConfig(m_tile=32, block_size=16, num_warps=1)
    if format_id == BlockScaledFormatId.MXFP4.value:
        return _Row1dCublasBlockedKernelConfig(m_tile=128, block_size=32, num_warps=8)
    if _is_mxfp8_format_id(format_id):
        return _Row1dCublasBlockedKernelConfig(m_tile=64, block_size=32, num_warps=8)
    return None


def _producer_has_tile2d_math(producer_id: int) -> bool:
    return producer_id not in (
        BlockScaledProducer.IDENTITY.value,
        BlockScaledProducer.ZEROCOPY_GATHER.value,
    )


def _get_nvfp4_tile2d_kernel_config(
    *,
    src_dtype: torch.dtype,
    has_axis0: bool,
    has_axis1: bool,
    is_two_d: bool,
) -> _Tile2dKernelConfig:
    if src_dtype == torch.float32:
        return _Tile2dKernelConfig(max_m_tile=1, max_k_tile=1, num_warps=8)
    if is_two_d:
        return _Tile2dKernelConfig(max_m_tile=2, max_k_tile=8, num_warps=8)
    if not has_axis0:
        return _Tile2dKernelConfig(max_m_tile=1, max_k_tile=16, num_warps=1)
    return _Tile2dKernelConfig(
        max_m_tile=8 if has_axis1 else 16,
        max_k_tile=1,
        num_warps=4,
    )


def _get_mxfp4_tile2d_kernel_config(
    *,
    has_axis0: bool,
    has_axis1: bool,
    is_two_d: bool,
    producer_id: int,
) -> _Tile2dKernelConfig:
    if is_two_d:
        return _Tile2dKernelConfig(max_m_tile=1, max_k_tile=8, num_warps=4)
    if not has_axis0:
        return _Tile2dKernelConfig(max_m_tile=1, max_k_tile=4, num_warps=1)
    if has_axis1 and _producer_has_tile2d_math(producer_id):
        return _Tile2dKernelConfig(max_m_tile=1, max_k_tile=1, num_warps=1)
    return _Tile2dKernelConfig(
        max_m_tile=2 if has_axis1 else 4,
        max_k_tile=2,
        num_warps=1,
    )


def _get_mxfp8_tile2d_kernel_config(
    *,
    has_axis0: bool,
    has_axis1: bool,
    is_two_d: bool,
    producer_id: int,
) -> _Tile2dKernelConfig:
    if is_two_d:
        return _Tile2dKernelConfig(max_m_tile=2, max_k_tile=4, num_warps=4)
    if not has_axis0:
        return _Tile2dKernelConfig(max_m_tile=4, max_k_tile=4, num_warps=8)
    if not has_axis1:
        if producer_id == BlockScaledProducer.SWIGLU_FWD.value:
            return _Tile2dKernelConfig(max_m_tile=2, max_k_tile=1, num_warps=2)
        return _Tile2dKernelConfig(max_m_tile=2, max_k_tile=4, num_warps=2)
    if _producer_has_tile2d_math(producer_id):
        warps = 2 if producer_id == BlockScaledProducer.SWIGLU_BWD_DXY.value else 1
        return _Tile2dKernelConfig(max_m_tile=1, max_k_tile=1, num_warps=warps)
    return _Tile2dKernelConfig(max_m_tile=1, max_k_tile=4, num_warps=1)


def _get_tile2d_kernel_config(
    *,
    format_id: int,
    src_dtype: torch.dtype,
    has_axis0: bool,
    has_axis1: bool,
    is_two_d: bool,
    producer_id: int,
) -> _Tile2dKernelConfig:
    """Return the tile limits used by the 2-D tile kernel launch.

    The callsite shrinks ``max_{m,k}_tile`` when shape divisibility requires it.
    The values here are tuned for the output kind first; producer math only
    changes both1d, where SwiGLU needs smaller CTAs than load-only producers.
    """
    if format_id == BlockScaledFormatId.NVFP4.value:
        return _get_nvfp4_tile2d_kernel_config(
            src_dtype=src_dtype,
            has_axis0=has_axis0,
            has_axis1=has_axis1,
            is_two_d=is_two_d,
        )

    if format_id == BlockScaledFormatId.MXFP4.value:
        return _get_mxfp4_tile2d_kernel_config(
            has_axis0=has_axis0,
            has_axis1=has_axis1,
            is_two_d=is_two_d,
            producer_id=producer_id,
        )

    if _is_mxfp8_format_id(format_id):
        return _get_mxfp8_tile2d_kernel_config(
            has_axis0=has_axis0,
            has_axis1=has_axis1,
            is_two_d=is_two_d,
            producer_id=producer_id,
        )

    raise ValueError(f"Unsupported format_id={format_id}")


def _can_flatten_outer_tma_dims(tensor: torch.Tensor) -> bool:
    return tensor.ndim != 3 or tensor.stride(0) == tensor.stride(1) * tensor.shape[1]


def _has_tma_load_layout(tensor: torch.Tensor) -> bool:
    return tensor.stride(-1) == 1 and _can_flatten_outer_tma_dims(tensor)


def _use_tma_swiglu_fwd_load_for_tile2d(
    *,
    x: torch.Tensor,
    producer_b: torch.Tensor | None,
    format_id: int,
    producer_id: int,
    is_col1d: bool,
) -> bool:
    if producer_b is None:
        return False
    return (
        producer_id == BlockScaledProducer.SWIGLU_FWD.value
        and is_col1d
        and _is_mxfp8_format_id(format_id)
        and x.dtype == torch.bfloat16
        and _has_tma_load_layout(x)
        and _has_tma_load_layout(producer_b)
        and supports_tma()
    )


def _make_tma_descriptor(tensor: torch.Tensor, block_shape: list[int]) -> object:
    tensor_2d = tensor.view(-1, tensor.shape[-1]) if tensor.ndim == 3 else tensor
    return TensorDescriptor(
        tensor_2d,
        list(tensor_2d.shape),
        list(tensor_2d.stride()),
        block_shape,
    )


def _use_nvfp4_recip_lut_for_tile2d(
    *,
    format_id: int,
    is_cublas_blocked_layout: bool,
) -> bool:
    return is_cublas_blocked_layout and format_id == BlockScaledFormatId.NVFP4.value


def _compute_row1d_scale_output_shape(
    x_shape: torch.Size,
    scale_cols: int,
    layout: int,
    *,
    is_a: bool,
) -> tuple[tuple[int, ...], int, bool]:
    if layout == ScaleFactorLayoutId.NATURAL.value:
        rows_per_group = x_shape[-2] if len(x_shape) >= 2 else 1
        return (*x_shape[:-1], scale_cols), rows_per_group, len(x_shape) == 3

    if len(x_shape) == 2:
        rows_per_group = x_shape[0]
        groups = 1
        has_groups = False
    elif len(x_shape) == 3 and not is_a:
        groups, rows_per_group, _ = x_shape
        has_groups = True
    else:
        raise ValueError(
            "CUBLAS_BLOCKED quantization expects a 2D SFA tensor or a 3D "
            f"SFB tensor with is_a=False; got shape={tuple(x_shape)}, is_a={is_a}"
        )

    padded_rows_per_group, padded_scale_cols = _padded_cublas_blocked_scale_shape(
        rows_per_group,
        scale_cols,
    )
    return (
        (
            groups * padded_rows_per_group,
            padded_scale_cols,
        ),
        rows_per_group,
        has_groups,
    )


# =============================================================================
# Row-1D quantization (row1d_tile)
# =============================================================================


@triton.jit(
    repr=make_dtype_repr(
        _build_quantize_spec_name(
            "natural",
            "SCALE_FACTOR_LAYOUT",
            "BLOCK_SIZE",
            producer_id_constexpr="PRODUCER_ID",
        ),
        [],
    ),
)
def _triton_quantize_blockscaled_row1d_tile(  # noqa: C901, TR001 -- format-specific tiles are selected explicitly
    # Input.
    x_ptr,
    global_scale_ptr,
    global_scale_stride0,
    global_scale_stride1,
    x_row_stride,
    x_col_stride,
    # Producer.
    producer_b_ptr,
    producer_b_row_stride,
    producer_b_col_stride,
    PRODUCER_ID: tl.constexpr,
    # Output.
    q_ptr,
    scale_ptr,
    SCALE_FACTOR_LAYOUT: tl.constexpr,
    # Quant / format config.
    FORMAT_ID: tl.constexpr,
    USE_HALF_RANGE_SCALE: tl.constexpr,
    FAST_MATH: tl.constexpr,
    CLAMPED: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    # Shape & addressing. n_rows / rows_per_group / group_output_elems are runtime,
    # not constexpr: they scale with the per-expert row count, so as constexpr they
    # recompiled the quant kernel for every distinct MoE routing shape. Used only in
    # masks and group-index arithmetic, never a tile/array shape.
    n_rows,
    N_COLS: tl.constexpr,
    Q_COLS: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    rows_per_group,
    N_COL_BLOCKS: tl.constexpr,
    group_output_elems,
    HAS_GROUPS: tl.constexpr,
    USE_64BIT_OFFSETS: tl.constexpr,
    HAS_GLOBAL_SCALE: tl.constexpr,
    GLOBAL_SCALE_RANK: tl.constexpr,
    GLOBAL_SCALE_ROWS_PER_GROUP: tl.constexpr,
    # Tiling.
    GRID_COL_TILES: tl.constexpr,
    ROW_TILE: tl.constexpr,
    SCALE_COL_TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    n_scale_blocks: tl.constexpr = ROW_TILE * SCALE_COL_TILE
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Each program covers a row x scale-column tile in the generic layout.
    # block_ids flatten that 2D tile so scale math stays vectorized.
    block_ids = tl.arange(0, n_scale_blocks)
    rows = pid_row * ROW_TILE + block_ids // SCALE_COL_TILE
    scale_col = pid_col * SCALE_COL_TILE + block_ids % SCALE_COL_TILE
    valid_blocks = (rows < n_rows) & (scale_col < SCALE_COLS)

    # Map logical (row, scale_col) to the requested scale-factor layout before
    # widening row addresses below.
    scale_offsets = _compute_row1d_scale_store_offsets(
        rows,
        scale_col,
        rows_per_group=rows_per_group,
        SCALE_COLS=SCALE_COLS,
        N_COL_BLOCKS=N_COL_BLOCKS,
        group_output_elems=group_output_elems,
        SCALE_FACTOR_LAYOUT=SCALE_FACTOR_LAYOUT,
        HAS_GROUPS=HAS_GROUPS,
    )
    if USE_64BIT_OFFSETS:
        rows = rows.to(tl.int64)
    global_scale = _load_global_scale_rows(
        global_scale_ptr,
        global_scale_stride0,
        global_scale_stride1,
        rows,
        rows < n_rows,
        HAS_GLOBAL_SCALE=HAS_GLOBAL_SCALE,
        GLOBAL_SCALE_RANK=GLOBAL_SCALE_RANK,
        GLOBAL_SCALE_ROWS_PER_GROUP=GLOBAL_SCALE_ROWS_PER_GROUP,
    )

    # Load one semantic scale vector per block; invalid rows/columns read zero.
    elem_offsets = tl.arange(0, BLOCK_SIZE)
    x_offsets = scale_col[:, None] * BLOCK_SIZE + elem_offsets[None, :]
    load_mask = valid_blocks[:, None] & (x_offsets < N_COLS)
    x = _load_row1d_tile_with_simple_producer(
        x_ptr,
        producer_b_ptr,
        rows[:, None] * x_row_stride + x_offsets * x_col_stride,
        rows[:, None] * producer_b_row_stride + x_offsets * producer_b_col_stride,
        load_mask,
        PRODUCER_ID=PRODUCER_ID,
        FAST_MATH=FAST_MATH,
        CLAMPED=CLAMPED,
        ALPHA=ALPHA,
        LIMIT=LIMIT,
    )

    # Per-vector block amax with NaN-preserving reduction.
    max_abs = _compute_block_amax_nonfinite(x, AXIS=1)
    if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_NVFP4 and HAS_GLOBAL_SCALE:
        scale_fp8, recip_scale = _nvfp4_scale_and_recip(
            max_abs,
            global_scale,
            HAS_GLOBAL_SCALE=HAS_GLOBAL_SCALE,
            USE_NVFP4_NO_CLIP_SCALE=False,
        )
        scale_u8 = scale_fp8.to(tl.uint8, bitcast=True)
    else:
        scale_u8, recip_scale = _compute_scale_and_recip_from_amax(
            max_abs,
            FORMAT_ID=FORMAT_ID,
            USE_HALF_RANGE_SCALE=USE_HALF_RANGE_SCALE,
            USE_NVFP4_NO_CLIP_SCALE=False,
        )
    tl.store(
        scale_ptr + scale_offsets,
        scale_u8,
        mask=valid_blocks,
        cache_modifier=".cs",
    )
    _store_row1d_qdata_row_major(
        x_values=x,
        recip_scale=recip_scale,
        rows=rows,
        scale_col=scale_col,
        elem_mask=load_mask,
        valid_blocks=valid_blocks,
        q_ptr=q_ptr,
        FORMAT_ID=FORMAT_ID,
        Q_COLS=Q_COLS,
        SCALE_COLS=SCALE_COLS,
        N_BLOCKS=n_scale_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# =============================================================================
# Given-grid MXFP8 quantization (dim0, shared 32x32 E8M0 scale)
# =============================================================================


@triton.jit
def _triton_quantize_blockscaled_tile2d_given_grid(  # noqa: TR001
    # Input.
    x_ptr,
    x_row_stride,
    x_col_stride,
    # Caller-supplied natural E8M0 scale grid (one byte per 32x32 tile).
    e8m0_grid_ptr,
    e8m0_grid_row_stride,
    e8m0_grid_col_stride,
    # Output (FP8 data only; no scale frames).
    q_ptr,
    # Quant / format config.
    FORMAT_ID: tl.constexpr,
    # Shape & addressing.
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    Q_COLS: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    # Tiling.
    GRID_COL_TILES: tl.constexpr,
    ROW_TILE: tl.constexpr,
    SCALE_COL_TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    ROW_OFFSET: tl.constexpr = 0,
):
    """Quantize weights to MXFP8 against a caller-supplied E8M0 grid.

    One E8M0 byte per 32x32 tile, broadcast over all 32 rows (``row // ROW_BLOCK``)
    and cols. Emits FP8 data only.
    """
    n_scale_blocks: tl.constexpr = ROW_TILE * SCALE_COL_TILE
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    block_ids = tl.arange(0, n_scale_blocks)
    rows = pid_row * ROW_TILE + block_ids // SCALE_COL_TILE
    scale_col = pid_col * SCALE_COL_TILE + block_ids % SCALE_COL_TILE
    valid_blocks = (rows < N_ROWS) & (scale_col < SCALE_COLS)

    # ``ROW_OFFSET`` shifts local rows to their GLOBAL row so a non-32-aligned
    # shard indexes the correct tiles of the global grid.
    grid_row = (rows + ROW_OFFSET) // ROW_BLOCK
    grid_offsets = grid_row * e8m0_grid_row_stride + scale_col * e8m0_grid_col_stride
    scale_u8 = tl.load(e8m0_grid_ptr + grid_offsets, mask=valid_blocks, other=0)
    recip_scale = _recip_from_e8m0_byte(scale_u8)

    rows = rows.to(tl.int64)
    elem_offsets = tl.arange(0, BLOCK_SIZE)
    x_offsets = scale_col[:, None] * BLOCK_SIZE + elem_offsets[None, :]
    load_mask = valid_blocks[:, None] & (x_offsets < N_COLS)
    x = tl.load(
        x_ptr + rows[:, None] * x_row_stride + x_offsets * x_col_stride,
        mask=load_mask,
        other=_TL_FP32_ZERO,
    ).to(tl.float32)

    _store_row1d_qdata_row_major(
        x_values=x,
        recip_scale=recip_scale,
        rows=rows,
        scale_col=scale_col,
        elem_mask=load_mask,
        valid_blocks=valid_blocks,
        q_ptr=q_ptr,
        FORMAT_ID=FORMAT_ID,
        Q_COLS=Q_COLS,
        SCALE_COLS=SCALE_COLS,
        N_BLOCKS=n_scale_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit(
    repr=make_dtype_repr(
        _build_quantize_spec_name(
            "blocked",
            None,
            "BLOCK_SIZE",
            producer_id_constexpr="PRODUCER_ID",
        ),
        [],
    ),
)
def _triton_quantize_blockscaled_row1d_cublas_blocked(  # noqa: C901, TR001 -- constexpr branches are fully specialized by format
    # Input.
    x_ptr,
    global_scale_ptr,
    global_scale_stride0,
    global_scale_stride1,
    x_row_stride,
    x_col_stride,
    # Producer.
    producer_b_ptr,
    producer_b_row_stride,
    producer_b_col_stride,
    PRODUCER_ID: tl.constexpr,
    # Output.
    q_ptr,
    scale_ptr,
    Q_COLS: tl.constexpr,
    # Quant / format config.
    FORMAT_ID: tl.constexpr,
    USE_HALF_RANGE_SCALE: tl.constexpr,
    FAST_MATH: tl.constexpr,
    CLAMPED: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    # Shape & addressing. n_rows / rows_per_group / group_output_elems are runtime,
    # not constexpr (row-count-derived) so the kernel is not recompiled per routing
    # shape; used only in masks and group-index arithmetic.
    n_rows,
    N_COLS: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    rows_per_group,
    N_COL_BLOCKS: tl.constexpr,
    group_output_elems,
    HAS_GROUPS: tl.constexpr,
    USE_64BIT_OFFSETS: tl.constexpr,
    HAS_GLOBAL_SCALE: tl.constexpr,
    GLOBAL_SCALE_RANK: tl.constexpr,
    GLOBAL_SCALE_ROWS_PER_GROUP: tl.constexpr,
    # Tiling.
    GRID_COL_TILES: tl.constexpr,
    M_TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize four scale columns and directly emit one cuBLAS SF atom tile.

    ``BLOCK_SIZE`` is the semantic SF vector width: 16 for NVFP4 and 32 for
    MX formats. Triton specializes this kernel with constexprs, so the FP8,
    MXFP4, and NVFP4 paths compile down to separate dead-code-eliminated
    device kernels while sharing one source implementation.

    TMA-load fast path was evaluated for this kernel and gave a 1.8x
    regression on the dist_moe row-1D fwd shape (T=98304, K=3584,
    M_TILE=64, k_tile=128 → 8 KB tile). The plain ``tl.load`` is already
    coalesced and the per-CTA descriptor cost outweighs the TMA savings
    at this tile size, so we kept the original path.
    """
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)
    k_tile: tl.constexpr = BLOCK_SIZE * _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    n_scale_blocks: tl.constexpr = M_TILE * _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM

    # One CTA covers four adjacent scale columns and an M tile of rows, matching
    # one cuBLAS-blocked scale atom in the output scale tensor.
    row_offsets = pid_row * M_TILE + tl.arange(0, M_TILE)[:, None]
    col_offsets = pid_col * k_tile + tl.arange(0, k_tile)[None, :]
    mask = (row_offsets < n_rows) & (col_offsets < N_COLS)
    if USE_64BIT_OFFSETS:
        row_offsets = row_offsets.to(tl.int64)
        col_offsets = col_offsets.to(tl.int64)
    global_scale = _load_global_scale_rows(
        global_scale_ptr,
        global_scale_stride0,
        global_scale_stride1,
        row_offsets,
        row_offsets < n_rows,
        HAS_GLOBAL_SCALE=HAS_GLOBAL_SCALE,
        GLOBAL_SCALE_RANK=GLOBAL_SCALE_RANK,
        GLOBAL_SCALE_ROWS_PER_GROUP=GLOBAL_SCALE_ROWS_PER_GROUP,
    )
    x = _load_row1d_tile_with_simple_producer(
        x_ptr,
        producer_b_ptr,
        row_offsets * x_row_stride + col_offsets * x_col_stride,
        row_offsets * producer_b_row_stride + col_offsets * producer_b_col_stride,
        mask,
        PRODUCER_ID=PRODUCER_ID,
        FAST_MATH=FAST_MATH,
        CLAMPED=CLAMPED,
        ALPHA=ALPHA,
        LIMIT=LIMIT,
    )

    x_blocks = x.reshape(M_TILE, _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM, BLOCK_SIZE)
    max_abs = _compute_block_amax_nonfinite(x_blocks, AXIS=2)

    block_ids = tl.arange(0, n_scale_blocks)
    rows = pid_row * M_TILE + block_ids // _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    scale_col = (
        pid_col * _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
        + block_ids % _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    )
    valid_blocks = (rows < n_rows) & (scale_col < SCALE_COLS)
    scale_offsets = _compute_row1d_scale_store_offsets(
        rows,
        scale_col,
        rows_per_group=rows_per_group,
        SCALE_COLS=SCALE_COLS,
        N_COL_BLOCKS=N_COL_BLOCKS,
        group_output_elems=group_output_elems,
        SCALE_FACTOR_LAYOUT=_TL_SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED,
        HAS_GROUPS=HAS_GROUPS,
    )
    if USE_64BIT_OFFSETS:
        rows = rows.to(tl.int64)

    if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_NVFP4 and HAS_GLOBAL_SCALE:
        scale_fp8, recip_scale = _nvfp4_scale_and_recip(
            max_abs,
            global_scale,
            HAS_GLOBAL_SCALE=HAS_GLOBAL_SCALE,
            USE_NVFP4_NO_CLIP_SCALE=False,
        )
        scale_u8 = scale_fp8.to(tl.uint8, bitcast=True)
    else:
        scale_u8, recip_scale = _compute_scale_and_recip_from_amax(
            max_abs,
            FORMAT_ID=FORMAT_ID,
            USE_HALF_RANGE_SCALE=USE_HALF_RANGE_SCALE,
            USE_NVFP4_NO_CLIP_SCALE=False,
        )
    tl.store(
        scale_ptr + scale_offsets,
        scale_u8.reshape(n_scale_blocks),
        mask=valid_blocks,
        cache_modifier=".cs",
    )
    if (
        FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E4M3
        or FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2
    ):
        _store_row1d_cublas_blocked_qdata_row_major_fp8(
            x_blocks=x_blocks,
            recip_scale=recip_scale,
            row_offsets=row_offsets,
            col_offsets=col_offsets,
            mask=mask,
            q_ptr=q_ptr,
            FORMAT_ID=FORMAT_ID,
            Q_COLS=Q_COLS,
            M_TILE=M_TILE,
            K_TILE=k_tile,
        )
    else:
        if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_NVFP4:
            # NVFP4 reciprocal recomputed at point of use. The fp8 reciprocal
            # carried from the shared top-level scale+recip helper is corrupted
            # once it co-exists with the bitcast scale byte across the scale
            # ``tl.store`` (Triton miscompile, NVFP4-only; the e8m0 MX reciprocal is
            # unaffected). Mirrors trunk's in-branch recompute.
            if HAS_GLOBAL_SCALE:
                _, recip_scale = _nvfp4_scale_and_recip(
                    max_abs,
                    global_scale,
                    HAS_GLOBAL_SCALE=HAS_GLOBAL_SCALE,
                    USE_NVFP4_NO_CLIP_SCALE=False,
                )
            else:
                _, recip_scale = _compute_nvfp4_scale_rtne(max_abs, False)
        _store_row1d_qdata_row_major(
            x_values=tl.reshape(x_blocks, (n_scale_blocks, BLOCK_SIZE)),
            recip_scale=tl.reshape(recip_scale, (n_scale_blocks,)),
            rows=rows,
            scale_col=scale_col,
            elem_mask=tl.reshape(
                tl.reshape(
                    mask, (M_TILE, _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM, BLOCK_SIZE)
                ),
                (n_scale_blocks, BLOCK_SIZE),
            ),
            valid_blocks=valid_blocks,
            q_ptr=q_ptr,
            FORMAT_ID=FORMAT_ID,
            Q_COLS=Q_COLS,
            SCALE_COLS=SCALE_COLS,
            N_BLOCKS=n_scale_blocks,
            BLOCK_SIZE=BLOCK_SIZE,
        )


# =============================================================================
# Multi-axis quantization (tile2d): axis=0 / axis=1 / 2D-tile reduction.
# =============================================================================


@triton.jit(
    repr=make_dtype_repr(
        _build_quantize_spec_name(
            "",
            "SCALE_FACTOR_LAYOUT",
            "SF_VEC",
            axis_mask_constexpr="AXIS_MASK",
            scale_reduction_constexpr="SCALE_REDUCTION",
            producer_id_constexpr="PRODUCER_ID",
        ),
        [],
    ),
)
def _triton_quantize_blockscaled_tile2d(  # noqa: C901, TR001 -- format/mode branches are constexpr
    # Input.
    x_ptr,  # ZEROCOPY_GATHER: int64* [n_rows] remote addresses. Else: source tile.
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    # Producer.
    producer_b_ptr,
    producer_c_ptr,
    x_desc,
    y_desc,
    producer_b_batch_stride,
    producer_b_row_stride,
    producer_b_col_stride,
    producer_c_batch_stride,
    producer_c_row_stride,
    producer_c_col_stride,
    PRODUCER_ID: tl.constexpr,
    # Output.
    q0_ptr,
    q1_ptr,
    s0_ptr,
    s1_ptr,
    q0_batch_stride,
    q0_row_stride,
    q1_batch_stride,
    q1_row_stride,
    s0_batch_stride,
    s0_row_stride,
    s1_batch_stride,
    s1_row_stride,
    SCALE_FACTOR_LAYOUT: tl.constexpr,
    global_scale_ptr,
    global_scale_stride0,
    global_scale_stride1,
    # Quant / format config.
    nvfp4_recip_lut_ptr,
    FORMAT_ID: tl.constexpr,
    AXIS_MASK: tl.constexpr,
    SCALE_REDUCTION: tl.constexpr,
    USE_HALF_RANGE_SCALE: tl.constexpr,
    USE_NVFP4_RECIP_LUT: tl.constexpr,
    FAST_MATH: tl.constexpr,
    CLAMPED: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    SRC_DTYPE: tl.constexpr,  # tl.bfloat16 / tl.float16 / tl.float32 — needed for
    # ZEROCOPY_GATHER pointer casts; ignored for non-gather modes.
    USE_TMA_LOAD: tl.constexpr,
    local_rank,
    WORLD_SIZE: tl.constexpr,
    # Shape & addressing.
    n_rows,
    N_COLS: tl.constexpr,
    SOURCE_COLS: tl.constexpr,
    S0_N_COL_BLOCKS,  # runtime: M-derived (ragged rows) — keeps one compile across shapes
    S1_N_COL_BLOCKS: tl.constexpr,
    HAS_GLOBAL_SCALE: tl.constexpr,
    GLOBAL_SCALE_RANK: tl.constexpr,
    GLOBAL_SCALE_ROWS_PER_GROUP: tl.constexpr,
    # Tiling.
    SF_VEC: tl.constexpr,
    M_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
    # Runtime, not constexpr: used for gather-rank index math. As a constexpr
    # its value (grid_m ~ n_rows/SF_VEC) recompiled
    # the kernel for every distinct row count.
    grid_m_tiles,
    GRID_K_TILES: tl.constexpr,
):
    # Each CTA covers (M_TILE * SF_VEC) rows x (K_TILE * SF_VEC) cols of input
    # — M_TILE x K_TILE scale-blocks. n-d reshape lets the single tl.load feed
    # all per-block reductions, keeping the kernel one big vectorized memory
    # transaction.
    #
    # Grid axis assignment: the row-tile count is unbounded (grows with the
    # gathered token count, e.g. dist_moe activations >2M rows), so it maps to
    # grid axis 0 (the only axis with a 2^31 extent). The col-tile count and the
    # outer batch dim (G) are both small (feature dim / expert count), so they
    # take grid axes 1 and 2, which CUDA caps at 65535. 2D callers launch G=1 and
    # pass zero batch strides.
    BLOCK_M: tl.constexpr = M_TILE * SF_VEC
    BLOCK_K: tl.constexpr = K_TILE * SF_VEC

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_g = tl.program_id(2)
    if PRODUCER_ID == _TL_BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER and WORLD_SIZE > 1:
        rank_start_m = (grid_m_tiles // WORLD_SIZE) * local_rank
        pid_m += rank_start_m
        if pid_m >= grid_m_tiles:
            pid_m -= grid_m_tiles
    pid_g = pid_g.to(tl.int64)
    x_ptr = x_ptr + pid_g * x_batch_stride
    producer_b_ptr = producer_b_ptr + pid_g * producer_b_batch_stride
    producer_c_ptr = producer_c_ptr + pid_g * producer_c_batch_stride
    q0_ptr = q0_ptr + pid_g * q0_batch_stride
    q1_ptr = q1_ptr + pid_g * q1_batch_stride
    s0_ptr = s0_ptr + pid_g * s0_batch_stride
    s1_ptr = s1_ptr + pid_g * s1_batch_stride
    row_start = pid_m * BLOCK_M
    col_start = pid_k * BLOCK_K

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_K)
    row_mask = rows < n_rows
    col_mask = cols < N_COLS
    rows = rows.to(tl.int64)
    cols = cols.to(tl.int64)
    global_scale_row_ids = rows
    global_scale_row_mask = row_mask
    if SCALE_REDUCTION == _TL_SCALE_REDUCTION_TWO_D:
        global_scale_row_ids = row_start + tl.arange(0, M_TILE) * SF_VEC
        global_scale_row_mask = global_scale_row_ids < n_rows
    global_scale_rows = _load_global_scale_rows(
        global_scale_ptr,
        global_scale_stride0,
        global_scale_stride1,
        pid_g * n_rows + global_scale_row_ids,
        global_scale_row_mask,
        HAS_GLOBAL_SCALE=HAS_GLOBAL_SCALE,
        GLOBAL_SCALE_RANK=GLOBAL_SCALE_RANK,
        GLOBAL_SCALE_ROWS_PER_GROUP=GLOBAL_SCALE_ROWS_PER_GROUP,
    )

    if (
        SCALE_REDUCTION == _TL_SCALE_REDUCTION_ONE_D
        and AXIS_MASK == _TL_AXIS_MASK_M
        and SCALE_FACTOR_LAYOUT == _TL_SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED
        and PRODUCER_ID == _TL_BLOCK_SCALED_PRODUCER_IDENTITY
        and (
            FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_NVFP4
            or FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP4
        )
    ):
        # Pure FP4 axis=0 writes qdata in column-major order. Load in that
        # order too, so packing no longer needs a BLOCK_M x BLOCK_K transpose.
        x_offsets_t = rows[None, :] * x_row_stride + cols[:, None] * x_col_stride
        x_t = tl.load(
            x_ptr + x_offsets_t,
            mask=col_mask[:, None] & row_mask[None, :],
            other=_TL_FP32_ZERO,
            cache_modifier=".cg",
        ).to(tl.float32)
        x_abs_t = tl.abs(x_t)
        x_abs_t = tl.where(x_abs_t != x_abs_t, float("inf"), x_abs_t)
        if HAS_GLOBAL_SCALE:
            global_scale_t = _safe_global_scale(global_scale_rows[None, :])
            x_abs_t = x_abs_t * global_scale_t
            x_t = x_t * global_scale_t

        x_groups_t = tl.reshape(x_abs_t, (BLOCK_K, M_TILE, SF_VEC))
        col_amax_t = tl.max(x_groups_t, axis=2)
        scale_u8_t, recip_t = (
            _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                col_amax_t,
                nvfp4_recip_lut_ptr,
                FORMAT_ID,
                USE_HALF_RANGE_SCALE,
                USE_NVFP4_RECIP_LUT,
                False,
            )
        )
        recip_groups_t = tl.reshape(recip_t, (BLOCK_K, M_TILE, 1)) + tl.zeros(
            (BLOCK_K, M_TILE, SF_VEC), tl.float32
        )
        q_t = x_t * tl.reshape(recip_groups_t, (BLOCK_K, BLOCK_M))

        packed = _pack_fp4_e2m1_blocks_to_u32_rn(
            q_t,
            N_BLOCKS=BLOCK_K,
            BLOCK_SIZE=BLOCK_M,
        )
        q_words = q0_ptr.to(tl.pointer_type(tl.uint32))
        word_offsets = row_start // 8 + tl.arange(0, BLOCK_M // 8)
        row_mask_words = (
            tl.max(row_mask.reshape(BLOCK_M // 8, 8).to(tl.int32), axis=1) != 0
        )
        q_offsets = cols[:, None] * (q0_row_stride // 4) + word_offsets[None, :]
        tl.store(
            q_words + q_offsets,
            packed,
            mask=col_mask[:, None] & row_mask_words[None, :],
            cache_modifier=".cs",
        )

        s0_block_rows = pid_m * M_TILE + tl.arange(0, M_TILE)
        # Axis-0 BLOCKED in MMA-A frame: atom-MN spans K positions
        # (``cols``), atom-K spans M/V scale positions (``s0_block_rows``).
        s0_offsets_t = _compute_tile2d_scale_store_offsets(
            cols[:, None],
            s0_block_rows[None, :],
            s0_row_stride,
            S0_N_COL_BLOCKS,
            SCALE_FACTOR_LAYOUT,
        )
        tl.store(
            s0_ptr + s0_offsets_t,
            scale_u8_t,
            mask=col_mask[:, None] & (s0_block_rows[None, :] * SF_VEC < n_rows),
            cache_modifier=".cs",
        )
        return

    x_offsets = rows[:, None] * x_row_stride + cols[None, :] * x_col_stride
    if PRODUCER_ID == _TL_BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY:
        # Logical output is [dx, dy]. Both halves read the same source column
        # from dz/x/y, then the producer selects the requested gradient.
        src_cols = tl.where(cols < SOURCE_COLS, cols, cols - SOURCE_COLS)
        src_mask = (
            row_mask[:, None] & col_mask[None, :] & (src_cols[None, :] < SOURCE_COLS)
        )
        dz = tl.load(
            x_ptr + rows[:, None] * x_row_stride + src_cols[None, :] * x_col_stride,
            mask=src_mask,
            other=_TL_FP32_ZERO,
        )
        x_saved = tl.load(
            producer_b_ptr
            + rows[:, None] * producer_b_row_stride
            + src_cols[None, :] * producer_b_col_stride,
            mask=src_mask,
            other=_TL_FP32_ZERO,
        )
        y_saved = tl.load(
            producer_c_ptr
            + rows[:, None] * producer_c_row_stride
            + src_cols[None, :] * producer_c_col_stride,
            mask=src_mask,
            other=_TL_FP32_ZERO,
        )
        x = _apply_block_scaled_quant_dxy_concat_producer(
            dz,
            x_saved,
            y_saved,
            col_start >= SOURCE_COLS,
            FAST_MATH=FAST_MATH,
            CLAMPED=CLAMPED,
            ALPHA=ALPHA,
            LIMIT=LIMIT,
        )
    elif PRODUCER_ID == _TL_BLOCK_SCALED_PRODUCER_SWIGLU_FWD:
        if USE_TMA_LOAD:
            x_raw = x_desc.load([row_start, col_start])
            y_raw = y_desc.load([row_start, col_start])
        else:
            load_mask = row_mask[:, None] & col_mask[None, :]
            x_raw = tl.load(x_ptr + x_offsets, mask=load_mask, other=_TL_FP32_ZERO)
            y_raw = tl.load(
                producer_b_ptr
                + rows[:, None] * producer_b_row_stride
                + cols[None, :] * producer_b_col_stride,
                mask=load_mask,
                other=_TL_FP32_ZERO,
            )
        x = _apply_block_scaled_quant_producer(
            x_raw,
            y_raw,
            PRODUCER_ID=PRODUCER_ID,
            FAST_MATH=FAST_MATH,
            CLAMPED=CLAMPED,
            ALPHA=ALPHA,
            LIMIT=LIMIT,
        )
    elif PRODUCER_ID == _TL_BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER:
        # ZEROCOPY_GATHER reuses ``x_ptr`` as an ``int64*`` array of remote
        # row addresses (the caller passes ``gather_ptrs`` in the ``x``
        # slot — see ``quantize_block_scaled_axes`` validation). Each
        # output row's source data lives at ``x_ptr[row]`` in a peer's
        # symm-mem buffer. Build a per-row pointer tensor and let Triton
        # issue gather loads. Sentinel pointer ``0`` (padding row produced
        # by ``dist_dispatch_routing`` under ``m_multiple_of``) is masked
        # to ``other=_TL_FP32_ZERO`` so the output row is quantized as
        # zeros — matches the host-side ``zerocopy_gather`` padding
        # contract bitwise.
        gather_ptrs_ptr = x_ptr.to(tl.pointer_type(tl.int64))
        row_addrs = tl.load(gather_ptrs_ptr + rows, mask=row_mask, other=0)
        is_padding_row = row_addrs == 0
        row_ptrs = row_addrs.to(tl.pointer_type(SRC_DTYPE))
        align: tl.constexpr = 16 // (SRC_DTYPE.primitive_bitwidth // 8)
        row_ptrs = tl.multiple_of(row_ptrs, align)
        # Per-row pointer points at the row's first element; add column
        # offsets in element units (Triton scales by ``elem_size``).
        addrs_2d = row_ptrs[:, None] + cols[None, :]
        load_mask = row_mask[:, None] & col_mask[None, :] & ~is_padding_row[:, None]
        x = tl.load(addrs_2d, mask=load_mask, other=_TL_FP32_ZERO).to(tl.float32)
    else:
        x = tl.load(
            x_ptr + x_offsets,
            mask=row_mask[:, None] & col_mask[None, :],
            other=_TL_FP32_ZERO,
        ).to(tl.float32)
    x_abs = tl.abs(x)
    x_abs = tl.where(x_abs != x_abs, float("inf"), x_abs)
    if HAS_GLOBAL_SCALE and SCALE_REDUCTION != _TL_SCALE_REDUCTION_TWO_D:
        global_scale_2d = _safe_global_scale(global_scale_rows[:, None])
        x_abs = x_abs * global_scale_2d
        x = x * global_scale_2d

    if SCALE_REDUCTION == _TL_SCALE_REDUCTION_TWO_D:
        x_4d = tl.reshape(x_abs, (M_TILE, SF_VEC, K_TILE, SF_VEC))
        tile_max = tl.max(tl.max(x_4d, axis=3), axis=1)
        if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_NVFP4 and HAS_GLOBAL_SCALE:
            tile_scale_fp8, tile_recip = _nvfp4_scale_and_recip(
                tile_max,
                global_scale_rows[:, None],
                HAS_GLOBAL_SCALE=HAS_GLOBAL_SCALE,
                USE_NVFP4_NO_CLIP_SCALE=False,
            )
            tile_scale_u8 = tile_scale_fp8.to(tl.uint8, bitcast=True)
        else:
            tile_scale_u8, tile_recip = (
                _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                    tile_max,
                    nvfp4_recip_lut_ptr,
                    FORMAT_ID,
                    USE_HALF_RANGE_SCALE,
                    False,
                    False,
                )
            )
        recip_4d = tl.reshape(tile_recip, (M_TILE, 1, K_TILE, 1)) + tl.zeros(
            (M_TILE, SF_VEC, K_TILE, SF_VEC), tl.float32
        )
        recip = tl.reshape(recip_4d, (BLOCK_M, BLOCK_K))
        q_values = x * recip
        if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP4:
            _store_tile2d_qdata_fp4_both_layouts(
                q0_ptr=q0_ptr,
                q1_ptr=q1_ptr,
                q_values=q_values,
                rows=rows,
                row_mask=row_mask,
                col_mask=col_mask,
                row_start=row_start,
                col_start=col_start,
                N_COLS=N_COLS,
                q0_row_stride=q0_row_stride,
                q1_row_stride=q1_row_stride,
                BLOCK_M=BLOCK_M,
                BLOCK_K=BLOCK_K,
            )
        else:
            _store_tile2d_qdata_row_major(
                q_ptr=q1_ptr,
                q_values=q_values,
                rows=rows,
                cols=cols,
                row_mask=row_mask,
                col_mask=col_mask,
                col_start=col_start,
                q_row_stride=q1_row_stride,
                BLOCK_M=BLOCK_M,
                BLOCK_K=BLOCK_K,
                FORMAT_ID=FORMAT_ID,
            )
            if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_NVFP4:
                _store_tile2d_qdata_col_major_fp4(
                    q_ptr=q0_ptr,
                    q_values=q_values,
                    cols=cols,
                    row_mask=row_mask,
                    col_mask=col_mask,
                    row_start=row_start,
                    q_row_stride=q0_row_stride,
                    batch_idx=pid_g,
                    n_rows=n_rows,
                    N_COLS=N_COLS,
                    BLOCK_M=BLOCK_M,
                    BLOCK_K=BLOCK_K,
                )
        s0_block_rows = pid_m * M_TILE + tl.arange(0, M_TILE)
        if (
            SCALE_FACTOR_LAYOUT == _TL_SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED
            and K_TILE == 4
            and (
                FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E4M3
                or FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2
            )
        ):
            _store_tile2d_scales_cublas_blocked(
                s0_ptr=s0_ptr,
                s1_ptr=s1_ptr,
                tile_scale_u8=tile_scale_u8,
                rows=rows,
                s0_block_rows=s0_block_rows,
                col_start=col_start,
                n_rows=n_rows,
                s0_row_stride=s0_row_stride,
                s1_row_stride=s1_row_stride,
                N_COLS=N_COLS,
                S0_N_COL_BLOCKS=S0_N_COL_BLOCKS,
                S1_N_COL_BLOCKS=S1_N_COL_BLOCKS,
                M_TILE=M_TILE,
                K_TILE=K_TILE,
                SF_VEC=SF_VEC,
                BLOCK_M=BLOCK_M,
                BLOCK_K=BLOCK_K,
            )
            return
        s0_col_idx_2d = tl.reshape(cols, (K_TILE, SF_VEC))
        s0_scale_3d = tl.reshape(tile_scale_u8, (M_TILE, K_TILE, 1)) + tl.zeros(
            (M_TILE, K_TILE, SF_VEC), tl.uint8
        )
        if SCALE_FACTOR_LAYOUT == _TL_SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED:
            # MMA-A frame: atom-MN spans K positions, atom-K spans M/V.
            s0_offsets_3d = _compute_tile2d_scale_store_offsets(
                s0_col_idx_2d[None, :, :],
                s0_block_rows[:, None, None],
                s0_row_stride,
                S0_N_COL_BLOCKS,
                SCALE_FACTOR_LAYOUT,
            )
        else:
            s0_offsets_3d = _compute_tile2d_scale_store_offsets(
                s0_block_rows[:, None, None],
                s0_col_idx_2d[None, :, :],
                s0_row_stride,
                S0_N_COL_BLOCKS,
                SCALE_FACTOR_LAYOUT,
            )
        s0_block_in_bounds = s0_block_rows[:, None, None] * SF_VEC < n_rows
        s0_col_in_bounds = s0_col_idx_2d[None, :, :] < N_COLS
        tl.store(
            s0_ptr + s0_offsets_3d,
            s0_scale_3d,
            mask=s0_block_in_bounds & s0_col_in_bounds,
            cache_modifier=".cs",
        )
        s1_block_cols = pid_k * K_TILE + tl.arange(0, K_TILE)
        s1_row_idx_2d = tl.reshape(rows, (M_TILE, SF_VEC))
        s1_scale_3d = tl.reshape(tile_scale_u8, (M_TILE, 1, K_TILE)) + tl.zeros(
            (M_TILE, SF_VEC, K_TILE), tl.uint8
        )
        s1_offsets_3d = _compute_tile2d_scale_store_offsets(
            s1_row_idx_2d[:, :, None],
            s1_block_cols[None, None, :],
            s1_row_stride,
            S1_N_COL_BLOCKS,
            SCALE_FACTOR_LAYOUT,
        )
        s1_row_in_bounds = s1_row_idx_2d[:, :, None] < n_rows
        s1_block_in_bounds = s1_block_cols[None, None, :] * SF_VEC < N_COLS
        tl.store(
            s1_ptr + s1_offsets_3d,
            s1_scale_3d,
            mask=s1_row_in_bounds & s1_block_in_bounds,
            cache_modifier=".cs",
        )
        return

    if AXIS_MASK & _TL_AXIS_MASK_K:
        x_3d = tl.reshape(x_abs, (BLOCK_M, K_TILE, SF_VEC))
        row_amax = tl.max(x_3d, axis=2)
        row_scale_u8, row_recip = (
            _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                row_amax,
                nvfp4_recip_lut_ptr,
                FORMAT_ID,
                USE_HALF_RANGE_SCALE,
                False,
                False,
            )
        )
        # Store the scale byte before computing qdata so ``row_scale_u8`` is
        # dead before the fp32 reciprocal is used (see the axis-M note: the
        # co-live bitcast scale byte + fp8 reciprocal miscompiles on this
        # Triton build).
        s1_block_cols = pid_k * K_TILE + tl.arange(0, K_TILE)
        s1_offsets = _compute_tile2d_scale_store_offsets(
            rows[:, None],
            s1_block_cols[None, :],
            s1_row_stride,
            S1_N_COL_BLOCKS,
            SCALE_FACTOR_LAYOUT,
        )
        s1_block_in_bounds = s1_block_cols[None, :] * SF_VEC < N_COLS
        tl.store(
            s1_ptr + s1_offsets,
            row_scale_u8,
            mask=row_mask[:, None] & s1_block_in_bounds,
            cache_modifier=".cs",
        )
        if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_NVFP4:
            _, row_recip = _compute_nvfp4_scale_rtne(row_amax, False)
        recip_3d = tl.reshape(row_recip, (BLOCK_M, K_TILE, 1)) + tl.zeros(
            (BLOCK_M, K_TILE, SF_VEC), tl.float32
        )
        recip = tl.reshape(recip_3d, (BLOCK_M, BLOCK_K))
        q1_values = x * recip
        _store_tile2d_qdata_row_major(
            q_ptr=q1_ptr,
            q_values=q1_values,
            rows=rows,
            cols=cols,
            row_mask=row_mask,
            col_mask=col_mask,
            col_start=col_start,
            q_row_stride=q1_row_stride,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            FORMAT_ID=FORMAT_ID,
        )

    if AXIS_MASK & _TL_AXIS_MASK_M:
        x_3d_n = tl.reshape(x_abs, (M_TILE, SF_VEC, BLOCK_K))
        col_amax = tl.max(x_3d_n, axis=1)
        col_scale_u8, col_recip = (
            _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                col_amax,
                nvfp4_recip_lut_ptr,
                FORMAT_ID,
                USE_HALF_RANGE_SCALE,
                USE_NVFP4_RECIP_LUT,
                False,
            )
        )
        # Store the scale byte before computing qdata so ``col_scale_u8`` (the
        # bitcast fp8->uint8 scale value) is dead by the time the fp32
        # reciprocal is used below. Keeping the bitcast scale byte co-live with
        # a reciprocal derived from the same fp8 scale miscompiles on this
        # Triton build in the col-major store path (the reciprocal is
        # corrupted, producing saturated FP4 codes); storing early frees it.
        s0_block_rows = pid_m * M_TILE + tl.arange(0, M_TILE)
        s0_block_in_bounds = s0_block_rows[:, None] * SF_VEC < n_rows
        if SCALE_FACTOR_LAYOUT == _TL_SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED:
            # Axis-0 BLOCKED is emitted in MMA-A frame: atom-MN spans K
            # positions, atom-K spans M/V scale positions. Swap the
            # offset args (cols/M_v) compared to the NATURAL path. u32-packing
            # 4 adjacent K-elements doesn't help here — the contiguous-in-atom
            # direction is now M/V (typically M_TILE=1 in fused-producer paths),
            # so byte stores are the right primitive.
            s0_offsets = _compute_tile2d_scale_store_offsets(
                cols[None, :],
                s0_block_rows[:, None],
                s0_row_stride,
                S0_N_COL_BLOCKS,
                SCALE_FACTOR_LAYOUT,
            )
            tl.store(
                s0_ptr + s0_offsets,
                col_scale_u8,
                mask=s0_block_in_bounds & col_mask[None, :],
                cache_modifier=".cs",
            )
        else:
            s0_offsets = _compute_tile2d_scale_store_offsets(
                s0_block_rows[:, None],
                cols[None, :],
                s0_row_stride,
                S0_N_COL_BLOCKS,
                SCALE_FACTOR_LAYOUT,
            )
            tl.store(
                s0_ptr + s0_offsets,
                col_scale_u8,
                mask=s0_block_in_bounds & col_mask[None, :],
                cache_modifier=".cs",
            )
        if FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_NVFP4:
            # Direct reciprocal (== the precomputed LUT value, 1/fp8_scale),
            # recomputed here so it is decoupled from the stored bitcast scale
            # byte — both the direct and LUT reciprocals corrupt when co-live
            # with that byte in the col-major store path on this Triton build.
            _, col_recip = _compute_nvfp4_scale_rtne(col_amax, False)
        recip_3d_n = tl.reshape(col_recip, (M_TILE, 1, BLOCK_K)) + tl.zeros(
            (M_TILE, SF_VEC, BLOCK_K), tl.float32
        )
        recip_n = tl.reshape(recip_3d_n, (BLOCK_M, BLOCK_K))
        q0_values = x * recip_n
        if (
            FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_NVFP4
            or FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP4
        ):
            _store_tile2d_qdata_col_major_fp4(
                q_ptr=q0_ptr,
                q_values=q0_values,
                cols=cols,
                row_mask=row_mask,
                col_mask=col_mask,
                row_start=row_start,
                q_row_stride=q0_row_stride,
                N_COLS=N_COLS,
                BLOCK_M=BLOCK_M,
                BLOCK_K=BLOCK_K,
            )
        else:
            _store_tile2d_qdata_col_major_fp8(
                q_ptr=q0_ptr,
                q_values=q0_values,
                rows=rows,
                cols=cols,
                row_mask=row_mask,
                col_mask=col_mask,
                q_row_stride=q0_row_stride,
                FORMAT_ID=FORMAT_ID,
            )


@triton.jit
def _store_swiglu_fused_axis1_mxfp8(
    values,
    q1_ptr,
    s1_ptr,
    rows,
    cols,
    row_mask,
    col_mask,
    logical_col_start,
    q1_row_stride,
    s1_row_stride,
    s1_n_col_blocks: tl.constexpr,
    scale_factor_layout: tl.constexpr,
    format_id: tl.constexpr,
    use_half_range_scale: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    k_tile: tl.constexpr,
    sf_vec: tl.constexpr,
):
    x_abs = tl.abs(values)
    x_abs = tl.where(x_abs != x_abs, float("inf"), x_abs)
    x_3d = tl.reshape(x_abs, (block_m, k_tile, sf_vec))
    row_amax = tl.max(x_3d, axis=2)
    row_scale_u8, row_recip = _compute_scale_and_recip_from_amax(
        row_amax,
        FORMAT_ID=format_id,
        USE_HALF_RANGE_SCALE=use_half_range_scale,
    )
    s1_block_cols = logical_col_start // sf_vec + tl.arange(0, k_tile)
    s1_offsets = _compute_tile2d_scale_store_offsets(
        rows[:, None],
        s1_block_cols[None, :],
        s1_row_stride,
        s1_n_col_blocks,
        scale_factor_layout,
    )
    tl.store(
        s1_ptr + s1_offsets,
        row_scale_u8,
        mask=row_mask[:, None],
        cache_modifier=".cs",
    )
    recip_3d = tl.reshape(row_recip, (block_m, k_tile, 1)) + tl.zeros(
        (block_m, k_tile, sf_vec), tl.float32
    )
    q_values = values * tl.reshape(recip_3d, (block_m, block_k))
    _store_tile2d_qdata_row_major(
        q_ptr=q1_ptr,
        q_values=q_values,
        rows=rows,
        cols=cols,
        row_mask=row_mask,
        col_mask=col_mask,
        col_start=logical_col_start,
        q_row_stride=q1_row_stride,
        FORMAT_ID=format_id,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
    )


@triton.jit
def _store_swiglu_fused_axis0_mxfp8(
    values,
    q0_ptr,
    s0_ptr,
    rows,
    cols,
    row_mask,
    col_mask,
    row_start,
    logical_col_start,
    q0_row_stride,
    s0_row_stride,
    s0_n_col_blocks,  # runtime: M-derived (ragged rows) — keeps one compile across shapes
    scale_factor_layout: tl.constexpr,
    format_id: tl.constexpr,
    use_half_range_scale: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    m_tile: tl.constexpr,
    sf_vec: tl.constexpr,
):
    x_abs = tl.abs(values)
    x_abs = tl.where(x_abs != x_abs, float("inf"), x_abs)
    x_3d_n = tl.reshape(x_abs, (m_tile, sf_vec, block_k))
    col_amax = tl.max(x_3d_n, axis=1)
    col_scale_u8, col_recip = _compute_scale_and_recip_from_amax(
        col_amax,
        FORMAT_ID=format_id,
        USE_HALF_RANGE_SCALE=use_half_range_scale,
    )
    s0_block_rows = row_start // sf_vec + tl.arange(0, m_tile)
    if scale_factor_layout == _TL_SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED:
        s0_offsets = _compute_tile2d_scale_store_offsets(
            cols[None, :],
            s0_block_rows[:, None],
            s0_row_stride,
            s0_n_col_blocks,
            scale_factor_layout,
        )
    else:
        s0_offsets = _compute_tile2d_scale_store_offsets(
            s0_block_rows[:, None],
            cols[None, :],
            s0_row_stride,
            s0_n_col_blocks,
            scale_factor_layout,
        )
    tl.store(
        s0_ptr + s0_offsets,
        col_scale_u8,
        mask=col_mask[None, :],
        cache_modifier=".cs",
    )
    recip_3d_n = tl.reshape(col_recip, (m_tile, 1, block_k)) + tl.zeros(
        (m_tile, sf_vec, block_k), tl.float32
    )
    q_values = values * tl.reshape(recip_3d_n, (block_m, block_k))
    _store_tile2d_qdata_col_major_fp8(
        q_ptr=q0_ptr,
        q_values=q_values,
        rows=rows,
        cols=cols,
        row_mask=row_mask,
        col_mask=col_mask,
        q_row_stride=q0_row_stride,
        FORMAT_ID=format_id,
    )


@triton.jit(
    repr=make_dtype_repr(
        _build_quantize_spec_name(
            "swiglu_bwd_fwd",
            "SCALE_FACTOR_LAYOUT",
            "SF_VEC",
            axis_mask_constexpr="AXIS_MASK",
            scale_reduction_constexpr="SCALE_REDUCTION",
            producer_id_constexpr="PRODUCER_ID",
            producer_tag_override="swiglu_bwd_fwd",
        ),
        [],
    ),
)
def _triton_quantize_swiglu_bwd_dxy_fwd_col_mxfp8(
    dz_ptr,
    dz_row_stride,
    dz_col_stride,
    x_ptr,
    x_row_stride,
    x_col_stride,
    y_ptr,
    y_row_stride,
    y_col_stride,
    dxy_q0_ptr,
    dxy_q1_ptr,
    h2_q0_ptr,
    dxy_s0_ptr,
    dxy_s1_ptr,
    h2_s0_ptr,
    dxy_q0_row_stride,
    dxy_q1_row_stride,
    h2_q0_row_stride,
    dxy_s0_row_stride,
    dxy_s1_row_stride,
    h2_s0_row_stride,
    SCALE_FACTOR_LAYOUT: tl.constexpr,
    FORMAT_ID: tl.constexpr,
    AXIS_MASK: tl.constexpr,
    SCALE_REDUCTION: tl.constexpr,
    PRODUCER_ID: tl.constexpr,
    USE_HALF_RANGE_SCALE: tl.constexpr,
    FAST_MATH: tl.constexpr,
    CLAMPED: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    n_rows,  # runtime: M-derived (ragged rows) — only a row mask, keeps one compile across shapes
    SOURCE_COLS: tl.constexpr,
    DXY_S0_N_COL_BLOCKS,  # runtime: M-derived (ragged rows) — keeps one compile across shapes
    DXY_S1_N_COL_BLOCKS: tl.constexpr,
    H2_S0_N_COL_BLOCKS,  # runtime: M-derived (ragged rows) — keeps one compile across shapes
    SF_VEC: tl.constexpr,
    M_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
    GRID_K_TILES: tl.constexpr,
):
    tl.static_assert(
        FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E4M3
        or FORMAT_ID == _TL_BLOCK_SCALED_FORMAT_MXFP8_E5M2,
        "fused swiglu bwd+fwd col quant supports MXFP8 formats only",
    )
    tl.static_assert(AXIS_MASK == (_TL_AXIS_MASK_M | _TL_AXIS_MASK_K))
    tl.static_assert(SCALE_REDUCTION == _TL_SCALE_REDUCTION_ONE_D)
    tl.static_assert(PRODUCER_ID == _TL_BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY)
    BLOCK_M: tl.constexpr = M_TILE * SF_VEC
    BLOCK_K: tl.constexpr = K_TILE * SF_VEC
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row_start = pid_m * BLOCK_M
    col_start = pid_k * BLOCK_K
    rows = row_start + tl.arange(0, BLOCK_M)
    source_cols = col_start + tl.arange(0, BLOCK_K)
    row_mask = rows < n_rows
    col_mask = source_cols < SOURCE_COLS
    rows = rows.to(tl.int64)
    source_cols = source_cols.to(tl.int64)

    load_mask = row_mask[:, None] & col_mask[None, :]
    dz = tl.load(
        dz_ptr + rows[:, None] * dz_row_stride + source_cols[None, :] * dz_col_stride,
        mask=load_mask,
        other=_TL_FP32_ZERO,
    )
    x = tl.load(
        x_ptr + rows[:, None] * x_row_stride + source_cols[None, :] * x_col_stride,
        mask=load_mask,
        other=_TL_FP32_ZERO,
    )
    y = tl.load(
        y_ptr + rows[:, None] * y_row_stride + source_cols[None, :] * y_col_stride,
        mask=load_mask,
        other=_TL_FP32_ZERO,
    )
    dx, dy, h2 = _apply_block_scaled_quant_dxy_fwd_producer(
        dz,
        x,
        y,
        FAST_MATH=FAST_MATH,
        CLAMPED=CLAMPED,
        ALPHA=ALPHA,
        LIMIT=LIMIT,
    )

    dxy_cols_dx = source_cols
    dxy_cols_dy = source_cols + SOURCE_COLS
    h2_cols = source_cols
    _store_swiglu_fused_axis1_mxfp8(
        dx,
        dxy_q1_ptr,
        dxy_s1_ptr,
        rows,
        dxy_cols_dx,
        row_mask,
        col_mask,
        col_start,
        dxy_q1_row_stride,
        dxy_s1_row_stride,
        DXY_S1_N_COL_BLOCKS,
        SCALE_FACTOR_LAYOUT,
        FORMAT_ID,
        USE_HALF_RANGE_SCALE,
        BLOCK_M,
        BLOCK_K,
        K_TILE,
        SF_VEC,
    )
    _store_swiglu_fused_axis1_mxfp8(
        dy,
        dxy_q1_ptr,
        dxy_s1_ptr,
        rows,
        dxy_cols_dy,
        row_mask,
        col_mask,
        SOURCE_COLS + col_start,
        dxy_q1_row_stride,
        dxy_s1_row_stride,
        DXY_S1_N_COL_BLOCKS,
        SCALE_FACTOR_LAYOUT,
        FORMAT_ID,
        USE_HALF_RANGE_SCALE,
        BLOCK_M,
        BLOCK_K,
        K_TILE,
        SF_VEC,
    )
    _store_swiglu_fused_axis0_mxfp8(
        dx,
        dxy_q0_ptr,
        dxy_s0_ptr,
        rows,
        dxy_cols_dx,
        row_mask,
        col_mask,
        row_start,
        col_start,
        dxy_q0_row_stride,
        dxy_s0_row_stride,
        DXY_S0_N_COL_BLOCKS,
        SCALE_FACTOR_LAYOUT,
        FORMAT_ID,
        USE_HALF_RANGE_SCALE,
        BLOCK_M,
        BLOCK_K,
        M_TILE,
        SF_VEC,
    )
    _store_swiglu_fused_axis0_mxfp8(
        dy,
        dxy_q0_ptr,
        dxy_s0_ptr,
        rows,
        dxy_cols_dy,
        row_mask,
        col_mask,
        row_start,
        SOURCE_COLS + col_start,
        dxy_q0_row_stride,
        dxy_s0_row_stride,
        DXY_S0_N_COL_BLOCKS,
        SCALE_FACTOR_LAYOUT,
        FORMAT_ID,
        USE_HALF_RANGE_SCALE,
        BLOCK_M,
        BLOCK_K,
        M_TILE,
        SF_VEC,
    )
    _store_swiglu_fused_axis0_mxfp8(
        h2,
        h2_q0_ptr,
        h2_s0_ptr,
        rows,
        h2_cols,
        row_mask,
        col_mask,
        row_start,
        col_start,
        h2_q0_row_stride,
        h2_s0_row_stride,
        H2_S0_N_COL_BLOCKS,
        SCALE_FACTOR_LAYOUT,
        FORMAT_ID,
        USE_HALF_RANGE_SCALE,
        BLOCK_M,
        BLOCK_K,
        M_TILE,
        SF_VEC,
    )


def quantize_swiglu_bwd_dxy_fwd_col(  # noqa: C901 -- host validation/launch setup is format-gated
    dz: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    format_id: int,
    layout_id: int = ScaleFactorLayoutId.NATURAL.value,
    half_range_scale: bool = False,
    fast_math: bool = False,
    clamped: bool = False,
    alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
):
    """Quantize SwiGLU backward DXY and forward h2-col1d in one Triton kernel."""
    clamped, alpha, limit = canonical_swiglu_clamp(clamped, alpha, limit)
    if format_id not in _MXFP8_FORMAT_IDS:
        raise NotImplementedError(
            "Triton fused SwiGLU backward+forward quantization currently supports MXFP8 only"
        )
    if layout_id not in (
        ScaleFactorLayoutId.NATURAL.value,
        ScaleFactorLayoutId.CUBLAS_BLOCKED.value,
    ):
        raise ValueError(
            f"Unsupported scale layout id for axes quantization: {layout_id}"
        )
    if dz.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Unsupported input dtype for axes quantization: {dz.dtype}")
    if x.dtype != dz.dtype or y.dtype != dz.dtype:
        raise TypeError(
            f"dz/x/y dtypes must match: dz={dz.dtype}, x={x.dtype}, y={y.dtype}"
        )
    if dz.shape != x.shape or dz.shape != y.shape:
        raise ValueError(f"dz/x/y shapes must match: {dz.shape}, {x.shape}, {y.shape}")
    if dz.ndim != 2:
        raise ValueError(
            f"fused SwiGLU backward+forward quantization expects 2D tensors: {dz.shape}"
        )
    if dz.stride(-1) != 1 or x.stride(-1) != 1 or y.stride(-1) != 1:
        raise ValueError(
            "fused SwiGLU backward+forward quantization requires contiguous last dim"
        )
    sf_vec = _get_sf_vec_size_for_format_id(format_id)
    n_rows, source_cols = dz.shape
    if n_rows % sf_vec != 0:
        raise ValueError(f"axis=0 dimension {n_rows} not divisible by {sf_vec}")
    if source_cols % sf_vec != 0:
        raise ValueError(f"source_cols {source_cols} not divisible by {sf_vec}")
    dxy_cols = source_cols * 2
    q_dtype = _get_qdata_dtype_for_format_id(format_id)
    sdtype = _get_scale_dtype_for_format_id(format_id)
    dxy_q0 = torch.empty((dxy_cols, n_rows), dtype=q_dtype, device=dz.device)
    dxy_q1 = torch.empty((n_rows, dxy_cols), dtype=q_dtype, device=dz.device)
    h2_q0 = torch.empty((source_cols, n_rows), dtype=q_dtype, device=dz.device)

    is_blocked = layout_id == ScaleFactorLayoutId.CUBLAS_BLOCKED.value

    def _alloc_scale(logical_rows: int, logical_cols: int) -> torch.Tensor:
        if is_blocked:
            padded_rows, padded_cols = _padded_cublas_blocked_scale_shape(
                logical_rows,
                logical_cols,
            )
            scale_factory = (
                torch.zeros
                if padded_rows != logical_rows or padded_cols != logical_cols
                else torch.empty
            )
            return scale_factory(
                (padded_rows, padded_cols), dtype=sdtype, device=dz.device
            )
        return torch.empty((logical_rows, logical_cols), dtype=sdtype, device=dz.device)

    if is_blocked:
        dxy_s0_logical = (dxy_cols, n_rows // sf_vec)
        h2_s0_logical = (source_cols, n_rows // sf_vec)
    else:
        dxy_s0_logical = (n_rows // sf_vec, dxy_cols)
        h2_s0_logical = (n_rows // sf_vec, source_cols)
    dxy_s1_logical = (n_rows, dxy_cols // sf_vec)

    dxy_s0 = _alloc_scale(*dxy_s0_logical)
    dxy_s1 = _alloc_scale(*dxy_s1_logical)
    h2_s0 = _alloc_scale(*h2_s0_logical)

    def _scale_stride(logical_rows: int, logical_cols: int) -> int:
        if is_blocked:
            return _padded_cublas_blocked_scale_shape(logical_rows, logical_cols)[1]
        return logical_cols

    m_tile = 1
    k_tile = 1
    block_m = m_tile * sf_vec
    block_k = k_tile * sf_vec
    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(source_cols, block_k))
    _triton_quantize_swiglu_bwd_dxy_fwd_col_mxfp8[grid](
        dz_ptr=dz,
        dz_row_stride=dz.stride(-2),
        dz_col_stride=dz.stride(-1),
        x_ptr=x,
        x_row_stride=x.stride(-2),
        x_col_stride=x.stride(-1),
        y_ptr=y,
        y_row_stride=y.stride(-2),
        y_col_stride=y.stride(-1),
        dxy_q0_ptr=dxy_q0,
        dxy_q1_ptr=dxy_q1,
        h2_q0_ptr=h2_q0,
        dxy_s0_ptr=dxy_s0.view(torch.uint8),
        dxy_s1_ptr=dxy_s1.view(torch.uint8),
        h2_s0_ptr=h2_s0.view(torch.uint8),
        dxy_q0_row_stride=dxy_q0.stride(0),
        dxy_q1_row_stride=dxy_q1.stride(0),
        h2_q0_row_stride=h2_q0.stride(0),
        dxy_s0_row_stride=_scale_stride(*dxy_s0_logical),
        dxy_s1_row_stride=_scale_stride(*dxy_s1_logical),
        h2_s0_row_stride=_scale_stride(*h2_s0_logical),
        SCALE_FACTOR_LAYOUT=layout_id,
        FORMAT_ID=format_id,
        AXIS_MASK=AxisMask.M.value | AxisMask.K.value,
        SCALE_REDUCTION=ScaleReduction.ONE_D.value,
        PRODUCER_ID=BlockScaledProducer.SWIGLU_BWD_DXY.value,
        USE_HALF_RANGE_SCALE=half_range_scale,
        FAST_MATH=fast_math,
        CLAMPED=clamped,
        ALPHA=alpha,
        LIMIT=limit,
        n_rows=n_rows,
        SOURCE_COLS=source_cols,
        DXY_S0_N_COL_BLOCKS=ceil_div(
            dxy_s0_logical[1],
            CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
        ),
        DXY_S1_N_COL_BLOCKS=ceil_div(
            dxy_s1_logical[1],
            CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
        ),
        H2_S0_N_COL_BLOCKS=ceil_div(
            h2_s0_logical[1],
            CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
        ),
        SF_VEC=sf_vec,
        M_TILE=m_tile,
        K_TILE=k_tile,
        GRID_K_TILES=grid[1],
        num_warps=2,
    )
    return (dxy_q0, dxy_s0), (dxy_q1, dxy_s1), (h2_q0, h2_s0)


def quantize_block_scaled(  # noqa: C901 -- host validation/launch setup is format-gated
    x: torch.Tensor,
    *,
    format_id: int,
    layout_id: int,
    is_a: bool,
    half_range_scale: bool = False,
    fast_math: bool = False,
    clamped: bool = False,
    alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    producer_b: torch.Tensor | None = None,
    producer_id: int = BlockScaledProducer.IDENTITY.value,
    output_shape: tuple[int, ...] | torch.Size | None = None,
    source_cols: int | None = None,
    make_contiguous: bool = True,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor, optionally evaluating an elementwise producer first."""
    clamped, alpha, limit = canonical_swiglu_clamp(clamped, alpha, limit)
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(
            f"Unsupported input dtype for block-scaled quantization: {x.dtype}"
        )
    if x.ndim < 2 or x.ndim > 3:
        raise ValueError(f"Expected a 2D or 3D tensor, got shape={tuple(x.shape)}")

    sf_vec_size = _get_sf_vec_size_for_format_id(format_id)
    logical_shape = tuple(output_shape) if output_shape is not None else tuple(x.shape)
    K = logical_shape[-1]
    input_k = source_cols if source_cols is not None else x.shape[-1]
    if _is_fp4_format_id(format_id):
        if K % 2 != 0:
            raise ValueError(f"FP4 output requires an even K dimension, got {K}")
    if producer_id != BlockScaledProducer.IDENTITY.value:
        if producer_b is None:
            raise ValueError("producer_b is required for producer mode")
        if K != input_k:
            raise ValueError("fast producer quantization expects output K == input K")
    if global_scale is not None:
        if format_id != BlockScaledFormatId.NVFP4.value:
            raise NotImplementedError(
                "global_scale is currently supported only for NVFP4"
            )
        if global_scale.device != x.device:
            raise ValueError("global_scale must be on the input device")

    if x.ndim == 2:
        x_storage = x
        x_2d = x
    elif x.ndim == 3 and K == x.shape[-1] and x.stride(0) == x.shape[1] * x.stride(1):
        x_storage = x
        x_2d = x.as_strided(
            (x.shape[0] * x.shape[1], K),
            (x.stride(1), x.stride(2)),
        )
    else:
        x_storage = x.contiguous() if make_contiguous else x
        if make_contiguous:
            x_2d = x_storage.view(-1, K)
        elif x_storage.ndim != 2:
            raise ValueError("producer quantization expects pre-flattened 2D input")
        else:
            x_2d = x_storage
    producer_b_2d = x_2d if producer_b is None else producer_b
    rows = x_2d.shape[0]
    global_scale_info = _global_scale_launch_info(
        global_scale,
        rows=rows,
        device=x.device,
    )
    scale_cols = ceil_div(K, sf_vec_size)

    q = _empty_quantized_qdata(
        logical_shape, format_id=format_id, device=x_storage.device
    )
    scale_dtype = _get_scale_dtype_for_format_id(format_id)
    scale_shape, rows_per_group, has_groups = _compute_row1d_scale_output_shape(
        torch.Size(logical_shape),
        scale_cols,
        layout_id,
        is_a=is_a,
    )
    scale_factory = (
        torch.zeros
        if layout_id == ScaleFactorLayoutId.CUBLAS_BLOCKED.value
        else torch.empty
    )
    scale = scale_factory(scale_shape, dtype=scale_dtype, device=x.device)
    if rows == 0 or scale.numel() == 0:
        return q, scale

    n_col_blocks = ceil_div(scale_cols, CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM)
    padded_rows_per_group, padded_scale_cols = _padded_cublas_blocked_scale_shape(
        rows_per_group,
        scale_cols,
    )
    group_output_elems = padded_rows_per_group * padded_scale_cols
    q_storage = q.view(torch.uint8) if _is_fp4_format_id(format_id) else q
    # Both tiled kernels write scale bytes through a uint8 view (the shared
    # `_compute_scale_and_recip_from_amax` returns the NVFP4 FP8 scale bitcast to uint8),
    # so view the buffer as uint8 regardless of its logical scale dtype.
    scale_storage = scale.view(torch.uint8)
    # Kernels write scale bytes through a flat contiguous view, including NATURAL
    # 3D SFB scales. Keep the fresh scale allocation contiguous if this factory
    # is refactored; the offset math assumes row-major storage.
    use_64bit_offsets = _needs_64bit_offsets(
        _compute_max_strided_offset(rows, K, x_2d.stride(-2), x_2d.stride(-1)),
        _compute_max_strided_offset(
            rows,
            K,
            producer_b_2d.stride(-2),
            producer_b_2d.stride(-1),
        ),
        _compute_max_strided_offset(rows, q_storage.shape[-1], q_storage.shape[-1], 1),
        0 if global_scale_info is None else global_scale_info.max_offset,
    )
    common_row1d_kernel_args = {
        "x_row_stride": x_2d.stride(-2),
        "x_col_stride": x_2d.stride(-1),
        "x_ptr": x_2d,
        **_global_scale_kernel_kwargs(global_scale_info, fallback_ptr=x_2d),
        "producer_b_ptr": producer_b_2d,
        "producer_b_row_stride": producer_b_2d.stride(-2),
        "producer_b_col_stride": producer_b_2d.stride(-1),
        "PRODUCER_ID": producer_id,
        "q_ptr": q_storage.view(-1, q_storage.shape[-1]),
        "scale_ptr": scale_storage.view(-1),
        "Q_COLS": q_storage.shape[-1],
        "FORMAT_ID": format_id,
        "USE_HALF_RANGE_SCALE": half_range_scale,
        "FAST_MATH": fast_math,
        "CLAMPED": clamped,
        "ALPHA": alpha,
        "LIMIT": limit,
        "n_rows": rows,
        "N_COLS": K,
        "SCALE_COLS": scale_cols,
        "rows_per_group": rows_per_group,
        "N_COL_BLOCKS": n_col_blocks,
        "group_output_elems": group_output_elems,
        "HAS_GROUPS": has_groups,
    }

    blocked_config = (
        _get_row1d_cublas_blocked_kernel_config(format_id)
        if layout_id == ScaleFactorLayoutId.CUBLAS_BLOCKED.value
        else None
    )
    if blocked_config is not None:
        grid = (
            triton.cdiv(
                K, blocked_config.block_size * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
            ),
            triton.cdiv(rows, blocked_config.m_tile),
        )
        _triton_quantize_blockscaled_row1d_cublas_blocked[grid](
            **common_row1d_kernel_args,
            GRID_COL_TILES=grid[0],
            M_TILE=blocked_config.m_tile,
            BLOCK_SIZE=blocked_config.block_size,
            USE_64BIT_OFFSETS=use_64bit_offsets,
            num_warps=blocked_config.num_warps,
        )
        return q, scale

    config = _get_row1d_tile_kernel_config(format_id)
    grid = (
        triton.cdiv(rows, config.row_tile),
        triton.cdiv(scale_cols, config.scale_col_tile),
    )
    _triton_quantize_blockscaled_row1d_tile[grid](
        **common_row1d_kernel_args,
        SCALE_FACTOR_LAYOUT=layout_id,
        GRID_COL_TILES=grid[1],
        ROW_TILE=config.row_tile,
        SCALE_COL_TILE=config.scale_col_tile,
        USE_64BIT_OFFSETS=use_64bit_offsets,
        BLOCK_SIZE=sf_vec_size,
        num_warps=config.num_warps,
    )
    return q, scale


# The 32x32 shared-max tile: 32 cols (the MXFP8 SF vector) x 32 rows.
_MXFP8_SHARED_TILE = 32


def _check_blockscaled_dims(x: torch.Tensor) -> None:
    """Validate a given-grid MXFP8 input."""
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D tensor, got shape={tuple(x.shape)}")
    rows, k = x.shape[-2], x.shape[-1]
    if k % _MXFP8_SHARED_TILE != 0:
        raise ValueError(
            f"given-grid MXFP8 requires K divisible by {_MXFP8_SHARED_TILE}; got K={k}"
        )
    if x.ndim == 3 and rows % _MXFP8_SHARED_TILE != 0:
        raise ValueError(
            "given-grid MXFP8 requires the per-group row dim divisible by "
            f"{_MXFP8_SHARED_TILE} for 3D grouped input; got rows={rows}"
        )


def _validate_given_e8m0_grid(
    e8m0_grid: torch.Tensor,
    x: torch.Tensor,
    row_offset: int,
    row_dim: int,
    scale_cols: int,
) -> None:
    """Validate ``e8m0_grid``'s shape for ``quantize_mxfp8_given_grid``.

    2D: any grid tall enough to cover ``[row_offset, row_offset + rows)`` -- the
    FULL global grid is accepted for ANY ``row_offset`` incl 0 (every shard,
    rank 0 included, gets the full grid). 3D grouped: the exact per-group grid.
    """
    if row_offset < 0:
        raise ValueError(f"row_offset must be non-negative; got {row_offset}")
    if e8m0_grid.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError(
            f"e8m0_grid must be uint8 or float8_e8m0fnu; got {e8m0_grid.dtype}"
        )
    if x.ndim == 3:
        if row_offset != 0:
            raise NotImplementedError(
                "given-grid MXFP8 with row_offset>0 is only supported for 2D input"
            )
        expected_grid = (x.shape[0], row_dim // _MXFP8_SHARED_TILE, scale_cols)
        if tuple(e8m0_grid.shape) != expected_grid:
            raise ValueError(
                f"e8m0_grid shape {tuple(e8m0_grid.shape)} does not match the expected "
                f"grouped natural grid {expected_grid} for x shape {tuple(x.shape)}"
            )
        return
    # 2D: accept any grid tall enough to cover [row_offset, row_offset + row_dim).
    # The full global grid is passed to every shard, incl. rank 0 (row_offset=0).
    min_grid_rows = (row_offset + row_dim - 1) // _MXFP8_SHARED_TILE + 1
    if e8m0_grid.ndim != 2 or e8m0_grid.shape[1] != scale_cols:
        raise ValueError(
            f"e8m0_grid shape {tuple(e8m0_grid.shape)} is not a 2D grid with "
            f"{scale_cols} scale columns for x shape {tuple(x.shape)}"
        )
    if e8m0_grid.shape[0] < min_grid_rows:
        raise ValueError(
            f"e8m0_grid has {e8m0_grid.shape[0]} rows but row_offset={row_offset} "
            f"with {row_dim} shard rows requires at least {min_grid_rows} grid rows"
        )


def quantize_mxfp8_given_grid(
    x: torch.Tensor,
    e8m0_grid: torch.Tensor,
    *,
    row_offset: int = 0,
) -> torch.Tensor:
    """Quantize ``x`` to MXFP8_E4M3 against a caller-supplied natural E8M0 grid.

    ``e8m0_grid`` is the per-32x32-tile grid (``uint8`` or ``float8_e8m0fnu``),
    one byte broadcast over each tile -- avoids the FP8 -> BF16 -> FP8 round-trip.
    ``row_offset`` (2D only) is the GLOBAL row index of ``x``'s local row 0, so a
    non-32-aligned shard indexes the right tiles of the FULL (possibly taller)
    global grid. Returns FP8 (``float8_e4m3fn``) data only; the caller derives the
    cuBLAS-blocked SF frames from the grid via ``transpose_sf_layout``.
    """
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"given-grid MXFP8 expects bf16 or fp32 input, got {x.dtype}")
    _check_blockscaled_dims(x)

    format_id = BlockScaledFormatId.MXFP8_E4M3.value
    k = x.shape[-1]
    row_dim = x.shape[-2]
    scale_cols = k // _MXFP8_SHARED_TILE
    _validate_given_e8m0_grid(e8m0_grid, x, row_offset, row_dim, scale_cols)
    if e8m0_grid.device != x.device:
        raise ValueError("e8m0_grid must be on the input device")
    x_2d = x.reshape(-1, k)
    n_rows = x_2d.shape[0]
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    q_2d = q.view(-1, k)
    # Flat uint8 view; grid row r maps to x's 32-row band r (rows_per_group % 32 == 0).
    grid_2d = e8m0_grid.reshape(-1, scale_cols).contiguous().view(torch.uint8)
    if n_rows == 0:
        return q

    config = _get_row1d_tile_kernel_config(format_id)
    grid = (
        triton.cdiv(n_rows, config.row_tile),
        triton.cdiv(scale_cols, config.scale_col_tile),
    )
    _triton_quantize_blockscaled_tile2d_given_grid[grid](
        x_ptr=x_2d,
        x_row_stride=x_2d.stride(0),
        x_col_stride=x_2d.stride(1),
        e8m0_grid_ptr=grid_2d,
        e8m0_grid_row_stride=grid_2d.stride(0),
        e8m0_grid_col_stride=grid_2d.stride(1),
        q_ptr=q_2d,
        FORMAT_ID=format_id,
        N_ROWS=n_rows,
        N_COLS=k,
        Q_COLS=k,
        SCALE_COLS=scale_cols,
        GRID_COL_TILES=grid[1],
        ROW_TILE=config.row_tile,
        SCALE_COL_TILE=config.scale_col_tile,
        BLOCK_SIZE=_MXFP8_SHARED_TILE,
        ROW_BLOCK=_MXFP8_SHARED_TILE,
        ROW_OFFSET=row_offset,
        num_warps=config.num_warps,
    )
    return q


def quantize_block_scaled_axes(  # noqa: C901 -- dispatch setup is format/axis gated
    x: torch.Tensor,
    *,
    format_id: int,
    axis_mask: int,
    scale_reduction_id: int,
    layout_id: int = ScaleFactorLayoutId.NATURAL.value,
    half_range_scale: bool = False,
    fast_math: bool = False,
    clamped: bool = False,
    alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    producer_b: torch.Tensor | None = None,
    producer_c: torch.Tensor | None = None,
    producer_id: int = BlockScaledProducer.IDENTITY.value,
    output_shape: tuple[int, ...] | None = None,
    source_cols: int | None = None,
    gather_dtype: torch.dtype | None = None,
    local_rank: int = 0,
    world_size: int = 1,
    global_scale: torch.Tensor | None = None,
):
    """Quantize x with axis=0 / axis=1 / 2D-tile scale reduction.

    Returns ``(qdata, scale)`` when ``axis_mask`` selects a single axis, or
    two ``(qdata, scale)`` pairs when both axes are requested. For
    ``SCALE_REDUCTION_TWO_D`` the same qdata tensor is aliased into both
    returned pairs since one tile scale is broadcast across both layouts.

    When ``producer_id == ZEROCOPY_GATHER``, ``x`` is reinterpreted as the
    ``int64`` 1-D gather-pointer table (one remote symm-mem address per
    output row); the source data dtype must then be passed via
    ``gather_dtype`` and the logical ``(n_rows, n_cols)`` is taken from
    ``output_shape``. ``x`` itself is not dereferenced for its values; the
    kernel only reads it as a pointer table.
    """
    clamped, alpha, limit = canonical_swiglu_clamp(clamped, alpha, limit)
    is_zerocopy_gather = producer_id == BlockScaledProducer.ZEROCOPY_GATHER.value
    if is_zerocopy_gather:
        if world_size <= 0:
            raise ValueError(f"world_size must be positive; got {world_size}")
        if local_rank < 0 or local_rank >= world_size:
            raise ValueError(
                f"local_rank must be in [0, world_size); got {local_rank=} {world_size=}"
            )
    elif local_rank != 0 or world_size != 1:
        raise ValueError("local_rank/world_size are only valid for ZEROCOPY_GATHER")
    if is_zerocopy_gather:
        if x.dtype != torch.int64:
            raise TypeError(
                "ZEROCOPY_GATHER: x must be int64 (1-D gather_ptrs table); "
                f"got {x.dtype}"
            )
        if x.ndim != 1:
            raise ValueError(
                f"ZEROCOPY_GATHER: x must be 1-D [n_rows]; got shape {tuple(x.shape)}"
            )
        if output_shape is None:
            raise ValueError("ZEROCOPY_GATHER requires output_shape=(n_rows, n_cols)")
        if gather_dtype is None:
            raise ValueError(
                "ZEROCOPY_GATHER requires gather_dtype (source-data dtype)"
            )
        if gather_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(
                f"ZEROCOPY_GATHER: gather_dtype must be bf16/fp16/fp32; got {gather_dtype}"
            )
    else:
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"Unsupported input dtype for axes quantization: {x.dtype}")
        if x.ndim not in (2, 3):
            raise NotImplementedError(
                f"axes quantization expects 2D or 3D input; got {tuple(x.shape)}"
            )
        if gather_dtype is not None:
            raise ValueError(
                "gather_dtype is only valid when producer_id=ZEROCOPY_GATHER"
            )
    sf_vec = _get_sf_vec_size_for_format_id(format_id)
    logical_shape = (
        tuple(output_shape)
        if output_shape is not None
        else tuple(x.shape)
        if not is_zerocopy_gather
        else None  # unreachable: gather requires output_shape
    )
    if len(logical_shape) not in (2, 3):
        raise ValueError(
            f"axes quantization output shape must be 2D or 3D: {logical_shape}"
        )
    if producer_id != BlockScaledProducer.IDENTITY.value and not is_zerocopy_gather:
        if producer_b is None or producer_c is None:
            raise ValueError("producer_b and producer_c are required for producer mode")
        if source_cols is None:
            raise ValueError("source_cols is required for producer mode")
    n_batch = logical_shape[0] if len(logical_shape) == 3 else 1
    n_rows, n_cols = logical_shape[-2], logical_shape[-1]
    if source_cols is None:
        source_cols = n_cols if is_zerocopy_gather else x.shape[-1]

    is_two_d = scale_reduction_id == ScaleReduction.TWO_D.value
    is_cublas_blocked_layout = layout_id == ScaleFactorLayoutId.CUBLAS_BLOCKED.value
    if layout_id not in (
        ScaleFactorLayoutId.NATURAL.value,
        ScaleFactorLayoutId.CUBLAS_BLOCKED.value,
    ):
        raise ValueError(
            f"Unsupported scale layout id for axes quantization: {layout_id}"
        )
    has_axis0 = (axis_mask & AxisMask.M.value) != 0
    has_axis1 = (axis_mask & AxisMask.K.value) != 0
    is_col1d = has_axis0 and not has_axis1 and not is_two_d
    if global_scale is not None:
        if format_id != BlockScaledFormatId.NVFP4.value:
            raise NotImplementedError(
                "global_scale is currently supported only for NVFP4"
            )
        if is_zerocopy_gather:
            raise NotImplementedError(
                "global_scale is not supported with ZEROCOPY_GATHER"
            )
        if not is_two_d and (
            scale_reduction_id != ScaleReduction.ONE_D.value
            or axis_mask not in (AxisMask.M.value, AxisMask.K.value)
        ):
            raise NotImplementedError(
                "NVFP4 global_scale axes quantization is supported only for "
                "single-axis 1D or square 2D blocks"
            )
        global_scale_info = _global_scale_launch_info(
            global_scale,
            rows=n_batch * n_rows,
            device=x.device,
            require_block_constant=has_axis0,
        )
    else:
        global_scale_info = None
    if (has_axis1 or is_two_d) and n_cols % sf_vec != 0:
        raise ValueError(f"axis=1 dimension {n_cols} not divisible by {sf_vec}")
    if (
        scale_reduction_id == ScaleReduction.ONE_D.value
        and axis_mask == AxisMask.K.value
        and producer_id == BlockScaledProducer.IDENTITY.value
    ):
        # Axis-1-only path uses the dedicated row-1D kernel.
        return quantize_block_scaled(
            x,
            format_id=format_id,
            layout_id=layout_id,
            is_a=x.ndim == 2,
            half_range_scale=half_range_scale,
            fast_math=fast_math,
            global_scale=global_scale,
        )
    if (has_axis0 or is_two_d) and n_rows % sf_vec != 0:
        raise ValueError(f"axis=0 dimension {n_rows} not divisible by {sf_vec}")

    is_fp4 = _is_fp4_format_id(format_id)
    sdtype = _get_scale_dtype_for_format_id(format_id)
    batch_shape = (n_batch,) if x.ndim == 3 else ()

    def _alloc_scale(logical_rows: int, logical_cols: int):
        if is_cublas_blocked_layout:
            padded_rows, padded_cols = _padded_cublas_blocked_scale_shape(
                logical_rows,
                logical_cols,
            )
            scale_factory = (
                torch.zeros
                if padded_rows != logical_rows or padded_cols != logical_cols
                else torch.empty
            )
            return scale_factory(
                (n_batch * padded_rows, padded_cols),
                dtype=sdtype,
                device=x.device,
            )
        return torch.empty(
            (*batch_shape, logical_rows, logical_cols), dtype=sdtype, device=x.device
        )

    def _alloc_s0():
        # Axis-0 reduces over M (n_rows). The grouped-GEMM wgrad consumer
        # reads this scale in MMA-A frame: atom-MN spans K (= n_cols) and
        # atom-K spans M/V (= n_rows / sf_vec). For BLOCKED layout we allocate
        # in MMA-A frame directly so callers can hand the buffer to wgrad
        # without a ``.t() + transpose_sf_layout`` swizzle. NATURAL stays in
        # the data-shaped ``[M/V, K]`` layout that downstream code expects.
        if is_cublas_blocked_layout:
            return _alloc_scale(n_cols, n_rows // sf_vec)
        return _alloc_scale(n_rows // sf_vec, n_cols)

    def _alloc_s1():
        return _alloc_scale(n_rows, n_cols // sf_vec)

    # FP8 TWO_D still passes q0 through a compiled dead branch; keep a real
    # typed buffer even though the caller receives None for q0.
    need_q0 = has_axis0 or is_two_d
    need_q1 = has_axis1 or is_two_d

    def _alloc_qdata(*shape: int) -> torch.Tensor:
        return _empty_quantized_qdata(
            (*batch_shape, *shape),
            format_id=format_id,
            device=x.device,
        )

    q0 = _alloc_qdata(n_cols, n_rows) if need_q0 else None
    q1 = _alloc_qdata(n_rows, n_cols) if need_q1 else None
    s0 = _alloc_s0() if (has_axis0 or is_two_d) else None
    s1 = _alloc_s1() if (has_axis1 or is_two_d) else None

    # Triton needs real pointers; substitute a 1-element dummy for unused buffers.
    dummy = torch.empty(1, dtype=torch.uint8, device=x.device)

    def _get_strides(t):
        # Return (batch_stride, row_stride) for the kernel; batch_stride is
        # zero when the tensor doesn't carry a batch dim.
        if t is None:
            return 0, 0
        # Use the original tensor's strides (FP4 view shares storage with
        # its uint8 backing, so strides match).
        if t.ndim == 3:
            return t.stride(0), t.stride(1)
        return 0, t.stride(0)

    def _get_scale_strides(t, logical_rows: int, logical_cols: int):
        if t is None:
            return 0, 0
        if is_cublas_blocked_layout:
            padded_rows, padded_cols = _padded_cublas_blocked_scale_shape(
                logical_rows,
                logical_cols,
            )
            return padded_rows * padded_cols, padded_cols
        return _get_strides(t)

    def _make_arg_q(t):
        if t is None:
            return dummy
        return t.view(torch.uint8) if is_fp4 else t

    def _make_arg_s(t):
        if t is None:
            return dummy
        return t.view(torch.uint8)

    q0_batch_stride, q0_row_stride = _get_strides(q0)
    q1_batch_stride, q1_row_stride = _get_strides(q1)
    # Axis-0 s0 logical layout: NATURAL is data-shape ``[M/V, K]``; BLOCKED
    # is MMA-A frame ``[K, M/V]`` (see ``_alloc_s0``).
    if is_cublas_blocked_layout:
        s0_logical_rows = n_cols
        s0_logical_cols = n_rows // sf_vec
    else:
        s0_logical_rows = n_rows // sf_vec
        s0_logical_cols = n_cols
    s1_logical_rows = n_rows
    s1_logical_cols = n_cols // sf_vec
    s0_batch_stride, s0_row_stride = _get_scale_strides(
        s0, s0_logical_rows, s0_logical_cols
    )
    s1_batch_stride, s1_row_stride = _get_scale_strides(
        s1, s1_logical_rows, s1_logical_cols
    )
    # For ZEROCOPY_GATHER, x is the int64 gather_ptrs vector; the kernel
    # only ever dereferences it as a per-row pointer table, so the strides
    # below are unused (kept zero to make stride math a constexpr-zero
    # in the gather branch). The producer_{b,c} slots are also unused;
    # default them to dummy.
    if is_zerocopy_gather:
        x_batch_stride = 0
        x_row_stride = 0
        x_col_stride = 0
    else:
        x_batch_stride = x.stride(0) if x.ndim == 3 else 0
        x_row_stride = x.stride(-2)
        x_col_stride = x.stride(-1)
    # ``x`` is 1-D in gather mode, so it cannot serve as the producer_{b,c}
    # default — substitute the 2-D dummy buffer the kernel already uses
    # for unused tensor slots.
    if producer_b is None:
        producer_b_arg = dummy if is_zerocopy_gather else x
    else:
        producer_b_arg = producer_b
    if producer_c is None:
        producer_c_arg = dummy if is_zerocopy_gather else x
    else:
        producer_c_arg = producer_c
    producer_b_batch_stride = (
        producer_b_arg.stride(0) if producer_b_arg.ndim == 3 else 0
    )
    producer_c_batch_stride = (
        producer_c_arg.stride(0) if producer_c_arg.ndim == 3 else 0
    )
    # producer_{b,c} strides are unused in gather and identity modes; zero them
    # when the producer arg is the 1-D dummy so ``stride(-2)`` never touches it.
    producer_b_strides = (
        (0, 0)
        if producer_b_arg is dummy
        else (producer_b_arg.stride(-2), producer_b_arg.stride(-1))
    )
    producer_c_strides = (
        (0, 0)
        if producer_c_arg is dummy
        else (producer_c_arg.stride(-2), producer_c_arg.stride(-1))
    )

    # Pick (M_TILE, K_TILE) so each CTA covers roughly 16 scale-blocks. Bigger
    # tiles amortize launch overhead and let Triton issue wider vectorized
    # memory transactions; too big blows register usage. Empirically 4x4
    # scale-blocks per CTA (i.e. 128x128 for MXFP8, 64x64 for NVFP4) is the
    # sweet spot on Blackwell. Tile size is capped by divisibility of n_rows
    # and n_cols.
    def _pick_tile(n: int, max_tile: int) -> int:
        t = max_tile
        while t > 1 and n % (t * sf_vec) != 0:
            t //= 2
        return t

    src_dtype = gather_dtype or x.dtype
    tile_config = _get_tile2d_kernel_config(
        format_id=format_id,
        src_dtype=src_dtype,
        has_axis0=has_axis0,
        has_axis1=has_axis1,
        is_two_d=is_two_d,
        producer_id=producer_id,
    )
    m_tile = _pick_tile(n_rows, tile_config.max_m_tile) if n_rows % sf_vec == 0 else 1
    k_tile = _pick_tile(n_cols, tile_config.max_k_tile)
    block_m = m_tile * sf_vec
    block_k = k_tile * sf_vec

    # Axis 0 (2^31 extent) carries the unbounded row-tile count; axes 1 and 2
    # (65535 cap) carry the small col-tile and batch dims. See the kernel's grid
    # axis assignment comment.
    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(n_cols, block_k), n_batch)
    nvfp4_recip_lut = (
        _build_nvfp4_recip_lut(x.device)
        if format_id == BlockScaledFormatId.NVFP4.value
        else dummy
    )
    use_nvfp4_recip_lut = _use_nvfp4_recip_lut_for_tile2d(
        format_id=format_id,
        is_cublas_blocked_layout=is_cublas_blocked_layout,
    )
    use_tma_load = _use_tma_swiglu_fwd_load_for_tile2d(
        x=x,
        producer_b=producer_b,
        format_id=format_id,
        producer_id=producer_id,
        is_col1d=is_col1d,
    )

    src_dtype_constexpr = _convert_torch_dtype_to_tl_dtype(src_dtype)
    if use_tma_load:
        assert producer_b is not None
        block_shape = [block_m, block_k]
        x_tma_desc = _make_tma_descriptor(x, block_shape)
        y_tma_desc = _make_tma_descriptor(producer_b, block_shape)
    else:
        x_tma_desc = x
        y_tma_desc = producer_b_arg
    _triton_quantize_blockscaled_tile2d[grid](
        # Input.
        x_ptr=x,
        x_batch_stride=x_batch_stride,
        x_row_stride=x_row_stride,
        x_col_stride=x_col_stride,
        # Producer.
        producer_b_ptr=producer_b_arg,
        producer_c_ptr=producer_c_arg,
        x_desc=x_tma_desc,
        y_desc=y_tma_desc,
        producer_b_batch_stride=producer_b_batch_stride,
        producer_b_row_stride=producer_b_strides[0],
        producer_b_col_stride=producer_b_strides[1],
        producer_c_batch_stride=producer_c_batch_stride,
        producer_c_row_stride=producer_c_strides[0],
        producer_c_col_stride=producer_c_strides[1],
        PRODUCER_ID=producer_id,
        # Output.
        q0_ptr=_make_arg_q(q0),
        q1_ptr=_make_arg_q(q1),
        s0_ptr=_make_arg_s(s0),
        s1_ptr=_make_arg_s(s1),
        q0_batch_stride=q0_batch_stride,
        q0_row_stride=q0_row_stride,
        q1_batch_stride=q1_batch_stride,
        q1_row_stride=q1_row_stride,
        s0_batch_stride=s0_batch_stride,
        s0_row_stride=s0_row_stride,
        s1_batch_stride=s1_batch_stride,
        s1_row_stride=s1_row_stride,
        SCALE_FACTOR_LAYOUT=layout_id,
        **_global_scale_kernel_kwargs(global_scale_info, fallback_ptr=dummy),
        # Quant / format config.
        nvfp4_recip_lut_ptr=nvfp4_recip_lut,
        FORMAT_ID=format_id,
        AXIS_MASK=axis_mask,
        SCALE_REDUCTION=scale_reduction_id,
        USE_HALF_RANGE_SCALE=half_range_scale,
        USE_NVFP4_RECIP_LUT=use_nvfp4_recip_lut,
        FAST_MATH=fast_math,
        CLAMPED=clamped,
        ALPHA=alpha,
        LIMIT=limit,
        SRC_DTYPE=src_dtype_constexpr,
        USE_TMA_LOAD=use_tma_load,
        local_rank=local_rank,
        WORLD_SIZE=world_size,
        # Shape & addressing.
        n_rows=n_rows,
        N_COLS=n_cols,
        SOURCE_COLS=source_cols,
        S0_N_COL_BLOCKS=ceil_div(s0_logical_cols, CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM),
        S1_N_COL_BLOCKS=ceil_div(s1_logical_cols, CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM),
        # Tiling.
        SF_VEC=sf_vec,
        M_TILE=m_tile,
        K_TILE=k_tile,
        grid_m_tiles=grid[0],
        GRID_K_TILES=grid[1],
        num_warps=tile_config.num_warps,
    )

    if is_two_d:
        # FP8 TWO_D: caller transposes qdata_row for axis=0 use, so return
        # None for the col-major slot. FP4 TWO_D needs both layouts.
        q0_ret = q0 if is_fp4 else None
        return (q0_ret, s0), (q1, s1)
    if has_axis0 and has_axis1:
        return (q0, s0), (q1, s1)
    if has_axis0:
        return q0, s0
    return q1, s1
