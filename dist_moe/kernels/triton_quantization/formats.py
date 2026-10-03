# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block-scaled quantization enums and constants.

This module is intentionally limited to enum definitions, integer constants,
and pure lookup tables. Kernel and interface implementations may import these
values, but this file must not import backend kernel modules.

The public enums are:

* ``BlockScaledFormat`` — operand precision / SF-vec-size triple
  (MXFP8_E4M3 / MXFP8_E5M2 / NVFP4 / MXFP4 / MXFP6_E3M2 / MXFP6_E2M3).
* ``ScaleFactorLayout`` — in-memory layout of the scale tensor (NATURAL,
  CUBLAS_BLOCKED). Both layouts share the same per-atom byte
  swizzle, so CUBLAS_BLOCKED feeds the cuBLAS / MSLK CUTLASS path
  AND the cute kernel directly (the latter reinterprets it as a 1D
  byte stream via ``tile_atom_to_shape_SF``).
* ``Backend`` — interface backend selector shared by quantize and dequantize.
* ``BlockScaledFormatId`` — kernel ABI IDs for ``BlockScaledFormat``.
* ``ScaleFactorLayoutId`` — kernel ABI IDs for ``ScaleFactorLayout``.
* ``AxisMask`` — kernel ABI bitmask for axis-mode quantization.
* ``ScaleReduction`` — kernel ABI mode for 1D vs 2D scale reductions.
* ``BlockScaledProducer`` — kernel ABI IDs for fused producer variants.

Plus the block-scaled format → ``(operand_dtype, sf_dtype, sf_vec_size)``
lookup, shared type aliases, and host-side naming helpers.
"""

import enum
from dataclasses import dataclass
from typing import TypeAlias

import torch


class BlockScaledFormatId(enum.IntEnum):
    """Kernel ABI IDs for ``BlockScaledFormat`` values; do not reorder."""

    MXFP8_E4M3 = 0
    MXFP8_E5M2 = 1
    NVFP4 = 2
    MXFP4 = 3
    MXFP6_E3M2 = 4
    MXFP6_E2M3 = 5
    NVFP4_UE5M3 = 6


class ScaleFactorLayoutId(enum.IntEnum):
    """Kernel ABI IDs for ``ScaleFactorLayout`` values; do not reorder."""

    NATURAL = 0
    CUBLAS_BLOCKED = 1


class AxisMask(enum.IntFlag):
    """Kernel ABI mask for selecting quantization axes."""

    M = 1
    K = 2


class ScaleReduction(enum.IntEnum):
    """Scale-reduction mode for axis-mode quantization.

    The integer values are the kernel ABI IDs consumed by the Triton /
    CuTe quantize kernels — do not reorder.

    Attributes:
        ONE_D: One set of scales per axis (one reduction along each
            requested axis).
        TWO_D: A single qdata tensor is shared by both axes, with a
            2D scale tile.
    """

    ONE_D = 0
    TWO_D = 1


BlockSize: TypeAlias = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _AxisPlan:
    """Internal axis-mode plan shared by quantize and dequantize interfaces."""

    axes: tuple[int, ...]
    scale_reduction: ScaleReduction


class BlockScaledProducer(enum.IntEnum):
    """Elementwise producer evaluated before block-scaled quantization.

    The integer values are the kernel ABI IDs consumed by the Triton
    fused quant + producer kernels — do not reorder.

    Attributes:
        IDENTITY: No producer; quantize the source operand directly.
        SWIGLU_FWD: SwiGLU forward (``x * sigmoid(x) * y``).
        SWIGLU_BWD_DXY: SwiGLU backward producing logical
            ``concat(dx, dy)`` for the fc13 backprop path.
        ZEROCOPY_GATHER: Per-row NVLink remote-pointer gather. The source
            tile is loaded by indexing ``gather_ptrs[row]`` into peer
            symmetric-memory buffers — no transform after load. Sentinel
            pointer ``0`` zero-fills the row (matches ``zerocopy_gather``
            padding semantics). Composes with all ``axis_mask`` /
            ``scale_reduction`` modes; only ``producer_id=IDENTITY`` is
            valid alongside ``ZEROCOPY_GATHER`` since both transform the
            load-time data.
    """

    IDENTITY = 0
    SWIGLU_FWD = 1
    SWIGLU_BWD_DXY = 2
    ZEROCOPY_GATHER = 3


# GPT-OSS clamped SwiGLU defaults. Shared by the bf16 kernels and the fused
# SwiGLU+quant kernels so both compile against the same constants.
SWIGLU_CLAMP_ALPHA_DEFAULT = 1.702
SWIGLU_CLAMP_LIMIT_DEFAULT = 7.0


def canonical_swiglu_clamp(
    clamped: bool,
    alpha: float,
    limit: float,
) -> tuple[bool, float, float]:
    """The `(clamped, alpha, limit)` triple the SwiGLU kernels specialize on.

    Plain SwiGLU ignores `alpha` / `limit`, so pin them to the defaults. That
    keeps one kernel cache entry for every plain call.
    """
    if not clamped:
        return False, SWIGLU_CLAMP_ALPHA_DEFAULT, SWIGLU_CLAMP_LIMIT_DEFAULT
    return True, float(alpha), float(limit)


class BlockScaledFormat(enum.StrEnum):
    """Operand precision / SF-vec-size triple for block-scaled GEMM.

    The choice of format fixes ``(operand_dtype, sf_dtype, sf_vec_size)``
    for the kernel.

    Attributes:
        MXFP8_E4M3: FP8 E4M3 operands, FP8_E8M0FNU scales, vec_size=32.
            Standard MoE-FFN low-precision path.
        MXFP8_E5M2: FP8 E5M2 operands, FP8_E8M0FNU scales, vec_size=32.
            Wider exponent for activations with larger dynamic range.
        NVFP4: FP4 E2M1FN operands, FP8_E4M3FN scales, vec_size=16.
            NVIDIA NVFP4 spec.
        MXFP4: FP4 E2M1FN operands, FP8_E8M0FNU scales, vec_size=32.
            OCP MXFP4 spec — same operand dtype as NVFP4 but different
            scale dtype + vec_size.
        MXFP6_E3M2: FP6 E3M2 operands stored as uint8 codes, FP8_E8M0FNU
            scales, vec_size=32. The low six bits of each uint8 hold one
            sign/exponent/mantissa code; the high two bits are zero.
        MXFP6_E2M3: FP6 E2M3 operands stored as uint8 codes, FP8_E8M0FNU
            scales, vec_size=32. The low six bits of each uint8 hold one
            sign/exponent/mantissa code; the high two bits are zero.
        NVFP4_UE5M3: FP4 E2M1FN operands, unsigned E5M3 scales (PTX ISA 9.4
            ``ue5m3``; no torch dtype, so scales are uint8 codes),
            vec_size=16. Same data format as NVFP4 with a wider-exponent
            scale; fake-quant only (no packed GEMM kernel).
    """

    MXFP8_E4M3 = enum.auto()
    MXFP8_E5M2 = enum.auto()
    NVFP4 = enum.auto()
    MXFP4 = enum.auto()
    MXFP6_E3M2 = enum.auto()
    MXFP6_E2M3 = enum.auto()
    NVFP4_UE5M3 = enum.auto()


class Backend(enum.StrEnum):
    """Backend selector shared by block-scaled quantize/dequantize interfaces."""

    NATIVE = enum.auto()
    TRITON = enum.auto()
    CUTE = enum.auto()


class ScaleFactorLayout(enum.StrEnum):
    """SF layout in memory.

    Different backends consume scales in different layouts. The
    interface accepts scales in any of these formats and re-swizzles
    on the fly when the chosen backend wants something else.

    Attributes:
        NATURAL: Per-token scales in ``[M, K // sf_vec_size]`` shape,
            row-major. What ``triton_to_mxfp8_dim0`` /
            ``triton_quant_mxfp8`` produce. Converted on-the-fly for
            CuTeDSL / CUTLASS / NATIVE.
        CUBLAS_BLOCKED: cuBLAS FP8 block-scaling layout (the
            ``(padded_rows, n_col_blocks * 16)`` swizzled form
            produced by ``to_blocked_2d`` / ``to_blocked``). The
            per-atom byte swizzle ``(row%32)*16 + (row//32)*4 + col``
            and outer M-slow / K-fast atom ordering match the cutlass
            ``BlockScaledBasicChunk`` layout, so the same byte stream
            feeds CUTLASS / MSLK / ``torch._scaled_mm`` AND the
            CuTeDSL kernel (which reinterprets it as a 1D byte stream
            via ``tile_atom_to_shape_SF``).
    """

    NATURAL = enum.auto()
    CUBLAS_BLOCKED = enum.auto()


BLOCK_SCALED_FORMAT_IDS = {
    BlockScaledFormat.MXFP8_E4M3: BlockScaledFormatId.MXFP8_E4M3.value,
    BlockScaledFormat.MXFP8_E5M2: BlockScaledFormatId.MXFP8_E5M2.value,
    BlockScaledFormat.NVFP4: BlockScaledFormatId.NVFP4.value,
    BlockScaledFormat.MXFP4: BlockScaledFormatId.MXFP4.value,
    BlockScaledFormat.MXFP6_E3M2: BlockScaledFormatId.MXFP6_E3M2.value,
    BlockScaledFormat.MXFP6_E2M3: BlockScaledFormatId.MXFP6_E2M3.value,
    BlockScaledFormat.NVFP4_UE5M3: BlockScaledFormatId.NVFP4_UE5M3.value,
}

SCALE_FACTOR_LAYOUT_IDS = {
    ScaleFactorLayout.NATURAL: ScaleFactorLayoutId.NATURAL.value,
    ScaleFactorLayout.CUBLAS_BLOCKED: ScaleFactorLayoutId.CUBLAS_BLOCKED.value,
}

FP8_FORMATS: tuple[BlockScaledFormat, ...] = (
    BlockScaledFormat.MXFP8_E4M3,
    BlockScaledFormat.MXFP8_E5M2,
)

# NVFP4_UE5M3 counts as FP4 everywhere (E2M1 payload and fake-quant dispatch)
# but has no real quantize kernels: real-path consumers
# must exclude ``FAKE_QUANT_ONLY_FORMATS``.
FP4_FORMATS: tuple[BlockScaledFormat, ...] = (
    BlockScaledFormat.NVFP4,
    BlockScaledFormat.MXFP4,
    BlockScaledFormat.NVFP4_UE5M3,
)

# Logical elements per storage byte, keyed by torch dtype (not by format, unlike
# `FP4_FORMATS`). Re-deriving this elsewhere risks missing a sub-byte dtype, and
# a miss is a silently halved shape: `view(G, -1, logical_dim)` succeeds on
# packed codes.
ELEMENTS_PER_BYTE: dict[torch.dtype, int] = {torch.float4_e2m1fn_x2: 2}


def elements_per_byte(dtype: torch.dtype) -> int:
    """Logical elements packed into one byte of `dtype`; 1 for every plain dtype."""
    return ELEMENTS_PER_BYTE.get(dtype, 1)


def logical_trailing_dim(tensor: torch.Tensor) -> int:
    """Trailing extent of `tensor` in logical elements, undoing sub-byte packing."""
    return tensor.shape[-1] * elements_per_byte(tensor.dtype)


def logical_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    """Shape of `tensor` in logical elements, undoing sub-byte packing."""
    return (*tensor.shape[:-1], logical_trailing_dim(tensor))


MXFP8_FORMAT_IDS: tuple[int, ...] = tuple(
    BLOCK_SCALED_FORMAT_IDS[fmt] for fmt in FP8_FORMATS
)
FP4_FORMAT_IDS: tuple[int, ...] = tuple(
    BLOCK_SCALED_FORMAT_IDS[fmt] for fmt in FP4_FORMATS
)

FP6_FORMATS: tuple[BlockScaledFormat, ...] = (
    BlockScaledFormat.MXFP6_E3M2,
    BlockScaledFormat.MXFP6_E2M3,
)
MXFP6_FORMAT_IDS: tuple[int, ...] = tuple(
    BLOCK_SCALED_FORMAT_IDS[fmt] for fmt in FP6_FORMATS
)

ALL_BLOCKSCALED_FORMATS: tuple[BlockScaledFormat, ...] = (
    FP8_FORMATS + FP4_FORMATS + FP6_FORMATS
)

# Formats with a fake-quantize implementation only. No packed quantize /
# dequantize kernels exist (the UE5M3 scale has no torch dtype and no
# hardware below sm_107f): ``quantize_for_format`` rejects these while
# ``fake_quantize_for_format`` accepts them.
FAKE_QUANT_ONLY_FORMATS: tuple[BlockScaledFormat, ...] = (
    BlockScaledFormat.NVFP4_UE5M3,
)

# NVFP4 and its UE5M3-scaled sibling: same E2M1 data path and sf_vec_size,
# same optional two-level global scale; only the scale dtype differs.
NVFP4_VARIANT_FORMATS: tuple[BlockScaledFormat, ...] = (
    BlockScaledFormat.NVFP4,
    BlockScaledFormat.NVFP4_UE5M3,
)
NVFP4_VARIANT_FORMAT_IDS: tuple[int, ...] = tuple(
    BLOCK_SCALED_FORMAT_IDS[f] for f in NVFP4_VARIANT_FORMATS
)

# Formats that default to the half-range exponent shift rather than plain
# rceil. The quantize interfaces take ``half_range_scale=None`` and resolve it
# through ``resolve_half_range_scale``; an explicit bool always wins.
#
# Adding MXFP8 here would break two things: bitwise parity with the
# pinned reference scale rule and the fused SwiGLU
# quant kernels, which bail out whenever the flag is set and are reachable only
# for MXFP8. Only E8M0-scaled formats belong here at all: NVFP4 is a no-op (its
# E4M3 scale already lands on the finer encoding) so listing it would imply a
# behavior change that does not exist, and MXFP6 is rejected outright.
HALF_RANGE_SCALE_DEFAULT_FORMATS: tuple[BlockScaledFormat, ...] = (
    BlockScaledFormat.MXFP4,
)

FORMAT_TAG = {
    BlockScaledFormatId.MXFP8_E4M3.value: "mxfp8_e4m3",
    BlockScaledFormatId.MXFP8_E5M2.value: "mxfp8_e5m2",
    BlockScaledFormatId.NVFP4.value: "nvfp4",
    BlockScaledFormatId.MXFP4.value: "mxfp4",
    BlockScaledFormatId.NVFP4_UE5M3.value: "nvfp4_ue5m3",
}
LAYOUT_TAG = {
    ScaleFactorLayoutId.NATURAL.value: "natural",
    ScaleFactorLayoutId.CUBLAS_BLOCKED.value: "blocked",
}
FORMAT_SF_VEC = {
    BlockScaledFormatId.MXFP8_E4M3.value: 32,
    BlockScaledFormatId.MXFP8_E5M2.value: 32,
    BlockScaledFormatId.NVFP4.value: 16,
    BlockScaledFormatId.MXFP4.value: 32,
    BlockScaledFormatId.NVFP4_UE5M3.value: 16,
}
PRODUCER_TAG = {
    BlockScaledProducer.IDENTITY.value: "id",
    BlockScaledProducer.SWIGLU_FWD.value: "swiglu_fwd",
    BlockScaledProducer.SWIGLU_BWD_DXY.value: "swiglu_bwd",
    BlockScaledProducer.ZEROCOPY_GATHER.value: "zerocopy_gather",
}


def build_block_tile_tag(
    axis_mask: int | None,
    scale_reduction: int | None,
    sf_vec: int,
) -> str:
    """Format the SF tile shape as ``RxC`` (rows x cols)."""
    has_m = axis_mask is not None and bool(axis_mask & AxisMask.M.value)
    has_k = axis_mask is not None and bool(axis_mask & AxisMask.K.value)
    if axis_mask is None:
        return f"1x{sf_vec}"
    if has_m and has_k:
        if scale_reduction == ScaleReduction.TWO_D.value:
            return f"{sf_vec}x{sf_vec}"
        return f"{sf_vec}x1_1x{sf_vec}"
    if has_m:
        return f"{sf_vec}x1"
    if has_k:
        return f"1x{sf_vec}"
    return f"1x{sf_vec}"


# Numeric limits of the supported block-scaled element formats. Single source
# of truth: the Triton kernels mirror these as ``tl.constexpr`` and the CuTe
# kernels as ``cutlass.Constexpr[float]`` (a plain Python global cannot be read
# from inside a ``@triton.jit`` kernel, so each backend wraps these values);
# ``quantize.py``'s native path uses them directly.
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
FP4_E2M1_MAX = 6.0
# fl(1/6) for the NVFP4 raw-scale order ``(amax * gs) * fl(1/6)``. Public and
# single-copy for the same reason as ``NVFP4_MAX``: every producer kernel must
# multiply by the identical fp32 reciprocal (the order and the constant are
# load-bearing at E4M3 RTNE ties — see the Triton ``_nvfp4_scale_and_recip``).
FP4_E2M1_MAX_RECIP = 1.0 / FP4_E2M1_MAX
# NVFP4 two-level numerator: FP8 E4M3 max (448) times FP4 E2M1 max (6).
# Public: every producer synthesizing an NVFP4 global scale (and any bench or
# test mirroring the production scale) must share this value.
NVFP4_MAX = FP8_E4M3_MAX * FP4_E2M1_MAX
# Keeps an E4M3-rounded NVFP4 scale from shrinking below the no-clipping scale.
NVFP4_NO_CLIP_TARGET_MAX = FP4_E2M1_MAX * (16.0 / 17.0)
# Unsigned E5M3 scale, per PTX ISA 9.4: 5-bit exponent, 3-bit mantissa, no
# sign, no infinity, NaN limited to the single byte 0xFF. The ISA gives only
# the structure; the numbers below follow its unsigned siblings, where "NaN
# limited to 0xNN" reserves exactly one code (ue4m3 max 0x7E = 448, ue8m0
# max 0xFE = 2^127): exponent bias 15 (E5M2's), max normal 0xFE =
# 1.75 * 2^16 = 114688, min normal 2^-14, min subnormal 2^-17, ~34 binades.
# (arXiv:2609.02846 section 4 assumes an E5M2-style reserved top exponent
# instead, max 61440. Block scales are amax/6 and never get near either
# bound.)
UE5M3_EXP_BIAS = 15
UE5M3_MAX = 114688.0
UE5M3_MIN_SUBNORMAL = 2.0**-17
UE5M3_NAN_BYTE = 255
UE5M3_MAX_FINITE_CODE = 0xFE
# NVFP4_UE5M3 two-level numerator, analogous to ``NVFP4_MAX``.
NVFP4_UE5M3_MAX = UE5M3_MAX * FP4_E2M1_MAX
MXFP_SCALE_BLOCK_SIZE = 32
# Half-range exponent-shift round boundaries (enabled by ``half_range_scale``).
FP8_E4M3_HALF_RANGE_ROUND_BOUNDARY = 232.0
FP8_E5M2_HALF_RANGE_ROUND_BOUNDARY = 30720.0
FP4_E2M1_HALF_RANGE_ROUND_BOUNDARY = 3.5
# Smallest subnormal E4M3 value (2^-9); NVFP4 scales clamp here before the FP8
# cast so very small block maxes keep fine scale resolution (the E4M3 scale
# dtype supports subnormals, so there is no reason to leave that range unused).
E4M3_MIN_SUBNORMAL = 0.001953125
# Floor applied to NVFP4 per-token / per-segment amax before the reciprocal:
# a denormal-only row (bf16 subnormals reach ~1e-41) would otherwise overflow
# the fp32 division to inf. Matches the torch reference's max(amax, 1e-8).
NVFP4_TOKEN_AMAX_FLOOR = 1.0e-8
# Amax floor shared by every NVFP4 weight / fake-quant / whole-tensor
# global-scale producer (`nvfp4_weight_global_scale`, the grouped and
# emulated per-expert gs): keeps an all-zero operand from dividing to inf
# while staying far below any real amax. Public because the floor value
# participates in the bitwise gs contract wherever it engages.
NVFP4_GLOBAL_SCALE_EPS = 1e-12
# E8M0 (MX) scale-byte exponent parameters.
E8M0_EXP_BIAS = 127.0
# Biased E8M0 byte encoding NaN (inf/NaN block maxes map here, not saturation).
E8M0_NAN_BYTE = 255
E8M0_MIN_UNBIASED_EXP = -127.0
E8M0_MAX_UNBIASED_EXP = 127.0

# cuBLAS-blocked scale-factor "atom" swizzle geometry (the layout cuBLAS expects
# for MX scale factors): 128 rows x 4 scale-columns per atom, each atom split
# into 32 row-lanes of 16 entries. Drives the CUBLAS_BLOCKED scale-store offset
# math in both the Triton and CuTe kernels.
CUBLAS_BLOCKED_ROWS_PER_ATOM = 128
CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM = 4
CUBLAS_BLOCKED_ROW_LANES = 32
CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE = 16
CUBLAS_BLOCKED_ENTRIES_PER_ATOM = (
    CUBLAS_BLOCKED_ROWS_PER_ATOM * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
)

# E2M1FN LUT (sign-magnitude): values 0..7 are positive, 8..15 share
# the same magnitude as 0..7 with the sign bit set.
FP4_E2M1_TABLE: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# FP6 LUTs use sign-magnitude code order: codes 0..31 are non-negative,
# and codes 32..63 mirror them with the sign bit set.
FP6_E3M2_TABLE: tuple[float, ...] = (
    0.0,
    0.0625,
    0.125,
    0.1875,
    0.25,
    0.3125,
    0.375,
    0.4375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    10.0,
    12.0,
    14.0,
    16.0,
    20.0,
    24.0,
    28.0,
    -0.0,
    -0.0625,
    -0.125,
    -0.1875,
    -0.25,
    -0.3125,
    -0.375,
    -0.4375,
    -0.5,
    -0.625,
    -0.75,
    -0.875,
    -1.0,
    -1.25,
    -1.5,
    -1.75,
    -2.0,
    -2.5,
    -3.0,
    -3.5,
    -4.0,
    -5.0,
    -6.0,
    -7.0,
    -8.0,
    -10.0,
    -12.0,
    -14.0,
    -16.0,
    -20.0,
    -24.0,
    -28.0,
)

FP6_E2M3_TABLE: tuple[float, ...] = (
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.125,
    1.25,
    1.375,
    1.5,
    1.625,
    1.75,
    1.875,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    -0.0,
    -0.125,
    -0.25,
    -0.375,
    -0.5,
    -0.625,
    -0.75,
    -0.875,
    -1.0,
    -1.125,
    -1.25,
    -1.375,
    -1.5,
    -1.625,
    -1.75,
    -1.875,
    -2.0,
    -2.25,
    -2.5,
    -2.75,
    -3.0,
    -3.25,
    -3.5,
    -3.75,
    -4.0,
    -4.5,
    -5.0,
    -5.5,
    -6.0,
    -6.5,
    -7.0,
    -7.5,
)


def resolve_half_range_scale(
    format: BlockScaledFormat,
    half_range_scale: bool | None,
) -> bool:
    """Resolve ``half_range_scale``, mapping ``None`` to the format default.

    The scale is amax-based so fused and composed quantization of the same
    tensor produce byte-identical scales.
    """
    if half_range_scale is not None:
        return half_range_scale
    return format in HALF_RANGE_SCALE_DEFAULT_FORMATS


def block_scaled_format_constants(
    format: BlockScaledFormat,
) -> tuple[torch.dtype, torch.dtype, int]:
    """Return ``(operand_dtype, sf_dtype, sf_vec_size)`` for the format."""
    match format:
        case BlockScaledFormat.MXFP8_E4M3:
            return torch.float8_e4m3fn, torch.float8_e8m0fnu, 32
        case BlockScaledFormat.MXFP8_E5M2:
            return torch.float8_e5m2, torch.float8_e8m0fnu, 32
        case BlockScaledFormat.NVFP4:
            return torch.float4_e2m1fn_x2, torch.float8_e4m3fn, 16
        case BlockScaledFormat.MXFP4:
            return torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, 32
        case BlockScaledFormat.NVFP4_UE5M3:
            # No torch e5m3 dtype: scales are uint8 codes (fake-quant only).
            return torch.float4_e2m1fn_x2, torch.uint8, 16
        case BlockScaledFormat.MXFP6_E3M2 | BlockScaledFormat.MXFP6_E2M3:
            return torch.uint8, torch.float8_e8m0fnu, 32
        case _:
            raise ValueError(f"Unsupported block-scaled format: {format}")
