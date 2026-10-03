# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL block-scaled quantization kernels.

Supported formats: MXFP8 (E4M3 / E5M2), MXFP4, and NVFP4.

Design: shared quantization numerics, CuTe kernel orchestration
===============================================================
Scale rules, NaN propagation, FP8/FP4 conversion, and half-range
canonicalization are expressed by the helpers below. The launched CuTe kernels
add tile, layout, and address calculations without changing those rules.

Per-block numerics pipeline
===========================
Each block flows through the shared helpers in this order (bracketed steps are
optional):

    amax -> scale + reciprocal ->[half-range]-> cast / clamp -> pack

1. Reduction
   - ``_frag_amax_nonfinite`` -- SM100 3-input ``max.NaN`` amax trees with a
     single ``NaN -> inf`` substitution per block (``_compute_block_amax_nonfinite``
     applied once to the reduced value; bitwise identical to the per-element form).

2. Scale + reciprocal
   - ``_compute_fp8_e8m0_scale_rceil`` -- MXFP8 / MXFP4 round-up E8M0 scale byte +
     reciprocal.
   - ``_compute_nvfp4_scale_byte`` -- NVFP4 E4M3 scale policy; its reciprocal is
     reconstructed directly or read from a precomputed LUT.

3. Half-range canonicalization -- optional, ``half_range_scale=True``
   - ``_canonicalize_mx_half_range`` -- E8M0 exponent shift across MXFP8
     (E4M3/E5M2) and MXFP4, selecting the per-format midpoint via
     ``format_id``.

4. Cast / clamp and pack
   - ``_clamp_to_target_max`` -- clamp scaled qdata into the target dtype range
     while keeping NaN visible to the downstream PTX conversion.

CuTeDSL needs extra PTX-wrapper primitives such as ``_cvt_f32x2_to_fp8_e8m0x2_rp``,
and ``_cvt_f32x8_to_fp4_e2m1x8_u32_rn`` because the language has no equivalent of
Triton's ``.to(tl.float8e4nv)`` builtin. Those conversion and packing helpers
live in the sibling ``_quant_conversion`` and ``_quant_packing`` modules; this
file keeps tile/layout orchestration,
producer routing, host helpers, and public entries.
"""

import math
from dataclasses import dataclass
from typing import cast

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass._mlir.dialects import nvvm

from ..formats import (
    AxisMask,
    block_scaled_format_constants,
    BLOCK_SCALED_FORMAT_IDS,
    BlockScaledFormat,
    BlockScaledFormatId,
    BlockScaledProducer,
    build_block_tile_tag as _build_block_tile_tag,
    canonical_swiglu_clamp,
    CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE,
    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
    FORMAT_SF_VEC as _FORMAT_SF_VEC,
    FORMAT_TAG as _FORMAT_TAG,
    FP4_E2M1_MAX,
    FP4_FORMAT_IDS as _FP4_FORMAT_IDS,
    LAYOUT_TAG as _LAYOUT_TAG,
    MXFP8_FORMAT_IDS as _MXFP8_FORMAT_IDS,
    PRODUCER_TAG as _PRODUCER_TAG,
    SCALE_FACTOR_LAYOUT_IDS,
    ScaleFactorLayout,
    ScaleFactorLayoutId,
    ScaleReduction,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)
from . import _dsl_compat as _cute_extern  # noqa: F401
from ._cuda_context import ensure_cuda_driver_context
from ._jit_cache import jit_cache
from ._kernel_name_prefix import scoped_kernel_name_prefixes
from ._quack_compile_utils import make_fake_stream, make_fake_tensor as fake_tensor
from ._quant_conversion import (
    _abs_f32,
    _ceil_div_i32,
    _cvt_f32x4_to_fp8x4_u32_rn,
    _unpack_b16x2,
)
from ._swiglu_quant import (
    _load_swiglu_fwd_nvfp4_block,
    _quantize_swiglu_bwd_dxy_fp8_tile,
    _quantize_swiglu_fwd_fp8_tile,
)
from .blockscaled_quantization_common import (
    _apply_block_scaled_quant_dxy_producer,
    _apply_block_scaled_quant_producer,
    _AXES_KERNEL_MODE_BOTH1D_VECTOR,
    _AXES_KERNEL_MODE_SCALAR,
    _AXES_KERNEL_MODE_SWIGLU_BWD_DXY_F32X2_VECTOR,
    _AXES_KERNEL_MODE_TILE2D_FP4_VECTOR,
    _AXES_KERNEL_MODE_TILE2D_FP8_VECTOR,
    _clamp_to_target_max,
    _compute_amax_nonfinite_b16x2,
    _compute_block_amax_nonfinite,
    _compute_nvfp4_token_scales,
    _compute_row_amax_b16x2_x16,
    _compute_scale_and_recip_from_amax,
    _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut,
    _copy_load_f16xn_as_f32,
    _cublas_blockscaled_qscale_offset,
    _fmax_nan,
    _frag_amax_nonfinite,
    _load_gather_row_as_b16x2,
    _max_b16x2,
    _quantize_fp8_tile_values,
    _reduce_amax_b16x2,
    _reduce_amax_f32,
    _reduce_nvfp4_block_amaxes,
    _scale_nvfp4_per_token_block,
    _store_blockscaled_xn_words,
    _store_fp8_qwords,
    _store_qdata_axis0_words as _store_qdata_col_words,
    _store_qdata_col_major_fp4,
    _store_qdata_col_major_fp8,
    _store_qdata_row,
    _store_qdata_row_words,
    _target_max_for_format_id,
    _warp_reduce_amax_f32,
)

_INPUT_DTYPE_TAG = {
    BFloat16: "bf16",
    Float16: "f16",
    Float32: "f32",
}

_ROW_COL_KERNEL_MODE_SWIGLU_FWD_FP8_TILE = 5
_ROW_COL_KERNEL_MODE_SWIGLU_BWD_DXY_FP8_TILE = 6
_FP8_QDATA_ROW_ALIGNMENT_BYTES = 16


# =============================================================================
# Constants (DSL-visible integer views + layout sizes)
# =============================================================================


_TORCH_TO_CUTE_INPUT_DTYPE = {
    torch.bfloat16: BFloat16,
    torch.float16: Float16,
    torch.float32: Float32,
}

# CuTe JIT branches need module-level Python integer constants for
# specialization. The public enums in ``interfaces.quant.formats`` remain the
# source of truth; these aliases are the DSL-visible views of those values.
BLOCK_SCALED_FORMAT_MXFP8_E4M3 = BlockScaledFormatId.MXFP8_E4M3.value
BLOCK_SCALED_FORMAT_MXFP8_E5M2 = BlockScaledFormatId.MXFP8_E5M2.value
BLOCK_SCALED_FORMAT_NVFP4 = BlockScaledFormatId.NVFP4.value
BLOCK_SCALED_FORMAT_MXFP4 = BlockScaledFormatId.MXFP4.value

SCALE_FACTOR_LAYOUT_NATURAL = ScaleFactorLayoutId.NATURAL.value
SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED = ScaleFactorLayoutId.CUBLAS_BLOCKED.value
BLOCK_SCALED_PRODUCER_IDENTITY = BlockScaledProducer.IDENTITY.value
BLOCK_SCALED_PRODUCER_SWIGLU_FWD = BlockScaledProducer.SWIGLU_FWD.value
BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY = BlockScaledProducer.SWIGLU_BWD_DXY.value
BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER = BlockScaledProducer.ZEROCOPY_GATHER.value

AXIS_MASK_M = AxisMask.M.value
AXIS_MASK_K = AxisMask.K.value
SCALE_REDUCTION_ONE_D = ScaleReduction.ONE_D.value
SCALE_REDUCTION_TWO_D = ScaleReduction.TWO_D.value

_BLOCK_SCALED_FORMAT_BY_ID: dict[int, BlockScaledFormat] = {
    v: cast(BlockScaledFormat, k) for k, v in BLOCK_SCALED_FORMAT_IDS.items()
}
_SCALE_FACTOR_LAYOUT_BY_ID: dict[int, ScaleFactorLayout] = {
    v: cast(ScaleFactorLayout, k) for k, v in SCALE_FACTOR_LAYOUT_IDS.items()
}

_NVFP4_RECIP_LUT_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}

_FP32_ZERO: cutlass.Constexpr[float] = 0.0
_FP32_ONE: cutlass.Constexpr[float] = 1.0

# =============================================================================
# CuTe Helpers
# =============================================================================


def _build_swiglu_clamp_tag(clamped: bool, alpha: float, limit: float) -> str:
    """Kernel-name tag for a clamped specialization.

    Kernel names become PTX symbols, which cannot contain `.` or `-`, so the
    constants are written as hex fp32 bit patterns.
    """
    if not clamped:
        return ""
    bits = torch.tensor([alpha, limit], dtype=torch.float32).view(torch.int32)
    return f"_clamp{int(bits[0]) & 0xFFFFFFFF:08x}_{int(bits[1]) & 0xFFFFFFFF:08x}"


def _make_input_fake_tensor(
    input_dtype: type[cutlass.Numeric],
    shape: tuple[object, object],
) -> cute.Tensor:
    return cute.runtime.make_fake_tensor(
        input_dtype,
        shape,
        stride=(cute.sym_int64(), cute.sym_int64()),
        assumed_align=input_dtype.width // 8,
    )


def _make_row_col_thread_config(
    row_col_kernel_mode: int,
    axis_mask: int,
    producer_id: int,
    is_fp4: bool,
    col1d_use_row_vector: bool,
) -> tuple[int, int]:
    mode_configs = {
        _AXES_KERNEL_MODE_SCALAR: (32, 1),
        _AXES_KERNEL_MODE_SWIGLU_BWD_DXY_F32X2_VECTOR: (128, 4),
        _ROW_COL_KERNEL_MODE_SWIGLU_FWD_FP8_TILE: (128, 4),
        _ROW_COL_KERNEL_MODE_SWIGLU_BWD_DXY_FP8_TILE: (128, 4),
        _AXES_KERNEL_MODE_TILE2D_FP8_VECTOR: (128, 4),
    }
    if row_col_kernel_mode in mode_configs:
        return mode_configs[row_col_kernel_mode]
    if row_col_kernel_mode == _AXES_KERNEL_MODE_BOTH1D_VECTOR:
        if is_fp4:
            return 64, 2
        if axis_mask == AXIS_MASK_M:
            if producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER:
                return 64, 2
            if col1d_use_row_vector:
                return 128, 4
            return 32, 1
        if (
            axis_mask == (AXIS_MASK_M | AXIS_MASK_K)
            and producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
        ):
            return 32, 1
        return 128, 4
    return 64, 2


def _use_row_major_col_qdata(
    *,
    format_id: int,
    scale_reduction_id: int,
    has_col_axis: bool,
    has_row_axis: bool,
    col1d_use_row_vector: bool,
) -> bool:
    return (
        format_id in _MXFP8_FORMAT_IDS
        and scale_reduction_id == SCALE_REDUCTION_ONE_D
        and has_col_axis
        and (has_row_axis or col1d_use_row_vector)
    )


def _use_row_vector_col1d(
    *,
    format_id: int,
    axis_mask: int,
    scale_reduction_id: int,
) -> bool:
    return (
        format_id in _MXFP8_FORMAT_IDS
        and scale_reduction_id == SCALE_REDUCTION_ONE_D
        and axis_mask == AXIS_MASK_M
    )


@cute.jit
def _load_f16_words_as_f32(
    mWords: cute.Tensor,
    row,
    col_start,
    source_dtype: type[cutlass.Numeric],
    n_values: cutlass.Constexpr[int],
):
    if cutlass.const_expr(source_dtype != BFloat16 and source_dtype != Float16):
        raise TypeError("f16 word row loads require BF16 or FP16 source dtype")
    if cutlass.const_expr(n_values < 8 or n_values % 8 != 0):
        raise ValueError("f16 word row loads require a positive multiple of 8 values")

    values = cute.make_rmem_tensor(n_values, Float32)
    chunk_elems: cutlass.Constexpr[int] = 16
    full_chunks: cutlass.Constexpr[int] = n_values // chunk_elems
    tail_elems: cutlass.Constexpr[int] = n_values - full_chunks * chunk_elems
    if cutlass.const_expr(tail_elems != 0 and tail_elems != 8):
        raise ValueError("f16 word row load tail must be 0 or 8 values")

    for chunk in cutlass.range_constexpr(full_chunks):
        base: cutlass.Constexpr[int] = chunk * chunk_elems
        loaded = _copy_load_f16xn_as_f32(
            mWords,
            row,
            (col_start + Int32(base)) // Int32(2),
            src_dtype=source_dtype,
            num_elems=chunk_elems,
        )
        for i in cutlass.range_constexpr(chunk_elems):
            values[base + i] = loaded[i]

    if cutlass.const_expr(tail_elems == 8):
        base: cutlass.Constexpr[int] = full_chunks * chunk_elems
        loaded = _copy_load_f16xn_as_f32(
            mWords,
            row,
            (col_start + Int32(base)) // Int32(2),
            src_dtype=source_dtype,
            num_elems=tail_elems,
        )
        for i in cutlass.range_constexpr(tail_elems):
            values[base + i] = loaded[i]
    return values


# =============================================================================
# CuTe Classes
# =============================================================================


# -----------------------------------------------------------------------------
# Row-1D quantization
# -----------------------------------------------------------------------------


class _QuantizeBlockScaledRow:
    def __init__(
        self,
        input_dtype: type[cutlass.Numeric],
        format_id: int,
        layout_id: int,
        producer_id: int,
        half_range_scale: bool,
        fast_math: bool,
        clamped: bool,
        alpha: float,
        limit: float,
        x_words_aligned: bool,
        producer_b_words_aligned: bool,
    ) -> None:
        self.input_dtype = input_dtype
        self.format_id = format_id
        self.layout_id = layout_id
        self.producer_id = producer_id
        self.half_range_scale = half_range_scale
        self.fast_math = fast_math
        self.clamped = clamped
        self.alpha = alpha
        self.limit = limit
        self.x_words_aligned = x_words_aligned
        self.producer_b_words_aligned = producer_b_words_aligned
        self.is_fp4 = format_id in _FP4_FORMAT_IDS
        self.sf_vec_size = _get_sf_vec_size_for_format_id(format_id)
        self.lane_elems = 16
        self.lane_group = self.sf_vec_size // self.lane_elems
        self.num_threads = 128
        if format_id == BLOCK_SCALED_FORMAT_NVFP4:
            self.m_tile = 128
        elif format_id == BLOCK_SCALED_FORMAT_MXFP4:
            self.m_tile = 64
        else:
            self.m_tile = 32
        self.cta_iters = (self.m_tile * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM) // (
            self.num_threads // self.lane_group
        )

    @staticmethod
    def can_implement(
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQBytes: cute.Tensor,
        mQWords: cute.Tensor,
        mScale: cute.Tensor,
        stream: cuda.CUstream,
    ) -> bool:
        return (
            mX.element_type in (BFloat16, Float16, Float32, Int64)
            and mProducerB.element_type in (BFloat16, Float16, Float32)
            and mProducerBWords.element_type == Uint32
            and mXWords.element_type == Uint32
            and mQBytes.element_type == Uint8
            and mQWords.element_type == Uint32
            and mScale.element_type == Uint8
            and stream is not None
        )

    @staticmethod
    @jit_cache
    def compile(
        input_dtype: type[cutlass.Numeric],
        format_id: int,
        layout_id: int,
        producer_id: int,
        half_range_scale: bool,
        fast_math: bool,
        clamped: bool,
        alpha: float,
        limit: float,
        x_words_aligned: bool,
        producer_b_words_aligned: bool,
    ) -> object:
        rows = cute.sym_int()
        K = cute.sym_int()
        q_cols = cute.sym_int()
        x_word_cols = cute.sym_int()
        q_word_cols = cute.sym_int()
        scale_elems = cute.sym_int()
        is_zerocopy_gather = producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
        x_cute = (
            fake_tensor(Int64, (rows,), 1)
            if is_zerocopy_gather
            else _make_input_fake_tensor(input_dtype, (rows, K))
        )
        producer_b_cute = (
            _make_input_fake_tensor(input_dtype, (cute.sym_int(), cute.sym_int()))
            if is_zerocopy_gather
            else _make_input_fake_tensor(input_dtype, (rows, K))
        )
        producer_b_words_div = 4 if producer_b_words_aligned else 1
        word_rows = cute.sym_int() if is_zerocopy_gather else rows
        producer_b_words_cute = fake_tensor(
            Uint32, (word_rows, x_word_cols), producer_b_words_div
        )
        x_words_div = 4 if x_words_aligned else 1
        x_words_cute = fake_tensor(Uint32, (word_rows, x_word_cols), x_words_div)
        q_bytes_cute = fake_tensor(Uint8, (rows, q_cols), 1)
        q_words_cute = fake_tensor(Uint32, (rows, q_word_cols), 4)
        scale_cute = (
            fake_tensor(Uint8, (rows, scale_elems), 1)
            if layout_id == SCALE_FACTOR_LAYOUT_NATURAL
            else fake_tensor(Uint8, (scale_elems,), 1)
        )
        layout_tag = _LAYOUT_TAG.get(layout_id, f"layout{layout_id}")
        fmt_tag = _FORMAT_TAG.get(format_id, f"fmt{format_id}")
        dtype_tag = _INPUT_DTYPE_TAG[input_dtype]
        producer_tag = _PRODUCER_TAG.get(producer_id, f"prod{producer_id}")
        producer_prefix = f"{producer_tag}_"
        sf_vec = _get_sf_vec_size_for_format_id(format_id)
        # Single-axis kernel: K-axis quant, scales shared across V cols per row.
        block_tile = _build_block_tile_tag(None, None, sf_vec)
        fastmath_tag = "_fastmath" if fast_math else ""
        clamp_tag = _build_swiglu_clamp_tag(clamped, alpha, limit)
        x_words_tag = "_xpack" if x_words_aligned else "_xscalar"
        name_prefix = f"_cute_quantize_{producer_prefix}{dtype_tag}_{fmt_tag}_{layout_tag}_{block_tile}{fastmath_tag}{clamp_tag}{x_words_tag}"
        with scoped_kernel_name_prefixes(
            ((_QuantizeBlockScaledRow.kernel, name_prefix),)
        ):
            return cute.compile(
                _QuantizeBlockScaledRow(
                    input_dtype,
                    format_id,
                    layout_id,
                    producer_id,
                    half_range_scale,
                    fast_math,
                    clamped,
                    alpha,
                    limit,
                    x_words_aligned,
                    producer_b_words_aligned,
                ),
                x_cute,
                producer_b_cute,
                producer_b_words_cute,
                x_words_cute,
                q_bytes_cute,
                q_words_cute,
                scale_cute,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(1),
                make_fake_stream(),
                options="--enable-tvm-ffi",
            )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQBytes: cute.Tensor,
        mQWords: cute.Tensor,
        mScale: cute.Tensor,
        source_cols: Int32,
        local_rank: Int32,
        world_size: Int32,
        stream: cuda.CUstream,
    ) -> None:
        assert self.can_implement(
            mX,
            mProducerB,
            mProducerBWords,
            mXWords,
            mQBytes,
            mQWords,
            mScale,
            stream,
        )
        assert (
            mX.element_type == Int64
            if cutlass.const_expr(
                self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
            )
            else mX.element_type == self.input_dtype
        )
        K = (
            source_cols
            if cutlass.const_expr(
                self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
            )
            else mX.shape[1]
        )
        scale_cols = K // self.sf_vec_size
        if cutlass.const_expr(self.layout_id == SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED):
            grid = [
                scale_cols // CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
                mX.shape[0] // self.m_tile,
                1,
            ]
        else:
            groups_per_cta: cutlass.Constexpr[int] = self.num_threads // self.lane_group
            total_groups = mX.shape[0] * scale_cols
            grid = [(total_groups + groups_per_cta - 1) // groups_per_cta, 1, 1]
        self.kernel(
            mX,
            mProducerB,
            mProducerBWords,
            mXWords,
            mQWords,
            mScale,
            source_cols,
            local_rank,
            world_size,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQWords: cute.Tensor,
        mScale: cute.Tensor,
        source_cols: Int32,
        local_rank: Int32,
        world_size: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pid_x, pid_y, _ = cute.arch.block_idx()
        rows = mX.shape[0]
        K = (
            source_cols
            if cutlass.const_expr(
                self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
            )
            else mX.shape[1]
        )
        scale_cols = K // self.sf_vec_size

        lane_group: cutlass.Constexpr[int] = self.lane_group
        groups_per_cta: cutlass.Constexpr[int] = self.num_threads // lane_group
        lane = tidx % lane_group
        group_in_cta = tidx // lane_group

        if cutlass.const_expr(self.layout_id == SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED):
            n_col_blocks = scale_cols // CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
            for iter_idx in cutlass.range_constexpr(self.cta_iters):
                block_id = iter_idx * groups_per_cta + group_in_cta
                col_in_atom = block_id % CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
                row = (
                    pid_y * self.m_tile + block_id // CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
                )
                scale_col = pid_x * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM + col_in_atom
                if row < rows and scale_col < scale_cols:
                    self._quantize_row_values(
                        mX,
                        mProducerB,
                        mProducerBWords,
                        mXWords,
                        mQWords,
                        mScale,
                        row,
                        scale_col,
                        lane,
                        scale_cols,
                        pid_x,
                        col_in_atom,
                        n_col_blocks,
                        source_cols,
                        local_rank,
                        world_size,
                    )
        else:
            total_blocks = rows * scale_cols
            idx = pid_x * groups_per_cta + group_in_cta
            row = idx // scale_cols
            scale_col = idx - row * scale_cols
            if idx < total_blocks:
                self._quantize_row_values(
                    mX,
                    mProducerB,
                    mProducerBWords,
                    mXWords,
                    mQWords,
                    mScale,
                    row,
                    scale_col,
                    lane,
                    scale_cols,
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    source_cols,
                    local_rank,
                    world_size,
                )

    @cute.jit
    def _quantize_row_values(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQWords: cute.Tensor,
        mScale: cute.Tensor,
        row: Int32,
        scale_col: Int32,
        lane: Int32,
        scale_cols: Int32,
        col_block: Int32,
        col_in_atom: Int32,
        n_col_blocks: Int32,
        source_cols: Int32,
        local_rank: Int32,
        world_size: Int32,
    ) -> None:
        row = self._get_rank_balanced_row(row, mX.shape[0], local_rank, world_size)
        k_base = scale_col * self.sf_vec_size + lane * self.lane_elems
        values, max_abs = self._load_row_values_and_compute_amax(
            mX,
            mProducerB,
            mProducerBWords,
            mXWords,
            row,
            k_base,
            source_cols,
        )

        if cutlass.const_expr(self.lane_group != 1):
            max_abs = _reduce_amax_f32(
                max_abs,
                1,
                self.lane_group.bit_length() - 1,
            )

        scale_byte, recip_scale = _compute_scale_and_recip_from_amax(
            max_abs,
            self.format_id,
            self.half_range_scale,
            use_nvfp4_no_clip_scale=False,
        )
        if lane == Int32(0):
            scale_offset = self._get_scale_offset(
                row=row,
                scale_col=scale_col,
                scale_cols=scale_cols,
                col_block=col_block,
                col_in_atom=col_in_atom,
                n_col_blocks=n_col_blocks,
            )
            if cutlass.const_expr(self.layout_id == SCALE_FACTOR_LAYOUT_NATURAL):
                mScale[row, scale_col] = scale_byte
            else:
                mScale[scale_offset] = scale_byte
        self._store_row_qdata_words(
            mQWords=mQWords,
            values=values,
            recip_scale=recip_scale,
            row=row,
            scale_col=scale_col,
            lane=lane,
        )

    @cute.jit
    def _get_rank_balanced_row(
        self,
        row: Int32,
        rows: Int32,
        local_rank: Int32,
        world_size: Int32,
    ) -> Int32:
        if cutlass.const_expr(
            self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
        ):
            if world_size > Int32(1):
                row += (rows // world_size) * local_rank
                if row >= rows:
                    row -= rows
        return row

    @cute.jit
    def _get_scale_offset(
        self,
        row: Int32,
        scale_col: Int32,
        scale_cols: Int32,
        col_block: Int32,
        col_in_atom: Int32,
        n_col_blocks: Int32,
    ) -> Int32:
        if cutlass.const_expr(self.layout_id == SCALE_FACTOR_LAYOUT_NATURAL):
            return row * scale_cols + scale_col
        return _cublas_blockscaled_qscale_offset(
            row,
            col_block * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM + col_in_atom,
            n_col_blocks,
        )

    @cute.jit
    def _load_row_values_and_compute_amax(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mXWords: cute.Tensor,
        row: Int32,
        k_base: Int32,
        source_cols: Int32,
    ):
        values = cute.make_rmem_tensor(self.lane_elems, Float32)
        for i in cutlass.range_constexpr(self.lane_elems):
            values[i] = Float32(_FP32_ZERO)

        if cutlass.const_expr(
            self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
        ):
            row_addr = mX[row]
            if row_addr != Int64(0):
                row_ptr = cute.make_ptr(
                    self.input_dtype,
                    row_addr,
                    mem_space=cute.AddressSpace.gmem,
                    assumed_align=128,
                )
                row_tensor = cute.make_tensor(
                    row_ptr,
                    cute.make_ordered_layout((source_cols,), order=(0,)),
                )
                chunk_elems: cutlass.Constexpr[int] = (
                    8
                    if self.input_dtype == BFloat16 or self.input_dtype == Float16
                    else 4
                )
                row_chunks = cute.tiled_divide(row_tensor, (chunk_elems,))
                copy_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.input_dtype,
                    num_bits_per_copy=128,
                )
                for chunk in cutlass.range_constexpr(self.lane_elems // chunk_elems):
                    loaded_values = cute.make_rmem_tensor(chunk_elems, self.input_dtype)
                    cute.copy(
                        copy_atom,
                        row_chunks[None, k_base // Int32(chunk_elems) + chunk],
                        loaded_values,
                    )
                    for i in cutlass.range_constexpr(chunk_elems):
                        value = Float32(loaded_values[i])
                        values[chunk * chunk_elems + i] = value
        elif cutlass.const_expr(
            (self.input_dtype == BFloat16 or self.input_dtype == Float16)
            and self.x_words_aligned
        ):
            loaded_values = _load_f16_words_as_f32(
                mWords=mXWords,
                row=row,
                col_start=k_base,
                source_dtype=self.input_dtype,
                n_values=self.lane_elems,
            )
            producer_values = cute.make_rmem_tensor(self.lane_elems, Float32)
            if cutlass.const_expr(
                self.producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD
                and self.producer_b_words_aligned
            ):
                producer_values = _load_f16_words_as_f32(
                    mWords=mProducerBWords,
                    row=row,
                    col_start=k_base,
                    source_dtype=self.input_dtype,
                    n_values=self.lane_elems,
                )
            for i in cutlass.range_constexpr(self.lane_elems):
                value = loaded_values[i]
                if cutlass.const_expr(
                    self.producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD
                ):
                    if cutlass.const_expr(self.producer_b_words_aligned):
                        producer_b = producer_values[i]
                    else:
                        producer_b = Float32(mProducerB[row, k_base + i])
                    value = _apply_block_scaled_quant_producer(
                        value,
                        producer_b,
                        self.producer_id,
                        self.fast_math,
                        self.clamped,
                        self.alpha,
                        self.limit,
                    )
                values[i] = value
        else:
            for i in cutlass.range_constexpr(self.lane_elems):
                col = k_base + i
                value = Float32(mX[row, col])
                if cutlass.const_expr(
                    self.producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD
                ):
                    producer_b = Float32(mProducerB[row, col])
                    value = _apply_block_scaled_quant_producer(
                        value,
                        producer_b,
                        self.producer_id,
                        self.fast_math,
                        self.clamped,
                        self.alpha,
                        self.limit,
                    )
                values[i] = value
        max_abs = _frag_amax_nonfinite(values, 0, self.lane_elems)
        return values, max_abs

    @cute.jit
    def _store_row_qdata_words(
        self,
        mQWords: cute.Tensor,
        values,
        recip_scale: Float32,
        row: Int32,
        scale_col: Int32,
        lane: Int32,
    ) -> None:
        target_max = Float32(FP4_E2M1_MAX)
        qdata_elems_per_word: cutlass.Constexpr[int] = 8
        if cutlass.const_expr(not self.is_fp4):
            target_max = _target_max_for_format_id(self.format_id)
            qdata_elems_per_word = 4

        q_values = cute.make_rmem_tensor(self.lane_elems, Float32)
        for i in cutlass.range_constexpr(self.lane_elems):
            q_values[i] = _clamp_to_target_max(values[i], recip_scale, target_max)

        q_word_col = scale_col * (self.sf_vec_size // qdata_elems_per_word) + lane * (
            self.lane_elems // qdata_elems_per_word
        )
        _store_blockscaled_xn_words(
            mQWords,
            row,
            q_word_col,
            q_values,
            recip=Float32(_FP32_ONE),
            num_elems=self.lane_elems,
            is_fp4=self.is_fp4,
            format_id=self.format_id,
            scale_values=False,
        )


# -----------------------------------------------------------------------------
# Dynamic per-token NVFP4 quantization
# -----------------------------------------------------------------------------


class _QuantizeNvfp4PerTokenRow:
    """Quantize one activation row per 128-thread CTA.

    Each thread retains disjoint 16-value blocks in registers. For the SwiGLU
    producer, FP32 results are rounded to the input 16-bit dtype before they
    are retained, so the quantized bytes match quantizing a materialized
    16-bit SwiGLU output.
    Threads reduce their block amaxes locally, then ``redux.sync`` reduces
    within each of the four warps. Warp leaders write four maxima to shared
    memory; warp 0 loads and reduces them to the row amax. Warp 0 lane 0
    computes the token and row scales, writes both beside the warp maxima,
    and the second CTA barrier broadcasts the row scale before threads
    quantize their retained blocks.
    """

    @staticmethod
    def _validate_k(K: int) -> None:
        if K <= 0 or K % 256 != 0:
            raise ValueError(f"K must be a positive multiple of 256, got {K}")
        if K > 16384:
            raise ValueError(f"K must be at most 16384, got {K}")

    @staticmethod
    def _validate_producer_k(K: int, producer_id: int) -> None:
        _QuantizeNvfp4PerTokenRow._validate_k(K)
        if producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD and K > 8192:
            raise ValueError(f"SwiGLU producer K must be at most 8192, got {K}")

    def __init__(
        self,
        input_dtype: type[cutlass.Numeric],
        K: int,
        layout_id: int,
        producer_id: int,
        fast_math: bool,
        clamped: bool,
        alpha: float,
        limit: float,
    ) -> None:
        self._validate_producer_k(K, producer_id)
        self.input_dtype = input_dtype
        self.K: cutlass.Constexpr[int] = K
        self.layout_id: cutlass.Constexpr[int] = layout_id
        self.producer_id: cutlass.Constexpr[int] = producer_id
        self.fast_math: cutlass.Constexpr[bool] = fast_math
        self.clamped: cutlass.Constexpr[bool] = clamped
        self.alpha: cutlass.Constexpr[float] = alpha
        self.limit: cutlass.Constexpr[float] = limit
        self.num_threads: cutlass.Constexpr[int] = 128
        self.warp_size: cutlass.Constexpr[int] = 32
        assert self.num_threads % self.warp_size == 0
        self.num_warps: cutlass.Constexpr[int] = self.num_threads // self.warp_size
        self.token_scale_inv_slot: cutlass.Constexpr[int] = self.num_warps
        self.row_scale_slot: cutlass.Constexpr[int] = self.num_warps + 1
        # Round the warp maxima and two broadcast values to a 16-byte allocation.
        self.reduction_slots: cutlass.Constexpr[int] = (self.num_warps + 5) // 4 * 4
        self.words_per_block: cutlass.Constexpr[int] = 8
        self.words_per_copy: cutlass.Constexpr[int] = 4
        assert self.words_per_block % self.words_per_copy == 0
        self.copies_per_block: cutlass.Constexpr[int] = (
            self.words_per_block // self.words_per_copy
        )
        self.values_per_block: cutlass.Constexpr[int] = 16
        self.scale_cols: cutlass.Constexpr[int] = K // self.values_per_block
        # Ceiling division is intentional; inactive tail lanes load zeros.
        self.blocks_per_lane: cutlass.Constexpr[int] = (
            self.scale_cols + self.num_threads - 1
        ) // self.num_threads

    def can_implement(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mXWords: cute.Tensor,
        mQWords: cute.Tensor,
        mScale: cute.Tensor,
        mTokenScaleInv: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        stream: cuda.CUstream,
    ) -> bool:
        return (
            0 < self.K
            and self.K
            <= (8192 if self.producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD else 16384)
            and self.K % 256 == 0
            and mX.element_type == self.input_dtype
            and mProducerB.element_type == self.input_dtype
            and mXWords.element_type == Uint32
            and mQWords.element_type == Uint32
            and mScale.element_type == Uint8
            and mTokenScaleInv.element_type == Float32
            and mNvfp4RecipLut.element_type == Float32
            and stream is not None
        )

    @staticmethod
    @jit_cache
    def compile(
        input_dtype: type[cutlass.Numeric],
        K: int,
        layout_id: int,
        producer_id: int,
        fast_math: bool,
        clamped: bool,
        alpha: float,
        limit: float,
    ) -> object:
        _QuantizeNvfp4PerTokenRow._validate_producer_k(K, producer_id)
        rows = cute.sym_int()
        scale_elems = cute.sym_int()
        x = cute.runtime.make_fake_tensor(
            input_dtype,
            (rows, K),
            stride=(cute.sym_int64(), 1),
            assumed_align=64,
        )
        producer_b = cute.runtime.make_fake_tensor(
            input_dtype,
            (rows, K),
            stride=(cute.sym_int64(), 1),
            assumed_align=64,
        )
        x_words = (
            fake_tensor(Uint32, (rows, K // 8), 4)
            if producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD
            else fake_tensor(Uint32, (rows, K // 2), 16)
        )
        q_words = fake_tensor(Uint32, (rows, K // 8), 4)
        scale = fake_tensor(Uint8, (scale_elems,), 1)
        # Fully-dynamic stride: callers may point this at a strided column of a
        # packed dispatch buffer so the kernel writes the per-token scale in
        # place instead of paying a separate strided-copy launch.
        token_scale_inv = fake_tensor(Float32, (rows,), 1, leading_dim=None)
        nvfp4_recip_lut = fake_tensor(Float32, (256,), 4)
        dtype_tag = _INPUT_DTYPE_TAG[input_dtype]
        producer_tag = _PRODUCER_TAG.get(producer_id, f"prod{producer_id}")
        fastmath_tag = "_fastmath" if fast_math else ""
        clamp_tag = _build_swiglu_clamp_tag(clamped, alpha, limit)
        name_prefix = (
            f"_cute_quantize_{producer_tag}_{dtype_tag}_nvfp4_per_token_"
            f"{_LAYOUT_TAG[layout_id]}_k{K}{fastmath_tag}{clamp_tag}"
        )
        with scoped_kernel_name_prefixes(
            ((_QuantizeNvfp4PerTokenRow.kernel, name_prefix),)
        ):
            return cute.compile(
                _QuantizeNvfp4PerTokenRow(
                    input_dtype,
                    K,
                    layout_id,
                    producer_id,
                    fast_math,
                    clamped,
                    alpha,
                    limit,
                ),
                x,
                producer_b,
                x_words,
                q_words,
                scale,
                cutlass.Int64(1),
                token_scale_inv,
                nvfp4_recip_lut,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mXWords: cute.Tensor,
        mQWords: cute.Tensor,
        mScale: cute.Tensor,
        scale_row_stride: Int64,
        mTokenScaleInv: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        assert self.can_implement(
            mX,
            mProducerB,
            mXWords,
            mQWords,
            mScale,
            mTokenScaleInv,
            mNvfp4RecipLut,
            stream,
        )
        self.kernel(
            mX,
            mProducerB,
            mXWords,
            mQWords,
            mScale,
            scale_row_stride,
            mTokenScaleInv,
            mNvfp4RecipLut,
        ).launch(
            grid=[mXWords.shape[0], 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self.reduction_slots * 4,
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901 -- fused load/reduce/quantize control flow is trace-time
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mXWords: cute.Tensor,
        mQWords: cute.Tensor,
        mScale: cute.Tensor,
        scale_row_stride: Int64,
        mTokenScaleInv: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tidx % Int32(self.warp_size)
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        packed = cute.make_rmem_tensor(
            (self.blocks_per_lane, self.words_per_block),
            Int32,
        )
        block_amaxes = cute.make_rmem_tensor(self.blocks_per_lane, Float32)
        if cutlass.const_expr(self.producer_id != BLOCK_SCALED_PRODUCER_SWIGLU_FWD):
            row_chunks = cute.tiled_divide(
                mXWords[row, None],
                (self.words_per_copy,),
            )
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Uint32,
                num_bits_per_copy=128,
            )
        for block_rep in cutlass.range_constexpr(self.blocks_per_lane):
            scale_col = block_rep * self.num_threads + tidx
            active = scale_col < self.scale_cols
            if cutlass.const_expr(self.producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD):
                _load_swiglu_fwd_nvfp4_block(
                    mX,
                    mProducerB,
                    row,
                    scale_col * self.values_per_block,
                    active,
                    packed,
                    block_rep,
                    self.input_dtype,
                    self.fast_math,
                    self.clamped,
                    self.alpha,
                    self.limit,
                )
            else:
                for copy_idx in cutlass.range_constexpr(self.copies_per_block):
                    loaded = cute.make_rmem_tensor(self.words_per_copy, Uint32)
                    loaded.fill(Uint32(0))
                    if active:
                        cute.copy(
                            copy_atom,
                            row_chunks[
                                None,
                                scale_col * self.copies_per_block + copy_idx,
                            ],
                            loaded,
                        )
                    for word in cutlass.range_constexpr(self.words_per_copy):
                        packed[
                            block_rep,
                            copy_idx * self.words_per_copy + word,
                        ] = Int32(loaded[word])
            block_amax = _compute_row_amax_b16x2_x16(
                packed,
                block_rep,
                self.input_dtype,
            )
            block_amaxes[block_rep] = block_amax

        row_amax_local = _reduce_nvfp4_block_amaxes(
            block_amaxes,
            self.blocks_per_lane,
        )

        warp_amax = _warp_reduce_amax_f32(row_amax_local)
        smem = cutlass.utils.SmemAllocator()
        sReduction = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.reduction_slots,)),
            byte_alignment=16,
        )
        if lane == Int32(0):
            sReduction[warp] = warp_amax
        cute.arch.sync_threads()

        if warp == Int32(0):
            warpgroup_amax = (
                sReduction[lane]
                if lane < Int32(self.num_warps)
                else Float32(_FP32_ZERO)
            )
            warpgroup_amax = _warp_reduce_amax_f32(warpgroup_amax)
            if lane == Int32(0):
                row_scale, token_scale_inv = _compute_nvfp4_token_scales(warpgroup_amax)
                sReduction[self.token_scale_inv_slot] = token_scale_inv
                sReduction[self.row_scale_slot] = row_scale
                mTokenScaleInv[row] = token_scale_inv
        cute.arch.sync_threads()
        row_scale = sReduction[self.row_scale_slot]

        values = cute.make_rmem_tensor(self.values_per_block, Float32)
        n_col_blocks: cutlass.Constexpr[int] = self.scale_cols // 4
        for block_rep in cutlass.range_constexpr(self.blocks_per_lane):
            scale_col = block_rep * self.num_threads + tidx
            if scale_col < self.scale_cols:
                scale_byte = _scale_nvfp4_per_token_block(
                    packed,
                    block_rep,
                    block_amaxes[block_rep],
                    row_scale,
                    mNvfp4RecipLut,
                    self.input_dtype,
                    values,
                    use_nvfp4_no_clip_scale=False,
                )
                _store_blockscaled_xn_words(
                    mQWords,
                    row,
                    scale_col * 2,
                    values,
                    recip=Float32(_FP32_ONE),
                    num_elems=self.values_per_block,
                    is_fp4=True,
                    format_id=BLOCK_SCALED_FORMAT_NVFP4,
                    scale_values=False,
                )
                if cutlass.const_expr(
                    self.layout_id == SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED
                ):
                    scale_offset = _cublas_blockscaled_qscale_offset(
                        Int64(row),
                        Int64(scale_col),
                        Int64(n_col_blocks),
                    )
                else:
                    scale_offset = Int64(row) * scale_row_stride + Int64(scale_col)
                mScale[scale_offset] = scale_byte


# -----------------------------------------------------------------------------
# Row/col quantization: column / row / 2D-tile scale reduction.
# Mirrors the Triton ``_triton_quantize_blockscaled_2d_tile`` 1:1 — one CTA covers one
# [SF_VEC, SF_VEC] tile, with one thread per row and warp reductions for
# cross-row (column / 2D) reductions.
# -----------------------------------------------------------------------------


class _QuantizeBlockScaledRowCol:
    def __init__(
        self,
        input_dtype: type[cutlass.Numeric],
        format_id: int,
        producer_id: int,
        axis_mask: int,
        scale_reduction_id: int,
        layout_id: int,
        row_col_kernel_mode: int,
        grouped_rows_per_group: int,
        half_range_scale: bool,
        fast_math: bool,
        clamped: bool,
        alpha: float,
        limit: float,
        producer_b_words_aligned: bool,
        producer_c_words_aligned: bool,
    ) -> None:
        self.input_dtype = input_dtype
        self.format_id = format_id
        self.producer_id = producer_id
        self.axis_mask = axis_mask
        self.scale_reduction_id = scale_reduction_id
        self.layout_id = layout_id
        self.row_col_kernel_mode = row_col_kernel_mode
        self.grouped_rows_per_group = grouped_rows_per_group
        self.half_range_scale = half_range_scale
        self.fast_math = fast_math
        self.clamped = clamped
        self.alpha = alpha
        self.limit = limit
        self.producer_b_words_aligned = producer_b_words_aligned
        self.producer_c_words_aligned = producer_c_words_aligned
        self.sf_vec_size = _get_sf_vec_size_for_format_id(format_id)
        self.qdata_elems_per_word = 4
        self.lane_elems = (
            4
            if row_col_kernel_mode
            in (
                _ROW_COL_KERNEL_MODE_SWIGLU_FWD_FP8_TILE,
                _ROW_COL_KERNEL_MODE_SWIGLU_BWD_DXY_FP8_TILE,
            )
            else 16
        )
        self.lane_group = self.sf_vec_size // self.lane_elems
        self.col_blocks_per_scale = 8
        self.col_lanes_cfg = 8
        self.scale_cols_per_tile = self.col_lanes_cfg // self.col_blocks_per_scale
        self.col1d_m_blocks = 2
        self.gather_scale_cols_per_tile = 4
        self.is_fp4 = format_id in _FP4_FORMAT_IDS
        self.col1d_use_row_vector = _use_row_vector_col1d(
            format_id=format_id,
            axis_mask=axis_mask,
            scale_reduction_id=scale_reduction_id,
        )
        self.col_qdata_row_major = _use_row_major_col_qdata(
            format_id=format_id,
            scale_reduction_id=scale_reduction_id,
            has_col_axis=(axis_mask & AXIS_MASK_M) != 0,
            has_row_axis=(axis_mask & AXIS_MASK_K) != 0,
            col1d_use_row_vector=self.col1d_use_row_vector,
        )
        self.both1d_split_tile_groups = (
            2 if format_id == BLOCK_SCALED_FORMAT_NVFP4 else 1
        )
        self.num_threads, self.warps_per_cta = _make_row_col_thread_config(
            row_col_kernel_mode,
            axis_mask,
            producer_id,
            self.is_fp4,
            self.col1d_use_row_vector,
        )

    @staticmethod
    def can_implement(
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQ0: cute.Tensor,
        mQ0Words: cute.Tensor,
        mQ1: cute.Tensor,
        mQ1Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        stream: cuda.CUstream,
    ) -> bool:
        return (
            mX.element_type in (BFloat16, Float16, Float32, Int64)
            and mProducerB.element_type in (BFloat16, Float16, Float32)
            and mProducerC.element_type in (BFloat16, Float16, Float32)
            and mProducerBWords.element_type == Uint32
            and mProducerCWords.element_type == Uint32
            and mXWords.element_type == Uint32
            and mQ0.element_type == Uint8
            and mQ0Words.element_type == Uint32
            and mQ1.element_type == Uint8
            and mQ1Words.element_type == Uint32
            and mS0.element_type == Uint8
            and mS1.element_type == Uint8
            and mNvfp4RecipLut.element_type == Float32
            and stream is not None
        )

    @staticmethod
    @jit_cache
    def compile(
        input_dtype: type[cutlass.Numeric],
        format_id: int,
        producer_id: int,
        axis_mask: int,
        scale_reduction_id: int,
        layout_id: int,
        row_col_kernel_mode: int,
        grouped_rows_per_group: int,
        half_range_scale: bool,
        fast_math: bool,
        clamped: bool,
        alpha: float,
        limit: float,
        producer_b_words_aligned: bool,
        producer_c_words_aligned: bool,
    ) -> object:
        swiglu_fwd_fp8_tile = (
            row_col_kernel_mode == _ROW_COL_KERNEL_MODE_SWIGLU_FWD_FP8_TILE
        )
        swiglu_bwd_dxy_fp8_tile = (
            row_col_kernel_mode == _ROW_COL_KERNEL_MODE_SWIGLU_BWD_DXY_FP8_TILE
        )
        swiglu_fp8_tile = swiglu_fwd_fp8_tile or swiglu_bwd_dxy_fp8_tile
        rows = cute.sym_int()
        source_cols = cute.sym_int()
        cols = cute.sym_int()
        x_word_cols = cute.sym_int()
        q0_word_cols = cute.sym_int()
        q1_word_cols = cute.sym_int()
        q_row_cols = cute.sym_int()
        q_col_cols = cute.sym_int()
        s0_elems = cute.sym_int()
        s1_elems = cute.sym_int()

        def _source_fake(shape):
            if swiglu_fp8_tile:
                return cute.runtime.make_fake_tensor(
                    input_dtype,
                    shape,
                    stride=(cute.sym_int64(), cute.sym_int64()),
                    assumed_align=16,
                )
            return _make_input_fake_tensor(input_dtype, shape)

        x_cute = (
            fake_tensor(Int64, (rows,), 1)
            if producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
            else _source_fake((rows, source_cols))
        )
        if producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER:
            producer_shape = (cute.sym_int(), cute.sym_int())
            producer_b_cute = _make_input_fake_tensor(input_dtype, producer_shape)
            producer_c_cute = _make_input_fake_tensor(input_dtype, producer_shape)
        elif swiglu_bwd_dxy_fp8_tile:
            producer_b_cute = _source_fake((rows, cols))
            producer_c_cute = _source_fake((rows, cols))
        else:
            producer_b_cute = _source_fake((rows, source_cols))
            producer_c_cute = _source_fake((rows, source_cols))
        producer_b_words_div = 4 if producer_b_words_aligned else 1
        producer_c_words_div = 4 if producer_c_words_aligned else 1
        word_rows = (
            cute.sym_int()
            if producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
            else rows
        )
        producer_b_words_cute = fake_tensor(
            Uint32, (word_rows, x_word_cols), producer_b_words_div
        )
        producer_c_words_cute = fake_tensor(
            Uint32, (word_rows, x_word_cols), producer_c_words_div
        )
        x_words_cute = fake_tensor(Uint32, (word_rows, x_word_cols), 4)
        has_col_axis = (axis_mask & AXIS_MASK_M) != 0
        has_row_axis = (axis_mask & AXIS_MASK_K) != 0
        q0_rows = rows if swiglu_fp8_tile else cols
        q0_cute = fake_tensor(Uint8, (q0_rows, q_col_cols), 1)
        q0_words_cute = (
            fake_tensor(Uint32, (q0_rows, q0_word_cols), 4)
            if has_col_axis and row_col_kernel_mode != _AXES_KERNEL_MODE_SCALAR
            else fake_tensor(Uint32, (cute.sym_int(), cute.sym_int()), 4)
        )
        q1_cute = fake_tensor(Uint8, (rows, q_row_cols), 1)
        q1_words_cute = (
            fake_tensor(Uint32, (rows, q1_word_cols), 4)
            if row_col_kernel_mode != _AXES_KERNEL_MODE_SCALAR
            and (
                has_row_axis
                or row_col_kernel_mode
                in (
                    _AXES_KERNEL_MODE_TILE2D_FP8_VECTOR,
                    _AXES_KERNEL_MODE_TILE2D_FP4_VECTOR,
                )
            )
            else fake_tensor(Uint32, (cute.sym_int(), cute.sym_int()), 4)
        )
        s0_cute = fake_tensor(Uint8, (s0_elems,), 1)
        s1_cute = fake_tensor(Uint8, (s1_elems,), 1)
        nvfp4_recip_lut_cute = fake_tensor(Float32, (256,), 4)
        layout_tag = _LAYOUT_TAG.get(layout_id, f"layout{layout_id}")
        fmt_tag = _FORMAT_TAG.get(format_id, f"fmt{format_id}")
        dtype_tag = _INPUT_DTYPE_TAG[input_dtype]
        producer_tag = _PRODUCER_TAG.get(producer_id, f"prod{producer_id}")
        if swiglu_fwd_fp8_tile:
            producer_prefix = "swiglu_fwd_shared_"
        elif swiglu_bwd_dxy_fp8_tile:
            producer_prefix = "swiglu_bwd_shared_"
        else:
            producer_prefix = f"{producer_tag}_"
        sf_vec = _get_sf_vec_size_for_format_id(format_id)
        block_tile = _build_block_tile_tag(axis_mask, scale_reduction_id, sf_vec)
        fastmath_tag = "_fastmath" if fast_math else ""
        clamp_tag = _build_swiglu_clamp_tag(clamped, alpha, limit)
        name_prefix = f"_cute_quantize_{producer_prefix}{dtype_tag}_{fmt_tag}_{layout_tag}_{block_tile}{fastmath_tag}{clamp_tag}"
        with scoped_kernel_name_prefixes(
            ((_QuantizeBlockScaledRowCol.kernel, name_prefix),)
        ):
            return cute.compile(
                _QuantizeBlockScaledRowCol(
                    input_dtype,
                    format_id,
                    producer_id,
                    axis_mask,
                    scale_reduction_id,
                    layout_id,
                    row_col_kernel_mode,
                    grouped_rows_per_group,
                    half_range_scale,
                    fast_math,
                    clamped,
                    alpha,
                    limit,
                    producer_b_words_aligned,
                    producer_c_words_aligned,
                ),
                x_cute,
                producer_b_cute,
                producer_c_cute,
                producer_b_words_cute,
                producer_c_words_cute,
                x_words_cute,
                q0_cute,
                q0_words_cute,
                q1_cute,
                q1_words_cute,
                s0_cute,
                s1_cute,
                nvfp4_recip_lut_cute,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(1),
                make_fake_stream(),
                options="--enable-tvm-ffi",
            )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQ0: cute.Tensor,
        mQ0Words: cute.Tensor,
        mQ1: cute.Tensor,
        mQ1Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        source_cols: Int32,
        local_rank: Int32,
        world_size: Int32,
        stream: cuda.CUstream,
    ) -> None:
        assert self.can_implement(
            mX,
            mProducerB,
            mProducerC,
            mProducerBWords,
            mProducerCWords,
            mXWords,
            mQ0,
            mQ0Words,
            mQ1,
            mQ1Words,
            mS0,
            mS1,
            mNvfp4RecipLut,
            stream,
        )
        SF_VEC: cutlass.Constexpr = self.sf_vec_size
        logical_cols = self._get_logical_cols(mQ0, mQ1)
        if cutlass.const_expr(
            self.row_col_kernel_mode == _ROW_COL_KERNEL_MODE_SWIGLU_FWD_FP8_TILE
        ):
            scale_cols = source_cols // SF_VEC
            col_work_tiles = (
                scale_cols + self.scale_cols_per_tile - 1
            ) // self.scale_cols_per_tile
            total_tiles = (mX.shape[0] // SF_VEC) * col_work_tiles
            grid = [
                (total_tiles + self.warps_per_cta - 1) // self.warps_per_cta,
                1,
                1,
            ]
        elif cutlass.const_expr(
            self.row_col_kernel_mode == _ROW_COL_KERNEL_MODE_SWIGLU_BWD_DXY_FP8_TILE
        ):
            source_scale_cols = source_cols // SF_VEC
            total_tiles = (mX.shape[0] // SF_VEC) * source_scale_cols
            grid = [
                (total_tiles + self.warps_per_cta - 1) // self.warps_per_cta,
                1,
                1,
            ]
        elif cutlass.const_expr(
            self.row_col_kernel_mode == _AXES_KERNEL_MODE_TILE2D_FP8_VECTOR
        ):
            scale_cols = logical_cols // SF_VEC
            total_tiles = (mX.shape[0] // SF_VEC) * scale_cols
            tiles_per_cta: cutlass.Constexpr[int] = self.warps_per_cta * (32 // SF_VEC)
            grid = [total_tiles // tiles_per_cta, 1, 1]
        elif cutlass.const_expr(
            self.row_col_kernel_mode == _AXES_KERNEL_MODE_TILE2D_FP4_VECTOR
        ):
            scale_cols = logical_cols // SF_VEC
            total_tiles = (mX.shape[0] // SF_VEC) * scale_cols
            tiles_per_cta: cutlass.Constexpr[int] = self.warps_per_cta * (32 // SF_VEC)
            grid = [total_tiles // tiles_per_cta, 1, 1]
        elif cutlass.const_expr(
            self.row_col_kernel_mode == _AXES_KERNEL_MODE_BOTH1D_VECTOR
        ):
            scale_cols = logical_cols // SF_VEC
            total_tiles = (mX.shape[0] // SF_VEC) * scale_cols
            if cutlass.const_expr(
                self.axis_mask == AXIS_MASK_M
                and self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
                and not self.is_fp4
            ):
                tiles_per_cta: cutlass.Constexpr[int] = (
                    self.col1d_m_blocks * self.gather_scale_cols_per_tile
                )
            elif cutlass.const_expr(self.axis_mask == AXIS_MASK_M):
                if cutlass.const_expr(self.col1d_use_row_vector):
                    tiles_per_cta: cutlass.Constexpr[int] = self.warps_per_cta * (
                        32 // SF_VEC
                    )
                else:
                    tiles_per_cta: cutlass.Constexpr[int] = (
                        self.warps_per_cta * (32 // SF_VEC) * self.col1d_m_blocks
                    )
            elif cutlass.const_expr(
                self.axis_mask == (AXIS_MASK_M | AXIS_MASK_K)
                and self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
                and not self.is_fp4
            ):
                tiles_per_cta: cutlass.Constexpr[int] = self.gather_scale_cols_per_tile
            elif cutlass.const_expr(not self.is_fp4):
                tiles_per_cta: cutlass.Constexpr[int] = self.warps_per_cta * (
                    32 // SF_VEC
                )
            else:
                tiles_per_cta: cutlass.Constexpr[int] = (
                    self.both1d_split_tile_groups * (32 // SF_VEC)
                )
            grid = [total_tiles // tiles_per_cta, 1, 1]
        elif cutlass.const_expr(
            self.row_col_kernel_mode == _AXES_KERNEL_MODE_SWIGLU_BWD_DXY_F32X2_VECTOR
        ):
            source_scale_cols = source_cols // SF_VEC
            total_tiles = (mX.shape[0] // SF_VEC) * source_scale_cols
            tiles_per_cta: cutlass.Constexpr[int] = self.warps_per_cta * (32 // SF_VEC)
            grid = [total_tiles // tiles_per_cta, 1, 1]
        else:
            grid = [
                mX.shape[0] // SF_VEC,
                logical_cols // SF_VEC,
                1,
            ]
        self.kernel(
            mX,
            mProducerB,
            mProducerC,
            mProducerBWords,
            mProducerCWords,
            mXWords,
            mQ0,
            mQ0Words,
            mQ1,
            mQ1Words,
            mS0,
            mS1,
            mNvfp4RecipLut,
            source_cols,
            local_rank,
            world_size,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901 -- format/axis branches are constexpr-folded by the JIT
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQ0: cute.Tensor,
        mQ0Words: cute.Tensor,
        mQ1: cute.Tensor,
        mQ1Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        source_cols: Int32,
        local_rank: Int32,
        world_size: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pid_m, pid_k, _ = cute.arch.block_idx()
        SF_VEC: cutlass.Constexpr = self.sf_vec_size
        logical_cols = self._get_logical_cols(mQ0, mQ1)
        if cutlass.const_expr(
            self.row_col_kernel_mode == _ROW_COL_KERNEL_MODE_SWIGLU_FWD_FP8_TILE
        ):
            self._run_swiglu_fwd_fp8_tile(
                mX,
                mProducerB,
                mQ0Words,
                mQ1Words,
                mS0,
                mS1,
                source_cols,
                tidx,
                pid_m,
            )
            return

        if cutlass.const_expr(
            self.row_col_kernel_mode == _ROW_COL_KERNEL_MODE_SWIGLU_BWD_DXY_FP8_TILE
        ):
            self._run_swiglu_bwd_dxy_fp8_tile(
                mX,
                mProducerB,
                mQ0Words,
                mQ1Words,
                mS0,
                mS1,
                source_cols,
                tidx,
                pid_m,
            )
            return

        if cutlass.const_expr(
            self.row_col_kernel_mode == _AXES_KERNEL_MODE_TILE2D_FP8_VECTOR
        ):
            self._run_tile2d_fp8_vectorized(
                mX,
                mProducerB,
                mProducerC,
                mProducerBWords,
                mProducerCWords,
                mXWords,
                mQ1Words,
                mS0,
                mS1,
                mNvfp4RecipLut,
                logical_cols,
                source_cols,
                tidx,
                pid_m,
                local_rank,
                world_size,
            )
            return

        if cutlass.const_expr(
            self.row_col_kernel_mode == _AXES_KERNEL_MODE_TILE2D_FP4_VECTOR
        ):
            self._run_tile2d_fp4_vectorized(
                mX,
                mProducerB,
                mProducerC,
                mProducerBWords,
                mProducerCWords,
                mXWords,
                mQ0,
                mQ1Words,
                mS0,
                mS1,
                mNvfp4RecipLut,
                logical_cols,
                source_cols,
                tidx,
                pid_m,
                local_rank,
                world_size,
            )
            return

        if cutlass.const_expr(
            self.row_col_kernel_mode == _AXES_KERNEL_MODE_SWIGLU_BWD_DXY_F32X2_VECTOR
        ):
            self._run_swiglu_bwd_dxy_f32x2_vectorized(
                mX,
                mProducerB,
                mProducerC,
                mProducerBWords,
                mProducerCWords,
                mXWords,
                mQ1Words,
                mQ0,
                mQ0Words,
                mS0,
                mS1,
                mNvfp4RecipLut,
                logical_cols,
                tidx,
                pid_m,
            )
            return

        if cutlass.const_expr(
            self.row_col_kernel_mode == _AXES_KERNEL_MODE_BOTH1D_VECTOR
        ):
            if cutlass.const_expr(self.axis_mask == AXIS_MASK_M):
                if cutlass.const_expr(
                    self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
                    and not self.is_fp4
                ):
                    self._run_gather_col_fp8_vectorized(
                        mX,
                        mQ0Words,
                        mS0,
                        mNvfp4RecipLut,
                        logical_cols,
                        source_cols,
                        tidx,
                        pid_m,
                        local_rank,
                        world_size,
                    )
                elif cutlass.const_expr(self.col1d_use_row_vector):
                    self._run_both1d_fp8_vectorized(
                        mX,
                        mProducerB,
                        mProducerC,
                        mProducerBWords,
                        mProducerCWords,
                        mXWords,
                        mQ1Words,
                        mQ0,
                        mQ0Words,
                        mS0,
                        mS1,
                        mNvfp4RecipLut,
                        logical_cols,
                        source_cols,
                        tidx,
                        pid_m,
                        local_rank,
                        world_size,
                    )
                else:
                    self._run_col1d_vectorized(
                        mX,
                        mProducerB,
                        mProducerC,
                        mXWords,
                        mQ0,
                        mQ0Words,
                        mS0,
                        mNvfp4RecipLut,
                        logical_cols,
                        source_cols,
                        tidx,
                        pid_m,
                        local_rank,
                        world_size,
                    )
            elif cutlass.const_expr(not self.is_fp4):
                if cutlass.const_expr(
                    self.axis_mask == (AXIS_MASK_M | AXIS_MASK_K)
                    and self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
                ):
                    self._run_gather_row_col_fp8_vectorized(
                        mX,
                        mQ1Words,
                        mQ0Words,
                        mS0,
                        mS1,
                        mNvfp4RecipLut,
                        logical_cols,
                        source_cols,
                        tidx,
                        pid_m,
                        local_rank,
                        world_size,
                    )
                else:
                    self._run_both1d_fp8_vectorized(
                        mX,
                        mProducerB,
                        mProducerC,
                        mProducerBWords,
                        mProducerCWords,
                        mXWords,
                        mQ1Words,
                        mQ0,
                        mQ0Words,
                        mS0,
                        mS1,
                        mNvfp4RecipLut,
                        logical_cols,
                        source_cols,
                        tidx,
                        pid_m,
                        local_rank,
                        world_size,
                    )
            else:
                self._run_both1d_fp4_vectorized(
                    mX,
                    mProducerB,
                    mProducerC,
                    mProducerBWords,
                    mProducerCWords,
                    mXWords,
                    mQ0,
                    mQ0Words,
                    mQ1Words,
                    mS0,
                    mS1,
                    mNvfp4RecipLut,
                    logical_cols,
                    source_cols,
                    tidx,
                    pid_m,
                    local_rank,
                    world_size,
                )
            return

        rows = mX.shape[0]
        cols = logical_cols
        pid_m = self._get_rank_balanced_row_block(
            row_block=pid_m,
            num_row_blocks=rows // SF_VEC,
            local_rank=local_rank,
            world_size=world_size,
        )
        row_start = pid_m * SF_VEC
        col_start = pid_k * SF_VEC
        row = row_start + tidx
        active = tidx < Int32(SF_VEC)

        safe_row = row if active else row_start
        source_col_start = self._get_source_col(col_start, source_cols)
        vals = cute.make_rmem_tensor(SF_VEC, Float32)
        if active:
            for j in cutlass.range_constexpr(SF_VEC):
                vals[j] = self._load_source_value(
                    mX,
                    safe_row,
                    source_col_start + j,
                    source_cols,
                )
        else:
            for j in cutlass.range_constexpr(SF_VEC):
                vals[j] = Float32(_FP32_ZERO)
        vals = self._apply_producer_to_row_values(
            mProducerB,
            mProducerC,
            mProducerBWords,
            mProducerCWords,
            safe_row,
            col_start,
            source_col_start,
            source_cols,
            vals,
            active,
            SF_VEC,
        )

        # Per-row amax (row-quant / 2D contribution).
        row_amax = _frag_amax_nonfinite(vals, 0, SF_VEC)

        if cutlass.const_expr(self.scale_reduction_id == SCALE_REDUCTION_TWO_D):
            tile_amax = cute.arch.warp_reduction(
                row_amax,
                cute.arch.fmax,
                threads_in_group=self.sf_vec_size,
            )
            tile_scale_byte, tile_recip = (
                _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                    tile_amax,
                    self.format_id,
                    self.half_range_scale,
                    mNvfp4RecipLut,
                    use_nvfp4_no_clip_scale=False,
                )
            )
            # Compute the per-row clamped scaled values once; reuse the
            # codes for both qdata layouts when FP4 needs the col-major
            # variant.
            q_codes = cute.make_rmem_tensor(SF_VEC, Float32)
            for k in cutlass.range_constexpr(SF_VEC):
                q_codes[k] = vals[k] * tile_recip
            if active:
                _store_qdata_row(
                    mQ=mQ1,
                    vals=vals,
                    recip=tile_recip,
                    row=row,
                    col_start=col_start,
                    sf_vec_size=self.sf_vec_size,
                    is_fp4=self.is_fp4,
                    format_id=self.format_id,
                )
                if cutlass.const_expr(self.is_fp4):
                    _store_qdata_col_major_fp4(
                        mQ=mQ0,
                        q_codes=q_codes,
                        tidx=tidx,
                        row_start=row_start,
                        col_start=col_start,
                        sf_vec_size=self.sf_vec_size,
                    )
                self._store_col_scale(
                    mScale=mS0,
                    m_v_row=pid_m,
                    k_col=col_start + tidx,
                    rows=rows,
                    cols=cols,
                    value=tile_scale_byte,
                )
                self._store_row_scale(
                    mScale=mS1,
                    row=row,
                    k_col=pid_k,
                    scale_cols=cols // SF_VEC,
                    value=tile_scale_byte,
                )
            return

        if cutlass.const_expr((self.axis_mask & AXIS_MASK_K) != 0):
            row_scale_byte, row_recip = (
                _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                    row_amax,
                    self.format_id,
                    self.half_range_scale,
                    mNvfp4RecipLut,
                    use_nvfp4_no_clip_scale=False,
                )
            )
            if active:
                _store_qdata_row(
                    mQ=mQ1,
                    vals=vals,
                    recip=row_recip,
                    row=row,
                    col_start=col_start,
                    sf_vec_size=self.sf_vec_size,
                    is_fp4=self.is_fp4,
                    format_id=self.format_id,
                )
                self._store_row_scale(
                    mScale=mS1,
                    row=row,
                    k_col=pid_k,
                    scale_cols=cols // SF_VEC,
                    value=row_scale_byte,
                )

        if cutlass.const_expr((self.axis_mask & AXIS_MASK_M) != 0):
            my_col_amax = Float32(_FP32_ZERO)
            for j in cutlass.range_constexpr(SF_VEC):
                col_val = _compute_block_amax_nonfinite(vals[j])
                col_val = cute.arch.warp_reduction(
                    col_val,
                    cute.arch.fmax,
                    threads_in_group=self.sf_vec_size,
                )
                if tidx == Int32(j):
                    my_col_amax = col_val

            col_scale_byte, my_col_recip = (
                _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                    my_col_amax,
                    self.format_id,
                    self.half_range_scale,
                    mNvfp4RecipLut,
                    use_nvfp4_no_clip_scale=False,
                )
            )
            if tidx < Int32(SF_VEC):
                self._store_col_scale(
                    mScale=mS0,
                    m_v_row=pid_m,
                    k_col=col_start + tidx,
                    rows=rows,
                    cols=cols,
                    value=col_scale_byte,
                )

            # Each row thread pulls col_recip from thread j to scale its
            # element at column j, then writes the result in col-major
            # layout (qdata contiguous along the original N axis).
            recip_row = cute.make_rmem_tensor(SF_VEC, Float32)
            for j in cutlass.range_constexpr(SF_VEC):
                recip_row[j] = cute.arch.shuffle_sync(my_col_recip, j)

            q_codes = cute.make_rmem_tensor(SF_VEC, Float32)
            for k in cutlass.range_constexpr(SF_VEC):
                q_codes[k] = vals[k] * recip_row[k]
            if active:
                if cutlass.const_expr(self.is_fp4):
                    _store_qdata_col_major_fp4(
                        mQ=mQ0,
                        q_codes=q_codes,
                        tidx=tidx,
                        row_start=row_start,
                        col_start=col_start,
                        sf_vec_size=self.sf_vec_size,
                    )
                else:
                    _store_qdata_col_major_fp8(
                        mQ=mQ0,
                        q_codes=q_codes,
                        tidx=tidx,
                        row_start=row_start,
                        col_start=col_start,
                        sf_vec_size=self.sf_vec_size,
                        format_id=self.format_id,
                    )

    @cute.jit
    def _get_logical_cols(self, mQ0: cute.Tensor, mQ1: cute.Tensor):
        logical_cols = mQ0.shape[0]
        if cutlass.const_expr(
            (
                self.row_col_kernel_mode
                in (
                    _AXES_KERNEL_MODE_SWIGLU_BWD_DXY_F32X2_VECTOR,
                    _ROW_COL_KERNEL_MODE_SWIGLU_BWD_DXY_FP8_TILE,
                )
                and not self.is_fp4
            )
            or (
                self.row_col_kernel_mode == _AXES_KERNEL_MODE_BOTH1D_VECTOR
                and self.axis_mask != AXIS_MASK_M
                and not self.is_fp4
            )
            or self.col1d_use_row_vector
        ):
            logical_cols = mQ1.shape[1]
        return logical_cols

    @cute.jit
    def _get_scale_offset(self, row, col, logical_cols):
        if cutlass.const_expr(self.layout_id == SCALE_FACTOR_LAYOUT_NATURAL):
            return row * logical_cols + col

        # cuBLAS blocked SF layout stores 128-row x 4-column atoms. The
        # caller passes row/col in the logical NATURAL scale matrix.
        n_col_blocks = _ceil_div_i32(
            logical_cols,
            Int32(CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM),
        )
        return _cublas_blockscaled_qscale_offset(row, col, n_col_blocks)

    @cute.jit
    def _get_rank_balanced_row_block(
        self, row_block, num_row_blocks, local_rank: Int32, world_size: Int32
    ):
        if cutlass.const_expr(
            self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
        ):
            if world_size > Int32(1):
                row_block += (num_row_blocks // world_size) * local_rank
                if row_block >= num_row_blocks:
                    row_block -= num_row_blocks
        return row_block

    @cute.jit
    def _get_source_col(self, logical_col, source_cols):
        if cutlass.const_expr(self.producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY):
            return (
                logical_col - source_cols if logical_col >= source_cols else logical_col
            )
        return logical_col

    @cute.jit
    def _get_vector_tile_coords(
        self,
        tidx,
        bidx,
        scale_cols,
        rows,
        local_rank: Int32,
        world_size: Int32,
    ):
        warp_id = tidx // Int32(32)
        lane = tidx - warp_id * Int32(32)
        tiles_per_warp: cutlass.Constexpr[int] = 32 // self.sf_vec_size
        tile_in_warp = lane // Int32(self.sf_vec_size)
        lane_in_tile = lane - tile_in_warp * Int32(self.sf_vec_size)
        tile_idx = (bidx * self.warps_per_cta + warp_id) * tiles_per_warp
        tile_idx = tile_idx + tile_in_warp
        row_block = tile_idx // scale_cols
        col_block = tile_idx - row_block * scale_cols
        row_block = self._get_rank_balanced_row_block(
            row_block=row_block,
            num_row_blocks=rows // self.sf_vec_size,
            local_rank=local_rank,
            world_size=world_size,
        )
        row_start = row_block * self.sf_vec_size
        col_start = col_block * self.sf_vec_size
        row = row_start + lane_in_tile
        return lane, lane_in_tile, row_block, col_block, row_start, col_start, row

    @cute.jit
    def _load_source_value(
        self,
        mX: cute.Tensor,
        row,
        col,
        source_cols,
    ) -> Float32:
        if cutlass.const_expr(
            self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
        ):
            row_addr = mX[row]
            value = Float32(_FP32_ZERO)
            if row_addr != Int64(0):
                row_ptr = cute.make_ptr(
                    self.input_dtype,
                    row_addr,
                    mem_space=cute.AddressSpace.gmem,
                    assumed_align=128,
                )
                row_tensor = cute.make_tensor(row_ptr, cute.make_layout((source_cols,)))
                value = Float32(row_tensor[col])
            return value
        return Float32(mX[row, col])

    @cute.jit
    def _load_row_values_as_f32(
        self,
        mValues: cute.Tensor,
        mValueWords: cute.Tensor,
        row,
        col_start,
        source_cols,
        n_values: cutlass.Constexpr[int],
        use_gather: cutlass.Constexpr[bool],
        words_aligned: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(use_gather):
            values = cute.make_rmem_tensor(n_values, Float32)
            chunk_elems: cutlass.Constexpr[int] = 4
            copy_bits: cutlass.Constexpr[int] = 128
            if cutlass.const_expr(
                (self.input_dtype == BFloat16 or self.input_dtype == Float16)
                and n_values != 4
            ):
                chunk_elems = 8
            if cutlass.const_expr(
                (self.input_dtype == BFloat16 or self.input_dtype == Float16)
                and n_values == 4
            ):
                copy_bits = 64
            if cutlass.const_expr(n_values % chunk_elems != 0):
                raise ValueError("gather row loads require aligned vector width")
            for i in cutlass.range_constexpr(n_values):
                values[i] = Float32(_FP32_ZERO)
            row_addr = mValues[row]
            if row_addr != Int64(0):
                row_ptr = cute.make_ptr(
                    self.input_dtype,
                    row_addr,
                    mem_space=cute.AddressSpace.gmem,
                    assumed_align=128,
                )
                row_tensor = cute.make_tensor(
                    row_ptr,
                    cute.make_ordered_layout((source_cols,), order=(0,)),
                )
                row_chunks = cute.tiled_divide(row_tensor, (chunk_elems,))
                copy_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.input_dtype,
                    num_bits_per_copy=copy_bits,
                )
                for chunk in cutlass.range_constexpr(n_values // chunk_elems):
                    loaded = cute.make_rmem_tensor(chunk_elems, self.input_dtype)
                    cute.copy(
                        copy_atom,
                        row_chunks[None, col_start // Int32(chunk_elems) + chunk],
                        loaded,
                    )
                    for i in cutlass.range_constexpr(chunk_elems):
                        values[chunk * chunk_elems + i] = Float32(loaded[i])
            return values
        if cutlass.const_expr(words_aligned):
            return _load_f16_words_as_f32(
                mWords=mValueWords,
                row=row,
                col_start=col_start,
                source_dtype=self.input_dtype,
                n_values=n_values,
            )
        values = cute.make_rmem_tensor(n_values, Float32)
        for i in cutlass.range_constexpr(n_values):
            values[i] = Float32(mValues[row, col_start + i])
        return values

    @cute.jit
    def _load_swiglu_bwd_dxy_values(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        mXWords: cute.Tensor,
        row,
        source_col_start,
    ):
        source_cols = mX.shape[1]
        input_words_aligned: cutlass.Constexpr[bool] = (
            self.input_dtype == BFloat16 or self.input_dtype == Float16
        )
        dz_vals = self._load_row_values_as_f32(
            mValues=mX,
            mValueWords=mXWords,
            row=row,
            col_start=source_col_start,
            source_cols=source_cols,
            n_values=self.sf_vec_size,
            use_gather=False,
            words_aligned=input_words_aligned,
        )
        x_vals = self._load_row_values_as_f32(
            mValues=mProducerB,
            mValueWords=mProducerBWords,
            row=row,
            col_start=source_col_start,
            source_cols=source_cols,
            n_values=self.sf_vec_size,
            use_gather=False,
            words_aligned=self.producer_b_words_aligned,
        )
        y_vals = self._load_row_values_as_f32(
            mValues=mProducerC,
            mValueWords=mProducerCWords,
            row=row,
            col_start=source_col_start,
            source_cols=source_cols,
            n_values=self.sf_vec_size,
            use_gather=False,
            words_aligned=self.producer_c_words_aligned,
        )

        dx_vals = cute.make_rmem_tensor(self.sf_vec_size, Float32)
        dy_vals = cute.make_rmem_tensor(self.sf_vec_size, Float32)
        for i in cutlass.range_constexpr(self.sf_vec_size):
            dx_vals[i], dy_vals[i] = _apply_block_scaled_quant_dxy_producer(
                dz_vals[i],
                x_vals[i],
                y_vals[i],
                self.fast_math,
                self.clamped,
                self.alpha,
                self.limit,
            )
        return dx_vals, dy_vals

    @cute.jit
    def _load_gather_row_col_fp8_values(
        self,
        mX: cute.Tensor,
        vals: cute.Tensor,
        source_cols,
        row_start,
        col_start,
        lane_row,
        packed_b16: cutlass.Constexpr[bool],
    ):
        for rep in cutlass.range_constexpr(8):
            row = row_start + lane_row + Int32(rep * 4)
            if cutlass.const_expr(packed_b16):
                loaded_packed = cute.make_rmem_tensor(8, Int32)
                _load_gather_row_as_b16x2(
                    mX[row],
                    source_cols,
                    col_start,
                    loaded_packed,
                    16,
                    self.input_dtype,
                )
                for pair in cutlass.range_constexpr(8):
                    vals[rep, pair] = loaded_packed[pair]
            else:
                loaded = self._load_row_values_as_f32(
                    mValues=mX,
                    mValueWords=mX,
                    row=row,
                    col_start=col_start,
                    source_cols=source_cols,
                    n_values=16,
                    use_gather=True,
                    words_aligned=False,
                )
                for j in cutlass.range_constexpr(16):
                    vals[rep, j] = loaded[j]

    @cute.jit
    def _load_row_values_and_apply_producer(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        mXWords: cute.Tensor,
        row,
        col_start,
        source_cols,
        n_values: cutlass.Constexpr[int],
    ):
        source_col_start = self._get_source_col(col_start, source_cols)
        input_words_aligned: cutlass.Constexpr[bool] = (
            self.input_dtype == BFloat16 or self.input_dtype == Float16
        )
        values = self._load_row_values_as_f32(
            mValues=mX,
            mValueWords=mXWords,
            row=row,
            col_start=source_col_start,
            source_cols=source_cols,
            n_values=n_values,
            use_gather=self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER,
            words_aligned=input_words_aligned,
        )
        return self._apply_producer_to_row_values(
            mProducerB,
            mProducerC,
            mProducerBWords,
            mProducerCWords,
            row,
            col_start,
            source_col_start,
            source_cols,
            values,
            True,
            n_values,
        )

    @cute.jit
    def _apply_producer_to_row_values(
        self,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        row,
        col_start,
        source_col_start,
        source_cols,
        values,
        active,
        n_values: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(
            self.producer_id
            in (
                BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
                BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY,
            )
        ):
            produced = cute.make_rmem_tensor(n_values, Float32)
            if active:
                if cutlass.const_expr(
                    self.producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD
                ):
                    if cutlass.const_expr(self.producer_b_words_aligned):
                        producer_b_values = self._load_row_values_as_f32(
                            mValues=mProducerB,
                            mValueWords=mProducerBWords,
                            row=row,
                            col_start=col_start,
                            source_cols=source_cols,
                            n_values=n_values,
                            use_gather=False,
                            words_aligned=True,
                        )
                    for i in cutlass.range_constexpr(n_values):
                        if cutlass.const_expr(self.producer_b_words_aligned):
                            producer_b = producer_b_values[i]
                        else:
                            producer_b = Float32(mProducerB[row, col_start + i])
                        produced[i] = _apply_block_scaled_quant_producer(
                            values[i],
                            producer_b,
                            self.producer_id,
                            self.fast_math,
                            self.clamped,
                            self.alpha,
                            self.limit,
                        )
                else:
                    if cutlass.const_expr(self.producer_b_words_aligned):
                        producer_b_values = self._load_row_values_as_f32(
                            mValues=mProducerB,
                            mValueWords=mProducerBWords,
                            row=row,
                            col_start=source_col_start,
                            source_cols=source_cols,
                            n_values=n_values,
                            use_gather=False,
                            words_aligned=True,
                        )
                    if cutlass.const_expr(self.producer_c_words_aligned):
                        producer_c_values = self._load_row_values_as_f32(
                            mValues=mProducerC,
                            mValueWords=mProducerCWords,
                            row=row,
                            col_start=source_col_start,
                            source_cols=source_cols,
                            n_values=n_values,
                            use_gather=False,
                            words_aligned=True,
                        )
                    is_dy_tile = col_start >= source_cols
                    for i in cutlass.range_constexpr(n_values):
                        if cutlass.const_expr(self.producer_b_words_aligned):
                            producer_b = producer_b_values[i]
                        else:
                            producer_b = Float32(mProducerB[row, source_col_start + i])
                        if cutlass.const_expr(self.producer_c_words_aligned):
                            producer_c = producer_c_values[i]
                        else:
                            producer_c = Float32(mProducerC[row, source_col_start + i])
                        dx, dy = _apply_block_scaled_quant_dxy_producer(
                            values[i],
                            producer_b,
                            producer_c,
                            self.fast_math,
                            self.clamped,
                            self.alpha,
                            self.limit,
                        )
                        produced[i] = dy if is_dy_tile else dx
            else:
                for i in cutlass.range_constexpr(n_values):
                    produced[i] = Float32(_FP32_ZERO)
            return produced
        return values

    @cute.jit
    def _compute_amax_from_f32_values(
        self,
        values,
        n_values: cutlass.Constexpr[int],
    ):
        return _frag_amax_nonfinite(values, 0, n_values)

    @cute.jit
    def _compute_gather_row_col_fp8_col_recips(
        self,
        mS0: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        col_amax_source: cute.Tensor,
        col_recips: cute.Tensor,
        rows,
        cols,
        col_start,
        pid_m,
        lane,
        lane_row,
        packed_b16: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(packed_b16):
            for pair in cutlass.range_constexpr(8):
                col_amax_packed = Int32(0)
                for rep in cutlass.range_constexpr(8):
                    col_amax_packed = _max_b16x2(
                        col_amax_packed,
                        _compute_amax_nonfinite_b16x2(
                            col_amax_source[rep, pair],
                            self.input_dtype,
                        ),
                        self.input_dtype,
                    )
                col_amax0, col_amax1 = _reduce_amax_b16x2(
                    col_amax_packed,
                    8,
                    2,
                    self.input_dtype,
                )
                for pair_idx in cutlass.range_constexpr(2):
                    col_amax = col_amax0
                    if cutlass.const_expr(pair_idx == 1):
                        col_amax = col_amax1
                    j: cutlass.Constexpr[int] = 2 * pair + pair_idx
                    col_scale_byte, col_recip = (
                        _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                            Float32(col_amax),
                            self.format_id,
                            self.half_range_scale,
                            mNvfp4RecipLut,
                        )
                    )
                    col_recips[j] = col_recip
                    if lane_row == Int32(0):
                        self._store_col_scale(
                            mScale=mS0,
                            m_v_row=pid_m,
                            k_col=col_start + Int32(j),
                            rows=rows,
                            cols=cols,
                            value=col_scale_byte,
                        )
        else:
            for j in cutlass.range_constexpr(16):
                col_amax = _reduce_amax_f32(col_amax_source[j], 8, 2)
                col_scale_byte, col_recip = (
                    _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                        col_amax,
                        self.format_id,
                        self.half_range_scale,
                        mNvfp4RecipLut,
                    )
                )
                col_recips[j] = col_recip
                if lane_row == Int32(0):
                    self._store_col_scale(
                        mScale=mS0,
                        m_v_row=pid_m,
                        k_col=col_start + Int32(j),
                        rows=rows,
                        cols=cols,
                        value=col_scale_byte,
                    )

    @cute.jit
    def _compute_gather_row_col_fp8_f32_amax(
        self,
        vals: cute.Tensor,
        row_amax_locals: cute.Tensor,
        col_amax_locals: cute.Tensor,
    ):
        for j in cutlass.range_constexpr(16):
            col_amax_locals[j] = Float32(_FP32_ZERO)
        # NaN-carrying max.NaN accumulation, one NaN -> inf substitution per
        # accumulator (bitwise identical to the per-element form; see
        # _frag_amax_nonfinite).
        for rep in cutlass.range_constexpr(8):
            row_amax_local = Float32(_FP32_ZERO)
            for j in cutlass.range_constexpr(16):
                abs_value = _abs_f32(vals[rep, j])
                row_amax_local = _fmax_nan(row_amax_local, abs_value)
                col_amax_locals[j] = _fmax_nan(col_amax_locals[j], abs_value)
            row_amax_locals[rep] = _compute_block_amax_nonfinite(row_amax_local)
        for j in cutlass.range_constexpr(16):
            col_amax_locals[j] = _compute_block_amax_nonfinite(col_amax_locals[j])

    @cute.jit
    def _store_col_scale(
        self,
        mScale: cute.Tensor,
        m_v_row,
        k_col,
        rows,
        cols,
        value,
    ) -> None:
        """Store column-quant scale. NATURAL: data-shape ``[M/V, K]``; BLOCKED:
        MMA-A frame (atom-MN spans K, atom-K spans M/V). ``rows`` and ``cols``
        are the input tensor's M and K dimensions."""
        if cutlass.const_expr(self.layout_id == SCALE_FACTOR_LAYOUT_NATURAL):
            offset = m_v_row * cols + k_col
        else:
            if cutlass.const_expr(self.grouped_rows_per_group > 0):
                group_m_v_cols: cutlass.Constexpr[int] = (
                    self.grouped_rows_per_group // self.sf_vec_size
                )
                group = m_v_row // group_m_v_cols
                local_m_v_row = m_v_row - group * group_m_v_cols
                offset = group * cols * group_m_v_cols + self._get_scale_offset(
                    k_col,
                    local_m_v_row,
                    group_m_v_cols,
                )
            else:
                m_v_cols = rows // self.sf_vec_size
                offset = self._get_scale_offset(k_col, m_v_row, m_v_cols)
        mScale[offset] = value

    @cute.jit
    def _store_row_scale(
        self,
        mScale: cute.Tensor,
        row,
        k_col,
        scale_cols,
        value,
    ) -> None:
        mScale[self._get_scale_offset(row, k_col, scale_cols)] = value

    @cute.jit
    def _store_row_qdata_words(
        self,
        mQWords: cute.Tensor,
        vals,
        recip,
        row,
        pid_k,
        lane,
    ):
        _store_qdata_row_words(
            mQWords=mQWords,
            vals=vals,
            recip=recip,
            flat_row=row,
            scale_col=pid_k,
            lane=lane,
            axes_kernel_mode=self.row_col_kernel_mode,
            is_fp4=self.is_fp4,
            sf_vec_size=self.sf_vec_size,
            num_elems=self.lane_elems,
            format_id=self.format_id,
        )

    @cute.jit
    def _store_row_tile(
        self,
        vals,
        mQ1Words: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        row,
        pid_k,
        scale_cols,
    ):
        row_scale_byte, row_recip = (
            _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                self._compute_amax_from_f32_values(vals, self.sf_vec_size),
                self.format_id,
                self.half_range_scale,
                mNvfp4RecipLut,
                use_nvfp4_no_clip_scale=False,
            )
        )
        self._store_row_qdata_words(
            mQWords=mQ1Words,
            vals=vals,
            recip=row_recip,
            row=row,
            pid_k=pid_k,
            lane=Int32(0),
        )
        self._store_row_scale(
            mScale=mS1,
            row=row,
            k_col=pid_k,
            scale_cols=scale_cols,
            value=row_scale_byte,
        )

    @cute.jit
    def _store_col_tile(
        self,
        vals,
        mQ0: cute.Tensor,
        mQ0Words: cute.Tensor,
        mS0: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        row_start,
        col_start,
        pid_m,
        rows,
        cols,
        lane,
        lane_in_tile,
        store_row_major_words: cutlass.Constexpr[bool],
    ):
        my_col_amax = Float32(_FP32_ZERO)
        for j in cutlass.range_constexpr(self.sf_vec_size):
            col_amax = cute.arch.warp_reduction(
                _compute_block_amax_nonfinite(vals[j]),
                cute.arch.fmax,
                threads_in_group=self.sf_vec_size,
            )
            if lane_in_tile == Int32(j):
                my_col_amax = col_amax

        col_scale_byte, my_col_recip = (
            _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                my_col_amax,
                self.format_id,
                self.half_range_scale,
                mNvfp4RecipLut,
                use_nvfp4_no_clip_scale=False,
            )
        )
        self._store_col_scale(
            mScale=mS0,
            m_v_row=pid_m,
            k_col=col_start + lane_in_tile,
            rows=rows,
            cols=cols,
            value=col_scale_byte,
        )

        q_codes = cute.make_rmem_tensor(self.sf_vec_size, Float32)
        tile_lane_base = lane - lane_in_tile
        for k in cutlass.range_constexpr(self.sf_vec_size):
            recip = cute.arch.shuffle_sync(my_col_recip, tile_lane_base + Int32(k))
            q_codes[k] = vals[k] * recip

        if cutlass.const_expr(
            store_row_major_words and (not self.is_fp4) and self.col_qdata_row_major
        ):
            row = row_start + lane_in_tile
            pid_k = col_start // self.sf_vec_size
            self._store_row_qdata_words(
                mQWords=mQ0Words,
                vals=q_codes,
                recip=Float32(_FP32_ONE),
                row=row,
                pid_k=pid_k,
                lane=lane,
            )
        elif cutlass.const_expr(self.is_fp4):
            _store_qdata_col_major_fp4(
                mQ=mQ0,
                q_codes=q_codes,
                tidx=lane_in_tile,
                row_start=row_start,
                col_start=col_start,
                sf_vec_size=self.sf_vec_size,
            )
        else:
            _store_qdata_col_major_fp8(
                mQ=mQ0,
                q_codes=q_codes,
                tidx=lane_in_tile,
                row_start=row_start,
                col_start=col_start,
                sf_vec_size=self.sf_vec_size,
                format_id=self.format_id,
            )

    @cute.jit
    def _store_col_tile_from_input(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mXWords: cute.Tensor,
        mQ0: cute.Tensor,
        mQ0Words: cute.Tensor,
        mS0: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        row_start,
        col_start,
        pid_m,
        rows,
        cols,
        lane,
        lane_in_tile,
        source_cols,
    ):
        col = col_start + lane_in_tile
        source_col = self._get_source_col(col, source_cols)
        if cutlass.const_expr(
            self.producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
        ):
            vals = self._load_row_values_as_f32(
                mValues=mX,
                mValueWords=mXWords,
                row=row_start + lane_in_tile,
                col_start=col_start,
                source_cols=source_cols,
                n_values=self.sf_vec_size,
                use_gather=True,
                words_aligned=False,
            )
            self._store_col_tile(
                vals=vals,
                mQ0=mQ0,
                mQ0Words=mQ0Words,
                mS0=mS0,
                mNvfp4RecipLut=mNvfp4RecipLut,
                row_start=row_start,
                col_start=col_start,
                pid_m=pid_m,
                rows=rows,
                cols=cols,
                lane=lane,
                lane_in_tile=lane_in_tile,
                store_row_major_words=False,
            )
        else:
            vals = cute.make_rmem_tensor(self.sf_vec_size, Float32)
            if cutlass.const_expr(self.producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD):
                for r in cutlass.range_constexpr(self.sf_vec_size):
                    vals[r] = _apply_block_scaled_quant_producer(
                        Float32(mX[row_start + r, source_col]),
                        Float32(mProducerB[row_start + r, source_col]),
                        self.producer_id,
                        self.fast_math,
                        self.clamped,
                        self.alpha,
                        self.limit,
                    )
            else:
                for r in cutlass.range_constexpr(self.sf_vec_size):
                    vals[r] = Float32(mX[row_start + r, source_col])
            col_scale_byte, col_recip = (
                _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                    self._compute_amax_from_f32_values(vals, self.sf_vec_size),
                    self.format_id,
                    self.half_range_scale,
                    mNvfp4RecipLut,
                    use_nvfp4_no_clip_scale=False,
                )
            )
            self._store_col_scale(
                mScale=mS0,
                m_v_row=pid_m,
                k_col=col,
                rows=rows,
                cols=cols,
                value=col_scale_byte,
            )
            _store_qdata_col_words(
                mQWords=mQ0Words,
                vals=vals,
                recip=col_recip,
                col=col,
                row_start=row_start,
                sf_vec_size=self.sf_vec_size,
                is_fp4=self.is_fp4,
                format_id=self.format_id,
            )

    @cute.jit
    def _store_row_col_fp8_rows(
        self,
        mQWords: cute.Tensor,
        vals,
        recips,
        row_recip: Float32,
        row_start,
        lane_row,
        q_word_col,
        use_row_recip: cutlass.Constexpr[bool],
        packed_b16: cutlass.Constexpr[bool],
        rep_base: cutlass.Constexpr[int],
        row_reps: cutlass.Constexpr[int],
        num_elems: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(num_elems % 4 != 0):
            raise ValueError("row-col FP8 stores require multiples of 4 values")
        if cutlass.const_expr(packed_b16 and num_elems % 2 != 0):
            raise ValueError("packed b16 row-col FP8 stores require even values")
        b16_pairs: cutlass.Constexpr[int] = num_elems // 2
        for rep_offset in cutlass.range_constexpr(row_reps):
            rep: cutlass.Constexpr[int] = rep_base + rep_offset
            row = row_start + lane_row + Int32(rep * 4)
            q_vals = cute.make_rmem_tensor(num_elems, Float32)
            if cutlass.const_expr(packed_b16):
                for pair in cutlass.range_constexpr(b16_pairs):
                    v0, v1 = _unpack_b16x2(vals[rep, pair], self.input_dtype)
                    if cutlass.const_expr(use_row_recip):
                        q_vals[2 * pair] = Float32(v0) * row_recip
                        q_vals[2 * pair + 1] = Float32(v1) * row_recip
                    else:
                        q_vals[2 * pair] = Float32(v0) * recips[2 * pair]
                        q_vals[2 * pair + 1] = Float32(v1) * recips[2 * pair + 1]
            else:
                for j in cutlass.range_constexpr(num_elems):
                    if cutlass.const_expr(use_row_recip):
                        q_vals[j] = vals[rep, j] * row_recip
                    else:
                        q_vals[j] = vals[rep, j] * recips[j]
            _store_fp8_qwords(
                mQWords=mQWords,
                q_vals=q_vals,
                row=row,
                q_word_col=q_word_col,
                num_elems=num_elems,
                format_id=self.format_id,
            )

    @cute.jit
    def _quantize_gather_row_col_fp8_tile(
        self,
        mX: cute.Tensor,
        mQ1Words: cute.Tensor,
        mQ0Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        rows,
        cols,
        source_cols,
        row_start,
        col_start,
        pid_m,
        lane,
        lane_row,
        lane_col_block,
        lane_col_pair,
        q_word_col,
        row_scale_offset_base,
        row_scale_store_stride,
    ):
        if cutlass.const_expr(
            self.input_dtype == BFloat16 or self.input_dtype == Float16
        ):
            vals_packed = cute.make_rmem_tensor((8, 8), Int32)
            self._load_gather_row_col_fp8_values(
                mX=mX,
                vals=vals_packed,
                source_cols=source_cols,
                row_start=row_start,
                col_start=col_start,
                lane_row=lane_row,
                packed_b16=True,
            )
            row_recips = cute.make_rmem_tensor(8, Float32)
            row_scale_store_lane = (lane_col_block - lane_col_pair * Int32(2)) == Int32(
                0
            )
            for rep in cutlass.range_constexpr(8):
                row_amax_local = _compute_row_amax_b16x2_x16(
                    vals_packed,
                    rep,
                    self.input_dtype,
                )
                row_amax = _reduce_amax_f32(row_amax_local, 1, 1)
                row_scale_u8, row_recip = _compute_scale_and_recip_from_amax(
                    row_amax,
                    self.format_id,
                    self.half_range_scale,
                )
                row_recips[rep] = row_recip
                if row_scale_store_lane:
                    mS1[row_scale_offset_base + Int32(rep) * row_scale_store_stride] = (
                        row_scale_u8
                    )

            for rep in cutlass.range_constexpr(8):
                self._store_row_col_fp8_rows(
                    mQWords=mQ1Words,
                    vals=vals_packed,
                    recips=row_recips,
                    row_recip=row_recips[rep],
                    row_start=row_start,
                    lane_row=lane_row,
                    q_word_col=q_word_col,
                    use_row_recip=True,
                    packed_b16=True,
                    rep_base=rep,
                    row_reps=1,
                    num_elems=16,
                )
            col_recips = cute.make_rmem_tensor(16, Float32)
            self._compute_gather_row_col_fp8_col_recips(
                mS0=mS0,
                mNvfp4RecipLut=mNvfp4RecipLut,
                col_amax_source=vals_packed,
                col_recips=col_recips,
                rows=rows,
                cols=cols,
                col_start=col_start,
                pid_m=pid_m,
                lane=lane,
                lane_row=lane_row,
                packed_b16=True,
            )
            for rep in cutlass.range_constexpr(8):
                self._store_row_col_fp8_rows(
                    mQWords=mQ0Words,
                    vals=vals_packed,
                    recips=col_recips,
                    row_recip=Float32(_FP32_ZERO),
                    row_start=row_start,
                    lane_row=lane_row,
                    q_word_col=q_word_col,
                    use_row_recip=False,
                    packed_b16=True,
                    rep_base=rep,
                    row_reps=1,
                    num_elems=16,
                )
        else:
            vals = cute.make_rmem_tensor((8, 16), Float32)
            self._load_gather_row_col_fp8_values(
                mX=mX,
                vals=vals,
                source_cols=source_cols,
                row_start=row_start,
                col_start=col_start,
                lane_row=lane_row,
                packed_b16=False,
            )
            row_amax_locals = cute.make_rmem_tensor(8, Float32)
            col_amax_locals = cute.make_rmem_tensor(16, Float32)
            self._compute_gather_row_col_fp8_f32_amax(
                vals=vals,
                row_amax_locals=row_amax_locals,
                col_amax_locals=col_amax_locals,
            )

            row_q_vals = cute.make_rmem_tensor(16, Float32)
            row_q_f32x2_packed = cute.make_rmem_tensor(8, Uint64)
            _quantize_fp8_tile_values(
                vals=vals,
                mQWords=mQ1Words,
                mScale=mS1,
                scale_recips=row_amax_locals,
                q_vals=row_q_vals,
                q_f32x2_packed=row_q_f32x2_packed,
                row_start=row_start,
                row_lane=lane_row,
                q_word=q_word_col,
                scale_offset_base=row_scale_offset_base,
                scale_store_lane=(lane_col_block - lane_col_pair * Int32(2))
                == Int32(0),
                row_lanes_cfg=4,
                row_reps_cfg=8,
                num_elems=16,
                lane_pairs=8,
                reduce_lane_stride=1,
                reduce_stages=1,
                scale_store_stride=row_scale_store_stride,
                col_quant=False,
                packed_f32x2=False,
                precomputed_amax=True,
                paired_scale_compute=False,
                format_id=self.format_id,
                half_range_scale=self.half_range_scale,
            )

            col_recips = cute.make_rmem_tensor(16, Float32)
            self._compute_gather_row_col_fp8_col_recips(
                mS0=mS0,
                mNvfp4RecipLut=mNvfp4RecipLut,
                col_amax_source=col_amax_locals,
                col_recips=col_recips,
                rows=rows,
                cols=cols,
                col_start=col_start,
                pid_m=pid_m,
                lane=lane,
                lane_row=lane_row,
                packed_b16=False,
            )
            self._store_row_col_fp8_rows(
                mQWords=mQ0Words,
                vals=vals,
                recips=col_recips,
                row_recip=Float32(_FP32_ZERO),
                row_start=row_start,
                lane_row=lane_row,
                q_word_col=q_word_col,
                use_row_recip=False,
                packed_b16=False,
                rep_base=0,
                row_reps=8,
                num_elems=16,
            )

    @cute.jit
    def _run_swiglu_fwd_fp8_tile(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mQ0Words: cute.Tensor,
        mQ1Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        source_cols,
        tidx,
        bidx,
    ):
        warp_id = tidx // Int32(32)
        warp_lane = tidx - warp_id * Int32(32)
        row_blocks = mX.shape[0] // self.sf_vec_size
        scale_cols = source_cols // self.sf_vec_size
        col_work_tiles = (
            scale_cols + Int32(self.scale_cols_per_tile) - Int32(1)
        ) // Int32(self.scale_cols_per_tile)
        tile_idx = bidx * Int32(self.warps_per_cta) + warp_id
        total_tiles = row_blocks * col_work_tiles
        if tile_idx < total_tiles:
            row_block = tile_idx // col_work_tiles
            col_work_tile = tile_idx - row_block * col_work_tiles
            row_start = row_block * Int32(self.sf_vec_size)
            col_block = col_work_tile * Int32(self.scale_cols_per_tile)
            _quantize_swiglu_fwd_fp8_tile(
                warp_lane,
                mX,
                mProducerB,
                mQ1Words,
                mS1,
                mQ0Words,
                mS0,
                row_start,
                row_block,
                col_block,
                row_blocks,
                source_cols,
                self.input_dtype,
                self.sf_vec_size,
                self.format_id,
                self.fast_math,
                self.clamped,
                self.alpha,
                self.limit,
                self.qdata_elems_per_word,
                self.lane_elems,
                self.col_blocks_per_scale,
                self.col_lanes_cfg,
                (self.axis_mask & AXIS_MASK_K) != 0,
                (self.axis_mask & AXIS_MASK_M) != 0,
            )

    @cute.jit
    def _run_swiglu_bwd_dxy_fp8_tile(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mQ0Words: cute.Tensor,
        mQ1Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        source_cols,
        tidx,
        bidx,
    ):
        warp_id = tidx // Int32(32)
        warp_lane = tidx - warp_id * Int32(32)
        row_blocks = mX.shape[0] // self.sf_vec_size
        source_scale_cols = source_cols // self.sf_vec_size
        tile_idx = bidx * Int32(self.warps_per_cta) + warp_id
        total_tiles = row_blocks * source_scale_cols
        if tile_idx < total_tiles:
            row_block = tile_idx // source_scale_cols
            col_block = tile_idx - row_block * source_scale_cols
            row_start = row_block * Int32(self.sf_vec_size)
            _quantize_swiglu_bwd_dxy_fp8_tile(
                warp_lane=warp_lane,
                mDz=mX,
                mH1=mProducerB,
                mRowQWords=mQ1Words,
                mRowScale=mS1,
                mDxyColQWords=mQ0Words,
                mDxyColScale=mS0,
                row_start=row_start,
                row_block=row_block,
                col_block=col_block,
                row_blocks=row_blocks,
                K=mProducerB.shape[1],
                source_dtype=self.input_dtype,
                sf_vec_size=self.sf_vec_size,
                format_id=self.format_id,
                fast_math=self.fast_math,
                clamped=self.clamped,
                alpha=self.alpha,
                limit=self.limit,
                qdata_elems_per_word=self.qdata_elems_per_word,
                num_elems=self.lane_elems,
                col_blocks_per_scale=self.col_blocks_per_scale,
                col_lanes_cfg=self.col_lanes_cfg,
            )

    @cute.jit
    def _run_swiglu_bwd_dxy_f32x2_vectorized(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQ1Words: cute.Tensor,
        mQ0: cute.Tensor,
        mQ0Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        logical_cols,
        tidx,
        bidx,
    ):
        rows = mX.shape[0]
        source_cols = mX.shape[1]
        cols = logical_cols
        scale_cols = cols // self.sf_vec_size
        source_scale_cols = source_cols // self.sf_vec_size
        warp_id = tidx // Int32(32)
        lane = tidx - warp_id * Int32(32)
        tiles_per_warp: cutlass.Constexpr[int] = 32 // self.sf_vec_size
        tile_in_warp = lane // Int32(self.sf_vec_size)
        lane_in_tile = lane - tile_in_warp * Int32(self.sf_vec_size)
        tile_idx = (bidx * self.warps_per_cta + warp_id) * tiles_per_warp
        tile_idx = tile_idx + tile_in_warp
        pid_m = tile_idx // source_scale_cols
        pid_k_src = tile_idx - pid_m * source_scale_cols
        row_start = pid_m * self.sf_vec_size
        source_col_start = pid_k_src * self.sf_vec_size
        row = row_start + lane_in_tile

        dx_vals, dy_vals = self._load_swiglu_bwd_dxy_values(
            mX=mX,
            mProducerB=mProducerB,
            mProducerC=mProducerC,
            mProducerBWords=mProducerBWords,
            mProducerCWords=mProducerCWords,
            mXWords=mXWords,
            row=row,
            source_col_start=source_col_start,
        )

        if cutlass.const_expr((self.axis_mask & AXIS_MASK_K) != 0):
            self._store_row_tile(
                vals=dx_vals,
                mQ1Words=mQ1Words,
                mS1=mS1,
                mNvfp4RecipLut=mNvfp4RecipLut,
                row=row,
                pid_k=pid_k_src,
                scale_cols=scale_cols,
            )
            self._store_row_tile(
                vals=dy_vals,
                mQ1Words=mQ1Words,
                mS1=mS1,
                mNvfp4RecipLut=mNvfp4RecipLut,
                row=row,
                pid_k=source_scale_cols + pid_k_src,
                scale_cols=scale_cols,
            )
        if cutlass.const_expr((self.axis_mask & AXIS_MASK_M) != 0):
            self._store_col_tile(
                vals=dx_vals,
                mQ0=mQ0,
                mQ0Words=mQ0Words,
                mS0=mS0,
                mNvfp4RecipLut=mNvfp4RecipLut,
                row_start=row_start,
                col_start=source_col_start,
                pid_m=pid_m,
                rows=rows,
                cols=cols,
                lane=lane,
                lane_in_tile=lane_in_tile,
                store_row_major_words=True,
            )
            self._store_col_tile(
                vals=dy_vals,
                mQ0=mQ0,
                mQ0Words=mQ0Words,
                mS0=mS0,
                mNvfp4RecipLut=mNvfp4RecipLut,
                row_start=row_start,
                col_start=source_cols + source_col_start,
                pid_m=pid_m,
                rows=rows,
                cols=cols,
                lane=lane,
                lane_in_tile=lane_in_tile,
                store_row_major_words=True,
            )

    @cute.jit
    def _run_tile2d_fp8_vectorized(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQ1Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        logical_cols,
        source_cols,
        tidx,
        bidx,
        local_rank,
        world_size,
    ):
        rows = mX.shape[0]
        cols = logical_cols
        scale_cols = cols // self.sf_vec_size
        _lane, lane_in_tile, pid_m, pid_k, _row_start, col_start, row = (
            self._get_vector_tile_coords(
                tidx=tidx,
                bidx=bidx,
                scale_cols=scale_cols,
                rows=rows,
                local_rank=local_rank,
                world_size=world_size,
            )
        )
        values = self._load_row_values_and_apply_producer(
            mX=mX,
            mProducerB=mProducerB,
            mProducerC=mProducerC,
            mProducerBWords=mProducerBWords,
            mProducerCWords=mProducerCWords,
            mXWords=mXWords,
            row=row,
            col_start=col_start,
            source_cols=source_cols,
            n_values=32,
        )

        tile_amax = cute.arch.warp_reduction(
            self._compute_amax_from_f32_values(values, 32),
            cute.arch.fmax,
            threads_in_group=self.sf_vec_size,
        )

        tile_scale_byte, tile_recip = (
            _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                tile_amax,
                self.format_id,
                self.half_range_scale,
                mNvfp4RecipLut,
            )
        )
        self._store_col_scale(
            mScale=mS0,
            m_v_row=pid_m,
            k_col=col_start + lane_in_tile,
            rows=rows,
            cols=cols,
            value=tile_scale_byte,
        )
        self._store_row_scale(
            mScale=mS1,
            row=row,
            k_col=pid_k,
            scale_cols=scale_cols,
            value=tile_scale_byte,
        )
        self._store_row_qdata_words(
            mQWords=mQ1Words,
            vals=values,
            recip=tile_recip,
            row=row,
            pid_k=pid_k,
            lane=Int32(0),
        )

    # Axis/layout branches are constexpr-folded by CuTeDSL JIT.
    @cute.jit
    def _run_tile2d_fp4_vectorized(  # noqa: C901
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQ0: cute.Tensor,
        mQ1Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        logical_cols,
        source_cols,
        tidx,
        bidx,
        local_rank,
        world_size,
    ):
        rows = mX.shape[0]
        cols = logical_cols
        scale_cols = cols // self.sf_vec_size
        lane, lane_in_tile, pid_m, pid_k, row_start, col_start, row = (
            self._get_vector_tile_coords(
                tidx=tidx,
                bidx=bidx,
                scale_cols=scale_cols,
                rows=rows,
                local_rank=local_rank,
                world_size=world_size,
            )
        )
        values = self._load_row_values_and_apply_producer(
            mX=mX,
            mProducerB=mProducerB,
            mProducerC=mProducerC,
            mProducerBWords=mProducerBWords,
            mProducerCWords=mProducerCWords,
            mXWords=mXWords,
            row=row,
            col_start=col_start,
            source_cols=source_cols,
            n_values=self.sf_vec_size,
        )

        tile_amax = cute.arch.warp_reduction(
            self._compute_amax_from_f32_values(values, self.sf_vec_size),
            cute.arch.fmax,
            threads_in_group=self.sf_vec_size,
        )

        tile_scale_byte, tile_recip = (
            _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
                tile_amax,
                self.format_id,
                self.half_range_scale,
                mNvfp4RecipLut,
            )
        )
        self._store_row_qdata_words(
            mQWords=mQ1Words,
            vals=values,
            recip=tile_recip,
            row=row,
            pid_k=pid_k,
            lane=Int32(0),
        )

        q_values = cute.make_rmem_tensor(self.sf_vec_size, Float32)
        for i in cutlass.range_constexpr(self.sf_vec_size):
            q_values[i] = values[i] * tile_recip
        _store_qdata_col_major_fp4(
            mQ=mQ0,
            q_codes=q_values,
            tidx=lane_in_tile,
            row_start=row_start,
            col_start=col_start,
            sf_vec_size=self.sf_vec_size,
        )
        self._store_col_scale(
            mScale=mS0,
            m_v_row=pid_m,
            k_col=col_start + lane_in_tile,
            rows=rows,
            cols=cols,
            value=tile_scale_byte,
        )
        self._store_row_scale(
            mScale=mS1,
            row=row,
            k_col=pid_k,
            scale_cols=scale_cols,
            value=tile_scale_byte,
        )

    @cute.jit
    def _run_gather_col_fp8_vectorized(
        self,
        mX: cute.Tensor,
        mQ0Words: cute.Tensor,
        mS0: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        logical_cols,
        source_cols,
        tidx,
        bidx,
        local_rank,
        world_size,
    ):
        rows = mX.shape[0]
        cols = logical_cols
        scale_cols = cols // self.sf_vec_size
        warp_id = tidx // Int32(32)
        lane = tidx - warp_id * Int32(32)
        scale_col_groups = scale_cols // self.gather_scale_cols_per_tile
        pid_m_group = bidx // scale_col_groups
        pid_k_group = bidx - pid_m_group * scale_col_groups
        pid_m = pid_m_group * self.col1d_m_blocks + warp_id
        pid_m = self._get_rank_balanced_row_block(
            row_block=pid_m,
            num_row_blocks=rows // self.sf_vec_size,
            local_rank=local_rank,
            world_size=world_size,
        )
        row_start = pid_m * self.sf_vec_size
        col_group_base = (
            pid_k_group * self.gather_scale_cols_per_tile * self.sf_vec_size
        )
        col_base = col_group_base + lane * Int32(4)

        vals0 = cute.make_rmem_tensor(self.sf_vec_size, Float32)
        vals1 = cute.make_rmem_tensor(self.sf_vec_size, Float32)
        vals2 = cute.make_rmem_tensor(self.sf_vec_size, Float32)
        vals3 = cute.make_rmem_tensor(self.sf_vec_size, Float32)
        for r in cutlass.range_constexpr(self.sf_vec_size):
            loaded = self._load_row_values_as_f32(
                mValues=mX,
                mValueWords=mX,
                row=row_start + r,
                col_start=col_base,
                source_cols=source_cols,
                n_values=4,
                use_gather=True,
                words_aligned=False,
            )
            vals0[r] = loaded[0]
            vals1[r] = loaded[1]
            vals2[r] = loaded[2]
            vals3[r] = loaded[3]
        # 3-input max.NaN trees over the buffered columns (max is
        # order-insensitive, so this is bitwise identical to the interleaved
        # 2-input chain at half the max instructions).
        amax0 = _frag_amax_nonfinite(vals0, 0, self.sf_vec_size)
        amax1 = _frag_amax_nonfinite(vals1, 0, self.sf_vec_size)
        amax2 = _frag_amax_nonfinite(vals2, 0, self.sf_vec_size)
        amax3 = _frag_amax_nonfinite(vals3, 0, self.sf_vec_size)

        scale0, recip0 = _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
            amax0,
            self.format_id,
            self.half_range_scale,
            mNvfp4RecipLut,
        )
        scale1, recip1 = _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
            amax1,
            self.format_id,
            self.half_range_scale,
            mNvfp4RecipLut,
        )
        scale2, recip2 = _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
            amax2,
            self.format_id,
            self.half_range_scale,
            mNvfp4RecipLut,
        )
        scale3, recip3 = _compute_scale_and_recip_from_amax_with_optional_nvfp4_lut(
            amax3,
            self.format_id,
            self.half_range_scale,
            mNvfp4RecipLut,
        )

        self._store_col_scale(
            mScale=mS0,
            m_v_row=pid_m,
            k_col=col_base,
            rows=rows,
            cols=cols,
            value=scale0,
        )
        self._store_col_scale(
            mScale=mS0,
            m_v_row=pid_m,
            k_col=col_base + Int32(1),
            rows=rows,
            cols=cols,
            value=scale1,
        )
        self._store_col_scale(
            mScale=mS0,
            m_v_row=pid_m,
            k_col=col_base + Int32(2),
            rows=rows,
            cols=cols,
            value=scale2,
        )
        self._store_col_scale(
            mScale=mS0,
            m_v_row=pid_m,
            k_col=col_base + Int32(3),
            rows=rows,
            cols=cols,
            value=scale3,
        )
        q_word_col = col_base // Int32(4)
        for r in cutlass.range_constexpr(self.sf_vec_size):
            q0 = vals0[r] * recip0
            q1 = vals1[r] * recip1
            q2 = vals2[r] * recip2
            q3 = vals3[r] * recip3
            if cutlass.const_expr(self.format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2):
                packed = _cvt_f32x4_to_fp8x4_u32_rn(
                    q0,
                    q1,
                    q2,
                    q3,
                    dst_kind=nvvm.CVTPackFloatKind.E5M2x2,
                )
            else:
                packed = _cvt_f32x4_to_fp8x4_u32_rn(
                    q0,
                    q1,
                    q2,
                    q3,
                    dst_kind=nvvm.CVTPackFloatKind.E4M3x2,
                )
            mQ0Words[row_start + Int32(r), q_word_col] = packed

    @cute.jit
    def _run_gather_row_col_fp8_vectorized(
        self,
        mX: cute.Tensor,
        mQ1Words: cute.Tensor,
        mQ0Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        logical_cols,
        source_cols,
        tidx,
        bidx,
        local_rank,
        world_size,
    ):
        rows = mX.shape[0]
        cols = logical_cols
        scale_cols = cols // self.sf_vec_size
        row_blocks = rows // self.sf_vec_size
        pid_k_group = bidx // row_blocks
        pid_m = bidx - pid_k_group * row_blocks
        pid_m = self._get_rank_balanced_row_block(
            row_block=pid_m,
            num_row_blocks=row_blocks,
            local_rank=local_rank,
            world_size=world_size,
        )
        row_start = pid_m * self.sf_vec_size
        col_group_base = (
            pid_k_group * self.gather_scale_cols_per_tile * self.sf_vec_size
        )

        lane = tidx
        # Warp layout is 4 row lanes x 8 column lanes. Each lane owns 8 row
        # repetitions and 16 contiguous columns, covering one 32x128 FP8 tile
        # while keeping row and column reductions warp-local.
        lane_row = lane // Int32(8)
        lane_col_block = lane - lane_row * Int32(8)
        lane_col_pair = lane_col_block // Int32(2)
        col_start = col_group_base + lane_col_block * Int32(16)
        q_word_col = col_start // Int32(4)
        scale_col = pid_k_group * self.gather_scale_cols_per_tile + lane_col_pair
        row_scale_offset_base = self._get_scale_offset(
            row_start + lane_row,
            scale_col,
            scale_cols,
        )
        row_scale_store_stride = Int32(4) * scale_cols
        if cutlass.const_expr(self.layout_id == SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED):
            row_scale_store_stride = Int32(4 * CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE)

        self._quantize_gather_row_col_fp8_tile(
            mX=mX,
            mQ1Words=mQ1Words,
            mQ0Words=mQ0Words,
            mS0=mS0,
            mS1=mS1,
            mNvfp4RecipLut=mNvfp4RecipLut,
            rows=rows,
            cols=cols,
            source_cols=source_cols,
            row_start=row_start,
            col_start=col_start,
            pid_m=pid_m,
            lane=lane,
            lane_row=lane_row,
            lane_col_block=lane_col_block,
            lane_col_pair=lane_col_pair,
            q_word_col=q_word_col,
            row_scale_offset_base=row_scale_offset_base,
            row_scale_store_stride=row_scale_store_stride,
        )

    @cute.jit
    def _run_col1d_vectorized(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mXWords: cute.Tensor,
        mQ0: cute.Tensor,
        mQ0Words: cute.Tensor,
        mS0: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        logical_cols,
        source_cols,
        tidx,
        bidx,
        local_rank,
        world_size,
    ):
        rows = mX.shape[0]
        cols = logical_cols
        scale_cols = cols // self.sf_vec_size
        warp_id = tidx // Int32(32)
        lane = tidx - warp_id * Int32(32)
        tiles_per_warp: cutlass.Constexpr[int] = 32 // self.sf_vec_size
        tile_in_warp = lane // Int32(self.sf_vec_size)
        lane_in_tile = lane - tile_in_warp * Int32(self.sf_vec_size)
        tile_group_idx = (bidx * self.warps_per_cta + warp_id) * tiles_per_warp
        tile_group_idx = tile_group_idx + tile_in_warp
        pid_m_group = tile_group_idx // scale_cols
        pid_k = tile_group_idx - pid_m_group * scale_cols
        col_start = pid_k * self.sf_vec_size

        for m_iter in cutlass.range_constexpr(self.col1d_m_blocks):
            pid_m = pid_m_group * self.col1d_m_blocks + m_iter
            pid_m = self._get_rank_balanced_row_block(
                row_block=pid_m,
                num_row_blocks=rows // self.sf_vec_size,
                local_rank=local_rank,
                world_size=world_size,
            )
            row_start = pid_m * self.sf_vec_size
            self._store_col_tile_from_input(
                mX=mX,
                mProducerB=mProducerB,
                mXWords=mXWords,
                mQ0=mQ0,
                mQ0Words=mQ0Words,
                mS0=mS0,
                mNvfp4RecipLut=mNvfp4RecipLut,
                row_start=row_start,
                col_start=col_start,
                pid_m=pid_m,
                rows=rows,
                cols=cols,
                lane=lane,
                lane_in_tile=lane_in_tile,
                source_cols=source_cols,
            )

    @cute.jit
    def _run_both1d_fp8_vectorized(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQ1Words: cute.Tensor,
        mQ0: cute.Tensor,
        mQ0Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        logical_cols,
        source_cols,
        tidx,
        bidx,
        local_rank,
        world_size,
    ):
        rows = mX.shape[0]
        cols = logical_cols
        scale_cols = cols // self.sf_vec_size
        lane, lane_in_tile, pid_m, pid_k, row_start, col_start, row = (
            self._get_vector_tile_coords(
                tidx=tidx,
                bidx=bidx,
                scale_cols=scale_cols,
                rows=rows,
                local_rank=local_rank,
                world_size=world_size,
            )
        )
        vals = self._load_row_values_and_apply_producer(
            mX=mX,
            mProducerB=mProducerB,
            mProducerC=mProducerC,
            mProducerBWords=mProducerBWords,
            mProducerCWords=mProducerCWords,
            mXWords=mXWords,
            row=row,
            col_start=col_start,
            source_cols=source_cols,
            n_values=self.sf_vec_size,
        )

        if cutlass.const_expr((self.axis_mask & AXIS_MASK_K) != 0):
            self._store_row_tile(
                vals=vals,
                mQ1Words=mQ1Words,
                mS1=mS1,
                mNvfp4RecipLut=mNvfp4RecipLut,
                row=row,
                pid_k=pid_k,
                scale_cols=scale_cols,
            )
        if cutlass.const_expr((self.axis_mask & AXIS_MASK_M) != 0):
            self._store_col_tile(
                vals=vals,
                mQ0=mQ0,
                mQ0Words=mQ0Words,
                mS0=mS0,
                mNvfp4RecipLut=mNvfp4RecipLut,
                row_start=row_start,
                col_start=col_start,
                pid_m=pid_m,
                rows=rows,
                cols=cols,
                lane=lane,
                lane_in_tile=lane_in_tile,
                store_row_major_words=True,
            )

    @cute.jit
    def _run_both1d_fp4_vectorized(
        self,
        mX: cute.Tensor,
        mProducerB: cute.Tensor,
        mProducerC: cute.Tensor,
        mProducerBWords: cute.Tensor,
        mProducerCWords: cute.Tensor,
        mXWords: cute.Tensor,
        mQ0: cute.Tensor,
        mQ0Words: cute.Tensor,
        mQ1Words: cute.Tensor,
        mS0: cute.Tensor,
        mS1: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        logical_cols,
        source_cols,
        tidx,
        bidx,
        local_rank,
        world_size,
    ):
        rows = mX.shape[0]
        cols = logical_cols
        scale_cols = cols // self.sf_vec_size
        warp_id = tidx // Int32(32)
        lane = tidx - warp_id * Int32(32)
        tiles_per_cta: cutlass.Constexpr[int] = 32 // self.sf_vec_size
        tile_in_warp = lane // Int32(self.sf_vec_size)
        lane_in_tile = lane - tile_in_warp * Int32(self.sf_vec_size)
        for tile_group_iter in cutlass.range_constexpr(self.both1d_split_tile_groups):
            tile_idx = (
                bidx * self.both1d_split_tile_groups + tile_group_iter
            ) * tiles_per_cta + tile_in_warp
            pid_m = tile_idx // scale_cols
            pid_k = tile_idx - pid_m * scale_cols
            pid_m = self._get_rank_balanced_row_block(
                row_block=pid_m,
                num_row_blocks=rows // self.sf_vec_size,
                local_rank=local_rank,
                world_size=world_size,
            )
            row_start = pid_m * self.sf_vec_size
            col_start = pid_k * self.sf_vec_size

            if warp_id == Int32(0):
                row = row_start + lane_in_tile
                vals = self._load_row_values_and_apply_producer(
                    mX=mX,
                    mProducerB=mProducerB,
                    mProducerC=mProducerC,
                    mProducerBWords=mProducerBWords,
                    mProducerCWords=mProducerCWords,
                    mXWords=mXWords,
                    row=row,
                    col_start=col_start,
                    source_cols=source_cols,
                    n_values=self.sf_vec_size,
                )

                self._store_row_tile(
                    vals=vals,
                    mQ1Words=mQ1Words,
                    mS1=mS1,
                    mNvfp4RecipLut=mNvfp4RecipLut,
                    row=row,
                    pid_k=pid_k,
                    scale_cols=scale_cols,
                )

            else:
                self._store_col_tile_from_input(
                    mX=mX,
                    mProducerB=mProducerB,
                    mXWords=mXWords,
                    mQ0=mQ0,
                    mQ0Words=mQ0Words,
                    mS0=mS0,
                    mNvfp4RecipLut=mNvfp4RecipLut,
                    row_start=row_start,
                    col_start=col_start,
                    pid_m=pid_m,
                    rows=rows,
                    cols=cols,
                    lane=lane,
                    lane_in_tile=lane_in_tile,
                    source_cols=source_cols,
                )


# =============================================================================
# Host Classes
# =============================================================================


@dataclass(frozen=True)
class _RowColWordViews:
    row_col_kernel_mode: int
    producer_b_words_aligned: bool
    producer_c_words_aligned: bool
    producer_b_words: torch.Tensor
    producer_c_words: torch.Tensor
    x_words: torch.Tensor
    q0_words: torch.Tensor
    q1_words: torch.Tensor


@dataclass(frozen=True)
class _RowColShapeConfig:
    source_cols: int
    n_rows: int
    n_cols: int
    has_col_axis: bool
    has_row_axis: bool
    is_two_d: bool
    is_col1d: bool


# =============================================================================
# Host Helpers
# =============================================================================


# -----------------------------------------------------------------------------
# Get helpers
# -----------------------------------------------------------------------------


def _get_sf_vec_size_for_format_id(format_id: int) -> int:
    if format_id in _FORMAT_SF_VEC:
        return _FORMAT_SF_VEC[format_id]
    raise ValueError(f"Unsupported format_id={format_id}")


def _get_scale_dtype_for_format_id(format_id: int) -> torch.dtype:
    return (
        torch.float8_e4m3fn
        if format_id == BLOCK_SCALED_FORMAT_NVFP4
        else torch.float8_e8m0fnu
    )


def get_nvfp4_recip_lut(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    lut = _NVFP4_RECIP_LUT_CACHE.get(key)
    if lut is None:
        scale_bytes = torch.arange(256, dtype=torch.uint8, device=device)
        lut = 1.0 / scale_bytes.view(torch.float8_e4m3fn).to(torch.float32)
        _NVFP4_RECIP_LUT_CACHE[key] = lut
    return lut


# -----------------------------------------------------------------------------
# Make helpers
# -----------------------------------------------------------------------------


def _make_empty_quantized_qdata(
    shape: tuple[int, ...],
    *,
    format_id: int,
    device: torch.device,
    row_alignment_bytes: int | None = None,
) -> torch.Tensor:
    if format_id == BLOCK_SCALED_FORMAT_MXFP8_E4M3:
        dtype = torch.float8_e4m3fn
    elif format_id == BLOCK_SCALED_FORMAT_MXFP8_E5M2:
        dtype = torch.float8_e5m2
    else:
        dtype = torch.float4_e2m1fn_x2
    if dtype == torch.float4_e2m1fn_x2:
        if row_alignment_bytes is not None:
            raise ValueError("row-aligned qdata allocation requires an FP8 format")
        *prefix, cols = shape
        buf = torch.empty((*prefix, cols // 2), dtype=torch.uint8, device=device)
        return buf.view(torch.float4_e2m1fn_x2)
    if row_alignment_bytes is not None:
        if len(shape) != 2:
            raise ValueError("row-aligned qdata allocation requires a 2D shape")
        n_rows, row_bytes = shape
        padded_row_bytes = (
            (row_bytes + row_alignment_bytes - 1)
            // row_alignment_bytes
            * row_alignment_bytes
        )
        storage = torch.zeros(
            (n_rows, padded_row_bytes),
            dtype=torch.uint8,
            device=device,
        )
        return storage[:, :row_bytes].view(dtype)
    return torch.empty(shape, dtype=dtype, device=device)


def _make_row1d_scale_shape(
    shape: torch.Size,
    scale_cols: int,
    layout: ScaleFactorLayout,
) -> tuple[int, ...]:
    if layout == ScaleFactorLayout.NATURAL:
        return (*shape[:-1], scale_cols)
    return (math.prod(shape[:-1]), scale_cols)


# -----------------------------------------------------------------------------
# Prepare helpers
# -----------------------------------------------------------------------------


def _prepare_row_col_input_view(
    x: torch.Tensor,
    *,
    producer_id: int,
    output_shape: tuple[int, ...] | None,
    source_cols: int | None,
) -> tuple[torch.Tensor, torch.Size | None]:
    if x.ndim != 3:
        _validate_2d_128_aligned(x, "x")
        return x, None

    if producer_id != BLOCK_SCALED_PRODUCER_IDENTITY:
        raise NotImplementedError(
            "CuTe grouped 3D row/col quantization does not support producer mode"
        )
    if output_shape is not None or source_cols is not None:
        raise NotImplementedError(
            "CuTe grouped 3D row/col quantization does not support output_shape/source_cols"
        )
    _validate_row1d_128_aligned(x, "x")
    return x.view(-1, x.shape[-1]), x.shape


def _prepare_row_col_producers(
    x: torch.Tensor,
    producer_b: torch.Tensor | None,
    producer_c: torch.Tensor | None,
    producer_id: int,
) -> tuple[bool, torch.Tensor, torch.Tensor]:
    if producer_id not in (
        BLOCK_SCALED_PRODUCER_IDENTITY,
        BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
        BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY,
    ):
        raise NotImplementedError(
            f"Unsupported CuTe row/col quantization producer_id={producer_id}"
        )

    producer_requires_bc = producer_id in (
        BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
        BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY,
    )
    if not producer_requires_bc:
        if producer_b is not None or producer_c is not None:
            raise ValueError("producer_b/producer_c are only valid for producer mode")
        return False, x, x

    if producer_b is None:
        raise ValueError("producer_b is required for producer mode")
    if producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY and producer_c is None:
        _validate_packed_swiglu_bwd_dxy_producer(x, producer_b)
        return True, producer_b, producer_b

    _validate_producer_tensor(x, producer_b, "producer_b")

    if producer_id != BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY:
        return True, producer_b, producer_b

    if producer_c is None:
        raise ValueError("producer_c is required for SWIGLU_BWD_DXY producer mode")
    _validate_producer_tensor(x, producer_c, "producer_c")
    return True, producer_b, producer_c


def _prepare_row_col_shape_config(  # noqa: C901 -- host validation branches are format/mode-gated
    x: torch.Tensor,
    *,
    output_shape: tuple[int, ...] | None,
    source_cols: int | None,
    sf_vec: int,
    axis_mask: int,
    scale_reduction_id: int,
    is_zerocopy_gather: bool = False,
) -> _RowColShapeConfig:
    logical_shape = tuple(output_shape) if output_shape is not None else tuple(x.shape)
    if len(logical_shape) != 2:
        raise ValueError(
            f"output_shape rank must match x rank: output_shape={logical_shape}, x={tuple(x.shape)}"
        )
    if is_zerocopy_gather:
        if x.ndim != 1:
            raise ValueError(
                f"ZEROCOPY_GATHER expects 1-D gather_ptrs; got {tuple(x.shape)}"
            )
        if logical_shape[0] != x.shape[0]:
            raise ValueError(
                f"output_shape prefix must match gather_ptrs length: output_shape={logical_shape}, x={tuple(x.shape)}"
            )
        if logical_shape[0] % 128 != 0 or logical_shape[1] % 128 != 0:
            raise NotImplementedError(
                f"output_shape dimensions must be multiples of 128 for CuTe gather quantization: {logical_shape}"
            )
        source_cols = logical_shape[1] if source_cols is None else source_cols
        if source_cols != logical_shape[1]:
            raise ValueError(
                f"source_cols must match output_shape last dim for CuTe gather quantization: {source_cols} != {logical_shape[1]}"
            )
    else:
        if logical_shape[0] != x.shape[0]:
            raise ValueError(
                f"output_shape prefix must match x shape: output_shape={logical_shape}, x={tuple(x.shape)}"
            )
        if logical_shape[1] % 128 != 0:
            raise ValueError(
                f"output_shape last dimension must be a multiple of 128 for CuTe row/col quantization: {logical_shape}"
            )

        source_cols = x.shape[-1] if source_cols is None else source_cols
        if source_cols != x.shape[-1]:
            raise ValueError(
                f"source_cols must match x last dim for CuTe row/col quantization: {source_cols} != {x.shape[-1]}"
            )

    n_rows, n_cols = logical_shape
    is_two_d = scale_reduction_id == SCALE_REDUCTION_TWO_D
    has_col_axis = (axis_mask & AXIS_MASK_M) != 0
    has_row_axis = (axis_mask & AXIS_MASK_K) != 0
    if source_cols % sf_vec != 0:
        raise ValueError(f"source_cols {source_cols} not divisible by {sf_vec}")
    if (has_row_axis or is_two_d) and n_cols % sf_vec != 0:
        raise ValueError(f"row-quant dimension {n_cols} not divisible by {sf_vec}")
    if (has_col_axis or is_two_d) and n_rows % sf_vec != 0:
        raise ValueError(f"column-quant dimension {n_rows} not divisible by {sf_vec}")

    return _RowColShapeConfig(
        source_cols,
        n_rows,
        n_cols,
        has_col_axis,
        has_row_axis,
        is_two_d,
        has_col_axis and not has_row_axis and not is_two_d,
    )


def _prepare_row_col_word_views(
    *,
    x: torch.Tensor,
    q0: torch.Tensor,
    q1: torch.Tensor,
    producer_b_arg: torch.Tensor,
    producer_c_arg: torch.Tensor,
    producer_requires_bc: bool,
    producer_id: int,
    format_id: int,
    axis_mask: int,
    scale_reduction_id: int,
    row_col_kernel_mode: int,
    is_col1d: bool,
    source_cols: int,
    n_rows: int,
    n_cols: int,
) -> _RowColWordViews:
    dummy_words = torch.empty((1, 1), dtype=torch.uint32, device=x.device)
    producer_b_words_aligned = False
    producer_c_words_aligned = False
    producer_b_words = dummy_words
    producer_c_words = dummy_words

    if (
        producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY
        and row_col_kernel_mode != _ROW_COL_KERNEL_MODE_SWIGLU_BWD_DXY_FP8_TILE
        and scale_reduction_id == SCALE_REDUCTION_ONE_D
        and source_cols == x.shape[1]
    ):
        row_col_kernel_mode = _AXES_KERNEL_MODE_SWIGLU_BWD_DXY_F32X2_VECTOR

    col1d_use_row_vector = _use_row_vector_col1d(
        format_id=format_id,
        axis_mask=axis_mask,
        scale_reduction_id=scale_reduction_id,
    )
    uses_row_vector_loads = (
        not is_col1d
        or row_col_kernel_mode == _AXES_KERNEL_MODE_SWIGLU_BWD_DXY_F32X2_VECTOR
        or col1d_use_row_vector
    )

    use_producer_words = (
        producer_requires_bc
        and x.dtype in (torch.bfloat16, torch.float16)
        and uses_row_vector_loads
        and format_id != BLOCK_SCALED_FORMAT_MXFP4
        and row_col_kernel_mode != _ROW_COL_KERNEL_MODE_SWIGLU_BWD_DXY_FP8_TILE
    )
    if use_producer_words:
        producer_b_words_aligned = True
        producer_c_words_aligned = True
        producer_b_words = _view_f16_as_u32_rows(producer_b_arg)
        producer_c_words = _view_f16_as_u32_rows(producer_c_arg)

    if row_col_kernel_mode == _AXES_KERNEL_MODE_SCALAR:
        return _RowColWordViews(
            row_col_kernel_mode,
            False,
            False,
            dummy_words,
            dummy_words,
            dummy_words,
            dummy_words,
            dummy_words,
        )

    dummy_words = torch.empty((n_rows, 4), dtype=torch.uint32, device=x.device)
    use_x_words = x.dtype in (torch.bfloat16, torch.float16) and uses_row_vector_loads
    x_words = _view_f16_as_u32_rows(x) if use_x_words else dummy_words

    if not producer_b_words_aligned:
        producer_b_words = x_words
    if not producer_c_words_aligned:
        producer_c_words = x_words

    has_col_axis = (axis_mask & AXIS_MASK_M) != 0
    has_row_axis = (axis_mask & AXIS_MASK_K) != 0
    col_qdata_row_major = _use_row_major_col_qdata(
        format_id=format_id,
        scale_reduction_id=scale_reduction_id,
        has_col_axis=has_col_axis,
        has_row_axis=has_row_axis,
        col1d_use_row_vector=col1d_use_row_vector,
    )
    q0_word_rows = n_rows if col_qdata_row_major else n_cols
    q0_words = (
        q0.view(torch.uint8).view(torch.uint32).view(q0_word_rows, -1)
        if has_col_axis
        else torch.empty((n_cols, 4), dtype=torch.uint32, device=x.device)
    )
    q1_words = (
        _view_qdata_as_u32_rows(q1, n_rows)
        if has_row_axis
        or row_col_kernel_mode
        in (
            _AXES_KERNEL_MODE_TILE2D_FP8_VECTOR,
            _AXES_KERNEL_MODE_TILE2D_FP4_VECTOR,
        )
        else torch.empty((n_rows, 4), dtype=torch.uint32, device=x.device)
    )

    return _RowColWordViews(
        row_col_kernel_mode,
        producer_b_words_aligned,
        producer_c_words_aligned,
        producer_b_words,
        producer_c_words,
        x_words,
        q0_words,
        q1_words,
    )


# -----------------------------------------------------------------------------
# Select helpers
# -----------------------------------------------------------------------------


def _select_row_col_kernel_mode(
    *,
    scale_reduction_id: int,
    has_col_axis: bool,
    is_fp4: bool,
) -> int:
    if scale_reduction_id == SCALE_REDUCTION_TWO_D and not is_fp4:
        return _AXES_KERNEL_MODE_TILE2D_FP8_VECTOR
    if scale_reduction_id == SCALE_REDUCTION_TWO_D and is_fp4:
        return _AXES_KERNEL_MODE_TILE2D_FP4_VECTOR
    if scale_reduction_id == SCALE_REDUCTION_ONE_D and has_col_axis:
        return _AXES_KERNEL_MODE_BOTH1D_VECTOR
    return _AXES_KERNEL_MODE_SCALAR


# -----------------------------------------------------------------------------
# Try helpers
# -----------------------------------------------------------------------------


def _make_swiglu_fwd_quantize_result(
    *,
    do_col_quant: bool,
    do_row_quant: bool,
    col_q: torch.Tensor | None,
    col_scale: torch.Tensor | None,
    row_q: torch.Tensor | None,
    row_scale: torch.Tensor | None,
):
    if do_col_quant and (col_q is None or col_scale is None):
        raise AssertionError("SwiGLU fwd column output allocation invariant failed")
    if do_row_quant and (row_q is None or row_scale is None):
        raise AssertionError("SwiGLU fwd row output allocation invariant failed")
    if do_col_quant and do_row_quant:
        return (col_q, col_scale), (row_q, row_scale)
    if do_col_quant:
        return col_q, col_scale
    return row_q, row_scale


def _try_quantize_swiglu_fwd_fp8_tile(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    format_id: int,
    axis_mask: int,
    layout_id: int,
    half_range_scale: bool,
    fast_math: bool,
    clamped: bool,
    alpha: float,
    limit: float,
):
    if (
        format_id not in _MXFP8_FORMAT_IDS
        or layout_id != SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED
        or half_range_scale
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.ndim != 2
        or y.ndim != 2
    ):
        return None

    do_col_quant = (axis_mask & AXIS_MASK_M) != 0
    do_row_quant = (axis_mask & AXIS_MASK_K) != 0
    if not do_col_quant and not do_row_quant:
        return None

    _validate_2d_128_aligned(x, "x")
    _validate_producer_tensor(x, y, "producer_b")
    n_rows, n_cols = x.shape
    sf_vec = _get_sf_vec_size_for_format_id(format_id)
    scale_cols = n_cols // sf_vec
    row_blocks = n_rows // sf_vec
    sdtype = _get_scale_dtype_for_format_id(format_id)
    row_q = None
    row_scale = None
    row_q_words = None
    col_q = None
    col_scale = None
    col_q_words = None
    col_q_kernel = None

    if do_row_quant:
        row_q = _make_empty_quantized_qdata(
            (n_rows, n_cols),
            format_id=format_id,
            device=x.device,
            row_alignment_bytes=_FP8_QDATA_ROW_ALIGNMENT_BYTES,
        )
        row_q_words = _view_qdata_as_u32_rows(row_q, n_rows)
        row_scale = torch.empty((n_rows, scale_cols), dtype=sdtype, device=x.device)

    if do_col_quant:
        col_q_kernel = _make_empty_quantized_qdata(
            (n_rows, n_cols),
            format_id=format_id,
            device=x.device,
            row_alignment_bytes=_FP8_QDATA_ROW_ALIGNMENT_BYTES,
        )
        col_q = col_q_kernel.t()
        col_q_words = _view_qdata_as_u32_rows(col_q_kernel, n_rows)
        col_scale = torch.empty((n_cols, row_blocks), dtype=sdtype, device=x.device)

    if n_rows == 0:
        return _make_swiglu_fwd_quantize_result(
            do_col_quant=do_col_quant,
            do_row_quant=do_row_quant,
            col_q=col_q,
            col_scale=col_scale,
            row_q=row_q,
            row_scale=row_scale,
        )

    if row_q_words is None:
        row_q_words = col_q_words
        row_scale = col_scale
    if col_q_words is None:
        col_q_words = row_q_words
        col_scale = row_scale

    if (
        row_q_words is None
        or row_scale is None
        or col_q_words is None
        or col_scale is None
    ):
        raise AssertionError("SwiGLU fwd tile output allocation invariant failed")
    col_q_arg = col_q_kernel if col_q_kernel is not None else row_q
    row_q_arg = row_q if row_q is not None else col_q_kernel
    if col_q_arg is None or row_q_arg is None:
        raise AssertionError("SwiGLU fwd tile qdata argument invariant failed")

    dummy_words = torch.empty((n_rows, 4), dtype=torch.uint32, device=x.device)
    nvfp4_recip_lut = torch.empty(256, dtype=torch.float32, device=x.device)

    with ensure_cuda_driver_context():
        _QuantizeBlockScaledRowCol.compile(
            _TORCH_TO_CUTE_INPUT_DTYPE[x.dtype],
            format_id,
            BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
            axis_mask,
            SCALE_REDUCTION_ONE_D,
            layout_id,
            _ROW_COL_KERNEL_MODE_SWIGLU_FWD_FP8_TILE,
            0,
            half_range_scale,
            fast_math,
            clamped,
            alpha,
            limit,
            False,
            False,
        )(
            x,
            y,
            y,
            dummy_words,
            dummy_words,
            dummy_words,
            col_q_arg.view(torch.uint8),
            col_q_words,
            row_q_arg.view(torch.uint8),
            row_q_words,
            col_scale.view(torch.uint8).view(-1),
            row_scale.view(torch.uint8).view(-1),
            nvfp4_recip_lut,
            None,
            cutlass.Int64(0),
            cutlass.Int64(1),
            cutlass.Int32(n_cols),
            cutlass.Int32(0),
            cutlass.Int32(1),
        )

    return _make_swiglu_fwd_quantize_result(
        do_col_quant=do_col_quant,
        do_row_quant=do_row_quant,
        col_q=col_q,
        col_scale=col_scale,
        row_q=row_q,
        row_scale=row_scale,
    )


# -----------------------------------------------------------------------------
# Validate helpers
# -----------------------------------------------------------------------------


def _validate_zerocopy_rank_args(
    *,
    is_zerocopy_gather: bool,
    local_rank: int,
    world_size: int,
) -> None:
    if is_zerocopy_gather:
        if world_size <= 0:
            raise ValueError(f"world_size must be positive; got {world_size}")
        if local_rank < 0 or local_rank >= world_size:
            raise ValueError(
                f"local_rank must be in [0, world_size); got {local_rank=} {world_size=}"
            )
        return
    if local_rank != 0 or world_size != 1:
        raise ValueError("local_rank/world_size are only valid for ZEROCOPY_GATHER")


def _validate_2d_128_aligned(x: torch.Tensor, name: str) -> None:
    if x.ndim != 2:
        raise NotImplementedError(
            f"{name} must be 2D for CuTe quantization: {tuple(x.shape)}"
        )
    if x.shape[0] % 128 != 0 or x.shape[1] % 128 != 0:
        raise NotImplementedError(
            f"{name} dimensions must be multiples of 128 for CuTe quantization: {tuple(x.shape)}"
        )
    if x.stride(-1) != 1:
        raise NotImplementedError(
            f"{name} must have contiguous last dimension for CuTe quantization"
        )


def _validate_row1d_128_aligned(x: torch.Tensor, name: str) -> None:
    if x.ndim not in (2, 3):
        raise NotImplementedError(
            f"{name} must be 2D or contiguous 3D for CuTe quantization: {tuple(x.shape)}"
        )
    if x.shape[-2] % 128 != 0 or x.shape[-1] % 128 != 0:
        raise NotImplementedError(
            f"{name} row and last dimensions must be multiples of 128 for CuTe quantization: {tuple(x.shape)}"
        )
    if x.stride(-1) != 1:
        raise NotImplementedError(
            f"{name} must have contiguous last dimension for CuTe quantization"
        )
    if x.ndim == 3 and not x.is_contiguous():
        raise NotImplementedError(
            f"{name} must be contiguous to flatten grouped 3D CuTe quantization input"
        )


def _validate_row1d_partial_rows(x: torch.Tensor, name: str) -> None:
    if x.ndim != 2:
        raise NotImplementedError(
            f"{name} must be 2D for partial-row CuTe quantization: {tuple(x.shape)}"
        )
    if x.shape[0] <= 0 or x.shape[1] % 128 != 0:
        raise NotImplementedError(
            f"{name} must have positive rows and a last dimension divisible by 128 "
            f"for partial-row CuTe quantization: {tuple(x.shape)}"
        )
    if x.stride(-1) != 1:
        raise NotImplementedError(
            f"{name} must have contiguous rows for partial-row CuTe quantization"
        )


def _validate_grouped_blocked_row_col_shape(
    grouped_shape: torch.Size | None,
    *,
    layout_id: int,
    sf_vec: int,
) -> None:
    if grouped_shape is None or layout_id != SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED:
        return
    rows = grouped_shape[-2]
    cols = grouped_shape[-1]
    if (rows // sf_vec) % CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM != 0:
        raise NotImplementedError(
            "CuTe grouped 3D blocked row/col quantization requires each group's "
            "column-quant scale columns to align with 4-column CUBLAS_BLOCKED atoms"
        )
    if (cols // sf_vec) % CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM != 0:
        raise NotImplementedError(
            "CuTe grouped 3D blocked row/col quantization requires each group's "
            "row-quant scale columns to align with 4-column CUBLAS_BLOCKED atoms"
        )


def _validate_producer_tensor(
    x: torch.Tensor,
    producer: torch.Tensor,
    name: str,
) -> None:
    if producer.dtype != x.dtype:
        raise TypeError(
            f"{name} dtype must match x dtype: {producer.dtype} != {x.dtype}"
        )
    if producer.shape != x.shape:
        raise ValueError(
            f"{name} shape must match x shape: {producer.shape} != {x.shape}"
        )
    _validate_2d_128_aligned(producer, name)


def _validate_packed_swiglu_bwd_dxy_producer(
    x: torch.Tensor,
    producer_b: torch.Tensor,
) -> None:
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"Unsupported input dtype for packed-H1 SwiGLU DXY quantization: {x.dtype}"
        )
    if producer_b.dtype != x.dtype:
        raise TypeError(f"dz/h1 dtypes must match: dz={x.dtype}, h1={producer_b.dtype}")
    _validate_2d_128_aligned(x, "x")
    _validate_2d_128_aligned(producer_b, "producer_b")
    expected_shape = (x.shape[0], x.shape[1] * 2)
    if tuple(producer_b.shape) != expected_shape:
        raise ValueError(
            f"producer_b must have shape [M, 2N] for x [M, N]: x={x.shape}, producer_b={producer_b.shape}"
        )


# -----------------------------------------------------------------------------
# View helpers
# -----------------------------------------------------------------------------


def _view_qdata_as_u32_rows(qdata: torch.Tensor, n_rows: int) -> torch.Tensor:
    u32 = qdata.view(torch.uint8).view(torch.uint32)
    if u32.numel() == 0:
        # Preserve the real row pitch: reshaping an empty tile to (0, 0) gives
        # it a synthetic stride of 1, which violates the kernel's 16-byte row
        # alignment contract even when the underlying qdata is aligned.
        return u32.reshape(n_rows, u32.shape[-1])
    return u32.view(n_rows, -1)


def _view_f16_as_u32_rows(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype in (torch.bfloat16, torch.float16)
    assert x.stride(-1) == 1
    return x.view(torch.uint32).view(x.shape[0], -1)


def _view_row1d_input_as_2d(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 2 else x.view(-1, x.shape[-1])


def _view_grouped_col_qdata(
    qdata: torch.Tensor | None,
    grouped_shape: torch.Size,
) -> torch.Tensor | None:
    if qdata is None:
        return None
    groups, rows, _cols = grouped_shape
    # Column-quant qdata is produced as [K, G * rows] (or [K, G * rows / 2] for
    # packed FP4). Restore the grouped [G, K, rows] logical order.
    if qdata.dtype == torch.float4_e2m1fn_x2:
        qdata_u8 = qdata.view(torch.uint8)
        grouped_u8 = (
            qdata_u8.view(qdata_u8.shape[0], groups, -1).permute(1, 0, 2).contiguous()
        )
        return grouped_u8.view(torch.float4_e2m1fn_x2)
    grouped = qdata.view(qdata.shape[0], groups, -1).permute(1, 0, 2)
    if qdata.stride(0) == 1:
        return grouped
    return grouped.contiguous()


def _view_grouped_row_col_result(
    result,
    grouped_shape: torch.Size,
    *,
    layout_id: int,
    sf_vec: int,
    has_col_axis: bool,
    has_row_axis: bool,
    is_two_d: bool,
):
    groups, rows, _cols = grouped_shape

    def col_scale_view(scale):
        if layout_id == SCALE_FACTOR_LAYOUT_NATURAL:
            return scale.view(groups, rows // sf_vec, scale.shape[-1])
        return scale

    def row_scale_view(scale):
        if layout_id == SCALE_FACTOR_LAYOUT_NATURAL:
            return scale.view(groups, rows, -1)
        return scale

    def col_result_view(axis_result):
        qdata, scale = axis_result
        return (
            _view_grouped_col_qdata(qdata, grouped_shape),
            col_scale_view(scale),
        )

    def row_result_view(axis_result):
        qdata, scale = axis_result
        return (
            qdata.view(groups, rows, qdata.shape[-1]),
            row_scale_view(scale),
        )

    if is_two_d:
        col_result, row_result = result
        return col_result_view(col_result), row_result_view(row_result)
    if has_col_axis and has_row_axis:
        col_result, row_result = result
        return col_result_view(col_result), row_result_view(row_result)
    if has_col_axis:
        return col_result_view(result)
    return row_result_view(result)


# Public Host Functions
# =============================================================================


def _prepare_dynamic_nvfp4_q_output(
    x: torch.Tensor,
    rows: int,
    K: int,
    q_out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_shape = (rows, K // 2)
    if q_out is None:
        q_bytes = torch.empty(q_shape, dtype=torch.uint8, device=x.device)
        return q_bytes.view(torch.float4_e2m1fn_x2), q_bytes
    if q_out.shape != q_shape or q_out.dtype != torch.float4_e2m1fn_x2:
        raise ValueError(
            f"q_out must have shape={q_shape}, dtype={torch.float4_e2m1fn_x2}; "
            f"got shape={tuple(q_out.shape)}, dtype={q_out.dtype}"
        )
    if (
        q_out.device != x.device
        or q_out.stride(-1) != 1
        or q_out.stride(0) < q_shape[1]
        or q_out.stride(0) % 16 != 0
        or q_out.data_ptr() % 16 != 0
    ):
        raise ValueError(
            "q_out must be on x.device with non-overlapping, 16-byte-aligned rows"
        )
    return q_out, q_out.view(torch.uint8)


def _prepare_dynamic_nvfp4_scale_output(
    x: torch.Tensor,
    rows: int,
    K: int,
    layout: ScaleFactorLayout,
    scale_out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_rows = (
        rows if layout == ScaleFactorLayout.NATURAL else (rows + 127) // 128 * 128
    )
    scale_shape = (scale_rows, K // 16)
    if scale_out is None:
        # Consumers index only logical rows; blocked-layout padding is unspecified.
        scale_bytes = torch.empty(
            scale_shape,
            dtype=torch.uint8,
            device=x.device,
        )
        return scale_bytes.view(torch.float8_e4m3fn), scale_bytes
    if scale_out.shape != scale_shape or scale_out.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"scale_out must have shape={scale_shape}, dtype={torch.float8_e4m3fn}; "
            f"got shape={tuple(scale_out.shape)}, dtype={scale_out.dtype}"
        )
    # Scale stores are byte-addressed; unlike qdata's uint32 vector stores,
    # they require no stronger row or base alignment.
    if (
        scale_out.device != x.device
        or scale_out.stride(-1) != 1
        or scale_out.stride(0) < scale_shape[1]
    ):
        raise ValueError("scale_out must be on x.device with non-overlapping rows")
    if layout == ScaleFactorLayout.CUBLAS_BLOCKED and not scale_out.is_contiguous():
        raise ValueError("blocked scale_out must be contiguous")
    return scale_out, scale_out.view(torch.uint8)


def _dynamic_nvfp4_scale_storage(scale_bytes: torch.Tensor) -> torch.Tensor:
    storage_elems = (scale_bytes.shape[0] - 1) * scale_bytes.stride(
        0
    ) + scale_bytes.shape[1]
    return scale_bytes.as_strided((storage_elems,), (1,))


def _quantize_nvfp4_per_token(  # noqa: C901 -- producer-specific host validation
    x: torch.Tensor,
    *,
    layout: ScaleFactorLayout,
    producer_b: torch.Tensor | None = None,
    producer_id: int = BLOCK_SCALED_PRODUCER_IDENTITY,
    fast_math: bool = False,
    clamped: bool = False,
    alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    q_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    token_scale_inv_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize each activation row with a fused dynamic NVFP4 global scale."""
    clamped, alpha, limit = canonical_swiglu_clamp(clamped, alpha, limit)
    is_swiglu = producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD
    if producer_id not in (
        BLOCK_SCALED_PRODUCER_IDENTITY,
        BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
    ):
        raise NotImplementedError(
            f"dynamic per-token NVFP4 producer_id={producer_id} is not supported"
        )
    if x.ndim != 2 or x.stride(-1) != 1:
        raise ValueError(f"x must be a 2D row-major tensor, got shape={tuple(x.shape)}")
    if not is_swiglu and not x.is_contiguous():
        raise ValueError("identity dynamic per-token NVFP4 input must be contiguous")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"x must be bfloat16 or float16, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    rows, K = x.shape
    _QuantizeNvfp4PerTokenRow._validate_producer_k(K, producer_id)
    if is_swiglu:
        if producer_b is None:
            raise ValueError("SwiGLU dynamic per-token NVFP4 requires producer_b")
        if producer_b.shape != x.shape:
            raise ValueError(
                f"x and producer_b must have the same shape: {x.shape} != {producer_b.shape}"
            )
        if producer_b.dtype != x.dtype:
            raise TypeError(
                f"x and producer_b dtypes must match: {x.dtype} != {producer_b.dtype}"
            )
        if not producer_b.is_cuda or producer_b.device != x.device:
            raise ValueError("x and producer_b must be on the same CUDA device")
        if producer_b.stride(-1) != 1:
            raise ValueError("producer_b must be row-major")
        for name, tensor in (("x", x), ("producer_b", producer_b)):
            row_stride_bytes = tensor.stride(0) * tensor.element_size()
            if tensor.data_ptr() % 8 != 0 or row_stride_bytes % 8 != 0:
                raise ValueError(
                    f"SwiGLU dynamic per-token NVFP4 requires 8-byte-aligned "
                    f"{name} row starts"
                )
    else:
        producer_b = x

    q, q_bytes = _prepare_dynamic_nvfp4_q_output(x, rows, K, q_out)
    scale, scale_bytes = _prepare_dynamic_nvfp4_scale_output(
        x,
        rows,
        K,
        layout,
        scale_out,
    )
    if token_scale_inv_out is not None:
        if (
            token_scale_inv_out.dtype != torch.float32
            or token_scale_inv_out.shape != (rows,)
            or not token_scale_inv_out.is_cuda
            or token_scale_inv_out.device != x.device
        ):
            raise ValueError(
                "token_scale_inv_out must be a float32 CUDA tensor of shape "
                f"({rows},) on {x.device}; got shape="
                f"{tuple(token_scale_inv_out.shape)}, "
                f"dtype={token_scale_inv_out.dtype}, "
                f"device={token_scale_inv_out.device}"
            )
        if token_scale_inv_out.data_ptr() % 4 != 0:
            raise ValueError("token_scale_inv_out must be 4-byte aligned")
        if token_scale_inv_out.stride(0) <= 0:
            raise ValueError("token_scale_inv_out must not have overlapping elements")
        token_scale_inv = token_scale_inv_out
    else:
        token_scale_inv = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if rows == 0:
        return q, scale, token_scale_inv

    scale_storage = _dynamic_nvfp4_scale_storage(scale_bytes)

    with ensure_cuda_driver_context():
        _QuantizeNvfp4PerTokenRow.compile(
            _TORCH_TO_CUTE_INPUT_DTYPE[x.dtype],
            K,
            SCALE_FACTOR_LAYOUT_IDS[layout],
            producer_id,
            fast_math,
            clamped,
            alpha,
            limit,
        )(
            x,
            producer_b,
            q_bytes.view(torch.uint32) if is_swiglu else x.view(torch.uint32),
            q_bytes.view(torch.uint32),
            scale_storage,
            cutlass.Int64(scale_bytes.stride(0)),
            token_scale_inv,
            get_nvfp4_recip_lut(x.device),
        )
    return q, scale, token_scale_inv


def quantize_block_scaled(  # noqa: C901 -- host validation/launch setup is format-gated
    x: torch.Tensor,
    *,
    format: BlockScaledFormat,
    layout: ScaleFactorLayout,
    half_range_scale: bool = False,
    fast_math: bool = False,
    clamped: bool = False,
    alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    producer_b: torch.Tensor | None = None,
    producer_id: int = BLOCK_SCALED_PRODUCER_IDENTITY,
    output_shape: tuple[int, ...] | None = None,
    gather_dtype: torch.dtype | None = None,
    local_rank: int = 0,
    world_size: int = 1,
    q_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    allow_partial_rows: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    clamped, alpha, limit = canonical_swiglu_clamp(clamped, alpha, limit)
    is_zerocopy_gather = producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
    _validate_zerocopy_rank_args(
        is_zerocopy_gather=is_zerocopy_gather,
        local_rank=local_rank,
        world_size=world_size,
    )
    if producer_id not in (
        BLOCK_SCALED_PRODUCER_IDENTITY,
        BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
        BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER,
    ):
        raise NotImplementedError(
            f"Unsupported CuTe quantization producer_id={producer_id}"
        )

    if is_zerocopy_gather:
        if x.dtype != torch.int64:
            raise TypeError(
                f"ZEROCOPY_GATHER x must be an int64 gather pointer table; got {x.dtype}"
            )
        if x.ndim != 1 or not x.is_contiguous():
            raise NotImplementedError(
                f"ZEROCOPY_GATHER expects contiguous 1D gather_ptrs; got {tuple(x.shape)}"
            )
        if gather_dtype not in _TORCH_TO_CUTE_INPUT_DTYPE:
            raise TypeError(
                f"ZEROCOPY_GATHER gather_dtype must be bf16/fp16/fp32; got {gather_dtype}"
            )
        if output_shape is None or len(output_shape) != 2:
            raise ValueError(
                f"ZEROCOPY_GATHER output_shape must be rank-2; got {output_shape}"
            )
        if output_shape[0] != x.shape[0]:
            raise ValueError(
                f"ZEROCOPY_GATHER output rows must match gather_ptrs length: {output_shape[0]} != {x.shape[0]}"
            )
        if output_shape[0] % 128 != 0 or output_shape[1] % 128 != 0:
            raise NotImplementedError(
                f"output_shape dimensions must be multiples of 128 for CuTe gather quantization: {output_shape}"
            )
        if producer_b is not None:
            raise ValueError("producer_b is not valid for ZEROCOPY_GATHER")
        input_torch_dtype = gather_dtype
        q_shape = tuple(output_shape)
    else:
        if output_shape is not None:
            raise ValueError("output_shape is only valid for ZEROCOPY_GATHER")
        if gather_dtype is not None:
            raise ValueError("gather_dtype is only valid for ZEROCOPY_GATHER")
        if x.dtype not in _TORCH_TO_CUTE_INPUT_DTYPE:
            raise TypeError(f"Unsupported input dtype for CuTe quantization: {x.dtype}")
        input_torch_dtype = x.dtype
        q_shape = tuple(x.shape)

    if producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD:
        if producer_b is None:
            raise ValueError("producer_b is required for SWIGLU_FWD producer mode")
        if producer_b.dtype != x.dtype:
            raise TypeError(
                f"producer_b dtype must match x dtype: {producer_b.dtype} != {x.dtype}"
            )
        if producer_b.shape != x.shape:
            raise ValueError(
                f"producer_b shape must match x shape: {producer_b.shape} != {x.shape}"
            )
    elif producer_b is not None:
        raise ValueError("producer_b is only valid for SWIGLU_FWD producer mode")
    if allow_partial_rows:
        if is_zerocopy_gather or producer_id != BLOCK_SCALED_PRODUCER_IDENTITY:
            raise ValueError(
                "allow_partial_rows only supports identity row quantization"
            )
        if layout != ScaleFactorLayout.NATURAL:
            raise ValueError("allow_partial_rows requires NATURAL scale layout")
        _validate_row1d_partial_rows(x, "x")
    elif not is_zerocopy_gather:
        _validate_row1d_128_aligned(x, "x")
    if producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD:
        assert producer_b is not None
        _validate_row1d_128_aligned(producer_b, "producer_b")

    q_dtype, scale_dtype, sf_vec_size = block_scaled_format_constants(format)
    K = q_shape[-1]
    if format in (BlockScaledFormat.NVFP4, BlockScaledFormat.MXFP4):
        if K % 2 != 0:
            raise ValueError(f"FP4 output requires an even K dimension, got {K}")
    if is_zerocopy_gather:
        x_2d = x
        assert gather_dtype is not None
        producer_b_2d = torch.empty((1, 1), dtype=gather_dtype, device=x.device)
    else:
        x_2d = _view_row1d_input_as_2d(x)
        producer_b_2d = (
            producer_b if producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD else x
        )
        producer_b_2d = _view_row1d_input_as_2d(producer_b_2d)
    rows = q_shape[0] if is_zerocopy_gather else x_2d.shape[0]
    scale_cols = K // sf_vec_size

    if producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD and x.ndim == 2:
        specialized = _try_quantize_swiglu_fwd_fp8_tile(
            x_2d,
            producer_b_2d,
            format_id=BLOCK_SCALED_FORMAT_IDS[format],
            axis_mask=AXIS_MASK_K,
            layout_id=SCALE_FACTOR_LAYOUT_IDS[layout],
            half_range_scale=half_range_scale,
            fast_math=fast_math,
            clamped=clamped,
            alpha=alpha,
            limit=limit,
        )
        if specialized is not None:
            return specialized

    if q_out is not None:
        q_storage_shape = (
            (*q_shape[:-1], K // 2)
            if format in (BlockScaledFormat.NVFP4, BlockScaledFormat.MXFP4)
            else q_shape
        )
        if q_out.shape != q_storage_shape or q_out.dtype != q_dtype:
            raise ValueError(
                f"q_out must have shape={q_storage_shape}, dtype={q_dtype}; "
                f"got shape={tuple(q_out.shape)}, dtype={q_out.dtype}"
            )
        if q_out.device != x.device or q_out.stride(-1) != 1:
            raise ValueError("q_out must be on x.device with contiguous rows")
        q = q_out
        q_byte = q.view(torch.uint8)
    elif q_dtype == torch.float4_e2m1fn_x2:
        q_byte = torch.empty(
            (*q_shape[:-1], K // 2),
            dtype=torch.uint8,
            device=x.device,
        )
        q = q_byte.view(torch.float4_e2m1fn_x2)
    else:
        q = torch.empty(*q_shape, dtype=q_dtype, device=x.device)
        q_byte = q.view(torch.uint8)
    scale_shape = _make_row1d_scale_shape(q_shape, scale_cols, layout)
    if scale_out is not None:
        if scale_out.shape != scale_shape or scale_out.dtype != scale_dtype:
            raise ValueError(
                f"scale_out must have shape={scale_shape}, dtype={scale_dtype}; "
                f"got shape={tuple(scale_out.shape)}, dtype={scale_out.dtype}"
            )
        if scale_out.device != x.device or scale_out.stride(-1) != 1:
            raise ValueError("scale_out must be on x.device with contiguous rows")
        scale = scale_out
    else:
        scale = torch.empty(scale_shape, dtype=scale_dtype, device=x.device)

    x_words_aligned = (not is_zerocopy_gather) and x.dtype in (
        torch.bfloat16,
        torch.float16,
    )
    x_words = (
        _view_f16_as_u32_rows(x_2d)
        if x_words_aligned
        else torch.empty((rows, 4), dtype=torch.uint32, device=x.device)
    )
    producer_b_words_aligned = (
        producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD
        and producer_b_2d.dtype in (torch.bfloat16, torch.float16)
    )
    if producer_id == BLOCK_SCALED_PRODUCER_IDENTITY:
        producer_b_words = x_words
    elif producer_b_words_aligned:
        producer_b_words = _view_f16_as_u32_rows(producer_b_2d)
    else:
        producer_b_words = torch.empty((rows, 4), dtype=torch.uint32, device=x.device)
    q_bytes = q_byte if q_byte.ndim == 2 else q_byte.view(rows, q_byte.shape[-1])
    q_words = q_bytes.view(torch.uint32)
    if layout == ScaleFactorLayout.NATURAL:
        scale_arg = (
            scale.view(torch.uint8)
            if scale.ndim == 2
            else scale.view(rows, scale_cols).view(torch.uint8)
        )
    else:
        scale_arg = scale.view(torch.uint8).view(-1)
    with ensure_cuda_driver_context():
        _QuantizeBlockScaledRow.compile(
            _TORCH_TO_CUTE_INPUT_DTYPE[input_torch_dtype],
            BLOCK_SCALED_FORMAT_IDS[format],
            SCALE_FACTOR_LAYOUT_IDS[layout],
            producer_id,
            half_range_scale,
            fast_math,
            clamped,
            alpha,
            limit,
            x_words_aligned,
            producer_b_words_aligned,
        )(
            x_2d,
            producer_b_2d,
            producer_b_words,
            x_words,
            q_bytes,
            q_words,
            scale_arg,
            cutlass.Int32(K),
            cutlass.Int32(local_rank),
            cutlass.Int32(world_size),
        )
    return q, scale


def quantize_block_scaled_axes(  # noqa: C901 -- host launch setup is format/mode-gated
    x: torch.Tensor,
    *,
    format_id: int,
    axis_mask: int,
    scale_reduction_id: int,
    layout_id: int = SCALE_FACTOR_LAYOUT_NATURAL,
    half_range_scale: bool = False,
    fast_math: bool = False,
    clamped: bool = False,
    alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    producer_b: torch.Tensor | None = None,
    producer_c: torch.Tensor | None = None,
    producer_id: int = BLOCK_SCALED_PRODUCER_IDENTITY,
    output_shape: tuple[int, ...] | None = None,
    source_cols: int | None = None,
    gather_dtype: torch.dtype | None = None,
    local_rank: int = 0,
    world_size: int = 1,
):
    """Quantize x with column / row / 2D-tile scale reduction (CuTeDSL)."""
    clamped, alpha, limit = canonical_swiglu_clamp(clamped, alpha, limit)
    is_zerocopy_gather = producer_id == BLOCK_SCALED_PRODUCER_ZEROCOPY_GATHER
    packed_swiglu_bwd_dxy = (
        producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_BWD_DXY and producer_c is None
    )
    _validate_zerocopy_rank_args(
        is_zerocopy_gather=is_zerocopy_gather,
        local_rank=local_rank,
        world_size=world_size,
    )
    grouped_shape = None
    if is_zerocopy_gather:
        if x.dtype != torch.int64:
            raise TypeError(
                f"ZEROCOPY_GATHER x must be an int64 gather pointer table; got {x.dtype}"
            )
        if gather_dtype not in _TORCH_TO_CUTE_INPUT_DTYPE:
            raise TypeError(
                f"ZEROCOPY_GATHER gather_dtype must be bf16/fp16/fp32; got {gather_dtype}"
            )
        if producer_b is not None or producer_c is not None:
            raise ValueError("producer_b/producer_c are not valid for ZEROCOPY_GATHER")
        input_dtype = _TORCH_TO_CUTE_INPUT_DTYPE[gather_dtype]
        producer_requires_bc = False
        producer_b_arg = torch.empty((1, 1), dtype=gather_dtype, device=x.device)
        producer_c_arg = producer_b_arg
    else:
        if gather_dtype is not None:
            raise ValueError(
                "gather_dtype is only valid when producer_id=ZEROCOPY_GATHER"
            )
        if x.dtype not in _TORCH_TO_CUTE_INPUT_DTYPE:
            raise TypeError(
                f"Unsupported input dtype for CuTe row/col quantization: {x.dtype}"
            )
        x, grouped_shape = _prepare_row_col_input_view(
            x,
            producer_id=producer_id,
            output_shape=output_shape,
            source_cols=source_cols,
        )
        if packed_swiglu_bwd_dxy and producer_b is not None and output_shape is None:
            output_shape = tuple(producer_b.shape)
        input_dtype = _TORCH_TO_CUTE_INPUT_DTYPE[x.dtype]
        producer_requires_bc, producer_b_arg, producer_c_arg = (
            _prepare_row_col_producers(
                x,
                producer_b,
                producer_c,
                producer_id,
            )
        )
    if layout_id not in _SCALE_FACTOR_LAYOUT_BY_ID:
        raise ValueError(
            f"Unsupported scale layout id for row/col quantization: {layout_id}"
        )
    if format_id not in _BLOCK_SCALED_FORMAT_BY_ID:
        raise ValueError(f"Unsupported format_id={format_id}")
    if (
        scale_reduction_id == SCALE_REDUCTION_ONE_D
        and axis_mask == AXIS_MASK_K
        and producer_id == BLOCK_SCALED_PRODUCER_IDENTITY
        and grouped_shape is None
    ):
        return quantize_block_scaled(
            x,
            format=_BLOCK_SCALED_FORMAT_BY_ID[format_id],
            layout=_SCALE_FACTOR_LAYOUT_BY_ID[layout_id],
            half_range_scale=half_range_scale,
            fast_math=fast_math,
            clamped=clamped,
            alpha=alpha,
            limit=limit,
        )

    sf_vec = _get_sf_vec_size_for_format_id(format_id)
    _validate_grouped_blocked_row_col_shape(
        grouped_shape,
        layout_id=layout_id,
        sf_vec=sf_vec,
    )
    shape_config = _prepare_row_col_shape_config(
        x,
        output_shape=output_shape,
        source_cols=source_cols,
        sf_vec=sf_vec,
        axis_mask=axis_mask,
        scale_reduction_id=scale_reduction_id,
        is_zerocopy_gather=is_zerocopy_gather,
    )
    if packed_swiglu_bwd_dxy:
        if format_id not in _MXFP8_FORMAT_IDS:
            raise NotImplementedError(
                "CuTe packed-H1 SwiGLU DXY quantization supports MXFP8 only"
            )
        if layout_id != SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED:
            raise NotImplementedError(
                "CuTe packed-H1 SwiGLU DXY quantization supports CUBLAS_BLOCKED scales only"
            )
        if half_range_scale:
            raise NotImplementedError(
                "CuTe packed-H1 SwiGLU DXY quantization does not support half-range scales"
            )
        if (
            axis_mask != (AXIS_MASK_M | AXIS_MASK_K)
            or scale_reduction_id != SCALE_REDUCTION_ONE_D
        ):
            raise NotImplementedError(
                "CuTe packed-H1 SwiGLU DXY quantization supports both 1D axes only"
            )
        if (shape_config.n_rows, shape_config.n_cols) != tuple(producer_b_arg.shape):
            raise ValueError(
                f"output_shape must match packed H1 shape: output_shape={(shape_config.n_rows, shape_config.n_cols)}, producer_b={tuple(producer_b_arg.shape)}"
            )
    if (
        not is_zerocopy_gather
        and grouped_shape is None
        and producer_id == BLOCK_SCALED_PRODUCER_SWIGLU_FWD
        and scale_reduction_id == SCALE_REDUCTION_ONE_D
    ):
        specialized = _try_quantize_swiglu_fwd_fp8_tile(
            x,
            producer_b_arg,
            format_id=format_id,
            axis_mask=axis_mask,
            layout_id=layout_id,
            half_range_scale=half_range_scale,
            fast_math=fast_math,
            clamped=clamped,
            alpha=alpha,
            limit=limit,
        )
        if specialized is not None:
            return specialized
    source_cols = shape_config.source_cols
    n_rows = shape_config.n_rows
    n_cols = shape_config.n_cols
    is_blocked = layout_id == SCALE_FACTOR_LAYOUT_CUBLAS_BLOCKED
    grouped_rows_per_group = 0 if grouped_shape is None else grouped_shape[-2]

    is_fp4 = format_id in _FP4_FORMAT_IDS
    sdtype = _get_scale_dtype_for_format_id(format_id)

    col_qdata_row_major = _use_row_major_col_qdata(
        format_id=format_id,
        scale_reduction_id=scale_reduction_id,
        has_col_axis=shape_config.has_col_axis,
        has_row_axis=shape_config.has_row_axis,
        col1d_use_row_vector=_use_row_vector_col1d(
            format_id=format_id,
            axis_mask=axis_mask,
            scale_reduction_id=scale_reduction_id,
        ),
    )
    q0_kernel = _make_empty_quantized_qdata(
        (n_rows, n_cols) if col_qdata_row_major else (n_cols, n_rows),
        format_id=format_id,
        device=x.device,
    )
    q0 = q0_kernel.t() if col_qdata_row_major else q0_kernel
    q1 = _make_empty_quantized_qdata(
        (n_rows, n_cols), format_id=format_id, device=x.device
    )
    if is_blocked and grouped_shape is not None:
        s0_shape = (grouped_shape[0] * n_cols, grouped_shape[1] // sf_vec)
    else:
        s0_shape = (
            (n_cols, n_rows // sf_vec) if is_blocked else (n_rows // sf_vec, n_cols)
        )
    s0 = torch.empty(s0_shape, dtype=sdtype, device=x.device)
    s1 = torch.empty((n_rows, n_cols // sf_vec), dtype=sdtype, device=x.device)

    row_col_kernel_mode = _select_row_col_kernel_mode(
        scale_reduction_id=scale_reduction_id,
        has_col_axis=shape_config.has_col_axis,
        is_fp4=is_fp4,
    )
    if packed_swiglu_bwd_dxy:
        row_col_kernel_mode = _ROW_COL_KERNEL_MODE_SWIGLU_BWD_DXY_FP8_TILE
    nvfp4_recip_lut = (
        get_nvfp4_recip_lut(x.device)
        if format_id == BLOCK_SCALED_FORMAT_NVFP4
        else torch.empty(256, dtype=torch.float32, device=x.device)
    )
    word_views = _prepare_row_col_word_views(
        x=x,
        q0=q0_kernel,
        q1=q1,
        producer_b_arg=producer_b_arg,
        producer_c_arg=producer_c_arg,
        producer_requires_bc=producer_requires_bc,
        producer_id=producer_id,
        format_id=format_id,
        axis_mask=axis_mask,
        scale_reduction_id=scale_reduction_id,
        row_col_kernel_mode=row_col_kernel_mode,
        is_col1d=shape_config.is_col1d,
        source_cols=source_cols,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    row_col_kernel_mode = word_views.row_col_kernel_mode
    with ensure_cuda_driver_context():
        _QuantizeBlockScaledRowCol.compile(
            input_dtype,
            format_id,
            producer_id,
            axis_mask,
            scale_reduction_id,
            layout_id,
            row_col_kernel_mode,
            grouped_rows_per_group,
            half_range_scale,
            fast_math,
            clamped,
            alpha,
            limit,
            word_views.producer_b_words_aligned,
            word_views.producer_c_words_aligned,
        )(
            x,
            producer_b_arg,
            producer_c_arg,
            word_views.producer_b_words,
            word_views.producer_c_words,
            word_views.x_words,
            q0_kernel.view(torch.uint8),
            word_views.q0_words,
            q1.view(torch.uint8),
            word_views.q1_words,
            s0.view(torch.uint8).view(-1),
            s1.view(torch.uint8).view(-1),
            nvfp4_recip_lut,
            cutlass.Int32(source_cols),
            cutlass.Int32(local_rank),
            cutlass.Int32(world_size),
        )

    if shape_config.is_two_d:
        q0_ret = q0 if is_fp4 else None
        result = (q0_ret, s0), (q1, s1)
    elif shape_config.has_col_axis and shape_config.has_row_axis:
        result = (q0, s0), (q1, s1)
    elif shape_config.has_col_axis:
        result = q0, s0
    else:
        result = q1, s1

    if grouped_shape is not None:
        return _view_grouped_row_col_result(
            result,
            grouped_shape,
            layout_id=layout_id,
            sf_vec=sf_vec,
            has_col_axis=shape_config.has_col_axis,
            has_row_axis=shape_config.has_row_axis,
            is_two_d=shape_config.is_two_d,
        )
    return result
