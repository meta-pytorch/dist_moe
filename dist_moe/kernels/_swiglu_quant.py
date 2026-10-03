# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused SwiGLU producer + FP8/NVFP4 tile quantization helpers.

Shared by the standalone block-scaled quant kernels and the DistMoE fused
grouped-GEMM epilogues; builds on the generic tile quantizer in ``common.py``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8, Uint32, Uint64

from ..formats import (
    BLOCK_SCALED_FORMAT_IDS,
    CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE,
    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
    HALF_RANGE_SCALE_DEFAULT_FORMATS,
)
from ._quant_conversion import (
    _abs_f32,
    _ceil_div_i32,
    _FP32_ZERO,
    _pack_b16x2,
    _unpack_b16x2,
)
from .blockscaled_quantization_common import (
    _apply_block_scaled_quant_dxy_fwd_producer_f32x2,
    _apply_block_scaled_quant_dxy_producer_f32x2,
    _apply_block_scaled_quant_producer,
    _compute_block_amax_nonfinite,
    _cublas_blockscaled_qscale_offset,
    _fmax_nan,
    _load_tensor_row_as_b16x2,
    _quantize_fp8_tile_values,
    BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
)

# Format IDs whose default scale rule is the half-range shift. Derived from the
# interface's policy tuple so the fused in-kernel SwiGLU quant below cannot drift
# from what ``quantize_for_format`` resolves for the same format -- when it did,
# fused and composed MXFP4 diverged bitwise. Formats that do not default to the
# shift (MXFP8, NVFP4) resolve False here, so unrelated callers of these helpers
# (optimizer, router epilogue) are unaffected.
_HALF_RANGE_DEFAULT_FORMAT_IDS: tuple[int, ...] = tuple(
    BLOCK_SCALED_FORMAT_IDS[_fmt] for _fmt in HALF_RANGE_SCALE_DEFAULT_FORMATS
)


# -----------------------------------------------------------------------------
# SwiGLU lane and scale layout helpers.
# -----------------------------------------------------------------------------


@cute.jit
def _swiglu_fp8_tile_lane_coords(
    warp_lane: Int32,
    col_lanes_cfg: cutlass.Constexpr[int],
    col_blocks_per_scale: cutlass.Constexpr[int],
    num_elems: cutlass.Constexpr[int],
    col_block: Int32,
    sf_vec_size: cutlass.Constexpr[int],
    qdata_elems_per_word: cutlass.Constexpr[int],
):
    col_lanes = Int32(col_lanes_cfg)
    row_lane = warp_lane // col_lanes
    col_lane_block = warp_lane - row_lane * col_lanes
    col_lane_pair = col_lane_block // Int32(col_blocks_per_scale)
    col_start = col_block * Int32(sf_vec_size) + col_lane_block * Int32(num_elems)
    q_word = col_start // Int32(qdata_elems_per_word)
    return row_lane, col_lane_block, col_lane_pair, col_start, q_word


@cute.jit
def _swiglu_fp8_row_scale_offset_base(
    row_start: Int32,
    row_lane: Int32,
    col_block: Int32,
    col_lane_pair: Int32,
    col_lane_block: Int32,
    col_blocks_per_scale: cutlass.Constexpr[int],
    scale_cols: Int32,
    scale_col_offset: Int32,
):
    row_scale_group_first_col = col_lane_pair * Int32(col_blocks_per_scale)
    row_scale_lane = (col_lane_block - row_scale_group_first_col) == Int32(0)
    row_scale_offset_base = Int32(0)
    if row_scale_lane:
        row_scale_n_col_blocks = _ceil_div_i32(
            scale_cols,
            Int32(CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM),
        )
        row_scale_col = scale_col_offset + col_block + col_lane_pair
        row_scale_offset_base = _cublas_blockscaled_qscale_offset(
            row_start + row_lane,
            row_scale_col,
            row_scale_n_col_blocks,
        )
    return row_scale_offset_base, row_scale_lane


@cute.jit
def _swiglu_fp8_col_scale_offset_base(
    row_lane: Int32,
    col_start: Int32,
    row_block: Int32,
    row_blocks: Int32,
) -> Int32:
    col_scale_offset_base = Int32(0)
    if row_lane == Int32(0):
        col_scale_n_col_blocks = _ceil_div_i32(
            row_blocks,
            Int32(CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM),
        )
        col_scale_offset_base = _cublas_blockscaled_qscale_offset(
            col_start,
            row_block,
            col_scale_n_col_blocks,
        )
    return col_scale_offset_base


# -----------------------------------------------------------------------------
# SwiGLU load helpers.
# -----------------------------------------------------------------------------


@cute.jit
def _load_swiglu_fwd_nvfp4_block(
    mX: cute.Tensor,
    mProducerB: cute.Tensor,
    row: Int32,
    k_base: Int32,
    active,
    packed: cute.Tensor,
    rep: cutlass.Constexpr[int],
    source_dtype: type[cutlass.Numeric],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
) -> None:
    x_packed = cute.make_rmem_tensor(8, Int32)
    producer_b_packed = cute.make_rmem_tensor(8, Int32)
    x_packed.fill(Int32(0))
    producer_b_packed.fill(Int32(0))
    if active:
        for chunk in cutlass.range_constexpr(4):
            x_chunk = cute.make_rmem_tensor(2, Int32)
            producer_b_chunk = cute.make_rmem_tensor(2, Int32)
            _load_tensor_row_as_b16x2(
                mX,
                row,
                k_base + Int32(chunk * 4),
                x_chunk,
                4,
                source_dtype,
            )
            _load_tensor_row_as_b16x2(
                mProducerB,
                row,
                k_base + Int32(chunk * 4),
                producer_b_chunk,
                4,
                source_dtype,
            )
            for pair in cutlass.range_constexpr(2):
                x_packed[chunk * 2 + pair] = x_chunk[pair]
                producer_b_packed[chunk * 2 + pair] = producer_b_chunk[pair]
    for pair in cutlass.range_constexpr(8):
        x0, x1 = _unpack_b16x2(x_packed[pair], source_dtype)
        y0, y1 = _unpack_b16x2(producer_b_packed[pair], source_dtype)
        z0 = _apply_block_scaled_quant_producer(
            Float32(x0),
            Float32(y0),
            BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
            fast_math,
            clamped,
            alpha,
            limit,
        )
        z1 = _apply_block_scaled_quant_producer(
            Float32(x1),
            Float32(y1),
            BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
            fast_math,
            clamped,
            alpha,
            limit,
        )
        packed[rep, pair] = _pack_b16x2(
            source_dtype(z0),
            source_dtype(z1),
            source_dtype,
        )


@cute.jit
def _load_swiglu_fwd_fp8_tile_values(
    mFwdX: cute.Tensor,
    mFwdY: cute.Tensor,
    z_vals: cute.Tensor,
    row_recips: cute.Tensor,
    row_start: Int32,
    row_lane: Int32,
    col_start: Int32,
    row_lanes_cfg: cutlass.Constexpr[int],
    row_reps_cfg: cutlass.Constexpr[int],
    num_elems: cutlass.Constexpr[int],
    source_dtype: type[cutlass.Numeric],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    do_row_quant: cutlass.Constexpr[bool],
) -> None:
    x_vals_packed = cute.make_rmem_tensor(
        num_elems // 2,
        Int32,
    )
    y_vals_packed = cute.make_rmem_tensor(
        num_elems // 2,
        Int32,
    )
    if cutlass.const_expr(do_row_quant):
        row_recips.fill(Float32(_FP32_ZERO))
    for rep in cutlass.range_constexpr(row_reps_cfg):
        row_rep_offset = Int32(rep * row_lanes_cfg)
        row = row_start + row_lane + row_rep_offset
        _load_tensor_row_as_b16x2(
            mFwdX,
            row,
            col_start,
            x_vals_packed,
            num_elems,
            source_dtype,
        )
        _load_tensor_row_as_b16x2(
            mFwdY,
            row,
            col_start,
            y_vals_packed,
            num_elems,
            source_dtype,
        )
        for pair in cutlass.range_constexpr(num_elems // 2):
            x_val0, x_val1 = _unpack_b16x2(
                x_vals_packed[pair],
                source_dtype,
            )
            y_val0, y_val1 = _unpack_b16x2(
                y_vals_packed[pair],
                source_dtype,
            )
            z0 = _apply_block_scaled_quant_producer(
                Float32(x_val0),
                Float32(y_val0),
                BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
                fast_math,
                clamped,
                alpha,
                limit,
            )
            z1 = _apply_block_scaled_quant_producer(
                Float32(x_val1),
                Float32(y_val1),
                BLOCK_SCALED_PRODUCER_SWIGLU_FWD,
                fast_math,
                clamped,
                alpha,
                limit,
            )
            z_vals[rep, 2 * pair] = z0
            z_vals[rep, 2 * pair + 1] = z1
            if cutlass.const_expr(do_row_quant):
                row_recips[rep] = _fmax_nan(row_recips[rep], _abs_f32(z0), _abs_f32(z1))
    if cutlass.const_expr(do_row_quant):
        # One NaN -> inf substitution per row accumulator; bitwise identical
        # to substituting per element (see _frag_amax_nonfinite).
        for rep in cutlass.range_constexpr(row_reps_cfg):
            row_recips[rep] = _compute_block_amax_nonfinite(row_recips[rep])


@cute.jit
def _load_swiglu_bwd_dxy_h2_as_f32x2(
    mDz: cute.Tensor,
    mH1: cute.Tensor,
    row: Int32,
    source_K: Int32,
    source_col_start: Int32,
    dx_lane_vals: cute.Tensor,
    dy_lane_vals: cute.Tensor,
    h2_lane_vals: cute.Tensor,
    num_elems: cutlass.Constexpr[int],
    source_dtype: type[cutlass.Numeric],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    return_h2: cutlass.Constexpr[bool],
) -> None:
    dz_vals_packed = cute.make_rmem_tensor(
        num_elems // 2,
        Int32,
    )
    x_vals_packed = cute.make_rmem_tensor(
        num_elems // 2,
        Int32,
    )
    y_vals_packed = cute.make_rmem_tensor(
        num_elems // 2,
        Int32,
    )
    _load_tensor_row_as_b16x2(
        mDz,
        row,
        source_col_start,
        dz_vals_packed,
        num_elems,
        source_dtype,
    )
    _load_tensor_row_as_b16x2(
        mH1,
        row,
        source_col_start,
        x_vals_packed,
        num_elems,
        source_dtype,
    )
    _load_tensor_row_as_b16x2(
        mH1,
        row,
        source_K + source_col_start,
        y_vals_packed,
        num_elems,
        source_dtype,
    )

    for pair in cutlass.range_constexpr(num_elems // 2):
        dz0, dz1 = _unpack_b16x2(
            dz_vals_packed[pair],
            source_dtype,
        )
        x0, x1 = _unpack_b16x2(
            x_vals_packed[pair],
            source_dtype,
        )
        y0, y1 = _unpack_b16x2(
            y_vals_packed[pair],
            source_dtype,
        )
        if cutlass.const_expr(return_h2):
            dx_pair, dy_pair, h2_pair = (
                _apply_block_scaled_quant_dxy_fwd_producer_f32x2(
                    (Float32(dz0), Float32(dz1)),
                    (Float32(x0), Float32(x1)),
                    (Float32(y0), Float32(y1)),
                    fast_math,
                    clamped,
                    alpha,
                    limit,
                )
            )
        else:
            dx_pair, dy_pair = _apply_block_scaled_quant_dxy_producer_f32x2(
                (Float32(dz0), Float32(dz1)),
                (Float32(x0), Float32(x1)),
                (Float32(y0), Float32(y1)),
                fast_math,
                clamped,
                alpha,
                limit,
            )
        dx_lane_vals[2 * pair] = dx_pair[0]
        dx_lane_vals[2 * pair + 1] = dx_pair[1]
        dy_lane_vals[2 * pair] = dy_pair[0]
        dy_lane_vals[2 * pair + 1] = dy_pair[1]
        if cutlass.const_expr(return_h2):
            h2_lane_vals[2 * pair] = h2_pair[0]
            h2_lane_vals[2 * pair + 1] = h2_pair[1]


@cute.jit
def _load_swiglu_bwd_fp8_tile_values(
    mDz: cute.Tensor,
    mH1: cute.Tensor,
    dx_vals: cute.Tensor,
    dy_vals: cute.Tensor,
    h2_vals: cute.Tensor,
    row_start: Int32,
    row_lane: Int32,
    source_K: Int32,
    source_lane_col_start: Int32,
    row_lanes_cfg: cutlass.Constexpr[int],
    row_reps_cfg: cutlass.Constexpr[int],
    num_elems: cutlass.Constexpr[int],
    source_dtype: type[cutlass.Numeric],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    return_h2: cutlass.Constexpr[bool],
) -> None:
    dx_lane_vals = cute.make_rmem_tensor(num_elems, Float32)
    dy_lane_vals = cute.make_rmem_tensor(num_elems, Float32)
    h2_lane_vals = cute.make_rmem_tensor(num_elems, Float32)
    for rep in cutlass.range_constexpr(row_reps_cfg):
        row_rep_offset = Int32(rep * row_lanes_cfg)
        row = row_start + row_lane + row_rep_offset
        _load_swiglu_bwd_dxy_h2_as_f32x2(
            mDz,
            mH1,
            row,
            source_K,
            source_lane_col_start,
            dx_lane_vals,
            dy_lane_vals,
            h2_lane_vals,
            num_elems,
            source_dtype,
            fast_math,
            clamped,
            alpha,
            limit,
            return_h2,
        )
        for i in cutlass.range_constexpr(num_elems):
            dx_vals[rep, i] = dx_lane_vals[i]
            dy_vals[rep, i] = dy_lane_vals[i]
            if cutlass.const_expr(return_h2):
                h2_vals[rep, i] = h2_lane_vals[i]


# -----------------------------------------------------------------------------
# SwiGLU FP8 tile quantization.
# -----------------------------------------------------------------------------


@cute.jit
def _quantize_swiglu_fp8_row_tile_values(
    vals: cute.Tensor,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
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
    row_reduce_stages: cutlass.Constexpr[int],
    format_id: cutlass.Constexpr[int],
    packed_f32x2: cutlass.Constexpr[bool],
    precomputed_amax: cutlass.Constexpr[bool],
    paired_scale_compute: cutlass.Constexpr[bool],
    qdata_elems_per_word: cutlass.Constexpr[int],
    cache_modifier: cutlass.Constexpr = None,
) -> None:
    _quantize_fp8_tile_values(
        vals,
        mRowQWords,
        mRowScale,
        scale_recips,
        q_vals,
        q_f32x2_packed,
        row_start,
        row_lane,
        q_word,
        scale_offset_base,
        scale_store_lane,
        row_lanes_cfg,
        row_reps_cfg,
        num_elems,
        lane_pairs,
        1,
        row_reduce_stages,
        row_lanes_cfg * CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE,
        False,
        packed_f32x2,
        precomputed_amax,
        paired_scale_compute,
        format_id,
        cutlass.const_expr(format_id in _HALF_RANGE_DEFAULT_FORMAT_IDS),
        qdata_elems_per_word,
        cache_modifier=cache_modifier,
    )


@cute.jit
def _quantize_swiglu_fp8_col_tile_values(
    vals: cute.Tensor,
    mColQWords: cute.Tensor,
    mColScale: cute.Tensor,
    scale_recips: cute.Tensor,
    q_vals: cute.Tensor,
    q_f32x2_packed: cute.Tensor,
    row_start: Int32,
    row_lane: Int32,
    q_word: Int32,
    scale_offset_base: Int32,
    row_lanes_cfg: cutlass.Constexpr[int],
    row_reps_cfg: cutlass.Constexpr[int],
    num_elems: cutlass.Constexpr[int],
    lane_pairs: cutlass.Constexpr[int],
    col_lanes_cfg: cutlass.Constexpr[int],
    col_reduce_stages: cutlass.Constexpr[int],
    format_id: cutlass.Constexpr[int],
    packed_f32x2: cutlass.Constexpr[bool],
    paired_scale_compute: cutlass.Constexpr[bool],
    qdata_elems_per_word: cutlass.Constexpr[int],
) -> None:
    _quantize_fp8_tile_values(
        vals,
        mColQWords,
        mColScale,
        scale_recips,
        q_vals,
        q_f32x2_packed,
        row_start,
        row_lane,
        q_word,
        scale_offset_base,
        row_lane == Int32(0),
        row_lanes_cfg,
        row_reps_cfg,
        num_elems,
        lane_pairs,
        col_lanes_cfg,
        col_reduce_stages,
        CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE,
        True,
        packed_f32x2,
        False,
        paired_scale_compute,
        format_id,
        cutlass.const_expr(format_id in _HALF_RANGE_DEFAULT_FORMAT_IDS),
        qdata_elems_per_word,
    )


@cute.jit
def _quantize_swiglu_bwd_fp8_tile_values(
    vals: cute.Tensor,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
    mColQWords: cute.Tensor,
    mColScale: cute.Tensor,
    scale_recips: cute.Tensor,
    q_vals: cute.Tensor,
    q_f32x2_packed: cute.Tensor,
    row_start: Int32,
    row_lane: Int32,
    q_word: Int32,
    row_scale_offset_base: Int32,
    col_scale_offset_base: Int32,
    row_scale_lane,
    row_lanes_cfg: cutlass.Constexpr[int],
    row_reps_cfg: cutlass.Constexpr[int],
    num_elems: cutlass.Constexpr[int],
    lane_pairs: cutlass.Constexpr[int],
    row_reduce_stages: cutlass.Constexpr[int],
    col_lanes_cfg: cutlass.Constexpr[int],
    col_reduce_stages: cutlass.Constexpr[int],
    format_id: cutlass.Constexpr[int],
    qdata_elems_per_word: cutlass.Constexpr[int],
) -> None:
    _quantize_swiglu_fp8_row_tile_values(
        vals,
        mRowQWords,
        mRowScale,
        scale_recips,
        q_vals,
        q_f32x2_packed,
        row_start,
        row_lane,
        q_word,
        row_scale_offset_base,
        row_scale_lane,
        row_lanes_cfg,
        row_reps_cfg,
        num_elems,
        lane_pairs,
        row_reduce_stages,
        format_id,
        True,
        False,
        True,
        qdata_elems_per_word,
    )
    _quantize_swiglu_fp8_col_tile_values(
        vals,
        mColQWords,
        mColScale,
        scale_recips,
        q_vals,
        q_f32x2_packed,
        row_start,
        row_lane,
        q_word,
        col_scale_offset_base,
        row_lanes_cfg,
        row_reps_cfg,
        num_elems,
        lane_pairs,
        col_lanes_cfg,
        col_reduce_stages,
        format_id,
        True,
        True,
        qdata_elems_per_word,
    )


@cute.jit
def _quantize_swiglu_fwd_fp8_tile(  # noqa: C901
    warp_lane: Int32,
    mFwdX: cute.Tensor,
    mFwdY: cute.Tensor,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
    mColQWords: cute.Tensor,
    mColScale: cute.Tensor,
    row_start: Int32,
    row_block: Int32,
    col_block: Int32,
    row_blocks: Int32,
    K: Int32,
    source_dtype: type[cutlass.Numeric],
    sf_vec_size: cutlass.Constexpr[int],
    format_id: cutlass.Constexpr[int],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    qdata_elems_per_word: cutlass.Constexpr[int],
    num_elems: cutlass.Constexpr[int],
    col_blocks_per_scale: cutlass.Constexpr[int],
    col_lanes_cfg: cutlass.Constexpr[int],
    do_row_quant: cutlass.Constexpr[bool],
    do_col_quant: cutlass.Constexpr[bool],
) -> None:
    row_lanes_cfg: cutlass.Constexpr[int] = 32 // col_lanes_cfg
    row_reps_cfg: cutlass.Constexpr[int] = sf_vec_size // row_lanes_cfg
    col_reduce_stages: cutlass.Constexpr[int] = row_lanes_cfg.bit_length() - 1
    row_reduce_stages: cutlass.Constexpr[int] = col_blocks_per_scale.bit_length() - 1
    if cutlass.const_expr(sf_vec_size != num_elems * col_blocks_per_scale):
        raise ValueError("swiglu fwd quant lane layout must cover one scale column")
    if cutlass.const_expr(num_elems % qdata_elems_per_word != 0):
        raise ValueError("swiglu fwd lane elems must divide qdata words")
    lane_pairs: cutlass.Constexpr[int] = num_elems // 2
    if cutlass.const_expr(
        (do_row_quant and mRowScale.element_type != Uint8)
        or (do_col_quant and mColScale.element_type != Uint8)
    ):
        raise TypeError("swiglu fwd qscale stores require Uint8 tensors")

    row_lane, col_lane_block, col_lane_pair, col_start, col_q_word = (
        _swiglu_fp8_tile_lane_coords(
            warp_lane,
            col_lanes_cfg,
            col_blocks_per_scale,
            num_elems,
            col_block,
            sf_vec_size,
            qdata_elems_per_word,
        )
    )
    col_scale_cols = K // Int32(sf_vec_size)

    z_vals = cute.make_rmem_tensor(
        (row_reps_cfg, num_elems),
        Float32,
    )
    row_recips = cute.make_rmem_tensor(
        row_reps_cfg,
        Float32,
    )
    col_recips = cute.make_rmem_tensor(
        num_elems,
        Float32,
    )
    q_vals = cute.make_rmem_tensor(
        num_elems,
        Float32,
    )
    q_f32x2_packed = cute.make_rmem_tensor(lane_pairs, Uint64)
    _load_swiglu_fwd_fp8_tile_values(
        mFwdX,
        mFwdY,
        z_vals,
        row_recips,
        row_start,
        row_lane,
        col_start,
        row_lanes_cfg,
        row_reps_cfg,
        num_elems,
        source_dtype,
        fast_math,
        clamped,
        alpha,
        limit,
        do_row_quant,
    )
    if cutlass.const_expr(do_row_quant):
        row_scale_offset_base, row_scale_lane = _swiglu_fp8_row_scale_offset_base(
            row_start,
            row_lane,
            col_block,
            col_lane_pair,
            col_lane_block,
            col_blocks_per_scale,
            col_scale_cols,
            Int32(0),
        )
        _quantize_swiglu_fp8_row_tile_values(
            z_vals,
            mRowQWords,
            mRowScale,
            row_recips,
            q_vals,
            q_f32x2_packed,
            row_start,
            row_lane,
            col_q_word,
            row_scale_offset_base,
            row_scale_lane,
            row_lanes_cfg,
            row_reps_cfg,
            num_elems,
            lane_pairs,
            row_reduce_stages,
            format_id,
            False,
            True,
            False,
            qdata_elems_per_word,
        )

    if cutlass.const_expr(do_col_quant):
        col_scale_offset_base = _swiglu_fp8_col_scale_offset_base(
            row_lane,
            col_start,
            row_block,
            row_blocks,
        )
        _quantize_swiglu_fp8_col_tile_values(
            z_vals,
            mColQWords,
            mColScale,
            col_recips,
            q_vals,
            q_f32x2_packed,
            row_start,
            row_lane,
            col_q_word,
            col_scale_offset_base,
            row_lanes_cfg,
            row_reps_cfg,
            num_elems,
            lane_pairs,
            col_lanes_cfg,
            col_reduce_stages,
            format_id,
            False,
            False,
            qdata_elems_per_word,
            Uint32(0),
            Uint32(0),
            Int32(0),
            False,
            0,
        )


@cute.jit
def _quantize_swiglu_bwd_dxy_fp8_tile(  # noqa: C901
    warp_lane: Int32,
    mDz: cute.Tensor,
    mH1: cute.Tensor,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
    mDxyColQWords: cute.Tensor,
    mDxyColScale: cute.Tensor,
    row_start: Int32,
    row_block: Int32,
    col_block: Int32,
    row_blocks: Int32,
    K: Int32,
    source_dtype: type[cutlass.Numeric],
    sf_vec_size: cutlass.Constexpr[int],
    format_id: cutlass.Constexpr[int],
    fast_math: cutlass.Constexpr[bool],
    clamped: cutlass.Constexpr[bool],
    alpha: cutlass.Constexpr[float],
    limit: cutlass.Constexpr[float],
    qdata_elems_per_word: cutlass.Constexpr[int],
    num_elems: cutlass.Constexpr[int],
    col_blocks_per_scale: cutlass.Constexpr[int],
    col_lanes_cfg: cutlass.Constexpr[int],
) -> None:
    row_lanes_cfg: cutlass.Constexpr[int] = 32 // col_lanes_cfg
    row_reps_cfg: cutlass.Constexpr[int] = sf_vec_size // row_lanes_cfg
    col_reduce_stages: cutlass.Constexpr[int] = row_lanes_cfg.bit_length() - 1
    row_reduce_stages: cutlass.Constexpr[int] = col_blocks_per_scale.bit_length() - 1
    if cutlass.const_expr(sf_vec_size != num_elems * col_blocks_per_scale):
        raise ValueError("swiglu bwd quant lane layout must cover one scale column")
    if cutlass.const_expr(num_elems % qdata_elems_per_word != 0):
        raise ValueError("swiglu bwd lane elems must divide qdata words")
    if cutlass.const_expr(row_reps_cfg < num_elems):
        raise ValueError("swiglu bwd recip scratch must cover col scales")
    lane_pairs: cutlass.Constexpr[int] = num_elems // 2
    if cutlass.const_expr(mRowScale.element_type != Uint8):
        raise TypeError("swiglu bwd row scale stores require Uint8 tensor")
    if cutlass.const_expr(mDxyColScale.element_type != Uint8):
        raise TypeError("swiglu bwd dxy col scale store requires Uint8 tensor")

    row_lane, col_lane_block, col_lane_pair, source_lane_col_start, dx_q_word = (
        _swiglu_fp8_tile_lane_coords(
            warp_lane,
            col_lanes_cfg,
            col_blocks_per_scale,
            num_elems,
            col_block,
            sf_vec_size,
            qdata_elems_per_word,
        )
    )
    source_K = K // Int32(2)
    source_scale_cols = source_K // Int32(sf_vec_size)
    dxy_scale_cols = K // Int32(sf_vec_size)
    dx_lane_col_start = source_lane_col_start
    dy_lane_col_start = source_K + source_lane_col_start
    dy_q_word = dy_lane_col_start // Int32(qdata_elems_per_word)

    dx_vals = cute.make_rmem_tensor(
        (row_reps_cfg, num_elems),
        Float32,
    )
    dy_vals = cute.make_rmem_tensor(
        (row_reps_cfg, num_elems),
        Float32,
    )
    h2_vals = cute.make_rmem_tensor(
        (row_reps_cfg, num_elems),
        Float32,
    )
    scale_recips = cute.make_rmem_tensor(row_reps_cfg, Float32)
    q_vals = cute.make_rmem_tensor(num_elems, Float32)
    q_f32x2_packed = cute.make_rmem_tensor(lane_pairs, Uint64)
    _load_swiglu_bwd_fp8_tile_values(
        mDz,
        mH1,
        dx_vals,
        dy_vals,
        h2_vals,
        row_start,
        row_lane,
        source_K,
        source_lane_col_start,
        row_lanes_cfg,
        row_reps_cfg,
        num_elems,
        source_dtype,
        fast_math,
        clamped,
        alpha,
        limit,
        False,
    )

    dx_row_scale_offset_base, row_scale_lane = _swiglu_fp8_row_scale_offset_base(
        row_start,
        row_lane,
        col_block,
        col_lane_pair,
        col_lane_block,
        col_blocks_per_scale,
        dxy_scale_cols,
        Int32(0),
    )
    dy_row_scale_offset_base, _ = _swiglu_fp8_row_scale_offset_base(
        row_start,
        row_lane,
        col_block,
        col_lane_pair,
        col_lane_block,
        col_blocks_per_scale,
        dxy_scale_cols,
        source_scale_cols,
    )
    dx_col_scale_offset_base = _swiglu_fp8_col_scale_offset_base(
        row_lane,
        dx_lane_col_start,
        row_block,
        row_blocks,
    )
    dy_col_scale_offset_base = _swiglu_fp8_col_scale_offset_base(
        row_lane,
        dy_lane_col_start,
        row_block,
        row_blocks,
    )
    _quantize_swiglu_bwd_fp8_tile_values(
        dx_vals,
        mRowQWords,
        mRowScale,
        mDxyColQWords,
        mDxyColScale,
        scale_recips,
        q_vals,
        q_f32x2_packed,
        row_start,
        row_lane,
        dx_q_word,
        dx_row_scale_offset_base,
        dx_col_scale_offset_base,
        row_scale_lane,
        row_lanes_cfg,
        row_reps_cfg,
        num_elems,
        lane_pairs,
        row_reduce_stages,
        col_lanes_cfg,
        col_reduce_stages,
        format_id,
        qdata_elems_per_word,
    )
    _quantize_swiglu_bwd_fp8_tile_values(
        dy_vals,
        mRowQWords,
        mRowScale,
        mDxyColQWords,
        mDxyColScale,
        scale_recips,
        q_vals,
        q_f32x2_packed,
        row_start,
        row_lane,
        dy_q_word,
        dy_row_scale_offset_base,
        dy_col_scale_offset_base,
        row_scale_lane,
        row_lanes_cfg,
        row_reps_cfg,
        num_elems,
        lane_pairs,
        row_reduce_stages,
        col_lanes_cfg,
        col_reduce_stages,
        format_id,
        qdata_elems_per_word,
    )
