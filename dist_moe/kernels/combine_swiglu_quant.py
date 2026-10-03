# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side combine-SwiGLU quant producer helpers for the DistMoE kernels.

Free ``@cute.jit`` functions hoisted out of ``DistBlockScaledGroupedGemmKernel``.
The wide quantize/producer functions read their compile-time attributes through
a ``CombineSwigluQuantParams`` record (the ``arguments.py`` ``CuteParamsBase``
convention); the narrower helpers take explicit ``cutlass.Constexpr``
parameters (the ``swiglu_epilogue.py`` convention).
"""

import cutlass
import cutlass.cute as cute

from ._swiglu_quant import (
    _load_swiglu_bwd_fp8_tile_values,
    _load_swiglu_fwd_fp8_tile_values,
    _quantize_swiglu_fp8_col_tile_values,
    _quantize_swiglu_fp8_row_tile_values,
    _swiglu_fp8_col_scale_offset_base,
    _swiglu_fp8_row_scale_offset_base,
    _swiglu_fp8_tile_lane_coords,
)
from .blockscaled_grouped_gemm import (
    NVFP4,
)
from .dispatch_quant import (
    _mega_dispatch_quant_wgrad_col_tile_wait,
    _wait_counter_at_least,
)
from .grouped_gemm_kernel import (
    _activation_buffer_rows,
    _remap_m_tile_idx,
)
from .params import (
    ceil_div,
    CombineSwigluQuantParams,
)
from .tile_scheduler import (
    advance_blockscaled_scale_start,
    blockscaled_scale_row_start,
)


@cute.jit
def _combine_swiglu_quant_fetch_warp_tile_batch(
    mWorkCounter: cute.Tensor,
    warp_lane: cutlass.Int32,
    COMBINE_SWIGLU_WORK_TILES_PER_FETCH: cutlass.Constexpr[int],
) -> cutlass.Int32:
    tile_idx = cutlass.Int32(0)
    if warp_lane == cutlass.Int32(0):
        work_counter_ptr = cute.recast_ptr(
            mWorkCounter.iterator,
            dtype=cutlass.Uint32,
        )
        tile_idx = cutlass.Int32(
            cute.arch.atomic_add(
                work_counter_ptr,
                cutlass.Uint32(COMBINE_SWIGLU_WORK_TILES_PER_FETCH),
                sem="release",
                scope="gpu",
            )
        )
    return cute.arch.shuffle_sync(tile_idx, cutlass.Int32(0))


@cute.jit
def _combine_swiglu_quant_signal_warp_tile_done(
    mDoneCounter: cute.Tensor,
    act_tile_slot: cutlass.Int32,
    warp_lane: cutlass.Int32,
) -> None:
    # Publish this lane's generic-proxy gmem stores to the async proxy
    # before signalling: the released counter gates the GEMM's TMA loads
    # of the row-quant operand, and fence.proxy.async is thread-local, so
    # it must run on every producer lane ahead of the release below.
    # This fence and counter cover only the row-quant operand. Col-quant
    # stores are not gated by this counter and are published (where an
    # in-kernel consumer needs it) through their own col done counter --
    # e.g. the mega kernel's _mega_combine_quant_signal_col_tile_done,
    # which inherits this tile body but consumes col-quant in-kernel; do
    # not read this fence as covering every quant store of the tile. Same
    # protocol as _dispatch_quant_signal_group_tile_done above and the
    # mega-kernel combine-quant signal (D115294535).
    cute.arch.fence_proxy("async.global")
    cute.arch.fence_acq_rel_gpu()
    cute.arch.sync_warp()
    if warp_lane == cutlass.Int32(0):
        counter_slot_ptr = cute.recast_ptr(
            mDoneCounter.iterator + act_tile_slot,
            dtype=cutlass.Uint32,
        )
        cute.arch.atomic_add(
            counter_slot_ptr,
            cutlass.Uint32(1),
            sem="release",
            scope="gpu",
        )


@cute.jit
def _combine_swiglu_fwd_quantize_tile(
    params: CombineSwigluQuantParams,
    warp_lane: cutlass.Int32,
    mDoneCounter: cute.Tensor,
    mFwdX: cute.Tensor,
    mFwdY: cute.Tensor,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
    mColQWords: cute.Tensor,
    mColScale: cute.Tensor,
    act_tile_slot: cutlass.Int32,
    row_start: cutlass.Int32,
    row_scale_start: cutlass.Int32,
    row_block: cutlass.Int32,
    col_block: cutlass.Int32,
    row_blocks: cutlass.Int32,
    K: cutlass.Int32,
    mValidPtrs=None,
) -> None:
    lane_elems: cutlass.Constexpr[int] = (
        params.COMBINE_SWIGLU_FWD_ROW_ONLY_ELEMS_PER_LANE
        if params.combine_swiglu_row_quant_only
        else params.COMBINE_SWIGLU_FWD_ELEMS_PER_LANE
    )
    col_lanes: cutlass.Constexpr[int] = (
        params.COMBINE_SWIGLU_FWD_ROW_ONLY_COL_LANES
        if params.combine_swiglu_row_quant_only
        else params.COMBINE_SWIGLU_FWD_COL_LANES
    )
    scale_cols_per_subtile: cutlass.Constexpr[int] = (
        col_lanes * lane_elems // params.sf_vec_size
    )
    num_subtiles: cutlass.Constexpr[int] = (
        params.COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE // scale_cols_per_subtile
    )
    col_scale_cols = K // cutlass.Int32(params.sf_vec_size)
    valid_subtiles = min(
        cutlass.Int32(num_subtiles),
        col_scale_cols - col_block,
    )
    for subtile in cutlass.range(num_subtiles, unroll=1):
        subtile_col_block = col_block + subtile * cutlass.Int32(scale_cols_per_subtile)
        if subtile < valid_subtiles:
            _combine_swiglu_fwd_quantize_subtile(
                params,
                warp_lane,
                mDoneCounter,
                mFwdX,
                mFwdY,
                mRowQWords,
                mRowScale,
                mColQWords,
                mColScale,
                act_tile_slot,
                row_start,
                row_scale_start,
                row_block,
                subtile_col_block,
                row_blocks,
                K,
                subtile == valid_subtiles - cutlass.Int32(1),
                mValidPtrs,
            )


@cute.jit
def _combine_swiglu_fwd_quantize_subtile(  # noqa: C901
    params: CombineSwigluQuantParams,
    warp_lane: cutlass.Int32,
    mDoneCounter: cute.Tensor,
    mFwdX: cute.Tensor,
    mFwdY: cute.Tensor,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
    mColQWords: cute.Tensor,
    mColScale: cute.Tensor,
    act_tile_slot: cutlass.Int32,
    row_start: cutlass.Int32,
    row_scale_start: cutlass.Int32,
    row_block: cutlass.Int32,
    col_block: cutlass.Int32,
    row_blocks: cutlass.Int32,
    K: cutlass.Int32,
    signal_done,
    mValidPtrs=None,
) -> None:
    fwd_row_quant_only: cutlass.Constexpr[bool] = params.combine_swiglu_row_quant_only
    do_fwd_row_quant: cutlass.Constexpr[bool] = True
    do_fwd_col_quant: cutlass.Constexpr[bool] = not fwd_row_quant_only
    lane_elems: cutlass.Constexpr[int] = (
        params.COMBINE_SWIGLU_FWD_ROW_ONLY_ELEMS_PER_LANE
        if fwd_row_quant_only
        else params.COMBINE_SWIGLU_FWD_ELEMS_PER_LANE
    )
    col_blocks_per_scale: cutlass.Constexpr[int] = (
        params.COMBINE_SWIGLU_FWD_ROW_ONLY_COL_BLOCKS_PER_SCALE
        if fwd_row_quant_only
        else params.COMBINE_SWIGLU_FWD_COL_BLOCKS_PER_SCALE
    )
    col_lanes_cfg: cutlass.Constexpr[int] = (
        params.COMBINE_SWIGLU_FWD_ROW_ONLY_COL_LANES
        if fwd_row_quant_only
        else params.COMBINE_SWIGLU_FWD_COL_LANES
    )
    row_lanes_cfg: cutlass.Constexpr[int] = 32 // col_lanes_cfg
    row_reps_cfg: cutlass.Constexpr[int] = params.sf_vec_size // row_lanes_cfg
    col_reduce_stages: cutlass.Constexpr[int] = row_lanes_cfg.bit_length() - 1
    row_reduce_stages: cutlass.Constexpr[int] = col_blocks_per_scale.bit_length() - 1
    if cutlass.const_expr(params.sf_vec_size != lane_elems * col_blocks_per_scale):
        raise ValueError("swiglu fwd quant lane layout must cover one scale column")
    if cutlass.const_expr(
        lane_elems % params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD != 0
        and lane_elems * 2 != params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD
    ):
        raise ValueError("swiglu fwd lane elems must divide qdata words")
    lane_pairs: cutlass.Constexpr[int] = lane_elems // 2
    if cutlass.const_expr(
        (
            do_fwd_row_quant
            and mRowScale.element_type not in (cutlass.Uint8, cutlass.Int8)
        )
        or (
            do_fwd_col_quant
            and mColScale.element_type not in (cutlass.Uint8, cutlass.Int8)
        )
    ):
        raise TypeError("swiglu fwd qscale stores require byte tensors")

    row_lane, col_lane_block, col_lane_pair, col_start, col_q_word = (
        _swiglu_fp8_tile_lane_coords(
            warp_lane,
            col_lanes_cfg,
            col_blocks_per_scale,
            lane_elems,
            col_block,
            params.sf_vec_size,
            params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
        )
    )
    col_scale_cols = K // cutlass.Int32(params.sf_vec_size)
    z_vals = cute.make_rmem_tensor(
        (row_reps_cfg, lane_elems),
        cutlass.Float32,
    )
    row_recips = cute.make_rmem_tensor(
        row_reps_cfg,
        cutlass.Float32,
    )
    col_recips = cute.make_rmem_tensor(
        lane_elems,
        cutlass.Float32,
    )
    q_vals = cute.make_rmem_tensor(
        lane_elems,
        cutlass.Float32,
    )
    q_f32x2_packed = cute.make_rmem_tensor(lane_pairs, cutlass.Uint64)
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
        lane_elems,
        params.dispatch_quant_source_dtype,
        params.combine_swiglu_fast_math,
        params.combine_swiglu_clamped,
        params.combine_swiglu_alpha,
        params.combine_swiglu_limit,
        do_fwd_row_quant,
    )
    if cutlass.const_expr(mValidPtrs is not None):
        for row_rep in cutlass.range_constexpr(row_reps_cfg):
            row = row_start + row_lane + cutlass.Int32(row_rep * row_lanes_cfg)
            if mValidPtrs[row] == cutlass.Int64(0):
                for elem in cutlass.range_constexpr(lane_elems):
                    z_vals[row_rep, elem] = cutlass.Float32(0)
    if cutlass.const_expr(do_fwd_row_quant):
        row_scale_offset_base, row_scale_lane = _swiglu_fp8_row_scale_offset_base(
            row_scale_start,
            row_lane,
            col_block,
            col_lane_pair,
            col_lane_block,
            col_blocks_per_scale,
            col_scale_cols,
            cutlass.Int32(0),
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
            lane_elems,
            lane_pairs,
            row_reduce_stages,
            params.format_id,
            False,
            True,
            False,
            params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
            cache_modifier="cs",
        )
    if signal_done:
        _combine_swiglu_quant_signal_warp_tile_done(
            mDoneCounter,
            act_tile_slot,
            warp_lane,
        )
    if cutlass.const_expr(do_fwd_col_quant):
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
            lane_elems,
            lane_pairs,
            col_lanes_cfg,
            col_reduce_stages,
            params.format_id,
            False,
            False,
            params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
        )


@cute.jit
def _combine_swiglu_bwd_quantize_tile(  # noqa: C901
    params: CombineSwigluQuantParams,
    warp_lane: cutlass.Int32,
    mDoneCounter: cute.Tensor,
    mDz: cute.Tensor,
    mH1: cute.Tensor,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
    mDxyColQWords: cute.Tensor,
    mDxyColScale: cute.Tensor,
    act_tile_slot: cutlass.Int32,
    row_start: cutlass.Int32,
    row_block: cutlass.Int32,
    col_block: cutlass.Int32,
    row_blocks: cutlass.Int32,
    K: cutlass.Int32,
    mValidPtrs=None,
) -> None:
    lane_elems: cutlass.Constexpr[int] = params.COMBINE_SWIGLU_BWD_ELEMS_PER_LANE
    col_blocks_per_scale: cutlass.Constexpr[int] = (
        params.COMBINE_SWIGLU_BWD_COL_BLOCKS_PER_SCALE
    )
    col_lanes_cfg: cutlass.Constexpr[int] = params.COMBINE_SWIGLU_BWD_COL_LANES
    row_lanes_cfg: cutlass.Constexpr[int] = 32 // col_lanes_cfg
    row_reps_cfg: cutlass.Constexpr[int] = params.sf_vec_size // row_lanes_cfg
    col_reduce_stages: cutlass.Constexpr[int] = row_lanes_cfg.bit_length() - 1
    row_reduce_stages: cutlass.Constexpr[int] = col_blocks_per_scale.bit_length() - 1
    if cutlass.const_expr(params.sf_vec_size != lane_elems * col_blocks_per_scale):
        raise ValueError("swiglu bwd quant lane layout must cover one scale column")
    if cutlass.const_expr(
        lane_elems % params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD != 0
        and lane_elems * 2 != params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD
    ):
        raise ValueError("swiglu bwd lane elems must divide qdata words")
    if cutlass.const_expr(row_reps_cfg < lane_elems):
        raise ValueError("swiglu bwd recip scratch must cover col scales")
    lane_pairs: cutlass.Constexpr[int] = lane_elems // 2
    if cutlass.const_expr(mRowScale.element_type not in (cutlass.Uint8, cutlass.Int8)):
        raise TypeError("swiglu bwd row scale stores require a byte tensor")
    if cutlass.const_expr(
        mDxyColScale.element_type not in (cutlass.Uint8, cutlass.Int8)
    ):
        raise TypeError("swiglu bwd col scale stores require byte tensors")

    row_lane, col_lane_block, col_lane_pair, source_lane_col_start, dx_q_word = (
        _swiglu_fp8_tile_lane_coords(
            warp_lane,
            col_lanes_cfg,
            col_blocks_per_scale,
            lane_elems,
            col_block,
            params.sf_vec_size,
            params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
        )
    )
    source_K = K // cutlass.Int32(2)
    source_scale_cols = source_K // cutlass.Int32(params.sf_vec_size)
    dxy_scale_cols = K // cutlass.Int32(params.sf_vec_size)
    dx_lane_col_start = source_lane_col_start
    dy_lane_col_start = source_K + source_lane_col_start
    dy_q_word = dy_lane_col_start // cutlass.Int32(
        params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD
    )
    dx_vals = cute.make_rmem_tensor(
        (row_reps_cfg, lane_elems),
        cutlass.Float32,
    )
    dy_vals = cute.make_rmem_tensor(
        (row_reps_cfg, lane_elems),
        cutlass.Float32,
    )
    scale_recips = cute.make_rmem_tensor(row_reps_cfg, cutlass.Float32)
    q_vals = cute.make_rmem_tensor(lane_elems, cutlass.Float32)
    q_f32x2_packed = cute.make_rmem_tensor(lane_pairs, cutlass.Uint64)
    _load_swiglu_bwd_fp8_tile_values(
        mDz,
        mH1,
        dx_vals,
        dy_vals,
        dx_vals,
        row_start,
        row_lane,
        source_K,
        source_lane_col_start,
        row_lanes_cfg,
        row_reps_cfg,
        lane_elems,
        params.dispatch_quant_source_dtype,
        params.combine_swiglu_fast_math,
        params.combine_swiglu_clamped,
        params.combine_swiglu_alpha,
        params.combine_swiglu_limit,
        False,
    )
    if cutlass.const_expr(mValidPtrs is not None):
        for row_rep in cutlass.range_constexpr(row_reps_cfg):
            row = row_start + row_lane + cutlass.Int32(row_rep * row_lanes_cfg)
            if mValidPtrs[row] == cutlass.Int64(0):
                for elem in cutlass.range_constexpr(lane_elems):
                    dx_vals[row_rep, elem] = cutlass.Float32(0)
                    dy_vals[row_rep, elem] = cutlass.Float32(0)
    dx_row_scale_offset_base, row_scale_lane = _swiglu_fp8_row_scale_offset_base(
        row_start,
        row_lane,
        col_block,
        col_lane_pair,
        col_lane_block,
        col_blocks_per_scale,
        dxy_scale_cols,
        cutlass.Int32(0),
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
    _quantize_swiglu_fp8_row_tile_values(
        dx_vals,
        mRowQWords,
        mRowScale,
        scale_recips,
        q_vals,
        q_f32x2_packed,
        row_start,
        row_lane,
        dx_q_word,
        dx_row_scale_offset_base,
        row_scale_lane,
        row_lanes_cfg,
        row_reps_cfg,
        lane_elems,
        lane_pairs,
        row_reduce_stages,
        params.format_id,
        True,
        False,
        True,
        params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
    )
    _quantize_swiglu_fp8_col_tile_values(
        dx_vals,
        mDxyColQWords,
        mDxyColScale,
        scale_recips,
        q_vals,
        q_f32x2_packed,
        row_start,
        row_lane,
        dx_q_word,
        dx_col_scale_offset_base,
        row_lanes_cfg,
        row_reps_cfg,
        lane_elems,
        lane_pairs,
        col_lanes_cfg,
        col_reduce_stages,
        params.format_id,
        True,
        True,
        params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
    )
    _quantize_swiglu_fp8_row_tile_values(
        dy_vals,
        mRowQWords,
        mRowScale,
        scale_recips,
        q_vals,
        q_f32x2_packed,
        row_start,
        row_lane,
        dy_q_word,
        dy_row_scale_offset_base,
        row_scale_lane,
        row_lanes_cfg,
        row_reps_cfg,
        lane_elems,
        lane_pairs,
        row_reduce_stages,
        params.format_id,
        True,
        False,
        True,
        params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
    )
    _combine_swiglu_quant_signal_warp_tile_done(
        mDoneCounter,
        act_tile_slot,
        warp_lane,
    )
    _quantize_swiglu_fp8_col_tile_values(
        dy_vals,
        mDxyColQWords,
        mDxyColScale,
        scale_recips,
        q_vals,
        q_f32x2_packed,
        row_start,
        row_lane,
        dy_q_word,
        dy_col_scale_offset_base,
        row_lanes_cfg,
        row_reps_cfg,
        lane_elems,
        lane_pairs,
        col_lanes_cfg,
        col_reduce_stages,
        params.format_id,
        True,
        True,
        params.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
    )


@cute.jit
def _combine_swiglu_fwd_quant_producer_body(
    params: CombineSwigluQuantParams,
    mWorkCounter: cute.Tensor,
    mDoneCounter: cute.Tensor,
    tile_smem_ptr: cute.Pointer,
    mFwdX: cute.Tensor,
    mFwdY: cute.Tensor,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
    mColQWords: cute.Tensor,
    mColScale: cute.Tensor,
    split_sizes: cute.Tensor,
    G: cutlass.Constexpr[int],
    M: cutlass.Int32,
    K: cutlass.Int32,
    total_tiles: cutlass.Int32,
    local_rank: cutlass.Int32,
) -> None:
    del M, tile_smem_ptr, total_tiles
    tidx, _, _ = cute.arch.thread_idx()
    quant_lane = tidx - cutlass.Int32(
        params.COMBINE_SWIGLU_QUANT_FIRST_WARP * params.THREADS_PER_WARP
    )
    warp_lane = quant_lane % cutlass.Int32(params.THREADS_PER_WARP)
    col_scale_cols = K // cutlass.Int32(params.sf_vec_size)
    col_work_tiles = ceil_div(
        col_scale_cols,
        cutlass.Int32(params.COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE),
    )
    row_blocks_total = _activation_buffer_rows(
        split_sizes,
        G,
    ) // cutlass.Int32(params.sf_vec_size)
    ACT_BLOCK_SIZE: cutlass.Constexpr[int] = (
        params.BLOCK_SIZE_N if params.SWAP_AB else params.BLOCK_SIZE_M
    )
    if cutlass.const_expr(ACT_BLOCK_SIZE % params.sf_vec_size != 0):
        raise ValueError(
            "combine swiglu fwd quant act tile must be a multiple of sf_vec_size"
        )
    row_blocks_per_act_tile: cutlass.Constexpr[int] = (
        ACT_BLOCK_SIZE // params.sf_vec_size
    )

    tile_idx_base = _combine_swiglu_quant_fetch_warp_tile_batch(
        mWorkCounter,
        warp_lane,
        params.COMBINE_SWIGLU_WORK_TILES_PER_FETCH,
    )
    tile_idx = tile_idx_base
    tile_batch_end = tile_idx_base + cutlass.Int32(
        params.COMBINE_SWIGLU_WORK_TILES_PER_FETCH
    )
    tile_start = cutlass.Int32(0)
    act_tile_start = cutlass.Int32(0)
    start_m = cutlass.Int32(0)
    scale_start_m = cutlass.Int32(0)
    for g in cutlass.range(G, unroll=1):
        m_size = cutlass.Int32(split_sizes[g])
        group_row_blocks = m_size // cutlass.Int32(params.sf_vec_size)
        group_tiles = group_row_blocks * col_work_tiles
        group_act_tiles = ceil_div(m_size, cutlass.Int32(ACT_BLOCK_SIZE))
        tile_end = tile_start + group_tiles

        while (tile_idx >= tile_start) and (tile_idx < tile_end):
            local_tile = tile_idx - tile_start
            local_row_block = local_tile // col_work_tiles
            col_work_tile = local_tile - local_row_block * col_work_tiles
            token_off = _remap_m_tile_idx(
                cutlass.Int32(0),
                group_act_tiles,
                local_rank,
                params.world_size,
            )
            local_row_block = (
                local_row_block + token_off * cutlass.Int32(row_blocks_per_act_tile)
            ) % group_row_blocks
            col_first_block = col_work_tile * cutlass.Int32(
                params.COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE
            )
            row_block = start_m // cutlass.Int32(params.sf_vec_size) + local_row_block
            row_start = start_m + local_row_block * cutlass.Int32(params.sf_vec_size)
            row_scale_start = blockscaled_scale_row_start(
                scale_start_m,
                local_row_block * cutlass.Int32(params.sf_vec_size),
                params.SWAP_AB,
                params.BLOCK_SIZE_N,
            )
            act_tile_slot = act_tile_start + local_row_block // cutlass.Int32(
                row_blocks_per_act_tile
            )

            col_block = col_first_block
            if col_block < col_scale_cols:
                _combine_swiglu_fwd_quantize_tile(
                    params,
                    warp_lane,
                    mDoneCounter,
                    mFwdX,
                    mFwdY,
                    mRowQWords,
                    mRowScale,
                    mColQWords,
                    mColScale,
                    act_tile_slot,
                    row_start,
                    row_scale_start,
                    row_block,
                    col_block,
                    row_blocks_total,
                    K,
                )

            tile_idx += cutlass.Int32(1)
            if tile_idx == tile_batch_end:
                tile_idx_base = _combine_swiglu_quant_fetch_warp_tile_batch(
                    mWorkCounter,
                    warp_lane,
                    params.COMBINE_SWIGLU_WORK_TILES_PER_FETCH,
                )
                tile_idx = tile_idx_base
                tile_batch_end = tile_idx_base + cutlass.Int32(
                    params.COMBINE_SWIGLU_WORK_TILES_PER_FETCH
                )

        start_m += m_size
        scale_start_m = advance_blockscaled_scale_start(
            scale_start_m, m_size, params.SWAP_AB, params.BLOCK_SIZE_N
        )
        act_tile_start += group_act_tiles
        tile_start = tile_end


@cute.jit
def _combine_swiglu_bwd_quant_producer_body(
    params: CombineSwigluQuantParams,
    mWorkCounter: cute.Tensor,
    mDoneCounter: cute.Tensor,
    mDz: cute.Tensor,
    mH1: cute.Tensor,
    mRowQWords: cute.Tensor,
    mRowScale: cute.Tensor,
    mDxyColQWords: cute.Tensor,
    mDxyColScale: cute.Tensor,
    split_sizes: cute.Tensor,
    G: cutlass.Constexpr[int],
    M: cutlass.Int32,
    K: cutlass.Int32,
    local_rank: cutlass.Int32,
    tile_smem_ptr: cute.Pointer | None = None,
    mScatter: cute.Tensor | None = None,
    col_done_counter_offset: cutlass.Int32 | None = None,
    signal_col_tile_done: cutlass.Constexpr[bool] = False,
) -> None:
    del M, tile_smem_ptr
    tidx, _, _ = cute.arch.thread_idx()
    quant_lane = tidx - cutlass.Int32(
        params.COMBINE_SWIGLU_QUANT_FIRST_WARP * params.THREADS_PER_WARP
    )
    warp_lane = quant_lane % cutlass.Int32(params.THREADS_PER_WARP)
    source_scale_cols = (K // cutlass.Int32(2)) // cutlass.Int32(params.sf_vec_size)
    col_work_tiles = ceil_div(
        source_scale_cols,
        cutlass.Int32(params.COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE),
    )
    row_blocks_total = _activation_buffer_rows(
        split_sizes,
        G,
    ) // cutlass.Int32(params.sf_vec_size)
    ACT_BLOCK_SIZE: cutlass.Constexpr[int] = (
        params.BLOCK_SIZE_N if params.SWAP_AB else params.BLOCK_SIZE_M
    )
    if cutlass.const_expr(ACT_BLOCK_SIZE % params.sf_vec_size != 0):
        raise ValueError(
            "combine swiglu bwd quant act tile must be a multiple of sf_vec_size"
        )
    row_blocks_per_act_tile: cutlass.Constexpr[int] = (
        ACT_BLOCK_SIZE // params.sf_vec_size
    )

    tile_idx_base = _combine_swiglu_quant_fetch_warp_tile_batch(
        mWorkCounter,
        warp_lane,
        params.COMBINE_SWIGLU_WORK_TILES_PER_FETCH,
    )
    tile_idx = tile_idx_base
    tile_batch_end = tile_idx_base + cutlass.Int32(
        params.COMBINE_SWIGLU_WORK_TILES_PER_FETCH
    )
    tile_start = cutlass.Int32(0)
    act_tile_start = cutlass.Int32(0)
    start_m = cutlass.Int32(0)
    for g in cutlass.range(G, unroll=1):
        m_size = cutlass.Int32(split_sizes[g])
        group_row_blocks = m_size // cutlass.Int32(params.sf_vec_size)
        group_tiles = group_row_blocks * col_work_tiles
        group_act_tiles = ceil_div(m_size, cutlass.Int32(ACT_BLOCK_SIZE))
        tile_end = tile_start + group_tiles

        while (tile_idx >= tile_start) and (tile_idx < tile_end):
            local_tile = tile_idx - tile_start
            local_row_block = local_tile // col_work_tiles
            col_work_tile = local_tile - local_row_block * col_work_tiles
            token_off = _remap_m_tile_idx(
                cutlass.Int32(0),
                group_act_tiles,
                local_rank,
                params.world_size,
            )
            local_row_block = (
                local_row_block + token_off * cutlass.Int32(row_blocks_per_act_tile)
            ) % group_row_blocks
            col_first_block = col_work_tile * cutlass.Int32(
                params.COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE
            )
            row_block = start_m // cutlass.Int32(params.sf_vec_size) + local_row_block
            row_start = start_m + local_row_block * cutlass.Int32(params.sf_vec_size)
            act_tile_slot = act_tile_start + local_row_block // cutlass.Int32(
                row_blocks_per_act_tile
            )

            col_block = col_first_block
            if col_block < source_scale_cols:
                _combine_swiglu_bwd_quantize_tile(
                    params,
                    warp_lane,
                    mDoneCounter,
                    mDz,
                    mH1,
                    mRowQWords,
                    mRowScale,
                    mDxyColQWords,
                    mDxyColScale,
                    act_tile_slot,
                    row_start,
                    row_block,
                    col_block,
                    row_blocks_total,
                    K,
                    mScatter,
                )
                if cutlass.const_expr(signal_col_tile_done):
                    _mega_combine_quant_signal_col_tile_done(
                        mDoneCounter,
                        col_done_counter_offset,
                        g * col_work_tiles + col_work_tile,
                        warp_lane,
                    )

            tile_idx += cutlass.Int32(1)
            if tile_idx == tile_batch_end:
                tile_idx_base = _combine_swiglu_quant_fetch_warp_tile_batch(
                    mWorkCounter,
                    warp_lane,
                    params.COMBINE_SWIGLU_WORK_TILES_PER_FETCH,
                )
                tile_idx = tile_idx_base
                tile_batch_end = tile_idx_base + cutlass.Int32(
                    params.COMBINE_SWIGLU_WORK_TILES_PER_FETCH
                )

        start_m += m_size
        act_tile_start += group_act_tiles
        tile_start = tile_end


@cute.jit
def _combine_epilog_get_scatter_ptr(
    tidx: cutlass.Int32,
    sC: cute.Tensor,
    mScatter: cute.Tensor,
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    mma_m_idx: cutlass.Constexpr[int],
    mma_n_idx: cutlass.Constexpr[int],
    subtile_idx: cutlass.Constexpr[int],
    NUM_MMA_ATOMS_M: cutlass.Constexpr[int],
    NUM_MMA_ATOMS_N: cutlass.Constexpr[int],
    cm_start: cutlass.Int32,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    cluster_cta_rank: cutlass.Int32,
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    EPILOG_WG_THREADS: cutlass.Constexpr[int],
    NUM_CTAS: cutlass.Constexpr[int],
    SWAP_AB: cutlass.Constexpr[bool],
):
    EPILOGUE_M: cutlass.Constexpr[int] = cute.size(sC.shape, mode=[0])
    EPILOGUE_N: cutlass.Constexpr[int] = cute.size(sC.shape, mode=[1])
    ATOM_M: cutlass.Constexpr[int] = BLOCK_SIZE_M // NUM_MMA_ATOMS_M
    ATOM_CTA_M: cutlass.Constexpr[int] = ATOM_M // NUM_CTAS
    N_ATOM_SPAN: cutlass.Constexpr[int] = BLOCK_SIZE_N // NUM_MMA_ATOMS_N

    if cutlass.const_expr(SWAP_AB):
        TOKEN_DIM: cutlass.Constexpr[int] = EPILOGUE_N
    else:
        TOKEN_DIM: cutlass.Constexpr[int] = EPILOGUE_M
    if cutlass.const_expr(TOKEN_DIM > EPILOG_WG_THREADS):
        raise ValueError("fused combine scatter pointer tile exceeds epilogue WG")

    peer_base_i64 = cutlass.Int64(0)
    pointer_lane_active = tidx < cutlass.Int32(TOKEN_DIM)
    if pointer_lane_active:
        local_token = tidx
        if cutlass.const_expr(SWAP_AB):
            token_global = (
                tile_n_idx * cutlass.Int32(BLOCK_SIZE_N)
                + cutlass.Int32(mma_n_idx * N_ATOM_SPAN)
                + cutlass.Int32(subtile_idx * EPILOGUE_N)
                + local_token
            )
            token_limit = n_size
        else:
            token_global = (
                tile_m_idx * cutlass.Int32(BLOCK_SIZE_M)
                + cutlass.Int32(mma_m_idx * ATOM_M)
                + cluster_cta_rank * cutlass.Int32(ATOM_CTA_M)
                + local_token
            )
            token_limit = m_size

        if token_global < token_limit:
            peer_base_i64 = mScatter[cm_start + token_global]
    return peer_base_i64, pointer_lane_active


@cute.jit
def _apply_epilog_global_scale_inv(
    tRS_rAcc: cute.Tensor,
    tRS_cAcc: cute.Tensor,
    a_scales_smem: cute.Tensor,
    b_scale: cutlass.Float32,
    mma_m_idx: cutlass.Constexpr[int],
    mma_n_idx: cutlass.Constexpr[int],
    atom_m: cutlass.Constexpr[int],
    atom_n: cutlass.Constexpr[int],
    SWAP_AB: cutlass.Constexpr[bool],
) -> None:
    assert cute.size(tRS_rAcc) % 2 == 0
    for value_idx in cutlass.range(0, cute.size(tRS_rAcc), 2, unroll_full=True):
        a_scales: list = []
        for pair_idx in cutlass.range_constexpr(2):
            coord = tRS_cAcc[value_idx + pair_idx]
            if cutlass.const_expr(SWAP_AB):
                row_in_tile = cutlass.Int32(mma_n_idx * atom_n) + coord[1]
            else:
                row_in_tile = cutlass.Int32(mma_m_idx * atom_m) + coord[0]
            a_scales.append(a_scales_smem[row_in_tile])
        # Preserve reference rounding: per-row A scale, then per-group B scale.
        acc_pair = cute.arch.mul_packed_f32x2(
            (tRS_rAcc[value_idx], tRS_rAcc[value_idx + 1]),
            (a_scales[0], a_scales[1]),
            rnd="rn",
        )
        tRS_rAcc[value_idx], tRS_rAcc[value_idx + 1] = cute.arch.mul_packed_f32x2(
            acc_pair, (b_scale, b_scale), rnd="rn"
        )


@cute.jit
def _mega_combine_quant_signal_col_tile_done(
    mDoneCounter: cute.Tensor,
    col_done_counter_offset: cutlass.Int32,
    col_done_slot: cutlass.Int32,
    warp_lane: cutlass.Int32,
) -> None:
    # Quant stores use the generic proxy while the downstream GEMM
    # consumes them via TMA (async proxy); fence.proxy.async is
    # thread-local, so every producer lane must publish before the
    # warp-synchronized release below. Same protocol as
    # _mega_dispatch_quant_signal_group_tile_done and the dispatch-quant
    # signal in dist_blockscaled_grouped_gemm.py — a consumer-side proxy
    # fence cannot publish another thread's writes across the
    # generic->async boundary.
    cute.arch.fence_proxy("async.global")
    cute.arch.fence_acq_rel_gpu()
    cute.arch.sync_warp()
    if warp_lane == cutlass.Int32(0):
        counter_ptr = cute.recast_ptr(
            mDoneCounter.iterator + col_done_counter_offset + col_done_slot,
            dtype=cutlass.Uint32,
        )
        cute.arch.atomic_add(
            counter_ptr,
            cutlass.Uint32(1),
            sem="release",
            scope="gpu",
        )


@cute.jit
def _mega_forward_h2_tile_wait(
    mDoneCounter: cute.Tensor,
    row_done_counter_offset: cutlass.Int32,
    act_tile_idx: cutlass.Int32,
    act_tile_start_per_group: cutlass.Int32,
    act_rows: cutlass.Int32,
    h2_dim: cutlass.Int32,
    ACT_BLOCK_SIZE: cutlass.Constexpr[int],
    COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    format: cutlass.Constexpr,
    sf_vec_size: cutlass.Constexpr[int],
) -> None:
    tile_row_start = act_tile_idx * cutlass.Int32(ACT_BLOCK_SIZE)
    valid_rows = act_rows - tile_row_start
    if valid_rows > cutlass.Int32(ACT_BLOCK_SIZE):
        valid_rows = cutlass.Int32(ACT_BLOCK_SIZE)
    valid_row_blocks = valid_rows // cutlass.Int32(sf_vec_size)
    col_work_tiles = ceil_div(
        h2_dim // cutlass.Int32(sf_vec_size),
        cutlass.Int32(COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE),
    )
    expected = valid_row_blocks * col_work_tiles
    if cutlass.const_expr(format is NVFP4):
        expected = valid_rows
    ptr = cute.recast_ptr(
        mDoneCounter.iterator
        + row_done_counter_offset
        + act_tile_start_per_group
        + act_tile_idx,
        dtype=cutlass.Uint32,
    )
    _wait_counter_at_least(ptr, cutlass.Uint32(expected))


@cute.jit
def _mega_combine_quant_dgrad_tile_wait(
    mDoneCounter: cute.Tensor,
    act_tile_idx: cutlass.Int32,
    act_tile_start_per_group: cutlass.Int32,
    act_rows: cutlass.Int32,
    source_dim: cutlass.Int32,
    ACT_BLOCK_SIZE: cutlass.Constexpr[int],
    COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
) -> None:
    tile_row_start = act_tile_idx * cutlass.Int32(ACT_BLOCK_SIZE)
    valid_rows = act_rows - tile_row_start
    if valid_rows > cutlass.Int32(ACT_BLOCK_SIZE):
        valid_rows = cutlass.Int32(ACT_BLOCK_SIZE)
    valid_row_blocks = valid_rows // cutlass.Int32(sf_vec_size)
    col_work_tiles = ceil_div(
        source_dim // cutlass.Int32(sf_vec_size),
        cutlass.Int32(COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE),
    )
    valid_quant_tiles = valid_row_blocks * col_work_tiles
    ptr = cute.recast_ptr(
        mDoneCounter.iterator + act_tile_start_per_group + act_tile_idx,
        dtype=cutlass.Uint32,
    )
    _wait_counter_at_least(ptr, cutlass.Uint32(valid_quant_tiles))


@cute.jit
def _mega_combine_quant_wgrad_feature_tile_wait(
    mDoneCounter: cute.Tensor,
    col_done_counter_offset: cutlass.Int32,
    g: cutlass.Int32,
    feature_tile_idx: cutlass.Int32,
    feature_tile_elems: cutlass.Constexpr[int],
    source_dim: cutlass.Int32,
    group_row_blocks: cutlass.Int32,
    COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
    sf_vec_size: cutlass.Constexpr[int],
) -> None:
    feature_start = feature_tile_idx * cutlass.Int32(feature_tile_elems)
    feature_end = feature_start + cutlass.Int32(feature_tile_elems)
    col_work_tiles = ceil_div(
        source_dim // cutlass.Int32(sf_vec_size),
        cutlass.Int32(COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE),
    )
    for half in cutlass.range_constexpr(2):
        half_start = cutlass.Int32(half) * source_dim
        half_end = half_start + source_dim
        intersect_start = feature_start
        if intersect_start < half_start:
            intersect_start = half_start
        intersect_end = feature_end
        if intersect_end > half_end:
            intersect_end = half_end
        if intersect_start < intersect_end:
            source_start = intersect_start - half_start
            source_end = intersect_end - half_start
            first_scale_col = source_start // cutlass.Int32(sf_vec_size)
            last_scale_col_excl = ceil_div(
                source_end,
                cutlass.Int32(sf_vec_size),
            )
            first_col_work_tile = first_scale_col // cutlass.Int32(
                COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE
            )
            last_col_work_tile_excl = ceil_div(
                last_scale_col_excl,
                cutlass.Int32(COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE),
            )
            for col_work_tile in cutlass.range(
                first_col_work_tile,
                last_col_work_tile_excl,
                unroll=1,
            ):
                _mega_dispatch_quant_wgrad_col_tile_wait(
                    mDoneCounter=mDoneCounter,
                    col_done_counter_offset=col_done_counter_offset,
                    g=g,
                    col_work_tile=col_work_tile,
                    group_row_blocks=group_row_blocks,
                    col_work_tiles=col_work_tiles,
                )


@cute.jit
def _mega_combine_wait_quant_ready(
    wait_axis: cutlass.Constexpr[int],
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    dgrad_act_off_tiles: cutlass.Int32,
    g: cutlass.Int32,
    split_sizes: cute.Tensor,
    m_size: cutlass.Int32,
    n_size: cutlass.Int32,
    dxy_dim: cutlass.Int32,
    mDoneCounter: cute.Tensor,
    col_done_counter_offset: cutlass.Int32,
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    SWAP_AB: cutlass.Constexpr[bool],
    sf_vec_size: cutlass.Constexpr[int],
    COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int],
) -> None:
    if cutlass.const_expr(SWAP_AB):
        out_m_tile = tile_n_idx
        OUT_M_BLOCK: cutlass.Constexpr[int] = BLOCK_SIZE_N
        out_m_size = n_size
    else:
        out_m_tile = tile_m_idx
        OUT_M_BLOCK: cutlass.Constexpr[int] = BLOCK_SIZE_M
        out_m_size = m_size
    source_dim = dxy_dim // cutlass.Int32(2)
    if cutlass.const_expr(wait_axis == 0):
        _mega_combine_quant_dgrad_tile_wait(
            mDoneCounter=mDoneCounter,
            act_tile_idx=out_m_tile,
            act_tile_start_per_group=dgrad_act_off_tiles,
            act_rows=out_m_size,
            source_dim=source_dim,
            ACT_BLOCK_SIZE=OUT_M_BLOCK,
            COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE=COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE,
            sf_vec_size=sf_vec_size,
        )
    else:
        group_row_blocks = cutlass.Int32(split_sizes[g]) // cutlass.Int32(sf_vec_size)
        _mega_combine_quant_wgrad_feature_tile_wait(
            mDoneCounter=mDoneCounter,
            col_done_counter_offset=col_done_counter_offset,
            g=g,
            feature_tile_idx=out_m_tile,
            feature_tile_elems=OUT_M_BLOCK,
            source_dim=source_dim,
            group_row_blocks=group_row_blocks,
            COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE=COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE,
            sf_vec_size=sf_vec_size,
        )
