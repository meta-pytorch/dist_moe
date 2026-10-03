# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side per-tile TMA-load and MMA-consumer bodies of the block-scaled
grouped GEMM, shared by the dist/mega/chunked kernels.

Free ``@cute.jit`` functions hoisted out of ``BlockScaledGroupedGemmKernel``;
compile-time attributes are passed explicitly as ``cutlass.Constexpr``
parameters (the ``swiglu_epilogue.py`` convention).
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import (
    tcgen05,
)

from . import sm103_blockscaled_helpers as sm103
from .activation_buffer import (
    GEMM_OPERAND_OFFSET_COUNT,
)
from .config import (
    uses_paged_blockscaled_scale_rows,
)
from .grouped_gemm import (
    _WGRAD,
)
from .grouped_gemm_kernel import (
    _activation_buffer_rows,
    _BAR_EPILOG_SYNC,
    _clear_tma_oob_prefetch_bit,
    _get_group_sizes,
    _make_tensor_for_tensormap_update,
    _require_valid_activation_buffer_range,
)
from .params import (
    BlockscaledTensormapParams,
    BlockscaledTensormapProblem,
    GroupedGemmMmaPipeline,
    GroupedGemmPipelineSync,
    GroupedGemmTmaPipeline,
)
from .tile_scheduler import (
    _get_bufidx_phase,
)


@cute.jit
def _tma_wait_and_arm_expect_tx(
    empty_mbar: cute.Pointer,
    full_mbar: cute.Pointer,
    phase: cutlass.Int32,
    cluster_cta_rank: cutlass.Int32,
    num_tma_load_bytes: cutlass.Constexpr[int],
    NUM_CTAS: cutlass.Constexpr[int],
) -> None:
    """Wait for the consumer-empty barrier, then arm the full barrier with
    the expected TMA byte count (one elected lane; CTA 0 only for 2-CTA)."""
    cute.arch.mbarrier_wait(empty_mbar, phase ^ 1)
    if cutlass.const_expr(NUM_CTAS == 2):
        if cluster_cta_rank == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_mbar,
                    num_tma_load_bytes,
                )
    else:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                full_mbar,
                num_tma_load_bytes,
            )


@cute.jit
def _blockscaled_tma_wait_and_arm(
    pipeline: GroupedGemmTmaPipeline,
    sync: GroupedGemmPipelineSync,
    buf: cutlass.Int32,
    phase: cutlass.Int32,
    NUM_CTAS: cutlass.Constexpr[int],
) -> None:
    _tma_wait_and_arm_expect_tx(
        sync.ab_empty_mbar + buf,
        sync.ab_full_mbar + buf,
        phase,
        sync.cluster_cta_rank,
        num_tma_load_bytes=pipeline.num_tma_load_bytes,
        NUM_CTAS=NUM_CTAS,
    )


@cute.jit
def _blockscaled_tma_copy_weights(
    pipeline: GroupedGemmTmaPipeline,
    sync: GroupedGemmPipelineSync,
    buf: cutlass.Int32,
    kk: cutlass.Int32,
    b_axis: cutlass.Int32,
    sfb_axis: cutlass.Int32,
    b_l: cutlass.Int32,
    sfb_l: cutlass.Int32,
    tma_desc_ptrs: tuple,
    use_device_tensormaps: cutlass.Constexpr[bool],
) -> None:
    (
        _,
        b_full_mcast_mask,
        _,
        sfb_full_mcast_mask,
    ) = sync.tma_mcast_masks
    if cutlass.const_expr(use_device_tensormaps):
        _, b_desc_ptr, _, sfb_desc_ptr = tma_desc_ptrs
        cute.copy(
            pipeline.tma_atom_sfb,
            pipeline.gSFB[(None, sfb_axis, kk, sfb_l)],
            pipeline.sSFB[(None, buf)],
            tma_bar_ptr=sync.ab_full_mbar + buf,
            mcast_mask=sfb_full_mcast_mask,
            tma_desc_ptr=sfb_desc_ptr,
        )
        cute.copy(
            pipeline.tma_atom_b,
            pipeline.gB[(None, b_axis, kk, b_l)],
            pipeline.sB[(None, buf)],
            tma_bar_ptr=sync.ab_full_mbar + buf,
            mcast_mask=b_full_mcast_mask,
            tma_desc_ptr=b_desc_ptr,
        )
    else:
        cute.copy(
            pipeline.tma_atom_sfb,
            pipeline.gSFB[(None, sfb_axis, kk, sfb_l)],
            pipeline.sSFB[(None, buf)],
            tma_bar_ptr=sync.ab_full_mbar + buf,
            mcast_mask=sfb_full_mcast_mask,
        )
        cute.copy(
            pipeline.tma_atom_b,
            pipeline.gB[(None, b_axis, kk, b_l)],
            pipeline.sB[(None, buf)],
            tma_bar_ptr=sync.ab_full_mbar + buf,
            mcast_mask=b_full_mcast_mask,
        )


@cute.jit
def _blockscaled_tma_issue_weights_ahead(
    pipeline: GroupedGemmTmaPipeline,
    sync: GroupedGemmPipelineSync,
    weights_ahead: cutlass.Int32,
    accum_cnt_smem: cutlass.Int32,
    tile_axes: tuple,
    group_axes: tuple,
    tma_desc_ptrs: tuple,
    use_device_tensormaps: cutlass.Constexpr[bool],
    NUM_CTAS: cutlass.Constexpr[int],
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
) -> None:
    """DISPATCH decode prologue: the B/SFB (weight) operands depend only on
    the group tensormap, not on quant-gathered activations, so issue up to a
    ring's worth of their loads BEFORE the per-tile gather wait. The stages
    are armed with the full expect_tx here; ``_blockscaled_tma_load_tile``
    back-fills A/SFA into them after the wait (a stage completes only when
    all four copies land, so consumer ordering is unchanged). 1-CTA only."""
    _, b_axis, _, sfb_axis = tile_axes
    _, b_l, _, sfb_l = group_axes

    for kk in cutlass.range(weights_ahead, unroll=1):
        buf, phase = _get_bufidx_phase(accum_cnt_smem + kk, NUM_SMEM_BUFFERS)
        _blockscaled_tma_wait_and_arm(
            pipeline,
            sync,
            buf,
            phase,
            NUM_CTAS=NUM_CTAS,
        )
        _blockscaled_tma_copy_weights(
            pipeline,
            sync,
            buf,
            kk,
            b_axis,
            sfb_axis,
            b_l,
            sfb_l,
            tma_desc_ptrs,
            use_device_tensormaps,
        )


@cute.jit
def _blockscaled_tma_load_tile(
    pipeline: GroupedGemmTmaPipeline,
    sync: GroupedGemmPipelineSync,
    num_k_tiles: cutlass.Int32,
    accum_cnt_smem: cutlass.Int32,
    tile_axes: tuple,
    group_axes: tuple,
    tma_desc_ptrs: tuple,
    use_device_tensormaps: cutlass.Constexpr[bool],
    KLOOP_UNROLL: cutlass.Constexpr[int],
    NUM_CTAS: cutlass.Constexpr[int],
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
    weights_ahead: cutlass.Int32,
    WEIGHTS_AHEAD: cutlass.Constexpr[bool] = False,
) -> cutlass.Int32:
    a_axis, b_axis, sfa_axis, sfb_axis = tile_axes
    a_l, b_l, sfa_l, sfb_l = group_axes
    (
        a_full_mcast_mask,
        b_full_mcast_mask,
        sfa_full_mcast_mask,
        sfb_full_mcast_mask,
    ) = sync.tma_mcast_masks
    if cutlass.const_expr(use_device_tensormaps):
        a_desc_ptr, b_desc_ptr, sfa_desc_ptr, sfb_desc_ptr = tma_desc_ptrs

    # ``WEIGHTS_AHEAD`` compilations (DISPATCH decode) back-fill A/SFA into
    # the stages pre-armed by ``_blockscaled_tma_issue_weights_ahead`` and
    # only run the full issue for the remaining stages. Every other
    # compilation keeps the original loop verbatim: under CuTeDSL 4.6.1 even
    # an equivalent restructuring of this hot loop reschedules prefill
    # kernels measurably (+1.7% on staged MXFP8 at 16K tokens/rank).
    if cutlass.const_expr(WEIGHTS_AHEAD):
        for kk in cutlass.range(weights_ahead, unroll=1):
            buf, phase = _get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
            if cutlass.const_expr(use_device_tensormaps):
                cute.copy(
                    pipeline.tma_atom_sfa,
                    pipeline.gSFA[(None, sfa_axis, kk, sfa_l)],
                    pipeline.sSFA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=sfa_full_mcast_mask,
                    tma_desc_ptr=sfa_desc_ptr,
                )
                cute.copy(
                    pipeline.tma_atom_a,
                    pipeline.gA[(None, a_axis, kk, a_l)],
                    pipeline.sA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=a_full_mcast_mask,
                    tma_desc_ptr=a_desc_ptr,
                )
            else:
                cute.copy(
                    pipeline.tma_atom_sfa,
                    pipeline.gSFA[(None, sfa_axis, kk, sfa_l)],
                    pipeline.sSFA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=sfa_full_mcast_mask,
                )
                cute.copy(
                    pipeline.tma_atom_a,
                    pipeline.gA[(None, a_axis, kk, a_l)],
                    pipeline.sA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=a_full_mcast_mask,
                )
            accum_cnt_smem += cutlass.Int32(1)
        num_rest_k_tiles = num_k_tiles - weights_ahead
        for kk0 in cutlass.range(num_rest_k_tiles, unroll=1):
            kk = kk0 + weights_ahead
            buf, phase = _get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
            _blockscaled_tma_wait_and_arm(pipeline, sync, buf, phase, NUM_CTAS=NUM_CTAS)
            if cutlass.const_expr(use_device_tensormaps):
                cute.copy(
                    pipeline.tma_atom_sfa,
                    pipeline.gSFA[(None, sfa_axis, kk, sfa_l)],
                    pipeline.sSFA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=sfa_full_mcast_mask,
                    tma_desc_ptr=sfa_desc_ptr,
                )
                cute.copy(
                    pipeline.tma_atom_a,
                    pipeline.gA[(None, a_axis, kk, a_l)],
                    pipeline.sA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=a_full_mcast_mask,
                    tma_desc_ptr=a_desc_ptr,
                )
            else:
                cute.copy(
                    pipeline.tma_atom_sfa,
                    pipeline.gSFA[(None, sfa_axis, kk, sfa_l)],
                    pipeline.sSFA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=sfa_full_mcast_mask,
                )
                cute.copy(
                    pipeline.tma_atom_a,
                    pipeline.gA[(None, a_axis, kk, a_l)],
                    pipeline.sA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=a_full_mcast_mask,
                )
            _blockscaled_tma_copy_weights(
                pipeline,
                sync,
                buf,
                kk,
                b_axis,
                sfb_axis,
                b_l,
                sfb_l,
                tma_desc_ptrs,
                use_device_tensormaps,
            )
            accum_cnt_smem += cutlass.Int32(1)
    else:
        for kk in cutlass.range(num_k_tiles, unroll=KLOOP_UNROLL):
            buf, phase = _get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
            cute.arch.mbarrier_wait(sync.ab_empty_mbar + buf, phase ^ 1)
            if cutlass.const_expr(NUM_CTAS == 2):
                if sync.cluster_cta_rank == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            sync.ab_full_mbar + buf,
                            pipeline.num_tma_load_bytes,
                        )
            else:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        sync.ab_full_mbar + buf,
                        pipeline.num_tma_load_bytes,
                    )

            if cutlass.const_expr(use_device_tensormaps):
                cute.copy(
                    pipeline.tma_atom_sfa,
                    pipeline.gSFA[(None, sfa_axis, kk, sfa_l)],
                    pipeline.sSFA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=sfa_full_mcast_mask,
                    tma_desc_ptr=sfa_desc_ptr,
                )
                cute.copy(
                    pipeline.tma_atom_sfb,
                    pipeline.gSFB[(None, sfb_axis, kk, sfb_l)],
                    pipeline.sSFB[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=sfb_full_mcast_mask,
                    tma_desc_ptr=sfb_desc_ptr,
                )
                cute.copy(
                    pipeline.tma_atom_a,
                    pipeline.gA[(None, a_axis, kk, a_l)],
                    pipeline.sA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=a_full_mcast_mask,
                    tma_desc_ptr=a_desc_ptr,
                )
                cute.copy(
                    pipeline.tma_atom_b,
                    pipeline.gB[(None, b_axis, kk, b_l)],
                    pipeline.sB[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=b_full_mcast_mask,
                    tma_desc_ptr=b_desc_ptr,
                )
            else:
                cute.copy(
                    pipeline.tma_atom_sfa,
                    pipeline.gSFA[(None, sfa_axis, kk, sfa_l)],
                    pipeline.sSFA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=sfa_full_mcast_mask,
                )
                cute.copy(
                    pipeline.tma_atom_sfb,
                    pipeline.gSFB[(None, sfb_axis, kk, sfb_l)],
                    pipeline.sSFB[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=sfb_full_mcast_mask,
                )
                cute.copy(
                    pipeline.tma_atom_a,
                    pipeline.gA[(None, a_axis, kk, a_l)],
                    pipeline.sA[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    pipeline.tma_atom_b,
                    pipeline.gB[(None, b_axis, kk, b_l)],
                    pipeline.sB[(None, buf)],
                    tma_bar_ptr=sync.ab_full_mbar + buf,
                    mcast_mask=b_full_mcast_mask,
                )
            accum_cnt_smem += cutlass.Int32(1)

    return accum_cnt_smem


@cute.jit
def _blockscaled_mma_consumer_tile(  # noqa: C901
    pipeline: GroupedGemmMmaPipeline,
    sync: GroupedGemmPipelineSync,
    num_k_tiles: cutlass.Int32,
    accum_cnt_tile: cutlass.Int32,
    accum_cnt_smem: cutlass.Int32,
    KLOOP_UNROLL: cutlass.Constexpr[int],
    NUM_CTAS: cutlass.Constexpr[int],
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
    NUM_TMEM_BUFFERS: cutlass.Constexpr[int],
    OVERLAPPING_ACCUM: cutlass.Constexpr[bool],
    cta_group: cutlass.Constexpr,
    sf_dtype: cutlass.Constexpr[type[cutlass.Numeric]],
) -> cutlass.Int32:
    tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
        _mainloop_s2t_copy_and_partition(
            pipeline.sSFA, pipeline.tCtSFA, cta_group, sf_dtype
        )
    )
    tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
        _mainloop_s2t_copy_and_partition(
            pipeline.sSFB, pipeline.tCtSFB_copy, cta_group, sf_dtype
        )
    )
    tCtSFB_field = pipeline.tCtSFB
    tmem_buf, tmem_phase = _get_bufidx_phase(accum_cnt_tile, NUM_TMEM_BUFFERS)
    if sync.pred_cta0:
        cute.arch.mbarrier_wait(sync.tmem_empty_mbar + tmem_buf, tmem_phase ^ 1)
        if cutlass.const_expr(OVERLAPPING_ACCUM):
            cute.arch.mbarrier_wait(
                sync.cross_seam_mbar,
                (accum_cnt_tile - cutlass.Int32(1)) & 1,
            )

    for kk in cutlass.range(num_k_tiles, unroll=KLOOP_UNROLL):
        smem_buf, smem_phase = _get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
        if sync.pred_cta0:
            cute.arch.mbarrier_wait(sync.ab_full_mbar + smem_buf, smem_phase)
            s2t_stage_coord = (None, None, None, None, smem_buf)
            cute.copy(
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t[s2t_stage_coord],
                tCtSFA_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t[s2t_stage_coord],
                tCtSFB_compact_s2t,
            )
            tCtAcc_slot = pipeline.tCtAcc_base[(None, None, None, tmem_buf)]
            num_kblocks = cute.size(pipeline.tCrA, mode=[2])
            NUM_MMA_ATOMS_M = cute.size(tCtAcc_slot.shape, mode=[1])
            NUM_MMA_ATOMS_N = cute.size(tCtAcc_slot.shape, mode=[2])
            for kblock_idx in cutlass.range_constexpr(num_kblocks):
                tCrA_kblock = pipeline.tCrA[(None, None, kblock_idx, smem_buf)]
                tCrB_kblock = pipeline.tCrB[(None, None, kblock_idx, smem_buf)]
                for mma_m_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_M):
                    tCrA_atom = cute.local_tile(
                        tCrA_kblock,
                        (tCrA_kblock.shape[0], 1),
                        (0, mma_m_idx),
                    )
                    if cutlass.const_expr(NUM_MMA_ATOMS_M > 1):
                        sfa_iter = pipeline.tCtSFA[
                            (None, mma_m_idx, kblock_idx)
                        ].iterator
                    else:
                        sfa_iter = pipeline.tCtSFA[(None, None, kblock_idx)].iterator
                    pipeline.tiled_mma.set(tcgen05.Field.SFA, sfa_iter)
                    for mma_n_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_N):
                        tCtAcc_atom = cute.local_tile(
                            tCtAcc_slot,
                            (
                                tCtAcc_slot.shape[0],
                                1,
                                1,
                            ),
                            (0, mma_m_idx, mma_n_idx),
                        )
                        tCrB_atom = cute.local_tile(
                            tCrB_kblock,
                            (tCrB_kblock.shape[0], 1),
                            (0, mma_n_idx),
                        )
                        if cutlass.const_expr(NUM_MMA_ATOMS_N > 1):
                            sfb_iter = tCtSFB_field[
                                (None, mma_n_idx, None, kblock_idx)
                            ].iterator
                        else:
                            sfb_iter = tCtSFB_field[(None, None, kblock_idx)].iterator
                        pipeline.tiled_mma.set(tcgen05.Field.SFB, sfb_iter)
                        if cutlass.const_expr(kblock_idx > 0):
                            pipeline.tiled_mma.set(
                                tcgen05.Field.ACCUMULATE,
                                True,
                            )
                        else:
                            pipeline.tiled_mma.set(
                                tcgen05.Field.ACCUMULATE,
                                kk > 0,
                            )
                        cute.gemm(
                            pipeline.tiled_mma,
                            tCtAcc_atom,
                            tCrA_atom,
                            tCrB_atom,
                            tCtAcc_atom,
                        )
            with cute.arch.elect_one():
                tcgen05.commit(
                    sync.ab_empty_mbar + smem_buf,
                    sync.ab_empty_mcast_mask,
                    cta_group,
                )
        accum_cnt_smem += cutlass.Int32(1)

    if num_k_tiles == 0:
        if cutlass.const_expr(NUM_CTAS == 2):
            if sync.pred_cta0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(sync.tmem_full_mbar + tmem_buf)
                    cute.arch.mbarrier_arrive(
                        sync.tmem_full_mbar + tmem_buf,
                        peer_cta_rank_in_cluster=1,
                    )
        else:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(sync.tmem_full_mbar + tmem_buf)
    elif cutlass.const_expr(NUM_CTAS == 2):
        if sync.pred_cta0:
            with cute.arch.elect_one():
                tcgen05.commit(
                    sync.tmem_full_mbar + tmem_buf,
                    sync.acc_full_mcast_mask,
                    cta_group,
                )
    else:
        with cute.arch.elect_one():
            tcgen05.commit(
                sync.tmem_full_mbar + tmem_buf,
                None,
                cta_group,
            )
    return accum_cnt_smem


@cute.jit
def _sm103_mma_k_tile(
    pipeline: GroupedGemmMmaPipeline,
    tiled_mma: cute.TiledMma,
    sync: GroupedGemmPipelineSync,
    tiled_copy_s2t_sfa,
    tCsSFA_compact_s2t: cute.Tensor,
    tCtSFA_compact_s2t: cute.Tensor,
    tiled_copy_s2t_sfb,
    tCsSFB_compact_s2t: cute.Tensor,
    tCtSFB_compact_s2t: cute.Tensor,
    tCtAcc_slot: cute.Tensor,
    kk: cutlass.Int32,
    accum_cnt_smem: cutlass.Int32,
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
    SF_RING_TILES: cutlass.Constexpr[int],
    cta_group: cutlass.Constexpr,
    sf_vec_size: cutlass.Constexpr[int],
) -> None:
    SF_SEGMENTS: cutlass.Constexpr[int] = sm103.sf_segments(sf_vec_size)
    MMAS_PER_SF_SEGMENT: cutlass.Constexpr[int] = sm103.mmas_per_sf_segment(sf_vec_size)
    SF_KBLOCK_COLS: cutlass.Constexpr[int] = sm103.sf_kblock_cols(sf_vec_size)
    sf_slot_base = (accum_cnt_smem % cutlass.Int32(SF_RING_TILES)) * cutlass.Int32(
        SF_SEGMENTS
    )
    sf_phase = (accum_cnt_smem // cutlass.Int32(SF_RING_TILES)) & cutlass.Int32(1)
    ab_base_count = accum_cnt_smem * cutlass.Int32(sm103.SM103_AB_SEGMENTS)
    for mma_idx in cutlass.range_constexpr(sm103.SM103_MMA_COUNT):
        if cutlass.const_expr(mma_idx % MMAS_PER_SF_SEGMENT == 0):
            sf_slot = sf_slot_base + cutlass.Int32(mma_idx // MMAS_PER_SF_SEGMENT)
            cute.arch.mbarrier_wait(sync.sf_full_mbar + sf_slot, sf_phase)
            s2t_stage_coord = (None, None, None, None, sf_slot)
            cute.copy(
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t[s2t_stage_coord],
                tCtSFA_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t[s2t_stage_coord],
                tCtSFB_compact_s2t,
            )
            with cute.arch.elect_one():
                tcgen05.commit(
                    sync.sf_empty_mbar + sf_slot,
                    sync.ab_empty_mcast_mask,
                    cta_group,
                )
        ab_wait_offset = sm103.SM103_AB_WAIT_OFFSETS[mma_idx]
        if cutlass.const_expr(ab_wait_offset >= 0):
            ab_wait_count = ab_base_count + ab_wait_offset
            ab_wait_buf = ab_wait_count % cutlass.Int32(NUM_SMEM_BUFFERS)
            ab_wait_phase = (
                ab_wait_count // cutlass.Int32(NUM_SMEM_BUFFERS)
            ) & cutlass.Int32(1)
            cute.arch.mbarrier_wait(sync.ab_full_mbar + ab_wait_buf, ab_wait_phase)
        sf_kblock_coord = (
            None,
            None,
            (mma_idx % MMAS_PER_SF_SEGMENT) * SF_KBLOCK_COLS,
        )
        tiled_mma.set(
            tcgen05.Field.SFA,
            pipeline.tCtSFA[sf_kblock_coord].iterator,
        )
        tiled_mma.set(
            tcgen05.Field.SFB,
            pipeline.tCtSFB[sf_kblock_coord].iterator,
        )
        tiled_mma.set(
            tcgen05.Field.ACCUMULATE,
            kk > 0 if mma_idx == 0 else True,
        )
        ab_kblock = sm103.SM103_AB_K_BLOCKS[mma_idx]
        ab_stage_count = ab_base_count + sm103.SM103_AB_STAGES[mma_idx]
        ab_stage = ab_stage_count % cutlass.Int32(NUM_SMEM_BUFFERS)
        ab_next_stage_count = ab_base_count + sm103.SM103_AB_NEXT_STAGES[mma_idx]
        ab_next_stage = ab_next_stage_count % cutlass.Int32(NUM_SMEM_BUFFERS)
        sm103.make_desc_and_call_mma(
            tiled_mma,
            tCtAcc_slot,
            pipeline.tCrA[(None, 0, ab_kblock, ab_stage)],
            pipeline.tCrA[(None, 0, 0, ab_next_stage)],
            pipeline.tCrB[(None, 0, ab_kblock, ab_stage)],
            pipeline.tCrB[(None, 0, 0, ab_next_stage)],
            tCtAcc_slot,
        )
        ab_release_stage = sm103.SM103_AB_RELEASE_OFFSETS[mma_idx]
        if cutlass.const_expr(ab_release_stage >= 0):
            ab_release_count = ab_base_count + ab_release_stage
            ab_release_buf = ab_release_count % cutlass.Int32(NUM_SMEM_BUFFERS)
            with cute.arch.elect_one():
                tcgen05.commit(
                    sync.ab_empty_mbar + ab_release_buf,
                    sync.ab_empty_mcast_mask,
                    cta_group,
                )


@cute.jit
def _sm103_mma_consumer_tile(
    pipeline: GroupedGemmMmaPipeline,
    sync: GroupedGemmPipelineSync,
    num_k_tiles: cutlass.Int32,
    accum_cnt_tile: cutlass.Int32,
    accum_cnt_smem: cutlass.Int32,
    KLOOP_UNROLL: cutlass.Constexpr[int],
    NUM_CTAS: cutlass.Constexpr[int],
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
    NUM_TMEM_BUFFERS: cutlass.Constexpr[int],
    OVERLAPPING_ACCUM: cutlass.Constexpr[bool],
    SF_RING_TILES: cutlass.Constexpr[int],
    cta_group: cutlass.Constexpr,
    sf_dtype: cutlass.Constexpr[type[cutlass.Numeric]],
    sf_vec_size: cutlass.Constexpr[int],
    FRESH_TILED_MMA: cutlass.Constexpr[bool],
) -> cutlass.Int32:
    """SM103 ultra per-tile MMA consumer shared by the standalone and mega
    kernels. The segmented A/B waits and SF SMEM->TMEM staging happen per
    K=96 instruction inside ``_sm103_mma_k_tile``; this shell owns the TMEM
    accumulator slot handshake (identical to the non-ultra shell)."""
    tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
        _mainloop_s2t_copy_and_partition(
            pipeline.sSFA,
            pipeline.tCtSFA_copy,
            cta_group,
            sf_dtype,
            use_sm103_ultra=True,
        )
    )
    tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
        _mainloop_s2t_copy_and_partition(
            pipeline.sSFB,
            pipeline.tCtSFB_copy,
            cta_group,
            sf_dtype,
            use_sm103_ultra=True,
        )
    )
    tmem_buf, tmem_phase = _get_bufidx_phase(accum_cnt_tile, NUM_TMEM_BUFFERS)
    if cutlass.const_expr(FRESH_TILED_MMA):
        # Fresh per-invocation TiledMma: the SFA/SFB/ACCUMULATE set() chain
        # must not escape the calling subtile's trace region (the mega's
        # extra problem_idx / subtile scf.if nesting is not threaded by the
        # tracer, unlike the standalone kernel's flatter loop), so the
        # handle lives and dies here.
        tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(pipeline.tiled_mma.op))
    else:
        tiled_mma = pipeline.tiled_mma
    if sync.pred_cta0:
        cute.arch.mbarrier_wait(sync.tmem_empty_mbar + tmem_buf, tmem_phase ^ 1)
        if cutlass.const_expr(OVERLAPPING_ACCUM):
            cute.arch.mbarrier_wait(
                sync.cross_seam_mbar,
                (accum_cnt_tile - cutlass.Int32(1)) & 1,
            )
    for kk in cutlass.range(num_k_tiles, unroll=KLOOP_UNROLL):
        if sync.pred_cta0:
            tCtAcc_slot = pipeline.tCtAcc_base[(None, 0, 0, tmem_buf)]
            _sm103_mma_k_tile(
                pipeline,
                tiled_mma,
                sync,
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
                tCtAcc_slot,
                kk,
                accum_cnt_smem,
                NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                SF_RING_TILES=SF_RING_TILES,
                cta_group=cta_group,
                sf_vec_size=sf_vec_size,
            )
        accum_cnt_smem += cutlass.Int32(1)
    if num_k_tiles == 0:
        if cutlass.const_expr(NUM_CTAS == 2):
            if sync.pred_cta0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(sync.tmem_full_mbar + tmem_buf)
                    cute.arch.mbarrier_arrive(
                        sync.tmem_full_mbar + tmem_buf,
                        peer_cta_rank_in_cluster=1,
                    )
        else:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(sync.tmem_full_mbar + tmem_buf)
    elif cutlass.const_expr(NUM_CTAS == 2):
        if sync.pred_cta0:
            with cute.arch.elect_one():
                tcgen05.commit(
                    sync.tmem_full_mbar + tmem_buf,
                    sync.acc_full_mcast_mask,
                    cta_group,
                )
    else:
        with cute.arch.elect_one():
            tcgen05.commit(
                sync.tmem_full_mbar + tmem_buf,
                None,
                cta_group,
            )
    return accum_cnt_smem


_BITS_PER_BYTE: int = 8


@cute.jit
def _get_group_sizes_swap_aware(
    split_sizes: cute.Tensor,
    g: cutlass.Int32,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    problem_type: cutlass.Constexpr[int],
    SWAP_AB: cutlass.Constexpr[bool],
):
    """``_get_group_sizes`` with M/N swapped when ``SWAP_AB`` is on, so the
    varlen axis lines up with the operand actually holding the activations
    (the B side under SWAP_AB)."""
    m_size, n_size, k_size = _get_group_sizes(split_sizes, g, M, N, K, problem_type)
    if cutlass.const_expr(SWAP_AB):
        return n_size, m_size, k_size
    return m_size, n_size, k_size


def _sfb_scale_page_rows(block_size_n: int) -> int:
    return ((block_size_n + 127) // 128) * 128


@cute.jit
def _make_sf_tensor_for_tensormap_update(
    per_group_base_ptr: cutlass.Int64,
    sf_dtype: type[cutlass.Numeric],
    problem_shape_mnk: tuple,
    sf_vec_size: cutlass.Constexpr[int],
    k_whole: cutlass.Int32,
    *,
    tensor_index: cutlass.Constexpr[int],  # 0=SFA, 1=SFB
    problem_type: cutlass.Constexpr[int],
    use_sm103_ultra: cutlass.Constexpr[bool],
):
    """Per-group GMEM tensor in the SF atom-tile layout for a tensormap
    update. Caller passes a base ptr + cumulative byte offset so no
    prefix-sum / per-launch device tensor is needed.

    For WGRAD the SF source is a single whole-tensor blocked buffer
    ``tile_atom_to_shape_SF((M_or_N, GM, 1), sf_vec_size)``; the layout
    built here keeps per-group ``k_g`` extents (so the kernel iterates
    only the group's K atoms) but uses the whole-tensor K (``k_whole``)
    for the inter-atom-MN stride. ``k_whole`` is ignored on FPROP/DGRAD
    where each group already has a private buffer.
    """
    gmem_ptr = cute.make_ptr(
        sf_dtype, per_group_base_ptr, cute.AddressSpace.gmem, assumed_align=16
    )
    c1 = cutlass.Int32(1)
    m = problem_shape_mnk[0]
    n = problem_shape_mnk[1]
    k = problem_shape_mnk[2]
    if cutlass.const_expr(problem_type == _WGRAD):
        k_desc = cutlass.max(k, cutlass.Int32(4 * sf_vec_size))
        k_stride_basis = cutlass.max(k_whole, cutlass.Int32(4 * sf_vec_size))
        mn_extent = m if cutlass.const_expr(tensor_index == 0) else n
        return _make_wgrad_sf_view(
            gmem_ptr, mn_extent, k_desc, k_stride_basis, sf_vec_size
        )
    if cutlass.const_expr(use_sm103_ultra):
        mn_extent = m if cutlass.const_expr(tensor_index == 0) else n
        return cute.make_tensor(
            gmem_ptr,
            sm103.make_gmem_layout_sf((mn_extent, k, c1), sf_vec_size),
        )
    if cutlass.const_expr(tensor_index == 0):  # SFA
        sf_layout = blockscaled_utils.tile_atom_to_shape_SF((m, k, c1), sf_vec_size)
    else:  # SFB
        sf_layout = blockscaled_utils.tile_atom_to_shape_SF((n, k, c1), sf_vec_size)
    return cute.make_tensor(gmem_ptr, sf_layout)


@cute.jit
def _make_wgrad_sf_view(
    gmem_ptr,
    mn_extent: cutlass.Int32,
    k_g: cutlass.Int32,
    k_whole: cutlass.Int32,
    sf_vec_size: cutlass.Constexpr[int],
):
    """Build an SF tensor view with per-group K extent + whole-tensor K stride.

    Mirrors the ``cuBLAS`` BlockScaled atom structure produced by
    ``tile_atom_to_shape_SF`` but takes the ``Rest_MN`` stride from
    ``k_whole`` (i.e. whole-tensor K = ``GM`` for the wgrad SF source).
    The per-group ``Rest_K`` extent (ceil-dividing by ``4 * sf_vec_size``)
    bounds TMA's K reads to this group; ``mn_stride_basis`` ensures the
    atom-MN step jumps over the entire ``GM`` K-row, matching how
    qdata's ``a_s1`` stride works.
    """
    # cuBLAS BlockScaled atom (K-major): 128 SF lanes (= 32 outer × 4 inner)
    # along MN, broadcast-of-``sf_vec_size`` × 4 atom-K-cols along K. The
    # atom cosize is 32*16 = 512 bytes (the per-atom-K stride).
    rest_mn_size = (mn_extent + 127) // 128
    atom_k = cutlass.Int32(4 * sf_vec_size)
    rest_k_size = (k_g + atom_k - 1) // atom_k
    mn_stride_basis = (cutlass.Int32(128) * k_whole) // sf_vec_size
    # Match ``tile_atom_to_shape_SF``'s hierarchical structure
    # ``((Atom_MN, Rest_MN), (Atom_K, Rest_K), Rest_L)`` exactly — the TMA
    # atom built at host time encodes the atom-vs-rest split, so a flat
    # 3-tuple per mode would walk the bytes in the wrong order.
    sf_layout = cute.make_layout(
        (
            ((32, 4), rest_mn_size),
            ((sf_vec_size, 4), rest_k_size),
            1,
        ),
        stride=(
            ((16, 4), mn_stride_basis),
            ((0, 1), cutlass.Int32(512)),
            0,
        ),
    )
    return cute.make_tensor(gmem_ptr, sf_layout)


def _mainloop_s2t_copy_and_partition(
    sSF: cute.Tensor,
    tSF: cute.Tensor,
    cta_group: cutlass.Constexpr,
    sf_dtype: cutlass.Constexpr[type[cutlass.Numeric]],
    use_sm103_ultra: cutlass.Constexpr[bool] = False,
) -> tuple:
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    tCtSF_for_copy = tCtSF_compact
    if cutlass.const_expr(use_sm103_ultra):
        # Ultra stages one SF segment at a time: collapse the K-block
        # modes so the copy atom sees a single-segment TMEM target.
        tCtSF_for_copy = cute.make_tensor(
            tCtSF_compact.iterator,
            cute.append(
                cute.append(
                    tCtSF_compact[(None, 0, 0)].layout,
                    cute.make_layout(1),
                ),
                cute.make_layout(1),
            ),
        )
    copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), sf_dtype)
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_for_copy)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t, tCsSF_compact_s2t_
    )
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


@cute.jit
def _prepare_blockscaled_problem_tensormaps(  # noqa: C901
    params: BlockscaledTensormapParams,
    problem: BlockscaledTensormapProblem,
    tensormaps: cute.Tensor,
    tensormap_manager,
    tensormap_smem_ptr_base,
    activation_buffer_operand_offsets,
    activation_buffer_size_bytes: cutlass.Int64,
    use_activation_buffer: cutlass.Constexpr[bool],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    SWAP_AB: cutlass.Constexpr[bool],
    sf_dtype: cutlass.Constexpr[type[cutlass.Numeric]],
    sf_vec_size: cutlass.Constexpr[int],
    ring_b_rows: cutlass.Constexpr[int] = 0,
    use_sm103_ultra: cutlass.Constexpr[bool] = False,
    borrow_slots: cutlass.Constexpr[int] = 0,
    borrow_a_base: cutlass.Int64 = 0,
    borrow_a_row_bytes: cutlass.Int64 = 0,
    borrow_sfa_base: cutlass.Int64 = 0,
    borrow_sfa_row_bytes: cutlass.Int64 = 0,
):
    # With `ring_b_rows`, the activation (B/SFB) operand lives in a
    # schedule-position ring shared by every group: descriptors carry the
    # ring base with no per-group offset and span the ring extent. Only the
    # chunked-mega interleaved forward requests this (SWAP_AB fprop).
    #
    # With `borrow_slots`, the trailing groups are borrowed experts whose
    # weight rows live in the expert-borrow slot buffer instead of the local
    # weight table: their A/SFA descriptors take the `borrow_*` bases (one
    # row per slot) while B/SFB/C keep the split-prefix arithmetic, which
    # covers appended groups' token rows already.
    ring_b_operand: cutlass.Constexpr[bool] = ring_b_rows > 0 and SWAP_AB
    a_base_ptr, b_base_ptr, c_base_ptr, sfa_base_ptr, sfb_base_ptr = params.base_ptrs
    (
        (a_s0, a_s1, a_group_stride),
        (
            b_s0,
            b_s1,
            b_group_stride,
        ),
        (c_s0, c_s1, c_group_stride),
    ) = params.strides
    sfa_row_stride_bytes, sfb_per_group_bytes = params.sf_strides
    elem_size_bytes_a, elem_size_bytes_b, elem_size_bytes_c = params.elem_sizes
    a_dtype, b_dtype, c_dtype = params.dtypes
    M, N, K = problem.mnk
    k_whole = K
    if cutlass.const_expr(problem.problem_type == _WGRAD and use_activation_buffer):
        k_whole = _activation_buffer_rows(problem.split_sizes, problem.groups)
    tensormap_a_smem_ptr = tensormap_smem_ptr_base
    tensormap_b_smem_ptr = tensormap_smem_ptr_base + 16
    tensormap_sfa_smem_ptr = tensormap_smem_ptr_base + 32
    tensormap_sfb_smem_ptr = tensormap_smem_ptr_base + 48
    tensormap_c_smem_ptr = tensormap_smem_ptr_base + 64

    if problem.warp_idx == 0:
        tensormap_manager.init_tensormap_from_atom(
            params.tma_atom_a, tensormap_a_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            params.tma_atom_b, tensormap_b_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            params.tma_atom_sfa, tensormap_sfa_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            params.tma_atom_sfb, tensormap_sfb_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            params.tma_atom_c, tensormap_c_smem_ptr, 0
        )
        _clear_tma_oob_prefetch_bit(tensormap_a_smem_ptr)
        _clear_tma_oob_prefetch_bit(tensormap_b_smem_ptr)
        _clear_tma_oob_prefetch_bit(tensormap_sfa_smem_ptr)
        _clear_tma_oob_prefetch_bit(tensormap_sfb_smem_ptr)
        _clear_tma_oob_prefetch_bit(tensormap_c_smem_ptr)

    tensormap_manager.fence_tensormap_initialization()

    if problem.warp_idx == 0:
        a_off_bytes = cutlass.Int64(0)
        b_off_bytes = cutlass.Int64(0)
        c_off_bytes = cutlass.Int64(0)
        sfa_off_bytes = cutlass.Int64(0)
        sfb_off_bytes = cutlass.Int64(0)

        for prev_g in cutlass.range(problem.groups, unroll=1):
            if prev_g < problem.prepare_g:
                prev_m, prev_n, prev_k = _get_group_sizes_swap_aware(
                    problem.split_sizes,
                    prev_g,
                    M,
                    N,
                    K,
                    problem.problem_type,
                    SWAP_AB,
                )
                if cutlass.const_expr(problem.problem_type == _WGRAD):
                    a_logical_offset = cutlass.Int64(prev_k) * cutlass.Int64(a_s1)
                    b_logical_offset = cutlass.Int64(prev_k) * cutlass.Int64(b_s1)
                    if cutlass.const_expr(a_dtype.width == 4):
                        a_off_bytes += a_logical_offset // 2
                    else:
                        a_off_bytes += a_logical_offset * cutlass.Int64(
                            elem_size_bytes_a
                        )
                    if cutlass.const_expr(b_dtype.width == 4):
                        b_off_bytes += b_logical_offset // 2
                    else:
                        b_off_bytes += b_logical_offset * cutlass.Int64(
                            elem_size_bytes_b
                        )
                    c_off_bytes += c_group_stride * cutlass.Int64(elem_size_bytes_c)
                    # SFA/SFB live in a single whole-tensor blocked
                    # buffer. Each WGRAD group advances by a K-direction
                    # slab rather than the A/B/C logical group stride.
                    group_k_off_bytes = (
                        cutlass.Int64(prev_k)
                        * cutlass.Int64(128)
                        // cutlass.Int64(sf_vec_size)
                    )
                    sfa_off_bytes += group_k_off_bytes
                    sfb_off_bytes += group_k_off_bytes
                elif cutlass.const_expr(SWAP_AB):
                    a_off_bytes += a_group_stride * cutlass.Int64(elem_size_bytes_a)
                    if cutlass.const_expr(ring_b_rows == 0):
                        b_logical_offset = cutlass.Int64(prev_n) * cutlass.Int64(b_s0)
                        if cutlass.const_expr(b_dtype.width == 4):
                            b_off_bytes += b_logical_offset // 2
                        else:
                            b_off_bytes += b_logical_offset * cutlass.Int64(
                                elem_size_bytes_b
                            )
                    c_off_bytes += (
                        cutlass.Int64(prev_n)
                        * cutlass.Int64(c_s1)
                        * cutlass.Int64(elem_size_bytes_c)
                    )
                    sfa_off_bytes += cutlass.Int64(sfb_per_group_bytes)
                    if cutlass.const_expr(ring_b_operand):
                        pass
                    elif cutlass.const_expr(
                        uses_paged_blockscaled_scale_rows(BLOCK_SIZE_N)
                    ):
                        scale_page_rows: cutlass.Constexpr[int] = (
                            (BLOCK_SIZE_N + 127) // 128
                        ) * 128
                        padded_scale_rows = (
                            (cutlass.Int64(prev_n) + cutlass.Int64(BLOCK_SIZE_N - 1))
                            // cutlass.Int64(BLOCK_SIZE_N)
                        ) * cutlass.Int64(scale_page_rows)
                        sfb_off_bytes += padded_scale_rows * cutlass.Int64(
                            sfa_row_stride_bytes
                        )
                    else:
                        sfb_off_bytes += cutlass.Int64(prev_n) * cutlass.Int64(
                            sfa_row_stride_bytes
                        )
                else:
                    a_logical_offset = cutlass.Int64(prev_m) * cutlass.Int64(a_s0)
                    if cutlass.const_expr(a_dtype.width == 4):
                        a_off_bytes += a_logical_offset // 2
                    else:
                        a_off_bytes += a_logical_offset * cutlass.Int64(
                            elem_size_bytes_a
                        )
                    b_off_bytes += b_group_stride * cutlass.Int64(elem_size_bytes_b)
                    c_off_bytes += (
                        cutlass.Int64(prev_m)
                        * cutlass.Int64(c_s0)
                        * cutlass.Int64(elem_size_bytes_c)
                    )
                    sfa_off_bytes += cutlass.Int64(prev_m) * cutlass.Int64(
                        sfa_row_stride_bytes
                    )
                    sfb_off_bytes += cutlass.Int64(sfb_per_group_bytes)

        m_size, n_size, k_size = _get_group_sizes_swap_aware(
            problem.split_sizes,
            problem.prepare_g,
            M,
            N,
            K,
            problem.problem_type,
            SWAP_AB,
        )
        a_desc_base = a_base_ptr + a_off_bytes
        sfa_desc_base = sfa_base_ptr + sfa_off_bytes
        if cutlass.const_expr(borrow_slots > 0):
            first_borrow: cutlass.Constexpr[int] = problem.groups - borrow_slots
            if problem.prepare_g >= cutlass.Int32(first_borrow):
                borrow_row = cutlass.Int64(
                    problem.prepare_g - cutlass.Int32(first_borrow)
                )
                a_desc_base = borrow_a_base + borrow_row * borrow_a_row_bytes
                sfa_desc_base = borrow_sfa_base + borrow_row * borrow_sfa_row_bytes
        real_tensor_a = _make_tensor_for_tensormap_update(
            a_desc_base,
            a_dtype,
            (m_size, n_size, k_size),
            a_s0,
            a_s1,
            tensor_index=0,
            problem_type=problem.problem_type,
            use_sm103_ultra=use_sm103_ultra,
        )
        b_n_size = n_size
        if cutlass.const_expr(ring_b_operand):
            b_n_size = cutlass.Int32(ring_b_rows)
        real_tensor_b = _make_tensor_for_tensormap_update(
            b_base_ptr + b_off_bytes,
            b_dtype,
            (m_size, b_n_size, k_size),
            b_s0,
            b_s1,
            tensor_index=1,
            problem_type=problem.problem_type,
            use_sm103_ultra=use_sm103_ultra,
        )
        real_tensor_c = _make_tensor_for_tensormap_update(
            c_base_ptr + c_off_bytes,
            c_dtype,
            (m_size, n_size, k_size),
            c_s0,
            c_s1,
            tensor_index=2,
            problem_type=problem.problem_type,
            use_sm103_ultra=use_sm103_ultra,
        )
        real_tensor_sfa = _make_sf_tensor_for_tensormap_update(
            sfa_desc_base,
            sf_dtype,
            (m_size, n_size, k_size),
            sf_vec_size,
            k_whole,
            tensor_index=0,
            problem_type=problem.problem_type,
            use_sm103_ultra=use_sm103_ultra,
        )
        sfb_n_size = b_n_size
        if cutlass.const_expr(
            SWAP_AB
            and problem.problem_type != _WGRAD
            and uses_paged_blockscaled_scale_rows(BLOCK_SIZE_N)
        ):
            # The descriptor must span every padded scale page; describing
            # only the 128-row rounding of n_size makes later TMA boxes OOB.
            sfb_n_size = (
                (b_n_size + cutlass.Int32(BLOCK_SIZE_N - 1))
                // cutlass.Int32(BLOCK_SIZE_N)
            ) * cutlass.Int32(_sfb_scale_page_rows(BLOCK_SIZE_N))
        real_tensor_sfb = _make_sf_tensor_for_tensormap_update(
            sfb_base_ptr + sfb_off_bytes,
            sf_dtype,
            (m_size, sfb_n_size, k_size),
            sf_vec_size,
            k_whole,
            tensor_index=1,
            problem_type=problem.problem_type,
            use_sm103_ultra=use_sm103_ultra,
        )
        if cutlass.const_expr(use_activation_buffer):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == cutlass.Int32(0):
                group_offsets = (
                    a_off_bytes,
                    b_off_bytes,
                    c_off_bytes,
                    sfa_off_bytes,
                    sfb_off_bytes,
                )
                tensors = (
                    real_tensor_a,
                    real_tensor_b,
                    real_tensor_c,
                    real_tensor_sfa,
                    real_tensor_sfb,
                )
                has_group_work = (
                    (m_size > cutlass.Int32(0))
                    & (n_size > cutlass.Int32(0))
                    & (k_size > cutlass.Int32(0))
                )
                for operand_idx in cutlass.range_constexpr(GEMM_OPERAND_OFFSET_COUNT):
                    operand_offset = activation_buffer_operand_offsets[operand_idx]
                    if operand_offset >= cutlass.Int64(0):
                        tensor = tensors[operand_idx]
                        # Empty groups keep byte_extent at 0, which makes
                        # _require_valid_activation_buffer_range skip the
                        # range check: their descriptor offsets may alias
                        # byte zero and carry no addressable payload.
                        byte_extent = cutlass.Int64(0)
                        if cute.size(tensor) > 0:
                            if has_group_work:
                                bit_extent = cutlass.Int64(
                                    cute.cosize(tensor.layout)
                                ) * cutlass.Int64(tensor.element_type.width)
                                byte_extent = (
                                    bit_extent + cutlass.Int64(_BITS_PER_BYTE - 1)
                                ) // cutlass.Int64(_BITS_PER_BYTE)
                        _require_valid_activation_buffer_range(
                            byte_offset=operand_offset + group_offsets[operand_idx],
                            byte_extent=byte_extent,
                            activation_buffer_size_bytes=(activation_buffer_size_bytes),
                            warp_scoped_diagnostic=False,
                        )
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(problem.prepare_g, problem.tensormap_base, None)].iterator
        )
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(problem.prepare_g, problem.tensormap_base + 1, None)].iterator
        )
        tensormap_sfa_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(problem.prepare_g, problem.tensormap_base + 2, None)].iterator
        )
        tensormap_sfb_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(problem.prepare_g, problem.tensormap_base + 3, None)].iterator
        )
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(problem.prepare_g, problem.tensormap_base + 4, None)].iterator
        )
        tensormap_manager.update_tensormap(
            (
                real_tensor_a,
                real_tensor_b,
                real_tensor_sfa,
                real_tensor_sfb,
                real_tensor_c,
            ),
            (
                params.tma_atom_a,
                params.tma_atom_b,
                params.tma_atom_sfa,
                params.tma_atom_sfb,
                params.tma_atom_c,
            ),
            (
                tensormap_a_ptr,
                tensormap_b_ptr,
                tensormap_sfa_ptr,
                tensormap_sfb_ptr,
                tensormap_c_ptr,
            ),
            0,
            (
                tensormap_a_smem_ptr,
                tensormap_b_smem_ptr,
                tensormap_sfa_smem_ptr,
                tensormap_sfb_smem_ptr,
                tensormap_c_smem_ptr,
            ),
        )
        tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_b_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_sfa_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_sfb_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_c_ptr)


@cute.jit
def _stage_a_global_scale_inv(
    tidx: cutlass.Int32,
    a_global_scale_inv_ptr: cutlass.Int64,
    a_global_scale_inv_smem_ptr: cute.Pointer,
    split_prefix: cutlass.Int32,
    a_scale_m_size: cutlass.Int32,
    tile_m_idx: cutlass.Int32,
    tile_n_idx: cutlass.Int32,
    cluster_cta_rank: cutlass.Int32,
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    EPILOG_WARP_IDS: cutlass.Constexpr,
    SWAP_AB: cutlass.Constexpr[bool],
    a_global_scale_inv_smem_size: cutlass.Constexpr[int],
    cta_tile_shape_mnk: cutlass.Constexpr,
) -> None:
    num_epilog_threads: cutlass.Constexpr[int] = 32 * len(EPILOG_WARP_IDS)
    num_scales: cutlass.Constexpr[int] = a_global_scale_inv_smem_size
    if cutlass.const_expr(SWAP_AB):
        row_start = tile_n_idx * BLOCK_SIZE_N
    else:
        row_start = tile_m_idx * BLOCK_SIZE_M + cluster_cta_rank * cta_tile_shape_mnk[0]
    a_scale_ptr = cute.make_ptr(
        cutlass.Float32,
        a_global_scale_inv_ptr,
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    a_scales = cute.make_tensor(
        a_scale_ptr + split_prefix,
        cute.make_layout((a_scale_m_size,), stride=(1,)),
    )
    a_scales_smem = cute.make_tensor(
        a_global_scale_inv_smem_ptr,
        cute.make_layout((num_scales,), stride=(1,)),
    )
    for load_iter in cutlass.range_constexpr(
        (num_scales + num_epilog_threads - 1) // num_epilog_threads
    ):
        scale_idx = tidx + load_iter * num_epilog_threads
        if scale_idx < num_scales:
            row_idx = row_start + scale_idx
            scale = cutlass.Float32(1.0)
            if row_idx < a_scale_m_size:
                scale = a_scales[row_idx]
            a_scales_smem[scale_idx] = scale
    cute.arch.barrier(
        barrier_id=_BAR_EPILOG_SYNC,
        number_of_threads=num_epilog_threads,
    )
