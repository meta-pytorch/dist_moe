# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side kernel classes and topology helpers for the chunked Mega block-scaled grouped GEMM."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from . import (
    mega_blockscaled_grouped_gemm_kernel as base,
    sm103_blockscaled_helpers as sm103,
    weight_borrow,
)
from ._dsl_compat import thread_exit
from .activation_buffer import (
    ACTIVATION_A_Q_OFFSET,
    ACTIVATION_A_SCALE_OFFSET,
    ACTIVATION_COL_Q_OFFSET,
    ACTIVATION_COL_SCALE_OFFSET,
    ACTIVATION_SOURCE_X_OFFSET,
    ACTIVATION_SOURCE_Y_OFFSET,
    MEGA_FIRST_GEMM_OFFSET_BASE,
    MEGA_SECOND_GEMM_OFFSET_BASE,
)
from .activation_buffer_kernel import (
    _activation_buffer_col_scale_storage_byte_extent,
    _activation_buffer_row_global_scale_tensor,
    _activation_buffer_row_scale_storage_byte_extent,
    _activation_buffer_tensor_with_byte_extent,
)
from .blockscaled_grouped_gemm import (
    BlockScaledFormatSpec,
    NVFP4,
)
from .dispatch_quant import (
    _load_group_global_scale_inv,
    _wait_counter_at_least,
)
from .gemm_warp_roles import (
    _gemm_mma_warp,
    _gemm_tma_warp,
)
from .grouped_gemm_kernel import (
    _activation_buffer_rows,
    _BAR_EPILOG_SYNC,
    _BAR_FULL_CTA_SYNC,
)
from .params import (
    ceil_div,
    ChunkedGemmWarpParams,
    GroupedGemmPipelineSync,
    GroupedGemmProblem,
    params_from_kernel,
)
from .swiglu_epilogue import (
    _signal_fc13_epilogue,
    _wait_pending_epilog_store,
    MEGA_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD,
    STAGED_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD,
)
from .tile_scheduler import (
    _chunk_offset,
    _get_bufidx_phase,
    _subtile_coords,
    _subtile_is_valid,
    CHUNKED_MEGA_WORK_INFO_FIELDS,
    CHUNKED_MEGA_WORK_INFO_ROLE_EPILOGUE,
    load_chunked_mega_work_info,
    MegaDynamicScheduler,
    stage_expert_metadata,
)

_PIPELINE_CHUNK_ROW_ALIGNMENT = 128


_MAX_PIPELINE_CHUNK_ROWS = 8192


def _validate_pipeline_chunk_rows(chunk_rows: int) -> None:
    if (
        chunk_rows <= 0
        or chunk_rows > _MAX_PIPELINE_CHUNK_ROWS
        or chunk_rows % _PIPELINE_CHUNK_ROW_ALIGNMENT != 0
    ):
        raise ValueError(
            "chunk_rows must be a positive multiple of 128 no greater than 8192; "
            f"got {chunk_rows}"
        )


def _chunked_mega_producer_topology(
    *,
    format: BlockScaledFormatSpec,
    dispatch_quant_warps: int,
    nvfp4_high_throughput: bool,
    nvfp4_wide_row_quant: bool,
) -> tuple[int, int, int]:
    warps_per_group = 2
    row_reduction_slots = (
        base.DistBlockScaledGroupedGemmKernel.NVFP4_ROW_REDUCTION_SLOTS
    )
    if format is NVFP4 and not nvfp4_high_throughput:
        warps_per_group = 8 if nvfp4_wide_row_quant else 4
        if nvfp4_wide_row_quant:
            row_reduction_slots = warps_per_group + 2
    if dispatch_quant_warps % warps_per_group != 0:
        raise ValueError("dispatch quant warps must divide evenly into producer groups")
    return (
        warps_per_group,
        dispatch_quant_warps // warps_per_group,
        row_reduction_slots,
    )


class ChunkedMegaBlockScaledGroupedGemmKernel(base.MegaBlockScaledGroupedGemmKernel):
    """Forward-only MegaMoE kernel with expert-local FC13/FC2 waves."""

    PIPELINE_LEAD_CHUNKS: int = 2
    PIPELINE_CHUNK_ROWS: int = 512
    FC2_TILES_PER_SCHEDULE_ITEM: int = 2
    STAGE_ALL_FC13: bool = False
    # Cross-group lead window: > 0 replaces the per-group FC13/FC2 lead with
    # a schedule where FC2 trails FC13 by this many chunks ACROSS group
    # boundaries. Fixes the per-group lead collapse at 1 chunk/group (the
    # dominant prefill stall); 0 keeps the legacy per-group schedule.
    GLOBAL_LEAD_CHUNKS: int = 0
    # The wgrad column-quantized copies of the dispatched activations exist
    # only for callers that save them for backward; forward-only launches
    # (all inference) compile the stores out and the host shrinks the
    # storages to dummies.
    DISPATCH_COL_QUANT: bool = True
    # Activation ring: reuse a W-chunk window of the h2 buffers keyed by the
    # schedule position (0 = full-extent buffers, ring disabled). Deadlock-
    # free for W > PIPELINE_LEAD_CHUNKS because the schedule is one linear
    # item sequence: FC13(chunk s) sits at slot s and FC2(chunk s) at slot
    # s + LEAD, so the FC2 work that frees slot (s - W) % W always precedes
    # FC13(chunk s) in claim order. Requires interleaved FC13 + SWAP_AB +
    # device tensormaps; validated by the host wrapper.
    ACTIVATION_RING_CHUNKS: int = 0
    # Offsets (in Int32 elements) of the per-schedule-chunk FC2 (h2 ring)
    # and FC13 (x ring) consumption counters inside the activation
    # done-counter tensor; host-computed.
    ACTIVATION_RING_FC2_DONE_OFFSET: int = 0
    ACTIVATION_RING_FC13_DONE_OFFSET: int = 0
    # NVFP4 only: per-schedule-chunk consumption counters for the BF16
    # interleaved_h2 staging (credited by the row-quant producer).
    ACTIVATION_RING_H1_DONE_OFFSET: int = 0
    # Expert-borrow weight streaming: the trailing SLOTS groups hold borrowed
    # experts pulled from peer publish windows by the dispatch warps before
    # token work. Offsets index the activation counter region (like the ring
    # counters, shape-dependent trace-time constexprs keyed in the kernel
    # cache). 0 slots compiles the path out.
    WEIGHT_BORROW_SLOTS: int = 0
    WEIGHT_BORROW_WORK_OFFSET: int = 0
    WEIGHT_BORROW_DONE_OFFSET: int = 0
    WEIGHT_BORROW_CHUNK_BYTES: int = weight_borrow.WEIGHT_BORROW_CHUNK_BYTES
    QUANTIZE_DISPATCH: bool = False
    SWIGLU_VALUES_PER_THREAD: int = MEGA_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD
    SWAPPED_AXIS_PER_CTA: bool = True

    # Warp roles are laid out on warpgroup boundaries (0-3 / 4-7 / 8-11 / 12-15 /
    # 16-19) so a per-role setmaxnreg stays legal. The peer-token gather owns two
    # full warpgroups: it moves the whole dispatched activation operand and is the
    # throughput limiter for FC13, more so as hidden_dim grows.
    DISPATCH_TOTAL_WARPS: int = 20
    DISPATCH_QUANT_WARP_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    DISPATCH_QUANT_FIRST_WARP: int = 0
    DISPATCH_QUANT_WARPS: int = 8
    DISPATCH_QUANT_WARPS_PER_GROUP: int = 2
    DISPATCH_QUANT_GROUPS: int = 4
    DISPATCH_QUANT_GROUP_THREADS: int = 64
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: int = (
        DISPATCH_QUANT_WARPS_PER_GROUP
        * base.MegaBlockScaledGroupedGemmKernel.DISPATCH_QUANT_MICROTILES_PER_WARP
        * base.DistBlockScaledGroupedGemmKernel.DISPATCH_QUANT_SCALE_COLS_PER_WARP
    )
    DISPATCH_THREADS_PER_CTA: int = 32 * DISPATCH_TOTAL_WARPS

    TMA_AB_WARP_ID: int = 8
    TMA_B_WARP_ID: int = 10
    MMA_WARP_ID: int = 9
    EPILOG_WARP_IDS: tuple[int, ...] = (12, 13, 14, 15)
    EPILOG_WG_THREADS: int = 128
    FORWARD_QUANT_FIRST_WARP: int = 16
    FORWARD_QUANT_LAST_WARP: int = 20
    COMBINE_SWIGLU_QUANT_FIRST_WARP: int = FORWARD_QUANT_FIRST_WARP
    COMBINE_SWIGLU_WORK_TILES_PER_FETCH: int = 1
    COMBINE_SWIGLU_FWD_ELEMS_PER_LANE: int = 4
    COMBINE_SWIGLU_FWD_COL_BLOCKS_PER_SCALE: int = 8

    def _make_first_gemm_tma_atoms(
        self,
        *,
        layout,
        tensor_a,
        tensor_b,
        tensor_c_eff,
        tensor_sfa,
        tensor_sfb,
        output_accum=False,
    ):
        tma_atoms, tma_tensors = self._make_gemm_tma_atoms(
            layout=layout,
            tensor_a=tensor_a,
            tensor_b=tensor_b,
            tensor_c_eff=tensor_c_eff,
            tensor_sfa=tensor_sfa,
            tensor_sfb=tensor_sfb,
            output_accum=output_accum,
        )
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            # Ultra atoms are already CTA-local through the tv-shaped TMA
            # box, so the paired CTA-local FC13 rebuild is unnecessary.
            return tma_atoms, tma_tensors
        local_mma = base.sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            (
                self.BLOCK_SIZE_M // (self.NUM_CTAS * self.NUM_MMAS),
                self.BLOCK_SIZE_N,
            ),
        )
        local_mma_tiler = (
            self.BLOCK_SIZE_M // self.NUM_CTAS,
            self.BLOCK_SIZE_N,
            self.BLOCK_SIZE_K,
        )
        local_cluster_layout = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (local_mma.thr_id.shape,),
        )
        paired_a_atom, paired_a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO),
            tensor_a,
            cute.slice_(layout.a_smem_layout_staged, (None, None, None, 0)),
            local_mma_tiler,
            local_mma,
            local_cluster_layout.shape,
            internal_type=(
                cutlass.Uint8
                if layout.a_dtype.width != layout.b_dtype.width
                and layout.a_dtype.width < 8
                else None
            ),
        )
        tensor_sfa_view = cute.make_tensor(
            tensor_sfa.iterator,
            base.blockscaled_utils.tile_atom_to_shape_SF(
                tensor_a.shape,
                self.sf_vec_size,
            ),
        )
        paired_sfa_atom, paired_sfa_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO),
            tensor_sfa_view,
            cute.slice_(layout.sfa_smem_layout_staged, (None, None, None, 0)),
            local_mma_tiler,
            local_mma,
            local_cluster_layout.shape,
            internal_type=cutlass.Int16,
        )
        return (
            (paired_a_atom, tma_atoms[1], paired_sfa_atom, *tma_atoms[3:]),
            (paired_a_tensor, tma_tensors[1], paired_sfa_tensor, *tma_tensors[3:]),
        )

    def _make_first_gemm_prologue(self, params, storage, cluster_cta_rank, bid):
        state = self._make_gemm_prologue(
            params,
            storage,
            cluster_cta_rank,
            bid,
        )
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            # FC13 gate/up pairing under ultra: CTA r must load the 128-row
            # weight tile at (tile_idx + r * num_m_tiles), so the A g-view is
            # rebuilt at 128-row granularity with no CTA-embedded offset (the
            # load site adds the rank term to the axis, as on the non-ultra
            # paired path). SFA is already 128-row granular in the base ultra
            # prologue. The default view's automatic 256-row CTA split would
            # pair two adjacent gate tiles instead.
            sA = storage.sA.get_tensor(
                params.a_smem_layout_staged.outer,
                swizzle=params.a_smem_layout_staged.inner,
            )
            gA13 = cute.local_tile(
                params.mA,
                (
                    self.BLOCK_SIZE_M // self.NUM_CTAS,
                    sm103.SM103_TILE_K // 2,
                ),
                (None, None, None),
            )
            lay13 = gA13.layout
            region_tiled_mma = self._region_tiled_mma(params)
            tCgA13 = cute.make_tensor(
                gA13.iterator,
                cute.tiled_divide(
                    lay13,
                    (cute.size(region_tiled_mma.tv_layout_A[1][0]), 128),
                ),
            )
            local_cta_layout = cute.make_layout(1)
            state.tAsA, state.tAgA = cpasync.tma_partition(
                params.tma_atom_a,
                0,
                local_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA13, 0, 1),
            )
            # Same 128-row output-tile re-stride as the non-ultra paired path
            # (ultra pins NUM_MMAS == 1, so no atom-packing offset applies):
            # the scheduler's c_tile_m_idx is in BLOCK_SIZE_M // NUM_CTAS rows.
            state.tCgC = cute.make_tensor(
                state.tCgC.iterator,
                cute.make_layout(
                    state.tCgC.shape,
                    stride=(
                        state.tCgC.stride[0],
                        state.tCgC.stride[1],
                        state.tCgC.stride[2],
                        (self.BLOCK_SIZE_M // self.NUM_CTAS) * state.tCgC.stride[0][0],
                        state.tCgC.stride[4],
                        state.tCgC.stride[5],
                    ),
                ),
            )
            return state
        local_mma_tiler = (
            self.BLOCK_SIZE_M // self.NUM_CTAS,
            self.BLOCK_SIZE_N,
            self.BLOCK_SIZE_K,
        )
        gA = cute.local_tile(
            params.mA,
            cute.slice_(local_mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gSFA = cute.local_tile(
            params.mSFA,
            cute.slice_(local_mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        local_mma = base.sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            (
                self.BLOCK_SIZE_M // (self.NUM_CTAS * self.NUM_MMAS),
                self.BLOCK_SIZE_N,
            ),
        )
        local_thr_mma = local_mma.get_slice(0)
        tCgA = local_thr_mma.partition_A(gA)
        tCgSFA = local_thr_mma.partition_A(gSFA)
        local_cta_layout = cute.make_layout(1)
        state.tAsA, state.tAgA = cpasync.tma_partition(
            params.tma_atom_a,
            0,
            local_cta_layout,
            cute.group_modes(state.sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            params.tma_atom_sfa,
            0,
            local_cta_layout,
            cute.group_modes(state.sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        state.tAsSFA = cute.filter_zeros(tAsSFA)
        state.tAgSFA = cute.filter_zeros(tAgSFA)
        # Pack each CTA's MMA atoms into its local C tile. Mode 1 is the atom
        # axis, one atom of ``BLOCK_SIZE_M // (NUM_CTAS * NUM_MMAS)`` rows, and
        # mode 3 is the tile axis the epilogue indexes with ``c_tile_m_idx``.
        # With multiple atoms, CTA 1 also needs an atom offset to complement the
        # scheduler's ``num_m_tiles - 1`` tile origin without overlapping CTA 0's
        # tail. That offset spans every atom the tile origin left behind, so it
        # scales with ``NUM_MMAS - 1``, not with a single atom: sweeping the
        # multiplier confirms 1 atom is correct only at NUM_MMAS=2 and 2 atoms
        # only at NUM_MMAS=3, while a fixed single atom stores three-atom tiles
        # to the wrong rows and leaves the rest of the tile unwritten (every
        # output element comes back NaN).
        state.tCgC = cute.make_tensor(
            state.tCgC.iterator,
            cute.make_layout(
                state.tCgC.shape,
                stride=(
                    state.tCgC.stride[0],
                    (
                        (self.BLOCK_SIZE_M // (self.NUM_CTAS * self.NUM_MMAS))
                        * state.tCgC.stride[0][0]
                        if self.NUM_MMAS > 1
                        else state.tCgC.stride[1]
                    ),
                    state.tCgC.stride[2],
                    (self.BLOCK_SIZE_M // self.NUM_CTAS) * state.tCgC.stride[0][0],
                    state.tCgC.stride[4],
                    state.tCgC.stride[5],
                ),
            ),
        )
        if cutlass.const_expr(self.NUM_MMAS > 1):
            state.tCgC = cute.domain_offset(
                (
                    (
                        cluster_cta_rank
                        * (self.BLOCK_SIZE_M // (self.NUM_CTAS * self.NUM_MMAS))
                        * (self.NUM_MMAS - 1),
                        None,
                    ),
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                state.tCgC,
            )
        return state

    def _make_gemm_role_context(
        self,
        dgrad,
        wgrad,
        storage,
        cluster_cta_rank,
        pred_cta0,
        split_sizes,
        G,
        M,
        N,
        K,
        local_rank,
        counter_ptr,
    ):
        bid = cute.arch.block_idx()
        fc13_state = self._make_first_gemm_prologue(
            dgrad,
            storage,
            cluster_cta_rank,
            bid,
        )
        fc2_state = self._make_gemm_prologue(
            wgrad,
            storage,
            cluster_cta_rank,
            bid,
        )
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        ab_empty_mcast_mask = None
        acc_full_mcast_mask = None
        if cutlass.const_expr(self.NUM_CTAS == 2):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                dgrad.cluster_layout_vmnk,
                fc13_state.block_in_cluster_coord_vmnk,
                mcast_mode=2,
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                dgrad.cluster_layout_vmnk,
                fc13_state.block_in_cluster_coord_vmnk,
                mcast_mode=1,
            )
            sfa_full_mcast_mask = a_full_mcast_mask
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                dgrad.cluster_layout_sfb_vmnk,
                fc13_state.block_in_cluster_coord_sfb_vmnk,
                mcast_mode=1,
            )
            peer_coord = (
                fc13_state.block_in_cluster_coord_vmnk[0] ^ 1,
                *fc13_state.block_in_cluster_coord_vmnk[1:],
            )
            a_peer_mask = cpasync.create_tma_multicast_mask(
                dgrad.cluster_layout_vmnk,
                peer_coord,
                mcast_mode=2,
            )
            b_peer_mask = cpasync.create_tma_multicast_mask(
                dgrad.cluster_layout_vmnk,
                peer_coord,
                mcast_mode=1,
            )
            ab_empty_mcast_mask = (
                a_full_mcast_mask | b_full_mcast_mask | a_peer_mask | b_peer_mask
            )
            acc_full_mcast_mask = cute.make_layout_image_mask(
                dgrad.cluster_layout_vmnk,
                fc13_state.block_in_cluster_coord_vmnk,
                mode=0,
            )

        schedule_rank = (local_rank + cutlass.Int32(1)) % cutlass.Int32(self.world_size)
        problem = GroupedGemmProblem(
            split_sizes=split_sizes,
            groups=G,
            mnk=(M, N, K),
            local_rank=schedule_rank,
        )
        tile_cta_bar_mbar = (
            storage.tile_cta_bar_mbar.data_ptr() if self.NUM_TILE_CTA_BARS > 0 else None
        )
        cross_seam_mbar = (
            storage.cross_seam_mbar.data_ptr() if self.OVERLAPPING_ACCUM else None
        )
        sync = GroupedGemmPipelineSync(
            ab_full_mbar=storage.smem_full_mbar.data_ptr(),
            ab_empty_mbar=storage.smem_empty_mbar.data_ptr(),
            sf_full_mbar=(
                storage.sf_smem_full_mbar.data_ptr() if self.USE_SM103_ULTRA else None
            ),
            sf_empty_mbar=(
                storage.sf_smem_empty_mbar.data_ptr() if self.USE_SM103_ULTRA else None
            ),
            tmem_full_mbar=storage.tmem_full_mbar.data_ptr(),
            tmem_empty_mbar=storage.tmem_empty_mbar.data_ptr(),
            tile_consumer_mbar=storage.tile_consumer_mbar.data_ptr(),
            tile_producer_mbar=storage.tile_producer_mbar.data_ptr(),
            tile_cta_bar_mbar=tile_cta_bar_mbar,
            tile_id_smem_ptr=storage.tile_id_smem.data_ptr(),
            cross_seam_mbar=cross_seam_mbar,
            counter_ptr=counter_ptr,
            tma_mcast_masks=(
                a_full_mcast_mask,
                b_full_mcast_mask,
                sfa_full_mcast_mask,
                sfb_full_mcast_mask,
            ),
            ab_empty_mcast_mask=ab_empty_mcast_mask,
            acc_full_mcast_mask=acc_full_mcast_mask,
            cluster_cta_rank=cluster_cta_rank,
            pred_cta0=pred_cta0,
        )
        return fc13_state, fc2_state, problem, sync

    def _make_tma_role_pipelines(
        self,
        dgrad,
        wgrad,
        fc13_state,
        fc2_state,
        load_b: bool,
    ):
        fc13_tma_bytes = self._num_pipeline_tma_load_bytes(dgrad, fc13_state)
        fc13_tma = self._make_tma_pipeline(
            dgrad,
            fc13_state,
            fc13_tma_bytes,
        )
        fc2_tma_bytes = self._num_pipeline_tma_load_bytes(wgrad, fc2_state)
        fc2_tma = self._make_tma_pipeline(
            wgrad,
            fc2_state,
            fc2_tma_bytes,
        )
        fc13_tma_a_bytes = self._num_pipeline_tma_load_bytes(
            dgrad,
            fc13_state,
            load_b=False,
        )
        fc2_tma_a_bytes = self._num_pipeline_tma_load_bytes(
            wgrad,
            fc2_state,
            load_b=False,
        )
        if load_b:
            return (
                fc13_tma,
                fc2_tma,
                fc13_tma_bytes - fc13_tma_a_bytes,
                fc2_tma_bytes - fc2_tma_a_bytes,
            )
        return fc13_tma, fc2_tma, fc13_tma_a_bytes, fc2_tma_a_bytes

    def _sm103_pair_seg_tx_bytes(self, layout, load_b: bool):
        """Per-warp (data-segment, SF-segment) TMA byte counts for the
        ultra pipelines: one A/B smem stage holds one K=256 data segment and
        one SF stage holds one K=192 scale segment."""
        num_mma_ctas = cute.size(layout.tiled_mma.thr_id.shape)
        data_layout, sf_layout = (
            (layout.b_smem_layout_staged, layout.sfb_smem_layout_staged)
            if load_b
            else (layout.a_smem_layout_staged, layout.sfa_smem_layout_staged)
        )
        data_seg = cute.size_in_bytes(
            cutlass.Uint8,
            cute.slice_(data_layout, (None, None, None, 0)),
        )
        sf_seg = cute.size_in_bytes(
            self.sf_dtype,
            cute.slice_(sf_layout, (None, None, None, 0)),
        )
        return data_seg * num_mma_ctas, sf_seg * num_mma_ctas

    def _gemm_tma_role(
        self,
        dgrad,
        wgrad,
        storage,
        cluster_cta_rank,
        pred_cta0,
        split_sizes,
        G,
        M,
        N,
        K,
        local_rank,
        counter_ptr,
        dispatch_quant_done_counter,
        row_done_counter_offset,
        col_done_counter_offset,
        activation_quant_done_counter,
        activation_row_done_counter_offset,
        activation_col_done_counter_offset,
        tensormaps,
        use_device_tensormaps,
        work_info_smem_ptr,
        *,
        load_b: bool,
        weight_borrow_args=None,
    ) -> None:
        fc13_state, fc2_state, problem, sync = self._make_gemm_role_context(
            dgrad,
            wgrad,
            storage,
            cluster_cta_rank,
            pred_cta0,
            split_sizes,
            G,
            M,
            N,
            K,
            local_rank,
            counter_ptr,
        )
        fc13_tma, fc2_tma, fc13_tx_bytes, fc2_tx_bytes = self._make_tma_role_pipelines(
            dgrad,
            wgrad,
            fc13_state,
            fc2_state,
            load_b=load_b,
        )
        dispatch_quant_sync = base._MegaDispatchQuantSync(
            done_counter=dispatch_quant_done_counter,
            done_counter_offsets=(
                row_done_counter_offset,
                col_done_counter_offset,
            ),
        )
        activation_quant_sync = base._MegaDispatchQuantSync(
            done_counter=activation_quant_done_counter,
            done_counter_offsets=(
                activation_row_done_counter_offset,
                activation_col_done_counter_offset,
            ),
        )
        tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)
        fc13_seg_tx = (0, 0)
        fc2_seg_tx = (0, 0)
        if self.USE_SM103_ULTRA:
            fc13_seg_tx = self._sm103_pair_seg_tx_bytes(dgrad, load_b)
            fc2_seg_tx = self._sm103_pair_seg_tx_bytes(wgrad, load_b)
        _gemm_tma_warp(
            params_from_kernel(ChunkedGemmWarpParams, self),
            fc13_tma,
            fc2_tma,
            fc13_tx_bytes,
            fc2_tx_bytes,
            not load_b,
            problem,
            sync,
            dispatch_quant_sync,
            activation_quant_sync,
            tensormap_manager,
            tensormaps,
            use_device_tensormaps,
            work_info_smem_ptr,
            format=self.format,
            KLOOP_UNROLL=self.KLOOP_UNROLL,
            NUM_SMEM_BUFFERS=self.NUM_SMEM_BUFFERS,
            INTERLEAVED_FC13=self.INTERLEAVED_FC13,
            DISPATCH_QUANT_SCALE_COLS_PER_TILE=self.DISPATCH_QUANT_SCALE_COLS_PER_TILE,
            sf_vec_size=self.sf_vec_size,
            COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE=self.COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE,
            fc13_seg_tx=fc13_seg_tx,
            fc2_seg_tx=fc2_seg_tx,
            weight_borrow_args=weight_borrow_args,
        )

    def __init__(
        self,
        *,
        chunk_rows: int = PIPELINE_CHUNK_ROWS,
        wide_gather: bool = True,
        interleaved_fc13: bool = False,
        **kwargs,
    ) -> None:
        _validate_pipeline_chunk_rows(chunk_rows)
        super().__init__(**kwargs)
        if self.mode != self.MEGA_FORWARD_MODE:
            raise ValueError("chunked MegaMoE only supports forward mode")
        if not self.SWAP_AB:
            raise ValueError("chunked MegaMoE requires SWAP_AB=True")
        self.PIPELINE_CHUNK_ROWS = chunk_rows
        self.INTERLEAVED_FC13 = interleaved_fc13
        self._configure_warp_topology(wide_gather)
        self.PIPELINE_LEAD_CHUNKS = 4 if wide_gather else 2

    def _forward_act_block_size(self) -> int:
        return self.BLOCK_SIZE_N if self.SWAP_AB else self.BLOCK_SIZE_M

    @property
    def RING_CHUNK_ROW_TILES(self) -> int:
        """Activation row tiles (BLOCK_SIZE_N) per ring schedule chunk.

        The FC13 wait target and the FC2 consumption credits must agree on
        this quantity; both derive it from this single definition.
        """
        return self.PIPELINE_CHUNK_ROWS // self.BLOCK_SIZE_N

    def _validate_chunk_rows(self) -> None:
        # _forward_act_tile_offset converts a chunk offset into activation tiles,
        # which is only exact when a chunk is a whole number of activation tiles.
        act_block = self._forward_act_block_size()
        if self.PIPELINE_CHUNK_ROWS % act_block != 0:
            raise ValueError(
                "PIPELINE_CHUNK_ROWS must be a multiple of the activation block "
                f"size; got {self.PIPELINE_CHUNK_ROWS} and {act_block}"
            )

    def _configure_warp_topology(self, wide_gather: bool) -> None:
        self.VECTOR_DISPATCH_SCALE_COPY = wide_gather and self.format is NVFP4
        if wide_gather:
            self.DISPATCH_QUANT_WARP_IDS = (0, 1, 2, 3, 4, 5, 6, 7)
            self.TMA_AB_WARP_ID = 8
            self.MMA_WARP_ID = 9
            self.TMA_B_WARP_ID = 10
            self.EPILOG_WARP_IDS = (12, 13, 14, 15)
            self.FORWARD_QUANT_FIRST_WARP = 16
            self.FORWARD_QUANT_LAST_WARP = 20
            self.FORWARD_QUANT_SYNC_BAR = 7
        else:
            self.DISPATCH_QUANT_WARP_IDS = (0, 1, 2, 3)
            self.TMA_AB_WARP_ID = 4
            self.MMA_WARP_ID = 5
            self.TMA_B_WARP_ID = 6
            self.EPILOG_WARP_IDS = (8, 9, 10, 11)
            self.FORWARD_QUANT_FIRST_WARP = 12
            self.FORWARD_QUANT_LAST_WARP = 16
            self.FORWARD_QUANT_SYNC_BAR = 5
        self.DISPATCH_QUANT_WARPS = len(self.DISPATCH_QUANT_WARP_IDS)
        self.COMBINE_SWIGLU_QUANT_FIRST_WARP = self.FORWARD_QUANT_FIRST_WARP
        self.DISPATCH_QUANT_GROUPS = (
            self.DISPATCH_QUANT_WARPS // self.DISPATCH_QUANT_WARPS_PER_GROUP
        )
        self.DISPATCH_TOTAL_WARPS = (
            self.FORWARD_QUANT_FIRST_WARP
            if wide_gather and self.INTERLEAVED_FC13 and self.format is not NVFP4
            else self.FORWARD_QUANT_LAST_WARP
        )
        self.DISPATCH_THREADS_PER_CTA = 32 * self.DISPATCH_TOTAL_WARPS
        self.TOTAL_WARPS = self.DISPATCH_TOTAL_WARPS
        self.THREADS_PER_CTA = self.DISPATCH_THREADS_PER_CTA

    def _configure_nvfp4_producer_topology(
        self,
        high_throughput: bool,
        wide_row_quant: bool = False,
    ) -> None:
        self.NVFP4_GROUP_SOURCE_WAIT = wide_row_quant
        self.NVFP4_WARP_LEADER_ROW_SCALE_LOAD = wide_row_quant
        self.NVFP4_WARP_LOCAL_ROW_SCALE = not high_throughput and not wide_row_quant
        (
            self.DISPATCH_QUANT_WARPS_PER_GROUP,
            self.DISPATCH_QUANT_GROUPS,
            self.NVFP4_ROW_REDUCTION_SLOTS,
        ) = _chunked_mega_producer_topology(
            format=self.format,
            dispatch_quant_warps=self.DISPATCH_QUANT_WARPS,
            nvfp4_high_throughput=high_throughput,
            nvfp4_wide_row_quant=wide_row_quant,
        )
        self.NVFP4_ROW_SCALE_SLOT = self.NVFP4_ROW_REDUCTION_SLOTS - 1
        self.DISPATCH_QUANT_GROUP_THREADS = 32 * self.DISPATCH_QUANT_WARPS_PER_GROUP
        if high_throughput:
            self.FORWARD_COPY_COL_TILES_PER_WORK = 16
        else:
            if self.VECTOR_DISPATCH_SCALE_COPY:
                self.FORWARD_QUANT_LAST_WARP = self.FORWARD_QUANT_FIRST_WARP + 8
                self.DISPATCH_TOTAL_WARPS = self.FORWARD_QUANT_LAST_WARP
                self.DISPATCH_THREADS_PER_CTA = 32 * self.DISPATCH_TOTAL_WARPS
                self.TOTAL_WARPS = self.DISPATCH_TOTAL_WARPS
                self.THREADS_PER_CTA = self.DISPATCH_THREADS_PER_CTA
        self.DISPATCH_QUANT_SCALE_COLS_PER_TILE = (
            self.DISPATCH_QUANT_WARPS_PER_GROUP
            * self.DISPATCH_QUANT_MICROTILES_PER_WARP
            * self.DISPATCH_QUANT_SCALE_COLS_PER_WARP
        )
        self.nvfp4_swiglu_quant_groups = max(
            1,
            (self.TOTAL_WARPS - self.COMBINE_SWIGLU_QUANT_FIRST_WARP)
            // self.DISPATCH_QUANT_WARPS_PER_GROUP,
        )

    def _make_shared_storage(
        self,
        dgrad_layout,
        wgrad_layout,
        combine,
        G,
        use_a_global_scale_inv=False,
    ):
        return super()._make_shared_storage(
            dgrad_layout,
            wgrad_layout,
            True,
            G,
            CHUNKED_MEGA_WORK_INFO_FIELDS,
            use_a_global_scale_inv,
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        chunk_rows: int = PIPELINE_CHUNK_ROWS,
        wide_gather: bool = True,
        interleaved_fc13: bool = False,
        **kwargs,
    ) -> "ChunkedMegaBlockScaledGroupedGemmKernel":
        _validate_pipeline_chunk_rows(chunk_rows)
        kernel = super().from_config(config, **kwargs)
        kernel.PIPELINE_CHUNK_ROWS = chunk_rows
        kernel.INTERLEAVED_FC13 = interleaved_fc13
        kernel._configure_warp_topology(wide_gather)
        kernel.SWIGLU_VALUES_PER_THREAD = int(
            config.get(
                "SWIGLU_VALUES_PER_THREAD",
                MEGA_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD,
            )
        )
        if kernel.SWIGLU_VALUES_PER_THREAD not in (
            MEGA_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD,
            STAGED_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD,
        ):
            raise ValueError("SWIGLU_VALUES_PER_THREAD must select a known layout")
        kernel.PIPELINE_LEAD_CHUNKS = int(
            config.get("PIPELINE_LEAD_CHUNKS", 4 if wide_gather else 2)
        )
        kernel.FC2_TILES_PER_SCHEDULE_ITEM = int(
            config.get("FC2_TILES_PER_SCHEDULE_ITEM", 2)
        )
        kernel.STAGE_ALL_FC13 = bool(config.get("STAGE_ALL_FC13", False))
        kernel.GLOBAL_LEAD_CHUNKS = int(config.get("GLOBAL_LEAD_CHUNKS", 0))
        kernel.ACTIVATION_RING_CHUNKS = int(config.get("ACTIVATION_RING_CHUNKS", 0))
        kernel._validate_schedule_gates(interleaved_fc13)
        if kernel.ACTIVATION_RING_CHUNKS > 0:
            kernel.ACTIVATION_RING_B_ROWS = kernel.ACTIVATION_RING_CHUNKS * chunk_rows
        kernel.MUTABLE_COMBINE_ACC_FRAGMENT = bool(
            config.get("MUTABLE_COMBINE_ACC_FRAGMENT", False)
        )
        compiler_opt_level = config.get("CUTE_DSL_OPT_LEVEL")
        if compiler_opt_level is not None and (
            type(compiler_opt_level) is not int or compiler_opt_level not in (1, 3)
        ):
            raise ValueError("CUTE_DSL_OPT_LEVEL must be 1 or 3")
        if kernel.PIPELINE_LEAD_CHUNKS <= 0:
            raise ValueError("PIPELINE_LEAD_CHUNKS must be positive")
        if kernel.FC2_TILES_PER_SCHEDULE_ITEM not in (1, 2, 4):
            raise ValueError("FC2_TILES_PER_SCHEDULE_ITEM must be 1, 2, or 4")
        kernel._validate_chunk_rows()
        return kernel

    def _validate_schedule_gates(self, interleaved_fc13: bool) -> None:
        if self.GLOBAL_LEAD_CHUNKS < 0:
            raise ValueError("GLOBAL_LEAD_CHUNKS must be >= 0")
        if self.GLOBAL_LEAD_CHUNKS > 0 and self.STAGE_ALL_FC13:
            raise ValueError(
                "GLOBAL_LEAD_CHUNKS and STAGE_ALL_FC13 are mutually exclusive"
            )
        if self.USE_SM103_ULTRA:
            if not interleaved_fc13:
                raise ValueError(
                    "SM103 ultra chunked-mega requires the interleaved FC13 epilogue"
                )
            if self.STAGE_ALL_FC13:
                raise ValueError(
                    "SM103 ultra chunked-mega does not support STAGE_ALL_FC13"
                )
        if self.ACTIVATION_RING_CHUNKS > 0:
            if self.USE_SM103_ULTRA:
                raise ValueError(
                    "SM103 ultra chunked-mega does not support the activation ring yet"
                )
            if not interleaved_fc13:
                raise ValueError(
                    "the activation ring requires the interleaved FC13 epilogue"
                )
            if self.STAGE_ALL_FC13:
                raise ValueError(
                    "STAGE_ALL_FC13 leaves every chunk live at once; the "
                    "activation ring cannot bound that window"
                )
            if self.ACTIVATION_RING_CHUNKS <= self.PIPELINE_LEAD_CHUNKS:
                raise ValueError(
                    "ACTIVATION_RING_CHUNKS must exceed PIPELINE_LEAD_CHUNKS "
                    "for the interleaved schedule to stay deadlock-free"
                )
            if (
                self.GLOBAL_LEAD_CHUNKS > 0
                and self.ACTIVATION_RING_CHUNKS <= self.GLOBAL_LEAD_CHUNKS
            ):
                raise ValueError(
                    "ACTIVATION_RING_CHUNKS must exceed GLOBAL_LEAD_CHUNKS "
                    "for the cross-group schedule to stay deadlock-free"
                )

    @cute.jit
    def _forward_act_tile_offset(
        self,
        group_act_tiles,
        group_rows,
        local_rank,
    ):
        # FC13/FC2 walk a rank-rotated *chunk* order (ChunkedMegaProblemVisitor),
        # so every activation-tile rotation has to land on a chunk boundary too.
        # Rotating by whole activation tiles diverges from the chunk rotation once
        # world_size reaches the per-group activation-tile count, and the SwiGLU
        # producer then blocks on an h1 tile the GEMM has not scheduled yet while
        # the GEMM blocks on the h2 tile that producer owes it.
        num_chunks = ceil_div(
            group_rows,
            cutlass.Int32(self.PIPELINE_CHUNK_ROWS),
        )
        chunk_offset = _chunk_offset(num_chunks, local_rank, world_size=self.world_size)
        act_tile_offset = chunk_offset * cutlass.Int32(
            self.PIPELINE_CHUNK_ROWS // self._forward_act_block_size()
        )
        return act_tile_offset % group_act_tiles

    def _num_pipeline_tma_load_bytes(
        self,
        params,
        state,
        *,
        load_a: bool = True,
        load_b: bool = True,
    ):
        a_bytes = (
            cute.cosize(cute.slice_(params.a_smem_layout_staged, (None, None, None, 0)))
            * params.a_dtype_width
            // 8
        ) + cute.size_in_bytes(
            state.sSFA.element_type,
            cute.slice_(params.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        b_bytes = (
            cute.cosize(cute.slice_(params.b_smem_layout_staged, (None, None, None, 0)))
            * params.b_dtype_width
            // 8
        ) + cute.size_in_bytes(
            state.sSFB.element_type,
            cute.slice_(params.sfb_tma_smem_layout_staged, (None, None, None, 0)),
        )
        return ((a_bytes if load_a else 0) + (b_bytes if load_b else 0)) * cute.size(
            params.tiled_mma.thr_id.shape
        )

    @cute.jit
    def _gemm_epilogue_warpgroup(  # noqa: C901
        self,
        tidx,
        fc13: base._MegaEpilogPipeline,
        fc2: base._MegaEpilogPipeline,
        problem: GroupedGemmProblem,
        sync: GroupedGemmPipelineSync,
        activation_quant: base._MegaDispatchQuantSync,
        tensormap_manager,
        tensormaps: cute.Tensor,
        scatter_ptrs: cute.Tensor,
        fc13_postprocess_output: cute.Tensor,
        fc13_postprocess_scale: cute.Tensor,
        scatter_ptr_smem_ptr: cute.Pointer,
        use_device_tensormaps: cutlass.Constexpr[bool],
        work_info_smem_ptr: cute.Pointer,
        a_global_scale_inv_smem_ptr,
        dgrad_a_global_scale_inv_ptr: cutlass.Int64,
        wgrad_a_global_scale_inv_ptr: cutlass.Int64,
        dgrad_b_global_scale_inv_ptr: cutlass.Int64,
        wgrad_b_global_scale_inv_ptr: cutlass.Int64,
        use_global_scale_inv: cutlass.Constexpr[bool],
    ) -> None:
        M, N, K = problem.mnk
        scheduler = MegaDynamicScheduler.create_record_consumer(
            sync.tile_consumer_mbar,
            sync.tile_id_smem_ptr,
            self.NUM_CTAS,
            self.NUM_TILE_BUFFERS,
        )
        tile_idx = scheduler.initial_work_tile_info()
        accum_cnt_tile = cutlass.Int32(0)
        accum_cnt_out = cutlass.Int32(0)
        ring_waited_sched = cutlass.Int32(-1)
        epilog_copy_pair = (
            self._make_epilog_copy_partitions(tidx, fc13),
            self._make_epilog_copy_partitions(tidx, fc2),
        )
        c_desc_base_pair = (
            self.MEGA_DGRAD_TENSORMAP_BASE + 4,
            self.MEGA_WGRAD_TENSORMAP_BASE + 4,
        )
        work = load_chunked_mega_work_info(
            tile_idx,
            work_info_smem_ptr,
            cutlass.Int32(0),
            self.format is not NVFP4,
            CHUNKED_MEGA_WORK_INFO_ROLE_EPILOGUE,
        )

        while work.is_valid_tile:
            g = work.group_idx
            dgrad_b_scale = _load_group_global_scale_inv(
                dgrad_b_global_scale_inv_ptr, g, use_global_scale_inv
            )
            wgrad_b_scale = _load_group_global_scale_inv(
                wgrad_b_global_scale_inv_ptr, g, use_global_scale_inv
            )
            c_desc_ptr_pair: list = []
            for gemm_idx in cutlass.range_constexpr(2):
                if cutlass.const_expr(use_device_tensormaps):
                    c_desc_g_ptr = tensormap_manager.get_tensormap_ptr(
                        tensormaps[(g, c_desc_base_pair[gemm_idx], None)].iterator
                    )
                    c_desc_ptr_pair.append(
                        tensormap_manager.get_tensormap_ptr(
                            c_desc_g_ptr,
                            cute.AddressSpace.generic,
                        )
                    )
                    tensormap_manager.fence_tensormap_update(c_desc_g_ptr)
                else:
                    c_desc_ptr_pair.append(None)

            while work.is_valid_tile and work.group_idx == g:
                for gemm_idx in cutlass.range_constexpr(2):
                    if work.problem_idx == cutlass.Int32(gemm_idx):
                        pipeline = (fc13, fc2)[gemm_idx]
                        full_m_size = (2 * N, K)[gemm_idx]
                        tTR_rAcc, tiled_copy_r2s, tRS_rC, tRS_sC = epilog_copy_pair[
                            gemm_idx
                        ]
                        _wait_pending_epilog_store(
                            EPILOG_WARP_IDS=self.EPILOG_WARP_IDS,
                            EPILOG_WG_THREADS=self.EPILOG_WG_THREADS,
                        )
                        c_desc_ptr = c_desc_ptr_pair[gemm_idx]
                        tile_buf, _ = _get_bufidx_phase(
                            accum_cnt_out,
                            self.NUM_TILE_BUFFERS,
                        )
                        tile_m_idx = work.tile_m_idx
                        tile_n_idx = work.tile_n_idx
                        is_leader = cutlass.Boolean(False)
                        ring_item_credit = cutlass.Int32(0)
                        tiles_per_item: cutlass.Constexpr[int] = (
                            1,
                            self.FC2_TILES_PER_SCHEDULE_ITEM,
                        )[gemm_idx]
                        postprocess_cm_start = work.split_prefix
                        postprocess_scale_cm_start = work.scale_split_prefix
                        if cutlass.const_expr(
                            self.ACTIVATION_RING_CHUNKS > 0 and gemm_idx == 0
                        ):
                            ring_scale_page_rows: cutlass.Constexpr[int] = (
                                (self.BLOCK_SIZE_N + 127) // 128
                            ) * 128
                            # Overwriting ring slot (sched % W) requires FC2 to
                            # have fully consumed schedule chunk (sched - W),
                            # the slot's previous occupant. The target is the
                            # tail-padded constant every chunk's FC2 credits
                            # reach (see the FC2 arm below).
                            ring_prev_sched = work.sched_chunk_idx - cutlass.Int32(
                                self.ACTIVATION_RING_CHUNKS
                            )
                            if ring_prev_sched >= cutlass.Int32(0):
                                # One wait per schedule chunk: consecutive
                                # work items share a chunk, so the acquire
                                # spin re-runs only on chunk transitions.
                                if work.sched_chunk_idx != ring_waited_sched:
                                    if cutlass.const_expr(self.format is NVFP4):
                                        # FC13 stores BF16 into interleaved_h2;
                                        # the slot is free once the row-quant
                                        # producer consumed its previous chunk
                                        # (tile-batched credits, tail-padded).
                                        _wait_counter_at_least(
                                            cute.recast_ptr(
                                                activation_quant.done_counter.iterator
                                                + cutlass.Int32(
                                                    self.ACTIVATION_RING_H1_DONE_OFFSET
                                                )
                                                + ring_prev_sched,
                                                dtype=cutlass.Uint32,
                                            ),
                                            cutlass.Uint32(self.PIPELINE_CHUNK_ROWS),
                                        )
                                    # On NVFP4 this wait additionally covers
                                    # the row-quant producer's h2 qdata slot
                                    # overwrite: the producer reads this
                                    # chunk's interleaved_h2 only after the
                                    # stores below (source-tile wait), so it
                                    # inherits the FC2-consumed guarantee.
                                    ring_fc2_target = ceil_div(
                                        K, cutlass.Int32(self.BLOCK_SIZE_M)
                                    ) * cutlass.Int32(
                                        self.RING_CHUNK_ROW_TILES * self.NUM_CTAS
                                    )
                                    _wait_counter_at_least(
                                        cute.recast_ptr(
                                            activation_quant.done_counter.iterator
                                            + cutlass.Int32(
                                                self.ACTIVATION_RING_FC2_DONE_OFFSET
                                            )
                                            + ring_prev_sched,
                                            dtype=cutlass.Uint32,
                                        ),
                                        cutlass.Uint32(ring_fc2_target),
                                    )
                                    ring_waited_sched = work.sched_chunk_idx
                            ring_slot = work.sched_chunk_idx % cutlass.Int32(
                                self.ACTIVATION_RING_CHUNKS
                            )
                            postprocess_cm_start = (
                                ring_slot * cutlass.Int32(self.PIPELINE_CHUNK_ROWS)
                                - work.chunk_start
                            )
                            postprocess_scale_cm_start = ring_slot * cutlass.Int32(
                                self.RING_CHUNK_ROW_TILES * ring_scale_page_rows
                            ) - (
                                work.chunk_start // cutlass.Int32(self.BLOCK_SIZE_N)
                            ) * cutlass.Int32(ring_scale_page_rows)
                        for subtile_idx in cutlass.range_constexpr(tiles_per_item):
                            subtile_m_idx, subtile_n_idx = _subtile_coords(
                                tile_m_idx,
                                tile_n_idx,
                                subtile_idx,
                                SWAP_AB=self.SWAP_AB,
                            )
                            if _subtile_is_valid(
                                subtile_m_idx,
                                subtile_n_idx,
                                work.num_output_tiles,
                                SWAP_AB=self.SWAP_AB,
                            ):
                                cta_m_start = (
                                    subtile_m_idx * self.BLOCK_SIZE_M
                                    + sync.cluster_cta_rank
                                    * (self.BLOCK_SIZE_M // self.NUM_CTAS)
                                )
                                skip_store = cutlass.Boolean(False)
                                if cutlass.const_expr(not self.SWAP_AB):
                                    skip_store = cta_m_start >= full_m_size
                                c_tile_m_idx = subtile_m_idx
                                if cutlass.const_expr(gemm_idx == 0):
                                    c_tile_m_idx += sync.cluster_cta_rank * (
                                        work.num_m_tiles - cutlass.Int32(1)
                                    )
                                if cutlass.const_expr(gemm_idx == 0):
                                    if cutlass.const_expr(use_device_tensormaps):
                                        c_tile_indices = (
                                            c_tile_m_idx,
                                            subtile_n_idx,
                                        )
                                    elif cutlass.const_expr(self.SWAP_AB):
                                        c_tile_indices = (
                                            c_tile_m_idx,
                                            work.group_tile_prefix + subtile_n_idx,
                                        )
                                    else:
                                        c_tile_indices = (
                                            work.group_tile_prefix + subtile_m_idx,
                                            subtile_n_idx,
                                        )
                                    is_leader = self._blockscaled_epilog_consumer_tile(
                                        tidx,
                                        pipeline,
                                        sync,
                                        tTR_rAcc,
                                        tiled_copy_r2s,
                                        tRS_rC,
                                        tRS_sC,
                                        work.num_k_tiles,
                                        accum_cnt_tile,
                                        c_tile_indices,
                                        skip_store,
                                        (c_desc_ptr,),
                                        use_device_tensormaps,
                                        dgrad_b_scale,
                                        True,
                                        split_prefix=work.split_prefix,
                                        postprocess_scale_prefix=(
                                            postprocess_scale_cm_start
                                        ),
                                        postprocess_row_prefix=postprocess_cm_start,
                                        a_scale_m_size=(
                                            work.full_n_size
                                            if self.SWAP_AB
                                            else full_m_size
                                        ),
                                        tile_m_idx=subtile_m_idx,
                                        tile_n_idx=subtile_n_idx,
                                        cluster_cta_rank=sync.cluster_cta_rank,
                                        a_global_scale_inv_smem_ptr=(
                                            a_global_scale_inv_smem_ptr
                                        ),
                                        a_global_scale_inv_ptr=(
                                            dgrad_a_global_scale_inv_ptr
                                        ),
                                        use_a_global_scale_inv=(use_global_scale_inv),
                                        use_b_global_scale_inv=(use_global_scale_inv),
                                        postprocess_output=fc13_postprocess_output,
                                        postprocess_scale=fc13_postprocess_scale,
                                        postprocess_store=self.INTERLEAVED_FC13,
                                        postprocess_m_size=(
                                            work.full_n_size
                                            if self.SWAP_AB
                                            else full_m_size
                                        ),
                                        postprocess_n_size=(
                                            full_m_size
                                            if self.SWAP_AB
                                            else work.full_n_size
                                        ),
                                    )
                                    activation_tile_idx = (
                                        subtile_n_idx if self.SWAP_AB else subtile_m_idx
                                    )
                                    _signal_fc13_epilogue(
                                        activation_tile_idx,
                                        work.group_tile_prefix,
                                        is_leader,
                                        activation_quant,
                                        INTERLEAVED_FC13=self.INTERLEAVED_FC13,
                                        format=self.format,
                                        EPILOG_WG_THREADS=self.EPILOG_WG_THREADS,
                                    )
                                    if cutlass.const_expr(
                                        self.ACTIVATION_RING_CHUNKS > 0
                                    ):
                                        # Credit x consumption for this chunk so
                                        # the dispatch producer can reuse its
                                        # ring slot; same tail-padded constant-
                                        # target scheme as the FC2 credits.
                                        ring_x_row_tiles: cutlass.Constexpr[int] = (
                                            self.PIPELINE_CHUNK_ROWS
                                            // self.BLOCK_SIZE_N
                                        )
                                        ring_x_credit = cutlass.Int32(1)
                                        if (
                                            work.tile_m_idx == cutlass.Int32(0)
                                            and work.tile_n_idx
                                            == work.chunk_start
                                            // cutlass.Int32(self.BLOCK_SIZE_N)
                                        ):
                                            ring_x_credit += ceil_div(
                                                cutlass.Int32(2) * N,
                                                cutlass.Int32(self.BLOCK_SIZE_M),
                                            ) * (
                                                cutlass.Int32(ring_x_row_tiles)
                                                - work.num_n_tiles
                                            )
                                        if is_leader:
                                            with cute.arch.elect_one():
                                                cute.arch.atomic_add(
                                                    cute.recast_ptr(
                                                        activation_quant.done_counter.iterator
                                                        + cutlass.Int32(
                                                            self.ACTIVATION_RING_FC13_DONE_OFFSET
                                                        )
                                                        + work.sched_chunk_idx,
                                                        dtype=cutlass.Uint32,
                                                    ),
                                                    cutlass.Uint32(ring_x_credit),
                                                    sem="release",
                                                    scope="gpu",
                                                )
                                else:
                                    is_leader = self._mega_combine_epilog_consumer_tile(
                                        tidx=tidx,
                                        pipeline=pipeline,
                                        sync=sync,
                                        mScatter=scatter_ptrs,
                                        scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
                                        num_k_tiles=work.num_k_tiles,
                                        accum_cnt_tile=accum_cnt_tile,
                                        tile_m_idx=subtile_m_idx,
                                        tile_n_idx=subtile_n_idx,
                                        cm_start=work.split_prefix,
                                        m_size=full_m_size,
                                        n_size=work.full_n_size,
                                        a_global_scale_inv_smem_ptr=(
                                            a_global_scale_inv_smem_ptr
                                        ),
                                        a_global_scale_inv_ptr=(
                                            wgrad_a_global_scale_inv_ptr
                                        ),
                                        b_scale=wgrad_b_scale,
                                        use_global_scale_inv=use_global_scale_inv,
                                    )
                                    if cutlass.const_expr(
                                        self.ACTIVATION_RING_CHUNKS > 0
                                    ):
                                        # Credit h2 consumption for this chunk.
                                        # tmem_full (inside the consumer tile)
                                        # implies every FC2 B k-stage for this
                                        # subtile has been read from the ring.
                                        # Group-tail chunks are padded by the
                                        # designated first subtile so every
                                        # chunk reaches the same constant
                                        # target the FC13 waiter derives.
                                        ring_credit = cutlass.Int32(1)
                                        if cutlass.const_expr(subtile_idx == 0):
                                            if (
                                                work.tile_m_idx == cutlass.Int32(0)
                                                and work.tile_n_idx
                                                == work.chunk_start
                                                // cutlass.Int32(self.BLOCK_SIZE_N)
                                            ):
                                                ring_credit += ceil_div(
                                                    K,
                                                    cutlass.Int32(self.BLOCK_SIZE_M),
                                                ) * (
                                                    cutlass.Int32(
                                                        self.RING_CHUNK_ROW_TILES
                                                    )
                                                    - work.num_n_tiles
                                                )
                                        ring_item_credit += ring_credit
                                accum_cnt_tile += cutlass.Int32(1)
                        if cutlass.const_expr(
                            self.ACTIVATION_RING_CHUNKS > 0 and gemm_idx == 1
                        ):
                            # One batched FC2 h2-consumption credit per work
                            # item; all its subtiles share the schedule chunk.
                            if ring_item_credit > cutlass.Int32(0):
                                if is_leader:
                                    with cute.arch.elect_one():
                                        cute.arch.atomic_add(
                                            cute.recast_ptr(
                                                activation_quant.done_counter.iterator
                                                + cutlass.Int32(
                                                    self.ACTIVATION_RING_FC2_DONE_OFFSET
                                                )
                                                + work.sched_chunk_idx,
                                                dtype=cutlass.Uint32,
                                            ),
                                            cutlass.Uint32(ring_item_credit),
                                            sem="release",
                                            scope="gpu",
                                        )
                        scheduler.consumer_release_tile(
                            sync.tile_producer_mbar,
                            tile_buf,
                            sync.cluster_cta_rank,
                            is_leader,
                        )
                accum_cnt_out += cutlass.Int32(1)
                tile_idx = scheduler.advance_record_consumer(accum_cnt_out)
                work = load_chunked_mega_work_info(
                    tile_idx,
                    work_info_smem_ptr,
                    accum_cnt_out % self.NUM_TILE_BUFFERS,
                    self.format is not NVFP4,
                    CHUNKED_MEGA_WORK_INFO_ROLE_EPILOGUE,
                )
        _wait_pending_epilog_store(
            EPILOG_WARP_IDS=self.EPILOG_WARP_IDS,
            EPILOG_WG_THREADS=self.EPILOG_WG_THREADS,
        )

    @cute.jit
    def _dispatch_warpgroup(
        self,
        work_counter,
        done_counter,
        tile_smem_ptr,
        pointer_smem_ptr,
        gather_ptrs,
        row_q_words,
        row_scale,
        row_global_scale_inv,
        col_q_words,
        col_scale,
        split_sizes,
        groups,
        rows,
        hidden_dim,
        total_tiles,
        local_rank,
        row_done_offset,
        col_done_offset,
        use_global_scale_inv,
        ring_fc13_done_counter=None,
        ring_fc13_target=0,
        weight_borrow_args=None,
        activation_done_counter=None,
    ) -> None:
        if cutlass.const_expr(self.WEIGHT_BORROW_SLOTS > 0):
            # Pull the borrowed slots' weight rows before any token work: the
            # items are few, the counter is dedicated, and the trailing
            # borrowed groups' gates open while local groups still compute.
            tidx, _, _ = cute.arch.thread_idx()
            drain_lane = tidx - cutlass.Int32(
                self.DISPATCH_QUANT_FIRST_WARP * self.THREADS_PER_WARP
            )
            drain_group = drain_lane // cutlass.Int32(self.DISPATCH_QUANT_GROUP_THREADS)
            drain_group_lane = drain_lane - drain_group * cutlass.Int32(
                self.DISPATCH_QUANT_GROUP_THREADS
            )
            weight_borrow._drain_weight_fetch_items(
                weight_borrow_args,
                activation_done_counter,
                cutlass.Int32(self.WEIGHT_BORROW_WORK_OFFSET),
                cutlass.Int32(self.WEIGHT_BORROW_DONE_OFFSET),
                tile_smem_ptr,
                drain_group,
                drain_group_lane,
                SLOTS=self.WEIGHT_BORROW_SLOTS,
                CHUNK_BYTES=self.WEIGHT_BORROW_CHUNK_BYTES,
                GROUP_THREADS=self.DISPATCH_QUANT_GROUP_THREADS,
                SYNC_BAR=self.DISPATCH_QUANT_SYNC_BAR,
            )
        if cutlass.const_expr(self.QUANTIZE_DISPATCH):
            self._mega_dispatch_quant_producer_body(
                mWorkCounter=work_counter,
                mDoneCounter=done_counter,
                tile_smem_ptr=tile_smem_ptr,
                dispatch_ptr_smem_ptr=pointer_smem_ptr,
                mGatherPtrs=gather_ptrs,
                mRowQWords=row_q_words,
                mRowScale=row_scale,
                mColQWords=col_q_words,
                mColScale=col_scale,
                split_sizes=split_sizes,
                G=groups,
                M=rows,
                K=hidden_dim,
                total_tiles=total_tiles,
                local_rank=local_rank,
                row_done_counter_offset=row_done_offset,
                col_done_counter_offset=col_done_offset,
                col_quant=self.DISPATCH_COL_QUANT,
                mRingFc13Done=ring_fc13_done_counter,
                ring_fc13_done_offset=cutlass.Int32(
                    self.ACTIVATION_RING_FC13_DONE_OFFSET
                ),
                ring_fc13_target=ring_fc13_target,
            )
        else:
            self._mega_dispatch_blockscaled_copy_producer_body(
                mWorkCounter=work_counter,
                mDoneCounter=done_counter,
                tile_smem_ptr=tile_smem_ptr,
                dispatch_ptr_smem_ptr=pointer_smem_ptr,
                mGatherPtrs=gather_ptrs,
                mRowQWords=row_q_words,
                mRowScale=row_scale,
                mRowGlobalScaleInv=row_global_scale_inv,
                split_sizes=split_sizes,
                G=groups,
                M=rows,
                K=hidden_dim,
                local_rank=local_rank,
                row_done_counter_offset=row_done_offset,
                col_done_counter_offset=col_done_offset,
                use_global_scale_inv=use_global_scale_inv,
                mRingFc13Done=ring_fc13_done_counter,
                ring_fc13_done_offset=cutlass.Int32(
                    self.ACTIVATION_RING_FC13_DONE_OFFSET
                ),
                ring_fc13_target=ring_fc13_target,
            )

    @cute.kernel
    def dispatch_bprop_kernel(  # noqa: C901
        self,
        dgrad: base._MegaGemmKernelParams,
        wgrad: base._MegaGemmKernelParams,
        split_sizes: cute.Tensor,
        counter_ptr: cute.Pointer,
        tensormaps: cute.Tensor,
        mDispatchQuantGatherPtrs: cute.Tensor,
        mDispatchQuantWorkCounter: cute.Tensor,
        mDispatchQuantDoneCounter: cute.Tensor,
        mDispatchQuantRowQWords: cute.Tensor,
        mDispatchQuantRowScale: cute.Tensor,
        mDispatchQuantRowGlobalScaleInv: cute.Tensor,
        mDispatchQuantColQWords: cute.Tensor,
        mDispatchQuantColScale: cute.Tensor,
        mCombineScatterPtrs: cute.Tensor,
        mCombineDz: cute.Tensor,
        mCombineH1: cute.Tensor,
        mActivationGatherPtrs: cute.Tensor,
        mActivationQuantWorkCounter: cute.Tensor,
        mActivationQuantDoneCounter: cute.Tensor,
        mActivationQuantRowQWords: cute.Tensor,
        mActivationQuantRowScale: cute.Tensor,
        mActivationQuantRowGlobalScaleInv: cute.Tensor,
        mActivationQuantColQWords: cute.Tensor,
        mActivationQuantColScale: cute.Tensor,
        row_done_counter_offset: cutlass.Int32,
        col_done_counter_offset: cutlass.Int32,
        activation_quant_total_tiles: cutlass.Int32,
        activation_row_done_counter_offset: cutlass.Int32,
        activation_col_done_counter_offset: cutlass.Int32,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        dispatch_quant_total_tiles: cutlass.Int32,
        combine: cutlass.Constexpr[bool],
        activation_dispatch: cutlass.Constexpr[bool],
        use_device_tensormaps: cutlass.Constexpr[bool],
        activation_buffer_base_ptr: cutlass.Int64,
        activation_buffer_size_bytes: cutlass.Int64,
        activation_offsets: cute.Tensor,
        use_activation_buffer: cutlass.Constexpr[bool],
        conditional_execution: cute.Tensor,
        use_conditional_execution: cutlass.Constexpr[bool],
        dgrad_b_global_scale_inv_ptr: cutlass.Int64,
        wgrad_b_global_scale_inv_ptr: cutlass.Int64,
        use_global_scale_inv: cutlass.Constexpr[bool],
        nvfp4_recip_lut_ptr: cutlass.Int64,
        weight_borrow_args=None,
    ):
        if cutlass.const_expr(use_conditional_execution):
            if conditional_execution[0] == cutlass.Int32(0):
                base.thread_exit()
        if cutlass.const_expr(use_activation_buffer):
            activation_rows = _activation_buffer_rows(split_sizes, G)
            dispatch_value_count = cutlass.Int64(activation_rows) * cutlass.Int64(K)
            dispatch_q_byte_extent = (
                dispatch_value_count
                * cutlass.Int64(cutlass.const_expr(self.format.a_dtype.width))
                // cutlass.Int64(8)
            )
            dispatch_row_scale_byte_extent = (
                _activation_buffer_row_scale_storage_byte_extent(
                    activation_rows,
                    K,
                    sf_vec_size=self.sf_vec_size,
                    sf_dtype_width=self.format.sf_dtype.width,
                )
            )
            dispatch_col_scale_byte_extent = (
                _activation_buffer_col_scale_storage_byte_extent(
                    activation_rows,
                    K,
                    sf_vec_size=self.sf_vec_size,
                    sf_dtype_width=self.format.sf_dtype.width,
                )
            )
            activation_value_count = cutlass.Int64(activation_rows) * cutlass.Int64(N)
            activation_q_byte_extent = (
                activation_value_count
                * cutlass.Int64(cutlass.const_expr(self.format.a_dtype.width))
                // cutlass.Int64(8)
            )
            activation_row_scale_byte_extent = (
                _activation_buffer_row_scale_storage_byte_extent(
                    activation_rows,
                    N,
                    sf_vec_size=self.sf_vec_size,
                    sf_dtype_width=self.format.sf_dtype.width,
                )
            )
            activation_col_scale_byte_extent = (
                _activation_buffer_col_scale_storage_byte_extent(
                    activation_rows,
                    N,
                    sf_vec_size=self.sf_vec_size,
                    sf_dtype_width=self.format.sf_dtype.width,
                )
            )
            source_element_size = cutlass.Int64(
                cutlass.const_expr(mCombineDz.element_type.width // 8)
            )
            has_activation_rows = cutlass.Int64(activation_rows > cutlass.Int32(0))
            if cutlass.const_expr(self.INTERLEAVED_FC13):
                # Both source offsets point at the dense contiguous NVFP4
                # h2 staging slot rather than the strided halves of h1. N
                # counts logical BF16 columns, so size the extent from the
                # BF16 view (mCombineDz is the same bytes viewed as uint32
                # with half the columns).
                source_byte_extent = (
                    has_activation_rows
                    * cutlass.Int64(activation_rows)
                    * cutlass.Int64(N)
                    * cutlass.Int64(
                        cutlass.const_expr(mCombineH1.element_type.width // 8)
                    )
                )
            else:
                source_byte_extent = (
                    has_activation_rows
                    * (
                        cutlass.Int64(2) * cutlass.Int64(activation_rows)
                        - cutlass.Int64(1)
                    )
                    * cutlass.Int64(N)
                    * source_element_size
                )
            mDispatchQuantRowQWords = _activation_buffer_tensor_with_byte_extent(
                mDispatchQuantRowQWords,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_A_Q_OFFSET,
                cutlass.Uint32,
                dispatch_q_byte_extent,
            )
            mDispatchQuantRowScale = _activation_buffer_tensor_with_byte_extent(
                mDispatchQuantRowScale,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_A_SCALE_OFFSET,
                cutlass.Uint8,
                dispatch_row_scale_byte_extent,
            )
            if cutlass.const_expr(use_global_scale_inv):
                mDispatchQuantRowGlobalScaleInv = (
                    _activation_buffer_row_global_scale_tensor(
                        mDispatchQuantRowGlobalScaleInv,
                        activation_buffer_base_ptr,
                        activation_buffer_size_bytes,
                        activation_offsets,
                        activation_rows,
                        K,
                        sf_vec_size=self.sf_vec_size,
                        sf_dtype_width=self.format.sf_dtype.width,
                        scale_offset_idx=MEGA_FIRST_GEMM_OFFSET_BASE
                        + ACTIVATION_A_SCALE_OFFSET,
                    )
                )
            mDispatchQuantColQWords = _activation_buffer_tensor_with_byte_extent(
                mDispatchQuantColQWords,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_COL_Q_OFFSET,
                cutlass.Uint32,
                dispatch_q_byte_extent,
            )
            mDispatchQuantColScale = _activation_buffer_tensor_with_byte_extent(
                mDispatchQuantColScale,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_COL_SCALE_OFFSET,
                cutlass.Uint8,
                dispatch_col_scale_byte_extent,
            )
            # MX interleaved forwards quantize h2 inside the FC13 epilogue, so
            # the plan marks both source offsets missing and the host dummies
            # stand in as never-read descriptors; skip the rebase, whose range
            # check requires an addressable offset.
            if cutlass.const_expr(
                not (self.INTERLEAVED_FC13 and self.format is not NVFP4)
            ):
                mCombineDz = _activation_buffer_tensor_with_byte_extent(
                    mCombineDz,
                    activation_buffer_base_ptr,
                    activation_buffer_size_bytes,
                    activation_offsets,
                    MEGA_SECOND_GEMM_OFFSET_BASE + ACTIVATION_SOURCE_X_OFFSET,
                    mCombineDz.element_type,
                    source_byte_extent,
                )
                mCombineH1 = _activation_buffer_tensor_with_byte_extent(
                    mCombineH1,
                    activation_buffer_base_ptr,
                    activation_buffer_size_bytes,
                    activation_offsets,
                    MEGA_SECOND_GEMM_OFFSET_BASE + ACTIVATION_SOURCE_Y_OFFSET,
                    mCombineH1.element_type,
                    source_byte_extent,
                )
            mActivationQuantRowQWords = _activation_buffer_tensor_with_byte_extent(
                mActivationQuantRowQWords,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_SECOND_GEMM_OFFSET_BASE + ACTIVATION_A_Q_OFFSET,
                cutlass.Uint32,
                activation_q_byte_extent,
            )
            mActivationQuantRowScale = _activation_buffer_tensor_with_byte_extent(
                mActivationQuantRowScale,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_SECOND_GEMM_OFFSET_BASE + ACTIVATION_A_SCALE_OFFSET,
                cutlass.Uint8,
                activation_row_scale_byte_extent,
            )
            if cutlass.const_expr(use_global_scale_inv):
                mActivationQuantRowGlobalScaleInv = (
                    _activation_buffer_row_global_scale_tensor(
                        mActivationQuantRowGlobalScaleInv,
                        activation_buffer_base_ptr,
                        activation_buffer_size_bytes,
                        activation_offsets,
                        activation_rows,
                        N,
                        sf_vec_size=self.sf_vec_size,
                        sf_dtype_width=self.format.sf_dtype.width,
                        scale_offset_idx=MEGA_SECOND_GEMM_OFFSET_BASE
                        + ACTIVATION_A_SCALE_OFFSET,
                    )
                )
            mActivationQuantColQWords = _activation_buffer_tensor_with_byte_extent(
                mActivationQuantColQWords,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_SECOND_GEMM_OFFSET_BASE + ACTIVATION_COL_Q_OFFSET,
                cutlass.Uint32,
                activation_q_byte_extent,
            )
            mActivationQuantColScale = _activation_buffer_tensor_with_byte_extent(
                mActivationQuantColScale,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                MEGA_SECOND_GEMM_OFFSET_BASE + ACTIVATION_COL_SCALE_OFFSET,
                cutlass.Uint8,
                activation_col_scale_byte_extent,
            )
        del (
            mActivationGatherPtrs,
            activation_quant_total_tiles,
            combine,
            activation_dispatch,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        cluster_cta_rank = cutlass.Int32(0)
        pred_cta0 = True
        if cutlass.const_expr(self.NUM_CTAS == 2):
            cluster_cta_rank = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            pred_cta0 = cluster_cta_rank == 0

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ab_full_mbar = storage.smem_full_mbar.data_ptr()
        ab_empty_mbar = storage.smem_empty_mbar.data_ptr()
        sf_full_mbar = (
            storage.sf_smem_full_mbar.data_ptr() if self.USE_SM103_ULTRA else None
        )
        sf_empty_mbar = (
            storage.sf_smem_empty_mbar.data_ptr() if self.USE_SM103_ULTRA else None
        )
        tmem_full_mbar = storage.tmem_full_mbar.data_ptr()
        tmem_empty_mbar = storage.tmem_empty_mbar.data_ptr()
        tile_consumer_mbar = storage.tile_consumer_mbar.data_ptr()
        tile_producer_mbar = storage.tile_producer_mbar.data_ptr()
        tile_cta_bar_mbar = (
            storage.tile_cta_bar_mbar.data_ptr() if self.NUM_TILE_CTA_BARS > 0 else None
        )
        tmem_dealloc_mbar = (
            storage.tmem_dealloc_mbar.data_ptr() if self.NUM_CTAS == 2 else None
        )
        cross_seam_mbar = (
            storage.cross_seam_mbar.data_ptr() if self.OVERLAPPING_ACCUM else None
        )
        dispatch_tile_id_smem_ptr = storage.dispatch_tile_id_smem.data_ptr()
        dispatch_ptr_smem_ptr = storage.dispatch_ptr_smem.data_ptr()
        if warp_idx == self.EPILOG_WARP_IDS[0]:
            with cute.arch.elect_one():
                for i in range(self.NUM_SMEM_BUFFERS):
                    cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
                    cute.arch.mbarrier_init(ab_full_mbar + i, 2)
                if cutlass.const_expr(self.USE_SM103_ULTRA):
                    # sf_full is armed by both pair-loader warps; sf_empty is
                    # committed once by the MMA consumer.
                    for i in range(
                        self.SM103_SF_RING_TILES * sm103.sf_segments(self.sf_vec_size)
                    ):
                        cute.arch.mbarrier_init(sf_empty_mbar + i, 1)
                        cute.arch.mbarrier_init(sf_full_mbar + i, 2)
                for i in range(self.NUM_TMEM_BUFFERS):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, self.NUM_CTAS)
                for i in range(self.NUM_TILE_BUFFERS):
                    cute.arch.mbarrier_init(tile_consumer_mbar + i, 1)
                    cute.arch.mbarrier_init(tile_producer_mbar + i, self.NUM_CTAS)
                if cutlass.const_expr(self.NUM_TILE_CTA_BARS > 0):
                    for i in range(self.NUM_TILE_CTA_BARS):
                        cute.arch.mbarrier_init(tile_cta_bar_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_dealloc_mbar, 32)
                if cutlass.const_expr(self.OVERLAPPING_ACCUM):
                    cute.arch.mbarrier_init(cross_seam_mbar, self.NUM_CTAS)
        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.NUM_CTAS == 2):
            cute.arch.fence_proxy(kind="async.shared", space="cluster")
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        if warp_idx == self.EPILOG_WARP_IDS[0]:
            cute.arch.alloc_tmem(
                self.num_tmem_alloc_cols,
                storage.tmem_holding_buf,
                is_two_cta=(self.NUM_CTAS == 2),
            )
        metadata_smem_ptr = cute.recast_ptr(
            storage.tensormap_buffer.data_ptr(), dtype=cutlass.Int32
        )
        split_sizes = stage_expert_metadata(
            split_sizes,
            metadata_smem_ptr,
            G,
            synchronize=False,
        )
        work_info_smem_ptr = metadata_smem_ptr + cutlass.Int32(G)
        cute.arch.barrier(
            barrier_id=_BAR_FULL_CTA_SYNC,
            number_of_threads=self.THREADS_PER_CTA,
        )
        if cutlass.const_expr(
            self.DISPATCH_TOTAL_WARPS == 16 and self.DISPATCH_QUANT_WARPS == 4
        ):
            # The narrow, one-CTA/SM layout uses the full 64K-register budget.
            # WGs 0-3/4-7/8-11/12-15 are dispatch/TMA-MMA-idle/epilogue/quant:
            # 4*32*(168+40+168+136) = 65536 registers.
            if warp_idx < 4:
                cute.arch.setmaxregister_increase(168)
            elif warp_idx < 8:
                cute.arch.setmaxregister_decrease(40)
            elif warp_idx < 12:
                cute.arch.setmaxregister_increase(168)
            else:
                cute.arch.setmaxregister_increase(136)
        if warp_idx == self.TMA_AB_WARP_ID:
            for w in cutlass.range_constexpr(2):
                params = (dgrad, wgrad)[w]
                cpasync.prefetch_descriptor(params.tma_atom_a)
                cpasync.prefetch_descriptor(params.tma_atom_b)
                cpasync.prefetch_descriptor(params.tma_atom_sfa)
                cpasync.prefetch_descriptor(params.tma_atom_sfb)
                cpasync.prefetch_descriptor(params.tma_atom_c)

        if warp_idx < self.DISPATCH_QUANT_WARPS:
            schedule_rank = (local_rank + cutlass.Int32(1)) % cutlass.Int32(
                self.world_size
            )
            self._dispatch_warpgroup(
                mDispatchQuantWorkCounter,
                mDispatchQuantDoneCounter,
                dispatch_tile_id_smem_ptr,
                dispatch_ptr_smem_ptr,
                mDispatchQuantGatherPtrs,
                mDispatchQuantRowQWords,
                mDispatchQuantRowScale,
                mDispatchQuantRowGlobalScaleInv,
                mDispatchQuantColQWords,
                mDispatchQuantColScale,
                split_sizes,
                G,
                M,
                K,
                dispatch_quant_total_tiles,
                schedule_rank,
                row_done_counter_offset,
                col_done_counter_offset,
                use_global_scale_inv,
                ring_fc13_done_counter=(
                    mActivationQuantDoneCounter
                    if cutlass.const_expr(self.ACTIVATION_RING_CHUNKS > 0)
                    else None
                ),
                ring_fc13_target=(
                    ceil_div(cutlass.Int32(2) * N, cutlass.Int32(self.BLOCK_SIZE_M))
                    * cutlass.Int32(self.RING_CHUNK_ROW_TILES * self.NUM_CTAS)
                ),
                weight_borrow_args=weight_borrow_args,
                activation_done_counter=mActivationQuantDoneCounter,
            )
        elif warp_idx == self.TMA_AB_WARP_ID:
            self._gemm_tma_role(
                dgrad,
                wgrad,
                storage,
                cluster_cta_rank,
                pred_cta0,
                split_sizes,
                G,
                M,
                N,
                K,
                local_rank,
                counter_ptr,
                mDispatchQuantDoneCounter,
                row_done_counter_offset,
                col_done_counter_offset,
                mActivationQuantDoneCounter,
                activation_row_done_counter_offset,
                activation_col_done_counter_offset,
                tensormaps,
                use_device_tensormaps,
                work_info_smem_ptr,
                load_b=False,
                weight_borrow_args=weight_borrow_args,
            )
        elif warp_idx == self.TMA_B_WARP_ID:
            self._gemm_tma_role(
                dgrad,
                wgrad,
                storage,
                cluster_cta_rank,
                pred_cta0,
                split_sizes,
                G,
                M,
                N,
                K,
                local_rank,
                counter_ptr,
                mDispatchQuantDoneCounter,
                row_done_counter_offset,
                col_done_counter_offset,
                mActivationQuantDoneCounter,
                activation_row_done_counter_offset,
                activation_col_done_counter_offset,
                tensormaps,
                use_device_tensormaps,
                work_info_smem_ptr,
                load_b=True,
            )
        elif warp_idx == self.MMA_WARP_ID:
            fc13_state, fc2_state, problem, sync = self._make_gemm_role_context(
                dgrad,
                wgrad,
                storage,
                cluster_cta_rank,
                pred_cta0,
                split_sizes,
                G,
                M,
                N,
                K,
                local_rank,
                counter_ptr,
            )
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            _gemm_mma_warp(
                params_from_kernel(ChunkedGemmWarpParams, self),
                self._make_mma_pipeline(dgrad, fc13_state, tmem_ptr),
                self._make_mma_pipeline(wgrad, fc2_state, tmem_ptr),
                problem,
                sync,
                work_info_smem_ptr,
                format=self.format,
                cta_group=self.cta_group,
                sf_dtype=self.sf_dtype,
            )
        elif warp_idx in self.EPILOG_WARP_IDS:
            fc13_state, fc2_state, problem, sync = self._make_gemm_role_context(
                dgrad,
                wgrad,
                storage,
                cluster_cta_rank,
                pred_cta0,
                split_sizes,
                G,
                M,
                N,
                K,
                local_rank,
                counter_ptr,
            )
            activation_quant_sync = base._MegaDispatchQuantSync(
                done_counter=mActivationQuantDoneCounter,
                done_counter_offsets=(
                    activation_row_done_counter_offset,
                    activation_col_done_counter_offset,
                ),
            )
            tensormap_manager = utils.TensorMapManager(
                utils.TensorMapUpdateMode.SMEM, 128
            )
            scatter_ptr_smem_ptr = storage.scatter_ptr_smem.data_ptr()
            a_global_scale_inv_smem_ptr = (
                storage.a_global_scale_inv.data_ptr() if use_global_scale_inv else None
            )
            epilogue_tidx = tidx - cutlass.Int32(
                self.EPILOG_WARP_IDS[0] * self.THREADS_PER_WARP
            )
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            self._gemm_epilogue_warpgroup(
                epilogue_tidx,
                self._make_epilog_pipeline(
                    params=dgrad,
                    state=fc13_state,
                    sC=fc13_state.sC,
                    tidx=epilogue_tidx,
                    tmem_ptr=tmem_ptr,
                ),
                self._make_epilog_pipeline(
                    params=wgrad,
                    state=fc2_state,
                    sC=fc2_state.combine_sC,
                    tidx=epilogue_tidx,
                    tmem_ptr=tmem_ptr,
                ),
                problem,
                sync,
                activation_quant_sync,
                tensormap_manager,
                tensormaps,
                mCombineScatterPtrs,
                (
                    mCombineDz
                    if cutlass.const_expr(
                        self.INTERLEAVED_FC13 and self.format is NVFP4
                    )
                    else mActivationQuantRowQWords
                ),
                mActivationQuantRowScale,
                scatter_ptr_smem_ptr,
                use_device_tensormaps,
                work_info_smem_ptr,
                a_global_scale_inv_smem_ptr,
                cutlass.Int64(mDispatchQuantRowGlobalScaleInv.iterator.toint()),
                cutlass.Int64(mActivationQuantRowGlobalScaleInv.iterator.toint()),
                dgrad_b_global_scale_inv_ptr,
                wgrad_b_global_scale_inv_ptr,
                use_global_scale_inv,
            )
            if warp_idx == self.EPILOG_WARP_IDS[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=(self.NUM_CTAS == 2))
            cute.arch.barrier(
                barrier_id=_BAR_EPILOG_SYNC,
                number_of_threads=self.EPILOG_WG_THREADS,
            )
            if warp_idx == self.EPILOG_WARP_IDS[0]:
                if cutlass.const_expr(self.NUM_CTAS == 2):
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar,
                        peer_cta_rank_in_cluster=cluster_cta_rank ^ 1,
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar, 0)
                cute.arch.dealloc_tmem(
                    tmem_ptr,
                    self.num_tmem_alloc_cols,
                    is_two_cta=(self.NUM_CTAS == 2),
                )
        elif (warp_idx >= self.FORWARD_QUANT_FIRST_WARP) and (
            warp_idx < self.FORWARD_QUANT_LAST_WARP
        ):
            schedule_rank = (local_rank + cutlass.Int32(1)) % cutlass.Int32(
                self.world_size
            )
            if cutlass.const_expr(self.INTERLEAVED_FC13 and self.format is not NVFP4):
                # The shared pipeline fixes this warp topology, and SMEM already
                # limits the kernel to one CTA per SM, so these warps exit early.
                thread_exit()
            elif cutlass.const_expr(self.format is NVFP4):
                nvfp4_row_reduce_smem_ptr = (
                    storage.nvfp4_row_reduce.data_ptr()
                    if use_global_scale_inv
                    else None
                )
                mNvfp4RecipLut = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        nvfp4_recip_lut_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((256,), stride=(1,)),
                )
                self._combine_swiglu_nvfp4_row_quant_producer_body(
                    mWorkCounter=mActivationQuantWorkCounter,
                    mDoneCounter=mActivationQuantDoneCounter,
                    tile_smem_ptr=nvfp4_row_reduce_smem_ptr,
                    mFwdX=mCombineDz,
                    mFwdY=mCombineH1,
                    mRowQWords=mActivationQuantRowQWords,
                    mRowScale=mActivationQuantRowScale,
                    mRowGlobalScaleInv=mActivationQuantRowGlobalScaleInv,
                    mNvfp4RecipLut=mNvfp4RecipLut,
                    split_sizes=split_sizes,
                    G=G,
                    M=M,
                    local_rank=schedule_rank,
                    wait_for_source_tile=True,
                    source_done_counter_offset=activation_col_done_counter_offset,
                    barrier_id_base=self.FORWARD_QUANT_SYNC_BAR,
                    precomputed_swiglu=self.INTERLEAVED_FC13,
                    ring_h1_done_offset=cutlass.Int32(
                        self.ACTIVATION_RING_H1_DONE_OFFSET
                    ),
                )
            else:
                self._mega_forward_swiglu_quant_producer_body(
                    mWorkCounter=mActivationQuantWorkCounter,
                    mDoneCounter=mActivationQuantDoneCounter,
                    h1_done_counter_offset=activation_col_done_counter_offset,
                    mFwdX=mCombineDz,
                    mFwdY=mCombineH1,
                    mScatter=mCombineScatterPtrs,
                    mRowQWords=mActivationQuantRowQWords,
                    mRowScale=mActivationQuantRowScale,
                    mColQWords=mActivationQuantColQWords,
                    mColScale=mActivationQuantColScale,
                    split_sizes=split_sizes,
                    G=G,
                    M=M,
                    K=N,
                    local_rank=schedule_rank,
                )

        if cutlass.const_expr(self.NUM_CTAS == 2):
            # Early thread_exit calls remove only completed producer roles. Every
            # still-active thread in every CTA, including subclasses, reaches this
            # aligned rendezvous before peer DSMEM can be reclaimed.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()


class _ReleaseOnlyChunkedMegaBlockScaledGroupedGemmKernel(
    ChunkedMegaBlockScaledGroupedGemmKernel
):
    """Publish wide-row NVFP4 tiles without a device-wide acquire-release fence.

    Every writer joins the quant-group barrier after its proxy fence, so the
    elected lane's release is sufficient for the TMA consumer. Other producer
    topologies retain the base protocol because their writers do not share this
    group-wide publication point.
    """

    @cute.jit
    def _dispatch_quant_signal_group_tile_done(
        self,
        mDoneCounter: cute.Tensor,
        act_tile_slot: cutlass.Int32,
        quant_group: cutlass.Int32,
        quant_group_lane: cutlass.Int32,
        completed_quant_col_tiles: cutlass.Int32,
        barrier_id_base: cutlass.Constexpr[int] = 3,
    ) -> cutlass.Uint32:
        cute.arch.fence_proxy("async.global")
        cute.arch.barrier(
            barrier_id=barrier_id_base + quant_group,
            number_of_threads=self.DISPATCH_QUANT_GROUP_THREADS,
        )
        # Mirror the base contract: return the pre-add counter value
        # (meaningful on group lane 0 only) for tile-completion detection.
        prev_count = cutlass.Uint32(0)
        if quant_group_lane == cutlass.Int32(0):
            # The group barrier joins every writer's proxy publication before
            # this lane's release publishes the tile to the TMA consumer.
            counter_slot_ptr = cute.recast_ptr(
                mDoneCounter.iterator + act_tile_slot,
                dtype=cutlass.Uint32,
            )
            prev_count = cute.arch.atomic_add(
                counter_slot_ptr,
                cutlass.Uint32(completed_quant_col_tiles),
                sem="release",
                scope="gpu",
            )
        return prev_count
