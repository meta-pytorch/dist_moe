# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side kernel class and plan types for the Mega block-scaled grouped GEMM."""

import weakref
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import cpasync

from . import sm103_blockscaled_helpers as sm103
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
    MISSING_ACTIVATION_OFFSET,
)
from .activation_buffer_kernel import (
    _activation_buffer_col_scale_storage_byte_extent,
    _activation_buffer_row_scale_storage_byte_extent,
    _activation_buffer_tensor_with_byte_extent,
)
from .blockscaled_gemm_tiles import (
    _prepare_blockscaled_problem_tensormaps,
    _stage_a_global_scale_inv,
)
from .blockscaled_grouped_gemm_kernel import (
    _activation_buffer_operand_offsets,
    _activation_buffer_pointer_strides,
)
from .combine_swiglu_quant import (
    _apply_epilog_global_scale_inv,
    _combine_swiglu_bwd_quant_producer_body,
    _combine_swiglu_fwd_quantize_tile,
    _combine_swiglu_quant_fetch_warp_tile_batch,
)
from .dispatch_quant import (
    _dispatch_copy_blockscaled_work_tile,
    _dispatch_quant_fetch_group_tile_id,
    _dispatch_quantize_tile,
    _mega_dispatch_quant_signal_group_tile_done,
    _mega_forward_wait_h1_tile,
    _wait_counter_at_least,
)
from .dist_blockscaled_grouped_gemm import (
    DistBlockScaledGroupedGemmKernel,
)
from .gemm_warp_roles import (
    _mega_mma_consumer_body,
    _mega_problem_specs,
    _mega_tma_producer_body,
)
from .grouped_gemm_kernel import (
    _activation_buffer_rows,
    _BAR_EPILOG_SYNC,
    _BAR_FULL_CTA_SYNC,
    _epilog_wait_pending_tma_store,
    _swap_if,
    _transpose_c_if_swap,
)
from .params import (
    BlockscaledGemmKernelParams as _MegaGemmKernelParams,
    BlockscaledGemmLayout as _MegaGemmLayoutBundle,
    BlockscaledGemmPrologue as _MegaGemmPrologue,
    BlockscaledPointerStrideArgs,
    BlockscaledTensormapParams as _MegaTensormapParams,
    BlockscaledTensormapProblem,
    ceil_div,
    CombineSwigluQuantParams,
    DispatchQuantParams,
    DispatchQuantSync as _MegaDispatchQuantSync,
    GroupedGemmEpilogPipeline as _MegaEpilogPipeline,
    GroupedGemmMmaPipeline as _MegaMmaPipeline,
    GroupedGemmPipelineSync as _MegaPipelineSync,
    GroupedGemmProblem as _MegaProblem,
    GroupedGemmTmaPipeline as _MegaTmaPipeline,
    MegaPipelineParams,
    params_from_kernel,
    swap_ab_pointer_strides,
)
from .tile_scheduler import (
    _DGRAD,
    _FPROP,
    _get_bufidx_phase,
    _ring_cached_slot,
    _WGRAD,
    advance_blockscaled_scale_start,
    blockscaled_scale_row_start,
    load_mega_work_info,
    MEGA_WORK_INFO_FIELDS,
    MegaDynamicScheduler,
    MegaProblemVisitor,
    MegaStaticScheduler,
    stage_expert_metadata,
)


class _MegaFusedUnsupportedError(NotImplementedError):
    pass


class MegaBlockScaledGroupedGemmKernel(DistBlockScaledGroupedGemmKernel):
    """Backward mega-kernel with fused quant, DGRAD, and WGRAD."""

    MEGA_BACKWARD_DISPATCH_MODE: int = 5
    MEGA_BACKWARD_COMBINE_MODE: int = 6
    MEGA_FORWARD_MODE: int = 7
    MUTABLE_COMBINE_ACC_FRAGMENT: bool = False
    COMBINE_SWIGLU_WORK_TILES_PER_FETCH: int = 1
    MEGA_DGRAD_PROBLEM_TYPE: int = _DGRAD
    MEGA_WGRAD_PROBLEM_TYPE: int = _WGRAD
    MEGA_TENSORMAP_DESCRIPTOR_COUNT: int = 10
    MEGA_DGRAD_TENSORMAP_BASE: int = 0
    MEGA_WGRAD_TENSORMAP_BASE: int = 5
    # Row extent of the activation ring, shared by both ringed activation
    # operands (the FC13 x and FC2 h2 B tensormaps); 0 keeps full-extent
    # per-group descriptors. Set by the chunked forward kernel only.
    ACTIVATION_RING_B_ROWS: int = 0
    # Expert-borrow weight streaming (chunked forward only): the trailing
    # SLOTS groups' A/SFA descriptors point at the borrow slot buffer.
    WEIGHT_BORROW_SLOTS: int = 0

    MEGA_DGRAD_LAYOUT_IDX: int = 0
    MEGA_WGRAD_LAYOUT_IDX: int = 1

    DISPATCH_TOTAL_WARPS: int = 16
    # Warps 0-5 are reserved for GEMM. Five producer groups preload gather
    # pointers independently; each group quantizes two 32x64 microtiles per warp.
    DISPATCH_QUANT_WARP_IDS: tuple[int, ...] = tuple(range(6, 16))
    DISPATCH_QUANT_FIRST_WARP: int = DISPATCH_QUANT_WARP_IDS[0]
    DISPATCH_QUANT_WARPS: int = len(DISPATCH_QUANT_WARP_IDS)
    DISPATCH_QUANT_WARPS_PER_GROUP: int = 2
    FORWARD_SWIGLU_QUANT_WARPS: int = 2
    FORWARD_INPUT_QUANT_FIRST_WARP: int = (
        DISPATCH_QUANT_FIRST_WARP + FORWARD_SWIGLU_QUANT_WARPS
    )
    COMBINE_ACTIVATION_DISPATCH_FIRST_WARP: int = 12
    FORWARD_COPY_COL_TILES_PER_WORK: int = 4
    DISPATCH_QUANT_GROUPS: int = DISPATCH_QUANT_WARPS // DISPATCH_QUANT_WARPS_PER_GROUP
    DISPATCH_QUANT_GROUP_THREADS: int = (
        DistBlockScaledGroupedGemmKernel.THREADS_PER_WARP
        * DISPATCH_QUANT_WARPS_PER_GROUP
    )
    DISPATCH_QUANT_MICROTILES_PER_WARP: int = 2
    DISPATCH_THREADS_PER_CTA: int = (
        DistBlockScaledGroupedGemmKernel.THREADS_PER_WARP * DISPATCH_TOTAL_WARPS
    )
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: int = (
        DISPATCH_QUANT_WARPS_PER_GROUP
        * DISPATCH_QUANT_MICROTILES_PER_WARP
        * DistBlockScaledGroupedGemmKernel.DISPATCH_QUANT_SCALE_COLS_PER_WARP
    )

    def _setup_gemm_layout_bundle(
        self,
        *,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_c: cute.Tensor,
        tensor_c_eff: cute.Tensor,
    ) -> _MegaGemmLayoutBundle:
        self.a_dtype = tensor_a.element_type
        self.b_dtype = tensor_b.element_type
        self.c_dtype = tensor_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(tensor_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(tensor_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(tensor_c_eff)
        self._setup_attributes()
        return _MegaGemmLayoutBundle(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            c_dtype=self.c_dtype,
            c_layout=self.c_layout,
            tiled_mma=self.tiled_mma,
            tiled_mma_sfb=self.tiled_mma_sfb,
            mma_tiler=self.mma_tiler,
            mma_tiler_sfb=self.mma_tiler_sfb,
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_layout_vmnk=self.cluster_layout_vmnk,
            cluster_layout_sfb_vmnk=self.cluster_layout_sfb_vmnk,
            epi_tile=self.epi_tile,
            a_smem_layout_staged=self.a_smem_layout_staged,
            b_smem_layout_staged=self.b_smem_layout_staged,
            sfa_smem_layout_staged=self.sfa_smem_layout_staged,
            sfb_smem_layout_staged=self.sfb_smem_layout_staged,
            sfb_tma_smem_layout_staged=self.sfb_tma_smem_layout_staged,
            epi_smem_layout_staged=self.epi_smem_layout_staged,
        )

    def _make_gemm_tma_atoms(
        self,
        *,
        layout: _MegaGemmLayoutBundle,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_c_eff: cute.Tensor,
        tensor_sfa: cute.Tensor,
        tensor_sfb: cute.Tensor,
        output_accum: cutlass.Constexpr[bool] = False,
    ) -> tuple[tuple, tuple]:
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            tensor_sfa_view = cute.make_tensor(
                tensor_sfa.iterator,
                sm103.make_gmem_layout_sf(tensor_a.shape, self.sf_vec_size),
            )
        else:
            tensor_sfa_view = cute.make_tensor(
                tensor_sfa.iterator,
                blockscaled_utils.tile_atom_to_shape_SF(
                    tensor_a.shape, self.sf_vec_size
                ),
            )
        tensor_sfb_data_shape = tensor_b.shape
        # The 4x/2x inflation gives each compact token tile its own 128-row
        # scale page. The resulting SMEM overhead is too small to change the
        # legal pipeline depth, so retain the simpler private-page geometry.
        if self.SWAP_AB and self.BLOCK_SIZE_N in (32, 64):
            scale_storage_factor = 4 if self.BLOCK_SIZE_N == 32 else 2
            tensor_sfb_data_shape = (
                tensor_b.shape[0] * scale_storage_factor,
                tensor_b.shape[1],
                tensor_b.shape[2],
            )
        elif self.SWAP_AB and self.BLOCK_SIZE_N >= 96 and self.BLOCK_SIZE_N % 128 != 0:
            scale_page_rows = ((self.BLOCK_SIZE_N + 127) // 128) * 128
            tensor_sfb_data_shape = (
                tensor_b.shape[0] // self.BLOCK_SIZE_N * scale_page_rows,
                tensor_b.shape[1],
                tensor_b.shape[2],
            )
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            tensor_sfb_view = cute.make_tensor(
                tensor_sfb.iterator,
                sm103.make_gmem_layout_sf(tensor_sfb_data_shape, self.sf_vec_size),
            )
        else:
            tensor_sfb_view = cute.make_tensor(
                tensor_sfb.iterator,
                blockscaled_utils.tile_atom_to_shape_SF(
                    tensor_sfb_data_shape,
                    self.sf_vec_size,
                ),
            )
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            mma_sf_tiler = (
                layout.cta_tile_shape_mnk[0],
                cute.round_up(layout.cta_tile_shape_mnk[1], 128),
                layout.cta_tile_shape_mnk[2] // sm103.sf_segments(self.sf_vec_size),
            )
            tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
                sm100_utils.cluster_shape_to_tma_atom_A(
                    self.cluster_shape_mn, layout.tiled_mma.thr_id
                ),
                cute.recast_tensor(tensor_a, cutlass.Uint8),
                sm103.adapt_layout_for_tma_ab(
                    sm103.make_smem_layout_ab(
                        layout.tiled_mma,
                        layout.mma_tiler,
                        sm103.SM103_AB_SEGMENTS,
                        is_a=True,
                    )
                ),
                (cute.size(layout.tiled_mma.tv_layout_A[1][0]), 384),
                self.cluster_shape_mn[1],
                internal_type=cutlass.Uint8,
            )
            tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
                sm100_utils.cluster_shape_to_tma_atom_B(
                    self.cluster_shape_mn, layout.tiled_mma.thr_id
                ),
                cute.recast_tensor(tensor_b, cutlass.Uint8),
                sm103.adapt_layout_for_tma_ab(
                    sm103.make_smem_layout_ab(
                        layout.tiled_mma,
                        layout.mma_tiler,
                        sm103.SM103_AB_SEGMENTS,
                        is_a=False,
                    )
                ),
                (cute.size(layout.tiled_mma.tv_layout_B[1][0]), 384),
                self.cluster_shape_mn[0] // cute.size(layout.tiled_mma.thr_id.shape),
                internal_type=cutlass.Uint8,
            )
            tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(
                sm100_utils.cluster_shape_to_tma_atom_A(
                    self.cluster_shape_mn, layout.tiled_mma.thr_id
                ),
                tensor_sfa_view,
                sm103.adapt_layout_for_tma_sf(
                    cute.slice_(layout.sfa_smem_layout_staged, (None, None, None, 0))
                ),
                (mma_sf_tiler[0], mma_sf_tiler[2]),
                self.cluster_shape_mn[1],
                internal_type=cutlass.Uint8,
            )
            tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
                sm100_utils.cluster_shape_to_tma_atom_SFB(
                    self.cluster_shape_mn, layout.tiled_mma.thr_id
                ),
                tensor_sfb_view,
                sm103.adapt_layout_for_tma_sf(
                    cute.slice_(layout.sfb_smem_layout_staged, (None, None, None, 0))
                ),
                (mma_sf_tiler[1], mma_sf_tiler[2]),
                self.cluster_shape_mn[0]
                // cute.size(layout.tiled_mma_sfb.thr_id.shape),
                internal_type=cutlass.Uint8,
            )
        else:
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                sm100_utils.cluster_shape_to_tma_atom_A(
                    self.cluster_shape_mn, layout.tiled_mma.thr_id
                ),
                tensor_a,
                cute.slice_(layout.a_smem_layout_staged, (None, None, None, 0)),
                layout.mma_tiler,
                layout.tiled_mma,
                layout.cluster_layout_vmnk.shape,
                internal_type=(
                    cutlass.Uint8
                    if layout.a_dtype.width != layout.b_dtype.width
                    and layout.a_dtype.width < 8
                    else None
                ),
            )
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                sm100_utils.cluster_shape_to_tma_atom_B(
                    self.cluster_shape_mn, layout.tiled_mma.thr_id
                ),
                tensor_b,
                cute.slice_(layout.b_smem_layout_staged, (None, None, None, 0)),
                layout.mma_tiler,
                layout.tiled_mma,
                layout.cluster_layout_vmnk.shape,
                internal_type=(
                    cutlass.Uint8
                    if layout.a_dtype.width != layout.b_dtype.width
                    and layout.b_dtype.width < 8
                    else None
                ),
            )
            tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
                sm100_utils.cluster_shape_to_tma_atom_A(
                    self.cluster_shape_mn, layout.tiled_mma.thr_id
                ),
                tensor_sfa_view,
                cute.slice_(layout.sfa_smem_layout_staged, (None, None, None, 0)),
                layout.mma_tiler,
                layout.tiled_mma,
                layout.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sm100_utils.cluster_shape_to_tma_atom_SFB(
                    self.cluster_shape_mn, layout.tiled_mma.thr_id
                ),
                tensor_sfb_view,
                cute.slice_(layout.sfb_tma_smem_layout_staged, (None, None, None, 0)),
                layout.mma_tiler_sfb,
                layout.tiled_mma_sfb,
                layout.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp()
            if output_accum
            else cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c_eff,
            cute.slice_(layout.epi_smem_layout_staged, (None, None, 0)),
            cute.composition(
                cute.make_identity_layout(tensor_c_eff.shape),
                layout.epi_tile,
            ),
        )
        return (
            (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb, tma_atom_c),
            (tma_tensor_a, tma_tensor_b, tma_tensor_sfa, tma_tensor_sfb, tma_tensor_c),
        )

    def _make_first_gemm_tma_atoms(self, **kwargs) -> tuple[tuple, tuple]:
        return self._make_gemm_tma_atoms(**kwargs)

    def _make_gemm_kernel_params(
        self,
        layout: _MegaGemmLayoutBundle,
        tma_atoms,
        tma_tensors,
    ) -> _MegaGemmKernelParams:
        return _MegaGemmKernelParams(
            *tma_atoms,
            *tma_tensors,
            a_smem_layout_staged=layout.a_smem_layout_staged,
            b_smem_layout_staged=layout.b_smem_layout_staged,
            sfa_smem_layout_staged=layout.sfa_smem_layout_staged,
            sfb_smem_layout_staged=layout.sfb_smem_layout_staged,
            sfb_tma_smem_layout_staged=layout.sfb_tma_smem_layout_staged,
            epi_smem_layout_staged=layout.epi_smem_layout_staged,
            epi_tile=layout.epi_tile,
            cluster_layout_vmnk=layout.cluster_layout_vmnk,
            cluster_layout_sfb_vmnk=layout.cluster_layout_sfb_vmnk,
            tiled_mma=layout.tiled_mma,
            tiled_mma_sfb=layout.tiled_mma_sfb,
            mma_tiler=layout.mma_tiler,
            mma_tiler_sfb=layout.mma_tiler_sfb,
            cta_tile_shape_mnk=layout.cta_tile_shape_mnk,
            c_layout=layout.c_layout,
            c_dtype=layout.c_dtype,
            a_dtype_width=layout.a_dtype.width,
            b_dtype_width=layout.b_dtype.width,
            combine_c_smem_layout_staged=self._make_mega_combine_c_smem_layout(
                layout.epi_tile
            ),
        )

    def _make_mega_combine_c_smem_layout(self, epi_tile: cute.Tile) -> cute.Layout:
        epi_m = cute.size(epi_tile[0])
        epi_n = cute.size(epi_tile[1])
        skew = 8 if self.NUM_TMEM_BUFFERS == 1 else 0
        if self.SWAP_AB:
            col_stride = epi_m + skew
            return cute.make_layout(
                (epi_m, epi_n, 1),
                stride=(1, col_stride, epi_m * col_stride),
            )
        row_stride = epi_n + skew
        return cute.make_layout(
            (epi_m, epi_n, 1),
            stride=(row_stride, 1, epi_m * row_stride),
        )

    def _make_tensormap_params(
        self,
        tma_atoms,
        pointer_strides: BlockscaledPointerStrideArgs,
        elem_sizes,
        dtypes,
    ) -> _MegaTensormapParams:
        base_ptrs, tensor_strides, sf_strides = pointer_strides
        return _MegaTensormapParams(
            *tma_atoms,
            base_ptrs=base_ptrs,
            strides=tensor_strides,
            sf_strides=sf_strides,
            elem_sizes=elem_sizes,
            dtypes=dtypes,
        )

    def _region_tiled_mma(self, params) -> cute.TiledMma:
        """Per-trace-region tiled MMA. The SM103 ultra MMA materializes its
        IR handle at first use, so sharing one instance across warp-role
        regions trips MLIR dominance; rebuild it fresh per region."""
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            return sm103.make_tiled_mma(
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                (
                    self.BLOCK_SIZE_M // self.NUM_MMAS,
                    self.MMA_ATOM_N,
                ),
            )
        return params.tiled_mma

    def _make_gemm_prologue(
        self,
        params: _MegaGemmKernelParams,
        storage,
        cluster_cta_rank: cutlass.Int32,
        bid,
    ) -> _MegaGemmPrologue:
        region_tiled_mma = self._region_tiled_mma(params)
        sA = storage.sA.get_tensor(
            params.a_smem_layout_staged.outer,
            swizzle=params.a_smem_layout_staged.inner,
        )
        sB = storage.sB.get_tensor(
            params.b_smem_layout_staged.outer,
            swizzle=params.b_smem_layout_staged.inner,
        )
        sC = cute.make_tensor(
            cute.recast_ptr(
                storage.sC.data_ptr(),
                params.epi_smem_layout_staged.inner,
                dtype=params.c_dtype,
            ),
            params.epi_smem_layout_staged.outer,
        )
        assert params.combine_c_smem_layout_staged is not None
        combine_sC = cute.make_tensor(
            cute.recast_ptr(storage.sC.data_ptr(), dtype=params.c_dtype),
            params.combine_c_smem_layout_staged,
        )
        sSFA = storage.sSFA.get_tensor(params.sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(params.sfb_smem_layout_staged)
        sSFB_tma = storage.sSFB.get_tensor(params.sfb_tma_smem_layout_staged)

        if cutlass.const_expr(self.USE_SM103_ULTRA):
            ab_mma_tiler = (
                params.mma_tiler[0],
                params.mma_tiler[1],
                params.mma_tiler[2] // 2,
            )
            mma_sf_tiler = (
                params.cta_tile_shape_mnk[0],
                cute.round_up(params.cta_tile_shape_mnk[1], 128),
                params.cta_tile_shape_mnk[2] // sm103.sf_segments(self.sf_vec_size),
            )
            gA = cute.local_tile(
                params.mA,
                cute.slice_(ab_mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            gB = cute.local_tile(
                params.mB,
                cute.slice_(ab_mma_tiler, (0, None, None)),
                (None, None, None),
            )
            gSFA = cute.local_tile(
                params.mSFA,
                cute.slice_(mma_sf_tiler, (None, 0, None)),
                (None, None, None),
            )
            gSFB = cute.local_tile(
                params.mSFB,
                cute.slice_(mma_sf_tiler, (0, None, None)),
                (None, None, None),
            )
        else:
            gA = cute.local_tile(
                params.mA,
                cute.slice_(params.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            gB = cute.local_tile(
                params.mB,
                cute.slice_(params.mma_tiler, (0, None, None)),
                (None, None, None),
            )
            gSFA = cute.local_tile(
                params.mSFA,
                cute.slice_(params.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            gSFB = cute.local_tile(
                params.mSFB,
                cute.slice_(params.mma_tiler_sfb, (0, None, None)),
                (None, None, None),
            )
        gC = cute.local_tile(
            params.mC,
            cute.slice_(params.mma_tiler, (None, None, 0)),
            (None, None, None),
        )

        mma_tile_coord_v = bid[0] % cute.size(region_tiled_mma.thr_id.shape)
        block_in_cluster_coord_vmnk = params.cluster_layout_vmnk.get_flat_coord(
            cluster_cta_rank
        )
        block_in_cluster_coord_sfb_vmnk = params.cluster_layout_sfb_vmnk.get_flat_coord(
            cluster_cta_rank
        )
        thr_mma = region_tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = params.tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgC = thr_mma.partition_C(gC)
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            tCgA_tmp = thr_mma.partition_A(gA)
            cta_tCgA = cute.make_tensor(
                tCgA_tmp.iterator,
                sm103.append_coalesce_layout(tCgA_tmp.layout),
            )
            tCgA = cute.make_tensor(
                cta_tCgA.iterator,
                cute.tiled_divide(
                    cta_tCgA.layout,
                    (cute.size(region_tiled_mma.tv_layout_A[1][0]), 128),
                ),
            )
            tCgB_tmp = thr_mma.partition_B(gB)
            cta_tCgB = cute.make_tensor(
                tCgB_tmp.iterator,
                sm103.append_coalesce_layout(tCgB_tmp.layout),
            )
            tCgB = cute.make_tensor(
                cta_tCgB.iterator,
                cute.tiled_divide(
                    cta_tCgB.layout,
                    (cute.size(region_tiled_mma.tv_layout_B[1][0]), 128),
                ),
            )
            tCgSFA = cute.make_tensor(
                gSFA.iterator,
                cute.tiled_divide(
                    gSFA.layout,
                    (mma_sf_tiler[0], mma_sf_tiler[2]),
                ),
            )
            tCgSFB = cute.make_tensor(
                gSFB.iterator,
                cute.tiled_divide(
                    gSFB.layout,
                    (mma_sf_tiler[1], mma_sf_tiler[2]),
                ),
            )
        else:
            tCgA = thr_mma.partition_A(gA)
            tCgB = thr_mma.partition_B(gB)
            tCgSFA = thr_mma.partition_A(gSFA)
            tCgSFB = thr_mma_sfb.partition_B(gSFB)
        a_cta_layout = cute.make_layout(
            cute.slice_(params.cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(params.cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            tAsA, tAgA = cpasync.tma_partition(
                params.tma_atom_a,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 1),
            )
            tBsB, tBgB = cpasync.tma_partition(
                params.tma_atom_b,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 1),
            )
        else:
            tAsA, tAgA = cpasync.tma_partition(
                params.tma_atom_a,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            tBsB, tBgB = cpasync.tma_partition(
                params.tma_atom_b,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            params.tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        sfb_cta_layout = cute.make_layout(
            cute.slice_(params.cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            params.tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB_tma, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        if self.BLOCK_SIZE_N in (32, 64):
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            # Ultra consumes raw Uint8 SMEM through explicit descriptors in
            # `sm103.make_desc_and_call_mma`; no MMA fragments.
            tCrA = sA
            tCrB = sB
        else:
            tCrA = region_tiled_mma.make_fragment_A(sA)
            tCrB = region_tiled_mma.make_fragment_B(sB)
        acc_shape = region_tiled_mma.partition_shape_C(
            (self.BLOCK_SIZE_M, self.BLOCK_SIZE_N)
        )
        tCtAcc_fake = region_tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.NUM_TMEM_BUFFERS)
        )
        if cutlass.const_expr(self.OVERLAPPING_ACCUM):
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (self.BLOCK_SIZE_N - self._num_sf_tmem_cols)
                        * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        return _MegaGemmPrologue(
            sA=sA,
            sB=sB,
            sC=sC,
            sSFA=sSFA,
            sSFB=sSFB,
            tCgC=tCgC,
            tAsA=tAsA,
            tAgA=tAgA,
            tBsB=tBsB,
            tBgB=tBgB,
            tAsSFA=cute.filter_zeros(tAsSFA),
            tAgSFA=(
                tAgSFA
                if cutlass.const_expr(self.USE_SM103_ULTRA)
                else cute.filter_zeros(tAgSFA)
            ),
            tBsSFB=cute.filter_zeros(tBsSFB),
            tBgSFB=(
                tBgSFB
                if cutlass.const_expr(self.USE_SM103_ULTRA)
                else cute.filter_zeros(tBgSFB)
            ),
            tCrA=tCrA,
            tCrB=tCrB,
            tCtAcc_fake=tCtAcc_fake,
            block_in_cluster_coord_vmnk=block_in_cluster_coord_vmnk,
            block_in_cluster_coord_sfb_vmnk=block_in_cluster_coord_sfb_vmnk,
            combine_sC=combine_sC,
        )

    def _make_first_gemm_prologue(self, *args, **kwargs) -> _MegaGemmPrologue:
        return self._make_gemm_prologue(*args, **kwargs)

    def _make_tma_pipeline(
        self,
        params: _MegaGemmKernelParams,
        state: _MegaGemmPrologue,
        num_tma_load_bytes: int,
    ) -> _MegaTmaPipeline:
        return _MegaTmaPipeline(
            tma_atom_a=params.tma_atom_a,
            tma_atom_b=params.tma_atom_b,
            tma_atom_sfa=params.tma_atom_sfa,
            tma_atom_sfb=params.tma_atom_sfb,
            gA=state.tAgA,
            gB=state.tBgB,
            gSFA=state.tAgSFA,
            gSFB=state.tBgSFB,
            sA=state.tAsA,
            sB=state.tBsB,
            sSFA=state.tAsSFA,
            sSFB=state.tBsSFB,
            num_tma_load_bytes=num_tma_load_bytes,
        )

    def _make_mma_pipeline(
        self,
        params: _MegaGemmKernelParams,
        state: _MegaGemmPrologue,
        tmem_ptr: cute.Pointer,
    ) -> _MegaMmaPipeline:
        region_tiled_mma = self._region_tiled_mma(params)
        tCtAcc_base = cute.make_tensor(tmem_ptr, state.tCtAcc_fake.layout)
        tCtSFA, tCtSFB, tCtSFB_field = self._scale_tmem_tensors(
            tmem_ptr,
            tCtAcc_base,
            region_tiled_mma,
            params.mma_tiler,
            params.sfa_smem_layout_staged,
            params.sfb_smem_layout_staged,
        )
        tCtSFA_field = tCtSFA
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            sfa_field_layout, sfb_field_layout = sm103.make_sf_tmem_field_layouts(
                region_tiled_mma,
                params.mma_tiler,
                params.cta_tile_shape_mnk,
                self.sf_vec_size,
            )
            tCtSFA_field = cute.make_tensor(tCtSFA.iterator, sfa_field_layout)
            tCtSFB_field = cute.make_tensor(tCtSFB.iterator, sfb_field_layout)
        return _MegaMmaPipeline(
            tiled_mma=region_tiled_mma,
            tCrA=state.tCrA,
            tCrB=state.tCrB,
            tCtAcc_base=tCtAcc_base,
            tCtSFA=tCtSFA_field,
            tCtSFA_copy=tCtSFA,
            tCtSFB=tCtSFB_field,
            tCtSFB_copy=tCtSFB,
            sSFA=state.sSFA,
            sSFB=state.sSFB,
        )

    def _make_epilog_pipeline(
        self,
        *,
        params: _MegaGemmKernelParams,
        state: _MegaGemmPrologue,
        sC: cute.Tensor,
        tidx: cutlass.Int32,
        tmem_ptr: cute.Pointer,
    ) -> _MegaEpilogPipeline:
        tCtAcc_base = cute.make_tensor(tmem_ptr, state.tCtAcc_fake.layout)
        return _MegaEpilogPipeline(
            c_atom=params.tma_atom_c,
            tCtAcc_base=tCtAcc_base,
            c_tensor=state.tCgC,
            sC=sC,
            epi_tile=params.epi_tile,
            cta_tile_shape_mnk=params.cta_tile_shape_mnk,
            c_layout=params.c_layout,
            c_dtype=params.c_dtype,
            a_dtype_width=params.a_dtype_width,
        )

    def _make_epilog_copy_partitions(
        self,
        tidx: cutlass.Int32,
        pipeline: _MegaEpilogPipeline,
    ) -> tuple:
        tiled_copy_t2r, _, tTR_rAcc = self._epilog_tmem_copy_and_partition(
            tidx=tidx,
            tAcc=pipeline.tCtAcc_base,
            gC_mnl=pipeline.c_tensor,
            epi_tile=pipeline.epi_tile,
            cta_tile_shape_mnk=pipeline.cta_tile_shape_mnk,
            c_layout=pipeline.c_layout,
            c_dtype=pipeline.c_dtype,
            use_2cta_instrs=self.NUM_CTAS == 2,
            mma_m_idx=0,
            mma_n_idx=0,
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, pipeline.sC.element_type)
        tiled_copy_r2s, tRS_rC, tRS_sC = self._epilog_smem_copy_and_partition(
            tiled_copy_t2r=tiled_copy_t2r,
            tTR_rC=tTR_rC,
            tidx=tidx,
            sC=pipeline.sC,
            c_layout=pipeline.c_layout,
            c_dtype=pipeline.c_dtype,
        )
        return tTR_rAcc, tiled_copy_r2s, tRS_rC, tRS_sC

    @cute.jit
    def _mega_combine_epilog_consumer_tile(  # noqa: C901
        self,
        tidx: cutlass.Int32,
        pipeline: _MegaEpilogPipeline,
        sync: _MegaPipelineSync,
        mScatter: cute.Tensor,
        scatter_ptr_smem_ptr: cute.Pointer,
        num_k_tiles: cutlass.Int32,
        accum_cnt_tile: cutlass.Int32,
        tile_m_idx: cutlass.Int32,
        tile_n_idx: cutlass.Int32,
        cm_start: cutlass.Int32,
        m_size: cutlass.Int32,
        n_size: cutlass.Int32,
        a_global_scale_inv_smem_ptr=None,
        a_global_scale_inv_ptr: cutlass.Int64 = 0,
        b_scale: cutlass.Float32 = 1.0,
        use_global_scale_inv: cutlass.Constexpr[bool] = False,
    ) -> cutlass.Boolean:
        tmem_buf, tmem_phase = _get_bufidx_phase(
            accum_cnt_tile,
            self.NUM_TMEM_BUFFERS,
        )
        cute.arch.mbarrier_wait(sync.tmem_full_mbar + tmem_buf, tmem_phase)
        if cutlass.const_expr(use_global_scale_inv):
            _stage_a_global_scale_inv(
                tidx,
                a_global_scale_inv_ptr,
                a_global_scale_inv_smem_ptr,
                cm_start,
                n_size if self.SWAP_AB else m_size,
                tile_m_idx,
                tile_n_idx,
                sync.cluster_cta_rank,
                BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                EPILOG_WARP_IDS=self.EPILOG_WARP_IDS,
                SWAP_AB=self.SWAP_AB,
                a_global_scale_inv_smem_size=self._a_global_scale_inv_smem_size,
                cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            )
            a_scales_smem = cute.make_tensor(
                a_global_scale_inv_smem_ptr,
                cute.make_layout((self._a_global_scale_inv_smem_size,), stride=(1,)),
            )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        NUM_MMA_ATOMS_M = cute.size(pipeline.tCtAcc_base.shape, mode=[1])
        NUM_MMA_ATOMS_N = cute.size(pipeline.tCtAcc_base.shape, mode=[2])
        for mma_m_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_M):
            for mma_n_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_N):
                tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                    self._epilog_tmem_copy_and_partition(
                        tidx=tidx,
                        tAcc=pipeline.tCtAcc_base,
                        gC_mnl=pipeline.c_tensor,
                        epi_tile=pipeline.epi_tile,
                        cta_tile_shape_mnk=pipeline.cta_tile_shape_mnk,
                        c_layout=pipeline.c_layout,
                        c_dtype=pipeline.c_dtype,
                        use_2cta_instrs=self.NUM_CTAS == 2,
                        mma_m_idx=mma_m_idx,
                        mma_n_idx=mma_n_idx,
                    )
                )
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, pipeline.sC.element_type)
                tiled_copy_r2s, tRS_rC, tRS_sC = self._epilog_smem_copy_and_partition(
                    tiled_copy_t2r=tiled_copy_t2r,
                    tTR_rC=tTR_rC,
                    tidx=tidx,
                    sC=pipeline.sC,
                    c_layout=pipeline.c_layout,
                    c_dtype=pipeline.c_dtype,
                )
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, tmem_buf)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                if cutlass.const_expr(use_global_scale_inv):
                    atom_m: cutlass.Constexpr[int] = (
                        pipeline.cta_tile_shape_mnk[0] // NUM_MMA_ATOMS_M
                    )
                    atom_n: cutlass.Constexpr[int] = (
                        pipeline.cta_tile_shape_mnk[1] // NUM_MMA_ATOMS_N
                    )
                    c_acc = cute.make_identity_tensor((atom_m, atom_n))
                    c_acc_epi = cute.flat_divide(c_acc, pipeline.epi_tile)
                    tTR_cAcc = tiled_copy_t2r.get_slice(tidx).partition_D(c_acc_epi)
                    tTR_cAcc = cute.group_modes(tTR_cAcc, 3, cute.rank(tTR_cAcc))
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                cols_per_subtile: cutlass.Constexpr[int] = cute.size(
                    pipeline.epi_tile[1]
                )
                iter_acc_early_release: cutlass.Constexpr[int] = (
                    self._num_sf_tmem_cols + cols_per_subtile - 1
                ) // cols_per_subtile - 1
                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.OVERLAPPING_ACCUM):
                        if tmem_buf == 0:
                            real_subtile_idx = subtile_cnt - 1 - subtile_idx

                    self._combine_epilog_load_scatter_ptrs(
                        tidx=tidx,
                        sC=pipeline.sC,
                        mScatter=mScatter,
                        scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
                        tile_m_idx=tile_m_idx,
                        tile_n_idx=tile_n_idx,
                        mma_m_idx=mma_m_idx,
                        mma_n_idx=mma_n_idx,
                        subtile_idx=real_subtile_idx,
                        NUM_MMA_ATOMS_M=NUM_MMA_ATOMS_M,
                        NUM_MMA_ATOMS_N=NUM_MMA_ATOMS_N,
                        cm_start=cm_start,
                        m_size=m_size,
                        n_size=n_size,
                        cluster_cta_rank=sync.cluster_cta_rank,
                    )

                    if num_k_tiles != 0:
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, real_subtile_idx)],
                            tTR_rAcc,
                        )

                    if cutlass.const_expr(self.OVERLAPPING_ACCUM):
                        release_overlap_atom = cutlass.Boolean(True)
                        if cutlass.const_expr(NUM_MMA_ATOMS_N > 1):
                            if cutlass.const_expr(mma_n_idx == 0):
                                release_overlap_atom = tmem_buf != 0
                            elif cutlass.const_expr(mma_n_idx == NUM_MMA_ATOMS_N - 1):
                                release_overlap_atom = tmem_buf == 0
                            else:
                                release_overlap_atom = cutlass.Boolean(False)
                        if (
                            subtile_idx == iter_acc_early_release
                        ) and release_overlap_atom:
                            cute.arch.fence_view_async_tmem_load()
                            cute.arch.barrier(
                                barrier_id=_BAR_EPILOG_SYNC,
                                number_of_threads=self.EPILOG_WG_THREADS,
                            )
                            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
                            if warp_idx == self.EPILOG_WARP_IDS[0]:
                                with cute.arch.elect_one():
                                    if cutlass.const_expr(self.NUM_CTAS == 2):
                                        cute.arch.mbarrier_arrive(
                                            sync.cross_seam_mbar,
                                            peer_cta_rank_in_cluster=0,
                                        )
                                    else:
                                        cute.arch.mbarrier_arrive(sync.cross_seam_mbar)

                    if cutlass.const_expr(use_global_scale_inv):
                        tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
                        if num_k_tiles == 0:
                            tRS_rAcc.store(cute.zeros_like(tRS_rAcc.load()))
                        tRS_cAcc = tiled_copy_r2s.retile(
                            tTR_cAcc[(None, None, None, real_subtile_idx)]
                        )
                        _apply_epilog_global_scale_inv(
                            tRS_rAcc,
                            tRS_cAcc,
                            a_scales_smem,
                            b_scale,
                            mma_m_idx,
                            mma_n_idx,
                            atom_m,
                            atom_n,
                            SWAP_AB=self.SWAP_AB,
                        )
                        acc_vec = tRS_rAcc.load()
                    elif cutlass.const_expr(self.MUTABLE_COMBINE_ACC_FRAGMENT):
                        tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
                        if num_k_tiles == 0:
                            tRS_rAcc.store(cute.zeros_like(tRS_rAcc.load()))
                        acc_vec = tRS_rAcc.load()
                    else:
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        if num_k_tiles == 0:
                            acc_vec = cute.zeros_like(acc_vec)
                    tRS_rC.store(acc_vec.to(pipeline.sC.element_type))
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, 0)],
                    )
                    cute.arch.fence_proxy(kind="async.shared", space="cta")
                    if cutlass.const_expr(
                        mma_m_idx == NUM_MMA_ATOMS_M - 1
                        and mma_n_idx == NUM_MMA_ATOMS_N - 1
                        and subtile_idx == subtile_cnt - 1
                    ):
                        cute.arch.fence_view_async_tmem_load()
                    cute.arch.barrier(
                        barrier_id=_BAR_EPILOG_SYNC,
                        number_of_threads=self.EPILOG_WG_THREADS,
                    )
                    if cutlass.const_expr(
                        mma_m_idx == NUM_MMA_ATOMS_M - 1
                        and mma_n_idx == NUM_MMA_ATOMS_N - 1
                        and subtile_idx == subtile_cnt - 1
                    ):
                        if warp_idx == self.EPILOG_WARP_IDS[0]:
                            with cute.arch.elect_one():
                                if cutlass.const_expr(self.NUM_CTAS == 2):
                                    cute.arch.mbarrier_arrive(
                                        sync.tmem_empty_mbar + tmem_buf,
                                        peer_cta_rank_in_cluster=0,
                                    )
                                else:
                                    cute.arch.mbarrier_arrive(
                                        sync.tmem_empty_mbar + tmem_buf
                                    )
                    self._combine_epilog_scatter_smem_tile(
                        tidx=tidx,
                        sC=pipeline.sC,
                        scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
                        tile_m_idx=tile_m_idx,
                        tile_n_idx=tile_n_idx,
                        mma_m_idx=mma_m_idx,
                        mma_n_idx=mma_n_idx,
                        subtile_idx=real_subtile_idx,
                        NUM_MMA_ATOMS_M=NUM_MMA_ATOMS_M,
                        NUM_MMA_ATOMS_N=NUM_MMA_ATOMS_N,
                        cm_start=cm_start,
                        m_size=m_size,
                        n_size=n_size,
                        elem_size_bytes_c=pipeline.c_dtype.width // 8,
                        cluster_cta_rank=sync.cluster_cta_rank,
                    )
                    cute.arch.barrier(
                        barrier_id=_BAR_EPILOG_SYNC,
                        number_of_threads=self.EPILOG_WG_THREADS,
                    )

        return warp_idx == self.EPILOG_WARP_IDS[0]

    @cute.jit
    def _mega_epilog_consumer_body(  # noqa: C901
        self,
        tidx: cutlass.Int32,
        dgrad: _MegaEpilogPipeline,
        wgrad: _MegaEpilogPipeline,
        problem: _MegaProblem,
        sync: _MegaPipelineSync,
        activation_quant: _MegaDispatchQuantSync,
        tensormap_manager,
        tensormaps: cute.Tensor,
        mScatter: cute.Tensor,
        scatter_ptr_smem_ptr: cute.Pointer,
        combine: cutlass.Constexpr[bool],
        use_device_tensormaps: cutlass.Constexpr[bool],
        work_info_smem_ptr: cute.Pointer,
    ) -> None:
        M, N, K = problem.mnk
        dgrad_copy = self._make_epilog_copy_partitions(tidx, dgrad)
        wgrad_copy = self._make_epilog_copy_partitions(tidx, wgrad)
        accum_cnt_tile = cutlass.Int32(0)

        if cutlass.const_expr(self.STATIC_SCHEDULER):
            scheduler = MegaStaticScheduler.create_consumer(self.NUM_CTAS)
        else:
            scheduler = MegaDynamicScheduler.create_consumer(
                sync.tile_consumer_mbar,
                sync.tile_id_smem_ptr,
                self.NUM_CTAS,
                self.NUM_TILE_BUFFERS,
            )
        tile_idx = scheduler.initial_work_tile_info()

        gemm_c_base_pair = (
            self.MEGA_DGRAD_TENSORMAP_BASE + 4,
            self.MEGA_WGRAD_TENSORMAP_BASE + 4,
        )
        if cutlass.const_expr(self.STATIC_SCHEDULER):
            gemm_mnk_pair, gemm_problem_type_pair = _mega_problem_specs(
                M,
                N,
                K,
                MEGA_FORWARD_MODE=self.MEGA_FORWARD_MODE,
                mode=self.mode,
                MEGA_DGRAD_PROBLEM_TYPE=self.MEGA_DGRAD_PROBLEM_TYPE,
                MEGA_WGRAD_PROBLEM_TYPE=self.MEGA_WGRAD_PROBLEM_TYPE,
            )
            group_tile_size: cutlass.Constexpr[int] = (
                self.BLOCK_SIZE_N if self.SWAP_AB else self.BLOCK_SIZE_M
            )
            visitor = MegaProblemVisitor.create(
                problem.split_sizes,
                gemm_mnk_pair,
                problem.groups,
                gemm_problem_type_pair,
                self.BLOCK_SIZE_M,
                self.BLOCK_SIZE_N,
                self.BLOCK_SIZE_K,
                self.force_n_major,
                self.num_n_clusters,
                problem.local_rank,
                self.world_size,
                self.SWAP_AB,
                group_tile_size,
            )
            work = visitor.get_work(tile_idx)
        else:
            work = load_mega_work_info(
                tile_idx,
                work_info_smem_ptr,
                cutlass.Int32(0),
            )

        while work.is_valid_tile:
            g = work.group_idx
            cm_start = work.split_prefix
            act_tile_start = work.group_tile_prefix
            for w in cutlass.range_constexpr(2):
                pipeline = (dgrad, wgrad)[w]
                if cutlass.const_expr(self.SWAP_AB):
                    c_m_off_tiles = cutlass.Int32(0)
                    c_n_off_tiles = work.n_tile_prefix
                else:
                    c_m_off_tiles = work.m_tile_prefix
                    c_n_off_tiles = cutlass.Int32(0)
                m_size = work.m_size
                n_size = work.n_size
                num_k_tiles = work.num_k_tiles
                c_base = gemm_c_base_pair[w]
                if work.problem_idx == cutlass.Int32(w):
                    tTR_rAcc, tiled_copy_r2s, tRS_rC, tRS_sC = (
                        dgrad_copy,
                        wgrad_copy,
                    )[w]
                    c_desc_ptr = None
                    if cutlass.const_expr(
                        (self.mode == self.MEGA_FORWARD_MODE and w == 0)
                        or (
                            self.mode != self.MEGA_FORWARD_MODE
                            and (not combine or w != 0)
                        )
                    ):
                        _epilog_wait_pending_tma_store()
                        if cutlass.const_expr(use_device_tensormaps):
                            tensormap_c_g_ptr = tensormap_manager.get_tensormap_ptr(
                                tensormaps[(g, c_base, None)].iterator
                            )
                            c_desc_ptr = tensormap_manager.get_tensormap_ptr(
                                tensormap_c_g_ptr,
                                cute.AddressSpace.generic,
                            )
                            tensormap_manager.fence_tensormap_update(tensormap_c_g_ptr)
                    while (
                        work.is_valid_tile
                        and work.group_idx == g
                        and work.problem_idx == cutlass.Int32(w)
                    ):
                        tile_buf, _ = _get_bufidx_phase(
                            accum_cnt_tile,
                            self.NUM_TILE_BUFFERS,
                        )
                        tile_m_idx = work.tile_m_idx
                        tile_n_idx = work.tile_n_idx
                        if cutlass.const_expr(
                            combine
                            and (
                                (self.mode == self.MEGA_FORWARD_MODE and w == 1)
                                or (self.mode != self.MEGA_FORWARD_MODE and w == 0)
                            )
                        ):
                            is_leader = self._mega_combine_epilog_consumer_tile(
                                tidx=tidx,
                                pipeline=pipeline,
                                sync=sync,
                                mScatter=mScatter,
                                scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
                                num_k_tiles=num_k_tiles,
                                accum_cnt_tile=accum_cnt_tile,
                                tile_m_idx=tile_m_idx,
                                tile_n_idx=tile_n_idx,
                                cm_start=cm_start,
                                m_size=m_size,
                                n_size=n_size,
                            )
                        else:
                            cta_m_start = (
                                tile_m_idx * self.BLOCK_SIZE_M
                                + sync.cluster_cta_rank
                                * (self.BLOCK_SIZE_M // self.NUM_CTAS)
                            )
                            if cutlass.const_expr(self.SWAP_AB):
                                skip_tma_store = cutlass.Boolean(False)
                            else:
                                skip_tma_store = cta_m_start >= m_size
                            if cutlass.const_expr(use_device_tensormaps):
                                c_tile_indices = (tile_m_idx, tile_n_idx)
                            else:
                                c_tile_indices = (
                                    c_m_off_tiles + tile_m_idx,
                                    c_n_off_tiles + tile_n_idx,
                                )
                            is_leader = self._blockscaled_epilog_consumer_tile(
                                tidx,
                                pipeline,
                                sync,
                                tTR_rAcc,
                                tiled_copy_r2s,
                                tRS_rC,
                                tRS_sC,
                                num_k_tiles,
                                accum_cnt_tile,
                                c_tile_indices,
                                skip_tma_store,
                                (c_desc_ptr,),
                                use_device_tensormaps,
                                cutlass.Float32(1.0),
                            )
                            if cutlass.const_expr(
                                self.mode == self.MEGA_FORWARD_MODE and w == 0
                            ):
                                forward_act_tile_idx = (
                                    tile_n_idx if self.SWAP_AB else tile_m_idx
                                )
                                cute.arch.fence_acq_rel_gpu()
                                if is_leader:
                                    with cute.arch.elect_one():
                                        ptr = cute.recast_ptr(
                                            activation_quant.done_counter.iterator
                                            + activation_quant.done_counter_offsets[1]
                                            + act_tile_start
                                            + forward_act_tile_idx,
                                            dtype=cutlass.Uint32,
                                        )
                                        cute.arch.atomic_add(
                                            ptr,
                                            cutlass.Uint32(1),
                                            sem="release",
                                            scope="gpu",
                                        )
                        scheduler.consumer_release_tile(
                            sync.tile_producer_mbar,
                            tile_buf,
                            sync.cluster_cta_rank,
                            is_leader,
                        )
                        accum_cnt_tile += cutlass.Int32(1)
                        tile_idx = scheduler.advance_consumer(accum_cnt_tile)
                        if cutlass.const_expr(self.STATIC_SCHEDULER):
                            work = visitor.get_work(tile_idx)
                        else:
                            work = load_mega_work_info(
                                tile_idx,
                                work_info_smem_ptr,
                                accum_cnt_tile % self.NUM_TILE_BUFFERS,
                            )

        _epilog_wait_pending_tma_store()

    @cute.jit
    def _remap_forward_row_block(
        self,
        local_row_block,
        row_blocks_per_act_tile,
        group_act_tiles,
        group_rows,
        local_rank,
    ):
        act_tile_offset = self._forward_act_tile_offset(
            group_act_tiles,
            group_rows,
            local_rank,
        )
        group_row_blocks = group_rows // cutlass.Int32(self.sf_vec_size)
        return (
            local_row_block + act_tile_offset * cutlass.Int32(row_blocks_per_act_tile)
        ) % group_row_blocks

    @cute.jit
    def _dispatch_ring_store_positions(
        self,
        activation_ring: cutlass.Constexpr[bool],
        local_row_block: cutlass.Int32,
        m_size: cutlass.Int32,
        ring_chunk_prefix: cutlass.Int32,
        local_rank: cutlass.Int32,
        row_start: cutlass.Int32,
        row_scale_start: cutlass.Int32,
        mRingFc13Done,
        ring_fc13_done_offset: cutlass.Int32,
        ring_fc13_target: cutlass.Int32,
        ring_cached_chunk: cutlass.Int32,
        ring_cached_row_base: cutlass.Int32,
        ring_cached_scale_base: cutlass.Int32,
    ):
        """Map a dispatch work tile's stores to its activation-ring slot.

        Returns ``(store_row_start, store_row_scale_start, cached_chunk,
        cached_row_base, cached_scale_base)``; ring off passes the logical
        positions through. The slot bases and the FC13-consumption wait for
        the slot's previous occupant (``sched - W``) only change at chunk
        granularity, so they refresh on chunk transitions and steady tiles
        pay only the cached-base arithmetic.
        """
        store_row_start = row_start
        store_row_scale_start = row_scale_start
        if cutlass.const_expr(not activation_ring):
            return (
                store_row_start,
                store_row_scale_start,
                ring_cached_chunk,
                ring_cached_row_base,
                ring_cached_scale_base,
            )
        ring_phys_row = local_row_block * cutlass.Int32(self.sf_vec_size)
        (
            store_row_start,
            store_row_scale_start,
            ring_refreshed,
            ring_cached_chunk,
            ring_sched,
            ring_cached_row_base,
            ring_cached_scale_base,
        ) = _ring_cached_slot(
            ring_phys_row,
            m_size,
            ring_chunk_prefix,
            local_rank,
            ring_cached_chunk,
            # This caller does not carry the schedule chunk across calls: it
            # is consumed only on refresh, for the consumption wait below.
            cutlass.Int32(0),
            ring_cached_row_base,
            ring_cached_scale_base,
            PIPELINE_CHUNK_ROWS=self.PIPELINE_CHUNK_ROWS,
            ACTIVATION_RING_CHUNKS=self.ACTIVATION_RING_CHUNKS,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            world_size=self.world_size,
            SWAP_AB=self.SWAP_AB,
        )
        if ring_refreshed:
            ring_prev_sched = ring_sched - cutlass.Int32(self.ACTIVATION_RING_CHUNKS)
            if ring_prev_sched >= cutlass.Int32(0):
                _wait_counter_at_least(
                    cute.recast_ptr(
                        mRingFc13Done.iterator
                        + ring_fc13_done_offset
                        + ring_prev_sched,
                        dtype=cutlass.Uint32,
                    ),
                    cutlass.Uint32(ring_fc13_target),
                )
        return (
            store_row_start,
            store_row_scale_start,
            ring_cached_chunk,
            ring_cached_row_base,
            ring_cached_scale_base,
        )

    @cute.jit
    def _mega_dispatch_quant_producer_body(
        self,
        mWorkCounter: cute.Tensor,
        mDoneCounter: cute.Tensor,
        tile_smem_ptr: cute.Pointer,
        dispatch_ptr_smem_ptr,
        mGatherPtrs: cute.Tensor,
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
        row_done_counter_offset: cutlass.Int32,
        col_done_counter_offset: cutlass.Int32,
        row_quant: cutlass.Constexpr[bool] = True,
        col_quant: cutlass.Constexpr[bool] = True,
        mRingFc13Done: cute.Tensor | None = None,
        ring_fc13_done_offset: cutlass.Int32 = 0,
        ring_fc13_target: cutlass.Int32 = 0,
    ) -> None:
        del M
        activation_ring: cutlass.Constexpr[bool] = (
            self.ACTIVATION_RING_CHUNKS > 0 and mRingFc13Done is not None
        )
        if cutlass.const_expr(activation_ring and dispatch_ptr_smem_ptr is None):
            raise ValueError(
                "the activation ring writes quantized rows at ring positions "
                "and needs the SMEM-staged gather pointers for the sources"
            )
        tidx, _, _ = cute.arch.thread_idx()
        quant_lane = tidx - cutlass.Int32(
            self.DISPATCH_QUANT_FIRST_WARP * self.THREADS_PER_WARP
        )
        quant_group = quant_lane // cutlass.Int32(self.DISPATCH_QUANT_GROUP_THREADS)
        quant_group_lane = quant_lane - quant_group * cutlass.Int32(
            self.DISPATCH_QUANT_GROUP_THREADS
        )
        warp_in_quant_group = quant_group_lane // cutlass.Int32(self.THREADS_PER_WARP)
        warp_lane = quant_group_lane % cutlass.Int32(self.THREADS_PER_WARP)
        col_scale_cols = K // cutlass.Int32(self.sf_vec_size)
        col_work_tiles = ceil_div(
            col_scale_cols,
            cutlass.Int32(self.DISPATCH_QUANT_SCALE_COLS_PER_TILE),
        )
        row_blocks_total = _activation_buffer_rows(
            split_sizes,
            G,
        ) // cutlass.Int32(self.sf_vec_size)
        ACT_BLOCK_SIZE: cutlass.Constexpr[int] = (
            self.BLOCK_SIZE_N if self.SWAP_AB else self.BLOCK_SIZE_M
        )
        if cutlass.const_expr(ACT_BLOCK_SIZE % self.sf_vec_size != 0):
            raise ValueError(
                "mega dispatch quant act tile must be a multiple of sf_vec_size"
            )
        row_blocks_per_act_tile: cutlass.Constexpr[int] = (
            ACT_BLOCK_SIZE // self.sf_vec_size
        )

        tile_idx_base = _dispatch_quant_fetch_group_tile_id(
            mWorkCounter,
            tile_smem_ptr,
            quant_group,
            quant_group_lane,
            DISPATCH_QUANT_SYNC_BAR=self.DISPATCH_QUANT_SYNC_BAR,
            DISPATCH_QUANT_GROUP_THREADS=self.DISPATCH_QUANT_GROUP_THREADS,
        )
        tile_start = cutlass.Int32(0)
        act_tile_start = cutlass.Int32(0)
        start_m = cutlass.Int32(0)
        scale_start_m = cutlass.Int32(0)
        ring_chunk_prefix = cutlass.Int32(0)
        ring_cached_chunk = cutlass.Int32(-1)
        ring_cached_row_base = cutlass.Int32(0)
        ring_cached_scale_base = cutlass.Int32(0)
        for g in cutlass.range(G, unroll=1):
            m_size = cutlass.Int32(split_sizes[g])
            group_row_blocks = m_size // cutlass.Int32(self.sf_vec_size)
            group_tiles = group_row_blocks * col_work_tiles
            group_act_tiles = ceil_div(m_size, cutlass.Int32(ACT_BLOCK_SIZE))
            tile_end = tile_start + group_tiles
            ring_cached_chunk = cutlass.Int32(-1)

            while (tile_idx_base >= tile_start) and (tile_idx_base < tile_end):
                local_tile = tile_idx_base - tile_start
                local_row_block = local_tile // col_work_tiles
                col_work_tile = local_tile - local_row_block * col_work_tiles
                local_row_block = self._remap_forward_row_block(
                    local_row_block,
                    row_blocks_per_act_tile,
                    group_act_tiles,
                    m_size,
                    local_rank,
                )
                col_first_block = col_work_tile * cutlass.Int32(
                    self.DISPATCH_QUANT_SCALE_COLS_PER_TILE
                )
                row_block_start = scale_start_m
                if cutlass.const_expr(self.BLOCK_SIZE_N == 32):
                    row_block_start = start_m
                row_block = (
                    row_block_start // cutlass.Int32(self.sf_vec_size) + local_row_block
                )
                row_start = start_m + local_row_block * cutlass.Int32(self.sf_vec_size)
                row_scale_start = blockscaled_scale_row_start(
                    scale_start_m,
                    local_row_block * cutlass.Int32(self.sf_vec_size),
                    self.SWAP_AB,
                    self.BLOCK_SIZE_N,
                )
                act_tile_slot = act_tile_start + local_row_block // cutlass.Int32(
                    row_blocks_per_act_tile
                )
                # With the ring on, quantized rows land at the chunk's
                # schedule-slot position; `row_start` stays logical for the
                # gather pointer staging.
                (
                    store_row_start,
                    store_row_scale_start,
                    ring_cached_chunk,
                    ring_cached_row_base,
                    ring_cached_scale_base,
                ) = self._dispatch_ring_store_positions(
                    activation_ring,
                    local_row_block,
                    m_size,
                    ring_chunk_prefix,
                    local_rank,
                    row_start,
                    row_scale_start,
                    mRingFc13Done,
                    ring_fc13_done_offset,
                    ring_fc13_target,
                    ring_cached_chunk,
                    ring_cached_row_base,
                    ring_cached_scale_base,
                )

                if col_first_block < col_scale_cols:
                    gather_ptr_smem_ptr = None
                    if cutlass.const_expr(dispatch_ptr_smem_ptr is not None):
                        gather_ptr_smem_ptr = (
                            dispatch_ptr_smem_ptr
                            + quant_group * cutlass.Int32(self.sf_vec_size)
                        )
                        if quant_group_lane < cutlass.Int32(self.sf_vec_size):
                            cute.arch.store(
                                gather_ptr_smem_ptr + quant_group_lane,
                                mGatherPtrs[row_start + quant_group_lane],
                                ss="cta",
                            )
                        cute.arch.barrier(
                            barrier_id=self.DISPATCH_QUANT_SYNC_BAR + quant_group,
                            number_of_threads=self.DISPATCH_QUANT_GROUP_THREADS,
                        )

                    for microtile in cutlass.range_constexpr(
                        self.DISPATCH_QUANT_MICROTILES_PER_WARP
                    ):
                        col_block = col_first_block + (
                            warp_in_quant_group
                            * cutlass.Int32(self.DISPATCH_QUANT_MICROTILES_PER_WARP)
                            + cutlass.Int32(microtile)
                        ) * cutlass.Int32(self.DISPATCH_QUANT_SCALE_COLS_PER_WARP)
                        if col_block < col_scale_cols:
                            _dispatch_quantize_tile(
                                params_from_kernel(DispatchQuantParams, self),
                                warp_lane,
                                mGatherPtrs,
                                gather_ptr_smem_ptr,
                                mRowQWords,
                                mRowScale,
                                mColQWords,
                                mColScale,
                                store_row_start,
                                store_row_scale_start,
                                row_block,
                                col_block,
                                row_blocks_total,
                                K,
                                row_quant,
                                col_quant,
                            )

                _mega_dispatch_quant_signal_group_tile_done(
                    mDoneCounter,
                    row_done_counter_offset,
                    col_done_counter_offset,
                    act_tile_slot,
                    g * col_work_tiles + col_work_tile,
                    quant_group,
                    quant_group_lane,
                    DISPATCH_QUANT_GROUP_THREADS=self.DISPATCH_QUANT_GROUP_THREADS,
                    DISPATCH_QUANT_SYNC_BAR=self.DISPATCH_QUANT_SYNC_BAR,
                    MEGA_FORWARD_MODE=self.MEGA_FORWARD_MODE,
                    mode=self.mode,
                )

                tile_idx_base = _dispatch_quant_fetch_group_tile_id(
                    mWorkCounter,
                    tile_smem_ptr,
                    quant_group,
                    quant_group_lane,
                    DISPATCH_QUANT_SYNC_BAR=self.DISPATCH_QUANT_SYNC_BAR,
                    DISPATCH_QUANT_GROUP_THREADS=self.DISPATCH_QUANT_GROUP_THREADS,
                )

            start_m += m_size
            scale_start_m = advance_blockscaled_scale_start(
                scale_start_m, m_size, self.SWAP_AB, self.BLOCK_SIZE_N
            )
            act_tile_start += group_act_tiles
            tile_start = tile_end
            if cutlass.const_expr(activation_ring):
                ring_chunk_prefix += ceil_div(
                    m_size, cutlass.Int32(self.PIPELINE_CHUNK_ROWS)
                )

    @cute.jit
    def _mega_dispatch_blockscaled_copy_producer_body(
        self,
        mWorkCounter: cute.Tensor,
        mDoneCounter: cute.Tensor,
        tile_smem_ptr: cute.Pointer,
        dispatch_ptr_smem_ptr: cute.Pointer,
        mGatherPtrs: cute.Tensor,
        mRowQWords: cute.Tensor,
        mRowScale: cute.Tensor,
        mRowGlobalScaleInv: cute.Tensor,
        split_sizes: cute.Tensor,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        row_done_counter_offset: cutlass.Int32,
        col_done_counter_offset: cutlass.Int32,
        use_global_scale_inv: cutlass.Constexpr[bool] = False,
        mRingFc13Done: cute.Tensor | None = None,
        ring_fc13_done_offset: cutlass.Int32 = 0,
        ring_fc13_target: cutlass.Int32 = 0,
    ) -> None:
        activation_ring: cutlass.Constexpr[bool] = (
            self.ACTIVATION_RING_CHUNKS > 0 and mRingFc13Done is not None
        )
        tidx, _, _ = cute.arch.thread_idx()
        quant_lane = tidx - cutlass.Int32(
            self.DISPATCH_QUANT_FIRST_WARP * self.THREADS_PER_WARP
        )
        quant_group = quant_lane // cutlass.Int32(self.DISPATCH_QUANT_GROUP_THREADS)
        quant_group_lane = quant_lane - quant_group * cutlass.Int32(
            self.DISPATCH_QUANT_GROUP_THREADS
        )
        scale_cols = K // cutlass.Int32(self.sf_vec_size)
        col_work_tiles = ceil_div(
            scale_cols,
            cutlass.Int32(self.DISPATCH_QUANT_SCALE_COLS_PER_TILE),
        )
        copy_col_work_tiles = ceil_div(
            col_work_tiles,
            cutlass.Int32(self.FORWARD_COPY_COL_TILES_PER_WORK),
        )
        ACT_BLOCK_SIZE: cutlass.Constexpr[int] = (
            self.BLOCK_SIZE_N if self.SWAP_AB else self.BLOCK_SIZE_M
        )
        row_blocks_per_act_tile: cutlass.Constexpr[int] = (
            ACT_BLOCK_SIZE // self.sf_vec_size
        )
        scale_cols_per_work_tile: cutlass.Constexpr[int] = (
            self.DISPATCH_QUANT_SCALE_COLS_PER_TILE
            * self.FORWARD_COPY_COL_TILES_PER_WORK
        )

        tile_idx = _dispatch_quant_fetch_group_tile_id(
            mWorkCounter,
            tile_smem_ptr,
            quant_group,
            quant_group_lane,
            DISPATCH_QUANT_SYNC_BAR=self.DISPATCH_QUANT_SYNC_BAR,
            DISPATCH_QUANT_GROUP_THREADS=self.DISPATCH_QUANT_GROUP_THREADS,
        )
        tile_start = cutlass.Int32(0)
        act_tile_start = cutlass.Int32(0)
        start_m = cutlass.Int32(0)
        scale_start_m = cutlass.Int32(0)
        ring_chunk_prefix = cutlass.Int32(0)
        ring_cached_chunk = cutlass.Int32(-1)
        ring_cached_row_base = cutlass.Int32(0)
        ring_cached_scale_base = cutlass.Int32(0)
        for g in cutlass.range(G, unroll=1):
            m_size = cutlass.Int32(split_sizes[g])
            group_row_blocks = m_size // cutlass.Int32(self.sf_vec_size)
            group_tiles = group_row_blocks * copy_col_work_tiles
            group_act_tiles = ceil_div(m_size, cutlass.Int32(ACT_BLOCK_SIZE))
            tile_end = tile_start + group_tiles
            ring_cached_chunk = cutlass.Int32(-1)

            while (tile_idx >= tile_start) and (tile_idx < tile_end):
                local_tile = tile_idx - tile_start
                local_row_block = local_tile // copy_col_work_tiles
                copy_col_work_tile = local_tile - local_row_block * copy_col_work_tiles
                col_work_tile = copy_col_work_tile * cutlass.Int32(
                    self.FORWARD_COPY_COL_TILES_PER_WORK
                )
                local_row_block = self._remap_forward_row_block(
                    local_row_block,
                    row_blocks_per_act_tile,
                    group_act_tiles,
                    m_size,
                    local_rank,
                )
                row_start = start_m + local_row_block * cutlass.Int32(self.sf_vec_size)
                row_scale_start = blockscaled_scale_row_start(
                    scale_start_m,
                    local_row_block * cutlass.Int32(self.sf_vec_size),
                    self.SWAP_AB,
                    self.BLOCK_SIZE_N,
                )
                # Same ring placement as the quantizing producer; `row_start`
                # stays logical for the gather pointer staging.
                (
                    store_row_start,
                    store_row_scale_start,
                    ring_cached_chunk,
                    ring_cached_row_base,
                    ring_cached_scale_base,
                ) = self._dispatch_ring_store_positions(
                    activation_ring,
                    local_row_block,
                    m_size,
                    ring_chunk_prefix,
                    local_rank,
                    row_start,
                    row_scale_start,
                    mRingFc13Done,
                    ring_fc13_done_offset,
                    ring_fc13_target,
                    ring_cached_chunk,
                    ring_cached_row_base,
                    ring_cached_scale_base,
                )
                scale_col_start = col_work_tile * cutlass.Int32(
                    self.DISPATCH_QUANT_SCALE_COLS_PER_TILE
                )
                gather_ptrs = dispatch_ptr_smem_ptr + quant_group * cutlass.Int32(
                    self.sf_vec_size
                )
                if quant_group_lane < cutlass.Int32(self.sf_vec_size):
                    cute.arch.store(
                        gather_ptrs + quant_group_lane,
                        mGatherPtrs[row_start + quant_group_lane],
                        ss="cta",
                    )
                cute.arch.barrier(
                    barrier_id=self.DISPATCH_QUANT_SYNC_BAR + quant_group,
                    number_of_threads=self.DISPATCH_QUANT_GROUP_THREADS,
                )

                qdata_prefetch: cutlass.Constexpr[int] = (
                    4
                    if self.format.a_dtype == cutlass.Float8E4M3FN
                    and self.format.b_dtype == cutlass.Float4E2M1FN
                    else 8
                    if self.sf_vec_size >= 32
                    else 4
                )
                _dispatch_copy_blockscaled_work_tile(
                    gather_ptrs,
                    mRowQWords,
                    mRowScale,
                    mRowGlobalScaleInv,
                    store_row_start,
                    store_row_scale_start,
                    scale_col_start,
                    K,
                    quant_group_lane,
                    self.DISPATCH_QUANT_GROUP_THREADS,
                    scale_cols_per_work_tile,
                    qdata_prefetch,
                    use_global_scale_inv,
                    DISPATCH_QUANT_QDATA_ELEMS_PER_WORD=self.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
                    VECTOR_DISPATCH_SCALE_COPY=self.VECTOR_DISPATCH_SCALE_COPY,
                    sf_vec_size=self.sf_vec_size,
                    DISPATCH_QUANT_SCALE_COLS_PER_TILE=self.DISPATCH_QUANT_SCALE_COLS_PER_TILE,
                    gsi_row_start=(
                        row_start if cutlass.const_expr(activation_ring) else None
                    ),
                )

                _mega_dispatch_quant_signal_group_tile_done(
                    mDoneCounter,
                    row_done_counter_offset,
                    col_done_counter_offset,
                    act_tile_start
                    + local_row_block // cutlass.Int32(row_blocks_per_act_tile),
                    g * col_work_tiles + col_work_tile,
                    quant_group,
                    quant_group_lane,
                    DISPATCH_QUANT_GROUP_THREADS=self.DISPATCH_QUANT_GROUP_THREADS,
                    DISPATCH_QUANT_SYNC_BAR=self.DISPATCH_QUANT_SYNC_BAR,
                    MEGA_FORWARD_MODE=self.MEGA_FORWARD_MODE,
                    mode=self.mode,
                    col_tiles_per_work=self.FORWARD_COPY_COL_TILES_PER_WORK,
                    col_done_limit=(g + cutlass.Int32(1)) * col_work_tiles,
                )
                tile_idx = _dispatch_quant_fetch_group_tile_id(
                    mWorkCounter,
                    tile_smem_ptr,
                    quant_group,
                    quant_group_lane,
                    DISPATCH_QUANT_SYNC_BAR=self.DISPATCH_QUANT_SYNC_BAR,
                    DISPATCH_QUANT_GROUP_THREADS=self.DISPATCH_QUANT_GROUP_THREADS,
                )

            start_m += m_size
            scale_start_m = advance_blockscaled_scale_start(
                scale_start_m, m_size, self.SWAP_AB, self.BLOCK_SIZE_N
            )
            act_tile_start += group_act_tiles
            tile_start = tile_end
            if cutlass.const_expr(activation_ring):
                ring_chunk_prefix += ceil_div(
                    m_size, cutlass.Int32(self.PIPELINE_CHUNK_ROWS)
                )

    @cute.jit
    def _mega_forward_swiglu_quant_producer_body(
        self,
        mWorkCounter: cute.Tensor,
        mDoneCounter: cute.Tensor,
        h1_done_counter_offset: cutlass.Int32,
        mFwdX: cute.Tensor,
        mFwdY: cute.Tensor,
        mScatter: cute.Tensor,
        mRowQWords: cute.Tensor,
        mRowScale: cute.Tensor,
        mColQWords: cute.Tensor,
        mColScale: cute.Tensor,
        split_sizes: cute.Tensor,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
    ) -> None:
        del M
        tidx, _, _ = cute.arch.thread_idx()
        quant_lane = tidx - cutlass.Int32(
            self.COMBINE_SWIGLU_QUANT_FIRST_WARP * self.THREADS_PER_WARP
        )
        warp_lane = quant_lane % cutlass.Int32(self.THREADS_PER_WARP)
        col_scale_cols = K // cutlass.Int32(self.sf_vec_size)
        col_work_tiles = ceil_div(
            col_scale_cols,
            cutlass.Int32(self.COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE),
        )
        row_blocks_total = _activation_buffer_rows(
            split_sizes,
            G,
        ) // cutlass.Int32(self.sf_vec_size)
        ACT_BLOCK_SIZE: cutlass.Constexpr[int] = (
            self.BLOCK_SIZE_N if self.SWAP_AB else self.BLOCK_SIZE_M
        )
        row_blocks_per_act_tile: cutlass.Constexpr[int] = (
            ACT_BLOCK_SIZE // self.sf_vec_size
        )

        tile_idx_base = _combine_swiglu_quant_fetch_warp_tile_batch(
            mWorkCounter,
            warp_lane,
            COMBINE_SWIGLU_WORK_TILES_PER_FETCH=self.COMBINE_SWIGLU_WORK_TILES_PER_FETCH,
        )
        tile_idx = tile_idx_base
        tile_batch_end = tile_idx_base + cutlass.Int32(
            self.COMBINE_SWIGLU_WORK_TILES_PER_FETCH
        )
        tile_start = cutlass.Int32(0)
        act_tile_start = cutlass.Int32(0)
        start_m = cutlass.Int32(0)
        scale_start_m = cutlass.Int32(0)
        for g in cutlass.range(G, unroll=1):
            m_size = cutlass.Int32(split_sizes[g])
            group_row_blocks = m_size // cutlass.Int32(self.sf_vec_size)
            group_tiles = group_row_blocks * col_work_tiles
            group_act_tiles = ceil_div(m_size, cutlass.Int32(ACT_BLOCK_SIZE))
            tile_end = tile_start + group_tiles

            while (tile_idx >= tile_start) and (tile_idx < tile_end):
                local_tile = tile_idx - tile_start
                local_row_block = local_tile // col_work_tiles
                col_work_tile = local_tile - local_row_block * col_work_tiles
                local_row_block = self._remap_forward_row_block(
                    local_row_block,
                    row_blocks_per_act_tile,
                    group_act_tiles,
                    m_size,
                    local_rank,
                )
                row_block_start = scale_start_m
                if cutlass.const_expr(self.BLOCK_SIZE_N == 32):
                    row_block_start = start_m
                row_block = (
                    row_block_start // cutlass.Int32(self.sf_vec_size) + local_row_block
                )
                row_start = start_m + local_row_block * cutlass.Int32(self.sf_vec_size)
                row_scale_start = blockscaled_scale_row_start(
                    scale_start_m,
                    local_row_block * cutlass.Int32(self.sf_vec_size),
                    self.SWAP_AB,
                    self.BLOCK_SIZE_N,
                )
                act_tile_slot = act_tile_start + local_row_block // cutlass.Int32(
                    row_blocks_per_act_tile
                )
                col_block = col_work_tile * cutlass.Int32(
                    self.COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE
                )
                _mega_forward_wait_h1_tile(
                    mDoneCounter,
                    h1_done_counter_offset,
                    act_tile_slot,
                    K,
                    BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                    BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                    NUM_CTAS=self.NUM_CTAS,
                    SWAP_AB=self.SWAP_AB,
                )
                if col_block < col_scale_cols:
                    _combine_swiglu_fwd_quantize_tile(
                        params_from_kernel(CombineSwigluQuantParams, self),
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
                        mScatter,
                    )

                tile_idx += cutlass.Int32(1)
                if tile_idx == tile_batch_end:
                    tile_idx_base = _combine_swiglu_quant_fetch_warp_tile_batch(
                        mWorkCounter,
                        warp_lane,
                        COMBINE_SWIGLU_WORK_TILES_PER_FETCH=self.COMBINE_SWIGLU_WORK_TILES_PER_FETCH,
                    )
                    tile_idx = tile_idx_base
                    tile_batch_end = tile_idx_base + cutlass.Int32(
                        self.COMBINE_SWIGLU_WORK_TILES_PER_FETCH
                    )

            start_m += m_size
            scale_start_m = advance_blockscaled_scale_start(
                scale_start_m, m_size, self.SWAP_AB, self.BLOCK_SIZE_N
            )
            act_tile_start += group_act_tiles
            tile_start = tile_end

    @cute.jit
    def _dispatch_quant_producer_body(
        self,
        mWorkCounter: cute.Tensor,
        mDoneCounter: cute.Tensor,
        tile_smem_ptr: cute.Pointer,
        mGatherPtrs: cute.Tensor,
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
        row_done_counter_offset = cutlass.Int32(0)
        col_done_counter_offset = M // cutlass.Int32(self.sf_vec_size)
        self._mega_dispatch_quant_producer_body(
            mWorkCounter=mWorkCounter,
            mDoneCounter=mDoneCounter,
            tile_smem_ptr=tile_smem_ptr,
            dispatch_ptr_smem_ptr=None,
            mGatherPtrs=mGatherPtrs,
            mRowQWords=mRowQWords,
            mRowScale=mRowScale,
            mColQWords=mColQWords,
            mColScale=mColScale,
            split_sizes=split_sizes,
            G=G,
            M=M,
            K=K,
            total_tiles=total_tiles,
            local_rank=local_rank,
            row_done_counter_offset=row_done_counter_offset,
            col_done_counter_offset=col_done_counter_offset,
        )

    def _make_shared_storage(
        self,
        dgrad_layout: _MegaGemmLayoutBundle,
        wgrad_layout: _MegaGemmLayoutBundle,
        combine: cutlass.Constexpr[bool],
        G: int,
        work_info_fields: int = MEGA_WORK_INFO_FIELDS,
        use_a_global_scale_inv: bool = False,
    ):
        NUM_SMEM = self.NUM_SMEM_BUFFERS
        NUM_TMEM = self.NUM_TMEM_BUFFERS
        NUM_TILE = self.NUM_TILE_BUFFERS
        NUM_CTAS = self.NUM_CTAS

        a_dtype = (
            cutlass.Uint8
            if self.USE_SM103_ULTRA
            or (
                dgrad_layout.a_dtype.width != dgrad_layout.b_dtype.width
                and dgrad_layout.a_dtype.width < 8
            )
            else dgrad_layout.a_dtype
        )
        b_dtype = (
            cutlass.Uint8
            if self.USE_SM103_ULTRA
            or (
                dgrad_layout.a_dtype.width != dgrad_layout.b_dtype.width
                and dgrad_layout.b_dtype.width < 8
            )
            else dgrad_layout.b_dtype
        )
        sf_dtype = self.sf_dtype

        a_smem_elems = max(
            cute.cosize(dgrad_layout.a_smem_layout_staged.outer),
            cute.cosize(wgrad_layout.a_smem_layout_staged.outer),
        )
        b_smem_elems = max(
            cute.cosize(dgrad_layout.b_smem_layout_staged.outer),
            cute.cosize(wgrad_layout.b_smem_layout_staged.outer),
        )
        # DGRAD and WGRAD alias this storage with independent element widths.
        c_smem_bytes = max(
            cute.cosize(dgrad_layout.epi_smem_layout_staged.outer)
            * dgrad_layout.c_dtype.width
            // 8,
            cute.cosize(wgrad_layout.epi_smem_layout_staged.outer)
            * wgrad_layout.c_dtype.width
            // 8,
        )
        if cutlass.const_expr(combine):
            c_smem_bytes = max(
                c_smem_bytes,
                cute.cosize(
                    self._make_mega_combine_c_smem_layout(dgrad_layout.epi_tile)
                )
                * dgrad_layout.c_dtype.width
                // 8,
            )
        c_smem_storage_elems = (c_smem_bytes + cutlass.Float32.width // 8 - 1) // (
            cutlass.Float32.width // 8
        )
        sfa_smem_elems = max(
            cute.cosize(dgrad_layout.sfa_smem_layout_staged),
            cute.cosize(wgrad_layout.sfa_smem_layout_staged),
        )
        sfb_smem_elems = max(
            cute.cosize(dgrad_layout.sfb_smem_layout_staged),
            cute.cosize(dgrad_layout.sfb_tma_smem_layout_staged),
            cute.cosize(wgrad_layout.sfb_smem_layout_staged),
            cute.cosize(wgrad_layout.sfb_tma_smem_layout_staged),
        )
        n_smem_empty = NUM_SMEM
        n_smem_full = NUM_SMEM
        n_tmem_full = NUM_TMEM
        n_tmem_empty = NUM_TMEM
        n_tile_consumer = NUM_TILE
        n_tile_producer = NUM_TILE
        n_tile_cta_bar = self.NUM_TILE_CTA_BARS
        n_tmem_dealloc = 1 if NUM_CTAS == 2 else 0
        n_cross_seam = 1 if self.OVERLAPPING_ACCUM else 0
        n_sf_smem = (
            self.SM103_SF_RING_TILES * sm103.sf_segments(self.sf_vec_size)
            if self.USE_SM103_ULTRA
            else 0
        )
        n_dispatch_ptr_smem = self.DISPATCH_QUANT_GROUPS * self.sf_vec_size
        n_dispatch_tile_id = self.DISPATCH_QUANT_GROUPS
        n_a_global_scale_inv = (
            self._a_global_scale_inv_smem_size if use_a_global_scale_inv else 0
        )
        # The SwiGLU producer indexes one reduction span per quant group. Size
        # it for the widest configured producer topology rather than one slot.
        n_nvfp4_row_reduce = (
            self.NVFP4_ROW_REDUCTION_SLOTS
            * (self.DISPATCH_QUANT_WARPS // self.DISPATCH_QUANT_WARPS_PER_GROUP)
            if use_a_global_scale_inv
            else 0
        )
        n_scatter_ptr_smem = 0
        if cutlass.const_expr(combine):
            n_scatter_ptr_smem = max(
                cute.size(dgrad_layout.epi_tile[0]),
                cute.size(dgrad_layout.epi_tile[1]),
            )
        n_tensormap_buffer = max(
            16 * self.MEGA_TENSORMAP_DESCRIPTOR_COUNT,
            (G + work_info_fields * NUM_TILE + 1) // 2,
        )

        @cute.struct
        class MegaBlockScaledSharedStorage:
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, c_smem_storage_elems], 1024
            ]
            sA: cute.struct.Align[cute.struct.MemRange[a_dtype, a_smem_elems], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[b_dtype, b_smem_elems], 1024]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfa_smem_elems], 1024
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfb_smem_elems], 1024
            ]
            tile_id_smem: cute.struct.MemRange[cutlass.Int32, NUM_TILE]
            dispatch_tile_id_smem: cute.struct.MemRange[
                cutlass.Int32, n_dispatch_tile_id
            ]
            dispatch_ptr_smem: cute.struct.MemRange[cutlass.Int64, n_dispatch_ptr_smem]
            scatter_ptr_smem: cute.struct.MemRange[cutlass.Int64, n_scatter_ptr_smem]
            a_global_scale_inv: cute.struct.MemRange[
                cutlass.Float32, n_a_global_scale_inv
            ]
            nvfp4_row_reduce: cute.struct.MemRange[cutlass.Int32, n_nvfp4_row_reduce]
            tensormap_buffer: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, n_tensormap_buffer],
                128,
            ]

            smem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_smem_empty]
            smem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_smem_full]
            sf_smem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_sf_smem]
            sf_smem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_sf_smem]
            tmem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_empty]
            tmem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_full]
            tile_consumer_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_consumer]
            tile_producer_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_producer]
            tile_cta_bar_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_cta_bar]
            tmem_dealloc_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_dealloc]
            cross_seam_mbar: cute.struct.MemRange[cutlass.Int64, n_cross_seam]

            tmem_holding_buf: cutlass.Int32

        return MegaBlockScaledSharedStorage

    def _make_prepare_shared_storage(self):
        @cute.struct
        class MegaPrepareSharedStorage:
            tensormap_buffer: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int64, 16 * self.MEGA_TENSORMAP_DESCRIPTOR_COUNT
                ],
                128,
            ]

        return MegaPrepareSharedStorage

    @cute.kernel
    def prepare_dispatch_bprop_kernel(  # noqa: C901
        self,
        dgrad: _MegaTensormapParams,
        wgrad: _MegaTensormapParams,
        split_sizes: cute.Tensor,
        counter_ptr: cute.Pointer,
        tensormaps: cute.Tensor,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        extra_counter_ptr: cute.Pointer,
        extra_counter_count: cutlass.Int32,
        activation_buffer_base_ptr: cutlass.Int64,
        activation_buffer_size_bytes: cutlass.Int64,
        activation_offsets: cute.Tensor,
        use_activation_buffer: cutlass.Constexpr[bool],
        weight_borrow_prepare=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        prepare_g, prepare_w, _ = cute.arch.block_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.prepare_shared_storage)
        tensormap_buffer_ptr = storage.tensormap_buffer.data_ptr()
        dgrad_tensormap_smem_ptr = tensormap_buffer_ptr
        wgrad_tensormap_smem_ptr = (
            tensormap_buffer_ptr + 16 * self.MEGA_TENSORMAP_DESCRIPTOR_COUNT // 2
        )

        tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)

        if (
            prepare_g == cutlass.Int32(0)
            and prepare_w == cutlass.Int32(0)
            and tidx == cutlass.Int32(0)
        ):
            cute.arch.store(counter_ptr, cutlass.Int32(0))

        extra_counters = cute.make_tensor(
            extra_counter_ptr,
            cute.make_ordered_layout((extra_counter_count,), order=(0,)),
        )
        block_dim_x, _, _ = cute.arch.block_dim()
        counter_zero_thread_idx = (
            prepare_w * cutlass.Int32(G) + prepare_g
        ) * block_dim_x + tidx
        counter_zero_threads = cutlass.Int32(G) * cutlass.Int32(2) * block_dim_x
        zero_iters = (
            extra_counter_count + counter_zero_threads - cutlass.Int32(1)
        ) // counter_zero_threads
        for i in cutlass.range(zero_iters, unroll=1):
            counter_idx = i * counter_zero_threads + counter_zero_thread_idx
            if counter_idx < extra_counter_count:
                extra_counters[counter_idx] = cutlass.Int32(0)

        tensormap_smem_ptr_pair = (
            dgrad_tensormap_smem_ptr,
            wgrad_tensormap_smem_ptr,
        )
        gemm_mnk_pair, problem_type_pair = _mega_problem_specs(
            M,
            N,
            K,
            MEGA_FORWARD_MODE=self.MEGA_FORWARD_MODE,
            mode=self.mode,
            MEGA_DGRAD_PROBLEM_TYPE=self.MEGA_DGRAD_PROBLEM_TYPE,
            MEGA_WGRAD_PROBLEM_TYPE=self.MEGA_WGRAD_PROBLEM_TYPE,
        )
        tensormap_base_pair = (
            self.MEGA_DGRAD_TENSORMAP_BASE,
            self.MEGA_WGRAD_TENSORMAP_BASE,
        )
        activation_offset_base_pair = (
            MEGA_FIRST_GEMM_OFFSET_BASE,
            MEGA_SECOND_GEMM_OFFSET_BASE,
        )
        for w in cutlass.range_constexpr(2):
            if prepare_w == cutlass.Int32(w):
                params = (dgrad, wgrad)[w]
                activation_buffer_operand_offsets = (
                    cutlass.Int64(MISSING_ACTIVATION_OFFSET),
                    cutlass.Int64(MISSING_ACTIVATION_OFFSET),
                    cutlass.Int64(MISSING_ACTIVATION_OFFSET),
                    cutlass.Int64(MISSING_ACTIVATION_OFFSET),
                    cutlass.Int64(MISSING_ACTIVATION_OFFSET),
                )
                if cutlass.const_expr(use_activation_buffer):
                    activation_buffer_operand_offsets = (
                        _activation_buffer_operand_offsets(
                            activation_offsets,
                            activation_offset_base_pair[w],
                            self.SWAP_AB,
                        )
                    )
                    pointer_strides = _activation_buffer_pointer_strides(
                        (params.base_ptrs, params.strides, params.sf_strides),
                        activation_buffer_base_ptr,
                        activation_offsets,
                        activation_offset_base_pair[w],
                        self.SWAP_AB,
                    )
                    base_ptrs, tensor_strides, sf_strides = pointer_strides
                    params = _MegaTensormapParams(
                        params.tma_atom_a,
                        params.tma_atom_b,
                        params.tma_atom_sfa,
                        params.tma_atom_sfb,
                        params.tma_atom_c,
                        base_ptrs=base_ptrs,
                        strides=tensor_strides,
                        sf_strides=sf_strides,
                        elem_sizes=params.elem_sizes,
                        dtypes=params.dtypes,
                    )
                prob_m, prob_n, prob_k = gemm_mnk_pair[w]
                problem = BlockscaledTensormapProblem(
                    split_sizes=split_sizes,
                    groups=G,
                    mnk=(prob_m, prob_n, prob_k),
                    problem_type=problem_type_pair[w],
                    tensormap_base=tensormap_base_pair[w],
                    prepare_g=prepare_g,
                    warp_idx=warp_idx,
                )
                borrow_a_base = cutlass.Int64(0)
                borrow_a_row_bytes = cutlass.Int64(0)
                borrow_sfa_base = cutlass.Int64(0)
                borrow_sfa_row_bytes = cutlass.Int64(0)
                if cutlass.const_expr(self.WEIGHT_BORROW_SLOTS > 0):
                    borrow_a_base = weight_borrow_prepare[4 * w + 0]
                    borrow_a_row_bytes = weight_borrow_prepare[4 * w + 1]
                    borrow_sfa_base = weight_borrow_prepare[4 * w + 2]
                    borrow_sfa_row_bytes = weight_borrow_prepare[4 * w + 3]
                _prepare_blockscaled_problem_tensormaps(
                    params=params,
                    problem=problem,
                    tensormaps=tensormaps,
                    tensormap_manager=tensormap_manager,
                    tensormap_smem_ptr_base=tensormap_smem_ptr_pair[w],
                    activation_buffer_operand_offsets=(
                        activation_buffer_operand_offsets
                    ),
                    activation_buffer_size_bytes=activation_buffer_size_bytes,
                    use_activation_buffer=use_activation_buffer,
                    BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                    SWAP_AB=self.SWAP_AB,
                    sf_dtype=self.sf_dtype,
                    sf_vec_size=self.sf_vec_size,
                    ring_b_rows=self.ACTIVATION_RING_B_ROWS,
                    use_sm103_ultra=self.USE_SM103_ULTRA,
                    borrow_slots=self.WEIGHT_BORROW_SLOTS,
                    borrow_a_base=borrow_a_base,
                    borrow_a_row_bytes=borrow_a_row_bytes,
                    borrow_sfa_base=borrow_sfa_base,
                    borrow_sfa_row_bytes=borrow_sfa_row_bytes,
                )

    @cute.jit
    def __call__(  # noqa: C901
        self,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_c: cute.Tensor,
        tensor_sfa: cute.Tensor,
        tensor_sfb: cute.Tensor,
        wgrad_tensor_a: cute.Tensor,
        wgrad_tensor_b: cute.Tensor,
        wgrad_tensor_c: cute.Tensor,
        wgrad_tensor_sfa: cute.Tensor,
        wgrad_tensor_sfb: cute.Tensor,
        split_sizes: cute.Tensor,
        counter: cute.Tensor,
        tensormaps: cute.Tensor,
        gather_ptrs: cute.Tensor,
        dispatch_quant_work_counter: cute.Tensor,
        dispatch_quant_done_counter: cute.Tensor,
        dispatch_quant_row_q_words: cute.Tensor,
        dispatch_quant_row_scale: cute.Tensor,
        dispatch_quant_row_global_scale_inv: cute.Tensor,
        dispatch_quant_col_q_words: cute.Tensor,
        dispatch_quant_col_scale: cute.Tensor,
        combine_scatter_ptrs: cute.Tensor,
        combine_dz: cute.Tensor,
        combine_h1: cute.Tensor,
        activation_gather_ptrs: cute.Tensor,
        activation_quant_work_counter: cute.Tensor,
        activation_quant_done_counter: cute.Tensor,
        activation_quant_row_q_words: cute.Tensor,
        activation_quant_row_scale: cute.Tensor,
        activation_quant_row_global_scale_inv: cute.Tensor,
        activation_quant_col_q_words: cute.Tensor,
        activation_quant_col_scale: cute.Tensor,
        dgrad_pointer_strides: BlockscaledPointerStrideArgs,
        wgrad_pointer_strides: BlockscaledPointerStrideArgs,
        dgrad_elem_sizes: cutlass.Constexpr[tuple[int, int, int]],
        wgrad_elem_sizes: cutlass.Constexpr[tuple[int, int, int]],
        wgrad_c_dtype: cutlass.Constexpr[type[cutlass.Numeric]],
        wgrad_output_accum: cutlass.Constexpr[bool],
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        num_clusters: int,
        dispatch_quant_total_tiles: cutlass.Int32,
        dispatch_quant_counter_zero_count: cutlass.Int32,
        row_done_counter_offset: cutlass.Int32,
        col_done_counter_offset: cutlass.Int32,
        activation_quant_total_tiles: cutlass.Int32,
        activation_row_done_counter_offset: cutlass.Int32,
        activation_col_done_counter_offset: cutlass.Int32,
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
        stream: cuda.CUstream,
        weight_borrow_window_ptrs: cute.Tensor | None = None,
        weight_borrow_src_rank: cute.Tensor | None = None,
        weight_borrow_src_slot: cute.Tensor | None = None,
        weight_borrow_scalars=None,
        weight_borrow_prepare=None,
    ):
        # Tensors cross the launch boundary as top-level args; re-bundle them
        # with the scalar half for the kernel-side plumbing.
        weight_borrow_args = None
        if cutlass.const_expr(weight_borrow_window_ptrs is not None):
            from .weight_borrow import WeightBorrowArgs

            s = weight_borrow_scalars
            weight_borrow_args = WeightBorrowArgs(
                window_ptrs=weight_borrow_window_ptrs,
                src_rank=weight_borrow_src_rank,
                src_slot=weight_borrow_src_slot,
                slot_buf_base=s[0],
                w13_win_off=s[1],
                w13_slot_off=s[2],
                w13_row_bytes=s[3],
                s13_win_off=s[4],
                s13_slot_off=s[5],
                s13_row_bytes=s[6],
                w2_win_off=s[7],
                w2_slot_off=s[8],
                w2_row_bytes=s[9],
                s2_win_off=s[10],
                s2_slot_off=s[11],
                s2_row_bytes=s[12],
            )
        eff_tensor_a, eff_tensor_b = _swap_if(self.SWAP_AB, tensor_a, tensor_b)
        eff_tensor_sfa, eff_tensor_sfb = _swap_if(self.SWAP_AB, tensor_sfa, tensor_sfb)
        dgrad_eff_pointer_strides = swap_ab_pointer_strides(
            dgrad_pointer_strides,
            self.SWAP_AB,
        )
        dgrad_elem_size_a, dgrad_elem_size_b, dgrad_elem_size_c = dgrad_elem_sizes
        eff_elem_a, eff_elem_b = _swap_if(
            self.SWAP_AB,
            dgrad_elem_size_a,
            dgrad_elem_size_b,
        )
        dgrad_eff_elem_sizes = (eff_elem_a, eff_elem_b, dgrad_elem_size_c)
        wgrad_eff_tensor_a, wgrad_eff_tensor_b = _swap_if(
            self.SWAP_AB,
            wgrad_tensor_a,
            wgrad_tensor_b,
        )
        wgrad_eff_tensor_sfa, wgrad_eff_tensor_sfb = _swap_if(
            self.SWAP_AB,
            wgrad_tensor_sfa,
            wgrad_tensor_sfb,
        )
        wgrad_eff_pointer_strides = swap_ab_pointer_strides(
            wgrad_pointer_strides,
            self.SWAP_AB,
        )
        wgrad_elem_size_a, wgrad_elem_size_b, wgrad_elem_size_c = wgrad_elem_sizes
        wgrad_eff_elem_a, wgrad_eff_elem_b = _swap_if(
            self.SWAP_AB,
            wgrad_elem_size_a,
            wgrad_elem_size_b,
        )
        wgrad_eff_elem_sizes = (
            wgrad_eff_elem_a,
            wgrad_eff_elem_b,
            wgrad_elem_size_c,
        )

        tensor_c_eff = _transpose_c_if_swap(tensor_c, self.SWAP_AB)
        wgrad_tensor_c_eff = _transpose_c_if_swap(wgrad_tensor_c, self.SWAP_AB)

        dgrad_layout = self._setup_gemm_layout_bundle(
            tensor_a=eff_tensor_a,
            tensor_b=eff_tensor_b,
            tensor_c=tensor_c,
            tensor_c_eff=tensor_c_eff,
        )
        wgrad_layout = self._setup_gemm_layout_bundle(
            tensor_a=wgrad_eff_tensor_a,
            tensor_b=wgrad_eff_tensor_b,
            tensor_c=wgrad_tensor_c,
            tensor_c_eff=wgrad_tensor_c_eff,
        )

        dgrad_tma_atoms, dgrad_tma_tensors = self._make_first_gemm_tma_atoms(
            layout=dgrad_layout,
            tensor_a=eff_tensor_a,
            tensor_b=eff_tensor_b,
            tensor_c_eff=tensor_c_eff,
            tensor_sfa=eff_tensor_sfa,
            tensor_sfb=eff_tensor_sfb,
            output_accum=False,
        )
        wgrad_tma_atoms, wgrad_tma_tensors = self._make_gemm_tma_atoms(
            layout=wgrad_layout,
            tensor_a=wgrad_eff_tensor_a,
            tensor_b=wgrad_eff_tensor_b,
            tensor_c_eff=wgrad_tensor_c_eff,
            tensor_sfa=wgrad_eff_tensor_sfa,
            tensor_sfb=wgrad_eff_tensor_sfb,
            output_accum=wgrad_output_accum,
        )
        dgrad_kernel_params = self._make_gemm_kernel_params(
            dgrad_layout,
            dgrad_tma_atoms,
            dgrad_tma_tensors,
        )
        wgrad_kernel_params = self._make_gemm_kernel_params(
            wgrad_layout,
            wgrad_tma_atoms,
            wgrad_tma_tensors,
        )
        dgrad_tensormap_params = self._make_tensormap_params(
            dgrad_tma_atoms,
            dgrad_eff_pointer_strides,
            dgrad_eff_elem_sizes,
            (dgrad_layout.a_dtype, dgrad_layout.b_dtype, dgrad_layout.c_dtype),
        )
        wgrad_tensormap_params = self._make_tensormap_params(
            wgrad_tma_atoms,
            wgrad_eff_pointer_strides,
            wgrad_eff_elem_sizes,
            (wgrad_layout.a_dtype, wgrad_layout.b_dtype, wgrad_c_dtype),
        )

        self.shared_storage = self._make_shared_storage(
            dgrad_layout,
            wgrad_layout,
            combine,
            G,
            use_a_global_scale_inv=use_global_scale_inv,
        )
        self.prepare_shared_storage = self._make_prepare_shared_storage()
        grid = (num_clusters * self.NUM_CTAS, 1, 1)

        if cutlass.const_expr(use_device_tensormaps):
            self.prepare_dispatch_bprop_kernel(
                dgrad=dgrad_tensormap_params,
                wgrad=wgrad_tensormap_params,
                split_sizes=split_sizes,
                counter_ptr=counter.iterator,
                tensormaps=tensormaps,
                G=G,
                M=M,
                N=N,
                K=K,
                extra_counter_ptr=dispatch_quant_work_counter.iterator,
                extra_counter_count=dispatch_quant_counter_zero_count,
                activation_buffer_base_ptr=activation_buffer_base_ptr,
                activation_buffer_size_bytes=activation_buffer_size_bytes,
                activation_offsets=activation_offsets,
                use_activation_buffer=use_activation_buffer,
                weight_borrow_prepare=weight_borrow_prepare,
            ).launch(
                grid=(G, 2, 1),
                block=(128, 1, 1),
                stream=stream,
            )

        self.dispatch_bprop_kernel(
            dgrad=dgrad_kernel_params,
            wgrad=wgrad_kernel_params,
            split_sizes=split_sizes,
            counter_ptr=counter.iterator,
            tensormaps=tensormaps,
            mDispatchQuantGatherPtrs=gather_ptrs,
            mDispatchQuantWorkCounter=dispatch_quant_work_counter,
            mDispatchQuantDoneCounter=dispatch_quant_done_counter,
            mDispatchQuantRowQWords=dispatch_quant_row_q_words,
            mDispatchQuantRowScale=dispatch_quant_row_scale,
            mDispatchQuantRowGlobalScaleInv=(dispatch_quant_row_global_scale_inv),
            mDispatchQuantColQWords=dispatch_quant_col_q_words,
            mDispatchQuantColScale=dispatch_quant_col_scale,
            mCombineScatterPtrs=combine_scatter_ptrs,
            mCombineDz=combine_dz,
            mCombineH1=combine_h1,
            row_done_counter_offset=row_done_counter_offset,
            col_done_counter_offset=col_done_counter_offset,
            mActivationGatherPtrs=activation_gather_ptrs,
            mActivationQuantWorkCounter=activation_quant_work_counter,
            mActivationQuantDoneCounter=activation_quant_done_counter,
            mActivationQuantRowQWords=activation_quant_row_q_words,
            mActivationQuantRowScale=activation_quant_row_scale,
            mActivationQuantRowGlobalScaleInv=(activation_quant_row_global_scale_inv),
            mActivationQuantColQWords=activation_quant_col_q_words,
            mActivationQuantColScale=activation_quant_col_scale,
            activation_quant_total_tiles=activation_quant_total_tiles,
            activation_row_done_counter_offset=activation_row_done_counter_offset,
            activation_col_done_counter_offset=activation_col_done_counter_offset,
            G=G,
            M=M,
            N=N,
            K=K,
            local_rank=local_rank,
            dispatch_quant_total_tiles=dispatch_quant_total_tiles,
            combine=combine,
            activation_dispatch=activation_dispatch,
            use_device_tensormaps=use_device_tensormaps,
            activation_buffer_base_ptr=activation_buffer_base_ptr,
            activation_buffer_size_bytes=activation_buffer_size_bytes,
            activation_offsets=activation_offsets,
            use_activation_buffer=use_activation_buffer,
            conditional_execution=conditional_execution,
            use_conditional_execution=use_conditional_execution,
            dgrad_b_global_scale_inv_ptr=dgrad_b_global_scale_inv_ptr,
            wgrad_b_global_scale_inv_ptr=wgrad_b_global_scale_inv_ptr,
            use_global_scale_inv=use_global_scale_inv,
            nvfp4_recip_lut_ptr=nvfp4_recip_lut_ptr,
            weight_borrow_args=weight_borrow_args,
        ).launch(
            grid=grid,
            block=(self.THREADS_PER_CTA, 1, 1),
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def dispatch_bprop_kernel(  # noqa: C901
        self,
        dgrad: _MegaGemmKernelParams,
        wgrad: _MegaGemmKernelParams,
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
                thread_exit()
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
            if cutlass.const_expr(combine):
                source_element_size = cutlass.Int64(
                    cutlass.const_expr(mCombineDz.element_type.width // 8)
                )
                if cutlass.const_expr(self.mode == self.MEGA_FORWARD_MODE):
                    has_activation_rows = cutlass.Int64(
                        activation_rows > cutlass.Int32(0)
                    )
                    source_byte_extent = (
                        has_activation_rows
                        * (
                            cutlass.Int64(2) * cutlass.Int64(activation_rows)
                            - cutlass.Int64(1)
                        )
                        * cutlass.Int64(N)
                        * source_element_size
                    )
                    source_x_byte_extent = source_byte_extent
                    source_y_byte_extent = source_byte_extent
                else:
                    source_x_byte_extent = (
                        cutlass.Int64(activation_rows)
                        * cutlass.Int64(K // cutlass.Int32(2))
                        * source_element_size
                    )
                    source_y_byte_extent = (
                        cutlass.Int64(activation_rows)
                        * cutlass.Int64(K)
                        * source_element_size
                    )
                mCombineDz = _activation_buffer_tensor_with_byte_extent(
                    mCombineDz,
                    activation_buffer_base_ptr,
                    activation_buffer_size_bytes,
                    activation_offsets,
                    MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_SOURCE_X_OFFSET,
                    mCombineDz.element_type,
                    source_x_byte_extent,
                )
                mCombineH1 = _activation_buffer_tensor_with_byte_extent(
                    mCombineH1,
                    activation_buffer_base_ptr,
                    activation_buffer_size_bytes,
                    activation_offsets,
                    MEGA_FIRST_GEMM_OFFSET_BASE + ACTIVATION_SOURCE_Y_OFFSET,
                    mCombineH1.element_type,
                    source_y_byte_extent,
                )
            if cutlass.const_expr(activation_dispatch):
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
        tile_id_smem_ptr = storage.tile_id_smem.data_ptr()
        dispatch_tile_id_smem_ptr = storage.dispatch_tile_id_smem.data_ptr()
        dispatch_ptr_smem_ptr = storage.dispatch_ptr_smem.data_ptr()
        scatter_ptr_smem_ptr = storage.dispatch_ptr_smem.data_ptr()
        if cutlass.const_expr(combine):
            scatter_ptr_smem_ptr = storage.scatter_ptr_smem.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf

        if warp_idx == self.EPILOG_WARP_IDS[0]:
            with cute.arch.elect_one():
                for i in range(self.NUM_SMEM_BUFFERS):
                    cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
                    cute.arch.mbarrier_init(ab_full_mbar + i, 1)
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
                tmem_holding_buf,
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
        bid = cute.arch.block_idx()
        dgrad_state = self._make_gemm_prologue(
            dgrad,
            storage,
            cluster_cta_rank,
            bid,
        )
        wgrad_state = self._make_gemm_prologue(
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
                dgrad_state.block_in_cluster_coord_vmnk,
                mcast_mode=2,
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                dgrad.cluster_layout_vmnk,
                dgrad_state.block_in_cluster_coord_vmnk,
                mcast_mode=1,
            )
            sfa_full_mcast_mask = a_full_mcast_mask
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                dgrad.cluster_layout_sfb_vmnk,
                dgrad_state.block_in_cluster_coord_sfb_vmnk,
                mcast_mode=1,
            )
            block_in_cluster_coord_vmnk_peer = (
                dgrad_state.block_in_cluster_coord_vmnk[0] ^ 1,
                *dgrad_state.block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                dgrad.cluster_layout_vmnk,
                block_in_cluster_coord_vmnk_peer,
                mcast_mode=2,
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                dgrad.cluster_layout_vmnk,
                block_in_cluster_coord_vmnk_peer,
                mcast_mode=1,
            )
            ab_empty_mcast_mask = (
                a_full_mcast_mask
                | b_full_mcast_mask
                | a_full_mcast_mask_peer
                | b_full_mcast_mask_peer
            )
            acc_full_mcast_mask = cute.make_layout_image_mask(
                dgrad.cluster_layout_vmnk,
                dgrad_state.block_in_cluster_coord_vmnk,
                mode=0,
            )

        dgrad_num_tma_load_bytes, _, _ = self._num_tma_load_bytes(
            sA=dgrad_state.sA,
            sB=dgrad_state.sB,
            sSFA=dgrad_state.sSFA,
            sSFB=dgrad_state.sSFB,
            a_smem_layout_staged=dgrad.a_smem_layout_staged,
            b_smem_layout_staged=dgrad.b_smem_layout_staged,
            sfa_smem_layout_staged=dgrad.sfa_smem_layout_staged,
            sfb_tma_smem_layout_staged=dgrad.sfb_tma_smem_layout_staged,
            tiled_mma=dgrad.tiled_mma,
        )
        wgrad_num_tma_load_bytes, _, _ = self._num_tma_load_bytes(
            sA=wgrad_state.sA,
            sB=wgrad_state.sB,
            sSFA=wgrad_state.sSFA,
            sSFB=wgrad_state.sSFB,
            a_smem_layout_staged=wgrad.a_smem_layout_staged,
            b_smem_layout_staged=wgrad.b_smem_layout_staged,
            sfa_smem_layout_staged=wgrad.sfa_smem_layout_staged,
            sfb_tma_smem_layout_staged=wgrad.sfb_tma_smem_layout_staged,
            tiled_mma=wgrad.tiled_mma,
        )
        dgrad_tma = self._make_tma_pipeline(
            dgrad, dgrad_state, dgrad_num_tma_load_bytes
        )
        wgrad_tma = self._make_tma_pipeline(
            wgrad, wgrad_state, wgrad_num_tma_load_bytes
        )
        problem = _MegaProblem(
            split_sizes=split_sizes,
            groups=G,
            mnk=(M, N, K),
            local_rank=local_rank,
        )
        sync = _MegaPipelineSync(
            ab_full_mbar=ab_full_mbar,
            ab_empty_mbar=ab_empty_mbar,
            tmem_full_mbar=tmem_full_mbar,
            tmem_empty_mbar=tmem_empty_mbar,
            tile_consumer_mbar=tile_consumer_mbar,
            tile_producer_mbar=tile_producer_mbar,
            tile_cta_bar_mbar=tile_cta_bar_mbar,
            tile_id_smem_ptr=tile_id_smem_ptr,
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
        dispatch_quant_sync = _MegaDispatchQuantSync(
            done_counter=mDispatchQuantDoneCounter,
            done_counter_offsets=(
                row_done_counter_offset,
                col_done_counter_offset,
            ),
        )
        activation_quant_sync = _MegaDispatchQuantSync(
            done_counter=mActivationQuantDoneCounter,
            done_counter_offsets=(
                activation_row_done_counter_offset,
                activation_col_done_counter_offset,
            ),
        )

        tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)
        if warp_idx == 0:
            for w in cutlass.range_constexpr(2):
                params = (dgrad, wgrad)[w]
                cpasync.prefetch_descriptor(params.tma_atom_a)
                cpasync.prefetch_descriptor(params.tma_atom_b)
                cpasync.prefetch_descriptor(params.tma_atom_sfa)
                cpasync.prefetch_descriptor(params.tma_atom_sfb)
                cpasync.prefetch_descriptor(params.tma_atom_c)

        if warp_idx == self.TMA_AB_WARP_ID:
            _mega_tma_producer_body(
                params_from_kernel(MegaPipelineParams, self),
                dgrad=dgrad_tma,
                wgrad=wgrad_tma,
                problem=problem,
                sync=sync,
                dispatch_quant=dispatch_quant_sync,
                activation_quant=activation_quant_sync,
                tensormap_manager=tensormap_manager,
                tensormaps=tensormaps,
                combine=combine,
                activation_dispatch=activation_dispatch,
                work_info_smem_ptr=work_info_smem_ptr,
                use_device_tensormaps=use_device_tensormaps,
                format=self.format,
            )
        elif warp_idx == self.MMA_WARP_ID:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            dgrad_mma = self._make_mma_pipeline(dgrad, dgrad_state, tmem_ptr)
            wgrad_mma = self._make_mma_pipeline(wgrad, wgrad_state, tmem_ptr)
            _mega_mma_consumer_body(
                params_from_kernel(MegaPipelineParams, self),
                dgrad=dgrad_mma,
                wgrad=wgrad_mma,
                problem=problem,
                sync=sync,
                work_info_smem_ptr=work_info_smem_ptr,
                cta_group=self.cta_group,
                sf_dtype=self.sf_dtype,
            )
        elif warp_idx in self.EPILOG_WARP_IDS:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            if cutlass.const_expr(combine and self.mode != self.MEGA_FORWARD_MODE):
                assert dgrad_state.combine_sC is not None
                dgrad_sC = dgrad_state.combine_sC
            else:
                dgrad_sC = dgrad_state.sC
            dgrad_epilog = self._make_epilog_pipeline(
                params=dgrad,
                state=dgrad_state,
                sC=dgrad_sC,
                tidx=tidx,
                tmem_ptr=tmem_ptr,
            )
            if cutlass.const_expr(combine and self.mode == self.MEGA_FORWARD_MODE):
                assert wgrad_state.combine_sC is not None
                wgrad_sC = wgrad_state.combine_sC
            else:
                wgrad_sC = wgrad_state.sC
            wgrad_epilog = self._make_epilog_pipeline(
                params=wgrad,
                state=wgrad_state,
                sC=wgrad_sC,
                tidx=tidx,
                tmem_ptr=tmem_ptr,
            )
            self._mega_epilog_consumer_body(
                tidx=tidx,
                dgrad=dgrad_epilog,
                wgrad=wgrad_epilog,
                problem=problem,
                sync=sync,
                activation_quant=activation_quant_sync,
                tensormap_manager=tensormap_manager,
                tensormaps=tensormaps,
                mScatter=mCombineScatterPtrs,
                scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
                combine=combine,
                work_info_smem_ptr=work_info_smem_ptr,
                use_device_tensormaps=use_device_tensormaps,
            )
            if warp_idx == self.EPILOG_WARP_IDS[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=(self.NUM_CTAS == 2))
            cute.arch.barrier(
                barrier_id=_BAR_EPILOG_SYNC,
                number_of_threads=32 * len(self.EPILOG_WARP_IDS),
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
        else:
            if cutlass.const_expr(self.mode == self.MEGA_FORWARD_MODE):
                if (warp_idx >= self.DISPATCH_QUANT_FIRST_WARP) & (
                    warp_idx < self.FORWARD_INPUT_QUANT_FIRST_WARP
                ):
                    self._mega_forward_swiglu_quant_producer_body(
                        mWorkCounter=mActivationQuantWorkCounter,
                        mDoneCounter=mActivationQuantDoneCounter,
                        h1_done_counter_offset=(activation_col_done_counter_offset),
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
                        local_rank=local_rank,
                    )
                elif warp_idx >= self.FORWARD_INPUT_QUANT_FIRST_WARP:
                    self._mega_dispatch_quant_producer_body(
                        mWorkCounter=mDispatchQuantWorkCounter,
                        mDoneCounter=mDispatchQuantDoneCounter,
                        tile_smem_ptr=dispatch_tile_id_smem_ptr,
                        dispatch_ptr_smem_ptr=dispatch_ptr_smem_ptr,
                        mGatherPtrs=mDispatchQuantGatherPtrs,
                        mRowQWords=mDispatchQuantRowQWords,
                        mRowScale=mDispatchQuantRowScale,
                        mColQWords=mDispatchQuantColQWords,
                        mColScale=mDispatchQuantColScale,
                        split_sizes=split_sizes,
                        G=G,
                        M=M,
                        K=K,
                        total_tiles=dispatch_quant_total_tiles,
                        local_rank=local_rank,
                        row_done_counter_offset=row_done_counter_offset,
                        col_done_counter_offset=col_done_counter_offset,
                    )
            elif cutlass.const_expr(combine):
                if cutlass.const_expr(activation_dispatch):
                    if (warp_idx >= self.COMBINE_SWIGLU_QUANT_FIRST_WARP) & (
                        warp_idx < self.COMBINE_ACTIVATION_DISPATCH_FIRST_WARP
                    ):
                        _combine_swiglu_bwd_quant_producer_body(
                            params_from_kernel(CombineSwigluQuantParams, self),
                            mWorkCounter=mDispatchQuantWorkCounter,
                            mDoneCounter=mDispatchQuantDoneCounter,
                            mDz=mCombineDz,
                            mH1=mCombineH1,
                            mScatter=mCombineScatterPtrs,
                            mRowQWords=mDispatchQuantRowQWords,
                            mRowScale=mDispatchQuantRowScale,
                            mDxyColQWords=mDispatchQuantColQWords,
                            mDxyColScale=mDispatchQuantColScale,
                            split_sizes=split_sizes,
                            G=G,
                            M=M,
                            K=K,
                            local_rank=local_rank,
                            col_done_counter_offset=col_done_counter_offset,
                            signal_col_tile_done=True,
                        )
                    if warp_idx >= self.COMBINE_ACTIVATION_DISPATCH_FIRST_WARP:
                        self._mega_dispatch_quant_producer_body(
                            mWorkCounter=mActivationQuantWorkCounter,
                            mDoneCounter=mActivationQuantDoneCounter,
                            tile_smem_ptr=dispatch_tile_id_smem_ptr,
                            dispatch_ptr_smem_ptr=dispatch_ptr_smem_ptr,
                            mGatherPtrs=mActivationGatherPtrs,
                            mRowQWords=mActivationQuantRowQWords,
                            mRowScale=mActivationQuantRowScale,
                            mColQWords=mActivationQuantColQWords,
                            mColScale=mActivationQuantColScale,
                            split_sizes=split_sizes,
                            G=G,
                            M=M,
                            K=N,
                            total_tiles=activation_quant_total_tiles,
                            local_rank=local_rank,
                            row_done_counter_offset=(
                                activation_row_done_counter_offset
                            ),
                            col_done_counter_offset=(
                                activation_col_done_counter_offset
                            ),
                            row_quant=False,
                        )
                elif warp_idx >= self.COMBINE_SWIGLU_QUANT_FIRST_WARP:
                    _combine_swiglu_bwd_quant_producer_body(
                        params_from_kernel(CombineSwigluQuantParams, self),
                        mWorkCounter=mDispatchQuantWorkCounter,
                        mDoneCounter=mDispatchQuantDoneCounter,
                        mDz=mCombineDz,
                        mH1=mCombineH1,
                        mScatter=mCombineScatterPtrs,
                        mRowQWords=mDispatchQuantRowQWords,
                        mRowScale=mDispatchQuantRowScale,
                        mDxyColQWords=mDispatchQuantColQWords,
                        mDxyColScale=mDispatchQuantColScale,
                        split_sizes=split_sizes,
                        G=G,
                        M=M,
                        K=K,
                        local_rank=local_rank,
                        col_done_counter_offset=col_done_counter_offset,
                        signal_col_tile_done=True,
                    )
            elif warp_idx >= self.DISPATCH_QUANT_FIRST_WARP:
                self._mega_dispatch_quant_producer_body(
                    mWorkCounter=mDispatchQuantWorkCounter,
                    mDoneCounter=mDispatchQuantDoneCounter,
                    tile_smem_ptr=dispatch_tile_id_smem_ptr,
                    dispatch_ptr_smem_ptr=dispatch_ptr_smem_ptr,
                    mGatherPtrs=mDispatchQuantGatherPtrs,
                    mRowQWords=mDispatchQuantRowQWords,
                    mRowScale=mDispatchQuantRowScale,
                    mColQWords=mDispatchQuantColQWords,
                    mColScale=mDispatchQuantColScale,
                    split_sizes=split_sizes,
                    G=G,
                    M=M,
                    K=K,
                    total_tiles=dispatch_quant_total_tiles,
                    local_rank=local_rank,
                    row_done_counter_offset=row_done_counter_offset,
                    col_done_counter_offset=col_done_counter_offset,
                )

        if cutlass.const_expr(self.NUM_CTAS == 2):
            # Keep every active thread alive until peer DSMEM accesses complete.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    def __init__(
        self,
        *,
        mode: int = MEGA_BACKWARD_DISPATCH_MODE,
        **kwargs,
    ) -> None:
        if mode not in (
            self.MEGA_BACKWARD_DISPATCH_MODE,
            self.MEGA_FORWARD_MODE,
        ):
            raise ValueError(f"unsupported MegaBlockScaledGroupedGemm mode={mode}")
        kwargs.setdefault(
            "problem_type",
            _FPROP if mode == self.MEGA_FORWARD_MODE else self.MEGA_DGRAD_PROBLEM_TYPE,
        )
        super().__init__(mode=self.DISPATCH_MODE, **kwargs)
        self.mode = mode
        self.TOTAL_WARPS = self.DISPATCH_TOTAL_WARPS
        self.THREADS_PER_CTA = self.DISPATCH_THREADS_PER_CTA

    @classmethod
    def from_config(
        cls,
        config: dict,
        **kwargs,
    ) -> "MegaBlockScaledGroupedGemmKernel":
        mode = kwargs.pop("mode", cls.MEGA_BACKWARD_DISPATCH_MODE)
        if mode not in (cls.MEGA_BACKWARD_DISPATCH_MODE, cls.MEGA_FORWARD_MODE):
            raise ValueError(f"unsupported MegaBlockScaledGroupedGemm mode={mode}")
        kwargs.setdefault(
            "problem_type",
            _FPROP if mode == cls.MEGA_FORWARD_MODE else cls.MEGA_DGRAD_PROBLEM_TYPE,
        )
        return super().from_config(
            config,
            mode=mode,
            **kwargs,
        )


class _TensorValidationCache:
    """Memoize validation without retaining tensors or allocator addresses."""

    def __init__(self) -> None:
        self._entries: dict[
            int, tuple[weakref.ReferenceType[torch.Tensor], set[tuple]]
        ] = {}

    def __contains__(self, item: tuple[torch.Tensor, tuple]) -> bool:
        tensor, key = item
        entry = self._entries.get(id(tensor))
        return entry is not None and entry[0]() is tensor and key in entry[1]

    def add(self, tensor: torch.Tensor, key: tuple) -> None:
        tensor_id = id(tensor)
        entry = self._entries.get(tensor_id)
        if entry is None or entry[0]() is not tensor:

            def discard(
                ref: weakref.ReferenceType[torch.Tensor],
                tensor_id: int = tensor_id,
            ) -> None:
                current = self._entries.get(tensor_id)
                if current is not None and current[0] is ref:
                    self._entries.pop(tensor_id, None)

            entry = (weakref.ref(tensor, discard), set())
            self._entries[tensor_id] = entry
        entry[1].add(key)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class _MegaDispatchPlan:
    rows: int
    dim: int
    dispatch_quant_total_tiles: int
    row_done_counter_offset: int
    row_done_counter_size: int
    col_done_counter_offset: int
    col_done_counter_size: int
    done_counter_size: int
    tensormap_descriptor_count: int
