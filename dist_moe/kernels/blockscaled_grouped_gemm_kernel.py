# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side kernel class and CuTeDSL helpers for the block-scaled grouped GEMM."""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05

from . import sm103_blockscaled_helpers as sm103
from .activation_buffer import (
    ACTIVATION_A_Q_OFFSET,
    ACTIVATION_A_SCALE_OFFSET,
    ACTIVATION_B_Q_OFFSET,
    ACTIVATION_B_SCALE_OFFSET,
    ACTIVATION_C_OFFSET,
    MISSING_ACTIVATION_OFFSET,
)
from .blockscaled_gemm_tiles import (
    _blockscaled_mma_consumer_tile,
    _blockscaled_tma_issue_weights_ahead,
    _blockscaled_tma_load_tile,
    _prepare_blockscaled_problem_tensormaps,
    _sm103_mma_consumer_tile,
    _stage_a_global_scale_inv,
    _tma_wait_and_arm_expect_tx,
)
from .config import (
    blockscaled_epilogue_subtile_divisor,
    EPILOGUE_SUBTILE_AUTO,
    MMA_INST_TILE_K,
    uses_paged_blockscaled_scale_rows,
)
from .grouped_gemm_kernel import (
    _BAR_EPILOG_SYNC,
    _BAR_FULL_CTA_SYNC,
    _cap_tmem_ld_repetition,
    _epilog_wait_pending_tma_store,
    _swap_if,
    _transpose_c_if_swap,
)
from .params import (
    BlockScaledFormatSpec,
    BlockscaledPointerStrideArgs,
    BlockscaledTensormapParams,
    BlockscaledTensormapProblem,
    GroupedGemmEpilogPipeline,
    GroupedGemmMmaPipeline,
    GroupedGemmPipelineSync,
    GroupedGemmProblem,
    GroupedGemmTmaPipeline,
    swap_ab_pointer_strides,
)
from .tile_scheduler import (
    _FPROP,
    _get_bufidx_phase,
    _WGRAD,
    DynamicTileScheduler,
    GroupedProblemVisitor,
    stage_expert_metadata,
    StaticTileScheduler,
)

_TENSORMAP_DESCRIPTOR_INT64S: int = 16


_BLOCKSCALED_TENSORMAP_DESCRIPTOR_COUNT: int = 5


_BLOCKSCALED_TENSORMAP_STAGING_INT64S: int = (
    _TENSORMAP_DESCRIPTOR_INT64S * _BLOCKSCALED_TENSORMAP_DESCRIPTOR_COUNT
)


@cute.jit
def _activation_buffer_operand_offsets(
    activation_offsets: cute.Tensor,
    offset_base: cutlass.Constexpr[int],
    swap_ab: cutlass.Constexpr[bool],
):
    a_idx = offset_base + (
        ACTIVATION_B_Q_OFFSET if cutlass.const_expr(swap_ab) else ACTIVATION_A_Q_OFFSET
    )
    b_idx = offset_base + (
        ACTIVATION_A_Q_OFFSET if cutlass.const_expr(swap_ab) else ACTIVATION_B_Q_OFFSET
    )
    sfa_idx = offset_base + (
        ACTIVATION_B_SCALE_OFFSET
        if cutlass.const_expr(swap_ab)
        else ACTIVATION_A_SCALE_OFFSET
    )
    sfb_idx = offset_base + (
        ACTIVATION_A_SCALE_OFFSET
        if cutlass.const_expr(swap_ab)
        else ACTIVATION_B_SCALE_OFFSET
    )
    c_idx = offset_base + ACTIVATION_C_OFFSET

    return (
        cutlass.Int64(activation_offsets[a_idx]),
        cutlass.Int64(activation_offsets[b_idx]),
        cutlass.Int64(activation_offsets[c_idx]),
        cutlass.Int64(activation_offsets[sfa_idx]),
        cutlass.Int64(activation_offsets[sfb_idx]),
    )


@cute.jit
def _activation_buffer_pointer_strides(
    pointer_strides: BlockscaledPointerStrideArgs,
    activation_buffer_base_ptr: cutlass.Int64,
    activation_offsets: cute.Tensor,
    offset_base: cutlass.Constexpr[int],
    swap_ab: cutlass.Constexpr[bool],
) -> BlockscaledPointerStrideArgs:
    base_ptrs, tensor_strides, sf_strides = pointer_strides

    a_base, b_base, c_base, sfa_base, sfb_base = base_ptrs
    a_offset, b_offset, c_offset, sfa_offset, sfb_offset = (
        _activation_buffer_operand_offsets(
            activation_offsets,
            offset_base,
            swap_ab,
        )
    )
    if a_offset >= cutlass.Int64(0):
        a_base = activation_buffer_base_ptr + a_offset
    if b_offset >= cutlass.Int64(0):
        b_base = activation_buffer_base_ptr + b_offset
    if c_offset >= cutlass.Int64(0):
        c_base = activation_buffer_base_ptr + c_offset
    if sfa_offset >= cutlass.Int64(0):
        sfa_base = activation_buffer_base_ptr + sfa_offset
    if sfb_offset >= cutlass.Int64(0):
        sfb_base = activation_buffer_base_ptr + sfb_offset
    return (
        (a_base, b_base, c_base, sfa_base, sfb_base),
        tensor_strides,
        sf_strides,
    )


def _make_flexible_smem_layout_sfa(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
) -> cute.Layout:
    """Build the stable scale-factor hierarchy used by flexible MMA tiles.

    The kernel's SFB TMA layout, per-N-half field rebinding, and s2t copies
    require the N-atom factor to remain nested inside operand mode 0. Derive
    the K-instruction count from the tiled MMA instead of assuming a fixed
    instruction shape.
    """
    sfa_tile_shape = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[2],
    )
    smem_layout = cute.tile_to_shape(
        blockscaled_utils.BlockScaledBasicChunk(sf_vec_size).layout,
        sfa_tile_shape,
        (2, 1),
    )
    mma_tile_inst_k = mma_tiler_mnk[2] // cute.size(tiled_mma.shape_mnk, mode=[2])
    sfa_tile_shape = cute.shape_div(sfa_tile_shape, (1, mma_tile_inst_k))
    smem_layout = cute.tiled_divide(smem_layout, sfa_tile_shape)
    atom_m = 128
    tiler_inst = ((atom_m, sf_vec_size),)
    smem_layout = cute.logical_divide(smem_layout, tiler_inst)
    return cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )


def _make_flexible_smem_layout_sfb(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
) -> cute.Layout:
    """Build the stable SFB hierarchy used by flexible MMA tiles."""
    sfb_tile_shape = (
        cute.round_up(mma_tiler_mnk[1], 128),
        mma_tiler_mnk[2],
    )
    smem_layout = cute.tile_to_shape(
        blockscaled_utils.BlockScaledBasicChunk(sf_vec_size).layout,
        sfb_tile_shape,
        (2, 1),
    )
    mma_tile_inst_k = mma_tiler_mnk[2] // cute.size(tiled_mma.shape_mnk, mode=[2])
    sfb_tile_shape = cute.shape_div(sfb_tile_shape, (1, mma_tile_inst_k))
    smem_layout = cute.tiled_divide(smem_layout, sfb_tile_shape)
    atom_n = 128
    tiler_inst = ((atom_n, sf_vec_size),)
    smem_layout = cute.logical_divide(smem_layout, tiler_inst)
    return cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )


MXFP8_E4M3 = BlockScaledFormatSpec(
    name="mxfp8_e4m3",
    a_dtype=cutlass.Float8E4M3FN,
    b_dtype=cutlass.Float8E4M3FN,
    sf_dtype=cutlass.Float8E8M0FNU,
    sf_vec_size=32,
)


def _format_has_fp4_b(format: BlockScaledFormatSpec) -> bool:
    return format.b_dtype == cutlass.Float4E2M1FN


def _make_blockscaled_tiled_mma(
    *,
    a_dtype: type[cutlass.Numeric],
    b_dtype: type[cutlass.Numeric],
    a_major_mode: tcgen05.OperandMajorMode,
    b_major_mode: tcgen05.OperandMajorMode,
    sf_dtype: type[cutlass.Numeric],
    sf_vec_size: int,
    cta_group: tcgen05.CtaGroup,
    mma_tiler_mn: tuple[int, int],
    use_sm103_ultra: bool,
) -> cute.TiledMma:
    if use_sm103_ultra:
        if a_dtype != b_dtype:
            raise TypeError("SM103 ultra MMA requires matching A/B dtypes")
        return sm103.make_tiled_mma(
            sf_dtype,
            sf_vec_size,
            cta_group,
            mma_tiler_mn,
        )
    return sm100_utils.make_blockscaled_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        a_major_mode,
        b_major_mode,
        sf_dtype,
        sf_vec_size,
        cta_group,
        mma_tiler_mn,
    )


class BlockScaledGroupedGemmKernel:
    """Persistent SM100 block-scaled grouped GEMM (FPROP).

    Host-side ``__call__`` builds SharedStorage + 5 TMA atoms (A, B, SFA,
    SFB, C) and launches; the ``@cute.kernel`` device entry runs the
    warp-specialized loop. ``problem_type`` is a constexpr so DGRAD/WGRAD
    stubs share plumbing with ``GroupedGemmKernel``.
    """

    # 6 active warps + 2 idle per CTA:
    #   0-3  epilog (TMEM → reg → SMEM → TMA store)
    #   4    MMA (tcgen05.mma + s2t SF copy)
    #   5    TMA (SFA/SFB then A/B onto ab_full_mbar)
    #   6-7  idle (cluster prologue arrive only)
    EPILOG_WARP_IDS: tuple[int, int, int, int] = (0, 1, 2, 3)
    MMA_WARP_ID: int = 4
    TMA_AB_WARP_ID: int = 5
    IDLE_WARP_IDS: tuple[int, int] = (6, 7)
    TOTAL_WARPS: int = 8
    THREADS_PER_CTA: int = 32 * TOTAL_WARPS
    # SM103 ultra SF ring depth in K-tiles. One tile (four segment slots)
    # suffices when the MMA warp serves a single problem; the mega kernels
    # can raise it so SF prefetch runs K-tiles ahead of consumption (a
    # 2-tile ring measured perf-neutral so far, so every kernel currently
    # runs the default of 1).
    SM103_SF_RING_TILES: int = 1

    def __init__(
        self,
        *,
        # ---- Problem config ----------------------------------------------
        problem_type: int = _FPROP,
        format: BlockScaledFormatSpec = MXFP8_E4M3,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        force_n_major: bool = False,
        num_n_clusters: int = 1,
        world_size: int = 1,
        # ---- Tile shape --------------------------------------------------
        num_ctas: int,
        block_m: int,
        block_n: int,
        block_k: int,
        mma_atom_n: int | None = None,
        num_mmas: int = 1,
        # ---- Pipeline depths --------------------------------------------
        num_smem_buffers: int,
        num_tmem_buffers: int,
        num_tile_buffers: int,
        num_c_stages: int | None = None,
        # ---- Epilogue + scheduling --------------------------------------
        epilogue_subtile: int = 0,
        epilogue_tile_full_width: bool = False,
        swap_ab: bool = False,
        overlapping_accum: bool = False,
        static_scheduler: bool = False,
        kloop_unroll: int = 2,
        num_warps: int | None = None,
        use_sm103_ultra: bool = False,
    ):
        self.problem_type: int = problem_type
        self.format = format
        self.acc_dtype = acc_dtype
        self.force_n_major = force_n_major
        self.num_n_clusters = num_n_clusters
        self.world_size = world_size

        # SF dtype + vec-size are frozen at construction.
        self.sf_dtype: type[cutlass.Numeric] = format.sf_dtype
        self.sf_vec_size: int = format.sf_vec_size

        # ---- Tile shape constexpr state ---------------------------------
        self.NUM_CTAS: int = num_ctas
        self.NUM_MMAS: int = num_mmas
        self.BLOCK_SIZE_M: int = block_m
        self.BLOCK_SIZE_N: int = block_n
        self.BLOCK_SIZE_K: int = block_k
        self._mma_atom_n: int | None = mma_atom_n

        # ---- Pipeline-depth constexpr state -----------------------------
        self.NUM_SMEM_BUFFERS: int = num_smem_buffers
        self.NUM_TMEM_BUFFERS: int = num_tmem_buffers
        self.NUM_TILE_BUFFERS: int = num_tile_buffers
        self.NUM_C_STAGES: int = (
            num_c_stages if num_c_stages is not None else num_tmem_buffers
        )

        # ---- Epilogue + scheduling constexpr state ----------------------
        self.EPILOGUE_SUBTILE: int = epilogue_subtile
        self.EPILOGUE_TILE_FULL_WIDTH: bool = epilogue_tile_full_width
        # Ping-pong pair of CTA-sync barriers for the cluster DSMEM
        # tile-id handoff in 2CTA mode; 1CTA needs no rendezvous.
        self.NUM_TILE_CTA_BARS: int = 2 if num_ctas == 2 else 0
        self.SWAP_AB: bool = swap_ab
        self.OVERLAPPING_ACCUM: bool = overlapping_accum
        self.STATIC_SCHEDULER: bool = static_scheduler
        self.KLOOP_UNROLL: int = kloop_unroll
        self.USE_SM103_ULTRA: bool = use_sm103_ultra
        self.TOTAL_WARPS = num_warps if num_warps is not None else self.TOTAL_WARPS
        self.THREADS_PER_CTA = 32 * self.TOTAL_WARPS
        if self.USE_SM103_ULTRA:
            if getattr(self.format, "name", "").lower() not in ("nvfp4", "mxfp4"):
                raise ValueError("SM103 ultra MMA supports only NVFP4 or MXFP4")
            if self.problem_type == _WGRAD:
                raise ValueError("SM103 ultra MMA does not support WGRAD")
            if (
                self.NUM_CTAS,
                self.NUM_MMAS,
                block_m,
                block_n,
                block_k,
                self.NUM_SMEM_BUFFERS,
            ) != (
                2,
                1,
                256,
                256,
                sm103.SM103_TILE_K,
                sm103.SM103_AB_PIPELINE_STAGES,
            ):
                raise ValueError(
                    "SM103 ultra MMA requires a 2CTA 256x256x768 tile, one "
                    "MMA, and five A/B shared-memory stages"
                )
        mma_instruction_k = (
            sm103.SM103_MMA_K
            if self.USE_SM103_ULTRA
            else (64 if self.format.a_dtype.width == 4 else 32)
        )
        if self.BLOCK_SIZE_K % mma_instruction_k != 0:
            raise ValueError(
                f"BLOCK_SIZE_K={self.BLOCK_SIZE_K} must be divisible by "
                f"MMA K={mma_instruction_k}"
            )
        # SF TMEM columns follow the cutlass invariant of 4 mma-atom-K
        # instructions per BLOCK_K on the legacy formats (see
        # ``derive_blockscaled_tmem_cols``): the SF TMEM layout produced by
        # the tiled-mma partition keeps that shape even when BLOCK_K holds
        # more K instructions, so recomputing the column count from
        # BLOCK_K // mma_instruction_k mis-places the accumulator/SF seam
        # and costs ~6% on the A8W4 mega prefill. Only the SM103 ultra
        # path, whose segmented SF staging really consumes one column set
        # per K=96 instruction, uses the exact quotient.
        self._mma_inst_tile_k = (
            self.BLOCK_SIZE_K // mma_instruction_k
            if self.USE_SM103_ULTRA
            else MMA_INST_TILE_K
        )

        # SF TMEM-column geometry for the overlap layout + early-release.
        sf_atom_mn = 32
        self._num_sfa_tmem_cols = (
            self.BLOCK_SIZE_M // sf_atom_mn
        ) * self._mma_inst_tile_k
        self._num_sfb_tmem_cols = (
            self.BLOCK_SIZE_N // sf_atom_mn
        ) * self._mma_inst_tile_k
        self._num_sf_tmem_cols = self._num_sfa_tmem_cols + self._num_sfb_tmem_cols

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.NUM_CTAS == 2 else tcgen05.CtaGroup.ONE
        )
        self.cluster_shape_mn: tuple[int, int] = (
            (2, 1) if self.NUM_CTAS == 2 else (1, 1)
        )
        self.mma_tiler_mn: tuple[int, int] = (
            self.BLOCK_SIZE_M,
            self.BLOCK_SIZE_N,
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        problem_type: int = _FPROP,
        format: BlockScaledFormatSpec = MXFP8_E4M3,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        force_n_major: bool = False,
        num_n_clusters: int = 1,
        world_size: int = 1,
    ) -> "BlockScaledGroupedGemmKernel":
        """Build a kernel from a config dict (``auto_blockscaled_config`` /
        ``build_blockscaled_config`` output)."""
        return cls(
            problem_type=problem_type,
            format=format,
            acc_dtype=acc_dtype,
            force_n_major=force_n_major,
            num_n_clusters=num_n_clusters,
            world_size=world_size,
            num_ctas=config["NUM_CTAS"],
            num_mmas=config["NUM_MMAS"],
            block_m=config["BLOCK_SIZE_M"],
            block_n=config["BLOCK_SIZE_N"],
            block_k=config["BLOCK_SIZE_K"],
            mma_atom_n=config.get("MMA_ATOM_N"),
            num_smem_buffers=config["NUM_SMEM_BUFFERS"],
            num_tmem_buffers=config["NUM_TMEM_BUFFERS"],
            num_tile_buffers=config["NUM_TILE_BUFFERS"],
            num_c_stages=config.get("NUM_C_STAGES"),
            epilogue_subtile=config.get("EPILOGUE_SUBTILE", EPILOGUE_SUBTILE_AUTO),
            epilogue_tile_full_width=config.get("EPILOGUE_TILE_FULL_WIDTH", False),
            swap_ab=config.get("SWAP_AB", False),
            overlapping_accum=config.get("OVERLAPPING_ACCUM", False),
            static_scheduler=config.get("STATIC_SCHEDULER", False),
            kloop_unroll=config.get("KLOOP_UNROLL", 2),
            num_warps=config.get("NUM_WARPS"),
            use_sm103_ultra=config.get("USE_SM103_ULTRA", False),
        )

    def _setup_attributes(self):
        self._needs_unpack_tma = self.a_dtype.width != self.b_dtype.width
        self.smem_alloc_a_dtype = (
            cutlass.Uint8
            if self._needs_unpack_tma and self.a_dtype.width < 8
            else self.a_dtype
        )
        self.smem_alloc_b_dtype = (
            cutlass.Uint8
            if self._needs_unpack_tma and self.b_dtype.width < 8
            else self.b_dtype
        )
        atom_m = self.BLOCK_SIZE_M // self.NUM_MMAS
        # ``SM100_MMA_MXF4_SS`` accepts any N in [8, 256] that is a multiple
        # of 8, so a BN=256 FP4 tile is a single MMA atom. The historical
        # atom-N <= 128 cap (dispatching BN=256 as two 128-wide MMA-N atoms
        # with per-N-half ``tcgen05.Field.SFB`` rebinding) existed only
        # because the vendored 4.4.2 SF SMEM hierarchy keeps the N-atom
        # factor as a separate MMA_N mode, which no longer matches the
        # single-atom CTA V-map on 4.5+.
        atom_n = self.BLOCK_SIZE_N
        if _format_has_fp4_b(self.format) and not self.USE_SM103_ULTRA:
            atom_n = (
                self._mma_atom_n
                if self._mma_atom_n is not None
                else min(128, self.BLOCK_SIZE_N)
            )
        if atom_n <= 0 or self.BLOCK_SIZE_N % atom_n != 0:
            raise ValueError("MMA_ATOM_N must be a positive divisor of BLOCK_SIZE_N")
        self.MMA_ATOM_N: int = atom_n
        atom_mma_tiler_mn = (atom_m, atom_n)
        atom_mma_tiler_mn_sfb = (
            atom_m // (2 if self.NUM_CTAS == 2 else 1),
            cute.round_up(atom_n, 128),
        )

        self.tiled_mma = _make_blockscaled_tiled_mma(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            a_major_mode=self.a_major_mode,
            b_major_mode=self.b_major_mode,
            sf_dtype=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
            cta_group=self.cta_group,
            mma_tiler_mn=atom_mma_tiler_mn,
            use_sm103_ultra=self.USE_SM103_ULTRA,
        )
        # SFB uses cta_group=ONE so the SFB partition's per-CTA M
        # projection matches the cutlass reference.
        self.tiled_mma_sfb = _make_blockscaled_tiled_mma(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            a_major_mode=self.a_major_mode,
            b_major_mode=self.b_major_mode,
            sf_dtype=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
            cta_group=tcgen05.CtaGroup.ONE,
            mma_tiler_mn=atom_mma_tiler_mn_sfb,
            use_sm103_ultra=self.USE_SM103_ULTRA,
        )

        self.mma_tiler = (*self.mma_tiler_mn, self.BLOCK_SIZE_K)
        self.mma_tiler_sfb = (
            atom_mma_tiler_mn_sfb[0],
            max(self.BLOCK_SIZE_N, atom_mma_tiler_mn_sfb[1]),
            self.BLOCK_SIZE_K,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(self.tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self.tiled_mma_sfb.thr_id.shape,),
        )

        if self.EPILOGUE_SUBTILE != EPILOGUE_SUBTILE_AUTO:
            cta_m = self.cta_tile_shape_mnk[0]
            cta_n = self.cta_tile_shape_mnk[1]
            warp_m, warp_n = (2, 2) if (cta_m == 64 and self.NUM_CTAS == 2) else (4, 1)
            epi_subtile = blockscaled_epilogue_subtile_divisor(
                epilogue_subtile=self.EPILOGUE_SUBTILE,
                c_width=self.c_dtype.width,
                a_width=self.a_dtype.width,
                full_width=self.EPILOGUE_TILE_FULL_WIDTH,
            )
            tile_m = min(cta_m, 32 * warp_m)
            tile_n = cta_n // epi_subtile
            tile_m_layout = cute.make_layout(tile_m)
            tile_n_layout = cute.make_layout(
                (tile_n // warp_n, warp_n), stride=(1, cta_n // warp_n)
            )
            self.epi_tile = (tile_m_layout, cute.coalesce(tile_n_layout))
        else:
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.NUM_CTAS == 2,
                self.c_layout,
                self.c_dtype,
            )

        if self.USE_SM103_ULTRA:
            self.a_smem_layout_staged = sm103.make_smem_layout_ab(
                self.tiled_mma,
                self.mma_tiler,
                self.NUM_SMEM_BUFFERS,
                is_a=True,
            )
            self.b_smem_layout_staged = sm103.make_smem_layout_ab(
                self.tiled_mma,
                self.mma_tiler,
                self.NUM_SMEM_BUFFERS,
                is_a=False,
            )
            self.sfa_smem_layout_staged = sm103.make_smem_layout_sfa(
                self.tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                self.SM103_SF_RING_TILES * sm103.sf_segments(self.sf_vec_size),
            )
            self.sfb_smem_layout_staged = sm103.make_smem_layout_sfb(
                self.mma_tiler,
                self.sf_vec_size,
                self.SM103_SF_RING_TILES * sm103.sf_segments(self.sf_vec_size),
            )
            self.sfb_tma_smem_layout_staged = self.sfb_smem_layout_staged
        else:
            self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
                self.tiled_mma,
                self.mma_tiler,
                self.smem_alloc_a_dtype,
                self.NUM_SMEM_BUFFERS,
            )
            self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                self.tiled_mma,
                self.mma_tiler,
                self.smem_alloc_b_dtype,
                self.NUM_SMEM_BUFFERS,
            )
            if self.BLOCK_SIZE_N == 64 or self.BLOCK_SIZE_N == self.MMA_ATOM_N:
                self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                    self.tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    self.NUM_SMEM_BUFFERS,
                )
                self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                    self.tiled_mma_sfb if self.BLOCK_SIZE_N == 32 else self.tiled_mma,
                    self.mma_tiler_sfb if self.BLOCK_SIZE_N == 32 else self.mma_tiler,
                    self.sf_vec_size,
                    self.NUM_SMEM_BUFFERS,
                )
            else:
                self.sfa_smem_layout_staged = _make_flexible_smem_layout_sfa(
                    self.tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    self.NUM_SMEM_BUFFERS,
                )
                self.sfb_smem_layout_staged = _make_flexible_smem_layout_sfb(
                    self.tiled_mma_sfb if self.BLOCK_SIZE_N == 32 else self.tiled_mma,
                    self.mma_tiler_sfb if self.BLOCK_SIZE_N == 32 else self.mma_tiler,
                    self.sf_vec_size,
                    self.NUM_SMEM_BUFFERS,
                )
            self.sfb_tma_smem_layout_staged = self._make_sfb_tma_smem_layout_staged(
                self.sfb_smem_layout_staged
            )
        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.NUM_C_STAGES
        )

        # Block-scaled SFA/SFB live in TMEM after the accumulator, so the
        # acc-only allocation size is not enough — the s2t SF copy would
        # write OOB. Always grab the full 512 TMEM cols. (Bug: BN=256 IMA
        # — at BN=128 the SFB copy happened to fit; at BN=256 it didn't.)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

    @property
    def _a_global_scale_inv_smem_size(self) -> int:
        return self.BLOCK_SIZE_N if self.SWAP_AB else self.cta_tile_shape_mnk[0]

    def _make_shared_storage(
        self,
        a_dtype,
        b_dtype,
        c_dtype,
        sf_dtype,
        G: int,
        use_a_global_scale_inv: bool,
    ):
        NUM_SMEM = self.NUM_SMEM_BUFFERS
        NUM_TMEM = self.NUM_TMEM_BUFFERS
        NUM_TILE = self.NUM_TILE_BUFFERS
        NUM_CTAS = self.NUM_CTAS

        a_smem_elems = cute.cosize(self.a_smem_layout_staged.outer)
        b_smem_elems = cute.cosize(self.b_smem_layout_staged.outer)
        c_smem_elems = cute.cosize(self.epi_smem_layout_staged.outer)
        sfa_smem_elems = cute.cosize(self.sfa_smem_layout_staged)
        sfb_smem_elems = cute.cosize(self.sfb_smem_layout_staged)

        n_smem_empty = NUM_SMEM
        n_smem_full = NUM_SMEM
        n_sf_smem = (
            self.SM103_SF_RING_TILES * sm103.sf_segments(self.sf_vec_size)
            if self.USE_SM103_ULTRA
            else 0
        )
        n_tmem_full = NUM_TMEM
        n_tmem_empty = NUM_TMEM
        n_tile_consumer = NUM_TILE
        n_tile_producer = NUM_TILE
        n_tile_cta_bar = self.NUM_TILE_CTA_BARS
        n_tmem_dealloc = 1 if NUM_CTAS == 2 else 0
        # OVERLAPPING_ACCUM shares a TMEM seam across acc stages, so the
        # per-stage tmem_empty bar can't gate the cross-stage hazard.
        # cross_seam_mbar is the shared bar epi arrives on after the
        # early-release drain; next round's MMA waits on it.
        n_cross_seam = 1 if self.OVERLAPPING_ACCUM else 0
        n_tensormap_buffer = max(
            _BLOCKSCALED_TENSORMAP_STAGING_INT64S,
            (G + 1) // 2,
        )
        n_a_global_scale_inv = (
            self._a_global_scale_inv_smem_size if use_a_global_scale_inv else 0
        )
        a_smem_dtype = cutlass.Uint8 if self.USE_SM103_ULTRA else a_dtype
        b_smem_dtype = cutlass.Uint8 if self.USE_SM103_ULTRA else b_dtype

        @cute.struct
        class SharedStorage:
            sC: cute.struct.Align[cute.struct.MemRange[c_dtype, c_smem_elems], 1024]
            sA: cute.struct.Align[
                cute.struct.MemRange[a_smem_dtype, a_smem_elems], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[b_smem_dtype, b_smem_elems], 1024
            ]
            # SF SMEM — 1 byte per ``sf_vec_size`` operand elements.
            # Layout built by ``blockscaled_utils.make_smem_layout_sf{a,b}``.
            sSFA: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfa_smem_elems], 1024
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[sf_dtype, sfb_smem_elems], 1024
            ]
            a_global_scale_inv: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, n_a_global_scale_inv], 128
            ]
            tile_id_smem: cute.struct.MemRange[cutlass.Int32, NUM_TILE]
            # Prepare uses five 128-byte A/B/SFA/SFB/C descriptors. The main
            # kernel reuses this range for the staged expert metadata cache.
            tensormap_buffer: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, n_tensormap_buffer], 128
            ]

            smem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_smem_empty]
            smem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_smem_full]
            sf_smem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_sf_smem]
            sf_smem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_sf_smem]
            tmem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_full]
            tmem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_empty]
            tile_id_consumer_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_consumer]
            tile_id_producer_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_producer]
            tile_cta_bar_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_cta_bar]
            tmem_dealloc_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_dealloc]
            cross_seam_mbar: cute.struct.MemRange[cutlass.Int64, n_cross_seam]

            tmem_holding_buf: cutlass.Int32

        return SharedStorage

    def _make_prepare_shared_storage(self):
        @cute.struct
        class PrepareSharedStorage:
            tensormap_buffer: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int64, _BLOCKSCALED_TENSORMAP_STAGING_INT64S
                ],
                128,
            ]

        return PrepareSharedStorage

    @cute.kernel
    def prepare_kernel(  # noqa: C901
        self,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tma_atom_sfa: cute.CopyAtom,
        tma_atom_sfb: cute.CopyAtom,
        tma_atom_c: cute.CopyAtom,
        split_sizes: cute.Tensor,
        counter_ptr: cute.Pointer,
        tensormaps: cute.Tensor,
        pointer_strides: BlockscaledPointerStrideArgs,
        elem_sizes: cutlass.Constexpr[tuple[int, int, int]],
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        extra_counter_ptr: cute.Pointer,
        extra_counter_count: cutlass.Int32,
        zero_extra_counters: cutlass.Constexpr[bool],
        activation_buffer_base_ptr: cutlass.Int64,
        activation_buffer_size_bytes: cutlass.Int64,
        activation_offsets: cute.Tensor,
        use_activation_buffer: cutlass.Constexpr[bool],
    ):
        base_ptrs, tensor_strides, sf_strides = pointer_strides
        activation_buffer_operand_offsets = (
            cutlass.Int64(MISSING_ACTIVATION_OFFSET),
            cutlass.Int64(MISSING_ACTIVATION_OFFSET),
            cutlass.Int64(MISSING_ACTIVATION_OFFSET),
            cutlass.Int64(MISSING_ACTIVATION_OFFSET),
            cutlass.Int64(MISSING_ACTIVATION_OFFSET),
        )
        if cutlass.const_expr(use_activation_buffer):
            activation_buffer_operand_offsets = _activation_buffer_operand_offsets(
                activation_offsets,
                0,
                self.SWAP_AB,
            )
            pointer_strides = _activation_buffer_pointer_strides(
                pointer_strides,
                activation_buffer_base_ptr,
                activation_offsets,
                0,
                self.SWAP_AB,
            )
            base_ptrs, tensor_strides, sf_strides = pointer_strides
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        prepare_g = cute.arch.block_idx()[0]

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.prepare_shared_storage)
        tensormap_buffer_ptr = storage.tensormap_buffer.data_ptr()

        tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)

        if prepare_g == 0 and tidx == cutlass.Int32(0):
            cute.arch.store(
                counter_ptr,
                cutlass.Int32(0),
            )

        if cutlass.const_expr(zero_extra_counters):
            extra_counters = cute.make_tensor(
                extra_counter_ptr,
                cute.make_ordered_layout((extra_counter_count,), order=(0,)),
            )
            block_dim_x, _, _ = cute.arch.block_dim()
            counter_zero_thread_idx = prepare_g * block_dim_x + tidx
            counter_zero_threads = cutlass.Int32(G) * block_dim_x
            zero_iters = (
                extra_counter_count + counter_zero_threads - cutlass.Int32(1)
            ) // counter_zero_threads
            for i in cutlass.range(zero_iters, unroll=1):
                counter_idx = i * counter_zero_threads + counter_zero_thread_idx
                if counter_idx < extra_counter_count:
                    extra_counters[counter_idx] = cutlass.Int32(0)

        params = BlockscaledTensormapParams(
            tma_atom_a,
            tma_atom_b,
            tma_atom_sfa,
            tma_atom_sfb,
            tma_atom_c,
            base_ptrs=base_ptrs,
            strides=tensor_strides,
            sf_strides=sf_strides,
            elem_sizes=elem_sizes,
            dtypes=(self.a_dtype, self.b_dtype, self.c_dtype),
        )
        problem = BlockscaledTensormapProblem(
            split_sizes=split_sizes,
            groups=G,
            mnk=(M, N, K),
            problem_type=self.problem_type,
            tensormap_base=0,
            prepare_g=prepare_g,
            warp_idx=warp_idx,
        )
        _prepare_blockscaled_problem_tensormaps(
            params=params,
            problem=problem,
            tensormaps=tensormaps,
            tensormap_manager=tensormap_manager,
            tensormap_smem_ptr_base=tensormap_buffer_ptr,
            activation_buffer_operand_offsets=activation_buffer_operand_offsets,
            activation_buffer_size_bytes=activation_buffer_size_bytes,
            use_activation_buffer=use_activation_buffer,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            SWAP_AB=self.SWAP_AB,
            sf_dtype=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
            use_sm103_ultra=self.USE_SM103_ULTRA,
        )

    @cute.jit
    def __call__(
        self,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_c: cute.Tensor,
        tensor_sfa: cute.Tensor,
        tensor_sfb: cute.Tensor,
        split_sizes: cute.Tensor,
        counter: cute.Tensor,
        tensormaps: cute.Tensor,
        pointer_strides: BlockscaledPointerStrideArgs,
        elem_sizes: cutlass.Constexpr[tuple[int, int, int]],
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        num_clusters: int,
        output_accum: cutlass.Constexpr[bool],
        use_device_tensormaps: cutlass.Constexpr[bool],
        activation_buffer_base_ptr: cutlass.Int64,
        activation_buffer_size_bytes: cutlass.Int64,
        activation_offsets: cute.Tensor,
        use_activation_buffer: cutlass.Constexpr[bool],
        a_global_scale_inv_ptr: cutlass.Int64,
        b_global_scale_inv_ptr: cutlass.Int64,
        use_a_global_scale_inv: cutlass.Constexpr[bool],
        use_b_global_scale_inv: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        eff_tensor_a, eff_tensor_b = _swap_if(self.SWAP_AB, tensor_a, tensor_b)
        eff_tensor_sfa, eff_tensor_sfb = _swap_if(self.SWAP_AB, tensor_sfa, tensor_sfb)
        eff_pointer_strides = swap_ab_pointer_strides(pointer_strides, self.SWAP_AB)
        elem_size_bytes_a, elem_size_bytes_b, elem_size_bytes_c = elem_sizes
        eff_elem_a, eff_elem_b = _swap_if(
            self.SWAP_AB,
            elem_size_bytes_a,
            elem_size_bytes_b,
        )
        eff_elem_sizes = (eff_elem_a, eff_elem_b, elem_size_bytes_c)

        # Under SWAP_AB, build the C TMA atom on a transposed Y so the
        # accumulator's (m_tile, n_tile) coords write the right cells.
        # ``c_layout`` must derive from ``tensor_c_eff`` or epi SMEM
        # staging disagrees with the GMEM TMA atom -> garbage.
        tensor_c_eff = _transpose_c_if_swap(tensor_c, self.SWAP_AB)

        self.a_dtype = eff_tensor_a.element_type
        self.b_dtype = eff_tensor_b.element_type
        self.c_dtype = tensor_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(eff_tensor_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(eff_tensor_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(tensor_c_eff)

        # FP4/FP8 always run K-major operands in FPROP.
        self._setup_attributes()

        # SF tensors in the atom-tile layout for TMA atom construction.
        # ``tensor_{a,b}.shape`` is in logical-element units (the host
        # passes the FP4 fake-tensor with logical K so this helper sees
        # the right K regardless of operand precision).
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            sfa_layout = sm103.make_gmem_layout_sf(eff_tensor_a.shape, self.sf_vec_size)
        else:
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                eff_tensor_a.shape, self.sf_vec_size
            )
        tensor_sfa_view = cute.make_tensor(eff_tensor_sfa.iterator, sfa_layout)
        sfb_data_shape = eff_tensor_b.shape
        if self.SWAP_AB and self.BLOCK_SIZE_N in (32, 64):
            scale_storage_factor = 4 if self.BLOCK_SIZE_N == 32 else 2
            sfb_data_shape = (
                eff_tensor_b.shape[0] * scale_storage_factor,
                eff_tensor_b.shape[1],
                eff_tensor_b.shape[2],
            )
        elif self.SWAP_AB and self.BLOCK_SIZE_N >= 96 and self.BLOCK_SIZE_N % 128 != 0:
            scale_page_rows = ((self.BLOCK_SIZE_N + 127) // 128) * 128
            sfb_data_shape = (
                eff_tensor_b.shape[0] // self.BLOCK_SIZE_N * scale_page_rows,
                eff_tensor_b.shape[1],
                eff_tensor_b.shape[2],
            )
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            sfb_layout = sm103.make_gmem_layout_sf(sfb_data_shape, self.sf_vec_size)
        else:
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
                sfb_data_shape, self.sf_vec_size
            )
        tensor_sfb_view = cute.make_tensor(eff_tensor_sfb.iterator, sfb_layout)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
                a_op,
                cute.recast_tensor(eff_tensor_a, cutlass.Uint8),
                sm103.adapt_layout_for_tma_ab(
                    sm103.make_smem_layout_ab(
                        self.tiled_mma,
                        self.mma_tiler,
                        sm103.SM103_AB_SEGMENTS,
                        is_a=True,
                    )
                ),
                (cute.size(self.tiled_mma.tv_layout_A[1][0]), 384),
                self.cluster_shape_mn[1],
                internal_type=cutlass.Uint8,
            )
            tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
                b_op,
                cute.recast_tensor(eff_tensor_b, cutlass.Uint8),
                sm103.adapt_layout_for_tma_ab(
                    sm103.make_smem_layout_ab(
                        self.tiled_mma,
                        self.mma_tiler,
                        sm103.SM103_AB_SEGMENTS,
                        is_a=False,
                    )
                ),
                (cute.size(self.tiled_mma.tv_layout_B[1][0]), 384),
                self.cluster_shape_mn[0] // cute.size(self.tiled_mma.thr_id.shape),
                internal_type=cutlass.Uint8,
            )
        else:
            a_smem_layout = cute.slice_(
                self.a_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                eff_tensor_a,
                a_smem_layout,
                self.mma_tiler,
                self.tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=(
                    self.smem_alloc_a_dtype
                    if self._needs_unpack_tma and self.a_dtype.width < 8
                    else None
                ),
            )
            b_smem_layout = cute.slice_(
                self.b_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                eff_tensor_b,
                b_smem_layout,
                self.mma_tiler,
                self.tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=(
                    self.smem_alloc_b_dtype
                    if self._needs_unpack_tma and self.b_dtype.width < 8
                    else None
                ),
            )

        # SFA/SFB TMA atoms — ``internal_type=Int16`` because blockscaled
        # SFs are packed that way for cp.async.bulk (cutlass reference).
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            mma_sf_tiler = sm103.mma_sf_tiler(self.cta_tile_shape_mnk, self.sf_vec_size)
            tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(
                sfa_op,
                tensor_sfa_view,
                sm103.adapt_layout_for_tma_sf(
                    cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
                ),
                (mma_sf_tiler[0], mma_sf_tiler[2]),
                self.cluster_shape_mn[1],
                internal_type=cutlass.Uint8,
            )
            tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
                sfb_op,
                tensor_sfb_view,
                sm103.adapt_layout_for_tma_sf(
                    cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
                ),
                (mma_sf_tiler[1], mma_sf_tiler[2]),
                self.cluster_shape_mn[0] // cute.size(self.tiled_mma_sfb.thr_id.shape),
                internal_type=cutlass.Uint8,
            )
        else:
            sfa_smem_layout = cute.slice_(
                self.sfa_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                tensor_sfa_view,
                sfa_smem_layout,
                self.mma_tiler,
                self.tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            sfb_smem_layout = cute.slice_(
                self.sfb_tma_smem_layout_staged, (None, None, None, 0)
            )
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                tensor_sfb_view,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                self.tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        # ``tensor_c_eff`` is built up top so its layout matches the
        # SMEM staging built downstream.
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(tensor_c_eff.shape), self.epi_tile
        )
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp()
            if output_accum
            else cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c_eff,
            epi_smem_layout,
            c_cta_v_layout,
        )

        self.shared_storage = self._make_shared_storage(
            self.smem_alloc_a_dtype,
            self.smem_alloc_b_dtype,
            self.c_dtype,
            self.sf_dtype,
            G,
            use_a_global_scale_inv,
        )
        self.prepare_shared_storage = self._make_prepare_shared_storage()

        grid = (num_clusters * self.NUM_CTAS, 1, 1)

        if cutlass.const_expr(use_device_tensormaps):
            self.prepare_kernel(
                tma_atom_a=tma_atom_a,
                tma_atom_b=tma_atom_b,
                tma_atom_sfa=tma_atom_sfa,
                tma_atom_sfb=tma_atom_sfb,
                tma_atom_c=tma_atom_c,
                split_sizes=split_sizes,
                counter_ptr=counter.iterator,
                tensormaps=tensormaps,
                pointer_strides=eff_pointer_strides,
                elem_sizes=eff_elem_sizes,
                G=G,
                M=M,
                N=N,
                K=K,
                extra_counter_ptr=counter.iterator,
                extra_counter_count=cutlass.Int32(0),
                zero_extra_counters=False,
                activation_buffer_base_ptr=activation_buffer_base_ptr,
                activation_buffer_size_bytes=activation_buffer_size_bytes,
                activation_offsets=activation_offsets,
                use_activation_buffer=use_activation_buffer,
            ).launch(
                grid=(G, 1, 1),
                block=(32, 1, 1),
                stream=stream,
            )

        self.kernel(
            tma_atom_a=tma_atom_a,
            tma_atom_b=tma_atom_b,
            tma_atom_sfa=tma_atom_sfa,
            tma_atom_sfb=tma_atom_sfb,
            tma_atom_c=tma_atom_c,
            mA=tma_tensor_a,
            mB=tma_tensor_b,
            mSFA=tma_tensor_sfa,
            mSFB=tma_tensor_sfb,
            mC=tma_tensor_c,
            a_smem_layout_staged=self.a_smem_layout_staged,
            b_smem_layout_staged=self.b_smem_layout_staged,
            sfa_smem_layout_staged=self.sfa_smem_layout_staged,
            sfb_smem_layout_staged=self.sfb_smem_layout_staged,
            sfb_tma_smem_layout_staged=self.sfb_tma_smem_layout_staged,
            epi_smem_layout_staged=self.epi_smem_layout_staged,
            epi_tile=self.epi_tile,
            cluster_layout_vmnk=self.cluster_layout_vmnk,
            cluster_layout_sfb_vmnk=self.cluster_layout_sfb_vmnk,
            tiled_mma=self.tiled_mma,
            tiled_mma_sfb=self.tiled_mma_sfb,
            split_sizes=split_sizes,
            counter_ptr=counter.iterator,
            tensormaps=tensormaps,
            G=G,
            M=M,
            N=N,
            K=K,
            local_rank=local_rank,
            use_device_tensormaps=use_device_tensormaps,
            a_global_scale_inv_ptr=a_global_scale_inv_ptr,
            b_global_scale_inv_ptr=b_global_scale_inv_ptr,
            use_a_global_scale_inv=use_a_global_scale_inv,
            use_b_global_scale_inv=use_b_global_scale_inv,
        ).launch(
            grid=grid,
            block=(self.THREADS_PER_CTA, 1, 1),
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        # ---- TMA atoms / global tensors ---------------------------------
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tma_atom_sfa: cute.CopyAtom,
        tma_atom_sfb: cute.CopyAtom,
        tma_atom_c: cute.CopyAtom,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mSFA: cute.Tensor,
        mSFB: cute.Tensor,
        mC: cute.Tensor,
        # ---- Layouts -----------------------------------------------------
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        sfb_tma_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        # ---- Group + scheduler inputs -----------------------------------
        split_sizes: cute.Tensor,
        counter_ptr: cute.Pointer,
        tensormaps: cute.Tensor,
        # ---- Problem dimensions -----------------------------------------
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        use_device_tensormaps: cutlass.Constexpr[bool],
        a_global_scale_inv_ptr: cutlass.Int64,
        b_global_scale_inv_ptr: cutlass.Int64,
        use_a_global_scale_inv: cutlass.Constexpr[bool],
        use_b_global_scale_inv: cutlass.Constexpr[bool],
        # ---- Constexpr flags --------------------------------------------
    ):
        # ----- Warp / cluster identity ----------------------------------
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        cluster_cta_rank = cutlass.Int32(0)
        pred_cta0 = True
        if cutlass.const_expr(self.NUM_CTAS == 2):
            cluster_cta_rank = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            pred_cta0 = cluster_cta_rank == 0

        # ----- SMEM / mbarrier allocation -------------------------------
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
        tile_consumer_mbar = storage.tile_id_consumer_mbar.data_ptr()
        tile_producer_mbar = storage.tile_id_producer_mbar.data_ptr()
        tile_cta_bar_mbar = (
            storage.tile_cta_bar_mbar.data_ptr() if self.NUM_TILE_CTA_BARS > 0 else None
        )
        tmem_dealloc_mbar = (
            storage.tmem_dealloc_mbar.data_ptr() if self.NUM_CTAS == 2 else None
        )
        cross_seam_mbar = (
            storage.cross_seam_mbar.data_ptr() if self.OVERLAPPING_ACCUM else None
        )
        a_global_scale_inv_smem_ptr = (
            storage.a_global_scale_inv.data_ptr() if use_a_global_scale_inv else None
        )
        tile_id_smem_ptr = storage.tile_id_smem.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf

        # mbarrier arrive_count mirrors the base kernel: the unified TMA
        # warp arms expect_tx with the combined A+B+SF byte count, so no
        # extra arrives are needed for the SF half.
        if warp_idx == self.EPILOG_WARP_IDS[0]:
            with cute.arch.elect_one():
                for i in range(self.NUM_SMEM_BUFFERS):
                    cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
                    cute.arch.mbarrier_init(ab_full_mbar + i, 1)
                if cutlass.const_expr(self.USE_SM103_ULTRA):
                    for i in range(
                        self.SM103_SF_RING_TILES * sm103.sf_segments(self.sf_vec_size)
                    ):
                        cute.arch.mbarrier_init(sf_empty_mbar + i, 1)
                        cute.arch.mbarrier_init(sf_full_mbar + i, 1)
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

        # ----- TMEM allocation (epilogue warps own it) -------------------
        if warp_idx == self.EPILOG_WARP_IDS[0]:
            cute.arch.alloc_tmem(
                self.num_tmem_alloc_cols,
                tmem_holding_buf,
                is_two_cta=(self.NUM_CTAS == 2),
            )

        split_sizes = stage_expert_metadata(
            split_sizes,
            cute.recast_ptr(storage.tensormap_buffer.data_ptr(), dtype=cutlass.Int32),
            G,
            synchronize=False,
        )
        cute.arch.barrier(
            barrier_id=_BAR_FULL_CTA_SYNC, number_of_threads=self.THREADS_PER_CTA
        )

        # Prologue partitioning: A/B identical to the base kernel; SFA
        # rides the A partition (M axis), SFB rides the cta_group=ONE
        # ``tiled_mma_sfb`` so its M partition matches the SFB layout.
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sSFB_tma = storage.sSFB.get_tensor(sfb_tma_smem_layout_staged)

        # SFA/SFB use the same slicing pattern as A/B so we can reuse
        # ``partition_A/B`` from tiled_mma / tiled_mma_sfb.
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            ab_mma_tiler = (
                self.mma_tiler[0],
                self.mma_tiler[1],
                self.mma_tiler[2] // 2,
            )
            mma_sf_tiler = sm103.mma_sf_tiler(self.cta_tile_shape_mnk, self.sf_vec_size)
            gA = cute.local_tile(
                mA,
                cute.slice_(ab_mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            gB = cute.local_tile(
                mB,
                cute.slice_(ab_mma_tiler, (0, None, None)),
                (None, None, None),
            )
            gSFA = cute.local_tile(
                mSFA,
                cute.slice_(mma_sf_tiler, (None, 0, None)),
                (None, None, None),
            )
            gSFB = cute.local_tile(
                mSFB,
                cute.slice_(mma_sf_tiler, (0, None, None)),
                (None, None, None),
            )
        else:
            gA = cute.local_tile(
                mA,
                cute.slice_(self.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            gB = cute.local_tile(
                mB,
                cute.slice_(self.mma_tiler, (0, None, None)),
                (None, None, None),
            )
            gSFA = cute.local_tile(
                mSFA,
                cute.slice_(self.mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            gSFB = cute.local_tile(
                mSFB,
                cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                (None, None, None),
            )
        gC = cute.local_tile(
            mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cluster_cta_rank
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cluster_cta_rank
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
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
                    (cute.size(tiled_mma.tv_layout_A[1][0]), 128),
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
                    (cute.size(tiled_mma.tv_layout_B[1][0]), 128),
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

        # TMA partitioning: A/B identical to base. SFA reuses the A
        # cta_layout; SFB needs the SFB cluster layout for the
        # cta_group=ONE mma_sfb tiler.
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 1),
            )
        else:
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 1),
            )
        else:
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )

        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        # ``filter_zeros`` collapses zero-strided modes that come from SF
        # tensors sharing A's tile layout but with a different atom (the
        # cutlass blockscaled reference does the same).
        tAsSFA = cute.filter_zeros(tAsSFA)
        if cutlass.const_expr(not self.USE_SM103_ULTRA):
            tAgSFA = cute.filter_zeros(tAgSFA)

        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB_tma, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        if cutlass.const_expr(not self.USE_SM103_ULTRA):
            tBgSFB = cute.filter_zeros(tBgSFB)

        # MMA fragments.
        if cutlass.const_expr(self.USE_SM103_ULTRA):
            tCrA = sA
            tCrB = sB
        else:
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((self.BLOCK_SIZE_M, self.BLOCK_SIZE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.NUM_TMEM_BUFFERS)
        )
        # OVERLAPPING_ACCUM packs both acc stages into the seam shared
        # with the SF block by shrinking the stage stride from BLOCK_N
        # to ``BLOCK_N - num_sf_tmem_cols``. Mirrors cutlass dense ref
        # ``dense_blockscaled_gemm_persistent.py:960-977``.
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

        # Multicast masks: A/B as in base kernel. SFA reuses A's mask
        # (same M cohort); SFB needs its own image on the SFB cluster
        # layout.
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        ab_empty_mcast_mask = None
        acc_full_mcast_mask = None
        if cutlass.const_expr(self.NUM_CTAS == 2):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = a_full_mcast_mask
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk,
                block_in_cluster_coord_sfb_vmnk,
                mcast_mode=1,
            )
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1,
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
            )
            ab_empty_mcast_mask = (
                a_full_mcast_mask
                | b_full_mcast_mask
                | a_full_mcast_mask_peer
                | b_full_mcast_mask_peer
            )
            acc_full_mcast_mask = cute.make_layout_image_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0
            )

        (
            num_tma_load_bytes,
            num_tma_load_bytes_ab,
            num_tma_load_bytes_sf,
        ) = self._num_tma_load_bytes(
            sA,
            sB,
            sSFA,
            sSFB,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_tma_smem_layout_staged,
            tiled_mma,
        )

        tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)

        # Prefetch the 5 host-built TMA descriptors into L1 before the
        # producer loops (per flash_fwd.py:1192-1196).
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        problem = GroupedGemmProblem(
            split_sizes=split_sizes,
            groups=G,
            mnk=(M, N, K),
            local_rank=local_rank,
        )
        sync = GroupedGemmPipelineSync(
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
            sf_full_mbar=sf_full_mbar,
            sf_empty_mbar=sf_empty_mbar,
        )

        # =================================================================
        # Warp-group dispatch.
        # =================================================================
        if warp_idx == self.TMA_AB_WARP_ID:
            pipeline = GroupedGemmTmaPipeline(
                tma_atom_a=tma_atom_a,
                tma_atom_b=tma_atom_b,
                tma_atom_sfa=tma_atom_sfa,
                tma_atom_sfb=tma_atom_sfb,
                gA=tAgA,
                gB=tBgB,
                gSFA=tAgSFA,
                gSFB=tBgSFB,
                sA=tAsA,
                sB=tBsB,
                sSFA=tAsSFA,
                sSFB=tBsSFB,
                num_tma_load_bytes=num_tma_load_bytes,
                num_tma_load_bytes_ab=num_tma_load_bytes_ab,
                num_tma_load_bytes_sf=num_tma_load_bytes_sf,
            )
            self._tma_producer_body(
                pipeline=pipeline,
                problem=problem,
                sync=sync,
                tensormap_manager=tensormap_manager,
                tensormaps=tensormaps,
                tile_producer_hook_tensor=split_sizes,
                use_device_tensormaps=use_device_tensormaps,
            )
        elif warp_idx == self.MMA_WARP_ID:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tCtSFA, tCtSFB, tCtSFB_field = self._scale_tmem_tensors(
                tmem_ptr,
                tCtAcc_base,
                tiled_mma,
                self.mma_tiler,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
            )
            tCtSFA_field = tCtSFA
            if cutlass.const_expr(self.USE_SM103_ULTRA):
                sfa_field_layout, sfb_field_layout = sm103.make_sf_tmem_field_layouts(
                    tiled_mma,
                    self.mma_tiler,
                    self.cta_tile_shape_mnk,
                    self.sf_vec_size,
                )
                tCtSFA_field = cute.make_tensor(
                    tCtSFA.iterator,
                    sfa_field_layout,
                )
                tCtSFB_field = cute.make_tensor(
                    tCtSFB.iterator,
                    sfb_field_layout,
                )

            pipeline = GroupedGemmMmaPipeline(
                tiled_mma=tiled_mma,
                tCrA=tCrA,
                tCrB=tCrB,
                tCtAcc_base=tCtAcc_base,
                tCtSFA=tCtSFA_field,
                tCtSFA_copy=tCtSFA,
                tCtSFB=tCtSFB_field,
                tCtSFB_copy=tCtSFB,
                sSFA=sSFA,
                sSFB=sSFB,
            )
            self._mma_consumer_body(pipeline, problem, sync)
        elif warp_idx in self.EPILOG_WARP_IDS:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            (
                tiled_copy_t2r,
                tTR_tAcc_base_00,
                tTR_rAcc,
            ) = self._epilog_tmem_copy_and_partition(
                tidx=tidx,
                tAcc=tCtAcc_base,
                gC_mnl=tCgC,
                epi_tile=epi_tile,
                use_2cta_instrs=self.NUM_CTAS == 2,
                mma_m_idx=0,
                mma_n_idx=0,
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, sC.element_type)
            (
                tiled_copy_r2s,
                tRS_rC,
                tRS_sC,
            ) = self._epilog_smem_copy_and_partition(
                tiled_copy_t2r=tiled_copy_t2r,
                tTR_rC=tTR_rC,
                tidx=tidx,
                sC=sC,
            )
            pipeline = GroupedGemmEpilogPipeline(
                c_atom=tma_atom_c,
                tCtAcc_base=tCtAcc_base,
                c_tensor=tCgC,
                sC=sC,
                epi_tile=epi_tile,
                cta_tile_shape_mnk=self.cta_tile_shape_mnk,
                c_layout=self.c_layout,
                c_dtype=self.c_dtype,
                a_dtype_width=self.a_dtype.width,
            )
            epilog_copy = (tTR_rAcc, tiled_copy_r2s, tRS_rC, tRS_sC)
            self._epilog_consumer_body(
                tidx=tidx,
                pipeline=pipeline,
                epilog_copy=epilog_copy,
                problem=problem,
                sync=sync,
                tensormap_manager=tensormap_manager,
                tensormaps=tensormaps,
                use_device_tensormaps=use_device_tensormaps,
                a_global_scale_inv_smem_ptr=a_global_scale_inv_smem_ptr,
                a_global_scale_inv_ptr=a_global_scale_inv_ptr,
                b_global_scale_inv_ptr=b_global_scale_inv_ptr,
                use_a_global_scale_inv=use_a_global_scale_inv,
                use_b_global_scale_inv=use_b_global_scale_inv,
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

        if cutlass.const_expr(self.NUM_CTAS == 2):
            # Keep every active thread alive until peer DSMEM accesses complete.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    # TMA producer: standard MMA uses one combined A/B/SF barrier per K tile;
    # SM103 rotates each tile's three A/B segments through a five-stage ring
    # and pipelines its four SF segments independently.
    @cute.jit
    def _before_tma_producer_tile(
        self,
        tile_m_idx: cutlass.Int32,
        tile_n_idx: cutlass.Int32,
        act_off_tiles: cutlass.Int32,
        m_size: cutlass.Int32,
        n_size: cutlass.Int32,
        K: cutlass.Int32,
        tile_producer_hook_tensor: cute.Tensor,
    ) -> None:
        pass

    @cute.jit
    def _sm103_tma_load_ab_segment(
        self,
        pipeline: GroupedGemmTmaPipeline,
        sync: GroupedGemmPipelineSync,
        kk: cutlass.Int32,
        ab_segment: cutlass.Constexpr[int],
        ab_buf: cutlass.Int32,
        phase: cutlass.Int32,
        a_axis: cutlass.Int32,
        b_axis: cutlass.Int32,
        group_axes: tuple,
        tma_desc_ptrs: tuple,
        use_device_tensormaps: cutlass.Constexpr[bool],
    ) -> None:
        a_l, b_l, _, _ = group_axes
        a_full_mcast_mask, b_full_mcast_mask, _, _ = sync.tma_mcast_masks
        if cutlass.const_expr(use_device_tensormaps):
            a_desc_ptr, b_desc_ptr, _, _ = tma_desc_ptrs
        _tma_wait_and_arm_expect_tx(
            sync.ab_empty_mbar + ab_buf,
            sync.ab_full_mbar + ab_buf,
            phase,
            sync.cluster_cta_rank,
            num_tma_load_bytes=pipeline.num_tma_load_bytes_ab,
            NUM_CTAS=self.NUM_CTAS,
        )
        gA_segment = cute.group_modes(
            pipeline.gA[(None, None, ab_segment, a_axis, kk, a_l)], 0, 2
        )
        gB_segment = cute.group_modes(
            pipeline.gB[(None, None, ab_segment, b_axis, kk, b_l)], 0, 2
        )
        if cutlass.const_expr(use_device_tensormaps):
            cute.copy(
                pipeline.tma_atom_a,
                gA_segment,
                pipeline.sA[(None, ab_buf)],
                tma_bar_ptr=sync.ab_full_mbar + ab_buf,
                mcast_mask=a_full_mcast_mask,
                tma_desc_ptr=a_desc_ptr,
            )
            cute.copy(
                pipeline.tma_atom_b,
                gB_segment,
                pipeline.sB[(None, ab_buf)],
                tma_bar_ptr=sync.ab_full_mbar + ab_buf,
                mcast_mask=b_full_mcast_mask,
                tma_desc_ptr=b_desc_ptr,
            )
        else:
            cute.copy(
                pipeline.tma_atom_a,
                gA_segment,
                pipeline.sA[(None, ab_buf)],
                tma_bar_ptr=sync.ab_full_mbar + ab_buf,
                mcast_mask=a_full_mcast_mask,
            )
            cute.copy(
                pipeline.tma_atom_b,
                gB_segment,
                pipeline.sB[(None, ab_buf)],
                tma_bar_ptr=sync.ab_full_mbar + ab_buf,
                mcast_mask=b_full_mcast_mask,
            )

    @cute.jit
    def _sm103_tma_load_sf_segment(
        self,
        pipeline: GroupedGemmTmaPipeline,
        sync: GroupedGemmPipelineSync,
        sf_k: cutlass.Int32,
        sf_stage: cutlass.Constexpr[int],
        phase: cutlass.Int32,
        sfa_axis: cutlass.Int32,
        sfb_axis: cutlass.Int32,
        group_axes: tuple,
        tma_desc_ptrs: tuple,
        use_device_tensormaps: cutlass.Constexpr[bool],
    ) -> None:
        _, _, sfa_l, sfb_l = group_axes
        _, _, sfa_full_mcast_mask, sfb_full_mcast_mask = sync.tma_mcast_masks
        if cutlass.const_expr(use_device_tensormaps):
            _, _, sfa_desc_ptr, sfb_desc_ptr = tma_desc_ptrs
        _tma_wait_and_arm_expect_tx(
            sync.sf_empty_mbar + sf_stage,
            sync.sf_full_mbar + sf_stage,
            phase,
            sync.cluster_cta_rank,
            num_tma_load_bytes=pipeline.num_tma_load_bytes_sf,
            NUM_CTAS=self.NUM_CTAS,
        )
        gSFA_segment = cute.filter_zeros(pipeline.gSFA[(None, sfa_axis, sf_k, sfa_l)])
        gSFB_segment = cute.filter_zeros(pipeline.gSFB[(None, sfb_axis, sf_k, sfb_l)])
        if cutlass.const_expr(use_device_tensormaps):
            cute.copy(
                pipeline.tma_atom_sfa,
                gSFA_segment,
                pipeline.sSFA[(None, sf_stage)],
                tma_bar_ptr=sync.sf_full_mbar + sf_stage,
                mcast_mask=sfa_full_mcast_mask,
                tma_desc_ptr=sfa_desc_ptr,
            )
            cute.copy(
                pipeline.tma_atom_sfb,
                gSFB_segment,
                pipeline.sSFB[(None, sf_stage)],
                tma_bar_ptr=sync.sf_full_mbar + sf_stage,
                mcast_mask=sfb_full_mcast_mask,
                tma_desc_ptr=sfb_desc_ptr,
            )
        else:
            cute.copy(
                pipeline.tma_atom_sfa,
                gSFA_segment,
                pipeline.sSFA[(None, sf_stage)],
                tma_bar_ptr=sync.sf_full_mbar + sf_stage,
                mcast_mask=sfa_full_mcast_mask,
            )
            cute.copy(
                pipeline.tma_atom_sfb,
                gSFB_segment,
                pipeline.sSFB[(None, sf_stage)],
                tma_bar_ptr=sync.sf_full_mbar + sf_stage,
                mcast_mask=sfb_full_mcast_mask,
            )

    @cute.jit
    def _sm103_tma_load_k_tile(
        self,
        pipeline: GroupedGemmTmaPipeline,
        sync: GroupedGemmPipelineSync,
        kk: cutlass.Int32,
        accum_cnt_smem: cutlass.Int32,
        tile_axes: tuple,
        group_axes: tuple,
        tma_desc_ptrs: tuple,
        use_device_tensormaps: cutlass.Constexpr[bool],
    ) -> None:
        a_axis, b_axis, sfa_axis, sfb_axis = tile_axes
        # The ultra SFA descriptor owns one CTA's 128 rows, so expand the
        # cluster-M tile into the descriptor's CTA-local row-tile coordinate.
        sfa_axis = sfa_axis * cutlass.Int32(self.NUM_CTAS) + sync.cluster_cta_rank
        phase = accum_cnt_smem & cutlass.Int32(1)
        SF_SEGMENTS: cutlass.Constexpr[int] = sm103.sf_segments(self.sf_vec_size)
        LOAD_STEPS: cutlass.Constexpr[int] = sm103.sf_load_steps(self.sf_vec_size)
        for sf_stage in cutlass.range_constexpr(LOAD_STEPS):
            if cutlass.const_expr(sf_stage < sm103.SM103_AB_SEGMENTS):
                ab_count = (
                    accum_cnt_smem * cutlass.Int32(sm103.SM103_AB_SEGMENTS) + sf_stage
                )
                ab_buf = ab_count % cutlass.Int32(self.NUM_SMEM_BUFFERS)
                ab_phase = (
                    ab_count // cutlass.Int32(self.NUM_SMEM_BUFFERS)
                ) & cutlass.Int32(1)
                self._sm103_tma_load_ab_segment(
                    pipeline,
                    sync,
                    kk,
                    sf_stage,
                    ab_buf,
                    ab_phase,
                    a_axis,
                    b_axis,
                    group_axes,
                    tma_desc_ptrs,
                    use_device_tensormaps,
                )
            if cutlass.const_expr(sf_stage < SF_SEGMENTS):
                self._sm103_tma_load_sf_segment(
                    pipeline,
                    sync,
                    kk * cutlass.Int32(SF_SEGMENTS) + sf_stage,
                    sf_stage,
                    phase,
                    sfa_axis,
                    sfb_axis,
                    group_axes,
                    tma_desc_ptrs,
                    use_device_tensormaps,
                )

    def _tma_weights_ahead_enabled(self) -> bool:
        """Whether to issue B/SFB (weight) TMA loads ahead of the
        ``_before_tma_producer_tile`` gate. Only meaningful when that hook
        actually blocks on activation availability; the base hook is a no-op,
        so the base kernel keeps the plain issue order."""
        return False

    @cute.jit
    def _tma_producer_body(  # noqa: C901
        self,
        pipeline: GroupedGemmTmaPipeline,
        problem: GroupedGemmProblem,
        sync: GroupedGemmPipelineSync,
        tensormap_manager,
        tensormaps: cute.Tensor,
        tile_producer_hook_tensor: cute.Tensor,
        use_device_tensormaps: cutlass.Constexpr[bool],
    ):
        M, N, K = problem.mnk
        accum_cnt_smem = cutlass.Int32(0)
        accum_cnt_out = cutlass.Int32(0)

        if cutlass.const_expr(self.STATIC_SCHEDULER):
            scheduler = StaticTileScheduler.create_producer(self.NUM_CTAS)
        else:
            scheduler = DynamicTileScheduler.create_producer(
                sync.counter_ptr,
                sync.tile_cta_bar_mbar,
                sync.tile_id_smem_ptr,
                sync.tile_consumer_mbar,
                sync.tile_producer_mbar,
                sync.cluster_cta_rank,
                self.NUM_CTAS,
                self.NUM_TILE_BUFFERS,
                self.NUM_TILE_CTA_BARS,
            )
        tile_idx = scheduler.initial_work_tile_info()
        visitor = GroupedProblemVisitor.create(
            problem.split_sizes,
            M,
            N,
            K,
            problem.groups,
            self.problem_type,
            self.BLOCK_SIZE_M,
            self.BLOCK_SIZE_N,
            self.BLOCK_SIZE_K,
            self.force_n_major,
            self.num_n_clusters,
            problem.local_rank,
            self.world_size,
            self.SWAP_AB,
        )
        work = visitor.get_work(tile_idx)

        while work.is_valid_tile:
            g = work.group_idx
            m_size = work.m_size
            n_size = work.n_size
            num_k_tiles = work.num_k_tiles
            if cutlass.const_expr(self.SWAP_AB):
                act_off_tiles = work.n_tile_prefix
            else:
                act_off_tiles = work.m_tile_prefix
            sf_act_off_tiles = act_off_tiles
            if cutlass.const_expr(
                self.SWAP_AB and uses_paged_blockscaled_scale_rows(self.BLOCK_SIZE_N)
            ):
                scale_page_rows: cutlass.Constexpr[int] = (
                    (self.BLOCK_SIZE_N + 127) // 128
                ) * 128
                sf_act_off_tiles = work.scale_split_prefix // cutlass.Int32(
                    scale_page_rows
                )

            if cutlass.const_expr(use_device_tensormaps):
                tensormap_a_g_ptr = tensormap_manager.get_tensormap_ptr(
                    tensormaps[(g, 0, None)].iterator
                )
                tensormap_b_g_ptr = tensormap_manager.get_tensormap_ptr(
                    tensormaps[(g, 1, None)].iterator
                )
                tensormap_sfa_g_ptr = tensormap_manager.get_tensormap_ptr(
                    tensormaps[(g, 2, None)].iterator
                )
                tensormap_sfb_g_ptr = tensormap_manager.get_tensormap_ptr(
                    tensormaps[(g, 3, None)].iterator
                )
                tma_desc_ptrs = (
                    tensormap_manager.get_tensormap_ptr(
                        tensormap_a_g_ptr, cute.AddressSpace.generic
                    ),
                    tensormap_manager.get_tensormap_ptr(
                        tensormap_b_g_ptr, cute.AddressSpace.generic
                    ),
                    tensormap_manager.get_tensormap_ptr(
                        tensormap_sfa_g_ptr, cute.AddressSpace.generic
                    ),
                    tensormap_manager.get_tensormap_ptr(
                        tensormap_sfb_g_ptr, cute.AddressSpace.generic
                    ),
                )
                # Acquire the descriptors published by the prepare kernel
                # before this kernel's TMA proxy consumes them.
                tensormap_manager.fence_tensormap_update(tensormap_a_g_ptr)
                tensormap_manager.fence_tensormap_update(tensormap_b_g_ptr)
                tensormap_manager.fence_tensormap_update(tensormap_sfa_g_ptr)
                tensormap_manager.fence_tensormap_update(tensormap_sfb_g_ptr)
            else:
                tma_desc_ptrs = ()

            while work.is_valid_tile and work.group_idx == g:
                scheduler.producer_publish_tile(accum_cnt_out)

                tile_m_idx = work.tile_m_idx
                tile_n_idx = work.tile_n_idx

                if cutlass.const_expr(use_device_tensormaps):
                    tile_axes = (
                        tile_m_idx,
                        tile_n_idx,
                        tile_m_idx,
                        tile_n_idx,
                    )
                    group_axes = (0, 0, 0, 0)
                elif cutlass.const_expr(self.SWAP_AB):
                    tile_axes = (
                        tile_m_idx,
                        act_off_tiles + tile_n_idx,
                        tile_m_idx,
                        sf_act_off_tiles + tile_n_idx,
                    )
                    group_axes = (g, 0, g, 0)
                else:
                    tile_axes = (
                        act_off_tiles + tile_m_idx,
                        tile_n_idx,
                        sf_act_off_tiles + tile_m_idx,
                        tile_n_idx,
                    )
                    group_axes = (0, g, 0, g)

                # Decode regime only (strictly under one M-tile per group):
                # at prefill the per-tile quant gather completes just-in-time
                # and pre-arming a ring of stages costs 1.3-1.4% on staged
                # MXFP8 16K-32K tokens/rank, and even the exactly-one-full-
                # tile boundary row (m_size == BLOCK_M) measures +2.3% - at
                # decode the gather gate dominates instead.
                weights_ahead = cutlass.Int32(0)
                if cutlass.const_expr(self._tma_weights_ahead_enabled()):
                    if m_size < cutlass.Int32(self.BLOCK_SIZE_M):
                        weights_ahead = cutlass.min(
                            cutlass.Int32(self.NUM_SMEM_BUFFERS), num_k_tiles
                        )
                        _blockscaled_tma_issue_weights_ahead(
                            pipeline,
                            sync,
                            weights_ahead,
                            accum_cnt_smem,
                            tile_axes,
                            group_axes,
                            tma_desc_ptrs,
                            use_device_tensormaps,
                            NUM_CTAS=self.NUM_CTAS,
                            NUM_SMEM_BUFFERS=self.NUM_SMEM_BUFFERS,
                        )

                self._before_tma_producer_tile(
                    tile_m_idx,
                    tile_n_idx,
                    act_off_tiles,
                    m_size,
                    n_size,
                    K,
                    tile_producer_hook_tensor,
                )

                if cutlass.const_expr(self.USE_SM103_ULTRA):
                    for kk in cutlass.range(num_k_tiles, unroll=self.KLOOP_UNROLL):
                        self._sm103_tma_load_k_tile(
                            pipeline,
                            sync,
                            kk,
                            accum_cnt_smem,
                            tile_axes,
                            group_axes,
                            tma_desc_ptrs,
                            use_device_tensormaps,
                        )
                        accum_cnt_smem += cutlass.Int32(1)
                else:
                    accum_cnt_smem = _blockscaled_tma_load_tile(
                        pipeline,
                        sync,
                        num_k_tiles,
                        accum_cnt_smem,
                        tile_axes,
                        group_axes,
                        tma_desc_ptrs,
                        use_device_tensormaps,
                        KLOOP_UNROLL=self.KLOOP_UNROLL,
                        NUM_CTAS=self.NUM_CTAS,
                        NUM_SMEM_BUFFERS=self.NUM_SMEM_BUFFERS,
                        weights_ahead=weights_ahead,
                        WEIGHTS_AHEAD=self._tma_weights_ahead_enabled(),
                    )

                accum_cnt_out += cutlass.Int32(1)

                scheduler.producer_wait_tile_released(accum_cnt_out)
                tile_idx = scheduler.advance_producer(accum_cnt_out)
                work = visitor.get_work(tile_idx)

        scheduler.producer_publish_termination(accum_cnt_out)

    # MMA consumer: each k-block first stages SFA/SFB SMEM->TMEM, then
    # calls ``cute.gemm`` with ``tcgen05.Field.SFA / SFB`` pointing at
    # the matching k-block slice of TMEM.
    @cute.jit
    def _mma_consumer_body(  # noqa: C901
        self,
        pipeline: GroupedGemmMmaPipeline,
        problem: GroupedGemmProblem,
        sync: GroupedGemmPipelineSync,
    ):
        M, N, K = problem.mnk
        accum_cnt_tile = cutlass.Int32(0)
        accum_cnt_smem = cutlass.Int32(0)

        if cutlass.const_expr(self.STATIC_SCHEDULER):
            scheduler = StaticTileScheduler.create_consumer(self.NUM_CTAS)
        else:
            scheduler = DynamicTileScheduler.create_consumer(
                sync.tile_consumer_mbar,
                sync.tile_id_smem_ptr,
                self.NUM_CTAS,
                self.NUM_TILE_BUFFERS,
            )
        tile_idx = scheduler.initial_work_tile_info()
        visitor = GroupedProblemVisitor.create(
            problem.split_sizes,
            M,
            N,
            K,
            problem.groups,
            self.problem_type,
            self.BLOCK_SIZE_M,
            self.BLOCK_SIZE_N,
            self.BLOCK_SIZE_K,
            self.force_n_major,
            self.num_n_clusters,
            problem.local_rank,
            self.world_size,
            self.SWAP_AB,
        )
        work = visitor.get_mma_work(tile_idx)

        while work.is_valid_tile:
            if cutlass.const_expr(self.USE_SM103_ULTRA):
                accum_cnt_smem = _sm103_mma_consumer_tile(
                    pipeline,
                    sync,
                    work.num_k_tiles,
                    accum_cnt_tile,
                    accum_cnt_smem,
                    KLOOP_UNROLL=self.KLOOP_UNROLL,
                    NUM_CTAS=self.NUM_CTAS,
                    NUM_SMEM_BUFFERS=self.NUM_SMEM_BUFFERS,
                    NUM_TMEM_BUFFERS=self.NUM_TMEM_BUFFERS,
                    OVERLAPPING_ACCUM=self.OVERLAPPING_ACCUM,
                    SF_RING_TILES=self.SM103_SF_RING_TILES,
                    cta_group=self.cta_group,
                    sf_dtype=self.sf_dtype,
                    sf_vec_size=self.sf_vec_size,
                    FRESH_TILED_MMA=False,
                )
            else:
                accum_cnt_smem = _blockscaled_mma_consumer_tile(
                    pipeline,
                    sync,
                    work.num_k_tiles,
                    accum_cnt_tile,
                    accum_cnt_smem,
                    KLOOP_UNROLL=self.KLOOP_UNROLL,
                    NUM_CTAS=self.NUM_CTAS,
                    NUM_SMEM_BUFFERS=self.NUM_SMEM_BUFFERS,
                    NUM_TMEM_BUFFERS=self.NUM_TMEM_BUFFERS,
                    OVERLAPPING_ACCUM=self.OVERLAPPING_ACCUM,
                    cta_group=self.cta_group,
                    sf_dtype=self.sf_dtype,
                )
            accum_cnt_tile += cutlass.Int32(1)
            tile_idx = scheduler.advance_consumer(accum_cnt_tile)
            work = visitor.get_mma_work(tile_idx)

    @cute.jit
    def _blockscaled_epilog_consumer_tile(  # noqa: C901
        self,
        tidx: cutlass.Int32,
        pipeline: GroupedGemmEpilogPipeline,
        sync: GroupedGemmPipelineSync,
        tTR_rAcc: cute.Tensor,
        tiled_copy_r2s,
        tRS_rC: cute.Tensor,
        tRS_sC: cute.Tensor,
        num_k_tiles: cutlass.Int32,
        accum_cnt_tile: cutlass.Int32,
        c_tile_indices: tuple,
        skip_tma_store: cutlass.Boolean,
        tma_desc_ptrs: tuple,
        use_device_tensormaps: cutlass.Constexpr[bool],
        b_scale: cutlass.Float32,
        evict_first_tma_store: cutlass.Constexpr[bool] = False,
        split_prefix: cutlass.Int32 = 0,
        postprocess_scale_prefix: cutlass.Int32 = 0,
        a_scale_m_size: cutlass.Int32 = 0,
        tile_m_idx: cutlass.Int32 = 0,
        tile_n_idx: cutlass.Int32 = 0,
        cluster_cta_rank: cutlass.Int32 = 0,
        a_global_scale_inv_smem_ptr=None,
        a_global_scale_inv_ptr: cutlass.Int64 = 0,
        use_a_global_scale_inv: cutlass.Constexpr[bool] = False,
        use_b_global_scale_inv: cutlass.Constexpr[bool] = False,
        postprocess_output: cute.Tensor | None = None,
        postprocess_scale: cute.Tensor | None = None,
        postprocess_store: cutlass.Constexpr[bool] = False,
        postprocess_m_size: cutlass.Int32 = 0,
        postprocess_n_size: cutlass.Int32 = 0,
        postprocess_row_prefix: cutlass.Int32 | None = None,
    ) -> cutlass.Boolean:
        # The postprocess store may target an activation-ring row base while
        # `split_prefix` stays logical for the NVFP4 per-row global scales.
        postprocess_cm = split_prefix
        if cutlass.const_expr(postprocess_row_prefix is not None):
            postprocess_cm = postprocess_row_prefix
        tmem_buf, tmem_phase = _get_bufidx_phase(accum_cnt_tile, self.NUM_TMEM_BUFFERS)
        cute.arch.mbarrier_wait(sync.tmem_full_mbar + tmem_buf, tmem_phase)
        if cutlass.const_expr(use_device_tensormaps):
            (c_desc_ptr,) = tma_desc_ptrs
        if cutlass.const_expr(use_a_global_scale_inv):
            _stage_a_global_scale_inv(
                tidx,
                a_global_scale_inv_ptr,
                a_global_scale_inv_smem_ptr,
                split_prefix,
                a_scale_m_size,
                tile_m_idx,
                tile_n_idx,
                cluster_cta_rank,
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

        NUM_MMA_ATOMS_M = cute.size(pipeline.tCtAcc_base.shape, mode=[1])
        NUM_MMA_ATOMS_N = cute.size(pipeline.tCtAcc_base.shape, mode=[2])
        for mma_m_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_M):
            for mma_n_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_N):
                tiled_copy_t2r, tTR_tAcc_base, _ = self._epilog_tmem_copy_and_partition(
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
                _, bSG_sC, bSG_gC_partitioned = self._epilog_gmem_copy_and_partition(
                    pipeline.c_atom,
                    pipeline.c_tensor,
                    pipeline.epi_tile,
                    pipeline.sC,
                    mma_m_idx=mma_m_idx,
                    mma_n_idx=mma_n_idx,
                )
                c_tile_m_idx, c_tile_n_idx = c_tile_indices
                bSG_gC = bSG_gC_partitioned[
                    (None, None, None, c_tile_m_idx, c_tile_n_idx, 0)
                ]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, tmem_buf)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                atom_m: cutlass.Constexpr[int] = (
                    self.cta_tile_shape_mnk[0] // NUM_MMA_ATOMS_M
                )
                atom_n: cutlass.Constexpr[int] = (
                    self.cta_tile_shape_mnk[1] // NUM_MMA_ATOMS_N
                )
                c_acc = cute.make_identity_tensor((atom_m, atom_n))
                c_acc_epi = cute.flat_divide(c_acc, pipeline.epi_tile)
                tTR_cAcc = tiled_copy_t2r.get_slice(tidx).partition_D(c_acc_epi)
                tTR_cAcc = cute.group_modes(tTR_cAcc, 3, cute.rank(tTR_cAcc))
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = accum_cnt_tile * subtile_cnt
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
                                number_of_threads=32 * len(self.EPILOG_WARP_IDS),
                            )
                            warp_idx_local = cute.arch.make_warp_uniform(
                                cute.arch.warp_idx()
                            )
                            if warp_idx_local == self.EPILOG_WARP_IDS[0]:
                                with cute.arch.elect_one():
                                    if cutlass.const_expr(self.NUM_CTAS == 2):
                                        cute.arch.mbarrier_arrive(
                                            sync.cross_seam_mbar,
                                            peer_cta_rank_in_cluster=0,
                                        )
                                    else:
                                        cute.arch.mbarrier_arrive(sync.cross_seam_mbar)
                    tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
                    if num_k_tiles == 0:
                        tRS_rAcc.store(cute.zeros_like(tRS_rAcc.load()))
                    if cutlass.const_expr(
                        use_a_global_scale_inv or use_b_global_scale_inv
                    ):
                        tRS_cAcc = tiled_copy_r2s.retile(
                            tTR_cAcc[(None, None, None, real_subtile_idx)]
                        )
                        assert cute.size(tRS_rAcc) % 2 == 0
                        for value_idx in cutlass.range(
                            0,
                            cute.size(tRS_rAcc),
                            2,
                            unroll_full=True,
                        ):
                            a_scales: list = []
                            for pair_idx in cutlass.range_constexpr(2):
                                a_scale = cutlass.Float32(1.0)
                                if cutlass.const_expr(use_a_global_scale_inv):
                                    coord = tRS_cAcc[value_idx + pair_idx]
                                    if cutlass.const_expr(self.SWAP_AB):
                                        row_in_tile = (
                                            cutlass.Int32(mma_n_idx * atom_n) + coord[1]
                                        )
                                    else:
                                        row_in_tile = (
                                            cutlass.Int32(mma_m_idx * atom_m) + coord[0]
                                        )
                                    a_scale = a_scales_smem[row_in_tile]
                                a_scales.append(a_scale)
                            if cutlass.const_expr(use_a_global_scale_inv):
                                (
                                    tRS_rAcc[value_idx],
                                    tRS_rAcc[value_idx + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (
                                        tRS_rAcc[value_idx],
                                        tRS_rAcc[value_idx + 1],
                                    ),
                                    (a_scales[0], a_scales[1]),
                                )
                            if cutlass.const_expr(use_b_global_scale_inv):
                                (
                                    tRS_rAcc[value_idx],
                                    tRS_rAcc[value_idx + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (
                                        tRS_rAcc[value_idx],
                                        tRS_rAcc[value_idx + 1],
                                    ),
                                    (b_scale, b_scale),
                                )
                    tRS_rC.store(tRS_rAcc.load().to(pipeline.sC.element_type))
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.NUM_C_STAGES
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    cute.arch.fence_proxy(kind="async.shared", space="cta")
                    cute.arch.barrier(
                        barrier_id=_BAR_EPILOG_SYNC,
                        number_of_threads=32 * len(self.EPILOG_WARP_IDS),
                    )
                    warp_idx_local = cute.arch.make_warp_uniform(cute.arch.warp_idx())
                    if cutlass.const_expr(postprocess_store):
                        if not skip_tma_store:
                            self._blockscaled_epilog_postprocess_store(
                                tidx,
                                pipeline.sC[(None, None, c_buffer)],
                                postprocess_output,
                                postprocess_scale,
                                tile_m_idx,
                                tile_n_idx,
                                c_tile_indices,
                                mma_m_idx,
                                mma_n_idx,
                                real_subtile_idx,
                                NUM_MMA_ATOMS_M,
                                NUM_MMA_ATOMS_N,
                                postprocess_cm,
                                postprocess_scale_prefix,
                                postprocess_m_size,
                                postprocess_n_size,
                                cluster_cta_rank,
                            )
                    elif warp_idx_local == self.EPILOG_WARP_IDS[0]:
                        if not skip_tma_store:
                            if cutlass.const_expr(use_device_tensormaps):
                                if cutlass.const_expr(evict_first_tma_store):
                                    # Keep a runtime-zero dependency so the TMA
                                    # cache policy remains register-valued.
                                    cache_policy = cutlass.Int64(
                                        0x12F0000000000000
                                    ) + cutlass.Int64(accum_cnt_tile) * cutlass.Int64(0)
                                    cute.copy(
                                        pipeline.c_atom,
                                        bSG_sC[(None, c_buffer)],
                                        bSG_gC[(None, real_subtile_idx)],
                                        tma_desc_ptr=c_desc_ptr,
                                        cache_policy=cache_policy,
                                    )
                                else:
                                    cute.copy(
                                        pipeline.c_atom,
                                        bSG_sC[(None, c_buffer)],
                                        bSG_gC[(None, real_subtile_idx)],
                                        tma_desc_ptr=c_desc_ptr,
                                    )
                            else:
                                cute.copy(
                                    pipeline.c_atom,
                                    bSG_sC[(None, c_buffer)],
                                    bSG_gC[(None, real_subtile_idx)],
                                )
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(
                                self.NUM_C_STAGES - 1,
                                read=True,
                            )
                    if cutlass.const_expr(
                        mma_m_idx == NUM_MMA_ATOMS_M - 1
                        and mma_n_idx == NUM_MMA_ATOMS_N - 1
                        and subtile_idx == subtile_cnt - 1
                    ):
                        cute.arch.fence_view_async_tmem_load()
                    cute.arch.barrier(
                        barrier_id=_BAR_EPILOG_SYNC,
                        number_of_threads=32 * len(self.EPILOG_WARP_IDS),
                    )

        warp_idx_local = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx_local == self.EPILOG_WARP_IDS[0]:
            with cute.arch.elect_one():
                if cutlass.const_expr(self.NUM_CTAS == 2):
                    cute.arch.mbarrier_arrive(
                        sync.tmem_empty_mbar + tmem_buf,
                        peer_cta_rank_in_cluster=0,
                    )
                else:
                    cute.arch.mbarrier_arrive(sync.tmem_empty_mbar + tmem_buf)
        return warp_idx_local == self.EPILOG_WARP_IDS[0]

    @cute.jit
    def _epilog_consumer_body(  # noqa: C901
        self,
        tidx: cutlass.Int32,
        pipeline: GroupedGemmEpilogPipeline,
        epilog_copy: tuple,
        problem: GroupedGemmProblem,
        sync: GroupedGemmPipelineSync,
        tensormap_manager,
        tensormaps: cute.Tensor,
        use_device_tensormaps: cutlass.Constexpr[bool],
        a_global_scale_inv_smem_ptr=None,
        a_global_scale_inv_ptr: cutlass.Int64 = 0,
        b_global_scale_inv_ptr: cutlass.Int64 = 0,
        use_a_global_scale_inv: cutlass.Constexpr[bool] = False,
        use_b_global_scale_inv: cutlass.Constexpr[bool] = False,
        postprocess_output: cute.Tensor | None = None,
        postprocess_scale: cute.Tensor | None = None,
        postprocess_store: cutlass.Constexpr[bool] = False,
    ):
        M, N, K = problem.mnk
        tTR_rAcc, tiled_copy_r2s, tRS_rC, tRS_sC = epilog_copy
        accum_cnt_tile = cutlass.Int32(0)

        if cutlass.const_expr(self.STATIC_SCHEDULER):
            scheduler = StaticTileScheduler.create_consumer(self.NUM_CTAS)
        else:
            scheduler = DynamicTileScheduler.create_consumer(
                sync.tile_consumer_mbar,
                sync.tile_id_smem_ptr,
                self.NUM_CTAS,
                self.NUM_TILE_BUFFERS,
            )
        tile_idx = scheduler.initial_work_tile_info()
        visitor = GroupedProblemVisitor.create(
            problem.split_sizes,
            M,
            N,
            K,
            problem.groups,
            self.problem_type,
            self.BLOCK_SIZE_M,
            self.BLOCK_SIZE_N,
            self.BLOCK_SIZE_K,
            self.force_n_major,
            self.num_n_clusters,
            problem.local_rank,
            self.world_size,
            self.SWAP_AB,
        )
        work = visitor.get_work(tile_idx)
        b_scale_ptr = cute.make_ptr(
            cutlass.Float32,
            b_global_scale_inv_ptr,
            cute.AddressSpace.gmem,
            assumed_align=4,
        )

        while work.is_valid_tile:
            g = work.group_idx
            m_size = work.m_size
            a_scale_m_size = work.n_size if self.SWAP_AB else work.m_size
            num_k_tiles = work.num_k_tiles
            if cutlass.const_expr(self.SWAP_AB):
                c_m_off_tiles = cutlass.Int32(0)
                c_n_off_tiles = work.n_tile_prefix
            else:
                c_m_off_tiles = work.m_tile_prefix
                c_n_off_tiles = cutlass.Int32(0)

            _epilog_wait_pending_tma_store()
            if cutlass.const_expr(use_device_tensormaps):
                tensormap_c_g_ptr = tensormap_manager.get_tensormap_ptr(
                    tensormaps[(g, 4, None)].iterator
                )
                c_desc_ptr = tensormap_manager.get_tensormap_ptr(
                    tensormap_c_g_ptr, cute.AddressSpace.generic
                )
                tma_desc_ptrs = (c_desc_ptr,)
                # Acquire the descriptor published by the prepare kernel
                # before this kernel's TMA proxy consumes it.
                tensormap_manager.fence_tensormap_update(tensormap_c_g_ptr)
            else:
                tma_desc_ptrs = ()

            b_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(use_b_global_scale_inv):
                b_scale = cute.arch.load(b_scale_ptr + g, cutlass.Float32)

            while work.is_valid_tile and work.group_idx == g:
                tile_buf, _ = _get_bufidx_phase(accum_cnt_tile, self.NUM_TILE_BUFFERS)

                tile_m_idx = work.tile_m_idx
                tile_n_idx = work.tile_n_idx

                # Follower-skip: when a 2-CTA cluster's per-CTA row
                # range falls entirely past m_size (group ends
                # mid-cluster), the follower's acc is garbage —
                # skip its TMA store. Callers pad per-group M to a
                # ``BLOCK_M // NUM_CTAS`` multiple so the in-range
                # CTA stays fully aligned. SWAP_AB maps kernel-M to
                # the uniform N axis, so this never fires there.
                cta_m_start = tile_m_idx * self.BLOCK_SIZE_M + sync.cluster_cta_rank * (
                    self.BLOCK_SIZE_M // self.NUM_CTAS
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
                    tma_desc_ptrs,
                    use_device_tensormaps,
                    b_scale,
                    split_prefix=work.split_prefix,
                    postprocess_scale_prefix=work.scale_split_prefix,
                    a_scale_m_size=a_scale_m_size,
                    tile_m_idx=tile_m_idx,
                    tile_n_idx=tile_n_idx,
                    cluster_cta_rank=sync.cluster_cta_rank,
                    a_global_scale_inv_smem_ptr=a_global_scale_inv_smem_ptr,
                    a_global_scale_inv_ptr=a_global_scale_inv_ptr,
                    use_a_global_scale_inv=use_a_global_scale_inv,
                    use_b_global_scale_inv=use_b_global_scale_inv,
                    postprocess_output=postprocess_output,
                    postprocess_scale=postprocess_scale,
                    postprocess_store=postprocess_store,
                    postprocess_m_size=(work.n_size if self.SWAP_AB else work.m_size),
                    postprocess_n_size=(work.m_size if self.SWAP_AB else work.n_size),
                )

                scheduler.consumer_release_tile(
                    sync.tile_producer_mbar,
                    tile_buf,
                    sync.cluster_cta_rank,
                    is_leader,
                )

                accum_cnt_tile += cutlass.Int32(1)
                tile_idx = scheduler.advance_consumer(accum_cnt_tile)
                work = visitor.get_work(tile_idx)

        _epilog_wait_pending_tma_store()

    @cute.jit
    def _idle_body(self):
        return

    def _num_tma_load_bytes(
        self,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_tma_smem_layout_staged,
        tiled_mma: cute.TiledMma,
        a_dtype=None,
        b_dtype=None,
    ):
        a_dtype = self.a_dtype if a_dtype is None else a_dtype
        b_dtype = self.b_dtype if b_dtype is None else b_dtype
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)
        if self.USE_SM103_ULTRA:
            # Ultra arms A/B and SF transactions on separate mbarriers, so
            # the per-kind byte counts are needed individually.
            num_tma_load_bytes_ab = (
                cute.size_in_bytes(
                    sA.element_type,
                    cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
                )
                + cute.size_in_bytes(
                    sB.element_type,
                    cute.slice_(b_smem_layout_staged, (None, None, None, 0)),
                )
            ) * num_mma_ctas
            num_tma_load_bytes_sf = (
                cute.size_in_bytes(
                    sSFA.element_type,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                )
                + cute.size_in_bytes(
                    sSFB.element_type,
                    cute.slice_(sfb_tma_smem_layout_staged, (None, None, None, 0)),
                )
            ) * num_mma_ctas
            return (
                num_tma_load_bytes_ab + num_tma_load_bytes_sf,
                num_tma_load_bytes_ab,
                num_tma_load_bytes_sf,
            )
        num_tma_load_bytes = (
            cute.size_in_bytes(
                a_dtype,
                cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
            )
            + cute.size_in_bytes(
                b_dtype,
                cute.slice_(b_smem_layout_staged, (None, None, None, 0)),
            )
            + cute.size_in_bytes(
                sSFA.element_type,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            + cute.size_in_bytes(
                sSFB.element_type,
                cute.slice_(sfb_tma_smem_layout_staged, (None, None, None, 0)),
            )
        ) * num_mma_ctas
        return num_tma_load_bytes, 0, 0

    def _scale_tmem_tensors(
        self,
        tmem_ptr: cute.Pointer,
        tCtAcc_base: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mma_tiler: tuple[int, int, int],
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
    ):
        sfa_tmem_ptr = cute.recast_ptr(
            tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
            dtype=self.sf_dtype,
        )
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=self.sf_dtype,
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            self.tiled_mma_sfb if self.BLOCK_SIZE_N == 32 else tiled_mma,
            self.mma_tiler_sfb if self.BLOCK_SIZE_N == 32 else mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
        tCtSFB_field = cute.make_tensor(
            tCtSFB.iterator,
            self._make_sfb_tmem_field_layout(tCtSFB_layout),
        )
        return tCtSFA, tCtSFB, tCtSFB_field

    # Mainloop SF SMEM->TMEM copy. ``tcgen05.Cp4x32x128bOp`` is the only
    # SF copy atom in tcgen05; MXFP8 and NVFP4 share it.
    # Tensormap helpers — A/B/C match the bf16 kernel; SFA/SFB use
    # Epilog copy / partition helpers — verbatim from the base kernel.
    def _epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs,
        mma_m_idx: int = 0,
        mma_n_idx: int = 0,
        *,
        cta_tile_shape_mnk=None,
        c_layout=None,
        c_dtype=None,
    ) -> tuple:
        cta_tile_shape_mnk = cta_tile_shape_mnk or self.cta_tile_shape_mnk
        c_layout = c_layout or self.c_layout
        c_dtype = c_dtype or self.c_dtype
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk,
            c_layout,
            c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        copy_atom_t2r = _cap_tmem_ld_repetition(copy_atom_t2r, self.acc_dtype)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), mma_m_idx, mma_n_idx, None)], epi_tile
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), mma_m_idx, mma_n_idx, None, None, None)],
            epi_tile,
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def _epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
        *,
        c_layout=None,
        c_dtype=None,
    ) -> tuple:
        c_layout = c_layout or self.c_layout
        c_dtype = c_dtype or self.c_dtype
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            c_layout, c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def _epilog_gmem_copy_and_partition(
        self,
        tma_atom_c,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
        mma_m_idx: int = 0,
        mma_n_idx: int = 0,
    ) -> tuple:
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.flat_divide(
            gC_mnl[((None, None), mma_m_idx, mma_n_idx, None, None, None)],
            epi_tile,
        )
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            cute.group_modes(gC_for_tma_partition, 0, 2),
        )
        return tma_atom_c, bSG_sC, bSG_gC

    def _make_sfb_tma_smem_layout_staged(
        self, sfb_smem_layout_staged: cute.Layout
    ) -> cute.Layout:
        """SFB SMEM view with the MMA-N atom factor lifted to the top
        level for TMA (FP4 BN=256 only). Byte-identical to the input
        layout; just rearranges the hierarchy for TMA's shape checker."""
        if self.BLOCK_SIZE_N <= self.MMA_ATOM_N or self.USE_SM103_ULTRA:
            return sfb_smem_layout_staged

        per_stage = cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
        shape = per_stage.shape
        stride = per_stage.stride
        tma_per_stage = cute.make_layout(
            ((shape[0][0][0], shape[0][1]), shape[0][0][1], shape[2]),
            stride=((stride[0][0][0], stride[0][1]), stride[0][0][1], stride[2]),
        )
        return cute.append(
            tma_per_stage,
            cute.make_layout(
                self.NUM_SMEM_BUFFERS,
                stride=cute.cosize(cute.filter_zeros(tma_per_stage)),
            ),
        )

    def _make_sfb_tmem_field_layout(self, sfb_tmem_layout: cute.Layout) -> cute.Layout:
        """Expose FP4 BN=256's N-atom factor for ``tcgen05.Field.SFB``."""
        if self.BLOCK_SIZE_N <= self.MMA_ATOM_N or self.USE_SM103_ULTRA:
            return sfb_tmem_layout

        shape = sfb_tmem_layout.shape
        stride = sfb_tmem_layout.stride
        num_n_atoms = self.BLOCK_SIZE_N // 128
        n_chunks = shape[0][0][0][1]
        chunks_per_atom = n_chunks // num_n_atoms
        chunk_stride = stride[0][0][0][1]
        atom_n_stride = chunk_stride * chunks_per_atom

        atom_shape = (
            ((shape[0][0][0][0], chunks_per_atom), shape[0][0][1]),
            shape[0][1],
        )
        atom_stride = (
            ((stride[0][0][0][0], chunk_stride), stride[0][0][1]),
            stride[0][1],
        )
        return cute.make_layout(
            (atom_shape, num_n_atoms, shape[1], shape[2]),
            stride=(atom_stride, atom_n_stride, stride[1], stride[2]),
        )


@dataclass(frozen=True)
class BlockScaledProductionConfig:
    """Format-family-specific production config.

    These are already final kernel config dictionaries. The auto-picker
    only returns one of the entries in ``PRODUCTION_BLOCKSCALED_CONFIGS``.
    """

    label: str
    format_names: frozenset[str]
    config: dict[str, int | bool]
    required_compute_capability: tuple[int, int] | None = None
    directions: frozenset[str] = frozenset({"fprop", "dgrad", "wgrad"})

    def supports_format(self, format: BlockScaledFormatSpec) -> bool:
        return getattr(format, "name", "").lower() in self.format_names

    def supports_device(self, device: torch.device | None = None) -> bool:
        if self.required_compute_capability is None:
            return True
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(device)
            == self.required_compute_capability
        )

    def build(
        self,
        format: BlockScaledFormatSpec = MXFP8_E4M3,
        *,
        static_scheduler: bool | None = None,
    ) -> dict:
        if not self.supports_format(format):
            raise ValueError(
                f"{self.label} does not support format "
                f"{getattr(format, 'name', format)!r}"
            )
        cfg = dict(self.config)
        if static_scheduler is not None:
            cfg["STATIC_SCHEDULER"] = static_scheduler
        return cfg

    @property
    def num_ctas(self) -> int:
        return int(self.config["NUM_CTAS"])

    @property
    def block_m(self) -> int:
        return int(self.config["BLOCK_SIZE_M"])

    @property
    def block_n(self) -> int:
        return int(self.config["BLOCK_SIZE_N"])

    @property
    def swap_ab(self) -> bool:
        return bool(self.config["SWAP_AB"])

    @property
    def overlapping_accum(self) -> bool:
        return bool(self.config["OVERLAPPING_ACCUM"])


@dataclass(frozen=True)
class _ParsedGemmProblem:
    G: int
    GM: int
    N: int
    K: int
    c_shape: tuple[int, ...]


@dataclass(frozen=True)
class _LaunchTensorBundle:
    placeholders: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    compile_tensors: tuple
    runtime_tensors: tuple
