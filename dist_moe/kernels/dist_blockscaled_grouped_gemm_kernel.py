# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side kernel class and CuTeDSL helpers for the distributed block-scaled grouped GEMM."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync

from ..formats import (
    BlockScaledFormat,
    BlockScaledFormatId,
    canonical_swiglu_clamp,
    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
    resolve_half_range_scale,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)
from ._dsl_compat import thread_exit
from ._quant_packing import _pack_fp4_e2m1_xn_to_u32_rn
from ._swiglu_quant import _load_swiglu_fwd_nvfp4_block
from .activation_buffer import (
    _FP4_FORMAT_NAMES,
    ACTIVATION_A_Q_OFFSET,
    ACTIVATION_A_SCALE_OFFSET,
    ACTIVATION_COL_Q_OFFSET,
    ACTIVATION_COL_SCALE_OFFSET,
    ACTIVATION_SOURCE_X_OFFSET,
    ACTIVATION_SOURCE_Y_OFFSET,
)
from .activation_buffer_kernel import (
    _activation_buffer_col_scale_storage_byte_extent,
    _activation_buffer_row_global_scale_tensor,
    _activation_buffer_row_scale_storage_byte_extent,
    _activation_buffer_tensor_with_byte_extent,
)
from .blockscaled_gemm_tiles import (
    _stage_a_global_scale_inv,
)
from .blockscaled_grouped_gemm import (
    NVFP4,
)
from .blockscaled_grouped_gemm_kernel import (
    _BLOCKSCALED_TENSORMAP_STAGING_INT64S,
    BlockScaledFormatSpec,
    BlockScaledGroupedGemmKernel,
    MXFP8_E4M3,
)
from .blockscaled_quantization_common import (
    _compute_nvfp4_token_scales,
    _compute_row_amax_b16x2_x16,
    _cublas_blockscaled_qscale_offset,
    _load_tensor_row_as_b16x2,
    _reduce_nvfp4_block_amaxes,
    _scale_nvfp4_per_token_block,
    _warp_reduce_amax_f32,
)
from .combine_swiglu_quant import (
    _apply_epilog_global_scale_inv,
    _combine_epilog_get_scatter_ptr,
    _combine_swiglu_bwd_quant_producer_body,
    _combine_swiglu_fwd_quant_producer_body,
)
from .config import EPILOGUE_SUBTILE_AUTO
from .dispatch_quant import (
    _dispatch_copy_blockscaled_work_tile,
    _dispatch_quant_fetch_group_tile_id,
    _dispatch_quant_tma_per_tile_wait,
    _dispatch_quantize_tile,
    _load_group_global_scale_inv,
    _make_u32x4_vector_type,
    _mega_forward_wait_h1_tile,
)
from .grouped_gemm_kernel import (
    _activation_buffer_rows,
    _BAR_EPILOG_SYNC,
    _BAR_FULL_CTA_SYNC,
    _remap_m_tile_idx,
    _swap_if,
    _transpose_c_if_swap,
)
from .params import (
    BlockscaledPointerStrideArgs,
    ceil_div,
    CombineSwigluQuantParams,
    DispatchQuantParams,
    GroupedGemmEpilogPipeline,
    GroupedGemmMmaPipeline,
    GroupedGemmPipelineSync,
    GroupedGemmProblem,
    GroupedGemmTmaPipeline,
    params_from_kernel,
    swap_ab_pointer_strides,
)
from .swiglu_epilogue import (
    blockscaled_epilog_postprocess_store,
    STAGED_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD,
)
from .tile_scheduler import (
    _FPROP,
    _get_bufidx_phase,
    _ring_cached_slot,
    advance_blockscaled_scale_start,
    blockscaled_scale_row_start,
    DynamicTileScheduler,
    GroupedProblemVisitor,
    stage_expert_metadata,
    StaticTileScheduler,
)


def _format_id(format: BlockScaledFormatSpec) -> int:
    match format.name.split(";", 1)[0]:
        case "mxfp8_e4m3":
            return BlockScaledFormatId.MXFP8_E4M3.value
        case "mxfp8_e5m2":
            return BlockScaledFormatId.MXFP8_E5M2.value
        case "nvfp4":
            return BlockScaledFormatId.NVFP4.value
        case "mxfp4":
            return BlockScaledFormatId.MXFP4.value
        case _:
            raise ValueError(f"Unsupported block-scaled format: {format.name}")


def _activation_format_name(format: BlockScaledFormatSpec) -> str:
    return format.name.split(";", 1)[0]


class DistBlockScaledGroupedGemmKernel(BlockScaledGroupedGemmKernel):
    """Block-scaled grouped GEMM with distributed DISPATCH/COMBINE fusion.

    COMBINE reuses the base block-scaled TMA producer and MMA consumer, then
    scatters accumulator subtiles directly to peer combine buffers. DISPATCH
    adds a producer warpgroup that gathers BF16 rows through ``gather_ptrs``,
    quantizes them into block-scaled A/SFA buffers with the CuTe quantization
    numerics helpers, and hands those buffers to the base GEMM pipeline.
    """

    # Overridden by the chunked Mega forward; the shared postprocess-store
    # override keys the h2 store cache policy off it.
    ACTIVATION_RING_CHUNKS: int = 0

    # Mode ids.
    DISPATCH_MODE: int = 1
    COMBINE_MODE: int = 2
    COMBINE_SWIGLU_FWD_MODE: int = 3
    COMBINE_SWIGLU_BWD_MODE: int = 4

    # Common CTA geometry.
    THREADS_PER_WARP: int = 32
    EPILOG_WG_THREADS: int = THREADS_PER_WARP * len(
        BlockScaledGroupedGemmKernel.EPILOG_WARP_IDS
    )

    # Plain COMBINE keeps the baseline 8-warp CTA layout and only changes C
    # staging depth.
    COMBINE_C_STAGES: int = 1

    # DISPATCH CTA scheduling: 8 gather/quant warps, 1 TMA warp, 1 MMA warp,
    # 2 idle warps, and the baseline 4 epilogue warps.
    DISPATCH_TOTAL_WARPS: int = 16
    DISPATCH_THREADS_PER_CTA: int = THREADS_PER_WARP * DISPATCH_TOTAL_WARPS
    DISPATCH_QUANT_WARP_IDS: tuple[int, ...] = tuple(range(8, 16))
    DISPATCH_QUANT_FIRST_WARP: int = DISPATCH_QUANT_WARP_IDS[0]
    DISPATCH_QUANT_WARPS: int = len(DISPATCH_QUANT_WARP_IDS)
    DISPATCH_QUANT_WARPS_PER_GROUP: int = 4
    DISPATCH_QUANT_GROUPS: int = DISPATCH_QUANT_WARPS // DISPATCH_QUANT_WARPS_PER_GROUP
    DISPATCH_QUANT_GROUP_THREADS: int = (
        THREADS_PER_WARP * DISPATCH_QUANT_WARPS_PER_GROUP
    )
    DISPATCH_QUANT_SYNC_BAR: int = 3

    # DISPATCH quant tile layout. Each quant warp owns a 32x64 source tile:
    # 4 row lanes x 8 row reps x 8 elements/lane. Four warps then form one
    # producer group over a 32x256 tile before signaling the TMA producer.
    DISPATCH_QUANT_ELEMS_PER_LANE: int = 8
    DISPATCH_QUANT_QDATA_ELEMS_PER_WORD: int = 4
    DISPATCH_QUANT_COL_BLOCKS_PER_SCALE: int = 4
    DISPATCH_QUANT_ELEMS_PER_SCALE_COL: int = (
        DISPATCH_QUANT_ELEMS_PER_LANE * DISPATCH_QUANT_COL_BLOCKS_PER_SCALE
    )
    DISPATCH_QUANT_SCALE_COLS_PER_WARP: int = 2
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: int = (
        DISPATCH_QUANT_WARPS_PER_GROUP * DISPATCH_QUANT_SCALE_COLS_PER_WARP
    )
    DISPATCH_QUANT_COL_LANES: int = 8
    DISPATCH_QUANT_ROW_LANES: int = 4
    DISPATCH_QUANT_ROW_REPS: int = 8
    DISPATCH_QUANT_COL_REDUCE_STAGES: int = DISPATCH_QUANT_ROW_LANES.bit_length() - 1
    DISPATCH_QUANT_ROW_REDUCE_STAGES: int = (
        DISPATCH_QUANT_COL_BLOCKS_PER_SCALE.bit_length() - 1
    )
    DISPATCH_BLOCKSCALED_COPY_COL_TILES_PER_WORK: int = 8
    DISPATCH_BLOCKSCALED_COPY_PREFETCH: int = 8
    VECTOR_DISPATCH_SCALE_COPY: bool = False

    # COMBINE_SWIGLU CTA scheduling. The GEMM and epilogue warp roles are
    # unchanged; warps 6-15 materialize the FC2 activation operand before TMA
    # reads it.
    COMBINE_SWIGLU_TOTAL_WARPS: int = 16
    COMBINE_SWIGLU_QUANT_FIRST_WARP: int = 6
    COMBINE_SWIGLU_WORK_TILES_PER_FETCH: int = 2

    # COMBINE_SWIGLU forward quant tile layouts. Row+col quant uses 8x8 per
    # warp; row-only quant uses 16x4 to keep row reductions within fewer lanes.
    COMBINE_SWIGLU_FWD_ELEMS_PER_LANE: int = 8
    COMBINE_SWIGLU_FWD_COL_BLOCKS_PER_SCALE: int = 4
    COMBINE_SWIGLU_FWD_COL_LANES: int = 8
    COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE: int = 2
    COMBINE_SWIGLU_FWD_ROW_ONLY_ELEMS_PER_LANE: int = 16
    COMBINE_SWIGLU_FWD_ROW_ONLY_COL_BLOCKS_PER_SCALE: int = 2
    COMBINE_SWIGLU_FWD_ROW_ONLY_COL_LANES: int = 4

    # Slot 0 carries the scheduled row, slots 1-4 carry warp maxima, and
    # slot 5 broadcasts the reciprocal token scale.
    NVFP4_ROW_REDUCTION_SLOTS: int = 6
    NVFP4_ROW_SCALE_SLOT: int = 5
    NVFP4_WARP_LOCAL_ROW_SCALE: bool = False
    NVFP4_GROUP_SOURCE_WAIT: bool = False
    NVFP4_WARP_LEADER_ROW_SCALE_LOAD: bool = False

    # COMBINE_SWIGLU backward quant tile layout: 8 column lanes x 4
    # elements/lane, with 8 adjacent lane blocks reduced for each column scale.
    COMBINE_SWIGLU_BWD_ELEMS_PER_LANE: int = 4
    COMBINE_SWIGLU_BWD_COL_BLOCKS_PER_SCALE: int = 8
    COMBINE_SWIGLU_BWD_COL_LANES: int = 8
    COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE: int = 1
    SWIGLU_VALUES_PER_THREAD: int = STAGED_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD
    # Swapped-axis epilogue geometry: chunked splits the swapped N axis per
    # CTA; the staged (and Mega) kernels treat it cluster-wide.
    SWAPPED_AXIS_PER_CTA: bool = False

    @classmethod
    def _dispatch_quant_topology(
        cls,
        *,
        format: BlockScaledFormatSpec,
        blockscaled_dispatch: bool,
        nvfp4_high_throughput: bool,
    ) -> tuple[int, int]:
        if format is NVFP4 and blockscaled_dispatch and nvfp4_high_throughput:
            return 1, cls.DISPATCH_QUANT_WARPS
        return cls.DISPATCH_QUANT_WARPS_PER_GROUP, cls.DISPATCH_QUANT_GROUPS

    @classmethod
    def _nvfp4_swiglu_quant_topology(
        cls,
        *,
        nvfp4_high_throughput: bool,
        combine_swiglu_k: int | None,
    ) -> tuple[int, int]:
        warps_per_group = (
            2
            if (
                nvfp4_high_throughput
                and combine_swiglu_k is not None
                and combine_swiglu_k <= 4096
            )
            else cls.DISPATCH_QUANT_WARPS_PER_GROUP
        )
        groups = max(
            1,
            (cls.COMBINE_SWIGLU_TOTAL_WARPS - cls.COMBINE_SWIGLU_QUANT_FIRST_WARP)
            // warps_per_group,
        )
        return warps_per_group, groups

    @staticmethod
    def _c_smem_skew(num_tmem_buffers: int) -> int:
        return 8 if num_tmem_buffers == 1 else 0

    def __init__(
        self,
        *,
        mode: int = COMBINE_MODE,
        dispatch_source_dtype: type[cutlass.Numeric] | None = None,
        blockscaled_dispatch: bool = False,
        nvfp4_high_throughput: bool = False,
        combine_swiglu_fast_math: bool = False,
        combine_swiglu_clamped: bool = False,
        combine_swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
        combine_swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
        combine_swiglu_row_quant_only: bool = False,
        combine_precomputed_swiglu: bool = False,
        combine_swiglu_k: int | None = None,
        interleaved_fc13: bool = False,
        prefetch_first_scatter_ptr: bool = False,
        combine_full_hidden_tiles: bool = False,
        **kwargs,
    ) -> None:
        if mode not in (
            self.DISPATCH_MODE,
            self.COMBINE_MODE,
            self.COMBINE_SWIGLU_FWD_MODE,
            self.COMBINE_SWIGLU_BWD_MODE,
        ):
            raise ValueError(f"unsupported DistBlockScaledGroupedGemm mode={mode}")
        super().__init__(**kwargs)
        self.mode = mode
        self.format_id = _format_id(self.format)
        # Derived from the format rather than threaded from the host: the fused
        # DISPATCH/SwiGLU quant below must use the same scale rule as the
        # host-side weight/activation quant, or the two halves of one fused path
        # disagree (MXFP4 diverged by rel_l2 ~0.14 when this was hardcoded off).
        # The kernel cache key already includes ``format.name``, so no extra key.
        activation_format_name = _activation_format_name(self.format)
        self.half_range_scale = resolve_half_range_scale(
            BlockScaledFormat(activation_format_name), None
        )
        self.is_fp4 = activation_format_name in _FP4_FORMAT_NAMES
        self.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD = 8 if self.is_fp4 else 4
        self.dispatch_quant_source_dtype = dispatch_source_dtype or cutlass.BFloat16
        self.blockscaled_dispatch = blockscaled_dispatch
        (
            self.DISPATCH_QUANT_WARPS_PER_GROUP,
            self.DISPATCH_QUANT_GROUPS,
        ) = self._dispatch_quant_topology(
            format=self.format,
            blockscaled_dispatch=blockscaled_dispatch,
            nvfp4_high_throughput=nvfp4_high_throughput,
        )
        self.DISPATCH_QUANT_GROUP_THREADS = (
            self.THREADS_PER_WARP * self.DISPATCH_QUANT_WARPS_PER_GROUP
        )
        if self.DISPATCH_QUANT_GROUPS == self.DISPATCH_QUANT_WARPS:
            # Give each warp one 16-row block and batch 4096 columns per work
            # item. Eight independent groups amortize counter and barrier traffic.
            self.DISPATCH_QUANT_SCALE_COLS_PER_TILE = 2
            self.DISPATCH_BLOCKSCALED_COPY_COL_TILES_PER_WORK = 128
            self.DISPATCH_BLOCKSCALED_COPY_PREFETCH = 16
        self.combine_swiglu_fast_math = combine_swiglu_fast_math
        (
            self.combine_swiglu_clamped,
            self.combine_swiglu_alpha,
            self.combine_swiglu_limit,
        ) = canonical_swiglu_clamp(
            combine_swiglu_clamped, combine_swiglu_alpha, combine_swiglu_limit
        )
        self.combine_swiglu_row_quant_only = combine_swiglu_row_quant_only
        self.combine_precomputed_swiglu = combine_precomputed_swiglu
        self.combine_swiglu_k = combine_swiglu_k
        self.interleaved_fc13 = interleaved_fc13
        self.prefetch_first_scatter_ptr = prefetch_first_scatter_ptr
        self.combine_full_hidden_tiles = combine_full_hidden_tiles
        if self.format is NVFP4 and self.mode == self.COMBINE_SWIGLU_FWD_MODE:
            if (
                combine_swiglu_k is None
                or combine_swiglu_k <= 0
                or combine_swiglu_k > 8192
                or combine_swiglu_k % 256 != 0
            ):
                raise ValueError(
                    "NVFP4 SwiGLU quantization requires K to be a positive "
                    "multiple of 256 no larger than 8192"
                )
        if self.dispatch_quant_source_dtype not in (cutlass.BFloat16, cutlass.Float16):
            raise TypeError("dispatch quant source dtype must be BF16 or FP16")
        if self.mode == self.DISPATCH_MODE:
            self.TOTAL_WARPS = self.DISPATCH_TOTAL_WARPS
            self.THREADS_PER_CTA = self.DISPATCH_THREADS_PER_CTA
        elif self.mode in (self.COMBINE_SWIGLU_FWD_MODE, self.COMBINE_SWIGLU_BWD_MODE):
            self.TOTAL_WARPS = self.COMBINE_SWIGLU_TOTAL_WARPS
            self.THREADS_PER_CTA = self.THREADS_PER_WARP * self.TOTAL_WARPS
        if self.format is NVFP4 and self.mode == self.COMBINE_SWIGLU_FWD_MODE:
            (
                self.DISPATCH_QUANT_WARPS_PER_GROUP,
                self.nvfp4_swiglu_quant_groups,
            ) = self._nvfp4_swiglu_quant_topology(
                nvfp4_high_throughput=nvfp4_high_throughput,
                combine_swiglu_k=combine_swiglu_k,
            )
            self.DISPATCH_QUANT_GROUP_THREADS = (
                self.THREADS_PER_WARP * self.DISPATCH_QUANT_WARPS_PER_GROUP
            )
        else:
            self.nvfp4_swiglu_quant_groups = max(
                1,
                (self.TOTAL_WARPS - self.COMBINE_SWIGLU_QUANT_FIRST_WARP)
                // self.DISPATCH_QUANT_WARPS_PER_GROUP,
            )

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        mode: int = COMBINE_MODE,
        problem_type: int = _FPROP,
        format: BlockScaledFormatSpec = MXFP8_E4M3,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        force_n_major: bool = False,
        num_n_clusters: int = 1,
        world_size: int = 1,
        dispatch_source_dtype: type[cutlass.Numeric] | None = None,
        blockscaled_dispatch: bool = False,
        nvfp4_high_throughput: bool = False,
        combine_swiglu_k: int | None = None,
        interleaved_fc13: bool = False,
    ) -> "DistBlockScaledGroupedGemmKernel":
        return cls(
            mode=mode,
            dispatch_source_dtype=dispatch_source_dtype,
            blockscaled_dispatch=blockscaled_dispatch,
            nvfp4_high_throughput=nvfp4_high_throughput,
            combine_swiglu_fast_math=bool(
                config.get("COMBINE_SWIGLU_FAST_MATH", False)
            ),
            combine_swiglu_clamped=bool(config.get("COMBINE_SWIGLU_CLAMPED", False)),
            combine_swiglu_alpha=float(
                config.get("COMBINE_SWIGLU_ALPHA", SWIGLU_CLAMP_ALPHA_DEFAULT)
            ),
            combine_swiglu_limit=float(
                config.get("COMBINE_SWIGLU_LIMIT", SWIGLU_CLAMP_LIMIT_DEFAULT)
            ),
            combine_swiglu_row_quant_only=bool(
                config.get("COMBINE_SWIGLU_ROW_QUANT_ONLY", False)
            ),
            combine_precomputed_swiglu=bool(
                config.get("COMBINE_PRECOMPUTED_SWIGLU", False)
            ),
            combine_swiglu_k=combine_swiglu_k,
            interleaved_fc13=interleaved_fc13,
            prefetch_first_scatter_ptr=bool(
                config.get("PREFETCH_FIRST_SCATTER_PTR", False)
            ),
            combine_full_hidden_tiles=bool(
                config.get("COMBINE_FULL_HIDDEN_TILES", False)
            ),
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
            use_sm103_ultra=config.get("USE_SM103_ULTRA", False),
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
        )

    @cute.jit
    def _blockscaled_epilog_postprocess_store(
        self,
        tidx,
        sC_stage,
        output_words,
        output_scale,
        tile_m_idx,
        tile_n_idx,
        c_tile_indices,
        mma_m_idx,
        mma_n_idx,
        subtile_idx,
        num_mma_atoms_m,
        num_mma_atoms_n,
        cm_start,
        scale_cm_start,
        m_size,
        n_size,
        cluster_cta_rank,
    ) -> None:
        blockscaled_epilog_postprocess_store(
            epilogue_tidx=tidx,
            sC_stage=sC_stage,
            output_words=output_words,
            output_scale=output_scale,
            tile_m_idx=tile_m_idx,
            tile_n_idx=tile_n_idx,
            swapped_tile_n_idx=(
                c_tile_indices[0] + cluster_cta_rank
                if self.SWAPPED_AXIS_PER_CTA
                else c_tile_indices[0]
            ),
            mma_m_idx=mma_m_idx,
            mma_n_idx=mma_n_idx,
            subtile_idx=subtile_idx,
            num_mma_atoms_m=num_mma_atoms_m,
            num_mma_atoms_n=num_mma_atoms_n,
            cm_start=cm_start,
            scale_cm_start=scale_cm_start,
            m_size=m_size,
            n_size=n_size,
            cluster_cta_rank=cluster_cta_rank,
            swapped_cluster_cta_rank=(
                cutlass.Int32(0) if self.SWAPPED_AXIS_PER_CTA else cluster_cta_rank
            ),
            swapped_block_size_n=(
                self.BLOCK_SIZE_M // self.NUM_CTAS
                if self.SWAPPED_AXIS_PER_CTA
                else self.BLOCK_SIZE_M
            ),
            values_per_thread=self.SWIGLU_VALUES_PER_THREAD,
            is_nvfp4=self.format is NVFP4,
            block_size_m=self.BLOCK_SIZE_M,
            block_size_n=self.BLOCK_SIZE_N,
            num_ctas=self.NUM_CTAS,
            epilogue_threads=self.EPILOG_WG_THREADS,
            fast_math=self.combine_swiglu_fast_math,
            clamped=self.combine_swiglu_clamped,
            alpha=self.combine_swiglu_alpha,
            limit=self.combine_swiglu_limit,
            swap_ab=self.SWAP_AB,
            format_id=self.format_id,
            qdata_elems_per_word=self.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
            # Ring h2 lines are re-read by FC2 shortly after being written;
            # keep them cacheable instead of evict-first streaming.
            store_cache_modifier=(None if self.ACTIVATION_RING_CHUNKS > 0 else "cs"),
        )

    def _setup_attributes(self):
        super()._setup_attributes()
        epi_m = cute.size(self.epi_tile[0])
        epi_n = cute.size(self.epi_tile[1])
        skew = self._c_smem_skew(self.NUM_TMEM_BUFFERS)
        c_stages = self.NUM_C_STAGES
        if self.SWAP_AB:
            col_stride = epi_m + skew
            self.c_smem_layout_staged = cute.make_layout(
                (epi_m, epi_n, c_stages),
                stride=(1, col_stride, epi_n * col_stride),
            )
        else:
            row_stride = epi_n + skew
            self.c_smem_layout_staged = cute.make_layout(
                (epi_m, epi_n, c_stages),
                stride=(row_stride, 1, epi_m * row_stride),
            )

    def _make_shared_storage(
        self,
        a_dtype,
        b_dtype,
        c_dtype,
        sf_dtype,
        G: int,
        use_global_scale_inv: bool,
    ):
        NUM_SMEM = self.NUM_SMEM_BUFFERS
        NUM_TMEM = self.NUM_TMEM_BUFFERS
        NUM_TILE = self.NUM_TILE_BUFFERS
        NUM_CTAS = self.NUM_CTAS

        a_smem_elems = cute.cosize(self.a_smem_layout_staged.outer)
        b_smem_elems = cute.cosize(self.b_smem_layout_staged.outer)
        if self.mode == self.DISPATCH_MODE:
            c_smem_elems = cute.cosize(self.epi_smem_layout_staged.outer)
        else:
            c_smem_elems = cute.cosize(self.c_smem_layout_staged)
        sfa_smem_elems = cute.cosize(self.sfa_smem_layout_staged)
        sfb_smem_elems = cute.cosize(self.sfb_smem_layout_staged)
        n_smem_empty = NUM_SMEM
        n_smem_full = NUM_SMEM
        n_tmem_full = NUM_TMEM
        n_tmem_empty = NUM_TMEM
        n_tile_consumer = NUM_TILE
        n_tile_producer = NUM_TILE
        n_tile_cta_bar = self.NUM_TILE_CTA_BARS
        n_tmem_dealloc = 1 if NUM_CTAS == 2 else 0
        n_cross_seam = 1 if self.OVERLAPPING_ACCUM else 0
        n_scatter_ptr_smem = max(
            cute.size(self.epi_tile[0]), cute.size(self.epi_tile[1])
        )
        n_dispatch_quant_tile = (
            self.DISPATCH_QUANT_GROUPS
            if self.mode
            in (
                self.DISPATCH_MODE,
                self.COMBINE_SWIGLU_FWD_MODE,
                self.COMBINE_SWIGLU_BWD_MODE,
            )
            else 0
        )
        if self.format is NVFP4 and self.mode == self.COMBINE_SWIGLU_FWD_MODE:
            n_dispatch_quant_tile = max(
                n_dispatch_quant_tile,
                self.nvfp4_swiglu_quant_groups * self.NVFP4_ROW_REDUCTION_SLOTS,
            )
        n_dispatch_quant_gather_ptr = (
            self.DISPATCH_QUANT_GROUPS * self.sf_vec_size
            if self.mode == self.DISPATCH_MODE and self.blockscaled_dispatch
            else 0
        )
        n_tensormap_buffer = max(
            _BLOCKSCALED_TENSORMAP_STAGING_INT64S,
            (G + 1) // 2,
        )
        n_a_global_scale_inv = (
            self._a_global_scale_inv_smem_size if use_global_scale_inv else 0
        )

        @cute.struct
        class DistBlockScaledSharedStorage:
            sC: cute.struct.Align[cute.struct.MemRange[c_dtype, c_smem_elems], 1024]
            sA: cute.struct.Align[cute.struct.MemRange[a_dtype, a_smem_elems], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[b_dtype, b_smem_elems], 1024]
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
            dispatch_quant_tile_smem: cute.struct.MemRange[
                cutlass.Int32, n_dispatch_quant_tile
            ]
            dispatch_quant_gather_ptr_smem: cute.struct.MemRange[
                cutlass.Int64, n_dispatch_quant_gather_ptr
            ]
            scatter_ptr_smem: cute.struct.MemRange[cutlass.Int64, n_scatter_ptr_smem]
            tensormap_buffer: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, n_tensormap_buffer], 128
            ]

            smem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_smem_empty]
            smem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_smem_full]
            tmem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_empty]
            tmem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_full]
            tile_consumer_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_consumer]
            tile_producer_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_producer]
            tile_cta_bar_mbar: cute.struct.MemRange[cutlass.Int64, n_tile_cta_bar]
            tmem_dealloc_mbar: cute.struct.MemRange[cutlass.Int64, n_tmem_dealloc]
            cross_seam_mbar: cute.struct.MemRange[cutlass.Int64, n_cross_seam]

            tmem_holding_buf: cutlass.Int32

        return DistBlockScaledSharedStorage

    # Source load helpers.
    # Producer scheduling helpers.
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
        # Publish this lane's generic-proxy gmem stores to the async proxy
        # before signalling: the consumer reads these tiles via TMA (async
        # proxy) and fence.proxy.async is thread-local, so the fence must run
        # on every producer lane that wrote, sequenced by the barrier below
        # ahead of the release. A consumer-side proxy fence cannot publish
        # another thread's writes across the generic->async boundary.
        cute.arch.fence_proxy("async.global")
        cute.arch.fence_acq_rel_gpu()
        cute.arch.barrier(
            barrier_id=barrier_id_base + quant_group,
            number_of_threads=self.DISPATCH_QUANT_GROUP_THREADS,
        )
        # The pre-add counter value is returned (meaningful on group lane 0
        # only) so callers can detect the signal that completed a tile.
        prev_count = cutlass.Uint32(0)
        if quant_group_lane == cutlass.Int32(0):
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

    # Quantization tile bodies.
    # Quantization producer loops.
    @cute.jit
    def _forward_act_tile_offset(
        self,
        group_act_tiles,
        group_rows,
        local_rank,
    ):
        del group_rows
        return _remap_m_tile_idx(
            cutlass.Int32(0),
            group_act_tiles,
            local_rank,
            self.world_size,
        )

    @cute.jit
    def _combine_swiglu_nvfp4_row_quant_producer_body(  # noqa: C901 -- CuTe trace-time producer control flow
        self,
        mWorkCounter: cute.Tensor,
        mDoneCounter: cute.Tensor,
        tile_smem_ptr: cute.Pointer,
        mFwdX: cute.Tensor,
        mFwdY: cute.Tensor,
        mRowQWords: cute.Tensor,
        mRowScale: cute.Tensor,
        mRowGlobalScaleInv: cute.Tensor,
        mNvfp4RecipLut: cute.Tensor,
        split_sizes: cute.Tensor,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        local_rank: cutlass.Int32,
        wait_for_source_tile: cutlass.Constexpr[bool] = False,
        source_done_counter_offset: cutlass.Int32 = 0,
        barrier_id_base: cutlass.Constexpr[int] = 3,
        precomputed_swiglu: cutlass.Constexpr[bool] = False,
        ring_h1_done_offset: cutlass.Int32 = 0,
    ) -> None:
        # Activation ring (chunked NVFP4 interleaved forward): this producer
        # reads the BF16 interleaved_h2 staging and writes h2 qdata/scales at
        # ring positions; the per-row global scales stay at logical rows.
        # It carries no ring wait of its own — the FC13 epilogue's
        # fc2_done wait covers the h2 slot overwrite transitively through
        # the source-tile wait below.
        activation_ring: cutlass.Constexpr[bool] = (
            self.ACTIVATION_RING_CHUNKS > 0 and precomputed_swiglu
        )
        K: cutlass.Constexpr[int] = self.combine_swiglu_k
        num_threads: cutlass.Constexpr[int] = self.DISPATCH_QUANT_GROUP_THREADS
        num_warps: cutlass.Constexpr[int] = self.DISPATCH_QUANT_WARPS_PER_GROUP
        words_per_block: cutlass.Constexpr[int] = 8
        values_per_block: cutlass.Constexpr[int] = 16
        scale_cols: cutlass.Constexpr[int] = K // values_per_block
        blocks_per_lane: cutlass.Constexpr[int] = (
            scale_cols + num_threads - 1
        ) // num_threads
        # Retaining each row in registers avoids an HBM reload; the K <= 8192
        # producer contract bounds the resulting register pressure.
        reduction_stride: cutlass.Constexpr[int] = self.NVFP4_ROW_REDUCTION_SLOTS

        tidx, _, _ = cute.arch.thread_idx()
        quant_lane = tidx - cutlass.Int32(
            self.COMBINE_SWIGLU_QUANT_FIRST_WARP * self.THREADS_PER_WARP
        )
        quant_group = quant_lane // cutlass.Int32(num_threads)
        group_lane = quant_lane - quant_group * cutlass.Int32(num_threads)
        warp_in_group = group_lane // cutlass.Int32(self.THREADS_PER_WARP)
        warp_lane = group_lane % cutlass.Int32(self.THREADS_PER_WARP)
        group_smem = tile_smem_ptr + quant_group * cutlass.Int32(reduction_stride)
        reduction_smem = cute.recast_ptr(group_smem, dtype=cutlass.Float32)
        barrier_id = barrier_id_base + quant_group

        row_idx = _dispatch_quant_fetch_group_tile_id(
            mWorkCounter,
            tile_smem_ptr,
            quant_group,
            group_lane,
            barrier_id_base=barrier_id_base,
            num_threads=num_threads,
            smem_stride=reduction_stride,
            relaxed_work_counter=True,
            DISPATCH_QUANT_SYNC_BAR=self.DISPATCH_QUANT_SYNC_BAR,
            DISPATCH_QUANT_GROUP_THREADS=self.DISPATCH_QUANT_GROUP_THREADS,
        )

        ACT_BLOCK_SIZE: cutlass.Constexpr[int] = (
            self.BLOCK_SIZE_N if self.SWAP_AB else self.BLOCK_SIZE_M
        )
        row_start = cutlass.Int32(0)
        scale_row_start = cutlass.Int32(0)
        act_tile_start = cutlass.Int32(0)
        ring_chunk_prefix = cutlass.Int32(0)
        ring_cached_chunk = cutlass.Int32(-1)
        ring_cached_sched = cutlass.Int32(0)
        ring_cached_row_base = cutlass.Int32(0)
        ring_cached_scale_base = cutlass.Int32(0)
        for g in cutlass.range(G, unroll=1):
            m_size = cutlass.Int32(split_sizes[g])
            row_end = row_start + m_size
            group_act_tiles = ceil_div(m_size, cutlass.Int32(ACT_BLOCK_SIZE))
            if cutlass.const_expr(activation_ring):
                ring_cached_chunk = cutlass.Int32(-1)
            while (row_idx >= row_start) and (row_idx < row_end):
                local_row = row_idx - row_start
                token_off = self._forward_act_tile_offset(
                    group_act_tiles,
                    m_size,
                    local_rank,
                )
                local_row = (
                    local_row + token_off * cutlass.Int32(ACT_BLOCK_SIZE)
                ) % m_size
                row = row_start + local_row
                scale_row = blockscaled_scale_row_start(
                    scale_row_start, local_row, self.SWAP_AB, self.BLOCK_SIZE_N
                )
                act_tile_slot = act_tile_start + local_row // cutlass.Int32(
                    ACT_BLOCK_SIZE
                )
                ring_row = row
                ring_scale_row = scale_row
                ring_sched = cutlass.Int32(0)
                if cutlass.const_expr(activation_ring):
                    # Chunk-cached slot derivation shared with the dispatch
                    # producers via `_ring_cached_slot`. The refresh flag is
                    # unused: no consumption wait is needed here, because the
                    # FC13 epilogue waits `fc2_done[sched - W]` before
                    # storing this chunk's interleaved_h2, and this producer
                    # reaches these rows only after that store (source-tile
                    # wait), so the h2 ring slot's previous occupant is
                    # already consumed.
                    (
                        ring_row,
                        ring_scale_row,
                        _ring_refreshed,
                        ring_cached_chunk,
                        ring_cached_sched,
                        ring_cached_row_base,
                        ring_cached_scale_base,
                    ) = _ring_cached_slot(
                        local_row,
                        m_size,
                        ring_chunk_prefix,
                        local_rank,
                        ring_cached_chunk,
                        ring_cached_sched,
                        ring_cached_row_base,
                        ring_cached_scale_base,
                        PIPELINE_CHUNK_ROWS=self.PIPELINE_CHUNK_ROWS,
                        ACTIVATION_RING_CHUNKS=self.ACTIVATION_RING_CHUNKS,
                        BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                        world_size=self.world_size,
                        SWAP_AB=self.SWAP_AB,
                    )
                    ring_sched = ring_cached_sched
                if cutlass.const_expr(wait_for_source_tile):
                    if cutlass.const_expr(self.NVFP4_GROUP_SOURCE_WAIT):
                        if warp_in_group == cutlass.Int32(0):
                            _mega_forward_wait_h1_tile(
                                mDoneCounter,
                                source_done_counter_offset,
                                act_tile_slot,
                                cutlass.Int32(K),
                                BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                                BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                                NUM_CTAS=self.NUM_CTAS,
                                SWAP_AB=self.SWAP_AB,
                            )
                        cute.arch.barrier(
                            barrier_id=barrier_id,
                            number_of_threads=num_threads,
                        )
                    else:
                        _mega_forward_wait_h1_tile(
                            mDoneCounter,
                            source_done_counter_offset,
                            act_tile_slot,
                            cutlass.Int32(K),
                            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                            NUM_CTAS=self.NUM_CTAS,
                            SWAP_AB=self.SWAP_AB,
                        )

                # One warpgroup owns the row. Each lane retains strided 16-value
                # blocks in registers so the quantization pass does not reload HBM.
                packed = cute.make_rmem_tensor(
                    (blocks_per_lane, words_per_block),
                    cutlass.Int32,
                )
                block_amaxes = cute.make_rmem_tensor(
                    blocks_per_lane,
                    cutlass.Float32,
                )
                for block_rep in cutlass.range_constexpr(blocks_per_lane):
                    scale_col = block_rep * num_threads + group_lane
                    active = scale_col < scale_cols
                    if cutlass.const_expr(precomputed_swiglu):
                        loaded = cute.make_rmem_tensor(8, cutlass.Int32)
                        loaded.fill(cutlass.Int32(0))
                        if active:
                            for chunk in cutlass.range_constexpr(4):
                                chunk_values = cute.make_rmem_tensor(2, cutlass.Int32)
                                _load_tensor_row_as_b16x2(
                                    mFwdY,
                                    ring_row,
                                    scale_col * values_per_block
                                    + cutlass.Int32(chunk * 4),
                                    chunk_values,
                                    4,
                                    self.dispatch_quant_source_dtype,
                                )
                                loaded[chunk * 2] = chunk_values[0]
                                loaded[chunk * 2 + 1] = chunk_values[1]
                        for pair in cutlass.range_constexpr(8):
                            packed[block_rep, pair] = loaded[pair]
                    else:
                        _load_swiglu_fwd_nvfp4_block(
                            mFwdX,
                            mFwdY,
                            row,
                            scale_col * values_per_block,
                            active,
                            packed,
                            block_rep,
                            self.dispatch_quant_source_dtype,
                            self.combine_swiglu_fast_math,
                            self.combine_swiglu_clamped,
                            self.combine_swiglu_alpha,
                            self.combine_swiglu_limit,
                        )
                    block_amaxes[block_rep] = _compute_row_amax_b16x2_x16(
                        packed,
                        block_rep,
                        self.dispatch_quant_source_dtype,
                    )

                # Reduce lane blocks within each warp, then reduce warp leaders
                # through shared memory and broadcast the row scale.
                row_amax_local = _reduce_nvfp4_block_amaxes(
                    block_amaxes,
                    blocks_per_lane,
                )
                warp_amax = _warp_reduce_amax_f32(row_amax_local)
                if warp_lane == cutlass.Int32(0):
                    cute.arch.store(
                        reduction_smem + cutlass.Int32(1) + warp_in_group,
                        warp_amax,
                        ss="cta",
                    )
                cute.arch.barrier(
                    barrier_id=barrier_id,
                    number_of_threads=num_threads,
                )
                if cutlass.const_expr(self.NVFP4_WARP_LOCAL_ROW_SCALE):
                    warpgroup_amax = (
                        cute.arch.load(
                            reduction_smem + cutlass.Int32(1) + warp_lane,
                            cutlass.Float32,
                            ss="cta",
                        )
                        if warp_lane < cutlass.Int32(num_warps)
                        else cutlass.Float32(0.0)
                    )
                    warpgroup_amax = _warp_reduce_amax_f32(warpgroup_amax)
                    row_scale = cutlass.Float32(0.0)
                    if warp_lane == cutlass.Int32(0):
                        row_scale, token_scale_inv = _compute_nvfp4_token_scales(
                            warpgroup_amax
                        )
                        if warp_in_group == cutlass.Int32(0):
                            mRowGlobalScaleInv[row] = token_scale_inv
                    row_scale = cute.arch.shuffle_sync(row_scale, cutlass.Int32(0))
                else:
                    row_scale = cutlass.Float32(0.0)
                    warpgroup_amax = cutlass.Float32(0.0)
                    if warp_in_group == cutlass.Int32(0):
                        warpgroup_amax = (
                            cute.arch.load(
                                reduction_smem + cutlass.Int32(1) + warp_lane,
                                cutlass.Float32,
                                ss="cta",
                            )
                            if warp_lane < cutlass.Int32(num_warps)
                            else cutlass.Float32(0.0)
                        )
                        warpgroup_amax = _warp_reduce_amax_f32(warpgroup_amax)
                        if warp_lane == cutlass.Int32(0):
                            row_scale, token_scale_inv = _compute_nvfp4_token_scales(
                                warpgroup_amax
                            )
                            cute.arch.store(
                                reduction_smem + self.NVFP4_ROW_SCALE_SLOT,
                                row_scale,
                                ss="cta",
                            )
                            mRowGlobalScaleInv[row] = token_scale_inv
                    cute.arch.barrier(
                        barrier_id=barrier_id,
                        number_of_threads=num_threads,
                    )
                    if cutlass.const_expr(self.NVFP4_WARP_LEADER_ROW_SCALE_LOAD):
                        if warp_lane == cutlass.Int32(0):
                            row_scale = cute.arch.load(
                                reduction_smem + self.NVFP4_ROW_SCALE_SLOT,
                                cutlass.Float32,
                                ss="cta",
                            )
                        row_scale = cute.arch.shuffle_sync(
                            row_scale,
                            cutlass.Int32(0),
                        )
                    else:
                        row_scale = cute.arch.load(
                            reduction_smem + self.NVFP4_ROW_SCALE_SLOT,
                            cutlass.Float32,
                            ss="cta",
                        )

                # Re-scan the retained blocks with the row scale and write packed
                # qdata plus one E4M3 scale byte per 16 input values.
                values = cute.make_rmem_tensor(values_per_block, cutlass.Float32)
                n_col_blocks: cutlass.Constexpr[int] = ceil_div(
                    scale_cols,
                    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
                )
                for block_rep in cutlass.range_constexpr(blocks_per_lane):
                    scale_col = block_rep * num_threads + group_lane
                    if scale_col < scale_cols:
                        scale_byte = _scale_nvfp4_per_token_block(
                            packed,
                            block_rep,
                            block_amaxes[block_rep],
                            row_scale,
                            mNvfp4RecipLut,
                            self.dispatch_quant_source_dtype,
                            values,
                            use_nvfp4_no_clip_scale=False,
                        )
                        q_words = cute.make_rmem_tensor(2, cutlass.Uint32)
                        _pack_fp4_e2m1_xn_to_u32_rn(
                            values,
                            q_words,
                            base=0,
                            packed_base=0,
                            recip=cutlass.Float32(1.0),
                            num_elems=values_per_block,
                            scale_values=False,
                        )
                        q_word_col = scale_col * 2
                        q_ptr = mRowQWords.iterator + cute.crd2idx(
                            (ring_row, q_word_col),
                            mRowQWords.layout,
                        )
                        cute.arch.store(q_ptr, q_words[0], cop="cs")
                        cute.arch.store(q_ptr + cutlass.Int32(1), q_words[1], cop="cs")
                        scale_offset = _cublas_blockscaled_qscale_offset(
                            cutlass.Int64(ring_scale_row),
                            cutlass.Int64(scale_col),
                            cutlass.Int64(n_col_blocks),
                        )
                        cute.arch.store(
                            mRowScale.iterator + scale_offset,
                            scale_byte,
                            cop="cs",
                        )

                prev_tile_count = self._dispatch_quant_signal_group_tile_done(
                    mDoneCounter,
                    act_tile_slot,
                    quant_group,
                    group_lane,
                    cutlass.Int32(1),
                    barrier_id_base,
                )
                if cutlass.const_expr(activation_ring):
                    # One h1 credit per completed act tile: the row whose
                    # signal above filled the tile counter credits all of the
                    # tile's rows at once, and the chunk's first tile also
                    # carries the tail padding, so the FC13 waiter keeps its
                    # constant CHUNK_ROWS target. The acquire re-read imports
                    # the other rows' interleaved_h2 loads (published by
                    # their release signals) ahead of the release credit.
                    if group_lane == cutlass.Int32(0):
                        ring_act_tile = local_row // cutlass.Int32(ACT_BLOCK_SIZE)
                        ring_tile_rows = cutlass.min(
                            m_size - ring_act_tile * cutlass.Int32(ACT_BLOCK_SIZE),
                            cutlass.Int32(ACT_BLOCK_SIZE),
                        )
                        if (
                            cutlass.Int32(prev_tile_count) + cutlass.Int32(1)
                            == ring_tile_rows
                        ):
                            _ = cute.arch.load(
                                cute.recast_ptr(
                                    mDoneCounter.iterator + act_tile_slot,
                                    dtype=cutlass.Uint32,
                                ),
                                cutlass.Uint32,
                                sem="acquire",
                                scope="gpu",
                            )
                            ring_credit = ring_tile_rows
                            tiles_per_chunk: cutlass.Constexpr[int] = (
                                self.PIPELINE_CHUNK_ROWS // ACT_BLOCK_SIZE
                            )
                            if ring_act_tile % cutlass.Int32(
                                tiles_per_chunk
                            ) == cutlass.Int32(0):
                                ring_phys_chunk = local_row // cutlass.Int32(
                                    self.PIPELINE_CHUNK_ROWS
                                )
                                ring_chunk_valid = cutlass.min(
                                    m_size
                                    - ring_phys_chunk
                                    * cutlass.Int32(self.PIPELINE_CHUNK_ROWS),
                                    cutlass.Int32(self.PIPELINE_CHUNK_ROWS),
                                )
                                ring_credit += (
                                    cutlass.Int32(self.PIPELINE_CHUNK_ROWS)
                                    - ring_chunk_valid
                                )
                            cute.arch.atomic_add(
                                cute.recast_ptr(
                                    mDoneCounter.iterator
                                    + ring_h1_done_offset
                                    + ring_sched,
                                    dtype=cutlass.Uint32,
                                ),
                                cutlass.Uint32(ring_credit),
                                sem="release",
                                scope="gpu",
                            )
                row_idx = _dispatch_quant_fetch_group_tile_id(
                    mWorkCounter,
                    tile_smem_ptr,
                    quant_group,
                    group_lane,
                    barrier_id_base=barrier_id_base,
                    num_threads=num_threads,
                    smem_stride=reduction_stride,
                    relaxed_work_counter=True,
                    DISPATCH_QUANT_SYNC_BAR=self.DISPATCH_QUANT_SYNC_BAR,
                    DISPATCH_QUANT_GROUP_THREADS=self.DISPATCH_QUANT_GROUP_THREADS,
                )

            row_start = row_end
            if cutlass.const_expr(activation_ring):
                ring_chunk_prefix += ceil_div(
                    m_size, cutlass.Int32(self.PIPELINE_CHUNK_ROWS)
                )
            scale_row_start = advance_blockscaled_scale_start(
                scale_row_start, m_size, self.SWAP_AB, self.BLOCK_SIZE_N
            )
            act_tile_start += group_act_tiles

    @cute.jit
    def _dispatch_quant_producer_body(
        self,
        mWorkCounter: cute.Tensor,
        mDoneCounter: cute.Tensor,
        tile_smem_ptr: cute.Pointer,
        dispatch_ptr_smem_ptr: cute.Pointer,
        mGatherPtrs: cute.Tensor,
        mRowQWords: cute.Tensor,
        mRowScale: cute.Tensor,
        mRowGlobalScaleInv: cute.Tensor,
        mColQWords: cute.Tensor,
        mColScale: cute.Tensor,
        split_sizes: cute.Tensor,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        K: cutlass.Int32,
        total_tiles: cutlass.Int32,
        local_rank: cutlass.Int32,
        use_global_scale_inv: cutlass.Constexpr[bool] = False,
    ) -> None:
        del M, total_tiles
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
        quant_col_tile_count = ceil_div(
            col_scale_cols,
            cutlass.Int32(self.DISPATCH_QUANT_SCALE_COLS_PER_TILE),
        )
        quant_col_tiles_per_work: cutlass.Constexpr[int] = (
            self.DISPATCH_BLOCKSCALED_COPY_COL_TILES_PER_WORK
            if self.blockscaled_dispatch
            else 1
        )
        scheduled_work_count_per_row_block = ceil_div(
            quant_col_tile_count,
            cutlass.Int32(quant_col_tiles_per_work),
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
                "dispatch quant act tile must be a multiple of sf_vec_size"
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
        for g in cutlass.range(G, unroll=1):
            m_size = cutlass.Int32(split_sizes[g])
            group_row_blocks = m_size // cutlass.Int32(self.sf_vec_size)
            group_tiles = group_row_blocks * scheduled_work_count_per_row_block
            group_act_tiles = ceil_div(m_size, cutlass.Int32(ACT_BLOCK_SIZE))
            tile_end = tile_start + group_tiles

            while (tile_idx_base >= tile_start) and (tile_idx_base < tile_end):
                local_tile = tile_idx_base - tile_start
                local_row_block = local_tile // scheduled_work_count_per_row_block
                scheduled_col_work_idx = (
                    local_tile - local_row_block * scheduled_work_count_per_row_block
                )
                first_quant_col_tile = scheduled_col_work_idx * cutlass.Int32(
                    quant_col_tiles_per_work
                )
                # Reuse the consumer's per-rank token-axis rotation so the
                # gather produces activation tiles in the TMA's request order.
                # The token axis is kernel-M normally and kernel-N under
                # SWAP_AB; group_act_tiles is that axis' tile count here.
                token_off = _remap_m_tile_idx(
                    cutlass.Int32(0),
                    group_act_tiles,
                    local_rank,
                    self.world_size,
                )
                local_row_block = (
                    local_row_block + token_off * cutlass.Int32(row_blocks_per_act_tile)
                ) % group_row_blocks
                first_scale_col = first_quant_col_tile * cutlass.Int32(
                    self.DISPATCH_QUANT_SCALE_COLS_PER_TILE
                )
                completed_quant_col_tiles = min(
                    cutlass.Int32(quant_col_tiles_per_work),
                    quant_col_tile_count - first_quant_col_tile,
                )
                row_block = start_m // cutlass.Int32(self.sf_vec_size) + local_row_block
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

                if cutlass.const_expr(self.blockscaled_dispatch):
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
                    _dispatch_copy_blockscaled_work_tile(
                        gather_ptrs,
                        mRowQWords,
                        mRowScale,
                        mRowGlobalScaleInv,
                        row_start,
                        row_scale_start,
                        first_scale_col,
                        K,
                        quant_group_lane,
                        self.DISPATCH_QUANT_GROUP_THREADS,
                        self.DISPATCH_QUANT_SCALE_COLS_PER_TILE
                        * quant_col_tiles_per_work,
                        self.DISPATCH_BLOCKSCALED_COPY_PREFETCH,
                        use_global_scale_inv,
                        DISPATCH_QUANT_QDATA_ELEMS_PER_WORD=self.DISPATCH_QUANT_QDATA_ELEMS_PER_WORD,
                        DISPATCH_QUANT_SCALE_COLS_PER_TILE=self.DISPATCH_QUANT_SCALE_COLS_PER_TILE,
                        VECTOR_DISPATCH_SCALE_COPY=self.VECTOR_DISPATCH_SCALE_COPY,
                        sf_vec_size=self.sf_vec_size,
                    )
                else:
                    col_block = first_scale_col + warp_in_quant_group * cutlass.Int32(
                        self.DISPATCH_QUANT_SCALE_COLS_PER_WARP
                    )
                    if col_block < col_scale_cols:
                        _dispatch_quantize_tile(
                            params_from_kernel(DispatchQuantParams, self),
                            warp_lane,
                            mGatherPtrs,
                            None,
                            mRowQWords,
                            mRowScale,
                            mColQWords,
                            mColScale,
                            row_start,
                            row_scale_start,
                            row_block,
                            col_block,
                            row_blocks_total,
                            K,
                        )

                self._dispatch_quant_signal_group_tile_done(
                    mDoneCounter,
                    act_tile_slot,
                    quant_group,
                    quant_group_lane,
                    completed_quant_col_tiles,
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

    def _tma_weights_ahead_enabled(self) -> bool:
        # DISPATCH only: there the per-tile hook waits on the quant gather,
        # and with SWAP_AB off B/SFB really are the gather-independent weight
        # operands. SWAP_AB inverts the roles (gathered tokens become B), and
        # the COMBINE_SWIGLU modes gate on locally produced h2 tiles whose
        # producer is the same kernel wave - out of scope. BLOCK_SIZE_M <= 64
        # limits the machinery to decode-class compilations: prefill-class
        # kernels stay branch-free (the residual `kk >= weights_ahead` check
        # in the k-loop measured +1.3% on staged MXFP8 at 16K tokens/rank).
        return (
            self.mode == self.DISPATCH_MODE
            and not self.SWAP_AB
            and self.NUM_CTAS == 1
            and self.BLOCK_SIZE_M <= 64
        )

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
        if cutlass.const_expr(
            self.mode == self.DISPATCH_MODE
            or self.mode == self.COMBINE_SWIGLU_FWD_MODE
            or self.mode == self.COMBINE_SWIGLU_BWD_MODE
        ):
            if cutlass.const_expr(self.SWAP_AB):
                _dispatch_quant_tma_per_tile_wait(
                    tile_n_idx,
                    act_off_tiles,
                    n_size,
                    tile_producer_hook_tensor,
                    K,
                    self.BLOCK_SIZE_N,
                    COMBINE_SWIGLU_BWD_MODE=self.COMBINE_SWIGLU_BWD_MODE,
                    COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE=self.COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE,
                    COMBINE_SWIGLU_FWD_MODE=self.COMBINE_SWIGLU_FWD_MODE,
                    COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE=self.COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE,
                    DISPATCH_QUANT_SCALE_COLS_PER_TILE=self.DISPATCH_QUANT_SCALE_COLS_PER_TILE,
                    format=self.format,
                    mode=self.mode,
                    sf_vec_size=self.sf_vec_size,
                )
            else:
                _dispatch_quant_tma_per_tile_wait(
                    tile_m_idx,
                    act_off_tiles,
                    m_size,
                    tile_producer_hook_tensor,
                    K,
                    self.BLOCK_SIZE_M,
                    COMBINE_SWIGLU_BWD_MODE=self.COMBINE_SWIGLU_BWD_MODE,
                    COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE=self.COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE,
                    COMBINE_SWIGLU_FWD_MODE=self.COMBINE_SWIGLU_FWD_MODE,
                    COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE=self.COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE,
                    DISPATCH_QUANT_SCALE_COLS_PER_TILE=self.DISPATCH_QUANT_SCALE_COLS_PER_TILE,
                    format=self.format,
                    mode=self.mode,
                    sf_vec_size=self.sf_vec_size,
                )

    @cute.kernel
    def dispatch_kernel(  # noqa: C901
        self,
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
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        dispatch_quant_total_tiles: cutlass.Int32,
        use_device_tensormaps: cutlass.Constexpr[bool],
        activation_buffer_base_ptr: cutlass.Int64,
        activation_buffer_size_bytes: cutlass.Int64,
        activation_offsets: cute.Tensor,
        use_activation_buffer: cutlass.Constexpr[bool],
        conditional_execution: cute.Tensor,
        use_conditional_execution: cutlass.Constexpr[bool],
        b_global_scale_inv_ptr: cutlass.Int64,
        use_global_scale_inv: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(use_conditional_execution):
            if conditional_execution[0] == cutlass.Int32(0):
                thread_exit()
        if cutlass.const_expr(use_activation_buffer):
            activation_rows = _activation_buffer_rows(split_sizes, G)
            activation_value_count = cutlass.Int64(activation_rows) * cutlass.Int64(K)
            activation_q_byte_extent = (
                activation_value_count
                * cutlass.Int64(cutlass.const_expr(self.format.a_dtype.width))
                // cutlass.Int64(8)
            )
            row_scale_byte_extent = _activation_buffer_row_scale_storage_byte_extent(
                activation_rows,
                K,
                sf_vec_size=self.sf_vec_size,
                sf_dtype_width=self.format.sf_dtype.width,
            )
            col_scale_byte_extent = _activation_buffer_col_scale_storage_byte_extent(
                activation_rows,
                K,
                sf_vec_size=self.sf_vec_size,
                sf_dtype_width=self.format.sf_dtype.width,
            )
            mDispatchQuantRowQWords = _activation_buffer_tensor_with_byte_extent(
                mDispatchQuantRowQWords,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                ACTIVATION_A_Q_OFFSET,
                cutlass.Uint32,
                activation_q_byte_extent,
            )
            mDispatchQuantRowScale = _activation_buffer_tensor_with_byte_extent(
                mDispatchQuantRowScale,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                ACTIVATION_A_SCALE_OFFSET,
                cutlass.Uint8,
                row_scale_byte_extent,
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
                    )
                )
            mDispatchQuantColQWords = _activation_buffer_tensor_with_byte_extent(
                mDispatchQuantColQWords,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                ACTIVATION_COL_Q_OFFSET,
                cutlass.Uint32,
                activation_q_byte_extent,
            )
            mDispatchQuantColScale = _activation_buffer_tensor_with_byte_extent(
                mDispatchQuantColScale,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                ACTIVATION_COL_SCALE_OFFSET,
                cutlass.Uint8,
                col_scale_byte_extent,
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
        dispatch_quant_tile_smem_ptr = storage.dispatch_quant_tile_smem.data_ptr()
        dispatch_quant_gather_ptr_smem_ptr = (
            storage.dispatch_quant_gather_ptr_smem.data_ptr()
        )
        a_global_scale_inv_smem_ptr = (
            storage.a_global_scale_inv.data_ptr() if use_global_scale_inv else None
        )
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

        split_sizes = stage_expert_metadata(
            split_sizes,
            cute.recast_ptr(storage.tensormap_buffer.data_ptr(), dtype=cutlass.Int32),
            G,
            synchronize=False,
        )
        cute.arch.barrier(
            barrier_id=_BAR_FULL_CTA_SYNC, number_of_threads=self.THREADS_PER_CTA
        )

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

        gA = cute.local_tile(
            mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB = cute.local_tile(
            mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC = cute.local_tile(
            mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gSFA = cute.local_tile(
            mSFA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB = cute.local_tile(
            mSFB,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
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
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)
        tCgSFA = thr_mma.partition_A(gSFA)
        tCgSFB = thr_mma_sfb.partition_B(gSFB)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
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
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
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
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((self.BLOCK_SIZE_M, self.BLOCK_SIZE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(
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

        num_tma_load_bytes, _, _ = self._num_tma_load_bytes(
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
        )

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
            )
            self._tma_producer_body(
                pipeline=pipeline,
                problem=problem,
                sync=sync,
                tensormap_manager=tensormap_manager,
                tensormaps=tensormaps,
                tile_producer_hook_tensor=mDispatchQuantDoneCounter,
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
            pipeline = GroupedGemmMmaPipeline(
                tiled_mma=tiled_mma,
                tCrA=tCrA,
                tCrB=tCrB,
                tCtAcc_base=tCtAcc_base,
                tCtSFA=tCtSFA,
                tCtSFA_copy=tCtSFA,
                tCtSFB=tCtSFB_field,
                tCtSFB_copy=tCtSFB,
                sSFA=sSFA,
                sSFB=sSFB,
            )
            self._mma_consumer_body(pipeline, problem, sync)
        elif warp_idx in self.EPILOG_WARP_IDS:
            epilogue_tidx = tidx - cutlass.Int32(
                self.EPILOG_WARP_IDS[0] * self.THREADS_PER_WARP
            )
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            (
                tiled_copy_t2r,
                _tTR_tAcc_base_00,
                tTR_rAcc,
            ) = self._epilog_tmem_copy_and_partition(
                tidx=epilogue_tidx,
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
                tidx=epilogue_tidx,
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
                tidx=epilogue_tidx,
                pipeline=pipeline,
                epilog_copy=epilog_copy,
                problem=problem,
                sync=sync,
                tensormap_manager=tensormap_manager,
                tensormaps=tensormaps,
                use_device_tensormaps=use_device_tensormaps,
                a_global_scale_inv_smem_ptr=a_global_scale_inv_smem_ptr,
                a_global_scale_inv_ptr=cutlass.Int64(
                    mDispatchQuantRowGlobalScaleInv.iterator.toint()
                ),
                b_global_scale_inv_ptr=b_global_scale_inv_ptr,
                use_a_global_scale_inv=use_global_scale_inv,
                use_b_global_scale_inv=use_global_scale_inv,
                postprocess_output=mDispatchQuantColQWords,
                postprocess_scale=mDispatchQuantColScale,
                postprocess_store=self.interleaved_fc13,
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
            if warp_idx >= self.DISPATCH_QUANT_FIRST_WARP:
                self._dispatch_quant_producer_body(
                    mWorkCounter=mDispatchQuantWorkCounter,
                    mDoneCounter=mDispatchQuantDoneCounter,
                    tile_smem_ptr=dispatch_quant_tile_smem_ptr,
                    dispatch_ptr_smem_ptr=dispatch_quant_gather_ptr_smem_ptr,
                    mGatherPtrs=mDispatchQuantGatherPtrs,
                    mRowQWords=mDispatchQuantRowQWords,
                    mRowScale=mDispatchQuantRowScale,
                    mRowGlobalScaleInv=mDispatchQuantRowGlobalScaleInv,
                    mColQWords=mDispatchQuantColQWords,
                    mColScale=mDispatchQuantColScale,
                    split_sizes=split_sizes,
                    G=G,
                    M=M,
                    K=K,
                    total_tiles=dispatch_quant_total_tiles,
                    local_rank=local_rank,
                    use_global_scale_inv=use_global_scale_inv,
                )

        if cutlass.const_expr(self.NUM_CTAS == 2):
            # Keep every active thread alive until peer DSMEM accesses complete.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

    @cute.jit
    def _combine_epilog_scatter_smem_tile(
        self,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
        scatter_ptr_smem_ptr: cute.Pointer,
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
        elem_size_bytes_c: cutlass.Constexpr[int],
        cluster_cta_rank: cutlass.Int32,
    ) -> None:
        """Scatter one staged epilogue subtile through the row pointer table.

        Output columns (hidden) are the contiguous SMEM axis, stored as
        16-byte vectors.
        """
        EPILOGUE_M: cutlass.Constexpr[int] = cute.size(sC.shape, mode=[0])
        EPILOGUE_N: cutlass.Constexpr[int] = cute.size(sC.shape, mode=[1])
        ATOM_M: cutlass.Constexpr[int] = self.BLOCK_SIZE_M // NUM_MMA_ATOMS_M
        ATOM_CTA_M: cutlass.Constexpr[int] = ATOM_M // self.NUM_CTAS
        N_ATOM_SPAN: cutlass.Constexpr[int] = self.BLOCK_SIZE_N // NUM_MMA_ATOMS_N
        # One 128-bit (16-byte) store per lane -> elements per vector chunk.
        ELEMS_PER_VEC: cutlass.Constexpr[int] = 16 // elem_size_bytes_c

        # Hidden is the contiguous (16-byte-vectorized) axis; SWAP_AB makes it
        # kernel-M, otherwise kernel-N. Token is the other axis.
        if cutlass.const_expr(self.SWAP_AB):
            TOKEN_DIM: cutlass.Constexpr[int] = EPILOGUE_N
            HIDDEN_DIM: cutlass.Constexpr[int] = EPILOGUE_M
        else:
            TOKEN_DIM: cutlass.Constexpr[int] = EPILOGUE_M
            HIDDEN_DIM: cutlass.Constexpr[int] = EPILOGUE_N

        CHUNKS_PER_ROW: cutlass.Constexpr[int] = HIDDEN_DIM // ELEMS_PER_VEC
        if cutlass.const_expr(CHUNKS_PER_ROW > self.EPILOG_WG_THREADS):
            raise ValueError(
                "fused combine epilogue has more 16-byte chunks per row "
                "than epilogue threads"
            )
        ROWS_PER_PASS: cutlass.Constexpr[int] = self.EPILOG_WG_THREADS // CHUNKS_PER_ROW
        if cutlass.const_expr(TOKEN_DIM % ROWS_PER_PASS != 0):
            raise ValueError(
                "fused combine scatter requires TOKEN_DIM to be a multiple of "
                "ROWS_PER_PASS"
            )
        NUM_ROW_PASSES: cutlass.Constexpr[int] = TOKEN_DIM // ROWS_PER_PASS
        chunk = tidx % CHUNKS_PER_ROW
        token_lane = tidx // CHUNKS_PER_ROW
        local_hidden = chunk * cutlass.Int32(ELEMS_PER_VEC)

        for row_pass in cutlass.range_constexpr(NUM_ROW_PASSES):
            local_token = token_lane + cutlass.Int32(row_pass * ROWS_PER_PASS)
            # Map the (token, hidden) lane to kernel (M, N) locals, then to
            # global coords with the fixed per-axis formulas.
            if cutlass.const_expr(self.SWAP_AB):
                m_local = local_hidden
                n_local = local_token
            else:
                m_local = local_token
                n_local = local_hidden
            m_global = (
                tile_m_idx * cutlass.Int32(self.BLOCK_SIZE_M)
                + cutlass.Int32(mma_m_idx * ATOM_M)
                + cluster_cta_rank * cutlass.Int32(ATOM_CTA_M)
                + m_local
            )
            n_global = (
                tile_n_idx * cutlass.Int32(self.BLOCK_SIZE_N)
                + cutlass.Int32(mma_n_idx * N_ATOM_SPAN)
                + cutlass.Int32(subtile_idx * EPILOGUE_N)
                + n_local
            )
            if cutlass.const_expr(self.SWAP_AB):
                hidden_global = m_global
                hidden_limit = m_size
            else:
                hidden_global = n_global
                hidden_limit = n_size

            chunk_in_bounds = cutlass.Boolean(True)
            if cutlass.const_expr(not self.combine_full_hidden_tiles):
                chunk_in_bounds = (
                    hidden_global + cutlass.Int32(ELEMS_PER_VEC) <= hidden_limit
                )
            if chunk_in_bounds:
                peer_base_i64 = cute.arch.load(
                    scatter_ptr_smem_ptr + local_token,
                    cutlass.Int64,
                    ss="cta",
                )
                if peer_base_i64 != cutlass.Int64(0):
                    # Explicit 128-bit SMEM->GMEM output scatter store.
                    ptr_vec = cute.recast_ptr(
                        sC.iterator
                        + cute.crd2idx(
                            (m_local, n_local, cutlass.Int32(0)),
                            sC.layout,
                        ),
                        dtype=cutlass.Uint32,
                    )
                    vec = cute.arch.load(ptr_vec, _make_u32x4_vector_type(), ss="cta")
                    dst_addr = peer_base_i64 + cutlass.Int64(
                        hidden_global
                    ) * cutlass.Int64(elem_size_bytes_c)
                    cute.arch.store(
                        cute.make_ptr(cutlass.Uint32, dst_addr, cute.AddressSpace.gmem),
                        vec,
                        cop="cg",
                    )

    @cute.jit
    def _combine_epilog_load_scatter_ptrs(
        self,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
        mScatter: cute.Tensor,
        scatter_ptr_smem_ptr: cute.Pointer,
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
    ) -> None:
        peer_base_i64, pointer_lane_active = _combine_epilog_get_scatter_ptr(
            tidx,
            sC,
            mScatter,
            tile_m_idx,
            tile_n_idx,
            mma_m_idx,
            mma_n_idx,
            subtile_idx,
            NUM_MMA_ATOMS_M,
            NUM_MMA_ATOMS_N,
            cm_start,
            m_size,
            n_size,
            cluster_cta_rank,
            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            EPILOG_WG_THREADS=self.EPILOG_WG_THREADS,
            NUM_CTAS=self.NUM_CTAS,
            SWAP_AB=self.SWAP_AB,
        )
        if pointer_lane_active:
            cute.arch.store(
                scatter_ptr_smem_ptr + tidx,
                peer_base_i64,
                ss="cta",
            )

    @cute.jit
    def _combine_epilog_consumer_body(  # noqa: C901
        self,
        tidx: cutlass.Int32,
        tCtAcc_base: cute.Tensor,
        tCgC: cute.Tensor,
        sC: cute.Tensor,
        tTR_rAcc: cute.Tensor,
        tiled_copy_r2s,
        tRS_rC: cute.Tensor,
        tRS_sC: cute.Tensor,
        epi_tile: cute.Tile,
        tmem_full_mbar: cute.Pointer,
        tmem_empty_mbar: cute.Pointer,
        tile_consumer_mbar: cute.Pointer,
        tile_producer_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        cross_seam_mbar: cute.Pointer,
        mScatter: cute.Tensor,
        scatter_ptr_smem_ptr: cute.Pointer,
        elem_size_bytes_c: cutlass.Constexpr[int],
        split_sizes: cute.Tensor,
        cluster_cta_rank: cutlass.Int32,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        a_global_scale_inv_smem_ptr=None,
        a_global_scale_inv_ptr: cutlass.Int64 = 0,
        b_global_scale_inv_ptr: cutlass.Int64 = 0,
        use_global_scale_inv: cutlass.Constexpr[bool] = False,
    ):
        accum_cnt_tile = cutlass.Int32(0)

        if cutlass.const_expr(self.STATIC_SCHEDULER):
            scheduler = StaticTileScheduler.create_consumer(self.NUM_CTAS)
        else:
            scheduler = DynamicTileScheduler.create_consumer(
                tile_consumer_mbar,
                tile_id_smem_ptr,
                self.NUM_CTAS,
                self.NUM_TILE_BUFFERS,
            )
        tile_idx = scheduler.initial_work_tile_info()
        visitor = GroupedProblemVisitor.create(
            split_sizes,
            M,
            N,
            K,
            G,
            self.problem_type,
            self.BLOCK_SIZE_M,
            self.BLOCK_SIZE_N,
            self.BLOCK_SIZE_K,
            self.force_n_major,
            self.num_n_clusters,
            local_rank,
            self.world_size,
            self.SWAP_AB,
        )
        work = visitor.get_work(tile_idx)

        while work.is_valid_tile:
            g = work.group_idx
            m_size = work.m_size
            n_size = work.n_size
            num_k_tiles = work.num_k_tiles
            cm_start = work.split_prefix
            b_scale = _load_group_global_scale_inv(
                b_global_scale_inv_ptr, g, use_global_scale_inv
            )

            if work.is_valid_tile:
                while work.is_valid_tile and work.group_idx == g:
                    tmem_buf, tmem_phase = _get_bufidx_phase(
                        accum_cnt_tile, self.NUM_TMEM_BUFFERS
                    )
                    tile_buf, _ = _get_bufidx_phase(
                        accum_cnt_tile, self.NUM_TILE_BUFFERS
                    )

                    tile_m_idx = work.tile_m_idx
                    tile_n_idx = work.tile_n_idx

                    defer_a_scale_stage: cutlass.Constexpr[bool] = (
                        self.format is NVFP4
                        and self.mode == self.COMBINE_SWIGLU_FWD_MODE
                    )
                    if cutlass.const_expr(
                        use_global_scale_inv and not defer_a_scale_stage
                    ):
                        _stage_a_global_scale_inv(
                            tidx,
                            a_global_scale_inv_ptr,
                            a_global_scale_inv_smem_ptr,
                            cm_start,
                            n_size if self.SWAP_AB else m_size,
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
                    NUM_MMA_ATOMS_M = cute.size(tCtAcc_base.shape, mode=[1])
                    NUM_MMA_ATOMS_N = cute.size(tCtAcc_base.shape, mode=[2])
                    prefetched_scatter_ptr = cutlass.Int64(0)
                    prefetched_scatter_ptr_valid = cutlass.Boolean(False)
                    if cutlass.const_expr(self.prefetch_first_scatter_ptr):
                        (
                            prefetched_scatter_ptr,
                            prefetched_scatter_ptr_valid,
                        ) = _combine_epilog_get_scatter_ptr(
                            tidx,
                            sC,
                            mScatter,
                            tile_m_idx,
                            tile_n_idx,
                            0,
                            0,
                            0,
                            NUM_MMA_ATOMS_M,
                            NUM_MMA_ATOMS_N,
                            cm_start,
                            m_size,
                            n_size,
                            cluster_cta_rank,
                            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                            EPILOG_WG_THREADS=self.EPILOG_WG_THREADS,
                            NUM_CTAS=self.NUM_CTAS,
                            SWAP_AB=self.SWAP_AB,
                        )
                    cute.arch.mbarrier_wait(tmem_full_mbar + tmem_buf, tmem_phase)

                    if cutlass.const_expr(use_global_scale_inv and defer_a_scale_stage):
                        _stage_a_global_scale_inv(
                            tidx,
                            a_global_scale_inv_ptr,
                            a_global_scale_inv_smem_ptr,
                            cm_start,
                            n_size if self.SWAP_AB else m_size,
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
                    if cutlass.const_expr(use_global_scale_inv):
                        a_scales_smem = cute.make_tensor(
                            a_global_scale_inv_smem_ptr,
                            cute.make_layout(
                                (self._a_global_scale_inv_smem_size,), stride=(1,)
                            ),
                        )

                    for mma_m_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_M):
                        for mma_n_idx in cutlass.range_constexpr(NUM_MMA_ATOMS_N):
                            (
                                _tiled_copy_t2r_atom,
                                tTR_tAcc_base,
                                _tTR_rAcc_atom,
                            ) = self._epilog_tmem_copy_and_partition(
                                tidx=tidx,
                                tAcc=tCtAcc_base,
                                gC_mnl=tCgC,
                                epi_tile=epi_tile,
                                use_2cta_instrs=self.NUM_CTAS == 2,
                                mma_m_idx=mma_m_idx,
                                mma_n_idx=mma_n_idx,
                            )
                            tTR_tAcc = tTR_tAcc_base[
                                (None, None, None, None, None, tmem_buf)
                            ]
                            tTR_tAcc = cute.group_modes(
                                tTR_tAcc, 3, cute.rank(tTR_tAcc)
                            )
                            atom_m: cutlass.Constexpr[int] = (
                                self.cta_tile_shape_mnk[0] // NUM_MMA_ATOMS_M
                            )
                            atom_n: cutlass.Constexpr[int] = (
                                self.cta_tile_shape_mnk[1] // NUM_MMA_ATOMS_N
                            )
                            c_acc = cute.make_identity_tensor((atom_m, atom_n))
                            c_acc_epi = cute.flat_divide(c_acc, epi_tile)
                            tTR_cAcc = _tiled_copy_t2r_atom.get_slice(tidx).partition_D(
                                c_acc_epi
                            )
                            tTR_cAcc = cute.group_modes(
                                tTR_cAcc, 3, cute.rank(tTR_cAcc)
                            )
                            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                            # OVERLAPPING_ACCUM early-release threshold:
                            # identical to the base blockscaled epilogue, but
                            # the post-stage action is scatter instead of C TMA.
                            cols_per_subtile: cutlass.Constexpr[int] = cute.size(
                                epi_tile[1]
                            )
                            iter_acc_early_release: cutlass.Constexpr[int] = (
                                self._num_sf_tmem_cols + cols_per_subtile - 1
                            ) // cols_per_subtile - 1
                            for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                                real_subtile_idx = subtile_idx
                                if cutlass.const_expr(self.OVERLAPPING_ACCUM):
                                    if tmem_buf == 0:
                                        real_subtile_idx = subtile_cnt - 1 - subtile_idx

                                if (
                                    cutlass.const_expr(
                                        self.prefetch_first_scatter_ptr
                                        and mma_m_idx == 0
                                        and mma_n_idx == 0
                                    )
                                    and real_subtile_idx == 0
                                ):
                                    if prefetched_scatter_ptr_valid:
                                        # The async-shared fence and warpgroup barrier
                                        # below publish this pointer with the C tile.
                                        cute.arch.store(
                                            scatter_ptr_smem_ptr + tidx,
                                            prefetched_scatter_ptr,
                                            ss="cta",
                                        )
                                else:
                                    self._combine_epilog_load_scatter_ptrs(
                                        tidx=tidx,
                                        sC=sC,
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
                                        cluster_cta_rank=cluster_cta_rank,
                                    )

                                if num_k_tiles != 0:
                                    cute.copy(
                                        _tiled_copy_t2r_atom,
                                        tTR_tAcc[(None, None, None, real_subtile_idx)],
                                        tTR_rAcc,
                                    )

                                if cutlass.const_expr(self.OVERLAPPING_ACCUM):
                                    release_overlap_atom = cutlass.Boolean(True)
                                    if cutlass.const_expr(NUM_MMA_ATOMS_N > 1):
                                        if cutlass.const_expr(mma_n_idx == 0):
                                            release_overlap_atom = tmem_buf != 0
                                        elif cutlass.const_expr(
                                            mma_n_idx == NUM_MMA_ATOMS_N - 1
                                        ):
                                            release_overlap_atom = tmem_buf == 0
                                        else:
                                            release_overlap_atom = cutlass.Boolean(
                                                False
                                            )
                                    if (
                                        subtile_idx == iter_acc_early_release
                                    ) and release_overlap_atom:
                                        cute.arch.fence_view_async_tmem_load()
                                        warp_idx_local = cute.arch.make_warp_uniform(
                                            cute.arch.warp_idx()
                                        )
                                        # Synchronize the whole epilogue warpgroup
                                        # before releasing the shared TMEM seam: the
                                        # per-thread fence above only orders THIS
                                        # lane's t2r loads, so without this barrier
                                        # the elected lane can arrive cross_seam_mbar
                                        # (freeing the seam to the peer MMA) while
                                        # sibling lanes are still reading it, letting
                                        # the MMA overwrite the seam under them.
                                        cute.arch.barrier(
                                            barrier_id=_BAR_EPILOG_SYNC,
                                            number_of_threads=self.EPILOG_WG_THREADS,
                                        )
                                        if warp_idx_local == self.EPILOG_WARP_IDS[0]:
                                            with cute.arch.elect_one():
                                                if cutlass.const_expr(
                                                    self.NUM_CTAS == 2
                                                ):
                                                    cute.arch.mbarrier_arrive(
                                                        cross_seam_mbar,
                                                        peer_cta_rank_in_cluster=0,
                                                    )
                                                else:
                                                    cute.arch.mbarrier_arrive(
                                                        cross_seam_mbar
                                                    )

                                tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
                                if num_k_tiles == 0:
                                    tRS_rAcc.store(cute.zeros_like(tRS_rAcc.load()))
                                if cutlass.const_expr(use_global_scale_inv):
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
                                tRS_rC.store(tRS_rAcc.load().to(sC.element_type))
                                cute.copy(
                                    tiled_copy_r2s,
                                    tRS_rC,
                                    tRS_sC[(None, None, None, 0)],
                                )

                                cute.arch.fence_proxy(kind="async.shared", space="cta")
                                cute.arch.barrier(
                                    barrier_id=_BAR_EPILOG_SYNC,
                                    number_of_threads=self.EPILOG_WG_THREADS,
                                )
                                self._combine_epilog_scatter_smem_tile(
                                    tidx=tidx,
                                    sC=sC,
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
                                    elem_size_bytes_c=elem_size_bytes_c,
                                    cluster_cta_rank=cluster_cta_rank,
                                )
                                cute.arch.barrier(
                                    barrier_id=_BAR_EPILOG_SYNC,
                                    number_of_threads=self.EPILOG_WG_THREADS,
                                )

                    warp_idx_local = cute.arch.make_warp_uniform(cute.arch.warp_idx())
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.barrier(
                        barrier_id=_BAR_EPILOG_SYNC,
                        number_of_threads=self.EPILOG_WG_THREADS,
                    )
                    if warp_idx_local == self.EPILOG_WARP_IDS[0]:
                        with cute.arch.elect_one():
                            if cutlass.const_expr(self.NUM_CTAS == 2):
                                cute.arch.mbarrier_arrive(
                                    tmem_empty_mbar + tmem_buf,
                                    peer_cta_rank_in_cluster=0,
                                )
                            else:
                                cute.arch.mbarrier_arrive(tmem_empty_mbar + tmem_buf)

                    scheduler.consumer_release_tile(
                        tile_producer_mbar,
                        tile_buf,
                        cluster_cta_rank,
                        warp_idx_local == self.EPILOG_WARP_IDS[0],
                    )

                    accum_cnt_tile += cutlass.Int32(1)
                    tile_idx = scheduler.advance_consumer(accum_cnt_tile)
                    work = visitor.get_work(tile_idx)

    @cute.kernel
    def combine_kernel(  # noqa: C901
        self,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tma_atom_sfa: cute.CopyAtom,
        tma_atom_sfb: cute.CopyAtom,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mSFA: cute.Tensor,
        mSFB: cute.Tensor,
        mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        sfb_tma_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        split_sizes: cute.Tensor,
        counter_ptr: cute.Pointer,
        mScatter: cute.Tensor,
        mCombineSwigluX: cute.Tensor,
        mCombineSwigluY: cute.Tensor,
        mCombineSwigluWorkCounter: cute.Tensor,
        mCombineSwigluDoneCounter: cute.Tensor,
        mCombineSwigluRowQWords: cute.Tensor,
        mCombineSwigluRowScale: cute.Tensor,
        mCombineSwigluRowGlobalScaleInv: cute.Tensor,
        mCombineSwigluColQWords: cute.Tensor,
        mCombineSwigluColScale: cute.Tensor,
        tensormaps: cute.Tensor,
        elem_size_bytes_c: cutlass.Constexpr[int],
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        use_device_tensormaps: cutlass.Constexpr[bool],
        activation_buffer_base_ptr: cutlass.Int64,
        activation_buffer_size_bytes: cutlass.Int64,
        activation_offsets: cute.Tensor,
        use_activation_buffer: cutlass.Constexpr[bool],
        conditional_execution: cute.Tensor,
        use_conditional_execution: cutlass.Constexpr[bool],
        b_global_scale_inv_ptr: cutlass.Int64,
        use_global_scale_inv: cutlass.Constexpr[bool],
        nvfp4_recip_lut_ptr: cutlass.Int64,
    ):
        if cutlass.const_expr(use_conditional_execution):
            if conditional_execution[0] == cutlass.Int32(0):
                thread_exit()
        if cutlass.const_expr(use_activation_buffer):
            activation_rows = _activation_buffer_rows(split_sizes, G)
            source_element_size = cutlass.Int64(
                cutlass.const_expr(mCombineSwigluX.element_type.width // 8)
            )
            if cutlass.const_expr(self.mode == self.COMBINE_SWIGLU_FWD_MODE):
                has_activation_rows = cutlass.Int64(activation_rows > cutlass.Int32(0))
                source_x_byte_extent = (
                    has_activation_rows
                    * (
                        cutlass.Int64(2) * cutlass.Int64(activation_rows)
                        - cutlass.Int64(1)
                    )
                    * cutlass.Int64(K)
                    * source_element_size
                )
                source_y_byte_extent = source_x_byte_extent
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
            activation_value_count = cutlass.Int64(activation_rows) * cutlass.Int64(K)
            activation_q_byte_extent = (
                activation_value_count
                * cutlass.Int64(cutlass.const_expr(self.format.a_dtype.width))
                // cutlass.Int64(8)
            )
            row_scale_byte_extent = _activation_buffer_row_scale_storage_byte_extent(
                activation_rows,
                K,
                sf_vec_size=self.sf_vec_size,
                sf_dtype_width=self.format.sf_dtype.width,
            )
            col_scale_byte_extent = _activation_buffer_col_scale_storage_byte_extent(
                activation_rows,
                K,
                sf_vec_size=self.sf_vec_size,
                sf_dtype_width=self.format.sf_dtype.width,
            )
            mCombineSwigluX = _activation_buffer_tensor_with_byte_extent(
                mCombineSwigluX,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                ACTIVATION_SOURCE_X_OFFSET,
                mCombineSwigluX.element_type,
                source_x_byte_extent,
            )
            mCombineSwigluY = _activation_buffer_tensor_with_byte_extent(
                mCombineSwigluY,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                ACTIVATION_SOURCE_Y_OFFSET,
                mCombineSwigluY.element_type,
                source_y_byte_extent,
            )
            mCombineSwigluRowQWords = _activation_buffer_tensor_with_byte_extent(
                mCombineSwigluRowQWords,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                ACTIVATION_A_Q_OFFSET,
                cutlass.Uint32,
                activation_q_byte_extent,
            )
            mCombineSwigluRowScale = _activation_buffer_tensor_with_byte_extent(
                mCombineSwigluRowScale,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                ACTIVATION_A_SCALE_OFFSET,
                cutlass.Uint8,
                row_scale_byte_extent,
            )
            if cutlass.const_expr(use_global_scale_inv):
                mCombineSwigluRowGlobalScaleInv = (
                    _activation_buffer_row_global_scale_tensor(
                        mCombineSwigluRowGlobalScaleInv,
                        activation_buffer_base_ptr,
                        activation_buffer_size_bytes,
                        activation_offsets,
                        activation_rows,
                        K,
                        sf_vec_size=self.sf_vec_size,
                        sf_dtype_width=self.format.sf_dtype.width,
                    )
                )
            mCombineSwigluColQWords = _activation_buffer_tensor_with_byte_extent(
                mCombineSwigluColQWords,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                ACTIVATION_COL_Q_OFFSET,
                cutlass.Uint32,
                activation_q_byte_extent,
            )
            mCombineSwigluColScale = _activation_buffer_tensor_with_byte_extent(
                mCombineSwigluColScale,
                activation_buffer_base_ptr,
                activation_buffer_size_bytes,
                activation_offsets,
                ACTIVATION_COL_SCALE_OFFSET,
                cutlass.Uint8,
                col_scale_byte_extent,
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
        scatter_ptr_smem_ptr = storage.scatter_ptr_smem.data_ptr()
        a_global_scale_inv_smem_ptr = (
            storage.a_global_scale_inv.data_ptr() if use_global_scale_inv else None
        )
        combine_swiglu_tile_smem_ptr = None
        if cutlass.const_expr(
            self.mode == self.COMBINE_SWIGLU_FWD_MODE
            or self.mode == self.COMBINE_SWIGLU_BWD_MODE
        ):
            combine_swiglu_tile_smem_ptr = storage.dispatch_quant_tile_smem.data_ptr()
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
        else:
            cute.arch.barrier(
                barrier_id=_BAR_FULL_CTA_SYNC,
                number_of_threads=self.THREADS_PER_CTA,
            )

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
            barrier_id=_BAR_FULL_CTA_SYNC,
            number_of_threads=self.THREADS_PER_CTA,
        )

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(c_smem_layout_staged)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sSFB_tma = storage.sSFB.get_tensor(sfb_tma_smem_layout_staged)

        gA = cute.local_tile(
            mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB = cute.local_tile(
            mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC = cute.local_tile(
            mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gSFA = cute.local_tile(
            mSFA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB = cute.local_tile(
            mSFB,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
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
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)
        tCgSFA = thr_mma.partition_A(gSFA)
        tCgSFB = thr_mma_sfb.partition_B(gSFB)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
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
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
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
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((self.BLOCK_SIZE_M, self.BLOCK_SIZE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(
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

        num_tma_load_bytes, _, _ = self._num_tma_load_bytes(
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

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        if cutlass.const_expr(
            self.mode == self.COMBINE_SWIGLU_FWD_MODE
            or self.mode == self.COMBINE_SWIGLU_BWD_MODE
        ):
            tile_producer_hook_tensor = mCombineSwigluDoneCounter
        else:
            tile_producer_hook_tensor = split_sizes

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
        )

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
            )
            self._tma_producer_body(
                pipeline=pipeline,
                problem=problem,
                sync=sync,
                tensormap_manager=tensormap_manager,
                tensormaps=tensormaps,
                tile_producer_hook_tensor=tile_producer_hook_tensor,
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
            pipeline = GroupedGemmMmaPipeline(
                tiled_mma=tiled_mma,
                tCrA=tCrA,
                tCrB=tCrB,
                tCtAcc_base=tCtAcc_base,
                tCtSFA=tCtSFA,
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
                _tTR_tAcc_base_00,
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
            ) = self._epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, tidx, sC)
            self._combine_epilog_consumer_body(
                tidx=tidx,
                tCtAcc_base=tCtAcc_base,
                tCgC=tCgC,
                sC=sC,
                tTR_rAcc=tTR_rAcc,
                tiled_copy_r2s=tiled_copy_r2s,
                tRS_rC=tRS_rC,
                tRS_sC=tRS_sC,
                epi_tile=epi_tile,
                tmem_full_mbar=tmem_full_mbar,
                tmem_empty_mbar=tmem_empty_mbar,
                tile_consumer_mbar=tile_consumer_mbar,
                tile_producer_mbar=tile_producer_mbar,
                tile_id_smem_ptr=tile_id_smem_ptr,
                cross_seam_mbar=cross_seam_mbar,
                mScatter=mScatter,
                scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
                elem_size_bytes_c=elem_size_bytes_c,
                split_sizes=split_sizes,
                cluster_cta_rank=cluster_cta_rank,
                G=G,
                M=M,
                N=N,
                K=K,
                local_rank=local_rank,
                a_global_scale_inv_smem_ptr=a_global_scale_inv_smem_ptr,
                a_global_scale_inv_ptr=cutlass.Int64(
                    mCombineSwigluRowGlobalScaleInv.iterator.toint()
                ),
                b_global_scale_inv_ptr=b_global_scale_inv_ptr,
                use_global_scale_inv=use_global_scale_inv,
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
        else:
            if cutlass.const_expr(self.mode == self.COMBINE_SWIGLU_FWD_MODE):
                if warp_idx >= self.COMBINE_SWIGLU_QUANT_FIRST_WARP:
                    if cutlass.const_expr(self.format is NVFP4):
                        quant_warp = warp_idx - self.COMBINE_SWIGLU_QUANT_FIRST_WARP
                        if quant_warp < cutlass.Int32(
                            self.nvfp4_swiglu_quant_groups
                            * self.DISPATCH_QUANT_WARPS_PER_GROUP
                        ):
                            recip_lut_ptr = cute.make_ptr(
                                cutlass.Float32,
                                nvfp4_recip_lut_ptr,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            mNvfp4RecipLut = cute.make_tensor(
                                recip_lut_ptr,
                                cute.make_ordered_layout((256,), order=(0,)),
                            )
                            self._combine_swiglu_nvfp4_row_quant_producer_body(
                                mWorkCounter=mCombineSwigluWorkCounter,
                                mDoneCounter=mCombineSwigluDoneCounter,
                                tile_smem_ptr=combine_swiglu_tile_smem_ptr,
                                mFwdX=mCombineSwigluX,
                                mFwdY=mCombineSwigluY,
                                mRowQWords=mCombineSwigluRowQWords,
                                mRowScale=mCombineSwigluRowScale,
                                mRowGlobalScaleInv=mCombineSwigluRowGlobalScaleInv,
                                mNvfp4RecipLut=mNvfp4RecipLut,
                                split_sizes=split_sizes,
                                G=G,
                                M=M,
                                local_rank=local_rank,
                                precomputed_swiglu=self.combine_precomputed_swiglu,
                            )
                    else:
                        _combine_swiglu_fwd_quant_producer_body(
                            params_from_kernel(CombineSwigluQuantParams, self),
                            mWorkCounter=mCombineSwigluWorkCounter,
                            mDoneCounter=mCombineSwigluDoneCounter,
                            tile_smem_ptr=combine_swiglu_tile_smem_ptr,
                            mFwdX=mCombineSwigluX,
                            mFwdY=mCombineSwigluY,
                            mRowQWords=mCombineSwigluRowQWords,
                            mRowScale=mCombineSwigluRowScale,
                            mColQWords=mCombineSwigluColQWords,
                            mColScale=mCombineSwigluColScale,
                            split_sizes=split_sizes,
                            G=G,
                            M=M,
                            K=K,
                            total_tiles=cutlass.Int32(0),
                            local_rank=local_rank,
                        )
            elif cutlass.const_expr(self.mode == self.COMBINE_SWIGLU_BWD_MODE):
                if warp_idx >= self.COMBINE_SWIGLU_QUANT_FIRST_WARP:
                    _combine_swiglu_bwd_quant_producer_body(
                        params_from_kernel(CombineSwigluQuantParams, self),
                        mWorkCounter=mCombineSwigluWorkCounter,
                        mDoneCounter=mCombineSwigluDoneCounter,
                        tile_smem_ptr=combine_swiglu_tile_smem_ptr,
                        mDz=mCombineSwigluX,
                        mH1=mCombineSwigluY,
                        mRowQWords=mCombineSwigluRowQWords,
                        mRowScale=mCombineSwigluRowScale,
                        mDxyColQWords=mCombineSwigluColQWords,
                        mDxyColScale=mCombineSwigluColScale,
                        split_sizes=split_sizes,
                        G=G,
                        M=M,
                        K=K,
                        local_rank=local_rank,
                    )

        if cutlass.const_expr(self.NUM_CTAS == 2):
            # Keep every active thread alive until peer DSMEM accesses complete.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

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
        scatter_ptrs: cute.Tensor,
        combine_swiglu_x: cute.Tensor,
        combine_swiglu_y: cute.Tensor,
        dispatch_quant_work_counter: cute.Tensor,
        dispatch_quant_done_counter: cute.Tensor,
        dispatch_quant_row_q_words: cute.Tensor,
        dispatch_quant_row_scale: cute.Tensor,
        dispatch_quant_row_global_scale_inv: cute.Tensor,
        dispatch_quant_col_q_words: cute.Tensor,
        dispatch_quant_col_scale: cute.Tensor,
        pointer_strides: BlockscaledPointerStrideArgs,
        elem_sizes: cutlass.Constexpr[tuple[int, int, int]],
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        N: cutlass.Int32,
        K: cutlass.Int32,
        local_rank: cutlass.Int32,
        num_clusters: int,
        dispatch_quant_total_tiles: cutlass.Int32,
        dispatch_quant_counter_zero_count: cutlass.Int32,
        use_device_tensormaps: cutlass.Constexpr[bool],
        activation_buffer_base_ptr: cutlass.Int64,
        activation_buffer_size_bytes: cutlass.Int64,
        activation_offsets: cute.Tensor,
        use_activation_buffer: cutlass.Constexpr[bool],
        conditional_execution: cute.Tensor,
        use_conditional_execution: cutlass.Constexpr[bool],
        b_global_scale_inv_ptr: cutlass.Int64,
        use_global_scale_inv: cutlass.Constexpr[bool],
        nvfp4_recip_lut_ptr: cutlass.Int64,
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
        tensor_c_eff = _transpose_c_if_swap(tensor_c, self.SWAP_AB)

        self.a_dtype = eff_tensor_a.element_type
        self.b_dtype = eff_tensor_b.element_type
        self.c_dtype = tensor_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(eff_tensor_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(eff_tensor_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(tensor_c_eff)

        self._setup_attributes()

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            eff_tensor_a.shape, self.sf_vec_size
        )
        tensor_sfa_view = cute.make_tensor(eff_tensor_sfa.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            eff_tensor_b.shape, self.sf_vec_size
        )
        tensor_sfb_view = cute.make_tensor(eff_tensor_sfb.iterator, sfb_layout)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
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
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
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

        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self.tiled_mma.thr_id
        )
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
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, self.tiled_mma.thr_id
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

        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(tensor_c_eff.shape), self.epi_tile
        )
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
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
            use_global_scale_inv,
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
                extra_counter_ptr=dispatch_quant_work_counter.iterator,
                extra_counter_count=dispatch_quant_counter_zero_count,
                zero_extra_counters=True,
                activation_buffer_base_ptr=activation_buffer_base_ptr,
                activation_buffer_size_bytes=activation_buffer_size_bytes,
                activation_offsets=activation_offsets,
                use_activation_buffer=use_activation_buffer,
            ).launch(
                grid=(G, 1, 1),
                block=(128, 1, 1),
                stream=stream,
            )

        if cutlass.const_expr(self.mode == self.DISPATCH_MODE):
            self.dispatch_kernel(
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
                mDispatchQuantGatherPtrs=scatter_ptrs,
                mDispatchQuantWorkCounter=dispatch_quant_work_counter,
                mDispatchQuantDoneCounter=dispatch_quant_done_counter,
                mDispatchQuantRowQWords=dispatch_quant_row_q_words,
                mDispatchQuantRowScale=dispatch_quant_row_scale,
                mDispatchQuantRowGlobalScaleInv=dispatch_quant_row_global_scale_inv,
                mDispatchQuantColQWords=dispatch_quant_col_q_words,
                mDispatchQuantColScale=dispatch_quant_col_scale,
                G=G,
                M=M,
                N=N,
                K=K,
                local_rank=local_rank,
                dispatch_quant_total_tiles=dispatch_quant_total_tiles,
                use_device_tensormaps=use_device_tensormaps,
                activation_buffer_base_ptr=activation_buffer_base_ptr,
                activation_buffer_size_bytes=activation_buffer_size_bytes,
                activation_offsets=activation_offsets,
                use_activation_buffer=use_activation_buffer,
                conditional_execution=conditional_execution,
                use_conditional_execution=use_conditional_execution,
                b_global_scale_inv_ptr=b_global_scale_inv_ptr,
                use_global_scale_inv=use_global_scale_inv,
            ).launch(
                grid=grid,
                block=(self.THREADS_PER_CTA, 1, 1),
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
            )
            return

        self.combine_kernel(
            tma_atom_a=tma_atom_a,
            tma_atom_b=tma_atom_b,
            tma_atom_sfa=tma_atom_sfa,
            tma_atom_sfb=tma_atom_sfb,
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
            c_smem_layout_staged=self.c_smem_layout_staged,
            epi_tile=self.epi_tile,
            cluster_layout_vmnk=self.cluster_layout_vmnk,
            cluster_layout_sfb_vmnk=self.cluster_layout_sfb_vmnk,
            tiled_mma=self.tiled_mma,
            tiled_mma_sfb=self.tiled_mma_sfb,
            split_sizes=split_sizes,
            counter_ptr=counter.iterator,
            mScatter=scatter_ptrs,
            mCombineSwigluX=combine_swiglu_x,
            mCombineSwigluY=combine_swiglu_y,
            mCombineSwigluWorkCounter=dispatch_quant_work_counter,
            mCombineSwigluDoneCounter=dispatch_quant_done_counter,
            mCombineSwigluRowQWords=dispatch_quant_row_q_words,
            mCombineSwigluRowScale=dispatch_quant_row_scale,
            mCombineSwigluRowGlobalScaleInv=dispatch_quant_row_global_scale_inv,
            mCombineSwigluColQWords=dispatch_quant_col_q_words,
            mCombineSwigluColScale=dispatch_quant_col_scale,
            tensormaps=tensormaps,
            elem_size_bytes_c=elem_size_bytes_c,
            G=G,
            M=M,
            N=N,
            K=K,
            local_rank=local_rank,
            use_device_tensormaps=use_device_tensormaps,
            activation_buffer_base_ptr=activation_buffer_base_ptr,
            activation_buffer_size_bytes=activation_buffer_size_bytes,
            activation_offsets=activation_offsets,
            use_activation_buffer=use_activation_buffer,
            conditional_execution=conditional_execution,
            use_conditional_execution=use_conditional_execution,
            b_global_scale_inv_ptr=b_global_scale_inv_ptr,
            use_global_scale_inv=use_global_scale_inv,
            nvfp4_recip_lut_ptr=nvfp4_recip_lut_ptr,
        ).launch(
            grid=grid,
            block=(self.THREADS_PER_CTA, 1, 1),
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
