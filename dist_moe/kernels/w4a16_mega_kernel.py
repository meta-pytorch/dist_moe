# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused W4A16 expert-parallel MoE forward in one persistent kernel.

One launch runs dispatch, FC13 with SwiGLU, FC2 and combine for the local
experts. Weights are NVFP4 (E2M1 codes, E4M3 16-element block scales and a
per-expert FP32 inverse global scale); activations stay BF16.

Warp roles (one CTA, 512 threads):

* warps 0-3: first the gather (copy routed token rows from the peer ranks'
  dispatch buffers into the local ``x`` rows, the dist BF16 dispatch
  protocol), then the epilogue. An FC13 tile scales the accumulator by its
  expert's inverse global scale, applies SwiGLU from the staged tile, writes
  h2 and counts the token tile done. An FC2 tile scales, stages and scatters
  each token row to its source rank's combine buffer.
* warp 4, MMA: BF16 ``tcgen05.mma`` with A (decoded weights) in TMEM; waits
  for the stage's decoded A and its B tile.
* warp 5, weight TMA: packed FP4 weights and scales of every K stage. It
  never waits on a dependency, so weights stream and decode ahead.
* warp 6, token TMA: the B tile of every K stage. An FC13 tile waits for its
  token tile's gather, an FC2 tile for every FC13 tile of its token tile.
* warps 8-15, two transform warpgroups: decode each stage's FP4 tile into a
  BF16 TMEM A stage, alternating stages.

Tiles are statically scheduled: CTA ``c`` takes linear tiles ``c, c + grid,
...`` over all FC13 tiles followed by all FC2 tiles, so a CTA finishes its
FC13 tiles before its first FC2 tile and the FC2 waits cannot deadlock.
SWAP_AB holds throughout: features are the MMA M (rows of TMEM), tokens N.
"""

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.mixed_input_helpers as mixed_input_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils.mixed_input_helpers import TransformMode

from ._nvfp4_decode import decode_nvfp4_fragment_to_bf16
from .combine_scatter import CombineScatterMixin
from .dispatch_quant import _dispatch_gather_producer_body
from .grouped_gemm_kernel import (
    _BAR_EPILOG_SYNC,
    _BAR_FULL_CTA_SYNC,
    _transpose_c_if_swap,
    EPILOG_WARP_IDS,
    GroupedGemmKernel,
    MMA_WARP_ID,
    TMA_WARP_ID,
)
from .grouped_gemm_tiles import split_atom_accum
from .grouped_problem_visitor import GroupedProblemVisitor
from .tile_scheduler import _FPROP, _get_bufidx_phase, stage_expert_metadata

# W4A16 (NVFP4 weights x BF16 activations) adds two transform warpgroups that
# decode TMA stages' FP4 weight tiles into the BF16 A stages the MMA reads.
# CUTLASS's shared-memory transform partition covers exactly one warpgroup
# (128 threads), so the two warpgroups take alternating stages.
W4A16_TRANSFORM_WARP_IDS: tuple[int, ...] = tuple(range(8, 16))
W4A16_NUM_TRANSFORM_GROUPS: int = 2
W4A16_TRANSFORM_GROUP_THREADS: int = 128
W4A16_THREADS_PER_CTA: int = 32 * 16  # 512
# K columns a transform warpgroup decodes and stores per pass.
W4A16_DECODE_K = 128
W4A16_SF_VEC_SIZE: int = 16
# Registers per thread decoded in one step: whole NVFP4 blocks only.
W4A16_TRANSFORM_FRAGMENT: int = 32


class _W4A16TransformPartition(NamedTuple):
    """One transform thread's decode partitions (``_w4a16_transform_setup``).

    The SMEM and TMEM stages are divided into ``num_passes`` passes of
    W4A16_DECODE_K columns; the register tensors hold one pass, the ``_load``
    and ``_store`` views in ``num_fragments`` fragments of ``frag_tiler``.
    """

    transform_group: cutlass.Int32
    num_passes: int
    num_fragments: int
    frag_tiler: cute.Layout
    dst_copy_a: cute.TiledCopy
    tAsA_input: cute.Tensor
    tA_transform: cute.Tensor
    tSsS: cute.Tensor
    tSrS_copy: cute.Tensor
    tSrS_load: cute.Tensor
    tArA_load: cute.Tensor
    tArA_transform: cute.Tensor
    tArA_transform_store: cute.Tensor


# The weight TMA is the base kernel's TMA_WARP_ID; the token TMA takes the
# first idle warp.
MEGA_TMA_B_WARP_ID: int = 6
# Gate/up interleave of the W13 rows: NVFP4 inference preparation interleaves
# groups of eight (`_interleave_w13_rows`).
MEGA_SWIGLU_GROUP: int = 8
# Gather rows per work item and K columns per gather chunk.
MEGA_GATHER_ROWS: int = 16
MEGA_GATHER_K: int = 256
_BAR_GATHER: int = 6


class _MegaSchedule(NamedTuple):
    """Static tile schedule shared by the warp roles.

    CTA ``cta`` of ``grid`` takes linear tiles ``cta + i * grid``: FC13 tiles
    for ``i < n13``, then FC2 tiles (numbered from ``total13``, the FC13 tile
    count of all CTAs) for ``n13 <= i < n_all``. FC13 is ``two_f`` features
    by ``D`` in ``k13`` K stages, FC2 ``D`` by ``F`` in ``k2``.
    """

    split_sizes: cute.Tensor
    n13: cutlass.Int32
    n_all: cutlass.Int32
    total13: cutlass.Int32
    cta: cutlass.Int32
    grid: cutlass.Int32
    two_f: cutlass.Int32
    D: cutlass.Int32
    F: cutlass.Int32
    k13: cutlass.Int32
    k2: cutlass.Int32
    local_rank: cutlass.Int32
    G: int
    WORLD_SIZE: int


class _MegaEpilogue(NamedTuple):
    """Epilogue thread state shared by every ``_epilogue_tile`` call.

    The partitions and copies of ``_epilogue``; ``sC_tokens`` is the staged
    tile as (tokens, features, stage).
    """

    tidx: cutlass.Int32
    tTR_tAcc_base: cute.Tensor
    tTR_rAcc: cute.Tensor
    tiled_copy_t2r: cute.TiledCopy
    tiled_copy_r2s: cute.TiledCopy
    tRS_rC: cute.Tensor
    tRS_sC: cute.Tensor
    sC_tokens: cute.Tensor
    tmem_full_mbar: cute.Pointer
    tmem_empty_mbar: cute.Pointer
    mH2Words: cute.Tensor
    h2_done_addr: cutlass.Int64
    scatter_ptrs_addr: cutlass.Int64
    padding_dst_addr: cutlass.Int64
    scatter_ptr_smem_ptr: cute.Pointer


class W4A16MegaKernel(CombineScatterMixin, GroupedGemmKernel):
    """Fused W4A16 EP MoE forward (see module docstring)."""

    EPILOG_WG_THREADS: int = 32 * len(EPILOG_WARP_IDS)
    # The dispatch rows carry padding sentinels (gather and combine).
    has_padding_sentinels: bool = True
    # Gather loads in flight per lane.
    GATHER_LOADS_PER_LANE: int = 16

    def __init__(self, config: dict, *, world_size: int):
        """Kernel for one W4A16 tile ``config`` and EP ``world_size``."""
        if config["NUM_CTAS"] != 1:
            raise ValueError("W4A16 Mega requires a one-CTA config")
        if split_atom_accum(config["NUM_TMEM_BUFFERS"], config["NUM_MMAS"]):
            raise ValueError("W4A16 Mega does not support split-atom accumulation")
        if config.get("W4A16_A_STAGES", 0) < 1:
            raise ValueError("W4A16 Mega needs config['W4A16_A_STAGES'] >= 1")
        super().__init__(
            config=config,
            problem_type=_FPROP,
            world_size=world_size,
            swap_ab=True,
        )
        if self.BLOCK_SIZE_N % MEGA_GATHER_ROWS:
            raise ValueError("token tile must be a multiple of the gather rows")
        self.threads_per_cta = W4A16_THREADS_PER_CTA
        # BF16 A stages in tensor memory, after the accumulators.
        self.W4A16_A_STAGES: int = config["W4A16_A_STAGES"]
        # Programmatic dependent launch: the prologue (mbarrier init, TMEM
        # alloc) overlaps the previous kernel's tail.
        self.USE_PDL: bool = bool(config.get("USE_PDL", False))
        self.W4A16_SWIGLU_FAST_MATH: bool = bool(
            config.get("W4A16_SWIGLU_FAST_MATH", False)
        )

    def _setup_attributes(self):
        """Base layouts, with A decoded into TMEM and a plain staged C layout."""
        super()._setup_attributes()
        # The MMA reads A, the decoded BF16 weights, from tensor memory.
        self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            (self.BLOCK_SIZE_M // self.NUM_MMAS, self.BLOCK_SIZE_N),
            tcgen05.OperandSource.TMEM,
        )
        # BF16 A stages: the layout the decoded weights take in TMEM.
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.W4A16_A_STAGES,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.NUM_SMEM_BUFFERS,
        )
        # One packed FP4 weight stage and one E4M3 scale stage per K stage,
        # filled by TMA; the transform warps decode them into the A stages.
        self.a_fp4_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma,
            self.mma_tiler,
            self.a_input_dtype,
            self.NUM_SMEM_BUFFERS,
        )
        self.sf_smem_layout_tma_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma,
            self.mma_tiler,
            W4A16_SF_VEC_SIZE,
            self.NUM_SMEM_BUFFERS,
        )
        # The transform reads the same scale bytes through a per-element
        # view: one scale broadcast over its 16 K values.
        sf_transform_layout = blockscaled_utils.make_smem_layout_sf(
            (self.cta_tile_shape_mnk[0], self.mma_tiler[2]),
            W4A16_SF_VEC_SIZE,
            self.NUM_SMEM_BUFFERS,
        )
        self.sf_smem_layout_transform_staged = cute.make_composed_layout(
            cute.make_swizzle(0, 4, 3), 0, sf_transform_layout
        )
        # TMEM: accumulators first (one FP32 column per output column), then
        # the BF16 A stages (two values per 32-bit column).
        self.w4a16_acc_tmem_cols = self.NUM_TMEM_BUFFERS * self.BLOCK_SIZE_N
        a_cols = self.W4A16_A_STAGES * self.BLOCK_SIZE_K * self.a_dtype.width // 32
        total = self.w4a16_acc_tmem_cols + a_cols
        if total > 512:
            raise ValueError(
                f"W4A16 TMEM needs {total} columns: {self.NUM_TMEM_BUFFERS} "
                f"accumulators x {self.BLOCK_SIZE_N} + {a_cols} for A"
            )
        self.num_tmem_alloc_cols = max(32, 1 << (total - 1).bit_length())
        # Neither output is TMA-stored: keep the staged tile in a plain layout
        # with the feature (M) axis contiguous, so a token's run of features
        # is one 16-byte SMEM load. A 16-byte skew keeps rows on distinct banks.
        epi_m = cute.size(self.epi_tile[0])
        epi_n = cute.size(self.epi_tile[1])
        col_stride = epi_m + 8
        self.c_smem_layout_staged = cute.make_layout(
            (epi_m, epi_n, 1), stride=(1, col_stride, epi_n * col_stride)
        )

    def _make_shared_storage(self, a_dtype, b_dtype, c_dtype, G: int):  # noqa: ANN001
        """SMEM struct: staged C, B stages, scatter pointers and barriers."""
        NUM_SMEM = self.NUM_SMEM_BUFFERS
        b_smem_elems = cute.cosize(self.b_smem_layout_staged.outer)
        c_smem_elems = cute.cosize(self.c_smem_layout_staged)
        scatter_rows = cute.size(self.epi_tile[1])
        n_acc = self.NUM_TMEM_BUFFERS
        n_trans = self.W4A16_A_STAGES

        @cute.struct
        class MegaSharedStorage:
            sC: cute.struct.Align[cute.struct.MemRange[c_dtype, c_smem_elems], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[b_dtype, b_smem_elems], 1024]
            split_sizes_smem: cute.struct.MemRange[cutlass.Int32, G]
            gather_tile_idx_smem: cute.struct.MemRange[cutlass.Int32, 1]
            scatter_ptr_smem: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, scatter_rows], 128
            ]
            smem_empty_mbar: cute.struct.MemRange[cutlass.Int64, NUM_SMEM]
            smem_full_mbar: cute.struct.MemRange[cutlass.Int64, NUM_SMEM]
            b_full_mbar: cute.struct.MemRange[cutlass.Int64, NUM_SMEM]
            tmem_full_mbar: cute.struct.MemRange[cutlass.Int64, n_acc]
            tmem_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_acc]
            trans_full_mbar: cute.struct.MemRange[cutlass.Int64, n_trans]
            trans_empty_mbar: cute.struct.MemRange[cutlass.Int64, n_trans]
            tmem_holding_buf: cutlass.Int32

        return MegaSharedStorage

    # ------------------------------------------------------------------
    # Host-side entry: build both problems' TMA atoms and launch.
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        x_rows: cute.Tensor,  # [M, D] bf16 gathered tokens (FC13 B), (M, D, 1)
        h2_rows: cute.Tensor,  # [M, F] bf16 SwiGLU output (FC2 B), (M, F, 1)
        split_sizes: cute.Tensor,  # [G] int32
        counters: cute.Tensor,  # int32, zeroed: [1 + 2 * token_tile_cap]
        w13_ptr: cute.Pointer,  # FP4 [G, 2F, D/2]
        w13_sf_ptr: cute.Pointer,
        w13_alpha_ptr: cute.Pointer,  # f32 [G]
        w2_ptr: cute.Pointer,  # FP4 [G, D, F/2]
        w2_sf_ptr: cute.Pointer,
        w2_alpha_ptr: cute.Pointer,
        h2_words_ptr: cute.Pointer,  # same storage as h2_rows, as u32
        gather_ptrs_addr: cutlass.Int64,  # int64 [M]
        scatter_ptrs_addr: cutlass.Int64,  # int64 [M]
        padding_src_addr: cutlass.Int64,
        padding_dst_addr: cutlass.Int64,
        x_rows_addr: cutlass.Int64,
        token_tile_cap: cutlass.Int32,
        G: cutlass.Constexpr[int],
        M: cutlass.Int32,
        D: cutlass.Int32,
        F: cutlass.Int32,
        local_rank: cutlass.Int32,
        num_ctas: int,
        stream: cuda.CUstream,
    ):
        """Build both problems' weight tensors and TMA atoms, then launch."""
        tensor_w13, tensor_sf13, alpha13 = self._w4a16_weight_tensors(
            w13_ptr, w13_sf_ptr, w13_alpha_ptr, F * cutlass.Int32(2), D, G
        )
        tensor_w2, tensor_sf2, alpha2 = self._w4a16_weight_tensors(
            w2_ptr, w2_sf_ptr, w2_alpha_ptr, D, F, G
        )
        h2_words = F // cutlass.Int32(2)
        tensor_h2_words = cute.make_tensor(
            h2_words_ptr,
            cute.make_layout((M, h2_words), stride=(cutlass.Int64(h2_words), 1)),
        )
        # The epilogue layout and partitions follow the transposed FC13 output.
        tensor_c = _transpose_c_if_swap(h2_rows, True)

        self.a_input_dtype = tensor_w13.element_type
        self.a_dtype = x_rows.element_type
        self.b_dtype = x_rows.element_type
        self.c_dtype = h2_rows.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(tensor_w13).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(x_rows).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(tensor_c)
        self._setup_attributes()

        # (atom, tensor[, sf atom, sf tensor]) for W13, W2, x rows, h2 rows.
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), self.tiled_mma.thr_id)
        atoms = [
            self._w4a16_tma_atoms_a(a_op, tensor_w13, tensor_sf13),
            self._w4a16_tma_atoms_a(a_op, tensor_w2, tensor_sf2),
        ]
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), self.tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        for rows in (x_rows, h2_rows):
            atoms.append(
                cute.nvgpu.make_tiled_tma_atom_B(
                    b_op,
                    rows,
                    b_smem_layout,
                    self.mma_tiler,
                    self.tiled_mma,
                    self.cluster_layout_vmnk.shape,
                )
            )

        self.shared_storage = self._make_shared_storage(
            self.a_dtype, self.b_dtype, self.c_dtype, G
        )
        self.mega_kernel(
            *atoms[0],
            *atoms[1],
            *atoms[2],
            *atoms[3],
            tensor_c,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.a_fp4_smem_layout_staged,
            self.sf_smem_layout_tma_staged,
            self.sf_smem_layout_transform_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tiled_mma,
            split_sizes,
            alpha13,
            alpha2,
            tensor_h2_words,
            counters,
            gather_ptrs_addr,
            scatter_ptrs_addr,
            padding_src_addr,
            padding_dst_addr,
            x_rows_addr,
            token_tile_cap,
            G,
            D,
            F,
            local_rank,
            self.schedule_world_size,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            use_pdl=self.USE_PDL,
        )

    # ------------------------------------------------------------------
    # Device entry.
    # ------------------------------------------------------------------
    @cute.kernel
    def mega_kernel(  # noqa: C901
        self,
        tma_atom_a13: cute.CopyAtom,
        mA13: cute.Tensor,
        tma_atom_sf13: cute.CopyAtom,
        mSF13: cute.Tensor,
        tma_atom_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_atom_sf2: cute.CopyAtom,
        mSF2: cute.Tensor,
        tma_atom_bx: cute.CopyAtom,
        mBx: cute.Tensor,
        tma_atom_bh: cute.CopyAtom,
        mBh: cute.Tensor,
        mC: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        a_fp4_smem_layout_staged: cute.ComposedLayout,
        sf_smem_layout_tma_staged,
        sf_smem_layout_transform_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tiled_mma: cute.TiledMma,
        split_sizes: cute.Tensor,
        alpha13: cute.Tensor,
        alpha2: cute.Tensor,
        mH2Words: cute.Tensor,
        counters: cute.Tensor,
        gather_ptrs_addr: cutlass.Int64,
        scatter_ptrs_addr: cutlass.Int64,
        padding_src_addr: cutlass.Int64,
        padding_dst_addr: cutlass.Int64,
        x_rows_addr: cutlass.Int64,
        token_tile_cap: cutlass.Int32,
        G: cutlass.Constexpr[int],
        D: cutlass.Int32,
        F: cutlass.Int32,
        local_rank: cutlass.Int32,
        WORLD_SIZE: cutlass.Constexpr[int],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        NUM_SMEM_BUFFERS = self.NUM_SMEM_BUFFERS
        NUM_TMEM_BUFFERS = self.NUM_TMEM_BUFFERS
        BLOCK_SIZE_M = self.BLOCK_SIZE_M
        BLOCK_SIZE_N = self.BLOCK_SIZE_N
        BLOCK_SIZE_K = self.BLOCK_SIZE_K

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ab_full_mbar = storage.smem_full_mbar.data_ptr()  # weights + scales
        b_full_mbar = storage.b_full_mbar.data_ptr()  # token tile
        ab_empty_mbar = storage.smem_empty_mbar.data_ptr()
        tmem_full_mbar = storage.tmem_full_mbar.data_ptr()
        tmem_empty_mbar = storage.tmem_empty_mbar.data_ptr()
        trans_full_mbar = storage.trans_full_mbar.data_ptr()
        trans_empty_mbar = storage.trans_empty_mbar.data_ptr()
        if warp_idx == EPILOG_WARP_IDS[0]:
            with cute.arch.elect_one():
                for i in range(NUM_SMEM_BUFFERS):
                    cute.arch.mbarrier_init(ab_empty_mbar + i, 1)
                    cute.arch.mbarrier_init(ab_full_mbar + i, 1)
                    cute.arch.mbarrier_init(b_full_mbar + i, 1)
                for i in range(NUM_TMEM_BUFFERS):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, 1)
                for i in range(self.W4A16_A_STAGES):
                    cute.arch.mbarrier_init(
                        trans_full_mbar + i, W4A16_TRANSFORM_GROUP_THREADS
                    )
                    cute.arch.mbarrier_init(trans_empty_mbar + i, 1)
        cute.arch.mbarrier_init_fence()
        if warp_idx == EPILOG_WARP_IDS[0]:
            cute.arch.alloc_tmem(self.num_tmem_alloc_cols, storage.tmem_holding_buf)
        if cutlass.const_expr(self.USE_PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        split_sizes = stage_expert_metadata(
            split_sizes, storage.split_sizes_smem.data_ptr(), G, synchronize=False
        )
        cute.arch.barrier(
            barrier_id=_BAR_FULL_CTA_SYNC, number_of_threads=self.threads_per_cta
        )

        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(c_smem_layout_staged)
        sA_fp4 = smem.allocate_tensor(
            element_type=self.a_input_dtype,
            layout=a_fp4_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_fp4_smem_layout_staged.inner,
        )
        sSF_transform = smem.allocate_tensor(
            element_type=cutlass.Float8E4M3FN,
            layout=sf_smem_layout_transform_staged.outer,
            byte_alignment=128,
            swizzle=sf_smem_layout_transform_staged.inner,
        )
        sSF_tma = cute.make_tensor(sSF_transform.iterator, sf_smem_layout_tma_staged)

        # Tile counts: every expert group has ceil(rows / BN) token tiles and
        # ceil(features / BM) feature tiles per problem.
        two_f = F * cutlass.Int32(2)
        num_m13 = (two_f + cutlass.Int32(BLOCK_SIZE_M - 1)) // cutlass.Int32(
            BLOCK_SIZE_M
        )
        num_m2 = (D + cutlass.Int32(BLOCK_SIZE_M - 1)) // cutlass.Int32(BLOCK_SIZE_M)
        token_tiles = cutlass.Int32(0)
        for g in cutlass.range(G, unroll=1):
            token_tiles += (
                cutlass.Int32(split_sizes[g]) + cutlass.Int32(BLOCK_SIZE_N - 1)
            ) // cutlass.Int32(BLOCK_SIZE_N)
        total13 = token_tiles * num_m13
        total = total13 + token_tiles * num_m2
        cta = cute.arch.block_idx()[0]
        grid = cute.arch.grid_dim()[0]
        n13 = cutlass.Int32(0)
        if total13 > cta:
            n13 = (total13 - cta + grid - cutlass.Int32(1)) // grid
        n_all = cutlass.Int32(0)
        if total > cta:
            n_all = (total - cta + grid - cutlass.Int32(1)) // grid
        k13 = (D + cutlass.Int32(BLOCK_SIZE_K - 1)) // cutlass.Int32(BLOCK_SIZE_K)
        k2 = (F + cutlass.Int32(BLOCK_SIZE_K - 1)) // cutlass.Int32(BLOCK_SIZE_K)
        sched = _MegaSchedule(
            split_sizes=split_sizes,
            n13=n13,
            n_all=n_all,
            total13=total13,
            cta=cta,
            grid=grid,
            two_f=two_f,
            D=D,
            F=F,
            k13=k13,
            k2=k2,
            local_rank=local_rank,
            G=G,
            WORLD_SIZE=WORLD_SIZE,
        )

        gather_counter_addr = cutlass.Int64(counters.iterator.toint())
        gather_done_addr = gather_counter_addr + cutlass.Int64(4)
        h2_done_addr = gather_done_addr + cutlass.Int64(token_tile_cap) * cutlass.Int64(
            4
        )

        # Register budget (512 threads, 128 each at launch): MMA and TMA
        # release 32 to the gather/epilogue; the transform warpgroups keep 128.
        if warp_idx < 4:
            cute.arch.setmaxregister_increase(160)
        elif warp_idx < 8:
            cute.arch.setmaxregister_decrease(96)

        if warp_idx == TMA_WARP_ID:
            self._producer_a(
                sched,
                tma_atom_a13,
                mA13,
                tma_atom_sf13,
                mSF13,
                tma_atom_a2,
                mA2,
                tma_atom_sf2,
                mSF2,
                tiled_mma,
                sA_fp4,
                sSF_tma,
                ab_full_mbar,
                ab_empty_mbar,
            )
        elif warp_idx == MEGA_TMA_B_WARP_ID:
            self._producer_b(
                sched,
                tma_atom_bx,
                mBx,
                tma_atom_bh,
                mBh,
                tiled_mma,
                sB,
                b_full_mbar,
                ab_empty_mbar,
                gather_done_addr,
                h2_done_addr,
                num_m13,
            )
        elif warp_idx == MMA_WARP_ID:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            acc_shape = tiled_mma.partition_shape_C((BLOCK_SIZE_M, BLOCK_SIZE_N))
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, NUM_TMEM_BUFFERS)
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCrA = self._w4a16_tmem_a(tiled_mma, tmem_ptr, a_smem_layout_staged)
            tCrB = tiled_mma.make_fragment_B(sB)
            accum_cnt_smem = cutlass.Int32(0)
            for i in cutlass.range(n_all, unroll=1):
                num_k = k2
                if i < n13:
                    num_k = k13
                tmem_buf, tmem_phase = _get_bufidx_phase(i, NUM_TMEM_BUFFERS)
                # The token tile has its own full barrier; K is never empty.
                accum_cnt_smem = self._w4a16_mma_tile(
                    tiled_mma,
                    tCrA,
                    tCrB,
                    tCtAcc_base,
                    trans_full_mbar,
                    trans_empty_mbar,
                    ab_empty_mbar,
                    tmem_full_mbar,
                    tmem_empty_mbar,
                    num_k,
                    tmem_buf,
                    tmem_phase,
                    accum_cnt_smem,
                    NUM_SMEM_BUFFERS,
                    b_full_mbar=b_full_mbar,
                    ALLOW_EMPTY_K=False,
                )
        elif warp_idx >= W4A16_TRANSFORM_WARP_IDS[0]:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            # Every K stage of the CTA's tiles, in the MMA's order.
            num_stages = n13 * k13 + (n_all - n13) * k2
            xf = self._w4a16_transform_setup(
                tiled_mma,
                sA_fp4,
                sSF_transform,
                self._w4a16_tmem_a(tiled_mma, tmem_ptr, a_smem_layout_staged),
                tidx,
            )
            for stage in cutlass.range(num_stages, unroll=1):
                self._w4a16_transform_stage(
                    stage,
                    xf,
                    ab_full_mbar,
                    trans_full_mbar,
                    trans_empty_mbar,
                    NUM_SMEM_BUFFERS,
                )
        elif warp_idx < 4:
            _dispatch_gather_producer_body(
                gather_counter_addr_i64=gather_counter_addr,
                a_buff_counter_addr_i64=gather_done_addr,
                gather_a_ptrs_addr_i64=gather_ptrs_addr,
                padding_row_addr_i64=padding_src_addr,
                a_local_addr_i64=x_rows_addr,
                stride_am_elems=D,
                gather_tile_idx_smem_ptr=storage.gather_tile_idx_smem.data_ptr(),
                split_sizes=split_sizes,
                elem_size_bytes_a=2,
                G=G,
                K=D,
                local_rank=local_rank,
                WORLD_SIZE=WORLD_SIZE,
                AG_BLOCK_SIZE_M=MEGA_GATHER_ROWS,
                AG_BLOCK_SIZE_K=MEGA_GATHER_K,
                BLOCK_SIZE_M=BLOCK_SIZE_N,
                has_padding_sentinels=self.has_padding_sentinels,
                _BAR_GATHER_WG_INTERNAL=_BAR_GATHER,
                _DIST_GATHER_WG_THREADS=self.EPILOG_WG_THREADS,
                SEGMENT_SIGNALS=False,
                LOADS_PER_LANE=self.GATHER_LOADS_PER_LANE,
                THREAD_BASE=0,
            )
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            self._epilogue(
                sched,
                tidx,
                tiled_mma,
                tmem_ptr,
                mC,
                sC,
                epi_tile,
                tmem_full_mbar,
                tmem_empty_mbar,
                alpha13,
                alpha2,
                mH2Words,
                h2_done_addr,
                scatter_ptrs_addr,
                padding_dst_addr,
                storage.scatter_ptr_smem.data_ptr(),
            )
            cute.arch.barrier(
                barrier_id=_BAR_EPILOG_SYNC, number_of_threads=self.EPILOG_WG_THREADS
            )
            if warp_idx == EPILOG_WARP_IDS[0]:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, self.num_tmem_alloc_cols)

    # ------------------------------------------------------------------
    # Schedule.
    # ------------------------------------------------------------------
    @cute.jit
    def _visitor(self, sched: _MegaSchedule, N, K):
        """Tile visitor of one problem: ``N`` features by ``K`` per expert."""
        # FPROP visitors never read the problem M.
        return GroupedProblemVisitor.create(
            sched.split_sizes,
            cutlass.Int32(0),
            N,
            K,
            sched.G,
            _FPROP,
            self.BLOCK_SIZE_M,
            self.BLOCK_SIZE_N,
            self.BLOCK_SIZE_K,
            False,
            1,
            sched.local_rank,
            sched.WORLD_SIZE,
            True,
        )

    @cute.jit
    def _phase_schedule(self, sched: _MegaSchedule, fc13: cutlass.Constexpr[bool]):
        """``(visitor, first, last, num_k)`` of FC13 if ``fc13``, else FC2.

        The CTA's tiles of the problem are ``first <= i < last``, each of
        ``num_k`` K stages.
        """
        if cutlass.const_expr(fc13):
            visitor = self._visitor(sched, sched.two_f, sched.D)
            first = 0
            last = sched.n13
            num_k = sched.k13
        else:
            visitor = self._visitor(sched, sched.D, sched.F)
            first = sched.n13
            last = sched.n_all
            num_k = sched.k2
        return visitor, first, last, num_k

    @cute.jit
    def _phase_tile_idx(self, sched: _MegaSchedule, fc13: cutlass.Constexpr[bool], i):
        """Tile index within FC13 (``fc13``) or FC2 of the CTA's ``i``-th tile."""
        tile = sched.cta + i * sched.grid
        if cutlass.const_expr(not fc13):
            tile = tile - sched.total13
        return tile

    @cute.jit
    def _wait_counter(self, addr: cutlass.Int64, target: cutlass.Int32):
        """Warp-wide spin until the int32 counter at ``addr`` reaches ``target``."""
        ptr = cute.make_ptr(cutlass.Int32, addr, cute.AddressSpace.gmem)
        ready = False
        while not ready:
            v = cute.arch.load(ptr, cutlass.Int32, sem="acquire", scope="gpu")
            ready = v >= target
        cute.arch.fence_proxy("async.global")

    # ------------------------------------------------------------------
    # Roles.
    # ------------------------------------------------------------------
    @cute.jit
    def _issue_a(
        self,
        tma_atom_a,
        tAgA,
        tAsA,
        tma_atom_sf,
        tSFgSF,
        tSFsSF,
        tile_m_idx,
        g,
        num_k_tiles,
        num_bytes,
        a_full_mbar,
        ab_empty_mbar,
        accum_cnt_smem,
    ):
        """Weight and scale TMA loads of every K stage of one tile."""
        for kk in cutlass.range(num_k_tiles, unroll=1):
            buf, phase = _get_bufidx_phase(accum_cnt_smem, self.NUM_SMEM_BUFFERS)
            cute.arch.mbarrier_wait(ab_empty_mbar + buf, phase ^ 1)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(a_full_mbar + buf, num_bytes)
            cute.copy(
                tma_atom_a,
                tAgA[(None, tile_m_idx, kk, g)],
                tAsA[(None, buf)],
                tma_bar_ptr=a_full_mbar + buf,
            )
            cute.copy(
                tma_atom_sf,
                tSFgSF[(None, tile_m_idx, kk, g)],
                tSFsSF[(None, buf)],
                tma_bar_ptr=a_full_mbar + buf,
            )
            accum_cnt_smem += cutlass.Int32(1)
        return accum_cnt_smem

    @cute.jit
    def _issue_b(
        self,
        tma_atom_b,
        tBgB,
        tBsB,
        num_k_tiles,
        num_bytes,
        b_full_mbar,
        ab_empty_mbar,
        accum_cnt_smem,
    ):
        """Token TMA loads of every K stage of one tile."""
        for kk in cutlass.range(num_k_tiles, unroll=1):
            buf, phase = _get_bufidx_phase(accum_cnt_smem, self.NUM_SMEM_BUFFERS)
            cute.arch.mbarrier_wait(ab_empty_mbar + buf, phase ^ 1)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(b_full_mbar + buf, num_bytes)
            cute.copy(
                tma_atom_b,
                tBgB[(None, kk)],
                tBsB[(None, buf)],
                tma_bar_ptr=b_full_mbar + buf,
            )
            accum_cnt_smem += cutlass.Int32(1)
        return accum_cnt_smem

    @cute.jit
    def _partition_a(self, tiled_mma, tma_atom, mA, sA_tma):
        """TMA partition of an (N, K, G) weight or scale tensor."""
        gA = cute.local_tile(
            mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sA_tma, 0, 3),
            cute.group_modes(tiled_mma.get_slice(0).partition_A(gA), 0, 3),
        )
        return tAsA, tAgA

    @cute.jit
    def _partition_weights(
        self, tiled_mma, tma_atom_a, mA, tma_atom_sf, mSF, sA_fp4, sSF_tma
    ):
        """TMA partitions ``(tAsA, tAgA, tSFsSF, tSFgSF)`` of one problem."""
        tAsA, tAgA = self._partition_a(tiled_mma, tma_atom_a, mA, sA_fp4)
        tSFsSF, tSFgSF = self._partition_a(tiled_mma, tma_atom_sf, mSF, sSF_tma)
        tSFsSF = cute.filter_zeros(tSFsSF)
        tSFgSF = cute.filter_zeros(tSFgSF)
        return tAsA, tAgA, tSFsSF, tSFgSF

    @cute.jit
    def _producer_a(
        self,
        sched: _MegaSchedule,
        tma_atom_a13,
        mA13,
        tma_atom_sf13,
        mSF13,
        tma_atom_a2,
        mA2,
        tma_atom_sf2,
        mSF2,
        tiled_mma,
        sA_fp4,
        sSF_tma,
        a_full_mbar,
        ab_empty_mbar,
    ):
        """Weights and scales of every K stage, FC13 tiles then FC2 tiles."""
        parts13 = self._partition_weights(
            tiled_mma, tma_atom_a13, mA13, tma_atom_sf13, mSF13, sA_fp4, sSF_tma
        )
        parts2 = self._partition_weights(
            tiled_mma, tma_atom_a2, mA2, tma_atom_sf2, mSF2, sA_fp4, sSF_tma
        )
        a_bytes = cute.size_in_bytes(
            self.a_input_dtype, cute.slice_(sA_fp4.layout, (None, None, None, 0))
        ) + cute.size_in_bytes(
            cutlass.Float8E4M3FN, cute.slice_(sSF_tma.layout, (None, None, None, 0))
        )
        accum_cnt_smem = cutlass.Int32(0)
        accum_cnt_smem = self._producer_a_phase(
            True,
            sched,
            tma_atom_a13,
            tma_atom_sf13,
            parts13,
            a_bytes,
            a_full_mbar,
            ab_empty_mbar,
            accum_cnt_smem,
        )
        self._producer_a_phase(
            False,
            sched,
            tma_atom_a2,
            tma_atom_sf2,
            parts2,
            a_bytes,
            a_full_mbar,
            ab_empty_mbar,
            accum_cnt_smem,
        )

    @cute.jit
    def _producer_a_phase(
        self,
        fc13: cutlass.Constexpr[bool],
        sched: _MegaSchedule,
        tma_atom_a,
        tma_atom_sf,
        parts,
        a_bytes,
        a_full_mbar,
        ab_empty_mbar,
        accum_cnt_smem,
    ):
        """Weights and scales of the CTA's FC13 (``fc13``) or FC2 tiles.

        ``parts`` is the problem's ``_partition_weights``. Returns the K
        stage count ``accum_cnt_smem`` after the problem's tiles.
        """
        tAsA, tAgA, tSFsSF, tSFgSF = parts
        visitor, first, last, num_k = self._phase_schedule(sched, fc13)
        for i in cutlass.range(first, last, unroll=1):
            work = visitor.get_work(self._phase_tile_idx(sched, fc13, i))
            accum_cnt_smem = self._issue_a(
                tma_atom_a,
                tAgA,
                tAsA,
                tma_atom_sf,
                tSFgSF,
                tSFsSF,
                work.tile_m_idx,
                work.group_idx,
                num_k,
                a_bytes,
                a_full_mbar,
                ab_empty_mbar,
                accum_cnt_smem,
            )
        return accum_cnt_smem

    @cute.jit
    def _producer_b(
        self,
        sched: _MegaSchedule,
        tma_atom_bx,
        mBx,
        tma_atom_bh,
        mBh,
        tiled_mma,
        sB,
        b_full_mbar,
        ab_empty_mbar,
        gather_done_addr,
        h2_done_addr,
        num_m13,
    ):
        """Token tiles of every K stage: gathered x for FC13, h2 for FC2."""
        b_bytes = cute.size_in_bytes(
            self.b_dtype, cute.slice_(sB.layout, (None, None, None, 0))
        )
        thr_mma = tiled_mma.get_slice(0)
        gather_target = cutlass.Int32(self.BLOCK_SIZE_N // MEGA_GATHER_ROWS)
        accum_cnt_smem = cutlass.Int32(0)
        accum_cnt_smem = self._producer_b_phase(
            True,
            sched,
            tma_atom_bx,
            mBx,
            thr_mma,
            sB,
            gather_done_addr,
            gather_target,
            b_bytes,
            b_full_mbar,
            ab_empty_mbar,
            accum_cnt_smem,
        )
        self._producer_b_phase(
            False,
            sched,
            tma_atom_bh,
            mBh,
            thr_mma,
            sB,
            h2_done_addr,
            num_m13,
            b_bytes,
            b_full_mbar,
            ab_empty_mbar,
            accum_cnt_smem,
        )

    @cute.jit
    def _producer_b_phase(
        self,
        fc13: cutlass.Constexpr[bool],
        sched: _MegaSchedule,
        tma_atom_b,
        mB,
        thr_mma,
        sB,
        done_addr,
        target,
        b_bytes,
        b_full_mbar,
        ab_empty_mbar,
        accum_cnt_smem,
    ):
        """Token tiles of the CTA's FC13 (``fc13``) or FC2 tiles, rows of ``mB``.

        A tile first waits until its token tile's int32 counter at
        ``done_addr`` reaches ``target``. Returns the K stage count
        ``accum_cnt_smem`` after the problem's tiles.
        """
        visitor, first, last, num_k = self._phase_schedule(sched, fc13)
        for i in cutlass.range(first, last, unroll=1):
            work = visitor.get_work(self._phase_tile_idx(sched, fc13, i))
            tile_n = work.tile_n_idx
            self._wait_counter(
                done_addr
                + cutlass.Int64(work.n_tile_prefix + tile_n) * cutlass.Int64(4),
                target,
            )
            tBsB, tBgB = self._w4a16_partition_b_rows(
                thr_mma,
                tma_atom_b,
                mB,
                sB,
                work.split_prefix + tile_n * cutlass.Int32(self.BLOCK_SIZE_N),
            )
            accum_cnt_smem = self._issue_b(
                tma_atom_b,
                tBgB,
                tBsB,
                num_k,
                b_bytes,
                b_full_mbar,
                ab_empty_mbar,
                accum_cnt_smem,
            )
        return accum_cnt_smem

    @cute.jit
    def _epilogue(
        self,
        sched: _MegaSchedule,
        tidx,
        tiled_mma,
        tmem_ptr,
        mC,
        sC,
        epi_tile,
        tmem_full_mbar,
        tmem_empty_mbar,
        alpha13,
        alpha2,
        mH2Words,
        h2_done_addr,
        scatter_ptrs_addr,
        padding_dst_addr,
        scatter_ptr_smem_ptr,
    ):
        """Epilogue of every tile, FC13 tiles then FC2 tiles."""
        BLOCK_SIZE_M = self.BLOCK_SIZE_M
        BLOCK_SIZE_N = self.BLOCK_SIZE_N
        acc_shape = tiled_mma.partition_shape_C((BLOCK_SIZE_M, BLOCK_SIZE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.NUM_TMEM_BUFFERS)
        )
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        gC = cute.local_tile(
            mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        tCgC = tiled_mma.get_slice(0).partition_C(gC)
        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self._epilog_tmem_copy_and_partition(
            tidx=tidx,
            tAcc=tCtAcc_base,
            gC_mnl=tCgC,
            epi_tile=epi_tile,
            use_2cta_instrs=False,
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, sC.element_type)
        tiled_copy_r2s, tRS_rC, tRS_sC = self._epilog_smem_copy_and_partition(
            tiled_copy_t2r=tiled_copy_t2r, tTR_rC=tTR_rC, tidx=tidx, sC=sC
        )
        ep = _MegaEpilogue(
            tidx=tidx,
            tTR_tAcc_base=tTR_tAcc_base,
            tTR_rAcc=tTR_rAcc,
            tiled_copy_t2r=tiled_copy_t2r,
            tiled_copy_r2s=tiled_copy_r2s,
            tRS_rC=tRS_rC,
            tRS_sC=tRS_sC,
            sC_tokens=_transpose_c_if_swap(sC, True),  # (tokens, features, stage)
            tmem_full_mbar=tmem_full_mbar,
            tmem_empty_mbar=tmem_empty_mbar,
            mH2Words=mH2Words,
            h2_done_addr=h2_done_addr,
            scatter_ptrs_addr=scatter_ptrs_addr,
            padding_dst_addr=padding_dst_addr,
            scatter_ptr_smem_ptr=scatter_ptr_smem_ptr,
        )
        self._epilogue_phase(True, sched, ep, alpha13)
        self._epilogue_phase(False, sched, ep, alpha2)

    @cute.jit
    def _epilogue_phase(
        self,
        fc13: cutlass.Constexpr[bool],
        sched: _MegaSchedule,
        ep: _MegaEpilogue,
        alpha,
    ):
        """Epilogue of the CTA's FC13 (``fc13``) or FC2 tiles, scaled by ``alpha``."""
        visitor, first, last, _ = self._phase_schedule(sched, fc13)
        for i in cutlass.range(first, last, unroll=1):
            work = visitor.get_work(self._phase_tile_idx(sched, fc13, i))
            self._epilogue_tile(fc13, work, i, ep, alpha)

    @cute.jit
    def _epilogue_tile(
        self,
        fc13: cutlass.Constexpr[bool],
        work,
        tile_count,
        ep: _MegaEpilogue,
        alpha,
    ):
        """Scale, stage and store one tile: h2 for FC13, combine rows for FC2.

        ``tile_count``, the CTA's tile count so far, picks the TMEM buffer;
        ``alpha`` holds the problem's FP32 per-expert inverse global scales.
        """
        (
            tidx,
            tTR_tAcc_base,
            tTR_rAcc,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tRS_rC,
            tRS_sC,
            sC_tokens,
            tmem_full_mbar,
            tmem_empty_mbar,
            mH2Words,
            h2_done_addr,
            scatter_ptrs_addr,
            padding_dst_addr,
            scatter_ptr_smem_ptr,
        ) = ep
        BLOCK_SIZE_M = self.BLOCK_SIZE_M
        BLOCK_SIZE_N = self.BLOCK_SIZE_N
        threads = self.EPILOG_WG_THREADS
        tmem_buf, tmem_phase = _get_bufidx_phase(tile_count, self.NUM_TMEM_BUFFERS)
        g = work.group_idx
        tile_m = work.tile_m_idx  # feature tile
        tile_n = work.tile_n_idx  # token tile
        features = work.m_size
        tokens = work.n_size
        cm_start = work.split_prefix
        alpha_g = alpha[g]
        cute.arch.mbarrier_wait(tmem_full_mbar + tmem_buf, tmem_phase)
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, tmem_buf)]
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
            cute.copy(
                tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc
            )
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load() * alpha_g
            tRS_rC.store(acc_vec.to(tRS_rC.element_type))
            if cutlass.const_expr(subtile_idx == subtile_cnt - 1):
                # Every accumulator value is in registers: hand the TMEM
                # buffer back to the MMA before the stores.
                cute.arch.fence_view_async_tmem_load()
                cute.arch.barrier(
                    barrier_id=_BAR_EPILOG_SYNC, number_of_threads=threads
                )
                if tidx < cutlass.Int32(32):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(tmem_empty_mbar + tmem_buf)
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, 0)])
            if cutlass.const_expr(not fc13):
                # Peer base pointers of this subtile's token rows.
                self._combine_epilog_load_scatter_ptrs(
                    tidx,
                    sC_tokens,
                    scatter_ptr_smem_ptr,
                    tile_n,
                    0,
                    cm_start,
                    tokens,
                    scatter_ptrs_addr,
                    padding_dst_addr,
                    BLOCK_SIZE_N,
                    1,
                    1,
                    cutlass.Int32(0),
                    subtile_idx=subtile_idx,
                    subtile_axis_m=True,
                )
            cute.arch.barrier(barrier_id=_BAR_EPILOG_SYNC, number_of_threads=threads)
            if cutlass.const_expr(fc13):
                # One MMA atom and one CTA: the atom offsets are zero.
                self._w4a16_swiglu_subtile(
                    tidx=tidx,
                    sC_stage=sC_tokens[(None, None, 0)],
                    w4a16_h2=mH2Words,
                    tile_m_idx=tile_m,
                    tile_n_idx=tile_n,
                    mma_m_idx=0,
                    mma_n_idx=0,
                    subtile_idx=subtile_idx,
                    NUM_MMA_ATOMS_M=1,
                    NUM_MMA_ATOMS_N=1,
                    cm_start=cm_start,
                    m_size=features,
                    n_size=tokens,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    NUM_CTAS=1,
                    cluster_cta_rank=cutlass.Int32(0),
                    group=MEGA_SWIGLU_GROUP,
                )
            else:
                self._combine_epilog_scatter_smem_tile(
                    tidx,
                    sC_tokens,
                    scatter_ptr_smem_ptr,
                    tile_n,
                    tile_m,
                    0,
                    0,
                    subtile_idx,
                    1,
                    1,
                    cm_start,
                    tokens,
                    features,
                    2,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_M,
                    1,
                    cutlass.Int32(0),
                    subtile_axis_m=True,
                )
            # The next subtile overwrites the staged tile.
            cute.arch.barrier(barrier_id=_BAR_EPILOG_SYNC, number_of_threads=threads)
        if cutlass.const_expr(fc13):
            # Publish this feature tile's h2 rows to the FC2 TMA loads.
            cute.arch.fence_proxy("async.global")
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier(barrier_id=_BAR_EPILOG_SYNC, number_of_threads=threads)
            if tidx == cutlass.Int32(0):
                cute.arch.atomic_add(
                    cute.make_ptr(
                        cutlass.Int32,
                        h2_done_addr
                        + cutlass.Int64(work.n_tile_prefix + tile_n) * cutlass.Int64(4),
                        cute.AddressSpace.gmem,
                    ),
                    cutlass.Int32(1),
                    sem="release",
                    scope="gpu",
                )

    # ------------------------------------------------------------------
    # W4A16 mainloop: weight tensors and TMA atoms, the token rows of a
    # tile, the transform warps' NVFP4 -> BF16 decode into TMEM, the MMA
    # over TMEM A stages, and the fused SwiGLU of an FC13 subtile.
    # ------------------------------------------------------------------
    @cute.jit
    def _w4a16_swiglu_subtile(
        self,
        tidx: cutlass.Int32,
        sC_stage: cute.Tensor,  # (tokens, features) view of the staged tile
        w4a16_h2: cute.Tensor,  # h2 words [M, N / 4]
        tile_m_idx: cutlass.Int32,  # feature tile
        tile_n_idx: cutlass.Int32,  # token tile
        mma_m_idx: cutlass.Constexpr[int],
        mma_n_idx: cutlass.Constexpr[int],
        subtile_idx,
        NUM_MMA_ATOMS_M: cutlass.Constexpr[int],
        NUM_MMA_ATOMS_N: cutlass.Constexpr[int],
        cm_start: cutlass.Int32,
        m_size: cutlass.Int32,  # W13 rows (2F) of this group
        n_size: cutlass.Int32,  # tokens of this group
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        NUM_CTAS: cutlass.Constexpr[int],
        cluster_cta_rank: cutlass.Int32,
        group: cutlass.Constexpr[int],
    ):
        """h2 of one staged SWAP_AB subtile.

        The subtile's columns are W13 rows in interleaved order: groups of
        ``group`` gate rows, then the matching up rows. Every epilogue thread
        takes 8 outputs of one token (8 gate and 8 up values, one 16-byte
        SMEM load each) so all threads stay busy and the exact sigmoid runs
        batched; invalid tokens compute on SMEM garbage and store nothing,
        keeping the sigmoid's warp vote convergent. ``group`` is a multiple
        of the 8-value chunk; at 8 a thread's chunk is a whole group.
        """
        from .blockscaled_quantization_common import (
            _copy_store_u32_words,
            _load_tensor_row_as_b16x2,
        )
        from .swiglu_fragment import _swiglu_words, SwigluH2Epilogue

        chunk: cutlass.Constexpr[int] = 8
        threads: cutlass.Constexpr[int] = 32 * len(EPILOG_WARP_IDS)
        epi_tokens: cutlass.Constexpr[int] = cute.size(sC_stage.shape, mode=[0])
        epi_feats: cutlass.Constexpr[int] = cute.size(sC_stage.shape, mode=[1])
        atom_m: cutlass.Constexpr[int] = BLOCK_SIZE_M // NUM_MMA_ATOMS_M
        atom_n: cutlass.Constexpr[int] = BLOCK_SIZE_N // NUM_MMA_ATOMS_N
        chunks_per_token: cutlass.Constexpr[int] = epi_feats // (2 * chunk)
        total: cutlass.Constexpr[int] = epi_tokens * chunks_per_token
        if cutlass.const_expr(
            NUM_CTAS != 1
            or epi_feats != atom_m
            or group % chunk != 0
            or epi_feats % (2 * group) != 0
            or total % threads != 0
        ):
            raise ValueError(
                "W4A16 SwiGLU needs one CTA and epilogue subtiles spanning the "
                f"feature atom with {threads}-thread multiples of work; got "
                f"{epi_tokens}x{epi_feats} subtile, atom {atom_m}, {NUM_CTAS} CTAs"
            )
        swiglu = SwigluH2Epilogue(
            w4a16_h2, None, self.W4A16_SWIGLU_FAST_MATH, False, 0.0, 0.0
        )
        feat_base = tile_m_idx * cutlass.Int32(BLOCK_SIZE_M) + cutlass.Int32(
            mma_m_idx * atom_m
        )
        token_base = (
            tile_n_idx * cutlass.Int32(BLOCK_SIZE_N)
            + cutlass.Int32(mma_n_idx * atom_n)
            + subtile_idx * cutlass.Int32(epi_tokens)
        )
        for work_pass in cutlass.range_constexpr(total // threads):
            linear = tidx + cutlass.Int32(work_pass * threads)
            local_token = linear // cutlass.Int32(chunks_per_token)
            chunk_idx = linear - local_token * cutlass.Int32(chunks_per_token)
            if cutlass.const_expr(group == chunk):
                group_idx = chunk_idx
            else:
                group_idx = chunk_idx // cutlass.Int32(group // chunk)
                half = chunk_idx - group_idx * cutlass.Int32(group // chunk)
            gate_col = group_idx * cutlass.Int32(2 * group)
            if cutlass.const_expr(group != chunk):
                gate_col = gate_col + half * cutlass.Int32(chunk)
            gate_words = cute.make_rmem_tensor(chunk // 2, cutlass.Int32)
            up_words = cute.make_rmem_tensor(chunk // 2, cutlass.Int32)
            _load_tensor_row_as_b16x2(
                sC_stage,
                local_token,
                gate_col,
                gate_words,
                chunk,
                sC_stage.element_type,
            )
            _load_tensor_row_as_b16x2(
                sC_stage,
                local_token,
                gate_col + cutlass.Int32(group),
                up_words,
                chunk,
                sC_stage.element_type,
            )
            words = cute.make_rmem_tensor(chunk, cutlass.Int32)
            for i in cutlass.range_constexpr(chunk // 2):
                words[i] = gate_words[i]
                words[chunk // 2 + i] = up_words[i]
            out = cute.make_rmem_tensor(chunk // 2, cutlass.Uint32)
            _swiglu_words(words, out, swiglu, sC_stage.element_type)
            token = token_base + local_token
            if (token < n_size) and (feat_base + gate_col < m_size):
                out_col = feat_base // cutlass.Int32(2) + group_idx * cutlass.Int32(
                    group
                )
                if cutlass.const_expr(group != chunk):
                    out_col = out_col + half * cutlass.Int32(chunk)
                _copy_store_u32_words(
                    w4a16_h2,
                    cutlass.Int64(cm_start + token),
                    out_col // cutlass.Int32(2),
                    out,
                    chunk // 2,
                )

    def _w4a16_weight_tensors(
        self,
        weight_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        N: cutlass.Int32,
        K: cutlass.Int32,
        G: int,
    ) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
        """Tensors over one W4A16 GEMM's NVFP4 weights.

        Args:
            weight_ptr: Packed E2M1 weights ``[G, N, K / 2]``.
            sf_ptr: CUBLAS-blocked E4M3 scales over ``(N, K, G)``.
            alpha_ptr: FP32 per-expert inverse global scales ``[G]``.
            N: Output features per expert (the SWAP_AB MMA M).
            K: Contraction length.
            G: Expert count.

        Returns:
            The K-major ``(N, K, G)`` weights, their scales and ``(G,)`` alpha.
        """
        tensor_w = cute.make_tensor(
            weight_ptr,
            cute.make_ordered_layout((N, K, G), order=(1, 0, 2)),
        )
        tensor_sf = cute.make_tensor(
            sf_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(tensor_w.shape, W4A16_SF_VEC_SIZE),
        )
        tensor_alpha = cute.make_tensor(alpha_ptr, cute.make_layout((G,)))
        return tensor_w, tensor_sf, tensor_alpha

    def _w4a16_tma_atoms_a(
        self,
        a_op,  # TMA load operation of operand A
        tensor_w: cute.Tensor,
        tensor_sf: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.CopyAtom, cute.Tensor]:
        """TMA atoms of ``_w4a16_weight_tensors``' weights and scales.

        Call after ``_setup_attributes``: the atoms load one stage of
        ``a_fp4_smem_layout_staged`` and ``sf_smem_layout_tma_staged``.

        Returns:
            ``(tma_atom_a, tma_tensor_a, tma_atom_sf, tma_tensor_sf)``.
        """
        tma_atom_sf, tma_tensor_sf = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            tensor_sf,
            cute.slice_(self.sf_smem_layout_tma_staged, (None, None, None, 0)),
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            tensor_w,
            cute.slice_(self.a_fp4_smem_layout_staged, (None, None, None, 0)),
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        return tma_atom_a, tma_tensor_a, tma_atom_sf, tma_tensor_sf

    @cute.jit
    def _w4a16_partition_b_rows(
        self,
        thr_mma: cute.ThrMma,
        tma_atom_b: cute.CopyAtom,
        mB_rows: cute.Tensor,
        sB: cute.Tensor,
        row0: cutlass.Int32,
    ) -> tuple:
        """TMA partition ``(tBsB, tBgB)`` of the token tile at row ``row0``.

        W4A16 B uses one host descriptor over all token rows ``mB_rows``; a
        tile is addressed by its first row instead of a per-group descriptor.
        ``thr_mma`` is ``tiled_mma.get_slice(0)``.
        """
        gB = cute.local_tile(
            cute.domain_offset((row0, 0, 0), mB_rows),
            cute.slice_(self.mma_tiler, (0, None, None)),
            (0, None, 0),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )
        return tBsB, tBgB

    @cute.jit
    def _w4a16_mma_tile(
        self,
        tiled_mma: cute.TiledMma,
        tCrA: cute.Tensor,  # TMEM (MMA, MMA_M, MMA_K, W4A16_A_STAGES)
        tCrB: cute.Tensor,  # (MMA, MMA_N, MMA_K, NUM_SMEM_BUFFERS)
        tCtAcc_base: cute.Tensor,
        trans_full_mbar: cute.Pointer,
        trans_empty_mbar: cute.Pointer,
        ab_empty_mbar: cute.Pointer,
        tmem_full_mbar: cute.Pointer,
        tmem_empty_mbar: cute.Pointer,
        num_k_tiles: cutlass.Int32,
        tmem_buf: cutlass.Int32,
        tmem_phase: cutlass.Int32,
        accum_cnt_smem: cutlass.Int32,
        NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
        b_full_mbar: cute.Pointer = None,
        ALLOW_EMPTY_K: cutlass.Constexpr[bool] = True,
    ) -> cutlass.Int32:
        """One-CTA ``native_mma_tile`` with decoupled W4A16 A stages.

        K stage ``i`` reads B from TMA stage ``i % NUM_SMEM_BUFFERS`` and
        decoded A from BF16 stage ``i % W4A16_A_STAGES``. ``trans_full`` gates
        both: the transform warpgroup waited for the TMA stage before decoding.
        One commit frees the TMA stage, another frees the BF16 A stage.

        ``b_full_mbar``, when given, is a separate full barrier per TMA stage
        for B, waited after ``trans_full``. ``ALLOW_EMPTY_K=False`` drops the
        ``num_k_tiles == 0`` path for callers whose K is never empty.
        """
        cute.arch.mbarrier_wait(tmem_empty_mbar + tmem_buf, tmem_phase ^ 1)
        tCtAcc_slot = tCtAcc_base[(None, None, None, tmem_buf)]
        for kk in cutlass.range(num_k_tiles, unroll=1):
            smem_buf, smem_phase = _get_bufidx_phase(accum_cnt_smem, NUM_SMEM_BUFFERS)
            a_buf, a_phase = _get_bufidx_phase(accum_cnt_smem, self.W4A16_A_STAGES)
            cute.arch.mbarrier_wait(trans_full_mbar + a_buf, a_phase)
            if cutlass.const_expr(b_full_mbar is not None):
                cute.arch.mbarrier_wait(b_full_mbar + smem_buf, smem_phase)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, kk > 0)
            for kblock_idx in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                cute.gemm(
                    tiled_mma,
                    tCtAcc_slot,
                    tCrA[(None, None, kblock_idx, a_buf)],
                    tCrB[(None, None, kblock_idx, smem_buf)],
                    tCtAcc_slot,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(ab_empty_mbar + smem_buf, None, self.cta_group)
                tcgen05.commit(trans_empty_mbar + a_buf, None, self.cta_group)
            accum_cnt_smem += cutlass.Int32(1)
        if cutlass.const_expr(ALLOW_EMPTY_K):
            # Empty contraction: no MMA was issued, so arrive by hand.
            if num_k_tiles == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tmem_full_mbar + tmem_buf)
            else:
                with cute.arch.elect_one():
                    tcgen05.commit(tmem_full_mbar + tmem_buf, None, self.cta_group)
        else:
            with cute.arch.elect_one():
                tcgen05.commit(tmem_full_mbar + tmem_buf, None, self.cta_group)
        return accum_cnt_smem

    @cute.jit
    def _w4a16_tmem_a(
        self,
        tiled_mma: cute.TiledMma,
        tmem_ptr: cute.Pointer,
        a_smem_layout_staged: cute.ComposedLayout,
    ) -> cute.Tensor:
        """BF16 A stages in TMEM, after the accumulator columns."""
        return cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.w4a16_acc_tmem_cols, dtype=self.a_dtype),
            tiled_mma.make_fragment_A(a_smem_layout_staged.outer).layout,
        )

    @cute.jit
    def _w4a16_transform_setup(
        self,
        tiled_mma: cute.TiledMma,
        sA_fp4: cute.Tensor,  # packed E2M1 TMA stages
        sSF_transform: cute.Tensor,  # E4M3 scales, per-element broadcast view
        tA_tmem: cute.Tensor,  # BF16 A stages in TMEM the MMA reads
        tidx: cutlass.Int32,
    ) -> _W4A16TransformPartition:
        """Partition the decode of transform thread ``tidx``.

        Thread ``t`` of a warpgroup owns TMEM lane ``t`` (row ``t``). The
        partition and fragment decode follow FlashInfer's W4A16 MoE kernel.

        Returns:
            The thread's partitions for ``_w4a16_transform_stage``.
        """
        transform_group = (
            tidx // cutlass.Int32(32) - cutlass.Int32(W4A16_TRANSFORM_WARP_IDS[0])
        ) // cutlass.Int32(4)
        transform_tidx = (
            tidx
            - cutlass.Int32(32 * W4A16_TRANSFORM_WARP_IDS[0])
            - transform_group * cutlass.Int32(W4A16_TRANSFORM_GROUP_THREADS)
        )
        copy_atom_input = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.a_input_dtype, num_bits_per_copy=32
        )
        copy_atom_transform = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
            self.a_dtype,
        )
        src_copy_a, dst_copy_a, tAsA_input, tA_transform = (
            mixed_input_utils.transform_partition(
                tcgen05.OperandSource.TMEM,
                TransformMode.ConvertScale,
                copy_atom_input,
                copy_atom_transform,
                sA_fp4,
                tA_tmem,
                transform_tidx,
            )
        )
        tCsS = tiled_mma.get_slice(0).partition_A(sSF_transform)
        tSsS = src_copy_a.get_slice(transform_tidx).partition_S(tCsS)
        # A stage deeper than W4A16_DECODE_K is decoded and stored in passes
        # of W4A16_DECODE_K columns, which bounds the register fragments.
        assert cute.rank(tAsA_input) == 5, "expected (CPY, M, K, BLOCK, STAGE)"
        num_passes = max(1, self.BLOCK_SIZE_K // W4A16_DECODE_K)
        num_blocks = cute.size(tAsA_input, mode=[3])
        assert num_blocks % num_passes == 0
        pass_tiler = (None, None, None, num_blocks // num_passes, None)
        tAsA_input = cute.logical_divide(tAsA_input, pass_tiler)
        tA_transform = cute.logical_divide(tA_transform, pass_tiler)
        tSsS = cute.logical_divide(tSsS, pass_tiler)
        pass0 = (None, None, None, (None, 0), None)
        s_stage0 = (None,) * (cute.rank(tSsS[pass0]) - 1) + (0,)
        tSsS_layout_per_stage = tSsS[pass0][s_stage0].layout
        tSrS_copy = cute.make_rmem_tensor(
            cute.filter_zeros(tSsS_layout_per_stage).shape, cutlass.Float8E4M3FN
        )
        a_stage0 = (None,) * (cute.rank(tAsA_input[pass0]) - 1) + (0,)
        tArA = cute.make_rmem_tensor(
            tAsA_input[pass0][a_stage0].shape, self.a_input_dtype
        )
        tArA_transform = cute.make_rmem_tensor(
            tAsA_input[pass0][a_stage0].shape, self.a_dtype
        )
        # Decode fragments of whole NVFP4 blocks.
        fragment = min(
            cute.size(cute.coalesce(tAsA_input[pass0].layout), mode=[0]),
            W4A16_TRANSFORM_FRAGMENT,
        )
        assert fragment % W4A16_SF_VEC_SIZE == 0, "fragments cover whole blocks"
        frag_tiler = cute.make_layout(fragment)
        tArA_load = cute.flat_divide(tArA, frag_tiler)
        tArA_load = cute.group_modes(tArA_load, 1, cute.rank(tArA_load))
        tSrS_load = cute.flat_divide(
            tSrS_copy, cute.make_layout(fragment // W4A16_SF_VEC_SIZE)
        )
        tSrS_load = cute.group_modes(tSrS_load, 1, cute.rank(tSrS_load))
        tArA_transform_store = cute.flat_divide(tArA_transform, frag_tiler)
        tArA_transform_store = cute.group_modes(
            tArA_transform_store, 1, cute.rank(tArA_transform_store)
        )
        num_fragments = cute.size(tArA_load, mode=[1])
        return _W4A16TransformPartition(
            transform_group=transform_group,
            num_passes=num_passes,
            num_fragments=num_fragments,
            frag_tiler=frag_tiler,
            dst_copy_a=dst_copy_a,
            tAsA_input=tAsA_input,
            tA_transform=tA_transform,
            tSsS=tSsS,
            tSrS_copy=tSrS_copy,
            tSrS_load=tSrS_load,
            tArA_load=tArA_load,
            tArA_transform=tArA_transform,
            tArA_transform_store=tArA_transform_store,
        )

    @cute.jit
    def _w4a16_transform_stage(
        self,
        stage: cutlass.Int32,
        xf: _W4A16TransformPartition,
        ab_full_mbar: cute.Pointer,
        trans_full_mbar: cute.Pointer,
        trans_empty_mbar: cute.Pointer,
        NUM_SMEM_BUFFERS: cutlass.Constexpr[int],
    ):
        """Decode K stage ``stage`` (the MMA's ``accum_cnt_smem``) if it is ours.

        The two warpgroups take alternating stages. For K stage ``i`` (TMA
        stage ``buf``, TMEM A stage ``a_buf = i % W4A16_A_STAGES``) a
        warpgroup waits ``ab_full[buf]``, decodes ``code * scale`` (exact in
        BF16) in registers, waits ``trans_empty[a_buf]`` (the MMA's commit
        after reading that stage), stores to TMEM with ``tcgen05.st``, fences
        the stores and arrives on ``trans_full[a_buf]``, which the MMA waits
        on. ``xf`` is the thread's ``_w4a16_transform_setup``.
        """
        transform_group = xf.transform_group
        num_passes = xf.num_passes
        num_fragments = xf.num_fragments
        frag_tiler = xf.frag_tiler
        dst_copy_a = xf.dst_copy_a
        tAsA_input = xf.tAsA_input
        tA_transform = xf.tA_transform
        tSsS = xf.tSsS
        tSrS_copy = xf.tSrS_copy
        tSrS_load = xf.tSrS_load
        tArA_load = xf.tArA_load
        tArA_transform = xf.tArA_transform
        tArA_transform_store = xf.tArA_transform_store
        if stage % cutlass.Int32(W4A16_NUM_TRANSFORM_GROUPS) == transform_group:
            buf, phase = _get_bufidx_phase(stage, NUM_SMEM_BUFFERS)
            cute.arch.mbarrier_wait(ab_full_mbar + buf, phase)
            a_buf, a_phase = _get_bufidx_phase(stage, self.W4A16_A_STAGES)
            for p in cutlass.range_constexpr(num_passes):
                part = (None, None, None, (None, p), None)
                tAsA_p = tAsA_input[part]
                a_coord = (None,) * (cute.rank(tAsA_p) - 1) + (buf,)
                tAsA_slice = cute.flat_divide(tAsA_p[a_coord], frag_tiler)
                tAsA_slice = cute.group_modes(tAsA_slice, 1, cute.rank(tAsA_slice))
                tSsS_p = tSsS[part]
                s_coord = (None,) * (cute.rank(tSsS_p) - 1) + (buf,)
                tSsS_stage = tSsS_p[s_coord]
                cute.autovec_copy(
                    cute.make_tensor(
                        tSsS_stage.iterator,
                        cute.filter_zeros(tSsS_stage.layout),
                    ),
                    tSrS_copy,
                )
                for idx in cutlass.range_constexpr(num_fragments):
                    cute.autovec_copy(tAsA_slice[(None, idx)], tArA_load[(None, idx)])
                    packed = cute.recast_tensor(
                        tArA_load[(None, idx)], cutlass.Uint8
                    ).load()
                    scale_slice = tSrS_load[(None, idx)]
                    scale = cute.recast_tensor(
                        cute.make_tensor(
                            scale_slice.iterator,
                            cute.filter_zeros(scale_slice.layout),
                        ),
                        cutlass.Uint8,
                    ).load()
                    tArA_transform_store[(None, idx)].store(
                        decode_nvfp4_fragment_to_bf16(packed, scale)
                    )
                if cutlass.const_expr(p == 0):
                    cute.arch.mbarrier_wait(trans_empty_mbar + a_buf, a_phase ^ 1)
                tA_p = tA_transform[part]
                t_coord = (None,) * (cute.rank(tA_p) - 1) + (a_buf,)
                mixed_input_utils.store_transformed_a(
                    tArA_transform, tA_p[t_coord], dst_copy_a
                )
            # Make the tcgen05.st stores visible to the MMA before
            # trans_full can be observed.
            cute.arch.fence_view_async_tmem_store()
            cute.arch.mbarrier_arrive(trans_full_mbar + a_buf)
