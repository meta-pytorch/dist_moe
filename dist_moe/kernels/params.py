# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe argument records shared by distributed grouped GEMM kernels."""

from dataclasses import dataclass, fields
from typing import TypeVar

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import torch
from cutlass.cutlass_dsl import NumericMeta


class BlockScaledFormatSpec:
    """Operand + SF dtype + SF vec-size for one block-scaled format. Drives
    ``make_blockscaled_trivial_tiled_mma`` and ``tile_atom_to_shape_SF``."""

    name: str
    a_dtype: type[cutlass.Numeric]
    b_dtype: type[cutlass.Numeric]
    sf_dtype: type[cutlass.Numeric]
    sf_vec_size: int

    def __init__(
        self,
        name: str,
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        sf_dtype: type[cutlass.Numeric],
        sf_vec_size: int,
    ):
        self.name = name
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.sf_dtype = sf_dtype
        self.sf_vec_size = sf_vec_size


def _torch_dtype(dtype: type[cutlass.Numeric]) -> torch.dtype:
    if dtype == cutlass.Float4E2M1FN:
        return torch.float4_e2m1fn_x2
    return cutlass_torch.dtype(dtype)


BlockscaledBasePtrs = tuple[
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
]
BlockscaledTensorStride = tuple[cutlass.Int32, cutlass.Int32, cutlass.Int64]
BlockscaledTensorStrides = tuple[
    BlockscaledTensorStride,
    BlockscaledTensorStride,
    BlockscaledTensorStride,
]
BlockscaledScaleFactorStrides = tuple[cutlass.Int32, cutlass.Int32]
BlockscaledPointerStrideArgs = tuple[
    BlockscaledBasePtrs,
    BlockscaledTensorStrides,
    BlockscaledScaleFactorStrides,
]


@dataclass(frozen=True)
class BlockscaledGemmLayout:
    a_dtype: type[cutlass.Numeric]
    b_dtype: type[cutlass.Numeric]
    c_dtype: type[cutlass.Numeric]
    c_layout: utils.LayoutEnum
    tiled_mma: cute.TiledMma
    tiled_mma_sfb: cute.TiledMma
    mma_tiler: tuple[int, int, int]
    mma_tiler_sfb: tuple[int, int, int]
    cta_tile_shape_mnk: tuple[int, int, int]
    cluster_layout_vmnk: cute.Layout
    cluster_layout_sfb_vmnk: cute.Layout
    epi_tile: cute.Tile
    a_smem_layout_staged: cute.ComposedLayout
    b_smem_layout_staged: cute.ComposedLayout
    sfa_smem_layout_staged: cute.Layout
    sfb_smem_layout_staged: cute.Layout
    sfb_tma_smem_layout_staged: cute.Layout
    epi_smem_layout_staged: cute.ComposedLayout


_CUTE_STATIC_TYPES = (
    cutlass.Constexpr,
    NumericMeta,
    int,
    bool,
    str,
    float,
    type(None),
    utils.LayoutEnum,
)


def _is_cute_static(value) -> bool:
    if isinstance(value, _CUTE_STATIC_TYPES):
        return True
    return isinstance(value, tuple) and all(_is_cute_static(item) for item in value)


@dataclass
class CuteParamsBase:
    def __extract_mlir_values__(self):
        dynamic_fields = [
            getattr(self, field.name)
            for field in fields(self)
            if not _is_cute_static(getattr(self, field.name))
        ]
        values = []
        self._values_pos = []
        for obj in dynamic_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values.extend(obj_values)
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        static_fields = {}
        dynamic_fields = {}
        for field in fields(self):
            value = getattr(self, field.name)
            target = static_fields if _is_cute_static(value) else dynamic_fields
            target[field.name] = value
        for (name, value), size in zip(dynamic_fields.items(), self._values_pos):
            dynamic_fields[name] = cutlass.new_from_mlir_values(value, values[:size])
            values = values[size:]
        return self.__class__(**dynamic_fields, **static_fields)


_P = TypeVar("_P", bound=CuteParamsBase)


def params_from_kernel(cls: type[_P], kernel) -> _P:
    """Collect the compile-time kernel attributes named by ``cls``'s fields."""
    return cls(**{f.name: getattr(kernel, f.name) for f in fields(cls)})


def ceil_div(numerator: cutlass.Int32, denominator: cutlass.Int32) -> cutlass.Int32:
    return (numerator + denominator - cutlass.Int32(1)) // denominator


def swap_ab_pointer_strides(
    pointer_strides: BlockscaledPointerStrideArgs,
    swap_ab: cutlass.Constexpr[bool],
) -> BlockscaledPointerStrideArgs:
    base_ptrs, tensor_strides, sf_strides = pointer_strides
    a_base_ptr, b_base_ptr, c_base_ptr, sfa_base_ptr, sfb_base_ptr = base_ptrs
    a_strides, b_strides, c_strides = tensor_strides
    c_s0, c_s1, c_group_stride = c_strides

    if cutlass.const_expr(swap_ab):
        return (
            (b_base_ptr, a_base_ptr, c_base_ptr, sfb_base_ptr, sfa_base_ptr),
            (b_strides, a_strides, (c_s1, c_s0, c_group_stride)),
            sf_strides,
        )
    return pointer_strides


@dataclass
class BlockscaledTensormapParams(CuteParamsBase):
    tma_atom_a: cute.CopyAtom
    tma_atom_b: cute.CopyAtom
    tma_atom_sfa: cute.CopyAtom
    tma_atom_sfb: cute.CopyAtom
    tma_atom_c: cute.CopyAtom
    base_ptrs: BlockscaledBasePtrs
    strides: BlockscaledTensorStrides
    sf_strides: BlockscaledScaleFactorStrides
    elem_sizes: tuple[int, int, int]
    dtypes: cutlass.Constexpr[
        tuple[
            type[cutlass.Numeric],
            type[cutlass.Numeric],
            type[cutlass.Numeric],
        ]
    ]


@dataclass
class BlockscaledTensormapProblem(CuteParamsBase):
    split_sizes: cute.Tensor
    groups: int
    mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
    problem_type: int
    tensormap_base: int
    prepare_g: cutlass.Int32
    warp_idx: cutlass.Int32


@dataclass
class BlockscaledGemmKernelParams(CuteParamsBase):
    tma_atom_a: cute.CopyAtom
    tma_atom_b: cute.CopyAtom
    tma_atom_sfa: cute.CopyAtom
    tma_atom_sfb: cute.CopyAtom
    tma_atom_c: cute.CopyAtom
    mA: cute.Tensor
    mB: cute.Tensor
    mSFA: cute.Tensor
    mSFB: cute.Tensor
    mC: cute.Tensor
    a_smem_layout_staged: cute.ComposedLayout
    b_smem_layout_staged: cute.ComposedLayout
    sfa_smem_layout_staged: cute.Layout
    sfb_smem_layout_staged: cute.Layout
    sfb_tma_smem_layout_staged: cute.Layout
    epi_smem_layout_staged: cute.ComposedLayout
    epi_tile: cute.Tile
    cluster_layout_vmnk: cute.Layout
    cluster_layout_sfb_vmnk: cute.Layout
    tiled_mma: cute.TiledMma
    tiled_mma_sfb: cute.TiledMma
    mma_tiler: tuple[int, int, int]
    mma_tiler_sfb: tuple[int, int, int]
    cta_tile_shape_mnk: tuple[int, int, int]
    c_layout: utils.LayoutEnum
    c_dtype: type[cutlass.Numeric]
    a_dtype_width: int
    b_dtype_width: int
    combine_c_smem_layout_staged: cute.Layout | None = None


@dataclass
class BlockscaledGemmPrologue:
    sA: cute.Tensor
    sB: cute.Tensor
    sC: cute.Tensor
    sSFA: cute.Tensor
    sSFB: cute.Tensor
    tCgC: cute.Tensor
    tAsA: cute.Tensor
    tAgA: cute.Tensor
    tBsB: cute.Tensor
    tBgB: cute.Tensor
    tAsSFA: cute.Tensor
    tAgSFA: cute.Tensor
    tBsSFB: cute.Tensor
    tBgSFB: cute.Tensor
    tCrA: cute.Tensor
    tCrB: cute.Tensor
    tCtAcc_fake: cute.Tensor
    block_in_cluster_coord_vmnk: object
    block_in_cluster_coord_sfb_vmnk: object
    combine_sC: cute.Tensor | None = None


@dataclass
class GroupedGemmTmaPipeline(CuteParamsBase):
    tma_atom_a: cute.CopyAtom
    tma_atom_b: cute.CopyAtom
    tma_atom_sfa: cute.CopyAtom
    tma_atom_sfb: cute.CopyAtom
    gA: cute.Tensor
    gB: cute.Tensor
    gSFA: cute.Tensor
    gSFB: cute.Tensor
    sA: cute.Tensor
    sB: cute.Tensor
    sSFA: cute.Tensor
    sSFB: cute.Tensor
    num_tma_load_bytes: int
    num_tma_load_bytes_ab: int = 0
    num_tma_load_bytes_sf: int = 0


@dataclass
class GroupedGemmMmaPipeline(CuteParamsBase):
    tiled_mma: cute.TiledMma
    tCrA: cute.Tensor
    tCrB: cute.Tensor
    tCtAcc_base: cute.Tensor
    tCtSFA: cute.Tensor
    tCtSFA_copy: cute.Tensor
    tCtSFB: cute.Tensor
    tCtSFB_copy: cute.Tensor
    sSFA: cute.Tensor
    sSFB: cute.Tensor


@dataclass
class GroupedGemmEpilogPipeline(CuteParamsBase):
    c_atom: cute.CopyAtom
    tCtAcc_base: cute.Tensor
    c_tensor: cute.Tensor
    sC: cute.Tensor
    epi_tile: cute.Tile
    cta_tile_shape_mnk: tuple[int, int, int]
    c_layout: utils.LayoutEnum
    c_dtype: type[cutlass.Numeric]
    a_dtype_width: int


@dataclass
class GroupedGemmProblem(CuteParamsBase):
    split_sizes: cute.Tensor
    groups: int
    mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
    local_rank: cutlass.Int32


@dataclass
class GroupedGemmSecondaryOperands(CuteParamsBase):
    """Runtime metadata for a second GEMM sharing the primary pipeline."""

    a_base_ptr: cutlass.Int64
    b_base_ptr: cutlass.Int64
    a_strides: tuple[cutlass.Int32, cutlass.Int32]
    b_strides: tuple[cutlass.Int32, cutlass.Int32]
    elem_sizes: tuple[int, int]
    ready_counter: cute.Tensor
    ready_feature_tiles: cutlass.Int32
    schedule_rank: cutlass.Int32


@dataclass
class GroupedGemmPipelineSync(CuteParamsBase):
    ab_full_mbar: cute.Pointer
    ab_empty_mbar: cute.Pointer
    tmem_full_mbar: cute.Pointer
    tmem_empty_mbar: cute.Pointer
    tile_consumer_mbar: cute.Pointer
    tile_producer_mbar: cute.Pointer
    tile_cta_bar_mbar: cute.Pointer | None
    tile_id_smem_ptr: cute.Pointer
    cross_seam_mbar: cute.Pointer | None
    counter_ptr: cute.Pointer
    tma_mcast_masks: tuple
    ab_empty_mcast_mask: object
    acc_full_mcast_mask: object
    cluster_cta_rank: cutlass.Int32
    pred_cta0: cutlass.Boolean
    sf_full_mbar: cute.Pointer | None = None
    sf_empty_mbar: cute.Pointer | None = None


@dataclass
class DispatchQuantSync(CuteParamsBase):
    done_counter: cute.Tensor
    done_counter_offsets: tuple[cutlass.Int32, cutlass.Int32]


@dataclass
class DispatchQuantParams(CuteParamsBase):
    DISPATCH_QUANT_COL_BLOCKS_PER_SCALE: cutlass.Constexpr[int]
    DISPATCH_QUANT_COL_LANES: cutlass.Constexpr[int]
    DISPATCH_QUANT_COL_REDUCE_STAGES: cutlass.Constexpr[int]
    DISPATCH_QUANT_ELEMS_PER_LANE: cutlass.Constexpr[int]
    DISPATCH_QUANT_ELEMS_PER_SCALE_COL: cutlass.Constexpr[int]
    DISPATCH_QUANT_QDATA_ELEMS_PER_WORD: cutlass.Constexpr[int]
    DISPATCH_QUANT_ROW_LANES: cutlass.Constexpr[int]
    DISPATCH_QUANT_ROW_REDUCE_STAGES: cutlass.Constexpr[int]
    DISPATCH_QUANT_ROW_REPS: cutlass.Constexpr[int]
    dispatch_quant_source_dtype: cutlass.Constexpr[type[cutlass.Numeric]]
    format_id: cutlass.Constexpr[int]
    half_range_scale: cutlass.Constexpr[bool]
    is_fp4: cutlass.Constexpr[bool]
    sf_vec_size: cutlass.Constexpr[int]


@dataclass
class CombineSwigluQuantParams(CuteParamsBase):
    BLOCK_SIZE_M: cutlass.Constexpr[int]
    BLOCK_SIZE_N: cutlass.Constexpr[int]
    COMBINE_SWIGLU_BWD_COL_BLOCKS_PER_SCALE: cutlass.Constexpr[int]
    COMBINE_SWIGLU_BWD_COL_LANES: cutlass.Constexpr[int]
    COMBINE_SWIGLU_BWD_ELEMS_PER_LANE: cutlass.Constexpr[int]
    COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int]
    COMBINE_SWIGLU_FWD_COL_BLOCKS_PER_SCALE: cutlass.Constexpr[int]
    COMBINE_SWIGLU_FWD_COL_LANES: cutlass.Constexpr[int]
    COMBINE_SWIGLU_FWD_ELEMS_PER_LANE: cutlass.Constexpr[int]
    COMBINE_SWIGLU_FWD_ROW_ONLY_COL_BLOCKS_PER_SCALE: cutlass.Constexpr[int]
    COMBINE_SWIGLU_FWD_ROW_ONLY_COL_LANES: cutlass.Constexpr[int]
    COMBINE_SWIGLU_FWD_ROW_ONLY_ELEMS_PER_LANE: cutlass.Constexpr[int]
    COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int]
    COMBINE_SWIGLU_QUANT_FIRST_WARP: cutlass.Constexpr[int]
    COMBINE_SWIGLU_WORK_TILES_PER_FETCH: cutlass.Constexpr[int]
    DISPATCH_QUANT_QDATA_ELEMS_PER_WORD: cutlass.Constexpr[int]
    SWAP_AB: cutlass.Constexpr[bool]
    THREADS_PER_WARP: cutlass.Constexpr[int]
    combine_swiglu_alpha: cutlass.Constexpr[float]
    combine_swiglu_clamped: cutlass.Constexpr[bool]
    combine_swiglu_fast_math: cutlass.Constexpr[bool]
    combine_swiglu_limit: cutlass.Constexpr[float]
    combine_swiglu_row_quant_only: cutlass.Constexpr[bool]
    dispatch_quant_source_dtype: cutlass.Constexpr[type[cutlass.Numeric]]
    format_id: cutlass.Constexpr[int]
    sf_vec_size: cutlass.Constexpr[int]
    world_size: cutlass.Constexpr[int]


@dataclass
class MegaPipelineParams(CuteParamsBase):
    BLOCK_SIZE_K: cutlass.Constexpr[int]
    BLOCK_SIZE_M: cutlass.Constexpr[int]
    BLOCK_SIZE_N: cutlass.Constexpr[int]
    COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int]
    COMBINE_SWIGLU_FWD_SCALE_COLS_PER_TILE: cutlass.Constexpr[int]
    DISPATCH_QUANT_SCALE_COLS_PER_TILE: cutlass.Constexpr[int]
    force_n_major: cutlass.Constexpr[bool]
    KLOOP_UNROLL: cutlass.Constexpr[int]
    MEGA_DGRAD_PROBLEM_TYPE: cutlass.Constexpr[int]
    MEGA_DGRAD_TENSORMAP_BASE: cutlass.Constexpr[int]
    MEGA_FORWARD_MODE: cutlass.Constexpr[int]
    MEGA_WGRAD_PROBLEM_TYPE: cutlass.Constexpr[int]
    MEGA_WGRAD_TENSORMAP_BASE: cutlass.Constexpr[int]
    mode: cutlass.Constexpr[int]
    NUM_CTAS: cutlass.Constexpr[int]
    num_n_clusters: cutlass.Constexpr[int]
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int]
    NUM_TILE_BUFFERS: cutlass.Constexpr[int]
    NUM_TILE_CTA_BARS: cutlass.Constexpr[int]
    NUM_TMEM_BUFFERS: cutlass.Constexpr[int]
    OVERLAPPING_ACCUM: cutlass.Constexpr[bool]
    sf_vec_size: cutlass.Constexpr[int]
    STATIC_SCHEDULER: cutlass.Constexpr[bool]
    SWAP_AB: cutlass.Constexpr[bool]
    world_size: cutlass.Constexpr[int]


@dataclass
class ChunkedGemmWarpParams(CuteParamsBase):
    ACTIVATION_RING_CHUNKS: cutlass.Constexpr[int]
    USE_SM103_ULTRA: cutlass.Constexpr[bool]
    SM103_SF_RING_TILES: cutlass.Constexpr[int]
    BLOCK_SIZE_K: cutlass.Constexpr[int]
    BLOCK_SIZE_M: cutlass.Constexpr[int]
    BLOCK_SIZE_N: cutlass.Constexpr[int]
    FC2_TILES_PER_SCHEDULE_ITEM: cutlass.Constexpr[int]
    force_n_major: cutlass.Constexpr[bool]
    GLOBAL_LEAD_CHUNKS: cutlass.Constexpr[int]
    KLOOP_UNROLL: cutlass.Constexpr[int]
    MEGA_DGRAD_TENSORMAP_BASE: cutlass.Constexpr[int]
    MEGA_WGRAD_TENSORMAP_BASE: cutlass.Constexpr[int]
    NUM_CTAS: cutlass.Constexpr[int]
    num_n_clusters: cutlass.Constexpr[int]
    NUM_SMEM_BUFFERS: cutlass.Constexpr[int]
    NUM_TILE_BUFFERS: cutlass.Constexpr[int]
    NUM_TILE_CTA_BARS: cutlass.Constexpr[int]
    NUM_TMEM_BUFFERS: cutlass.Constexpr[int]
    OVERLAPPING_ACCUM: cutlass.Constexpr[bool]
    PIPELINE_CHUNK_ROWS: cutlass.Constexpr[int]
    PIPELINE_LEAD_CHUNKS: cutlass.Constexpr[int]
    STAGE_ALL_FC13: cutlass.Constexpr[bool]
    SWAP_AB: cutlass.Constexpr[bool]
    # Expert-borrow weight streaming: trailing WEIGHT_BORROW_SLOTS groups'
    # weights arrive via in-kernel fetch and the weight loader gates on the
    # done counters at WEIGHT_BORROW_DONE_OFFSET in the activation counter
    # region. 0 compiles the whole path out.
    WEIGHT_BORROW_SLOTS: cutlass.Constexpr[int]
    WEIGHT_BORROW_DONE_OFFSET: cutlass.Constexpr[int]
    WEIGHT_BORROW_CHUNK_BYTES: cutlass.Constexpr[int]
    world_size: cutlass.Constexpr[int]
