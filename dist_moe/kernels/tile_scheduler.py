# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DistMoE grouped GEMM tile schedulers.

Mirrors the layered design of the official CuTeDSL MoE example
(``cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/moe/moe_persistent_scheduler.py``):
a small ``WorkTileInfo`` data class, host-side ``Params``, and a device-side
scheduler with create factories plus ``initial_work_tile_info`` /
``advance_producer`` and ``advance_consumer`` instance methods.

Two scheduler classes are provided:

* :class:`DynamicTileScheduler` wraps the base grouped-GEMM atomic-counter +
  tile-id mbarrier protocol used by the original dist-MoE kernel. The TMA
  warp owns the atomic counter; MMA / epilog warps pre-read the same
  ``tile_id_smem`` ring via the consumer mbarrier.
* :class:`StaticTileScheduler` derives each persistent cluster's tile
  sequence from ``blockIdx.x // NUM_CTAS`` and advances by
  ``grid_dim.x // NUM_CTAS``. No counter, no tile-id mbarriers — works
  back-to-back with the dynamic ring kept in shared state for the
  dynamic kernel mode.

Both schedulers expose the same instance API so the warp bodies stay
mode-agnostic at every call site::

    scheduler = StaticTileScheduler.create_producer(...)  # or Dynamic*
    tile_idx = scheduler.initial_work_tile_info()
    while ...:
        ...
        tile_idx = scheduler.advance_producer(accum_cnt)

``__extract_mlir_values__`` / ``__new_from_mlir_values__`` are provided so
both schedulers can flow through ``@cute.jit`` boundaries via SSA — same
convention the MoE example uses. ``TileScheduler`` defines the shared
structural interface for type checking; concrete schedulers intentionally do
not inherit from a runtime base class so CuTeDSL sees only concrete JIT value
types.
"""

from typing import List, Protocol, TYPE_CHECKING

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import (
    extract_mlir_values,
    Int32,
    new_from_mlir_values,
)

from .config import (  # noqa: F401
    _DGRAD,
    _FPROP,
    _WGRAD,
    uses_paged_blockscaled_scale_rows,
)
from .params import (
    ceil_div,
)

_HAS_MAP_DSMEM_PTR = hasattr(cute.arch, "map_dsmem_ptr")


@cute.jit
def blockscaled_scale_row_start(
    scale_start_m: Int32,
    row_offset: Int32,
    SWAP_MN: cutlass.Constexpr[bool],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
) -> Int32:
    if cutlass.const_expr(not SWAP_MN):
        return scale_start_m + row_offset
    if cutlass.const_expr(uses_paged_blockscaled_scale_rows(BLOCK_SIZE_N)):
        scale_page_rows: cutlass.Constexpr[int] = ((BLOCK_SIZE_N + 127) // 128) * 128
        return (
            scale_start_m
            + row_offset // Int32(BLOCK_SIZE_N) * Int32(scale_page_rows)
            + row_offset % Int32(BLOCK_SIZE_N)
        )
    return scale_start_m + row_offset


@cute.jit
def advance_blockscaled_scale_start(
    scale_start_m: Int32,
    group_rows: Int32,
    SWAP_MN: cutlass.Constexpr[bool],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
) -> Int32:
    if cutlass.const_expr(SWAP_MN and uses_paged_blockscaled_scale_rows(BLOCK_SIZE_N)):
        scale_page_rows: cutlass.Constexpr[int] = ((BLOCK_SIZE_N + 127) // 128) * 128
        return scale_start_m + (
            (group_rows + Int32(BLOCK_SIZE_N - 1)) // Int32(BLOCK_SIZE_N)
        ) * Int32(scale_page_rows)
    return scale_start_m + group_rows


def _map_remote_smem_ptr(smem_ptr, cta_rank_in_cluster):
    """Map peer SMEM to the addrspace-3 pointer expected by arch.store."""
    if cutlass.const_expr(_HAS_MAP_DSMEM_PTR):
        dsmem_ptr = cute.arch.map_dsmem_ptr(smem_ptr, cta_rank_in_cluster)
        return llvm.addrspacecast(
            llvm.PointerType.get(cute.AddressSpace.smem),
            dsmem_ptr.to_llvm_ptr(),
        )
    return cute.arch.mapa(smem_ptr, cta_rank_in_cluster=cta_rank_in_cluster)


# Termination sentinel for tile_info. INT32_MAX - 1 leaves headroom for the
# 2-CTA `tile_idx + cluster_cta_rank` read.
_TILE_SENTINEL = 0x7FFFFFFF - 1


class GroupedWorkTileInfo:
    """Decoded grouped-GEMM work owned by one persistent cluster."""

    def __init__(
        self,
        group_idx: Int32,
        tile_m_idx: Int32,
        tile_n_idx: Int32,
        subtile_idx: Int32,
        num_k_tiles: Int32,
        m_size: Int32,
        n_size: Int32,
        k_size: Int32,
        split_prefix: Int32,
        m_tile_prefix: Int32,
        n_tile_prefix: Int32,
        scale_split_prefix: Int32,
    ):
        self.group_idx = group_idx
        self.tile_m_idx = tile_m_idx
        self.tile_n_idx = tile_n_idx
        self.subtile_idx = subtile_idx
        self.num_k_tiles = num_k_tiles
        self.m_size = m_size
        self.n_size = n_size
        self.k_size = k_size
        self.split_prefix = split_prefix
        self.m_tile_prefix = m_tile_prefix
        self.n_tile_prefix = n_tile_prefix
        self.scale_split_prefix = scale_split_prefix

    @property
    def is_valid_tile(self):
        return self.group_idx >= Int32(0)

    def _values(self):
        return (
            self.group_idx,
            self.tile_m_idx,
            self.tile_n_idx,
            self.subtile_idx,
            self.num_k_tiles,
            self.m_size,
            self.n_size,
            self.k_size,
            self.split_prefix,
            self.m_tile_prefix,
            self.n_tile_prefix,
            self.scale_split_prefix,
        )

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for value in self._values():
            values.extend(extract_mlir_values(value))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "GroupedWorkTileInfo":
        old_values = self._values()
        assert len(values) == len(old_values)
        return GroupedWorkTileInfo(
            *[
                new_from_mlir_values(old_value, [value])
                for old_value, value in zip(old_values, values)
            ]
        )


class GroupedMmaWorkTileInfo:
    """Minimal grouped schedule state consumed by an MMA warp.

    ``tile_n_idx`` is the group-local token tile the MMA is about to issue. The
    paged SFB layouts pack two token tiles into one 128-row scale page, so the
    MMA needs the tile's parity to bind the matching ``tcgen05`` SFB field.
    """

    def __init__(self, group_idx: Int32, num_k_tiles: Int32, tile_n_idx: Int32):
        self.group_idx = group_idx
        self.num_k_tiles = num_k_tiles
        self.tile_n_idx = tile_n_idx

    @property
    def is_valid_tile(self):
        return self.group_idx >= Int32(0)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        return [
            *extract_mlir_values(self.group_idx),
            *extract_mlir_values(self.num_k_tiles),
            *extract_mlir_values(self.tile_n_idx),
        ]

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "GroupedMmaWorkTileInfo":
        assert len(values) == 3
        return GroupedMmaWorkTileInfo(
            new_from_mlir_values(self.group_idx, [values[0]]),
            new_from_mlir_values(self.num_k_tiles, [values[1]]),
            new_from_mlir_values(self.tile_n_idx, [values[2]]),
        )


@cute.jit
def _group_sizes(
    split_sizes: cute.Tensor,
    group_idx: Int32,
    M: Int32,
    N: Int32,
    K: Int32,
    problem_type: cutlass.Constexpr[int],
    SWAP_MN: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(problem_type == _WGRAD):
        m_size, n_size, k_size = M, N, Int32(split_sizes[group_idx])
    else:
        m_size, n_size, k_size = Int32(split_sizes[group_idx]), N, K
    if cutlass.const_expr(SWAP_MN):
        return n_size, m_size, k_size
    return m_size, n_size, k_size


@cute.jit
def _tile_grid(
    m_size: Int32,
    n_size: Int32,
    k_size: Int32,
    BLOCK_SIZE_M: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    BLOCK_SIZE_K: cutlass.Constexpr[int],
    FORCE_N_MAJOR: cutlass.Constexpr[bool],
    NUM_N_CLUSTERS: cutlass.Constexpr[int],
    SWAP_MN: cutlass.Constexpr[bool],
):
    num_m_tiles = (m_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n_tiles = (n_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_k_tiles = (k_size + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    logical_num_m_tiles = num_m_tiles
    logical_num_n_tiles = num_n_tiles
    logical_m_size = m_size
    logical_n_size = n_size
    if cutlass.const_expr(SWAP_MN):
        logical_num_m_tiles = num_n_tiles
        logical_num_n_tiles = num_m_tiles
        logical_m_size = n_size
        logical_n_size = m_size

    num_tiles = num_m_tiles * num_n_tiles
    if cutlass.const_expr(NUM_N_CLUSTERS > 1):
        if cutlass.const_expr(FORCE_N_MAJOR) or not (logical_m_size < logical_n_size):
            n_tiles_per_cluster = (
                logical_num_n_tiles + NUM_N_CLUSTERS - 1
            ) // NUM_N_CLUSTERS
            num_tiles = (
                logical_num_m_tiles * n_tiles_per_cluster * Int32(NUM_N_CLUSTERS)
            )
    return num_m_tiles, num_n_tiles, num_k_tiles, num_tiles


@cute.jit
def _tile_coords(
    tile_idx: Int32,
    num_m_tiles: Int32,
    num_n_tiles: Int32,
    m_size: Int32,
    n_size: Int32,
    FORCE_N_MAJOR: cutlass.Constexpr[bool],
    NUM_N_CLUSTERS: cutlass.Constexpr[int],
    local_rank: Int32,
    WORLD_SIZE: cutlass.Constexpr[int],
    SWAP_MN: cutlass.Constexpr[bool],
):
    logical_tile_m_idx = Int32(0)
    logical_tile_n_idx = Int32(0)
    logical_num_m_tiles = num_m_tiles
    logical_num_n_tiles = num_n_tiles
    logical_m_size = m_size
    logical_n_size = n_size
    if cutlass.const_expr(SWAP_MN):
        logical_num_m_tiles = num_n_tiles
        logical_num_n_tiles = num_m_tiles
        logical_m_size = n_size
        logical_n_size = m_size

    if cutlass.const_expr(FORCE_N_MAJOR) or not (logical_m_size < logical_n_size):
        n_tiles_per_cluster = (
            logical_num_n_tiles + NUM_N_CLUSTERS - 1
        ) // NUM_N_CLUSTERS
        cluster_tiles = logical_num_m_tiles * n_tiles_per_cluster
        cluster_idx = tile_idx // cluster_tiles
        within_cluster = tile_idx % cluster_tiles
        logical_tile_m_idx = within_cluster // n_tiles_per_cluster
        logical_tile_n_idx = (
            cluster_idx * n_tiles_per_cluster + within_cluster % n_tiles_per_cluster
        )
    else:
        logical_tile_m_idx = tile_idx % logical_num_m_tiles
        logical_tile_n_idx = tile_idx // logical_num_m_tiles

    if cutlass.const_expr(WORLD_SIZE > 1):
        tiles_per_rank = (logical_num_m_tiles + WORLD_SIZE - 1) // WORLD_SIZE
        m_offset = (local_rank * tiles_per_rank) % logical_num_m_tiles
        logical_tile_m_idx = (logical_tile_m_idx + m_offset) % logical_num_m_tiles
    if cutlass.const_expr(SWAP_MN):
        return logical_tile_n_idx, logical_tile_m_idx
    return logical_tile_m_idx, logical_tile_n_idx


class GroupedProblemVisitor:
    """Stateful linear-tile decoder with a cached current expert."""

    def __init__(
        self,
        split_sizes: cute.Tensor,
        M: Int32,
        N: Int32,
        K: Int32,
        group_idx: Int32,
        tile_start: Int32,
        tile_end: Int32,
        split_prefix: Int32,
        m_tile_prefix: Int32,
        n_tile_prefix: Int32,
        scale_split_prefix: Int32,
        m_size: Int32,
        n_size: Int32,
        k_size: Int32,
        num_m_tiles: Int32,
        num_n_tiles: Int32,
        num_k_tiles: Int32,
        G: cutlass.Constexpr[int],
        problem_type: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        local_rank: Int32,
        WORLD_SIZE: cutlass.Constexpr[int],
        SWAP_MN: cutlass.Constexpr[bool],
        M_SUBTILES: cutlass.Constexpr[int],
    ):
        self.split_sizes = split_sizes
        self.M = M
        self.N = N
        self.K = K
        self.group_idx = group_idx
        self.tile_start = tile_start
        self.tile_end = tile_end
        self.split_prefix = split_prefix
        self.m_tile_prefix = m_tile_prefix
        self.n_tile_prefix = n_tile_prefix
        self.scale_split_prefix = scale_split_prefix
        self.m_size = m_size
        self.n_size = n_size
        self.k_size = k_size
        self.num_m_tiles = num_m_tiles
        self.num_n_tiles = num_n_tiles
        self.num_k_tiles = num_k_tiles
        self.G = G
        self.problem_type = problem_type
        self.BLOCK_SIZE_M = BLOCK_SIZE_M
        self.BLOCK_SIZE_N = BLOCK_SIZE_N
        self.BLOCK_SIZE_K = BLOCK_SIZE_K
        self.FORCE_N_MAJOR = FORCE_N_MAJOR
        self.NUM_N_CLUSTERS = NUM_N_CLUSTERS
        self.local_rank = local_rank
        self.WORLD_SIZE = WORLD_SIZE
        self.SWAP_MN = SWAP_MN
        self.M_SUBTILES = M_SUBTILES

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = list(extract_mlir_values(self.split_sizes))
        for value in (
            self.M,
            self.N,
            self.K,
            self.group_idx,
            self.tile_start,
            self.tile_end,
            self.split_prefix,
            self.m_tile_prefix,
            self.n_tile_prefix,
            self.scale_split_prefix,
            self.m_size,
            self.n_size,
            self.k_size,
            self.num_m_tiles,
            self.num_n_tiles,
            self.num_k_tiles,
            self.local_rank,
        ):
            values.extend(extract_mlir_values(value))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "GroupedProblemVisitor":
        tensor_value_count = len(extract_mlir_values(self.split_sizes))
        split_sizes = new_from_mlir_values(
            self.split_sizes, values[:tensor_value_count]
        )
        old_values = (
            self.M,
            self.N,
            self.K,
            self.group_idx,
            self.tile_start,
            self.tile_end,
            self.split_prefix,
            self.m_tile_prefix,
            self.n_tile_prefix,
            self.scale_split_prefix,
            self.m_size,
            self.n_size,
            self.k_size,
            self.num_m_tiles,
            self.num_n_tiles,
            self.num_k_tiles,
            self.local_rank,
        )
        runtime_values = values[tensor_value_count:]
        assert len(runtime_values) == len(old_values)
        new_values = [
            new_from_mlir_values(old_value, [value])
            for old_value, value in zip(old_values, runtime_values)
        ]
        return GroupedProblemVisitor(
            split_sizes=split_sizes,
            M=new_values[0],
            N=new_values[1],
            K=new_values[2],
            group_idx=new_values[3],
            tile_start=new_values[4],
            tile_end=new_values[5],
            split_prefix=new_values[6],
            m_tile_prefix=new_values[7],
            n_tile_prefix=new_values[8],
            scale_split_prefix=new_values[9],
            m_size=new_values[10],
            n_size=new_values[11],
            k_size=new_values[12],
            num_m_tiles=new_values[13],
            num_n_tiles=new_values[14],
            num_k_tiles=new_values[15],
            G=self.G,
            problem_type=self.problem_type,
            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            BLOCK_SIZE_K=self.BLOCK_SIZE_K,
            FORCE_N_MAJOR=self.FORCE_N_MAJOR,
            NUM_N_CLUSTERS=self.NUM_N_CLUSTERS,
            local_rank=new_values[16],
            WORLD_SIZE=self.WORLD_SIZE,
            SWAP_MN=self.SWAP_MN,
            M_SUBTILES=self.M_SUBTILES,
        )

    @staticmethod
    @cute.jit
    def create(
        split_sizes: cute.Tensor,
        M: Int32,
        N: Int32,
        K: Int32,
        G: cutlass.Constexpr[int],
        problem_type: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        local_rank: Int32,
        WORLD_SIZE: cutlass.Constexpr[int],
        SWAP_MN: cutlass.Constexpr[bool] = False,
        M_SUBTILES: cutlass.Constexpr[int] = 1,
    ) -> "GroupedProblemVisitor":
        group_idx = Int32(0)
        m_size, n_size, k_size = _group_sizes(
            split_sizes, group_idx, M, N, K, problem_type, SWAP_MN
        )
        num_m_tiles, num_n_tiles, num_k_tiles, num_tiles = _tile_grid(
            m_size,
            n_size,
            k_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            FORCE_N_MAJOR,
            NUM_N_CLUSTERS,
            SWAP_MN,
        )
        num_tiles *= Int32(M_SUBTILES)
        return GroupedProblemVisitor(
            split_sizes,
            M,
            N,
            K,
            group_idx,
            Int32(0),
            num_tiles,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            m_size,
            n_size,
            k_size,
            num_m_tiles,
            num_n_tiles,
            num_k_tiles,
            G,
            problem_type,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            FORCE_N_MAJOR,
            NUM_N_CLUSTERS,
            local_rank,
            WORLD_SIZE,
            SWAP_MN,
            M_SUBTILES,
        )

    @cute.jit
    def _advance_group_to_contain(self, tile_idx: Int32) -> None:
        while tile_idx >= self.tile_end and self.group_idx < self.G:
            group_rows = Int32(self.split_sizes[self.group_idx])
            self.split_prefix += group_rows
            self.scale_split_prefix = advance_blockscaled_scale_start(
                self.scale_split_prefix,
                group_rows,
                self.SWAP_MN,
                self.BLOCK_SIZE_N,
            )
            self.m_tile_prefix += self.num_m_tiles
            self.n_tile_prefix += self.num_n_tiles
            self.tile_start = self.tile_end
            self.group_idx += Int32(1)
            if self.group_idx < self.G:
                self.m_size, self.n_size, self.k_size = _group_sizes(
                    self.split_sizes,
                    self.group_idx,
                    self.M,
                    self.N,
                    self.K,
                    self.problem_type,
                    self.SWAP_MN,
                )
                (
                    self.num_m_tiles,
                    self.num_n_tiles,
                    self.num_k_tiles,
                    num_tiles,
                ) = _tile_grid(
                    self.m_size,
                    self.n_size,
                    self.k_size,
                    self.BLOCK_SIZE_M,
                    self.BLOCK_SIZE_N,
                    self.BLOCK_SIZE_K,
                    self.FORCE_N_MAJOR,
                    self.NUM_N_CLUSTERS,
                    self.SWAP_MN,
                )
                num_tiles *= Int32(self.M_SUBTILES)
                self.tile_end += num_tiles

    @cute.jit
    def _work_tile_coords(self, parent_tile_idx: Int32):
        return _tile_coords(
            parent_tile_idx,
            self.num_m_tiles,
            self.num_n_tiles,
            self.m_size,
            self.n_size,
            self.FORCE_N_MAJOR,
            self.NUM_N_CLUSTERS,
            self.local_rank,
            self.WORLD_SIZE,
            self.SWAP_MN,
        )

    @cute.jit
    def get_work(self, tile_idx: Int32) -> GroupedWorkTileInfo:
        self._advance_group_to_contain(tile_idx)
        work = GroupedWorkTileInfo(Int32(-1), *([Int32(0)] * 11))
        if self.group_idx < self.G and tile_idx < self.tile_end:
            local_tile_idx = tile_idx - self.tile_start
            parent_tile_idx = local_tile_idx // Int32(self.M_SUBTILES)
            subtile_idx = local_tile_idx % Int32(self.M_SUBTILES)
            tile_m_idx, tile_n_idx = self._work_tile_coords(parent_tile_idx)
            work = GroupedWorkTileInfo(
                self.group_idx,
                tile_m_idx,
                tile_n_idx,
                subtile_idx,
                self.num_k_tiles,
                self.m_size,
                self.n_size,
                self.k_size,
                self.split_prefix,
                self.m_tile_prefix,
                self.n_tile_prefix,
                self.scale_split_prefix,
            )
        return work

    @cute.jit
    def get_mma_work(self, tile_idx: Int32) -> GroupedMmaWorkTileInfo:
        self._advance_group_to_contain(tile_idx)
        work = GroupedMmaWorkTileInfo(Int32(-1), Int32(0), Int32(0))
        if self.group_idx < self.G and tile_idx < self.tile_end:
            parent_tile_idx = (tile_idx - self.tile_start) // Int32(self.M_SUBTILES)
            _, tile_n_idx = self._work_tile_coords(parent_tile_idx)
            work = GroupedMmaWorkTileInfo(self.group_idx, self.num_k_tiles, tile_n_idx)
        return work


class MegaWorkTileInfo:
    """Decoded work for one of two GEMMs interleaved per expert."""

    def __init__(
        self,
        group_idx: Int32,
        problem_idx: Int32,
        tile_idx: Int32,
        tile_m_idx: Int32,
        tile_n_idx: Int32,
        num_m_tiles: Int32,
        num_n_tiles: Int32,
        num_k_tiles: Int32,
        m_size: Int32,
        n_size: Int32,
        k_size: Int32,
        split_prefix: Int32,
        m_tile_prefix: Int32,
        n_tile_prefix: Int32,
        group_tile_prefix: Int32,
        act_tile_prefix_0: Int32,
        act_tile_prefix_1: Int32,
    ):
        self.group_idx = group_idx
        self.problem_idx = problem_idx
        self.tile_idx = tile_idx
        self.tile_m_idx = tile_m_idx
        self.tile_n_idx = tile_n_idx
        self.num_m_tiles = num_m_tiles
        self.num_n_tiles = num_n_tiles
        self.num_k_tiles = num_k_tiles
        self.m_size = m_size
        self.n_size = n_size
        self.k_size = k_size
        self.split_prefix = split_prefix
        self.m_tile_prefix = m_tile_prefix
        self.n_tile_prefix = n_tile_prefix
        self.group_tile_prefix = group_tile_prefix
        self.act_tile_prefix_0 = act_tile_prefix_0
        self.act_tile_prefix_1 = act_tile_prefix_1

    def _values(self):
        return (
            self.group_idx,
            self.problem_idx,
            self.tile_idx,
            self.tile_m_idx,
            self.tile_n_idx,
            self.num_m_tiles,
            self.num_n_tiles,
            self.num_k_tiles,
            self.m_size,
            self.n_size,
            self.k_size,
            self.split_prefix,
            self.m_tile_prefix,
            self.n_tile_prefix,
            self.group_tile_prefix,
            self.act_tile_prefix_0,
            self.act_tile_prefix_1,
        )

    @property
    def is_valid_tile(self):
        return self.group_idx >= Int32(0)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for value in self._values():
            values.extend(extract_mlir_values(value))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MegaWorkTileInfo":
        old_values = self._values()
        assert len(values) == len(old_values)
        return MegaWorkTileInfo(
            *[
                new_from_mlir_values(old_value, [value])
                for old_value, value in zip(old_values, values)
            ]
        )


MEGA_WORK_INFO_FIELDS: int = 17


@cute.jit
def publish_mega_work_info(
    work: MegaWorkTileInfo,
    work_info_smem_ptr: cute.Pointer,
    tile_buf: Int32,
) -> None:
    base = tile_buf * Int32(MEGA_WORK_INFO_FIELDS)
    with cute.arch.elect_one():
        for field_idx, value in enumerate(work._values()):
            cute.arch.store(
                work_info_smem_ptr + base + Int32(field_idx),
                value,
                ss="cta",
            )


@cute.jit
def load_mega_work_info(
    tile_idx: Int32,
    work_info_smem_ptr: cute.Pointer,
    tile_buf: Int32,
) -> MegaWorkTileInfo:
    work = MegaWorkTileInfo(Int32(-1), *([Int32(0)] * 16))
    if tile_idx < Int32(_TILE_SENTINEL):
        base = tile_buf * Int32(MEGA_WORK_INFO_FIELDS)
        values = [
            cute.arch.load(
                work_info_smem_ptr + base + Int32(field_idx),
                Int32,
                ss="cta",
            )
            for field_idx in range(MEGA_WORK_INFO_FIELDS)
        ]
        work = MegaWorkTileInfo(*values)
    return work


class MegaProblemVisitor:
    """Stateful decoder for two grouped GEMMs interleaved per expert."""

    def __init__(
        self,
        split_sizes: cute.Tensor,
        problem_mnk: tuple,
        group_idx: Int32,
        problem_idx: Int32,
        tile_start: Int32,
        tile_end: Int32,
        split_prefix: Int32,
        group_tile_prefix: Int32,
        m_tile_prefix_0: Int32,
        n_tile_prefix_0: Int32,
        m_tile_prefix_1: Int32,
        n_tile_prefix_1: Int32,
        m_size: Int32,
        n_size: Int32,
        k_size: Int32,
        num_m_tiles: Int32,
        num_n_tiles: Int32,
        num_k_tiles: Int32,
        local_rank: Int32,
        G: cutlass.Constexpr[int],
        problem_types: cutlass.Constexpr[tuple[int, int]],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        WORLD_SIZE: cutlass.Constexpr[int],
        SWAP_MN: cutlass.Constexpr[bool],
        GROUP_TILE_SIZE: cutlass.Constexpr[int],
    ):
        self.split_sizes = split_sizes
        self.problem_mnk = problem_mnk
        self.group_idx = group_idx
        self.problem_idx = problem_idx
        self.tile_start = tile_start
        self.tile_end = tile_end
        self.split_prefix = split_prefix
        self.group_tile_prefix = group_tile_prefix
        self.m_tile_prefix_0 = m_tile_prefix_0
        self.n_tile_prefix_0 = n_tile_prefix_0
        self.m_tile_prefix_1 = m_tile_prefix_1
        self.n_tile_prefix_1 = n_tile_prefix_1
        self.m_size = m_size
        self.n_size = n_size
        self.k_size = k_size
        self.num_m_tiles = num_m_tiles
        self.num_n_tiles = num_n_tiles
        self.num_k_tiles = num_k_tiles
        self.local_rank = local_rank
        self.G = G
        self.problem_types = problem_types
        self.BLOCK_SIZE_M = BLOCK_SIZE_M
        self.BLOCK_SIZE_N = BLOCK_SIZE_N
        self.BLOCK_SIZE_K = BLOCK_SIZE_K
        self.FORCE_N_MAJOR = FORCE_N_MAJOR
        self.NUM_N_CLUSTERS = NUM_N_CLUSTERS
        self.WORLD_SIZE = WORLD_SIZE
        self.SWAP_MN = SWAP_MN
        self.GROUP_TILE_SIZE = GROUP_TILE_SIZE

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = list(extract_mlir_values(self.split_sizes))
        for value in (*self.problem_mnk[0], *self.problem_mnk[1]):
            values.extend(extract_mlir_values(value))
        for value in (
            self.group_idx,
            self.problem_idx,
            self.tile_start,
            self.tile_end,
            self.split_prefix,
            self.group_tile_prefix,
            self.m_tile_prefix_0,
            self.n_tile_prefix_0,
            self.m_tile_prefix_1,
            self.n_tile_prefix_1,
            self.m_size,
            self.n_size,
            self.k_size,
            self.num_m_tiles,
            self.num_n_tiles,
            self.num_k_tiles,
            self.local_rank,
        ):
            values.extend(extract_mlir_values(value))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MegaProblemVisitor":
        tensor_value_count = len(extract_mlir_values(self.split_sizes))
        split_sizes = new_from_mlir_values(
            self.split_sizes, values[:tensor_value_count]
        )
        old_problem_values = (*self.problem_mnk[0], *self.problem_mnk[1])
        value_idx = tensor_value_count
        problem_values = [
            new_from_mlir_values(old_value, [values[value_idx + idx]])
            for idx, old_value in enumerate(old_problem_values)
        ]
        value_idx += len(old_problem_values)
        old_state_values = (
            self.group_idx,
            self.problem_idx,
            self.tile_start,
            self.tile_end,
            self.split_prefix,
            self.group_tile_prefix,
            self.m_tile_prefix_0,
            self.n_tile_prefix_0,
            self.m_tile_prefix_1,
            self.n_tile_prefix_1,
            self.m_size,
            self.n_size,
            self.k_size,
            self.num_m_tiles,
            self.num_n_tiles,
            self.num_k_tiles,
            self.local_rank,
        )
        state_values = [
            new_from_mlir_values(old_value, [values[value_idx + idx]])
            for idx, old_value in enumerate(old_state_values)
        ]
        assert value_idx + len(state_values) == len(values)
        return MegaProblemVisitor(
            split_sizes=split_sizes,
            problem_mnk=(tuple(problem_values[:3]), tuple(problem_values[3:])),
            group_idx=state_values[0],
            problem_idx=state_values[1],
            tile_start=state_values[2],
            tile_end=state_values[3],
            split_prefix=state_values[4],
            group_tile_prefix=state_values[5],
            m_tile_prefix_0=state_values[6],
            n_tile_prefix_0=state_values[7],
            m_tile_prefix_1=state_values[8],
            n_tile_prefix_1=state_values[9],
            m_size=state_values[10],
            n_size=state_values[11],
            k_size=state_values[12],
            num_m_tiles=state_values[13],
            num_n_tiles=state_values[14],
            num_k_tiles=state_values[15],
            local_rank=state_values[16],
            G=self.G,
            problem_types=self.problem_types,
            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            BLOCK_SIZE_K=self.BLOCK_SIZE_K,
            FORCE_N_MAJOR=self.FORCE_N_MAJOR,
            NUM_N_CLUSTERS=self.NUM_N_CLUSTERS,
            WORLD_SIZE=self.WORLD_SIZE,
            SWAP_MN=self.SWAP_MN,
            GROUP_TILE_SIZE=self.GROUP_TILE_SIZE,
        )

    @staticmethod
    @cute.jit
    def create(
        split_sizes: cute.Tensor,
        problem_mnk: tuple,
        G: cutlass.Constexpr[int],
        problem_types: cutlass.Constexpr[tuple[int, int]],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        local_rank: Int32,
        WORLD_SIZE: cutlass.Constexpr[int],
        SWAP_MN: cutlass.Constexpr[bool],
        GROUP_TILE_SIZE: cutlass.Constexpr[int],
    ) -> "MegaProblemVisitor":
        m_size, n_size, k_size = _group_sizes(
            split_sizes,
            Int32(0),
            *problem_mnk[0],
            problem_types[0],
            SWAP_MN,
        )
        num_m_tiles, num_n_tiles, num_k_tiles, num_tiles = _tile_grid(
            m_size,
            n_size,
            k_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            FORCE_N_MAJOR,
            NUM_N_CLUSTERS,
            SWAP_MN,
        )
        return MegaProblemVisitor(
            split_sizes,
            problem_mnk,
            Int32(0),
            Int32(0),
            Int32(0),
            num_tiles,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            m_size,
            n_size,
            k_size,
            num_m_tiles,
            num_n_tiles,
            num_k_tiles,
            local_rank,
            G,
            problem_types,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            FORCE_N_MAJOR,
            NUM_N_CLUSTERS,
            WORLD_SIZE,
            SWAP_MN,
            GROUP_TILE_SIZE,
        )

    @cute.jit
    def _advance_problem_to_contain(self, tile_idx: Int32) -> None:
        while tile_idx >= self.tile_end and self.group_idx < self.G:
            if self.problem_idx == Int32(0):
                self.m_tile_prefix_0 += self.num_m_tiles
                self.n_tile_prefix_0 += self.num_n_tiles
                self.problem_idx = Int32(1)
            else:
                self.m_tile_prefix_1 += self.num_m_tiles
                self.n_tile_prefix_1 += self.num_n_tiles
                split_size = Int32(self.split_sizes[self.group_idx])
                self.split_prefix += split_size
                self.group_tile_prefix += (
                    split_size + self.GROUP_TILE_SIZE - 1
                ) // self.GROUP_TILE_SIZE
                self.problem_idx = Int32(0)
                self.group_idx += Int32(1)
            self.tile_start = self.tile_end
            if self.group_idx < self.G:
                if self.problem_idx == Int32(0):
                    self.m_size, self.n_size, self.k_size = _group_sizes(
                        self.split_sizes,
                        self.group_idx,
                        *self.problem_mnk[0],
                        self.problem_types[0],
                        self.SWAP_MN,
                    )
                else:
                    self.m_size, self.n_size, self.k_size = _group_sizes(
                        self.split_sizes,
                        self.group_idx,
                        *self.problem_mnk[1],
                        self.problem_types[1],
                        self.SWAP_MN,
                    )
                (
                    self.num_m_tiles,
                    self.num_n_tiles,
                    self.num_k_tiles,
                    num_tiles,
                ) = _tile_grid(
                    self.m_size,
                    self.n_size,
                    self.k_size,
                    self.BLOCK_SIZE_M,
                    self.BLOCK_SIZE_N,
                    self.BLOCK_SIZE_K,
                    self.FORCE_N_MAJOR,
                    self.NUM_N_CLUSTERS,
                    self.SWAP_MN,
                )
                self.tile_end += num_tiles

    @cute.jit
    def get_work(self, tile_idx: Int32) -> MegaWorkTileInfo:
        self._advance_problem_to_contain(tile_idx)
        work = MegaWorkTileInfo(Int32(-1), *([Int32(0)] * 16))
        if self.group_idx < self.G and tile_idx < self.tile_end:
            local_tile_idx = tile_idx - self.tile_start
            tile_m_idx, tile_n_idx = _tile_coords(
                local_tile_idx,
                self.num_m_tiles,
                self.num_n_tiles,
                self.m_size,
                self.n_size,
                self.FORCE_N_MAJOR,
                self.NUM_N_CLUSTERS,
                self.local_rank,
                self.WORLD_SIZE,
                self.SWAP_MN,
            )
            m_tile_prefix = self.m_tile_prefix_0
            n_tile_prefix = self.n_tile_prefix_0
            if self.problem_idx == Int32(1):
                m_tile_prefix = self.m_tile_prefix_1
                n_tile_prefix = self.n_tile_prefix_1
            if cutlass.const_expr(self.SWAP_MN):
                act_tile_prefix_0 = self.n_tile_prefix_0
                act_tile_prefix_1 = self.n_tile_prefix_1
            else:
                act_tile_prefix_0 = self.m_tile_prefix_0
                act_tile_prefix_1 = self.m_tile_prefix_1
            work = MegaWorkTileInfo(
                self.group_idx,
                self.problem_idx,
                local_tile_idx,
                tile_m_idx,
                tile_n_idx,
                self.num_m_tiles,
                self.num_n_tiles,
                self.num_k_tiles,
                self.m_size,
                self.n_size,
                self.k_size,
                self.split_prefix,
                m_tile_prefix,
                n_tile_prefix,
                self.group_tile_prefix,
                act_tile_prefix_0,
                act_tile_prefix_1,
            )
        return work


class ChunkedMegaWorkTileInfo:
    """Decoded schedule item for the chunk-pipelined forward Mega kernel."""

    def __init__(
        self,
        group_idx: Int32,
        problem_idx: Int32,
        tile_idx: Int32,
        tile_m_idx: Int32,
        tile_n_idx: Int32,
        num_m_tiles: Int32,
        num_n_tiles: Int32,
        num_k_tiles: Int32,
        num_output_tiles: Int32,
        m_size: Int32,
        n_size: Int32,
        k_size: Int32,
        full_m_size: Int32,
        full_n_size: Int32,
        group_rows: Int32,
        chunk_start: Int32,
        split_prefix: Int32,
        group_tile_prefix: Int32,
        scale_split_prefix: Int32,
        sched_chunk_idx: Int32,
    ):
        self.group_idx = group_idx
        self.problem_idx = problem_idx
        self.tile_idx = tile_idx
        self.tile_m_idx = tile_m_idx
        self.tile_n_idx = tile_n_idx
        self.num_m_tiles = num_m_tiles
        self.num_n_tiles = num_n_tiles
        self.num_k_tiles = num_k_tiles
        self.num_output_tiles = num_output_tiles
        self.m_size = m_size
        self.n_size = n_size
        self.k_size = k_size
        self.full_m_size = full_m_size
        self.full_n_size = full_n_size
        self.group_rows = group_rows
        self.chunk_start = chunk_start
        self.split_prefix = split_prefix
        self.group_tile_prefix = group_tile_prefix
        self.scale_split_prefix = scale_split_prefix
        self.sched_chunk_idx = sched_chunk_idx

    @property
    def is_valid_tile(self):
        return self.group_idx >= Int32(0)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for value in self._values():
            values.extend(extract_mlir_values(value))
        return values

    def _values(self):
        return (
            self.group_idx,
            self.problem_idx,
            self.tile_idx,
            self.tile_m_idx,
            self.tile_n_idx,
            self.num_m_tiles,
            self.num_n_tiles,
            self.num_k_tiles,
            self.num_output_tiles,
            self.m_size,
            self.n_size,
            self.k_size,
            self.full_m_size,
            self.full_n_size,
            self.group_rows,
            self.chunk_start,
            self.split_prefix,
            self.group_tile_prefix,
            self.scale_split_prefix,
            self.sched_chunk_idx,
        )

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "ChunkedMegaWorkTileInfo":
        old_values = self._values()
        assert len(values) == len(old_values)
        return ChunkedMegaWorkTileInfo(
            *[
                new_from_mlir_values(old_value, [value])
                for old_value, value in zip(old_values, values)
            ]
        )


CHUNKED_MEGA_WORK_INFO_FIELD_NAMES: tuple[str, ...] = (
    "group_idx",
    "problem_idx",
    "tile_idx",
    "tile_m_idx",
    "tile_n_idx",
    "num_m_tiles",
    "num_n_tiles",
    "num_k_tiles",
    "num_output_tiles",
    "m_size",
    "n_size",
    "k_size",
    "full_m_size",
    "full_n_size",
    "group_rows",
    "chunk_start",
    "split_prefix",
    "group_tile_prefix",
    "scale_split_prefix",
    "sched_chunk_idx",
)
CHUNKED_MEGA_WORK_INFO_FIELDS: int = len(CHUNKED_MEGA_WORK_INFO_FIELD_NAMES)
_CHUNKED_MEGA_WORK_INFO_FIELD_INDEX = {
    name: index for index, name in enumerate(CHUNKED_MEGA_WORK_INFO_FIELD_NAMES)
}


def _chunked_mega_work_info_fields(*names: str) -> tuple[int, ...]:
    return tuple(_CHUNKED_MEGA_WORK_INFO_FIELD_INDEX[name] for name in names)


# Fields absent from a role tuple deserialize as zero. Every new consumer read
# must therefore add its field here instead of relying on the full work record.
_CHUNKED_MEGA_CONSUMER_WORK_INFO_FIELDS = _chunked_mega_work_info_fields(
    "group_idx",
    "problem_idx",
    "tile_m_idx",
    "tile_n_idx",
    "num_m_tiles",
    "num_n_tiles",
    "num_k_tiles",
    "num_output_tiles",
    "full_n_size",
    "chunk_start",
    "split_prefix",
    "group_tile_prefix",
    "scale_split_prefix",
    "sched_chunk_idx",
)
CHUNKED_MEGA_WORK_INFO_ROLE_ALL: int = 0
CHUNKED_MEGA_WORK_INFO_ROLE_TMA_B: int = 1
CHUNKED_MEGA_WORK_INFO_ROLE_MMA: int = 2
CHUNKED_MEGA_WORK_INFO_ROLE_EPILOGUE: int = 3
_CHUNKED_MEGA_TMA_B_WORK_INFO_FIELDS = _chunked_mega_work_info_fields(
    "group_idx",
    "problem_idx",
    "tile_m_idx",
    "tile_n_idx",
    "num_k_tiles",
    "num_output_tiles",
    "full_n_size",
    "chunk_start",
    "group_tile_prefix",
    "scale_split_prefix",
    "sched_chunk_idx",
)
_CHUNKED_MEGA_MMA_WORK_INFO_FIELDS = _chunked_mega_work_info_fields(
    "group_idx",
    "problem_idx",
    "tile_m_idx",
    "tile_n_idx",
    "num_k_tiles",
    "num_output_tiles",
    "group_tile_prefix",
)
_CHUNKED_MEGA_EPILOGUE_WORK_INFO_FIELDS = _chunked_mega_work_info_fields(
    "group_idx",
    "problem_idx",
    "tile_m_idx",
    "tile_n_idx",
    "num_m_tiles",
    "num_n_tiles",
    "num_k_tiles",
    "num_output_tiles",
    "full_n_size",
    "chunk_start",
    "split_prefix",
    "group_tile_prefix",
    "scale_split_prefix",
    "sched_chunk_idx",
)


@cute.jit
def _load_chunked_mega_work_field(
    work_info_smem_ptr: cute.Pointer,
    field_offset: Int32,
    load_field: cutlass.Constexpr[bool],
) -> Int32:
    value = Int32(0)
    if cutlass.const_expr(load_field):
        if cute.arch.lane_idx() == Int32(0):
            value = cute.arch.load(
                work_info_smem_ptr + field_offset,
                Int32,
                ss="cta",
            )
        value = cute.arch.shuffle_sync(value, Int32(0))
    return value


@cute.jit
def publish_chunked_mega_work_info(
    work: ChunkedMegaWorkTileInfo,
    work_info_smem_ptr: cute.Pointer,
    tile_buf: Int32,
) -> None:
    base = tile_buf * Int32(CHUNKED_MEGA_WORK_INFO_FIELDS)
    with cute.arch.elect_one():
        for field_idx, value in enumerate(work._values()):
            if field_idx in _CHUNKED_MEGA_CONSUMER_WORK_INFO_FIELDS:
                cute.arch.store(
                    work_info_smem_ptr + base + Int32(field_idx),
                    value,
                    ss="cta",
                )


@cute.jit
def load_chunked_mega_work_info(
    tile_idx: Int32,
    work_info_smem_ptr: cute.Pointer,
    tile_buf: Int32,
    broadcast_within_warp: cutlass.Constexpr[bool] = False,
    consumer_role: cutlass.Constexpr[int] = CHUNKED_MEGA_WORK_INFO_ROLE_ALL,
) -> ChunkedMegaWorkTileInfo:
    work = ChunkedMegaWorkTileInfo(
        Int32(-1), *([Int32(0)] * (CHUNKED_MEGA_WORK_INFO_FIELDS - 1))
    )
    if tile_idx < Int32(_TILE_SENTINEL):
        base = tile_buf * Int32(CHUNKED_MEGA_WORK_INFO_FIELDS)
        consumer_fields = _CHUNKED_MEGA_CONSUMER_WORK_INFO_FIELDS
        if cutlass.const_expr(consumer_role == CHUNKED_MEGA_WORK_INFO_ROLE_TMA_B):
            consumer_fields = _CHUNKED_MEGA_TMA_B_WORK_INFO_FIELDS
        elif cutlass.const_expr(consumer_role == CHUNKED_MEGA_WORK_INFO_ROLE_MMA):
            consumer_fields = _CHUNKED_MEGA_MMA_WORK_INFO_FIELDS
        elif cutlass.const_expr(consumer_role == CHUNKED_MEGA_WORK_INFO_ROLE_EPILOGUE):
            consumer_fields = _CHUNKED_MEGA_EPILOGUE_WORK_INFO_FIELDS
        if cutlass.const_expr(broadcast_within_warp):
            values = [
                _load_chunked_mega_work_field(
                    work_info_smem_ptr,
                    base + Int32(field_idx),
                    field_idx in consumer_fields,
                )
                for field_idx in range(CHUNKED_MEGA_WORK_INFO_FIELDS)
            ]
        else:
            values = [
                (
                    cute.arch.load(
                        work_info_smem_ptr + base + Int32(field_idx),
                        Int32,
                        ss="cta",
                    )
                    if field_idx in consumer_fields
                    else Int32(0)
                )
                for field_idx in range(CHUNKED_MEGA_WORK_INFO_FIELDS)
            ]
        work = ChunkedMegaWorkTileInfo(*values)
    return work


class ChunkedMegaProblemVisitor:
    """Stateful decoder for chunk-pipelined FC13/FC2 schedule items."""

    def __init__(
        self,
        split_sizes: cute.Tensor,
        N: Int32,
        K: Int32,
        group_idx: Int32,
        lead_chunk_idx: Int32,
        problem_idx: Int32,
        initialized: Int32,
        tile_start: Int32,
        tile_end: Int32,
        split_prefix: Int32,
        group_tile_prefix: Int32,
        scale_split_prefix: Int32,
        group_rows: Int32,
        chunk_start: Int32,
        m_size: Int32,
        n_size: Int32,
        k_size: Int32,
        full_m_size: Int32,
        full_n_size: Int32,
        num_m_tiles: Int32,
        num_n_tiles: Int32,
        num_k_tiles: Int32,
        num_output_tiles: Int32,
        local_rank: Int32,
        chunk_prefix: Int32,
        sched_chunk_idx: Int32,
        a0_group_idx: Int32,
        a0_linear_chunk: Int32,
        a0_chunk_prefix: Int32,
        a0_split_prefix: Int32,
        a0_group_tile_prefix: Int32,
        a0_scale_split_prefix: Int32,
        a1_group_idx: Int32,
        a1_linear_chunk: Int32,
        a1_chunk_prefix: Int32,
        a1_split_prefix: Int32,
        a1_group_tile_prefix: Int32,
        a1_scale_split_prefix: Int32,
        G: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        WORLD_SIZE: cutlass.Constexpr[int],
        SWAP_MN: cutlass.Constexpr[bool],
        CHUNK_ROWS: cutlass.Constexpr[int],
        LEAD_CHUNKS: cutlass.Constexpr[int],
        FC2_TILES_PER_ITEM: cutlass.Constexpr[int],
        STAGE_ALL_FC13: cutlass.Constexpr[bool],
        GLOBAL_LEAD_CHUNKS: cutlass.Constexpr[int] = 0,
    ):
        assert CHUNK_ROWS % BLOCK_SIZE_N == 0, (
            "CHUNK_ROWS must be divisible by BLOCK_SIZE_N"
        )
        assert not (STAGE_ALL_FC13 and GLOBAL_LEAD_CHUNKS > 0), (
            "GLOBAL_LEAD_CHUNKS and STAGE_ALL_FC13 are mutually exclusive"
        )
        self.split_sizes = split_sizes
        self.N = N
        self.K = K
        self.group_idx = group_idx
        self.lead_chunk_idx = lead_chunk_idx
        self.problem_idx = problem_idx
        self.initialized = initialized
        self.tile_start = tile_start
        self.tile_end = tile_end
        self.split_prefix = split_prefix
        self.group_tile_prefix = group_tile_prefix
        self.scale_split_prefix = scale_split_prefix
        self.group_rows = group_rows
        self.chunk_start = chunk_start
        self.m_size = m_size
        self.n_size = n_size
        self.k_size = k_size
        self.full_m_size = full_m_size
        self.full_n_size = full_n_size
        self.num_m_tiles = num_m_tiles
        self.num_n_tiles = num_n_tiles
        self.num_k_tiles = num_k_tiles
        self.num_output_tiles = num_output_tiles
        self.local_rank = local_rank
        self.chunk_prefix = chunk_prefix
        self.sched_chunk_idx = sched_chunk_idx
        # Cross-group lead-window cursors (GLOBAL_LEAD_CHUNKS > 0 only): arm 0
        # walks the FC13 chunk stream, arm 1 trails it by the window for FC2.
        self.a0_group_idx = a0_group_idx
        self.a0_linear_chunk = a0_linear_chunk
        self.a0_chunk_prefix = a0_chunk_prefix
        self.a0_split_prefix = a0_split_prefix
        self.a0_group_tile_prefix = a0_group_tile_prefix
        self.a0_scale_split_prefix = a0_scale_split_prefix
        self.a1_group_idx = a1_group_idx
        self.a1_linear_chunk = a1_linear_chunk
        self.a1_chunk_prefix = a1_chunk_prefix
        self.a1_split_prefix = a1_split_prefix
        self.a1_group_tile_prefix = a1_group_tile_prefix
        self.a1_scale_split_prefix = a1_scale_split_prefix
        self.G = G
        self.BLOCK_SIZE_M = BLOCK_SIZE_M
        self.BLOCK_SIZE_N = BLOCK_SIZE_N
        self.BLOCK_SIZE_K = BLOCK_SIZE_K
        self.FORCE_N_MAJOR = FORCE_N_MAJOR
        self.NUM_N_CLUSTERS = NUM_N_CLUSTERS
        self.WORLD_SIZE = WORLD_SIZE
        self.SWAP_MN = SWAP_MN
        self.CHUNK_ROWS = CHUNK_ROWS
        self.LEAD_CHUNKS = LEAD_CHUNKS
        self.FC2_TILES_PER_ITEM = FC2_TILES_PER_ITEM
        self.STAGE_ALL_FC13 = STAGE_ALL_FC13
        self.GLOBAL_LEAD_CHUNKS = GLOBAL_LEAD_CHUNKS

    def _state_values(self):
        return (
            self.N,
            self.K,
            self.group_idx,
            self.lead_chunk_idx,
            self.problem_idx,
            self.initialized,
            self.tile_start,
            self.tile_end,
            self.split_prefix,
            self.group_tile_prefix,
            self.scale_split_prefix,
            self.group_rows,
            self.chunk_start,
            self.m_size,
            self.n_size,
            self.k_size,
            self.full_m_size,
            self.full_n_size,
            self.num_m_tiles,
            self.num_n_tiles,
            self.num_k_tiles,
            self.num_output_tiles,
            self.local_rank,
            self.chunk_prefix,
            self.sched_chunk_idx,
            self.a0_group_idx,
            self.a0_linear_chunk,
            self.a0_chunk_prefix,
            self.a0_split_prefix,
            self.a0_group_tile_prefix,
            self.a0_scale_split_prefix,
            self.a1_group_idx,
            self.a1_linear_chunk,
            self.a1_chunk_prefix,
            self.a1_split_prefix,
            self.a1_group_tile_prefix,
            self.a1_scale_split_prefix,
        )

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = list(extract_mlir_values(self.split_sizes))
        for value in self._state_values():
            values.extend(extract_mlir_values(value))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "ChunkedMegaProblemVisitor":
        tensor_value_count = len(extract_mlir_values(self.split_sizes))
        split_sizes = new_from_mlir_values(
            self.split_sizes, values[:tensor_value_count]
        )
        old_values = self._state_values()
        runtime_values = values[tensor_value_count:]
        assert len(runtime_values) == len(old_values)
        state_values = [
            new_from_mlir_values(old_value, [value])
            for old_value, value in zip(old_values, runtime_values)
        ]
        return ChunkedMegaProblemVisitor(
            split_sizes,
            *state_values,
            G=self.G,
            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            BLOCK_SIZE_K=self.BLOCK_SIZE_K,
            FORCE_N_MAJOR=self.FORCE_N_MAJOR,
            NUM_N_CLUSTERS=self.NUM_N_CLUSTERS,
            WORLD_SIZE=self.WORLD_SIZE,
            SWAP_MN=self.SWAP_MN,
            CHUNK_ROWS=self.CHUNK_ROWS,
            LEAD_CHUNKS=self.LEAD_CHUNKS,
            FC2_TILES_PER_ITEM=self.FC2_TILES_PER_ITEM,
            STAGE_ALL_FC13=self.STAGE_ALL_FC13,
            GLOBAL_LEAD_CHUNKS=self.GLOBAL_LEAD_CHUNKS,
        )

    @staticmethod
    @cute.jit
    def create(
        split_sizes: cute.Tensor,
        N: Int32,
        K: Int32,
        G: cutlass.Constexpr[int],
        BLOCK_SIZE_M: cutlass.Constexpr[int],
        BLOCK_SIZE_N: cutlass.Constexpr[int],
        BLOCK_SIZE_K: cutlass.Constexpr[int],
        FORCE_N_MAJOR: cutlass.Constexpr[bool],
        NUM_N_CLUSTERS: cutlass.Constexpr[int],
        local_rank: Int32,
        WORLD_SIZE: cutlass.Constexpr[int],
        SWAP_MN: cutlass.Constexpr[bool],
        CHUNK_ROWS: cutlass.Constexpr[int],
        LEAD_CHUNKS: cutlass.Constexpr[int],
        FC2_TILES_PER_ITEM: cutlass.Constexpr[int],
        STAGE_ALL_FC13: cutlass.Constexpr[bool],
        GLOBAL_LEAD_CHUNKS: cutlass.Constexpr[int] = 0,
    ) -> "ChunkedMegaProblemVisitor":
        return ChunkedMegaProblemVisitor(
            split_sizes,
            N,
            K,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            local_rank,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            G,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            FORCE_N_MAJOR,
            NUM_N_CLUSTERS,
            WORLD_SIZE,
            SWAP_MN,
            CHUNK_ROWS,
            LEAD_CHUNKS,
            FC2_TILES_PER_ITEM,
            STAGE_ALL_FC13,
            GLOBAL_LEAD_CHUNKS,
        )

    @cute.jit
    def _advance_slot(self) -> None:
        if cutlass.const_expr(self.GLOBAL_LEAD_CHUNKS > 0):
            # Cross-group lead window: consuming a slot advances that arm's
            # chunk cursor; parity alternates FC13/FC2 arms.
            if self.problem_idx == Int32(0):
                self.a0_linear_chunk += Int32(1)
                self.problem_idx = Int32(1)
            else:
                self.a1_linear_chunk += Int32(1)
                self.problem_idx = Int32(0)
        elif cutlass.const_expr(self.STAGE_ALL_FC13):
            self.lead_chunk_idx += Int32(1)
        else:
            if self.problem_idx == Int32(0):
                self.problem_idx = Int32(1)
            else:
                self.problem_idx = Int32(0)
                self.lead_chunk_idx += Int32(1)

    @cute.jit
    def _advance_group_prefixes(
        self,
        rows: Int32,
        chunk_prefix: Int32,
        split_prefix: Int32,
        group_tile_prefix: Int32,
        scale_split_prefix: Int32,
    ):
        """Advance the per-group prefix cursors past a group of ``rows``.
        Single source of the prefix arithmetic for both the per-group
        (``_advance_group``) and cross-group (``_arm_normalize``)
        schedules."""
        chunk_prefix += (rows + Int32(self.CHUNK_ROWS) - Int32(1)) // Int32(
            self.CHUNK_ROWS
        )
        split_prefix += rows
        scale_split_prefix = advance_blockscaled_scale_start(
            scale_split_prefix,
            rows,
            self.SWAP_MN,
            self.BLOCK_SIZE_N,
        )
        if cutlass.const_expr(self.SWAP_MN):
            group_tile_prefix += (rows + self.BLOCK_SIZE_N - 1) // self.BLOCK_SIZE_N
        else:
            group_tile_prefix += (rows + self.BLOCK_SIZE_M - 1) // self.BLOCK_SIZE_M
        return chunk_prefix, split_prefix, group_tile_prefix, scale_split_prefix

    @cute.jit
    def _advance_group(self) -> None:
        (
            self.chunk_prefix,
            self.split_prefix,
            self.group_tile_prefix,
            self.scale_split_prefix,
        ) = self._advance_group_prefixes(
            self.group_rows,
            self.chunk_prefix,
            self.split_prefix,
            self.group_tile_prefix,
            self.scale_split_prefix,
        )
        self.group_idx += Int32(1)
        self.lead_chunk_idx = Int32(0)
        if cutlass.const_expr(self.STAGE_ALL_FC13):
            if self.group_idx >= self.G and self.problem_idx == Int32(0):
                self.group_idx = Int32(0)
                self.problem_idx = Int32(1)
                self.split_prefix = Int32(0)
                self.group_tile_prefix = Int32(0)
                self.scale_split_prefix = Int32(0)
                self.chunk_prefix = Int32(0)
        else:
            self.problem_idx = Int32(0)

    @cute.jit
    def _arm_normalize(
        self,
        g: Int32,
        linear: Int32,
        chunk_prefix: Int32,
        split_prefix: Int32,
        group_tile_prefix: Int32,
        scale_split_prefix: Int32,
    ):
        """Advance an arm cursor past exhausted and empty groups."""
        settled = cutlass.Boolean(False)
        while (not settled) and g < Int32(self.G):
            rows = Int32(self.split_sizes[g])
            num_chunks = (rows + Int32(self.CHUNK_ROWS) - Int32(1)) // Int32(
                self.CHUNK_ROWS
            )
            if linear < num_chunks:
                settled = cutlass.Boolean(True)
            else:
                (
                    chunk_prefix,
                    split_prefix,
                    group_tile_prefix,
                    scale_split_prefix,
                ) = self._advance_group_prefixes(
                    rows,
                    chunk_prefix,
                    split_prefix,
                    group_tile_prefix,
                    scale_split_prefix,
                )
                g += Int32(1)
                linear = Int32(0)
        return (
            g,
            linear,
            chunk_prefix,
            split_prefix,
            group_tile_prefix,
            scale_split_prefix,
        )

    @cute.jit
    def _decode_global_slot(
        self,
        g: Int32,
        linear: Int32,
        chunk_prefix: Int32,
        split_prefix: Int32,
        group_tile_prefix: Int32,
        scale_split_prefix: Int32,
        is_fc2: cutlass.Constexpr[bool],
    ) -> None:
        """Publish one arm's current chunk as the decoded schedule slot.

        Keep in sync with the per-group decode inlined in
        ``_seek_valid_slot``, its runtime-``problem_idx`` twin: chunk
        rotation, ``sched_chunk_idx``, m/n/k sizing, the ``_tile_grid``
        dispatch and ``num_output_tiles`` must stay field-for-field
        identical so ring slots and credits pair the two schedules'
        chunks the same way."""
        rows = Int32(self.split_sizes[g])
        num_chunks = (rows + Int32(self.CHUNK_ROWS) - Int32(1)) // Int32(
            self.CHUNK_ROWS
        )
        # Same per-rank chunk rotation as the per-group schedule; FC13 and
        # FC2 walk identical rotated orders, so the schedule index pairs
        # them (ring slots and credits key off sched_chunk_idx).
        chunk_offset = Int32(0)
        if num_chunks > Int32(0):
            chunk_offset = (
                self.local_rank
                * ((num_chunks + self.WORLD_SIZE - 1) // self.WORLD_SIZE)
            ) % num_chunks
        chunk_idx = (linear + chunk_offset) % num_chunks
        self.sched_chunk_idx = chunk_prefix + linear
        self.chunk_start = chunk_idx * Int32(self.CHUNK_ROWS)
        chunk_rows = min(
            Int32(self.CHUNK_ROWS),
            rows - self.chunk_start,
        )
        if cutlass.const_expr(is_fc2):
            output_dim = self.K
            contraction_dim = self.N
        else:
            output_dim = 2 * self.N
            contraction_dim = self.K
        if cutlass.const_expr(self.SWAP_MN):
            self.m_size = output_dim
            self.n_size = chunk_rows
            self.full_m_size = output_dim
            self.full_n_size = rows
        else:
            self.m_size = chunk_rows
            self.n_size = output_dim
            self.full_m_size = rows
            self.full_n_size = output_dim
        self.k_size = contraction_dim
        num_tiles = Int32(0)
        if cutlass.const_expr(not is_fc2):
            (
                self.num_m_tiles,
                self.num_n_tiles,
                self.num_k_tiles,
                num_tiles,
            ) = _tile_grid(
                self.m_size,
                self.n_size,
                self.k_size,
                self.BLOCK_SIZE_M,
                self.BLOCK_SIZE_N,
                self.BLOCK_SIZE_K,
                self.FORCE_N_MAJOR,
                self.NUM_N_CLUSTERS,
                self.SWAP_MN,
            )
        else:
            if cutlass.const_expr(self.SWAP_MN):
                (
                    self.num_m_tiles,
                    self.num_n_tiles,
                    self.num_k_tiles,
                    num_tiles,
                ) = _tile_grid(
                    self.m_size,
                    self.n_size,
                    self.k_size,
                    self.BLOCK_SIZE_M * self.FC2_TILES_PER_ITEM,
                    self.BLOCK_SIZE_N,
                    self.BLOCK_SIZE_K,
                    self.FORCE_N_MAJOR,
                    self.NUM_N_CLUSTERS,
                    self.SWAP_MN,
                )
            else:
                (
                    self.num_m_tiles,
                    self.num_n_tiles,
                    self.num_k_tiles,
                    num_tiles,
                ) = _tile_grid(
                    self.m_size,
                    self.n_size,
                    self.k_size,
                    self.BLOCK_SIZE_M,
                    self.BLOCK_SIZE_N * self.FC2_TILES_PER_ITEM,
                    self.BLOCK_SIZE_K,
                    self.FORCE_N_MAJOR,
                    self.NUM_N_CLUSTERS,
                    self.SWAP_MN,
                )
        if cutlass.const_expr(self.SWAP_MN):
            self.num_output_tiles = (
                self.m_size + self.BLOCK_SIZE_M - 1
            ) // self.BLOCK_SIZE_M
        else:
            self.num_output_tiles = (
                self.n_size + self.BLOCK_SIZE_N - 1
            ) // self.BLOCK_SIZE_N
        self.tile_end = self.tile_start + num_tiles
        self.group_idx = g
        self.group_rows = rows
        self.split_prefix = split_prefix
        self.group_tile_prefix = group_tile_prefix
        self.scale_split_prefix = scale_split_prefix

    @cute.jit
    def _seek_valid_slot_global(self) -> None:
        found = cutlass.Boolean(False)
        done = cutlass.Boolean(False)
        while (not found) and (not done):
            if self.problem_idx == Int32(0):
                (
                    self.a0_group_idx,
                    self.a0_linear_chunk,
                    self.a0_chunk_prefix,
                    self.a0_split_prefix,
                    self.a0_group_tile_prefix,
                    self.a0_scale_split_prefix,
                ) = self._arm_normalize(
                    self.a0_group_idx,
                    self.a0_linear_chunk,
                    self.a0_chunk_prefix,
                    self.a0_split_prefix,
                    self.a0_group_tile_prefix,
                    self.a0_scale_split_prefix,
                )
                if self.a0_group_idx < Int32(self.G):
                    self._decode_global_slot(
                        self.a0_group_idx,
                        self.a0_linear_chunk,
                        self.a0_chunk_prefix,
                        self.a0_split_prefix,
                        self.a0_group_tile_prefix,
                        self.a0_scale_split_prefix,
                        is_fc2=False,
                    )
                    found = cutlass.Boolean(True)
                else:
                    # FC13 arm exhausted; remaining slots all belong to FC2.
                    self.problem_idx = Int32(1)
            else:
                (
                    self.a1_group_idx,
                    self.a1_linear_chunk,
                    self.a1_chunk_prefix,
                    self.a1_split_prefix,
                    self.a1_group_tile_prefix,
                    self.a1_scale_split_prefix,
                ) = self._arm_normalize(
                    self.a1_group_idx,
                    self.a1_linear_chunk,
                    self.a1_chunk_prefix,
                    self.a1_split_prefix,
                    self.a1_group_tile_prefix,
                    self.a1_scale_split_prefix,
                )
                if self.a1_group_idx >= Int32(self.G):
                    done = cutlass.Boolean(True)
                else:
                    # FC2 may trail FC13 by no less than the window; once the
                    # FC13 arm is exhausted the tail drains unconditionally.
                    a0_pos = self.a0_chunk_prefix + self.a0_linear_chunk
                    a1_pos = self.a1_chunk_prefix + self.a1_linear_chunk
                    window_open = (
                        a1_pos + Int32(self.GLOBAL_LEAD_CHUNKS) <= a0_pos
                    ) or (self.a0_group_idx >= Int32(self.G))
                    if window_open:
                        self._decode_global_slot(
                            self.a1_group_idx,
                            self.a1_linear_chunk,
                            self.a1_chunk_prefix,
                            self.a1_split_prefix,
                            self.a1_group_tile_prefix,
                            self.a1_scale_split_prefix,
                            is_fc2=True,
                        )
                        found = cutlass.Boolean(True)
                    else:
                        self.problem_idx = Int32(0)
        if done:
            # Sentinel: get_work and the containment loop key off group_idx.
            self.group_idx = Int32(self.G)

    @cute.jit
    def _seek_valid_slot(self) -> None:  # noqa: C901
        if cutlass.const_expr(self.GLOBAL_LEAD_CHUNKS > 0):
            self._seek_valid_slot_global()
            return
        found = cutlass.Boolean(False)
        while (not found) and self.group_idx < self.G:
            self.group_rows = Int32(self.split_sizes[self.group_idx])
            num_chunks = (self.group_rows + self.CHUNK_ROWS - 1) // self.CHUNK_ROWS
            num_slots = num_chunks
            if cutlass.const_expr(not self.STAGE_ALL_FC13):
                num_slots += Int32(self.LEAD_CHUNKS)
            if self.lead_chunk_idx >= num_slots:
                self._advance_group()
            else:
                linear_chunk_idx = self.lead_chunk_idx
                if cutlass.const_expr(not self.STAGE_ALL_FC13):
                    linear_chunk_idx -= self.problem_idx * Int32(self.LEAD_CHUNKS)
                if linear_chunk_idx >= Int32(0) and linear_chunk_idx < num_chunks:
                    # Ring slots key off the schedule position, not the
                    # physical chunk: the rank rotation permutes chunks
                    # within a group, but FC13 and FC2 walk the same
                    # rotated order, so the schedule index pairs them.
                    # Keep this decode in sync with ``_decode_global_slot``,
                    # the cross-group schedule's constexpr-``is_fc2`` twin.
                    self.sched_chunk_idx = self.chunk_prefix + linear_chunk_idx
                    chunk_offset = Int32(0)
                    if num_chunks > Int32(0):
                        chunk_offset = (
                            self.local_rank
                            * ((num_chunks + self.WORLD_SIZE - 1) // self.WORLD_SIZE)
                        ) % num_chunks
                    chunk_idx = (linear_chunk_idx + chunk_offset) % num_chunks
                    self.chunk_start = chunk_idx * Int32(self.CHUNK_ROWS)
                    chunk_rows = min(
                        Int32(self.CHUNK_ROWS),
                        self.group_rows - self.chunk_start,
                    )
                    output_dim = 2 * self.N
                    contraction_dim = self.K
                    if self.problem_idx == Int32(1):
                        output_dim = self.K
                        contraction_dim = self.N
                    if cutlass.const_expr(self.SWAP_MN):
                        self.m_size = output_dim
                        self.n_size = chunk_rows
                        self.full_m_size = output_dim
                        self.full_n_size = self.group_rows
                    else:
                        self.m_size = chunk_rows
                        self.n_size = output_dim
                        self.full_m_size = self.group_rows
                        self.full_n_size = output_dim
                    self.k_size = contraction_dim
                    num_tiles = Int32(0)
                    if self.problem_idx == Int32(0):
                        (
                            self.num_m_tiles,
                            self.num_n_tiles,
                            self.num_k_tiles,
                            num_tiles,
                        ) = _tile_grid(
                            self.m_size,
                            self.n_size,
                            self.k_size,
                            self.BLOCK_SIZE_M,
                            self.BLOCK_SIZE_N,
                            self.BLOCK_SIZE_K,
                            self.FORCE_N_MAJOR,
                            self.NUM_N_CLUSTERS,
                            self.SWAP_MN,
                        )
                    else:
                        if cutlass.const_expr(self.SWAP_MN):
                            (
                                self.num_m_tiles,
                                self.num_n_tiles,
                                self.num_k_tiles,
                                num_tiles,
                            ) = _tile_grid(
                                self.m_size,
                                self.n_size,
                                self.k_size,
                                self.BLOCK_SIZE_M * self.FC2_TILES_PER_ITEM,
                                self.BLOCK_SIZE_N,
                                self.BLOCK_SIZE_K,
                                self.FORCE_N_MAJOR,
                                self.NUM_N_CLUSTERS,
                                self.SWAP_MN,
                            )
                        else:
                            (
                                self.num_m_tiles,
                                self.num_n_tiles,
                                self.num_k_tiles,
                                num_tiles,
                            ) = _tile_grid(
                                self.m_size,
                                self.n_size,
                                self.k_size,
                                self.BLOCK_SIZE_M,
                                self.BLOCK_SIZE_N * self.FC2_TILES_PER_ITEM,
                                self.BLOCK_SIZE_K,
                                self.FORCE_N_MAJOR,
                                self.NUM_N_CLUSTERS,
                                self.SWAP_MN,
                            )
                    if cutlass.const_expr(self.SWAP_MN):
                        self.num_output_tiles = (
                            self.m_size + self.BLOCK_SIZE_M - 1
                        ) // self.BLOCK_SIZE_M
                    else:
                        self.num_output_tiles = (
                            self.n_size + self.BLOCK_SIZE_N - 1
                        ) // self.BLOCK_SIZE_N
                    self.tile_end = self.tile_start + num_tiles
                    found = cutlass.Boolean(True)
                else:
                    self._advance_slot()

    @cute.jit
    def _advance_problem_to_contain(self, tile_idx: Int32) -> None:
        while (
            self.initialized == Int32(0) or tile_idx >= self.tile_end
        ) and self.group_idx < self.G:
            if self.initialized == Int32(0):
                self.initialized = Int32(1)
            else:
                self.tile_start = self.tile_end
                self._advance_slot()
            self._seek_valid_slot()

    @cute.jit
    def get_work(self, tile_idx: Int32) -> ChunkedMegaWorkTileInfo:
        self._advance_problem_to_contain(tile_idx)
        work = ChunkedMegaWorkTileInfo(
            Int32(-1), *([Int32(0)] * (CHUNKED_MEGA_WORK_INFO_FIELDS - 1))
        )
        if self.group_idx < self.G and tile_idx < self.tile_end:
            local_tile_idx = tile_idx - self.tile_start
            tile_m_idx, tile_n_idx = _tile_coords(
                local_tile_idx,
                self.num_m_tiles,
                self.num_n_tiles,
                self.m_size,
                self.n_size,
                self.FORCE_N_MAJOR,
                self.NUM_N_CLUSTERS,
                Int32(0),
                self.WORLD_SIZE,
                self.SWAP_MN,
            )
            tiles_per_item = Int32(1)
            if self.problem_idx == Int32(1):
                tiles_per_item = Int32(self.FC2_TILES_PER_ITEM)
            if cutlass.const_expr(self.SWAP_MN):
                tile_m_idx *= tiles_per_item
                tile_n_idx += self.chunk_start // self.BLOCK_SIZE_N
            else:
                tile_m_idx += self.chunk_start // self.BLOCK_SIZE_M
                tile_n_idx *= tiles_per_item
            work = ChunkedMegaWorkTileInfo(
                self.group_idx,
                self.problem_idx,
                local_tile_idx,
                tile_m_idx,
                tile_n_idx,
                self.num_m_tiles,
                self.num_n_tiles,
                self.num_k_tiles,
                self.num_output_tiles,
                self.m_size,
                self.n_size,
                self.k_size,
                self.full_m_size,
                self.full_n_size,
                self.group_rows,
                self.chunk_start,
                self.split_prefix,
                self.group_tile_prefix,
                self.scale_split_prefix,
                self.sched_chunk_idx,
            )
        return work


@cute.jit
def stage_expert_metadata(
    split_sizes: cute.Tensor,
    split_sizes_smem_ptr: cute.Pointer,
    G: cutlass.Constexpr[int],
    synchronize: cutlass.Constexpr[bool] = True,
) -> cute.Tensor:
    """Cooperatively cache per-expert token counts in CTA-local SMEM."""
    tidx, _, _ = cute.arch.thread_idx()
    block_dim, _, _ = cute.arch.block_dim()
    g = cutlass.Int32(tidx)
    while g < G:
        cute.arch.store(
            split_sizes_smem_ptr + g,
            cutlass.Int32(split_sizes[g]),
            ss="cta",
        )
        g += cutlass.Int32(block_dim)
    if cutlass.const_expr(synchronize):
        cute.arch.sync_threads()
    return cute.make_tensor(
        split_sizes_smem_ptr,
        cute.make_layout((G,), stride=(1,)),
    )


@cute.jit
def _get_bufidx_phase(accum_cnt: Int32, NUM_BUFFERS_KV: cutlass.Constexpr[int]):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@cute.jit
def _producer_fetch_tile_idx(
    counter_ptr: cute.Pointer,
    cluster_cta_rank: Int32,
    tile_cta_bar_ptr: cute.Pointer,
    tile_id_smem_ptr: cute.Pointer,
    tile_buf: Int32,
    cta_sync_cnt: Int32,
    NUM_CTAS: cutlass.Constexpr[int],
    NUM_CTA_SYNC_BARS: cutlass.Constexpr[int],
):
    bar_idx = Int32(0)
    bar_phase = Int32(0)
    if cutlass.const_expr(NUM_CTAS == 2):
        bar_idx, bar_phase = _get_bufidx_phase(cta_sync_cnt, NUM_CTA_SYNC_BARS)

    if cluster_cta_rank == 0:
        with cute.arch.elect_one():
            tile_idx_local = cute.arch.atomic_add(
                counter_ptr,
                Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            cute.arch.store(tile_id_smem_ptr + tile_buf, tile_idx_local, ss="cta")
            if cutlass.const_expr(NUM_CTAS == 2):
                remote_ptr = _map_remote_smem_ptr(
                    tile_id_smem_ptr + tile_buf,
                    cta_rank_in_cluster=1,
                )
                cute.arch.store(remote_ptr, tile_idx_local, ss="cluster")
                cute.arch.fence_proxy(kind="async.shared", space="cluster")
                cute.arch.mbarrier_arrive(
                    tile_cta_bar_ptr + bar_idx,
                    peer_cta_rank_in_cluster=1,
                )
        if cutlass.const_expr(NUM_CTAS == 2):
            cute.arch.mbarrier_wait(tile_cta_bar_ptr + bar_idx, bar_phase)
    else:
        if cutlass.const_expr(NUM_CTAS == 2):
            cute.arch.mbarrier_wait(tile_cta_bar_ptr + bar_idx, bar_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(
                    tile_cta_bar_ptr + bar_idx,
                    peer_cta_rank_in_cluster=0,
                )
    cute.arch.sync_warp()
    tile_idx = cute.arch.load(tile_id_smem_ptr + tile_buf, Int32, ss="cta")
    cute.arch.sync_warp()
    return tile_idx


@cute.jit
def _producer_wait_tile_empty(
    tile_id_producer_bars_ptr: cute.Pointer,
    accum_cnt: Int32,
    cluster_cta_rank: Int32,
    NUM_TILE_BUFFERS: cutlass.Constexpr[int],
):
    buf, phase = _get_bufidx_phase(accum_cnt, NUM_TILE_BUFFERS)
    if cluster_cta_rank == 0:
        cute.arch.mbarrier_wait(tile_id_producer_bars_ptr + buf, phase ^ 1)


@cute.jit
def _producer_signal_tile_ready(
    tile_id_smem_ptr: cute.Pointer,
    tile_id_consumer_bars_ptr: cute.Pointer,
    tile_buf: Int32,
    tile_idx: Int32,
    WRITE_SMEM: cutlass.Constexpr[bool],
):
    with cute.arch.elect_one():
        if cutlass.const_expr(WRITE_SMEM):
            cute.arch.store(tile_id_smem_ptr + tile_buf, tile_idx, ss="cta")
        cute.arch.mbarrier_arrive(tile_id_consumer_bars_ptr + tile_buf)


@cute.jit
def _epi_signal_tile_done(
    tile_id_producer_bars_ptr: cute.Pointer,
    tile_buf: Int32,
    cluster_cta_rank: Int32,
    NUM_CTAS: cutlass.Constexpr[int],
):
    with cute.arch.elect_one():
        if cutlass.const_expr(NUM_CTAS == 2):
            if cluster_cta_rank == 0:
                cute.arch.mbarrier_arrive(tile_id_producer_bars_ptr + tile_buf)
            else:
                cute.arch.mbarrier_arrive(
                    tile_id_producer_bars_ptr + tile_buf,
                    peer_cta_rank_in_cluster=0,
                )
        else:
            cute.arch.mbarrier_arrive(tile_id_producer_bars_ptr + tile_buf)


@cute.jit
def _wait_tile_ready(
    accum_cnt: Int32,
    NUM_TILE_BUFFERS: cutlass.Constexpr[int],
    tile_id_consumer_bars_ptr: cute.Pointer,
) -> Int32:
    buf, phase = _get_bufidx_phase(accum_cnt, NUM_TILE_BUFFERS)
    cute.arch.mbarrier_wait(tile_id_consumer_bars_ptr + buf, phase)
    return buf


@cute.jit
def _preread_tile_idx(
    accum_cnt: Int32,
    NUM_TILE_BUFFERS: cutlass.Constexpr[int],
    tile_id_consumer_bars_ptr: cute.Pointer,
    tile_id_smem_ptr: cute.Pointer,
):
    buf = _wait_tile_ready(
        accum_cnt,
        NUM_TILE_BUFFERS,
        tile_id_consumer_bars_ptr,
    )
    return cute.arch.load(tile_id_smem_ptr + buf, Int32, ss="cta")


class TileScheduler(Protocol):
    def __extract_mlir_values__(self) -> List[ir.Value]: ...

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "TileScheduler": ...

    def initial_work_tile_info(self) -> Int32: ...

    def producer_publish_tile(self, accum_cnt_out: Int32) -> None: ...

    def producer_wait_tile_released(self, accum_cnt_out: Int32) -> None: ...

    def producer_publish_termination(self, accum_cnt_out: Int32) -> None: ...

    def consumer_release_tile(
        self,
        tile_producer_mbar: cute.Pointer,
        tile_buf: Int32,
        cluster_cta_rank: Int32,
        is_leader_warp: cutlass.Boolean,
    ) -> None: ...

    def advance_producer(self, accum_cnt_out: Int32) -> Int32: ...

    def advance_consumer(self, accum_cnt_tile: Int32) -> Int32: ...


# =============================================================================
# Dynamic (atomic-counter) tile scheduler
# =============================================================================


class DynamicTileScheduler:
    """Atomic-counter persistent tile scheduler.

    Mirrors the base ``grouped_gemm.py`` protocol: the TMA producer warp
    atomic-fetches the next ``tile_idx`` into ``tile_id_smem`` and arms
    ``tile_consumer_mbar``; MMA / epilog consumer warps pre-read the same
    SMEM slot. ``__init__`` carries the cluster-scoped pointers + the
    runtime ``tile_idx`` / ``cta_sync_cnt`` state so each warp role holds
    its own scheduler instance.

    Producers track ``cta_sync_cnt`` so the leader CTA's atomic fetch can
    pair with the ping-pong CTA-sync barrier. Consumers ignore it.
    """

    def __init__(
        self,
        tile_idx: Int32,
        cta_sync_cnt: Int32,
        counter_ptr: cute.Pointer,
        tile_cta_bar_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        tile_consumer_mbar: cute.Pointer,
        tile_producer_mbar: cute.Pointer,
        cluster_cta_rank: Int32,
        NUM_CTAS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        NUM_TILE_CTA_BARS: cutlass.Constexpr[int],
    ):
        self.tile_idx = tile_idx
        self.cta_sync_cnt = cta_sync_cnt
        self.counter_ptr = counter_ptr
        self.tile_cta_bar_mbar = tile_cta_bar_mbar
        self.tile_id_smem_ptr = tile_id_smem_ptr
        self.tile_consumer_mbar = tile_consumer_mbar
        self.tile_producer_mbar = tile_producer_mbar
        self.cluster_cta_rank = cluster_cta_rank
        self.NUM_CTAS = NUM_CTAS
        self.NUM_TILE_BUFFERS = NUM_TILE_BUFFERS
        self.NUM_TILE_CTA_BARS = NUM_TILE_CTA_BARS

    # --------------------------------------------------------------------- #
    # MLIR value serialization. Only ``tile_idx`` and ``cta_sync_cnt`` are
    # SSA values; everything else is a constexpr or a pointer (constants
    # at JIT time).
    # --------------------------------------------------------------------- #

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = list(extract_mlir_values(self.tile_idx))
        values.extend(extract_mlir_values(self.cta_sync_cnt))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "DynamicTileScheduler":
        assert len(values) == 2
        return DynamicTileScheduler(
            tile_idx=new_from_mlir_values(self.tile_idx, [values[0]]),
            cta_sync_cnt=new_from_mlir_values(self.cta_sync_cnt, [values[1]]),
            counter_ptr=self.counter_ptr,
            tile_cta_bar_mbar=self.tile_cta_bar_mbar,
            tile_id_smem_ptr=self.tile_id_smem_ptr,
            tile_consumer_mbar=self.tile_consumer_mbar,
            tile_producer_mbar=self.tile_producer_mbar,
            cluster_cta_rank=self.cluster_cta_rank,
            NUM_CTAS=self.NUM_CTAS,
            NUM_TILE_BUFFERS=self.NUM_TILE_BUFFERS,
            NUM_TILE_CTA_BARS=self.NUM_TILE_CTA_BARS,
        )

    # --------------------------------------------------------------------- #
    # Factories. ``create_producer`` issues the first atomic fetch and
    # increments ``cta_sync_cnt`` for the next leader-rank ping-pong slot.
    # ``create_consumer`` waits on ``tile_consumer_mbar[0]`` and reads
    # ``tile_id_smem[0]`` to seed the loop with the first tile_idx.
    # --------------------------------------------------------------------- #

    @staticmethod
    @cute.jit
    def create_producer(
        counter_ptr: cute.Pointer,
        tile_cta_bar_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        tile_consumer_mbar: cute.Pointer,
        tile_producer_mbar: cute.Pointer,
        cluster_cta_rank: Int32,
        NUM_CTAS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        NUM_TILE_CTA_BARS: cutlass.Constexpr[int],
    ) -> "DynamicTileScheduler":
        cta_sync_cnt = Int32(0)
        tile_idx = _producer_fetch_tile_idx(
            counter_ptr,
            cluster_cta_rank,
            tile_cta_bar_mbar,
            tile_id_smem_ptr,
            Int32(0),
            cta_sync_cnt,
            NUM_CTAS,
            NUM_CTA_SYNC_BARS=NUM_TILE_CTA_BARS,
        )
        cta_sync_cnt += Int32(1)
        return DynamicTileScheduler(
            tile_idx=tile_idx,
            cta_sync_cnt=cta_sync_cnt,
            counter_ptr=counter_ptr,
            tile_cta_bar_mbar=tile_cta_bar_mbar,
            tile_id_smem_ptr=tile_id_smem_ptr,
            tile_consumer_mbar=tile_consumer_mbar,
            tile_producer_mbar=tile_producer_mbar,
            cluster_cta_rank=cluster_cta_rank,
            NUM_CTAS=NUM_CTAS,
            NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
            NUM_TILE_CTA_BARS=NUM_TILE_CTA_BARS,
        )

    @staticmethod
    @cute.jit
    def create_consumer(
        tile_consumer_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        NUM_CTAS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
    ) -> "DynamicTileScheduler":
        tile_idx = _preread_tile_idx(
            Int32(0), NUM_TILE_BUFFERS, tile_consumer_mbar, tile_id_smem_ptr
        )
        # ``counter_ptr`` and ``tile_cta_bar_mbar`` are producer-only; the
        # consumer path never dereferences them, but the class field types
        # require valid ``cute.Pointer`` objects. Re-use one of the
        # consumer pointers as a harmless placeholder.
        return DynamicTileScheduler(
            tile_idx=tile_idx,
            cta_sync_cnt=Int32(0),
            counter_ptr=tile_consumer_mbar,
            tile_cta_bar_mbar=tile_consumer_mbar,
            tile_id_smem_ptr=tile_id_smem_ptr,
            tile_consumer_mbar=tile_consumer_mbar,
            tile_producer_mbar=tile_consumer_mbar,
            cluster_cta_rank=Int32(0),
            NUM_CTAS=NUM_CTAS,
            NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
            NUM_TILE_CTA_BARS=0,
        )

    # --------------------------------------------------------------------- #
    # Tile iteration. Both ``advance_producer`` and ``advance_consumer``
    # leave ``self.tile_idx`` updated in place — return value is provided
    # for chaining and parity with the static scheduler.
    # --------------------------------------------------------------------- #

    @cute.jit
    def initial_work_tile_info(self) -> Int32:
        return self.tile_idx

    @cute.jit
    def producer_publish_tile(self, accum_cnt_out: Int32) -> None:
        tile_buf = accum_cnt_out % self.NUM_TILE_BUFFERS
        _producer_signal_tile_ready(
            tile_id_smem_ptr=self.tile_id_smem_ptr,
            tile_id_consumer_bars_ptr=self.tile_consumer_mbar,
            tile_buf=tile_buf,
            tile_idx=self.tile_idx,
            WRITE_SMEM=False,
        )

    @cute.jit
    def producer_wait_tile_released(self, accum_cnt_out: Int32) -> None:
        _producer_wait_tile_empty(
            self.tile_producer_mbar,
            accum_cnt_out,
            self.cluster_cta_rank,
            self.NUM_TILE_BUFFERS,
        )

    @cute.jit
    def producer_publish_termination(self, accum_cnt_out: Int32) -> None:
        tile_buf = accum_cnt_out % self.NUM_TILE_BUFFERS
        self.producer_wait_tile_released(accum_cnt_out)
        _producer_signal_tile_ready(
            tile_id_smem_ptr=self.tile_id_smem_ptr,
            tile_id_consumer_bars_ptr=self.tile_consumer_mbar,
            tile_buf=tile_buf,
            tile_idx=Int32(_TILE_SENTINEL),
            WRITE_SMEM=True,
        )

    @cute.jit
    def consumer_release_tile(
        self,
        tile_producer_mbar: cute.Pointer,
        tile_buf: Int32,
        cluster_cta_rank: Int32,
        is_leader_warp: cutlass.Boolean,
    ) -> None:
        if is_leader_warp:
            _epi_signal_tile_done(
                tile_producer_mbar,
                tile_buf,
                cluster_cta_rank,
                self.NUM_CTAS,
            )

    @cute.jit
    def advance_producer(self, accum_cnt_out: Int32) -> Int32:
        self.tile_idx = _producer_fetch_tile_idx(
            self.counter_ptr,
            self.cluster_cta_rank,
            self.tile_cta_bar_mbar,
            self.tile_id_smem_ptr,
            accum_cnt_out % self.NUM_TILE_BUFFERS,
            self.cta_sync_cnt,
            self.NUM_CTAS,
            NUM_CTA_SYNC_BARS=self.NUM_TILE_CTA_BARS,
        )
        self.cta_sync_cnt += Int32(1)
        return self.tile_idx

    @cute.jit
    def advance_consumer(self, accum_cnt_tile: Int32) -> Int32:
        self.tile_idx = _preread_tile_idx(
            accum_cnt_tile,
            self.NUM_TILE_BUFFERS,
            self.tile_consumer_mbar,
            self.tile_id_smem_ptr,
        )
        return self.tile_idx


# =============================================================================
# Static persistent tile scheduler
# =============================================================================


class StaticTileScheduler:
    """Static persistent tile scheduler.

    Each persistent cluster derives its starting tile_idx from
    ``blockIdx.x // NUM_CTAS`` and advances by ``grid_dim.x // NUM_CTAS``.
    The atomic-counter ring is unused (mbarriers stay quiescent), so this
    path avoids the leader-rank serialization that dominates the dynamic
    scheduler on memory-bound shapes.

    The class is intentionally stateless beyond ``tile_idx`` /
    ``stride``; producer and consumer instances differ only in their
    factory entry point so callers stay symmetric with the dynamic path.
    """

    def __init__(
        self,
        tile_idx: Int32,
        stride: Int32,
        NUM_CTAS: cutlass.Constexpr[int],
    ):
        self.tile_idx = tile_idx
        self.stride = stride
        self.NUM_CTAS = NUM_CTAS

    # --------------------------------------------------------------------- #
    # MLIR value serialization.
    # --------------------------------------------------------------------- #

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = list(extract_mlir_values(self.tile_idx))
        values.extend(extract_mlir_values(self.stride))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "StaticTileScheduler":
        assert len(values) == 2
        return StaticTileScheduler(
            tile_idx=new_from_mlir_values(self.tile_idx, [values[0]]),
            stride=new_from_mlir_values(self.stride, [values[1]]),
            NUM_CTAS=self.NUM_CTAS,
        )

    # --------------------------------------------------------------------- #
    # Factories. Producer and consumer share the same seed; the split is
    # kept for API parity with the dynamic scheduler.
    # --------------------------------------------------------------------- #

    @staticmethod
    @cute.jit
    def _initial(NUM_CTAS: cutlass.Constexpr[int]):
        bid = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()
        return bid[0] // NUM_CTAS, grid_dim[0] // NUM_CTAS

    @staticmethod
    @cute.jit
    def create_producer(
        NUM_CTAS: cutlass.Constexpr[int],
    ) -> "StaticTileScheduler":
        tile_idx, stride = StaticTileScheduler._initial(NUM_CTAS)
        return StaticTileScheduler(tile_idx=tile_idx, stride=stride, NUM_CTAS=NUM_CTAS)

    @staticmethod
    @cute.jit
    def create_consumer(
        NUM_CTAS: cutlass.Constexpr[int],
    ) -> "StaticTileScheduler":
        tile_idx, stride = StaticTileScheduler._initial(NUM_CTAS)
        return StaticTileScheduler(tile_idx=tile_idx, stride=stride, NUM_CTAS=NUM_CTAS)

    # --------------------------------------------------------------------- #
    # Tile iteration. ``accum_cnt_*`` args are unused in the static path
    # but kept in the signature so the kernel body can dispatch on a
    # ``cutlass.const_expr(STATIC_SCHEDULER)`` switch without rewriting
    # the call site.
    # --------------------------------------------------------------------- #

    @cute.jit
    def initial_work_tile_info(self) -> Int32:
        return self.tile_idx

    @cute.jit
    def producer_publish_tile(self, accum_cnt_out: Int32) -> None:
        del accum_cnt_out

    @cute.jit
    def producer_wait_tile_released(self, accum_cnt_out: Int32) -> None:
        del accum_cnt_out

    @cute.jit
    def producer_publish_termination(self, accum_cnt_out: Int32) -> None:
        del accum_cnt_out

    @cute.jit
    def consumer_release_tile(
        self,
        tile_producer_mbar: cute.Pointer,
        tile_buf: Int32,
        cluster_cta_rank: Int32,
        is_leader_warp: cutlass.Boolean,
    ) -> None:
        del tile_producer_mbar, tile_buf, cluster_cta_rank, is_leader_warp

    @cute.jit
    def advance_producer(self, accum_cnt_out: Int32) -> Int32:
        del accum_cnt_out  # static schedule ignores the dynamic-ring counter
        self.tile_idx += self.stride
        return self.tile_idx

    @cute.jit
    def advance_consumer(self, accum_cnt_tile: Int32) -> Int32:
        del accum_cnt_tile  # static schedule ignores the dynamic-ring counter
        self.tile_idx += self.stride
        return self.tile_idx


# =============================================================================
# Mega persistent tile schedulers
# =============================================================================


class MegaDynamicScheduler(DynamicTileScheduler):
    """Dynamic scheduler for a flattened multi-problem GEMM stream."""

    def __init__(
        self,
        tile_idx: Int32,
        cta_sync_cnt: Int32,
        counter_ptr: cute.Pointer,
        tile_cta_bar_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        tile_consumer_mbar: cute.Pointer,
        tile_producer_mbar: cute.Pointer,
        cluster_cta_rank: Int32,
        NUM_CTAS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        NUM_TILE_CTA_BARS: cutlass.Constexpr[int],
    ):
        super().__init__(
            tile_idx=tile_idx,
            cta_sync_cnt=cta_sync_cnt,
            counter_ptr=counter_ptr,
            tile_cta_bar_mbar=tile_cta_bar_mbar,
            tile_id_smem_ptr=tile_id_smem_ptr,
            tile_consumer_mbar=tile_consumer_mbar,
            tile_producer_mbar=tile_producer_mbar,
            cluster_cta_rank=cluster_cta_rank,
            NUM_CTAS=NUM_CTAS,
            NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
            NUM_TILE_CTA_BARS=NUM_TILE_CTA_BARS,
        )

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "MegaDynamicScheduler":
        assert len(values) == 2
        return MegaDynamicScheduler(
            tile_idx=new_from_mlir_values(self.tile_idx, [values[0]]),
            cta_sync_cnt=new_from_mlir_values(self.cta_sync_cnt, [values[1]]),
            counter_ptr=self.counter_ptr,
            tile_cta_bar_mbar=self.tile_cta_bar_mbar,
            tile_id_smem_ptr=self.tile_id_smem_ptr,
            tile_consumer_mbar=self.tile_consumer_mbar,
            tile_producer_mbar=self.tile_producer_mbar,
            cluster_cta_rank=self.cluster_cta_rank,
            NUM_CTAS=self.NUM_CTAS,
            NUM_TILE_BUFFERS=self.NUM_TILE_BUFFERS,
            NUM_TILE_CTA_BARS=self.NUM_TILE_CTA_BARS,
        )

    @staticmethod
    @cute.jit
    def create_producer(
        counter_ptr: cute.Pointer,
        tile_cta_bar_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        tile_consumer_mbar: cute.Pointer,
        tile_producer_mbar: cute.Pointer,
        cluster_cta_rank: Int32,
        NUM_CTAS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
        NUM_TILE_CTA_BARS: cutlass.Constexpr[int],
    ) -> "MegaDynamicScheduler":
        cta_sync_cnt = Int32(0)
        tile_idx = _producer_fetch_tile_idx(
            counter_ptr,
            cluster_cta_rank,
            tile_cta_bar_mbar,
            tile_id_smem_ptr,
            Int32(0),
            cta_sync_cnt,
            NUM_CTAS,
            NUM_CTA_SYNC_BARS=NUM_TILE_CTA_BARS,
        )
        cta_sync_cnt += Int32(1)
        return MegaDynamicScheduler(
            tile_idx=tile_idx,
            cta_sync_cnt=cta_sync_cnt,
            counter_ptr=counter_ptr,
            tile_cta_bar_mbar=tile_cta_bar_mbar,
            tile_id_smem_ptr=tile_id_smem_ptr,
            tile_consumer_mbar=tile_consumer_mbar,
            tile_producer_mbar=tile_producer_mbar,
            cluster_cta_rank=cluster_cta_rank,
            NUM_CTAS=NUM_CTAS,
            NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
            NUM_TILE_CTA_BARS=NUM_TILE_CTA_BARS,
        )

    @staticmethod
    @cute.jit
    def create_consumer(
        tile_consumer_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        NUM_CTAS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
    ) -> "MegaDynamicScheduler":
        tile_idx = _preread_tile_idx(
            Int32(0), NUM_TILE_BUFFERS, tile_consumer_mbar, tile_id_smem_ptr
        )
        return MegaDynamicScheduler(
            tile_idx=tile_idx,
            cta_sync_cnt=Int32(0),
            counter_ptr=tile_consumer_mbar,
            tile_cta_bar_mbar=tile_consumer_mbar,
            tile_id_smem_ptr=tile_id_smem_ptr,
            tile_consumer_mbar=tile_consumer_mbar,
            tile_producer_mbar=tile_consumer_mbar,
            cluster_cta_rank=Int32(0),
            NUM_CTAS=NUM_CTAS,
            NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
            NUM_TILE_CTA_BARS=0,
        )

    @staticmethod
    @cute.jit
    def create_record_consumer(
        tile_consumer_mbar: cute.Pointer,
        tile_id_smem_ptr: cute.Pointer,
        NUM_CTAS: cutlass.Constexpr[int],
        NUM_TILE_BUFFERS: cutlass.Constexpr[int],
    ) -> "MegaDynamicScheduler":
        _wait_tile_ready(Int32(0), NUM_TILE_BUFFERS, tile_consumer_mbar)
        return MegaDynamicScheduler(
            tile_idx=Int32(0),
            cta_sync_cnt=Int32(0),
            counter_ptr=tile_consumer_mbar,
            tile_cta_bar_mbar=tile_consumer_mbar,
            tile_id_smem_ptr=tile_id_smem_ptr,
            tile_consumer_mbar=tile_consumer_mbar,
            tile_producer_mbar=tile_consumer_mbar,
            cluster_cta_rank=Int32(0),
            NUM_CTAS=NUM_CTAS,
            NUM_TILE_BUFFERS=NUM_TILE_BUFFERS,
            NUM_TILE_CTA_BARS=0,
        )

    @cute.jit
    def advance_record_consumer(self, accum_cnt_tile: Int32) -> Int32:
        _wait_tile_ready(
            accum_cnt_tile,
            self.NUM_TILE_BUFFERS,
            self.tile_consumer_mbar,
        )
        return Int32(0)


class MegaStaticScheduler(StaticTileScheduler):
    """Static scheduler for a flattened multi-problem GEMM stream."""

    def __init__(
        self,
        tile_idx: Int32,
        stride: Int32,
        NUM_CTAS: cutlass.Constexpr[int],
    ):
        super().__init__(tile_idx=tile_idx, stride=stride, NUM_CTAS=NUM_CTAS)

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MegaStaticScheduler":
        assert len(values) == 2
        return MegaStaticScheduler(
            tile_idx=new_from_mlir_values(self.tile_idx, [values[0]]),
            stride=new_from_mlir_values(self.stride, [values[1]]),
            NUM_CTAS=self.NUM_CTAS,
        )

    @staticmethod
    @cute.jit
    def create_producer(
        NUM_CTAS: cutlass.Constexpr[int],
    ) -> "MegaStaticScheduler":
        tile_idx, stride = StaticTileScheduler._initial(NUM_CTAS)
        return MegaStaticScheduler(
            tile_idx=tile_idx,
            stride=stride,
            NUM_CTAS=NUM_CTAS,
        )

    @staticmethod
    @cute.jit
    def create_consumer(
        NUM_CTAS: cutlass.Constexpr[int],
    ) -> "MegaStaticScheduler":
        tile_idx, stride = StaticTileScheduler._initial(NUM_CTAS)
        return MegaStaticScheduler(
            tile_idx=tile_idx,
            stride=stride,
            NUM_CTAS=NUM_CTAS,
        )


if TYPE_CHECKING:

    def _check_dynamic_scheduler_interface(
        scheduler: DynamicTileScheduler,
    ) -> TileScheduler:
        return scheduler

    def _check_static_scheduler_interface(
        scheduler: StaticTileScheduler,
    ) -> TileScheduler:
        return scheduler

    def _check_mega_dynamic_scheduler_interface(
        scheduler: MegaDynamicScheduler,
    ) -> TileScheduler:
        return scheduler

    def _check_mega_static_scheduler_interface(
        scheduler: MegaStaticScheduler,
    ) -> TileScheduler:
        return scheduler


@cute.jit
def _chunk_offset(
    num_chunks,
    local_rank,
    world_size: cutlass.Constexpr[int],
):
    chunk_offset = cutlass.Int32(0)
    if num_chunks > cutlass.Int32(0):
        chunk_offset = (
            local_rank * ceil_div(num_chunks, cutlass.Int32(world_size))
        ) % num_chunks
    return chunk_offset


@cute.jit
def _ring_schedule_slot(
    phys_row,
    m_size,
    ring_chunk_prefix,
    local_rank,
    PIPELINE_CHUNK_ROWS: cutlass.Constexpr[int],
    ACTIVATION_RING_CHUNKS: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
):
    """Map a group-local physical row to its activation-ring placement.

    The rank rotation (`_chunk_offset`) permutes whole chunks within a group,
    so the schedule chunk derives from the row's physical (post-remap) chunk
    by inverting the rotation and adding the group's chunk prefix. Returns
    ``(ring_sched, ring_row_base, ring_scale_row_base)``: the global schedule
    chunk (ring waits and credits key on it) and the ring slot's row and
    scale-row bases (callers add their within-chunk offsets).
    """
    ring_num_chunks = ceil_div(m_size, cutlass.Int32(PIPELINE_CHUNK_ROWS))
    ring_chunk_off = _chunk_offset(ring_num_chunks, local_rank, world_size=world_size)
    ring_phys_chunk = phys_row // cutlass.Int32(PIPELINE_CHUNK_ROWS)
    ring_sched = (
        ring_chunk_prefix
        + (ring_phys_chunk - ring_chunk_off + ring_num_chunks) % ring_num_chunks
    )
    ring_slot = ring_sched % cutlass.Int32(ACTIVATION_RING_CHUNKS)
    ring_scale_page_rows: cutlass.Constexpr[int] = ((BLOCK_SIZE_N + 127) // 128) * 128
    ring_row_base = ring_slot * cutlass.Int32(PIPELINE_CHUNK_ROWS)
    ring_scale_row_base = ring_slot * cutlass.Int32(
        (PIPELINE_CHUNK_ROWS // BLOCK_SIZE_N) * ring_scale_page_rows
    )
    return ring_sched, ring_row_base, ring_scale_row_base


@cute.jit
def _ring_cached_slot(
    phys_row,
    m_size,
    ring_chunk_prefix,
    local_rank,
    ring_cached_chunk,
    ring_cached_sched,
    ring_cached_row_base,
    ring_cached_scale_base,
    PIPELINE_CHUNK_ROWS: cutlass.Constexpr[int],
    ACTIVATION_RING_CHUNKS: cutlass.Constexpr[int],
    BLOCK_SIZE_N: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
    SWAP_AB: cutlass.Constexpr[bool],
):
    """Chunk-cached `_ring_schedule_slot` plus the within-chunk placement.

    The slot bases only change at chunk granularity, so the slot is
    re-derived only when ``phys_row`` crosses into a new physical chunk;
    steady rows pay just the cached-base arithmetic. Returns ``(ring_row,
    ring_scale_row, refreshed, cached_chunk, cached_sched, cached_row_base,
    cached_scale_base)``: the store positions for ``phys_row``, whether the
    slot was re-derived this call, and the updated cache fields to carry. A
    caller that must wait for the slot's previous occupant to be consumed
    performs that wait when ``refreshed`` is set, before its first store to
    the returned positions.
    """
    ring_phys_chunk = phys_row // cutlass.Int32(PIPELINE_CHUNK_ROWS)
    refreshed = ring_phys_chunk != ring_cached_chunk
    if refreshed:
        ring_cached_sched, ring_cached_row_base, ring_cached_scale_base = (
            _ring_schedule_slot(
                phys_row,
                m_size,
                ring_chunk_prefix,
                local_rank,
                PIPELINE_CHUNK_ROWS=PIPELINE_CHUNK_ROWS,
                ACTIVATION_RING_CHUNKS=ACTIVATION_RING_CHUNKS,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                world_size=world_size,
            )
        )
        ring_cached_chunk = ring_phys_chunk
    ring_within = phys_row - ring_phys_chunk * cutlass.Int32(PIPELINE_CHUNK_ROWS)
    ring_row = ring_cached_row_base + ring_within
    ring_scale_row = blockscaled_scale_row_start(
        ring_cached_scale_base,
        ring_within,
        SWAP_AB,
        BLOCK_SIZE_N,
    )
    return (
        ring_row,
        ring_scale_row,
        refreshed,
        ring_cached_chunk,
        ring_cached_sched,
        ring_cached_row_base,
        ring_cached_scale_base,
    )


@cute.jit
def _subtile_coords(
    tile_m_idx,
    tile_n_idx,
    subtile_idx: cutlass.Constexpr[int],
    SWAP_AB: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(SWAP_AB):
        tile_m_idx += cutlass.Int32(subtile_idx)
    else:
        tile_n_idx += cutlass.Int32(subtile_idx)
    return tile_m_idx, tile_n_idx


@cute.jit
def _subtile_is_valid(
    tile_m_idx,
    tile_n_idx,
    num_output_tiles,
    SWAP_AB: cutlass.Constexpr[bool],
):
    if cutlass.const_expr(SWAP_AB):
        return tile_m_idx < num_output_tiles
    return tile_n_idx < num_output_tiles
