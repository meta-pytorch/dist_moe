# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Low-level SM103 ultra FP4 MMA layout and descriptor helpers."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import tcgen05

SM103_MMA_K = 96
SM103_MMA_COUNT = 8
SM103_TILE_K = SM103_MMA_K * SM103_MMA_COUNT
SM103_AB_SEGMENTS = 3
SM103_AB_PIPELINE_STAGES = 5


def sf_segments(sf_vec_size: int) -> int:
    """SF segments per K=768 tile: one per adjacent MMA pair at vec 16
    (four K=192 segments), one per four MMAs at vec 32 (two K=384
    segments) -- an SF atom spans K = 4 * sf_vec_size and a segment is
    three atoms either way."""
    return SM103_TILE_K // (3 * 4 * sf_vec_size)


def mmas_per_sf_segment(sf_vec_size: int) -> int:
    """K=96 MMA instructions sharing one SF segment."""
    return SM103_MMA_COUNT // sf_segments(sf_vec_size)


def sf_kblock_cols(sf_vec_size: int) -> int:
    """TMEM SF k-block columns consumed per K=96 MMA instruction."""
    return SM103_MMA_K // sf_vec_size


def sf_load_steps(sf_vec_size: int) -> int:
    """Loader steps per K=768 tile: the A/B and SF segment loads are
    interleaved step-for-step, so the loop spans the longer of the two."""
    return max(sf_segments(sf_vec_size), SM103_AB_SEGMENTS)


# One K=768 tile is three K=256 A/B segments. Each segment is 128 bytes per
# FP4 row and matches the K_SW128 SMEM layout. Eight K=96 MMAs start at offsets
# 0, 96, ..., 672; the MMAs at 192 and 480 cross a K=256 segment boundary, so
# their descriptors name both the starting segment and the segment holding the
# tail. Scale-factor segmentation is per ``sf_segments(sf_vec_size)``.
SM103_AB_STAGES = (0, 0, 0, 1, 1, 1, 2, 2)
SM103_AB_K_BLOCKS = (0, 3, 6, 1, 4, 7, 2, 5)
SM103_AB_NEXT_STAGES = (0, 0, 1, 1, 1, 2, 2, 2)
# Sibling schedules for the same interleaved MMA order: at ``mma_idx`` the
# consumer waits for A/B segment ``SM103_AB_WAIT_OFFSETS[mma_idx]`` before
# issuing, and releases segment ``SM103_AB_RELEASE_OFFSETS[mma_idx]`` after
# its last reader; -1 means no wait/release at that MMA. Segment s is waited
# on right before its first reader (MMAs 0/2/5 read segments 0/1/2 first) and
# released after its last (MMAs 2/5/7).
SM103_AB_WAIT_OFFSETS = (0, -1, 1, -1, -1, 2, -1, -1)
SM103_AB_RELEASE_OFFSETS = (-1, -1, 0, -1, -1, 1, -1, 2)

SM103_COMPUTE_CAPABILITY = (10, 3)


def make_tiled_mma(
    sf_dtype: type[cutlass.Numeric],
    sf_vec_size: int,
    cta_group: tcgen05.CtaGroup,
    mma_tiler_mn: tuple[int, int],
) -> cute.TiledMma:
    if sf_vec_size == 16:
        mma_op = tcgen05.SM103MmaMXF4NVF4Op(
            sf_dtype,
            (*mma_tiler_mn, SM103_MMA_K),
            cta_group,
            tcgen05.OperandSource.SMEM,
        )
    elif sf_vec_size == 32:
        mma_op = tcgen05.SM103MmaMXF4Op(
            (*mma_tiler_mn, SM103_MMA_K),
            cta_group,
            tcgen05.OperandSource.SMEM,
        )
    else:
        raise ValueError(
            f"SM103 ultra MMA requires SF vector size 16 or 32, got {sf_vec_size}"
        )
    return cute.make_tiled_mma(cute.make_mma_atom(mma_op))


def make_smem_layout_ab(
    tiled_mma: cute.TiledMma,
    mma_tiler: cute.Tile,
    num_stages: int,
    *,
    is_a: bool,
) -> cute.ComposedLayout:
    mn_extent = mma_tiler[0] if is_a else mma_tiler[1]
    if is_a:
        mn_extent //= cute.size(tiled_mma.thr_layout_vmnk.shape[0])
    else:
        mn_extent //= cute.size(tiled_mma.thr_id.shape)
    return tcgen05.tile_to_mma_shape(
        tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128,
            cutlass.Uint8,
        ),
        cute.append(((mn_extent, 16), 1, 8), num_stages),
        order=(0, 1, 2),
    )


def _make_sf_atom_layout(sf_vec_size: int) -> cute.Layout:
    return cute.make_layout(
        shape=((8, 4, 4), (sf_vec_size, 4)),
        stride=((16, 128, 4), (0, 1)),
    )


def make_gmem_layout_sf(shape: cute.Shape, sf_vec_size: int) -> cute.Layout:
    return cute.tile_to_shape(
        _make_sf_atom_layout(sf_vec_size),
        shape,
        (2, 1, 3),
    )


def make_smem_layout_sfa(
    tiled_mma: cute.TiledMma,
    mma_tiler: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
) -> cute.Layout:
    mma_shape_mk = tiled_mma.partition_shape_A((mma_tiler[0], mma_tiler[2]))
    sf_atom = _make_sf_atom_layout(sf_vec_size)
    k_divisor = 4 if sf_vec_size == 16 else 2
    mma_sfa_tiler = (
        mma_shape_mk[0][0] * mma_shape_mk[1],
        mma_shape_mk[0][1] * mma_shape_mk[2] // k_divisor,
    )
    per_stage = cute.tiled_product(
        sf_atom,
        cute.make_layout(
            cute.shape_div(mma_sfa_tiler, cute.product_each(sf_atom.shape))
        ),
    )
    return cute.make_layout(
        shape=cute.append(per_stage.shape, num_stages),
        stride=cute.append(
            per_stage.stride,
            cute.size(cute.filter_zeros(per_stage)),
        ),
    )


def make_smem_layout_sfb(
    mma_tiler: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
) -> cute.Layout:
    k_divisor = 4 if sf_vec_size == 16 else 2
    mma_sfb_tiler = (mma_tiler[1], mma_tiler[2] // k_divisor)
    sf_k_major_atom256 = cute.make_layout(
        shape=((32, 4, 2), (sf_vec_size, 4)),
        stride=(
            (16, 4, mma_sfb_tiler[1] // sf_vec_size // 4 * 512),
            (0, 1),
        ),
    )
    per_stage = cute.tiled_product(
        sf_k_major_atom256,
        cute.make_layout(
            cute.shape_div(
                mma_sfb_tiler,
                cute.product_each(sf_k_major_atom256.shape),
            )
        ),
    )
    return cute.make_layout(
        shape=cute.append(per_stage.shape, num_stages),
        stride=cute.append(
            per_stage.stride,
            cute.size(cute.filter_zeros(per_stage)),
        ),
    )


def append_coalesce_layout(layout: cute.Layout) -> cute.Layout:
    result = cute.append(
        cute.coalesce(cute.append(layout[0][0], layout[1])),
        cute.coalesce(cute.append(layout[0][1], layout[2])),
    )
    result = cute.append(result, layout[3])
    result = cute.append(result, layout[4])
    return cute.append(result, layout[5])


def adapt_layout_for_tma_ab(
    composed_layout: cute.ComposedLayout,
) -> cute.ComposedLayout:
    layout = composed_layout.outer
    part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
    part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
    result = cute.append(part1, cute.append(part2, layout[3]))
    return cute.make_composed_layout(
        composed_layout.inner,
        composed_layout.offset,
        result,
    )


def mma_sf_tiler(
    cta_tile_shape_mnk: tuple[int, int, int], sf_vec_size: int
) -> tuple[int, int, int]:
    """SF tile shape for one CTA tile: N padded to the 128-row SF atom and
    one SF segment along K (``K // sf_segments(sf_vec_size)``)."""
    return (
        cta_tile_shape_mnk[0],
        cute.round_up(cta_tile_shape_mnk[1], 128),
        cta_tile_shape_mnk[2] // sf_segments(sf_vec_size),
    )


def adapt_layout_for_tma_sf(layout: cute.Layout) -> cute.Layout:
    part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
    part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
    return cute.append(cute.group_modes(part1, 0, cute.rank(part1)), part2)


def make_sf_tmem_field_layouts(
    tiled_mma: cute.TiledMma,
    mma_tiler: tuple[int, int, int],
    cta_tile_shape_mnk: tuple[int, int, int],
    sf_vec_size: int,
) -> tuple[cute.Layout, cute.Layout]:
    mma_m = cta_tile_shape_mnk[0]
    mma_n = mma_sf_tiler(cta_tile_shape_mnk, sf_vec_size)[1]
    mma_k_sf = cta_tile_shape_mnk[2] // 2
    mn_basic_block = (32, 4)
    k_basic_block = (sf_vec_size, 1)
    sfa_iter_layout = cute.make_layout(
        (((mn_basic_block, mma_m // 128), k_basic_block), 1, mma_k_sf // sf_vec_size)
    )
    sfb_iter_layout = cute.make_layout(
        (((mn_basic_block, mma_n // 128), k_basic_block), 1, mma_k_sf // sf_vec_size)
    )
    return (
        blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, mma_tiler, sf_vec_size, sfa_iter_layout
        ),
        blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, mma_tiler, sf_vec_size, sfb_iter_layout
        ),
    )


def make_desc_and_call_mma(
    tiled_mma: cute.TiledMma,
    d: cute.Tensor,
    sA_cur: cute.Tensor,
    sA_next: cute.Tensor,
    sB_cur: cute.Tensor,
    sB_next: cute.Tensor,
    c: cute.Tensor,
) -> None:
    a_desc = tcgen05.make_umma_smem_desc(
        sA_cur.iterator,
        sA_cur.layout,
        "k",
        next_src=sA_next.iterator,
    )
    b_desc = tcgen05.make_umma_smem_desc(
        sB_cur.iterator,
        sB_cur.layout,
        "k",
        next_src=sB_next.iterator,
    )
    view_layout = cute.make_layout(1, stride=0)
    cute.mma_atom_call(
        tiled_mma,
        d,
        cute.make_tensor(a_desc, view_layout),
        cute.make_tensor(b_desc, view_layout),
        c,
    )
