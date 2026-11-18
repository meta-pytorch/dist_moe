# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host launcher of the fused W4A16 EP MoE forward (``W4A16MegaKernel``).

The caller owns routing: for every routed row ``r`` (grouped by local expert,
``split_sizes`` rows per expert) ``gather_ptrs[r]`` is the address of its
BF16 token row in a peer's dispatch buffer and ``scatter_ptrs[r]`` the
address of its output row in the source rank's combine buffer. Peers must
publish dispatch rows before the launch and read combine rows only after
every rank's launch completes.
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack, make_ptr

from ._environment import num_sms_per_device
from .config import (
    _align_up,
    BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES,
    derive_grouped_gemm_config,
    estimate_grouped_gemm_smem_bytes,
)
from .dist_grouped_gemm import _PADDING_SINK_CACHE, _PADDING_SOURCE_CACHE, _padding_row
from .grouped_gemm import _compile_or_get
from .w4a16_mega_kernel import (
    W4A16_NUM_TRANSFORM_GROUPS,
    W4A16_SF_VEC_SIZE,
    W4A16MegaKernel,
)

_STORAGE_ALIGNMENT_BYTES = 128
# SMEM the W4A16 estimate leaves for alignment padding of its stages.
_W4A16_SMEM_ALIGN_SLACK = 2048
# Deepest SMEM ring the configuration search starts from.
_W4A16_MAX_SMEM_STAGES = 12
# Smallest token tile; it bounds the per-tile counters of every block_n.
_MIN_BLOCK_N = 16


def _token_tile_cap(num_rows: int, num_experts: int, block_n: int) -> int:
    """Bound the token tiles of routed rows split over expert groups.

    Args:
        num_rows: Routed-row capacity.
        num_experts: Local expert groups; each adds at most one partial tile.
        block_n: Token tile width.

    Returns:
        Upper bound of ``sum(ceil(rows_g / block_n))`` over the groups.
    """
    return num_rows // block_n + num_experts


def _operand_offsets(
    num_rows: int, num_experts: int, hidden_dim: int, intermediate_dim: int
) -> tuple[int, int, int]:
    """Lay out the gathered rows, SwiGLU rows and counters in operand storage.

    Args:
        num_rows: Routed-row capacity.
        num_experts: Local expert count.
        hidden_dim: Model hidden dimension ``D``.
        intermediate_dim: Expert intermediate dimension ``F``.

    Returns:
        Byte offsets of the h2 rows and of the counters, and the total bytes.
        The gathered token rows start at offset zero.
    """
    h2_offset = _align_up(num_rows * hidden_dim * 2, _STORAGE_ALIGNMENT_BYTES)
    counters_offset = h2_offset + _align_up(
        num_rows * intermediate_dim * 2, _STORAGE_ALIGNMENT_BYTES
    )
    num_counters = 1 + 2 * _token_tile_cap(num_rows, num_experts, _MIN_BLOCK_N)
    return (
        h2_offset,
        counters_offset,
        counters_offset + _align_up(num_counters * 4, _STORAGE_ALIGNMENT_BYTES),
    )


def w4a16_mega_operand_bytes(
    *, num_rows: int, num_experts: int, hidden_dim: int, intermediate_dim: int
) -> int:
    """Return the bytes ``w4a16_mega_fprop`` carves from ``operand_storage``.

    The storage holds BF16 gathered token rows ``[num_rows, D]``, BF16 SwiGLU
    rows ``[num_rows, F]`` and the gather/FC13 tile counters, for any token
    tile.

    Args:
        num_rows: Routed-row capacity.
        num_experts: Local expert count.
        hidden_dim: Model hidden dimension ``D``.
        intermediate_dim: Expert intermediate dimension ``F``.

    Returns:
        Required storage size in bytes.
    """
    return _operand_offsets(num_rows, num_experts, hidden_dim, intermediate_dim)[2]


@functools.cache
def w4a16_mega_config(block_n: int) -> dict:
    """Return the W4A16 tile geometry: 128 features by ``block_n`` tokens.

    K stages are 256 deep, or 128 from 64-token tiles up, whose BF16 token
    stages would otherwise leave room for only two SMEM stages.

    TMEM holds the accumulators and ``W4A16_A_STAGES`` BF16 A stages (two per
    transform warpgroup at ``block_k=128``, one at 256), so the accumulator
    buffer count is capped to leave room for A. SMEM holds no BF16 A: a TMA
    stage is the packed FP4 tile, its E4M3 scales and the BF16 token tile,
    and the ring takes the deepest count that fits.

    Args:
        block_n: Token tile width (16, 32, 64 or 128).

    Returns:
        Grouped-GEMM configuration shared by both Mega problems.

    Raises:
        ValueError: If no even SMEM ring depth fits the SMEM budget.
    """
    block_m = 128
    block_k = 128 if block_n >= 64 else 256
    config = derive_grouped_gemm_config(
        num_ctas=1, num_mmas=1, block_m=block_m, block_n=block_n, block_k=block_k
    )
    config["STATIC_SCHEDULER"] = True
    # Programmatic dependent launch overlaps each launch's prologue with the
    # previous kernel's tail.
    config["USE_PDL"] = True
    a_stages = W4A16_NUM_TRANSFORM_GROUPS * max(1, 256 // block_k)
    config["W4A16_A_STAGES"] = a_stages
    a_tmem_cols = a_stages * block_k // 2
    acc_buffers = max(1, (512 - a_tmem_cols) // block_n)
    acc_buffers = 1 << (acc_buffers.bit_length() - 1)
    config["NUM_TMEM_BUFFERS"] = min(config["NUM_TMEM_BUFFERS"], acc_buffers)
    config["NUM_TILE_BUFFERS"] = config["NUM_TMEM_BUFFERS"] + 1
    config["NUM_C_STAGES"] = min(config["NUM_C_STAGES"], config["NUM_TMEM_BUFFERS"])
    a_bf16_bytes = block_m * block_k * 2
    fp4_sf_bytes = block_m * block_k // 2 + block_m * block_k // W4A16_SF_VEC_SIZE
    # The transform warpgroups take alternating K stages, so each must own whole
    # TMA buffers: with a ring of N buffers, a group sees buffer b at every
    # N-th stage of its own only when N is a multiple of the group count.
    # Otherwise it skips a phase of ab_full[b], and its parity wait can pass on
    # the phase before (a TMA that lands out of order makes that happen),
    # decoding a stale weight stage.
    groups = W4A16_NUM_TRANSFORM_GROUPS
    max_stages = _W4A16_MAX_SMEM_STAGES
    for stages in range(max_stages - max_stages % groups, groups - 1, -groups):
        bf16_bytes = estimate_grouped_gemm_smem_bytes(
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_ctas=1,
            num_smem_buffers=stages,
            num_tmem_buffers=config["NUM_TMEM_BUFFERS"],
            num_tile_buffers=config["NUM_TILE_BUFFERS"],
            epilogue_subtile=config["EPILOGUE_SUBTILE"],
            num_c_stages=config["NUM_C_STAGES"],
        )
        total = (
            bf16_bytes
            - stages * a_bf16_bytes
            + stages * fp4_sf_bytes
            + 2 * a_stages * 8
            + _W4A16_SMEM_ALIGN_SLACK
        )
        if total <= BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES:
            config["NUM_SMEM_BUFFERS"] = stages
            return config
    raise ValueError(f"W4A16 SMEM does not fit: block_n={block_n}")


@functools.cache
def _mega_kernel(block_n: int, world_size: int) -> W4A16MegaKernel:
    """Return the cached kernel object for one token tile and EP size.

    Args:
        block_n: Token tile width.
        world_size: Expert-parallel world size.

    Returns:
        The ``W4A16MegaKernel`` for this configuration.
    """
    return W4A16MegaKernel(w4a16_mega_config(block_n), world_size=world_size)


# Largest mean routed rows per expert for each token tile, smallest first.
# Each 128-feature weight tile is decoded once per token tile, so wider tiles
# pay off once the rows fill them (single B200, per-rank 16..2048 rows/expert).
_MEGA_BLOCK_N_MAX_ROWS = ((16, 8), (32, 32), (64, 64))


def w4a16_mega_block_n(num_rows: int, num_experts: int) -> int:
    """Select the token tile from the mean routed rows per expert.

    Args:
        num_rows: Expected routed rows (not the capacity).
        num_experts: Local expert count.

    Returns:
        Token tile width: 16, 32, 64 or 128.
    """
    rows_per_expert = num_rows / max(num_experts, 1)
    for block_n, max_rows in _MEGA_BLOCK_N_MAX_ROWS:
        if rows_per_expert <= max_rows:
            return block_n
    return 128


def w4a16_mega_fprop(
    *,
    w13: tuple,  # (FP4 bytes [G, 2F, D/2], E4M3 blocked scales, f32 alpha [G])
    w2: tuple,  # (FP4 bytes [G, D, F/2], E4M3 blocked scales, f32 alpha [G])
    split_sizes: torch.Tensor,  # [G] int32
    gather_ptrs: torch.Tensor,  # int64 [num_rows]
    scatter_ptrs: torch.Tensor,  # int64 [num_rows]
    num_rows: int,
    hidden_dim: int,
    intermediate_dim: int,
    local_rank: int = 0,
    world_size: int = 1,
    block_n: int | None = None,
    expected_rows: int | None = None,
    num_sms: int | None = None,
    operand_storage: torch.Tensor | None = None,
) -> None:
    """Run dispatch, FC13 + SwiGLU, FC2 and combine in one launch.

    The launch is enqueued on the current stream and does not synchronize
    peers: dispatch rows must be published before it, and combine rows read
    only after every rank's launch has completed.

    Args:
        w13: Packed FP4 bytes ``[G, 2F, D / 2]`` in NVFP4 inference
            ``fuse_swiglu`` row order (gate/up groups of eight), their
            CUBLAS-blocked E4M3 scale bytes, and the FP32 per-expert inverse
            global scale ``[G]``.
        w2: Packed FP4 bytes ``[G, D, F / 2]``, blocked scale bytes and the
            FP32 inverse global scale ``[G]``.
        split_sizes: Int32 routed rows per local expert ``[G]``.
        gather_ptrs: Int64 address of each routed row's BF16 token row in a
            local or peer dispatch buffer, grouped by expert.
        scatter_ptrs: Int64 address of each routed row's BF16 output row in
            its source rank's combine buffer.
        num_rows: Routed-row capacity (length of the pointer vectors).
        hidden_dim: Model hidden dimension ``D``, a multiple of 256.
        intermediate_dim: Expert intermediate dimension ``F``, a multiple of
            256.
        local_rank: This rank's index in the expert-parallel group.
        world_size: Expert-parallel world size.
        block_n: Optional token tile; ``None`` selects it from
            ``expected_rows``.
        expected_rows: Typical routed rows used for the tile choice;
            defaults to ``num_rows``.
        num_sms: Persistent grid size; defaults to the device SM count.
        operand_storage: Optional context-owned, 128-byte-aligned uint8 vector
            of at least ``w4a16_mega_operand_bytes`` bytes for the gathered
            rows, SwiGLU rows and counters. Without it they are allocated per
            call. Its counters are cleared on the current stream.

    Raises:
        ValueError: If the weight shapes, dimensions or storage are invalid.
    """
    w13_q, w13_sf, w13_alpha = w13
    w2_q, w2_sf, w2_alpha = w2
    G = split_sizes.numel()
    D, F = hidden_dim, intermediate_dim
    device = split_sizes.device
    if w13_q.shape != (G, 2 * F, D // 2) or w2_q.shape != (G, D, F // 2):
        raise ValueError(
            f"weights must be [G, 2F, D/2] and [G, D, F/2]; got "
            f"{tuple(w13_q.shape)} and {tuple(w2_q.shape)}"
        )
    if D % 256 or F % 256:
        raise ValueError("W4A16 Mega needs D and F multiples of 256")
    if block_n is None:
        block_n = w4a16_mega_block_n(
            num_rows if expected_rows is None else expected_rows, G
        )
    if num_sms is None:
        num_sms = num_sms_per_device(device)
    token_tile_cap = _token_tile_cap(num_rows, G, block_n)
    num_counters = 1 + 2 * token_tile_cap
    if operand_storage is None:
        counters = torch.zeros(num_counters, dtype=torch.int32, device=device)
        x_rows = torch.empty((num_rows, D), dtype=torch.bfloat16, device=device)
        h2_rows = torch.empty((num_rows, F), dtype=torch.bfloat16, device=device)
    else:
        h2_offset, counters_offset, required = _operand_offsets(num_rows, G, D, F)
        if (
            operand_storage.dtype != torch.uint8
            or operand_storage.dim() != 1
            or operand_storage.data_ptr() % _STORAGE_ALIGNMENT_BYTES
            or operand_storage.numel() < required
        ):
            raise ValueError(
                f"operand_storage must be a 128-byte-aligned uint8 vector of at "
                f"least {required} bytes"
            )
        x_rows = (
            operand_storage[: num_rows * D * 2].view(torch.bfloat16).view(num_rows, D)
        )
        h2_rows = (
            operand_storage[h2_offset : h2_offset + num_rows * F * 2]
            .view(torch.bfloat16)
            .view(num_rows, F)
        )
        counters = operand_storage[
            counters_offset : counters_offset + num_counters * 4
        ].view(torch.int32)
        counters.zero_()
    padding_src = _padding_row(
        _PADDING_SOURCE_CACHE, device=device, dtype=torch.bfloat16, numel=D, zero=True
    )
    padding_dst = _padding_row(
        _PADDING_SINK_CACHE, device=device, dtype=torch.bfloat16, numel=D, zero=False
    )

    def rows_cute(t: torch.Tensor):
        """Wrap a row-major ``[rows, cols]`` tensor as a dynamic CuTe tensor."""
        return from_dlpack(t.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )

    def ptr(dtype, t: torch.Tensor, align: int):
        """Return a global-memory CuTe pointer to ``t``'s storage."""
        return make_ptr(
            dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
        )

    args = [
        rows_cute(x_rows),
        rows_cute(h2_rows),
        from_dlpack(split_sizes, assumed_align=4),
        from_dlpack(counters, assumed_align=4),
        ptr(cutlass.Float4E2M1FN, w13_q, 32),
        ptr(cutlass.Float8E4M3FN, w13_sf, 16),
        ptr(cutlass.Float32, w13_alpha, 16),
        ptr(cutlass.Float4E2M1FN, w2_q, 32),
        ptr(cutlass.Float8E4M3FN, w2_sf, 16),
        ptr(cutlass.Float32, w2_alpha, 16),
        ptr(cutlass.Uint32, h2_rows, 16),
        cutlass.Int64(gather_ptrs.data_ptr()),
        cutlass.Int64(scatter_ptrs.data_ptr()),
        cutlass.Int64(padding_src.data_ptr()),
        cutlass.Int64(padding_dst.data_ptr()),
        cutlass.Int64(x_rows.data_ptr()),
        cutlass.Int32(token_tile_cap),
    ]
    tail = [
        cutlass.Int32(num_rows),
        cutlass.Int32(D),
        cutlass.Int32(F),
        cutlass.Int32(local_rank),
        num_sms,
        cutlass_torch.current_stream(),
    ]
    compiled = _compile_or_get(
        ("w4a16_mega", block_n, world_size, G, num_sms),
        _mega_kernel(block_n, world_size),
        *args,
        G,
        *tail,
    )
    compiled(*args, *tail)
