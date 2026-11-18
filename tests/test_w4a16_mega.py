# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused W4A16 MoE forward equals BF16 grouped GEMMs on the dequantized weights.

Single rank: the gather and scatter pointers address a local token tensor
and a local combine buffer, so the kernel runs its full dispatch, FC13 +
SwiGLU, FC2 and combine protocol without peers.
"""

import pytest
import torch

from dist_moe._blockscaled_weight import _interleave_w13_rows
from dist_moe.kernels.config import (
    derive_grouped_gemm_config,
    grouped_gemm_config_name,
    GROUPED_GEMM_CONFIGS,
)
from dist_moe.kernels.grouped_gemm import grouped_gemm_fprop
from dist_moe.kernels.triton.swiglu import swiglu_fwd
from dist_moe.kernels.w4a16_mega import (
    w4a16_mega_fprop,
    w4a16_mega_operand_bytes,
)

pytestmark = [
    pytest.mark.gpus_needed_1,
    pytest.mark.gb10x,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="W4A16 Mega requires an SM100-family GPU",
    ),
]

_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _blocked_scale_offsets(N: int, B: int, device: torch.device) -> torch.Tensor:
    """Byte offset of each natural-order scale in the CUBLAS 128x4 layout.

    Args:
        N: Rows per expert (a multiple of 128).
        B: Scale columns per row (``K / 16``, a multiple of 4).
        device: Device of the returned offsets.

    Returns:
        Offsets ``[N * B]`` in row-major natural order.
    """
    r = torch.arange(N, device=device).view(N, 1)
    c = torch.arange(B, device=device).view(1, B)
    offset = (
        ((r // 128) * (B // 4) + c // 4) * 512
        + (r % 32) * 16
        + (r % 128 // 32) * 4
        + c % 4
    )
    return offset.flatten()


def blocked_scale_bytes(sf_GNB: torch.Tensor) -> torch.Tensor:
    """Lay per-16 scales out as CUBLAS 128x4 blocked bytes.

    Args:
        sf_GNB: E4M3 scale bytes ``[G, N, K / 16]`` in natural order.

    Returns:
        Blocked scale bytes ``[G, N * K / 16]``.
    """
    G, N, B = sf_GNB.shape
    assert N % 128 == 0 and B % 4 == 0
    out = torch.empty(G, N * B, dtype=torch.uint8, device=sf_GNB.device)
    out[:, _blocked_scale_offsets(N, B, sf_GNB.device)] = sf_GNB.reshape(G, -1)
    return out


def unblock_scale_bytes(blocked: torch.Tensor, G: int, N: int, B: int) -> torch.Tensor:
    """Invert ``blocked_scale_bytes``.

    Args:
        blocked: Blocked scale bytes with ``G * N * B`` elements.
        G: Expert count.
        N: Rows per expert.
        B: Scale columns per row (``K / 16``).

    Returns:
        Natural-order scale bytes ``[G, N, B]``.
    """
    offsets = _blocked_scale_offsets(N, B, blocked.device)
    return blocked.reshape(G, N * B)[:, offsets].view(G, N, B)


def dequantize_nvfp4(
    packed_GNQ: torch.Tensor, blocked_scale: torch.Tensor
) -> torch.Tensor:
    """Return ``code * block_scale`` of NVFP4 weights, exact in BF16.

    Args:
        packed_GNQ: Packed E2M1 bytes ``[G, N, K / 2]``, even column in the low
            nibble.
        blocked_scale: CUBLAS-blocked E4M3 scale bytes, ``G * N * K / 16``.

    Returns:
        BF16 weights ``[G, N, K]`` without the per-expert global scale.
    """
    G, N, Q = packed_GNQ.shape
    nibble = torch.stack((packed_GNQ & 15, packed_GNQ >> 4), dim=-1).flatten(-2)
    values = _E2M1_VALUES.to(packed_GNQ.device)[(nibble & 7).long()]
    values = torch.where(nibble >= 8, -values, values)
    scale = unblock_scale_bytes(blocked_scale, G, N, 2 * Q // 16)
    scale = scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return (values * scale).to(torch.bfloat16)


def bf16_swap_ab_fprop(
    x: torch.Tensor,
    w_GNK: torch.Tensor,
    split_sizes: torch.Tensor,
    alpha_G: torch.Tensor,
    *,
    block_n: int,
) -> torch.Tensor:
    """BF16 SWAP_AB grouped FPROP in the W4A16 Mega tile, scaled per expert.

    Both kernels issue the same MMA sequence per output element (128 features
    by ``block_n`` tokens, one MMA per 16 K values, in order), so with weights
    equal to ``code * block_scale`` their FP32 accumulators match. ``alpha_G``
    must hold powers of two: then scaling the BF16 output equals Mega's
    scaling of the FP32 accumulator before rounding.

    Args:
        x: BF16 rows ``[GM, K]`` grouped by expert.
        w_GNK: BF16 weights ``[G, N, K]``.
        split_sizes: Int32 rows per expert ``[G]``.
        alpha_G: FP32 per-expert power-of-two scales ``[G]``.
        block_n: Token tile of the Mega launch under comparison.

    Returns:
        BF16 output ``[GM, N]``.
    """
    assert bool((torch.frexp(alpha_G).mantissa == 0.5).all())
    config = derive_grouped_gemm_config(
        num_ctas=1, num_mmas=1, block_m=128, block_n=block_n, block_k=128
    )
    name = grouped_gemm_config_name(config)
    GROUPED_GEMM_CONFIGS.setdefault(name, config)
    out = grouped_gemm_fprop(x, w_GNK, split_sizes, config=name, swap_ab=True)
    rows = torch.repeat_interleave(
        torch.arange(alpha_G.numel(), device=x.device), split_sizes.long()
    )
    return (out.float() * alpha_G[rows].unsqueeze(1)).to(torch.bfloat16)


def _make_nvfp4(G: int, N: int, K: int, gen: torch.Generator, device: str):
    """Draw random NVFP4 weights and their exact BF16 dequantization.

    Codes cover all sixteen E2M1 values and the E4M3 block scales span nine
    binades, so ``code * scale`` is exact in BF16.

    Args:
        G: Expert count.
        N: Output features per expert.
        K: Contraction length.
        gen: CPU generator.
        device: Device of the returned tensors.

    Returns:
        Packed E2M1 bytes ``[G, N, K / 2]``, blocked E4M3 scale bytes and the
        dequantized BF16 weights ``[G, N, K]``.
    """
    nibble = torch.randint(0, 16, (G, N, K), generator=gen, device="cpu")
    packed = (nibble[..., 0::2] | (nibble[..., 1::2] << 4)).to(torch.uint8)
    mant = 1.0 + torch.randint(0, 8, (G, N, K // 16), generator=gen) / 8.0
    expo = torch.randint(-6, 3, (G, N, K // 16), generator=gen).float()
    scale = (mant * torch.pow(2.0, expo)).to(torch.float8_e4m3fn)
    sf_bytes = blocked_scale_bytes(scale.view(torch.uint8).to(device))
    packed = packed.to(device)
    return packed, sf_bytes, dequantize_nvfp4(packed, sf_bytes)


@pytest.mark.parametrize(
    "G, D, F, T, K, block_n",
    [
        (4, 512, 256, 24, 2, 16),
        (8, 512, 512, 40, 4, 32),
        (32, 2048, 768, 64, 8, 16),
        (8, 512, 512, 160, 4, 64),
        (32, 2048, 768, 512, 8, 128),
    ],
)
def test_w4a16_mega_matches_bf16_on_dequantized_weights(G, D, F, T, K, block_n) -> None:
    """Mega equals BF16 grouped GEMMs on ``code * scale`` bit for bit.

    The reference runs BF16 SWAP_AB FC13 on the canonical W13, scales each
    expert's rows by its power-of-two inverse global scale, applies SwiGLU,
    and does the same for FC2; Mega runs on the inference (gate/up groups of
    eight) W13. Distinct per-expert scales catch a misattributed scale. Each
    shape runs with per-call intermediates and with stale context storage,
    twice each, so the counter reset is covered too. The shapes cover every
    token tile and the 32-expert ``D=2048, F=768`` shape of one rank at
    expert parallelism four.
    """
    device = "cuda"
    gen = torch.Generator().manual_seed(3)
    x = (0.5 * torch.randn(T, D, generator=gen)).bfloat16().to(device)
    ids = torch.stack([torch.randperm(G, generator=gen)[:K] for _ in range(T)]).to(
        device
    )
    w13_q, w13_sf, w13_deq = _make_nvfp4(G, 2 * F, D, gen, device)
    w2_q, w2_sf, w2_deq = _make_nvfp4(G, D, F, gen, device)
    a13 = torch.pow(2.0, torch.randint(-3, 3, (G,), generator=gen).float()).to(device)
    a2 = torch.pow(2.0, torch.randint(-3, 3, (G,), generator=gen).float()).to(device)

    flat = ids.flatten()
    order = torch.argsort(flat, stable=True)  # routed row -> (token, k) pair
    split = torch.bincount(flat, minlength=G).int()
    tokens = order // K
    rows = T * K

    h1 = bf16_swap_ab_fprop(x[tokens], w13_deq, split, a13, block_n=block_n)
    y = bf16_swap_ab_fprop(swiglu_fwd(h1), w2_deq, split, a2, block_n=block_n)
    expected = torch.empty(rows, D, dtype=torch.bfloat16, device=device)
    expected[order] = y

    w13_inference = (
        _interleave_w13_rows(w13_q),
        blocked_scale_bytes(
            _interleave_w13_rows(unblock_scale_bytes(w13_sf, G, 2 * F, D // 16))
        ),
        a13,
    )
    combine = torch.zeros(rows, D, dtype=torch.bfloat16, device=device)
    gather_ptrs = x.data_ptr() + tokens * (D * 2)
    scatter_ptrs = combine.data_ptr() + order * (D * 2)
    num_bytes = w4a16_mega_operand_bytes(
        num_rows=rows, num_experts=G, hidden_dim=D, intermediate_dim=F
    )
    # Context storage holds stale bytes from earlier calls; the launcher must
    # clear the counters it carves from it.
    stale_storage = torch.full((num_bytes,), 0xFF, dtype=torch.uint8, device=device)
    for operand_storage in (None, stale_storage):
        for _ in range(2):
            combine.zero_()
            w4a16_mega_fprop(
                w13=w13_inference,
                w2=(w2_q, w2_sf, a2),
                split_sizes=split,
                gather_ptrs=gather_ptrs,
                scatter_ptrs=scatter_ptrs,
                num_rows=rows,
                hidden_dim=D,
                intermediate_dim=F,
                block_n=block_n,
                operand_storage=operand_storage,
            )
        torch.testing.assert_close(
            combine.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )
