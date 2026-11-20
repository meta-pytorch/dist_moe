# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Four Over Six NVFP4 quantization of weight-only (W4A16) experts.

Both grouped launches are held bitwise equal to a PyTorch transcription of
the quantizer's FP32 arithmetic, and weight-only preparation is held to the
quantize launch. The arithmetic is described in docs/nvfp4_kernel_design.md.
"""

import dist_moe
import pytest
import torch
from test_w4a16_mega import unblock_scale_bytes

from dist_moe._blockscaled_weight import _interleave_w13_rows
from dist_moe.kernels.nvfp4_four_over_six import (
    four_over_six_global_scales,
    four_over_six_qdq,
    four_over_six_quantize,
)

pytestmark = [
    pytest.mark.gpus_needed_1,
    pytest.mark.gb10x,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="Four Over Six quantization requires an SM100-family GPU",
    ),
]

_BLOCK = 16
_E4M3_MAX = 448.0
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_POLICY = dist_moe.BlockScaledConfig(
    format=dist_moe.BlockScaledFormat.NVFP4, pipeline="mega", weight_only=True
)


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 values to E2M1, nearest-even with saturation (``cvt.rn``)."""
    magnitude = x.abs()
    rounded = torch.where(magnitude > 5.0, 6.0, 4.0)
    rounded = torch.where(magnitude < 3.5, 3.0, rounded)
    rounded = torch.where(magnitude <= 2.5, 2.0, rounded)
    rounded = torch.where(magnitude < 1.75, 1.5, rounded)
    rounded = torch.where(magnitude <= 1.25, 1.0, rounded)
    rounded = torch.where(magnitude < 0.75, 0.5, rounded)
    rounded = torch.where(magnitude <= 0.25, 0.0, rounded)
    return torch.copysign(rounded, x)


def _reference_quantize(
    weight_GNK: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Four Over Six in PyTorch ops, operation for operation as the kernel.

    Args:
        weight_GNK: BF16 expert weights ``[G, N, K]``.

    Returns:
        E2M1 values ``[G, N, K / 16, 16]``, FP32-valued E4M3 block scales
        ``[G, N, K / 16]`` and each expert's FP32 amax ``[G]``.
    """
    amax_G, encode_G, decode_G = four_over_six_global_scales(weight_GNK)
    encode_G11, decode_G11 = encode_G.view(-1, 1, 1), decode_G.view(-1, 1, 1)
    x_GNBV = weight_GNK.float().unflatten(-1, (-1, _BLOCK))
    x_scaled_GNBV = x_GNBV * encode_G11.unsqueeze(-1)
    block_amax_GNB = x_GNBV.abs().amax(dim=-1)
    # Divide by a tensor: division by a Python scalar becomes a multiplication
    # by its reciprocal, which rounds differently.
    scale6_GNB = (block_amax_GNB / torch.full_like(block_amax_GNB, 6.0)) * encode_G11
    candidates = []
    for scale_hp_GNB in (scale6_GNB * 1.5, scale6_GNB):  # map-to-4, map-to-6
        scale_GNB = scale_hp_GNB.clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn).float()
        inverse_GNB = (1.0 / (scale_GNB * decode_G11)).clamp(
            max=torch.finfo(torch.float32).max
        )
        codes_GNBV = _round_to_e2m1(x_GNBV * inverse_GNB.unsqueeze(-1))
        # code * scale in FP16 is exact up to its single FP16 rounding.
        diff_GNBV = (
            codes_GNBV.half() * scale_GNB.half().unsqueeze(-1)
        ).float() - x_scaled_GNBV
        squared_GNBV = diff_GNBV * diff_GNBV
        error_GNB = squared_GNBV[..., 0]
        for value in range(1, _BLOCK):  # sequential FP32 sum, in order
            error_GNB = error_GNB + squared_GNBV[..., value]
        candidates.append((codes_GNBV, scale_GNB, error_GNB))
    (codes4, scale4, error4), (codes6, scale6, error6) = candidates
    map_to_4_GNB = error4 < error6
    return (
        torch.where(map_to_4_GNB.unsqueeze(-1), codes4, codes6),
        torch.where(map_to_4_GNB, scale4, scale6),
        amax_G,
    )


def _pack_e2m1(values_GNK: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 values two per byte, the even column in the low nibble."""
    index = values_GNK.abs().unsqueeze(-1) == _E2M1_VALUES.to(values_GNK.device)
    nibble = index.int().argmax(dim=-1) | torch.where(torch.signbit(values_GNK), 8, 0)
    return (nibble[..., 0::2] | (nibble[..., 1::2] << 4)).to(torch.uint8)


def _weights() -> torch.Tensor:
    """BF16 experts spanning magnitudes, blocks with outliers, zero and ties."""
    gen = torch.Generator().manual_seed(11)
    G, N, K = 6, 256, 512
    w = torch.randn(G, N, K, generator=gen)
    w *= torch.pow(10.0, torch.arange(G).float() - 3.0).view(G, 1, 1)
    w[1, :, ::37] *= 40.0  # one outlier every few blocks
    w[2].zero_()  # all-zero expert: global scale falls back to one
    # Exact E2M1 grid points times a power of two exercise the rounding ties.
    w[4] = _E2M1_VALUES[torch.randint(0, 8, (N, K), generator=gen)] * torch.where(
        torch.rand(N, K, generator=gen) < 0.5, -0.25, 0.25
    )
    return w.bfloat16().cuda()


def test_qdq_matches_reference() -> None:
    """The QDQ launch equals ``code * (scale * amax / 2688)`` bit for bit.

    Experts span six decades, outlier blocks, an all-zero expert and values on
    the E2M1 grid, so both candidates, saturation and ties are exercised. The
    ``out`` form writes the same bits into caller storage.
    """
    w = _weights()
    codes, scale, amax = _reference_quantize(w)
    final_scale = (scale * amax.view(-1, 1, 1)) * (1.0 / (6 * _E4M3_MAX))
    expected = (codes * final_scale.unsqueeze(-1)).flatten(-2).to(w.dtype)
    out = torch.empty_like(w)
    for got in (four_over_six_qdq(w), four_over_six_qdq(w, out=out)):
        torch.testing.assert_close(
            got.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
        )
    assert four_over_six_qdq(w, out=out) is out


def test_quantize_matches_reference() -> None:
    """The quantize launch stores the reference codes and scales bit for bit.

    Codes pack two per byte with the even column in the low nibble; scales
    are E4M3 bytes in natural order. The returned global scales are
    ``2688 / amax`` and its reciprocal, one for the all-zero expert.
    """
    w = _weights()
    codes, scale, _ = _reference_quantize(w)
    got_codes, got_scale, encode, decode = four_over_six_quantize(w)
    assert torch.equal(got_codes, _pack_e2m1(codes.flatten(-2)))
    assert torch.equal(got_scale, scale.to(torch.float8_e4m3fn).view(torch.uint8))
    _, ref_encode, ref_decode = four_over_six_global_scales(w)
    assert torch.equal(encode, ref_encode) and torch.equal(decode, ref_decode)
    assert encode[2].item() == 1.0


@pytest.mark.parametrize("fuse_swiglu", [False, True])
def test_weight_only_preparation_holds_four_over_six_operands(fuse_swiglu) -> None:
    """Weight-only preparation stores the quantize launch's operands.

    The prepared codes and unblocked scales equal the quantize launch (gate/up
    interleaved for fused SwiGLU), and the global scales are its encode and
    decode scales, so ``quantize_dequantize_weight`` dequantizes exactly the
    inference operands. A non-weight-only policy is rejected.
    """
    w = _weights()
    G, N, K = w.shape
    prepared = dist_moe.prepare_block_scaled_weight(
        w, _POLICY, inference=True, fuse_swiglu=fuse_swiglu
    )
    codes, scale, encode, decode = four_over_six_quantize(w)
    if fuse_swiglu:
        codes, scale = _interleave_w13_rows(codes), _interleave_w13_rows(scale)
    assert torch.equal(prepared.fprop_data.view(torch.uint8), codes)
    assert torch.equal(
        unblock_scale_bytes(prepared.fprop_scale.view(torch.uint8), G, N, K // 16),
        scale,
    )
    assert torch.equal(prepared.global_scale, encode)
    assert torch.equal(prepared.global_scale_inv, decode)
    assert torch.equal(
        dist_moe.quantize_dequantize_weight(w, _POLICY), four_over_six_qdq(w)
    )
    with pytest.raises(ValueError, match="weight_only NVFP4"):
        dist_moe.quantize_dequantize_weight(
            w,
            dist_moe.BlockScaledConfig(
                format=dist_moe.BlockScaledFormat.NVFP4, pipeline="mega"
            ),
        )
