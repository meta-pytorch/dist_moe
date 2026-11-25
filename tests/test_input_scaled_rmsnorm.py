# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import weakref

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import dist_moe
from dist_moe._input_scaled_rmsnorm import _resolve_input_scaled_rmsnorm
from dist_moe._postprocess import _resolve_experts_output_postprocess_fn

_EPS = 1e-6
_GAIN_CENTER = 1.0


def _policy(gain: torch.Tensor) -> dist_moe.InputScaledRMSNormPostprocess:
    return dist_moe.InputScaledRMSNormPostprocess(
        weight_getter=lambda: gain, eps=_EPS, gain_center=_GAIN_CENTER
    )


def _resolve(policy, num_tokens, hidden_dim, topk, score_dtype=torch.float32):
    x_TD = torch.empty((num_tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)
    return _resolve_input_scaled_rmsnorm(
        policy,
        x_TD=x_TD,
        topk_expert_ids_TK=torch.empty(
            (num_tokens, topk), device="cuda", dtype=torch.int64
        ),
        topk_scores_TK=torch.empty(
            (num_tokens, topk), device="cuda", dtype=score_dtype
        ),
    )


def test_input_scaled_rmsnorm_policy_is_an_eager_callback_with_validated_fields():
    """Outside the fused kernels the policy is its own eager callback.

    Its output must match an FP32 ``rms_norm`` of the input-scaled rows, and
    gradients must reach the gain through the getter so the callback path
    trains it too. Invalid epsilon is rejected at construction.
    """
    gain = (0.1 * torch.randn(16)).requires_grad_()
    rows = torch.randn((4, 16))
    policy = _policy(gain)
    assert dist_moe.ExecutionOptions(experts_output_postprocess=policy).requires_eager
    output = policy(rows)
    expected = torch.nn.functional.rms_norm(
        rows * (gain + _GAIN_CENTER), (16,), eps=_EPS
    )
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    output.sum().backward()
    assert gain.grad is not None
    with pytest.raises(ValueError, match="eps"):
        dist_moe.InputScaledRMSNormPostprocess(
            weight_getter=policy.weight_getter, eps=0.0, gain_center=1.0
        )


def test_input_scaled_rmsnorm_resolves_supported_shapes_only():
    """Metadata selects the fused kernels only for shapes they fit.

    Unsupported shapes resolve to ``None``, which the public resolver turns
    into the policy's eager callback. A wrong selection would launch a kernel
    whose ``tl.arange`` cannot represent the shape or whose tiles exceed the
    register bound; a missed one silently costs speed. A gain that breaks its
    storage contract must fail at resolution rather than in a kernel.
    """
    with FakeTensorMode():
        for num_tokens, hidden_dim, topk, score_dtype, fused in (
            (64, 4096, 8, torch.float32, True),
            (64, 512, 2, torch.float32, True),
            (64, 1536, 8, torch.float32, False),
            (64, 8192, 4, torch.float32, False),
            (64, 4096, 16, torch.float32, False),
            (64, 4096, 6, torch.float32, False),
            (64, 4096, 8, torch.bfloat16, False),
        ):
            gain = torch.empty(hidden_dim, device="cuda", dtype=torch.bfloat16)
            resolved = _resolve(
                _policy(gain), num_tokens, hidden_dim, topk, score_dtype
            )
            assert (resolved is not None) == fused, (hidden_dim, topk)
        with pytest.raises(ValueError, match="gain"):
            _resolve(_policy(gain.float()), 64, 4096, 8)


def test_input_scaled_rmsnorm_hooks_do_not_retain_the_input():
    """The resolved hooks must not keep the Dist-MoE input alive.

    Autograd keeps the hooks from forward until backward, so a reference to
    ``x_TD`` would hold every MoE layer's input until its backward runs.
    """
    with FakeTensorMode():
        x_TD = torch.empty((64, 4096), device="cuda", dtype=torch.bfloat16)
        ids_TK = torch.empty((64, 8), device="cuda", dtype=torch.int64)
        scores_TK = torch.empty((64, 8), device="cuda", dtype=torch.float32)
        policy = _policy(torch.empty(4096, device="cuda", dtype=torch.bfloat16))
        resolved = _resolve_experts_output_postprocess_fn(
            policy, x_TD=x_TD, topk_expert_ids_TK=ids_TK, topk_scores_TK=scores_TK
        )
    assert resolved.includes_scale_and_sum
    x_ref = weakref.ref(x_TD)
    del x_TD
    gc.collect()
    assert x_ref() is None


def _reference(h3_MD, gain_D, scores_TK, expert_ids_TK):
    num_tokens, topk = expert_ids_TK.shape
    x_TKD = h3_MD.view(num_tokens, topk, -1)
    scaled_TKD = x_TKD * (gain_D + _GAIN_CENTER)
    rstd_TK1 = torch.rsqrt(scaled_TKD.square().mean(-1, keepdim=True) + _EPS)
    unit_TKD = scaled_TKD * rstd_TK1
    # The kernels round each normalized route to BF16 before score weighting
    # and pass gradients straight through the rounding.
    unit_TKD = unit_TKD + (unit_TKD.to(torch.bfloat16).float() - unit_TKD).detach()
    weights_TK = scores_TK * (expert_ids_TK >= 0)
    return (unit_TKD * weights_TK[..., None]).sum(1)


@pytest.mark.gpus_needed_1
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("gain", ["parameter", "cast", "frozen"])
@pytest.mark.parametrize("hidden_dim,topk", [(4096, 8), (1024, 4), (16, 2)])
@pytest.mark.parametrize("h3_saved", [False, True])
def test_input_scaled_rmsnorm_kernels_match_reference(hidden_dim, topk, h3_saved, gain):
    """Fused forward and backward match an FP32 autograd reference.

    The forward saves H3 into the activation buffer when the planner keeps it,
    and backward reads either that copy or the replayed H3 in place. NaN fills
    every byte the kernels must not read: padded routes, the unselected H3
    source, and the buffer when nothing is saved. The gain is a BF16
    parameter, a BF16 cast of an FP32 master (a non-leaf), or frozen, in
    which case backward computes route and score gradients only. Tolerances
    are BF16-level because the kernels round normalized routes and their
    gradients to BF16. A second backward must be bitwise identical, which
    guards the fixed gain-reduction order.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens = 2048
    expert_ids_TK = torch.randint(0, 64, (num_tokens, topk), device=device)
    expert_ids_TK[::7, -1] = -1
    valid_M1 = (expert_ids_TK >= 0).view(-1, 1)
    valid_M = valid_M1[:, 0]
    h3_MD = torch.randn(
        (num_tokens * topk, hidden_dim), device=device, dtype=torch.bfloat16
    ).masked_fill(~valid_M1, float("nan"))
    scores_TK = torch.rand((num_tokens, topk), device=device)
    master_D = 0.1 * torch.randn(hidden_dim, device=device)
    gain_D = master_D.to(torch.bfloat16)
    grad_owner = {"parameter": gain_D, "cast": master_D, "frozen": None}[gain]
    if grad_owner is not None:
        grad_owner.requires_grad_(True)
    policy = dist_moe.InputScaledRMSNormPostprocess(
        weight_getter=(
            (lambda: master_D.to(torch.bfloat16)) if gain == "cast" else lambda: gain_D
        ),
        eps=_EPS,
        gain_center=_GAIN_CENTER,
    )
    grad_output_TD = torch.randn(
        (num_tokens, hidden_dim), device=device, dtype=torch.bfloat16
    )
    x_TD = torch.empty((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16)
    resolved = _resolve_experts_output_postprocess_fn(
        policy,
        x_TD=x_TD,
        topk_expert_ids_TK=expert_ids_TK,
        topk_scores_TK=scores_TK,
    )
    assert resolved.includes_scale_and_sum

    offset = 256
    # 0xFFFF is a BF16 NaN.
    buffer = torch.full(
        (offset + h3_MD.numel() * 2,), 0xFF, device=device, dtype=torch.uint8
    )
    saved_MD = buffer[offset:].view(torch.bfloat16).view_as(h3_MD)
    offset_1 = torch.tensor([offset], device=device)
    need_replay_1 = torch.tensor([not h3_saved], device=device)
    output_TD, _, _, _ = resolved.forward(
        h3_MD,
        scores_TK,
        output_dtype=torch.bfloat16,
        save=True,
        save_buffer=buffer,
        save_offset=offset_1,
        save_condition=need_replay_1,
    )
    if h3_saved:
        assert torch.equal(saved_MD[valid_M], h3_MD[valid_M])
        assert saved_MD[~valid_M].isnan().all()
    else:
        assert saved_MD.isnan().all()

    h3_ref_MD = h3_MD.float().masked_fill(~valid_M1, 0).requires_grad_(True)
    gain_ref_D = gain_D.detach().float().requires_grad_(True)
    scores_ref_TK = scores_TK.clone().requires_grad_(True)
    expected_TD = _reference(h3_ref_MD, gain_ref_D, scores_ref_TK, expert_ids_TK)
    torch.testing.assert_close(
        output_TD, expected_TD.to(torch.bfloat16), rtol=2e-2, atol=2e-2
    )
    expected_TD.backward(grad_output_TD.float())

    def backward():
        combine_MD = torch.full_like(h3_MD, float("nan")) if h3_saved else h3_MD.clone()
        if grad_owner is not None:
            grad_owner.grad = None
        grad_h3_MD, grad_scores_TK, published = resolved.backward(
            grad_output_TD,
            combine_MD,
            scores_TK,
            postprocess_output_dtype=torch.bfloat16,
            publish_view=lambda shape, dtype: combine_MD,
            x_buffer=buffer,
            x_buffer_offset=offset_1,
            x_buffer_condition=need_replay_1,
        )
        assert published and grad_h3_MD.data_ptr() == combine_MD.data_ptr()
        return (
            grad_h3_MD,
            grad_scores_TK,
            None if grad_owner is None else grad_owner.grad,
        )

    grad_h3_MD, grad_scores_TK, grad_gain_D = backward()
    torch.testing.assert_close(grad_h3_MD.float(), h3_ref_MD.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        grad_scores_TK, scores_ref_TK.grad * (expert_ids_TK >= 0), rtol=2e-2, atol=2e-2
    )
    if grad_owner is None:
        assert grad_gain_D is None and gain_D.grad is None
    else:
        expected_gain_D = gain_ref_D.grad
        torch.testing.assert_close(
            grad_gain_D.float(),
            expected_gain_D,
            rtol=2e-2,
            atol=2e-2 * expected_gain_D.abs().max().item(),
        )
    for actual, repeated in zip(
        (grad_h3_MD, grad_scores_TK, grad_gain_D), backward(), strict=True
    ):
        assert (actual is None and repeated is None) or torch.equal(actual, repeated)
