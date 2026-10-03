"""Shared distributed setup for the public examples."""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

import dist_moe
import torch
import torch.distributed as dist


@dataclasses.dataclass(frozen=True)
class ExampleInputs:
    """Small routed-expert tensors and their static dimensions.

    Args:
        device: CUDA device owned by this local rank.
        group: Expert-parallel process group.
        x_TD: Local BF16 token activations.
        topk_expert_ids_TK: Global expert IDs selected for each token.
        topk_scores_TK: Router weights for the selected experts.
        w13_EFD: Local gate/up expert weights.
        w2_EDF: Local down-projection expert weights.
        num_experts: Global expert count.
        num_local_experts: Experts owned by this rank.
        hidden_dim: Model hidden width.
        intermediate_dim: Per-expert intermediate width.
        top_k: Experts selected per token.
    """

    device: torch.device
    group: dist.ProcessGroup
    x_TD: torch.Tensor
    topk_expert_ids_TK: torch.Tensor
    topk_scores_TK: torch.Tensor
    w13_EFD: torch.Tensor
    w2_EDF: torch.Tensor
    num_experts: int
    num_local_experts: int
    hidden_dim: int
    intermediate_dim: int
    top_k: int


def initialize_inputs(*, requires_grad: bool) -> ExampleInputs:
    """Initialize NCCL and create one deterministic two-rank example case.

    Args:
        requires_grad: Whether floating-point inputs participate in autograd.

    Returns:
        Routed tokens, scores, and local expert weights for this rank.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group = dist.group.WORLD

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    torch.manual_seed(17)

    num_tokens = 16
    hidden_dim = 256
    intermediate_dim = 256
    top_k = 2
    num_local_experts = 4
    num_experts = num_local_experts * world_size

    x_TD = torch.randn(
        num_tokens,
        hidden_dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=requires_grad,
    )
    global_token_ids_T = rank * num_tokens + torch.arange(num_tokens, device=device)
    topk_expert_ids_TK = (
        global_token_ids_T[:, None] * top_k
        + torch.arange(top_k, device=device)[None, :]
    ) % num_experts
    topk_scores_TK = torch.full(
        (num_tokens, top_k),
        1.0 / top_k,
        dtype=torch.float32,
        device=device,
        requires_grad=requires_grad,
    )
    w13_EFD = torch.randn(
        num_local_experts,
        2 * intermediate_dim,
        hidden_dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=requires_grad,
    )
    w2_EDF = torch.randn(
        num_local_experts,
        hidden_dim,
        intermediate_dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=requires_grad,
    )
    return ExampleInputs(
        device=device,
        group=group,
        x_TD=x_TD,
        topk_expert_ids_TK=topk_expert_ids_TK,
        topk_scores_TK=topk_scores_TK,
        w13_EFD=w13_EFD,
        w2_EDF=w2_EDF,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        top_k=top_k,
    )


def create_config(
    inputs: ExampleInputs,
    *,
    inference: bool = False,
    block_scaled: dist_moe.BlockScaledConfig | None = None,
    activation: Literal["swiglu", "swiglu_clamped"] = "swiglu",
    activation_slot_capacity_factor: float | None = 1.0,
    max_moe_layers_per_activation_slot: int = 1,
    num_activation_slots: int = 1,
    vmm: dist_moe.VmmConfig | None = None,
) -> dist_moe.Config:
    """Create the common static configuration with feature overrides.

    Args:
        inputs: Static shapes and expert topology.
        inference: Whether to select inference-only execution.
        block_scaled: Optional MXFP8 or NVFP4 policy.
        activation: SwiGLU activation variant.
        activation_slot_capacity_factor: Optional saved-state capacity factor.
        max_moe_layers_per_activation_slot: Maximum layers sharing one slot.
        num_activation_slots: Independent saved-state slots.
        vmm: Optional host-backed scratch policy.

    Returns:
        Validated public configuration for the example context.
    """
    return dist_moe.Config(
        num_local_input_tokens=inputs.x_TD.shape[0],
        hidden_dim=inputs.hidden_dim,
        intermediate_dim=inputs.intermediate_dim,
        top_k=inputs.top_k,
        num_experts=inputs.num_experts,
        max_moe_layers_per_activation_slot=max_moe_layers_per_activation_slot,
        device_scratch_capacity_factor=1.0,
        activation_slot_capacity_factor=(
            None if inference else activation_slot_capacity_factor
        ),
        num_activation_slots=0 if inference else num_activation_slots,
        block_scaled=block_scaled,
        activation=activation,
        inference=inference,
        vmm=vmm,
    )


def close(
    context: dist_moe.Context | None,
    *,
    destroy_group: bool = True,
) -> None:
    """Close the context and destroy the example process group.

    Args:
        context: Context to close, or ``None`` if construction failed.
        destroy_group: Whether to destroy the process group after closing.
    """
    try:
        if context is not None:
            context.close()
    finally:
        if destroy_group and dist.is_initialized():
            dist.destroy_process_group()
