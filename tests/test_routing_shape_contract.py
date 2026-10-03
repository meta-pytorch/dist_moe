# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Isolated fatal routing-contract tests."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import dist_moe
import pytest
import torch
import torch.distributed as dist
from dist_moe._buffers import _routing_token_count_view

_EXPECTED_TRAP_ERRORS = (
    "device-side assert",
    "illegal instruction",
    "unspecified launch failure",
)


def _make_inputs(
    *,
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    requires_grad: bool,
) -> tuple[torch.Tensor, ...]:
    """Create one local-expert input tuple for a routing-safety worker."""
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    return (
        torch.randn(
            num_tokens,
            hidden_dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=requires_grad,
        ),
        torch.zeros(num_tokens, 1, dtype=torch.int64, device=device),
        torch.ones(
            num_tokens,
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        ),
        torch.randn(
            1,
            2 * intermediate_dim,
            hidden_dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=requires_grad,
        ),
        torch.randn(
            1,
            hidden_dim,
            intermediate_dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=requires_grad,
        ),
    )


def _run_mismatched_worker(*, inference: bool) -> None:
    """Corrupt one peer header and require the routing kernel to trap."""
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    num_tokens, hidden_dim, intermediate_dim = 4, 128, 128
    config = dist_moe.Config(
        num_local_input_tokens=num_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        top_k=1,
        num_experts=dist.get_world_size(),
        max_moe_layers_per_activation_slot=1,
        num_activation_slots=0 if inference else 1,
        block_scaled=dist_moe.BlockScaledConfig() if inference else None,
        inference=inference,
    )
    context = dist_moe.create_context(
        group=dist.group.WORLD,
        config=config,
        device=device,
    )
    if local_rank == 1:
        _routing_token_count_view(context.buffers.routing, local_rank).fill_(
            num_tokens + 1
        )
    dist.barrier()

    x_TD, topk_expert_ids_TK, topk_scores_TK, w13_EFD, w2_EDF = _make_inputs(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        requires_grad=not inference,
    )
    if inference:
        assert config.block_scaled is not None
        w13_weight = dist_moe.prepare_block_scaled_weight(
            w13_EFD,
            config.block_scaled,
            inference=True,
        )
        w2_weight = dist_moe.prepare_block_scaled_weight(
            w2_EDF,
            config.block_scaled,
            inference=True,
        )
    else:
        w13_weight, w2_weight = w13_EFD, w2_EDF
    try:
        with torch.no_grad() if inference else torch.enable_grad():
            dist_moe.routed_experts(
                x_TD,
                topk_expert_ids_TK,
                topk_scores_TK,
                w13_weight,
                w2_weight,
                context,
            )
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        if not any(
            fragment in str(error).lower() for fragment in _EXPECTED_TRAP_ERRORS
        ):
            raise
        print("EXPECTED_DIST_MOE_TOKEN_COUNT_TRAP", flush=True)
        return
    raise AssertionError("unequal peer token counts did not trigger a device trap")


def _run_capacity_overflow_worker(*, block_scaled: bool) -> None:
    """Route every token to rank zero and require the earliest capacity trap."""
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    num_tokens = 512 if block_scaled else 8
    hidden_dim = intermediate_dim = 256 if block_scaled else 128
    config = dist_moe.Config(
        num_local_input_tokens=num_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        top_k=1,
        num_experts=dist.get_world_size(),
        max_moe_layers_per_activation_slot=1,
        device_scratch_capacity_factor=0.5,
        num_activation_slots=0 if block_scaled else 1,
        block_scaled=dist_moe.BlockScaledConfig() if block_scaled else None,
        inference=block_scaled,
    )
    context = dist_moe.create_context(
        group=dist.group.WORLD,
        config=config,
        device=device,
    )
    x_TD, topk_expert_ids_TK, topk_scores_TK, w13_EFD, w2_EDF = _make_inputs(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        requires_grad=False,
    )
    if block_scaled:
        assert config.block_scaled is not None
        w13_weight = dist_moe.prepare_block_scaled_weight(
            w13_EFD,
            config.block_scaled,
            inference=True,
        )
        w2_weight = dist_moe.prepare_block_scaled_weight(
            w2_EDF,
            config.block_scaled,
            inference=True,
        )
    else:
        w13_weight, w2_weight = w13_EFD, w2_EDF
    try:
        with torch.no_grad():
            dist_moe.routed_experts(
                x_TD,
                topk_expert_ids_TK,
                topk_scores_TK,
                w13_weight,
                w2_weight,
                context,
            )
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        if not any(
            fragment in str(error).lower() for fragment in _EXPECTED_TRAP_ERRORS
        ):
            raise
        print("EXPECTED_DIST_MOE_SCRATCH_CAPACITY_TRAP", flush=True)
        raise
    raise AssertionError("routing beyond total scratch capacity did not trap")


@pytest.mark.gpus_needed_4
@pytest.mark.gb10x
@pytest.mark.parametrize("inference", (False, True))
def test_unequal_peer_token_counts_trap_before_routing(inference: bool) -> None:
    """Prove both routing pipelines fail before consuming mismatched metadata."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--local-addr=127.0.0.1",
        "--nproc-per-node=4",
        str(Path(__file__).resolve()),
        "--worker",
        "--inference" if inference else "--training",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert output.count("EXPECTED_DIST_MOE_TOKEN_COUNT_TRAP") == 4, output


@pytest.mark.gpus_needed_4
@pytest.mark.gb10x
@pytest.mark.parametrize("block_scaled", (False, True))
def test_scratch_capacity_overflow_traps_before_execution(
    block_scaled: bool,
) -> None:
    """Prove general and direct-decode routing fail with one stable diagnostic."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--local-addr=127.0.0.1",
        "--nproc-per-node=4",
        str(Path(__file__).resolve()),
        "--worker",
        "--capacity-overflow",
        *(["--block-scaled"] if block_scaled else []),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert "EXPECTED_DIST_MOE_SCRATCH_CAPACITY_TRAP" in output, output
    assert "DistMoE scratch capacity exceeded" in output, output
    assert "required_receive_rows:" in output, output
    assert "total_scratch_capacity_rows:" in output, output
    assert "illegal memory access" not in output.lower(), output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--training", action="store_true")
    mode.add_argument("--inference", action="store_true")
    mode.add_argument("--capacity-overflow", action="store_true")
    parser.add_argument("--block-scaled", action="store_true")
    args = parser.parse_args()
    if args.capacity_overflow:
        _run_capacity_overflow_worker(block_scaled=args.block_scaled)
    else:
        _run_mismatched_worker(inference=args.inference)
