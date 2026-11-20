# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Time one inference MoE layer through the public API, per expert format.

Run: ``torchrun --standalone --nproc-per-node=4 benchmarks/w4a16_mega.py``.

Each format gets its own inference context at the same shape: BF16, MXFP8
Mega, NVFP4 Mega and weight-only NVFP4 (W4A16) Mega, with prepared
block-scaled weights. The defaults are a 128-expert top-8 model with
``D=2048`` and ``F=768``. Routing is random top-k, drawn per rank; scratch
covers twice the balanced receive rows by default. For each
token count, ``--chain`` back-to-back ``dist_moe.routed_experts`` calls are
captured in one CUDA graph; each window times ``--iters`` replays, takes the
slowest rank, and the median window is reported per layer. Rank zero prints
one JSON line per format and token count, including the prepared weight bytes
per rank and the output's relative L2 distance to BF16.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence

import dist_moe
import torch
import torch.distributed as dist

_FORMATS = ("bf16", "mxfp8", "nvfp4", "w4a16")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the layer shape, formats and measurement options.

    Args:
        argv: Optional command-line arguments, otherwise read from the process.

    Returns:
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--intermediate-dim", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    parser.add_argument("--formats", nargs="+", choices=_FORMATS, default=_FORMATS)
    parser.add_argument("--scratch-capacity-factor", type=float, default=2.0)
    parser.add_argument("--chain", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args(argv)


def build_layer(
    precision: str,
    args: argparse.Namespace,
    w13_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
) -> tuple[Callable[..., torch.Tensor], dist_moe.Context, int]:
    """Create one format's context and prepared weights.

    Args:
        precision: One of ``bf16``, ``mxfp8``, ``nvfp4`` or ``w4a16``.
        args: Parsed shape options.
        w13_EFD: Local BF16 W13 ``[E_local, 2F, D]``.
        w2_EDF: Local BF16 W2 ``[E_local, D, F]``.

    Returns:
        The layer call, its context, and the bytes of its weight operands.
    """
    policy = {
        "bf16": None,
        "mxfp8": dist_moe.BlockScaledConfig(pipeline="mega"),
        "nvfp4": dist_moe.BlockScaledConfig(
            format=dist_moe.BlockScaledFormat.NVFP4, pipeline="mega"
        ),
        "w4a16": dist_moe.BlockScaledConfig(
            format=dist_moe.BlockScaledFormat.NVFP4,
            pipeline="mega",
            weight_only=True,
        ),
    }[precision]
    context = dist_moe.create_context(
        group=dist.group.WORLD,
        config=dist_moe.Config(
            max_num_local_input_tokens=max(args.tokens),
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            top_k=args.top_k,
            num_experts=args.num_experts,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=args.scratch_capacity_factor,
            num_activation_slots=0,
            block_scaled=policy,
            inference=True,
        ),
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    if policy is None:
        w13, w2 = w13_EFD, w2_EDF
        weight_bytes = w13.nbytes + w2.nbytes
    else:
        w13 = dist_moe.prepare_block_scaled_weight(
            w13_EFD, policy, inference=True, fuse_swiglu=True
        )
        w2 = dist_moe.prepare_block_scaled_weight(w2_EDF, policy, inference=True)
        weight_bytes = sum(
            tensor.nbytes for weight in (w13, w2) for tensor in weight.storage_tensors()
        )

    def run(
        x_TD: torch.Tensor, ids_TK: torch.Tensor, scores_TK: torch.Tensor
    ) -> torch.Tensor:
        """Run one inference layer."""
        return dist_moe.routed_experts(x_TD, ids_TK, scores_TK, w13, w2, context)

    return run, context, weight_bytes


def layer_us(fn: Callable[[], object], chain: int, iters: int, repeats: int) -> float:
    """Return the median per-layer CUDA-graph time, slowest rank per window.

    Args:
        fn: Runs one layer on the current stream.
        chain: Layers captured back to back in the graph.
        iters: Graph replays per timed window.
        repeats: Timed windows.

    Returns:
        Median over windows of the slowest rank's microseconds per layer.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(chain):
            fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    windows = []
    for _ in range(repeats):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        elapsed = torch.tensor(
            start.elapsed_time(end) * 1e3 / (iters * chain), device="cuda"
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        windows.append(elapsed.item())
    return sorted(windows)[len(windows) // 2]


def main(argv: Sequence[str] | None = None) -> None:
    """Benchmark every format and token count and print JSON lines on rank 0.

    Args:
        argv: Optional command-line arguments.
    """
    args = parse_args(argv)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_experts = args.num_experts // world
    D, F = args.hidden_dim, args.intermediate_dim
    generator = torch.Generator(device="cuda").manual_seed(100 + rank)
    w13_EFD = (
        0.02 * torch.randn(local_experts, 2 * F, D, device="cuda", generator=generator)
    ).bfloat16()
    w2_EDF = (
        0.02 * torch.randn(local_experts, D, F, device="cuda", generator=generator)
    ).bfloat16()
    inputs = {}
    for tokens in args.tokens:
        logits = torch.rand(
            tokens, args.num_experts, device="cuda", generator=generator
        )
        values, ids = logits.topk(args.top_k, dim=-1)
        inputs[tokens] = (
            torch.randn(tokens, D, device="cuda", generator=generator).bfloat16(),
            ids.int().contiguous(),
            torch.softmax(values, dim=-1).float().contiguous(),
        )
    layers = {
        precision: build_layer(precision, args, w13_EFD, w2_EDF)
        for precision in args.formats
    }
    if rank == 0:
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "world_size": world,
                    "torch": torch.__version__,
                    "hidden_dim": D,
                    "intermediate_dim": F,
                    "num_experts": args.num_experts,
                    "top_k": args.top_k,
                }
            ),
            flush=True,
        )
    with torch.no_grad():
        for tokens, case in inputs.items():
            reference = None
            if "bf16" in layers:
                reference = layers["bf16"][0](*case).float()
            for precision, (run, context, weight_bytes) in layers.items():
                output = run(*case).float()
                relative_l2 = None
                if reference is not None:
                    error = torch.stack(
                        ((output - reference).square().sum(), reference.square().sum())
                    )
                    dist.all_reduce(error)
                    relative_l2 = round((error[0] / error[1]).sqrt().item(), 5)
                us = layer_us(
                    lambda run=run, case=case: run(*case),
                    args.chain,
                    args.iters,
                    args.repeats,
                )
                if rank == 0:
                    row = {
                        "format": precision,
                        "tokens_per_rank": tokens,
                        "layer_us": round(us, 2),
                        "weight_bytes_per_rank": weight_bytes,
                        "context_device_bytes": (
                            context.memory_plan.total_device_buffer_bytes
                        ),
                        "relative_l2_vs_bf16": relative_l2,
                    }
                    print(json.dumps(row), flush=True)
    for _, context, _ in layers.values():
        context.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
