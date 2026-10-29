# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure matched SwiGLU-family expert forward/backward calls.

See docs/swiglu_benchmark.md for the measurement and ownership contracts.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import dist_moe
import torch
import torch.distributed as dist

ACTIVATIONS = ("swiglu", "swiglu_clamped", "situ_glu")
GRADIENT_INPUTS = ((0, "x_TD"), (2, "scores_TK"), (3, "w13_EFD"), (4, "w2_EDF"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a fixed-shape benchmark, rejecting empty measurement windows.

    Args:
        argv: Optional command-line arguments, otherwise read from the process.

    Returns:
        Validated benchmark options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--activation", choices=(*ACTIVATIONS, "all"), default="all")
    parser.add_argument(
        "--pipeline", choices=("bf16", "staged", "mega"), default="bf16"
    )
    parser.add_argument("--fast-math", action="store_true")
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--fake-ep-size", type=int, choices=(1, 2, 4))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--source-label", default="current")
    parser.add_argument("--source-revision", default="unspecified")
    args = parser.parse_args(argv)
    for field in (
        "tokens",
        "hidden",
        "intermediate",
        "experts",
        "top_k",
        "warmup",
        "windows",
        "steps_per_window",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.profile_steps < 0:
        parser.error("--profile-steps cannot be negative")
    if args.top_k > args.experts:
        parser.error("--top-k cannot exceed --experts")
    if args.fast_math and args.pipeline == "bf16":
        parser.error("--fast-math is an MXFP8-only option")
    return args


def make_config(args: argparse.Namespace, activation: str) -> dist_moe.Config:
    """Construct the same saved-state and scratch policy for every activation.

    Args:
        args: Validated shape and execution options.
        activation: Selected SwiGLU-family activation.

    Returns:
        One-layer, one-slot training configuration without VMM.
    """
    return dist_moe.Config(
        max_num_local_input_tokens=args.tokens,
        hidden_dim=args.hidden,
        intermediate_dim=args.intermediate,
        top_k=args.top_k,
        num_experts=args.experts,
        max_moe_layers_per_activation_slot=1,
        num_activation_slots=1,
        activation_slot_capacity_factor=1.0,
        device_scratch_capacity_factor=1.0,
        block_scaled=(
            None
            if args.pipeline == "bf16"
            else dist_moe.BlockScaledConfig(
                format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
                pipeline=args.pipeline,
                fast_math=args.fast_math,
            )
        ),
        activation=activation,
        wgrad_dtype=torch.bfloat16,
    )


def make_inputs(
    args: argparse.Namespace, rank: int, world_size: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Create deterministic BF16 operands and globally balanced unique routes.

    Args:
        args: Validated tensor dimensions.
        rank: Logical expert-parallel rank.
        world_size: Real or fake expert-parallel group size.
        device: Local device; fake ranks may all use CUDA device zero metadata.

    Returns:
        BF16 ``x_TD``, integer IDs and FP32 scores ``[T,K]``, BF16
        ``w13_EFD`` and ``w2_EDF``, and BF16 upstream ``grad_TD``. Input,
        score, and weight tensors require gradients.
    """
    if args.experts % world_size:
        raise ValueError("expert count must be divisible by EP size")
    if args.tokens * args.top_k % args.experts:
        raise ValueError("tokens * top_k must be divisible by expert count")
    torch.manual_seed(42 + rank)
    x_TD = torch.randn(args.tokens, args.hidden, dtype=torch.bfloat16, device=device)
    ids_TK = (
        (torch.arange(args.tokens, device=device)[:, None] + rank * args.tokens)
        * args.top_k
        + torch.arange(args.top_k, device=device)[None, :]
    ) % args.experts
    scores_TK = torch.full((args.tokens, args.top_k), 1.0 / args.top_k, device=device)
    local_experts = args.experts // world_size
    w13_EFD = torch.randn(
        local_experts,
        2 * args.intermediate,
        args.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    w2_EDF = torch.randn(
        local_experts,
        args.hidden,
        args.intermediate,
        dtype=torch.bfloat16,
        device=device,
    )
    w13_EFD.mul_(args.hidden**-0.5)
    w2_EDF.mul_(args.intermediate**-0.5)
    for tensor in (x_TD, scores_TK, w13_EFD, w2_EDF):
        tensor.requires_grad_()
    return x_TD, ids_TK, scores_TK, w13_EFD, w2_EDF, torch.randn_like(x_TD)


def make_step(
    context: dist_moe.Context, inputs: tuple[torch.Tensor, ...]
) -> Callable[[], torch.Tensor]:
    """Bind one complete forward/backward call without an optimizer.

    Args:
        context: Caller-owned context, retained throughout measurement.
        inputs: Fixed-shape operands from ``make_inputs``.

    Returns:
        Callable that replaces functional gradients and rewinds the planner
        through backward. No synchronization or cache clearing occurs inside.
    """
    x_TD, ids_TK, scores_TK, w13_EFD, w2_EDF, grad_TD = inputs

    def step() -> torch.Tensor:
        """Execute one training call, including dynamic weight preparation."""
        for tensor in (x_TD, scores_TK, w13_EFD, w2_EDF):
            tensor.grad = None
        output_TD = dist_moe.routed_experts(
            x_TD, ids_TK, scores_TK, w13_EFD, w2_EDF, context
        )
        output_TD.backward(grad_TD)
        return output_TD

    return step


def snapshot_gradients(inputs: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    """Clone the four gradients required for eager/graph parity.

    Args:
        inputs: Fixed-shape operands from ``make_inputs`` after backward.

    Returns:
        Independent gradient snapshots in input, score, and weight order.

    Raises:
        RuntimeError: If backward did not populate a required gradient.
    """
    snapshots = []
    for index, name in GRADIENT_INPUTS:
        gradient = inputs[index].grad
        if gradient is None:
            raise RuntimeError(f"backward did not populate {name}.grad (input {index})")
        snapshots.append(gradient.clone())
    return snapshots


def measure(step: Callable[[], Any], args: argparse.Namespace) -> dict[str, Any]:
    """Measure unprofiled windows and reduce each latency by maximum rank.

    Args:
        step: Eager training call or graph replay.
        args: Warmup and measurement window sizes.

    Returns:
        Per-step CUDA-event and wall-clock latency distributions in milliseconds.
    """
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    windows = []
    for _ in range(args.windows):
        dist.barrier()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        wall_start = time.perf_counter()
        for _ in range(args.steps_per_window):
            step()
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000 / args.steps_per_window
        cuda_ms = start.elapsed_time(end) / args.steps_per_window
        timing_2 = torch.tensor([cuda_ms, wall_ms], device="cuda", dtype=torch.float64)
        dist.all_reduce(timing_2, op=dist.ReduceOp.MAX)
        windows.append(timing_2.tolist())
    return {
        "max_rank_cuda_ms_per_step": [window[0] for window in windows],
        "max_rank_wall_ms_per_step": [window[1] for window in windows],
        "median_cuda_ms_per_step": statistics.median(window[0] for window in windows),
        "median_wall_ms_per_step": statistics.median(window[1] for window in windows),
    }


def profile_kernels(
    step: Callable[[], Any], args: argparse.Namespace, name: str
) -> list[dict[str, Any]]:
    """Profile rank zero outside timed windows, optionally exporting a trace.

    Args:
        step: Eager training call or graph replay.
        args: Profile length and optional output directory.
        name: Activation, pipeline, and execution-mode tag.

    Returns:
        Rank-zero CUDA kernel counts and total durations in microseconds.
        Fused kernel durations include their other work, not activation alone.
    """
    if args.profile_steps == 0:
        return []
    dist.barrier()
    kernels = {}
    if dist.get_rank() == 0:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(args.profile_steps):
                step()
            torch.cuda.synchronize()
        for event in prof.events():
            if event.device_type != torch.autograd.DeviceType.CUDA:
                continue
            record = kernels.setdefault(
                event.name, {"name": event.name, "count": 0, "total_us": 0.0}
            )
            record["count"] += 1
            record["total_us"] += event.time_range.elapsed_us()
        if args.trace_dir is not None:
            args.trace_dir.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(args.trace_dir / f"{name}-rank0.json"))
    else:
        for _ in range(args.profile_steps):
            step()
        torch.cuda.synchronize()
    dist.barrier()
    return sorted(kernels.values(), key=lambda item: item["total_us"], reverse=True)


def fake_preflight(
    args: argparse.Namespace, activations: Sequence[str]
) -> dict[str, Any]:
    """Validate metadata and planner sizes for every logical fake EP rank.

    Args:
        args: Validated options with a requested fake EP size.
        activations: Activations to validate.

    Returns:
        Shape and memory plans only; fake execution supplies no timing estimate.
    """
    import torch.testing._internal.distributed.fake_pg  # noqa: F401
    from torch._subclasses.fake_tensor import FakeTensorMode

    world_size = args.fake_ep_size
    torch.cuda.set_device(0)
    records = []
    for rank in range(world_size):
        dist.init_process_group(
            "fake", store=dist.HashStore(), rank=rank, world_size=world_size
        )
        try:
            for activation in activations:
                config = make_config(args, activation)
                # Context construction owns real local buffers. FakeTensor
                # execution models the call without replacing their allocator.
                context = dist_moe.create_context(
                    group=dist.group.WORLD, config=config, device="cuda:0"
                )
                try:
                    with FakeTensorMode(allow_non_fake_inputs=True):
                        output_TD = make_step(
                            context,
                            make_inputs(args, rank, world_size, torch.device("cuda:0")),
                        )()
                        assert output_TD.shape == (args.tokens, args.hidden)
                        records.append(
                            {
                                "rank": rank,
                                "activation": activation,
                                "plan": dataclasses.asdict(
                                    dist_moe.plan_memory(config, world_size)
                                ),
                            }
                        )
                finally:
                    context.close()
        finally:
            dist.destroy_process_group()
    return {"fake_ep_size": world_size, "records": records, "timing": None}


def main(argv: Sequence[str] | None = None) -> None:
    """Run real CUDA measurements or metadata-only preflight and write JSON.

    Args:
        argv: Optional command-line arguments.
    """
    args = parse_args(argv)
    activations = ACTIVATIONS if args.activation == "all" else (args.activation,)
    if args.fake_ep_size is not None:
        result = fake_preflight(args, activations)
    else:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
        try:
            rank, world_size = dist.get_rank(), dist.get_world_size()
            props = torch.cuda.get_device_properties(device)
            result = {
                "source_label": args.source_label,
                "source_revision": args.source_revision,
                "source_file": str(Path(dist_moe.__file__).resolve()),
                "driver_file": str(Path(__file__).resolve()),
                "pipeline": args.pipeline,
                "fast_math": args.fast_math,
                "mode": args.mode,
                "world_size": world_size,
                "local_tokens": args.tokens,
                "hidden_dim": args.hidden,
                "intermediate_dim": args.intermediate,
                "global_experts": args.experts,
                "top_k": args.top_k,
                "warmup": args.warmup,
                "windows": args.windows,
                "steps_per_window": args.steps_per_window,
                "profile_steps": args.profile_steps,
                "weight_preparation": "dynamic, included in training call",
                "wgrad_dtype": "bfloat16",
                "routing": "balanced cyclic unique expert IDs, uniform scores",
                "gpu": props.name,
                "sm_count": props.multi_processor_count,
                "gpu_total_bytes": props.total_memory,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "cutlass": importlib.metadata.version("nvidia-cutlass-dsl"),
                "triton": importlib.metadata.version("triton"),
                "measurements": [],
            }
            for activation in activations:
                config = make_config(args, activation)
                context = dist_moe.create_context(
                    group=dist.group.WORLD, config=config, device=device
                )
                try:
                    inputs = make_inputs(args, rank, world_size, device)
                    step = make_step(context, inputs)
                    for _ in range(args.warmup):
                        step()
                    torch.cuda.synchronize()
                    graph = None
                    if args.mode == "graph":
                        reference_TD = step().detach().clone()
                        reference_gradients = snapshot_gradients(inputs)
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(args.warmup):
                                step()
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured_TD = step()
                        graph.replay()
                        torch.testing.assert_close(
                            captured_TD, reference_TD, rtol=0, atol=0
                        )
                        for (index, name), reference in zip(
                            GRADIENT_INPUTS, reference_gradients, strict=True
                        ):
                            gradient = inputs[index].grad
                            if gradient is None:
                                raise RuntimeError(
                                    f"graph replay did not populate {name}.grad (input {index})"
                                )
                            torch.testing.assert_close(
                                gradient, reference, rtol=0, atol=0
                            )
                        del captured_TD, reference_TD, reference_gradients, reference
                        timed_step = graph.replay
                    else:
                        timed_step = step
                    torch.cuda.reset_peak_memory_stats()
                    timing = measure(timed_step, args)
                    allocated = torch.cuda.max_memory_allocated()
                    reserved = torch.cuda.max_memory_reserved()
                    kernel_times = profile_kernels(
                        timed_step, args, f"{args.pipeline}-{activation}-{args.mode}"
                    )
                    result["measurements"].append(
                        {
                            "activation": activation,
                            **timing,
                            "tokens_per_second_per_gpu": args.tokens
                            * 1000
                            / timing["median_wall_ms_per_step"],
                            "peak_pytorch_allocated_bytes_local": allocated,
                            "peak_pytorch_reserved_bytes_local": reserved,
                            "memory_plan": dataclasses.asdict(
                                dist_moe.plan_memory(config, world_size)
                            ),
                            "rank0_kernels": kernel_times,
                        }
                    )
                    del timed_step, graph, step, inputs
                finally:
                    context.close()
        finally:
            dist.destroy_process_group()
        if rank != 0:
            return
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
