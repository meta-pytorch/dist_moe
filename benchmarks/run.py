# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Render campaigns or execute compatible cases under a distributed launcher."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib.metadata
import os
import sys
import traceback
from pathlib import Path

from .config import campaign_jobs, Case, compilation_key, load_shapes
from .metrics import (
    artifact_digest,
    effective_hbm_bandwidth,
    summarize_windows,
    write_json,
)
from .routing import routing_summary


def _select_cases(
    cases: list[Case],
    backend: str | None,
    schedule: str,
    fast_math: bool,
    execution_mode: str | None = None,
) -> list[Case]:
    """Apply one provider and execution policy to timing or fit cases.

    Args:
        cases: Generated workload cases.
        backend: Provider filter, or all providers.
        schedule: Shared-stream scheduling policy.
        fast_math: Whether selected MXFP8 cases use approximate sigmoid.
        execution_mode: Optional matched eager or graph execution policy.

    Returns:
        Cases with preserved shapes, routing and validation requirements.
    """
    return [
        dataclasses.replace(
            c,
            shared_expert_schedule=schedule,
            fast_math=fast_math and c.precision == "mxfp8" and c.backend == "dist_moe",
            execution_mode=execution_mode,
        )
        for c in cases
        if backend is None or c.backend == backend
    ]


def render_campaign(
    shape_file: Path,
    output: Path,
    *,
    scope: str | None = None,
    backend: str | None = None,
    shared_expert_schedule: str = "concurrent",
    mxfp8_fast_math: bool = False,
    execution_mode: str | None = None,
) -> dict:
    """Write selected cases and fit checks into compatible child batches.

    Args:
        shape_file: Published shapes and provenance.
        output: Destination manifest.
        scope: Optional isolated or shared-comparison phase.
        backend: Optional provider filter applied to both timing and fit cases.
        shared_expert_schedule: Dist-MoE shared-stream scheduling policy.
        mxfp8_fast_math: Enable approximate sigmoid for selected MXFP8 cases.
        execution_mode: Optional shared eager or CUDA-graph timing override.

    Returns:
        Complete campaign manifest for the selected execution scope.
    """
    if backend not in (None, "dist_moe", "mok"):
        raise ValueError("backend must be dist_moe or mok")
    if mxfp8_fast_math and backend not in (None, "dist_moe"):
        raise ValueError("MXFP8 fast-math campaigns require Dist-MoE")
    if shared_expert_schedule not in ("concurrent", "sequential"):
        raise ValueError("shared_expert_schedule must be concurrent or sequential")
    if shared_expert_schedule == "sequential" and (
        backend != "dist_moe" or scope != "comparison"
    ):
        raise ValueError("sequential campaigns require shared Dist-MoE comparisons")
    shapes, provenance = load_shapes(shape_file)
    manifest = {
        "provenance": provenance,
        "shape_file_sha256": artifact_digest(shape_file),
        "jobs": {},
        "scope": scope,
    }
    for smoke in (True, False):
        for name, cases in campaign_jobs(shapes, smoke=smoke, scope=scope).items():
            fit = []
            if smoke and cases[0].mode == "training" and scope != "isolated":
                fit = campaign_jobs(shapes)["packing-training-ep32"]
            cases = _select_cases(
                cases, backend, shared_expert_schedule, mxfp8_fast_math, execution_mode
            )
            fit = _select_cases(
                fit, backend, shared_expert_schedule, mxfp8_fast_math, execution_mode
            )
            if not cases:
                continue
            batches = {}
            for case in [*cases, *fit]:
                key = repr(compilation_key(case))
                batches.setdefault(key, []).append(case.identifier)
            manifest["jobs"][name] = {
                "ep": cases[0].ep,
                "smoke": smoke,
                "cases": {c.identifier: c.to_dict() for c in [*cases, *fit]},
                "validation_only": [c.identifier for c in fit],
                "batches": list(batches.values()),
            }
    write_json(output, manifest)
    return manifest


def _runtime_identity(case: Case | None = None) -> dict:
    """Record the actual imported provider versions and paths before allocation."""
    import dist_moe
    import torch
    import torch.distributed as dist

    prefix = Path(sys.prefix).resolve()
    configured = os.environ.get("CONDA_DIR")
    if configured and prefix != Path(configured).resolve():
        raise RuntimeError(
            f"GPU worker interpreter is outside the configured package: {prefix} != {configured}"
        )
    if configured and any(
        not Path(module.__file__).resolve().is_relative_to(prefix)
        for module in ((torch, dist_moe) if case is None else (torch,))
    ):
        raise RuntimeError(
            "GPU provider import resolved outside the configured package"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    backend = type(dist.group.WORLD._get_backend(device)).__name__
    if backend != "ProcessGroupNCCL":
        raise RuntimeError(
            f"campaign requires stock NCCL, resolved {backend}; set TORCH_DIST_USE_NCCL2=0 before importing Torch"
        )
    result = {
        "torch": torch.__version__,
        "prefix": str(prefix),
        "torch_sha": torch.version.git_version,
        "torch_path": torch.__file__,
        "dist_moe_path": dist_moe.__file__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "arch_list": torch.cuda.get_arch_list(),
        "cuda_module_loading_environment": os.environ.get(
            "CUDA_MODULE_LOADING", "runtime default"
        ),
        "current_cuda_device": torch.cuda.current_device(),
        "nccl_process_group": backend,
        "nccl_backend_selection": os.environ.get(
            "TORCH_DIST_USE_NCCL2", "runtime default"
        ),
        "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "compiler_cache_root": os.environ.get("CUTE_DSL_CACHE_DIR"),
        "nccl_control_transport": {
            name: os.environ.get(name, "runtime default")
            for name in ("NCCL_P2P_DISABLE", "NCCL_MNNVL_ENABLE", "NCCL_DMABUF_ENABLE")
        },
    }
    for name in ("nvidia-cutlass-dsl", "cuda-bindings", "triton", "nvidia-ml-py"):
        result[name] = importlib.metadata.version(name)
    if case is not None:
        from .provenance import verify_runtime

        if torch.cuda.get_device_capability(device) != (10, 3):
            raise RuntimeError("the frozen campaign requires a GB300 device")
        result["frozen_provider"] = verify_runtime(case.scope)
    return result


def run_case(  # noqa: C901
    case: Case,
    output: Path,
    *,
    smoke: bool,
    validation_only: bool,
    local_check: bool,
) -> None:
    """Commit each rank's diagnostics and require collective completion."""
    import torch
    import torch.distributed as dist

    from .measurement import GraphIteration, nvlink_bandwidth, profile, time_windows
    from .reference import verify
    from .runtime import Workload

    rank = dist.get_rank()
    directory = output / "cases" / case.backend / case.identifier / "attempt-0"
    rank_dir = directory / f"rank-{rank}"
    if rank_dir.exists():
        raise FileExistsError(
            f"attempt already exists: {rank_dir}; choose a fresh output root"
        )
    rank_dir.mkdir(parents=True)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    workload = iteration = None
    write_json(rank_dir / "status.json", {"state": "running", "case": case.to_dict()})
    try:
        identity = _runtime_identity(case)
        write_json(rank_dir / "runtime.json", identity)
        workload = Workload(case, device)
        write_json(
            rank_dir / "routing.json",
            [routing_summary(case, rank, i) for i in range(case.layers)],
        )
        if case.backend == "dist_moe":
            write_json(rank_dir / "memory_plan.json", dataclasses.asdict(workload.plan))
        else:
            write_json(rank_dir / "memory_plan.json", workload.mok_settings)

        numerical = verify(
            workload,
            progress=lambda phase: write_json(
                rank_dir / "status.json",
                {"state": "running", "phase": phase, "case": case.to_dict()},
            ),
        )
        write_json(rank_dir / "numerics.json", numerical)
        for _ in range(10):
            workload.step()
        iteration = (
            GraphIteration(workload)
            if case.timing_mode == "cuda_graph" and not local_check
            else workload
        )
        if isinstance(iteration, GraphIteration):
            from .measurement import verify_graph

            verify_graph(workload, iteration)
        timings = (
            []
            if validation_only or local_check
            else time_windows(workload, iteration, smoke=smoke)
        )
        paired_eager = (
            time_windows(workload, workload, smoke=smoke)
            if timings and case.execution_mode == "cuda_graph"
            else []
        )
        torch.cuda.memory._record_memory_history(max_entries=20000)
        try:
            memory = workload.memory_diagnostic()
            write_json(rank_dir / "allocator_stats.json", memory)
            if rank in (0, case.ep - 1):
                torch.cuda.memory._dump_snapshot(
                    str(rank_dir / "memory_snapshot.pickle")
                )
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
        if (
            case.layers > 1
            and case.backend == "dist_moe"
            and any(memory["capacity_recompute_layers"])
        ):
            raise AssertionError(
                "packing fits, but the requested factor2.5 did not retain all ten layers"
            )
        traces = (
            profile(workload, iteration, rank_dir)
            if not local_check
            else {"state": "Not Run", "reason": "focused local numerical/memory check"}
        )
        bandwidth = (
            nvlink_bandwidth(workload, iteration)
            if not validation_only and not local_check
            else {"state": "Not Run", "reason": "validation-only execution"}
        )
        latency = summarize_windows(case, timings) if timings else {}
        metrics = {
            "case": case.to_dict(),
            "scope": case.scope,
            "timing_mode": "eager_local_check" if local_check else case.timing_mode,
            "shared_expert_schedule": case.shared_expert_schedule,
            "graph_verification": "Passed: changed input values, stable addresses, output/gradient equivalence"
            if isinstance(iteration, GraphIteration)
            else "Not Applicable: eager execution",
            "numerics": "Passed",
            "full_forward_backward_fit": "Passed"
            if case.layers > 1
            else "Not Applicable",
            "latency": latency,
            "matched_eager_latency": summarize_windows(case, paired_eager)
            if paired_eager
            else None,
            "effective_hbm_bandwidth": effective_hbm_bandwidth(case, rank, latency),
            "trace": traces,
            "nvlink_bandwidth": bandwidth,
            "stream_priorities": {
                "routed": workload.routed_stream.priority,
                "shared": workload.shared_stream.priority
                if workload.shared_stream
                else None,
            },
            "graph_pool_ids": {
                "forward": list(iteration.forward_graph.pool()),
                "backward": list(iteration.backward_graph.pool())
                if iteration.backward_graph
                else None,
            }
            if isinstance(iteration, GraphIteration)
            else None,
            "iterations": {
                "warmup": 10,
                "timed": 0 if not timings else 10 if smoke else 150,
                "trace": 0 if local_check else 3,
                "memory_diagnostic": 1,
            },
        }
        if paired_eager:
            metrics["iterations"]["matched_eager_timed"] = 10 if smoke else 150
        write_json(rank_dir / "metrics.json", metrics)
    except BaseException:
        write_json(
            rank_dir / "status.json",
            {
                "state": "failed",
                "error": traceback.format_exc(),
                "case_id": case.identifier,
            },
        )
        if rank == 0:
            write_json(
                directory / "status.json",
                {"state": "failed", "case_id": case.identifier},
            )
        raise
    finally:
        try:
            if isinstance(iteration, GraphIteration):
                iteration.close()
            if workload is not None:
                workload.close()
        except BaseException:
            write_json(
                rank_dir / "status.json",
                {
                    "state": "failed",
                    "error": traceback.format_exc(),
                    "case_id": case.identifier,
                },
            )
            raise
    write_json(
        rank_dir / "cleanup.json",
        {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
        },
    )
    dist.barrier()
    state = "functional_pass" if validation_only or local_check else "complete"
    write_json(rank_dir / "status.json", {"state": state, "case_id": case.identifier})
    dist.barrier()
    if rank == 0:
        write_json(
            directory / "status.json",
            {"state": state, "ranks": case.ep, "case_id": case.identifier},
        )


def main() -> None:
    """Render the CPU-only campaign or execute an explicit distributed batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=Path,
        default=Path(__file__).parent / "configs/mok_default.json",
    )
    parser.add_argument("--render", type=Path)
    parser.add_argument("--scope", choices=("isolated", "comparison"))
    parser.add_argument("--backend", choices=("dist_moe", "mok"))
    parser.add_argument("--mxfp8-fast-math", action="store_true")
    parser.add_argument("--execution-mode", choices=("eager", "cuda_graph"))
    parser.add_argument(
        "--shared-expert-schedule",
        choices=("concurrent", "sequential"),
        default="concurrent",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--case", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--local-check",
        action="store_true",
        help="Numerical and memory execution only; never performance evidence",
    )
    args = parser.parse_args()
    if args.render:
        manifest = render_campaign(
            args.shapes,
            args.render,
            scope=args.scope,
            backend=args.backend,
            shared_expert_schedule=args.shared_expert_schedule,
            mxfp8_fast_math=args.mxfp8_fast_math,
            execution_mode=args.execution_mode,
        )
        counts = {name: len(job["cases"]) for name, job in manifest["jobs"].items()}
        print(counts)
        return
    if not args.manifest or not args.job or not args.case or not args.output:
        parser.error("execution requires --manifest, --job, --case and --output")
    import json

    import torch
    import torch.distributed as dist

    job = json.loads(args.manifest.read_text())["jobs"][args.job]
    cases = [Case.from_dict(job["cases"][identifier]) for identifier in args.case]
    if len({compilation_key(c) for c in cases}) != 1:
        parser.error("one child batch must contain compilation-compatible cases")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(minutes=10), device_id=device
    )
    try:
        for case in cases:
            run_case(
                case,
                args.output,
                smoke=job["smoke"],
                validation_only=case.identifier in job["validation_only"],
                local_check=args.local_check,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
