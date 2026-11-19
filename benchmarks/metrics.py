# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Portable artifact writing, trace validation and workload accounting."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import statistics
import uuid
from pathlib import Path

from .config import Case
from .routing import expert_quotas, rank_quotas


def _open_json_temporary(path: Path):
    """Create a fresh owned UUID object or finish its empty partial creation.

    Args:
        path: Newly generated temporary path owned by this publication.

    Returns:
        Writable stream positioned at the start of the empty object.

    Raises:
        FileExistsError: The conflicting object contains data.
    """
    try:
        return path.open("x")
    except FileExistsError:
        # Manifold can create our empty object before reporting a create error.
        if path.stat().st_size:
            raise
        return path.open("r+")


def write_json(path: Path, value: object) -> None:
    """Publish complete JSON using a fresh temporary object for each write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Object-backed mounts can retain the old name after publishing its object.
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    with _open_json_temporary(temporary) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def artifact_digest(path: Path) -> str:
    """Hash a completed artifact for independent persistent-storage readback."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval_union(events: list[dict]) -> list[tuple[float, float]]:
    """Merge trace intervals in microseconds so concurrent work is counted once."""
    merged = []
    for start, end in sorted((e["ts"], e["ts"] + e["dur"]) for e in events):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _busy_ms(events: list[dict]) -> float:
    """Return the union duration of one device's events in milliseconds."""
    return sum(end - start for start, end in _interval_union(events)) / 1000


def _overlap_ms(left: list[dict], right: list[dict]) -> float:
    """Intersect two unions without double-counting simultaneous GPU kernels."""
    a, b = _interval_union(left), _interval_union(right)
    i = j = 0
    overlap = 0.0
    while i < len(a) and j < len(b):
        overlap += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return overlap / 1000


def _kernel_kind(event: dict) -> str:
    """Classify known GEMM, communication and fused symbols conservatively."""
    name = event.get("name", "").lower()
    if (
        "dispatch_mlp_swiglu" in name
        or "dist_grouped_gemm" in name
        or ("mega" in name and "gemm" in name)
    ):
        return "fused_compute_communication"
    if any(
        s in name for s in ("nccl", "dispatch", "combine", "all_gather", "allgather")
    ):
        return "communication"
    if any(s in name for s in ("gemm", "mlp", "cutlass", "matmul", "nvjet", "xmma")):
        return "compute"
    return "other"


def _timeline(events: list[dict], kernels: list[dict], *, shared: bool) -> dict:
    """Attribute GPU branches through graph tags or eager launch correlations.

    Synchronous branch calls bound enqueue time, including autograd's device
    threads. CPU annotations identify which branch enqueued a kernel. Overlap is then
    computed from GPU timestamps; CPU scope duration is never GPU execution
    time. Fused communication/compute kernels remain a separate category.
    """
    scopes = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("dur", 0) > 0
        and e.get("name", "").startswith(("shared/", "routed/"))
    ]
    apis = [
        e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("dur", 0) > 0
    ]
    launches = {
        e["args"]["correlation"]: e
        for e in apis
        if "launch" in e.get("name", "").lower() and "correlation" in e.get("args", {})
    }
    branches = {"shared": [], "routed": []}
    cross_thread = 0
    for kernel in kernels:
        labels = json.dumps(kernel.get("args", {}))
        tagged = [name for name in branches if name + "/" in labels]
        if len(tagged) > 1:
            raise RuntimeError("graph kernel has ambiguous shared/routed ownership")
        if tagged:
            branches[tagged[0]].append(kernel)
            continue
        launch = launches.get(kernel.get("args", {}).get("correlation"))
        if launch is None:
            continue
        enclosing = [
            s
            for s in scopes
            if s["pid"] == launch["pid"]
            and s["ts"] <= launch["ts"] <= s["ts"] + s["dur"]
        ]
        if len({s["name"].split("/")[0] for s in enclosing}) > 1:
            raise RuntimeError(
                "trace launch has ambiguous shared/routed branch ownership"
            )
        if enclosing:
            scope = min(enclosing, key=lambda s: s["dur"])
            branches[scope["name"].split("/")[0]].append(kernel)
            cross_thread += scope["tid"] != launch["tid"]
        elif shared and _kernel_kind(kernel) != "other":
            raise RuntimeError(
                "shared trace contains target GPU work outside a branch call"
            )
    if shared and any(not branches[name] for name in branches):
        raise RuntimeError("shared trace lacks correlated GPU work for both branches")
    devices = {e.get("args", {}).get("device") for e in kernels}
    if len(devices) != 1:
        raise RuntimeError(f"rank trace contains multiple GPU devices: {devices}")
    device = next(iter(devices))
    activity = [
        e
        for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        and e.get("dur", 0) > 0
        and e.get("args", {}).get("device") == device
    ]
    classes = {
        kind: [e for e in kernels if _kernel_kind(e) == kind]
        for kind in ("compute", "communication", "fused_compute_communication", "other")
    }
    first = min(e["ts"] for e in activity)
    span_ms = (max(e["ts"] + e["dur"] for e in activity) - first) / 1000
    launch_apis = [e for e in apis if "launch" in e.get("name", "").lower()]
    waits = [
        e
        for e in apis
        if "wait" in e.get("name", "").lower()
        or "synchronize" in e.get("name", "").lower()
    ]
    return {
        "kernel_busy_ms": {kind: _busy_ms(value) for kind, value in classes.items()},
        "compute_communication_overlap_ms": _overlap_ms(
            classes["compute"], classes["communication"]
        ),
        "shared_routed_overlap_ms": _overlap_ms(branches["shared"], branches["routed"])
        if shared
        else None,
        "branch_kernels": {name: len(value) for name, value in branches.items()},
        "cross_thread_branch_kernels": cross_thread,
        "branch_busy_ms": {name: _busy_ms(value) for name, value in branches.items()},
        "device_activity_span_ms": span_ms,
        "device_activity_gap_ms": max(0, span_ms - _busy_ms(activity)),
        "host_launch_api_count": len(launch_apis),
        "host_launch_api_sum_ms": sum(e["dur"] for e in launch_apis) / 1000,
        "host_dependency_apis": [
            {
                "name": e["name"],
                "thread": e["tid"],
                "start_ms_from_first_gpu_activity": (e["ts"] - first) / 1000,
                "duration_ms": e["dur"] / 1000,
            }
            for e in waits
        ],
        "interpretation": "GPU interval unions; host API durations are CPU costs. Fused kernel time cannot be split into communication and compute; stream-wait API time is enqueue cost, not device wait time.",
    }


def trace_summary(path: Path, *, shared: bool, graph: bool) -> dict:
    """Require GPU compute, communication and semantic phase annotations.

    Fused distributed GEMM and MoK kernels perform communication and compute;
    their duration cannot be split into those components using a timeline alone.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        trace = json.load(stream)
    events = trace.get("traceEvents", [])
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
    compute = [
        e
        for e in kernels
        if _kernel_kind(e) in ("compute", "fused_compute_communication")
    ]
    communication = [
        e
        for e in kernels
        if _kernel_kind(e) in ("communication", "fused_compute_communication")
    ]
    annotations = [
        e
        for e in events
        if any(
            s in e.get("name", "")
            for s in ("routed/forward", "benchmark/forward", "shared/forward")
        )
    ]
    early_stop = [
        e
        for e in events
        if "collection" in e.get("name", "").lower()
        and "stop" in e.get("name", "").lower()
    ]
    if not kernels or not compute or not communication or not annotations or early_stop:
        raise RuntimeError(
            f"trace lacks required device/phase evidence: kernels={len(kernels)}, compute={len(compute)}, communication={len(communication)}, annotations={len(annotations)}, early_stop={early_stop}"
        )
    if graph and not any(
        "benchmark/" in json.dumps(e.get("args", {}))
        or "benchmark/" in e.get("name", "")
        for e in kernels
    ):
        raise RuntimeError("graph trace lacks GPU kernel semantic annotations")
    if shared and not any(
        "shared/forward" in e.get("name", "")
        or "shared/forward" in json.dumps(e.get("args", {}))
        for e in events
    ):
        raise RuntimeError("shared comparison trace lacks the shared branch annotation")
    return {
        "gpu_kernels": len(kernels),
        "compute_kernels": len(compute),
        "communication_or_fused_kernels": len(communication),
        "annotations": len(annotations),
        "timeline": _timeline(events, kernels, shared=shared),
        "streams": sorted(
            {
                e.get("args", {}).get("stream")
                for e in kernels
                if e.get("args", {}).get("stream") is not None
            }
        ),
        "sha256": artifact_digest(path),
    }


def summarize_windows(case: Case, windows: list[dict]) -> dict:
    """Report maximum-rank phase latency and useful average/aggregate TFLOP/s."""
    result = {}
    for phase in ("forward", "backward", "combined"):
        if phase == "backward" and case.mode == "inference":
            continue
        samples = [window[f"{phase}_ms"] for window in windows]
        ms = statistics.median(samples)
        if not ms > 0:
            raise ValueError(f"nonpositive {phase} timing: {samples}")
        flops = case.useful_flops(backward=phase == "backward")
        if phase == "combined" and case.mode == "training":
            flops += case.useful_flops(backward=True)
        result[phase] = {
            "max_rank_window_ms": samples,
            "median_ms": ms,
            "min_ms": min(samples),
            "max_ms": max(samples),
            "useful_gemm_flops_per_gpu": flops,
            "useful_tflops_per_gpu": flops / ms / 1e9,
            "useful_tflops_total": case.ep * flops / ms / 1e9,
        }
    return result


def effective_hbm_bandwidth(case: Case, rank: int, latency: dict) -> dict:
    """Estimate logical workload bytes per measured phase, without DRAM counters.

    Uses the internal benchmark's representative forward tensors and coarse
    1/2/3 forward/backward/combined byte multipliers. Actual destination quotas
    determine active weights and activation rows. Shared work remains BF16.
    Padding, replay, cache effects and physical tensor rereads are excluded.

    Args:
        case: Logical workload and precision, independent of native scheduling.
        rank: Destination rank whose useful bytes are counted.
        latency: Phase medians from ``summarize_windows`` in milliseconds.

    Returns:
        Explicit analytical method, forward byte breakdown and phase GB/s.
    """
    if not 0 <= rank < case.ep:
        raise ValueError("rank is outside the expert-parallel group")
    s = case.shape
    local = s.experts // case.ep
    bits, vector = {"bf16": (16, 0), "mxfp8": (8, 32), "nvfp4": (4, 16)}[case.precision]
    weight_per_expert = 2 * s.intermediate * (
        math.ceil(s.hidden * bits / 8) + (math.ceil(s.hidden / vector) if vector else 0)
    ) + s.hidden * (
        math.ceil(s.intermediate * bits / 8)
        + (math.ceil(s.intermediate / vector) if vector else 0)
    )
    parts = {
        "inputs": 0,
        "routed_weights": 0,
        "routing_scores": 0,
        "routed_activations": 0,
        "shared_work": 0,
    }
    for layer in range(case.layers):
        counts = expert_quotas(case, layer)[rank * local : (rank + 1) * local]
        rows = rank_quotas(case, layer)[0][rank]
        parts["inputs"] += s.tokens * s.hidden * 2
        parts["routed_weights"] += sum(n > 0 for n in counts) * weight_per_expert
        parts["routing_scores"] += s.tokens * s.top_k * 4
        parts["routed_activations"] += rows * (2 * s.intermediate + s.hidden) * 2
        if case.shared_experts:
            parts["shared_work"] += (
                s.tokens * s.hidden * 2
                + 3 * s.hidden * s.intermediate * 2
                + s.tokens * (2 * s.intermediate + s.hidden) * 2
            )
    forward = sum(parts.values())
    phases = {}
    for phase, measured in latency.items():
        multiplier = (
            2
            if phase == "backward"
            else 3
            if phase == "combined" and case.mode == "training"
            else 1
        )
        phases[phase] = {
            "logical_bytes_per_iteration": forward * multiplier,
            "gb_s": forward * multiplier / measured["median_ms"] / 1e6,
        }
    return {
        "state": "Passed" if phases else "Not Run",
        "method": "analytical_bytes_over_measured_latency",
        "hardware_counter_measurement": False,
        "forward_bytes": parts,
        "training_bytes": "coarse 3x forward model; not a backward traffic audit",
        "weight_qdata_bits": bits,
        "weight_scale_vector": vector or None,
        "activation_element_bytes": 2,
        "score_element_bytes": 4,
        "includes_padding_or_replay": False,
        "phases": phases,
    }
