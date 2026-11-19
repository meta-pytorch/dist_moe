# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Steady-state GPU event timing and separate profiler diagnostics."""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .metrics import trace_summary
from .runtime import Workload


class GraphIteration:
    """Capture forward/backward separately with one pool and fixed saved state."""

    def __init__(self, workload: Workload) -> None:
        """Require the runtime's real semantic annotation API before capture."""
        from torch.cuda import graph_annotations

        if not graph_annotations.is_available():
            raise RuntimeError("the packaged graph annotation provider is unavailable")
        graph_annotations.clear_kernel_annotations()
        self.workload = workload
        self.forward_graph = torch.cuda.CUDAGraph()
        self.backward_graph = torch.cuda.CUDAGraph() if workload.training else None
        stream = workload.routed_stream
        stream.wait_stream(torch.cuda.current_stream(workload.device))
        options = {"stream": stream, "enable_annotations": True}
        with torch.autograd.grad_mode.set_multithreading_enabled(False):
            with (
                torch.cuda.graph(self.forward_graph, **options),
                graph_annotations.mark_kernels("benchmark/forward"),
            ):
                self.output = workload.forward()
            self.frames = workload.frames
            if self.backward_graph:
                with (
                    torch.cuda.graph(
                        self.backward_graph, pool=self.forward_graph.pool(), **options
                    ),
                    graph_annotations.mark_kernels("benchmark/backward"),
                ):
                    self.gradients = workload.backward(retain_graph=True)
        torch.cuda.current_stream(workload.device).wait_stream(stream)
        torch.cuda.synchronize(workload.device)
        self.annotations = graph_annotations.get_kernel_annotations()
        if not self.annotations:
            raise RuntimeError("graph capture produced no semantic kernel annotations")

    def forward(self) -> None:
        """Replay forward with the captured routing values and persistent state."""
        self.forward_graph.replay()

    def backward(self) -> None:
        """Replay backward after its forward on the current stream."""
        self.backward_graph.replay()

    def step(self) -> None:
        """Replay one complete captured iteration."""
        self.forward()
        if self.backward_graph:
            self.backward()

    def close(self) -> None:
        """Drop graph and saved-state references before closing the workload."""
        self.forward_graph.reset()
        if self.backward_graph:
            self.backward_graph.reset()
        self.frames = None
        self.output = None
        self.gradients = None
        gc.collect()


def verify_graph(workload: Workload, iteration: GraphIteration) -> None:
    """Compare independent eager references after changing values at fixed addresses."""
    from .reference import compare_tensor

    iteration.step()
    torch.cuda.synchronize(workload.device)
    addresses = [x.data_ptr() for x in workload.inputs]
    with torch.no_grad():
        for x in workload.inputs:
            x.mul_(-0.5)
        if workload.training:
            for dy in workload.output_gradients:
                dy.mul_(-0.5)
    try:
        eager_y = workload.forward().detach().clone()
        eager_grads = (
            [
                tuple(grad.detach().clone() for grad in layer)
                for layer in workload.backward()
            ]
            if workload.training
            else []
        )
        iteration.step()
        torch.cuda.synchronize(workload.device)
        compare_tensor(
            "graph_changed_input_output", eager_y, iteration.output, (0.0, 0.0)
        )
        if workload.training:
            for i, (eager, captured) in enumerate(
                zip(eager_grads, iteration.gradients, strict=True)
            ):
                for j, (a, b) in enumerate(zip(eager, captured, strict=True)):
                    compare_tensor(
                        f"graph_changed_input_gradient/{i}/{j}", a, b, (0.0, 0.0)
                    )
        if addresses != [x.data_ptr() for x in workload.inputs]:
            raise AssertionError("graph verification changed captured input addresses")
    finally:
        # Power-of-two scaling restores BF16 inputs exactly without extra buffers.
        with torch.no_grad():
            for x in workload.inputs:
                x.mul_(-2)
            if workload.training:
                for dy in workload.output_gradients:
                    dy.mul_(-2)


def time_windows(workload: Workload, iteration, *, smoke: bool) -> list[dict]:
    """Measure10 smoke or5x30 suite iterations without profiler collection."""
    windows = []
    for _ in range(1 if smoke else 5):
        dist.barrier()
        torch.cuda.synchronize(workload.device)
        events = [
            tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
            for _ in range(10 if smoke else 30)
        ]
        for start, middle, end in events:
            start.record()
            iteration.forward()
            middle.record()
            if workload.training:
                iteration.backward()
            end.record()
        torch.cuda.synchronize(workload.device)
        local = [
            sum(a.elapsed_time(b) for a, b, _ in events) / len(events),
            sum(b.elapsed_time(c) for _, b, c in events) / len(events),
            sum(a.elapsed_time(c) for a, _, c in events) / len(events),
        ]
        values = torch.tensor(local, dtype=torch.float64, device=workload.device)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        windows.append(
            dict(
                zip(
                    ("forward_ms", "backward_ms", "combined_ms"),
                    values.tolist(),
                    strict=True,
                )
            )
        )
    return windows


def profile(workload: Workload, iteration, directory: Path) -> dict:
    """Write and parse three all-rank steady-state iterations outside timing."""
    dist.barrier()
    torch.cuda.synchronize(workload.device)
    path = directory / "trace.json.gz"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profiler:
        for _ in range(3):
            with torch.profiler.record_function("benchmark/forward"):
                iteration.forward()
            if workload.training:
                with torch.profiler.record_function("benchmark/backward"):
                    iteration.backward()
            profiler.step()
        torch.cuda.synchronize(workload.device)
    if isinstance(iteration, GraphIteration):
        profiler.export_chrome_trace(
            str(path), cuda_graph_annotations=iteration.annotations
        )
    else:
        profiler.export_chrome_trace(str(path))
    trace = trace_summary(
        path,
        shared=workload.case.shared_experts == 1
        and workload.case.backend == "dist_moe",
        graph=isinstance(iteration, GraphIteration),
    )
    if workload.case.shared_expert_schedule == "sequential":
        overlap = trace["timeline"]["shared_routed_overlap_ms"]
        if overlap > 1e-6:
            raise AssertionError(
                f"sequential shared/routed execution overlapped by {overlap} ms"
            )
    return trace


def nvlink_bandwidth(workload: Workload, iteration) -> dict:
    """Collect hardware NVLink TX/RX rates over at least one second of work.

    NVML GPM units are returned with the values; unsupported counters fail
    closed. Collection is outside latency timing and includes all GPU work
    during the sampling interval, so placement must exclude other GPU users.
    """
    import pynvml as nvml

    nvml.nvmlInit()
    samples = []
    try:
        uuid = torch.cuda.get_device_properties(workload.device).uuid
        handle = nvml.nvmlDeviceGetHandleByUUID(str(uuid))
        if not nvml.nvmlGpmQueryDeviceSupport(handle).isSupportedDevice:
            raise RuntimeError("NVML GPM is unsupported on the selected GPU")
        samples = [nvml.nvmlGpmSampleAlloc(), nvml.nvmlGpmSampleAlloc()]
        metrics = nvml.c_nvmlGpmMetricsGet_t()
        metrics.version = nvml.NVML_GPM_METRICS_GET_VERSION
        metrics.numMetrics = 2
        metrics.sample1, metrics.sample2 = samples
        metrics.metrics[0].metricId = nvml.NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC
        metrics.metrics[1].metricId = nvml.NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC
        dist.barrier()
        torch.cuda.synchronize(workload.device)
        first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        first.record()
        for _ in range(30):
            iteration.step()
        last.record()
        torch.cuda.synchronize(workload.device)
        calibrated = torch.tensor(first.elapsed_time(last) / 30, device=workload.device)
        dist.all_reduce(calibrated, op=dist.ReduceOp.MAX)
        count = max(30, math.ceil(2000 / calibrated.item()))
        dist.barrier()
        torch.cuda.synchronize(workload.device)
        nvml.nvmlGpmSampleGet(handle, samples[0])
        start = time.monotonic()
        for _ in range(count):
            iteration.step()
        torch.cuda.synchronize(workload.device)
        elapsed = time.monotonic() - start
        nvml.nvmlGpmSampleGet(handle, samples[1])
        minimum = torch.tensor(elapsed, device=workload.device)
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        if minimum.item() < 1:
            raise RuntimeError(
                f"NVLink collection window shorter than one second: {minimum.item()}"
            )
        nvml.nvmlGpmMetricsGet(metrics)
        result = {
            "seconds": elapsed,
            "iterations": count,
            "collector": "NVML GPM",
            "uuid": str(uuid),
        }
        for name, metric in zip(("tx", "rx"), metrics.metrics[:2], strict=True):
            if metric.nvmlReturn != nvml.NVML_SUCCESS:
                raise RuntimeError(
                    f"NVLink {name} counter failed: NVML error {metric.nvmlReturn}"
                )
            result[name] = {
                "value": metric.value,
                "unit": metric.metricInfo.unit.decode(),
            }
        return result
    finally:
        for sample in samples:
            nvml.nvmlGpmSampleFree(sample)
        nvml.nvmlShutdown()
