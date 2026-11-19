# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""BF16 all-to-all reference, independent of fused expert implementations."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.distributed as dist

from .runtime import dense_expert, Workload


def _exchange(tensor: torch.Tensor, send: list[int], recv: list[int]) -> torch.Tensor:
    """Exchange routed rows on NCCL with explicitly matched variable splits."""
    output = tensor.new_empty((sum(recv), *tensor.shape[1:]))
    dist.all_to_all_single(output, tensor.contiguous(), recv, send)
    return output


def reference_layer(
    workload: Workload, layer: int, x_TD: torch.Tensor, dy_TD: torch.Tensor
) -> tuple:
    """Evaluate one actual layer input and incoming gradient in ordinary Torch.

    Route output/gradient transport uses BF16, activation and route-weight sums
    use FP32, and weights remain the original BF16 inputs before quantization.
    Returns output, dx, router gradient, routed gate/up/down gradients and
    optional shared gradients in the same order as the native adapter.
    """
    ids, scores = workload.ids[layer], workload.scores[layer].detach()
    s = workload.case.shape
    local_experts = s.experts // workload.case.ep
    destinations = ids.flatten() // local_experts
    order = destinations.argsort(stable=True)
    counts = destinations.bincount(minlength=workload.case.ep)
    all_counts = torch.empty(
        (workload.case.ep, workload.case.ep), device=x_TD.device, dtype=torch.int64
    )
    dist.all_gather_into_tensor(all_counts, counts)
    send, recv = counts.tolist(), all_counts[:, workload.rank].tolist()
    x_recv = _exchange(x_TD.detach()[order // s.top_k], send, recv)
    expert_recv = _exchange((ids.flatten()[order] % local_experts), send, recv)
    dy_recv = _exchange(
        (dy_TD[:, None].float() * scores[:, :, None])
        .to(torch.bfloat16)
        .reshape(-1, s.hidden)[order],
        send,
        recv,
    )
    y_recv, dx_recv = torch.empty_like(x_recv), torch.empty_like(x_recv)
    gate_weights, up_weights, w2 = workload.reference_weights(layer)
    dw_gate, dw_up, dw2 = (
        torch.zeros_like(gate_weights),
        torch.zeros_like(up_weights),
        torch.zeros_like(w2),
    )
    # Each expert graph is retired immediately, so the reference does not retain
    # every routed expert's intermediate tensors across the whole rank.
    for expert in range(local_experts):
        rows = (expert_recv == expert).nonzero().flatten()
        if not rows.numel():
            continue
        x = x_recv[rows].requires_grad_()
        gate = gate_weights[expert].detach().requires_grad_()
        up = up_weights[expert].detach().requires_grad_()
        down = w2[expert].detach().requires_grad_()
        y = dense_expert(x, (gate, up, down))
        dx, dg, du, dd = torch.autograd.grad(y, (x, gate, up, down), dy_recv[rows])
        y_recv[rows], dx_recv[rows] = y.detach(), dx
        dw_gate[expert], dw_up[expert], dw2[expert] = (
            dg,
            du,
            dd,
        )
    flat_y, flat_dx = (
        torch.empty_like(x_TD).repeat_interleave(s.top_k, dim=0),
        torch.empty_like(x_TD).repeat_interleave(s.top_k, dim=0),
    )
    flat_y[order] = _exchange(y_recv, recv, send)
    flat_dx[order] = _exchange(dx_recv, recv, send)
    route_y = flat_y.reshape(s.tokens, s.top_k, s.hidden).float()
    y = (route_y * scores[:, :, None]).sum(1)
    dx = flat_dx.reshape(s.tokens, s.top_k, s.hidden).float().sum(1)
    ds = (route_y * dy_TD[:, None].float()).sum(2)
    shared_grads = ()
    if workload.case.shared_experts:
        x_shared = x_TD.detach().requires_grad_()
        shared_weights = tuple(
            w.detach().requires_grad_() for w in workload.shared_weights[layer]
        )
        shared_y = dense_expert(x_shared, shared_weights)
        shared_dx, *shared_grads = torch.autograd.grad(
            shared_y, (x_shared, *shared_weights), dy_TD
        )
        y += shared_y.detach().float()
        dx += shared_dx.float()
    return (
        y.to(torch.bfloat16),
        dx.to(torch.bfloat16),
        ds,
        dw_gate,
        dw_up,
        dw2,
        *shared_grads,
    )


def compare_tensor(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    tolerance: tuple[float, float],
) -> dict:
    """Check global maximum absolute and mean relative L1 error, as in MoK."""
    if expected.shape != actual.shape:
        raise AssertionError(
            f"{name}: shape mismatch {expected.shape} != {actual.shape}"
        )
    diff = (expected.float() - actual.float()).abs()
    stats = torch.stack(
        (
            diff.sum().double(),
            expected.float().abs().sum().double(),
            diff.new_tensor(diff.numel()).double(),
        )
    )
    maximum = diff.max()
    dist.all_reduce(stats)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    absolute = maximum.item()
    relative = (
        (stats[0] / stats[1]).item()
        if stats[1].item()
        else (0.0 if stats[0].item() == 0 else math.inf)
    )
    result = {
        "name": name,
        "max_absolute": absolute,
        "mean_absolute": (stats[0] / stats[2]).item(),
        "relative_l1": relative,
        "absolute_tolerance": tolerance[0],
        "relative_tolerance": tolerance[1],
    }
    if (
        not math.isfinite(absolute)
        or not math.isfinite(relative)
        or absolute > tolerance[0]
        or relative > tolerance[1]
    ):
        raise AssertionError(f"numerical verification failed: {result}")
    return result


def _compare_nvfp4(name: str, expected: torch.Tensor, actual: torch.Tensor) -> dict:
    """Use the public inference tests' relative-L2/cosine NVFP4 bounds."""
    ref, value = expected.float(), actual.float()
    stats = torch.stack(
        (
            (ref - value).square().sum(),
            ref.square().sum(),
            value.square().sum(),
            (ref * value).sum(),
        )
    ).double()
    dist.all_reduce(stats)
    l2 = (stats[0] / stats[1].clamp_min(1e-24)).sqrt().item()
    cosine = (stats[3] / (stats[1] * stats[2]).clamp_min(1e-24).sqrt()).item()
    if (
        not math.isfinite(l2)
        or not math.isfinite(cosine)
        or l2 >= 0.29
        or cosine <= 0.955
    ):
        raise AssertionError(
            f"NVFP4 {name}: relativeL2={l2}, cosine={cosine}; required L2<0.29/cosine>0.955"
        )
    return {
        "name": name,
        "relative_l2": l2,
        "relative_l2_bound": 0.29,
        "cosine": cosine,
        "cosine_floor": 0.955,
    }


def verify(
    workload: Workload, *, progress: Callable[[str], None] | None = None
) -> list[dict]:
    """Check outputs and every training gradient at each actual layer input.

    Args:
        workload: Collectively constructed workload on every EP rank.
        progress: Optional CPU-only callback before each untimed check phase.
    """
    if progress is not None:
        progress("forward")
    workload.forward()
    if workload.training:
        if progress is not None:
            progress("backward")
        workload.backward()
    if progress is not None:
        progress("device_completion")
    torch.cuda.synchronize(workload.device)
    tolerance = (0.5, 0.01) if workload.case.precision == "bf16" else (1.0, 0.1)
    checks = []
    names = (
        "output",
        "dx",
        "router",
        "routed_gate",
        "routed_up",
        "routed_down",
        "shared_gate",
        "shared_up",
        "shared_down",
    )
    for layer in reversed(range(workload.case.layers)):
        if progress is not None:
            progress(f"reference_layer{layer}")
        frame = workload.frames[layer]
        dy = workload.output_gradients[layer]
        expected = reference_layer(workload, layer, frame[0], dy)
        if not workload.training:
            expected = expected[:1]
        y = frame[1]
        if workload.case.backend == "dist_moe" and frame[2] is not None:
            y = (y.float() + frame[2].float()).to(torch.bfloat16)
        actual = (
            (y, *workload.gradients[workload.case.layers - 1 - layer])
            if workload.training
            else (y,)
        )
        for name, ref, value in zip(
            names[: len(expected)], expected, actual, strict=True
        ):
            label = f"layer{layer}/{name}"
            checks.append(
                _compare_nvfp4(label, ref, value)
                if workload.case.precision == "nvfp4"
                else compare_tensor(label, ref, value, tolerance)
            )
    return checks
