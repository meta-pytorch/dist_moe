# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""GPU workloads using public Dist-MoE and optional native MoK APIs."""

from __future__ import annotations

import dataclasses
import gc
import math
from contextlib import nullcontext

import dist_moe
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from .config import Case
from .routing import local_routes, mok_memory_settings


def _graph_kernel_scope(case: Case, label: str):
    """Attach comparison branch identity without replacing isolated phase tags.

    Args:
        case: Workload whose shared/routed branches require separate tags.
        label: Shared or routed phase/layer label.

    Returns:
        Graph annotation context, or an empty context outside capture.
    """
    if case.shared_experts and torch.cuda.is_current_stream_capturing():
        from torch.cuda.graph_annotations import mark_kernels

        return mark_kernels(label)
    return nullcontext()


def dense_expert(x_TD: torch.Tensor, weights: tuple) -> torch.Tensor:
    """Apply a BF16 dense SwiGLU expert with FP32 activation arithmetic."""
    gate_FD, up_FD, down_DF = weights
    gate_TF = F.linear(x_TD, gate_FD).float()
    up_TF = F.linear(x_TD, up_FD).float()
    hidden_TF = (F.silu(gate_TF) * up_TF).to(torch.bfloat16)
    return F.linear(hidden_TF, down_DF)


def _random_tensor(
    shape: tuple, device: torch.device, seed: int, scale: float, grad: bool
) -> torch.Tensor:
    """Initialize an independent BF16 tensor without perturbing other seeds."""
    generator = torch.Generator(device=device).manual_seed(seed)
    tensor = torch.randn(
        shape, generator=generator, device=device, dtype=torch.bfloat16
    )
    return tensor.mul_(scale).requires_grad_(grad)


def make_config(case: Case) -> dist_moe.Config:
    """Resolve the same public device-only configuration used by real workloads.

    Args:
        case: Dist-MoE workload, precision, pipeline and activation-slot depth.

    Returns:
        Validated public config with shape-aware defaults and no host VMM.
    """
    training = case.mode == "training"
    c, s = case, case.shape
    policy = None
    if c.precision != "bf16":
        fmt = (
            dist_moe.BlockScaledFormat.MXFP8_E4M3
            if c.precision == "mxfp8"
            else dist_moe.BlockScaledFormat.NVFP4
        )
        policy = dist_moe.BlockScaledConfig(
            format=fmt, pipeline=c.pipeline, fast_math=c.fast_math
        )
    capacity = min(8.0, c.max_imbalance)
    return dist_moe.Config(
        max_num_local_input_tokens=s.tokens,
        hidden_dim=s.hidden,
        intermediate_dim=s.intermediate,
        top_k=s.top_k,
        num_experts=s.experts,
        max_moe_layers_per_activation_slot=c.layers,
        device_scratch_capacity_factor=capacity,
        activation_slot_capacity_factor=(2.5 if c.layers > 1 else capacity)
        if training
        else None,
        num_activation_slots=1 if training else 0,
        block_scaled=policy,
        inference=not training,
    )


class RecomputeObserver(TorchDispatchMode):
    """Read actual planner flags from the pinned registered forward schemas.

    This diagnostic runs outside timing and keeps only scalar flag tensors.
    Unknown schemas fail closed instead of inferring recomputation from capacity.
    """

    def __init__(self) -> None:
        """Start an empty per-forward diagnostic."""
        super().__init__()
        self.flags: list[torch.Tensor] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Observe the native flag after delegating the unmodified operation."""
        result = func(*args, **(kwargs or {}))
        contracts = {
            "dist_moe.bf16_forward.default": (9, 6),
            "dist_moe.block_scaled_forward.default": (21, 16),
        }
        if str(func) in contracts:
            size, index = contracts[str(func)]
            state = result[1]
            if (
                len(state) != size
                or state[index].dtype != torch.bool
                or state[index].numel() != 1
            ):
                raise RuntimeError(
                    f"unsupported recomputation diagnostic schema: {func}"
                )
            self.flags.append(state[index])
        return result


class Workload:
    """Own one case's weights, routing, contexts, streams and layer lifetimes.

    All group ranks construct, execute and close the workload collectively in
    the same order. Ten-layer cases use distinct weights and retain all forward
    frames until reverse-order backward. Weight quantization is outside timing.
    """

    def __init__(self, case: Case, device: torch.device) -> None:
        """Allocate on the current device-bound NCCL group and local CUDA device."""
        self.case, self.device = case, device
        self.rank = dist.get_rank()
        if dist.get_world_size() != case.ep:
            raise ValueError("case EP must equal the real process-group size")
        self.training = case.mode == "training"
        self.autograd = self.training and case.backend == "dist_moe"
        self.context = None
        self.workspace = None
        self.frames = []
        self.gradients = []
        self.routed_stream = torch.cuda.Stream(device=device, priority=-1)
        self.shared_stream = (
            (
                self.routed_stream
                if case.shared_expert_schedule == "sequential"
                else torch.cuda.Stream(device=device, priority=0)
            )
            if case.shared_experts
            else None
        )
        s = case.shape
        self.inputs = [
            _random_tensor(
                (s.tokens, s.hidden),
                device,
                case.seed + 100 * i + self.rank,
                1,
                self.autograd,
            )
            for i in range(case.layers)
        ]
        self.output_gradients = [
            _random_tensor(
                (s.tokens, s.hidden),
                device,
                case.seed + 1000 + 100 * i + self.rank,
                1 / s.hidden,
                False,
            )
            for i in range(case.layers)
        ]
        self.ids = [
            torch.tensor(
                local_routes(case, self.rank, i), dtype=torch.int64, device=device
            )
            for i in range(case.layers)
        ]
        self.scores = [
            torch.full(
                (s.tokens, s.top_k),
                1 / s.top_k,
                dtype=torch.float32,
                device=device,
                requires_grad=self.autograd,
            )
            for _ in range(case.layers)
        ]
        self.weights = []
        self.shared_weights = []
        for i in range(case.layers):
            seed = case.seed + 10000 + 100 * i + self.rank
            self.weights.append(
                (
                    _random_tensor(
                        (s.experts // case.ep, 2 * s.intermediate, s.hidden),
                        device,
                        seed,
                        1 / math.sqrt(s.hidden),
                        self.autograd,
                    ),
                    _random_tensor(
                        (s.experts // case.ep, s.hidden, s.intermediate),
                        device,
                        seed + 1,
                        1 / math.sqrt(s.intermediate),
                        self.autograd,
                    ),
                )
            )
            if case.shared_experts:
                self.shared_weights.append(
                    (
                        _random_tensor(
                            (s.intermediate, s.hidden),
                            device,
                            seed + 2,
                            1 / math.sqrt(s.hidden),
                            self.autograd,
                        ),
                        _random_tensor(
                            (s.intermediate, s.hidden),
                            device,
                            seed + 3,
                            1 / math.sqrt(s.hidden),
                            self.autograd,
                        ),
                        _random_tensor(
                            (s.hidden, s.intermediate),
                            device,
                            seed + 4,
                            1 / math.sqrt(s.intermediate),
                            self.autograd,
                        ),
                    )
                )
        if case.backend == "dist_moe":
            self._initialize_dist_moe()
        else:
            self._initialize_mok()
        if torch.cuda.current_device() != device.index:
            raise RuntimeError("backend initialization changed the current CUDA device")
        torch.cuda.synchronize(device)

    def _initialize_dist_moe(self) -> None:
        """Use one device scratch region and one shared activation slot."""
        c = self.case
        self.config = make_config(c)
        self.policy = self.config.block_scaled
        self.plan = dist_moe.plan_memory(self.config, c.ep, device=self.device)
        self.context = dist_moe.create_context(
            group=dist.group.WORLD, config=self.config, device=self.device
        )
        self.prepared = self.weights
        if self.policy:
            self.prepared = [
                (
                    dist_moe.prepare_block_scaled_weight(
                        w13,
                        self.policy,
                        inference=not self.autograd,
                        fuse_swiglu=not self.training and c.pipeline == "mega",
                    ),
                    dist_moe.prepare_block_scaled_weight(
                        w2, self.policy, inference=not self.training
                    ),
                )
                for w13, w2 in self.weights
            ]

    def _initialize_mok(self) -> None:
        """Retain the native fused shared expert and published launch settings."""
        from mok import functional, ops

        self.mok = functional
        c, s = self.case, self.case.shape
        self.mok_settings = mok_memory_settings(c)
        self.config = functional.MoKConfig(
            fwd_num_comm_sms=24 if c.precision == "bf16" else 36,
            bwd_num_comm_sms=28 if c.precision == "bf16" else 36,
            minibatch_size=self.mok_settings["minibatch_size"],
            macrobatch_size=self.mok_settings["macrobatch_size"],
            schedule_capacity_multiplier=self.mok_settings[
                "schedule_capacity_multiplier"
            ],
        )
        self.workspace = functional.get_workspace(
            self.config,
            dist.group.WORLD,
            device=self.device,
            num_local_tokens=s.tokens,
            hidden_size=s.hidden,
            topk=s.top_k,
        )
        self.native_weights = []
        self.master_weights = []
        for w13, w2 in self.weights:
            gate = w13[:, : s.intermediate].contiguous()
            up = w13[:, s.intermediate :].contiguous()
            weights = (gate, up, w2)
            self.master_weights.append(weights)
            if c.precision == "mxfp8":
                weights = tuple(ops.mxfp8_quantize(w, True, True) for w in weights)
            self.native_weights.append(weights)
        self.weights.clear()

    def reference_weights(self, layer: int) -> tuple:
        """Expose BF16 gate/up/down views without retaining duplicate master layouts."""
        if self.case.backend == "mok":
            return self.master_weights[layer]
        w13, w2 = self.weights[layer]
        f = self.case.shape.intermediate
        return w13[:, :f], w13[:, f:], w2

    def forward(self) -> torch.Tensor:
        """Run all layers, enqueueing shared work before routed work per layer."""
        self.frames = []
        caller = torch.cuda.current_stream(self.device)
        with torch.set_grad_enabled(self.autograd):
            for i in range(self.case.layers):
                x = self.inputs[i]
                if self.case.backend == "mok":
                    schedule = self.mok.build_schedule(
                        self.workspace,
                        self.config,
                        self.ids[i],
                        num_local_experts=self.case.shape.experts // self.case.ep,
                    )
                    weights = self.native_weights[i]
                    if self.case.precision == "mxfp8":
                        weights = tuple(w[:2] for w in weights)
                    y, state = self.mok.forward(
                        self.config,
                        self.workspace,
                        schedule,
                        x,
                        self.scores[i],
                        *self.shared_weights[i],
                        *weights,
                    )
                    self.frames.append((x, y, schedule, state))
                else:
                    shared_y = None
                    if self.shared_stream is not None:
                        self.shared_stream.wait_stream(caller)
                        with (
                            torch.cuda.stream(self.shared_stream),
                            torch.profiler.record_function(f"shared/forward/layer{i}"),
                            _graph_kernel_scope(self.case, f"shared/forward/layer{i}"),
                        ):
                            shared_y = dense_expert(x, self.shared_weights[i])
                    self.routed_stream.wait_stream(caller)
                    with (
                        torch.cuda.stream(self.routed_stream),
                        torch.profiler.record_function(f"routed/forward/layer{i}"),
                        _graph_kernel_scope(self.case, f"routed/forward/layer{i}"),
                    ):
                        y = dist_moe.routed_experts(
                            x,
                            self.ids[i],
                            self.scores[i],
                            *self.prepared[i],
                            self.context,
                        )
                    caller.wait_stream(self.routed_stream)
                    if (
                        self.shared_stream is not None
                        and self.shared_stream is not self.routed_stream
                    ):
                        caller.wait_stream(self.shared_stream)
                    self.frames.append((x, y, shared_y))
                    if shared_y is not None:
                        y = (y.float() + shared_y.float()).to(torch.bfloat16)
        self.output_TD = y
        return y

    def backward(self, *, retain_graph: bool = False) -> list[tuple]:
        """Consume every retained layer in reverse, including shared gradients."""
        self.gradients = []
        caller = torch.cuda.current_stream(self.device)
        for i in reversed(range(self.case.layers)):
            frame = self.frames[i]
            dy = self.output_gradients[i]
            if self.case.backend == "mok":
                x, _, schedule, state = frame
                weights = self.native_weights[i]
                if self.case.precision == "mxfp8":
                    weights = (*weights[:2], weights[2][2:])
                grads = self.mok.backward(
                    self.config,
                    self.workspace,
                    schedule,
                    state,
                    dy,
                    x,
                    self.scores[i],
                    *self.shared_weights[i],
                    *weights,
                )
            else:
                x, routed_y, shared_y = frame
                shared_grads = ()
                if shared_y is not None:
                    self.shared_stream.wait_stream(caller)
                    with (
                        torch.cuda.stream(self.shared_stream),
                        torch.profiler.record_function(f"shared/backward/layer{i}"),
                        _graph_kernel_scope(self.case, f"shared/backward/layer{i}"),
                    ):
                        shared_grads = torch.autograd.grad(
                            shared_y,
                            (x, *self.shared_weights[i]),
                            dy,
                            retain_graph=retain_graph,
                        )
                self.routed_stream.wait_stream(caller)
                with (
                    torch.cuda.stream(self.routed_stream),
                    torch.profiler.record_function(f"routed/backward/layer{i}"),
                    _graph_kernel_scope(self.case, f"routed/backward/layer{i}"),
                ):
                    dx, dscores, dw13, dw2 = torch.autograd.grad(
                        routed_y,
                        (x, self.scores[i], *self.weights[i]),
                        dy,
                        retain_graph=retain_graph,
                    )
                caller.wait_stream(self.routed_stream)
                if shared_y is not None:
                    if self.shared_stream is not self.routed_stream:
                        caller.wait_stream(self.shared_stream)
                    dx = (dx.float() + shared_grads[0].float()).to(torch.bfloat16)
                f = self.case.shape.intermediate
                grads = (dx, dscores, dw13[:, :f], dw13[:, f:], dw2, *shared_grads[1:])
            self.gradients.append(grads)
        return self.gradients

    def step(self) -> None:
        """Run one complete forward and optional backward iteration."""
        self.forward()
        if self.training:
            self.backward()

    def memory_diagnostic(self) -> dict:
        """Observe full-save lifetimes and native capacity-induced replay."""
        observer = (
            RecomputeObserver()
            if self.case.backend == "dist_moe" and self.training
            else nullcontext()
        )
        torch.cuda.reset_peak_memory_stats(self.device)
        with observer:
            self.forward()
        torch.cuda.synchronize(self.device)
        forward_stats = dict(torch.cuda.memory_stats(self.device))
        result = {
            "after_all_forwards": forward_stats,
            "allocator_backend": torch.cuda.memory.get_allocator_backend(),
            "graph_pool_stats": {
                "state": "Not Available",
                "reason": "memory_stats reports device totals; graph pool IDs are recorded separately",
            },
        }
        if self.case.backend == "dist_moe":
            if self.training and len(observer.flags) != self.case.layers:
                raise RuntimeError(
                    "diagnostic did not observe every layer's forward planner"
                )
            flags = [bool(x.item()) for x in observer.flags] if self.training else []
            result["capacity_recompute_layers"] = flags
            if not self.training:
                result["recompute_diagnostic"] = (
                    "Not Applicable: inference retains no backward state"
                )
            buffer = self.context.activation_buffer
            result["arena"] = {
                "activation_slot_bytes": buffer.activation_slot_bytes,
                "buffer_offsets": buffer.buffer_offsets.cpu().tolist(),
                "peak_min_free_space": buffer.peak_min_free_space.cpu().tolist(),
                "saved_activation_bytes_per_rank": buffer.saved_activation_bytes_per_rank.cpu().tolist(),
                "memory_plan": dataclasses.asdict(self.plan),
            }
        else:
            rows = [int(frame[2].num_tokens.item()) for frame in self.frames]
            predicted = [
                layer[self.rank] for layer in self.mok_settings["padded_receive_rows"]
            ]
            if rows != predicted:
                raise RuntimeError(
                    f"native schedule rows {rows} differ from routing prediction {predicted}"
                )
            result["native_memory_settings"] = self.mok_settings
            result["observed_schedule_rows"] = rows
            result["source_policy_replay_rows"] = [
                max(0, n - self.config.macrobatch_size) for n in rows
            ]
        if self.training:
            self.backward()
        torch.cuda.synchronize(self.device)
        result["after_backward"] = dict(torch.cuda.memory_stats(self.device))
        result["free_total_bytes"] = list(torch.cuda.mem_get_info(self.device))
        return result

    def close(self) -> None:
        """Retire tensors/context references before releasing unused cache blocks."""
        torch.cuda.synchronize(self.device)
        self.frames.clear()
        self.gradients.clear()
        if self.context is not None:
            self.context.close()
            self.context = None
        if self.workspace is not None:
            self.mok.clear_workspace_cache()
            self.workspace = None
        for name in (
            "prepared",
            "native_weights",
            "master_weights",
            "weights",
            "shared_weights",
            "ids",
            "scores",
            "inputs",
            "output_gradients",
            "output_TD",
        ):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        torch.cuda.empty_cache()
