# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Portable workload specifications for routed-expert benchmarks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

PACKING_PROFILES = {
    "A": (7.5, 2.5, 4.5, 0.6, 1.8, 0.9, 2.2, 0.7, 1.6, 0.8),
    "B": (0.7, 5.8, 1.2, 0.6, 3.6, 1.0, 7.2, 0.8, 2.3, 0.7),
}


@dataclasses.dataclass(frozen=True)
class Shape:
    """Logical expert dimensions, independent of precision or allocation.

    Args:
        name: Published workload identifier.
        tokens: Input tokens per expert-parallel rank.
        hidden: Model hidden width.
        intermediate: Width of each gate/up projection.
        experts: Global routed-expert count.
        top_k: Distinct experts selected per token.
    """

    name: str
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    top_k: int

    def __post_init__(self) -> None:
        """Reject invalid logical dimensions before allocating device memory."""
        for field in dataclasses.fields(self):
            if field.name == "name":
                continue
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer: {value!r}")
        if self.top_k > self.experts:
            raise ValueError("top_k cannot exceed the global expert count")
        if not self.name or any(c in self.name for c in "/\\"):
            raise ValueError("shape name must be a nonempty path component")


@dataclasses.dataclass(frozen=True)
class Case:
    """One workload/pipeline configuration with deterministic routing.

    Args:
        shape: Logical routed-expert workload.
        mode: Training or inference.
        precision: bf16, mxfp8, or nvfp4.
        pipeline: Standard BF16, staged, or Mega launch policy.
        ep: Number of active expert-parallel ranks.
        category: single_layer, width, packing, or smoke.
        imbalance: Target maximum-rank receive imbalance for single layers.
        profile: Tracked-rank load profile for the ten-layer experiment.
        seed: Shared input/route seed for all backend implementations.
        shared_experts: Zero for isolated measurements, one for comparisons.
        backend: Public Dist-MoE or the optional native MoK provider.
        num_layers: Optional independent layer count for focused smoke checks.
            Packing defaults to its profile length; single-layer/width uses one.
        shared_expert_schedule: Concurrent streams or shared-before-routed
            serialization for Dist-MoE comparisons.
        fast_math: Enable the MXFP8 approximate sigmoid policy.
        execution_mode: Optional eager or CUDA-graph override for matched runs.
    """

    shape: Shape
    mode: str
    precision: str
    pipeline: str
    ep: int
    category: str
    imbalance: float = 1.0
    profile: str | None = None
    seed: int = 17
    shared_experts: int = 0
    backend: str = "dist_moe"
    num_layers: int | None = None
    shared_expert_schedule: str = "concurrent"
    fast_math: bool = False
    execution_mode: str | None = None

    def __post_init__(self) -> None:  # noqa: C901
        """Check mode, precision, topology, and per-token routing feasibility."""
        if self.mode not in ("training", "inference"):
            raise ValueError(f"unknown mode: {self.mode}")
        if self.precision not in ("bf16", "mxfp8", "nvfp4"):
            raise ValueError(f"unknown precision: {self.precision}")
        if self.precision == "nvfp4" and self.mode != "inference":
            raise ValueError("this precision is available only in inference")
        if self.backend not in ("dist_moe", "mok"):
            raise ValueError(f"unknown backend: {self.backend}")
        if type(self.shared_experts) is not int or self.shared_experts not in (0, 1):
            raise ValueError("shared_experts must be zero or one")
        self._validate_execution_policy()
        if self.backend == "mok" and (
            not self.shared_experts or self.precision == "nvfp4"
        ):
            raise ValueError(
                "native MoK requires a shared expert and supports bf16/mxfp8"
            )
        expected = (
            ("native",)
            if self.backend == "mok"
            else (("standard",) if self.precision == "bf16" else ("staged", "mega"))
        )
        if self.pipeline not in expected:
            raise ValueError(f"{self.precision} pipeline must be one of {expected}")
        if type(self.ep) is not int or self.ep < 1 or self.shape.experts % self.ep:
            raise ValueError("EP must be positive and divide the global expert count")
        if self.backend == "mok" and self.ep not in (1, 4, 8, 16, 32, 64):
            raise ValueError("native MoK supports EP1/4/8/16/32/64")
        if self.num_layers is not None and (
            type(self.num_layers) is not int or self.num_layers <= 0
        ):
            raise ValueError("num_layers must be a positive integer")
        if self.category not in ("single_layer", "width", "packing", "smoke"):
            raise ValueError(f"unknown category: {self.category}")
        if not math.isfinite(self.imbalance) or self.imbalance < 1:
            raise ValueError("maximum-rank imbalance must be finite and at least one")
        peak = self.imbalance
        if self.profile is not None:
            if self.category != "packing" or self.profile not in PACKING_PROFILES:
                raise ValueError("packing profile must be A or B in a packing case")
            peak = max(PACKING_PROFILES[self.profile])
        elif self.category == "packing":
            raise ValueError("packing requires an explicit profile")
        if self.profile and self.layers != len(PACKING_PROFILES[self.profile]):
            raise ValueError("layer count must match the packing profile")
        if self.category in ("single_layer", "width") and self.layers != 1:
            raise ValueError("single-layer and width cases require one layer")
        if peak > self.max_imbalance:
            raise ValueError(
                f"required receive factor {peak:g} exceeds topology bound "
                f"{self.max_imbalance:g} for EP={self.ep}, E={self.shape.experts}, "
                f"K={self.shape.top_k}"
            )

    def _validate_execution_policy(self) -> None:
        """Reject scheduling and math policies unsupported by this provider."""
        if self.execution_mode not in (None, "eager", "cuda_graph"):
            raise ValueError("execution_mode must be eager or cuda_graph")
        if self.shared_expert_schedule not in ("concurrent", "sequential"):
            raise ValueError("shared_expert_schedule must be concurrent or sequential")
        if self.shared_expert_schedule == "sequential" and (
            self.backend != "dist_moe" or not self.shared_experts
        ):
            raise ValueError("sequential shared scheduling requires shared Dist-MoE")
        if type(self.fast_math) is not bool:
            raise ValueError("fast_math must be a bool")
        if self.fast_math and (self.precision != "mxfp8" or self.backend != "dist_moe"):
            raise ValueError("fast_math requires MXFP8 Dist-MoE")

    @property
    def max_imbalance(self) -> float:
        """Return the rank-load bound imposed by distinct experts per token."""
        local_experts = self.shape.experts // self.ep
        return self.ep * min(self.shape.top_k, local_experts) / self.shape.top_k

    @property
    def layers(self) -> int:
        """Return the number of routed layers whose activations coexist."""
        return (
            self.num_layers
            if self.num_layers is not None
            else (10 if self.profile is not None else 1)
        )

    @property
    def identifier(self) -> str:
        """Return a stable case ID including pipeline, profile, and routing seed."""
        digest = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode())
        route = self.profile or f"i{self.imbalance:g}"
        return (
            f"{self.shape.name}-{self.mode}-{self.precision}-{self.pipeline}-"
            f"ep{self.ep}-{self.category}-{route}-F{self.shape.intermediate}-"
            f"{digest.hexdigest()[:10]}"
        )

    def to_dict(self) -> dict:
        """Serialize the complete logical configuration without runtime objects."""
        values = dataclasses.asdict(self)
        # Preserve existing concurrent case identities and serialized manifests.
        if self.shared_expert_schedule == "concurrent":
            del values["shared_expert_schedule"]
        if not self.fast_math:
            del values["fast_math"]
        if self.execution_mode is None:
            del values["execution_mode"]
        return values

    @classmethod
    def from_dict(cls, values: dict) -> Case:
        """Read a serialized case while retaining constructor validation."""
        return cls(**{**values, "shape": Shape(**values["shape"])})

    def useful_flops(self, *, backward: bool = False) -> int:
        """Count expert GEMM FLOPs per GPU, excluding padding and replay work."""
        s = self.shape
        return (
            (12 if backward else 6)
            * s.tokens
            * (s.top_k + self.shared_experts)
            * s.hidden
            * s.intermediate
            * self.layers
        )

    @property
    def scope(self) -> str:
        """Name the workload so isolated and shared comparison rows never mix."""
        return "comparison" if self.shared_experts else "isolated"

    @property
    def timing_mode(self) -> str:
        """Resolve an explicit execution mode or the existing scope default."""
        return self.execution_mode or ("eager" if self.shared_experts else "cuda_graph")


def load_shapes(path: str | Path) -> tuple[list[Shape], dict]:
    """Read portable published shapes and their source provenance."""
    data = json.loads(Path(path).read_text())
    return [Shape(**shape) for shape in data["shapes"]], data.get("provenance", {})


def make_cases(
    shapes: list[Shape],
    ep: int,
    mode: str,
    category: str,
    *,
    backend: str = "dist_moe",
    shared_experts: int = 0,
) -> list[Case]:
    """Expand one category/geometry, including local native-width controls.

    MXFP8 always emits both pipelines. Width controls remain in their own job
    so an allocation's sensitivity results have a reference on the same GPUs.
    """
    precisions = (
        ("bf16", "mxfp8")
        if mode == "training" or shared_experts
        else ("bf16", "mxfp8", "nvfp4")
    )
    cases = []
    for shape in shapes:
        for precision in precisions:
            pipelines = (
                ("native",)
                if backend == "mok"
                else (
                    ("standard",)
                    if precision == "bf16"
                    else (("staged", "mega") if precision == "mxfp8" else ("mega",))
                )
            )
            for pipeline in pipelines:
                if category == "packing":
                    if mode != "training":
                        raise ValueError("packing is a training category")
                    cases.extend(
                        Case(
                            shape,
                            mode,
                            precision,
                            pipeline,
                            ep,
                            category,
                            profile=p,
                            backend=backend,
                            shared_experts=shared_experts,
                        )
                        for p in PACKING_PROFILES
                    )
                elif category == "width":
                    for factor in (0.5, 1.0, 2.0):
                        width = shape.intermediate * factor
                        if not width.is_integer():
                            raise ValueError(
                                "width multiplier must produce an integer width"
                            )
                        cases.append(
                            Case(
                                dataclasses.replace(shape, intermediate=int(width)),
                                mode,
                                precision,
                                pipeline,
                                ep,
                                category,
                                backend=backend,
                                shared_experts=shared_experts,
                            )
                        )
                else:
                    for factor in (1.0,) if category == "smoke" else (1.0, 4.0, 8.0):
                        cases.append(
                            Case(
                                shape,
                                mode,
                                precision,
                                pipeline,
                                ep,
                                category,
                                factor,
                                backend=backend,
                                shared_experts=shared_experts,
                            )
                        )
    return cases


def compilation_key(case: Case) -> tuple:
    """Group cases sharing static shapes and precision, with routing still live."""
    s = case.shape
    return (
        case.backend,
        case.scope,
        case.timing_mode,
        case.mode,
        case.precision,
        case.pipeline,
        case.ep,
        s.tokens,
        s.hidden,
        s.intermediate,
        s.experts,
        s.top_k,
        case.layers,
        case.shared_expert_schedule,
        case.fast_math,
    )


def campaign_jobs(
    shapes: list[Shape], *, smoke: bool = False, scope: str | None = None
) -> dict[str, list[Case]]:
    """Expand cases, grouping selected scopes into one allocation per geometry.

    Args:
        shapes: Published logical workload shapes.
        smoke: Select the two smoke geometries instead of the full suite.
        scope: Isolated or comparison runtime phase; None retains category jobs.

    Returns:
        Explicit jobs with every original case and width control preserved.
    """
    if scope not in (None, "isolated", "comparison"):
        raise ValueError("scope must be isolated or comparison")
    specs = (
        (("smoke", 32, "training"), ("smoke", 8, "inference"))
        if smoke
        else (
            ("single_layer", 8, "inference"),
            ("single_layer", 16, "inference"),
            ("single_layer", 32, "training"),
            ("single_layer", 64, "training"),
            ("width", 16, "inference"),
            ("width", 64, "training"),
            ("packing", 32, "training"),
            ("packing", 64, "training"),
        )
    )
    jobs = {}
    for category, ep, mode in specs:
        cases = []
        if category != "packing":
            cases.extend(make_cases(shapes, ep, mode, category))
        cases.extend(make_cases(shapes, ep, mode, category, shared_experts=1))
        cases.extend(
            make_cases(shapes, ep, mode, category, backend="mok", shared_experts=1)
        )
        if scope is None:
            jobs[f"{category}-{mode}-ep{ep}"] = cases
        else:
            selected = [case for case in cases if case.scope == scope]
            if selected:
                name = f"{'smoke-' if smoke else ''}{scope}-{mode}-ep{ep}"
                jobs.setdefault(name, []).extend(selected)
    return jobs
