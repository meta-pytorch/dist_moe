# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""CPU invariants for benchmark routing, workload scopes and artifacts."""

import dataclasses
import gzip
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from benchmarks.config import campaign_jobs, Case, PACKING_PROFILES, Shape
from benchmarks.metrics import (
    effective_hbm_bandwidth,
    summarize_windows,
    trace_summary,
    write_json,
)
from benchmarks.report import collect, comparisons
from benchmarks.routing import (
    expert_quotas,
    local_routes,
    mok_memory_settings,
    rank_quotas,
)
from benchmarks.run import _runtime_identity, render_campaign


class BenchmarkTest(unittest.TestCase):
    """Independently verify generated traffic and measurement configuration."""

    def setUp(self) -> None:
        """Use the published geometry with few tokens for exact CPU histograms."""
        self.shape = Shape("published", 7, 7168, 3072, 384, 6)

    def test_quotas_and_distinct_token_experts(self) -> None:
        """Actual routes conserve all expert/rank quotas at each target scale."""
        for ep in (8, 16, 32, 64):
            for factor in (1.0, 4.0, 8.0):
                with self.subTest(ep=ep, factor=factor):
                    case = Case(
                        self.shape,
                        "training",
                        "bf16",
                        "standard",
                        ep,
                        "single_layer",
                        factor,
                    )
                    counts = Counter()
                    for rank in range(ep):
                        routes = local_routes(case, rank)
                        self.assertEqual(len(routes), self.shape.tokens)
                        for row in routes:
                            self.assertEqual(len(set(row)), self.shape.top_k)
                            counts.update(row)
                    expected = expert_quotas(case)
                    self.assertEqual(
                        [counts[e] for e in range(self.shape.experts)], expected
                    )
                    ranks, _ = rank_quotas(case)
                    actual = [
                        sum(
                            counts[e]
                            for e in range(
                                r * self.shape.experts // ep,
                                (r + 1) * self.shape.experts // ep,
                            )
                        )
                        for r in range(ep)
                    ]
                    self.assertEqual(actual, ranks)
                    self.assertAlmostEqual(
                        max(actual) / (self.shape.tokens * self.shape.top_k), factor
                    )

    def test_topology_bound_and_scope_validation(self) -> None:
        """Reject impossible routing and unsupported native comparison cells."""
        shape = dataclasses.replace(self.shape, experts=8)
        with self.assertRaisesRegex(ValueError, "topology bound"):
            Case(shape, "training", "bf16", "standard", 8, "single_layer", 4)
        with self.assertRaisesRegex(ValueError, "requires a shared"):
            Case(
                self.shape,
                "training",
                "bf16",
                "native",
                8,
                "single_layer",
                backend="mok",
            )
        with self.assertRaisesRegex(ValueError, "supports bf16/mxfp8"):
            Case(
                self.shape,
                "inference",
                "nvfp4",
                "native",
                8,
                "single_layer",
                backend="mok",
                shared_experts=1,
            )

    def test_campaign_counts_and_scope_keys(self) -> None:
        """Freeze the execution counts and prevent shared/isolated row reuse."""
        suite = campaign_jobs([self.shape])
        smoke = campaign_jobs([self.shape], smoke=True)
        self.assertEqual(len(suite), 8)
        self.assertEqual(sum(map(len, suite.values())), 173)
        self.assertEqual(sum(map(len, smoke.values())), 17)
        for cases in suite.values():
            self.assertEqual(len({c.identifier for c in cases}), len(cases))
            for c in cases:
                self.assertEqual(c.shared_experts, int(c.scope == "comparison"))
                if c.backend == "mok":
                    self.assertNotEqual(c.precision, "nvfp4")

    def test_graph_campaign_matches_provider_scopes_and_preserves_defaults(
        self,
    ) -> None:
        """All original comparison cases support explicit graph execution."""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "shape.json"
            write_json(path, {"shapes": [dataclasses.asdict(self.shape)]})
            graph = render_campaign(
                path,
                Path(root) / "graph.json",
                scope="comparison",
                execution_mode="cuda_graph",
                mxfp8_fast_math=True,
            )
        counts = {
            name: len(job["cases"])
            for name, job in graph["jobs"].items()
            if not name.startswith("smoke-")
        }
        self.assertEqual(sorted(counts.values()), [15, 25, 30, 40])
        for job in graph["jobs"].values():
            for values in job["cases"].values():
                case = Case.from_dict(values)
                self.assertEqual(case.shared_experts, 1)
                self.assertEqual(case.timing_mode, "cuda_graph")
                self.assertEqual(
                    case.fast_math,
                    case.backend == "dist_moe" and case.precision == "mxfp8",
                )
                default = dataclasses.replace(case, execution_mode=None)
                self.assertNotIn("execution_mode", default.to_dict())
                self.assertEqual(default.timing_mode, "eager")
                self.assertNotEqual(default.identifier, case.identifier)
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            Case(
                self.shape,
                "training",
                "bf16",
                "standard",
                32,
                "smoke",
                execution_mode="unknown",
            )

    def test_sequential_campaign_excludes_mok_and_preserves_controls(self) -> None:
        """Serial reruns retain all Dist cases and six packing fit checks."""
        with tempfile.TemporaryDirectory() as root:
            shape_file = Path(root) / "shapes.json"
            write_json(shape_file, {"shapes": [dataclasses.asdict(self.shape)]})
            campaign = render_campaign(
                shape_file,
                Path(root) / "campaign.json",
                scope="comparison",
                backend="dist_moe",
                shared_expert_schedule="sequential",
                mxfp8_fast_math=True,
            )
        expected = {
            "inference-ep8": 9,
            "inference-ep16": 18,
            "training-ep32": 15,
            "training-ep64": 24,
        }
        for suffix, count in expected.items():
            job = campaign["jobs"][f"comparison-{suffix}"]
            self.assertEqual(len(job["cases"]), count)
        fit = campaign["jobs"]["smoke-comparison-training-ep32"]
        self.assertEqual(len(fit["validation_only"]), 6)
        for job in campaign["jobs"].values():
            for values in job["cases"].values():
                case = Case.from_dict(values)
                self.assertEqual(case.backend, "dist_moe")
                self.assertEqual(case.shared_expert_schedule, "sequential")
                self.assertEqual(case.fast_math, case.precision == "mxfp8")
                concurrent = dataclasses.replace(
                    case, shared_expert_schedule="concurrent", fast_math=False
                )
                self.assertNotIn("shared_expert_schedule", concurrent.to_dict())
                self.assertEqual(Case.from_dict(concurrent.to_dict()), concurrent)
                self.assertNotEqual(case.identifier, concurrent.identifier)
        with self.assertRaisesRegex(ValueError, "requires shared Dist-MoE"):
            Case(
                self.shape,
                "training",
                "bf16",
                "standard",
                32,
                "smoke",
                shared_expert_schedule="sequential",
            )

    def test_packing_profile_and_native_rounding(self) -> None:
        """Both variable profiles fit aggregate2.5 while native layers round up."""
        shape = dataclasses.replace(self.shape, tokens=2048)
        for profile in PACKING_PROFILES:
            case = Case(
                shape,
                "training",
                "bf16",
                "native",
                32,
                "packing",
                profile=profile,
                backend="mok",
                shared_experts=1,
            )
            settings = mok_memory_settings(case)
            self.assertEqual(settings["minibatch_size"], 4096)
            self.assertEqual(settings["requested_saved_rows"], 30720)
            self.assertEqual(settings["macrobatch_size"], 32768)
            self.assertAlmostEqual(settings["effective_activation_factor"], 8 / 3)
            ratios = [
                rank_quotas(case, i)[0][0] / (shape.tokens * shape.top_k)
                for i in range(10)
            ]
            self.assertLess(sum(ratios) / 10, 2.5)
            self.assertGreater(max(ratios), settings["effective_activation_factor"])
            self.assertTrue(
                all(
                    max(rows) <= settings["schedule_capacity_rows"]
                    for rows in settings["padded_receive_rows"]
                )
            )

    def test_tflops_match_work_and_window(self) -> None:
        """Forward/backward GEMM counts include shared work only in comparisons."""
        isolated = Case(self.shape, "training", "bf16", "standard", 32, "single_layer")
        shared = dataclasses.replace(isolated, shared_experts=1)
        self.assertEqual(shared.useful_flops() * 6, isolated.useful_flops() * 7)
        result = summarize_windows(
            shared, [{"forward_ms": 1, "backward_ms": 2, "combined_ms": 3}]
        )
        self.assertEqual(
            result["combined"]["useful_gemm_flops_per_gpu"], 18 * 7 * 7 * 7168 * 3072
        )
        self.assertEqual(
            result["forward"]["useful_tflops_total"],
            32 * result["forward"]["useful_tflops_per_gpu"],
        )

    def test_trace_rejects_empty_device_collection(self) -> None:
        """A well-formed compressed trace cannot pass without actual GPU events."""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump(
                    {
                        "traceEvents": [
                            {"name": "benchmark/forward", "cat": "user_annotation"}
                        ]
                    },
                    stream,
                )
            with self.assertRaisesRegex(RuntimeError, "required device/phase evidence"):
                trace_summary(path, shared=False, graph=False)
            write_json(Path(root) / "status.json", {"state": "pending"})
            self.assertEqual(
                json.loads((Path(root) / "status.json").read_text())["state"], "pending"
            )

    def test_graph_trace_preserves_shared_and_routed_branch_identity(self) -> None:
        """Graph nodes carry branch identity without per-kernel CPU launches."""
        events = [
            {"name": "benchmark/forward", "cat": "user_annotation"},
            {
                "name": "bf16_gemm",
                "cat": "kernel",
                "ts": 10,
                "dur": 3,
                "args": {
                    "device": 0,
                    "annotation": "benchmark/forward/shared/forward/layer0",
                },
            },
            {
                "name": "dist_grouped_gemm",
                "cat": "kernel",
                "ts": 11,
                "dur": 3,
                "args": {
                    "device": 0,
                    "annotation": "benchmark/forward/routed/forward/layer0",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.json"
            write_json(path, {"traceEvents": events})
            result = trace_summary(path, shared=True, graph=True)
        self.assertEqual(
            result["timeline"]["branch_kernels"], {"shared": 1, "routed": 1}
        )
        self.assertAlmostEqual(result["timeline"]["shared_routed_overlap_ms"], 0.002)

    def test_json_writer_does_not_reuse_retained_temporary_name(self) -> None:
        """A stale temporary object must not prevent subsequent status updates."""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "status.json"
            stale = path.with_name(path.name + f".tmp-{os.getpid()}")
            stale.write_text("retained object")
            original_open = Path.open

            def mount_open(target, mode="r", *args, **kwargs):
                if mode in ("w", "x") and ".tmp-" in target.name and target.exists():
                    raise FileExistsError(f"retained temporary object: {target}")
                return original_open(target, mode, *args, **kwargs)

            with mock.patch.object(Path, "open", mount_open):
                write_json(path, {"phase": 1})
                write_json(path, {"phase": 2})
            self.assertEqual(json.loads(path.read_text()), {"phase": 2})
            self.assertEqual(stale.read_text(), "retained object")

    def test_json_writer_finishes_only_its_empty_partial_create(self) -> None:
        """Complete a committed empty create; preserve a nonempty conflict."""
        original_open = Path.open
        for partial in ("", "existing data"):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "status.json"
                created = []

                def mount_open(
                    target, mode="r", *args, partial=partial, created=created, **kwargs
                ):
                    if mode == "x" and ".tmp-" in target.name:
                        with original_open(target, "w") as stream:
                            stream.write(partial)
                        created.append(target)
                        raise FileExistsError(
                            f"create committed before error: {target}"
                        )
                    return original_open(target, mode, *args, **kwargs)

                with mock.patch.object(Path, "open", mount_open):
                    if partial:
                        with self.assertRaises(FileExistsError):
                            write_json(path, {"phase": "complete"})
                    else:
                        write_json(path, {"phase": "complete"})
                if partial:
                    self.assertFalse(path.exists())
                    self.assertEqual(created[0].read_text(), partial)
                else:
                    self.assertEqual(
                        json.loads(path.read_text()), {"phase": "complete"}
                    )

    def test_trace_identifies_distributed_grouped_gemm_as_fused(self) -> None:
        """Fused inference counts GEMM work while dispatch-only routing cannot."""
        kernel = {
            "cat": "kernel",
            "name": "_cute_dist_grouped_gemm_dispatch_fprop_kernel_cutlass_dispatch_kernel",
            "ts": 10,
            "dur": 3,
            "args": {"device": 0, "annotation": "benchmark/forward"},
        }
        events = [
            {"name": "benchmark/forward", "cat": "user_annotation"},
            kernel,
        ]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.json"
            write_json(path, {"traceEvents": events})
            result = trace_summary(path, shared=False, graph=True)
            self.assertEqual(result["compute_kernels"], 1)
            self.assertEqual(result["communication_or_fused_kernels"], 1)
            self.assertEqual(
                result["timeline"]["kernel_busy_ms"]["fused_compute_communication"],
                0.003,
            )
            self.assertEqual(result["timeline"]["compute_communication_overlap_ms"], 0)
            kernel["name"] = "_triton_decode_dispatch_routing"
            write_json(path, {"traceEvents": events})
            with self.assertRaisesRegex(RuntimeError, "compute=0"):
                trace_summary(path, shared=False, graph=True)

    def test_effective_hbm_counts_active_weights_and_actual_rank_load(self) -> None:
        """Logical bytes follow real routes and ignore native padding/replay."""
        shape = Shape("tiny", 1, 32, 16, 16, 2)
        base = Case(shape, "inference", "mxfp8", "staged", 4, "smoke")
        counts = Counter(
            e for r in range(4) for row in local_routes(base, r) for e in row
        )
        for rank in range(4):
            active = sum(counts[e] > 0 for e in range(rank * 4, (rank + 1) * 4))
            rows = sum(counts[e] for e in range(rank * 4, (rank + 1) * 4))
            result = effective_hbm_bandwidth(base, rank, {"forward": {"median_ms": 1}})
            parts = result["forward_bytes"]
            self.assertEqual(parts["routed_weights"], active * (32 * 33 + 32 * 17))
            self.assertEqual(parts["routed_activations"], rows * 64 * 2)
            self.assertEqual(parts["routing_scores"], 8)
            self.assertEqual(
                result["phases"]["forward"]["gb_s"], sum(parts.values()) / 1e6
            )
        shared = dataclasses.replace(base, shared_experts=1)
        native = dataclasses.replace(shared, backend="mok", pipeline="native")
        phases = {"combined": {"median_ms": 1}}
        self.assertEqual(
            effective_hbm_bandwidth(shared, 0, phases),
            effective_hbm_bandwidth(native, 0, phases),
        )
        trained = effective_hbm_bandwidth(
            dataclasses.replace(shared, mode="training"), 0, phases
        )
        self.assertEqual(
            trained["phases"]["combined"]["logical_bytes_per_iteration"],
            3 * sum(trained["forward_bytes"].values()),
        )
        skewed = dataclasses.replace(base, imbalance=4)
        quiet = effective_hbm_bandwidth(skewed, 3, phases)["forward_bytes"]
        self.assertEqual(quiet["routed_weights"], 0)
        self.assertEqual(quiet["routed_activations"], 0)

    def test_layer_count_is_independent_of_profile(self) -> None:
        """Focused smoke can retain ten layers while reported single layers stay one."""
        case = Case(
            self.shape, "training", "bf16", "standard", 32, "smoke", num_layers=10
        )
        self.assertEqual(case.layers, 10)
        with self.assertRaisesRegex(ValueError, "require one layer"):
            dataclasses.replace(case, category="single_layer")
        with self.assertRaisesRegex(ValueError, "match the packing profile"):
            dataclasses.replace(case, category="packing", profile="A", num_layers=9)

    def test_trace_overlap_uses_gpu_intervals_and_launch_correlations(self) -> None:
        """Non-overlapping CPU branches can launch overlapping GPU work."""
        events = [
            {
                "cat": "user_annotation",
                "name": "shared/forward/layer0",
                "ts": 0,
                "dur": 40,
                "pid": 1,
                "tid": 1,
            },
            {
                "cat": "user_annotation",
                "name": "routed/forward/layer0",
                "ts": 40,
                "dur": 50,
                "pid": 1,
                "tid": 1,
            },
        ]
        for correlation, launch, start, duration, name in (
            (1, 10, 100, 20, "nvjet_sm103"),
            (2, 50, 110, 30, "dispatch_kernel"),
            (3, 60, 130, 20, "grouped_gemm"),
            (4, 70, 200, 50, "dispatch_mlp_swiglu_combiner"),
        ):
            events.extend(
                [
                    {
                        "cat": "cuda_driver",
                        "name": "cuLaunchKernel",
                        "ts": launch,
                        "dur": 1,
                        "pid": 1,
                        "tid": 2 if correlation == 3 else 1,
                        "args": {"correlation": correlation},
                    },
                    {
                        "cat": "kernel",
                        "name": name,
                        "ts": start,
                        "dur": duration,
                        "args": {
                            "correlation": correlation,
                            "device": 0,
                            "stream": 1 if correlation == 1 else 2,
                        },
                    },
                ]
            )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.json"
            write_json(path, {"traceEvents": events})
            timeline = trace_summary(path, shared=True, graph=False)["timeline"]
            self.assertAlmostEqual(timeline["shared_routed_overlap_ms"], 0.01)
            self.assertAlmostEqual(timeline["compute_communication_overlap_ms"], 0.02)
            self.assertAlmostEqual(
                timeline["kernel_busy_ms"]["fused_compute_communication"], 0.05
            )
            self.assertAlmostEqual(timeline["device_activity_gap_ms"], 0.05)
            self.assertEqual(timeline["branch_kernels"], {"shared": 1, "routed": 3})
            self.assertEqual(timeline["cross_thread_branch_kernels"], 1)
            events[2]["args"]["correlation"] = 999
            write_json(path, {"traceEvents": events})
            with self.assertRaisesRegex(RuntimeError, "lacks correlated GPU work"):
                trace_summary(path, shared=True, graph=False)

    def test_report_pairs_scopes_and_excludes_failed_fast_pipeline(self) -> None:
        """Only complete shared rows can win, and overlapping windows stay inconclusive."""
        base = Case(
            self.shape,
            "training",
            "mxfp8",
            "staged",
            8,
            "single_layer",
            shared_experts=1,
        )
        cases = [
            base,
            dataclasses.replace(base, pipeline="mega"),
            dataclasses.replace(base, pipeline="native", backend="mok"),
            dataclasses.replace(base, shared_experts=0),
        ]
        manifest = {
            "jobs": {"test": {"cases": {c.identifier: c.to_dict() for c in cases}}}
        }
        with tempfile.TemporaryDirectory() as root:
            campaign = Path(root)
            pending = collect(manifest, campaign)
            self.assertTrue(all(r["state"] == "pending" for r in pending))
            rows = []
            for case, state, ms, windows in zip(
                cases,
                ("complete", "failed", "complete", "complete"),
                (2, 1, 2.1, 0.5),
                ([1.9, 2.1], [1, 1], [2, 2.2], [0.5, 0.5]),
                strict=True,
            ):
                rows.append(
                    {
                        "job": "test",
                        "case_id": case.identifier,
                        "backend": case.backend,
                        "scope": case.scope,
                        "pipeline": case.pipeline,
                        "state": state,
                        "combined_ms": ms,
                    }
                )
                directory = (
                    campaign
                    / "test/cases"
                    / case.backend
                    / case.identifier
                    / "attempt-0/rank-0"
                )
                write_json(
                    directory / "metrics.json",
                    {"latency": {"combined": {"max_rank_window_ms": windows}}},
                )
            result = comparisons(rows, manifest, campaign)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["best_valid_dist_pipeline"], "staged")
            self.assertAlmostEqual(result[0]["median_speedup"], 1.05)
            self.assertEqual(
                result[0]["interpretation"],
                "inconclusive: timing window ranges overlap",
            )

    def test_report_accepts_effective_hbm_without_dram_counters(self) -> None:
        """Core results require analytical HBM and real NVLink/trace evidence."""
        case = Case(self.shape, "training", "bf16", "standard", 8, "single_layer")
        manifest = {"jobs": {"test": {"cases": {case.identifier: case.to_dict()}}}}
        metric = {
            "numerics": "Passed",
            "trace": {"gpu_kernels": 2},
            "nvlink_bandwidth": {"tx": {"unit": "GB/s", "value": 3}},
            "effective_hbm_bandwidth": {
                "state": "Passed",
                "phases": {"combined": {"gb_s": 7}},
            },
            "latency": summarize_windows(
                case, [{"forward_ms": 1, "backward_ms": 2, "combined_ms": 3}]
            ),
        }
        with tempfile.TemporaryDirectory() as root:
            directory = (
                Path(root) / "test/cases/dist_moe" / case.identifier / "attempt-0"
            )
            write_json(directory / "status.json", {"state": "complete"})
            for rank in range(case.ep):
                write_json(directory / f"rank-{rank}/metrics.json", metric)
            row = collect(manifest, Path(root))[0]
            self.assertEqual(row["state"], "complete")
            self.assertEqual(row["effective_hbm_mean_gb_s"], 7)
            self.assertEqual(row["nvlink_tx_mean_gb_s"], 3)
            self.assertNotIn("hbm_total_mean_gb_s", row)
            metric["nvlink_bandwidth"] = {}
            write_json(directory / "rank-1/metrics.json", metric)
            with self.assertRaisesRegex(ValueError, "contradicts rank evidence"):
                collect(manifest, Path(root))

    def test_worker_rejects_wrong_packaged_interpreter_before_cuda(self) -> None:
        """A copied console-script shebang cannot bypass the configured prefix."""
        with mock.patch.dict(os.environ, {"CONDA_DIR": "/wrong/package/prefix"}):
            with self.assertRaisesRegex(RuntimeError, "outside the configured package"):
                _runtime_identity()

    def test_runtime_phases_preserve_cases_and_separate_smoke_gates(self) -> None:
        """Runtime phases keep all173 cases and exclude packing from isolated smoke."""
        counts = {}
        for scope, expected in (
            ("isolated", [9, 12, 18, 24]),
            ("comparison", [15, 25, 30, 40]),
        ):
            jobs = campaign_jobs([self.shape], scope=scope)
            self.assertEqual(sorted(map(len, jobs.values())), expected)
            self.assertTrue(
                all(c.scope == scope for cases in jobs.values() for c in cases)
            )
            counts[scope] = sum(map(len, jobs.values()))
        self.assertEqual(counts, {"isolated": 63, "comparison": 110})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape_file = root / "shapes.json"
            write_json(shape_file, {"shapes": [dataclasses.asdict(self.shape)]})
            isolated = render_campaign(
                shape_file, root / "isolated.json", scope="isolated"
            )
            comparison = render_campaign(
                shape_file, root / "comparison.json", scope="comparison"
            )
            self.assertEqual(
                sum(len(j["cases"]) for j in isolated["jobs"].values()), 70
            )
            self.assertEqual(
                sum(len(j["cases"]) for j in comparison["jobs"].values()), 130
            )
            self.assertTrue(
                all(not j["validation_only"] for j in isolated["jobs"].values())
            )
            fit = comparison["jobs"]["smoke-comparison-training-ep32"][
                "validation_only"
            ]
            self.assertEqual(len(fit), 10)


if __name__ == "__main__":
    unittest.main()
