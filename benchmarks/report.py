# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Read committed artifacts and report only cases with complete performance gates."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from .config import Case
from .metrics import write_json


def _gb_s(rate: dict) -> float:
    """Convert the observed NVML rate unit to decimal GB/s without assuming it."""
    multiplier = {"MiB/s": 1024**2, "MB/s": 1e6, "GiB/s": 1024**3, "GB/s": 1e9}
    if rate["unit"] not in multiplier:
        raise ValueError(f"unrecognized NVLink rate unit: {rate['unit']}")
    return rate["value"] * multiplier[rate["unit"]] / 1e9


def collect(manifest: dict, campaign: Path) -> list[dict]:
    """Collect every planned row with explicit missing/failed/validated states."""
    rows = []
    for job_name, job in manifest["jobs"].items():
        for identifier, definition in job["cases"].items():
            case = Case.from_dict(definition)
            if identifier != case.identifier:
                raise ValueError(
                    f"manifest case ID does not match its configuration: {identifier}"
                )
            directory = (
                campaign / job_name / "cases" / case.backend / identifier / "attempt-0"
            )
            status_path = directory / "status.json"
            state = (
                json.loads(status_path.read_text())["state"]
                if status_path.exists()
                else "pending"
            )
            row = {
                "job": job_name,
                "case_id": identifier,
                "backend": case.backend,
                "scope": case.scope,
                "mode": case.mode,
                "precision": case.precision,
                "pipeline": case.pipeline,
                "ep": case.ep,
                "intermediate": case.shape.intermediate,
                "imbalance": case.imbalance,
                "profile": case.profile,
                "layers": case.layers,
                "state": state,
                "artifacts": str(directory),
            }
            if state in ("complete", "functional_pass"):
                metrics = [
                    json.loads((directory / f"rank-{r}" / "metrics.json").read_text())
                    for r in range(case.ep)
                ]
                if state == "complete":
                    if any(
                        m["effective_hbm_bandwidth"].get("state") != "Passed"
                        or not m["nvlink_bandwidth"].get("tx")
                        or not m["trace"].get("gpu_kernels")
                        or not m["latency"]
                        or m["numerics"] != "Passed"
                        for m in metrics
                    ):
                        raise ValueError(
                            f"completed case contradicts rank evidence: {identifier}"
                        )
                    latency = metrics[0]["latency"]
                    row.update(
                        forward_ms=latency["forward"]["median_ms"],
                        combined_ms=latency["combined"]["median_ms"],
                        tflops_per_gpu=latency["combined"]["useful_tflops_per_gpu"],
                        tflops_total=latency["combined"]["useful_tflops_total"],
                        nvlink_tx_mean_gb_s=statistics.mean(
                            _gb_s(m["nvlink_bandwidth"]["tx"]) for m in metrics
                        ),
                        effective_hbm_mean_gb_s=statistics.mean(
                            m["effective_hbm_bandwidth"]["phases"]["combined"]["gb_s"]
                            for m in metrics
                        ),
                    )
                    if case.mode == "training":
                        row["backward_ms"] = latency["backward"]["median_ms"]
                if case.layers > 1:
                    memory = [
                        json.loads(
                            (
                                directory / f"rank-{r}" / "allocator_stats.json"
                            ).read_text()
                        )
                        for r in range(case.ep)
                    ]
                    row["max_rank_peak_allocated_bytes"] = max(
                        m["after_backward"]["allocated_bytes.all.peak"] for m in memory
                    )
                    row["max_rank_peak_reserved_bytes"] = max(
                        m["after_backward"]["reserved_bytes.all.peak"] for m in memory
                    )
                    if case.backend == "dist_moe":
                        row["capacity_recompute_layers_by_rank"] = [
                            m["capacity_recompute_layers"] for m in memory
                        ]
                    else:
                        row["source_policy_replay_rows_by_rank"] = [
                            m["source_policy_replay_rows"] for m in memory
                        ]
            rows.append(row)
    return rows


def _timing_windows(campaign: Path, job: str, row: dict) -> list[float]:
    """Read validated maximum-rank timing windows for one explicit job/case."""
    path = (
        campaign
        / job
        / "cases"
        / row["backend"]
        / row["case_id"]
        / "attempt-0/rank-0/metrics.json"
    )
    return json.loads(path.read_text())["latency"]["combined"]["max_rank_window_ms"]


def comparisons(rows: list[dict], manifest: dict, campaign: Path) -> list[dict]:
    """Pair shared workloads and name the fastest fully validated Dist pipeline.

    Window ranges describe the observed five-window variation, not confidence
    intervals. Overlapping ranges remain inconclusive even when medians differ.
    """
    groups = {}
    for row in rows:
        if row["scope"] != "comparison":
            continue
        definition = manifest["jobs"][row["job"]]["cases"][row["case_id"]]
        key = json.dumps(
            {
                k: v
                for k, v in definition.items()
                if k not in ("pipeline", "backend", "fast_math")
            },
            sort_keys=True,
        )
        groups.setdefault((row["job"], key), []).append(row)
    result = []
    for (job, _), candidates in groups.items():
        native = [
            r for r in candidates if r["backend"] == "mok" and r["state"] == "complete"
        ]
        dist_rows = [
            r
            for r in candidates
            if r["backend"] == "dist_moe" and r["state"] == "complete"
        ]
        record = {
            "job": job,
            "pipeline_rows": [
                {
                    "case_id": r["case_id"],
                    "backend": r["backend"],
                    "pipeline": r["pipeline"],
                    "state": r["state"],
                }
                for r in candidates
            ],
            "state": "pending",
        }
        if native and dist_rows:
            best = min(dist_rows, key=lambda r: r["combined_ms"])
            baseline = native[0]
            record.update(
                state="measured",
                best_valid_dist_pipeline=best["pipeline"],
                dist_case=best["case_id"],
                mok_case=baseline["case_id"],
                median_speedup=baseline["combined_ms"] / best["combined_ms"],
            )

            dist_windows = _timing_windows(campaign, job, best)
            mok_windows = _timing_windows(campaign, job, baseline)
            record["observed_speedup_range"] = [
                min(mok_windows) / max(dist_windows),
                max(mok_windows) / min(dist_windows),
            ]
            record["interpretation"] = (
                "Dist-MoE faster across all measured windows"
                if max(dist_windows) < min(mok_windows)
                else "MoK faster across all measured windows"
                if max(mok_windows) < min(dist_windows)
                else "inconclusive: timing window ranges overlap"
            )
        result.append(record)
    return result


def main() -> None:
    """Write a machine-readable report and a compact table of validated rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = collect(manifest, args.campaign)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "results.json", rows)
    write_json(
        args.output / "comparisons.json", comparisons(rows, manifest, args.campaign)
    )
    columns = (
        "job",
        "case_id",
        "backend",
        "scope",
        "precision",
        "pipeline",
        "ep",
        "intermediate",
        "imbalance",
        "profile",
        "layers",
        "state",
        "forward_ms",
        "backward_ms",
        "combined_ms",
        "tflops_per_gpu",
        "tflops_total",
        "nvlink_tx_mean_gb_s",
        "effective_hbm_mean_gb_s",
        "artifacts",
    )
    with (args.output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    passed = [row for row in rows if row["state"] == "complete"]
    lines = [
        "# Distributed expert benchmark results",
        "",
        f"Validated performance rows: {len(passed)} / {len(rows)}. Other rows retain their explicit state in results.json.",
        "",
        "Effective HBM uses analytical useful bytes divided by measured latency. NVLink uses measured hardware rates.",
        "",
        "| Job | Backend / pipeline | Combined ms | TFLOP/s per GPU | NVLink TX GB/s per GPU | Effective HBM GB/s per GPU |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in passed:
        lines.append(
            f"| {row['job']} | {row['backend']} / {row['pipeline']} | {row['combined_ms']:.3f} | {row['tflops_per_gpu']:.2f} | {row['nvlink_tx_mean_gb_s']:.2f} | {row['effective_hbm_mean_gb_s']:.2f} |"
        )
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
