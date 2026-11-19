# Frozen benchmark reproduction

This branch preserves the Dist-MoE provider and benchmark protocols measured
in October 2026. It is a study snapshot, not a maintained benchmark framework.
The normal repository branch remains independent.

```text
benchmarks/
  isolated/          63 routed-only configurations
  mok-comparison/    110 native comparison executions
  fb/                internal deployment support, excluded from public export
```

Shared routing, measurement, verification and reporting helpers remain at the
benchmark root. The production wheel contains the provider; run the benchmark modules from
this source checkout. Run commands from the project root after installing the exact
runtime for the selected family.

## Two separate environments

| Family | Shared expert | Runtime |
|---|---|---|
| Isolated | None | [isolated/runtime.json](isolated/runtime.json), CUDA 13.1 |
| MoK comparison | One BF16 expert on both | [mok-comparison/runtime.json](mok-comparison/runtime.json), CUDA 13.0 |

The provider snapshot is
`a9ac3c789c2f51d9be4d86e0ef664d210239e42c`.
The public branch contains that provider's source; its eventual Git commit
also includes these benchmark files. [versions.json](versions.json) records
the measured provider contents, original benchmark source digest and native
MoK revision/build artifact hashes separately.

Use the recorded dependency and compiler/driver composition. The runtime
check requires the original Torch revision, CUDA/NCCL, compiled architecture
list and required distributions. It validates imported Dist-MoE source hashes
and, for comparison, the actual installed MoK files. It accepts the pinned
provider from this checkout or an identical installed package.

The recorded comparison Torch wheel has generic Blackwell architecture
entries; the Dist CuTe kernels target the GB300 device. GPU execution requires
a real device with compute capability 10.3. Do not replace runtime identity
with a mocked capability or assume that a Torch architecture label alone
proves a kernel execution path.

## Run the reproduction matrix

Run from the project root on GB300 GPUs in one NVLink fabric, using the exact
installed environment for the selected family. The commands below do not
provision GPUs or install that environment.

| Term | Meaning |
|---|---|
| `provenance` | Checks the imported runtime versions, provider source hashes and comparison native extension. It does not install or benchmark anything. |
| `isolated` | Routed Dist-MoE experts alone, with no shared expert or MoK. The layer still spans multiple GPUs. |
| `mok-comparison` | Matched graph benchmarks of Dist-MoE and native MoK, with one shared BF16 expert. |
| Manifest | Generated JSON containing experiment settings, job names, exact case IDs and compatible batches. It is configuration, not measured results. |
| Job | A group for one family, training/inference mode and GPU geometry; the name itself does not create a scheduler allocation. |
| Case | One backend/precision/pipeline/width/routing configuration. |
| Batch | Cases that share static compilation settings and can run sequentially in the same Torchrun invocation. |

### Check the environment and generate the manifest

Use the isolated environment for the first pair and the separate comparison
environment for the second pair:

```bash
python -m benchmarks.provenance --scope isolated
python -m benchmarks.isolated.run --render isolated-manifest.json

python -m benchmarks.provenance --scope comparison
python -m benchmarks.mok-comparison.run --render comparison-manifest.json
```

`--scope` selects which recorded environment to check; it does not activate an
environment. `--render` uses CPU only. Generate each manifest once, before GPU
workers start, and make the same contents available on every host. The names
`*-manifest.json` distinguish these catalogs from the published result JSON.
The comparison environment must already contain the pinned native MoK wheel.

The complete required cases, geometry counts and exact case IDs are in:

- [Isolated reproduction matrix](isolated/README.md#required-measurement-matrix): 63 cases across inference EP8/16 and training EP32/64.
- [MoK comparison matrix](mok-comparison/README.md#required-measurement-matrix): 110 backend cases across the same geometries, including ten-layer profiles A/B.

Run every primary job in the selected family's table. The exact ID tables are
split into compatible batches. The prefix `mok_default` names the published
shape; it does not select the MoK backend. Optional smoke/fit jobs validate
setup and are excluded from the 63/110 measured case lists.

### Launch every required case

Reserve the GPU geometry from the family README. The original placement uses
four GPUs per host: EP8/16/32/64 require 2/4/8/16 hosts respectively. Run the
following on every host. Set NODE_RANK to 0 on the first host, 1 on the second,
and so on; keep all other values identical across hosts.

This example runs all twelve isolated EP8 inference cases. For another job,
use the module, manifest, job name and NNODES shown in its family README.
Use a unique RUN_ID and a fresh shared OUTPUT_ROOT for each reproduction run.

```bash
export BENCH_MODULE=benchmarks.isolated.run
export MANIFEST=isolated-manifest.json
export JOB=isolated-inference-ep8
export NNODES=2
export GPUS_PER_NODE=4
export NODE_RANK=0
export RDZV_ENDPOINT=node0:29500
export RUN_ID=dist-moe-reproduction-01
export OUTPUT_ROOT=/shared/benchmarks/reproduction-01
export TORCH_DIST_USE_NCCL2=0
```

Replace node0 with the first host's reachable hostname. OUTPUT_ROOT must name
the same persistent shared storage on all hosts. Torchrun starts one worker
per GPU; its c10d rendezvous coordinates the hosts, and the benchmark uses the
stock NCCL GPU backend. A fresh rendezvous ID is used for every batch.

```bash
python - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

manifest_path = os.environ["MANIFEST"]
manifest = json.loads(Path(manifest_path).read_text())
job_name = os.environ["JOB"]
job = manifest["jobs"][job_name]
nodes = int(os.environ["NNODES"])
gpus = int(os.environ["GPUS_PER_NODE"])
assert job["ep"] == nodes * gpus, "Job EP must equal total GPU workers"

for batch_number, case_ids in enumerate(job["batches"], start=1):
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes", str(nodes), "--nproc-per-node", str(gpus),
        "--node-rank", os.environ["NODE_RANK"],
        "--rdzv-backend", "c10d",
        "--rdzv-endpoint", os.environ["RDZV_ENDPOINT"],
        "--rdzv-id", f"{os.environ['RUN_ID']}-{job_name}-{batch_number}",
        "--max-restarts", "0", "--module", os.environ["BENCH_MODULE"],
        "--manifest", manifest_path, "--job", job_name,
        "--output", str(Path(os.environ["OUTPUT_ROOT"]) / job_name),
    ]
    for case_id in case_ids:
        command.extend(["--case", case_id])
    print(f"Running {job_name} batch {batch_number}/{len(job['batches'])}", flush=True)
    subprocess.run(command, check=True)
PY
```

Run this block with the selected environment's interpreter on every host. It
runs each batch once and every required case within that batch; it stops on
an unsuccessful child. Repeat for the remaining primary jobs in the
family table using their allocated geometry. The internal MAST supervisor
performs this sequencing and child retirement inside one allocation.

To run one case instead, pass the family's module with `--manifest`, `--job`,
`--case <exact-ID-from-the-family-README>` and `--output` to Torchrun. Repeat
`--case` only for IDs in the same numbered batch of the same job.

Each case runs numerical/graph checks, ten warmups, five 30-iteration timing
windows, three profiler iterations and separate memory/NVLink diagnostics.
Comparison cases also run five 30-iteration eager control windows. Output is
under `OUTPUT_ROOT/<job>/cases/<backend>/<case-ID>/attempt-0/rank-<rank>/`;
`metrics.json` contains timings and rates, `allocator_stats.json` contains
memory diagnostics, and `trace.json.gz` contains the real GPU trace.

CPU harness checks use the selected locked runtime:

```bash
python -m unittest discover -s benchmarks/tests -p 'test_*.py'
```

## Workloads and timing

T=2048 input tokens per rank, D=7168, E=384, K=6 and default F=3072. Keeping T
fixed per rank makes EP variation weak scaling. Inference measures forward
workload, not decode latency. Single-layer receive-imbalance targets are 1/4/8;
F sensitivity uses 1536/3072/6144. Exact integer quotas conserve routes and
each token selects distinct experts.

Primary timing uses CUDA graphs. Ten eager warmups precede capture and
changed-input/output-gradient verification with independent cloned references.
Five windows of 30 graph iterations are timed without the profiler. Comparison
cases then time five same-case eager windows of 30 while graph pools remain
resident. Three graph profiler iterations, one memory diagnostic and at least
one second of NVLink sampling are separate. Graph-then-eager order is fixed.

Shared Dist work runs on a separate stream before routed work is enqueued,
with joins before output/input-gradient sums. Capture retains these dependencies
in separate forward/backward graphs sharing one pool. The native MoK path
keeps its fused shared/routed implementation.

Ten-layer comparison profiles use independent inputs/weights/gradients; all
ten forwards precede reverse backward:

```text
A: 7.5,2.5,4.5,0.6,1.8,0.9,2.2,0.7,1.6,0.8  aggregate 2.31
B: 0.7,5.8,1.2,0.6,3.6,1.0,7.2,0.8,2.3,0.7  aggregate 2.39
```

These are tracked-rank load ratios; values below 1 are not a layer maximum.
Dist uses one factor 2.5 activation slot and device scratch for peak load 8.
MoK's native 4096-row minibatch rounds 30720 requested saved rows to 32768,
effective factor 2.667 per layer. Native padding/quantization policies remain
unchanged. This is a retained-activation experiment, not a connected model.

## Metrics and interpretation

Useful GEMM work per GPU on average is `6*T*(K+shared)*D*F` forward, twice
that backward and three times forward for complete training; multiply by
layer count. Rates divide by maximum-rank latency. Forward/backward/combined
rates remain separate. Padding and replay are excluded from useful work.

NVLink TX/RX comes from real NVML counters. Effective HBM is analytical useful
bytes divided by latency, with a coarse 1/2/3 forward/backward/training byte
model; it is not physical DRAM traffic or a cache/replay traffic audit.

Per-rank outputs include actual runtime/provenance, routing, numerical errors,
timing windows, allocator/arena statistics and a parsed real GPU trace.
Hot/quiet ranks also write memory snapshots. Capacity-induced Dist recompute
flags and MoK source-policy replay rows are distinct from ordinary backward
preparation. Total allocator peaks need not be lower for the packed provider.

## Accepted result artifacts

The [versioned public Gist](https://gist.github.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/105d4e5c9867e2752cc4306cd7356430f863f36d) contains the final 63 isolated and 110
comparison records and the consolidated report. It is separate from this
source snapshot. Download files from this revision and verify their bytes
against [SHA256SUMS](https://gist.githubusercontent.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/raw/105d4e5c9867e2752cc4306cd7356430f863f36d/SHA256SUMS)
(`cd79619cd088771b3328546f76948ab8d46f1b0159c34960ab41106812b7d17c`). CSV files use LF line endings.

| File | SHA256 |
|---|---|
| [isolated.csv](https://gist.githubusercontent.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/raw/105d4e5c9867e2752cc4306cd7356430f863f36d/isolated.csv) | `df37d9aa7a6c3d8f2d1825ddc3734319078aa352811a38b2b59a1b71ff4612cb` |
| [isolated.json](https://gist.githubusercontent.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/raw/105d4e5c9867e2752cc4306cd7356430f863f36d/isolated.json) | `0e14cfea690cb6c28af47b7244a6e6c6855f11045496cadd04714c8b37423445` |
| [mok-comparison.csv](https://gist.githubusercontent.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/raw/105d4e5c9867e2752cc4306cd7356430f863f36d/mok-comparison.csv) | `36b4a8d05dc0762158bca3cee6009537ba9189b32649c6c1398e7d73a11cda39` |
| [mok-comparison.json](https://gist.githubusercontent.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/raw/105d4e5c9867e2752cc4306cd7356430f863f36d/mok-comparison.json) | `2fec148c640c1e21f57c9595b6082492bc8cca840a1dc2bdf678bdc4bead2822` |
| [REPORT.md](https://gist.githubusercontent.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/raw/105d4e5c9867e2752cc4306cd7356430f863f36d/REPORT.md) | `33e5f41adb6f446635bc6efecd8ef0bcb8c31769528770c0981d212d2808ecaa` |

New output must be stored separately. A numerical failure or an empty/missing
target trace is not an accepted performance result.
