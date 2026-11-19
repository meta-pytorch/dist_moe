# Reproduce the isolated Dist-MoE results

Run all **63 cases** listed below to reproduce the isolated result matrix.
These measure routed experts without shared experts or MoK. Each case is a
single expert layer distributed across the listed GPU count. Inference measures
forward execution; training measures forward plus backward.

Use the exact Torch/CUDA/NCCL/CuTe versions in [runtime.json](runtime.json) on
GB300 GPUs in one NVLink fabric. This campaign uses CUDA 13.1. The source
revision and runtime pins are required; the comparison campaign uses a separate
environment. Run from the project root:

```bash
python -m benchmarks.provenance --scope isolated
python -m benchmarks.isolated.run --render isolated-manifest.json
```

The first command checks the installed runtime and provider source hashes;
it does not install an environment or measure performance. The second writes
the experiment catalog using CPU only. This generated manifest describes cases;
it is different from the published `isolated.json` measurement results.

## Required measurement matrix

Every row means all combinations of the listed pipelines, widths and routing
settings. Dimensions are T=2048 tokens per GPU, D=7168, E=384 routed experts,
K=6 and seed=17. F is the expert intermediate width. Shared experts=0.
Inference pipelines are BF16 standard, MXFP8 staged, MXFP8 Mega and NVFP4 Mega;
training pipelines are BF16 standard, MXFP8 staged and MXFP8 Mega. Isolated
MXFP8 retains the measured default math policy.

| Manifest job (`--job`) | GPUs / four-GPU hosts | Single layer: F3072, imbalance 1/4/8 | Width control: F1536/3072/6144, imbalance 1 | Total cases | Batches |
|---|---|---:|---:|---:|---:|
| `isolated-inference-ep8` | 8 / 2 | 12 | 0 | 12 | 4 |
| `isolated-inference-ep16` | 16 / 4 | 12 | 12 | 24 | 12 |
| `isolated-training-ep32` | 32 / 8 | 9 | 0 | 9 | 3 |
| `isolated-training-ep64` | 64 / 16 | 9 | 9 | 18 | 9 |

The F3072 width-control case and balanced single-layer case have distinct
case IDs and are both required to reproduce the original records. Smoke jobs
in the generated manifest check setup; they are not part of these 63 results.

## Run the required cases

Use the [shared batch-launch instructions](../README.md#launch-every-required-case)
on every host in the allocated geometry. For example, the EP8 allocation uses:

```bash
export BENCH_MODULE=benchmarks.isolated.run
export MANIFEST=isolated-manifest.json
export JOB=isolated-inference-ep8
export NNODES=2
```

Repeat for all four jobs, setting NNODES from the table and a fresh output root.
The launcher runs every batch below. It supplies each listed ID as `--case`;
you do not need to construct IDs or inspect the manifest schema. Run the same
batch on every host. Cases may share one invocation only when their batch
number is the same within the same job.

## Exact required case IDs

<details>
<summary>isolated-inference-ep8: 12 cases in 4 batches</summary>

| Batch | Required `--case` ID |
|---|---|
| 1 | `mok_default-inference-bf16-standard-ep8-single_layer-i1-F3072-c2c480ac46` |
| 1 | `mok_default-inference-bf16-standard-ep8-single_layer-i4-F3072-d9374188ec` |
| 1 | `mok_default-inference-bf16-standard-ep8-single_layer-i8-F3072-9479bd9a85` |
| 2 | `mok_default-inference-mxfp8-staged-ep8-single_layer-i1-F3072-13c1d443c0` |
| 2 | `mok_default-inference-mxfp8-staged-ep8-single_layer-i4-F3072-11b1817891` |
| 2 | `mok_default-inference-mxfp8-staged-ep8-single_layer-i8-F3072-a22f06f1c7` |
| 3 | `mok_default-inference-mxfp8-mega-ep8-single_layer-i1-F3072-e109e45ca2` |
| 3 | `mok_default-inference-mxfp8-mega-ep8-single_layer-i4-F3072-717ff65fa4` |
| 3 | `mok_default-inference-mxfp8-mega-ep8-single_layer-i8-F3072-799b033398` |
| 4 | `mok_default-inference-nvfp4-mega-ep8-single_layer-i1-F3072-ee9e38cf2b` |
| 4 | `mok_default-inference-nvfp4-mega-ep8-single_layer-i4-F3072-111585bafd` |
| 4 | `mok_default-inference-nvfp4-mega-ep8-single_layer-i8-F3072-12ef210596` |

</details>

<details>
<summary>isolated-inference-ep16: 24 cases in 12 batches</summary>

| Batch | Required `--case` ID |
|---|---|
| 1 | `mok_default-inference-bf16-standard-ep16-single_layer-i1-F3072-51b6dcd0d1` |
| 1 | `mok_default-inference-bf16-standard-ep16-single_layer-i4-F3072-9d5a0a1e89` |
| 1 | `mok_default-inference-bf16-standard-ep16-single_layer-i8-F3072-f295d873be` |
| 1 | `mok_default-inference-bf16-standard-ep16-width-i1-F3072-76b5d8e4ff` |
| 2 | `mok_default-inference-mxfp8-staged-ep16-single_layer-i1-F3072-29513fe75e` |
| 2 | `mok_default-inference-mxfp8-staged-ep16-single_layer-i4-F3072-0b3f616a1c` |
| 2 | `mok_default-inference-mxfp8-staged-ep16-single_layer-i8-F3072-dcc837987d` |
| 2 | `mok_default-inference-mxfp8-staged-ep16-width-i1-F3072-ddfff29f2d` |
| 3 | `mok_default-inference-mxfp8-mega-ep16-single_layer-i1-F3072-588173b90e` |
| 3 | `mok_default-inference-mxfp8-mega-ep16-single_layer-i4-F3072-5d90d5c13c` |
| 3 | `mok_default-inference-mxfp8-mega-ep16-single_layer-i8-F3072-702cc66951` |
| 3 | `mok_default-inference-mxfp8-mega-ep16-width-i1-F3072-bca6fd8396` |
| 4 | `mok_default-inference-nvfp4-mega-ep16-single_layer-i1-F3072-a993b830c7` |
| 4 | `mok_default-inference-nvfp4-mega-ep16-single_layer-i4-F3072-ff1e3f7476` |
| 4 | `mok_default-inference-nvfp4-mega-ep16-single_layer-i8-F3072-fc94b72615` |
| 4 | `mok_default-inference-nvfp4-mega-ep16-width-i1-F3072-20d1d60fc4` |
| 5 | `mok_default-inference-bf16-standard-ep16-width-i1-F1536-83d16f9337` |
| 6 | `mok_default-inference-bf16-standard-ep16-width-i1-F6144-99cc47d57d` |
| 7 | `mok_default-inference-mxfp8-staged-ep16-width-i1-F1536-6d8ba89008` |
| 8 | `mok_default-inference-mxfp8-staged-ep16-width-i1-F6144-33abbf7fcd` |
| 9 | `mok_default-inference-mxfp8-mega-ep16-width-i1-F1536-77a5a218ec` |
| 10 | `mok_default-inference-mxfp8-mega-ep16-width-i1-F6144-283c2093ee` |
| 11 | `mok_default-inference-nvfp4-mega-ep16-width-i1-F1536-55a95d0f68` |
| 12 | `mok_default-inference-nvfp4-mega-ep16-width-i1-F6144-a115862636` |

</details>

<details>
<summary>isolated-training-ep32: 9 cases in 3 batches</summary>

| Batch | Required `--case` ID |
|---|---|
| 1 | `mok_default-training-bf16-standard-ep32-single_layer-i1-F3072-0cac1d228e` |
| 1 | `mok_default-training-bf16-standard-ep32-single_layer-i4-F3072-a931119f67` |
| 1 | `mok_default-training-bf16-standard-ep32-single_layer-i8-F3072-5d5a9e8993` |
| 2 | `mok_default-training-mxfp8-staged-ep32-single_layer-i1-F3072-d12d2e04b9` |
| 2 | `mok_default-training-mxfp8-staged-ep32-single_layer-i4-F3072-dae9cb35b7` |
| 2 | `mok_default-training-mxfp8-staged-ep32-single_layer-i8-F3072-c1fe115f86` |
| 3 | `mok_default-training-mxfp8-mega-ep32-single_layer-i1-F3072-0873040f05` |
| 3 | `mok_default-training-mxfp8-mega-ep32-single_layer-i4-F3072-debf3d6363` |
| 3 | `mok_default-training-mxfp8-mega-ep32-single_layer-i8-F3072-864604c924` |

</details>

<details>
<summary>isolated-training-ep64: 18 cases in 9 batches</summary>

| Batch | Required `--case` ID |
|---|---|
| 1 | `mok_default-training-bf16-standard-ep64-single_layer-i1-F3072-cd2b8fa28a` |
| 1 | `mok_default-training-bf16-standard-ep64-single_layer-i4-F3072-df59aa79b9` |
| 1 | `mok_default-training-bf16-standard-ep64-single_layer-i8-F3072-fe46c815bc` |
| 1 | `mok_default-training-bf16-standard-ep64-width-i1-F3072-923679d6d1` |
| 2 | `mok_default-training-mxfp8-staged-ep64-single_layer-i1-F3072-a2e37e0f3c` |
| 2 | `mok_default-training-mxfp8-staged-ep64-single_layer-i4-F3072-2edf57f597` |
| 2 | `mok_default-training-mxfp8-staged-ep64-single_layer-i8-F3072-a7c57f0bd1` |
| 2 | `mok_default-training-mxfp8-staged-ep64-width-i1-F3072-683d9e76ff` |
| 3 | `mok_default-training-mxfp8-mega-ep64-single_layer-i1-F3072-7135c007e1` |
| 3 | `mok_default-training-mxfp8-mega-ep64-single_layer-i4-F3072-ac10be95c7` |
| 3 | `mok_default-training-mxfp8-mega-ep64-single_layer-i8-F3072-0c69c88181` |
| 3 | `mok_default-training-mxfp8-mega-ep64-width-i1-F3072-e8d70b21c6` |
| 4 | `mok_default-training-bf16-standard-ep64-width-i1-F1536-c53f0afbb6` |
| 5 | `mok_default-training-bf16-standard-ep64-width-i1-F6144-481c5e8308` |
| 6 | `mok_default-training-mxfp8-staged-ep64-width-i1-F1536-9c13e1792c` |
| 7 | `mok_default-training-mxfp8-staged-ep64-width-i1-F6144-3c908db36d` |
| 8 | `mok_default-training-mxfp8-mega-ep64-width-i1-F1536-aa25db9bb5` |
| 9 | `mok_default-training-mxfp8-mega-ep64-width-i1-F6144-003928e01a` |

</details>

## Accepted result artifacts

[Accepted measurements](https://gist.githubusercontent.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/raw/105d4e5c9867e2752cc4306cd7356430f863f36d/isolated.csv) and
[the report](https://gist.github.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/105d4e5c9867e2752cc4306cd7356430f863f36d#file-report-md) are historical
reference data. A new run records its actual runtime and every rank's numerical,
latency, NVLink, effective HBM, allocator and trace artifacts. Effective HBM is
an analytical byte/time metric rather than measured physical DRAM traffic.
