# Reproduce the native MoK comparison results

Run all **110 backend cases** listed below: 66 Dist-MoE pipeline cases and
44 native MoK cases. They form 44 matched workloads. Both implementations use
CUDA graphs and one BF16 shared expert. Inference measures forward execution;
training measures forward plus backward.

Use the measured environment in [runtime.json](runtime.json): Torch revision
`6aa9e2fc40d8c71dc7db1177abd8f65474d1221b`, CUDA 13.0/NCCL 2.30.7 and CuTe 4.7.1.
Use GB300 GPUs in one NVLink fabric. The separate isolated campaign uses its
own CUDA 13.1 environment.

## Native MoK setup

Build/install `cursor/mixture-of-kittens` at commit
`f3a8bd2432457c5a2a2ea13915f0dcd0224581e4` with Python 3.12 and the pinned Torch,
CUDA 13.0, target compiler and SM103a architecture. Apply the recorded
[namespace-only build fix](namespace_qualification.patch). Measured native
wheel/extension and provider file hashes are in [../versions.json](../versions.json).
The runner requires the measured wheel bytes; rebuilding from the source pin
alone does not guarantee identical bytes across compilers or build paths.
Preserve the original wheel to reproduce this frozen study.

```bash
python -m pip wheel --no-build-isolation --no-deps <pinned-mok-source> \
  --wheel-dir <wheel-output>
python -m pip install --no-deps <built-mok-wheel>
python -m benchmarks.provenance --scope comparison
python -m benchmarks.mok-comparison.run --render comparison-manifest.json
```

`provenance` checks the actual installed runtime/provider hashes and native
extension; it does not install or benchmark anything. `--render` writes the
experiment catalog using CPU only. This generated manifest is different from
the published result JSON.

## Required measurement matrix

Every row means all combinations of the five backend pipelines, listed widths
and routing settings. Dimensions are T=2048 tokens per GPU, D=7168, E=384 routed
experts, K=6 and seed=17. F is the expert intermediate width. The five pipelines
are Dist BF16 standard, Dist MXFP8 staged, Dist MXFP8 Mega, MoK BF16 native and
MoK MXFP8 native. Dist MXFP8 uses fast math; Dist BF16 uses automatic fused SwiGLU
backward. Dist shared/routed work uses the measured concurrent stream schedule.

| Manifest job (`--job`) | GPUs / four-GPU hosts | Single layer: F3072, imbalance 1/4/8 | Width control: F1536/3072/6144, imbalance 1 | Ten layers: F3072, profiles A/B | Total cases | Batches |
|---|---|---:|---:|---:|---:|---:|
| `comparison-inference-ep8` | 8 / 2 | 15 | 0 | 0 | 15 | 5 |
| `comparison-inference-ep16` | 16 / 4 | 15 | 15 | 0 | 30 | 15 |
| `comparison-training-ep32` | 32 / 8 | 15 | 0 | 10 | 25 | 10 |
| `comparison-training-ep64` | 64 / 16 | 15 | 15 | 10 | 40 | 20 |

The F3072 width-control case and balanced single-layer case have distinct IDs
and are both required. Smoke and validation-only fit jobs are setup checks;
they are not part of these 110 measured backend cases.

The ten-layer inputs are independent; retain all ten forwards before reverse
backward. Dist uses one activation slot at factor 2.5. MoK's native 4096-row
minibatch rounds its per-layer saved-state factor to 2.667. Required profiles:

```text
A: 7.5,2.5,4.5,0.6,1.8,0.9,2.2,0.7,1.6,0.8  aggregate 2.31
B: 0.7,5.8,1.2,0.6,3.6,1.0,7.2,0.8,2.3,0.7  aggregate 2.39
```

## Run the required cases

Use the [shared batch-launch instructions](../README.md#launch-every-required-case)
on every host in the allocated geometry. For example, the EP32 allocation uses:

```bash
export BENCH_MODULE=benchmarks.mok-comparison.run
export MANIFEST=comparison-manifest.json
export JOB=comparison-training-ep32
export NNODES=8
```

Repeat for all four jobs, setting NNODES from the table and a fresh output root.
The launcher runs every batch below, including both Dist and MoK. It supplies
each listed ID as `--case`; no manual ID construction or manifest inspection
is required. Run the same batch on every host. Only cases with the same batch
number in the same job can share one invocation.

Each measured case uses ten warmups and five windows of 30 graph iterations.
The same case also runs five windows of 30 eager control iterations. Profiling,
numerical checks and allocator diagnostics are outside the timing windows.

## Exact required case IDs

<details>
<summary>comparison-inference-ep8: 15 cases in 5 batches</summary>

| Batch | Required `--case` ID |
|---|---|
| 1 | `mok_default-inference-bf16-standard-ep8-single_layer-i1-F3072-ee0b9deeec` |
| 1 | `mok_default-inference-bf16-standard-ep8-single_layer-i4-F3072-f916e0253b` |
| 1 | `mok_default-inference-bf16-standard-ep8-single_layer-i8-F3072-45990e6283` |
| 2 | `mok_default-inference-mxfp8-staged-ep8-single_layer-i1-F3072-cdc865540a` |
| 2 | `mok_default-inference-mxfp8-staged-ep8-single_layer-i4-F3072-332643919d` |
| 2 | `mok_default-inference-mxfp8-staged-ep8-single_layer-i8-F3072-2a7c812ea7` |
| 3 | `mok_default-inference-mxfp8-mega-ep8-single_layer-i1-F3072-641534d579` |
| 3 | `mok_default-inference-mxfp8-mega-ep8-single_layer-i4-F3072-a7ca219578` |
| 3 | `mok_default-inference-mxfp8-mega-ep8-single_layer-i8-F3072-d4d0c90e85` |
| 4 | `mok_default-inference-bf16-native-ep8-single_layer-i1-F3072-281326603f` |
| 4 | `mok_default-inference-bf16-native-ep8-single_layer-i4-F3072-e577afd04b` |
| 4 | `mok_default-inference-bf16-native-ep8-single_layer-i8-F3072-00e17c04a7` |
| 5 | `mok_default-inference-mxfp8-native-ep8-single_layer-i1-F3072-bcd2c6143f` |
| 5 | `mok_default-inference-mxfp8-native-ep8-single_layer-i4-F3072-7da62b3697` |
| 5 | `mok_default-inference-mxfp8-native-ep8-single_layer-i8-F3072-b39667c0bf` |

</details>

<details>
<summary>comparison-inference-ep16: 30 cases in 15 batches</summary>

| Batch | Required `--case` ID |
|---|---|
| 1 | `mok_default-inference-bf16-standard-ep16-single_layer-i1-F3072-da2e7d9de1` |
| 1 | `mok_default-inference-bf16-standard-ep16-single_layer-i4-F3072-755a014285` |
| 1 | `mok_default-inference-bf16-standard-ep16-single_layer-i8-F3072-86c2c54615` |
| 1 | `mok_default-inference-bf16-standard-ep16-width-i1-F3072-aba8817ea3` |
| 2 | `mok_default-inference-mxfp8-staged-ep16-single_layer-i1-F3072-03cd080078` |
| 2 | `mok_default-inference-mxfp8-staged-ep16-single_layer-i4-F3072-040e44fcdf` |
| 2 | `mok_default-inference-mxfp8-staged-ep16-single_layer-i8-F3072-ea41cc6643` |
| 2 | `mok_default-inference-mxfp8-staged-ep16-width-i1-F3072-62e3b014bb` |
| 3 | `mok_default-inference-mxfp8-mega-ep16-single_layer-i1-F3072-af4643a8bf` |
| 3 | `mok_default-inference-mxfp8-mega-ep16-single_layer-i4-F3072-0ae24438b0` |
| 3 | `mok_default-inference-mxfp8-mega-ep16-single_layer-i8-F3072-01498ee335` |
| 3 | `mok_default-inference-mxfp8-mega-ep16-width-i1-F3072-51da9f78d0` |
| 4 | `mok_default-inference-bf16-native-ep16-single_layer-i1-F3072-f34e8b45e4` |
| 4 | `mok_default-inference-bf16-native-ep16-single_layer-i4-F3072-90f90756f8` |
| 4 | `mok_default-inference-bf16-native-ep16-single_layer-i8-F3072-9f6d932286` |
| 4 | `mok_default-inference-bf16-native-ep16-width-i1-F3072-81756a5735` |
| 5 | `mok_default-inference-mxfp8-native-ep16-single_layer-i1-F3072-8fe38391fc` |
| 5 | `mok_default-inference-mxfp8-native-ep16-single_layer-i4-F3072-242274849a` |
| 5 | `mok_default-inference-mxfp8-native-ep16-single_layer-i8-F3072-78f5fff826` |
| 5 | `mok_default-inference-mxfp8-native-ep16-width-i1-F3072-8a1528ed3b` |
| 6 | `mok_default-inference-bf16-standard-ep16-width-i1-F1536-ca21365783` |
| 7 | `mok_default-inference-bf16-standard-ep16-width-i1-F6144-3f79d2c4f8` |
| 8 | `mok_default-inference-mxfp8-staged-ep16-width-i1-F1536-c0d3fbb4c2` |
| 9 | `mok_default-inference-mxfp8-staged-ep16-width-i1-F6144-13565b84f8` |
| 10 | `mok_default-inference-mxfp8-mega-ep16-width-i1-F1536-91edbf48e7` |
| 11 | `mok_default-inference-mxfp8-mega-ep16-width-i1-F6144-1824d1c6e5` |
| 12 | `mok_default-inference-bf16-native-ep16-width-i1-F1536-20d397c162` |
| 13 | `mok_default-inference-bf16-native-ep16-width-i1-F6144-1ffa3c53bb` |
| 14 | `mok_default-inference-mxfp8-native-ep16-width-i1-F1536-c76a264ab0` |
| 15 | `mok_default-inference-mxfp8-native-ep16-width-i1-F6144-800f1e6fce` |

</details>

<details>
<summary>comparison-training-ep32: 25 cases in 10 batches</summary>

| Batch | Required `--case` ID |
|---|---|
| 1 | `mok_default-training-bf16-standard-ep32-single_layer-i1-F3072-c845d22859` |
| 1 | `mok_default-training-bf16-standard-ep32-single_layer-i4-F3072-f5fc8f09b4` |
| 1 | `mok_default-training-bf16-standard-ep32-single_layer-i8-F3072-dbce0e93c0` |
| 2 | `mok_default-training-mxfp8-staged-ep32-single_layer-i1-F3072-7c9003d4ff` |
| 2 | `mok_default-training-mxfp8-staged-ep32-single_layer-i4-F3072-98addc084b` |
| 2 | `mok_default-training-mxfp8-staged-ep32-single_layer-i8-F3072-23912625d5` |
| 3 | `mok_default-training-mxfp8-mega-ep32-single_layer-i1-F3072-2d52f81ed9` |
| 3 | `mok_default-training-mxfp8-mega-ep32-single_layer-i4-F3072-6c9bf2c23b` |
| 3 | `mok_default-training-mxfp8-mega-ep32-single_layer-i8-F3072-f949dded01` |
| 4 | `mok_default-training-bf16-native-ep32-single_layer-i1-F3072-8f2640c7a9` |
| 4 | `mok_default-training-bf16-native-ep32-single_layer-i4-F3072-89bb19ac4f` |
| 4 | `mok_default-training-bf16-native-ep32-single_layer-i8-F3072-016c4c257b` |
| 5 | `mok_default-training-mxfp8-native-ep32-single_layer-i1-F3072-cb63ebba0e` |
| 5 | `mok_default-training-mxfp8-native-ep32-single_layer-i4-F3072-155aa901a8` |
| 5 | `mok_default-training-mxfp8-native-ep32-single_layer-i8-F3072-bfc5cf1773` |
| 6 | `mok_default-training-bf16-standard-ep32-packing-A-F3072-cbb8697517` |
| 6 | `mok_default-training-bf16-standard-ep32-packing-B-F3072-3b4c5103b7` |
| 7 | `mok_default-training-mxfp8-staged-ep32-packing-A-F3072-78344dfa5e` |
| 7 | `mok_default-training-mxfp8-staged-ep32-packing-B-F3072-8e52086ff8` |
| 8 | `mok_default-training-mxfp8-mega-ep32-packing-A-F3072-f462af0c8a` |
| 8 | `mok_default-training-mxfp8-mega-ep32-packing-B-F3072-928567ccbb` |
| 9 | `mok_default-training-bf16-native-ep32-packing-A-F3072-76de8ebaa6` |
| 9 | `mok_default-training-bf16-native-ep32-packing-B-F3072-ba73d55bf1` |
| 10 | `mok_default-training-mxfp8-native-ep32-packing-A-F3072-04133f8ced` |
| 10 | `mok_default-training-mxfp8-native-ep32-packing-B-F3072-8a8cec4ad5` |

</details>

<details>
<summary>comparison-training-ep64: 40 cases in 20 batches</summary>

| Batch | Required `--case` ID |
|---|---|
| 1 | `mok_default-training-bf16-standard-ep64-single_layer-i1-F3072-404c17a44c` |
| 1 | `mok_default-training-bf16-standard-ep64-single_layer-i4-F3072-e3f70862ab` |
| 1 | `mok_default-training-bf16-standard-ep64-single_layer-i8-F3072-a1bed48358` |
| 1 | `mok_default-training-bf16-standard-ep64-width-i1-F3072-dc57e2437d` |
| 2 | `mok_default-training-mxfp8-staged-ep64-single_layer-i1-F3072-89e224ce90` |
| 2 | `mok_default-training-mxfp8-staged-ep64-single_layer-i4-F3072-3626f69507` |
| 2 | `mok_default-training-mxfp8-staged-ep64-single_layer-i8-F3072-b930dfb4fa` |
| 2 | `mok_default-training-mxfp8-staged-ep64-width-i1-F3072-804227b83a` |
| 3 | `mok_default-training-mxfp8-mega-ep64-single_layer-i1-F3072-d76c3813f1` |
| 3 | `mok_default-training-mxfp8-mega-ep64-single_layer-i4-F3072-06a45f2b54` |
| 3 | `mok_default-training-mxfp8-mega-ep64-single_layer-i8-F3072-1b841ab345` |
| 3 | `mok_default-training-mxfp8-mega-ep64-width-i1-F3072-a3b5275333` |
| 4 | `mok_default-training-bf16-native-ep64-single_layer-i1-F3072-e33b0d193c` |
| 4 | `mok_default-training-bf16-native-ep64-single_layer-i4-F3072-fa97f70fc1` |
| 4 | `mok_default-training-bf16-native-ep64-single_layer-i8-F3072-89d198ef38` |
| 4 | `mok_default-training-bf16-native-ep64-width-i1-F3072-0f9bb96e34` |
| 5 | `mok_default-training-mxfp8-native-ep64-single_layer-i1-F3072-6cffdbdc0a` |
| 5 | `mok_default-training-mxfp8-native-ep64-single_layer-i4-F3072-dd302b2055` |
| 5 | `mok_default-training-mxfp8-native-ep64-single_layer-i8-F3072-142dd863df` |
| 5 | `mok_default-training-mxfp8-native-ep64-width-i1-F3072-6922069aa2` |
| 6 | `mok_default-training-bf16-standard-ep64-width-i1-F1536-d9ef78ef88` |
| 7 | `mok_default-training-bf16-standard-ep64-width-i1-F6144-6cadb31702` |
| 8 | `mok_default-training-mxfp8-staged-ep64-width-i1-F1536-8a702dd4ac` |
| 9 | `mok_default-training-mxfp8-staged-ep64-width-i1-F6144-7be5fed5f7` |
| 10 | `mok_default-training-mxfp8-mega-ep64-width-i1-F1536-105783d969` |
| 11 | `mok_default-training-mxfp8-mega-ep64-width-i1-F6144-7a53b601c5` |
| 12 | `mok_default-training-bf16-native-ep64-width-i1-F1536-a4e5231b02` |
| 13 | `mok_default-training-bf16-native-ep64-width-i1-F6144-1bd9f95202` |
| 14 | `mok_default-training-mxfp8-native-ep64-width-i1-F1536-0234749818` |
| 15 | `mok_default-training-mxfp8-native-ep64-width-i1-F6144-36fab0230d` |
| 16 | `mok_default-training-bf16-standard-ep64-packing-A-F3072-f9cba50699` |
| 16 | `mok_default-training-bf16-standard-ep64-packing-B-F3072-aebb4a7579` |
| 17 | `mok_default-training-mxfp8-staged-ep64-packing-A-F3072-7997e2d39e` |
| 17 | `mok_default-training-mxfp8-staged-ep64-packing-B-F3072-d05d2a634f` |
| 18 | `mok_default-training-mxfp8-mega-ep64-packing-A-F3072-4be0213863` |
| 18 | `mok_default-training-mxfp8-mega-ep64-packing-B-F3072-9b4e766d6f` |
| 19 | `mok_default-training-bf16-native-ep64-packing-A-F3072-d45967ce7a` |
| 19 | `mok_default-training-bf16-native-ep64-packing-B-F3072-04efa5dbc9` |
| 20 | `mok_default-training-mxfp8-native-ep64-packing-A-F3072-a39e139d18` |
| 20 | `mok_default-training-mxfp8-native-ep64-packing-B-F3072-62ed3db118` |

</details>

## Accepted result artifacts

[Conclusions](https://gist.github.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/105d4e5c9867e2752cc4306cd7356430f863f36d#file-report-md) and
[all accepted pipeline measurements](https://gist.githubusercontent.com/sanketpurandare/830b95c363533b29c543ca9dc65f57ec/raw/105d4e5c9867e2752cc4306cd7356430f863f36d/mok-comparison.csv)
retain the original results. The comparison supports the tested overall
latency advantage and cross-layer capacity flexibility; it does not establish
universal phase superiority, total HBM savings or end-to-end model throughput.
