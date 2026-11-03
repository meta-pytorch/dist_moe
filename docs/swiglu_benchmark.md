# Matched activation benchmark

`benchmarks/swiglu.py` compares the activations supported by BF16, NVFP4
Staged, and NVFP4 Mega with the same local token count, global expert count,
widths, routing, scores, gradient dtype, and memory policy. BF16 covers plain,
clamped, and SiTU-GLU; NVFP4 covers plain and clamped SwiGLU because SiTU-GLU
is BF16-only. Run each group size on a real NVLink-connected allocation;
FakePG validates metadata but does not measure communication or predict
throughput.

```bash
for ranks in 1 2 4; do
  python -m torch.distributed.run --standalone --nproc-per-node="$ranks" \
    benchmarks/swiglu.py --mode graph --pipeline bf16 --activation all \
    --output "results/bf16-${ranks}.json" \
    --trace-dir "results/traces-bf16-${ranks}"

  for pipeline in nvfp4-staged nvfp4-mega; do
    for activation in swiglu swiglu_clamped; do
      python -m torch.distributed.run --standalone --nproc-per-node="$ranks" \
        benchmarks/swiglu.py --mode graph --pipeline "$pipeline" \
        --activation "$activation" \
        --output "results/${pipeline}-${activation}-${ranks}.json" \
        --trace-dir "results/traces-${pipeline}-${activation}-${ranks}"
    done
  done
done
```

Defaults are 128 tokens per rank, hidden width 3584, intermediate width 3072,
eight global experts, and top-k two. Without `--activation`, BF16 measures all
three supported activations while NVFP4 measures plain SwiGLU. Run NVFP4
activations in separate processes so each context has an isolated lifecycle.
Balanced cyclic routing uses unique expert IDs and uniform scores. Each layer
has one activation slot, full-save capacity factor 1.0, device scratch factor
1.0, no VMM, and BF16 WGRAD. NVFP4's native BF16 weights are quantized in
forward and again before the shared BF16 backward. The training call includes
forward, DGRAD, WGRAD, planner rewind, and dynamic weight preparation. It
excludes optimizer, FSDP, and model/framework overhead. For lifecycle details,
see
[BF16 execution](bf16_execution.md) and
[NVFP4 execution](nvfp4_execution.md).

Ten warmup calls precede compilation-free measurement. Graph mode additionally
warms a separate stream, captures a complete training call, and requires exact
eager/replay output and gradient parity. Each of five measurement windows
contains twenty calls. CUDA-event and wall-clock latencies are reduced by
maximum rank per window; throughput uses the median wall latency. Synchronize
only at warmup, capture, window, or profile boundaries, never inside a layer.
Compare baseline and modified source with the same driver and dependencies.

Profiling is outside the timed windows and selects rank zero only. JSON lists
all rank-zero CUDA kernel names, counts, and total durations. BF16 activation
kernels contain `_triton_swiglu` in their names; NVFP4 may fuse activation
arithmetic into larger expert kernels. Report fused-kernel durations as whole
kernels, not isolated activation-math time. Profiling can skew peer waiting;
the unprofiled windows, not the trace, determine throughput. Record GPU model,
SM count, interconnect/topology, software versions, and both source revisions
alongside the results.

Metadata-only preflight can run without a real collective group:

```bash
python benchmarks/swiglu.py --fake-ep-size 4 --output results/preflight-4.json
```

One CUDA device constructs each fake rank's local context; forward/backward
FakeTensor propagation validates metadata and memory planning without running
the expert kernels. Its `timing` field is intentionally null.
