# Matched activation benchmark

`benchmarks/swiglu.py` compares plain, clamped, and SiTU-GLU with the same
local token count, global expert count, widths, routing, scores, gradient
dtype, and memory policy. Run each group size on a real NVLink-connected
allocation; FakePG validates metadata but does not measure communication or
predict throughput.

```bash
for ranks in 1 2 4; do
  for pipeline in bf16 staged mega; do
    python -m torch.distributed.run --standalone --nproc-per-node="$ranks" \
      benchmarks/swiglu.py --mode graph --pipeline "$pipeline" \
      --output "results/${pipeline}-${ranks}.json" \
      --trace-dir "results/traces-${pipeline}-${ranks}"
  done
done
```

Defaults are 128 tokens per rank, hidden width 3584, intermediate width 3072,
eight global experts, and top-k two. Balanced cyclic routing uses unique
expert IDs and uniform scores. Each layer has one activation slot, full-save
capacity factor 1.0, device scratch factor 1.0, no VMM, and BF16 WGRAD. The
training call includes forward, DGRAD, WGRAD, planner rewind, and any dynamic
weight preparation. It excludes optimizer, FSDP, and model/framework overhead.
For lifecycle details, see [BF16 execution](bf16_execution.md).
MXFP8 defaults to `fast_math=False`; `--fast-math` explicitly selects the
approximate independent sigmoid for staged or Mega comparisons. Use the same
flag for all three activations and both source revisions. SiTU-GLU tanhs
remain precise, and a gate bound of four shares the tanh under either flag. For
fused-pipeline contracts, see [MXFP8 execution](mxfp8_execution.md).

Ten warmup calls precede compilation-free measurement. Graph mode additionally
warms a separate stream, captures a complete training call, and requires exact
eager/replay output and gradient parity. Each of five measurement windows
contains twenty calls. CUDA-event and wall-clock latencies are reduced by
maximum rank per window; throughput uses the median wall latency. Synchronize
only at warmup, capture, window, or profile boundaries, never inside a layer.
Compare baseline and modified source with the same driver and dependencies.

Profiling is outside the timed windows and selects rank zero only. JSON lists
all CUDA kernel names, counts, and total durations. BF16 activation kernels
contain `_triton_swiglu` in their names. Report fused-kernel durations as whole
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
