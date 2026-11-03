# NVFP4 Training Numerical Tests

Run these commands from the repository root on supported Blackwell GPUs in the
`pt_nightly` Conda environment.

```bash
conda activate pt_nightly
```

The one-GPU test executes the real NVFP4 forward and native BF16 backward
kernels without peer communication:

```bash
CUDA_VISIBLE_DEVICES=0 \
  python -m pytest -q \
  tests/test_dist_moe.py::DistMoeKernelTest::test_nvfp4_training_backward_matches_real_quant_reference
```

The two-GPU test adds real cross-rank dispatch, combine, NCCL synchronization,
and symmetric peer-memory access:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  python -m pytest -q \
  tests/test_dist_moe.py::DistMoeTwoRankTest::test_nvfp4_forward_and_backward_exercise_peer_paths
```

In short, run the one-GPU test for focused kernel, autograd, activation-saving,
and recompute coverage. Run the two-GPU test for the end-to-end distributed
path. Both compare against the same real-quantized NVFP4 forward and BF16
straight-through-estimator backward reference.

## Reference used by both tests

`_reference_moe_nvfp4_real_quant` groups routed rows by expert and evaluates
each expert independently. Its forward:

1. Computes the same per-token activation and per-expert weight outer scales as
   Dist-MoE.
2. Quantizes activations and weights to packed E2M1 with blocked E4M3 scales,
   using Dist-MoE's inner-scale operation order.
3. Pads expert row counts to the shape required by `torch.nn.functional.scaled_mm`.
4. Runs real NVFP4 tensor-core W13 and W2 GEMMs.
5. Removes the internal padding and applies routing scores.

The reference wraps each NVFP4 GEMM in a custom autograd function. Backward
recreates `DQ(Q(activation))` and `DQ(Q(weight))`, then computes DGRAD and WGRAD
with BF16 `torch.mm`. This models the Dist-MoE NVFP4 training contract rather
than differentiating through `scaled_mm`.

The supporting one-GPU primitive test proves that the reference quantizer
produces bitwise-identical packed FP4 data, E4M3 block scales, and outer scales:

```bash
CUDA_VISIBLE_DEVICES=0 \
  python -m pytest -q \
  tests/test_dist_moe.py::DistMoeKernelTest::test_nvfp4_reference_scale_order_matches_dist_moe_quantization
```

## One-GPU test

`test_nvfp4_training_backward_matches_real_quant_reference` uses a
world-size-one fake process group only for distributed API plumbing. All
forward, backward, quantization, dequantization, activation-buffer, and planner
work runs through the real CUDA implementation. It is numerical kernel
evidence, but not communication evidence.

The test covers:

- Staged and Mega NVFP4 forward pipelines.
- The normal full-save activation-planner decision.
- A minimum-capacity activation slot that forces forward recomputation during
  backward.
- Dynamic BF16 weight quantization in forward and requantization followed by
  dequantization in backward.
- Real NVFP4 W13 and W2 forward GEMMs.
- Native BF16 score reduction, DGRAD, SwiGLU backward, WGRAD, and input-route
  reduction.
- Planner rewind after backward.

For the current fixed shape and seed, it requires bitwise equality for:

- The BF16 forward output.
- The BF16 input gradient.
- The W13 gradient.

Two results use relative-L2 bounds because the implementation and reference
perform mathematically equivalent reductions with different kernel schedules:

- Router-score gradient: less than `1e-6`.
- W2 gradient: less than `5e-4`.

The router-score residual comes from the Triton hidden-dimension reduction
versus PyTorch autograd's reduction tree. W2 WGRAD uses Dist-MoE's grouped CuTe
kernel, while the reference uses a separate `torch.mm` for each expert. Their
FP32 accumulation orders need not be bitwise identical. The observed bitwise
W13 result is evidence for this test configuration, not a general API promise
about GEMM accumulation order.

## Two-GPU test

`test_nvfp4_forward_and_backward_exercise_peer_paths` launches two real
processes and assigns one CUDA device to each. Every token selects one expert
on each rank, so both dispatch and combine must access peer-owned rows rather
than accidentally exercising only local routing.

The test covers:

- Staged and Mega NVFP4 forward pipelines.
- Real two-rank routing metadata exchange and synchronization.
- Packed NVFP4 activation publication and peer gather.
- Remote expert execution and peer combine scatter.
- Native BF16 backward through remote routes.
- Gathering local input, score, W13, and W2 gradients for comparison with the
  global reference.
- Forward/backward weight preprocessing and both WGRAD postprocessing hooks.

The distributed endpoint uses tight tolerance-based checks:

| Result | Elementwise `rtol` | Elementwise `atol` | Relative L2 | Cosine |
| --- | ---: | ---: | ---: | ---: |
| Forward output | `5e-3` | `3.1e-5` | `< 2e-5` | `> 0.999999` |
| Every gradient | `1e-2` | `2e-3` | `< 5e-4` | `> 0.999999` |

The tolerances do not accommodate different quantized operands: those are
checked independently for bitwise equality. They accommodate floating-point
association differences between per-expert `scaled_mm` and Dist-MoE's grouped
NVFP4 kernels, between PyTorch and Triton route reductions, and between
per-expert `torch.mm` and grouped native WGRAD. Peer transport itself preserves
bits; the distributed test is needed because routing changes which rank owns
and consumes each row.

The current two-GPU forward comparison observes the final score-weighted
output. It does not separately identify whether a residual originated in an
expert's route-level H3 result or in the final route reduction. Comparing saved
H3 before score reduction would isolate those two stages.
