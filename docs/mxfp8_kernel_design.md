# Asynchronous MXFP8 Kernel Design

This guide explains MXFP8 representation and GPU execution below the public
API. Read [asynchronous MXFP8 execution](mxfp8_execution.md) first for context,
ownership, saved state, and the end-to-end call sequence.

**CTA** means cooperative thread array. **TMA** moves tiles between global and
shared memory. **TMEM** holds Blackwell tensor-core accumulators.

## Outline

- [Numeric format](#numeric-format)
- [Scale storage](#scale-storage)
- [Staged forward](#staged-forward)
- [Mega forward](#mega-forward)
- [Backward](#backward)
- [Warp roles](#warp-roles)
- [Saved state and recomputation](#saved-state-and-recomputation)

<a id="numeric-format"></a>
## Numeric format

MXFP8 E4M3 stores one FP8 value per element and one E8M0 power-of-two scale per
32 logical values. Tensor-core products use FP8 data and E8M0 scales and
accumulate in FP32 TMEM.

```text
BF16 block x[0:32]
  -> amax and E8M0 scale
  -> 32 E4M3 qdata bytes + 1 scale byte
```

Activations use one-dimensional scales along the contraction dimension:

- row layout `[M, K]`: one scale per 32 columns, used by FPROP/DGRAD;
- column layout for logical `[M, K]`: one scale per 32 rows, used when the
  activation is consumed as a transposed WGRAD operand.

Activation causality is preserved: no 32x32 activation scale depends on future
rows. Weights are static for a launch and use grouped 32x32 quantization, so one
qdata allocation can be shared by FPROP and DGRAD while each orientation owns
its required scale layout.

<a id="scale-storage"></a>
## Scale storage

Logical natural scales are converted to the cuBLAS/CUTLASS blocked atom:

```text
logical scale matrix
  rows: tensor rows
  cols: K / 32 scale columns

one blocked atom: 128 rows x 4 scale columns
storage offset within atom:
  (row % 32) * 16 + (row // 32) * 4 + column
```

Atoms are ordered M-major then K-major. CuTe consumes the same byte stream by
tiling a scale atom over the MMA shape. Transposed consumption changes the
logical scale orientation, not merely tensor strides; FPROP and DGRAD scales
are therefore distinct tensors.

<a id="staged-forward"></a>
## Staged forward

The staged path executes two distributed kernels:

1. `dist_blockscaled_grouped_gemm_fprop_dispatch`
2. `dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine`

Training publishes BF16 source rows in symmetric dispatch HBM. Dispatch
gather/quant warps read local or peer rows, create the row-quantized W13
operand and optional column WGRAD operand in the local activation buffer, then hand the
tile to TMA/MMA.

```text
BF16 source row in local/peer symmetric HBM
  -> gather + row/column quant producer
  -> E4M3 qdata + E8M0 scales in local HBM
  -> TMA qdata/scales to SMEM
  -> tcgen05 FP8 x FP8, FP32 TMEM accumulation
  -> h1 BF16 in local activation buffer
```

The second kernel loads BF16 `h1`, evaluates SwiGLU in FP32 registers,
quantizes `h2`, runs W2, and scatters BF16 route outputs directly to peer
combine HBM:

```text
h1 BF16 in local HBM
  -> SwiGLU FP32 registers
  -> h2 MXFP8 + E8M0 in local HBM/SMEM pipeline
  -> TMA to SMEM
  -> tcgen05 FP8 x FP8, FP32 TMEM accumulation
  -> BF16 in SMEM
  -> 16-byte stores to peer symmetric HBM for combine
```

The clamped-SwiGLU specialization changes only the FP32 register expression
before quantization. Forward, backward, and recomputation receive identical
static alpha/limit values; qdata, scale layouts, MMA accumulation, and peer
publication remain unchanged.

<a id="mega-forward"></a>
## Mega forward

`chunked_mega_blockscaled_grouped_gemm_fprop_swiglu_fwd` preserves the same
formats and route order but pipelines W13, SwiGLU quantization, W2, and combine
stores in one chunked kernel topology. Tile-scheduler records connect producer
and consumer phases. The Mega policy changes launch fusion, not the weight or
activation representation contract.

<a id="backward"></a>
## Backward

Staged backward executes:

1. `dist_blockscaled_grouped_gemm_dgrad_dispatch` for W2 DGRAD and column
   quantization of `grad_h3` when W2 WGRAD must be materialized;
2. `blockscaled_grouped_gemm_wgrad` for W2;
3. `dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine` for fused SwiGLU
   backward, W13 DGRAD, and peer scatter;
4. `blockscaled_grouped_gemm_wgrad` for W13;
5. `reduce_from_topk` for the local input gradient.

Mega backward replaces the separate DGRAD/WGRAD pairs with
`mega_blockscaled_grouped_gemm_dgrad_wgrad_dispatch` and
`mega_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine`.

WGRAD MMA accumulates in FP32. Its destination may be BF16 or FP32. Optional
reduce-add output stores accumulate directly into a caller-owned gradient
destination; this does not change the MMA precision.

<a id="warp-roles"></a>
## Warp roles

The distributed dispatch CTA extends the block-scaled grouped GEMM with a
gather/quant producer group:

```text
epilogue warps       FP32 TMEM -> output SMEM -> GMEM/peer stores
MMA warp/group       tcgen05 block-scaled MMA
TMA warp             qdata and scale SMEM pipeline
idle/control warps   cluster ordering
gather/quant warps   peer loads, amax, qdata/scale stores, tile publication
```

The exact warp count and split are configuration-dependent and frozen with the
selected production config. SMEM carries A/B qdata, SFA/SFB scale tiles,
epilogue storage, descriptor staging, and mbarriers. TMEM holds FP32
accumulators only.

<a id="saved-state-and-recomputation"></a>
## Saved state and recomputation

The activation planner allocates both row- and column-oriented operands only
when later consumers require them. When an activation slot cannot retain the
complete forward state, the fixed backward topology conditionally reruns the
forward producer into scratch. Weight qdata and both scale orientations are
saved independently of this activation decision unless the caller supplies
prepared operands.

The synchronous block-scaled implementation is not part of the public runtime.
