# Asynchronous MXFP8 Kernel Design

This guide explains MXFP8 representation and GPU execution below the public
API. Read [asynchronous MXFP8 execution](mxfp8_execution.md) first for context,
ownership, saved state, and the end-to-end call sequence. The
[memory planner](memory_planner.md) owns allocation limits and the distinction
between planner-backed scratch and device-only Mega static inference storage.

**CTA** means cooperative thread array. **TMA** moves tiles between global and
shared memory. **TMEM** holds Blackwell tensor-core accumulators.

## Outline

- [Numeric format](#numeric-format)
- [Scale storage](#scale-storage)
- [Kernel components](#kernel-components)
- [Staged forward](#staged-forward)
- [Mega forward](#mega-forward)
- [Inference layout and scratch](#inference-layout-and-scratch)
- [Backward](#backward)
- [Quantization readiness](#quantization-readiness)
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
rows. Training weights use grouped 32x32 quantization, so one qdata allocation
can be shared by FPROP and DGRAD while each orientation owns its required scale
layout. Inference preparation uses canonical row-wise quantization and retains
only the FPROP orientation.

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

Token tiles narrower than 128 rows, or wider tiles that are not multiples of
128, use physical scale pages. A token tile of `B` rows occupies
`P = ceil(B / 128) * 128` scale rows. Scale-prefix metadata advances by pages,
not by the compact qdata row count. The routing padding multiple, token tile,
and scale-page extent are different quantities.

<a id="kernel-components"></a>
## Kernel components

| Component | Role |
| --- | --- |
| `blockscaled_config`, `dist_blockscaled_config`, `chunked_mega_blockscaled_policy` | Format-specific geometry, pipeline depths, and topology selection |
| `inference_config` | One source-policy resolution shared by runtime inference and physical scratch planning |
| `blockscaled_gemm_tiles` | Qdata/scale TMA descriptors, operand tiles, and MMA operations |
| `blockscaled_entrypoints`, `dist_blockscaled_entrypoints`, `mega_blockscaled_entrypoints` | Matching launch and device entrypoints for each kernel family |
| `gemm_warp_roles` | Shared TMA, MMA, and scheduling roles |
| `grouped_problem_visitor`, `tile_scheduler` | Expert/problem traversal and work-record publication |
| `dispatch_quant`, `combine_swiglu_quant` | Quantization producers and their row/column readiness protocols |

With `SWAP_AB`, the config's `BLOCK_SIZE_N` is the token-axis tile and
`BLOCK_SIZE_M` is the feature-axis tile. These names describe the kernel's
swapped contraction, not the public tensor shape. The source configuration
helpers resolve the complete geometry before launch; the scratch planner does
not choose another width when a layout exceeds its bound.

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

The clamped-SwiGLU and SiTU-GLU specializations change only the FP32 register
expression before quantization. SiTU-GLU computes

```text
b_g = situ_glu_gate_beta * tanh(g / situ_glu_gate_beta)
b_u = situ_glu_up_beta * tanh(u / situ_glu_up_beta)
h2  = b_g * sigmoid(g) * b_u
```

The tanh operations use precise math; sigmoid follows the selected kernel
fast-math policy. Forward, backward, and recomputation receive identical static
activation parameters. Qdata, scale layouts, MMA accumulation, and peer
publication remain unchanged.

<a id="mega-forward"></a>
## Mega forward

`chunked_mega_blockscaled_grouped_gemm_fprop_swiglu_fwd` preserves the same
formats and route order but pipelines W13, SwiGLU quantization, W2, and combine
stores in one chunked kernel topology. Work records carry the expert, GEMM,
chunk, and physical qdata/scale prefixes to the producer and consumer roles.
W2 waits for the corresponding H2 quantization, rather than waiting for every
expert's W13 work to finish.

An eligible scheduler-relay topology publishes shared work records for the
TMA, MMA, and epilogue roles. It does not replace the quantization readiness
counters or the caller-owned peer barriers. Source policy selects the relay,
chunk lead, epilogue transfers, and pipeline depth together. Deriving pipeline
depth searches the legal shared-memory stages for that geometry; it is not a
scratch-capacity tuning loop.

Pending TMA stores must stop reading the C shared-memory stage before it is
reused. `_epilog_wait_pending_tma_store` and `_wait_pending_epilog_store`
perform `cp_async_bulk_wait_group(0, read=True)` in the issuing epilogue warp,
then synchronize the epilogue writers. Mega backward also drains these reads
before the combine stage aliases storage used by the other GEMM's C stages.
A TMEM accumulator becoming available does not prove that those older SMEM
reads have finished.

<a id="inference-layout-and-scratch"></a>
## Inference layout and scratch

Training weights and the default staged/Mega inference weights concatenate
gate and up rows. Opt-in `fuse_swiglu=True` preparation is restricted to Mega
inference. Its W13 feature rows are interleaved in groups of eight:

```text
canonical:   gate[0:F], up[0:F]
interleaved: gate[0:8], up[0:8], gate[8:16], up[8:16], ...
```

Canonical rows are quantized first. Qdata and natural scale rows then receive
the same permutation before scales are converted to blocked storage. This
preserves the original quantization groups; quantizing the already-interleaved
weight would be a different operation. W2 is not interleaved, and the logical
prepared-weight identity remains the original weight.

In admitted G8 configurations, the interleaved FC13 epilogue produces
row-quantized H2 directly. Dense H1/H2
are not retained, and unused column-quantized WGRAD buffers are only small
descriptor operands. This layout is forward-only, not a training checkpoint
or a substitute for concatenated weights on the staged path.

Compact inference routing keeps a 32-row logical alignment while each expert
occupies complete physical token tiles. If `C` is the hard receive capacity,
`G` the local expert count, and `B` the token tile, the source allocation uses
`R = align_up(C + (B - 32) * G, lcm(128, B))` physical qdata rows. For paged
scales it reserves `max(ceil(R / B) * P, G * P)` scale rows. These physical
tails do not enlarge the admitted routing capacity or become valid expert
inputs.

Automatic inference token tiles are multiples of 32 up to 256. The source
policy considers the host token count, local expert count, feature dimensions,
SM count, and routing capacity together. Its ordinary larger-row fallback uses
nonpaged 128/256-row tiles. These are geometry choices, not a guarantee that
the maximum runtime token count has the largest physical scale allocation.

Device-only Mega inference binds operands, scales, counters, and tensor maps
as aligned views of the context's existing byte scratch allocation. Concat
uses a full dense H1 view; interleaved W13 needs only the descriptor operand.
The internal source admission rules choose host or device tensor maps. Host
maps, when admitted, consume these actual views rather than zero-offset
placeholders; partial scale pages or other unsupported host geometry use
device preparation. Static storage does not force a host-map path.

This path skips the unused runtime `get_forward_plan()` and received-count
clone; GPU routing counts remain live inputs to descriptor preparation and
compute. Offset-backed operands with host tensor maps are rejected before
launch. The bridge does not read GPU activation offsets on the CPU or allocate a second
arena. Staged execution, training, and no-grad evaluation on a training
context retain their GPU-offset-backed operands. Every VMM-configured inference
context also keeps the GPU-offset planner and device tensor maps, rather than
reinterpreting its allocation as the device-only static layout. This choice is
per context; no public per-operand mapping or descriptor selector is exposed.
Interleaved source configurations with offset-backed storage retain their
existing admission limits and are not enabled merely by requesting Mega:
`COMPACT_DECODE_ROWS`, explicit `SCHEDULER_RELAY`, and
`ACTIVATION_RING_CHUNKS` are excluded with GPU offsets. Automatic relay is
ineligible and native tails are normalized off. These are internal source
admission rules, not public storage controls. Compact G8 additionally requires
host descriptors; static views alone do not admit it when the source selects
device maps.

The [memory planner](memory_planner.md#build-and-inspect-a-plan) reserves the
device-only Mega physical envelope across the supported runtime token counts;
the launcher checks its actual layout against that bound before use.
Planner-backed paths retain their offset-based scratch bounds.

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

For SiTU-GLU, the fused activation producer reuses the forward FP32 values and
applies

```text
d_b_g/dg = 1 - tanh(g / situ_glu_gate_beta)^2
d_b_u/du = 1 - tanh(u / situ_glu_up_beta)^2
d_gate/dg = (d_b_g/dg) * sigmoid(g)
             + b_g * sigmoid(g) * (1 - sigmoid(g))
grad_g = grad_h2 * b_u * d_gate/dg
grad_u = grad_h2 * b_g * sigmoid(g) * d_b_u/du
```

It quantizes these logical W13-input gradients through the same row/column
paths used by plain and clamped SwiGLU.

WGRAD MMA accumulates in FP32. Its destination may be BF16 or FP32. Optional
reduce-add output stores accumulate directly into a caller-owned gradient
destination; this does not change the MMA precision.

<a id="quantization-readiness"></a>
## Quantization readiness

Row- and column-quantized copies have separate completion counters. A row
counter gates a DGRAD/FPROP activation tile; a column counter gates the feature
range needed by WGRAD. Publishing one copy does not make the other copy ready.
Every producer lane fences its generic-proxy qdata/scale stores to the async
proxy before a warp/group joins the release. Consumers acquire the appropriate
counter before issuing dependent TMA loads.

For Mega W13 WGRAD, `_mega_combine_quant_wgrad_feature_tile_wait` intersects
the requested feature tile with both halves of logical `[dgate, dup]`. It
maps the resulting column-work counter ranges to lanes in batches of up to
32. `_wait_lane_counters_at_least` uses per-lane acquire probes and a
warp-uniform ballot until every active lane observes the required number of
row blocks. Inactive lanes still use readable counter addresses, and a warp
sync orders the completed acquires before dependent loads. This is a readiness
protocol, not evidence that compute and communication overlapped in a run.

The counter plans use logical receive capacity even when compact operand
storage has more physical rows. Ordinary descriptor preparation zeros the
dynamic work counter and the extra quantization counters. When descriptor
preparation is folded into the Mega kernel, a separate ready counter counts
published expert/GEMM descriptor pairs and consumers wait for all pairs. That
path requires counters to be cleared before the folded builders run. With
static storage, an unsupplied host-tensormap counter region is zeroed on the
caller stream; the ordinary device-tensormap path retains its prepare kernel.

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

The exact warp count and split are configuration-dependent and selected with
the complete kernel config. SMEM carries A/B qdata, SFA/SFB scale tiles,
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
