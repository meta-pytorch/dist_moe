# NVFP4 Kernel Design

This guide explains the packed NVFP4 representation and the GPU kernels that
consume it. Read [NVFP4 execution](nvfp4_execution.md) first for the
public API and end-to-end lifetime model.
The [memory planner](memory_planner.md) defines the hard routing and physical
scratch bounds; this guide explains the kernel views inside that allocation.

## Outline

- [Supported surface](#supported-surface)
- [Prepare weights](#prepare-weights)
- [Publish source rows](#publish-source-rows)
- [Run expert compute](#run-expert-compute)
- [Interleaved Mega inference](#interleaved-mega-inference)
- [Physical scales and scratch](#physical-scales-and-scratch)
- [Publication and readiness](#publication-and-readiness)
- [Compare with MXFP8](#compare-with-mxfp8)
- [Preserve numerical and layout invariants](#preserve-numerical-and-layout-invariants)

<a id="supported-surface"></a>
## Supported surface

NVFP4 supports Staged and Mega forward for training and inference. Training
uses native BF16 weights and delegates its gradient sequence to the BF16
kernels after reconstructing exact dequantized forward operands. Prepared
weights and interleaved W13 remain inference-only.

The format stores:

```text
qdata: two E2M1 values per byte
block scale: one E4M3 byte per 16 logical values
activation global scale: one FP32 inverse scale per source row
weight global scale: one FP32 scale per expert tensor
weight inverse global scale: one precomputed FP32 reciprocal per expert tensor
```

The global scales keep block-scale values representable in E4M3. They are
separate MMA epilogue inputs, not replacements for the per-16-value scales.

<a id="prepare-weights"></a>
## Prepare weights

NVFP4 execution accepts native BF16 or explicitly prepared weights. Native
weights are converted to packed FPROP qdata and CUBLAS-blocked scales during
every invocation. Explicit preparation creates the same operands once.
The inference-only prepared object does not allocate DGRAD qdata or scales.
It retains both the per-expert global scale and its
reciprocal. The runtime consumes the reciprocal directly, so prepared eager
execution and CUDA graph replay do not launch quantization or a reciprocal
kernel on every invocation.

Training accepts only native BF16 weights. Forward quantizes them for FPROP.
Backward quantizes them again, decodes packed E2M1 data and CUBLAS-blocked
E4M3 scales with the per-expert inverse global scale, and supplies BF16
`DQ(Q(W13))` and `DQ(Q(W2))` to native DGRAD. Packed weights are not saved.

The default W13 rows concatenate gate and up projections for both staged and
Mega execution. Opt-in `fuse_swiglu=True` preparation is restricted to Mega
inference. It quantizes canonical W13 first, interleaves qdata and natural
scale rows together in adjacent gate/up groups of eight, and then converts
scales to blocked storage. The per-expert global scale remains attached to
the original logical weight. W2 is never interleaved, and no backward operand
is created for this representation.
Mega G8 preparation is not a blanket admission of offset-backed execution;
the source topology restrictions below still apply.

<a id="publish-source-rows"></a>
## Publish source rows

Training MXFP8 publishes BF16 and quantizes after peer gather. NVFP4 quantizes
on the source rank before publication to reduce symmetric-memory and
peer-fabric traffic in both modes. `_stage_blockscaled_dispatch` writes each
row as:

```text
[packed E2M1 qdata | natural E4M3 block scales | FP32 inverse global scale]
```

The row stride is padded to 16 bytes. The activation quantizer writes the FP32
inverse scale directly into its strided slot in this row; there is no temporary
scale tensor or follow-up copy. Peer dispatch producers gather the packed row
directly and do not reconstruct a full BF16 row in peer HBM.

<a id="run-expert-compute"></a>
## Run expert compute

The staged and Mega policies use the same high-level kernel families as MXFP8:

```text
packed row in local/peer symmetric HBM
  -> gather packed qdata/scales/global inverse
  -> TMA qdata and scale atoms to SMEM
  -> tcgen05 FP4 x FP4, FP32 TMEM accumulation
  -> apply activation and weight global-scale factors
  -> BF16 expert output
  -> direct peer combine stores
```

The staged policy uses the distributed dispatch and fused
SwiGLU/W2/combine kernels. It keeps concatenated W13 and a dense BF16 H1
boundary. The Mega policy uses the chunked forward kernel. The source
configuration helpers jointly select FP4 MMA atom widths, token/feature tiles,
producer layouts, epilogue transfers, and legal pipeline depths. Under
`SWAP_AB`, `BLOCK_SIZE_N` names the token tile, not the public output feature
dimension. Deriving pipeline depth does not shrink that width to fit a
different scratch estimate.
The optional clamped-SwiGLU specialization is evaluated in FP32 registers
before the W2 activation is quantized and does not change the packed NVFP4
dispatch or scale layouts.

The shared `blockscaled_gemm_tiles` helpers build qdata/scale TMA operands;
the matching distributed and Mega entrypoint modules pass the complete
problem and pipeline records to `gemm_warp_roles` and the tile scheduler.
Those components share one descriptor, layout, and readiness contract rather
than maintaining separate training and inference arithmetic.

<a id="interleaved-mega-inference"></a>
## Interleaved Mega inference

Interleaved W13 places the gate/up values needed by SwiGLU beside one another
within the FC13 epilogue:

```text
gate[0:8], up[0:8], gate[8:16], up[8:16], ...
  -> FC13 FP32 accumulation
  -> plain or clamped SwiGLU
  -> dense BF16 H2 staging
  -> per-token amax/global scale and per-16-value block quantization
  -> W2
```

NVFP4 needs a global scale for the entire activation row. A feature-axis
epilogue tile cannot see the row's complete amax, so it stages dense H2.
Row-quantization producers wait for that source tile, reread the complete row,
reduce its block maxima, and publish packed H2, blocked E4M3 scales, and the
FP32 inverse row scale. MXFP8 has no per-token global scale and can quantize
interleaved H2 in the epilogue without this dense staging.

Eligible Mega configurations use a scheduler relay to publish shared chunk
work records to TMA, MMA, and epilogue roles. FC2 consumes H2 only after its
quantization counter is ready. Relay records do not replace the row-readiness
protocol, descriptor publication, or caller-owned peer barriers.

<a id="physical-scales-and-scratch"></a>
## Physical scales and scratch

Qdata packs two feature values per byte, but blocked scale storage still uses
128-row atoms with four scale columns. A token tile of `B` rows occupies
`ceil(B / 128) * 128` physical scale rows. Compact inference routing has a
32-row logical alignment; each expert's qdata and scale prefixes are expanded
to complete physical tiles/pages. A 96-row token tile therefore occupies a
128-row scale page, and a 192-row tile occupies 256 scale rows. These tails
are storage, not additional admitted routes.

For device-only Mega, the source config and static layout builder share this
geometry. The [device-resolved memory plan](memory_planner.md#build-and-inspect-a-plan)
covers all supported runtime T values, not only the largest T, because smaller
tiles can have larger scale-page expansion. The launcher checks the resolved
physical byte requirement before accessing scratch; a short allocation raises
instead of selecting a narrower kernel. VMM-configured execution retains its
GPU-offset scratch planner and source admission bounds.

Device-only Mega inference uses aligned, disjoint views of the context's
existing byte allocation for both concatenated and interleaved W13:

- gathered X and H2 packed rows, blocked scales, and inverse row scales;
- full dense BF16 H1 for concat, or dense BF16 H2 staging for interleaved
  FC13's full-row amax reduction;
- small never-stored descriptor operands and unused column-quantization views;
- the source counter splits and tensor-map table.

There is no second arena and no CPU read of the GPU activation-offset table.
Internal source admission selects host or device tensor maps. Host maps, when
legal, are constructed from these actual static views; static views with
partial scale pages or unsupported host geometry use device preparation.
The unused per-call `get_forward_plan()` and received-count clone are skipped;
GPU routing counts and pointers still drive descriptor preparation and compute.

Offset-backed operands with host tensor maps are rejected before launch.
Staged inference and every VMM-configured inference context retain the
GPU-offset scratch planner; VMM uses device tensor maps even when particular
operands lie in device-backed sections. There is no public descriptor/storage
selector or per-operand VMM mapping policy. None of these inference views
creates saved activation state, and concat remains the default weight layout.

Training always uses GPU-planned offsets. Its hybrid layout combines dense
BF16 saved operands with transient packed NVFP4 row bundles. Full-save retains
`DQ(Q(X_gathered))`, H1, and H3; recompute retains local BF16 X. The row-bundle
decoder converts Q(X) and Q(H2) before their scratch is reused.

G8 with GPU offsets requires the source high-throughput producer topology.
Its padded allocation row bound is at least `16 * num_sms`, and the
`(D, F) = (4096, 4096)` square shape is excluded. The narrow topology lacks the
required buffer-output path and raises before launch. `COMPACT_DECODE_ROWS`,
explicit `SCHEDULER_RELAY`, and `ACTIVATION_RING_CHUNKS` also require static
storage and are excluded with offsets; automatic relay is ineligible and
native tails are normalized off. These constraints do not create public
mapping or descriptor switches, and no VMM/G8 runtime support is implied
outside the admitted source combinations.

Compact G8 additionally requires host descriptors, not merely static views;
the internal source admission rule still decides whether that descriptor path
is legal.

<a id="publication-and-readiness"></a>
## Publication and readiness

Quantization writers publish their generic-proxy global stores to the async
proxy before releasing a tile counter that gates TMA. The wide-row producer
topology joins all writers at its quant-group barrier before an elected lane
releases the counter; other topologies retain their corresponding source
fences and synchronization. A consumer-side fence cannot publish another
thread's writes.

Counter plans use logical receive capacity, not the larger compact qdata/scale
extent. Ordinary device tensor-map preparation resets the work counter and
the extra quantization counters before compute. If preparation is folded into
the Mega kernel, builders publish each expert/GEMM descriptor pair through a
separate ready counter, and consumers acquire all pairs. Supplied counters
must be precleared. With no supplied counters, static host-tensormap storage
is zeroed on the caller stream; ordinary device-tensormap preparation keeps
its own reset path.

Before shared epilogue storage is reused, the issuing warp drains pending TMA
store reads with `cp_async_bulk_wait_group(0, read=True)` and the epilogue
writers synchronize. This protects the shared-memory lifetime; it is distinct
from peer publication and is not an overlap measurement. The common Mega
backward
[lane-parallel WGRAD readiness protocol](mxfp8_kernel_design.md#quantization-readiness)
belongs to MXFP8 training. NVFP4 uses native BF16 DGRAD/WGRAD after operand
reconstruction instead.

<a id="compare-with-mxfp8"></a>
## Compare with MXFP8

| Property | MXFP8 training | NVFP4 training |
| --- | --- | --- |
| qdata | E4M3, one byte/value | E2M1, two values/byte |
| block scale | E8M0 per 32 values | E4M3 per 16 values |
| row global scale | none | FP32 inverse per activation row |
| expert global scale | none | FP32 per expert weight tensor |
| dispatch traffic | BF16 source rows | packed qdata + scales |
| backward | block-scaled staged or Mega | native BF16 kernels |
| activation buffer | quantized saved state + scratch | dense saved state + packed scratch |

<a id="preserve-numerical-and-layout-invariants"></a>
## Preserve numerical and layout invariants

FP4 packing orientation, round-to-nearest E4M3 block scales and E2M1 qdata,
global-scale application order, and peer-store ordering are part of the kernel
contract. Activation and weight inverse global scales are applied in the
source operation order, not combined into a reassociated factor. A generic
transpose, `repeat_interleave`, or scale-layout reinterpretation is not
equivalent to the prepared operand. Changes to these rules require numerical,
capture, communication, and storage-lifetime validation.
