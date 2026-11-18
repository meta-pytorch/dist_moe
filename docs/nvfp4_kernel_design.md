# NVFP4 Inference Kernel Design

This guide explains the packed NVFP4 representation and the GPU kernels that
consume it. Read [NVFP4 inference execution](nvfp4_execution.md) first for the
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
- [W4A16 Mega inference](#w4a16-mega-inference)
- [Four Over Six quantization](#four-over-six-quantization)
- [Compare with MXFP8](#compare-with-mxfp8)
- [Preserve numerical and layout invariants](#preserve-numerical-and-layout-invariants)

<a id="supported-surface"></a>
## Supported surface

NVFP4 is inference-only.
`dist_moe.BlockScaledConfig(format=dist_moe.BlockScaledFormat.NVFP4)` requires
`dist_moe.Config(inference=True)`. Autograd, WGRAD, DGRAD, and training
recomputation are rejected.

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

The same weight representation also feeds weight-only (W4A16) Mega inference,
which keeps BF16 activations; see
[W4A16 Mega inference](#w4a16-mega-inference).

<a id="prepare-weights"></a>
## Prepare weights

NVFP4 weights must be prepared before execution. Preparation creates packed
FPROP qdata and CUBLAS-blocked scales. Because NVFP4 is inference-only, the
prepared object does not allocate DGRAD qdata or scales. It retains both the
per-expert global scale and its reciprocal. The runtime consumes the reciprocal
directly, so eager execution and CUDA graph replay do not launch a reciprocal
kernel on every invocation.

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

Training MXFP8 publishes BF16 and quantizes after peer gather. NVFP4 inference
quantizes on the source rank before publication to reduce symmetric-memory and
peer-fabric traffic. `_stage_blockscaled_dispatch` writes each row as:

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
than maintaining separate inference arithmetic.

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
backward [lane-parallel WGRAD readiness protocol](mxfp8_kernel_design.md#quantization-readiness)
belongs to MXFP8 training, not to NVFP4's supported surface.

<a id="w4a16-mega-inference"></a>
## W4A16 Mega inference

`W4A16MegaKernel` (`kernels/w4a16_mega_kernel.py`, launched by
`w4a16_mega_fprop()` in `kernels/w4a16_mega.py`) runs one rank's whole
weight-only expert forward in one persistent launch: the BF16 gather from
peer dispatch buffers, W13 with SwiGLU, W2, and the scatter into the source
ranks' combine buffers. It subclasses `GroupedGemmKernel` for its BF16 tile
geometry, epilogue copies and scheduling helpers; the W4A16 mainloop below
lives in the subclass, so the base kernel is unchanged. See
[weight-only W4A16 execution](nvfp4_execution.md#weight-only-w4a16-mega) for
the call chain.

Weight-only NVFP4 (W4A16) multiplies BF16 activations by NVFP4 weights:
packed E2M1 codes, one E4M3 scale per 16 K values, and one FP32 inverse
global scale (`alpha`) per expert, applied to the FP32 accumulator before
the BF16 conversion:

```text
y[rows of g] = alpha[g] * (x[rows of g] @ (code * block_scale)[g].T)
```

Both GEMMs are `SWAP_AB` FPROP with one-CTA clusters: the MMA M is the
expert's 128-feature tile and N its token tile (`BLOCK_SIZE_N` of 16, 32, 64
or 128).

### Decode into tensor memory

SMEM holds no BF16 weight tile. A TMA stage holds the packed FP4 tile, its
E4M3 scales and the BF16 token tile; the decoded weights live in tensor
memory, after the accumulator columns:

```text
GMEM E2M1 codes + blocked E4M3 scales        GMEM BF16 token rows
        | TMA, one host map each                   | TMA, one host map
        v                                           v
SMEM FP4 + scale stage [b] ------------------ SMEM token stage [b]
        | transform warpgroup i % 2:                |
        | code * scale -> BF16 (exact), tcgen05.st  |
        v                                           v
TMEM BF16 A stage [i % W4A16_A_STAGES] ----> tcgen05.mma, FP32 TMEM accumulator
```

K stage `i` uses TMA buffer `b = i % NUM_SMEM_BUFFERS` and BF16 A stage
`i % W4A16_A_STAGES`. Transform warpgroup `i % 2` waits for buffer `b`,
decodes it in registers, waits on `trans_empty` until the MMA has released
the A stage, stores it with `tcgen05.st`, fences, and arrives on `trans_full`
(128 arrivals). The MMA waits on `trans_full`, issues the stage's MMAs, and
commits both the TMA buffer (`ab_empty`) and the A stage (`trans_empty`).

Each transform thread owns one TMEM lane, one weight row of the tile. A K
stage deeper than 128 is decoded in 128-column passes of 32-value fragments
(two NVFP4 blocks), which bounds the transform registers. The fragment decode
in `kernels/_nvfp4_decode.py` is ported from FlashInfer. With CUDA 13.2 or
newer it converts E2M1 and E4M3 directly to BF16; earlier toolkits decode and
multiply in FP16 and widen to BF16. E2M1 times E4M3 is exact in both, so the
decoded A equals `code * scale` bit for bit.

The transform warpgroups take alternating K stages, so each must see every
TMA buffer at consecutive phases of its own: the SMEM ring depth is always
even. With an odd depth a warpgroup would skip a phase of a buffer's full
barrier, and its parity wait could pass on stale weights.

### Layouts and descriptors

| Operand | Layout |
| --- | --- |
| weights | uint8 `[G, N, K / 2]`, lower K index in the low nibble; host TMA map over `(N, K, G)` |
| block scales | CUBLAS 128x4 blocked E4M3 bytes over `(N, K / 16, G)`; host TMA map, read by the decode through a per-element broadcast view |
| `alpha` | FP32 `[G]`, read once per expert group by the epilogue |
| tokens | BF16 `[GM, K]`, one host TMA map; a tile starts at row `split_prefix + tile_n * BLOCK_SIZE_N` |

Because the weight maps are indexed by expert, no descriptor is patched per
group. Token rows past the group end belong to the next expert; the epilogue
stores only the group's rows.

TMEM holds the accumulators, then `W4A16_A_STAGES` BF16 A stages of
`BLOCK_SIZE_K / 2` columns each: two stages at a K stage of 256 and four at
128. `w4a16_mega_config()` caps the accumulator buffers to fit the 512
columns and picks the deepest even SMEM ring that fits.


### Mega warp roles

| Warps | Role |
| --- | --- |
| 0-3 | first the gather (the BF16 dispatch protocol: 16-row items copied in 256-column chunks), then the epilogue |
| 4 | MMA, A from TMEM |
| 5 | weight TMA: FP4 tiles and scales of every K stage; it never waits on a dependency, so weights stream and decode ahead |
| 6 | token TMA: the gathered rows for W13 tiles, the SwiGLU rows for W2 tiles |
| 7 | idle |
| 8-15 | two transform warpgroups: decode FP4 stages into BF16 TMEM A stages |

Weights and tokens have separate full barriers per TMA buffer, so a token
tile that waits for its gather does not delay the weight stream. The gather
and epilogue warps raise their register limit to 160, warps 4-7 lower theirs
to 96, and the transform warpgroups keep 128.

### Static schedule

Each expert's routed rows form `ceil(rows / BLOCK_SIZE_N)` token tiles.
W13 has `2F / 128` feature tiles per token tile and W2 has `D / 128`. CTA `c`
of a grid of `num_sms` CTAs takes linear tiles `c, c + grid, ...` over all W13
tiles followed by all W2 tiles, so every CTA finishes its W13 tiles before its
first W2 tile:

```text
W13 tile (token tile t): wait gather_done[t] == BLOCK_SIZE_N / 16
  -> MMA over D -> alpha13 -> SwiGLU -> h2 rows -> release h2_done[t] += 1
W2 tile (token tile t):  wait h2_done[t] == 2F / 128
  -> MMA over F -> alpha2 -> 16-byte stores to the scatter pointers
```

A W2 tile waits only on W13 tiles, which every CTA schedules before any W2
tile, and the grid is co-resident, so the waits cannot deadlock. The gather
releases `gather_done` after each 16-row item; the W13 epilogue publishes its
h2 rows to the async proxy and releases `h2_done` once per feature tile. The
token TMA acquires both counters before issuing the tile's loads.

The token tile follows the expected routed rows per local expert (the call's
`T * top_k`, not the capacity): 16 up to 8 rows, 32 up to 32, 64 up to 64,
then 128. K stages are 256 deep below 64-token tiles and 128 from there, where
the BF16 token stages would otherwise leave room for only two SMEM stages.
W13 must be prepared with `fuse_swiglu=True`: rows interleave eight gate rows
and the matching eight up rows, and the W13 epilogue evaluates plain SwiGLU in
FP32 on the staged BF16 gate and up values, as `swiglu_fwd` does, before
writing BF16 h2.

### Mega storage

`w4a16_mega_operand_bytes()` sizes three regions of context scratch: the
gathered BF16 token rows `[rows, D]`, the BF16 SwiGLU rows `[rows, F]`, and
the int32 counters (one gather work counter plus `gather_done` and `h2_done`
per possible token tile, bounded for the smallest tile). Rows are the padded
receive capacity. The launcher clears the counters on the caller stream
before each launch; there is no per-call allocation, device-side descriptor
preparation or host synchronization.

### Validation

`tests/test_w4a16_mega.py` holds the kernel bitwise equal to BF16 `SWAP_AB`
grouped GEMMs on `code * scale` in the same tile (one MMA per 16 K values, in
order) for W13, `swiglu_fwd` and W2, for every token tile, with fresh and stale
context storage. Its per-expert `alpha` values are powers of two, so scaling
the BF16 reference output equals scaling the FP32 accumulator.
`benchmarks/w4a16_mega.py` times the public weight-only path.

<a id="four-over-six-quantization"></a>
## Four Over Six quantization

Weight-only preparation and `quantize_dequantize_weight()` share one
quantizer (`kernels/nvfp4_four_over_six.py`): Four Over Six
([arXiv 2512.02010](https://arxiv.org/abs/2512.02010)) in Transformer Engine's
NVFP4 arithmetic, configured as the W4A16 RL recipe of radixark/miles does.
The block arithmetic is miles' fused CuTe DSL QDQ kernel, vendored unchanged
in `kernels/_miles_nvfp4_qdq.py` (Apache-2.0, recorded in NOTICE).

```text
encode[g] = 2688 / amax(w[g])            (FP32 division; 1 for an all-zero expert)
block b:  s6 = E4M3(amax_b / 6 * encode)   s4 = E4M3(amax_b / 4 * encode)
          codes_k = E2M1(w / (s_k / encode)) for k in (6, 4)
          keep map-to-4 only if its squared error (code * s in FP16) is lower
QDQ:      w_qdq = code * (s * amax / 2688)  (rounded once to the weight dtype)
```

Two launches cover a `[G, N, K]` expert stack in one grid of
`[G, ctas_per_expert]` CTAs of 128 threads, one 1x16 block per thread and
iteration, each expert with its own global scale. The QDQ launch stores the
dequantized block; the quantize launch stores its eight code bytes (the even
column in the low nibble) and its E4M3 scale byte in natural order. Weight-only
preparation then interleaves W13 rows for fused SwiGLU and blocks the scales,
so the prepared codes are exactly the ones the QDQ launch dequantizes. The
inference kernels apply `1 / encode` to the FP32 accumulator, so trainer and
generator differ only in how the global scale rounds.

`tests/test_nvfp4_four_over_six.py` holds both launches bitwise equal to a
PyTorch transcription of the arithmetic, and the prepared operands equal to
the quantize launch.

<a id="compare-with-mxfp8"></a>
## Compare with MXFP8

| Property | MXFP8 training | NVFP4 inference |
| --- | --- | --- |
| qdata | E4M3, one byte/value | E2M1, two values/byte |
| block scale | E8M0 per 32 values | E4M3 per 16 values |
| row global scale | none | FP32 inverse per activation row |
| expert global scale | none | FP32 per prepared weight |
| dispatch traffic | BF16 source rows | packed qdata + scales |
| backward | staged and Mega | unsupported |
| activation buffer | saved state + scratch | scratch only |

<a id="preserve-numerical-and-layout-invariants"></a>
## Preserve numerical and layout invariants

FP4 packing orientation, round-to-nearest E4M3 block scales and E2M1 qdata,
global-scale application order, and peer-store ordering are part of the kernel
contract. Activation and weight inverse global scales are applied in the
source operation order, not combined into a reassociated factor. A generic
transpose, `repeat_interleave`, or scale-layout reinterpretation is not
equivalent to the prepared operand. Changes to these rules require numerical,
capture, communication, and storage-lifetime validation.
