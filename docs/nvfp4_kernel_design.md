# NVFP4 Inference Kernel Design

This guide explains the packed NVFP4 representation and the GPU kernels that
consume it. Read [NVFP4 inference execution](nvfp4_execution.md) first for the
public API and end-to-end lifetime model.

## Outline

- [Supported surface](#supported-surface)
- [Prepare weights](#prepare-weights)
- [Publish source rows](#publish-source-rows)
- [Run expert compute](#run-expert-compute)
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

<a id="prepare-weights"></a>
## Prepare weights

NVFP4 weights must be prepared before execution. Preparation creates packed
FPROP qdata and CUBLAS-blocked scales. Because NVFP4 is inference-only, the
prepared object does not allocate DGRAD qdata or scales. It retains both the
per-expert global scale and its reciprocal. The runtime consumes the reciprocal
directly, so eager execution and CUDA graph replay do not launch a reciprocal
kernel on every invocation.

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
SwiGLU/W2/combine kernels. The Mega policy uses the chunked forward kernel.
Format-specific topology selectors choose FP4 MMA atom widths and producer
layouts; they do not add an alternate allocator or communication protocol.
The optional clamped-SwiGLU specialization is evaluated in FP32 registers
before the W2 activation is quantized and does not change the packed NVFP4
dispatch or scale layouts.

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

FP4 packing orientation, E4M3 scale rounding behavior, global-scale application
order, and direct peer store order are part of the validated numerical contract. A
generic transpose, `repeat_interleave`, or scale-layout reinterpretation is not
equivalent to the prepared operand.
