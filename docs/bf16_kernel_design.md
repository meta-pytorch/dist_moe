# BF16 Kernel Design

This guide explains the GPU kernels behind the BF16 execution flow. Read
[BF16 end-to-end execution](bf16_execution.md) first if you need the public API,
saved-state, or backward-lifetime model.

**CTA** means cooperative thread array, the unit of thread-block scheduling.
**TMA** moves tensor tiles between global memory (GMEM) and shared memory
(SMEM). **TMEM** is tensor memory used by Blackwell tensor cores.

## Outline

- [Scope](#scope)
- [Persistent grouped GEMM](#persistent-grouped-gemm)
- [CTA roles](#cta-roles)
- [SwiGLU](#swiglu)
- [WGRAD](#wgrad)
- [Synchronization](#synchronization)
- [Numerical invariants](#numerical-invariants)

<a id="scope"></a>
## Scope

The BF16 path fuses expert-parallel data movement with grouped expert GEMMs.
Routing scores and top-k IDs are inputs; routing itself is not part of the
expert kernels. Matrix products use BF16 operands and FP32 tensor-core
accumulation. Outputs are converted to BF16 at the epilogue unless WGRAD is
configured for FP32 output.

The kernel families are:

| File | Responsibility |
| --- | --- |
| `kernels/dist_grouped_gemm.py` | Peer gather + grouped GEMM dispatch and grouped GEMM + peer scatter combine |
| `kernels/grouped_gemm.py` | Local grouped FPROP/DGRAD/WGRAD implementation |
| `kernels/swiglu_epilogue.py` | Interleaved SwiGLU epilogue helpers |
| `kernels/tile_scheduler.py` | Persistent grouped tile assignment and producer/consumer handoff |

<a id="persistent-grouped-gemm"></a>
## Persistent grouped GEMM

One launch covers all local experts. `split_sizes[e]` gives the number of rows
owned by expert `e`; a persistent scheduler maps each work ID to:

```text
(expert, expert-local M tile, output N tile)
```

Rows never cross expert boundaries. Tail rows and columns are predicated.
Host-side configuration chooses one- or two-CTA clusters, tile M/N/K, pipeline
depth, epilogue subdivision, and static or dynamic scheduling. These choices
are compile-time CuTe parameters and form a performance-sensitive
specialization contract.

<a id="cta-roles"></a>
## CTA roles

The ordinary grouped GEMM divides work into pipelined roles:

```text
TMA producer     GMEM -> staged A/B SMEM, descriptor updates, full barriers
MMA consumer     SMEM -> tcgen05 MMA -> FP32 TMEM accumulators
epilogue group   TMEM -> registers -> C SMEM -> 16-byte/TMA GMEM stores
idle warps       participate only in required cluster/warpgroup ordering
```

The distributed dispatch kernel adds gather warps. They load source rows using
`gather_ptrs`; a pointer may address the local symmetric dispatch buffer or a
peer rank's dispatch buffer. Gathered BF16 rows are written to the local
activation buffer and signal a per-M-tile counter. The TMA producer waits for
that counter before staging the row tile.

```text
local/peer symmetric dispatch HBM
  -> gather warps
  -> x_gathered BF16 in local activation buffer
  -> TMA to SMEM
  -> tcgen05 BF16 x BF16, FP32 TMEM accumulation
  -> BF16 epilogue in SMEM
  -> local activation buffer (dispatch result)
```

The combine kernel reverses the communication direction. It reads expert-local
rows, runs the grouped GEMM, and resolves each route's `scatter_ptr` in its
epilogue. Sixteen-byte stores write the BF16 result directly to the owning
rank's symmetric combine HBM.

```text
h2 BF16 in local activation buffer
  -> TMA to SMEM
  -> tcgen05 BF16 x BF16, FP32 TMEM accumulation
  -> BF16 in SMEM
  -> 16-byte stores to local or peer symmetric combine HBM
```

<a id="swiglu"></a>
## SwiGLU

`swiglu_fwd` and `swiglu_bwd` are Triton kernels over planner-provided buffer
offsets. The nonlinear arithmetic is evaluated in FP32 registers and stored as
BF16:

```text
h1 BF16 in local HBM
  -> gate/up values promoted to FP32 registers
  -> sigmoid and multiply in FP32
  -> h2 BF16 in local HBM
```

The backward reads saved or recomputed `h1`, combines `grad_h2` with the
derivative in FP32 registers, and stores interleaved BF16 `grad_h1`.

The same kernels implement clamped SwiGLU when requested. They clamp the gate
above by `limit`, clamp the up branch to `[-limit, limit]`, add one to the up
branch, and evaluate the sigmoid with multiplier `alpha`; the arithmetic and
derivative remain in FP32 registers before BF16 stores.

<a id="wgrad"></a>
## WGRAD

`grouped_gemm_wgrad` computes, per expert:

```text
dW[e] = dY[e].T @ X[e]
```

Both operands may be ordinary tensors or device byte offsets into the shared
activation buffer. The kernel reads those offsets on device, so dynamic save or
recompute policy causes no CPU synchronization. MMA accumulates in FP32 TMEM.
The epilogue either overwrites a BF16/FP32 destination or uses reduce-add stores
for configured in-place accumulation.

<a id="synchronization"></a>
## Synchronization

- Symmetric-memory barriers establish visibility after source publication and
  before consuming peer-written combine output.
- Gather-completion counters order peer loads before TMA consumes a local tile.
- SMEM full/empty mbarriers order TMA and MMA stages.
- TMEM full/empty mbarriers order MMA and epilogue stages.
- Two-CTA configurations use cluster barriers and multicast masks for their
  cooperative tile.

These are GPU ordering operations. They are not CPU/GPU synchronizations.

<a id="numerical-invariants"></a>
## Numerical invariants

Byte parity depends on preserving the selected tile configuration, row order,
expert grouping, FP32 accumulation, BF16 conversion point, SwiGLU expression,
barrier order, and peer-store order. Refactoring host orchestration must not
change any of those properties.
