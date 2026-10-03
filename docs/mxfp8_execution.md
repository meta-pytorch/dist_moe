# Asynchronous MXFP8 End-To-End Execution

This guide describes MXFP8 E4M3 training with the activation-buffer-backed
asynchronous runtime. The synchronous implementation is not exported. See
[MXFP8 kernel design](mxfp8_kernel_design.md) for CTA and layout details.

Forward expert matrix multiplication is abbreviated **FPROP**. Backward uses
**DGRAD** for input gradients and **WGRAD** for weight gradients. **Staged** and
**Mega** name two launch pipelines; they do not name different numeric formats.

## Outline

- [Configure MXFP8](#configure-mxfp8)
- [Reuse the two-rank example](#reuse-the-two-rank-example)
- [Prepare and reuse weights](#prepare-and-reuse-weights)
- [Evaluate with a training context](#evaluate-with-a-training-context)
- [Use inference specialization](#use-inference-specialization)
- [Understand the graph boundary](#understand-the-graph-boundary)
- [Follow common forward setup](#follow-common-forward-setup)
- [Follow the staged forward](#follow-the-staged-forward)
- [Follow the Mega forward](#follow-the-mega-forward)
- [Understand saved state](#understand-saved-state)
- [Follow backward](#follow-backward)
- [Review memory residency](#review-memory-residency)

<a id="configure-mxfp8"></a>
## Configure MXFP8

This integration fragment assumes the imports and tensor names from the README
quickstart and defines `T`, `D`, `F`, `K`, `E`, and `L` from
the model configuration:

```python
config = dist_moe.Config(
    num_local_input_tokens=T,
    hidden_dim=D,
    intermediate_dim=F,
    top_k=K,
    num_experts=E,
    max_moe_layers_per_activation_slot=L,
    block_scaled=dist_moe.BlockScaledConfig(
        format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
        pipeline="staged",  # or "mega"
    ),
)
context = dist_moe.create_context(group=ep_group, config=config)
output_TD = dist_moe.routed_experts(
    x_TD,
    topk_expert_ids_TK,
    topk_scores_TK,
    w13_E2FD,
    w2_EDF,
    context,
)
```

`x`, route scores, and the returned output are ordinary PyTorch tensors.
Block-scaled activation bundles and planner offsets are context-owned buffer
state. Routing remains external.

`dist_moe.plan_memory()` exposes the padded row bound used by the planner and
grouped GEMMs. For example, 16 local tokens, top-2 routing, two local experts,
EP=2, and capacity factor 1 produce 32 logical rows. Each local expert can add
at most 127 padding rows, so the conservative capacity is 256:

```python
config = dist_moe.Config(
    num_local_input_tokens=16,
    hidden_dim=256,
    intermediate_dim=256,
    top_k=2,
    num_experts=4,
    max_moe_layers_per_activation_slot=1,
    device_scratch_capacity_factor=1.0,
    block_scaled=dist_moe.BlockScaledConfig(),
)
plan = dist_moe.plan_memory(config, ep_size=2)
assert plan.balanced_recv_rows == 32
assert plan.device_scratch_capacity_rows == 256
```

Execution does not materialize all 256 rows unconditionally. Routing records
the actual per-expert counts on device, independently rounds each count to 128,
and packs only that padded requirement into the activation buffer. The capacity
factor sets the maximum scratch bound; it does not replace dynamic activation
packing.

`dist_moe.BlockScaledKernelConfig` is an expert-only override for the final CuTe
launch parameters. Leave `kernel_config=None` for the production shape-aware
presets. When supplying it, every field must describe one complete tuned
configuration; partial overrides are intentionally unsupported:

```python
kernel_config = dist_moe.BlockScaledKernelConfig(
    num_ctas=2,
    block_m=128,
    block_n=128,
    block_k=128,
    num_smem_buffers=3,
    num_c_stages=2,
    num_tmem_buffers=2,
    num_tile_buffers=2,
    epilogue_subtile=64,
    overlapping_accum=True,
    swap_ab=False,
    kloop_unroll=1,
)
config = dataclasses.replace(
    config,
    block_scaled=dataclasses.replace(
        config.block_scaled,
        kernel_config=kernel_config,
    ),
)
```

<a id="reuse-the-two-rank-example"></a>
## Reuse the two-rank example

This guide uses the routing assignment from the
[BF16 running example](bf16_execution.md#running-example): rank 0 owns `E0`
and `E1` and receives expert rows `E0[A,C], E1[B,D]`; rank 1 owns `E2` and
`E3`. The communication pattern is unchanged. MXFP8 changes the representation
created after each peer gather:

| Point in the flow | Representation | Why it exists |
| --- | --- | --- |
| Source publication | BF16 token row | Peers gather the original training activation. |
| W13 input on expert rank | Row MXFP8 qdata/scales | FPROP consumes it without a second source-side copy. |
| Saved W13 input | Column MXFP8 qdata/scales when retained | W13 WGRAD consumes the activation in transposed orientation. |
| W13 output | BF16 `h1` | SwiGLU evaluates its nonlinear expression in FP32 registers. |
| W2 input | Row MXFP8 plus optional column MXFP8 | W2 FPROP and WGRAD need different scale orientations. |
| Route output | BF16 | It is scattered to the token owner's combine buffer. |

Thus rank 0 still computes `E0[A,C]` and `E1[B,D]`, but its local activation
buffer holds quantized operand bundles as well as the BF16 nonlinear boundary.
If the selected slot cannot retain a complete column bundle, the forward saves
the BF16 layer input and backward regenerates that bundle in scratch.

<a id="prepare-and-reuse-weights"></a>
## Prepare and reuse weights

`dist_moe.prepare_block_scaled_weight(weight, policy)` quantizes one grouped
BF16 weight into a `dist_moe.PreparedWeight`:

| Field | Role |
| --- | --- |
| `source` | Logical high-precision tensor and autograd gradient edge |
| `fprop_data` | Grouped E4M3 qdata used by forward |
| `fprop_scale` | E8M0 scales in forward orientation |
| `dgrad_data` | Same MXFP8 qdata object as FPROP |
| `dgrad_scale` | Separate E8M0 scales for transposed DGRAD consumption |

The two scale layouts differ because the contraction axis differs. Sharing the
qdata does not make the scales interchangeable.

Refilling an existing prepared weight quantizes into temporary results and
copies qdata plus both scale layouts into the caller-owned tensors. This keeps
the prepared tensor identities stable across FSDP unshard cycles without
retaining a full-size private workspace in the persistent prepared state.

```python
prepared_w13 = dist_moe.prepare_block_scaled_weight(w13_E2FD, config.block_scaled)
prepared_w2 = dist_moe.prepare_block_scaled_weight(w2_EDF, config.block_scaled)
output_TD = dist_moe.routed_experts(
    x_TD,
    topk_expert_ids_TK,
    topk_scores_TK,
    prepared_w13,
    prepared_w2,
    context,
)

# Refill the same graph-stable storage after a later weight unshard.
dist_moe.prepare_block_scaled_weight(w13_E2FD, config.block_scaled, out=prepared_w13)
dist_moe.prepare_block_scaled_weight(w2_EDF, config.block_scaled, out=prepared_w2)
```

Prepared weights are optional. If BF16 weights are passed directly, the
forward weight-preparation kernel creates the qdata and both scale layouts and
the registered custom-op state retains them only when backward will consume
them. A framework such as FSDP should prepare once per actual unshard and pass
prepared operands so PP microbatches can reuse them when the unsharded weight
remains resident.

<a id="evaluate-with-a-training-context"></a>
## Evaluate with a training context

`torch.no_grad()` does not select MXFP8 inference specialization. It keeps the
training FPROP representation, padding, routing, staged/Mega choice, and
postprocess semantics, while setting the invocation's backward-retention policy
to false. Consequently:

- routing omits backward gather pointers and the per-rank count snapshot;
- FPROP emits row-quantized operands only, not column-quantized WGRAD bundles;
- the planner places row operands and required dense H1 storage in shared
  scratch;
- no dispatch input, route output, recomputation predicate, or postprocess
  context is saved; and
- activation-slot offsets, saved-byte counters, and layer IDs remain unchanged.

Dynamic training weight preparation still uses the combined quantizer so its
FPROP qdata and scale values remain bitwise identical to grad-enabled training.
Its DGRAD scale result is not retained after the forward. Removing that extra
scale computation requires a separately qualified FPROP-only kernel mode; it is
not emulated with inference's different row-only quantization.

Repeated no-grad calls can therefore be followed immediately by a grad-enabled
forward/backward on the same context without `context.reset()`. Non-strict
`make_fx` records the registered forward with a static false retention flag.
CUDA graphs require separate captures for grad-enabled training and no-grad
evaluation because the latter intentionally omits backward-only work.

<a id="use-inference-specialization"></a>
## Use inference specialization

MXFP8 inference uses the same staged or Mega forward kernels with
`inference=True` and `num_activation_slots=0`. It saves no autograd or backward
state, and the complete activation buffer is reusable scratch. Prepared
inference weights contain FPROP qdata and scales only; DGRAD fields are absent.
Dynamic BF16 weights are also accepted and quantized for the call. Supplying
tensors that require gradients to an inference context is rejected.

<a id="understand-the-graph-boundary"></a>
## Understand the graph boundary

The public path uses ordered registered operations. The enclosing Python
autograd bridge preserves the logical BF16 weight edges and selects either a
functional WGRAD operation or an accumulating one:

```text
dist_moe.routed_experts
  -> dist_moe::block_scaled_forward
  -> registered autograd bridge
  -> dist_moe::block_scaled_backward
     or dist_moe::block_scaled_backward_accumulate_
```

Forward returns the model output plus fixed-shape private state. The setup
context resolves caller-supplied versus dynamically produced operands without
reading tensor data. FakeTensor implementations create metadata-only qdata,
scale, routing, and planner tensors.

Non-strict `make_fx` uses this registered boundary for both training and
inference. Training-context no-grad evaluation and inference both return empty
private state placeholders, but only inference changes the static execution
specialization.

<a id="follow-common-forward-setup"></a>
## Follow common forward setup

1. Local route IDs are published to symmetric routing HBM.
2. Training publishes BF16 `x` rows to symmetric dispatch HBM.
3. `dist_dispatch_routing()` produces padded per-expert row counts and GPU
   gather/scatter pointers.
4. `get_forward_plan()` assigns activation offsets and decides whether later
   backward will use saved column operands or recompute them.
5. The chosen staged or Mega pipeline executes with the same route order and
   activation ownership.
6. The selected post-expert stage consumes the BF16 W2 combine output. The
   default performs score weighting and top-k reduction; a direct callback
   transforms each route first; `dist_moe.RMSNormPostprocess` fuses route-wise RMSNorm,
   weighting, and reduction in its own kernel.

Training quantizes after peer gather. Source-side quantization would not remove
the column-oriented WGRAD work: the expert rank still needs scales grouped
along received-row order, which is known only after routing and padding.

<a id="follow-the-staged-forward"></a>
## Follow the staged forward

The staged pipeline realizes the running example in two distributed compute
launches. First it gathers `A, C, B, D` into rank 0's expert-major order while
quantizing the W13 operands. Then it transforms the local W13 results through
SwiGLU and W2 and scatters each route result back to its source rank.

| Order | API or kernel | Reads | Writes | Residency and save policy | Synchronization |
| ---: | --- | --- | --- | --- | --- |
| 1 | `dist_dispatch_routing()` | local routing IDs in symmetric HBM | GPU row counts and gather/scatter pointer tables | metadata is local HBM and saved for backward | follows the dispatch publication barrier |
| 2 | `get_forward_plan()` | row counts and device planner state | byte offsets in the selected activation slot and scratch | offsets and recompute predicate are saved | launch ordered; no host read |
| 3 | `dist_blockscaled_grouped_gemm_fprop_dispatch()` | BF16 route rows in local/peer symmetric dispatch HBM; prepared W13 in local HBM | BF16 `h1` plus row/column MXFP8 bundles | local activation buffer; column W13 input is saved when the slot fits, otherwise regenerated | pointer-driven peer reads after the barrier |
| 4 | `dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine()` | BF16 `h1` and prepared W2 in local HBM | BF16 route outputs in peer combine HBM plus W2 column input | local activation buffer and peer symmetric HBM; `h1`/column W2 follow the same planner decision | remote stores complete before the combine barrier |
| 5 | `scale_and_sum()` | local combine rows and router scores | BF16 `output_TD` | ordinary local HBM; only fixed routing/planner state reaches autograd | consumes combine rows after the EP barrier |

### W13 dispatch

`dist_blockscaled_grouped_gemm_fprop_dispatch()` gathers BF16 route rows from
local or peer symmetric HBM. Its producer creates:

- row MXFP8 qdata + E8M0 scales for W13 FPROP;
- column MXFP8 qdata + E8M0 scales when W13 WGRAD will consume the activation;
- the BF16 gathered view required by the save/recompute policy.

TMA stages qdata and scale tiles into SMEM. `tcgen05` multiplies MXFP8
activations by prepared MXFP8 W13 and accumulates in FP32 TMEM. The epilogue
stores BF16 `h1` in the local activation buffer.

### SwiGLU, W2, and combine

`dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine()` reads BF16 `h1`,
evaluates SwiGLU in FP32 registers, creates row-quantized `h2` for W2 FPROP and
the column operand needed by W2 WGRAD, then runs W2. Its epilogue converts FP32
accumulators to BF16 and scatters route rows directly to peer symmetric combine
HBM. `scale_and_sum()` produces the local BF16 output.

`dist_moe.Config(activation="swiglu_clamped")` selects the clamped formula in
this fused producer. Gate/up clamping and the sigmoid are evaluated
in FP32 before `h2` is quantized; backward and conditional recomputation receive
the same static `swiglu_alpha` and `swiglu_limit` values.

<a id="follow-the-mega-forward"></a>
## Follow the Mega forward

`chunked_mega_blockscaled_grouped_gemm_fprop_swiglu_fwd()` fuses W13, SwiGLU,
W2, and combine publication into one chunked pipeline. It produces the same
logical BF16 output and the same planner-owned saved state. Mega is a launch
topology choice, not a different quantization or memory API.

For the running example, the logical states remain `E0[A,C], E1[B,D]`; Mega
only pipelines their W13, activation, W2, and peer-store tiles inside one
launch. It does not alter routing order, saved-state ownership, or the final
BF16 combine rows.

| Order | API or kernel | Reads | Writes | Residency and save policy | Synchronization |
| ---: | --- | --- | --- | --- | --- |
| 1 | `dist_dispatch_routing()` + `get_forward_plan()` | symmetric routing HBM and local planner state | pointer tables, capacity, and Mega offset bundle | metadata and offsets are local HBM and saved | dispatch barrier precedes peer reads; planner has no host read |
| 2 | `dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_dispatch_combine()` | BF16 route rows from local/peer dispatch HBM and both prepared weights from local HBM | BF16 route outputs in peer combine HBM and both column WGRAD operands | local activation buffer and peer HBM; the complete column bundle is retained or regenerated together | fixed chunk-pipeline ordering followed by combine barrier |
| 3 | `scale_and_sum()` | local combine rows and router scores | BF16 `output_TD` | ordinary local HBM; no dynamically sized tensor reaches autograd | consumes combine rows after the EP barrier |

<a id="understand-saved-state"></a>
## Understand saved state

The graph-visible state contains:

- FPROP qdata and scale plus DGRAD scale for W13 and W2 when prepared in the
  call;
- route IDs/scores, expert row counts, gather/scatter pointers, and per-rank
  counts;
- `need_recompute`, packed forward offsets, activation offsets, recompute
  condition, and the forward-produced snapshot of the resolved activation
  slot;
- the fused RMSNorm rstd context when `dist_moe.RMSNormPostprocess` is selected;
- logical W13/W2 tensors as the normal autograd gradient owners.

For MXFP8, `dgrad_data` aliases `fprop_data`; it is not saved twice. Buffer
activation bundles are addressed by byte offsets and remain in local HBM or
VMM scratch according to the planner.

<a id="follow-backward"></a>
## Follow backward

Read backward as the reverse of the representation table above. Rank 0
receives route-output gradients for `E0[A,C]` and `E1[B,D]`, creates the
row/column forms needed by W2 DGRAD/WGRAD, propagates through SwiGLU, then
creates the corresponding W13 operands. Input-gradient routes return to the
token owners; W13 and W2 gradients remain with rank 0's local expert weights.

`get_backward_plan()` selects saved activation bundles or conditional
recompute scratch. The fixed recompute path regenerates W13 input quantization,
`h1`, SwiGLU, and W2 input quantization when `need_recompute` is true.

The staged kernel order is:

1. Compute `grad_topk_scores` and publish BF16 `grad_h3` in symmetric combine
   HBM. `dist_moe.RMSNormPostprocess` uses its fused backward here; a direct callback is
   replayed under autograd.
2. `dist_blockscaled_grouped_gemm_dgrad_dispatch()` gathers and quantizes
   `grad_h3`, runs W2 DGRAD, and produces the W2 WGRAD column operand.
3. `blockscaled_grouped_gemm_wgrad()` computes W2 WGRAD.
4. `dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine()` evaluates
   SwiGLU backward, quantizes `grad_h1`, runs W13 DGRAD, and scatters route
   input gradients into peer symmetric dispatch HBM.
5. `blockscaled_grouped_gemm_wgrad()` computes W13 WGRAD.
6. `reduce_from_topk()` reduces peer-written route gradients to `grad_x`.

| Order | API or kernel | Reads | Writes | Residency and save policy | Synchronization |
| ---: | --- | --- | --- | --- | --- |
| 1 | `get_backward_plan()` | saved forward offsets and planner state | saved-or-scratch offsets | device metadata; selects retained or recomputed state | launch ordered; no host read |
| 2 | conditional staged forward recompute | saved BF16 `x` | row/column quant bundles and BF16 intermediates | shared scratch only when the saved bundle was omitted | fixed launches use one device predicate and peer barrier |
| 3 | `broadcast_and_scale()` | `grad_output_TD`, router scores, and `h3` | BF16 `grad_h3` plus router-score gradient | symmetric combine HBM and ordinary local HBM | combine barrier precedes peer reads |
| 4 | `dist_blockscaled_grouped_gemm_dgrad_dispatch()` | peer `grad_h3` and W2 DGRAD operands | BF16 `grad_h2` plus its column WGRAD operand | shared scratch | pointer-driven peer reads |
| 5 | `blockscaled_grouped_gemm_wgrad()` | saved/recomputed H2 column data plus scratch `grad_h3` column data | W2 gradient | caller-selected gradient storage | same-stream dependency |
| 6 | `dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine()` | `grad_h2`, `h1`, and W13 DGRAD operands | peer BF16 input gradients plus W13 WGRAD operand | symmetric dispatch HBM and scratch | remote stores complete before dispatch barrier |
| 7 | `blockscaled_grouped_gemm_wgrad()` + `reduce_from_topk()` | column operands and peer route gradients | W13 gradient and `grad_x_TD` | caller gradient storage and ordinary local HBM | reduction follows the dispatch barrier |

Mega uses
`dist_blockscaled_grouped_gemm_dgrad_wgrad_dispatch()` for W2 DGRAD/WGRAD and
`dist_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine()` for SwiGLU,
W13 DGRAD/WGRAD, and peer scatter.

| Order | API or kernel | Reads | Writes | Residency and save policy | Synchronization |
| ---: | --- | --- | --- | --- | --- |
| 1 | `get_backward_plan()` + conditional Mega forward recompute | planner state and saved BF16 input | selected offsets and optional regenerated column bundle | activation slot or one fixed scratch bundle | device predicate and peer barrier |
| 2 | `broadcast_and_scale()` | output gradient, router scores, and `h3` | BF16 `grad_h3` | symmetric combine HBM | combine barrier precedes peer reads |
| 3 | `dist_blockscaled_grouped_gemm_dgrad_wgrad_dispatch()` | peer `grad_h3`, W2 DGRAD operands, and H2 column operands | W2 DGRAD and WGRAD | scratch plus caller gradient storage | one fused launch after combine barrier |
| 4 | `dist_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine()` | `grad_h2`, `h1`, W13 DGRAD operands, and X column operands | W13 DGRAD/WGRAD plus peer input gradients | scratch, caller gradient storage, and symmetric dispatch HBM | remote stores complete before dispatch barrier |
| 5 | `reduce_from_topk()` | peer route gradients | BF16 `grad_x_TD` | ordinary local HBM | follows the dispatch barrier |

The functional registered backward returns fresh W13 and W2 gradients. With
`ExecutionOptions(inplace_wgrad_accum=True)`, later serialized contributions
use `dist_moe::block_scaled_backward_accumulate_` and mutate both existing
`parameter.grad` tensors explicitly. Prepared and dynamically quantized
weights use the same gradient-ownership contract. See
[WGRAD accumulation](wgrad_accumulation.md).

All tensor-core products accumulate in FP32. WGRAD may be returned in BF16 or
FP32 and may reduce-add into an explicit destination. SwiGLU inputs and outputs
remain BF16 at activation-buffer boundaries; its nonlinear arithmetic is FP32 in registers.

<a id="review-memory-residency"></a>
## Review memory residency

```text
ordinary local HBM:
  x, ids, scores, logical weights, prepared weight tensors, final gradients

symmetric local/peer HBM:
  routing IDs, BF16 training dispatch rows, BF16 combine rows and gradients

rank-local activation slot:
  saved input and saved row/column quant bundles selected by planner

rank-local scratch:
  recompute bundles, h1/h2 temporaries, activation gradients, WGRAD operands
```

The context's virtual addresses remain stable under CUDA graph capture.
Device-side offsets decide which physical bytes a launch uses; no dynamically
sized PyTorch activation tensor is returned to autograd.
