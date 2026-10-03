# NVFP4 Inference Execution

This guide follows one NVFP4 Dist-MoE invocation from weight preparation to
the returned BF16 output. For the numeric format, scale swizzle, and warp-level
MMA design, see [`nvfp4_kernel_design.md`](nvfp4_kernel_design.md).

NVFP4 uses only forward propagation (**FPROP**). It has no input-gradient
(**DGRAD**) or weight-gradient (**WGRAD**) path in this release.

## Outline

- [Public contract](#public-contract)
- [Reuse the two-rank example](#reuse-the-two-rank-example)
- [Prepare weights](#prepare-weights)
- [Plan context memory](#plan-context-memory)
- [Follow the forward pass](#follow-the-forward-pass)
- [Understand the packed dispatch row](#understand-the-packed-dispatch-row)
- [Choose staged or Mega](#choose-staged-or-mega)
- [Capture and replay CUDA graphs](#capture-and-replay-cuda-graphs)
- [Understand failure boundaries](#understand-failure-boundaries)

<a id="public-contract"></a>
## Public contract

NVFP4 is an inference-only mode. The following fragment assumes the imports,
tensor names, dimensions, device, and expert-parallel group from the README
quickstart:

```python
policy = dist_moe.BlockScaledConfig(
    format=dist_moe.BlockScaledFormat.NVFP4,
    pipeline="staged",  # or "mega"
)
config = dist_moe.Config(
    num_local_input_tokens=T,
    hidden_dim=D,
    intermediate_dim=F,
    top_k=K,
    num_experts=E,
    max_moe_layers_per_activation_slot=1,
    num_activation_slots=0,
    block_scaled=policy,
    inference=True,
)
context = dist_moe.create_context(group=ep_group, config=config)
prepared_w13 = dist_moe.prepare_block_scaled_weight(w13_E2FD, policy, inference=True)
prepared_w2 = dist_moe.prepare_block_scaled_weight(w2_EDF, policy, inference=True)
y_TD = dist_moe.routed_experts(
    x_TD,
    topk_expert_ids_TK,
    topk_scores_TK,
    prepared_w13,
    prepared_w2,
    context,
)
```

`D` and `F` must satisfy the NVFP4 alignment and maximum-dimension checks.
Autograd is rejected. Weights must be prepared before execution, and prepared
weights remain caller-owned for repeated eager calls or CUDA-graph replay.
Unlike MXFP8 evaluation, wrapping a training context in `torch.no_grad()` does
not enable NVFP4: the format requires the static inference context because its
routing, packed dispatch, weight contract, and memory layout are
inference-specialized.

<a id="reuse-the-two-rank-example"></a>
## Reuse the two-rank example

Use the same routing assignment as the
[BF16 running example](bf16_execution.md#running-example): rank 0 owns `E0`
and `E1` and computes `E0[A,C], E1[B,D]`. Unlike training, each source rank
quantizes its tokens before publishing them:

```text
rank 0: BF16 A, B -> packed NVFP4 rows A, B in rank 0 dispatch HBM
rank 1: BF16 C, D -> packed NVFP4 rows C, D in rank 1 dispatch HBM

rank 0 gathers packed A, C for E0 and packed B, D for E1
  -> W13 -> SwiGLU -> W2
  -> BF16 route outputs scattered to the owners of A, B, C, and D
  -> rank 0 reduces the two route outputs for A and B
```

There is no saved-state or reverse flow. The packed dispatch rows and local
scratch may be overwritten after the invocation; only the returned BF16 token
outputs and caller-owned prepared weights survive.

<a id="prepare-weights"></a>
## Prepare weights

`dist_moe.prepare_block_scaled_weight()` calls `_prepare_nvfp4_weight_operands()` for
each grouped expert weight:

1. `nvfp4_weight_global_scale()` computes one FP32 global scale per expert.
2. `expand_global_scale_for_rows()` creates the row-oriented scale input used
   by the quantizer.
3. `quantize_block_scaled()` writes packed E2M1 qdata and CUBLAS-blocked E4M3
   block scales.
4. The reciprocal expert scale is computed once and retained because the CuTe
   GEMM consumes the inverse form directly.

The resulting `dist_moe.PreparedWeight` owns:

| Field | Shape and dtype | Purpose |
| --- | --- | --- |
| `source` | grouped BF16 weight | Logical weight identity; no WGRAD is produced in inference. |
| `fprop_data` | packed FP4, two values per byte | Tensor-core FPROP operand. |
| `fprop_scale` | CUBLAS-blocked E4M3 | Per-16-value FPROP scales in the kernel-native swizzle. |
| `global_scale` | FP32 `[E_local]` | Per-expert quantization scale retained for inspection and ownership. |
| `global_scale_inv` | FP32 `[E_local]` | Reciprocal consumed directly by FPROP kernels. |
| `dgrad_data`, `dgrad_scale` | `None` | NVFP4 has no backward path. |

Preparation is outside `dist_moe.routed_experts()`. Reusing the prepared object therefore
does not requantize weights during each invocation or CUDA-graph replay.
Non-strict `make_fx` captures the same registered block-scaled forward without
introducing autograd state.

<a id="plan-context-memory"></a>
## Plan context memory

`dist_moe.create_context()` derives the NVFP4 planner geometry from the public format:

- one byte stores two FP4 activation values;
- one E4M3 block scale covers 16 logical values;
- each packed activation row also stores one FP32 inverse global scale;
- row storage is padded to the dispatch alignment required by peer loads.

Inference uses `num_activation_slots=0`. The activation slot is therefore
empty and all temporary offsets refer to reusable scratch. With VMM disabled,
scratch is device HBM. With VMM enabled, one stable virtual allocation is
ordered from low to high addresses as a device-backed prefix, host-backed
overflow, and hot device-backed scratch suffix. Planner offsets address the
same logical bytes regardless of physical backing. VMM changes residency, not
the execution graph or packed row format.

<a id="follow-the-forward-pass"></a>
## Follow the forward pass

The public call crosses one registered operation. Private Python helpers may
change without changing this contract:

```text
dist_moe.routed_experts()
  -> dist_moe::block_scaled_forward
  -> staged or Mega NVFP4 kernels
```

The forward then executes the following phases.

For the running example, phases 1-3 publish packed `A/B`, discover the peer
rows `C/D`, and build rank 0's `E0[A,C], E1[B,D]` pointer order. Phases 4-5
assign scratch and compute those expert rows. Phase 6 reduces only rank 0's
local token results `A/B`; rank 1 performs the symmetric work for `C/D`.

| Phase | Callable | Reads | Writes | Residency | Saved or recomputed | Synchronization |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `_stage_blockscaled_dispatch` | local BF16 `x_TD` | packed qdata, E4M3 scales, and FP32 inverse row scale | local rank's symmetric dispatch HBM | not saved; inference scratch is reusable after the call | none inside quantization |
| 2 | `_ep_barrier` | dispatch and routing publication state | symmetric-memory signal state | device | no backward state | EP device barrier before peer reads |
| 3 | `dist_dispatch_routing` | global top-k IDs and published row addresses | per-expert counts, gather pointers, scatter pointers | local HBM metadata referring to local or peer HBM | invocation-local only | no CPU synchronization |
| 4 | `get_forward_plan` | received-row counts and static model geometry | device offsets for the fixed scratch layout | device HBM | scratch-only plan; nothing is retained for backward | one fixed-topology planner launch |
| 5a | staged distributed W13 GEMM | peer packed rows, prepared W13 qdata/scales/global inverse | W13 result and fused activation state at planner offsets | scratch; peer input remains in symmetric HBM | overwritten by the next invocation | kernel-internal producer/consumer ordering |
| 5b | staged fused SwiGLU/W2/combine | staged activation, prepared W2 qdata/scales/global inverse, scatter pointers | combined expert output | local output plus peer symmetric combine stores | no recompute path exists | device barrier before peer combine consumption |
| 5c | Mega fused forward | the same packed rows, both prepared weights, routing metadata, and scratch offsets | combined expert output | scratch and peer symmetric combine HBM | no backward state or recompute | fixed chunk-pipeline ordering and peer barrier |
| 6 | `_async_forward_postprocess` | combined rows and top-k scores | BF16 `y_TD` | local HBM | returned output only | no host synchronization |

Only one of the staged pair or the Mega launch executes. Both consume the same
prepared-weight layout, routing metadata, scratch plan, and packed symmetric
dispatch rows.

<a id="understand-the-packed-dispatch-row"></a>
## Understand the packed dispatch row

Each source token is quantized directly into its rank-local symmetric dispatch
buffer. No BF16 staging copy is published first:

```text
byte 0
  | packed FP4 qdata: D / 2 bytes
  | natural E4M3 block scales: D / 16 bytes
  | FP32 inverse token-global scale: 4 bytes
  | alignment padding to the next 16-byte row boundary
```

Routing gather pointers identify rows in local or peer symmetric HBM. A
consumer GEMM follows the pointer, loads the packed row, and applies its token
and expert global-scale factors during FP32 accumulation. The row remains in
the producing rank's symmetric allocation; it is not copied into a second
peer-owned BF16 tensor.

<a id="choose-staged-or-mega"></a>
## Choose staged or Mega

The staged path separates distributed W13 dispatch from fused
SwiGLU/W2/combine. Its intermediate values live at offsets returned by the
scratch planner and are overwritten by the next invocation after the required
device ordering.

The Mega path pipelines W13, SwiGLU, W2, and combine in one chunked forward
launch. It uses the same arithmetic, prepared weights, peer pointers, and
planner-owned scratch but keeps intermediate tiles within the fused pipeline
where possible.

Neither path saves activations, routing snapshots, DGRAD operands, or WGRAD
operands for backward. The returned BF16 output is the only user-visible
result.

<a id="capture-and-replay-cuda-graphs"></a>
## Capture and replay CUDA graphs

The context, prepared weights, input tensors, routing tensors, output tensor,
symmetric buffers, and scratch allocation must retain stable addresses across
capture and replay. Device-side routing and planning can change values while
their tensor shapes and launch topology remain fixed. A replay may therefore
use new activations and routing decisions without allocating a new dispatch
row buffer or changing the captured kernel sequence.

<a id="understand-failure-boundaries"></a>
## Understand failure boundaries

Construction or invocation fails before kernel execution when:

- `inference=False` is paired with NVFP4;
- a dimension violates NVFP4 alignment or supported maxima;
- either weight is not a prepared NVFP4 operand;
- required global scales or reciprocal scales are absent or malformed;
- input tokens differ from the fixed `num_local_input_tokens`; or
- the selected scratch capacity cannot represent the configured receive bound.

These checks prevent an unsupported training path, an incorrectly packed row,
or an undersized scratch plan from reaching the kernels.
