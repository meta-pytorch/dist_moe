# NVFP4 Execution

This guide follows NVFP4 Dist-MoE inference and training. Forward uses real
NVFP4 tensor-core GEMMs. Training recreates quantized forward operands and
then reuses the BF16 input-gradient (**DGRAD**) and weight-gradient
(**WGRAD**) sequence. For the numeric format, scale swizzle, and warp-level
MMA design, see [`nvfp4_kernel_design.md`](nvfp4_kernel_design.md).

## Outline

- [Public contract](#public-contract)
- [Reuse the two-rank example](#reuse-the-two-rank-example)
- [Prepare weights](#prepare-weights)
- [Plan context memory](#plan-context-memory)
- [Follow the forward pass](#follow-the-forward-pass)
- [Follow the backward pass](#follow-the-backward-pass)
- [Understand the packed dispatch row](#understand-the-packed-dispatch-row)
- [Choose staged or Mega](#choose-staged-or-mega)
- [Capture and replay CUDA graphs](#capture-and-replay-cuda-graphs)
- [Understand failure boundaries](#understand-failure-boundaries)

<a id="public-contract"></a>
## Public contract

NVFP4 supports training with native BF16 weights and inference with native
BF16 or explicitly prepared NVFP4 weights. Expert ID `-1` marks an inactive
route: it is absent from communication and expert kernels, and final reduction
masks the unwritten combine slot so an all-padding token produces an exact
zero output row. The following fragment assumes the imports,
tensor names, dimensions, device, and expert-parallel group from the README
quickstart:

```python
policy = dist_moe.BlockScaledConfig(
    format=dist_moe.BlockScaledFormat.NVFP4,
    pipeline="staged",  # or "mega"
)
config = dist_moe.Config(
    max_num_local_input_tokens=T,
    hidden_dim=D,
    intermediate_dim=F,
    top_k=K,
    num_experts=E,
    max_moe_layers_per_activation_slot=1,
    num_activation_slots=1,
    block_scaled=policy,
    inference=False,
)
context = dist_moe.create_context(group=ep_group, config=config)
y_TD = dist_moe.routed_experts(
    x_TD,
    topk_expert_ids_TK,
    topk_scores_TK,
    w13_E2FD.flatten(1, 2),
    w2_EDF,
    context,
)
y_TD.float().sum().backward()
```

`D` and `F` must satisfy the NVFP4 alignment and maximum-dimension checks.
Training applies straight-through-estimator semantics to quantization:
forward quantizes native BF16 weights, backward quantizes them again and
dequantizes the result for BF16 DGRAD. Prepared NVFP4 weights are rejected in
training.

For inference, use `inference=True` and `num_activation_slots=0`. Native BF16
weights are quantized inside every invocation, including every CUDA-graph
replay. Explicitly prepared weights remain caller-owned for repeated eager
calls or replay and avoid that repeated quantization.

Staged inference retains the asynchronous source's 128-row routing and launch
selection. Compact automatic decode tuning applies to the matched Mega
operand layout, not to staged offset-backed scratch.

<a id="reuse-the-two-rank-example"></a>
## Reuse the two-rank example

Use the same routing assignment as the
[BF16 running example](bf16_execution.md#running-example): rank 0 owns `E0`
and `E1` and computes `E0[A,C], E1[B,D]`. In both modes, each source rank
quantizes its tokens before publishing them:

```text
rank 0: BF16 A, B -> packed NVFP4 rows A, B in rank 0 dispatch HBM
rank 1: BF16 C, D -> packed NVFP4 rows C, D in rank 1 dispatch HBM

rank 0 gathers packed A, C for E0 and packed B, D for E1
  -> W13 -> SwiGLU -> W2
  -> BF16 route outputs scattered to the owners of A, B, C, and D
  -> rank 0 reduces the two route outputs for A and B
```

Inference has no saved-state or reverse flow. Training saves either dense
backward operands or the original local BF16 input plus routing/planner state;
backward uses the same peer paths in reverse for DGRAD.

<a id="prepare-weights"></a>
## Prepare weights

Native BF16 execution and `dist_moe.prepare_block_scaled_weight()` both call
`_prepare_nvfp4_weight_operands()` for each grouped expert weight:

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
| `source` | grouped BF16 metadata | Logical weight identity; readable storage is not required and no WGRAD is produced. |
| `fprop_data` | packed FP4, two values per byte | Tensor-core FPROP operand. |
| `fprop_scale` | CUBLAS-blocked E4M3 | Per-16-value FPROP scales in the kernel-native swizzle. |
| `global_scale` | FP32 `[E_local]` | Per-expert quantization scale retained for inspection and ownership. |
| `global_scale_inv` | FP32 `[E_local]` | Reciprocal consumed directly by FPROP kernels. |
| `dgrad_data`, `dgrad_scale` | `None` | Prepared NVFP4 is inference-only. |

Explicit preparation is outside `dist_moe.routed_experts()`. Reusing the
prepared object therefore does not requantize weights during each invocation
or CUDA-graph replay. Passing BF16 weights performs the identical preparation
inside the registered forward operation instead. Training repeats that
preparation in backward and exactly dequantizes the regenerated operands to
BF16; it never saves packed weights across the autograd boundary.
Non-strict `make_fx` captures the same registered block-scaled forward without
introducing autograd state.

Planning uses the context's logical dimensions. Packed W2 has physical shape
`[E_local, D, F/2]`, but its logical intermediate dimension remains `F`; packing
must not change routing capacity, scratch sizing or automatic kernel selection.

For supported Mega inference, `fuse_swiglu=True` prepares W13 in adjacent gate/up groups
of eight. Quantization happens before that rearrangement, so each canonical
16-value scale group is preserved. Qdata and natural scale rows are rearranged
together, then the scales are converted to blocked storage. W2 is never
interleaved. The default remains concatenated for both pipelines; interleaved
preparation is rejected for staged execution and training. Preparation itself
is a registered operation with metadata-only FakeTensor outputs.
Offset-backed G8 execution has additional source admission limits; preparation
does not promise that every VMM-configured Mega shape can consume this layout.
It requires the high-throughput producer topology: the padded allocation row
bound must be at least `16 * num_sms`, and `(D, F)` must differ from
`(4096, 4096)`. Compact decode, explicit scheduler relay, and activation rings
are excluded with offsets. A shape selecting the narrow topology is rejected
before launch instead of receiving a separate mapping or static workaround.

<a id="plan-context-memory"></a>
## Plan context memory

`dist_moe.create_context()` derives the NVFP4 planner geometry from the public format:

- one byte stores two FP4 activation values;
- one E4M3 block scale covers 16 logical values;
- each packed activation row also stores one FP32 inverse global scale;
- row storage is padded to the dispatch alignment required by peer loads.

Inference uses `num_activation_slots=0`. The activation slot is therefore
empty and scratch is reusable. With VMM disabled, scratch is device HBM:
Mega inference binds aligned static views and skips the unused per-call
`get_forward_plan()` launch and received-count clone. The internal source
rules select host or device tensor maps; static views do not force host maps.
Staged inference retains GPU-offset planning.

Every VMM-configured inference context retains the GPU-offset scratch planner
and device tensor maps. One stable virtual allocation is
ordered from low to high addresses as a device-backed prefix, host-backed
overflow, and hot device-backed scratch suffix. Planner offsets address the
same logical bytes regardless of physical backing. The context's VMM
configuration selects this planner-backed execution; operands are not
individually reclassified or separately mapped. The packed row format and
arithmetic are unchanged by that choice.

Training uses the hybrid activation plan. A full-save decision retains
`X_bw = DQ(Q(X_gathered))`, the BF16 H1 emitted by the real NVFP4 W13 GEMM,
and BF16 H3. A recompute decision retains only the original local BF16 X.
Both decisions keep fixed routing, planner, and postprocessing state. H2 and
all packed activation/weight tensors are transient scratch.

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
| 1 | `_stage_blockscaled_dispatch` | local BF16 `x_TD` | packed qdata, E4M3 scales, and FP32 inverse row scale | local rank's symmetric dispatch HBM | packed rows are transient | none inside quantization |
| 2 | `_ep_barrier` | dispatch and routing publication state | symmetric-memory signal state | device | routing state is retained for training | EP device barrier before peer reads |
| 3 | `dist_dispatch_routing` | global top-k IDs and published row addresses | per-expert counts, gather pointers, scatter pointers | local HBM metadata referring to local or peer HBM | retained for training | no CPU synchronization |
| 4 | planner binding | received-row counts and static model geometry | activation-slot and scratch offsets; static views only for device-only inference Mega | context-owned memory | training dynamically selects full-save or recompute | planner-backed paths run `get_forward_plan`; device-only inference Mega skips that launch and the unused count clone |
| 5a | staged distributed W13 GEMM | peer packed rows and dynamic/prepared W13 operands | BF16 H1 and optional dense X_bw | activation slot or scratch | full-save retains H1 and X_bw | kernel-internal producer/consumer ordering |
| 5b | staged fused SwiGLU/W2/combine | H1, dynamic/prepared W2 operands, scatter pointers | packed H2 scratch and BF16 H3 | activation slot, scratch, and peer combine HBM | full-save retains H3; H2 is never retained | device barrier before peer combine consumption |
| 5c | Mega fused forward | packed rows, both dynamic/prepared weights, routing metadata, and planner/static views | BF16 H1/H3 and optional dense X_bw | activation slot, scratch, and peer combine HBM | training retains the same logical full-save state as staged | fixed chunk-pipeline ordering and peer barrier |
| 6 | `_async_forward_postprocess` | combined rows and top-k scores | BF16 `y_TD` | local HBM | returned output only | no host synchronization |

Only one of the staged pair or the Mega launch executes. Both consume the same
prepared row format, routing metadata, and packed symmetric dispatch rows.
Scratch binding follows the context/pipeline contract above; G8 is Mega-only
and remains subject to its source topology and offset restrictions.

<a id="follow-the-backward-pass"></a>
## Follow the backward pass

The registered backward regenerates both NVFP4 weight operands from the native
BF16 weights, exactly dequantizes them, and delegates the dense gradient
sequence to the BF16 implementation.

If the device planner selected recompute, backward republishes the saved local
X as packed NVFP4, repeats staged W13 and W2 forward compute, and materializes
`X_bw`, H1, and H3. This staged reconstruction is used for both original
forward pipelines. If it selected full-save, those values already occupy the
activation slot.

For both decisions, backward reruns SwiGLU and the NVFP4 H2 quantizer, then
dequantizes the result as
`H2_bw = DQ(Q(SwiGLU(H1)))`. The existing BF16 sequence then performs:

```text
grad_Y, scores, H3
  -> score/reduction backward -> grad_scores, grad_H3
  -> BF16 W2 DGRAD with DQ(Q(W2)) -> grad_H2
  -> BF16 W2 WGRAD with H2_bw -> grad_W2
  -> SwiGLU backward with H1 -> grad_H1
  -> BF16 W13 DGRAD with DQ(Q(W13)) -> route grad_X
  -> BF16 W13 WGRAD with X_bw -> grad_W13
  -> peer scatter and top-k reduction -> grad_X
```

The planner rewind, WGRAD ownership/callback policy, postprocessing backward,
and output-gradient ownership are the BF16 contracts. Packed Q(X), Q(H2), and
Q(W) never cross the autograd boundary.

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
launch. It uses the same arithmetic, weight operands, peer pointers, and
context-owned scratch but keeps intermediate tiles within the fused pipeline
where possible. Device-only Mega uses static views; VMM-configured Mega keeps
the device-offset plan and device tensor maps. The G8 offset-backed source path
requires its admitted producer topology rather than remapping selected operands
to make an unsupported shape run.

Inference saves no activations or routing snapshots. Training exposes the same
fixed private state schema for either pipeline and retains only the dense
operands selected by the device planner.

<a id="capture-and-replay-cuda-graphs"></a>
## Capture and replay CUDA graphs

The context, input and weight tensors, routing tensors, output tensor,
symmetric buffers, and scratch allocation must retain stable addresses across
capture and replay. Prepared weights remain stable operands; native BF16
weights are quantized again by the captured preparation launches on every
replay. Device-side routing and planner-backed offsets can change values while
their tensor shapes and launch topology remain fixed. Device-only Mega instead
reuses static views with its captured automatic descriptor path.
A replay may therefore
use new activations and routing decisions without allocating a new dispatch
row buffer or changing the captured kernel sequence.

<a id="understand-failure-boundaries"></a>
## Understand failure boundaries

Construction or invocation fails before kernel execution when:

- a dimension violates NVFP4 alignment or supported maxima;
- a training weight is not native BF16;
- an inference weight is neither native BF16 nor a prepared NVFP4 operand;
- a prepared weight's required global scales or reciprocal scales are absent
  or malformed;
- input tokens exceed `max_num_local_input_tokens`; or
- the selected scratch capacity cannot represent the configured receive bound.

These checks prevent an invalid weight contract, incorrectly packed row, or
undersized activation/scratch plan from reaching the kernels.
