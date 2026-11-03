# Post-Expert Processing

Dist-MoE applies post-expert processing after the W2 expert projection has
produced one row per selected route and before those rows are reduced into one
output row per input token.

## Outline

- [Choose a policy](#choose-a-policy)
- [Follow one token](#follow-one-token)
- [Place fused RMSNorm](#place-fused-rmsnorm)
- [Select a kernel schedule](#select-a-kernel-schedule)
- [Configure the typed policy](#configure-the-typed-policy)
- [Understand saved state and recomputation](#understand-saved-state-and-recomputation)
- [Follow backward](#follow-backward)
- [Use observers and callbacks](#use-observers-and-callbacks)
- [Check supported combinations](#check-supported-combinations)

<a id="choose-a-policy"></a>
## Choose a policy

`dist_moe.ExecutionOptions.experts_output_postprocess` selects one of three
behaviors:

| Value | Forward behavior | Backward behavior | Graph support |
| --- | --- | --- | --- |
| `None` | Multiply route rows by router scores and reduce top-k. | Differentiate score weighting and reduction. | Registered op, FakeTensor, non-strict `make_fx`, activation checkpointing, CUDA graphs. |
| Callable | Apply the callback to each route row, then perform ordinary score weighting and reduction. | Recompute the callback under autograd and differentiate it with the reduction. | Eager only. |
| `dist_moe.RMSNormPostprocess` | Fuse RMSNorm, router-score weighting, and top-k reduction. | Use the fused RMSNorm/reduction backward. | Callback-free policies use the registered graph boundary; observers select eager execution. |

Routing scores and top-k expert IDs are inputs to Dist-MoE. Routing itself is
not part of postprocessing and is not computed by these kernels.

An expert ID of `-1` disables one route. Plain reduction and typed RMSNorm use
the IDs to exclude invalid operands before reductions. Some RMSNorm schedules
stage route rows before clearing inactive register fragments; they do not rely
on multiplying undefined values by a zero score. Inactive routes contribute
zero even when their expert output, saved reciprocal RMS, or score contains
NaNs. Their input and score gradients are also zero. Python callbacks and
observers are eager extension points:
when padding is possible, set
`ExecutionOptions.zero_out_padded_callback_inputs=True` to zero invalid `h3`
rows before user code. With the option disabled, callbacks require a caller
guarantee that no route is padded.

<a id="follow-one-token"></a>
## Follow one token

Consider one token routed to two experts. W2 returns route rows `h3_0_D` and
`h3_1_D`, and the router supplied scores `s0` and `s1`.

Without a postprocess policy, Dist-MoE computes:

```text
output_D = s0 * h3_0_D + s1 * h3_1_D
```

With scalarless fused RMSNorm, each route is normalized before weighting:

```text
n0_D = h3_0_D / sqrt(mean(h3_0_D**2) + eps)
n1_D = h3_1_D / sqrt(mean(h3_1_D**2) + eps)
output_D = s0 * n0_D + s1 * n1_D
```

The forward may retain the two `h3` rows and their reciprocal RMS values or
ask the activation planner to recreate them in backward. Backward receives one
`grad_output_D`, expands it through both scores, applies the RMSNorm derivative
to each route, and also computes gradients for `s0` and `s1`. The expert rows
remain separate until the final weighted reduction.

<a id="place-fused-rmsnorm"></a>
## Place fused RMSNorm

Let `h3_MD` be the route-wise W2 output, where `M = T * K`. The fused policy
computes:

```text
h3_MD
  -> optional input scaling by (weight_D + gain_center)
  -> RMSNorm over D
  -> multiply by topk_scores_TK
  -> reduce K routes
  -> output_TD
```

The normalization is therefore applied before the top-k combine reduction. It
is not a separate RMSNorm over the already combined `[T, D]` tensor. Both the
scalarless form and the optional gamma form execute inside the fused
RMSNorm/combine kernel; gamma does not require a separate pointwise launch.

<a id="select-a-kernel-schedule"></a>
## Select a kernel schedule

The typed policy accepts BF16 expert rows, FP32 scores and normalized values,
top-k two or eight, and hidden dimensions from 256 through 131072 in multiples
of 128. The reduced output must preserve the Dist-MoE input dtype.

Forward uses the following families without changing the policy's placement:

| Family | Behavior |
| --- | --- |
| General fused | Process top-k rows with vectorized loads, optional input scaling, and the requested ordinary or Kahan sum-of-squares. Wide rows use column clusters. |
| Parallel top-k | Normalize expert rows concurrently, then combine them with the standalone first-pair accumulation order. |
| Prefetched or streamed parallel top-k | Reuse the parallel reduction order while prefetching rows or pipelining token tiles through shared memory. |

Parallel schedules are selected only for the supported BF16/FP32/BF16 shapes on
SM10x/SM11x. Kahan accumulation, input-scale gamma, and a direct saved-input
copy select the general family. Batch size, width, top-k, input alignment, and
copy ownership select among the eligible families; there is no separate public
tuning switch.

`require_bitwise=True` preserves the standalone normalization and score
reduction orders. Backward selects a register exact schedule when it already
matches those orders, or a chunked exact schedule that independently reproduces
the normalization projection and the score gradient's 1024-column, four-warp
reduction. Large top-k-eight batches at supported widths can share one
`grad_output` row across several expert rows within a CTA. BF16 upstream
gradients remain BF16 in that shared-memory path and are promoted exactly
inside the kernel.

`require_bitwise=False` permits the source-selected non-bitwise forward
clustering and the optimized backward at compute capability 10 or newer.
Non-bitwise training remains limited to dimensions at most 12288; wider
dimensions are inference-only. Older device capabilities retain the applicable exact or
standalone backward path.

All selected fused RMSNorm/combine families receive the route-ID operand.
Register loads, shared-memory reuse, column shards, and statistics-only
recomputation preserve the same inactive-route contract. The package does not expose stochastic
rounding.

<a id="configure-the-typed-policy"></a>
## Configure the typed policy

This fragment assumes the imports, tensors, and context from the README
quickstart:

```python
options = dist_moe.ExecutionOptions(
    experts_output_postprocess=dist_moe.RMSNormPostprocess(
        eps=1e-6,
        norm_output_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        require_bitwise=True,
    )
)
output_TD = dist_moe.routed_experts(
    x_TD,
    topk_expert_ids_TK,
    topk_scores_TK,
    w13_E2FD,
    w2_EDF,
    context,
    options=options,
)
```

`dist_moe.RMSNormPostprocess` fields have the following roles:

| Field | Contract |
| --- | --- |
| `eps` | Positive epsilon added to the mean square. |
| `norm_output_dtype` | Precision of normalized route rows before score weighting. |
| `output_dtype` | Final reduced output dtype; it must match the Dist-MoE output contract. |
| `require_bitwise` | Preserve standalone reduction order when true; false permits faster numerically equivalent reductions. |
| `weight` | Optional contiguous `[D]` gamma applied to the RMSNorm input. It is inference-only. |
| `gain_center` | Scalar added to `weight`, including zero-centered-gamma policies. |
| `use_kahan` | Enable compensated sum-of-squares in every normalization launch. |
| `observe_expert_output_fn` | Optional read-only observer of pre-normalization route outputs. |
| `observe_expert_output_grad_fn` | Optional read-only observer of route outputs and their gradients. |
| `recompute_rstd` | Omit reciprocal RMS values from forward state and recompute them from the saved expert output in backward. |

The default is scalarless RMSNorm. Supplying `weight` computes
`rmsnorm((weight + gain_center) * h3)` and is supported only for inference;
training would require a gradient for gamma that the fused Dist-MoE backward
does not publish.

For inference, pass the contiguous gamma directly in the typed policy. This
fragment assumes `inference_context` was created from a config with
`inference=True`:

```python
gamma_D = torch.ones(D, dtype=torch.bfloat16, device="cuda")
weighted_options = dist_moe.ExecutionOptions(
    experts_output_postprocess=dist_moe.RMSNormPostprocess(
        weight=gamma_D,
        gain_center=1.0,
        eps=1e-6,
        norm_output_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
)
output_TD = dist_moe.routed_experts(
    x_TD,
    topk_expert_ids_TK,
    topk_scores_TK,
    prepared_w13,
    prepared_w2,
    inference_context,
    options=weighted_options,
)
```

<a id="understand-saved-state-and-recomputation"></a>
## Understand saved state and recomputation

For training, the selected postprocess participates in the same activation
planner as the expert kernels:

1. W2 produces route-wise `h3_MD`.
2. The postprocess forward returns `output_TD`, optional saved `h3_MD`, optional
   context, and the dtype entering reduction.
3. Plain reduction can fuse a conditional `h3_MD` copy into its reduction launch.
4. A callable saves the pre-callback route output, because backward must replay
   the callback under autograd.
5. Fused RMSNorm saves or recomputes the pre-normalization route output. By
   default it retains reciprocal RMS values `rstd_TK` as compact auxiliary
   context; `recompute_rstd=True` omits that context and regenerates it from
   `h3_MD` during backward.

A direct saved-input destination can receive its copy from the general fused
forward. Planner-owned saved activations use the existing device-offset,
conditionally executed activation-buffer copy. The planner supplies its byte
offset and save/recompute condition; postprocessing neither reads those device
values on the host nor enlarges the saved-state budget to select a copy
schedule. No new output destination or saved-state owner is introduced.

If the planner selects recomputation, W13, SwiGLU, and W2 reconstruct `h3_MD` in
shared scratch before postprocess backward. If it selects saved activations,
`h3_MD` is read from the selected activation slot. Backward always uses the
physical slot ID in the matching forward's produced planner state, even if a
later pipeline stage changes the context's current selection.

The callback-free registered boundary saves `rstd_TK` together with the original
forward's planner state. Selective activation-checkpoint recomputation does not
replace that state with the context's live slot or with a later normalization
result. If reciprocal RMS values were intentionally omitted, backward derives
them from the matching saved or recomputed `h3_MD`.

<a id="follow-backward"></a>
## Follow backward

The resolved postprocess returns three values:

```text
(grad_h3_MD, grad_topk_scores_TK, grad_h3_is_published)
```

- Plain reduction broadcasts `grad_output_TD` through router scores and can
  publish `grad_h3` directly into the symmetric combine buffer.
- A callable replays the callback on detached `h3_MD`, differentiates the callback
  and reduction with autograd, and returns an unpublished `grad_h3`.
- Fused RMSNorm consumes `h3_MD`, optional saved `rstd_TK`, scores, and
  `grad_output_TD`, then produces both route gradients and router-score
  gradients. When `rstd_TK` was omitted, the existing fused backward recomputes
  it from `h3_MD`: eligible shared-gradient schedules compute it in the backward
  launch, while other SM10x/SM11x exact schedules use a statistics-only
  normalization launch without materializing normalized expert rows or a
  temporary combined output. Remaining fallback schedules recompute the
  standalone composition. When supported, backward writes the route gradient
  directly into the requested publication view.

If a variant cannot publish directly, the Dist-MoE runtime performs the one
required copy into the symmetric combine buffer before the expert DGRAD path.

<a id="use-observers-and-callbacks"></a>
## Use observers and callbacks

Observers and arbitrary postprocess callables execute Python and therefore
select the eager autograd path. Observers are read-only: mutating their tensors
violates the execution contract. Callback closure parameters receive gradients
from the nested backward replay, but they are not explicit inputs to the outer
Dist-MoE autograd function.

When padded inputs are zeroed, callback values and derivatives must be
well-defined on those zeroed rows. Clearing their inputs does not make an
arbitrary callback with an undefined derivative at zero safe
for parameter-gradient accumulation.

Callback-free `dist_moe.RMSNormPostprocess` is flattened into tensor and scalar custom-op
arguments. FakeTensor and non-strict `make_fx` see one ordered Dist-MoE operator;
CUDA graphs replay fixed kernels and stable activation/communication-buffer addresses.

<a id="check-supported-combinations"></a>
## Check supported combinations

- BF16 and MXFP8 training: scalarless fused RMSNorm with forward/backward.
- BF16, MXFP8, and NVFP4: scalarless RMSNorm.
- Inference/forward-only execution also accepts input-scale gamma RMSNorm.
- Staged and Mega block-scaled execution: the same typed policy and output
  semantics.
- Arbitrary callables and observers: eager only.
- Weighted RMSNorm training: unsupported; use a callback when eager execution
  and ordinary autograd ownership are acceptable.
