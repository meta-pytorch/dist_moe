# Post-Expert Processing

Dist-MoE applies post-expert processing after the W2 expert projection has
produced one row per selected route and before those rows are reduced into one
output row per input token.

## Outline

- [Choose a policy](#choose-a-policy)
- [Follow one token](#follow-one-token)
- [Place fused RMSNorm](#place-fused-rmsnorm)
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

If the planner selects recomputation, W13, SwiGLU, and W2 reconstruct `h3_MD` in
shared scratch before postprocess backward. If it selects saved activations,
`h3_MD` is read from the selected activation slot. Backward always uses the
physical slot ID in the matching forward's produced planner state, even if a
later pipeline stage changes the context's current selection.

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
  it from `h3_MD`. When supported, it writes the route gradient directly into
  the requested publication view.

If a variant cannot publish directly, the Dist-MoE runtime performs the one
required copy into the symmetric combine buffer before the expert DGRAD path.

<a id="use-observers-and-callbacks"></a>
## Use observers and callbacks

Observers and arbitrary postprocess callables execute Python and therefore
select the eager autograd path. Observers are read-only: mutating their tensors
violates the execution contract. Callback closure parameters receive gradients
from the nested backward replay, but they are not explicit inputs to the outer
Dist-MoE autograd function.

Callback-free `dist_moe.RMSNormPostprocess` is flattened into tensor and scalar custom-op
arguments. FakeTensor and non-strict `make_fx` see one ordered Dist-MoE operator;
CUDA graphs replay fixed kernels and stable activation/communication-buffer addresses.

<a id="check-supported-combinations"></a>
## Check supported combinations

- BF16 and MXFP8 training: scalarless fused RMSNorm with forward/backward.
- BF16, MXFP8, and NVFP4 inference: scalarless or input-scale gamma RMSNorm.
- Staged and Mega block-scaled execution: the same typed policy and output
  semantics.
- Arbitrary callables and observers: eager only.
- Weighted RMSNorm training: unsupported; use a callback when eager execution
  and ordinary autograd ownership are acceptable.
