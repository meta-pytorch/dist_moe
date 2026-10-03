# WGRAD Accumulation

Weight-gradient (**WGRAD**) kernels can write a new gradient or add directly
into an existing destination. Direct accumulation removes a separate
full-gradient add, but it also makes ownership and execution order part of the
correctness contract.

This guide explains the public `parameter.grad` path first. The final section
covers frameworks that own a separate gradient buffer.

## Outline

- [Decide whether to enable accumulation](#decide-whether-to-enable-accumulation)
- [Use the standard parameter.grad contract](#use-the-standard-parametergrad-contract)
- [Follow first and later contributions](#follow-first-and-later-contributions)
- [Understand the registered operations](#understand-the-registered-operations)
- [Preserve precision and numerics](#preserve-precision-and-numerics)
- [Capture CUDA graphs](#capture-cuda-graphs)
- [Respect the limitations](#respect-the-limitations)
- [Integrate an external gradient buffer](#integrate-an-external-gradient-buffer)
- [Check an integration](#check-an-integration)

<a id="decide-whether-to-enable-accumulation"></a>
## Decide whether to enable accumulation

Use `dist_moe.ExecutionOptions(inplace_wgrad_accum=True)` when several
serialized backward calls contribute to the same unsharded expert gradients.
Pipeline-parallel microbatches inside one FSDP reduction window are the main
example. A single SPMD backward is correct but normally has no earlier WGRAD to
reuse.

```python
options = dist_moe.ExecutionOptions(inplace_wgrad_accum=True)
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

The option is not enabled by default. The caller must serialize writers to the
same expert parameters and retain their gradient storage for the complete
accumulation window.

<a id="use-the-standard-parametergrad-contract"></a>
## Use the standard `parameter.grad` contract

The public path keeps ordinary PyTorch gradient ownership. Before either WGRAD
kernel launches, Dist-MoE resolves the current gradients for both expert
parameters. Each destination must be a contiguous local tensor with the
expected shape, dtype, and device.

For ordinary tensors and views, Dist-MoE follows the view base to the leaf that
owns `.grad`. When a local compute tensor is a view of a distributed parameter,
pass the outer W13 and W2 owners explicitly:

```python
options = dist_moe.ExecutionOptions(
    inplace_wgrad_accum=True,
    wgrad_parameter_owners=(outer_w13, outer_w2),
)
```

Dist-MoE retains weak references to those owners. A public
`torch.distributed.tensor.DTensor` may own `.grad`; the grouped-GEMM kernel
receives its validated local contiguous tensor. The package does not import an
FSDP, pipeline, or training-framework API.

<a id="follow-first-and-later-contributions"></a>
## Follow first and later contributions

The first contribution follows ordinary autograd ownership:

```text
parameter.grad is None
  -> functional registered backward creates W13 and W2 gradients
  -> the autograd function returns both gradients
  -> AccumulateGrad attaches them to parameter.grad
```

Later serialized contributions use the existing storage:

```text
both parameter.grad tensors exist
  -> the autograd function passes them to backward_accumulate_
  -> WGRAD kernels add directly into those tensors
  -> the registered operation returns only input and router-score gradients
  -> the autograd function returns None for both logical weight gradients
  -> AccumulateGrad cannot add the result a second time
```

Both destinations are resolved and validated before either kernel runs. If
only one gradient exists, Dist-MoE uses the functional backward for both
weights and lets ordinary `AccumulateGrad` handle the existing side. It never
partially mutates one destination and then discovers that the other is invalid.

<a id="understand-the-registered-operations"></a>
## Understand the registered operations

BF16 and MXFP8 each expose two private operations with fixed semantics:

```text
functional backward
  inputs:  saved state and output gradient
  outputs: input gradient, router-score gradient, W13 gradient, W2 gradient
  mutation: none

accumulating backward
  inputs:  saved state, output gradient, W13 accumulator, W2 accumulator
  outputs: input gradient, router-score gradient
  mutation: both accumulators
```

The pair for each precision shares one state-reconstruction helper and one
kernel backward implementation. There is no optional destination, conditional
alias, public `beta`, or duplicate algorithm.

Because the accumulators are explicit mutable inputs, non-strict `make_fx`
records the storage dependency and mutation. The Python autograd bridge, not
the registered operation, decides whether logical weight gradients returned to
PyTorch are tensors or `None`.

<a id="preserve-precision-and-numerics"></a>
## Preserve precision and numerics

For standard `parameter.grad` ownership, a non-`None` parameter `grad_dtype`
is authoritative. An explicit `Config.wgrad_dtype` must match that declaration;
otherwise the resolver uses existing gradient storage and then the parameter
dtype. W13 and W2 must resolve to the same BF16 or FP32 dtype before either
kernel writes. The main README contains the complete precedence table.

Fused accumulation is mathematically equivalent to a GEMM followed by a
gradient add, but the rounding point differs:

```text
separate: BF16(previous + BF16(FP32 GEMM))
fused:    BF16(FP32(previous) + FP32 GEMM)
```

The disabled path remains byte-identical to ordinary execution. The enabled
path should be compared with a numerical tolerance appropriate for the
selected destination dtype.

<a id="capture-cuda-graphs"></a>
## Capture CUDA graphs

CUDA graph replay requires fixed control flow and stable gradient addresses.
Materialize both parameter gradients before capturing the steady-state
accumulation graph, then retain them with `zero_grad(set_to_none=False)`.

The first contribution and later contributions have different fixed
topologies:

- first contribution: functional backward returns fresh WGRAD tensors;
- later contribution: `backward_accumulate_` mutates existing tensors.

Do not switch between missing and materialized gradients inside one capture.
Direct non-strict `make_fx` supports the explicit mutation. Full
`torch.compile` through AOTAutograd is outside this contract because generic
custom-operation functionalization copies mutable inputs, defeating the memory
and bandwidth goal.

<a id="respect-the-limitations"></a>
## Respect the limitations

The standard path does not support:

- concurrent WGRAD writers for the same parameter;
- tied or shared expert parameters without a higher-level contribution
  coordinator;
- higher-order differentiation;
- accumulation across separate FSDP reduction windows unless the caller owns
  the complete unsharded lifetime; or
- simultaneous WGRAD postprocessing.

`wgrad_destination_fn`, `inplace_wgrad_accum`, and
`wgrad_postprocess_fn` are mutually exclusive.

<a id="integrate-an-external-gradient-buffer"></a>
## Integrate an external gradient buffer

Some frameworks attach a separate gradient buffer to each parameter. Megatron
Core 0.14, for example, documents a distributed-optimizer `main_grad`
contract. This is a framework-level ownership protocol, not a property of the
Dist-MoE kernels.

An integration that owns the complete lifecycle may provide
`wgrad_destination_fn`. Immediately before W13 or W2 WGRAD, Dist-MoE passes
the projection name, grouped output shape, dtype, and device to the callback.
The callback returns a contiguous destination and an explicit overwrite or
accumulate choice. Dist-MoE writes there and returns no expert-weight gradient
to ordinary autograd.

The framework must then own every related operation:

1. allocate the external gradient in the required precision;
2. retain it across all local contributions;
3. reduce that buffer instead of `parameter.grad`;
4. clear it at the optimizer-step boundary; and
5. integrate clipping, overflow handling, checkpointing, and hooks.

These callbacks execute in the eager extension path. Making them traceable
would require graph-visible ownership operations and a concrete integration
use case.

<a id="check-an-integration"></a>
## Check an integration

Before enabling direct accumulation, verify all of the following:

- W13 and W2 have one serialized writer at a time.
- Both destinations exist before the accumulating backward begins.
- Destination shape, dtype, device, contiguity, and lifetime match the logical
  parameters.
- The reduction starts only after the final local contribution.
- CUDA graph capture uses stable destination addresses.
- Gradient hooks observe the updated storage exactly once per contribution.
- Enabled numerics are compared against the functional path with an explicit
  tolerance.
