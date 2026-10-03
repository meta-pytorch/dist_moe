# Pipeline Activation Slots

This guide explains how a pipeline framework maps schedule lifetimes onto a
fixed number of Dist-MoE activation slots. Dist-MoE consumes the assignment; it
does not inspect or modify the pipeline schedule.

## Outline

- [Define the framework contract](#define-the-framework-contract)
- [Choose a slot granularity](#choose-a-slot-granularity)
- [Compute liveness and coloring](#compute-liveness-and-coloring)
- [Work through an interleaved schedule](#work-through-an-interleaved-schedule)
- [Follow runtime ordering](#follow-runtime-ordering)
- [Meet the framework obligations](#meet-the-framework-obligations)

<a id="define-the-framework-contract"></a>
## Define the framework contract

The package does not import a pipeline framework. It accepts a graph-stable
selection that identifies:

```text
(physical activation slot, number of MoE layers using that slot)
```

The framework derives this assignment once from its finalized schedule and
selects it before executing a forward through the public context API. For
stage-microbatch coloring, one slot belongs to one local stage interval:

```python
context.select_activation_slot(
    activation_slot=slot_assignment[(pipeline_stage_index, pipeline_microbatch_index)],
    num_moe_layers_in_slot=moe_layers_per_stage[pipeline_stage_index],
)
output = stage(...)
```

For microbatch coloring, one slot spans every local stage used by that
microbatch, so the bound must include all local MoE layers:

```python
context.select_activation_slot(
    activation_slot=slot_assignment[pipeline_microbatch_index],
    num_moe_layers_in_slot=total_local_moe_layers,
)
output = stage(...)
```

The context selects an immutable one-element view from its device slot-ID table
for each scheduled forward. The planner copies that slot ID into the
forward-produced planner state, and autograd saves the produced snapshot. Its
matching backward therefore retains the original physical slot even after
later forwards or selective activation-checkpoint recomputation select other
views.

A training-context call with no backward consumer does not acquire the selected
slot. Under `torch.no_grad()`, BF16 and MXFP8 place their forward temporaries in
shared scratch and leave slot offsets, saved-byte counters, and layer IDs
unchanged. The framework may run evaluation between completed training steps
and resume training without recoloring or resetting the context. This does not
change the static schedule assignment used by grad-enabled forwards.

PyTorch pipelining supplies the canonical action identity as two reserved
forward kwargs when the stage opts in:

```text
pipeline_stage_index       global logical stage, not physical rank
pipeline_microbatch_index  raw schedule microbatch index
```

The framework maps those integers through an immutable assignment table. The
package never reconstructs schedule IR, subclasses `PipelineStage`, or consults
thread-local/global "current microbatch" state.

<a id="choose-a-slot-granularity"></a>
## Choose a slot granularity

`microbatch` gives one lifetime to all tracked local stages for the same
microbatch. Its interval starts at the earliest local forward and ends after
the last local backward. Each slot must hold every local MoE layer.

`stage_microbatch` gives a distinct lifetime to each `(stage, microbatch)`.
Each slot needs capacity only for the largest number of MoE layers in one local
stage. This is normally more memory-efficient for interleaved schedules even
when it produces more slot IDs.

```text
microbatch bytes = peak live microbatches * all local MoE layers * layer bytes
stage-microbatch bytes = peak live pairs * max local-stage MoE layers * layer bytes
```

The framework should compute both and choose the smaller policy in `auto`
mode. Slot count alone is not the comparison metric.

Configure `num_activation_slots` to the coloring's physical slot count and
`max_moe_layers_per_activation_slot` to the maximum MoE-layer depth of any
resource assigned to one slot. For microbatch coloring, that depth is all local
MoE layers. For stage-microbatch coloring, it is the largest local stage's
MoE-layer count.

<a id="compute-liveness-and-coloring"></a>
## Compute liveness and coloring

The schedule utility consumes the actual finalized local action list. For each
tracked resource:

1. acquire immediately before `FORWARD`;
2. release after `FULL_BACKWARD`;
3. for split backward, retain through `BACKWARD_INPUT` and release after
   `BACKWARD_WEIGHT`;
4. ignore communication and FSDP actions for activation liveness;
5. sort intervals by start and assign the lowest currently free slot.

Duplicate forwards, duplicate terminal backwards, missing backwards, unknown
stages, or out-of-range microbatches fail during setup. Static assignment has
no modulo fallback, and the number of assigned slots must equal analytical peak
liveness.

<a id="work-through-an-interleaved-schedule"></a>
## Work through an interleaved schedule

For PP2/VPP2, physical rank 0 owns logical stages 0 and 2. `0F1` means stage 0
forward for microbatch 1 and `2B0` means stage 2 backward for microbatch 0:

```text
position   0    1    2    3    4    5    6    7    8
action    0F0  0F1  2F0  2F1  0F2  2B0  0F3  2B1  2F2

microbatch live       1    2    2    2    3    3    4    4    4
stage-microbatch live 1    2    3    4    5    4    5    4    5
```

A deterministic stage-microbatch coloring begins:

```text
(0,0)->S0  (0,1)->S1  (2,0)->S2  (2,1)->S3  (0,2)->S4
2B0 releases S2
(0,3)->S2
2B1 releases S3
(2,2)->S3
```

The stage-aware plan uses five smaller slots here. The microbatch plan uses
four larger slots because a microbatch remains live until all of its local
stages finish backward.

<a id="follow-runtime-ordering"></a>
## Follow runtime ordering

```text
finalized schedule
  -> liveness analysis and deterministic coloring
  -> allocate context with resolved slot count and slot depth
  -> stage forward receives canonical IDs
  -> framework selects the assigned immutable device-scalar view
  -> every Dist-MoE layer in the stage advances that slot's layer counter
  -> planner returns offsets plus a snapshot of the resolved slot
  -> autograd saves that forward-produced state
  -> backward consumes and releases the same slot in reverse layer order
```

The Python lookup executes while capturing the fixed PP schedule. Each
Dist-MoE launch records the pointer of its schedule-assigned immutable view;
full-step replay repeats those fixed per-call pointers without Python lookup,
GPU scalar reads, or mutation. `num_moe_layers_in_slot` remains static host
metadata because each scheduled action belongs to one logical stage.

If a schedule aborts after acquiring a slot but before its matching backward,
reset the context before retrying the step. Do not reset a successfully
completed step or while any backward still owns saved activation state:

```python
try:
    pipeline_schedule.step(input_batch)
except BaseException:
    context.reset()
    raise
```

Selecting an unknown slot, exceeding that slot's configured MoE-layer depth,
or attempting a second live forward for the same slot fails instead of falling
back to modulo assignment or another allocation.

<a id="meet-the-framework-obligations"></a>
## Meet the framework obligations

- Analyze the schedule after all compute, communication, and split-backward
  transformations are final.
- Track only stages that actually contain Dist-MoE layers.
- Supply the exact number of MoE layers owned by each logical stage.
- Pass both canonical identifiers on every executed stage forward.
- Reset the context after an aborted pipeline step.
- Keep one context per physical rank; slots partition its saved-activation
  storage, while scratch remains shared and stream ordered.
