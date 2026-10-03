# Activation And Scratch Memory Planner

This guide explains how Dist-MoE turns static model limits into one reusable
memory layout. Start here when choosing activation-slot or scratch capacity.
Read [VMM host-backed scratch](vmm.md) only after the device-only plan is clear.

## Outline

- [Start with ownership](#start-with-ownership)
- [Build and inspect a plan](#build-and-inspect-a-plan)
- [Understand activation-buffer layout](#understand-activation-buffer-layout)
- [Understand dynamic recomputation](#understand-dynamic-recomputation)
- [Add VMM host-backed scratch](#add-vmm-host-backed-scratch)
- [Handle capacity failures](#handle-capacity-failures)
- [Manage the lifecycle](#manage-the-lifecycle)

<a id="start-with-ownership"></a>
## Start with ownership

A `dist_moe.Context` owns three kinds of storage for one fixed model shape and
expert-parallel group:

| Storage | Visibility | Lifetime | Contents |
| --- | --- | --- | --- |
| Routing, dispatch, and combine buffers | Symmetric GPU HBM, peer-addressable | Context | Route IDs, source activations, route outputs, gradients, and signal pads |
| Activation slots | Rank-local GPU HBM | Context; suballocated per forward | Values retained until the matching backward |
| Scratch region | Rank-local device HBM, optionally extended by pinned host VMM pages | Context; reused by each local action | Forward temporaries, recompute temporaries, and activation gradients |

The activation buffer is not CUDA managed memory. With VMM, one stable virtual
address range is backed by explicit device, host, and device physical sections.
Pages do not migrate and the CPU never participates in an execution-time copy.

<a id="build-and-inspect-a-plan"></a>
## Build and inspect a plan

`dist_moe.plan_memory(config, ep_size, device=...)` is the source of truth for
the allocation. It derives:

```text
balanced_recv_rows = num_local_input_tokens * top_k
device_scratch_capacity_rows = topology-aware padded rows at device imbalance factor
total_scratch_capacity_rows = topology-aware padded rows at VMM imbalance factor
minimum activation slot = mandatory inputs for every layer sharing the slot
balanced full-save slot = all eligible state at balanced routing
maximum useful slot = all eligible state within total executable scratch capacity
total device buffer = slot count * selected slot bytes + one device scratch frame
```

BF16 execution packs the actual received rows contiguously; its capacity factor
only reserves the maximum scratch rows. MXFP8 and NVFP4 also allocate the
actual received work dynamically, but each local expert is independently
padded to the grouped-GEMM 128-row multiple. Consequently,
`device_scratch_capacity_rows` can exceed the unpadded
`balanced_recv_rows * device_scratch_capacity_factor`. Integrations obtain the
device and total padded bounds from `dist_moe.plan_memory()` rather than
reimplementing this topology-aware calculation.

`dist_moe.MemoryPlan.explain()` reports the logical bytes. Supplying `device`
also resolves CUDA VMM granularity, physical section padding, and the allocator
size-class adjustment used by `dist_moe.create_context()`.

For example, `num_local_input_tokens=4096`, `top_k=8`, EP 16, device
factor 1.5, and total factor 16 produce 32,768 balanced rows, 49,152
device-scratch rows, and 524,288 total scratch rows. The exact byte counts also
depend on hidden/intermediate dimensions and forward/backward precision; obtain
them from the plan instead of multiplying rows by one activation tensor.

For `D=4096`, `F=14336`, 128 global experts, 32 MoE layers per slot, and one
activation slot, the same configuration produces the following exact logical
plans. The intermediate row uses a caller-selected 16 GiB slot. The balanced
row uses `activation_slot_capacity_factor=1.0`.

| Precision and budget | Saved activations | Device scratch | Host scratch | Total device allocation |
| --- | ---: | ---: | ---: | ---: |
| BF16 minimum | 1,073,741,824 B | 7,449,083,904 B | 72,007,811,072 B | 8,522,825,728 B |
| BF16 16 GiB | 17,179,869,184 B | 7,449,083,904 B | 72,007,811,072 B | 24,628,953,088 B |
| BF16 balanced full-save | 77,309,411,328 B | 7,449,083,904 B | 72,007,811,072 B | 84,758,495,232 B |
| MXFP8 minimum | 1,073,741,824 B | 8,561,811,456 B | 81,282,465,792 B | 9,635,553,280 B |
| MXFP8 16 GiB | 17,179,869,184 B | 8,561,811,456 B | 81,282,465,792 B | 25,741,680,640 B |
| MXFP8 balanced full-save | 90,839,973,888 B | 8,561,811,456 B | 81,282,465,792 B | 99,401,785,344 B |

The minimum is not zero: every training layer input must remain available for
conditional replay. Increasing only slot capacity changes which forward
intermediates are retained; it does not change scratch rows or host-overflow
reservation. MXFP8 scratch is larger because each expert's received rows are
padded independently and each saved WGRAD operand includes quantized data plus
scales.

The public slot policies are mutually exclusive:

```text
both None       -> minimum_activation_slot_bytes
exact bytes     -> internally aligned activation_slot_bytes
capacity factor -> minimum_slot_bytes
                   + factor * (balanced_full_save_slot_bytes - minimum_slot_bytes)
```

`balanced_full_save_slot_bytes` uses the topology-aware padded receive count at
routing factor `1.0`; it does not inherit either scratch factor. Factor `1.0`
therefore covers all eligible state at balanced routing, while `1.5` covers
aggregate optional-state bytes up to 1.5 times the balanced amount. This is a
byte-capacity statement rather than a raw-row ratio because block-scaled
routing pads each local expert independently.

<a id="understand-activation-buffer-layout"></a>
## Understand activation-buffer layout

For `N` activation slots the allocation is:

```text
low address                                                   high address
+-----------+-----------+-----+-------------------------------+
| slot 0 -->| slot 1 -->| ... |              <- shared scratch|
+-----------+-----------+-----+-------------------------------+
```

Each activation slot grows toward higher addresses and has the effective
`activation_slot_bytes` reported by `MemoryPlan`. The public byte request does
not expose an alignment requirement; planning rounds it upward internally.
Aggregate activation storage is exactly `num_activation_slots *
activation_slot_bytes`, so no division remainder exists. Scratch grows toward
lower addresses and is shared because pipeline execution runs one local stage
action at a time.

The planner state is stored in small device tensors:

- `buffer_offsets[N + 1]`: next activation byte for each slot and the current
  scratch end;
- `saved_activation_bytes_per_rank[N * ep_size]`: rank-specific retained bytes
  used for look-ahead recompute decisions;
- `peak_min_free_space[N + 1]`: per-region low-water marks;
- `moe_layer_id[N]`: the next layer position in each selected slot;
- `activation_slot_ids_S[S]`: immutable device scalars for every physical slot;
- `activation_slot_id_1`: the current one-element selector view supplied to the
  next forward.

The selected slot's MoE-layer depth is a static host integer. A pipeline stage
has fixed local layer ownership, so tracing specializes that value while each
scheduled call receives its graph-visible immutable selector view. The planner
copies the selected slot ID into the forward-produced planner state. Autograd
saves that produced snapshot, not the context's live selector, so later
forwards and selective activation-checkpoint recomputation cannot change which
slot the matching backward releases.

Kernel inputs that name an activation-buffer value are device `int64`
byte-offset tensors, not pointer-sized Python integers. A kernel forms
`buffer.data_ptr() + offset` on device. This keeps data-dependent allocation
decisions and addresses inside the captured CUDA graph.

<a id="understand-dynamic-recomputation"></a>
## Understand dynamic recomputation

Forward always reserves enough activation space for the input needed to replay
the layer. The device planner then asks whether retaining the remaining
intermediates would leave enough space for the unvisited layers in the selected
slot. It writes `need_recompute` and all forward offsets on device.

- If space is available, forward retains the WGRAD operands and other required
  intermediates in the activation slot.
- Otherwise, forward retains the layer input and places transient values in
  scratch. Backward observes `need_recompute` and conditionally relaunches the
  fixed forward kernel sequence into scratch.

The host does not branch on `need_recompute`. Both branches remain in one fixed
launch topology, with device predicates controlling work. This is why the
policy composes with CUDA graph capture.

Backward activation gradients always use scratch. A saved forward value remains
in its activation slot until the matching backward has consumed it. After the
layer's final backward work, the planner restores that slot's offset; each slot
is LIFO.

### Forward-only calls on a training context

When a training context is invoked without a backward consumer, the planner
uses a third mode distinct from specialized inference. BF16 and MXFP8 retain
their training-compatible forward kernels and routing policy, but every live
forward operand is placed in shared scratch. The planner does not update
`buffer_offsets`, `saved_activation_bytes_per_rank`, or `moe_layer_id`, and it
does not create a routing-count snapshot for backward.

This mode is selected when grad mode is disabled or no differentiable operand
requires gradients. It permits repeated evaluation calls followed immediately
by ordinary training on the same context without `context.reset()`. Scratch
low-water marks still record peak temporary usage, and exceeding scratch
capacity remains an error. `Config.inference=True` remains a separate static
specialization with its own routing, formats, and allocation geometry.

### Pipeline slot example

A pipeline integration can assign slots to live `(stage, microbatch)` intervals
instead of reserving one slot for every microbatch. Consider a rank whose
schedule has four microbatches but at most three stage-microbatch lifetimes
overlap. If `plan.activation_slot_bytes == 436_207_616`, the two policies use:

```text
microbatch slots:       4 * 436,207,616 = 1,744,830,464 bytes
stage-microbatch slots: 3 * 436,207,616 = 1,308,622,848 bytes
saving:                                      436,207,616 bytes (25%)
```

The integration computes the static coloring before execution, selects an
immutable device slot ID, and supplies the slot's static MoE-layer depth.
The planner copies the device slot ID into forward-produced state for backward,
so both directions use the same graph-stable addresses without a host read in
the captured step. Slot reuse is legal only when the schedule proves that the
two stage-microbatch lifetimes do not overlap.

<a id="add-vmm-host-backed-scratch"></a>
## Add VMM host-backed scratch

With `dist_moe.VmmConfig`, the virtual range has three physical sections:

```text
low address                                                   high address
+----------------------+----------------------+----------------+
| activations + lower  | host-backed scratch  | hot HBM scratch|
| HBM scratch/padding  | pinned system memory | device HBM     |
+----------------------+----------------------+----------------+
                                             scratch grows <-
```

Normal balanced work stays in the hot HBM suffix. Only scratch demand beyond
the device capacity crosses into host-backed pages. Activation storage is
always device-backed. Symmetric communication buffers are separate allocations
and never point into host-backed VMM.

With `dist_moe.VmmConfig(prefetch=True)`, `dist_moe.create_context()` starts the
physical mapping before collective communication-buffer initialization and
consumes it through a dedicated CUDA memory pool when constructing the buffer.
With `prefetch=False`, the same plan is allocated synchronously after the
communication buffers are ready. Context construction owns the temporary
prefetch state and releases it on every failure path. Only one live Dist-MoE
VMM-backed activation buffer is supported per CUDA device.

<a id="handle-capacity-failures"></a>
## Handle capacity failures

The configured device factor is a performance boundary, not a global routing
guarantee. Without VMM, exceeding it exhausts scratch and is an error. With
VMM, rows up to the host factor remain correct but access pinned host pages and
are slower. No allocation can cover an imbalance above the topology maximum:

```text
max rows at one rank = T * ep_size * min(top_k, num_local_experts)
```

Every execution mode passes `total_scratch_capacity_rows` into routing. After
the final local per-expert counts and any grouped-GEMM row padding are known,
the routing prefix or direct-decode kernel prints the required and available
rows and traps if the total capacity is exceeded. This happens before final
routing offsets or pointers are published, so downstream kernels cannot use an
out-of-bounds scratch plan. The check adds no launch, allocation, collective,
or synchronization. It is rank-local: the overflowing rank reports the cause,
while peers may time out or be terminated by the job supervisor. The failed
CUDA context cannot be reused.

`ActivationBuffer.begin_check_overflow()` and `finish_check_overflow()` copy
only diagnostic low-water marks after execution. They are not part of the
forward or backward critical path.

<a id="manage-the-lifecycle"></a>
## Manage the lifecycle

1. Build `dist_moe.Config`.
2. Inspect `dist_moe.plan_memory()` and choose the default, exact-byte, or
   balanced-factor activation-slot policy.
3. Call `dist_moe.create_context()` once per device and shape; VMM prefetch follows the
   `dist_moe.VmmConfig.prefetch` policy.
4. Select an activation slot before each forward when multiple pipeline
   lifetimes share the context.
5. Call `context.reset()` after an aborted schedule before reusing the buffer.
6. Stop graph replay and traced execution, then call `context.close()`.

All real expert-parallel ranks must create matching contexts in the same
relative order because symmetric-memory rendezvous and barrier-workspace setup
are collective. One context owns mutable planner, scratch, and communication
state; do not overlap calls through it or use it concurrently from independent
streams.
