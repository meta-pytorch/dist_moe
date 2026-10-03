# VMM Host-Overflow Scratch

Dist-MoE can map one contiguous CUDA virtual address range to device and pinned
host memory. This is CUDA Virtual Memory Management (VMM), not managed memory:
the physical location of every section is fixed when the buffer is created and
pages do not migrate during execution.

VMM changes storage backing only. Routing, offsets, grouped GEMMs, SwiGLU, and
backward use the same kernels and the same virtual base pointer as the
device-only path. There are no VMM-specific kernel branches.

## Outline

- [Size logical scratch capacity](#size-logical-scratch-capacity)
- [Understand logical and physical layouts](#understand-logical-and-physical-layouts)
- [Keep activations on device](#keep-activations-on-device)
- [Create and close a VMM context](#create-and-close-a-vmm-context)
- [Choose when to use VMM](#choose-when-to-use-vmm)

<a id="size-logical-scratch-capacity"></a>
## Size logical scratch capacity

For the fixed `T` local input tokens and top-k `K`, balanced routing receives
`T * K` rows per rank. Two independent factors bound scratch:

```text
balanced_rows = T * K
device_rows = round(balanced_rows * device_scratch_capacity_factor)
total_rows = round(balanced_rows * total_scratch_capacity_factor)
```

`device_scratch_capacity_factor` belongs to `dist_moe.Config` and controls
the hot device-backed suffix. `total_scratch_capacity_factor` belongs to
`dist_moe.VmmConfig` and controls device plus host-backed scratch. The total
factor is clamped to the expert-parallel size, the largest possible receive
imbalance, and must not be smaller than the device factor.

The factors do not rebalance, truncate, or pad routing results. They reserve
capacity. Exceeding total capacity is an execution error.

MXFP8 additionally pads each local expert to its 128-row kernel group. The
memory planner applies that topology-aware padding before sizing both device
and total scratch. This changes byte counts, not VMM ownership: saved MXFP8
activation bundles remain device-backed and only overflow scratch may be
host-backed.

<a id="understand-logical-and-physical-layouts"></a>
## Understand logical and physical layouts

Training keeps saved forward state at low addresses and grows temporary
scratch backward from the high end:

```text
low address                                                     high address
+--------------------------+----------------------+-------------------------+
| saved activation slots   | host-backed scratch  | hot device scratch      |
| device HBM               | pinned host memory   | device HBM              |
+--------------------------+----------------------+-------------------------+
                                  scratch grows right to left <-------------
```

Inference saves no backward state, so the complete virtual range is scratch.
Without VMM, both saved activations and the complete scratch range reside in
device HBM.

The logical activation boundary is
`num_activation_slots * activation_slot_bytes`. CUDA requires each physical
mapping to use the device allocation granularity, and the PyTorch allocator may
reserve a larger block for a small request. `dist_moe.plan_memory(..., device=device)`
resolves those physical sections and reports:

- `vmm_device_prefix_bytes`: saved activations plus leading alignment;
- `vmm_host_section_bytes`: host-backed overflow scratch;
- `vmm_device_scratch_section_bytes`: hot device scratch;
- `vmm_padding_bytes`: physical bytes beyond the logical request;
- `total_virtual_bytes`: the final contiguous address range.

Offsets remain byte offsets from the same virtual base. A kernel may therefore
read an activation from local HBM, hot scratch from local HBM, or overflow
scratch from pinned host memory without a host-side pointer decision.
Symmetric routing, dispatch, combine, signal, and barrier storage always stays
in device memory; peer GPUs never access the rank-local host section.

<a id="keep-activations-on-device"></a>
## Keep activations on device

VMM does not change saved-activation policy. For training, the minimum useful
budget retains every layer input and recomputes all expert intermediates. The
maximum useful budget retains all planner-eligible intermediates. A value
between those limits lets the device-side planner save complete layer state
while space remains and select recomputation otherwise.

Each slot has the effective `activation_slot_bytes` returned by the planner;
the total device-backed activation prefix is their product. An explicit byte
request is aligned internally. A capacity factor scales only optional saved
state above mandatory layer inputs and is independent of both scratch factors.
`max_moe_layers_per_activation_slot` bounds the local layer depth that may
append state to one selected slot. Exhausting a slot selects recomputation; it
does not spill saved activations into host memory.

<a id="create-and-close-a-vmm-context"></a>
## Create and close a VMM context

All ranks must construct matching contexts collectively and in the same order.
The normal API derives the exact physical layout from one public config:

```python
import os

import torch
import torch.distributed as dist
import dist_moe

device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl")
ep_group = dist.group.WORLD

config = dist_moe.Config(
    num_local_input_tokens=4096,
    hidden_dim=4096,
    intermediate_dim=14336,
    top_k=8,
    num_experts=128,
    max_moe_layers_per_activation_slot=32,
    device_scratch_capacity_factor=1.5,
    activation_slot_capacity_factor=1.0,
    num_activation_slots=1,
    vmm=dist_moe.VmmConfig(
        total_scratch_capacity_factor=16.0,
        prefetch=True,
    ),
)
plan = dist_moe.plan_memory(
    config,
    ep_size=dist.get_world_size(ep_group),
    device=device,
)
context = dist_moe.create_context(
    group=ep_group,
    config=config,
    device=device,
)
try:
    print(plan.explain())
    # Run sequential Dist-MoE calls or CUDA-graph replay here.
finally:
    context.close()
    dist.destroy_process_group()
```

Context construction first reserves the CUDA device, then, with
`prefetch=True`, starts the exact physical VMM allocation before collective
communication-buffer initialization and consumes it when the activation buffer
is created. A concurrent constructor for that device therefore fails before
starting another allocation. With `prefetch=False`, context construction waits
until communication buffers are ready and creates the same physical layout
synchronously. No prefetch work remains active after `dist_moe.create_context()`
returns successfully.

Context closure must happen only after all traced execution and CUDA-graph
replay using its virtual addresses have ended.

`close()` is idempotent after successful teardown. If a CUDA unmap, allocation
release, or virtual-address release fails, the owner retains the unreleased
handles and address range so a later `close()` can retry exactly the remaining
work. It does not clear ownership bookkeeping or compact the section list after
a partial failure.

Failed context construction retries prefetch and allocator cleanup once and
releases the device reservation only after no VMM region remains. If cleanup
still fails, that cleanup error is primary, the triggering setup error remains
its exception context, and the device stays reserved. Because construction did
not return a context that can retry teardown, the process must not attempt
another Dist-MoE VMM context on that device; restart the process after recording
the CUDA Driver cleanup failure.

Only one live Dist-MoE VMM-backed activation buffer is supported per CUDA
device. This makes address ownership and teardown deterministic.

<a id="choose-when-to-use-vmm"></a>
## Choose when to use VMM

Host-backed scratch protects against receive imbalance without paying its full
HBM cost, but accesses over the GPU-host interconnect are slower than HBM.
Choose a device factor from measured routing and use host scratch for the
tolerated tail. If deterministic peak throughput is more important than HBM
capacity, reserve the full required scratch on device and leave `vmm=None`.
