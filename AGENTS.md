# Dist-MoE Contributor Guide

This directory is synchronized to `meta-pytorch/dist_moe`. Keep source, tests,
documentation, and commit messages suitable for public release. Do not add
private project names, repositories, infrastructure links, identifiers, or
code names.

## Where To Start

- `README.md` defines installation, supported features, the BF16 quickstart,
  and the public API index.
- `dist_moe/api.py` owns configuration, memory planning, context construction,
  and the public `dist_moe.routed_experts()` entry point.
- `dist_moe/_execution.py` and `dist_moe/_blockscaled.py` implement BF16 and
  block-scaled orchestration respectively.
- `docs/*_execution.md` explain complete forward/backward call chains.
- `docs/*_kernel_design.md` explain kernel arithmetic, layouts, and scheduling.
- `docs/memory_planner.md`, `docs/vmm.md`, and
  `docs/pipeline_activation_slots.md` own the memory-management contracts.
- `examples/` contains the complete runnable programs referenced by README.
- `tests/AGENTS.md` explains the test matrix and debugging workflow.

Follow links to the canonical guide instead of repeating its explanation in a
new comment or document.

## Design Boundaries

- Routing and top-k selection happen before `dist_moe.routed_experts()`;
  callers supply expert IDs and scores.
- The package owns symmetric communication buffers, activation slots, scratch,
  planner state, and the registered autograd boundary. It does not own model,
  optimizer, FSDP, or pipeline policy.
- Framework integration belongs in the consuming repository. Production
  package code must not depend on a training framework.
- Public APIs must state tensor shape, dtype, layout, ownership, mutation,
  collective participation, lifecycle, and unsupported-mode contracts.
- Fixed-shape tensor variables use suffixes such as `_TD`, `_TK`, `_EFD`,
  `_EDF`, and `_1`. Polymorphic tensors use semantic names and document their
  accepted layouts.

## Kernel Boundary

Preserve the retained CuTe and Triton kernel arithmetic, launch geometry,
barriers, layouts, scheduling, and operation order unless a reviewed kernel
change requires otherwise. The annex is an intentional deterministic fork, so
runtime parity rather than source hashes is authoritative. A kernel change
requires numerical, graph, memory, and performance evidence.

## Graph And Autograd Changes

The supported compiler boundary is FakeTensor metadata, non-strict `make_fx`,
activation checkpointing, and CUDA graph capture. Keep mutable context storage
explicit at registered-operation boundaries. Preserve fixed schemas and device
addresses across capture and replay. A backward change must account for saved
versus recomputed activations, DGRAD, WGRAD, planner rewind, and output-gradient
ownership.

## Documentation

Every public and private non-kernel callable has a concise Google-style
docstring. Use inline comments only for non-obvious ownership, distributed,
memory, or graph invariants. Execution guides own end-to-end flows; kernel
guides own low-level implementation; API docstrings own local contracts.

## Validation Order

Run focused checks before the complete matrix:

1. package/config and planner policy tests;
2. one-GPU non-VMM execution;
3. real two-rank communication;
4. internal parity where the reference is available;
5. formatting, lint, package, and documentation checks;
6. real VMM only on an isolated supported Blackwell worker.

Never weaken an assertion or classify a failure as infrastructure without an
independent reproduction.
