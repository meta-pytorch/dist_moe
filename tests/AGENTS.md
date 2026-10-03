# Dist-MoE Test Guide

Tests in this directory validate the standalone public package. They must not
import private reference implementations or framework integrations.

## Test Layout

- `test_dist_moe.py` covers configuration, buffers, BF16 and block-scaled
  execution, autograd, tracing, CUDA graphs, and real two-rank communication.
- `test_dist_moe_vmm.py` separates CPU planner/ownership tests from real VMM
  allocation and host-spill execution.
- `test_package_boundary.py` separates runtime-free source and artifact checks
  from documentation checks that import or execute the selected runtime.
- `test_examples.py` runs each checked-in example as an exit-status smoke test;
  its VMM case belongs only on an isolated worker.

The detailed feature contract belongs in the corresponding guide under
`docs/`; tests should link to it rather than restating its implementation.

## Test Contract

Every test docstring explains the exercised path, the assertion that proves the
invariant, and why that coverage is distinct. Parameterized tests describe the
invariant shared by all cases. Numerical tests state whether equality is exact
or tolerance-based.

Use the narrowest real execution that proves the behavior:

- CPU tests for configuration, validation, metadata, and packaging.
- Runtime-free packaging jobs must select `StaticPackageBoundaryTest` and
  `DistributionArtifactTest` explicitly; they must not install Torch merely to
  execute runtime documentation.
- CPU fbsource targets exclude the `distribution_artifact` and
  `gpus_needed_2` markers because they do not provide either prerequisite.
- One supported Blackwell GPU for kernel, autograd, tracing, and graph replay.
- Two real ranks for peer routing and symmetric-memory communication.
- Example smoke tests prove that documented commands execute; they do not
  duplicate numerical assertions from `test_dist_moe.py`.
- FakePG only for shape, allocation, and tracing behavior; it is not numerical
  communication evidence.
- Isolated scheduled Blackwell workers for real VMM allocation and host spill.

## Debugging

1. Reproduce the exact failing node in its declared environment.
2. Verify that the test actually selected the intended architecture and rank
   count.
3. Identify the violated invariant before changing code or expectations.
4. Fix the earliest owning implementation or test; do not add a child repair.
5. Rerun the focused node, then the affected class, then the complete lane.

Do not skip a real failure, broaden tolerances without numerical evidence, use
an overlay, or run real VMM on a shared development GPU.
