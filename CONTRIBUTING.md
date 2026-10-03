# Contributing

Thank you for improving `dist_moe`. See Meta's open source
[contribution guidelines](https://opensource.fb.com/how-to-contribute/) and
[code of conduct](https://opensource.fb.com/code-of-conduct/) before opening a
change.

## Design boundary

The `dist_moe` package is a standalone PyTorch extension. Production code must
not import TorchTitan or another training framework. Framework configuration,
FSDP policy, model conversion, and pipeline schedule ownership belong in the
consumer.

Routing is also outside the package: callers provide top-k expert IDs and
scores. Changes that add a router to the fused expert operation are outside the
current API contract.

## Kernel changes

The CuTe kernels have a deterministic runtime byte-parity contract with their
reference implementation. A change to arithmetic, data layout, launch geometry, tile
selection, barriers, operation order, or conversion points requires:

1. a focused explanation of the invariant being changed;
2. forward and backward numerical tests;
3. internal byte-parity validation where the reference is available;
4. before/after kernel, memory, and throughput evidence.

Do not mix a kernel change with packaging, documentation, or framework
integration cleanup.

## Tests and formatting

Add focused tests that exercise the real code path and prove the changed
invariant. See [`tests/AGENTS.md`](tests/AGENTS.md) for the test ownership,
evidence, and debugging contract. CPU source and package checks use the
checked-in CPU tool lock:

```bash
python -m pip install --requirement .github/requirements/ci-cpu.txt
pre-commit run --all-files
ruff check .
ruff format --check .
python -m build --no-isolation
twine check dist/*
DIST_MOE_ARTIFACT_DIR=dist python -m pytest -q \
  tests/test_package_boundary.py::StaticPackageBoundaryTest \
  tests/test_package_boundary.py::DistributionArtifactTest
```

Kernel execution requires a supported Blackwell runtime. Run the one-GPU
config, buffer, and kernel classes, then the real two-rank class with the
repository's distributed launcher. Public examples are smoke-tested through
`tests/test_examples.py`; distributed examples require two GPUs.
Real VMM mapping and host-spill tests must run on an isolated scheduled host
rather than a personal development server.

Packaging and dependency changes must build and inspect both the exact wheel
and source distribution.
Regenerate a lock with the `uv pip compile` command recorded in its header and
review the complete diff; do not hand-edit generated lock files.

Report security issues through Meta's
[security process](https://www.facebook.com/whitehat) rather than a public
issue.
