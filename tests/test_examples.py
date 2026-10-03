# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for the public example programs."""

from __future__ import annotations

import ast
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_ROOT = _PROJECT_ROOT / "examples"
_DISTRIBUTED_EXAMPLES = (
    "bf16_training.py",
    "mxfp8_training.py",
    "block_scaled_inference.py",
)


def _example_command(example_name: str) -> list[str]:
    """Return the launch command declared by one example module.

    Args:
        example_name: Filename below ``examples/``.

    Returns:
        Shell-free argument vector for the declared command.
    """
    source = (_EXAMPLES_ROOT / example_name).read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))
    if (
        docstring is None
        or "\n" in docstring
        or not docstring.startswith("Run: ")
        or not docstring.endswith(".")
    ):
        raise AssertionError(
            f"{example_name} must declare exactly one 'Run: <command>.' line"
        )
    return shlex.split(docstring[len("Run: ") : -1])


def _run_example(example_name: str) -> None:
    """Run one declared example in the current Python environment.

    Args:
        example_name: Filename below ``examples/``.
    """
    command = _example_command(example_name)
    if command[0] == "python":
        command[0] = sys.executable
    elif command[0] == "torchrun":
        command[:1] = [sys.executable, "-m", "torch.distributed.run"]
    else:
        raise AssertionError(f"unsupported example launcher: {command[0]}")
    subprocess.run(command, cwd=_PROJECT_ROOT, check=True)


def test_example_docstrings_declare_exact_launch_commands() -> None:
    """Require every public example to expose one directly executable command."""
    example_names = {
        path.name
        for path in _EXAMPLES_ROOT.glob("*.py")
        if not path.name.startswith("_")
    }
    expected = {
        "memory_planning.py",
        *_DISTRIBUTED_EXAMPLES,
    }
    assert example_names == expected
    for example_name in sorted(example_names):
        command = _example_command(example_name)
        assert command[-1] == f"examples/{example_name}"


def test_memory_planning_example_exits_zero() -> None:
    """Run the CPU-only planning example as executable documentation."""
    _run_example("memory_planning.py")


@pytest.mark.gpus_needed_2
@pytest.mark.gb10x
@pytest.mark.parametrize("example_name", _DISTRIBUTED_EXAMPLES)
def test_distributed_example_exits_zero(example_name: str) -> None:
    """Run each non-VMM distributed example without duplicating numerical tests."""
    _run_example(example_name)
