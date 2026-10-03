# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Version-tolerant import of the CuTe DSL IKET (in-kernel event tracing) API.

Kernel modules must import ``iket`` from here rather than doing their own
try/except: the DSL's AST preprocessor replays the kernel module's import
statements and only tolerates ``ImportError``, while older DSLs' experimental
gate can raise ``NotImplementedError`` — so the fallback logic has to live in
a module the kernel imports unconditionally.

IKET calls are stripped by the compiler unless lowering is enabled (a
``run-iket`` profiling run, ``CUTE_DSL_COMPILER_OPT=iket``, or
``cute.compile(..., options="iket")``), so guarded instrumentation is free in
production builds.
"""

import os

try:
    from cutlass.cute.experimental import iket  # noqa: F401  (DSL >= 4.6)
except (AttributeError, ImportError, NotImplementedError):
    from . import _iket_noop as iket  # noqa: F401


def iket_level(env_var: str) -> int:
    """Instrumentation level from ``env_var`` (0 = off, calls compile-time
    stripped). Levels are baked as trace-time constexprs and invisible to
    the cutedsl disk cache's key — run instrumented workloads with
    ``CUTEDSL_CACHE_ENABLED=0``."""
    return int(os.getenv(env_var, "0"))
