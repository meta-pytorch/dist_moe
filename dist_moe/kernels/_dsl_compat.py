# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility helpers for supported NVIDIA CuTe DSL releases."""

import inspect
from importlib.metadata import version

import cutlass.cute as cute
import cutlass.cute.math as cute_math
from cutlass._mlir.dialects import llvm, math as mlir_math, nvvm
from cutlass.cutlass_dsl import dsl_user_op
from packaging.version import Version

RoundingMode = getattr(nvvm, "RoundingModeKind", None) or nvvm.FPRoundingMode

CUTEDSL_VERSION = Version(version("nvidia-cutlass-dsl"))

# From 4.6.2, ``set_name_prefix`` stores an immutable ``_cute_dsl_name_options``
# record on the decorated function. Earlier releases set ``_name_prefix`` on
# the jit wrapper, and kernel launches copy it onto the shared DSL object.
# Compare the release segment only, so 4.6.2rc1 and 4.6.2.dev0 count as 4.6.2.
NAME_OPTIONS_ON_FUNCTION = CUTEDSL_VERSION.release >= (4, 6, 2)

# Packed FP32 helpers changed from NVVM enums to string literals in CuTe 4.5.
_packed_rounding_default = (
    inspect.signature(cute.arch.mul_packed_f32x2).parameters["rnd"].default
)
PACKED_F32_RN = "rn" if isinstance(_packed_rounding_default, str) else RoundingMode.RN


def _adapt_nvvm_builder(builder):
    """Preserve the CuTe 4.4 NVVM builder call shape on newer releases.

    CuTe 4.5 removed the leading explicit result type from generated NVVM
    builders. Kernel code uses the 4.4 call shape so both releases generate the
    same operation after this import-time adaptation.

    Args:
        builder: Generated NVVM operation builder.

    Returns:
        Builder accepting the CuTe 4.4 argument convention.
    """
    if next(iter(inspect.signature(builder).parameters)) == "res":
        return builder

    def drop_result_type(_result_type, *args, **kwargs):
        """Call a newer NVVM builder without the obsolete result type."""
        return builder(*args, **kwargs)

    return drop_result_type


cvt_packfloat = _adapt_nvvm_builder(nvvm.cvt_packfloat)
cvt_packfloat_f32 = _adapt_nvvm_builder(nvvm.cvt_packfloat_f32)
fmax = _adapt_nvvm_builder(nvvm.fmax)
fmin = _adapt_nvvm_builder(nvvm.fmin)

if hasattr(cute_math, "absf"):
    absf = cute_math.absf
else:

    def absf(value, fastmath: bool = False):
        """Evaluate absolute value across supported CuTe DSL releases.

        Args:
            value: CuTe scalar value.
            fastmath: Whether to enable fast-math lowering.

        Returns:
            Absolute value of ``value``.
        """
        return cute_math._math_op(mlir_math.absf, fastmath, value)


@dsl_user_op
def thread_exit(*, loc=None, ip=None) -> None:
    """Exit every thread in a uniformly predicated CTA branch.

    Args:
        loc: Optional MLIR source location.
        ip: Optional MLIR insertion point.
    """
    llvm.inline_asm(
        None,
        [],
        "exit;",
        "",
        has_side_effects=True,
        is_align_stack=False,
    )
