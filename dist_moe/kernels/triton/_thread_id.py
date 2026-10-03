# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone Triton thread-index binding used by routing kernels."""

import triton
import triton.language as tl

from .comm_utils import get_flat_tid


@triton.jit
def thread_id(axis: tl.constexpr):
    """Return the flattened thread ID used by the Dist-MoE routing kernel.

    Args:
        axis: Thread-index axis. The routing kernel requires axis zero.

    Returns:
        Flattened thread index within the CTA.
    """
    tl.static_assert(axis == 0)
    return get_flat_tid()
