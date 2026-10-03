# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin adapters over the bundled Triton kernels."""

from __future__ import annotations

import torch

from ._buffers import (
    get_multimem_barrier_workspace,
    is_fake_symmetric_memory,
    SymmetricMemoryBuffer,
)
from .kernels.triton.barrier import (
    triton_symm_mem_barrier_multimem_ptrs,
    triton_symm_mem_barrier_ptrs,
)
from .kernels.triton.broadcast_n_reduction import (
    broadcast_and_scale,
    reduce_from_topk,
    scale_and_sum,
)
from .kernels.triton.comm_utils import (
    get_flat_tid,
    send_signal,
    sync_threads,
    wait_signal,
)
from .kernels.triton.conditional_copy import (
    conditional_copy_activations,
    copy_activation_to_dispatch,
    copy_dispatch_to_activation,
    copy_routing_and_dispatch,
)
from .kernels.triton.swiglu import swiglu_bwd, swiglu_fwd

__all__ = [
    "broadcast_and_scale",
    "conditional_copy_activations",
    "copy_activation_to_dispatch",
    "copy_dispatch_to_activation",
    "copy_routing_and_dispatch",
    "get_flat_tid",
    "reduce_from_topk",
    "scale_and_sum",
    "send_signal",
    "swiglu_bwd",
    "swiglu_fwd",
    "symmetric_memory_barrier",
    "sync_threads",
    "wait_signal",
]


def symmetric_memory_barrier(
    buffer: SymmetricMemoryBuffer,
    channel: int = 0,
    conditional_execution: torch.Tensor | None = None,
) -> None:
    """Synchronize peers through the buffer's eagerly initialized barrier.

    Args:
        buffer: Symmetric-memory buffer whose ranks must synchronize.
        channel: Signal-pad or multicast flag channel reserved by the caller.
        conditional_execution: Optional common device-side predicate. Every
            rank must provide the same value.

    Raises:
        ValueError: If ``channel`` exceeds the multicast workspace.
    """
    if is_fake_symmetric_memory(buffer):
        return
    world_size = buffer.hdl.world_size
    workspace = get_multimem_barrier_workspace(buffer.group)
    if workspace is not None:
        if not 0 <= channel < workspace.num_channels:
            raise ValueError(
                f"barrier channel {channel} is outside [0, {workspace.num_channels})"
            )
        triton_symm_mem_barrier_multimem_ptrs(
            workspace.multicast_ptr,
            workspace.local_ptr,
            n_ranks=world_size,
            channel=channel,
            conditional_execution=conditional_execution,
        )
        return
    triton_symm_mem_barrier_ptrs(
        buffer.signal_pad_ptrs_tensor,
        rank=buffer.hdl.rank,
        n_ranks=world_size,
        channel=channel,
        conditional_execution=conditional_execution,
    )
