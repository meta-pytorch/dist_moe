# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process-local context lookup and shared expert-weight shape helpers."""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from .api import Context


_CONTEXTS: dict[str, object] = {}


def _register_context(context_id: str, context: Context) -> None:
    """Register a process-local execution context.

    Args:
        context_id: Identifier assigned during context construction.
        context: Live distributed MoE context.
    """
    _CONTEXTS[context_id] = context


def _unregister_context(context_id: str) -> None:
    """Remove a process-local execution context if it is registered.

    Args:
        context_id: Identifier assigned during context construction.
    """
    _CONTEXTS.pop(context_id, None)


def _get_context(context_id: str) -> Context:
    """Resolve a live process-local execution context.

    Args:
        context_id: Identifier assigned during context construction.

    Returns:
        Live distributed MoE context.

    Raises:
        RuntimeError: If the context is missing or closed.
    """
    # Context teardown is forbidden while an invocation is live. CPython dict
    # lookup is atomic under the GIL, so reads need no per-invocation lock.
    context = cast("Context | None", _CONTEXTS.get(context_id))
    if context is None or context._closed:
        raise RuntimeError(f"distributed MoE context {context_id} is not live")
    return context


def _reshape_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    context: Context,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return grouped expert views while preserving logical ownership.

    Args:
        w13: Fused gate/up weight in flattened, grouped, or gate-axis form.
        w2: Down-projection weight in flattened or grouped form.
        context: Execution context defining local expert dimensions.

    Returns:
        Three-dimensional grouped ``w13`` and ``w2`` views.

    Raises:
        ValueError: If the two weight layouts are incompatible.
    """
    if w13.ndim == 3 and w2.ndim == 3:
        return w13, w2
    if w13.ndim == 4 and w2.ndim == 3:
        if w13.shape[1] != 2:
            raise ValueError("gate-axis w13 must have shape [E, 2, F, D]")
        return w13.flatten(1, 2), w2
    if w13.ndim == 2 and w2.ndim == 2:
        num_local_experts = context.config.num_experts // dist.get_world_size(
            context.group
        )
        return (
            w13.view(num_local_experts, -1, context.hidden_dim),
            w2.view(num_local_experts, context.hidden_dim, -1),
        )
    raise ValueError("w13 and w2 use incompatible expert-weight layouts")
