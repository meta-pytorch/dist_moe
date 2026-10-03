# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

import torch

CollectiveContext = Callable[[Any, str | None, Any, Any], AbstractContextManager[None]]

_collective_context: ContextVar[CollectiveContext | None] = ContextVar(
    "collective_context",
    default=None,
)


@contextmanager
def collective_context(context: CollectiveContext) -> Iterator[None]:
    token = _collective_context.set(context)
    try:
        yield
    finally:
        _collective_context.reset(token)


def annotate_collective(
    group: Any,
    name: str | None = None,
    input_tensor: Any = None,
    output_tensor: Any = None,
) -> AbstractContextManager[None]:
    """Annotate Python launches during capture; Dynamo-traced regions are a no-op."""
    if torch.compiler.is_compiling():
        return nullcontext()

    context = _collective_context.get()
    if context is None:
        return nullcontext()

    return context(group, name, input_tensor, output_tensor)


def annotate_barrier(pg_name: str) -> AbstractContextManager[None]:
    return annotate_collective(group=pg_name)
