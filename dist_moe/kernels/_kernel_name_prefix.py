# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import inspect
import threading
from collections.abc import Iterator
from typing import Any

from ._dsl_compat import NAME_OPTIONS_ON_FUNCTION

_KERNEL_NAME_PREFIX_LOCK = threading.RLock()
_NAME_OPTIONS_ATTR = "_cute_dsl_name_options"


def _recorded_prefix(target: Any) -> str | None:
    """Return the prefix ``set_name_prefix`` recorded for this DSL release."""
    if NAME_OPTIONS_ON_FUNCTION:
        options = vars(inspect.unwrap(target)).get(_NAME_OPTIONS_ATTR)
        return None if options is None else options.name_prefix
    return getattr(target, "_name_prefix", None)


def _materialize_decorated_dsl_object(target: Any) -> Any | None:
    """Perform CuTeDSL's one-way lazy initialization before state capture."""
    decorated = inspect.unwrap(target)
    # Mirror BaseDSL._lazy_initialize_dsl: @cute.jit stores _dsl_cls directly
    # on the wrapped function, then permanently replaces it with _dsl_object
    # on first use. Materialize under our lock so that first use is also scoped.
    dsl_cls = vars(decorated).get("_dsl_cls")
    if dsl_cls is not None:
        dsl_cls._lazy_initialize_dsl(decorated)
    return getattr(decorated, "_dsl_object", None)


@contextlib.contextmanager
def scoped_kernel_name_prefixes(
    prefixes: tuple[tuple[Any, str], ...],
) -> Iterator[None]:
    """Restore CuTeDSL name prefixes set by ``set_name_prefix`` after use."""
    with _KERNEL_NAME_PREFIX_LOCK:
        missing = object()
        # fmt: off
        resolved_prefixes: tuple[tuple[Any, str, Any | None], ...] = tuple(
            (target, prefix, _materialize_decorated_dsl_object(target))
            for target, prefix in prefixes
        )
        # fmt: on
        target_states: dict[int, tuple[Any, Any]] = {}
        dsl_states: dict[int, tuple[Any, Any]] = {}
        option_states: dict[int, tuple[Any, Any]] = {}
        installed_prefixes: set[str] = set()

        for target, _prefix, dsl in resolved_prefixes:
            target_states.setdefault(
                id(target),
                (target, getattr(target, "_name_prefix", missing)),
            )
            if NAME_OPTIONS_ON_FUNCTION:
                decorated = inspect.unwrap(target)
                option_states.setdefault(
                    id(decorated),
                    (decorated, vars(decorated).get(_NAME_OPTIONS_ATTR, missing)),
                )
            if dsl is not None:
                dsl_states.setdefault(
                    id(dsl),
                    (dsl, getattr(dsl, "_name_prefix", missing)),
                )

        try:
            for target, prefix, _dsl in resolved_prefixes:
                installed_prefixes.add(prefix)
                target.set_name_prefix(prefix)
                if _recorded_prefix(target) != prefix:
                    # fmt: off
                    raise RuntimeError(
                        "CuTeDSL name-prefix contract changed: "
                        "set_name_prefix() did not record the prefix"
                    )
                    # fmt: on
            yield
        finally:
            for decorated, previous_options in option_states.values():
                if previous_options is missing:
                    vars(decorated).pop(_NAME_OPTIONS_ATTR, None)
                else:
                    setattr(decorated, _NAME_OPTIONS_ATTR, previous_options)
            for target, previous_prefix in target_states.values():
                if getattr(target, "_name_prefix", None) in installed_prefixes:
                    if previous_prefix is missing:
                        delattr(target, "_name_prefix")
                    else:
                        target._name_prefix = previous_prefix
            for dsl, previous_prefix in dsl_states.values():
                if getattr(dsl, "_name_prefix", None) in installed_prefixes:
                    if previous_prefix is missing:
                        delattr(dsl, "_name_prefix")
                    else:
                        dsl._name_prefix = previous_prefix
