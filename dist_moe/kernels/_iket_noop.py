# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""No-op stand-in for ``cutlass.cute.experimental.iket`` on DSL versions
without IKET. Only ever traced when an instrumentation level is force-enabled
on such a version; every call is a benign no-op."""


def mark(name, payload=None) -> None:
    pass


def range_push(name, payload=None) -> None:
    pass


def range_pop() -> None:
    pass


def range_start(name, payload=None):
    return None


def range_end(token, payload=None) -> None:
    pass


def sentinel_token(name):
    return None
