# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process-local compilation cache used by the standalone CuTe kernels."""

from functools import cache as jit_cache

__all__ = ["jit_cache"]
