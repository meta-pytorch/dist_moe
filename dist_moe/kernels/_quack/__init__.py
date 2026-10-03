# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal pinned Quack support closure used by the CuTe kernels."""

# Quack's register-fragment helpers are installed at package import time.
# Import the exact required helper before any copied kernel is compiled so its
# CuTe tensor operations retain the same contract.
from . import cute_tensor as _cute_tensor  # noqa: F401
