# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA driver context utilities for CuTe DSL kernels.

During PyTorch's autograd backward pass, the CUDA driver context may not be
bound to the current thread (cuCtxGetCurrent returns NULL). Code that uses the
CUDA driver API — such as cutlass HardwareInfo.get_max_active_clusters() — will
fail with CUDA_ERROR_INVALID_CONTEXT in this situation.

This module provides a context manager to ensure a valid driver context is
present before calling into such code.
"""

from contextlib import contextmanager
from typing import Generator

import torch
from cuda.bindings import driver


@contextmanager
def ensure_cuda_driver_context() -> Generator[None, None, None]:
    """Context manager that ensures a valid CUDA driver context on the current thread.

    Checks whether a driver context is already bound. If not, retains the
    device's primary context and pushes it for the duration of the block.

    Usage::

        with ensure_cuda_driver_context():
            cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    """
    ctx_result = driver.cuCtxGetCurrent()
    current_ctx = ctx_result[1]
    need_push = current_ctx is None or int(current_ctx) == 0
    if need_push:
        device_id = torch.cuda.current_device()
        device_handle = driver.cuDeviceGet(device_id)[1]
        primary_ctx = driver.cuDevicePrimaryCtxRetain(device_handle)[1]
        driver.cuCtxPushCurrent(primary_ctx)
    try:
        yield
    finally:
        if need_push:
            driver.cuCtxPopCurrent()
