# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA device properties consumed by the CuTe launchers."""

import torch


def is_blackwell_gpu(device: torch.device | int | str | None = None) -> bool:
    """Return whether a CUDA device has Blackwell compute capability.

    Args:
        device: CUDA device, defaulting to the current device.

    Returns:
        ``True`` for compute capability 10.x or newer.
    """
    if not torch.cuda.is_available():
        return False
    if device is not None and not isinstance(device, int):
        device = torch.device(device)
        if device.type != "cuda":
            return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 10


def num_sms_per_device(device: torch.device | int | None = None) -> int:
    """Return the streaming-multiprocessor count for a CUDA device.

    Args:
        device: CUDA device, defaulting to the current device.

    Returns:
        Number of streaming multiprocessors.

    Raises:
        RuntimeError: If CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CuTe distributed MoE requires CUDA")
    return torch.cuda.get_device_properties(device).multi_processor_count


def l2_cache_size(device: torch.device | int | None = None) -> int:
    """Return the CUDA device L2 capacity in bytes.

    Args:
        device: CUDA device, defaulting to the current device.

    Returns:
        L2 capacity in bytes.

    Raises:
        RuntimeError: If CUDA or the property is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CuTe distributed MoE requires CUDA")
    capacity = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0)
    if capacity <= 0:
        raise RuntimeError("CUDA device properties do not expose L2 cache size")
    return int(capacity)
