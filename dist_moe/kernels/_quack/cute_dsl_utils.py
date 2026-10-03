# Copyright (c) 2025, Tri Dao.
# Licensed under the Apache License, Version 2.0.
# Modified by Meta Platforms, Inc.

import os
import re
from functools import lru_cache
from typing import Tuple

import cutlass
import torch
from cutlass import BFloat16, Float8E4M3FN, Float8E5M2, Float16, Float32, Int32, Int64

torch2cute_dtype_map = {
    torch.uint8: cutlass.Uint8,
    # Packed fp4: dlpack presents the logical (doubled) K extent to the DSL.
    torch.float4_e2m1fn_x2: cutlass.Float4E2M1FN,
    torch.float8_e4m3fn: Float8E4M3FN,
    torch.float8_e5m2: Float8E5M2,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
    torch.float8_e8m0fnu: cutlass.Float8E8M0FNU,
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


def _parse_arch_str(arch_str: str) -> Tuple[int, int]:
    """Parse arch string (e.g. 'sm_90', 'sm90', '90', 'sm_100a') to (major, minor) tuple."""
    match = re.match(r"^(?:sm_?)?(\d+)(\d)([af]?)$", arch_str.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Invalid QUACK_ARCH format: {arch_str!r} (expected e.g. '90', 'sm_90')"
        )
    major, minor, _ = match.groups()
    return int(major), int(minor)


@lru_cache
def _get_device_capacity_cached(device: torch.device = None) -> Tuple[int, int]:
    """Return (major, minor) device capability.

    Override with QUACK_ARCH (e.g. 'sm_90' or '90') for CPU-only compilation
    without a GPU present.
    """
    arch_override = os.environ.get("QUACK_ARCH")
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    return torch.cuda.get_device_capability(device)


def get_device_capacity(
    device: torch.device | torch.Tensor | None = None,
) -> Tuple[int, int]:
    """Return (major, minor) device capability.

    Override with QUACK_ARCH (e.g. 'sm_90' or '90') for CPU-only compilation
    without a GPU present.

    Accepts either a ``torch.device`` or a tensor and canonicalizes to the
    underlying device before consulting the cached helper. This avoids leaking
    tensors through the LRU cache key.
    """
    if isinstance(device, torch.Tensor):
        device = device.device
    return _get_device_capacity_cached(device)
