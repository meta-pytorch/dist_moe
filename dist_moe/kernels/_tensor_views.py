# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch-only helpers for reasoning about tensor views."""

import torch


def _same_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return whether two tensors address the same logical view."""
    return (
        a.device == b.device
        and a.dtype == b.dtype
        and a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
    )


def _view_byte_bounds(tensor: torch.Tensor) -> tuple[int, int]:
    """Return conservative half-open byte bounds for a PyTorch strided view."""
    # PyTorch does not expose negative-stride tensors, so `data_ptr()` is the
    # lowest reachable address for the view.
    max_offset = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    element_size = tensor.element_size()
    return (
        tensor.data_ptr(),
        tensor.data_ptr() + (max_offset + 1) * element_size,
    )


def _views_overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return whether two views may share bytes, conservatively for strided layouts."""
    if a.device != b.device or a.numel() == 0 or b.numel() == 0:
        return False
    a_start, a_end = _view_byte_bounds(a)
    b_start, b_end = _view_byte_bounds(b)
    return a_start < b_end and b_start < a_end
