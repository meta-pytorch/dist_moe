# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Normalized native and prepared block-scaled expert weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockscaledWeightSpec:
    """Normalized native and optional prequantized expert weight operands.

    Args:
        w_native: Native grouped expert weight.
        w_q_fprop: Quantized FPROP operand.
        w_scale_fprop: FPROP scale-factor layout.
        w_q_dgrad: Optional quantized DGRAD operand.
        w_scale_dgrad: Optional DGRAD scale-factor layout.
        w_global_scale: Optional per-expert NVFP4 global scale.
        w_global_scale_inv: Optional precomputed reciprocal global scale.
    """

    w_native: torch.Tensor
    w_q_fprop: torch.Tensor | None = None
    w_scale_fprop: torch.Tensor | None = None
    w_q_dgrad: torch.Tensor | None = None
    w_scale_dgrad: torch.Tensor | None = None
    w_global_scale: torch.Tensor | None = None
    w_global_scale_inv: torch.Tensor | None = None

    @property
    def is_prequantized(self) -> bool:
        """Return whether FPROP quantized operands are present."""
        return self.w_q_fprop is not None
