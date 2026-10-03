# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Global-scale helpers for NVFP4 block-scaled quantization."""

import math
from enum import auto, IntEnum, StrEnum

import torch

from .formats import NVFP4_GLOBAL_SCALE_EPS, NVFP4_MAX


class GlobalScaleGranularity(StrEnum):
    """How an NVFP4 global-scale tensor is broadcast across logical tokens."""

    PER_TENSOR = auto()
    PER_EXPERT = auto()
    PER_TOKEN = auto()


class GlobalScaleRank(IntEnum):
    """Rank of the row-scale tensor passed through backend kernel metadata.

    The integer values are consumed by Triton constexpr branches, so do not
    reorder them.
    """

    SCALAR = 0
    ONE_D = 1
    TWO_D = 2


def nvfp4_weight_global_scale(
    weight: torch.Tensor, numerator: float = NVFP4_MAX
) -> torch.Tensor:
    """Return the FP32 per-expert scale used to quantize an NVFP4 weight.

    ``numerator`` is the scale-format max times the E2M1 max; pass
    ``NVFP4_UE5M3_MAX`` for the UE5M3-scaled variant.
    """
    if weight.ndim != 3:
        raise ValueError(
            "NVFP4 per-expert weight scaling requires a 3D grouped weight; "
            f"got shape={tuple(weight.shape)}"
        )
    # Fused inf-norm reduction: abs and the fp32 accumulation fold into one
    # reduction kernel with no full-size intermediate at the weight-transfer
    # memory peak (`.abs()` would materialize a weight-sized copy; `.float()`
    # first, a 2x-larger one). Bitwise-identical to abs->amax->upcast: abs is
    # exact, the monotone bf16->fp32 upcast commutes with max, and NaN/Inf
    # propagate the same way. Same idiom as the pinned fake-quant counterpart
    # (`_compute_nvfp4_global_scale_for_fake_quant`).
    amax = torch.linalg.vector_norm(
        weight, ord=float("inf"), dim=(-2, -1), dtype=torch.float32
    ).clamp(min=NVFP4_GLOBAL_SCALE_EPS)
    # AUTHORITY for the NVFP4 global-scale arithmetic: reciprocal-multiply.
    # It can differ from correctly-rounded division by 1 ulp at some amaxes,
    # and a 1-ulp global-scale mismatch between fake and real quantization
    # flips FP4 codes near rounding boundaries — so every producer synthesizes
    # with this same arithmetic: fake quantization
    # (`_compute_nvfp4_global_scale_for_fake_quant`) and the per-token/dynamic
    # serving producers (`calculate_group_max`, the token-scale reductions in
    # transport, stacked, and CuTe quantizers, plus the fused kernels; their
    # wire `token_scale_inv` is the correctly-rounded
    # inverse of the scale). Scope: the bitwise agreement is for FINITE
    # amaxes — the fused SwiGLU kernels' maxnum reductions floor a
    # NaN amax finite where the transport quantizers propagate it.
    return numerator * torch.reciprocal(amax)


def safe_global_scale(global_scale: torch.Tensor) -> torch.Tensor:
    """Clamp invalid global scales without host-side synchronization."""
    return torch.where(
        global_scale > NVFP4_GLOBAL_SCALE_EPS,
        global_scale,
        NVFP4_GLOBAL_SCALE_EPS,
    )


def wire_inverse(scale: torch.Tensor) -> torch.Tensor:
    """Correctly-rounded inverse of a scale — the wire ``token_scale_inv``
    contract. Written as tensor/tensor division (correctly rounded, matching
    the kernels' ``div_rn`` / ``rcp.rn``); a python-scalar numerator or
    divisor would take a different CUDA fast path."""
    return torch.div(torch.ones_like(scale), scale)


def _expand_per_tensor(
    scale: torch.Tensor,
    scale_shape: tuple[int, ...],
) -> torch.Tensor:
    if scale.numel() != 1:
        raise ValueError(
            "per-tensor global_scale must be scalar or one element; "
            f"got shape={scale_shape}"
        )
    return scale.reshape(1).contiguous()


def _expand_per_expert(
    scale: torch.Tensor,
    scale_shape: tuple[int, ...],
    logical_shape: tuple[int, ...],
    row_shape: tuple[int, ...],
) -> torch.Tensor:
    if len(logical_shape) != 3:
        raise ValueError("per-expert global_scale requires a grouped 3D tensor")
    if scale_shape != logical_shape[:-2]:
        raise ValueError(
            "per-expert global_scale must have shape logical_shape[:-2]; "
            f"got global_scale.shape={scale_shape} and logical_shape={logical_shape}"
        )
    return scale.reshape(*logical_shape[:-2], 1).expand(row_shape)


def _expand_per_token(
    scale: torch.Tensor,
    scale_shape: tuple[int, ...],
    logical_shape: tuple[int, ...],
    row_shape: tuple[int, ...],
    rows: int,
) -> torch.Tensor:
    if scale_shape == row_shape:
        return scale.contiguous()
    if scale.ndim == 1 and scale.numel() == rows:
        return scale.reshape(row_shape).contiguous()
    raise ValueError(
        "per-token global_scale must have shape logical_shape[:-1] or "
        f"({rows},); got global_scale.shape={scale_shape} "
        f"and logical_shape={logical_shape}"
    )


def _expand_per_granularity(
    scale: torch.Tensor,
    scale_shape: tuple[int, ...],
    logical_shape: tuple[int, ...],
    row_shape: tuple[int, ...],
    rows: int,
    granularity: GlobalScaleGranularity,
) -> torch.Tensor:
    if granularity == GlobalScaleGranularity.PER_TENSOR:
        return _expand_per_tensor(scale, scale_shape)
    if granularity == GlobalScaleGranularity.PER_EXPERT:
        return _expand_per_expert(scale, scale_shape, logical_shape, row_shape)
    if granularity == GlobalScaleGranularity.PER_TOKEN:
        return _expand_per_token(scale, scale_shape, logical_shape, row_shape, rows)
    raise AssertionError(f"unreachable granularity: {granularity}")


def expand_global_scale_for_rows(
    global_scale: torch.Tensor,
    logical_shape: torch.Size | tuple[int, ...],
    *,
    device: torch.device,
    granularity: GlobalScaleGranularity,
) -> torch.Tensor:
    """Expand scalar / per-expert / per-token scales for row-wise kernels.

    Supported shapes:
    * scalar / one element: per-tensor scale, kept as one element.
    * ``logical_shape[:-2]``: per-expert scale for grouped 3D inputs. For a
      2D ``(G, K)`` tensor, a 1-D scale of length ``G`` is per-token unless
      per-expert granularity is requested explicitly, which then raises.
    * ``logical_shape[:-1]``: per-token scale.
    * ``(prod(logical_shape[:-1]),)``: flattened per-token scale.

    ``granularity`` is explicit; the shape must match that granularity.
    """
    if global_scale.device != device:
        if (
            global_scale.numel() != 1
            or granularity != GlobalScaleGranularity.PER_TENSOR
        ):
            raise ValueError(
                "global_scale must be on the same device as the tensor being quantized"
            )
        global_scale = global_scale.to(device)

    shape = tuple(logical_shape)
    if len(shape) < 2:
        raise ValueError(f"Expected a tensor shape with at least 2 dims, got {shape}")
    row_shape = shape[:-1]
    rows = math.prod(row_shape)
    scale = global_scale.to(dtype=torch.float32)
    scale_shape = tuple(global_scale.shape)
    return _expand_per_granularity(
        scale,
        scale_shape,
        shape,
        row_shape,
        rows,
        granularity,
    )
