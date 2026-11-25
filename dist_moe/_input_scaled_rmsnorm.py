# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused learned input-scale RMSNorm postprocess for BF16 route outputs."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ._postprocess import _ResolvedExpertsPostprocess, InputScaledRMSNormPostprocess
from .kernels._environment import num_sms_per_device
from .kernels._tensor_views import _same_view, _views_overlap

# Routes per backward block.
_ROWS_PER_BLOCK = 4
# Warps per program, chosen for D=4096 and top-8: forward spreads one token's
# [K, D] FP32 tile over a full 1024-thread CTA; backward, a [4, D] tile over 512.
_FORWARD_WARPS = 32
_BACKWARD_WARPS = 16
# Each kernel's FP32 tile holds at most this many values per thread, as at
# D=4096 and top-8. A 1024-thread CTA gets at most 64 registers per thread, so
# larger tiles would spill; those shapes use the eager callback.
_MAX_TILE_VALUES_PER_THREAD = 32
# The gain reduction sums [64 partials, 32 channels] FP32 tiles, 16 values per
# thread at 4 warps, with one program per 32 channels.
_GAIN_REDUCE_PARTIALS = 64
_GAIN_REDUCE_CHANNELS = 32
_GAIN_REDUCE_WARPS = 4


def _is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return value > 0 and value & (value - 1) == 0


def _fits_tiles(hidden_dim: int, topk: int) -> bool:
    """Return whether both kernels' FP32 tiles fit the per-thread bound."""
    values_per_warp = 32 * _MAX_TILE_VALUES_PER_THREAD  # 32 threads per warp
    return (
        topk * hidden_dim <= _FORWARD_WARPS * values_per_warp
        and _ROWS_PER_BLOCK * hidden_dim <= _BACKWARD_WARPS * values_per_warp
    )


def _validate_gain(
    weight_D: torch.Tensor,
    *,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Validate the gain storage contract.

    Args:
        weight_D: Tensor returned by the policy's ``weight_getter``.
        hidden_dim: Expected length ``D``.
        dtype: Expected dtype.
        device: Expected device.

    Raises:
        ValueError: If the gain is not a contiguous ``[D]`` tensor with the
            input dtype on the input device.
    """
    if (
        not isinstance(weight_D, torch.Tensor)
        or weight_D.shape != (hidden_dim,)
        or weight_D.dtype != dtype
        or weight_D.device != device
        or not weight_D.is_contiguous()
    ):
        raise ValueError(
            f"expected a contiguous {dtype} input-scale gain of shape "
            f"({hidden_dim},) on {device}"
        )


def _resolve_input_scaled_rmsnorm(
    config: InputScaledRMSNormPostprocess,
    *,
    x_TD: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
) -> _ResolvedExpertsPostprocess | None:
    """Validate the gain and resolve the fused kernels when they apply.

    Args:
        config: Public input-scaled RMSNorm policy.
        x_TD: Dist-MoE input with shape ``[T, D]``.
        topk_expert_ids_TK: Routed expert IDs with shape ``[T, K]``; negative
            IDs mark padded routes.
        topk_scores_TK: Router scores with shape ``[T, K]``.

    Returns:
        Forward and backward hooks for the fused kernels, or ``None`` when the
        shapes need the policy's eager callback.

    Raises:
        ValueError: If the gain violates its storage contract.
    """
    num_tokens, hidden_dim = x_TD.shape
    topk = topk_expert_ids_TK.shape[1]
    # The hooks live until backward; capture metadata only, so they keep
    # neither each layer's input nor a resolved gain alive.
    device = x_TD.device
    _validate_gain(
        config.weight_getter(), hidden_dim=hidden_dim, dtype=x_TD.dtype, device=device
    )
    if not (
        x_TD.is_cuda
        and x_TD.dtype == torch.bfloat16
        and topk_scores_TK.dtype == torch.float32
        and _is_power_of_two(hidden_dim)
        and _is_power_of_two(topk)
        and _fits_tiles(hidden_dim, topk)
        and topk_scores_TK.shape == topk_expert_ids_TK.shape == (num_tokens, topk)
        and topk_expert_ids_TK.is_contiguous()
        and topk_scores_TK.is_contiguous()
    ):
        return None
    num_routes = num_tokens * topk

    def weight() -> torch.Tensor:
        """Resolve the live gain and validate its storage contract."""
        weight_D = config.weight_getter()
        _validate_gain(
            weight_D, hidden_dim=hidden_dim, dtype=torch.bfloat16, device=device
        )
        return weight_D

    def validate_routes(h3_MD: torch.Tensor) -> None:
        """Reject route outputs that the kernels would read incorrectly."""
        if (
            h3_MD.shape != (num_routes, hidden_dim)
            or h3_MD.dtype != torch.bfloat16
            or not h3_MD.is_contiguous()
        ):
            raise ValueError(
                "input-scaled RMSNorm expects contiguous BF16 route outputs of "
                f"shape ({num_routes}, {hidden_dim}); got {tuple(h3_MD.shape)} "
                f"{h3_MD.dtype}"
            )

    def forward(
        h3_MD: torch.Tensor,
        scores_TK: torch.Tensor,
        *,
        output_dtype: torch.dtype,
        save: bool,
        save_requires_copy: bool = False,
        save_buffer: torch.Tensor | None = None,
        save_offset: torch.Tensor | None = None,
        save_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, None, torch.dtype]:
        """Reduce the normalized routes, saving H3 in the same launch.

        Args:
            h3_MD: Route-wise W2 output with shape ``[T * K, D]``.
            scores_TK: Router scores with shape ``[T, K]``.
            output_dtype: Reduced-output dtype.
            save: Whether backward state is required.
            save_requires_copy: Unused; H3 is saved into ``save_buffer``.
            save_buffer: Activation buffer that receives H3 when saved.
            save_offset: Byte offset of H3 in ``save_buffer``.
            save_condition: Device predicate; H3 is saved when it is zero.

        Returns:
            Reduced output, no saved tensor or context, and the BF16 dtype
            of the normalized routes.

        Raises:
            ValueError: If saving lacks an activation buffer, or ``h3_MD``
                does not match the resolved shape.
        """
        from .kernels.triton.input_scaled_rmsnorm import route_forward

        validate_routes(h3_MD)
        if save and (
            save_buffer is None or save_offset is None or save_condition is None
        ):
            raise ValueError(
                "input-scaled RMSNorm saves H3 only into an activation buffer; "
                "pass save_buffer, save_offset, and save_condition"
            )
        output_TD = h3_MD.new_empty((num_tokens, hidden_dim), dtype=output_dtype)
        route_forward[(num_tokens,)](
            h3_MD,
            weight(),
            scores_TK,
            topk_expert_ids_TK,
            output_TD,
            save_buffer if save else None,
            save_offset if save else None,
            save_condition if save else None,
            hidden_dim,
            topk,
            config.eps,
            config.gain_center,
            num_warps=_FORWARD_WARPS,
            enable_fp_fusion=False,
        )
        return output_TD, None, None, torch.bfloat16

    def backward(
        grad_output_TD: torch.Tensor,
        h3_saved_MD: torch.Tensor,
        scores_TK: torch.Tensor,
        *,
        postprocess_output_dtype: torch.dtype,
        context: torch.Tensor | None = None,
        publish_view: Callable[[tuple[int, ...], torch.dtype], torch.Tensor]
        | None = None,
        x_buffer: torch.Tensor | None = None,
        x_buffer_offset: torch.Tensor | None = None,
        x_buffer_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Differentiate the fused norm and reduction, publishing H3 gradients.

        Args:
            grad_output_TD: Gradient of the reduced output with shape ``[T, D]``.
            h3_saved_MD: Replayed W2 output in the local combine buffer.
            scores_TK: Router scores with shape ``[T, K]``.
            postprocess_output_dtype: Unused; the routes are BF16.
            context: Unused auxiliary context.
            publish_view: Factory for the combine view that receives the H3
                gradient; it may alias ``h3_saved_MD`` exactly.
            x_buffer: Activation buffer holding H3 when it was saved.
            x_buffer_offset: Byte offset of H3 in ``x_buffer``.
            x_buffer_condition: Device predicate; ``x_buffer`` holds H3 when
                it is zero, and ``h3_saved_MD`` holds it otherwise.

        Returns:
            Published H3 gradient, router-score gradient, and ``True``.

        Raises:
            ValueError: If an argument is missing or the gradient view
                partially overlaps ``h3_saved_MD``.
        """
        from .kernels.triton.input_scaled_rmsnorm import reduce_gain, route_backward

        if (
            publish_view is None
            or x_buffer is None
            or x_buffer_offset is None
            or x_buffer_condition is None
        ):
            raise ValueError(
                "input-scaled RMSNorm backward requires publish_view and the "
                "activation buffer, offset, and condition"
            )
        validate_routes(h3_saved_MD)
        grad_h3_MD = publish_view(tuple(h3_saved_MD.shape), h3_saved_MD.dtype)
        # The kernel reads each block before overwriting it, so the gradient
        # may alias the replayed H3, but must not partially overlap it.
        if not _same_view(grad_h3_MD, h3_saved_MD) and _views_overlap(
            grad_h3_MD, h3_saved_MD
        ):
            raise ValueError(
                "the published H3 gradient must alias the replayed H3 exactly "
                "or not at all"
            )
        grad_output_TD = grad_output_TD.to(torch.bfloat16).contiguous()
        # Resolve the gain under grad mode so a framework-provided view keeps
        # its edge to the live parameter.
        with torch.enable_grad():
            weight_D = weight()
        # Two persistent programs per SM keep every SM busy whether one or two
        # fit; the grid, and so the gain reduction order, is fixed per device.
        num_programs = min(
            2 * num_sms_per_device(device), math.ceil(num_routes / _ROWS_PER_BLOCK)
        )
        # A frozen gain needs no partials, reduction, or autograd edge.
        gain_partials_PD = (
            torch.empty((num_programs, hidden_dim), device=device, dtype=torch.float32)
            if weight_D.requires_grad
            else None
        )
        grad_scores_TK = torch.empty_like(scores_TK)
        route_backward[(num_programs,)](
            h3_saved_MD,
            x_buffer,
            x_buffer_offset,
            x_buffer_condition,
            grad_output_TD,
            weight_D,
            scores_TK,
            topk_expert_ids_TK,
            grad_h3_MD,
            grad_scores_TK,
            gain_partials_PD,
            num_routes,
            hidden_dim,
            topk,
            _ROWS_PER_BLOCK,
            config.eps,
            config.gain_center,
            num_warps=_BACKWARD_WARPS,
            enable_fp_fusion=False,
        )
        if gain_partials_PD is not None:
            grad_weight_D = torch.empty_like(weight_D)
            channels = min(_GAIN_REDUCE_CHANNELS, hidden_dim)
            reduce_gain[(hidden_dim // channels,)](
                gain_partials_PD,
                grad_weight_D,
                num_programs,
                hidden_dim,
                _GAIN_REDUCE_PARTIALS,
                channels,
                num_warps=_GAIN_REDUCE_WARPS,
            )
            torch.autograd.backward(weight_D, grad_weight_D)
        return grad_h3_MD, grad_scores_TK, True

    return _ResolvedExpertsPostprocess(
        forward=forward,
        backward=backward,
        includes_scale_and_sum=True,
        fuses_saved_copy_into_reduction=True,
    )
