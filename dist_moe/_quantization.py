# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block-scaled quantization operations required by distributed MoE."""

from __future__ import annotations

import torch

from .formats import (
    AxisMask,
    block_scaled_format_constants,
    BLOCK_SCALED_FORMAT_IDS,
    BlockScaledFormat,
    BlockScaledProducer,
    resolve_half_range_scale,
    SCALE_FACTOR_LAYOUT_IDS,
    ScaleFactorLayout,
    ScaleReduction,
)
from .kernels.blockscaled_quantize import (
    quantize_block_scaled,
    quantize_block_scaled_axes,
)

QuantizedTensor = tuple[torch.Tensor, torch.Tensor]


def _axis_plan(
    block_size: tuple[tuple[int, int], ...] | None,
    vector_size: int,
) -> tuple[int, int, bool]:
    """Resolve the compact set of quantization modes used by DistMoE.

    Args:
        block_size: Requested block shapes.
        vector_size: Number of values sharing one scale factor.

    Returns:
        Axis mask, scale-reduction mode, and row-fast-path flag.

    Raises:
        NotImplementedError: If the requested shape is outside DistMoE's modes.
    """
    row = ((1, vector_size),)
    col = ((vector_size, 1),)
    both = ((vector_size, 1), (1, vector_size))
    tile = ((vector_size, vector_size),)
    normalized = row if block_size is None else tuple(tuple(x) for x in block_size)
    if normalized == row:
        return AxisMask.K.value, ScaleReduction.ONE_D.value, True
    if normalized == col:
        return AxisMask.M.value, ScaleReduction.ONE_D.value, False
    if normalized in (both, tuple(reversed(both))):
        return (AxisMask.M | AxisMask.K).value, ScaleReduction.ONE_D.value, False
    if normalized == tile:
        return (AxisMask.M | AxisMask.K).value, ScaleReduction.TWO_D.value, False
    raise NotImplementedError(
        f"block_size={block_size!r} is not a distributed MoE quantization mode"
    )


def quantize_for_format(
    x: torch.Tensor,
    format: BlockScaledFormat,
    *,
    layout: ScaleFactorLayout = ScaleFactorLayout.CUBLAS_BLOCKED,
    block_size: tuple[tuple[int, int], ...] | None = None,
    gather_output_shape: tuple[int, int] | None = None,
    gather_dtype: torch.dtype | None = None,
    local_rank: int = 0,
    world_size: int = 1,
) -> QuantizedTensor | tuple[QuantizedTensor, QuantizedTensor]:
    """Quantize a local tensor or zero-copy peer-gather pointer table.

    Args:
        x: BF16/FP16/FP32 source, or an int64 row-pointer table.
        format: Target block-scaled operand format.
        layout: Scale-factor layout. DistMoE requires CUBLAS-blocked scales.
        block_size: Row, column, both-axis, or two-dimensional block mode.
        gather_output_shape: Logical source shape for pointer-table input.
        gather_dtype: Logical source dtype for pointer-table input.
        local_rank: Rank encoded in zero-copy pointer metadata.
        world_size: Number of ranks encoded in zero-copy pointer metadata.
    Returns:
        One quantized pair, or column and row pairs for two-axis modes.

    Raises:
        ValueError: If a gather request or layout is invalid.
    """
    if layout != ScaleFactorLayout.CUBLAS_BLOCKED:
        raise ValueError("DistMoE quantization requires CUBLAS_BLOCKED scales")
    _, _, vector_size = block_scaled_format_constants(format)
    half_range_scale = resolve_half_range_scale(format, None)
    axis_mask, scale_reduction, row_fast_path = _axis_plan(block_size, vector_size)
    gather = gather_output_shape is not None or gather_dtype is not None
    if gather != (x.dtype == torch.int64):
        raise ValueError(
            "gather mode requires an int64 pointer table and shape metadata"
        )
    producer_id = (
        BlockScaledProducer.ZEROCOPY_GATHER.value
        if gather
        else BlockScaledProducer.IDENTITY.value
    )
    if row_fast_path:
        return quantize_block_scaled(
            x,
            format=format,
            layout=layout,
            producer_id=producer_id,
            output_shape=gather_output_shape,
            gather_dtype=gather_dtype,
            local_rank=local_rank,
            world_size=world_size,
            half_range_scale=half_range_scale,
        )
    return quantize_block_scaled_axes(
        x,
        format_id=BLOCK_SCALED_FORMAT_IDS[format],
        axis_mask=axis_mask,
        scale_reduction_id=scale_reduction,
        layout_id=SCALE_FACTOR_LAYOUT_IDS[layout],
        producer_id=producer_id,
        output_shape=gather_output_shape,
        gather_dtype=gather_dtype,
        local_rank=local_rank,
        world_size=world_size,
        half_range_scale=half_range_scale,
    )


def _split_swiglu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a non-interleaved fused gate/up activation.

    Args:
        x: Tensor whose last dimension contains gate followed by up values.

    Returns:
        Gate and up views.
    """
    if x.shape[-1] % 2:
        raise ValueError("SwiGLU input width must be even")
    return x.chunk(2, dim=-1)


def swiglu_fwd_quantize(
    h1_M2F: torch.Tensor,
    *,
    format: BlockScaledFormat,
    fast_math: bool,
) -> tuple[QuantizedTensor, QuantizedTensor]:
    """Compute SwiGLU and materialize column and row quantized operands.

    Args:
        h1_M2F: BF16 fused gate/up activation with shape ``[M, 2F]``.
        format: Target block-scaled format.
        fast_math: Whether to use the original approximate sigmoid path.

    Returns:
        Column-oriented WGRAD and row-oriented FPROP operands.
    """
    gate, up = _split_swiglu(h1_M2F)
    result = quantize_block_scaled_axes(
        gate,
        format_id=BLOCK_SCALED_FORMAT_IDS[format],
        axis_mask=(AxisMask.M | AxisMask.K).value,
        scale_reduction_id=ScaleReduction.ONE_D.value,
        layout_id=SCALE_FACTOR_LAYOUT_IDS[ScaleFactorLayout.CUBLAS_BLOCKED],
        producer_b=up,
        producer_id=BlockScaledProducer.SWIGLU_FWD.value,
        fast_math=fast_math,
        half_range_scale=resolve_half_range_scale(format, None),
    )
    assert isinstance(result, tuple) and len(result) == 2
    return result


def swiglu_bwd_quantize(
    grad_h2_MF: torch.Tensor,
    h1_M2F: torch.Tensor,
    *,
    format: BlockScaledFormat,
    fast_math: bool,
) -> tuple[QuantizedTensor, QuantizedTensor]:
    """Compute SwiGLU backward and quantize DXY for dgrad and wgrad.

    Args:
        grad_h2_MF: Gradient of the SwiGLU output with shape ``[M, F]``.
        h1_M2F: Saved fused gate/up activation with shape ``[M, 2F]``.
        format: Target block-scaled format.
        fast_math: Whether to use approximate sigmoid math.

    Returns:
        Column-oriented WGRAD and row-oriented DGRAD operands.
    """
    gate, up = _split_swiglu(h1_M2F)
    return quantize_block_scaled_axes(
        grad_h2_MF,
        format_id=BLOCK_SCALED_FORMAT_IDS[format],
        axis_mask=(AxisMask.M | AxisMask.K).value,
        scale_reduction_id=ScaleReduction.ONE_D.value,
        layout_id=SCALE_FACTOR_LAYOUT_IDS[ScaleFactorLayout.CUBLAS_BLOCKED],
        producer_b=gate,
        producer_c=up,
        producer_id=BlockScaledProducer.SWIGLU_BWD_DXY.value,
        output_shape=(grad_h2_MF.shape[0], 2 * grad_h2_MF.shape[1]),
        source_cols=grad_h2_MF.shape[1],
        fast_math=fast_math,
        half_range_scale=resolve_half_range_scale(format, None),
    )
