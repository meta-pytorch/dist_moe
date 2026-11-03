# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton dequantization kernels for NVFP4 backward operands."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .formats import (
    CUBLAS_BLOCKED_ENTRIES_PER_ATOM,
    CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE,
    CUBLAS_BLOCKED_ROW_LANES,
    CUBLAS_BLOCKED_ROWS_PER_ATOM,
    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM,
    ScaleFactorLayoutId,
)

_BLOCK_SIZE = 256
_TL_SCALE_FACTOR_LAYOUT_NATURAL = tl.constexpr(ScaleFactorLayoutId.NATURAL.value)
_TL_CUBLAS_BLOCKED_ENTRIES_PER_ATOM = tl.constexpr(CUBLAS_BLOCKED_ENTRIES_PER_ATOM)
_TL_CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE = tl.constexpr(
    CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE
)
_TL_CUBLAS_BLOCKED_ROW_LANES = tl.constexpr(CUBLAS_BLOCKED_ROW_LANES)
_TL_CUBLAS_BLOCKED_ROWS_PER_ATOM = tl.constexpr(CUBLAS_BLOCKED_ROWS_PER_ATOM)
_TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM = tl.constexpr(
    CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
)


@triton.jit
def _decode_fp4_e2m1(code):
    """Decode one sign-magnitude E2M1 nibble into FP32."""
    magnitude_code = code & 0x7
    magnitude = tl.where(
        magnitude_code <= 4,
        magnitude_code.to(tl.float32) * 0.5,
        tl.where(
            magnitude_code == 5,
            3.0,
            tl.where(magnitude_code == 6, 4.0, 6.0),
        ),
    )
    return tl.where((code & 0x8) != 0, -magnitude, magnitude)


@triton.jit
def _cublas_blocked_scale_offset(row, col, n_col_blocks):
    """Return the byte offset for one logical CUBLAS-blocked scale."""
    row_atom = row // _TL_CUBLAS_BLOCKED_ROWS_PER_ATOM
    row_in_atom = row - row_atom * _TL_CUBLAS_BLOCKED_ROWS_PER_ATOM
    col_atom = col // _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    col_in_atom = col - col_atom * _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM

    atom_id = row_atom * n_col_blocks + col_atom
    atom_base = atom_id * _TL_CUBLAS_BLOCKED_ENTRIES_PER_ATOM
    row_lane = row_in_atom % _TL_CUBLAS_BLOCKED_ROW_LANES
    row_band = row_in_atom // _TL_CUBLAS_BLOCKED_ROW_LANES
    lane_base = row_lane * _TL_CUBLAS_BLOCKED_ENTRIES_PER_ROW_LANE
    return (
        atom_base
        + lane_base
        + row_band * _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
        + col_in_atom
    )


@triton.jit
def _dequantize_nvfp4_kernel(
    qdata_ptr,
    scale_ptr,
    global_scale_inv_ptr,
    output_ptr,
    activation_buffer_ptr,
    activation_offset_ptr,
    condition_ptr,
    qdata_row_stride,
    scale_row_stride,
    global_scale_inv_stride,
    rows_per_group,
    group_scale_elements,
    n_col_blocks,
    n_cols,
    SCALE_LAYOUT: tl.constexpr,
    OUTER_SCALE_PER_ROW: tl.constexpr,
    WRITE_ACTIVATION_BUFFER: tl.constexpr,
    HAS_CONDITION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if HAS_CONDITION:
        if tl.load(condition_ptr, eviction_policy="evict_last") == 0:
            return

    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = cols < n_cols
    row_i64 = row.to(tl.int64)
    cols_i64 = cols.to(tl.int64)

    packed = tl.load(
        qdata_ptr + row_i64 * qdata_row_stride + cols_i64 // 2,
        mask=valid,
        other=0,
    )
    code = tl.where((cols & 1) == 0, packed & 0xF, packed >> 4)
    values = _decode_fp4_e2m1(code)

    scale_col = cols // 16
    if SCALE_LAYOUT == _TL_SCALE_FACTOR_LAYOUT_NATURAL:
        scale_offset = row_i64 * scale_row_stride + scale_col.to(tl.int64)
    else:
        group = row // rows_per_group
        row_in_group = row - group * rows_per_group
        scale_offset = group.to(
            tl.int64
        ) * group_scale_elements + _cublas_blocked_scale_offset(
            row_in_group,
            scale_col,
            n_col_blocks,
        ).to(tl.int64)
    inner_scale = tl.load(scale_ptr + scale_offset, mask=valid, other=0.0).to(
        tl.float32
    )

    outer_row = row if OUTER_SCALE_PER_ROW else row // rows_per_group
    global_scale_inv = tl.load(
        global_scale_inv_ptr + outer_row.to(tl.int64) * global_scale_inv_stride
    ).to(tl.float32)
    if WRITE_ACTIVATION_BUFFER:
        byte_offset = tl.load(
            activation_offset_ptr,
            eviction_policy="evict_last",
        ).to(tl.int64)
        output_base = (activation_buffer_ptr + byte_offset).to(
            tl.pointer_type(tl.bfloat16)
        )
    else:
        output_base = output_ptr
    tl.store(
        output_base + row_i64 * n_cols + cols_i64,
        values * inner_scale * global_scale_inv,
        mask=valid,
    )


@triton.jit
def _dequantize_nvfp4_gather_kernel(
    gather_ptrs,
    num_rows_ptr,
    output_buffer_ptr,
    output_offset_ptr,
    condition_ptr,
    n_cols,
    EXECUTE_WHEN_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    condition = tl.load(condition_ptr, eviction_policy="evict_last")
    should_execute = condition == 0 if EXECUTE_WHEN_ZERO else condition != 0
    if not should_execute:
        return

    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_rows = tl.load(num_rows_ptr, eviction_policy="evict_last")
    valid_col = cols < n_cols
    valid_row = row < num_rows
    row_address = tl.load(gather_ptrs + row)
    valid_source = valid_row & (row_address != 0)

    packed_ptr = row_address.to(tl.pointer_type(tl.uint8))
    packed = tl.load(
        packed_ptr + cols // 2,
        mask=valid_col & valid_source,
        other=0,
    )
    code = tl.where((cols & 1) == 0, packed & 0xF, packed >> 4)
    values = _decode_fp4_e2m1(code)

    q_cols = n_cols // 2
    scale_cols = n_cols // 16
    scale_ptr = row_address.to(tl.pointer_type(tl.float8e4nv))
    inner_scale = tl.load(
        scale_ptr + q_cols + cols // 16,
        mask=valid_col & valid_source,
        other=0.0,
    ).to(tl.float32)
    global_scale_inv_ptr = (row_address + q_cols + scale_cols).to(
        tl.pointer_type(tl.float32)
    )
    global_scale_inv = tl.load(
        global_scale_inv_ptr,
        mask=valid_source,
        other=0.0,
    ).to(tl.float32)

    output_offset = tl.load(
        output_offset_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    output = (output_buffer_ptr + output_offset).to(tl.pointer_type(tl.bfloat16))
    tl.store(
        output + row.to(tl.int64) * n_cols + cols.to(tl.int64),
        values * inner_scale * global_scale_inv,
        mask=valid_col & valid_row,
    )


@triton.jit
def _dequantize_nvfp4_activation_buffer_kernel(
    activation_buffer_ptr,
    qdata_offset_ptr,
    scale_offset_ptr,
    num_rows_ptr,
    output_offset_ptr,
    condition_ptr,
    n_cols,
    EXECUTE_WHEN_ZERO: tl.constexpr,
    ROW_SCALE_STORAGE_MULTIPLE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Decode one planner-owned row bundle into planner-owned BF16 storage."""
    condition = tl.load(condition_ptr, eviction_policy="evict_last")
    should_execute = condition == 0 if EXECUTE_WHEN_ZERO else condition != 0
    if not should_execute:
        return

    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_rows = tl.load(num_rows_ptr, eviction_policy="evict_last").to(tl.int64)
    valid = (row.to(tl.int64) < num_rows) & (cols < n_cols)
    row_i64 = row.to(tl.int64)
    cols_i64 = cols.to(tl.int64)
    qdata_offset = tl.load(qdata_offset_ptr, eviction_policy="evict_last").to(tl.int64)
    scale_offset = tl.load(scale_offset_ptr, eviction_policy="evict_last").to(tl.int64)
    qdata = (activation_buffer_ptr + qdata_offset).to(tl.pointer_type(tl.uint8))
    packed = tl.load(
        qdata + row_i64 * (n_cols // 2) + cols_i64 // 2,
        mask=valid,
        other=0,
    )
    code = tl.where((cols & 1) == 0, packed & 0xF, packed >> 4)
    values = _decode_fp4_e2m1(code)

    scale_cols = n_cols // 16
    scales = (activation_buffer_ptr + scale_offset).to(tl.pointer_type(tl.float8e4nv))
    n_col_blocks = (scale_cols + _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM - 1) // (
        _TL_CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    )
    inner_scale = tl.load(
        scales
        + _cublas_blocked_scale_offset(
            row,
            cols // 16,
            n_col_blocks,
        ).to(tl.int64),
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    global_scale_offset = (
        scale_offset + num_rows * ROW_SCALE_STORAGE_MULTIPLE * scale_cols
    )
    global_scale_inv = tl.load(
        (activation_buffer_ptr + global_scale_offset).to(tl.pointer_type(tl.float32))
        + row_i64,
        mask=row_i64 < num_rows,
        other=0.0,
    ).to(tl.float32)

    output_offset = tl.load(
        output_offset_ptr,
        eviction_policy="evict_last",
    ).to(tl.int64)
    output = (activation_buffer_ptr + output_offset).to(tl.pointer_type(tl.bfloat16))
    tl.store(
        output + row_i64 * n_cols + cols_i64,
        values * inner_scale * global_scale_inv,
        mask=valid,
    )


def dequantize_nvfp4(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    global_scale_inv: torch.Tensor,
    *,
    scale_layout: ScaleFactorLayoutId,
    rows_per_group: int,
    outer_scale_per_row: bool,
) -> torch.Tensor:
    """Dequantize packed NVFP4 rows into a contiguous BF16 tensor.

    Args:
        qdata: Rank-two packed E2M1 rows. Strided rows are supported.
        scale: Natural or CUBLAS-blocked E4M3 scales.
        global_scale_inv: FP32 inverse outer scales, per row or per group.
        scale_layout: Physical layout of ``scale``.
        rows_per_group: Logical rows belonging to one scale-layout group.
        outer_scale_per_row: Whether each logical row owns an outer scale.

    Returns:
        Contiguous BF16 rows with twice the packed column count.
    """
    rows, q_cols = qdata.shape
    n_cols = q_cols * 2
    output = torch.empty((rows, n_cols), dtype=torch.bfloat16, device=qdata.device)
    if rows == 0 or n_cols == 0:
        return output

    scale_cols = triton.cdiv(n_cols, 16)
    n_col_blocks = triton.cdiv(scale_cols, CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM)
    padded_rows = (
        triton.cdiv(rows_per_group, CUBLAS_BLOCKED_ROWS_PER_ATOM)
        * CUBLAS_BLOCKED_ROWS_PER_ATOM
    )
    group_scale_elements = (
        padded_rows * n_col_blocks * CUBLAS_BLOCKED_SCALE_COLS_PER_ATOM
    )
    scale_row_stride = (
        scale.stride(0) if scale_layout == ScaleFactorLayoutId.NATURAL else 0
    )
    grid = (rows, triton.cdiv(n_cols, _BLOCK_SIZE))
    _dequantize_nvfp4_kernel[grid](
        qdata.view(torch.uint8),
        scale,
        global_scale_inv,
        output,
        output,
        global_scale_inv,
        global_scale_inv,
        qdata.stride(0),
        scale_row_stride,
        global_scale_inv.stride(0),
        rows_per_group,
        group_scale_elements,
        n_col_blocks,
        n_cols,
        SCALE_LAYOUT=scale_layout.value,
        OUTER_SCALE_PER_ROW=outer_scale_per_row,
        WRITE_ACTIVATION_BUFFER=False,
        HAS_CONDITION=False,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
    return output


def dequantize_gathered_nvfp4_to_activation_buffer(
    gather_ptrs: torch.Tensor,
    num_rows: torch.Tensor,
    activation_buffer: torch.Tensor,
    activation_offset: torch.Tensor,
    condition: torch.Tensor,
    *,
    n_cols: int,
    execute_when_zero: bool,
) -> None:
    """Dequantize peer-addressed packed rows into planner-owned BF16 storage.

    Args:
        gather_ptrs: Capacity-sized table of packed symmetric-memory row addresses.
        num_rows: Device scalar containing the valid padded-row count.
        activation_buffer: Planner-owned one-dimensional byte storage.
        activation_offset: One-element device byte offset for the destination.
        condition: One-element device predicate.
        n_cols: Logical BF16 feature width.
        execute_when_zero: Whether a zero predicate selects this operation.
    """
    if gather_ptrs.numel() == 0 or n_cols == 0:
        return
    grid = (gather_ptrs.numel(), triton.cdiv(n_cols, _BLOCK_SIZE))
    _dequantize_nvfp4_gather_kernel[grid](
        gather_ptrs,
        num_rows,
        activation_buffer,
        activation_offset,
        condition,
        n_cols,
        EXECUTE_WHEN_ZERO=execute_when_zero,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )


def dequantize_nvfp4_activation_buffer_to_activation_buffer(
    activation_buffer: torch.Tensor,
    qdata_offset: torch.Tensor,
    scale_offset: torch.Tensor,
    num_rows: torch.Tensor,
    output_offset: torch.Tensor,
    condition: torch.Tensor,
    *,
    capacity_rows: int,
    n_cols: int,
    execute_when_zero: bool,
    row_scale_storage_multiple: int,
) -> None:
    """Dequantize one offset-backed NVFP4 row bundle into BF16 storage.

    Args:
        activation_buffer: Planner-owned byte-addressed allocation.
        qdata_offset: Device scalar byte offset of packed E2M1 rows.
        scale_offset: Device scalar byte offset of CUBLAS-blocked E4M3 scales.
        num_rows: Device scalar containing the valid padded-row count.
        output_offset: Device scalar byte offset of the BF16 destination.
        condition: Device predicate selecting this operation.
        capacity_rows: Static maximum row count used for graph-stable launch.
        n_cols: Logical BF16 feature width.
        execute_when_zero: Whether a zero predicate selects this operation.
        row_scale_storage_multiple: Physical row-scale padding multiplier.
    """
    if capacity_rows == 0 or n_cols == 0:
        return
    grid = (capacity_rows, triton.cdiv(n_cols, _BLOCK_SIZE))
    _dequantize_nvfp4_activation_buffer_kernel[grid](
        activation_buffer,
        qdata_offset,
        scale_offset,
        num_rows,
        output_offset,
        condition,
        n_cols,
        EXECUTE_WHEN_ZERO=execute_when_zero,
        ROW_SCALE_STORAGE_MULTIPLE=row_scale_storage_multiple,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=4,
    )
