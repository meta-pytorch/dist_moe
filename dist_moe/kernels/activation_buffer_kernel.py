# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cutlass-side activation-buffer support for the DistMoE kernels.

The quantized-operand staging shapes/placeholders and the device-side
buffer views and extents (free ``@cute.jit`` helpers). The torch-only
offset table and launch validation live in ``activation_buffer``.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .activation_buffer import (
    _ceil_div,
    _FP4_FORMAT_NAMES,
    ACTIVATION_A_SCALE_OFFSET,
    activation_buffer_placeholder as _activation_buffer_placeholder,
    ACTIVATION_ROW_SCALE_STORAGE_MULTIPLE,
    prepare_activation_buffer_launch,
)
from .config import (
    BLOCKSCALED_DISPATCH_DIM_ALIGNMENT,
    BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT,
    uses_paged_blockscaled_scale_rows,
)
from .grouped_gemm_kernel import (
    _require_valid_activation_buffer_range,
)
from .params import (
    _torch_dtype,
    BlockScaledFormatSpec,
    ceil_div,
)


def _compact_scale_storage_rows(rows: int, m_multiple_of: int) -> int:
    if m_multiple_of == 32:
        return rows * 4
    if m_multiple_of == 64:
        return rows * 2
    if uses_paged_blockscaled_scale_rows(m_multiple_of):
        scale_page_rows = _ceil_div(m_multiple_of, 128) * 128
        return _ceil_div(rows, m_multiple_of) * scale_page_rows
    return rows


def _dispatch_quant_source_dtype_from_torch(
    dtype: torch.dtype,
) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise NotImplementedError(f"Unsupported fused DISPATCH quant source dtype {dtype}")


def _dispatch_quant_scale_shape(rows: int, dim: int, format: BlockScaledFormatSpec):
    if (
        rows % BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT != 0
        or dim % BLOCKSCALED_DISPATCH_DIM_ALIGNMENT != 0
    ):
        raise NotImplementedError(
            "fused blockscaled DISPATCH quant requires rows to be a multiple of "
            f"{BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT} and dim to be a multiple of "
            f"{BLOCKSCALED_DISPATCH_DIM_ALIGNMENT}; got rows={rows}, dim={dim}"
        )
    return (rows, dim // format.sf_vec_size)


def _quantized_storage_dim(dim: int, format: BlockScaledFormatSpec) -> int:
    return dim // 2 if format.name in _FP4_FORMAT_NAMES else dim


def _column_quantized_storage_shape(
    rows: int,
    dim: int,
    format: BlockScaledFormatSpec,
) -> tuple[int, int]:
    if format.name in _FP4_FORMAT_NAMES:
        assert rows == 0 or rows % 8 == 0, f"FP4 rows must be 8-aligned; got {rows}"
        return (dim, max(1, _ceil_div(rows, 8)) * 4)
    return (rows, dim)


def _column_quantized_scale_shape(
    rows: int,
    dim: int,
    format: BlockScaledFormatSpec,
) -> tuple[int, int]:
    assert rows == 0 or rows % format.sf_vec_size == 0, (
        f"column quantization rows must be {format.sf_vec_size}-aligned; got {rows}"
    )
    return (dim, max(1, _ceil_div(rows, format.sf_vec_size)))


def _column_quantized_output(
    storage: torch.Tensor,
    format: BlockScaledFormatSpec,
) -> torch.Tensor:
    return storage if format.name in _FP4_FORMAT_NAMES else storage.t()


def _column_quantized_output_shape(
    rows: int,
    dim: int,
    format: BlockScaledFormatSpec,
) -> tuple[int, int]:
    storage_shape = _column_quantized_storage_shape(rows, dim, format)
    return storage_shape if format.name in _FP4_FORMAT_NAMES else storage_shape[::-1]


def _empty_qdata(
    shape: tuple[int, ...],
    format: BlockScaledFormatSpec,
    device: torch.device,
) -> torch.Tensor:
    dtype = _torch_dtype(format.a_dtype)
    if format.name in _FP4_FORMAT_NAMES:
        return torch.empty(shape, dtype=torch.uint8, device=device).view(dtype)
    return torch.empty(shape, dtype=dtype, device=device)


def _empty_dispatch_quantized_operand(
    *,
    rows: int,
    scale_rows: int | None = None,
    dim: int,
    format: BlockScaledFormatSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale_rows is None:
        scale_rows = rows
    q = _empty_qdata(
        (rows, _quantized_storage_dim(dim, format)),
        format,
        device,
    )
    scale = torch.empty(
        _dispatch_quant_scale_shape(scale_rows, dim, format),
        dtype=_torch_dtype(format.sf_dtype),
        device=device,
    )
    return q, scale.view(-1)


def _dispatch_quantized_operand_placeholder(
    *,
    rows: int,
    scale_rows: int | None = None,
    dim: int,
    format: BlockScaledFormatSpec,
    device: torch.device,
    activation_buffer: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale_rows is None:
        scale_rows = rows
    if activation_buffer is None:
        return _empty_dispatch_quantized_operand(
            rows=rows,
            scale_rows=scale_rows,
            dim=dim,
            format=format,
            device=device,
        )
    q = _activation_buffer_placeholder(
        activation_buffer,
        (rows, _quantized_storage_dim(dim, format)),
        _torch_dtype(format.a_dtype),
    )
    scale = _activation_buffer_placeholder(
        activation_buffer,
        _dispatch_quant_scale_shape(scale_rows, dim, format),
        _torch_dtype(format.sf_dtype),
    )
    return q, scale.view(-1)


def _prepare_activation_buffer_launch_args(
    *,
    activation_buffer: torch.Tensor | None,
    activation_offsets: torch.Tensor | None,
    conditional_execution: torch.Tensor | None,
    fallback_offsets: torch.Tensor,
    expected_offset_count: int,
    enable_tvm_ffi: bool,
):
    (
        activation_buffer_base_ptr,
        offsets,
        use_activation_buffer,
        condition,
        use_conditional_execution,
        condition_owner,
        activation_buffer_size_bytes,
    ) = prepare_activation_buffer_launch(
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        fallback_offsets=fallback_offsets,
        expected_offset_count=expected_offset_count,
    )
    return (
        activation_buffer_base_ptr,
        from_dlpack(
            offsets,
            assumed_align=8,
            enable_tvm_ffi=enable_tvm_ffi,
        ),
        use_activation_buffer,
        from_dlpack(
            condition,
            assumed_align=4,
            enable_tvm_ffi=enable_tvm_ffi,
        ),
        use_conditional_execution,
        condition_owner,
        activation_buffer_size_bytes,
    )


@cute.jit
def _activation_buffer_tensor_with_byte_extent(
    template: cute.Tensor,
    activation_buffer_base_ptr: cutlass.Int64,
    activation_buffer_size_bytes: cutlass.Int64,
    activation_offsets: cute.Tensor,
    offset_idx: cutlass.Constexpr[int],
    element_type: cutlass.Constexpr[type[cutlass.Numeric]],
    byte_extent: cutlass.Int64,
):
    """Rebase an operand whose ABI offset is required to be addressable."""
    if cutlass.const_expr(template.element_type != element_type):
        raise TypeError("activation buffer placeholder has the wrong element type")
    byte_offset = cutlass.Int64(activation_offsets[offset_idx])
    element_size = cutlass.const_expr(template.element_type.width // 8)
    if (cute.arch.block_idx()[0] == 0) & (cute.arch.thread_idx()[0] == 0):
        _require_valid_activation_buffer_range(
            byte_offset=byte_offset,
            byte_extent=byte_extent,
            activation_buffer_size_bytes=activation_buffer_size_bytes,
            warp_scoped_diagnostic=False,
        )
    if cutlass.const_expr(element_type in (cutlass.Uint8, cutlass.Uint32)):
        element_offset = byte_offset // cutlass.Int64(element_size)
        coord = (0,) * (cute.rank(template) - 1) + (element_offset,)
        return cute.domain_offset(coord, template)
    ptr = cute.make_ptr(
        element_type,
        activation_buffer_base_ptr + byte_offset,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return cute.make_tensor(ptr, template.layout)


@cute.jit
def _activation_buffer_row_scale_storage_byte_extent(
    rows: cutlass.Int32,
    K: cutlass.Int32,
    sf_vec_size: cutlass.Constexpr[int],
    sf_dtype_width: cutlass.Constexpr[int],
) -> cutlass.Int64:
    # Activation-buffer kernels are cached across row capacities, so this
    # extent must depend on runtime rows rather than the compile-time tensor.
    scale_cols = ceil_div(K, cutlass.Int32(sf_vec_size))
    return (
        cutlass.Int64(rows)
        * cutlass.Int64(ACTIVATION_ROW_SCALE_STORAGE_MULTIPLE)
        * cutlass.Int64(scale_cols)
        * cutlass.Int64(cutlass.const_expr(sf_dtype_width))
        // cutlass.Int64(8)
    )


@cute.jit
def _activation_buffer_col_scale_storage_byte_extent(
    rows: cutlass.Int32,
    K: cutlass.Int32,
    sf_vec_size: cutlass.Constexpr[int],
    sf_dtype_width: cutlass.Constexpr[int],
) -> cutlass.Int64:
    scale_rows = ceil_div(rows, cutlass.Int32(sf_vec_size))
    return (
        cutlass.Int64(K)
        * cutlass.Int64(scale_rows)
        * cutlass.Int64(cutlass.const_expr(sf_dtype_width))
        // cutlass.Int64(8)
    )


@cute.jit
def _activation_buffer_row_global_scale_tensor(
    template: cute.Tensor,
    activation_buffer_base_ptr: cutlass.Int64,
    activation_buffer_size_bytes: cutlass.Int64,
    activation_offsets: cute.Tensor,
    rows: cutlass.Int32,
    K: cutlass.Int32,
    sf_vec_size: cutlass.Constexpr[int],
    sf_dtype_width: cutlass.Constexpr[int],
    scale_offset_idx: cutlass.Constexpr[int] = ACTIVATION_A_SCALE_OFFSET,
) -> cute.Tensor:
    scale_bytes = _activation_buffer_row_scale_storage_byte_extent(
        rows,
        K,
        sf_vec_size,
        sf_dtype_width,
    )
    byte_offset = cutlass.Int64(activation_offsets[scale_offset_idx]) + cutlass.Int64(
        scale_bytes
    )
    if (cute.arch.block_idx()[0] == 0) & (cute.arch.thread_idx()[0] == 0):
        _require_valid_activation_buffer_range(
            byte_offset=byte_offset,
            byte_extent=cutlass.Int64(rows) * cutlass.Int64(4),
            activation_buffer_size_bytes=activation_buffer_size_bytes,
            warp_scoped_diagnostic=False,
        )
    ptr = cute.make_ptr(
        cutlass.Float32,
        activation_buffer_base_ptr + byte_offset,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return cute.make_tensor(ptr, template.layout)
