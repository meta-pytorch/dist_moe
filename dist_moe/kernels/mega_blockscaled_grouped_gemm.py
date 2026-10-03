# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mega fused block-scaled grouped GEMM entry points for dist_moe backward."""

import cutlass
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

from ..formats import (
    canonical_swiglu_clamp,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)
from ._environment import num_sms_per_device
from .activation_buffer import (
    _ceil_div,
    _qdata_words,
    MEGA_ACTIVATION_OFFSET_COUNT,
)
from .activation_buffer_kernel import (
    _column_quantized_output,
    _column_quantized_output_shape,
    _column_quantized_scale_shape,
    _column_quantized_storage_shape,
    _dispatch_quant_source_dtype_from_torch,
    _dispatch_quantized_operand_placeholder,
    _empty_dispatch_quantized_operand,
    _empty_qdata,
    _prepare_activation_buffer_launch_args,
)
from .blockscaled_grouped_gemm import (
    _compile_or_get,
    _config_cache_key,
    _cutlass_dtype_from_torch,
    _format_uses_fp4,
    _make_launch_tensor_bundle,
    _make_pointer_stride_args,
    _parse_blockscaled_problem,
    _resolve_output_tensor,
    _set_swiglu_clamp_config,
    _torch_dtype,
    _torch_layout_signature,
    _validate_2d_operand_layout,
    _validate_3d_operand_layout,
    _validate_blockscaled_launch_inputs,
    auto_blockscaled_config,
    BlockScaledFormatSpec,
    make_mixed_blockscaled_format,
    MXFP4,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
)
from .config import (
    DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    uses_paged_blockscaled_scale_rows,
)
from .dist_blockscaled_grouped_gemm import (
    _activation_buffer_placeholder,
    _can_use_fused_combine_swiglu_quant,
    _can_use_fused_dispatch_quant,
    _flatten_last_dim,
    _logical_dim_size,
    _maybe_slice_output,
    _resolve_combine_descriptor_tensor,
)
from .grouped_gemm import _DGRAD, _WGRAD
from .mega_blockscaled_grouped_gemm_kernel import (
    _MegaDispatchPlan,
    _MegaFusedUnsupportedError,
    _TensorValidationCache,
    MegaBlockScaledGroupedGemmKernel,
)

__all__ = [
    "MXFP4",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "NVFP4",
    "MegaBlockScaledGroupedGemmKernel",
    "_mega_dispatch_quant_split_metadata",
    "mega_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine",
    "mega_blockscaled_grouped_gemm_dgrad_wgrad_dispatch",
]


_MEGA_DISPATCH_QUANT_SPLIT_METADATA_CACHE: dict[tuple, int] = {}
_MEGA_DISPATCH_QUANT_SPLIT_METADATA_SHAPE_CACHE: dict[tuple, int | None] = {}


_MEGA_QUANT_SPLIT_ALIGNMENT_CHECKED = _TensorValidationCache()
_MEGA_PADDED_SCALE_CAPACITY_CHECKED = _TensorValidationCache()
_MEGA_BLOCKSCALED_KERNEL_CACHE: dict[tuple, MegaBlockScaledGroupedGemmKernel] = {}
_MEGA_SPLIT_UNSUPPORTED = -1


def _cache_shape_metadata(shape_cache: dict, shape_key: tuple, metadata) -> None:
    shape_cached = shape_cache.get(shape_key)
    if shape_cached is None and shape_key in shape_cache:
        return
    if shape_cached is None:
        shape_cache[shape_key] = metadata
    elif shape_cached != metadata:
        shape_cache[shape_key] = None


def _validate_mega_quant_split_alignment(
    split_sizes: torch.Tensor,
    *,
    dim: int,
    format: BlockScaledFormatSpec,
    mode: str,
    dim_name: str,
) -> None:
    """Reject quant splits whose per-group tail rows would be dropped.

    Quant producers derive per-group row blocks as ``m_size // sf_vec_size``,
    whether they are scheduled from host metadata or from the runtime split
    tensor, so a group that is not a multiple of ``sf_vec_size`` silently leaves
    its tail rows unquantized. Reading ``split_sizes`` costs a device sync and is
    illegal mid-capture, so values are checked once per split buffer and left to
    the pre-capture warmup while a graph is being captured.
    """
    if dim % format.sf_vec_size != 0:
        raise ValueError(
            f"mega blockscaled {mode} quant requires {dim_name} to be a multiple of "
            f"sf_vec_size={format.sf_vec_size}, got {dim_name}={dim}"
        )
    # CUDA split sizes come from routing, which rounds every expert count to
    # the GEMM's m_multiple_of. Reading them here serializes fresh eager routes.
    if split_sizes.is_cuda:
        return
    checked_key = (tuple(split_sizes.shape), format.sf_vec_size)
    if (split_sizes, checked_key) in _MEGA_QUANT_SPLIT_ALIGNMENT_CHECKED:
        return
    split_sizes_cpu = split_sizes.detach().to("cpu", non_blocking=False)
    if bool((split_sizes_cpu % format.sf_vec_size != 0).any().item()):
        raise ValueError(
            f"mega blockscaled {mode} quant requires split_sizes to be multiples "
            f"of sf_vec_size={format.sf_vec_size}"
        )
    _MEGA_QUANT_SPLIT_ALIGNMENT_CHECKED.add(split_sizes, checked_key)


def _validate_mega_padded_scale_capacity(
    split_sizes: torch.Tensor,
    *,
    scale_rows: int,
    block_size_n: int,
) -> None:
    # CUDA routing pads each split to BLOCK_SIZE_N, as required by
    # _validate_swap_ab_row_multiple, so the allocation covers one scale page
    # for every token tile.
    # Keep the value-dependent check for CPU callers without synchronizing routes.
    uses_scale_pages = uses_paged_blockscaled_scale_rows(block_size_n)
    if not uses_scale_pages or split_sizes.is_cuda:
        return
    checked_key = (tuple(split_sizes.shape), scale_rows, block_size_n)
    if (split_sizes, checked_key) in _MEGA_PADDED_SCALE_CAPACITY_CHECKED:
        return
    split_sizes_cpu = split_sizes.detach().to("cpu", non_blocking=False)
    scale_group_rows = block_size_n
    scale_page_rows = _ceil_div(block_size_n, 128) * 128
    padded_scale_rows = int(
        (
            ((split_sizes_cpu + scale_group_rows - 1) // scale_group_rows)
            * scale_page_rows
        )
        .sum()
        .item()
    )
    if padded_scale_rows > scale_rows:
        raise ValueError(
            f"N={block_size_n} MegaMoE scale storage requires at least "
            f"{padded_scale_rows} rows, got scale_rows={scale_rows}"
        )
    _MEGA_PADDED_SCALE_CAPACITY_CHECKED.add(split_sizes, checked_key)


def _mega_quant_split_metadata(
    split_sizes: torch.Tensor,
    *,
    rows: int,
    dim: int,
    format: BlockScaledFormatSpec,
    act_block_size: int,
    mode: str,
    dim_name: str,
    scale_cols_per_tile: int,
    cache: dict[tuple, int],
    shape_cache: dict[tuple, int | None],
    require_full_act_tiles: bool = True,
) -> int:
    _validate_mega_quant_split_alignment(
        split_sizes,
        dim=dim,
        format=format,
        mode=mode,
        dim_name=dim_name,
    )
    if not require_full_act_tiles:
        # Quant producers scan runtime splits; this legacy launch argument is unused.
        return 0
    cache_key = (
        split_sizes.data_ptr(),
        str(split_sizes.device),
        str(split_sizes.dtype),
        tuple(split_sizes.shape),
        rows,
        dim,
        format.sf_vec_size,
        act_block_size,
    )
    shape_key = cache_key[1:]
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    is_capturing = split_sizes.is_cuda and torch.cuda.is_current_stream_capturing()
    if is_capturing:
        shape_cached = shape_cache.get(shape_key)
        if shape_cached == _MEGA_SPLIT_UNSUPPORTED:
            raise _MegaFusedUnsupportedError(
                f"mega fused {mode} bprop does not support the cached split shape"
            )
        if shape_cached is not None:
            cache[cache_key] = shape_cached
            return shape_cached
        raise RuntimeError(
            f"mega {mode.lower()} quant split metadata must be cached before CUDA graph capture"
        )
    split_sizes_cpu = split_sizes.detach().to("cpu", non_blocking=False)
    if bool((split_sizes_cpu % act_block_size != 0).any().item()):
        _cache_shape_metadata(shape_cache, shape_key, _MEGA_SPLIT_UNSUPPORTED)
        raise _MegaFusedUnsupportedError(
            f"mega fused {mode} bprop requires split_sizes to be multiples of "
            f"{mode.lower()} act tile size={act_block_size}"
        )
    col_work_tiles = _ceil_div(
        dim // format.sf_vec_size,
        scale_cols_per_tile,
    )
    total_tiles = int(
        ((split_sizes_cpu // format.sf_vec_size) * col_work_tiles).sum().item()
    )
    cache[cache_key] = total_tiles
    _cache_shape_metadata(shape_cache, shape_key, total_tiles)
    return total_tiles


def _mega_dispatch_quant_split_metadata(
    split_sizes: torch.Tensor,
    *,
    rows: int,
    dim: int,
    format: BlockScaledFormatSpec,
    act_block_size: int,
    require_full_act_tiles: bool = True,
) -> int:
    return _mega_quant_split_metadata(
        split_sizes,
        rows=rows,
        dim=dim,
        format=format,
        act_block_size=act_block_size,
        mode="DISPATCH",
        dim_name="dim",
        scale_cols_per_tile=(
            MegaBlockScaledGroupedGemmKernel.DISPATCH_QUANT_SCALE_COLS_PER_TILE
        ),
        cache=_MEGA_DISPATCH_QUANT_SPLIT_METADATA_CACHE,
        shape_cache=_MEGA_DISPATCH_QUANT_SPLIT_METADATA_SHAPE_CACHE,
        require_full_act_tiles=require_full_act_tiles,
    )


def _mega_dispatch_done_counter_sizes(
    *,
    rows: int,
    dim: int,
    num_groups: int,
    format: BlockScaledFormatSpec,
) -> tuple[int, int]:
    col_blocks = dim // format.sf_vec_size
    col_work_tiles = _ceil_div(
        col_blocks,
        MegaBlockScaledGroupedGemmKernel.DISPATCH_QUANT_SCALE_COLS_PER_TILE,
    )
    return (
        max(1, rows // format.sf_vec_size),
        max(1, num_groups * col_work_tiles),
    )


def _make_mega_dispatch_plan(
    split_sizes: torch.Tensor,
    *,
    rows: int,
    dim: int,
    format: BlockScaledFormatSpec,
    config: dict | None = None,
    require_full_act_tiles: bool = True,
) -> _MegaDispatchPlan:
    if rows % format.sf_vec_size != 0:
        raise ValueError(
            f"mega dispatch rows must be divisible by {format.sf_vec_size}, got {rows}"
        )
    act_block_size = (
        int(config["BLOCK_SIZE_N"])
        if config is not None and bool(config.get("SWAP_AB", False))
        else int(config["BLOCK_SIZE_M"])
        if config is not None
        else format.sf_vec_size
    )
    total_tiles = _mega_dispatch_quant_split_metadata(
        split_sizes,
        rows=rows,
        dim=dim,
        format=format,
        act_block_size=act_block_size,
        require_full_act_tiles=require_full_act_tiles,
    )
    row_done_counter_size, col_done_counter_size = _mega_dispatch_done_counter_sizes(
        rows=rows,
        dim=dim,
        num_groups=int(split_sizes.numel()),
        format=format,
    )
    done_counter_size = row_done_counter_size + col_done_counter_size
    return _MegaDispatchPlan(
        rows=rows,
        dim=dim,
        dispatch_quant_total_tiles=total_tiles,
        row_done_counter_offset=0,
        row_done_counter_size=row_done_counter_size,
        col_done_counter_offset=row_done_counter_size,
        col_done_counter_size=col_done_counter_size,
        done_counter_size=done_counter_size,
        tensormap_descriptor_count=(
            MegaBlockScaledGroupedGemmKernel.MEGA_TENSORMAP_DESCRIPTOR_COUNT
        ),
    )


def _make_mega_combine_plan(
    split_sizes: torch.Tensor,
    *,
    rows: int,
    source_dim: int,
    format: BlockScaledFormatSpec,
) -> _MegaDispatchPlan:
    if rows % format.sf_vec_size != 0:
        raise ValueError(
            f"mega combine rows must be divisible by {format.sf_vec_size}, got {rows}"
        )
    _validate_mega_quant_split_alignment(
        split_sizes,
        dim=source_dim,
        format=format,
        mode="COMBINE",
        dim_name="source_dim",
    )
    col_work_tiles = _ceil_div(
        source_dim // format.sf_vec_size,
        MegaBlockScaledGroupedGemmKernel.COMBINE_SWIGLU_BWD_SCALE_COLS_PER_TILE,
    )
    row_done_counter_size = max(1, rows // format.sf_vec_size)
    col_done_counter_size = max(1, int(split_sizes.numel()) * col_work_tiles)
    done_counter_size = row_done_counter_size + col_done_counter_size
    dxy_dim = 2 * source_dim
    return _MegaDispatchPlan(
        rows=rows,
        dim=dxy_dim,
        # Combine quant producers derive their schedule from the runtime splits.
        dispatch_quant_total_tiles=0,
        row_done_counter_offset=0,
        row_done_counter_size=row_done_counter_size,
        col_done_counter_offset=row_done_counter_size,
        col_done_counter_size=col_done_counter_size,
        done_counter_size=done_counter_size,
        tensormap_descriptor_count=(
            MegaBlockScaledGroupedGemmKernel.MEGA_TENSORMAP_DESCRIPTOR_COUNT
        ),
    )


def _mega_forward_done_counter_sizes(
    *,
    rows: int,
    format: BlockScaledFormatSpec,
) -> tuple[int, int]:
    counter_size = max(1, rows // format.sf_vec_size)
    return counter_size, counter_size


def _make_mega_forward_h2_plan(
    split_sizes: torch.Tensor,
    *,
    rows: int,
    dim: int,
    format: BlockScaledFormatSpec,
) -> _MegaDispatchPlan:
    _validate_mega_quant_split_alignment(
        split_sizes,
        dim=dim,
        format=format,
        mode="FORWARD",
        dim_name="dim",
    )
    row_done_counter_size, col_done_counter_size = _mega_forward_done_counter_sizes(
        rows=rows, format=format
    )
    return _MegaDispatchPlan(
        rows=rows,
        dim=dim,
        # Forward h2 quant producers derive their schedule from the runtime splits.
        dispatch_quant_total_tiles=0,
        row_done_counter_offset=0,
        row_done_counter_size=row_done_counter_size,
        col_done_counter_offset=row_done_counter_size,
        col_done_counter_size=col_done_counter_size,
        done_counter_size=row_done_counter_size + col_done_counter_size,
        tensormap_descriptor_count=(
            MegaBlockScaledGroupedGemmKernel.MEGA_TENSORMAP_DESCRIPTOR_COUNT
        ),
    )


def _allocate_mega_dispatch_counter_workspace(
    *,
    num_clusters: int,
    num_ctas: int,
    device: torch.device,
    plan: _MegaDispatchPlan,
    activation_plan: _MegaDispatchPlan | None = None,
    min_tensormap_rows: int = 0,
    zero_in_prepare: bool = False,
):
    grid_size = max(num_clusters * num_ctas, min_tensormap_rows)
    counter_storage_fn = torch.empty if zero_in_prepare else torch.zeros
    activation_done_counter_size = (
        activation_plan.done_counter_size if activation_plan is not None else 1
    )
    counter_storage = counter_storage_fn(
        3 + plan.done_counter_size + activation_done_counter_size,
        dtype=torch.int32,
        device=device,
    )
    (
        counter,
        dispatch_quant_work_counter,
        dispatch_quant_done_counter,
        activation_quant_work_counter,
        activation_quant_done_counter,
    ) = torch.split(
        counter_storage,
        (1, 1, plan.done_counter_size, 1, activation_done_counter_size),
    )
    tensormaps = torch.empty(
        (grid_size, plan.tensormap_descriptor_count, 16),
        dtype=torch.int64,
        device=device,
    )
    extra_counter_zero_count = (
        2 + plan.done_counter_size + activation_done_counter_size
        if zero_in_prepare
        else 0
    )
    return (
        counter,
        tensormaps,
        dispatch_quant_work_counter,
        dispatch_quant_done_counter,
        activation_quant_work_counter,
        activation_quant_done_counter,
        extra_counter_zero_count,
    )


def _validate_wgrad_quant_pair(
    *,
    name: str,
    quant: tuple[torch.Tensor, torch.Tensor],
    dim: int,
    rows: int,
    format: BlockScaledFormatSpec,
    device: torch.device,
) -> None:
    qdata, scale = quant
    expected_dtype = _torch_dtype(format.a_dtype)
    expected_scale_dtype = _torch_dtype(format.sf_dtype)
    if qdata.device != device or scale.device != device:
        raise ValueError(f"{name} tensors must be on {device}")
    if qdata.dtype != expected_dtype:
        raise TypeError(f"{name}.qdata must have dtype {expected_dtype}")
    if scale.dtype != expected_scale_dtype:
        raise TypeError(f"{name}.scale must have dtype {expected_scale_dtype}")
    expected_qdata_shape = _column_quantized_output_shape(rows, dim, format)
    if tuple(qdata.shape) != expected_qdata_shape:
        raise ValueError(
            f"{name}.qdata must have shape {expected_qdata_shape}, "
            f"got {tuple(qdata.shape)}"
        )
    expected_scale_shape = _column_quantized_scale_shape(rows, dim, format)
    expected_scale_numel = expected_scale_shape[0] * expected_scale_shape[1]
    if scale.numel() != expected_scale_numel:
        raise ValueError(
            f"{name}.scale has {scale.numel()} elements, expected "
            f"{expected_scale_numel}"
        )


def _resolve_wgrad_out_dtype(
    wgrad_out_dtype: torch.dtype | None,
    dw: torch.Tensor | None,
) -> torch.dtype:
    return wgrad_out_dtype or (torch.float32 if dw is not None else torch.bfloat16)


def _empty_mega_bprop_quantized_operands(
    *,
    rows: int,
    dim: int,
    format: BlockScaledFormatSpec,
    device: torch.device,
    row_scale_rows: int | None = None,
    col_rows: int | None = None,
):
    row_scale_rows = rows if row_scale_rows is None else row_scale_rows
    col_rows = rows if col_rows is None else col_rows
    row_q, row_scale = _empty_dispatch_quantized_operand(
        rows=rows,
        scale_rows=row_scale_rows,
        dim=dim,
        format=format,
        device=device,
    )
    col_q_storage = _empty_qdata(
        _column_quantized_storage_shape(col_rows, dim, format),
        format,
        device,
    )
    col_scale_storage = torch.empty(
        _column_quantized_scale_shape(col_rows, dim, format),
        dtype=_torch_dtype(format.sf_dtype),
        device=device,
    )
    return (
        row_q,
        row_scale,
        (_column_quantized_output(col_q_storage, format), col_scale_storage.view(-1)),
        col_q_storage,
        col_scale_storage,
    )


def _mega_bprop_quantized_operands(
    *,
    rows: int,
    dim: int,
    format: BlockScaledFormatSpec,
    device: torch.device,
    activation_buffer: torch.Tensor | None,
    row_scale_rows: int | None = None,
    col_rows: int | None = None,
):
    # `col_rows` shrinks the column-quantized (wgrad) storages when the
    # caller guarantees they are never written (forward-only interleaved
    # inference gates the column stores off in the dispatch producer).
    row_scale_rows = rows if row_scale_rows is None else row_scale_rows
    if activation_buffer is None:
        return _empty_mega_bprop_quantized_operands(
            rows=rows,
            dim=dim,
            format=format,
            device=device,
            row_scale_rows=row_scale_rows,
            col_rows=col_rows,
        )
    if col_rows is not None and col_rows != rows:
        raise ValueError(
            "col_rows shrinking is only supported without an activation "
            "buffer: buffer-backed column storages are zero-cost descriptor "
            "placeholders over planner-carved offsets and keep the full "
            "row extent"
        )
    row_q, row_scale = _dispatch_quantized_operand_placeholder(
        rows=rows,
        scale_rows=row_scale_rows,
        dim=dim,
        format=format,
        device=device,
        activation_buffer=activation_buffer,
    )
    col_q_storage = _activation_buffer_placeholder(
        activation_buffer,
        _column_quantized_storage_shape(rows, dim, format),
        _torch_dtype(format.a_dtype),
    )
    col_scale_storage = _activation_buffer_placeholder(
        activation_buffer,
        _column_quantized_scale_shape(rows, dim, format),
        _torch_dtype(format.sf_dtype),
    )
    return (
        row_q,
        row_scale,
        (_column_quantized_output(col_q_storage, format), col_scale_storage.view(-1)),
        col_q_storage,
        col_scale_storage,
    )


def _to_cute_tensor(
    tensor: torch.Tensor,
    *,
    assumed_align: int,
    format: BlockScaledFormatSpec,
):
    return from_dlpack(
        tensor.detach(),
        assumed_align=assumed_align,
        enable_tvm_ffi=_format_uses_fp4(format),
    )


def _make_mega_two_gemm_kernel_args(  # noqa: C901
    *,
    dgrad_tensors: tuple[torch.Tensor, ...],
    wgrad_tensors: tuple[torch.Tensor, ...],
    split_sizes: torch.Tensor,
    workspace: tuple[torch.Tensor, ...],
    route_ptrs: torch.Tensor,
    combine_route_ptrs: torch.Tensor | None,
    combine_sources: tuple[torch.Tensor, torch.Tensor] | None,
    col_q_storage: torch.Tensor,
    col_scale_storage: torch.Tensor,
    activation_quant: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        _MegaDispatchPlan,
    ]
    | None,
    dgrad_problem,
    wgrad_problem,
    dgrad_out_dtype: torch.dtype,
    wgrad_out_dtype: torch.dtype,
    format: BlockScaledFormatSpec,
    output_accum: bool,
    symm_mem_buffer,
    num_clusters: int,
    plan: _MegaDispatchPlan,
    dgrad_weight_format: BlockScaledFormatSpec | None = None,
    wgrad_weight_format: BlockScaledFormatSpec | None = None,
    kernel_mnk: tuple[int, int, int] | None = None,
    use_device_tensormaps: bool = True,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    dispatch_row_global_scale_inv: torch.Tensor | None = None,
    activation_row_global_scale_inv: torch.Tensor | None = None,
    dgrad_b_global_scale_inv: torch.Tensor | None = None,
    wgrad_b_global_scale_inv: torch.Tensor | None = None,
    nvfp4_recip_lut: torch.Tensor | None = None,
):
    dgrad_weight_format = format if dgrad_weight_format is None else dgrad_weight_format
    wgrad_weight_format = format if wgrad_weight_format is None else wgrad_weight_format
    dgrad_kernel_format = make_mixed_blockscaled_format(format, dgrad_weight_format)
    wgrad_kernel_format = make_mixed_blockscaled_format(format, wgrad_weight_format)
    if dgrad_kernel_format.name != wgrad_kernel_format.name:
        raise ValueError(
            "Mega two-GEMM launches require the same mixed activation/weight "
            "format for both contractions"
        )
    kernel_format = dgrad_kernel_format
    dgrad_a, dgrad_b, dgrad_c, dgrad_sfa, dgrad_sfb = dgrad_tensors
    wgrad_a, wgrad_b, wgrad_c, wgrad_sfa, wgrad_sfb = wgrad_tensors
    (
        counter,
        tensormaps,
        quant_work_counter,
        quant_done_counter,
        activation_work_counter,
        activation_done_counter,
        zero_count,
    ) = workspace
    launch_tensors = _make_launch_tensor_bundle(
        a=dgrad_a,
        b=dgrad_b,
        c=dgrad_c,
        sfa=dgrad_sfa,
        sfb=dgrad_sfb,
        split_sizes=split_sizes,
        counter=counter,
        tensormaps=tensormaps,
        format=format,
        weight_format=dgrad_weight_format,
        out_dtype=dgrad_out_dtype,
    )
    wgrad_launch_tensors = _make_launch_tensor_bundle(
        a=wgrad_a,
        b=wgrad_b,
        c=wgrad_c,
        sfa=wgrad_sfa,
        sfb=wgrad_sfb,
        split_sizes=split_sizes,
        counter=counter,
        tensormaps=tensormaps,
        format=format,
        weight_format=wgrad_weight_format,
        out_dtype=wgrad_out_dtype,
    )
    route_cute = _to_cute_tensor(route_ptrs, assumed_align=8, format=kernel_format)
    combine_route_cute = (
        route_cute
        if combine_route_ptrs is None
        else _to_cute_tensor(combine_route_ptrs, assumed_align=8, format=kernel_format)
    )
    if combine_sources is None:
        combine_source_cute = (route_cute, route_cute)
    else:
        combine_source_cute = tuple(
            _to_cute_tensor(source, assumed_align=16, format=kernel_format)
            for source in combine_sources
        )
    if activation_quant is None:
        activation_route = route_ptrs
        activation_row_q = dgrad_a
        activation_row_scale = dgrad_sfa
        activation_col_q_storage = col_q_storage
        activation_col_scale_storage = col_scale_storage
        activation_plan = plan
    else:
        (
            activation_route,
            activation_row_q,
            activation_row_scale,
            activation_col_q_storage,
            activation_col_scale_storage,
            activation_plan,
        ) = activation_quant
    global_scale_dummy = counter.view(torch.float32)
    dispatch_row_global_scale_inv = (
        global_scale_dummy
        if dispatch_row_global_scale_inv is None
        else dispatch_row_global_scale_inv
    )
    activation_row_global_scale_inv = (
        global_scale_dummy
        if activation_row_global_scale_inv is None
        else activation_row_global_scale_inv
    )
    has_global_scale_inv = (
        dgrad_b_global_scale_inv is not None
        and wgrad_b_global_scale_inv is not None
        and nvfp4_recip_lut is not None
    )
    use_global_scale_inv = format is NVFP4
    if has_global_scale_inv != use_global_scale_inv:
        raise ValueError(
            "NVFP4 mega bprop requires both weight global scales and the "
            "NVFP4 reciprocal LUT; non-NVFP4 formats must not provide them"
        )
    dgrad_b_global_scale_inv_ptr = (
        0 if dgrad_b_global_scale_inv is None else dgrad_b_global_scale_inv.data_ptr()
    )
    wgrad_b_global_scale_inv_ptr = (
        0 if wgrad_b_global_scale_inv is None else wgrad_b_global_scale_inv.data_ptr()
    )
    nvfp4_recip_lut_ptr = 0 if nvfp4_recip_lut is None else nvfp4_recip_lut.data_ptr()
    quant_kernel_args = (
        route_cute,
        _to_cute_tensor(quant_work_counter, assumed_align=4, format=kernel_format),
        _to_cute_tensor(quant_done_counter, assumed_align=4, format=kernel_format),
        _to_cute_tensor(
            dgrad_a.view(torch.uint32), assumed_align=16, format=kernel_format
        ),
        _to_cute_tensor(
            dgrad_sfa.view(torch.uint8), assumed_align=16, format=kernel_format
        ),
        _to_cute_tensor(
            dispatch_row_global_scale_inv,
            assumed_align=4,
            format=kernel_format,
        ),
        _to_cute_tensor(
            _qdata_words(col_q_storage), assumed_align=16, format=kernel_format
        ),
        _to_cute_tensor(
            col_scale_storage.view(-1).view(torch.uint8),
            assumed_align=16,
            format=kernel_format,
        ),
        combine_route_cute,
        *combine_source_cute,
        _to_cute_tensor(activation_route, assumed_align=8, format=kernel_format),
        _to_cute_tensor(activation_work_counter, assumed_align=4, format=kernel_format),
        _to_cute_tensor(activation_done_counter, assumed_align=4, format=kernel_format),
        _to_cute_tensor(
            activation_row_q.view(torch.uint32),
            assumed_align=16,
            format=kernel_format,
        ),
        _to_cute_tensor(
            activation_row_scale.view(torch.uint8),
            assumed_align=16,
            format=kernel_format,
        ),
        _to_cute_tensor(
            activation_row_global_scale_inv,
            assumed_align=4,
            format=kernel_format,
        ),
        _to_cute_tensor(
            _qdata_words(activation_col_q_storage),
            assumed_align=16,
            format=kernel_format,
        ),
        _to_cute_tensor(
            activation_col_scale_storage.view(-1).view(torch.uint8),
            assumed_align=16,
            format=kernel_format,
        ),
    )
    dgrad_pointer_stride_args, dgrad_elem_sizes = _make_pointer_stride_args(
        a=dgrad_a,
        b=dgrad_b,
        c=dgrad_c,
        sfa=dgrad_sfa,
        sfb=dgrad_sfb,
        format=format,
        weight_format=dgrad_weight_format,
        N=dgrad_problem.N,
        K=dgrad_problem.K,
    )
    wgrad_pointer_stride_args, wgrad_elem_sizes = _make_pointer_stride_args(
        a=wgrad_a,
        b=wgrad_b,
        c=wgrad_c,
        sfa=wgrad_sfa,
        sfb=wgrad_sfb,
        format=format,
        weight_format=wgrad_weight_format,
        N=wgrad_problem.N,
        K=wgrad_problem.K,
    )
    (
        activation_buffer_base_ptr,
        activation_offsets_cute,
        use_activation_buffer,
        condition_cute,
        use_conditional_execution,
        condition_owner,
        activation_buffer_size_bytes,
    ) = _prepare_activation_buffer_launch_args(
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        fallback_offsets=tensormaps,
        expected_offset_count=MEGA_ACTIVATION_OFFSET_COUNT,
        enable_tvm_ffi=_format_uses_fp4(kernel_format),
    )
    stream = cutlass_torch.current_stream()
    compile_tensor_args = (
        launch_tensors.compile_tensors[:5]
        + wgrad_launch_tensors.compile_tensors[:5]
        + launch_tensors.compile_tensors[5:]
        + quant_kernel_args
    )
    runtime_tensor_args = (
        launch_tensors.runtime_tensors[:5]
        + wgrad_launch_tensors.runtime_tensors[:5]
        + launch_tensors.runtime_tensors[5:]
        + quant_kernel_args
    )
    kernel_m, kernel_n, kernel_k = kernel_mnk or (
        dgrad_problem.GM,
        dgrad_problem.N,
        dgrad_problem.K,
    )
    common_args = (
        dgrad_pointer_stride_args,
        wgrad_pointer_stride_args,
        kernel_m,
        kernel_n,
        kernel_k,
        symm_mem_buffer.hdl.rank,
        num_clusters,
        plan.dispatch_quant_total_tiles,
        zero_count,
        plan.row_done_counter_offset,
        plan.col_done_counter_offset,
        activation_plan.dispatch_quant_total_tiles,
        activation_plan.row_done_counter_offset,
        activation_plan.col_done_counter_offset,
    )
    compile_args = compile_tensor_args + (
        dgrad_pointer_stride_args,
        wgrad_pointer_stride_args,
        dgrad_elem_sizes,
        wgrad_elem_sizes,
        _cutlass_dtype_from_torch(wgrad_out_dtype),
        output_accum,
        dgrad_problem.G,
        *common_args[2:],
        combine_sources is not None,
        activation_quant is not None,
        use_device_tensormaps,
        activation_buffer_base_ptr,
        cutlass.Int64(activation_buffer_size_bytes),
        activation_offsets_cute,
        use_activation_buffer,
        condition_cute,
        use_conditional_execution,
        cutlass.Int64(dgrad_b_global_scale_inv_ptr),
        cutlass.Int64(wgrad_b_global_scale_inv_ptr),
        use_global_scale_inv,
        cutlass.Int64(nvfp4_recip_lut_ptr),
        stream,
    )
    runtime_args = (
        runtime_tensor_args
        + common_args
        + (
            activation_buffer_base_ptr,
            cutlass.Int64(activation_buffer_size_bytes),
            activation_offsets_cute,
            condition_cute,
            cutlass.Int64(dgrad_b_global_scale_inv_ptr),
            cutlass.Int64(wgrad_b_global_scale_inv_ptr),
            cutlass.Int64(nvfp4_recip_lut_ptr),
            stream,
        )
    )
    return (
        compile_args,
        runtime_args,
        dgrad_elem_sizes,
        wgrad_elem_sizes,
        launch_tensors.placeholders,
        use_activation_buffer,
        use_conditional_execution,
        condition_owner,
    )


def _get_mega_blockscaled_kernel(
    *,
    config: dict,
    format: BlockScaledFormatSpec,
    force_n_major: bool,
    num_n_clusters: int,
    world_size: int,
    dispatch_source_dtype: type[cutlass.Numeric],
    combine_swiglu_fast_math: bool = False,
    combine_swiglu_clamped: bool = False,
    combine_swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    combine_swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    mode: int = MegaBlockScaledGroupedGemmKernel.MEGA_BACKWARD_DISPATCH_MODE,
) -> MegaBlockScaledGroupedGemmKernel:
    swiglu_clamp = canonical_swiglu_clamp(
        combine_swiglu_clamped, combine_swiglu_alpha, combine_swiglu_limit
    )
    key = (
        _config_cache_key(config),
        format.name,
        force_n_major,
        num_n_clusters,
        world_size,
        dispatch_source_dtype,
        combine_swiglu_fast_math,
        swiglu_clamp,
        mode,
    )
    inst = _MEGA_BLOCKSCALED_KERNEL_CACHE.get(key)
    if inst is None:
        kernel_config = dict(config)
        if combine_swiglu_fast_math:
            kernel_config["COMBINE_SWIGLU_FAST_MATH"] = True
        _set_swiglu_clamp_config(kernel_config, *swiglu_clamp)
        inst = MegaBlockScaledGroupedGemmKernel.from_config(
            kernel_config,
            mode=mode,
            format=format,
            force_n_major=force_n_major,
            num_n_clusters=num_n_clusters,
            world_size=world_size,
            dispatch_source_dtype=dispatch_source_dtype,
        )
        _MEGA_BLOCKSCALED_KERNEL_CACHE[key] = inst
    return inst


def _cute_mega_blockscaled_grouped_gemm_dispatch_bprop(  # noqa: C901
    dy: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    x_wgrad_quant: tuple[torch.Tensor, torch.Tensor],
    dx: torch.Tensor | None = None,
    dw: torch.Tensor | None = None,
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    wgrad_out_dtype: torch.dtype | None = None,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    output_accum: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Run the fused backward dispatch DGRAD+WGRAD mega kernel."""
    if dx is not None and activation_buffer is not None:
        raise ValueError("dx must be None when activation_buffer is provided")
    if w.dim() != 3:
        raise ValueError(f"w must be 3D, got shape={tuple(w.shape)}")
    _, n_storage, k = w.shape
    n = n_storage * 2 if format in (NVFP4, MXFP4) else n_storage
    rows = int(gather_ptrs.numel()) if num_out_tokens is None else int(num_out_tokens)
    if rows % format.sf_vec_size != 0:
        raise ValueError(
            f"num_out_tokens/gather rows must be divisible by {format.sf_vec_size}, "
            f"got {rows}"
        )
    if num_sms is None:
        num_sms = num_sms_per_device()
    if config is None:
        config = dict(
            auto_blockscaled_config(
                GM=rows,
                G=int(split_sizes.numel()),
                N=k,
                K=n,
                num_sms=num_sms,
                format=format,
                m_multiple_of=m_multiple_of,
                problem_type=_DGRAD,
            )
        )
        config["STATIC_SCHEDULER"] = False
        if config.get("EPILOGUE_SUBTILE") == 2:
            # Mega bprop needs subtile 4 to avoid epilogue register spills.
            config["EPILOGUE_SUBTILE"] = 4
    else:
        config = dict(config)
        config.setdefault("STATIC_SCHEDULER", False)
    plan = _make_mega_dispatch_plan(
        split_sizes,
        rows=rows,
        dim=n,
        format=format,
        config=config,
        require_full_act_tiles=False,
    )
    _validate_wgrad_quant_pair(
        name="x_wgrad_quant",
        quant=x_wgrad_quant,
        dim=k,
        rows=rows,
        format=format,
        device=w.device,
    )

    wgrad_out_dtype = _resolve_wgrad_out_dtype(wgrad_out_dtype, dw)
    can_use_fused_dispatch_bprop = (
        out_dtype in (torch.bfloat16, torch.float16)
        and wgrad_out_dtype in (torch.bfloat16, torch.float32)
        and _can_use_fused_dispatch_quant(
            rows=rows,
            dim=n,
            dtype=dy.dtype,
            format=format,
        )
    )
    if not can_use_fused_dispatch_bprop:
        raise _MegaFusedUnsupportedError(
            "mega fused DISPATCH bprop requires bf16/fp16 DGRAD output, "
            "bf16/fp32 WGRAD output, and a fused DISPATCH quantizable dy input"
        )

    dy_2d = _flatten_last_dim(dy, n, "dy")
    if split_sizes.dtype != torch.int32:
        split_sizes = split_sizes.to(torch.int32)

    (
        dy_row_q,
        dy_row_scale,
        dy_wgrad_quant,
        dy_col_q_storage,
        dy_col_scale_storage,
    ) = _mega_bprop_quantized_operands(
        rows=rows,
        dim=n,
        format=format,
        device=w.device,
        activation_buffer=activation_buffer,
    )
    _validate_2d_operand_layout(
        "mega DISPATCH DGRAD dy",
        dy_row_q,
        format,
        contraction_axis=1,
    )
    w_dgrad = w.transpose(1, 2)
    _validate_3d_operand_layout(
        "mega DISPATCH DGRAD w",
        w_dgrad,
        format,
        contraction_axis=2,
    )
    _validate_blockscaled_launch_inputs(
        a=dy_row_q,
        b=w_dgrad,
        sfa=dy_row_scale,
        sfb=sfb,
        split_sizes=split_sizes,
        format=format,
        split_size_multiple_of=m_multiple_of,
    )
    dgrad_problem = _parse_blockscaled_problem(
        a=dy_row_q,
        b=w_dgrad,
        format=format,
        split_sizes=split_sizes,
        contraction_axes=(1, 2),
        problem_type=_DGRAD,
    )
    dx_placeholder = (
        None
        if activation_buffer is None or dx is not None
        else _activation_buffer_placeholder(
            activation_buffer,
            dgrad_problem.c_shape,
            out_dtype,
        )
    )
    dx, out_dtype = _resolve_output_tensor(
        c=dx if dx is not None else dx_placeholder,
        c_shape=dgrad_problem.c_shape,
        out_dtype=out_dtype,
        device=w.device,
        problem_type=_DGRAD,
        output_accum=False,
    )

    _validate_wgrad_quant_pair(
        name="dy_wgrad_quant",
        quant=dy_wgrad_quant,
        dim=n,
        rows=rows,
        format=format,
        device=w.device,
    )
    x_wgrad_q, x_wgrad_scale = x_wgrad_quant
    _validate_2d_operand_layout(
        "mega WGRAD dy.T",
        dy_wgrad_quant[0],
        format,
        contraction_axis=1,
    )
    _validate_2d_operand_layout(
        "mega WGRAD x.T",
        x_wgrad_q,
        format,
        contraction_axis=1,
    )
    _validate_blockscaled_launch_inputs(
        a=dy_wgrad_quant[0],
        b=x_wgrad_q,
        sfa=dy_wgrad_quant[1],
        sfb=x_wgrad_scale,
        split_sizes=split_sizes,
        format=format,
        split_size_multiple_of=m_multiple_of,
    )
    wgrad_problem = _parse_blockscaled_problem(
        a=dy_wgrad_quant[0],
        b=x_wgrad_q,
        format=format,
        split_sizes=split_sizes,
        contraction_axes=(1, 1),
        problem_type=_WGRAD,
    )
    dw, wgrad_out_dtype = _resolve_output_tensor(
        c=dw,
        c_shape=wgrad_problem.c_shape,
        out_dtype=wgrad_out_dtype,
        device=w.device,
        problem_type=_WGRAD,
        output_accum=output_accum,
    )

    force_n_major = True
    num_n_clusters = 1
    num_clusters = max(1, num_sms // config["NUM_CTAS"])
    world_size = symm_mem_buffer.hdl.world_size
    dispatch_source_dtype = _dispatch_quant_source_dtype_from_torch(dy_2d.dtype)
    kernel = _get_mega_blockscaled_kernel(
        config=config,
        format=format,
        force_n_major=force_n_major,
        num_n_clusters=num_n_clusters,
        world_size=world_size,
        dispatch_source_dtype=dispatch_source_dtype,
    )
    workspace = _allocate_mega_dispatch_counter_workspace(
        num_clusters=num_clusters,
        num_ctas=config["NUM_CTAS"],
        device=w.device,
        plan=plan,
        min_tensormap_rows=dgrad_problem.G,
        zero_in_prepare=True,
    )
    (
        compile_args,
        runtime_args,
        dgrad_elem_sizes,
        wgrad_elem_sizes,
        (a_placeholder, b_placeholder, c_placeholder),
        use_activation_buffer,
        use_conditional_execution,
        condition_owner,
    ) = _make_mega_two_gemm_kernel_args(
        dgrad_tensors=(dy_row_q, w_dgrad, dx, dy_row_scale, sfb),
        wgrad_tensors=(
            dy_wgrad_quant[0],
            x_wgrad_q,
            dw,
            dy_wgrad_quant[1],
            x_wgrad_scale,
        ),
        split_sizes=split_sizes,
        workspace=workspace,
        route_ptrs=gather_ptrs[:rows],
        combine_route_ptrs=None,
        combine_sources=None,
        col_q_storage=dy_col_q_storage,
        col_scale_storage=dy_col_scale_storage,
        activation_quant=None,
        dgrad_problem=dgrad_problem,
        wgrad_problem=wgrad_problem,
        dgrad_out_dtype=out_dtype,
        wgrad_out_dtype=wgrad_out_dtype,
        format=format,
        output_accum=output_accum,
        symm_mem_buffer=symm_mem_buffer,
        num_clusters=num_clusters,
        plan=plan,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
    )
    dgrad_elem_a, dgrad_elem_b, dgrad_elem_c = dgrad_elem_sizes
    wgrad_elem_a, wgrad_elem_b, wgrad_elem_c = wgrad_elem_sizes
    cache_key = (
        _config_cache_key(config),
        format.name,
        force_n_major,
        num_n_clusters,
        world_size,
        _torch_layout_signature(a_placeholder),
        _torch_layout_signature(b_placeholder),
        _torch_layout_signature(c_placeholder),
        _torch_layout_signature(dy_wgrad_quant[0]),
        _torch_layout_signature(x_wgrad_q),
        _torch_layout_signature(dw),
        dgrad_problem.G,
        dgrad_problem.N,
        dgrad_problem.K,
        wgrad_problem.N,
        wgrad_problem.K,
        dgrad_elem_a,
        dgrad_elem_b,
        dgrad_elem_c,
        wgrad_elem_a,
        wgrad_elem_b,
        wgrad_elem_c,
        dy_2d.dtype,
        output_accum,
        use_activation_buffer,
        use_conditional_execution,
        (
            "mega_dispatch_bprop",
            MegaBlockScaledGroupedGemmKernel.DISPATCH_QUANT_GROUPS,
            MegaBlockScaledGroupedGemmKernel.DISPATCH_QUANT_WARPS_PER_GROUP,
            MegaBlockScaledGroupedGemmKernel.DISPATCH_QUANT_MICROTILES_PER_WARP,
        ),
    )
    name_prefix = "_cute_mega_blockscaled_grouped_gemm_dispatch_bprop"
    compiled = _compile_or_get(
        cache_key,
        kernel,
        *compile_args,
        name_prefix=name_prefix,
    )
    compiled(*runtime_args)
    if condition_owner is not None:
        condition_owner.record_stream(torch.cuda.current_stream(condition_owner.device))
    return dx, dw, dy_wgrad_quant


def mega_blockscaled_grouped_gemm_dgrad_wgrad_dispatch(
    dy: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    x_wgrad_quant: tuple[torch.Tensor, torch.Tensor],
    dx: torch.Tensor | None = None,
    dw: torch.Tensor | None = None,
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    wgrad_out_dtype: torch.dtype | None = None,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    output_accum: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Run fused backward dispatch bprop.

    With `activation_buffer`, activation-backed return tensors are shape-only
    aliases at buffer offset zero; consume results through `activation_offsets`.
    """
    try:
        return _cute_mega_blockscaled_grouped_gemm_dispatch_bprop(
            dy=dy,
            w=w,
            sfb=sfb,
            split_sizes=split_sizes,
            gather_ptrs=gather_ptrs,
            num_out_tokens=num_out_tokens,
            symm_mem_buffer=symm_mem_buffer,
            x_wgrad_quant=x_wgrad_quant,
            dx=dx,
            dw=dw,
            format=format,
            out_dtype=out_dtype,
            wgrad_out_dtype=wgrad_out_dtype,
            num_sms=num_sms,
            config=config,
            m_multiple_of=m_multiple_of,
            output_accum=output_accum,
            activation_buffer=activation_buffer,
            activation_offsets=activation_offsets,
            conditional_execution=conditional_execution,
        )
    except _MegaFusedUnsupportedError as exc:
        raise NotImplementedError(str(exc)) from exc


def _cute_mega_blockscaled_grouped_gemm_combine_bprop(  # noqa: C901
    grad_h2: torch.Tensor,
    h1: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    x_wgrad_quant: tuple[torch.Tensor, torch.Tensor],
    dx: torch.Tensor | None = None,
    dw: torch.Tensor | None = None,
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    wgrad_out_dtype: torch.dtype | None = None,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    num_output_tokens: int | None = None,
    output_accum: bool = False,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if w.dim() != 3:
        raise ValueError(f"w must be 3D, got shape={tuple(w.shape)}")
    source_dim = int(grad_h2.shape[-1])
    dxy_dim = 2 * source_dim
    grad_h2_2d = _flatten_last_dim(grad_h2, source_dim, "grad_h2")
    h1_2d = _flatten_last_dim(h1, dxy_dim, "h1")
    if grad_h2_2d.shape[0] != h1_2d.shape[0]:
        raise ValueError(
            "grad_h2 and h1 must flatten to the same row count; "
            f"got {grad_h2_2d.shape[0]} and {h1_2d.shape[0]}"
        )
    if grad_h2_2d.device != w.device or h1_2d.device != w.device:
        raise ValueError("grad_h2, h1, and w must be on the same device")
    rows = int(grad_h2_2d.shape[0])
    _, w_dxy_dim_storage, hidden_dim = w.shape
    w_dxy_dim = _logical_dim_size(w_dxy_dim_storage, format)
    if w_dxy_dim != dxy_dim:
        raise ValueError(
            f"w.shape[1] must equal 2 * grad_h2.shape[-1]={dxy_dim}, got {w_dxy_dim}"
        )
    if scatter_ptrs.numel() < rows:
        raise ValueError(
            f"scatter_ptrs has {scatter_ptrs.numel()} entries, expected at least {rows}"
        )
    if num_sms is None:
        num_sms = num_sms_per_device()
    if config is None:
        config = dict(
            auto_blockscaled_config(
                GM=rows,
                G=int(split_sizes.numel()),
                N=hidden_dim,
                K=dxy_dim,
                num_sms=num_sms,
                format=format,
                m_multiple_of=m_multiple_of,
                problem_type=_DGRAD,
            )
        )
        config["STATIC_SCHEDULER"] = False
        config["NUM_SMEM_BUFFERS"] = 6
        config["NUM_C_STAGES"] = 1
        config["EPILOGUE_SUBTILE"] = 4
    else:
        config = dict(config)
        config.setdefault("STATIC_SCHEDULER", False)

    can_use_fused = (
        out_dtype in (torch.bfloat16, torch.float16)
        and (wgrad_out_dtype in (None, torch.bfloat16, torch.float32))
        and _can_use_fused_combine_swiglu_quant(
            rows=rows,
            dim=source_dim,
            dtype=grad_h2_2d.dtype,
            format=format,
        )
    )
    if not can_use_fused:
        raise _MegaFusedUnsupportedError(
            "mega fused COMBINE bprop requires bf16/fp16 DGRAD output, "
            "bf16/fp32 WGRAD output, and fused COMBINE SwiGLU-bwd quantization"
        )

    if split_sizes.dtype != torch.int32:
        split_sizes = split_sizes.to(torch.int32)
    force_n_major = False
    num_n_clusters = 2
    plan = _make_mega_combine_plan(
        split_sizes,
        rows=rows,
        source_dim=source_dim,
        format=format,
    )
    _validate_wgrad_quant_pair(
        name="x_wgrad_quant",
        quant=x_wgrad_quant,
        dim=hidden_dim,
        rows=rows,
        format=format,
        device=w.device,
    )
    wgrad_out_dtype = _resolve_wgrad_out_dtype(wgrad_out_dtype, dw)

    (
        dxy_row_q,
        dxy_row_scale,
        dxy_wgrad_quant,
        dxy_col_q_storage,
        dxy_col_scale_storage,
    ) = _mega_bprop_quantized_operands(
        rows=rows,
        dim=dxy_dim,
        format=format,
        device=w.device,
        activation_buffer=activation_buffer,
    )
    _validate_2d_operand_layout(
        "mega COMBINE DGRAD dxy",
        dxy_row_q,
        format,
        contraction_axis=1,
    )
    w_dgrad = w.transpose(1, 2)
    _validate_3d_operand_layout(
        "mega COMBINE DGRAD w",
        w_dgrad,
        format,
        contraction_axis=2,
    )
    _validate_blockscaled_launch_inputs(
        a=dxy_row_q,
        b=w_dgrad,
        sfa=dxy_row_scale,
        sfb=sfb,
        split_sizes=split_sizes,
        format=format,
        split_size_multiple_of=m_multiple_of,
    )
    dgrad_problem = _parse_blockscaled_problem(
        a=dxy_row_q,
        b=w_dgrad,
        format=format,
        split_sizes=split_sizes,
        contraction_axes=(1, 2),
        problem_type=_DGRAD,
    )
    if dx is not None:
        if dx.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"dx.dtype={dx.dtype} must be bfloat16 or float16")
        if tuple(dx.shape) != dgrad_problem.c_shape:
            raise ValueError(
                f"dx.shape={tuple(dx.shape)} does not match expected "
                f"{dgrad_problem.c_shape}"
            )
    if out_dtype not in (None, torch.bfloat16, torch.float16):
        raise NotImplementedError(f"out_dtype={out_dtype} not supported.")
    dgrad_c, out_dtype = _resolve_combine_descriptor_tensor(
        c=dx,
        c_shape=dgrad_problem.c_shape,
        out_dtype=out_dtype,
        device=w.device,
        problem_type=_DGRAD,
        config=config,
    )

    x_wgrad_q, x_wgrad_scale = x_wgrad_quant
    _validate_2d_operand_layout(
        "mega COMBINE WGRAD dxy.T",
        dxy_wgrad_quant[0],
        format,
        contraction_axis=1,
    )
    _validate_2d_operand_layout(
        "mega COMBINE WGRAD x.T",
        x_wgrad_q,
        format,
        contraction_axis=1,
    )
    _validate_blockscaled_launch_inputs(
        a=dxy_wgrad_quant[0],
        b=x_wgrad_q,
        sfa=dxy_wgrad_quant[1],
        sfb=x_wgrad_scale,
        split_sizes=split_sizes,
        format=format,
        split_size_multiple_of=m_multiple_of,
    )
    wgrad_problem = _parse_blockscaled_problem(
        a=dxy_wgrad_quant[0],
        b=x_wgrad_q,
        format=format,
        split_sizes=split_sizes,
        contraction_axes=(1, 1),
        problem_type=_WGRAD,
    )
    dw, wgrad_out_dtype = _resolve_output_tensor(
        c=dw,
        c_shape=wgrad_problem.c_shape,
        out_dtype=wgrad_out_dtype,
        device=w.device,
        problem_type=_WGRAD,
        output_accum=output_accum,
    )

    num_clusters = max(1, num_sms // config["NUM_CTAS"])
    world_size = symm_mem_buffer.hdl.world_size
    dispatch_source_dtype = _dispatch_quant_source_dtype_from_torch(grad_h2_2d.dtype)
    kernel = _get_mega_blockscaled_kernel(
        config=config,
        format=format,
        force_n_major=force_n_major,
        num_n_clusters=num_n_clusters,
        world_size=world_size,
        dispatch_source_dtype=dispatch_source_dtype,
        combine_swiglu_fast_math=swiglu_fast_math,
        combine_swiglu_clamped=swiglu_clamped,
        combine_swiglu_alpha=swiglu_alpha,
        combine_swiglu_limit=swiglu_limit,
    )
    workspace = _allocate_mega_dispatch_counter_workspace(
        num_clusters=num_clusters,
        num_ctas=config["NUM_CTAS"],
        device=w.device,
        plan=plan,
        min_tensormap_rows=dgrad_problem.G,
        zero_in_prepare=True,
    )
    (
        compile_args,
        runtime_args,
        dgrad_elem_sizes,
        wgrad_elem_sizes,
        (a_placeholder, b_placeholder, c_placeholder),
        use_activation_buffer,
        use_conditional_execution,
        condition_owner,
    ) = _make_mega_two_gemm_kernel_args(
        dgrad_tensors=(dxy_row_q, w_dgrad, dgrad_c, dxy_row_scale, sfb),
        wgrad_tensors=(
            dxy_wgrad_quant[0],
            x_wgrad_q,
            dw,
            dxy_wgrad_quant[1],
            x_wgrad_scale,
        ),
        split_sizes=split_sizes,
        workspace=workspace,
        route_ptrs=scatter_ptrs[:rows],
        combine_route_ptrs=None,
        combine_sources=(grad_h2_2d, h1_2d),
        col_q_storage=dxy_col_q_storage,
        col_scale_storage=dxy_col_scale_storage,
        activation_quant=None,
        dgrad_problem=dgrad_problem,
        wgrad_problem=wgrad_problem,
        dgrad_out_dtype=out_dtype,
        wgrad_out_dtype=wgrad_out_dtype,
        format=format,
        output_accum=output_accum,
        symm_mem_buffer=symm_mem_buffer,
        num_clusters=num_clusters,
        plan=plan,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
    )
    dgrad_elem_a, dgrad_elem_b, dgrad_elem_c = dgrad_elem_sizes
    wgrad_elem_a, wgrad_elem_b, wgrad_elem_c = wgrad_elem_sizes
    cache_key = (
        _config_cache_key(config),
        format.name,
        force_n_major,
        num_n_clusters,
        world_size,
        _torch_layout_signature(a_placeholder),
        _torch_layout_signature(b_placeholder),
        _torch_layout_signature(c_placeholder),
        _torch_layout_signature(dxy_wgrad_quant[0]),
        _torch_layout_signature(x_wgrad_q),
        _torch_layout_signature(dw),
        _torch_layout_signature(grad_h2_2d),
        _torch_layout_signature(h1_2d),
        dgrad_problem.G,
        dgrad_problem.N,
        dgrad_problem.K,
        wgrad_problem.N,
        wgrad_problem.K,
        dgrad_elem_a,
        dgrad_elem_b,
        dgrad_elem_c,
        wgrad_elem_a,
        wgrad_elem_b,
        wgrad_elem_c,
        grad_h2_2d.dtype,
        output_accum,
        swiglu_fast_math,
        canonical_swiglu_clamp(swiglu_clamped, swiglu_alpha, swiglu_limit),
        use_activation_buffer,
        use_conditional_execution,
        "mega_combine_bprop",
    )
    name_prefix = "_cute_mega_blockscaled_grouped_gemm_combine_bprop"
    compile_options = "--opt-level=1"
    cache_key = cache_key + (compile_options,)
    compiled = _compile_or_get(
        cache_key,
        kernel,
        *compile_args,
        name_prefix=name_prefix,
        options=compile_options,
    )
    compiled(*runtime_args)
    if condition_owner is not None:
        condition_owner.record_stream(torch.cuda.current_stream(condition_owner.device))
    return (
        _maybe_slice_output(symm_mem_buffer.local(), num_output_tokens),
        dw,
        dxy_wgrad_quant,
    )


def mega_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine(
    grad_h2: torch.Tensor,
    h1: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    x_wgrad_quant: tuple[torch.Tensor, torch.Tensor],
    dx: torch.Tensor | None = None,
    dw: torch.Tensor | None = None,
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    wgrad_out_dtype: torch.dtype | None = None,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    num_output_tokens: int | None = None,
    output_accum: bool = False,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Run fused backward combine bprop.

    With `activation_buffer`, activation-backed return tensors are shape-only
    aliases at buffer offset zero; consume results through `activation_offsets`.
    """
    try:
        return _cute_mega_blockscaled_grouped_gemm_combine_bprop(
            grad_h2=grad_h2,
            h1=h1,
            w=w,
            sfb=sfb,
            split_sizes=split_sizes,
            scatter_ptrs=scatter_ptrs,
            symm_mem_buffer=symm_mem_buffer,
            x_wgrad_quant=x_wgrad_quant,
            dx=dx,
            dw=dw,
            format=format,
            out_dtype=out_dtype,
            wgrad_out_dtype=wgrad_out_dtype,
            num_sms=num_sms,
            config=config,
            m_multiple_of=m_multiple_of,
            num_output_tokens=num_output_tokens,
            output_accum=output_accum,
            swiglu_fast_math=swiglu_fast_math,
            swiglu_clamped=swiglu_clamped,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            activation_buffer=activation_buffer,
            activation_offsets=activation_offsets,
            conditional_execution=conditional_execution,
        )
    except _MegaFusedUnsupportedError as exc:
        raise NotImplementedError(str(exc)) from exc
