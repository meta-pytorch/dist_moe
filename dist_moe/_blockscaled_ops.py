# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed block-scaled grouped GEMM interface for expert parallelism.

CuTe launchers require explicit caller-owned peer barriers. ``sync_peers=False``
is retained for signature compatibility; ``True`` is unsupported.

Dense tensors with one stable logical shape use shape suffixes. Packed weights,
peer-pointer sources, destinations, and offset bundles retain semantic names
because their physical layouts vary by format, pipeline, and execution mode.
"""

import functools
import math
from typing import Any

import torch

from ._buffers import SymmetricMemoryBuffer
from .formats import (
    BlockScaledFormat,
    ScaleFactorLayout,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)
from .kernels.config import (
    DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
)

DEFAULT_BACKEND: str = "cute"
DEFAULT_FORMAT: BlockScaledFormat = BlockScaledFormat.MXFP8_E4M3
DEFAULT_LAYOUT: ScaleFactorLayout = ScaleFactorLayout.CUBLAS_BLOCKED
DEFAULT_M_MULTIPLE_OF: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF


def _reject_implicit_peer_sync(sync_peers: bool) -> None:
    """Require peer publication ordering to remain caller-owned.

    Args:
        sync_peers: Whether a launcher-local peer synchronization was requested.

    Raises:
        NotImplementedError: If implicit synchronization was requested.
    """
    if sync_peers:
        raise NotImplementedError(
            "CuTe grouped GEMM requires explicit peer barriers at the caller"
        )


def _reject_owned_dispatch_staging(copy_inputs: bool) -> None:
    """Require dispatch inputs to be staged by the orchestration layer.

    Args:
        copy_inputs: Whether the launcher was asked to copy source inputs.

    Raises:
        NotImplementedError: If launcher-owned staging was requested.
    """
    if copy_inputs:
        raise NotImplementedError(
            "CuTe grouped GEMM dispatch requires caller-staged inputs; "
            "pass copy_inputs=False"
        )


@functools.lru_cache(maxsize=1)
def _get_available_backends() -> tuple[str, ...]:
    """Return importable distributed block-scaled kernel backends."""
    backends: list[str] = []
    try:
        from .kernels import dist_blockscaled_grouped_gemm as mod

        if mod.dist_blockscaled_grouped_gemm_fprop_dispatch is not None:
            backends.append("cute")
    except (ImportError, AttributeError, OSError):
        pass
    return tuple(backends)


def _resolve(backend: str) -> Any:
    """Import the requested kernel module.

    Args:
        backend: Backend name.

    Returns:
        Imported kernel module.

    Raises:
        ValueError: If the backend is unknown.
    """
    if backend == "cute":
        from .kernels import dist_blockscaled_grouped_gemm as mod
    else:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {_get_available_backends()}"
        )
    return mod


def _kernel_format_constant(mod: Any, format: BlockScaledFormat) -> Any:
    """Resolve the kernel module's format specialization.

    Args:
        mod: Imported kernel module.
        format: Internal block-scaled format.

    Returns:
        Kernel format constant.

    Raises:
        ValueError: If the format is unsupported.
    """
    match format:
        case BlockScaledFormat.MXFP8_E4M3:
            return mod.MXFP8_E4M3
        case BlockScaledFormat.MXFP8_E5M2:
            return mod.MXFP8_E5M2
        case BlockScaledFormat.NVFP4:
            return mod.NVFP4
        case BlockScaledFormat.MXFP4:
            return mod.MXFP4
        case _:
            raise ValueError(f"Unsupported block-scaled format: {format}")


def _require_cute_scale_layout(layout: ScaleFactorLayout) -> None:
    """Require the CUBLAS-blocked scale layout consumed by CuTe.

    Args:
        layout: Supplied scale-factor layout.

    Raises:
        NotImplementedError: If the layout is not CUBLAS-blocked.
    """
    if layout != ScaleFactorLayout.CUBLAS_BLOCKED:
        raise NotImplementedError(
            "dist_blockscaled_grouped_gemm currently expects CuTe-native "
            f"CUBLAS_BLOCKED scale layout, got {layout}"
        )


def _view_flat_scale(scale: torch.Tensor, name: str) -> torch.Tensor:
    """Return a flat view of a CuTe scale buffer.

    Args:
        scale: Scale tensor.
        name: Argument name used in errors.

    Returns:
        One-dimensional tensor view.

    Raises:
        ValueError: If the tensor cannot be flattened without a copy.
    """
    if scale.dim() == 1:
        return scale
    try:
        return scale.view(-1)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} must be 1D or viewable as a flat CuTe scale buffer; "
            f"got shape={tuple(scale.shape)}, stride={tuple(scale.stride())}"
        ) from exc


def dist_blockscaled_grouped_gemm_fprop_dispatch(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    num_tokens_per_local_expert_E: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer: SymmetricMemoryBuffer,
    *,
    topk: int,
    format: BlockScaledFormat = DEFAULT_FORMAT,
    layout: ScaleFactorLayout = DEFAULT_LAYOUT,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    sync_peers: bool = False,
    copy_inputs: bool = True,
    y_MN: torch.Tensor | None = None,
    config: dict | None = None,
    backend: str = DEFAULT_BACKEND,
    m_multiple_of: int = DEFAULT_M_MULTIPLE_OF,
    return_wgrad_quant: bool = False,
    blockscaled_dispatch: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    counter_storage: torch.Tensor | None = None,
    w_global_scale_inv: torch.Tensor | None = None,
    interleaved_fc13: bool = False,
    swiglu_fast_math: bool = False,
) -> Any:
    """Gather+quantize A from peers, then run block-scaled FPROP GEMM.

    ``counter_storage`` must be zeroed earlier on the launch stream.
    CuTe callers must pre-stage inputs, bracket peer access with barriers, and
    pass ``sync_peers=False, copy_inputs=False``.

    Args:
        x: Local activation source or peer pointer table.
        w: Prepared FPROP weight data.
        w_scale: Prepared FPROP weight scales.
        num_tokens_per_local_expert_E: Padded rows for each local expert.
        gather_ptrs: Gather pointer table.
        num_out_tokens: Optional output-row count.
        symm_mem_buffer: Symmetric dispatch allocation.
        topk: Routes per token.
        format: Block-scaled operand format.
        layout: Weight scale-factor layout.
        out_dtype: Dense output dtype.
        num_sms: Optional SM limit.
        sync_peers: Unsupported launcher-local synchronization request.
        copy_inputs: Unsupported launcher-local staging request.
        y_MN: Optional dense output destination with shape ``[M, N]``.
        config: Optional kernel tuning dictionary.
        backend: Kernel backend name.
        m_multiple_of: Per-expert row padding multiple.
        return_wgrad_quant: Whether to return column-oriented WGRAD input.
        blockscaled_dispatch: Whether ``x`` is already packed and quantized.
        activation_buffer: Optional activation buffer.
        activation_offsets: Optional device-side activation-buffer offsets.
        conditional_execution: Optional device recompute predicate.
        counter_storage: Pre-cleared decode counter storage.
        w_global_scale_inv: Optional reciprocal NVFP4 weight scale.
        interleaved_fc13: Whether FC13 uses interleaved NVFP4 storage.
        swiglu_fast_math: Whether the fused producer uses fast-math SwiGLU.

    Returns:
        Dense grouped-GEMM output and any requested WGRAD operand.
    """
    _reject_implicit_peer_sync(sync_peers)
    _reject_owned_dispatch_staging(copy_inputs)
    _require_cute_scale_layout(layout)
    mod = _resolve(backend)
    return mod.dist_blockscaled_grouped_gemm_fprop_dispatch(
        x=x,
        w=w,
        sfb=_view_flat_scale(w_scale, "w_scale"),
        split_sizes=num_tokens_per_local_expert_E,
        gather_ptrs=gather_ptrs,
        num_out_tokens=num_out_tokens,
        symm_mem_buffer=symm_mem_buffer,
        topk=topk,
        y=y_MN,
        format=_kernel_format_constant(mod, format),
        out_dtype=out_dtype,
        num_sms=num_sms,
        config=config,
        m_multiple_of=m_multiple_of,
        return_wgrad_quant=return_wgrad_quant,
        blockscaled_dispatch=blockscaled_dispatch,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        counter_storage=counter_storage,
        b_global_scale_inv=w_global_scale_inv,
        interleaved_fc13=interleaved_fc13,
        swiglu_fast_math=swiglu_fast_math,
    )


def dist_blockscaled_grouped_gemm_dgrad_dispatch(
    dy: torch.Tensor,
    w: torch.Tensor,
    w_scale_dgrad: torch.Tensor,
    num_tokens_per_local_expert_E: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer: SymmetricMemoryBuffer,
    *,
    format: BlockScaledFormat = DEFAULT_FORMAT,
    layout: ScaleFactorLayout = DEFAULT_LAYOUT,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    sync_peers: bool = False,
    copy_inputs: bool = True,
    dx_MK: torch.Tensor | None = None,
    config: dict | None = None,
    backend: str = DEFAULT_BACKEND,
    m_multiple_of: int = DEFAULT_M_MULTIPLE_OF,
    return_wgrad_quant: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
) -> Any:
    """Gather+quantize dy from peers, then run block-scaled DGRAD GEMM.

    CuTe callers must pre-stage inputs and pass ``copy_inputs=False``.

    Args:
        dy: Local output-gradient source or peer pointer table.
        w: Prepared DGRAD weight data.
        w_scale_dgrad: Prepared DGRAD weight scales.
        num_tokens_per_local_expert_E: Padded rows for each local expert.
        gather_ptrs: Gather pointer table.
        num_out_tokens: Optional output-row count.
        symm_mem_buffer: Symmetric dispatch allocation.
        format: Block-scaled operand format.
        layout: Weight scale-factor layout.
        out_dtype: Dense output dtype.
        num_sms: Optional SM limit.
        sync_peers: Unsupported launcher-local synchronization request.
        copy_inputs: Unsupported launcher-local staging request.
        dx_MK: Optional dense DGRAD destination with shape ``[M, K]``.
        config: Optional kernel tuning dictionary.
        backend: Kernel backend name.
        m_multiple_of: Per-expert row padding multiple.
        return_wgrad_quant: Whether to return column-oriented WGRAD input.
        activation_buffer: Optional activation buffer.
        activation_offsets: Optional device-side activation-buffer offsets.
        conditional_execution: Optional device recompute predicate.

    Returns:
        Dense DGRAD output and any requested WGRAD operand.
    """
    _reject_implicit_peer_sync(sync_peers)
    _reject_owned_dispatch_staging(copy_inputs)
    _require_cute_scale_layout(layout)
    mod = _resolve(backend)
    return mod.dist_blockscaled_grouped_gemm_dgrad_dispatch(
        dy=dy,
        w=w,
        sfb=_view_flat_scale(w_scale_dgrad, "w_scale_dgrad"),
        split_sizes=num_tokens_per_local_expert_E,
        gather_ptrs=gather_ptrs,
        num_out_tokens=num_out_tokens,
        symm_mem_buffer=symm_mem_buffer,
        dx=dx_MK,
        format=_kernel_format_constant(mod, format),
        out_dtype=out_dtype,
        num_sms=num_sms,
        config=config,
        m_multiple_of=m_multiple_of,
        return_wgrad_quant=return_wgrad_quant,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
    )


def dist_blockscaled_grouped_gemm_dgrad_wgrad_dispatch(
    dy: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    num_tokens_per_local_expert_E: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer: SymmetricMemoryBuffer,
    *,
    x_wgrad_quant: tuple[torch.Tensor, torch.Tensor],
    format: BlockScaledFormat = DEFAULT_FORMAT,
    layout: ScaleFactorLayout = DEFAULT_LAYOUT,
    out_dtype: torch.dtype = torch.bfloat16,
    wgrad_out_dtype: torch.dtype | None = None,
    num_sms: int | None = None,
    sync_peers: bool = False,
    copy_inputs: bool = True,
    dx_MK: torch.Tensor | None = None,
    dw_GNK: torch.Tensor | None = None,
    config: dict | None = None,
    backend: str = DEFAULT_BACKEND,
    m_multiple_of: int = DEFAULT_M_MULTIPLE_OF,
    output_accum: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Gather and quantize DY, then run fused DGRAD and WGRAD.

    Args:
        dy: Local output-gradient source or peer pointer table.
        w: Prepared DGRAD weight data.
        w_scale: Prepared DGRAD weight scales.
        num_tokens_per_local_expert_E: Padded rows for each local expert.
        gather_ptrs: Gather pointer table.
        num_out_tokens: Optional output-row count.
        symm_mem_buffer: Symmetric dispatch allocation.
        x_wgrad_quant: Column-oriented WGRAD activation and scales.
        format: Block-scaled operand format.
        layout: Weight scale-factor layout.
        out_dtype: Dense DGRAD dtype.
        wgrad_out_dtype: Optional WGRAD dtype.
        num_sms: Optional SM limit.
        sync_peers: Unsupported launcher-local synchronization request.
        copy_inputs: Unsupported launcher-local staging request.
        dx_MK: Optional dense DGRAD destination with shape ``[M, K]``.
        dw_GNK: Optional grouped WGRAD destination with shape ``[G, N, K]``.
        config: Optional kernel tuning dictionary.
        backend: Kernel backend name.
        m_multiple_of: Per-expert row padding multiple.
        output_accum: Whether to accumulate into ``dw_GNK``.
        activation_buffer: Optional activation buffer.
        activation_offsets: Optional device-side activation-buffer offsets.
        conditional_execution: Optional device recompute predicate.

    Returns:
        DGRAD, WGRAD, and column-oriented DY quantization.
    """
    _reject_implicit_peer_sync(sync_peers)
    _reject_owned_dispatch_staging(copy_inputs)
    _require_cute_scale_layout(layout)
    if backend == "cute":
        from .kernels import mega_blockscaled_grouped_gemm as mod
    else:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {_get_available_backends()}"
        )
    return mod.mega_blockscaled_grouped_gemm_dgrad_wgrad_dispatch(
        dy=dy,
        w=w,
        sfb=_view_flat_scale(w_scale, "w_scale"),
        split_sizes=num_tokens_per_local_expert_E,
        gather_ptrs=gather_ptrs,
        num_out_tokens=num_out_tokens,
        symm_mem_buffer=symm_mem_buffer,
        x_wgrad_quant=x_wgrad_quant,
        dx=dx_MK,
        dw=dw_GNK,
        format=_kernel_format_constant(mod, format),
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


def dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine(
    h1_M2F: torch.Tensor | None,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    num_tokens_per_local_expert_E: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer: SymmetricMemoryBuffer,
    *,
    format: BlockScaledFormat = DEFAULT_FORMAT,
    layout: ScaleFactorLayout = DEFAULT_LAYOUT,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    sync_peers: bool = False,
    y_TD: torch.Tensor | None = None,
    config: dict | None = None,
    backend: str = DEFAULT_BACKEND,
    m_multiple_of: int = DEFAULT_M_MULTIPLE_OF,
    num_output_tokens: int | None = None,
    return_wgrad_quant: bool = False,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    num_recv_tokens: int | None = None,
    counter_storage: torch.Tensor | None = None,
    w_global_scale_inv: torch.Tensor | None = None,
    h2_quant: (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
        ]
        | None
    ) = None,
    precomputed_swiglu: bool = False,
    return_row_quant: bool = False,
) -> Any:
    """Run SwiGLU fwd + block-scaled FPROP GEMM, then scatter C to peers.

    ``counter_storage`` must be zeroed earlier on the launch stream.
    ``return_row_quant`` additionally returns the row-quantized ``h2`` A operand.

    Args:
        h1_M2F: Dense FC13 output with shape ``[M, 2F]``, or ``None`` when
            read from the activation buffer.
        w: Prepared FC2 FPROP weight data.
        w_scale: Prepared FC2 FPROP weight scales.
        num_tokens_per_local_expert_E: Padded rows for each local expert.
        scatter_ptrs: Peer combine pointer table.
        symm_mem_buffer: Symmetric combine allocation.
        format: Block-scaled operand format.
        layout: Weight scale-factor layout.
        out_dtype: Dense output dtype.
        num_sms: Optional SM limit.
        sync_peers: Unsupported launcher-local synchronization request.
        y_TD: Optional combined output destination with shape ``[T, D]``.
        config: Optional kernel tuning dictionary.
        backend: Kernel backend name.
        m_multiple_of: Per-expert row padding multiple.
        num_output_tokens: Optional local output-row count.
        return_wgrad_quant: Whether to return column-oriented H2.
        swiglu_fast_math: Whether SwiGLU uses fast math.
        swiglu_clamped: Whether to use clamped SwiGLU.
        swiglu_alpha: Clamped-SwiGLU alpha.
        swiglu_limit: Clamped-SwiGLU input limit.
        activation_buffer: Optional activation buffer.
        activation_offsets: Optional device-side activation-buffer offsets.
        conditional_execution: Optional device recompute predicate.
        num_recv_tokens: Optional active received-row count.
        counter_storage: Pre-cleared decode counter storage.
        w_global_scale_inv: Optional reciprocal NVFP4 weight scale.
        h2_quant: Optional precomputed H2 quantization.
        precomputed_swiglu: Whether ``h2_quant`` replaces SwiGLU computation.
        return_row_quant: Whether to return row-oriented H2 quantization.

    Returns:
        Combined output and requested saved quantized operands.
    """
    _reject_implicit_peer_sync(sync_peers)
    _require_cute_scale_layout(layout)
    mod = _resolve(backend)
    return mod.dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine(
        h1=h1_M2F,
        w=w,
        sfb=_view_flat_scale(w_scale, "w_scale"),
        split_sizes=num_tokens_per_local_expert_E,
        scatter_ptrs=scatter_ptrs,
        symm_mem_buffer=symm_mem_buffer,
        y=y_TD,
        format=_kernel_format_constant(mod, format),
        out_dtype=out_dtype,
        num_sms=num_sms,
        config=config,
        m_multiple_of=m_multiple_of,
        num_output_tokens=num_output_tokens,
        return_wgrad_quant=return_wgrad_quant,
        swiglu_fast_math=swiglu_fast_math,
        swiglu_clamped=swiglu_clamped,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        num_recv_tokens=num_recv_tokens,
        counter_storage=counter_storage,
        b_global_scale_inv=w_global_scale_inv,
        h2_quant=h2_quant,
        precomputed_swiglu=precomputed_swiglu,
        return_row_quant=return_row_quant,
    )


def dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_dispatch_combine(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    num_tokens_per_local_expert_E: torch.Tensor,
    gather_ptrs: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    num_out_tokens: int,
    symm_mem_buffer: SymmetricMemoryBuffer,
    *,
    format: BlockScaledFormat = DEFAULT_FORMAT,
    layout: ScaleFactorLayout = DEFAULT_LAYOUT,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    sync_peers: bool = False,
    config: dict | None = None,
    backend: str = DEFAULT_BACKEND,
    m_multiple_of: int = DEFAULT_M_MULTIPLE_OF,
    num_output_tokens: int | None = None,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    return_x_wgrad_quant: bool = False,
    blockscaled_dispatch: bool = False,
    return_h2_wgrad_quant: bool = True,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    counter_storage: torch.Tensor | None = None,
    w13_global_scale_inv: torch.Tensor | None = None,
    w2_global_scale_inv: torch.Tensor | None = None,
    interleaved_fc13: bool = False,
    return_row_quant: bool = False,
) -> Any:
    """Run fused block-scaled dispatch, FC13, SwiGLU quant, FC2, and combine.

    ``counter_storage`` must be zeroed earlier on the launch stream.
    ``return_row_quant`` appends the row-quantized ``x_gathered`` and ``h2``
    A operands to the result tuple.

    Args:
        w13: Prepared fused gate/up FPROP weight data.
        w13_scale: Prepared W13 FPROP scales.
        w2: Prepared down-projection FPROP weight data.
        w2_scale: Prepared W2 FPROP scales.
        num_tokens_per_local_expert_E: Padded rows for each local expert.
        gather_ptrs: Peer dispatch pointer table.
        scatter_ptrs: Peer combine pointer table.
        num_out_tokens: Received-row capacity.
        symm_mem_buffer: Symmetric communication allocation.
        format: Block-scaled operand format.
        layout: Weight scale-factor layout.
        out_dtype: Dense output dtype.
        num_sms: Optional SM limit.
        sync_peers: Unsupported launcher-local synchronization request.
        config: Optional kernel tuning dictionary.
        backend: Kernel backend name.
        m_multiple_of: Per-expert row padding multiple.
        num_output_tokens: Optional local output-row count.
        swiglu_fast_math: Whether SwiGLU uses fast math.
        swiglu_clamped: Whether to use clamped SwiGLU.
        swiglu_alpha: Clamped-SwiGLU alpha.
        swiglu_limit: Clamped-SwiGLU input limit.
        return_x_wgrad_quant: Whether to return column-oriented dispatched X.
        blockscaled_dispatch: Whether dispatch input is already quantized.
        return_h2_wgrad_quant: Whether to return column-oriented H2.
        activation_buffer: Optional activation buffer.
        activation_offsets: Optional device-side activation-buffer offsets.
        conditional_execution: Optional device recompute predicate.
        counter_storage: Pre-cleared decode counter storage.
        w13_global_scale_inv: Optional reciprocal NVFP4 W13 scale.
        w2_global_scale_inv: Optional reciprocal NVFP4 W2 scale.
        interleaved_fc13: Whether FC13 uses interleaved NVFP4 storage.
        return_row_quant: Whether to return row-oriented activation quants.

    Returns:
        Combined output and requested saved quantized operands.
    """
    _reject_implicit_peer_sync(sync_peers)
    _require_cute_scale_layout(layout)
    if backend != "cute":
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {_get_available_backends()}"
        )
    from .kernels import chunked_mega_blockscaled_grouped_gemm as mod

    kernel_config = None if config is None else dict(config)
    chunk_rows = 512
    if kernel_config is not None:
        explicit_chunk_rows = "PIPELINE_CHUNK_ROWS" in kernel_config
        chunk_rows = int(kernel_config.pop("PIPELINE_CHUNK_ROWS", chunk_rows))
        token_tile = int(kernel_config["BLOCK_SIZE_N"])
        if chunk_rows % token_tile != 0:
            if explicit_chunk_rows:
                raise ValueError(
                    "PIPELINE_CHUNK_ROWS must be divisible by BLOCK_SIZE_N; "
                    f"got PIPELINE_CHUNK_ROWS={chunk_rows}, BLOCK_SIZE_N={token_tile}"
                )
            chunk_rows = math.lcm(128, token_tile)
    return mod.chunked_mega_blockscaled_grouped_gemm_fprop_swiglu_fwd(
        w13=w13,
        w13_scale=_view_flat_scale(w13_scale, "w13_scale"),
        w2=w2,
        w2_scale=_view_flat_scale(w2_scale, "w2_scale"),
        split_sizes=num_tokens_per_local_expert_E,
        gather_ptrs=gather_ptrs,
        scatter_ptrs=scatter_ptrs,
        num_out_tokens=num_out_tokens,
        symm_mem_buffer=symm_mem_buffer,
        num_output_tokens=num_output_tokens,
        format=_kernel_format_constant(mod, format),
        out_dtype=out_dtype,
        num_sms=num_sms,
        config=kernel_config,
        m_multiple_of=m_multiple_of,
        swiglu_fast_math=swiglu_fast_math,
        swiglu_clamped=swiglu_clamped,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        chunk_rows=chunk_rows,
        return_x_wgrad_quant=return_x_wgrad_quant,
        blockscaled_dispatch=blockscaled_dispatch,
        return_h2_wgrad_quant=return_h2_wgrad_quant,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        counter_storage=counter_storage,
        w13_global_scale_inv=w13_global_scale_inv,
        w2_global_scale_inv=w2_global_scale_inv,
        interleaved_fc13=interleaved_fc13,
        return_row_quant=return_row_quant,
    )


def dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine(
    grad_h2_MF: torch.Tensor,
    h1_M2F: torch.Tensor,
    w: torch.Tensor,
    w_scale_dgrad: torch.Tensor,
    num_tokens_per_local_expert_E: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer: SymmetricMemoryBuffer,
    *,
    format: BlockScaledFormat = DEFAULT_FORMAT,
    layout: ScaleFactorLayout = DEFAULT_LAYOUT,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    sync_peers: bool = False,
    dx_TD: torch.Tensor | None = None,
    config: dict | None = None,
    backend: str = DEFAULT_BACKEND,
    m_multiple_of: int = DEFAULT_M_MULTIPLE_OF,
    num_output_tokens: int | None = None,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    num_recv_tokens: int | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Run fused SwiGLU backward quantization, DGRAD, and peer combine.

    Args:
        grad_h2_MF: Dense SwiGLU-output gradient with shape ``[M, F]``.
        h1_M2F: Saved dense SwiGLU input with shape ``[M, 2F]``.
        w: Prepared W13 DGRAD weight data.
        w_scale_dgrad: Prepared W13 DGRAD weight scales.
        num_tokens_per_local_expert_E: Padded rows for each local expert.
        scatter_ptrs: Peer combine pointer table.
        symm_mem_buffer: Symmetric combine allocation.
        format: Block-scaled operand format.
        layout: Weight scale-factor layout.
        out_dtype: Dense DGRAD dtype.
        num_sms: Optional SM limit.
        sync_peers: Unsupported launcher-local synchronization request.
        dx_TD: Optional combined DGRAD destination with shape ``[T, D]``.
        config: Optional kernel tuning dictionary.
        backend: Kernel backend name.
        m_multiple_of: Per-expert row padding multiple.
        num_output_tokens: Optional local output-row count.
        swiglu_fast_math: Whether SwiGLU uses fast math.
        swiglu_clamped: Whether to use clamped SwiGLU.
        swiglu_alpha: Clamped-SwiGLU alpha.
        swiglu_limit: Clamped-SwiGLU input limit.
        activation_buffer: Optional activation buffer.
        activation_offsets: Optional device-side activation-buffer offsets.
        conditional_execution: Optional device recompute predicate.
        num_recv_tokens: Optional active received-row count.

    Returns:
        Dense DGRAD and column-oriented DXY quantization.
    """
    _reject_implicit_peer_sync(sync_peers)
    _require_cute_scale_layout(layout)
    mod = _resolve(backend)
    return mod.dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine(
        grad_h2=grad_h2_MF,
        h1=h1_M2F,
        w=w,
        sfb=_view_flat_scale(w_scale_dgrad, "w_scale_dgrad"),
        split_sizes=num_tokens_per_local_expert_E,
        scatter_ptrs=scatter_ptrs,
        symm_mem_buffer=symm_mem_buffer,
        dx=dx_TD,
        format=_kernel_format_constant(mod, format),
        out_dtype=out_dtype,
        num_sms=num_sms,
        config=config,
        m_multiple_of=m_multiple_of,
        num_output_tokens=num_output_tokens,
        swiglu_fast_math=swiglu_fast_math,
        swiglu_clamped=swiglu_clamped,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        num_recv_tokens=num_recv_tokens,
    )


def dist_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine(
    grad_h2_MF: torch.Tensor,
    h1_M2F: torch.Tensor,
    w: torch.Tensor,
    w_scale_dgrad: torch.Tensor,
    num_tokens_per_local_expert_E: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer: SymmetricMemoryBuffer,
    *,
    x_wgrad_quant: tuple[torch.Tensor, torch.Tensor],
    format: BlockScaledFormat = DEFAULT_FORMAT,
    layout: ScaleFactorLayout = DEFAULT_LAYOUT,
    out_dtype: torch.dtype = torch.bfloat16,
    wgrad_out_dtype: torch.dtype | None = None,
    num_sms: int | None = None,
    sync_peers: bool = False,
    dx_TD: torch.Tensor | None = None,
    dw_GNK: torch.Tensor | None = None,
    config: dict | None = None,
    backend: str = DEFAULT_BACKEND,
    m_multiple_of: int = DEFAULT_M_MULTIPLE_OF,
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
    """Run SwiGLU backward quantization with fused W13 DGRAD and WGRAD.

    Args:
        grad_h2_MF: Dense SwiGLU-output gradient with shape ``[M, F]``.
        h1_M2F: Saved dense SwiGLU input with shape ``[M, 2F]``.
        w: Prepared W13 DGRAD weight data.
        w_scale_dgrad: Prepared W13 DGRAD weight scales.
        num_tokens_per_local_expert_E: Padded rows for each local expert.
        scatter_ptrs: Peer combine pointer table.
        symm_mem_buffer: Symmetric combine allocation.
        x_wgrad_quant: Column-oriented W13 WGRAD input and scales.
        format: Block-scaled operand format.
        layout: Weight scale-factor layout.
        out_dtype: Dense DGRAD dtype.
        wgrad_out_dtype: Optional WGRAD dtype.
        num_sms: Optional SM limit.
        sync_peers: Unsupported launcher-local synchronization request.
        dx_TD: Optional combined DGRAD destination with shape ``[T, D]``.
        dw_GNK: Optional grouped WGRAD destination with shape ``[G, N, K]``.
        config: Optional kernel tuning dictionary.
        backend: Kernel backend name.
        m_multiple_of: Per-expert row padding multiple.
        num_output_tokens: Optional local output-row count.
        output_accum: Whether to accumulate into ``dw_GNK``.
        swiglu_fast_math: Whether SwiGLU uses fast math.
        swiglu_clamped: Whether to use clamped SwiGLU.
        swiglu_alpha: Clamped-SwiGLU alpha.
        swiglu_limit: Clamped-SwiGLU input limit.
        activation_buffer: Optional activation buffer.
        activation_offsets: Optional device-side activation-buffer offsets.
        conditional_execution: Optional device recompute predicate.

    Returns:
        Dense DGRAD, WGRAD, and column-oriented DXY quantization.
    """
    _reject_implicit_peer_sync(sync_peers)
    _require_cute_scale_layout(layout)
    if backend == "cute":
        from .kernels import mega_blockscaled_grouped_gemm as mod
    else:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {_get_available_backends()}"
        )
    return mod.mega_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine(
        grad_h2=grad_h2_MF,
        h1=h1_M2F,
        w=w,
        sfb=_view_flat_scale(w_scale_dgrad, "w_scale_dgrad"),
        split_sizes=num_tokens_per_local_expert_E,
        scatter_ptrs=scatter_ptrs,
        symm_mem_buffer=symm_mem_buffer,
        x_wgrad_quant=x_wgrad_quant,
        dx=dx_TD,
        dw=dw_GNK,
        format=_kernel_format_constant(mod, format),
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
