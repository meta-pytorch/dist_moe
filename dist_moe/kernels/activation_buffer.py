# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation-buffer handling for the DistMoE kernels.

The torch-only host layer owns the operand offset table, launch
preparation/validation, and symmetric-memory staging. It must stay importable
without ``cutlass`` because host-side planners import its offset constants;
CUTLASS-dependent staging shapes and device-side ``@cute.jit`` views live in
``activation_buffer_kernel``.
"""

import functools

import torch

ACTIVATION_BUFFER_ALIGNMENT_BYTES: int = 16
ACTIVATION_ROW_SCALE_STORAGE_MULTIPLE: int = 4
MISSING_ACTIVATION_OFFSET: int = -1

ACTIVATION_A_Q_OFFSET: int = 0
ACTIVATION_B_Q_OFFSET: int = 1
ACTIVATION_C_OFFSET: int = 2
ACTIVATION_A_SCALE_OFFSET: int = 3
ACTIVATION_B_SCALE_OFFSET: int = 4
ACTIVATION_SOURCE_X_OFFSET: int = 5
ACTIVATION_SOURCE_Y_OFFSET: int = 6
ACTIVATION_COL_Q_OFFSET: int = 7
ACTIVATION_COL_SCALE_OFFSET: int = 8
GEMM_OPERAND_OFFSET_COUNT: int = ACTIVATION_B_SCALE_OFFSET + 1
ACTIVATION_OFFSET_COUNT: int = ACTIVATION_COL_SCALE_OFFSET + 1

MEGA_FIRST_GEMM_OFFSET_BASE: int = 0
MEGA_SECOND_GEMM_OFFSET_BASE: int = ACTIVATION_OFFSET_COUNT
MEGA_ACTIVATION_OFFSET_COUNT: int = 2 * ACTIVATION_OFFSET_COUNT

FORWARD_DISPATCH_OFFSET_BASE: int = 0
FORWARD_COMBINE_OFFSET_BASE: int = ACTIVATION_OFFSET_COUNT
FORWARD_ACTIVATION_OFFSET_COUNT: int = 2 * ACTIVATION_OFFSET_COUNT

BACKWARD_RECOMPUTE_DISPATCH_OFFSET_BASE: int = 0
BACKWARD_RECOMPUTE_COMBINE_OFFSET_BASE: int = ACTIVATION_OFFSET_COUNT
BACKWARD_FC2_DGRAD_OFFSET_BASE: int = 2 * ACTIVATION_OFFSET_COUNT
BACKWARD_FC2_WGRAD_OFFSET_BASE: int = 3 * ACTIVATION_OFFSET_COUNT
BACKWARD_FC13_DGRAD_OFFSET_BASE: int = 4 * ACTIVATION_OFFSET_COUNT
BACKWARD_FC13_WGRAD_OFFSET_BASE: int = 5 * ACTIVATION_OFFSET_COUNT
BACKWARD_ACTIVATION_OFFSET_COUNT: int = 6 * ACTIVATION_OFFSET_COUNT


def _canonical_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _devices_match(lhs: torch.device, rhs: torch.device) -> bool:
    return _canonical_device(lhs) == _canonical_device(rhs)


@functools.lru_cache(maxsize=None)
def _unconditional_execution_placeholder(device: torch.device) -> torch.Tensor:
    return torch.ones(1, dtype=torch.int32, device=_canonical_device(device))


def validate_activation_buffer_storage(
    activation_buffer: torch.Tensor,
) -> None:
    if activation_buffer.dtype != torch.uint8:
        raise TypeError("activation_buffer must have dtype uint8")
    if activation_buffer.ndim != 1 or not activation_buffer.is_contiguous():
        raise ValueError("activation_buffer must be a contiguous 1D tensor")
    if activation_buffer.data_ptr() % ACTIVATION_BUFFER_ALIGNMENT_BYTES != 0:
        raise ValueError(
            "activation_buffer must be aligned to "
            f"{ACTIVATION_BUFFER_ALIGNMENT_BYTES} bytes"
        )


def activation_buffer_placeholder(
    activation_buffer: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Describe an offset-backed operand without allocating separate storage.

    Placeholders intentionally alias byte zero and are never dereferenced: the
    kernel replaces their base pointers with activation-buffer offsets.
    """
    validate_activation_buffer_storage(activation_buffer)
    numel = 1
    for size in shape:
        numel *= size
    required_bytes = numel * dtype.itemsize
    if required_bytes > activation_buffer.numel():
        raise ValueError(
            f"activation buffer cannot describe placeholder shape={shape}, dtype={dtype}: "
            f"requires {required_bytes} bytes, has {activation_buffer.numel()}"
        )
    return activation_buffer[:required_bytes].view(dtype).view(shape)


def validate_activation_buffer_launch(
    *,
    activation_buffer: torch.Tensor,
    activation_offsets: torch.Tensor,
    expected_offset_count: int,
    device: torch.device,
) -> None:
    """Validate activation-buffer and offset-table storage.

    Offset values are produced by CUDA planners and are not inspected here, so
    validation never introduces a device-to-host synchronization.
    """
    validate_activation_buffer_storage(activation_buffer)
    if not _devices_match(activation_buffer.device, device):
        raise ValueError("activation_buffer must be on the operand device")
    if activation_offsets.shape != (expected_offset_count,):
        raise ValueError(
            "activation_offsets must contain "
            f"{expected_offset_count} elements, got shape={tuple(activation_offsets.shape)}"
        )
    if activation_offsets.dtype != torch.int64:
        raise TypeError("activation_offsets must have dtype int64")
    if (
        not _devices_match(activation_offsets.device, device)
        or not activation_offsets.is_contiguous()
    ):
        raise ValueError(
            "activation_offsets must be contiguous and on the operand device"
        )


def validate_conditional_execution(
    conditional_execution: torch.Tensor,
    *,
    device: torch.device,
) -> None:
    """Validate a condition tensor used by conditional kernel launches.

    Bool conditions are normalized to a separate int32 launch tensor. Callers
    that mutate a live condition between launches must pass int32 storage.
    """
    if conditional_execution.dtype not in (torch.bool, torch.int32):
        raise TypeError("conditional_execution must have dtype bool or int32")
    if (
        conditional_execution.numel() != 1
        or not _devices_match(conditional_execution.device, device)
        or not conditional_execution.is_contiguous()
    ):
        raise ValueError(
            "conditional_execution must be a contiguous single-element tensor "
            "on the operand device"
        )


def prepare_activation_buffer_launch(
    *,
    activation_buffer: torch.Tensor | None,
    activation_offsets: torch.Tensor | None,
    conditional_execution: torch.Tensor | None,
    fallback_offsets: torch.Tensor,
    expected_offset_count: int,
) -> tuple[int, torch.Tensor, bool, torch.Tensor, bool, torch.Tensor | None, int]:
    """Normalize and validate activation-buffer launch storage.

    The returned tensors remain PyTorch-owned. CuTe callers convert them to
    runtime tensors immediately before launch.
    """
    device = fallback_offsets.device
    use_activation_buffer = activation_buffer is not None
    if use_activation_buffer:
        if activation_offsets is None:
            raise ValueError(
                "activation_buffer requires an activation_offsets tensor with "
                f"{expected_offset_count} elements"
            )
        validate_activation_buffer_launch(
            activation_buffer=activation_buffer,
            activation_offsets=activation_offsets,
            expected_offset_count=expected_offset_count,
            device=device,
        )
        activation_buffer_base_ptr = activation_buffer.data_ptr()
        activation_buffer_size_bytes = activation_buffer.nbytes
        offsets = activation_offsets
    else:
        if activation_offsets is not None:
            raise ValueError("activation_offsets requires activation_buffer")
        activation_buffer_base_ptr = 0
        activation_buffer_size_bytes = 0
        fallback_offsets = fallback_offsets.view(-1)
        if fallback_offsets.numel() < expected_offset_count:
            raise ValueError(
                "fallback_offsets must have at least "
                f"{expected_offset_count} elements, got {fallback_offsets.numel()}"
            )
        offsets = fallback_offsets.narrow(0, 0, expected_offset_count)

    use_conditional_execution = conditional_execution is not None
    condition_owner = None
    if conditional_execution is None:
        condition = _unconditional_execution_placeholder(device)
    else:
        validate_conditional_execution(conditional_execution, device=device)
        if conditional_execution.dtype == torch.bool:
            condition = conditional_execution.to(torch.int32)
            condition_owner = condition
        else:
            condition = conditional_execution
    return (
        activation_buffer_base_ptr,
        offsets,
        use_activation_buffer,
        condition,
        use_conditional_execution,
        condition_owner,
        activation_buffer_size_bytes,
    )


_FP4_FORMAT_NAMES: frozenset[str] = frozenset({"nvfp4", "mxfp4"})


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _qdata_words(qdata: torch.Tensor) -> torch.Tensor:
    """View qdata as words; its last-dimension byte count must divide by four.

    FP4 column storage pads each column to whole words to satisfy this contract.
    """
    return qdata.view(torch.uint32)


def _row_quant_tuple(a, sfa, a_global_scale_inv):
    """Row bundle, plus NVFP4's per-token inverse global scale when it has one."""
    if a_global_scale_inv is None:
        return (a, sfa)
    return (a, sfa, a_global_scale_inv)
