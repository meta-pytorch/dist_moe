# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared TMA (Tensor Memory Accelerator) utilities for Triton kernels."""

from typing import Optional

import torch
import triton
import triton.language as tl

from .env import is_cuda, TRITON_VERSION

TMA_ALIGNMENT = 128
TMA_ALIGNMENT_TL_CONSTEXPR = tl.constexpr(TMA_ALIGNMENT)

if TRITON_VERSION == (3, 3):

    @triton.jit
    def _make_tensor_descriptor(  # type: ignore
        ptr,
        shape,
        strides,
        block_shape,
    ):
        return tl._experimental_make_tensor_descriptor(
            ptr,
            shape=shape,
            strides=strides,
            block_shape=block_shape,
        )

elif TRITON_VERSION >= (3, 5):

    @triton.jit
    def _make_tensor_descriptor(
        ptr,
        shape,
        strides,
        block_shape,
    ):
        return tl.make_tensor_descriptor(  # type: ignore
            ptr,
            shape=shape,
            strides=strides,
            block_shape=block_shape,
        )

else:
    raise RuntimeError(f"Unsupported Triton version {TRITON_VERSION}")


if TRITON_VERSION == (3, 3):

    @triton.jit
    def _tma_store(  # type: ignore
        c_desc,
        offset_cm,
        offset_cn,
        acc,
        c_ptr,
        OUTPUT_ACCUM: tl.constexpr,
    ):
        if OUTPUT_ACCUM:
            c = c_desc.load([offset_cm, offset_cn])  # type: ignore
            c = c.to(tl.float32) + acc
        else:
            c = acc
        c = c.to(c_ptr.dtype.element_ty)
        c_desc.store([offset_cm, offset_cn], c)  # type: ignore

elif TRITON_VERSION >= (3, 5):

    @triton.jit
    def _tma_store(  # type: ignore
        c_desc,
        offset_cm,
        offset_cn,
        acc,
        c_ptr,
        OUTPUT_ACCUM: tl.constexpr,
    ):
        c = acc.to(c_ptr.dtype.element_ty)
        if OUTPUT_ACCUM:
            c_desc.atomic_add(
                [offset_cm, offset_cn],
                c,
            )  # type: ignore
        else:
            c_desc.store(
                [offset_cm, offset_cn],
                c,
            )  # type: ignore
else:
    raise RuntimeError(f"Unsupported Triton version {TRITON_VERSION}")


def supports_tma():
    """Check if the current GPU supports TMA (requires CUDA with compute capability >= 9.0)."""
    return (
        is_cuda()
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
    )


def set_triton_allocator():
    """Set the Triton allocator for TMA descriptors.

    Always set the allocator (as it may be reset between kernel calls) — but
    only outside of torch.compile traces.

    Dynamo treats ``triton.set_allocator`` as untraceable because it mutates
    global Triton allocator state. Callers must invoke this function eagerly
    during setup before entering a compiled region; tracing then skips the
    redundant mutation.
    """
    if torch.compiler.is_compiling():
        return

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
