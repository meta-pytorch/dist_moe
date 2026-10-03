# Copyright (c) 2025-2026, QuACK team.
# Licensed under the Apache License, Version 2.0.
# Modified by Meta Platforms, Inc.

from typing import Optional, Type

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, const_expr, Int32
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op

from .._quack_utils import make_vector


@dsl_user_op
def cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    retile: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    assert (
        isinstance(src.iterator, cute.Pointer)
        and src.memspace == cute.AddressSpace.rmem
    )
    if const_expr(src.element_type != dst.element_type):
        src = src.to(dst.element_type, loc=loc, ip=ip)
    if const_expr(retile):
        src = tiled_copy.retile(src)
    cute.copy(tiled_copy, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@dsl_user_op
def get_copy_atom(
    dtype: Type[cutlass.Numeric],
    num_copy_elems: int,
    is_async: bool = False,
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    num_copy_elems = src.shape[0][0]
    copy_atom = get_copy_atom(src.element_type, num_copy_elems, is_async)
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    assert num_threads % threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(
                tAcA[(0, rest_v), 0, rest_k][1], limit
            )
    return tApA


@dsl_user_op
@cute.jit
def store(
    ptr: cute.Pointer,
    val,
    pred: Optional[Boolean] = None,
    cop: cutlass.Constexpr = None,
    *,
    loc=None,
    ip=None,
):
    """Store a scalar value via cute.arch.store.

    ptr:  cute.Pointer (any address space).
    val:  DSL Numeric value.
    pred: None → unconditional.  DSL Boolean → skipped when pred == 0.
    cop:  Cache operator — "wb" (default), "cg", "cs" (streaming), "wt".
    """
    if const_expr(pred is None):
        cute.arch.store(ptr.llvm_ptr, type(val)(val), cop=cop, loc=loc, ip=ip)
    else:
        if pred:
            cute.arch.store(ptr.llvm_ptr, type(val)(val), cop=cop, loc=loc, ip=ip)


@dsl_user_op
@cute.jit
def store_v2(
    ptr: cute.Pointer,
    v0,
    v1,
    pred: Optional[Boolean] = None,
    cop: cutlass.Constexpr = None,
    *,
    loc=None,
    ip=None,
):
    """Vectorized store of 2 elements via cute.arch.store.

    Packs v0, v1 into an MLIR <2 x T> vector.
    ptr:  cute.Pointer (any address space, must be aligned for vector width).
    cop:  Cache operator — "wb" (default), "cg", "cs" (streaming), "wt".
    """
    vec = make_vector(type(v0), v0, v1, loc=loc, ip=ip)
    if const_expr(pred is None):
        cute.arch.store(ptr.llvm_ptr, vec, cop=cop, loc=loc, ip=ip)
    else:
        if pred:
            cute.arch.store(ptr.llvm_ptr, vec, cop=cop, loc=loc, ip=ip)
