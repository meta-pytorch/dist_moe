# Copyright (c) 2025, Tri Dao.
# Licensed under the Apache License, Version 2.0.
# Modified by Meta Platforms, Inc.

import operator
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32
from cutlass.base_dsl.arch import Arch

from .. import _quack_utils as utils

_operator_max = getattr(operator, "max", None)


_operator_min = getattr(operator, "min", None)


_cutlass_min = getattr(cutlass, "min", None)


@cute.jit
def warp_reduce(
    val: cute.Numeric,
    op: Callable,
    threads_in_group: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
    dtype: cutlass.Constexpr = None,
    abs: cutlass.Constexpr[bool] = False,
    nan: cutlass.Constexpr[bool] = False,
) -> cute.Numeric:
    """Reduce across the aligned ``threads_in_group``-lane subgroup this thread
    belongs to (all lanes receive the result). Lowers to one ``redux.sync``
    where the hardware has it — Int32 everywhere, Float32 min/max on the SM100
    family (redux fp32 does not exist on SM120) — with a per-subgroup member
    mask for groups smaller than a warp; shuffle-butterfly otherwise.

    ``abs``: reduce |val| (fp32 min/max only). Folded into the REDUX.MAXABS
    instruction on the redux path, an explicit absf on the fallback.
    ``nan``: propagate NaN (any NaN input poisons the result — the CUTLASS
    amax semantics). Redux path only; the fallback butterfly's max.f32 drops
    NaNs, so only rely on this where the SM100 family is guaranteed.
    """
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    val_dtype = dtype if const_expr(dtype is not None) else getattr(val, "dtype", None)
    is_max = const_expr(op is max or op is cutlass.max or op is _operator_max)
    is_min = const_expr(op is min or op is _cutlass_min or op is _operator_min)
    kind = None
    if const_expr(val_dtype == Int32):
        if const_expr(op is operator.add):
            kind = "add"
        elif const_expr(is_max):
            kind = "max"
        elif const_expr(is_min):
            kind = "min"
    elif const_expr(val_dtype == Float32 and arch.is_family_of(Arch.sm_100f)):
        if const_expr(is_max or op is cute.arch.fmax):
            kind = "fmax"
        elif const_expr(is_min or op is cute.arch.fmin):
            kind = "fmin"
    if const_expr(kind is not None):
        if const_expr(threads_in_group == cute.arch.WARP_SIZE):
            mask = 0xFFFFFFFF
        else:
            # Aligned contiguous subgroup (same grouping as the butterfly
            # below): this lane's group mask, shifted to its base lane
            # (32 - g == the 5-bit ~(g - 1), kept positive for the DSL).
            group_mask = (1 << threads_in_group) - 1
            base_lane = cute.arch.lane_idx() & (cute.arch.WARP_SIZE - threads_in_group)
            mask = cutlass.Uint32(group_mask) << base_lane
        # Only forward the qualifiers when set: the dsl_user_op wrapper
        # rejects explicit None keyword values.
        kwargs = {}
        if const_expr(abs):
            kwargs["abs"] = True
        if const_expr(nan):
            kwargs["nan"] = True
        return cute.arch.warp_redux_sync(val, kind, mask, **kwargs)
    if const_expr(abs):
        val = cute.math.absf(val)
    return cute.arch.warp_reduction(val, op, threads_in_group=threads_in_group)


@cute.jit
def block_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: cute.Numeric = 0.0,
    dtype: cutlass.Constexpr = None,
) -> cute.Numeric:
    """reduction_buffer has shape (num_warps / warp_per_row, warps_per_row)"""
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op, dtype=dtype)


@cute.jit
def cluster_reduce(
    val: cute.Numeric | tuple,
    op: Callable,
    reduction_buffer: cute.Tensor | tuple,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: Optional[Int32] = None,
    dtype: cutlass.Constexpr = None,
) -> cute.Numeric | tuple:
    """Each reduction buffer has shape (num_warps / warps_per_row, (warps_per_row, cluster_n)).

    A tuple of values (with a matching tuple of buffers) reduces through a SINGLE
    transaction barrier: it is armed once with the combined byte count, every STAS
    credits it, and one wait covers all buffers — one cluster round trip instead of
    one per value. Sync slots are therefore decoupled from buffer slots.
    """
    is_multi = const_expr(isinstance(val, tuple))
    vals = val if const_expr(is_multi) else (val,)
    bufs = reduction_buffer if const_expr(is_multi) else (reduction_buffer,)
    assert len(vals) == len(bufs), "one reduction buffer per value"
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = bufs[0].shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            tx_bytes = sum(
                num_warps * cluster_n * buf.element_type.width // 8 for buf in bufs
            )
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tx_bytes)
    if lane_idx < cluster_n:
        for buf, v in zip(bufs, vals):
            utils.store_shared_remote(
                v,
                utils.elem_pointer(buf, (row_idx, (col_idx, cta_rank_in_cluster))),
                mbar_ptr,
                peer_cta_rank_in_cluster=lane_idx,
            )
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    results = []
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for buf in bufs:
        block_reduce_val = init_val
        for i in cutlass.range_constexpr(num_iter):
            idx = lane_idx + i * cute.arch.WARP_SIZE
            if idx < cute.size(buf, mode=[1]):
                block_reduce_val = op(block_reduce_val, buf[row_idx, idx])
        results.append(warp_reduce(block_reduce_val, op, dtype=dtype))
    return tuple(results) if const_expr(is_multi) else results[0]


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric | tuple,
    op: Callable,
    reduction_buffer: cute.Tensor | tuple,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    dtype: cutlass.Constexpr = None,
) -> cute.Numeric | tuple:
    """Perform either block or cluster reduction based on whether mbar_ptr is provided."""
    if const_expr(mbar_ptr is None):
        # No cross-CTA sync to share: reduce each value through its own buffer.
        if const_expr(isinstance(val, tuple)):
            return tuple(
                block_reduce(v, op, buf, init_val=init_val, dtype=dtype)
                for v, buf in zip(val, reduction_buffer)
            )
        return block_reduce(
            val,
            op,
            reduction_buffer,
            init_val=init_val,
            dtype=dtype,
        )
    else:
        return cluster_reduce(
            val,
            op,
            reduction_buffer,
            mbar_ptr,
            phase=phase,
            init_val=init_val,
            dtype=dtype,
        )


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric | tuple,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor | tuple] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
) -> cute.Numeric | tuple:
    """Each reduction buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n)).

    A tuple x (with a matching tuple of reduction buffers) reduces every value with
    the same op through a single mbarrier / cluster round trip (see cluster_reduce)
    and returns a tuple.
    """
    is_multi = const_expr(isinstance(x, tuple))
    xs = x if const_expr(is_multi) else (x,)
    vals = []
    for xi in xs:
        if const_expr(isinstance(xi, cute.TensorSSA)):
            val = xi.reduce(op, init_val=init_val, reduction_profile=0)
        else:
            val = xi
        # Scalar inputs (e.g. an ArithValue from a prior TensorSSA.reduce) carry no
        # .dtype; None makes warp_reduce fall back to its generic reduction.
        val_dtype = xi.dtype if const_expr(isinstance(xi, cute.TensorSSA)) else None
        warp_op = {
            cute.ReductionOp.ADD: operator.add,
            cute.ReductionOp.MAX: cute.arch.fmax
            if const_expr(val_dtype == Float32)
            else max,
            cute.ReductionOp.MIN: cute.arch.fmin
            if const_expr(val_dtype == Float32)
            else min,
            cute.ReductionOp.MUL: operator.mul,
        }[op]
        val = warp_reduce(
            val,
            warp_op,
            threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
            dtype=val_dtype,
        )
        vals.append(val)
        buf_stage_dtype = val_dtype
    if const_expr(hook_fn is not None):
        hook_fn()
    if const_expr(reduction_buffer is not None):
        bufs = reduction_buffer if const_expr(is_multi) else (reduction_buffer,)
        warps_per_row, cluster_n = bufs[0].shape[1]
        assert cluster_n == 1 or mbar_ptr is not None, (
            "mbar_ptr must be provided for cluster reduction"
        )
        if const_expr(warps_per_row > 1 or cluster_n > 1):
            # The combine reads the (single-dtype) buffers, so one op suffices; for
            # multi that dtype is the buffers', for single keep the value's dtype
            # (preserves the exact redux selection of the scalar path).
            if const_expr(is_multi):
                buf_stage_dtype = bufs[0].element_type
            reduced = block_or_cluster_reduce(
                tuple(vals) if const_expr(is_multi) else vals[0],
                warp_op,
                reduction_buffer,
                mbar_ptr,
                phase=phase,
                init_val=init_val,
                dtype=buf_stage_dtype,
            )
            vals = list(reduced) if const_expr(is_multi) else [reduced]
    return tuple(vals) if const_expr(is_multi) else vals[0]
