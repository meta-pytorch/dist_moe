# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused deterministic dw finalize for the CuTe DSL RMSNorm backward."""

import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import const_expr, Float32, Int32

from .. import _dsl_compat as _cute_extern  # noqa: F401
from .._jit_cache import jit_cache
from .._quack_compile_utils import make_fake_stream, make_fake_tensor as fake_tensor


class FusedDwReduce:
    """Column-reduce FP32 dw partials and deterministically downcast in one kernel.

    Parallelism: each lane owns a ``cols_per_lane``-column strip and the block's
    ``NUM_WARPS`` warps split the partial rows into contiguous chunks (a
    per-lane serial P loop measured up to 2.9x slower than the torch
    composite it replaces — N/cols_per_lane lanes alone cannot cover the load
    latency). Warp w accumulates rows ``[w * Pc, (w + 1) * Pc)`` in
    ascending order and the chunk sums combine through SMEM in fixed warp
    order, so dw is bitwise deterministic for a fixed partial count.
    Accumulation stays fp32 end to end, matching the triton backend's dw
    kernel; against an fp64 reference the fixed-order sum sits within one
    rounding of the ``u * sum|partial|`` conditioning bound on every
    registry shape — the same noise envelope as the torch pairwise sum it
    replaces. Adjacent lanes hold adjacent strips, so every row visit is
    one coalesced 32-lane transaction.
    """

    NUM_WARPS = 8
    STRIPS_PER_CTA = 32
    NUM_THREADS = NUM_WARPS * 32

    def __init__(
        self,
        N: int,
        out_dtype: Type[cutlass.Numeric],
    ):
        self.N = N
        self.out_dtype = out_dtype
        # Widest per-lane column strip that divides N: vector loads stay
        # aligned and there is no ragged tail to predicate.
        self.cols_per_lane = math.gcd(N, 8)

    @classmethod
    @jit_cache
    def compile(cls, N, out_dtype):
        """Compile and cache a launch-specialized dw-reduce variant."""
        batch_sym = cute.sym_int()
        kernel = cls(N, out_dtype)
        partial_cute = fake_tensor(Float32, (batch_sym, N), kernel.cols_per_lane)
        out_cute = fake_tensor(out_dtype, (N,), kernel.cols_per_lane)
        return cute.compile(
            kernel,
            partial_cute,
            out_cute,
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )

    @cute.jit
    def __call__(
        self,
        mDwPartial: cute.Tensor,  # (P, N) fp32
        mDw: cute.Tensor,  # (N,) out_dtype
        stream: cuda.CUstream,
    ):
        num_strips = self.N // self.cols_per_lane
        grid = (num_strips + self.STRIPS_PER_CTA - 1) // self.STRIPS_PER_CTA
        self.kernel(mDwPartial, mDw).launch(
            grid=[grid, 1, 1], block=[self.NUM_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mDwPartial: cute.Tensor,
        mDw: cute.Tensor,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bidx, _, _ = cute.arch.block_idx()
        strip = bidx * self.STRIPS_PER_CTA + lane
        in_range = strip < const_expr(self.N // self.cols_per_lane)

        smem = cutlass.utils.SmemAllocator()
        sAcc = smem.allocate_tensor(
            Float32,
            cute.make_ordered_layout(
                (self.NUM_WARPS, self.STRIPS_PER_CTA, self.cols_per_lane),
                order=(2, 1, 0),
            ),
            byte_alignment=16,
        )

        frag = cute.make_rmem_tensor(self.cols_per_lane, Float32)
        acc = cute.make_rmem_tensor(self.cols_per_lane, Float32)
        acc.fill(0.0)
        num_partials = Int32(mDwPartial.shape[0])
        rows_per_warp = (num_partials + Int32(self.NUM_WARPS - 1)) // Int32(
            self.NUM_WARPS
        )
        p = warp * rows_per_warp
        p_end = p + rows_per_warp
        if p_end > num_partials:
            p_end = num_partials
        if in_range:
            # (cols_per_lane, P) view of this lane's column strip; adjacent lanes own
            # adjacent strips, so each row visit coalesces across the warp.
            gW = cute.local_tile(mDwPartial, (1, self.cols_per_lane), (None, strip))[
                0, None, None
            ]
            while p < p_end:
                cute.autovec_copy(gW[None, p], frag)
                acc.store(acc.load() + frag.load())
                p += Int32(1)
        cute.autovec_copy(acc, sAcc[warp, lane, None])
        cute.arch.barrier()

        if warp == 0:
            total = cute.make_rmem_tensor(self.cols_per_lane, Float32)
            self._fold_chunk_sums(sAcc, lane, total)
            if in_range:
                out_frag = cute.make_rmem_tensor(self.cols_per_lane, self.out_dtype)
                out_frag.store(total.load().to(self.out_dtype))
                gOut = cute.make_tensor(
                    mDw.iterator + strip * self.cols_per_lane,
                    cute.make_layout(self.cols_per_lane),
                )
                cute.autovec_copy(out_frag, gOut)

    @cute.jit
    def _fold_chunk_sums(self, sAcc: cute.Tensor, lane: Int32, total: cute.Tensor):
        """Fold the warps' fp32 chunk sums out of SMEM into ``total``.

        The fixed ascending-warp order keeps the reduction bitwise
        deterministic for a fixed partial count.
        """
        frag = cute.make_rmem_tensor(self.cols_per_lane, Float32)
        total.fill(0.0)
        for w in cutlass.range_constexpr(self.NUM_WARPS):
            cute.autovec_copy(sAcc[w, lane, None], frag)
            total.store(total.load() + frag.load())
