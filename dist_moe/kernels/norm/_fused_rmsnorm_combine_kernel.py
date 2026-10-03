# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL fused RMSNorm + weighted TOPK reduction kernels.

Forward:  y[b,d] = sum_k( x[b,k,d] / rms(x[b,k,:]) * w[b,k] )
Backward: grad_x[b,k,d], grad_w[b,k] via chain rule through the fused op.

Input shapes:
  x:       [B, TOPK, DIM]  (passed as [B*TOPK, DIM])
  weights: [B, TOPK]       (passed as [B*TOPK])
  y:       [B, DIM]

Forward TOPK rows are normalized and combined in one CTA. Backward loads the
shared output gradient once, then reuses scale_and_sum's scale-gradient dot
product for the RMSNorm projection. DIM is tiled across threads using
``tiled_copy`` with cp.async.

Supports optional Kahan-compensated sum-of-squares for improved numerical
accuracy on hardware where FP64 is slow (e.g. GB300).
"""

import math
import operator
from functools import partial
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr, Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

from .. import _dsl_compat as _cute_extern  # noqa: F401
from .._environment import is_blackwell_gpu
from .._quack import copy_utils
from .._quack.reduce import row_reduce
from .._quack.reduction_base import ReductionBase
from ._rmsnorm_common import _fake_div

# Exact backward reproduces the standalone 1024-column scale-gradient chunks.
# Its vector width must divide 1024 // threads_per_row, whose minimum is four
# across the supported 128- and 256-thread configurations. Pinning four also
# caps each vector copy at 16 bytes because `can_implement` rejects tensor
# element widths above 32 bits. `_compile_bwd` in `fused_rmsnorm_combine.py`
# derives fake-tensor divisibility from the same constant.
_EXACT_BWD_VECSIZE = 4


def _exact_bwd_vecsize(N: int) -> int:
    """Return the fixed-width exact-backward vector size for `N`."""
    return math.gcd(N, _EXACT_BWD_VECSIZE)


_BLACKWELL_REGISTERS_PER_SM = 65_536
_BLACKWELL_SHARED_BYTES_PER_SM = 228 * 1024
_REGISTER_ALLOCATION_GRANULARITY = 8
_REGISTER_OVERHEAD_ESTIMATE = 32
_REDUCTION_SCRATCH_BYTES_ESTIMATE = 1024


@dsl_user_op
def _mul_rn_f32(
    a: float | Float32,
    b: float | Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fma_rn_f32(
    a: float | Float32,
    b: float | Float32,
    c: float | Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _fma_dot_f32(x: cute.TensorSSA, y: cute.TensorSSA) -> Float32:
    acc = Float32(0.0)
    for i in cutlass.range(cute.size(x.shape), unroll_full=True):
        acc = _fma_rn_f32(x[i], y[i], acc)
    return acc


class FusedRMSNormCombineFwd(ReductionBase):
    """Forward kernel for fused RMSNorm + weighted TOPK reduction.

    Exact mode launches one block per batch element. Clustered mode launches
    ``cluster_n`` blocks per batch element, partitioned across DIM. Each batch
    element processes TOPK rows, computing per-row RMS normalization and
    weighted TOPK reduction into [DIM].

    Args:
        dtype: Element type of the input tensor (e.g. BFloat16).
        norm_dtype: Element type used for the normalized input.
        N: Feature dimension (DIM). Compile-time constant.
        TOPK: Number of expert rows per batch element. Compile-time constant.
        use_kahan: Use Kahan-compensated sum-of-squares instead of naive FP32.
        require_bitwise: Disable clustered reductions to preserve reduction order.
        tile_d: Match the tiled standalone scale-and-sum accumulation order.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        norm_dtype: Type[cutlass.Numeric],
        N: int,
        TOPK: int,
        use_kahan: bool = False,
        require_bitwise: bool = True,
        tile_d: bool = False,
        copy_input: bool = False,
        has_weight: bool = False,
    ):
        super().__init__(dtype, N, stage=2)
        # The input-scale gamma participates in the sum-of-squares, so it must
        # be applied before both reduction variants; only the plain FP32 path
        # carries it (its consumers are inference-only).
        assert not (has_weight and use_kahan), (
            "input-scale weight is not supported with Kahan summation"
        )
        assert not (has_weight and copy_input), (
            "input-scale weight is forward-only; the training input copy "
            "never coexists with it"
        )
        self.norm_dtype = norm_dtype
        self.TOPK = TOPK
        self.use_kahan = use_kahan
        self.require_bitwise = require_bitwise
        self.tile_d = tile_d
        self.copy_input = copy_input
        self.has_weight = has_weight
        self.cluster_n = 1
        capability = (
            torch.cuda.get_device_capability(torch.cuda.current_device())
            if torch.cuda.is_available()
            else (0, 0)
        )
        self._is_blackwell = 10 <= capability[0] < 12
        self._supports_clusters = capability[0] >= 9 and capability[0] != 12

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [
            (3072, 32),
            (6144, 64),
            (16384, 128),
        ]:
            if N <= limit:
                return threads
        return 256

    def _num_threads(self):
        return self._threads_per_row()

    def _load_stages(self, shard_n):
        threads_per_row = self._threads_per_row()
        elements_per_thread = cute.ceil_div(shard_n, threads_per_row)
        if (
            self.copy_input
            or shard_n <= 2304
            or threads_per_row < 64
            or elements_per_thread > 80
        ):
            return 1
        # Larger register fragments leave less shared-memory headroom for rows.
        return 4 if elements_per_thread <= 64 else 3

    def _min_blocks_per_mp(self):
        shard_n = cute.ceil_div(self.N, self.cluster_n)
        if shard_n <= 2304:
            return 1
        num_threads = self._num_threads()
        elements_per_thread = cute.ceil_div(shard_n, self._threads_per_row())
        # At D=4096 ptxas reports 96 registers/thread: 64 row elements plus
        # 32 registers of kernel state. Revisit this estimate when state changes.
        # The input-scale gamma keeps one 16-bit fragment resident per thread.
        weight_registers = elements_per_thread // 2 if self.has_weight else 0
        target_registers = (
            cute.ceil_div(
                elements_per_thread + weight_registers + _REGISTER_OVERHEAD_ESTIMATE,
                _REGISTER_ALLOCATION_GRANULARITY,
            )
            * _REGISTER_ALLOCATION_GRANULARITY
        )
        register_limited_blocks = _BLACKWELL_REGISTERS_PER_SM // (
            num_threads * target_registers
        )
        load_stages = self._load_stages(shard_n)
        shared_bytes_per_block = (
            load_stages * shard_n * self.dtype.width // 8
            + _REDUCTION_SCRATCH_BYTES_ESTIMATE
        )
        shared_limited_blocks = _BLACKWELL_SHARED_BYTES_PER_SM // shared_bytes_per_block
        return max(1, min(register_limited_blocks, shared_limited_blocks))

    def _set_cluster_n(self):
        if self.require_bitwise or not self._supports_clusters:
            self.cluster_n = 1
            return
        elements_per_thread = cute.ceil_div(self.N, self._threads_per_row())
        cluster_n = 1
        # The deeper 32-element clustering is measured for the wide D=12288
        # training shape; D=4096 inference measured ~240 us/iteration slower
        # with it, so narrower rows keep the original 64-element threshold.
        cluster_elements = 32 if self.N >= 12288 else 64
        while cute.ceil_div(elements_per_thread, cluster_n) > cluster_elements:
            cluster_n *= 2
        self.cluster_n = min(cluster_n, 16)

    @staticmethod
    def _is_16b_aligned(tensor: cute.Tensor, dim: int) -> bool:
        return (tensor.shape[dim] * tensor.element_type.width) % 128 == 0

    def can_implement(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mXCopy: cute.Tensor,
        mG: cute.Tensor | None = None,
    ) -> bool:
        if (mG is None) == self.has_weight:
            return False
        if mG is not None and not (
            mG.element_type == self.dtype
            and mG.shape[1] == self.N
            and self._is_16b_aligned(mG, dim=1)
        ):
            return False
        return (
            mX.element_type == self.dtype
            and mXCopy.element_type == self.dtype
            and mX.shape[1] == self.N
            and mXCopy.shape == mX.shape
            and mY.shape[1] == self.N
            and self._is_16b_aligned(mX, dim=1)
            and self._is_16b_aligned(mXCopy, dim=1)
            and self._is_16b_aligned(mY, dim=1)
            and self._vector_size(mY.element_type.width) % 2 == 0
            and mRstd.element_type == Float32
        )

    def _vector_size(self, output_dtype_width: int) -> int:
        largest_dtype_width = max(
            self.dtype.width,
            self.norm_dtype.width,
            output_dtype_width,
        )
        return math.gcd(self.N, 128 // largest_dtype_width)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mXCopy: cute.Tensor,
        mG: cute.Tensor | None,
        eps: Float32,
        gain_center: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert (mG is None) != self.has_weight
        self._set_cluster_n()
        vecsize = self._vector_size(mY.element_type.width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        load_stages = self._load_stages(tiler_mn[1])
        num_threads = tiled_copy.size
        B = mY.shape[0]
        self.kernel(
            mX,
            mW,
            mY,
            mRstd,
            mXCopy,
            mG,
            eps,
            gain_center,
            tiler_mn,
            tiled_copy,
            threads_per_row,
            load_stages,
        ).launch(
            grid=[B, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
            min_blocks_per_mp=self._min_blocks_per_mp(),
        )

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mRstd: cute.Tensor,
        mXCopy: cute.Tensor,
        mG: cute.Tensor | None,
        eps: Float32,
        gain_center: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        load_stages: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )
        TOPK = const_expr(self.TOPK)
        tv_layout = tiled_copy.layout_tv_tiled
        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        smem = cutlass.utils.SmemAllocator()
        sX0 = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        use_pipelined_load = const_expr(load_stages > 1)
        use_four_stage_pipeline = const_expr(load_stages == 4)
        if const_expr(use_pipelined_load):
            sX1 = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sX2 = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            if const_expr(use_four_stage_pipeline):
                sX3 = smem.allocate_tensor(
                    mX.element_type,
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                    byte_alignment=16,
                )
            else:
                sX3 = sX0
        else:
            sX1 = sX0
            sX2 = sX0
            sX3 = sX0
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )
        use_pairwise_init = const_expr(
            mY.element_type.width == 16 and self.norm_dtype.width == 32 and TOPK > 1
        )
        sW = smem.allocate_tensor(Float32, cute.make_layout(TOPK), byte_alignment=4)

        DIM = mX.shape[1]
        is_even_N = const_expr(DIM == tiler_mn[1] * self.cluster_n)
        thr_copy = tiled_copy.get_slice(tidx)
        self._initialize_cluster(tidx, mbar_ptr, num_warps)
        tXsX0 = thr_copy.partition_D(sX0)
        tXsX1 = thr_copy.partition_D(sX1)
        tXsX2 = thr_copy.partition_D(sX2)
        tXsX3 = thr_copy.partition_D(sX3)

        topk_tiler_mn = (TOPK, tiler_mn[1])
        gX = cute.local_tile(mX, topk_tiler_mn, (bidx, cluster_y))
        gXCopy = cute.local_tile(mXCopy, topk_tiler_mn, (bidx, cluster_y))
        gW = cute.local_tile(mW, (TOPK,), (bidx,))
        gRstd = cute.local_tile(mRstd, (TOPK,), (bidx,))
        idX = cute.make_identity_tensor(mX.shape)
        cX = cute.local_tile(idX, topk_tiler_mn, (bidx, cluster_y))
        gX_0 = cute.local_tile(gX, tiler_mn, (0, 0))
        cX_0 = cute.local_tile(cX, tiler_mn, (0, 0))
        tXcX_full = thr_copy.partition_S(cX_0)
        tXcX_0 = tXcX_full[(0, None), None, None]
        tXpX = copy_utils.predicate_k(tXcX_full, limit=DIM) if not is_even_N else None
        copy = partial(copy_utils.copy, pred=tXpX)
        use_direct_gmem_load = const_expr(tiler_mn[1] <= 2304)

        tXgX_0 = thr_copy.partition_S(gX_0)
        tXrX = cute.make_fragment_like(tXgX_0)
        tXrAcc = cute.make_fragment_like(tXgX_0, Float32)
        assert cute.size(tXrAcc) % 2 == 0
        tXrAcc.fill(0.0)

        if const_expr(self.has_weight):
            # Input-scale gamma: one [DIM]-shard fragment, loaded once and
            # shared by every TOPK row. The gain center is folded here with
            # torch's `weight + gain_center` semantics (fp32 add, one RN round
            # back to the input dtype), so the per-row multiplies below see the
            # same scaled weight the standalone composition materializes.
            gG = cute.local_tile(mG, tiler_mn, (0, cluster_y))
            tXgG = thr_copy.partition_S(gG)
            tXrG = cute.make_fragment_like(tXgG)
            if not is_even_N:
                tXrG.fill(0.0)
            copy(tXgG, tXrG)
            tXrG.store((tXrG.load().to(Float32) + gain_center).to(mX.element_type))

        if tidx < TOPK:
            sW[tidx] = gW[tidx]
        cute.arch.barrier()

        if const_expr(use_pipelined_load):
            first_k = const_expr(1 if TOPK > 1 else 0)
            gX_first = cute.local_tile(gX, tiler_mn, (first_k, 0))
            tXgX_first = thr_copy.partition_S(gX_first)
            copy(tXgX_first, tXsX0, is_async=True)
            cute.arch.cp_async_commit_group()
            if const_expr(TOPK > 1):
                gX_second = cute.local_tile(gX, tiler_mn, (0, 0))
                tXgX_second = thr_copy.partition_S(gX_second)
                copy(tXgX_second, tXsX1, is_async=True)
                cute.arch.cp_async_commit_group()
            if const_expr(use_four_stage_pipeline and TOPK > 2):
                gX_third = cute.local_tile(gX, tiler_mn, (2, 0))
                tXgX_third = thr_copy.partition_S(gX_third)
                copy(tXgX_third, tXsX2, is_async=True)
                cute.arch.cp_async_commit_group()

        for reduction_i in cutlass.range_constexpr(TOPK):
            if const_expr(TOPK > 1 and reduction_i == 0):
                k = const_expr(1)
            elif const_expr(TOPK > 1 and reduction_i == 1):
                k = const_expr(0)
            else:
                k = reduction_i

            gX_k = cute.local_tile(gX, tiler_mn, (k, 0))
            tXgX_k = thr_copy.partition_S(gX_k)
            if const_expr(use_direct_gmem_load):
                if not is_even_N:
                    tXrX.fill(0.0)
                copy(tXgX_k, tXrX)
            elif const_expr(use_pipelined_load):
                if const_expr(use_four_stage_pipeline):
                    cute.arch.cp_async_wait_group(min(2, TOPK - reduction_i - 1))
                    tXsX = (
                        tXsX0
                        if const_expr(reduction_i % 4 == 0)
                        else tXsX1
                        if const_expr(reduction_i % 4 == 1)
                        else tXsX2
                        if const_expr(reduction_i % 4 == 2)
                        else tXsX3
                    )
                    if const_expr(reduction_i + 3 < TOPK):
                        next_k = const_expr(reduction_i + 3)
                        gX_next = cute.local_tile(gX, tiler_mn, (next_k, 0))
                        tXgX_next = thr_copy.partition_S(gX_next)
                        tXsX_next = (
                            tXsX0
                            if const_expr((reduction_i + 3) % 4 == 0)
                            else tXsX1
                            if const_expr((reduction_i + 3) % 4 == 1)
                            else tXsX2
                            if const_expr((reduction_i + 3) % 4 == 2)
                            else tXsX3
                        )
                        copy(tXgX_next, tXsX_next, is_async=True)
                        cute.arch.cp_async_commit_group()
                else:
                    cute.arch.cp_async_wait_group(min(1, TOPK - reduction_i - 1))
                    tXsX = (
                        tXsX0
                        if const_expr(reduction_i % 3 == 0)
                        else tXsX1
                        if const_expr(reduction_i % 3 == 1)
                        else tXsX2
                    )
                    if const_expr(reduction_i + 2 < TOPK):
                        next_k = const_expr(reduction_i + 2)
                        gX_next = cute.local_tile(gX, tiler_mn, (next_k, 0))
                        tXgX_next = thr_copy.partition_S(gX_next)
                        tXsX_next = (
                            tXsX0
                            if const_expr((reduction_i + 2) % 3 == 0)
                            else tXsX1
                            if const_expr((reduction_i + 2) % 3 == 1)
                            else tXsX2
                        )
                        copy(tXgX_next, tXsX_next, is_async=True)
                        cute.arch.cp_async_commit_group()
                cute.autovec_copy(tXsX, tXrX)
            else:
                tXsX = tXsX0
                copy(tXgX_k, tXsX, is_async=True)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.autovec_copy(tXsX, tXrX)

            if const_expr(self.copy_input):
                gXCopy_k = cute.local_tile(gXCopy, tiler_mn, (k, 0))
                tXgXCopy_k = thr_copy.partition_D(gXCopy_k)
                copy(tXrX, tXgXCopy_k)

            reduction_stage = const_expr(reduction_i % self.stage)
            if const_expr(self.use_kahan):
                num_elems = const_expr(cute.size(tXrX))
                kahan_sum = Float32(0.0)
                kahan_comp = Float32(0.0)
                for i in cutlass.range_constexpr(num_elems):
                    x_val = tXrX[i].to(Float32)
                    x_sq = x_val * x_val
                    y_val = x_sq + kahan_comp
                    t_val = kahan_sum + y_val
                    kahan_comp = y_val - (t_val - kahan_sum)
                    kahan_sum = t_val
                sum_sq = row_reduce(
                    kahan_sum,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, reduction_stage],
                    mbar_ptr + reduction_stage
                    if const_expr(self.cluster_n > 1)
                    else mbar_ptr,
                    phase=const_expr((reduction_i // self.stage) % 2),
                    init_val=Float32(0.0),
                    hook_fn=cute.arch.cluster_wait
                    if const_expr(self.cluster_n > 1)
                    else None,
                )
            else:
                x = tXrX.load().to(Float32)
                if const_expr(self.has_weight):
                    # torch semantics of `w_scaled * x`: fp32 multiply, one RN
                    # round to the input dtype. This is the input-scaled row
                    # the standalone composition feeds its sum-of-squares.
                    x = (x * tXrG.load().to(Float32)).to(mX.element_type).to(Float32)
                # Preserve Hopper's pre-optimization reduction order.
                local_sum_sq = (
                    _fma_dot_f32(x, x) if const_expr(self._is_blackwell) else x * x
                )
                sum_sq = row_reduce(
                    local_sum_sq,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, reduction_stage],
                    mbar_ptr + reduction_stage
                    if const_expr(self.cluster_n > 1)
                    else mbar_ptr,
                    phase=const_expr((reduction_i // self.stage) % 2),
                    init_val=0.0,
                    hook_fn=cute.arch.cluster_wait
                    if const_expr(self.cluster_n > 1)
                    else None,
                )

            rstd = cute.math.rsqrt(sum_sq / DIM + eps, fastmath=False)
            if tidx == 0 and cluster_y == 0:
                gRstd[k] = rstd

            if const_expr(not use_direct_gmem_load):
                cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)
            if const_expr(self.has_weight):
                # Same rounding as the sum-of-squares scaling above, so the
                # normalized value is the scaled row divided by its own rms.
                x = (x * tXrG.load().to(Float32)).to(mX.element_type).to(Float32)
            x_norm = (x * rstd).to(self.norm_dtype).to(Float32)
            if const_expr(use_pairwise_init and reduction_i == 0):
                tXrAcc.store(x_norm)
            elif const_expr(use_pairwise_init and reduction_i == 1):
                w0 = sW[0]
                w1 = sW[1]
                if const_expr(self._is_blackwell):
                    # The even vector size makes fragment index parity match column
                    # parity and guarantees every element participates in a pair.
                    for i in cutlass.range_constexpr(0, cute.size(tXrAcc), 2):
                        x0 = (x_norm[i], x_norm[i + 1])
                        x1 = (tXrAcc[i], tXrAcc[i + 1])
                        x1w1 = cute.arch.mul_packed_f32x2(x1, (w1, w1), rnd="rn")
                        if const_expr(self.tile_d or self.N <= 1024):
                            acc = cute.arch.fma_packed_f32x2(
                                x0, (w0, w0), x1w1, rnd="rn"
                            )
                        else:
                            x0w0 = cute.arch.mul_packed_f32x2(x0, (w0, w0), rnd="rn")
                            acc = cute.arch.fma_packed_f32x2(
                                (x1[0], x0[1]),
                                (w1, w0),
                                (x0w0[0], x1w1[1]),
                                rnd="rn",
                            )
                        tXrAcc[i] = acc[0]
                        tXrAcc[i + 1] = acc[1]
                else:
                    for i in cutlass.range_constexpr(cute.size(tXrAcc)):
                        x0w0 = _mul_rn_f32(x_norm[i], w0)
                        x1w1 = _mul_rn_f32(tXrAcc[i], w1)
                        if (
                            const_expr(self.tile_d or self.N <= 1024)
                            or tXcX_full[i][1] % 2 != 0
                        ):
                            tXrAcc[i] = _fma_rn_f32(x_norm[i], w0, x1w1)
                        else:
                            tXrAcc[i] = _fma_rn_f32(tXrAcc[i], w1, x0w0)
            elif const_expr(use_pairwise_init):
                w_k = sW[k]
                if const_expr(self._is_blackwell):
                    for i in cutlass.range_constexpr(0, cute.size(tXrAcc), 2):
                        acc = cute.arch.fma_packed_f32x2(
                            (x_norm[i], x_norm[i + 1]),
                            (w_k, w_k),
                            (tXrAcc[i], tXrAcc[i + 1]),
                            rnd="rn",
                        )
                        tXrAcc[i] = acc[0]
                        tXrAcc[i + 1] = acc[1]
                else:
                    for i in cutlass.range_constexpr(cute.size(tXrAcc)):
                        tXrAcc[i] = _fma_rn_f32(x_norm[i], w_k, tXrAcc[i])
            else:
                tXrAcc.store(tXrAcc.load() + x_norm * sW[k])

        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        tXgY = thr_copy.partition_D(gY)
        row = tXcX_0[0][0]
        if row < mX.shape[0]:
            copy_utils.cvt_copy(tiled_copy, tXrAcc, tXgY, pred=tXpX, retile=True)


class FusedRMSNormCombineBwdExact(ReductionBase):
    """Bitwise-exact backward matching the two standalone kernels.

    The scale gradient follows Triton's 1024-column chunking and four-warp
    reduction. The input gradient independently follows the standalone
    RMSNorm row-reduction order.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        grad_y_dtype: Type[cutlass.Numeric],
        N: int,
        TOPK: int,
    ):
        super().__init__(dtype, N, stage=1)
        self.grad_y_dtype = grad_y_dtype
        self.TOPK = TOPK
        self.cluster_n = 1
        self._is_blackwell = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 10
        )
        if not self._is_blackwell and N > 4096:
            self._num_threads_val = 256
        else:
            self._num_threads_val = 128

    def _threads_per_row(self):
        return self._num_threads_val

    def _num_threads(self):
        return self._num_threads_val

    @staticmethod
    def _is_16b_aligned(tensor: cute.Tensor, dim: int) -> bool:
        return (tensor.shape[dim] * tensor.element_type.width) % 128 == 0

    def can_implement(
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
    ) -> bool:
        return (
            mX.element_type == self.dtype
            and mGradY.element_type == self.grad_y_dtype
            and mX.element_type.width <= 32
            and mGradY.element_type.width <= 32
            and mGradX.element_type.width <= 32
            and mX.shape[1] == self.N
            and mGradY.shape[1] == self.N
            and mGradX.shape[1] == self.N
            and self._is_16b_aligned(mGradY, dim=1)
            and self._is_16b_aligned(mX, dim=1)
            and self._is_16b_aligned(mGradX, dim=1)
            and mRstd.element_type == Float32
        )

    @cute.jit
    def __call__(
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        vecsize = _exact_bwd_vecsize(self.N)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        B = mGradY.shape[0]
        self.kernel(
            mGradY,
            mX,
            mW,
            mRstd,
            mGradX,
            mGradW,
            tiler_mn,
            tiled_copy,
            threads_per_row,
            vecsize,
        ).launch(
            grid=[B, self.TOPK, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        vecsize: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, kidx, _ = cute.arch.block_idx()
        TOPK = const_expr(self.TOPK)
        dscale_block_d = const_expr(1024)
        dscale_elems_per_thread = const_expr(dscale_block_d // threads_per_row)
        dscale_chunks = const_expr(cute.ceil_div(self.N, dscale_block_d))

        tv_layout = tiled_copy.layout_tv_tiled
        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        grad_w_num_warps = const_expr(4)
        smem = cutlass.utils.SmemAllocator()

        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )
        sGradWByWarp = smem.allocate_tensor(
            Float32,
            cute.make_layout((dscale_chunks, grad_w_num_warps)),
            byte_alignment=4,
        )
        if const_expr(threads_per_row == 256):
            sGradYHigh = smem.allocate_tensor(
                Float32,
                cute.make_layout((dscale_chunks, 128, dscale_elems_per_thread)),
                byte_alignment=4,
            )
            sXNormHigh = smem.allocate_tensor(
                Float32,
                cute.make_layout((dscale_chunks, 128, dscale_elems_per_thread)),
                byte_alignment=4,
            )
        else:
            sGradYHigh = None
            sXNormHigh = None

        DIM = mX.shape[1]
        is_even_N = const_expr(DIM == tiler_mn[1])

        thr_copy = tiled_copy.get_slice(tidx)
        self._initialize_cluster(tidx, mbar_ptr, num_warps)
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        topk_tiler_mn = (TOPK, tiler_mn[1])
        gX = cute.local_tile(mX, topk_tiler_mn, (bidx, 0))
        gRstd = cute.local_tile(mRstd, (TOPK,), (bidx,))
        gW = cute.local_tile(mW, (TOPK,), (bidx,))
        gGradW = cute.local_tile(mGradW, (TOPK,), (bidx,))
        gGradX = cute.local_tile(mGradX, topk_tiler_mn, (bidx, 0))
        gX_k = cute.local_tile(gX, tiler_mn, (kidx, 0))
        gGradX_k = cute.local_tile(gGradX, tiler_mn, (kidx, 0))

        idX = cute.make_identity_tensor(mX.shape)
        cX = cute.local_tile(idX, topk_tiler_mn, (bidx, 0))
        cX_0 = cute.local_tile(cX, tiler_mn, (0, 0))
        tXpX = (
            copy_utils.predicate_k(thr_copy.partition_S(cX_0), limit=DIM)
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        tXgX_k = thr_copy.partition_S(gX_k)
        tXrX = cute.make_fragment_like(tXgX_k)
        gGradY = cute.local_tile(mGradY, tiler_mn, (bidx, 0))
        tXgGradY = thr_copy.partition_S(gGradY)
        tXrGradY = cute.make_fragment_like(tXgGradY)
        tXrWdy = tXrGradY
        if const_expr(mGradY.element_type != Float32):
            tXrWdy = cute.make_fragment_like(tXrGradY, Float32)
        if not is_even_N:
            tXrGradY.fill(0.0)
            tXrX.fill(0.0)
        copy(tXgGradY, tXrGradY)
        copy(tXgX_k, tXrX)
        rstd_k = gRstd[kidx]
        w_k = gW[kidx]
        grad_y = tXrGradY.load().to(Float32)
        x = tXrX.load().to(Float32)
        x_norm = x * rstd_k

        frag_elems = const_expr(cute.size(tXrX))
        for chunk_idx in cutlass.range_constexpr(dscale_chunks):
            chunk_frag = const_expr(chunk_idx * dscale_elems_per_thread)
            grad_w_thread = Float32(0.0)

            if const_expr(threads_per_row == 256):
                assert sGradYHigh is not None
                assert sXNormHigh is not None
                for offset in cutlass.range_constexpr(dscale_elems_per_thread):
                    frag_idx = const_expr(chunk_frag + offset)
                    high_grad_y = Float32(0.0)
                    high_x_norm = Float32(0.0)
                    if const_expr(frag_idx < frag_elems):
                        high_grad_y = grad_y[frag_idx]
                        high_x_norm = x_norm[frag_idx]
                    if tidx >= 128:
                        high_tid = tidx - 128
                        sGradYHigh[chunk_idx, high_tid, offset] = high_grad_y
                        sXNormHigh[chunk_idx, high_tid, offset] = high_x_norm
                cute.arch.barrier()

            first_idx = const_expr(chunk_frag + 1)
            if const_expr(first_idx < frag_elems):
                grad_w_thread = _mul_rn_f32(grad_y[first_idx], x_norm[first_idx])

            second_idx = const_expr(chunk_frag)
            if const_expr(second_idx < frag_elems):
                grad_w_thread = _fma_rn_f32(
                    grad_y[second_idx], x_norm[second_idx], grad_w_thread
                )

            for offset in cutlass.range_constexpr(2, dscale_elems_per_thread):
                frag_idx = const_expr(chunk_frag + offset)
                if const_expr(frag_idx < frag_elems):
                    grad_w_thread = _fma_rn_f32(
                        grad_y[frag_idx], x_norm[frag_idx], grad_w_thread
                    )

            if const_expr(threads_per_row == 256):
                assert sGradYHigh is not None
                assert sXNormHigh is not None
                if tidx < 128:
                    for offset in cutlass.range_constexpr(dscale_elems_per_thread):
                        grad_w_thread = _fma_rn_f32(
                            sGradYHigh[chunk_idx, tidx, offset],
                            sXNormHigh[chunk_idx, tidx, offset],
                            grad_w_thread,
                        )

            grad_w_warp = cute.arch.warp_reduction(grad_w_thread, operator.add)
            if lane_idx == 0 and warp_idx < grad_w_num_warps:
                sGradWByWarp[chunk_idx, warp_idx] = grad_w_warp

        for i in cutlass.range_constexpr(cute.size(tXrWdy)):
            tXrWdy[i] = _mul_rn_f32(grad_y[i], w_k)
        wdy = tXrWdy.load()

        mean_xhat_wdy_thread = Float32(0.0)
        for i in cutlass.range_constexpr(cute.size(tXrX)):
            mean_xhat_wdy_thread = _fma_rn_f32(x_norm[i], wdy[i], mean_xhat_wdy_thread)
        mean_xhat_wdy = (
            row_reduce(
                mean_xhat_wdy_thread,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=0.0,
            )
            / DIM
        )

        grad_w_k = Float32(0.0)
        for chunk_idx in cutlass.range_constexpr(dscale_chunks):
            grad_w_chunk = Float32(0.0)
            if warp_idx == 0 and lane_idx < grad_w_num_warps:
                grad_w_chunk = sGradWByWarp[chunk_idx, lane_idx]
            grad_w_chunk = cute.arch.warp_reduction(
                grad_w_chunk,
                operator.add,
                threads_in_group=grad_w_num_warps,
            )
            if tidx == 0:
                grad_w_k += grad_w_chunk
        if tidx == 0:
            gGradW[kidx] = grad_w_k

        grad_x = (wdy - x_norm * mean_xhat_wdy) * rstd_k
        tXgGradX_k = thr_copy.partition_D(gGradX_k)
        tXrGradX = cute.make_fragment_like(tXgGradX_k)
        tXrGradX.store(grad_x.to(tXrGradX.element_type))
        copy(tXrGradX, tXgGradX_k)


class FusedRMSNormCombineBwd(ReductionBase):
    """Backward kernel for fused RMSNorm + weighted TOPK reduction.

    Given grad_y [B, DIM], computes:
      grad_x [B*TOPK, DIM] — gradient w.r.t. input
      grad_w [B*TOPK]      — gradient w.r.t. weights

    Math:
      x_norm[k] = x[k] * rstd[k]
      grad_w[k] = sum_d(grad_y[d] * x_norm[k,d])
      mean[k] = grad_w[k] * w[k] / DIM
      grad_x[k,d] = (grad_y[d] * w[k] - x_norm[k,d] * mean[k]) * rstd[k]
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        grad_y_dtype: Type[cutlass.Numeric],
        N: int,
        TOPK: int,
    ):
        super().__init__(dtype, N, stage=1)
        self.grad_y_dtype = grad_y_dtype
        self.TOPK = TOPK
        self.cluster_n = 1
        self._num_threads_val = 256 if is_blackwell_gpu() else 128

    def _threads_per_row(self):
        return self._num_threads_val

    def _num_threads(self):
        return self._num_threads_val

    @staticmethod
    def _is_16b_aligned(tensor: cute.Tensor, dim: int) -> bool:
        return (tensor.shape[dim] * tensor.element_type.width) % 128 == 0

    def can_implement(
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
    ) -> bool:
        return (
            mX.element_type == self.dtype
            and mGradY.element_type == self.grad_y_dtype
            and mX.shape[1] == self.N
            and mGradY.shape[1] == self.N
            and mGradX.shape[1] == self.N
            and self._is_16b_aligned(mGradY, dim=1)
            and self._is_16b_aligned(mX, dim=1)
            and self._is_16b_aligned(mGradX, dim=1)
            and mRstd.element_type == Float32
        )

    @cute.jit
    def __call__(
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        vecsize = const_expr(
            _fake_div(
                self.N,
                (mX.element_type, mGradY.element_type, mGradX.element_type),
            )
        )
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        B = mGradY.shape[0]
        self.kernel(
            mGradY,
            mX,
            mW,
            mRstd,
            mGradX,
            mGradW,
            tiler_mn,
            tiled_copy,
            threads_per_row,
            vecsize,
        ).launch(
            grid=[B, 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        mGradY: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mRstd: cute.Tensor,
        mGradX: cute.Tensor,
        mGradW: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        vecsize: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        TOPK = const_expr(self.TOPK)
        grad_w_num_warps = const_expr(cute.size(tiled_copy) // cute.arch.WARP_SIZE)
        smem = cutlass.utils.SmemAllocator()

        sGradY = smem.allocate_tensor(
            mGradY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        sGradWByWarp = smem.allocate_tensor(
            Float32,
            cute.make_layout((grad_w_num_warps,)),
            byte_alignment=4,
        )
        sGradW = smem.allocate_tensor(
            Float32,
            cute.make_layout((1,)),
            byte_alignment=4,
        )

        DIM = mX.shape[1]
        is_even_N = const_expr(DIM == tiler_mn[1])

        thr_copy = tiled_copy.get_slice(tidx)
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        # Carve this CTA's TOPK expert rows once, then slice by local k block.
        topk_tiler_mn = (TOPK, tiler_mn[1])
        gX = cute.local_tile(mX, topk_tiler_mn, (bidx, 0))
        gRstd = cute.local_tile(mRstd, (TOPK,), (bidx,))
        gW = cute.local_tile(mW, (TOPK,), (bidx,))
        gGradW = cute.local_tile(mGradW, (TOPK,), (bidx,))
        gGradX = cute.local_tile(mGradX, topk_tiler_mn, (bidx, 0))

        # Predication (shared for all rows — same DIM)
        idX = cute.make_identity_tensor(mX.shape)
        cX = cute.local_tile(idX, topk_tiler_mn, (bidx, 0))
        cX_0 = cute.local_tile(cX, tiler_mn, (0, 0))
        tXpX = (
            copy_utils.predicate_k(thr_copy.partition_S(cX_0), limit=DIM)
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        gX_0 = cute.local_tile(gX, tiler_mn, (0, 0))
        tXrX = cute.make_fragment_like(thr_copy.partition_S(gX_0))
        gGradY = cute.local_tile(mGradY, tiler_mn, (bidx, 0))
        tXgGradY = thr_copy.partition_S(gGradY)
        tXsGradY = thr_copy.partition_D(sGradY)
        # Predicated `CopyG2SOp` zero-fills out-of-bounds vector atoms.
        copy(tXgGradY, tXsGradY, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        tXrGradY = cute.make_fragment_like(tXsGradY)
        tXrWdy = tXrGradY
        if const_expr(mGradY.element_type != Float32):
            tXrWdy = cute.make_fragment_like(tXrGradY, Float32)

        for kidx in cutlass.range(TOPK, unroll=1):
            gX_k = cute.local_tile(gX, tiler_mn, (kidx, 0))
            gGradX_k = cute.local_tile(gGradX, tiler_mn, (kidx, 0))
            tXgX_k = thr_copy.partition_S(gX_k)
            if not is_even_N:
                tXrX.fill(0.0)
            copy(tXgX_k, tXrX)
            rstd_k = gRstd[kidx]
            w_k = gW[kidx]
            x_norm = tXrX.load().to(Float32) * rstd_k
            cute.autovec_copy(tXsGradY, tXrGradY)
            grad_y = tXrGradY.load().to(Float32)

            grad_w_thread = (grad_y * x_norm).reduce(
                cute.ReductionOp.ADD,
                init_val=Float32(0.0),
                reduction_profile=0,
            )
            grad_w_warp = cute.arch.warp_reduction(grad_w_thread, operator.add)
            if lane_idx == 0:
                sGradWByWarp[warp_idx] = grad_w_warp

            cute.arch.barrier()

            grad_w_k = Float32(0.0)
            if warp_idx == 0 and lane_idx < grad_w_num_warps:
                grad_w_k = sGradWByWarp[lane_idx]
            grad_w_k = cute.arch.warp_reduction(
                grad_w_k,
                operator.add,
                threads_in_group=grad_w_num_warps,
            )
            if tidx == 0:
                gGradW[kidx] = grad_w_k
                sGradW[0] = grad_w_k
            cute.arch.barrier()

            for i in cutlass.range(cute.size(tXrWdy), unroll_full=True):
                tXrWdy[i] = _mul_rn_f32(grad_y[i], w_k)
            wdy = tXrWdy.load()
            mean_xhat_wdy = _mul_rn_f32(sGradW[0], w_k) / DIM
            tXgGradX_k = thr_copy.partition_D(gGradX_k)
            grad_x = (wdy - x_norm * mean_xhat_wdy) * rstd_k
            tXrX.store(grad_x.to(tXrX.element_type))
            copy(tXrX, tXgGradX_k)
