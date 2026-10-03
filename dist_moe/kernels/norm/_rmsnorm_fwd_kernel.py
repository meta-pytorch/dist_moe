# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL RMSNorm forward kernels with fused gain_center.

Contains the forked forward kernel classes:
  - FusedRMSNormFwd: FP64/FP32 sum-of-squares reduction and fused gain_center.
  - RMSNormKahan: Kahan-compensated per-thread sum-of-squares for hardware
    where FP64 is slow (e.g. GB300 with 1/64 FP64 throughput).
"""

import math
from functools import partial
from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr, Float32, Float64

from .. import _dsl_compat as _cute_extern  # noqa: F401
from .._jit_cache import jit_cache
from .._quack import copy_utils, layout_utils
from .._quack.reduce import row_reduce
from .._quack.reduction_base import ReductionBase
from .._quack.rmsnorm_config import RmsNormFwdConfig
from .._quack_compile_utils import make_fake_stream, make_fake_tensor as fake_tensor
from ._rmsnorm_common import _fma_dot_f32


def _current_arch_major() -> Optional[int]:
    """Major CC of the current device; None defers to quack's GPU-blind
    QUACK_ARCH-aware fallback (compile-pool workers, CPU-only imports)."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(torch.cuda.current_device())[0]


# ---------------------------------------------------------------------------
# Forked forward kernel: FusedRMSNormFwd
# ---------------------------------------------------------------------------
# Extends Quack's ReductionBase with the same tiled_copy, cluster, and smem
# staging, but overrides the sum-of-squares reduction to use FP64 accumulation
# and adds a dynamic gain_center argument for fused zero-centered gamma.


class FusedRMSNormFwd(ReductionBase):
    """RMSNorm forward with fused gain_center and configurable reduction precision.

    Reuses ReductionBase for tiled_copy, cluster, and smem infrastructure.
    The reduction accumulates x*x in the specified reduction dtype (FP32 or FP64),
    and the gain_center is a dynamic Float32 argument fused into the weight multiply.

    Args:
        dtype: Element type of the input tensor.
        N: Feature dimension size.
        use_fp64_reduction: If True, use FP64 accumulation for sum-of-squares and
            FP64 rsqrt (for f32 inputs). If False, use FP32 (sufficient for bf16/f16).
        input_scale: Apply the effective weight before normalization instead of after.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        use_fp64_reduction: bool = False,
        input_scale: bool = False,
    ):
        reduction_dtype = Float64 if use_fp64_reduction else Float32
        super().__init__(dtype, N, stage=1, reduction_dtype=reduction_dtype)
        self.use_fp64_reduction = use_fp64_reduction
        self.input_scale = input_scale
        config = RmsNormFwdConfig.from_analytical_heuristic(
            N, dtype.width, arch_major=_current_arch_major()
        )
        self.reload_from = config.reload_from
        # input_scale consumes w inside the sum-of-squares reduction, so the
        # w load can never be deferred past it.
        self.delay_w_load = config.delay_w_load and not input_scale
        self._num_threads_val = config.num_threads
        self._threads_per_row_val = config.threads_per_row
        self._cluster_n_val = config.cluster_n

    @classmethod
    @jit_cache
    def compile(
        cls,
        dtype,
        out_dtype,
        weight_dtype,
        N,
        has_rstd,
        use_fp64_reduction,
        input_scale=False,
    ):
        """Compile and cache a launch-specialized forward kernel variant."""
        batch_sym = cute.sym_int()
        all_dtypes = [dtype, out_dtype, weight_dtype]
        div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        out_cute = fake_tensor(out_dtype, (batch_sym, N), div)
        weight_cute = fake_tensor(weight_dtype, (N,), div)
        rstd_cute = fake_tensor(Float32, (batch_sym,)) if has_rstd else None
        return cute.compile(
            cls(
                dtype,
                N,
                use_fp64_reduction=use_fp64_reduction,
                input_scale=input_scale,
            ),
            x_cute,
            weight_cute,
            out_cute,
            rstd_cute,
            Float32(0),  # eps
            Float32(0),  # gain_center
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )

    def _num_threads(self):
        return self._num_threads_val

    def _threads_per_row(self):
        return self._threads_per_row_val

    def _set_cluster_n(self):
        self.cluster_n = self._cluster_n_val

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        eps: Float32,
        gain_center: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mW, mO] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mW = (
            layout_utils.expand(mW, dim=0, size=tiler_mn[0])
            if const_expr(mW is not None)
            else None
        )
        mRstd = (
            layout_utils.expand(mRstd, dim=1, size=self.N)
            if const_expr(mRstd is not None)
            else None
        )
        self.kernel(
            mX,
            mW,
            mO,
            mRstd,
            eps,
            gain_center,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        eps: Float32,
        gain_center: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gO, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, mO, idX)
        ]
        gW = (
            cute.local_tile(mW, tiler_mn, (0, cluster_y))
            if const_expr(mW is not None)
            else None
        )

        thr_copy_X = tiled_copy.get_slice(tidx)
        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_X.partition_D(gO)
        tXrRstd = (
            thr_copy_X.partition_D(cute.local_tile(mRstd, tiler_mn, (bidx, cluster_y)))
            if const_expr(mRstd is not None)
            else None
        )
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrW = cute.make_fragment_like(tXgW) if const_expr(mW is not None) else None
        tXrX, tXrO = [cute.make_fragment_like(t) for t in (tXgX, tXgO)]

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)
        # OOB w lanes feed the reduction under input_scale (OOB x lanes are
        # zfilled, but 0 * garbage w is NaN when the garbage is Inf/NaN).
        if const_expr(self.input_scale and mW is not None and not is_even_N):
            tXrW.fill(0.0)

        row = tXcX[0][0]

        # Vectorized load: gmem -> smem -> rmem
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()

        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(self.input_scale and mW is not None):
            x *= tXrW.load().to(cute.Float32) + gain_center

        # --- Sum of squares reduction ---
        # FP64 mode: cast x*x to FP64, reduce in FP64, rsqrt in FP64.
        # FP32 mode: reduce x*x in FP32, rsqrt with fastmath (sufficient for bf16/f16).
        if const_expr(self.use_fp64_reduction):
            x_sq = (x * x).to(Float64)
            sum_sq_x = row_reduce(
                x_sq,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=Float64(0.0),
                hook_fn=cute.arch.cluster_wait
                if const_expr(self.cluster_n > 1)
                else None,
            )
            rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=False)
        else:
            sum_sq_x = row_reduce(
                _fma_dot_f32(x, x),
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=0.0,
                hook_fn=cute.arch.cluster_wait
                if const_expr(self.cluster_n > 1)
                else None,
            )
            rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=False)

        if const_expr(mRstd is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = (
                    rstd.to(Float32) if const_expr(self.use_fp64_reduction) else rstd
                )

        if const_expr(self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)

        if const_expr(self.reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)
            if const_expr(self.input_scale and mW is not None):
                x *= tXrW.load().to(cute.Float32) + gain_center

        y = x * rstd
        if const_expr(mW is not None and not self.input_scale):
            y *= tXrW.load().to(cute.Float32) + gain_center
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)


# ---------------------------------------------------------------------------
# Kahan-compensated forward kernel: RMSNormKahan
# ---------------------------------------------------------------------------
# Uses the same vectorized tiled_copy, cluster support, and smem staging as
# FusedRMSNormFwd, but replaces the naive per-thread tree reduction with Kahan
# compensated summation over register elements. Intended for hardware where
# FP64 is slow (e.g. GB300 with 1/64 FP64 throughput).
#
# Reference: https://en.wikipedia.org/wiki/Kahan_summation_algorithm


class RMSNormKahan(ReductionBase):
    """RMSNorm forward with Kahan-compensated per-thread sum-of-squares.

    Uses the same vectorized tiled_copy, cluster support, and smem staging as the
    standard quack RMSNorm. Replaces the naive per-thread tree reduction with Kahan
    compensated summation over register elements. Uses fastmath=False on rsqrt.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int, input_scale: bool = False):
        super().__init__(dtype, N, stage=1)
        self.input_scale = input_scale
        config = RmsNormFwdConfig.from_analytical_heuristic(
            N, dtype.width, arch_major=_current_arch_major()
        )
        self.reload_from = config.reload_from
        # input_scale consumes w inside the sum-of-squares reduction, so the
        # w load can never be deferred past it.
        self.delay_w_load = config.delay_w_load and not input_scale
        self._num_threads_val = config.num_threads
        self._threads_per_row_val = config.threads_per_row
        self._cluster_n_val = config.cluster_n

    @classmethod
    @jit_cache
    def compile(cls, dtype, out_dtype, weight_dtype, N, has_rstd, input_scale=False):
        """Compile and cache a launch-specialized Kahan forward variant."""
        batch_sym = cute.sym_int()
        all_dtypes = [dtype, out_dtype, weight_dtype]
        div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        out_cute = fake_tensor(out_dtype, (batch_sym, N), div)
        weight_cute = fake_tensor(weight_dtype, (N,), div)
        rstd_cute = fake_tensor(Float32, (batch_sym,)) if has_rstd else None
        return cute.compile(
            cls(dtype, N, input_scale=input_scale),
            x_cute,
            weight_cute,
            out_cute,
            rstd_cute,
            Float32(0),  # eps
            Float32(0),  # gain_center
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )

    def _num_threads(self):
        return self._num_threads_val

    def _threads_per_row(self):
        return self._threads_per_row_val

    def _set_cluster_n(self):
        self.cluster_n = self._cluster_n_val

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        eps: Float32,
        gain_center: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mW, mO] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mW = (
            layout_utils.expand(mW, dim=0, size=tiler_mn[0])
            if const_expr(mW is not None)
            else None
        )
        mRstd = (
            layout_utils.expand(mRstd, dim=1, size=self.N)
            if const_expr(mRstd is not None)
            else None
        )
        self.kernel(
            mX,
            mW,
            mO,
            mRstd,
            eps,
            gain_center,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        eps: Float32,
        gain_center: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gO, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, mO, idX)
        ]
        gW = (
            cute.local_tile(mW, tiler_mn, (0, cluster_y))
            if const_expr(mW is not None)
            else None
        )

        thr_copy_X = tiled_copy.get_slice(tidx)

        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_X.partition_D(gO)
        tXrRstd = (
            thr_copy_X.partition_D(cute.local_tile(mRstd, tiler_mn, (bidx, cluster_y)))
            if const_expr(mRstd is not None)
            else None
        )
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # Register fragments
        tXrW = cute.make_fragment_like(tXgW) if const_expr(mW is not None) else None
        tXrX, tXrO = [cute.make_fragment_like(t) for t in (tXgX, tXgO)]

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)
        # OOB w lanes feed the reduction under input_scale (OOB x lanes are
        # zfilled, but 0 * garbage w is NaN when the garbage is Inf/NaN).
        if const_expr(self.input_scale and mW is not None and not is_even_N):
            tXrW.fill(0.0)

        row = tXcX[0][0]

        # Vectorized load x from gmem -> smem -> rmem
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()

        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)

        # --- Kahan-compensated sum of squares ---
        # Iterate over register fragment elements with Kahan accumulation.
        # Reads directly from tXrX to avoid allocating a separate x_sq fragment.
        num_elems = const_expr(cute.size(tXrX))
        kahan_sum = cute.Float32(0.0)
        kahan_comp = cute.Float32(0.0)
        for i in cutlass.range_constexpr(num_elems):
            x_val = tXrX[i].to(cute.Float32)
            if const_expr(self.input_scale and mW is not None):
                x_val *= tXrW[i].to(cute.Float32) + gain_center
            x_sq_val = x_val * x_val
            # Fast2Sum: y = val + comp; t = sum + y; comp = y - (t - sum); sum = t
            y_val = x_sq_val + kahan_comp
            t_val = kahan_sum + y_val
            kahan_comp = y_val - (t_val - kahan_sum)
            kahan_sum = t_val

        # Cross-thread reduction of compensated partial sums
        sum_sq_x = row_reduce(
            kahan_sum,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=False)

        if const_expr(mRstd is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd

        if const_expr(self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)

        # Reload x from smem if registers were reused (large D)
        if const_expr(self.reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)

        x = tXrX.load().to(cute.Float32)
        if const_expr(self.input_scale and mW is not None):
            x *= tXrW.load().to(cute.Float32) + gain_center
        y = x * rstd
        if const_expr(mW is not None and not self.input_scale):
            y *= tXrW.load().to(cute.Float32) + gain_center
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)
