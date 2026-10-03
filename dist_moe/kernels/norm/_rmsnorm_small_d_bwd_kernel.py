# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL RMSNorm backward specialized for small feature dims (D <= 256).

The scaleless (w = None) backward companion of
``_rmsnorm_small_d_fwd_kernel.py``, built for the qk-norm regime: rows are just
the head dim, the row count is T * H, and — with no weight — there is no dw
accumulation, so the backward is purely per-row:

    x_hat = x * rstd
    mean  = sum(x_hat * dy) / N        (per-row warp reduction)
    dx    = (dy - x_hat * mean) * rstd

The load side reuses the forward's warp-owned pipeline: per-warp smem rings
for BOTH streams (x and dy), streamed with cp.async at 16B per lane and
pipelined across chunks with ``cp.async.wait_group`` + ``sync_warp`` only.
Head-sliced x is addressed per lane straight from its (T, H) strides (any H
binds); dy, rstd, and dx are flat ``(M, ...)`` tensors.

The compute geometry deliberately replicates the general backward's
Blackwell configuration for these shapes, so on Blackwell dx is **bitwise
identical** to the reshape-copy + ``FusedRMSNormBwd`` launch it replaces:
``vecsize = gcd(N, 128 // max_width)`` consecutive columns per lane (16B
for pure bf16), a 32-wide butterfly ``warp_reduction`` (the general
kernel's Blackwell ``threads_per_row = 32`` for every N this kernel
accepts), and lanes whose columns fall beyond N contribute exact zeros —
the same values the general kernel's predicated cp.async zfill feeds its
On other archs (Hopper), whose general launch table picks a different
threads_per_row, the trees diverge at round-off level — equally accurate,
not bitwise across the two kernels. Per-row numerics never depend on the
input layout, so strided and contiguous launches of THIS kernel agree
bitwise everywhere.

Supported: w = None; x, dy, and dx all of one width with ``N * width`` a
whole number of 16B lanes and ``N / vecsize <= 32``. Everything else,
including a wider (fp32) dx over
16-bit x/dy — stays on the general kernel.
"""

import math
import operator
from functools import lru_cache
from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr, Float32

from .. import _dsl_compat as _cute_extern  # noqa: F401
from .._iket_shim import iket, iket_level
from .._jit_cache import jit_cache
from .._quack import copy_utils
from .._quack_compile_utils import make_fake_stream, make_fake_tensor as fake_tensor
from .._quack_utils import elem_pointer
from ._rmsnorm_common import (
    _fake_div,
    _fma_dot_f32,
    _head_walk,
    _LANE_BYTES,
    _MAX_WARP_SMEM_BYTES,
    _NormLaunch,
    _plan_grid,
    _resolve_input,
    _walk_row_coords,
)

# Same trace-time IKET level knob as the forward (see iket_shim).
_IKET_LEVEL = iket_level("MSL_SMALLD_IKET")

# The general backward's Blackwell threads_per_row for every shape this
# kernel accepts (pure-bf16 / narrow-row branch of its launch heuristic);
# matching it is what makes Blackwell bf16 dx bitwise with the general
# kernel (the strided-vs-contiguous parity tests enforce it).
_REDUCE_WIDTH = 32


class RMSNormSmallDBwd:
    """Load-once scaleless RMSNorm backward for D <= 256 (module docstring).

    Args:
        dtype / dout_dtype / dx_dtype: Element types of x / dy / dx.
        N: Feature dimension size (the head dim for qk norms).
        T_hint: Expected flat row count; buckets the JIT cache per unique M
            and sizes the persistent grid.
        x_head_group / x_walk_t_minor / x_tokens: Head-sliced x binding,
            exactly as in :class:`RMSNormSmallDFwd`.
        stages / warps_per_cta / chunk_rows: Pipeline knobs, as in the
            forward (chunk bytes count both the x and dy rings).
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        dout_dtype: Optional[Type[cutlass.Numeric]] = None,
        dx_dtype: Optional[Type[cutlass.Numeric]] = None,
        T_hint: int = 0,
        x_head_group: Optional[int] = None,
        x_walk_t_minor: bool = False,
        x_tokens: int = 0,
        stages: Optional[int] = None,
        warps_per_cta: int = 8,
        chunk_rows: Optional[int] = None,
    ):
        self.dtype = dtype
        self.N = N
        self.dout_dtype = dout_dtype if dout_dtype is not None else dtype
        self.dx_dtype = dx_dtype if dx_dtype is not None else dtype
        self.x_head_group = x_head_group
        self.x_walk_t_minor = x_walk_t_minor
        self.x_tokens = x_tokens
        if x_walk_t_minor and x_tokens <= 0:
            raise ValueError("x_walk_t_minor requires the token count (x_tokens)")
        self.warps_per_cta = warps_per_cta
        geom = self._geometry(
            N, dtype.width, self.dout_dtype.width, self.dx_dtype.width
        )
        if geom is None:
            raise ValueError(
                f"RMSNormSmallDBwd does not support N={N} (widths "
                f"{dtype.width}/{self.dout_dtype.width}/{self.dx_dtype.width})"
            )
        self.vec, self.load_tpr, self.load_rps = geom
        # Defaults from GB200 shmoos (IKET shows the backward is
        # compute-bound per warp with memory fully hidden, so resident-warp
        # count dominates): 8 warps of shallow chunks at 2 stages beat the
        # forward's wide-chunk rule by 15-22% on the Q slices and tie at
        # the launch floor. Chunk depth caps at 8 rows AND ~4KB of combined
        # x+dy bytes — bf16 D<=128 rows take 8-row chunks (8KB/warp ring ->
        # 24 warps/SM); 1KB rows (fp32 D=128, bf16 D=256) prefer 4-row
        # chunks, worth 5-10% over depth 8. Explicit arguments override.
        if chunk_rows is None:
            combined_row_bytes = N * (dtype.width + self.dout_dtype.width) // 8
            chunk_rows = max(self.load_rps, min(8, 4096 // combined_row_bytes))
        if stages is None:
            stages = 2
        self.stages = stages
        self.chunk_rows = max(chunk_rows, self.load_rps)
        if self.chunk_rows % self.load_rps != 0:
            raise ValueError(
                f"chunk_rows={self.chunk_rows} must be a multiple of "
                f"rows_per_load_step={self.load_rps}"
            )
        self.load_steps = self.chunk_rows // self.load_rps
        chunk_bytes = self.chunk_rows * (N * (dtype.width + self.dout_dtype.width) // 8)
        self.num_ctas = _plan_grid(
            N,
            chunk_bytes,
            stages,
            warps_per_cta,
            self.chunk_rows,
            T_hint,
            2 * _MAX_WARP_SMEM_BYTES,
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _geometry(
        N: int, x_width: int, dout_width: int, dx_width: int
    ) -> Optional[tuple[int, int, int]]:
        """Resolve (vec, load_tpr, load_rps) or None when unsupported.

        ``vec`` follows the general backward's ``dx_epilogue_vecsize`` rule
        (gcd over all operand widths) so the per-lane column split — and
        with the fixed 32-wide butterfly, the whole reduction tree — matches
        the general kernel's Blackwell configuration bitwise. The load
        geometry is the forward's 16B-per-lane split of the x rows (x and
        dy must be of equal width to share it).
        """
        if x_width != dout_width:
            return None
        if dx_width != x_width:
            # A wider dx shrinks the compute vec below a full 16B lane, but
            # the launcher's alignment contract (``_fake_div``) then drops to
            # vec granularity while the load path still issues 16B-per-lane
            # copies — the compile-time fake tensors cannot honor both. Keep
            # mixed-width backwards on the general kernel.
            return None
        max_width = max(x_width, dout_width, dx_width)
        vec = math.gcd(N, 128 // max_width)
        if N % vec != 0 or N // vec > _REDUCE_WIDTH:
            return None
        row_bits = N * x_width
        if row_bits % (_LANE_BYTES * 8) != 0 or row_bits > 32 * _LANE_BYTES * 8:
            return None
        load_vec = _LANE_BYTES * 8 // x_width
        load_tpr = N // load_vec
        if load_tpr > 32 or (load_tpr & (load_tpr - 1)) != 0:
            return None
        return vec, load_tpr, 32 // load_tpr

    @classmethod
    def can_implement(cls, dtype, dout_dtype, dx_dtype, N: int) -> bool:
        """Return whether the operand widths and feature size are supported."""
        return (
            cls._geometry(N, dtype.width, dout_dtype.width, dx_dtype.width) is not None
        )

    @classmethod
    def for_input(
        cls,
        x: torch.Tensor,
        dtype,
        dout_dtype,
        dx_dtype,
        N: int,
        T_hint: int = 0,
        stages: Optional[int] = None,
        warps_per_cta: int = 8,
        chunk_rows: Optional[int] = None,
    ) -> _NormLaunch:
        """Bind a torch x to a backward variant (same contract as the
        forward's ``for_input``: the launcher takes x in its original
        layout). Any head count binds; callers select the kernel with
        :meth:`can_implement` first — an unsupported geometry raises here."""
        if not cls.can_implement(dtype, dout_dtype, dx_dtype, N):
            raise ValueError(
                f"RMSNormSmallDBwd does not support N={N} for these operand widths"
            )
        form, head_group = _resolve_input(
            x, _fake_div(N, (dtype, dout_dtype, dx_dtype))
        )
        x_walk_t_minor, x_tokens = (
            _head_walk(x) if head_group is not None else (False, 0)
        )
        return _NormLaunch(
            cls,
            (
                dtype,
                dout_dtype,
                dx_dtype,
                N,
                T_hint,
                head_group,
                x_walk_t_minor,
                x_tokens,
                stages,
                warps_per_cta,
                chunk_rows,
            ),
            form,
            head_group,
        )

    @classmethod
    @jit_cache
    def compile(
        cls,
        dtype,
        dout_dtype,
        dx_dtype,
        N,
        T_hint=0,
        x_head_group=None,
        x_walk_t_minor=False,
        x_tokens=0,
        stages=None,
        warps_per_cta=8,
        chunk_rows=None,
    ):
        """Compile and cache a launch-specialized small-D backward variant."""
        batch_sym = cute.sym_int()
        div = _fake_div(N, (dtype, dout_dtype, dx_dtype))
        if x_head_group is None:
            x_cute = fake_tensor(dtype, (batch_sym, N), div)
        else:
            x_cute = fake_tensor(dtype, (cute.sym_int(), x_head_group, N), div)
        dout_cute = fake_tensor(dout_dtype, (batch_sym, N), div)
        rstd_cute = fake_tensor(Float32, (batch_sym,))
        dx_cute = fake_tensor(dx_dtype, (batch_sym, N), div)
        return cute.compile(
            cls(
                dtype,
                N,
                dout_dtype=dout_dtype,
                dx_dtype=dx_dtype,
                T_hint=T_hint,
                x_head_group=x_head_group,
                x_walk_t_minor=x_walk_t_minor,
                x_tokens=x_tokens,
                stages=stages,
                warps_per_cta=warps_per_cta,
                chunk_rows=chunk_rows,
            ),
            x_cute,
            dout_cute,
            rstd_cute,
            dx_cute,
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mdO: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        load_vec = const_expr(_LANE_BYTES * 8 // self.dtype.width)
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype, self.load_tpr, cute.arch.WARP_SIZE, load_vec
        )
        tiler_mn = (self.load_rps, self.N)
        self.kernel(mX, mdO, mRstd, mdX, tiler_mn, tiled_copy).launch(
            grid=[self.num_ctas, 1, 1],
            block=[self.warps_per_cta * cute.arch.WARP_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901 -- constexpr-stripped IKET guards and the warp-local pipeline branches keep the hot loop inline
        self,
        mX: cute.Tensor,
        mdO: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tidx % cute.arch.WARP_SIZE
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        STAGES = const_expr(self.stages)
        LSTEPS = const_expr(self.load_steps)
        CROWS = const_expr(self.chunk_rows)
        WARPS = const_expr(self.warps_per_cta)
        VEC = const_expr(self.vec)
        LVEC = const_expr(_LANE_BYTES * 8 // self.dtype.width)
        LRPS = const_expr(self.load_rps)
        N = const_expr(self.N)

        shape = mX.shape
        if const_expr(self.x_head_group is None):
            M = cute.size(shape[0])
        else:
            M = cute.size(shape[0]) * cute.size(shape[1])
        n_chunks = cute.ceil_div(M, CROWS)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(
                (tiler_mn[0], tiler_mn[1], LSTEPS, STAGES, WARPS),
                order=(1, 0, 2, 3, 4),
            ),
            byte_alignment=16,
        )
        sdO = smem.allocate_tensor(
            mdO.element_type,
            cute.make_ordered_layout(
                (tiler_mn[0], tiler_mn[1], LSTEPS, STAGES, WARPS),
                order=(1, 0, 2, 3, 4),
            ),
            byte_alignment=16,
        )

        thr_copy = tiled_copy.get_slice(lane)
        gdO = cute.local_tile(mdO, tiler_mn, (None, 0))
        tXgdO = thr_copy.partition_S(gdO)
        tXsX = thr_copy.partition_D(sX[None, None, None, None, warp])
        tXsdO = thr_copy.partition_D(sdO[None, None, None, None, warp])
        if const_expr(self.x_head_group is None):
            gX = cute.local_tile(mX, tiler_mn, (None, 0))
            tXgX = thr_copy.partition_S(gX)
        else:
            tXgX = None

        # Load-side lane mapping (16B per lane, same as the forward).
        T_MINOR = const_expr(self.x_head_group is not None and self.x_walk_t_minor)
        HG = const_expr(self.x_head_group if self.x_head_group is not None else 1)
        TOKENS = const_expr(self.x_tokens if self.x_tokens > 0 else 1)
        l_lane_row = lane // const_expr(self.load_tpr)
        l_lane_col = (lane % const_expr(self.load_tpr)) * LVEC
        s_tok, s_head = mX.stride[0], mX.stride[1]
        g2s_atom = copy_utils.get_copy_atom(mX.element_type, LVEC, is_async=True)
        g2s_dO_atom = copy_utils.get_copy_atom(mdO.element_type, LVEC, is_async=True)
        load_vec_layout = cute.make_layout((LVEC,))

        # Compute-side lane mapping: ``vec`` consecutive columns per lane,
        # 32-wide butterfly; lanes whose columns fall beyond N contribute
        # exact zeros (the general kernel's zfilled lanes).
        c_col = lane * VEC
        c_active = c_col < N
        dx_atom = copy_utils.get_copy_atom(mdX.element_type, VEC, is_async=False)
        dx_vec_layout = cute.make_layout((VEC,))
        # A compute-side lane segment is vec * elem_bytes aligned — a full
        # 16B only when the operand mix does not shrink vec below a 16B lane
        # (fp32 dx over 16-bit x/dy gives 8B x/dy segments). The verifier
        # cannot see this through elem_pointer arithmetic, so assert exactly
        # that guarantee. (The load side is unconditionally 16B per lane.)
        xdy_seg_bytes = min(16, self.vec * mX.element_type.width // 8)
        dx_seg_bytes = min(16, self.vec * mdX.element_type.width // 8)

        fX = [
            cute.make_rmem_tensor(cute.make_layout((VEC,)), mX.element_type)
            for _ in range(2)
        ]
        fdO = [
            cute.make_rmem_tensor(cute.make_layout((VEC,)), mdO.element_type)
            for _ in range(2)
        ]
        fdX = [
            cute.make_rmem_tensor(cute.make_layout((VEC,)), mdX.element_type)
            for _ in range(2)
        ]
        # Inactive lanes (columns beyond N) never load; their zero-filled
        # fragments feed the 32-wide butterfly with the exact zeros the
        # general kernel's predicated-zfill lanes contribute.
        fX[0].fill(0.0)
        fX[1].fill(0.0)
        fdO[0].fill(0.0)
        fdO[1].fill(0.0)

        first_chunk = bidx * WARPS + warp
        chunk_stride = gdim * WARPS

        # --- cp.async prologue -------------------------------------------
        for j in cutlass.range_constexpr(STAGES - 1):
            cj = first_chunk + j * chunk_stride
            if cj < n_chunks:
                for st in cutlass.range_constexpr(LSTEPS):
                    tile = cj * LSTEPS + st
                    row = tile * LRPS + l_lane_row
                    if row < M:
                        if const_expr(self.x_head_group is None):
                            copy_utils.copy(
                                tXgX[None, None, None, tile],
                                tXsX[None, None, None, st, j],
                                is_async=True,
                            )
                        else:
                            t, h = _walk_row_coords(row, HG, TOKENS, T_MINOR)
                            src = cute.make_tensor(
                                (
                                    mX.iterator + (t * s_tok + h * s_head + l_lane_col)
                                ).align(16),
                                load_vec_layout,
                            )
                            dst = cute.make_tensor(
                                elem_pointer(
                                    sX, (l_lane_row, l_lane_col, st, j, warp)
                                ).align(16),
                                load_vec_layout,
                            )
                            cute.copy(g2s_atom, src, dst)
                        if const_expr(T_MINOR):
                            # dy is flat (M, N): the walk-row's x must pair
                            # with dy at its flat index t*HG + h (== row only
                            # in the h-minor walk), the same mapping rstd and
                            # the dx store use.
                            t, h = _walk_row_coords(row, HG, TOKENS, T_MINOR)
                            dy_src = cute.make_tensor(
                                (
                                    mdO.iterator
                                    + ((t * HG + h) * mdO.stride[0] + l_lane_col)
                                ).align(16),
                                load_vec_layout,
                            )
                            dy_dst = cute.make_tensor(
                                elem_pointer(
                                    sdO, (l_lane_row, l_lane_col, st, j, warp)
                                ).align(16),
                                load_vec_layout,
                            )
                            cute.copy(g2s_dO_atom, dy_src, dy_dst)
                        else:
                            copy_utils.copy(
                                tXgdO[None, None, None, tile],
                                tXsdO[None, None, None, st, j],
                                is_async=True,
                            )
            cute.arch.cp_async_commit_group()

        stage = cutlass.Int32(0)
        if const_expr(_IKET_LEVEL >= 1):
            iket.range_push("main")
        # --- main loop -----------------------------------------------------
        for c in cutlass.range(first_chunk, n_chunks, chunk_stride):
            if const_expr(_IKET_LEVEL >= 1):
                iket.range_push("wait", c)
            cute.arch.cp_async_wait_group(STAGES - 2)
            cute.arch.sync_warp()
            if const_expr(_IKET_LEVEL >= 1):
                iket.range_pop()

            if const_expr(_IKET_LEVEL >= 1):
                iket.range_push("issue")
            cn = c + (STAGES - 1) * chunk_stride
            issue_stage = stage - 1
            if issue_stage < 0:
                issue_stage = STAGES - 1
            if cn < n_chunks:
                for st in cutlass.range_constexpr(LSTEPS):
                    tile = cn * LSTEPS + st
                    row = tile * LRPS + l_lane_row
                    if row < M:
                        if const_expr(self.x_head_group is None):
                            copy_utils.copy(
                                tXgX[None, None, None, tile],
                                tXsX[None, None, None, st, issue_stage],
                                is_async=True,
                            )
                        else:
                            t, h = _walk_row_coords(row, HG, TOKENS, T_MINOR)
                            src = cute.make_tensor(
                                (
                                    mX.iterator + (t * s_tok + h * s_head + l_lane_col)
                                ).align(16),
                                load_vec_layout,
                            )
                            dst = cute.make_tensor(
                                elem_pointer(
                                    sX, (l_lane_row, l_lane_col, st, issue_stage, warp)
                                ).align(16),
                                load_vec_layout,
                            )
                            cute.copy(g2s_atom, src, dst)
                        if const_expr(T_MINOR):
                            t, h = _walk_row_coords(row, HG, TOKENS, T_MINOR)
                            dy_src = cute.make_tensor(
                                (
                                    mdO.iterator
                                    + ((t * HG + h) * mdO.stride[0] + l_lane_col)
                                ).align(16),
                                load_vec_layout,
                            )
                            dy_dst = cute.make_tensor(
                                elem_pointer(
                                    sdO, (l_lane_row, l_lane_col, st, issue_stage, warp)
                                ).align(16),
                                load_vec_layout,
                            )
                            cute.copy(g2s_dO_atom, dy_src, dy_dst)
                        else:
                            copy_utils.copy(
                                tXgdO[None, None, None, tile],
                                tXsdO[None, None, None, st, issue_stage],
                                is_async=True,
                            )
            cute.arch.cp_async_commit_group()
            if const_expr(_IKET_LEVEL >= 1):
                iket.range_pop()

            if const_expr(_IKET_LEVEL >= 1):
                iket.range_push("compute")
            # One row per step across the whole warp; LDS double-buffered.
            row0 = c * CROWS
            if c_active:
                cute.autovec_copy(
                    cute.make_tensor(
                        elem_pointer(sX, (0, c_col, 0, stage, warp)).align(
                            xdy_seg_bytes
                        ),
                        dx_vec_layout,
                    ),
                    fX[0],
                )
                cute.autovec_copy(
                    cute.make_tensor(
                        elem_pointer(sdO, (0, c_col, 0, stage, warp)).align(
                            xdy_seg_bytes
                        ),
                        dx_vec_layout,
                    ),
                    fdO[0],
                )
            for r in cutlass.range_constexpr(CROWS):
                if const_expr(r + 1 < CROWS):
                    nst = const_expr((r + 1) // LRPS)
                    nrow = const_expr((r + 1) % LRPS)
                    if c_active:
                        cute.autovec_copy(
                            cute.make_tensor(
                                elem_pointer(sX, (nrow, c_col, nst, stage, warp)).align(
                                    xdy_seg_bytes
                                ),
                                dx_vec_layout,
                            ),
                            fX[(r + 1) % 2],
                        )
                        cute.autovec_copy(
                            cute.make_tensor(
                                elem_pointer(
                                    sdO, (nrow, c_col, nst, stage, warp)
                                ).align(xdy_seg_bytes),
                                dx_vec_layout,
                            ),
                            fdO[(r + 1) % 2],
                        )
                row = row0 + r
                r_out = row
                if const_expr(T_MINOR):
                    t, h = _walk_row_coords(row, HG, TOKENS, T_MINOR)
                    r_out = t * HG + h
                rstd = Float32(0.0)
                if row < M:
                    rstd = mRstd[r_out]
                x = fX[r % 2].load().to(Float32)
                dout = fdO[r % 2].load().to(Float32)
                x_hat = x * rstd
                wdy = dout
                mean_xhat_wdy = (
                    cute.arch.warp_reduction(
                        _fma_dot_f32(x_hat, wdy),
                        operator.add,
                        threads_in_group=_REDUCE_WIDTH,
                    )
                    / N
                )
                dx = (wdy - x_hat * mean_xhat_wdy) * rstd
                fdX[r % 2].store(dx.to(fdX[r % 2].element_type))
                if c_active and row < M:
                    cute.copy(
                        dx_atom,
                        fdX[r % 2],
                        cute.make_tensor(
                            (mdX.iterator + (r_out * mdX.stride[0] + c_col)).align(
                                dx_seg_bytes
                            ),
                            dx_vec_layout,
                        ),
                    )
            if const_expr(_IKET_LEVEL >= 1):
                iket.range_pop()

            stage = stage + 1
            if stage == STAGES:
                stage = 0
        if const_expr(_IKET_LEVEL >= 1):
            iket.range_pop()
