# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL RMSNorm forward specialized for small feature dims (D <= 256).

The general kernels in ``_rmsnorm_fwd_kernel.py`` are shaped for the wide
model-dim norms (D up to 128k): CTA-wide row tiles, cluster support, smem
reload paths. The qk norm is the opposite regime — D is just the head dim
(64/128/256) while the row count is T * H (up to ~1M rows) — so this kernel
assigns work so that every D-element row is loaded from gmem exactly once
and never revisited:

  - **G2S**: each warp owns a private smem ring of ``stages`` chunks of
    ``chunk_rows`` full rows and streams them in with cp.async (LDGSTS),
    pipelined across chunks: while a chunk is being reduced, the next ones
    are in flight. No TMA, no CTA barriers — the pipeline is warp-local
    (cp.async groups are per-thread state, so ``cp.async.wait_group`` plus
    ``sync_warp`` is the full synchronization).
  - **Compute**: a warp reduces one row group at a time straight out of
    smem. Each row is spread across ``threads_per_row = row_bytes // 16``
    lanes (16B per lane), so a warp covers ``32 // threads_per_row`` rows
    per step. Lanes do an in-thread FMA sum-of-squares over their vector,
    then a butterfly ``warp_reduction`` over the ``threads_per_row`` lanes
    of the row. The smem -> register loads (LDS) are double-buffered across
    steps so the next row group's LDS issues before the current group's
    reduction, hiding one behind the other.
  - **Store**: ``y = x * rstd * (w + gain_center)`` is computed in the same
    per-lane vector registers and stored straight to gmem (STG) — the
    output never touches smem.

A single ``tiled_copy_2d(dtype, threads_per_row, 32, vec)`` tv-layout drives
all three phases, with ``vec`` sized so each lane moves 16 bytes: LDGSTS.128
on the gmem loads, LDS.128 on the smem reads, and STG.128 on the stores.
The geometry is a pure function of (N, operand widths) — never of the input
layout — so per-row numerics are layout-invariant. Head-sliced x is
addressed per lane straight from its (T, H) strides: cp.async is per-thread
gather and every row is a whole 128B-sector-aligned run, so no nested-layout
algebra — and no divisibility constraint on H — is involved. ``for_input``
picks the load-walk order (heads-fastest vs tokens-fastest) from whichever
row mode is memory-minor, so a warp's consecutive rows stay gmem-adjacent
for transposed layouts too; the tokens-fastest walk remaps rows to the flat
``t * H + h`` output order at store time.

Grid: a fixed CTA count (T_hint-derived, persistent-style) of
``warps_per_cta`` independent warps; warp ``w`` grid-strides over chunks
``w, w + total_warps, ...``.

Supported: ``16 <= D * dtype_width // 8 <= 512`` with 16-byte rows
(bf16/fp16 D in {8..256}, fp32 D in {4..128}) and ``D // vec`` a power of
two <= 32. Forward only; the backward keeps the general kernel (its dw
accumulation wants CTA-wide row tiles).
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
from .._quack import copy_utils, layout_utils
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

# In-kernel event tracing level (see iket_shim): 0 off (default), 1 traces
# per-chunk wait / issue / compute ranges per warp.
_IKET_LEVEL = iket_level("MSL_SMALLD_IKET")


class RMSNormSmallDFwd:
    """Load-once RMSNorm forward for D <= 256 (see module docstring).

    Args:
        dtype: Element type of the input tensor.
        N: Feature dimension size (the head dim for qk norms).
        out_dtype / weight_dtype: Output / weight element types; only their
            widths matter (the lane vector is sized by the widest operand,
            exactly like the general kernels' ``vecsize``).
        T_hint: Expected flat row count (T * H). Buckets the JIT cache per
            unique M (like ``FusedRMSNormBwd``) and sizes the persistent
            grid; 0 means "large".
        x_head_group: If set, x arrives as a head-sliced rank-3 (T, H, N)
            view (the qkv Q/K slice case) and is read in place, each lane's
            16B segment addressed from the (T, H) strides — any H binds.
        x_walk_t_minor / x_tokens: Load-walk order over the row modes,
            derived from the strides by ``for_input`` (see ``__init__``).
        stages: cp.async pipeline depth (chunks in flight per warp).
        warps_per_cta: Independent warps per CTA.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        out_dtype: Optional[Type[cutlass.Numeric]] = None,
        weight_dtype: Optional[Type[cutlass.Numeric]] = None,
        T_hint: int = 0,
        x_head_group: Optional[int] = None,
        x_walk_t_minor: bool = False,
        x_tokens: int = 0,
        stages: Optional[int] = None,
        warps_per_cta: int = 4,
        chunk_rows: Optional[int] = None,
    ):
        self.dtype = dtype
        self.N = N
        self.x_head_group = x_head_group
        # Load-walk order over the head-sliced (T, H) row modes, chosen by
        # ``for_input`` from the input's strides: walk the memory-minor mode
        # so a warp's consecutive rows are gmem-adjacent. h-minor (the fused
        # qkv slice case) walks rows in output order; t-minor (transposed
        # layouts) walks tokens fastest and remaps to output rows at store
        # time, which needs the token count baked (``x_tokens``).
        self.x_walk_t_minor = x_walk_t_minor
        self.x_tokens = x_tokens
        if x_walk_t_minor and x_tokens <= 0:
            raise ValueError("x_walk_t_minor requires the token count (x_tokens)")
        self.warps_per_cta = warps_per_cta
        max_width = max(
            dt.width for dt in (dtype, out_dtype, weight_dtype) if dt is not None
        )
        geom = self._geometry(N, dtype.width, max_width)
        if geom is None:
            raise ValueError(
                f"RMSNormSmallDFwd does not support N={N} "
                f"(dtype width {dtype.width}, widest operand {max_width})"
            )
        self.vec, self.threads_per_row, self.rows_per_step = geom
        # Chunk/depth defaults from a GB200 sweep of representative Q/K shapes:
        # large problems (>= 64MB of x) want wide 8KB chunks at 2 stages
        # (fewer, longer LDGSTS bursts per warp: representative Q 0.92x -> 0.99x);
        # smaller ones want 16-row chunks at 3 stages so the chunk supply
        # per warp stays deep enough to pipeline (one K workload improves 1.12x
        # while another remains at parity). Explicit arguments override.
        row_bytes = N * dtype.width // 8
        big = T_hint == 0 or T_hint * row_bytes >= 64 * 1024 * 1024
        if chunk_rows is None:
            chunk_rows = max(16, 8192 // row_bytes) if big else 16
        if stages is None:
            stages = 2 if big else 3
        self.stages = stages
        self.chunk_rows = max(chunk_rows, self.rows_per_step)
        if self.chunk_rows % self.rows_per_step != 0:
            raise ValueError(
                f"chunk_rows={self.chunk_rows} must be a multiple of "
                f"rows_per_step={self.rows_per_step}"
            )
        self.steps = self.chunk_rows // self.rows_per_step
        chunk_bytes = self.chunk_rows * N * dtype.width // 8
        self.num_ctas = _plan_grid(
            N,
            chunk_bytes,
            stages,
            warps_per_cta,
            self.chunk_rows,
            T_hint,
            _MAX_WARP_SMEM_BYTES,
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _geometry(
        N: int, x_width: int, max_width: int
    ) -> Optional[tuple[int, int, int]]:
        """Resolve (vec, threads_per_row, rows_per_step) for this shape.

        The lane vector is the widest the operand mix allows (16B of the
        widest dtype, the general kernels' ``vecsize`` rule) and is a pure
        function of (N, dtypes): threads_per_row — and with it the per-row
        reduction tree — never depends on the input layout or its strides,
        so a strided launch stays bitwise-identical to the contiguous one.
        None when threads_per_row is not a power of two in [8, 32]
        (warp_reduction group constraint; the >= 8 floor keeps every pure
        bf16 shape on the general kernel's threads_per_row table — 8/16/32
        at N = 64/128/256 — so bf16 dispatch is also bitwise-identical to
        the general kernel. Wider-operand mixes reduce over more lanes than
        the general kernel's table for the same N, e.g. 32 vs 16 for fp32
        at N=128: equally accurate, not bitwise across the two kernels) or
        when rows are not whole 16-byte multiples.
        """
        row_bits = N * x_width
        if row_bits % (_LANE_BYTES * 8) != 0 or row_bits > 32 * _LANE_BYTES * 8:
            return None
        vec = math.gcd(N, _LANE_BYTES * 8 // max_width)
        tpr = N // vec
        if tpr > 32 or tpr < 8 or (tpr & (tpr - 1)) != 0:
            return None
        return vec, tpr, 32 // tpr

    @classmethod
    def can_implement(cls, dtype, out_dtype, weight_dtype, N: int) -> bool:
        """Static dispatch predicate: pure function of (N, operand widths),
        never of the input layout — any layout of a supported shape binds
        (head-sliced views in place, 2D-viewable inputs as flat rows,
        misaligned ones via a contiguous copy)."""
        max_width = max(
            dt.width for dt in (dtype, out_dtype, weight_dtype) if dt is not None
        )
        return cls._geometry(N, dtype.width, max_width) is not None

    @classmethod
    def for_input(
        cls,
        x: torch.Tensor,
        dtype,
        out_dtype,
        weight_dtype,
        N: int,
        has_rstd: bool,
        T_hint: int = 0,
        stages: Optional[int] = None,
        warps_per_cta: int = 4,
        chunk_rows: Optional[int] = None,
    ) -> _NormLaunch:
        """Bind a torch x to a small-D forward variant.

        Same contract as the general kernels' ``for_input``: decode x's
        shape / strides / base alignment once and return a plan whose
        compiled launcher takes x in its original layout. Gmem rows are
        addressed per lane from the (H, T) strides, so any head count binds.
        Callers select the kernel with :meth:`can_implement` first; an
        unsupported (N, widths) geometry raises here.
        """
        if not cls.can_implement(dtype, out_dtype, weight_dtype, N):
            raise ValueError(
                f"RMSNormSmallDFwd does not support N={N} for these operand widths"
            )
        form, head_group = _resolve_input(
            x, _fake_div(N, (dtype, out_dtype, weight_dtype))
        )
        x_walk_t_minor, x_tokens = (
            _head_walk(x) if head_group is not None else (False, 0)
        )
        return _NormLaunch(
            cls,
            (
                dtype,
                out_dtype,
                weight_dtype,
                N,
                has_rstd,
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
        out_dtype,
        weight_dtype,
        N,
        has_rstd,
        T_hint=0,
        x_head_group=None,
        x_walk_t_minor=False,
        x_tokens=0,
        stages=None,
        warps_per_cta=4,
        chunk_rows=None,
    ):
        """Compile and cache a launch-specialized small-D forward variant."""
        batch_sym = cute.sym_int()
        div = _fake_div(N, (dtype, out_dtype, weight_dtype))
        if x_head_group is None:
            x_cute = fake_tensor(dtype, (batch_sym, N), div)
        else:
            x_cute = fake_tensor(dtype, (cute.sym_int(), x_head_group, N), div)
        out_cute = fake_tensor(out_dtype, (batch_sym, N), div)
        weight_cute = fake_tensor(weight_dtype, (N,), div)
        rstd_cute = fake_tensor(Float32, (batch_sym,)) if has_rstd else None
        return cute.compile(
            cls(
                dtype,
                N,
                out_dtype=out_dtype,
                weight_dtype=weight_dtype,
                T_hint=T_hint,
                x_head_group=x_head_group,
                x_walk_t_minor=x_walk_t_minor,
                x_tokens=x_tokens,
                stages=stages,
                warps_per_cta=warps_per_cta,
                chunk_rows=chunk_rows,
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
        # One tv-layout for all three phases: (rows_per_step, N) tiles, each
        # lane owning vec contiguous elements of one row. Head-sliced x stays
        # rank-3 — its rows are addressed per lane from the (T, H) strides,
        # so no nested-layout regroup (and no divisibility constraint on H).
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype, self.threads_per_row, cute.arch.WARP_SIZE, self.vec
        )
        tiler_mn = (self.rows_per_step, self.N)
        mW = (
            layout_utils.expand(mW, dim=0, size=tiler_mn[0])
            if const_expr(mW is not None)
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
        ).launch(
            grid=[self.num_ctas, 1, 1],
            block=[self.warps_per_cta * cute.arch.WARP_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901 -- constexpr-stripped IKET guards and the warp-local pipeline branches keep the hot loop inline
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        eps: Float32,
        gain_center: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tidx % cute.arch.WARP_SIZE
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        STAGES = const_expr(self.stages)
        STEPS = const_expr(self.steps)
        WARPS = const_expr(self.warps_per_cta)
        tpr = const_expr(self.threads_per_row)

        shape = mX.shape
        if const_expr(self.x_head_group is None):
            M = cute.size(shape[0])
        else:
            M = cute.size(shape[0]) * cute.size(shape[1])
        # (rows_per_step, N) row groups; a chunk is STEPS consecutive groups.
        n_chunks = cute.ceil_div(M, const_expr(self.chunk_rows))

        smem = cutlass.utils.SmemAllocator()
        # Per-warp private ring: (tile) x steps x stages x warps, tile
        # row-major so lane vectors are contiguous 16B runs (conflict-free
        # for both the LDGSTS writes and the LDS row reads).
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(
                (tiler_mn[0], tiler_mn[1], STEPS, STAGES, WARPS),
                order=(1, 0, 2, 3, 4),
            ),
            byte_alignment=16,
        )

        thr_copy = tiled_copy.get_slice(lane)
        gO = cute.local_tile(mO, tiler_mn, (None, 0))
        tXgO = thr_copy.partition_D(gO)
        tXsX = thr_copy.partition_D(sX[None, None, None, None, warp])
        # Every name the issue/store closures capture must be bound on all
        # const_expr paths: the DSL inspects closure cells when building
        # dynamic-if regions, and a name assigned only under one branch
        # leaves an empty cell that fails the trace.
        if const_expr(self.x_head_group is None):
            gX = cute.local_tile(mX, tiler_mn, (None, 0))
            tXgX = thr_copy.partition_S(gX)
        else:
            tXgX = None

        # This lane's position inside every (rows_per_step, N) row group is
        # fixed by the tv layout (thread = row * threads_per_row +
        # col_group), so global row indices and gmem/smem addresses are
        # plain arithmetic — no per-step identity-tensor reads, and for
        # head-sliced x no nested-layout tiling (each lane's 16B segment is
        # addressed individually from the (T, H) strides; rows are whole
        # 128B-sector-aligned runs, so per-lane addressing loses no
        # coalescing).
        RPS = const_expr(self.rows_per_step)
        VEC = const_expr(self.vec)
        T_MINOR = const_expr(self.x_head_group is not None and self.x_walk_t_minor)
        HG = const_expr(self.x_head_group if self.x_head_group is not None else 1)
        TOKENS = const_expr(self.x_tokens if self.x_tokens > 0 else 1)
        lane_row = lane // tpr
        lane_col = (lane % tpr) * VEC
        is_col0 = lane % tpr == 0
        s_tok, s_head = mX.stride[0], mX.stride[1]
        g2s_atom = copy_utils.get_copy_atom(mX.element_type, VEC, is_async=True)
        r2g_atom = copy_utils.get_copy_atom(mO.element_type, VEC, is_async=False)
        lane_vec = cute.make_layout((VEC,))
        # The decode gate guarantees base and strides are vec-multiples, so a
        # lane segment is vec * elem_bytes aligned — a full 16B only when the
        # operand mix does not shrink vec below a 16B lane (mixed widths give
        # e.g. 8B bf16 segments). The verifier cannot see either through
        # dynamic pointer arithmetic, so assert exactly that guarantee.
        x_seg_bytes = min(16, self.vec * mX.element_type.width // 8)
        o_seg_bytes = min(16, self.vec * mO.element_type.width // 8)

        # Weight: one vec-slice per lane, loaded once for the whole grid
        # stride (every row this lane touches uses the same columns).
        tXrW = None
        wf32 = None
        if const_expr(mW is not None):
            gW = cute.local_tile(mW, tiler_mn, (0, 0))
            tXgW = thr_copy.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            cute.autovec_copy(tXgW, tXrW)
            wf32 = tXrW.load().to(Float32) + gain_center

        tXrX = [cute.make_fragment_like(tXsX[None, None, None, 0, 0]) for _ in range(2)]
        # Two output fragments, alternating per step: a single fragment's
        # reuse is a WAR dependency that chains every step's reduction to the
        # previous step's store, serializing the unrolled compute loop.
        tXrO = [cute.make_fragment_like(tXgO[None, None, None, 0]) for _ in range(2)]

        first_chunk = bidx * WARPS + warp
        chunk_stride = gdim * WARPS

        # --- cp.async prologue: put stages-1 chunks in flight -------------
        # (2D inputs keep the affine partitioned copies; head-sliced inputs
        # address each lane's 16B segment from the (T, H) strides.)
        for j in cutlass.range_constexpr(STAGES - 1):
            cj = first_chunk + j * chunk_stride
            if cj < n_chunks:
                for st in cutlass.range_constexpr(STEPS):
                    tile = cj * STEPS + st
                    row = tile * RPS + lane_row
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
                                    mX.iterator + (t * s_tok + h * s_head + lane_col)
                                ).align(x_seg_bytes),
                                lane_vec,
                            )
                            dst = cute.make_tensor(
                                elem_pointer(
                                    sX, (lane_row, lane_col, st, j, warp)
                                ).align(x_seg_bytes),
                                lane_vec,
                            )
                            cute.copy(g2s_atom, src, dst)
            cute.arch.cp_async_commit_group()

        stage = cutlass.Int32(0)
        if const_expr(_IKET_LEVEL >= 1):
            iket.range_push("main")
        # --- main loop: one chunk per iteration, grid-strided -------------
        for c in cutlass.range(first_chunk, n_chunks, chunk_stride):
            # Oldest in-flight group (this chunk) has landed; the sync_warp
            # also fences last iteration's LDS reads of the stage the
            # prefetch below overwrites.
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
                for st in cutlass.range_constexpr(STEPS):
                    tile = cn * STEPS + st
                    row = tile * RPS + lane_row
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
                                    mX.iterator + (t * s_tok + h * s_head + lane_col)
                                ).align(x_seg_bytes),
                                lane_vec,
                            )
                            dst = cute.make_tensor(
                                elem_pointer(
                                    sX, (lane_row, lane_col, st, issue_stage, warp)
                                ).align(x_seg_bytes),
                                lane_vec,
                            )
                            cute.copy(g2s_atom, src, dst)
            cute.arch.cp_async_commit_group()
            if const_expr(_IKET_LEVEL >= 1):
                iket.range_pop()

            if const_expr(_IKET_LEVEL >= 1):
                iket.range_push("compute")
            # Reduce the chunk one row group at a time, LDS double-buffered
            # so step s+1's loads issue before step s's reduction.
            cute.autovec_copy(tXsX[None, None, None, 0, stage], tXrX[0])
            for st in cutlass.range_constexpr(STEPS):
                if const_expr(st + 1 < STEPS):
                    cute.autovec_copy(
                        tXsX[None, None, None, st + 1, stage], tXrX[(st + 1) % 2]
                    )
                tile = c * STEPS + st
                row = tile * RPS + lane_row
                x = tXrX[st % 2].load().to(Float32)
                sum_sq = cute.arch.warp_reduction(
                    _fma_dot_f32(x, x), operator.add, threads_in_group=tpr
                )
                rstd = cute.math.rsqrt(
                    sum_sq / const_expr(self.N) + eps, fastmath=False
                )
                r_out = row
                if const_expr(T_MINOR):
                    t, h = _walk_row_coords(row, HG, TOKENS, T_MINOR)
                    r_out = t * HG + h
                if const_expr(mRstd is not None):
                    if is_col0 and row < M:
                        mRstd[r_out] = rstd
                y = x * rstd
                if const_expr(mW is not None):
                    y *= wf32
                tXrO[st % 2].store(y.to(tXrO[st % 2].element_type))
                if row < M:
                    if const_expr(T_MINOR):
                        # The t-minor walk's rows are scattered in output
                        # space — store this lane's segment by address.
                        gO_dst = cute.make_tensor(
                            (mO.iterator + (r_out * mO.stride[0] + lane_col)).align(
                                o_seg_bytes
                            ),
                            lane_vec,
                        )
                        rO_src = cute.make_tensor(
                            tXrO[st % 2].iterator.align(o_seg_bytes), lane_vec
                        )
                        cute.copy(r2g_atom, rO_src, gO_dst)
                    else:
                        copy_utils.copy(tXrO[st % 2], tXgO[None, None, None, tile])

            if const_expr(_IKET_LEVEL >= 1):
                iket.range_pop()

            stage = stage + 1
            if stage == STAGES:
                stage = 0
        if const_expr(_IKET_LEVEL >= 1):
            iket.range_pop()
