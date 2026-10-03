# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL RMSNorm backward kernel with fused gain_center.

Contains the forked `FusedRMSNormBwd` kernel class. The gain center is added
in Float32 at each weight use, matching the forward's
``w.to(Float32) + gain_center``. The optional MXFP8 epilogue emits E4M3 data
and E8M0 scales directly from the computed input gradient.
"""

import math
from functools import lru_cache, partial
from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass import (
    Boolean,
    const_expr,
    Float8E4M3FN,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
)
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync

from ...formats import FP8_E4M3_MAX
from .. import (
    _dsl_compat as _cute_extern,  # noqa: F401
    _quack_utils as utils,
)
from .._jit_cache import jit_cache
from .._quack import copy_utils, layout_utils
from .._quack.pipeline import make_pipeline_state, PipelineStasAsync
from .._quack.reduce import row_reduce
from .._quack.reduction_base import ReductionBase
from .._quack_compile_utils import make_fake_stream, make_fake_tensor as fake_tensor
from .._quant_conversion import (
    _cvt_f32x4_to_fp8x4_u32_rn,
)
from ..blockscaled_quantization_common import (
    _compute_fp8_e8m0_scale_rceil,
    _frag_amax_nan,
    _reduce_amax_f32,
)
from ._rmsnorm_common import _fma_dot_f32

# MXFP8 scale-factor vector size: one E8M0 scale byte per 32 columns.
_MX_SF_VEC_SIZE = 32


@cute.jit
def _cvt_e4m3x4_word(
    src: cute.Tensor,
    base: cutlass.Constexpr[int],
    recip: Float32,
) -> Uint32:
    q0 = src[base] * recip
    q1 = src[base + 1] * recip
    q2 = src[base + 2] * recip
    q3 = src[base + 3] * recip
    return _cvt_f32x4_to_fp8x4_u32_rn(
        q0, q1, q2, q3, dst_kind=nvvm.CVTPackFloatKind.E4M3x2
    )


@cute.jit
def _mxfp8_emit_vec(
    src: cute.Tensor,
    j: cutlass.Constexpr[int],
    col_off: cutlass.Constexpr[int],
    q_base_ptr,
    sf_base_ptr,
    is_sf_lane: Boolean,
    vec_ok: Boolean,
    vecsize: cutlass.Constexpr[int],
    reduce_stages: cutlass.Constexpr[int],
):
    """Scale and store one MXFP8 vector: block amax (butterfly-reduced across
    the ``32 // vecsize`` lanes of the scale block, every lane deriving the
    same rceil E8M0 byte, NaN routed to 0xFF), predicated scale-byte store by
    the block's first lane, then the E4M3x4 data cast(s). Data words go through
    ``q_base_ptr`` straight to gmem, predicated per vector.

    The amax uses SM100 3-input ``max.NaN`` chains: a NaN element survives
    the max tree and the cross-lane butterfly and reaches the E8M0 rceil cvt
    (NaN -> 0xFF, the byte a ``NaN -> inf`` substitution would produce), so
    each element costs one |x| plus half a max instead of an
    abs + NaN-test + select + max."""
    amax = _frag_amax_nan(src, j * vecsize, vecsize)
    amax = _reduce_amax_f32(amax, 1, reduce_stages, nan_propagate=True)
    scale_u8, recip = _compute_fp8_e8m0_scale_rceil(amax, Float32(1.0 / FP8_E4M3_MAX))
    copy_utils.store(
        sf_base_ptr + const_expr(col_off // _MX_SF_VEC_SIZE),
        scale_u8,
        pred=vec_ok and is_sf_lane,
    )
    data_pred = vec_ok
    if const_expr(vecsize == 8):
        word0 = _cvt_e4m3x4_word(src, j * 8, recip)
        word1 = _cvt_e4m3x4_word(src, j * 8 + 4, recip)
        copy_utils.store_v2(
            q_base_ptr + const_expr(col_off // 4), word0, word1, pred=data_pred
        )
    else:
        word0 = _cvt_e4m3x4_word(src, j * 4, recip)
        copy_utils.store(q_base_ptr + const_expr(col_off // 4), word0, pred=data_pred)


@cute.jit
def _quantize_frag_to_mxfp8(
    src: cute.Tensor,
    q_base_ptr,
    mdXSF: cute.Tensor,
    tXcX_cur: cute.Tensor,
    row: Int32,
    row_ok: Boolean,
    N: Int32,
    vecsize: cutlass.Constexpr[int],
    vec_col_stride: cutlass.Constexpr[int],
):
    """Quantize an FP32 dx fragment to MXFP8 (E4M3 data + E8M0 scales).

    Data bytes are packed with round-to-nearest E4M3 conversion and written as
    32/64-bit words straight to gmem through ``q_base_ptr``.
    Buffering them in a register fragment for a deferred tiled copy pushes
    the kernel past the 255-register cap and into spills. ``q_base_ptr``
    must be a Uint32-recast pointer at this thread's (row, col_base) with
    row-contiguous columns.
    """
    num_vecs = const_expr(cute.size(src) // vecsize)
    lanes_per_block = const_expr(_MX_SF_VEC_SIZE // vecsize)
    reduce_stages = const_expr(int(math.log2(lanes_per_block)))
    # One base address per row; per-vector offsets are compile-time constants
    # (vectors stride by threads_per_row * vecsize columns), so the stores use
    # [reg + imm] addressing instead of one live 64-bit chain per vector.
    col_base = tXcX_cur[0][1]
    sf_base_ptr = mdXSF.iterator + cute.crd2idx(
        (row, col_base // _MX_SF_VEC_SIZE), mdXSF.layout
    )
    is_sf_lane = col_base % _MX_SF_VEC_SIZE == 0
    for j in cutlass.range_constexpr(num_vecs):
        col_off = const_expr(j * vec_col_stride)
        vec_ok = row_ok and col_base + col_off < N
        _mxfp8_emit_vec(
            src,
            j,
            col_off,
            q_base_ptr,
            sf_base_ptr,
            is_sf_lane,
            vec_ok,
            vecsize,
            reduce_stages,
        )


# ---------------------------------------------------------------------------
# Forked backward kernel: FusedRMSNormBwd
# ---------------------------------------------------------------------------
# Divergences from upstream Quack:
#   - Fused gain_center: added in Float32 at each weight use so the backward
#     multiplies by the same (w + gain_center) the forward used; dw =
#     sum(dout * x_hat) is independent of the constant.
#   - Blackwell launch heuristic tuned from a broad CUDA-graph sweep (see
#     ``_blackwell_launch_config``); Hopper path keeps Quack's defaults.
#   - TMA tensor-tile load + PipelineTmaAsync (N-stage) for the gmem->smem
#     path, with a cp.async fallback for shapes that don't meet the TMA
#     16-byte inner-dim alignment.


class FusedRMSNormBwd(ReductionBase):
    """RMSNorm backward with fused gain_center, Blackwell-tuned heuristic,
    and a TMA-pipelined load path (with cp.async fallback).

    ``input_scale`` switches the derivative to RMSNorm of
    ``(w + gain_center) * x`` rather than an affine output weight.

    Optional ``dx_dtype=Float8E4M3FN`` emits dx as MXFP8: E4M3 data bytes
        plus NATURAL-layout ``[M, N // 32]`` E8M0 scale bytes (rceil scales,
        bit-compatible with the retained block-scaled quantization rule).
    """

    @staticmethod
    def dx_epilogue_vecsize(N: int, largest_dtype_width: int) -> int:
        """Per-thread vector width of the backward epilogues.

        Single source for the kernel's vecsize and the python wrapper's
        pre-launch validation of the MXFP8 vecsize-in-(4, 8) constraint.
        """
        return math.gcd(N, 128 // largest_dtype_width)

    @staticmethod
    def _make_row_tma_atom(op, m: cute.Tensor, tiler_mn):
        """Row-tile TMA atom with the row mode pre-split into (box, tN // box).

        The split (box = gcd(tN, 256), mirrored on the gmem view and the smem
        layout — the descriptor rank follows the gmem leaf modes) fits the
        whole (tile_m, tN) tile in one multi-dim TMA box, so each
        ``cute.copy`` lowers to a single ``cp.async.bulk.tensor`` instead of
        tN/box elect_one-guarded box copies whose issue cost hinges on an
        unreliable ptxas elect-merge. Smem bytes are laid out identically.
        Falls back to the flat (unrolled) form when the rest mode exceeds the
        256 box cap or box does not divide N.
        """
        tile_m, tN = tiler_mn
        M, N = m.shape
        box = math.gcd(tN, 256)
        rest = tN // box
        if rest == 1 or rest > 256 or N % box != 0:
            smem_layout = cute.make_ordered_layout(tiler_mn, order=(1, 0))
        else:
            smem_layout = cute.make_layout((tile_m, (box, rest)), stride=(tN, (1, box)))
            m = cute.make_tensor(
                m.iterator,
                cute.make_layout((M, (box, N // box)), stride=(N, (1, box))),
            )
        return cpasync.make_tiled_tma_atom(op, m, smem_layout, tiler_mn)

    @staticmethod
    @cute.jit
    def _dw_row_contrib(
        input_scale: cutlass.Constexpr,
        dx_inner,
        x,
        dout,
        x_hat,
    ):
        """Per-row dw contribution shared by quantized and plain accumulation.

        Keeping this derivative in one helper prevents the input-scale path from
        diverging between them again."""
        if const_expr(input_scale):
            return dx_inner * x
        return dout * x_hat

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        dout_dtype: Optional[Type[cutlass.Numeric]] = None,
        T_hint: int = 0,
        dx_dtype: Optional[Type[cutlass.Numeric]] = None,
        use_fused_norm_reductions: bool = False,
        input_scale: bool = False,
    ):
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)
        self.input_scale = input_scale
        # dx defaults to the input dtype; Float8E4M3FN selects the fused MXFP8
        # epilogue (its data bytes are launched through a Uint8 view).
        self.dx_dtype = dx_dtype if dx_dtype is not None else dtype
        self.quant_dx = self.dx_dtype is Float8E4M3FN
        self._is_blackwell = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 10
        )
        if self.quant_dx and N % _MX_SF_VEC_SIZE != 0:
            raise ValueError(
                f"MXFP8 dx requires N divisible by {_MX_SF_VEC_SIZE}, got {N}"
            )
        if self.quant_dx and not self._is_blackwell:
            raise ValueError("MXFP8 dx quantization requires SM100+")
        cfg = self._resolve_launch_config(
            N,
            in_bits=dtype.width,
            out_bits=dout_dtype.width if dout_dtype is not None else dtype.width,
            dx_bits=self.dx_dtype.width,
            T_hint=T_hint,
            is_blackwell=self._is_blackwell,
            use_fused_norm_reductions=use_fused_norm_reductions,
        )
        self.reload_wdy = cfg["reload_wdy"]
        self.reload_x = cfg["reload_x"]
        self._num_threads_val = cfg["num_threads"]
        self._threads_per_row_val = cfg["threads_per_row"]
        self._cluster_n_val = cfg["cluster_n"]
        self._tma_stages_val = cfg.get("tma_stages", 2)
        if self.N > 128 * 1024 and self.dtype.width >= 32:
            raise ValueError(
                "FusedRMSNormBwd does not support N > 128k with dtype >= 32 bits"
            )
        if not torch.cuda.is_available():
            raise ValueError("FusedRMSNormBwd requires CUDA")
        # Blackwell heuristic returns sm_mult; Hopper falls back to
        # get_sm_count's per-N table.
        sm_mult = cfg.get("sm_mult")
        if sm_mult is not None:
            base_sm = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            self.sm_count = max(1, int(base_sm * sm_mult))
        else:
            self.sm_count = self.get_sm_count(N, torch.cuda.current_device())
        # TMA tile load (cp.async.bulk.tensor.2d) handles single-row and
        # multi-row tiles uniformly with hardware OOB zfill; the only
        # constraint is 16-byte inner-dim alignment on gmem and smem
        # (tile_mn[1] * element_width % 16 == 0).
        dout_w = dout_dtype.width if dout_dtype is not None else dtype.width
        tN = N // max(1, self._cluster_n_val)
        row_bytes_x = tN * dtype.width // 8
        row_bytes_do = tN * dout_w // 8
        self.USE_TMA = row_bytes_x % 16 == 0 and row_bytes_do % 16 == 0

    @classmethod
    @jit_cache
    def compile(
        cls,
        N,
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        has_db_partial,
        dres_dtype,
        dres_out_dtype,
        has_dw_partial,
        T_hint=0,
        use_fused_norm_reductions=False,
        input_scale=False,
    ):
        """Compile and cache a launch-specialized backward kernel variant."""
        batch_sym, batch_partial_sym = cute.sym_int(), cute.sym_int()
        # A Float8E4M3FN dx selects the fused MXFP8 epilogue; its data bytes
        # are launched through a Uint8 view, so the fake dx uses the storage
        # dtype.
        quant_dx = dx_dtype is Float8E4M3FN
        dx_storage_dtype = Uint8 if quant_dx else dx_dtype
        all_dtypes = [dtype, dout_dtype, dx_dtype, dres_dtype, dres_out_dtype]
        div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
        x_cute, dout_cute, dx_cute, dres_out_cute, dres_cute = [
            fake_tensor(dt, (batch_sym, N), div)
            for dt in [dtype, dout_dtype, dx_storage_dtype, dres_out_dtype, dres_dtype]
        ]
        weight_cute = fake_tensor(weight_dtype, (N,), div)
        rstd_cute = fake_tensor(Float32, (batch_sym,))
        dx_sf_cute = (
            fake_tensor(Uint8, (batch_sym, N // _MX_SF_VEC_SIZE)) if quant_dx else None
        )
        dw_partial_cute = (
            fake_tensor(Float32, (batch_partial_sym, N), div)
            if has_dw_partial
            else None
        )
        db_partial_cute = (
            fake_tensor(Float32, (batch_partial_sym, N), div)
            if has_db_partial
            else None
        )
        return cute.compile(
            cls(
                dtype,
                N,
                dout_dtype=dout_dtype,
                T_hint=T_hint,
                dx_dtype=dx_dtype,
                use_fused_norm_reductions=use_fused_norm_reductions,
                input_scale=input_scale,
            ),
            x_cute,
            weight_cute,
            dout_cute,
            dres_out_cute,
            rstd_cute,
            dx_cute,
            dx_sf_cute,
            dw_partial_cute,
            dres_cute,
            db_partial_cute,
            Float32(0),  # gain_center
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )

    @classmethod
    def _resolve_launch_config(
        cls,
        N: int,
        *,
        in_bits: int,
        out_bits: int,
        dx_bits: int,
        T_hint: int,
        is_blackwell: bool,
        use_fused_norm_reductions: bool = False,
    ) -> dict:
        """Single source of the launch heuristic's parameter surface.

        Both ``__init__`` and ``resolve_sm_count`` resolve through here, so
        the dw_partial row count the wrapper allocates cannot drift from the
        persistent grid the kernel actually launches.
        """
        if is_blackwell:
            return cls._blackwell_launch_config(
                N,
                dtype_in=in_bits,
                dtype_out=out_bits,
                T_hint=T_hint,
                dtype_dx=dx_bits,
                use_fused_norm_reductions=use_fused_norm_reductions,
            )
        return cls._hopper_launch_config(N)

    @classmethod
    @lru_cache(maxsize=1024)
    def resolve_sm_count(
        cls,
        D: int,
        in_bits: int,
        out_bits: int,
        dx_bits: int,
        T_hint: int,
        device: torch.device,
        use_fused_norm_reductions: bool = False,
    ) -> int:
        """Resolve the backward launch's sm_count: the Blackwell heuristic's
        sm_mult when available, else the Hopper per-N get_sm_count table.

        Pure function of shape, dtype widths, T_hint, and device;
        cached so the eager hot path skips the config recompute and
        device-property queries. T_hint buckets naturally (one entry per
        unique M, mirroring the JIT cache).
        """
        cfg = cls._resolve_launch_config(
            D,
            in_bits=in_bits,
            out_bits=out_bits,
            dx_bits=dx_bits,
            T_hint=T_hint,
            is_blackwell=(
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(device)[0] >= 10
            ),
            use_fused_norm_reductions=use_fused_norm_reductions,
        )
        sm_mult = cfg.get("sm_mult")
        if sm_mult is not None:
            base_sm = torch.cuda.get_device_properties(device).multi_processor_count
            return max(1, int(base_sm * sm_mult))
        return cls.get_sm_count(D, device)

    @staticmethod
    def _epilogue_launch_config(
        N: int,
        dtype_in: int,
        dtype_out: int,
        T_hint: int,
        dtype_dx: int | None = None,
    ) -> dict:
        """Launch heuristic for the MXFP8 dx epilogue (SM10).

        Dtypes arrive as width bits like ``dtype_in``/``dtype_out``;
        ``dtype_dx == 8`` means the MXFP8 E4M3 epilogue and ``None`` means dx
        matches ``dtype_in``.

        Two fits: mixed-dtype rows (an fp32 operand somewhere) use the
        3.8k-point v17 registry shmoo (48 tier-1..6 shapes x 3 epilogue
        variants, CUDA-graph + L2-rotation timing, within 9% geomean of the
        per-shape oracle); pure-bf16 rows take a dedicated branch below.
        Mixed-dtype decisions derive from the per-row on-chip I/O traffic:

          ``row_kb = N * (in_bytes + dout_bytes) / 1024``  (x + dy stream)

        - ``tile_m`` (rows per TMA tile): batch narrow rows so one tile
          carries >= ~24 KB of in-flight bytes -- 4 rows while row_kb <= 6,
          2 rows while <= 24, else single-row.
        - ``cluster_n``: split only very wide rows. The per-thread fp32 dw
          accumulator holds ``N / (cn * tpr)`` elements; cn=2 above 48 KB
          rows and cn=8 above 56 KB keeps it under ~50 registers per thread.
          The quantized epilogue's register pressure starves 8-wide cluster
          co-scheduling on those wide rows, so cap it at cn=2.
        - reloads: none. The epilogue re-reads both x and w*dy at store time;
          an smem round-trip adds bandwidth without freeing enough registers.
          (Inheriting the plain kernel's large-T ``reload_x='smem'`` rule was
          the single biggest epilogue cliff -- up to 2x on the expert norms.)
        - ``sm_mult`` (persistence vs occupancy): with T >= 32K rows and
          narrow (<= 8 KB) rows the tiles are small and plentiful, so 4
          CTAs/SM overlap TMA latency; mid-T keeps the plain kernel's 2x;
          small-T very-wide rows (>= 48 KB, e.g. the D=12288 tp8 slice)
          saturate the row supply already -- 1x avoids launch overhead.
        """
        quant = dtype_dx == 8
        in_b, out_b = dtype_in // 8, dtype_out // 8
        row_kb = N * (in_b + out_b) / 1024
        is_pure_bf16 = dtype_in == 16 and dtype_out == 16
        if is_pure_bf16:
            # Pure 16-bit rows (the bf16-in/bf16-out production regime; fp16
            # shares the byte/register profile and takes the same bands): fit
            # on a dedicated 1.4k-point sweep of the 24 v17 tier norm+expert
            # shapes (2026-07-07, current kernel), geomean within 3% of the
            # per-cell oracle. Bands are COLUMN-denominated: the register
            # mechanism is the per-thread fp32 dw accumulator (N / (cn *
            # tpr)), and byte-denominated bands sent the D=12288 io=4 shape
            # into a 12x spill cliff (96 accum regs at cn=1). At io=4 one
            # smem reload returns as the register-pressure valve on specific
            # width bands (non-monotone in N -- empirical, from the sweep).
            tile_m = 4 if N <= 1024 else (2 if N <= 3584 else 1)
            cn = 1 if N <= 9216 else 2
            while cn > 1 and N % cn != 0:
                cn //= 2
            rl_x = None
            rl_wdy = "smem" if 1024 < N <= 1792 or 6144 < N <= 7168 else None
            sm_mult = 4.0 if (T_hint >= 32 * 1024 and N <= 1536) else 2.0
            return {
                "cluster_n": cn,
                "num_threads": 128,
                "threads_per_row": 128 // tile_m,
                "reload_wdy": rl_wdy,
                "reload_x": rl_x,
                "tma_stages": 2,
                "sm_mult": sm_mult,
            }
        tile_m = 4 if row_kb <= 6 else (2 if row_kb <= 24 else 1)
        cn = 1 if row_kb <= 48 else (2 if row_kb <= 56 else 8)
        if quant:
            cn = min(cn, 2)
        while cn > 1 and N % cn != 0:
            cn //= 2
        if T_hint >= 32 * 1024:
            sm_mult = 4.0 if row_kb <= 8 else 2.0
        elif T_hint >= 8 * 1024 or T_hint == 0:
            sm_mult = 2.0
        else:
            sm_mult = 1.0 if row_kb >= 48 else 2.0
        return {
            "cluster_n": cn,
            "num_threads": 128,
            "threads_per_row": 128 // tile_m,
            "reload_wdy": None,
            "reload_x": None,
            "tma_stages": 2,
            "sm_mult": sm_mult,
        }

    @staticmethod
    def _blackwell_launch_config(  # noqa: C901
        N: int,
        dtype_in: int = 32,
        dtype_out: int = 16,
        T_hint: int = 0,
        dtype_dx: int | None = None,
        use_fused_norm_reductions: bool = False,
    ) -> dict:
        """TMA launch heuristic for Blackwell (SM10).

        ``use_fused_norm_reductions`` opts into the retuned pure-16-bit plain
        launch bands. The retune changes the persistent grid and therefore
        the dw partial grouping, so it is part of the opt-in
        fused-norm-reductions numerics bundle; when False, pure-16-bit rows
        take the general bands below (bitwise-stable launch geometry).

        The MXFP8 dx epilogue (8-bit ``dtype_dx``) changes the store phase's
        register and compute profile enough to need its own heuristic; see
        :meth:`_epilogue_launch_config`.

        Tuned from a 4,140-point CUDA-graph and L2-rotation sweep plus an
        earlier 70-shape sweep for fallback bands. All decisions are
        byte-derived (``rb_in``, ``rb_io``, and ``T_hint``).

        Under cgraph measurement, this picks rank 1 in the shmoo oracle
        for four of five target shapes (pre_norm 69.1%, post_norm 52.6%,
        output_norm 62.2%, pre_expert 70.3%) and rank 2 within 0.3 pp of
        rank 1 for post_expert (80.9% vs 81.2%).

        Key findings baked in:
        - ``tile_m = 1`` with ``nt = tpr = 128`` wins for all wide shapes.
        - Aggressive ``cn`` (up to 8) shrinks per-CTA tile and raises
          occupancy from ~18% -> ~50% for the register-pressure-bound
          norm shapes (NCU-flagged bottleneck).
        - ``sm_mult = 0.5`` for wide D at small T (T=2K already has enough
          rows for 152 SMs; more CTAs just add launch overhead).
        - Large-T bf16 expert shapes always want ``reload_x='smem'``
          (drops regs/thread -> more warps/scheduler).
        """
        if dtype_dx == 8:
            return FusedRMSNormBwd._epilogue_launch_config(
                N,
                dtype_in,
                dtype_out,
                T_hint,
                dtype_dx=dtype_dx,
            )
        # TODO: drop the flag gate (keeping these bands) once a numerics
        # ladder derisks use_fused_norm_reductions and it becomes the default
        if use_fused_norm_reductions and dtype_in == 16 and dtype_out == 16:
            # Pure 16-bit plain path (bf16-in / bf16-RN-out production regime;
            # fp16 shares the byte/register profile and takes the same bands):
            # refit on the dedicated 1.4k-point 2026-07-07 sweep of the 24
            # v17 tier norm+expert shapes, geomean within 2% of the per-cell
            # oracle (up to 1.5x over the general bands below on the narrow
            # expert rows, which want 4-row tiles + 4x persistence). Bands
            # are column-denominated; reload bands are empirical
            # register-pressure valves from the sweep (non-monotone in N).
            tile_m = 4 if N <= 1792 else (2 if N <= 3584 else 1)
            cn = 1 if N <= 8192 else 2
            while cn > 1 and N % cn != 0:
                cn //= 2
            rl_x = (
                "smem" if N <= 1024 or 3072 < N <= 4608 or 9216 < N <= 12288 else None
            )
            rl_wdy = "smem" if 1024 < N <= 1536 or 6144 < N <= 7168 else None
            if T_hint >= 16 * 1024 and N <= 768:
                sm_mult = 4.0
            elif 0 < T_hint <= 4096 and 8192 < N <= 9216:
                sm_mult = 1.0
            else:
                sm_mult = 2.0
            return {
                "cluster_n": cn,
                "num_threads": 128,
                "threads_per_row": 128 // tile_m,
                "reload_wdy": rl_wdy,
                "reload_x": rl_x,
                "tma_stages": 2,
                "sm_mult": sm_mult,
            }
        is_bf16_in = dtype_in == 16
        is_pure_bf16 = is_bf16_in and dtype_out == 16
        is_bf16_to_f32 = is_bf16_in and dtype_out == 32
        in_b = dtype_in // 8
        out_b = dtype_out // 8
        rb_in = N * in_b
        rb_io = N * (in_b + out_b)
        small_T = (T_hint == 0) or (T_hint <= 2048)
        large_T = T_hint >= 8192

        # - cluster_n --------------------------
        # Aggressive cn=8 wins for measured D in [14K, 28K]. Gate it
        # on N>=14K so we don't apply it to mid-D shapes where the shmoo
        # didn't run  those regressed 5-8 pp under an ungated rule.
        # Outside that range, use the general table validated on mid-D shapes.
        if small_T and N >= 14 * 1024 and rb_io >= 48 * 1024:
            cn = 8
        elif is_bf16_in:
            # N in (8K, 12K] bf16->fp32 at cn=1 hits a register-pressure cliff
            # on the TMA path (~96 fp32 accum elem/thread); cn=2 avoids it
            # and matches master within noise.
            if N <= 8 * 1024:
                cn = 1
            elif N <= 16 * 1024:
                cn = 2
            else:
                cn = 8
        else:
            # FP32 input uses the established bands outside the wide-shape range.
            if N <= 8 * 1024:
                cn = 1
            elif N <= 16 * 1024:
                cn = 2
            else:
                cn = 8
        while cn > 1 and N % cn != 0:
            cn //= 2

        # - num_threads / threads_per_row ----------------
        # nt=tpr=128 single-row (tile_m=1) wins for all measured wide
        # shapes. Only very-small bf16_bf16 uses multi-row.
        if is_pure_bf16 and N <= 1536:
            nt, tpr = 128, 32  # tile_m=4 (small bf16_bf16 expert)
        elif is_pure_bf16 and N <= 2560:
            nt, tpr = 128, 64  # tile_m=2
        else:
            nt, tpr = 128, 128  # tile_m=1 single-row (shmoo winner)

        # - reload_wdy -------------------------
        # bf16_in small-T: mid-D band wants smem reload; large-T prefers
        # rl_x instead (shmoo: pre_expert's rl_wdy=None, rl_x=smem wins).
        # FP32 input uses SMEM reload at very wide D (measured at D=28K).
        if is_bf16_in:
            if large_T:
                rl_wdy = None  # expert shapes: rl_x instead
            elif N <= 1536 or 4608 <= N <= 8192:
                rl_wdy = "smem"
            else:
                rl_wdy = None
        else:
            # fp32 input: smem reload at very wide D or 6K mid-band.
            rl_wdy = "smem" if (N >= 24 * 1024 or 6144 <= N <= 8192) else None

        # - reload_x --------------------------
        # rl_x=smem wins for large-T BF16 expert shapes
        # (pre/post_expert at D=7168) cuts register pressure. Also for
        # very-wide fp32 pre_norm and small bf16 shapes
        if is_pure_bf16:
            if N <= 2560 or N == 7168 or large_T:
                rl_x = "smem"
            else:
                rl_x = None
        elif is_bf16_to_f32:
            if N <= 768 or large_T:
                rl_x = "smem"
            else:
                rl_x = None
        else:
            # FP32 input reloads x from SMEM at very wide D.
            rl_x = "smem" if (N >= 24 * 1024 or N == 4096) else None

        # - tma_stages -------------------------
        # Shmoo: 2 stages optimal for ~95% of shapes. KISS.
        stages = 2

        # - sm_mult --------------------------─
        # Wide small-T shapes prefer a lean sm_count (0.5-1.0x base)
        # shmoo showed extra CTAs just add launch overhead once T=2K rows
        # already saturate the ~152 SMs. Mid-D small-T still wants master's
        # base*2 boost (the Blackwell path in get_sm_count). Very small D
        # ramps the multiplier up to keep the grid populated.
        # Large-T expert shapes use the measured 2.0 multiplier.
        if T_hint == 0:
            sm_mult = 2.0
        elif small_T:
            if rb_in <= 1024:
                sm_mult = 16.0
            elif rb_in <= 2 * 1024:
                sm_mult = 8.0
            elif rb_in <= 4 * 1024:
                sm_mult = 4.0
            elif N >= 14 * 1024:
                # Sweep-tuned lean SM count for wide shapes.
                sm_mult = 0.5 if rb_in <= 64 * 1024 else 1.0
            else:
                sm_mult = 2.0  # master's base*2 for mid-D small-T
        else:
            sm_mult = 2.0  # large-T expert

        return {
            "cluster_n": cn,
            "num_threads": nt,
            "threads_per_row": tpr,
            "reload_wdy": rl_wdy,
            "reload_x": rl_x,
            "tma_stages": stages,
            "sm_mult": sm_mult,
        }

    @staticmethod
    def _hopper_launch_config(N: int) -> dict:
        """Upstream Quack defaults for Hopper and earlier."""
        # cluster_n
        for limit, candidate_cn in [
            (8192, 1),
            (16384, 2),
            (32768, 4),
            (65536, 8),
        ]:
            if N <= limit:
                cn = candidate_cn
                break
        else:
            cn = 16
        # num_threads / threads_per_row
        if N <= 4096:
            nt = 128
        else:
            nt = 256
        for limit, candidate_tpr in [
            (64, 8),
            (128, 16),
            (256, 32),
            (512, 64),
            (4096, 128),
        ]:
            if N <= limit:
                tpr = candidate_tpr
                break
        else:
            tpr = 256
        return {
            "cluster_n": cn,
            "num_threads": nt,
            "threads_per_row": tpr,
            "reload_wdy": None if N <= 16 * 1024 else "smem",
            "reload_x": None,
        }

    @staticmethod
    def get_sm_count(N: int, device) -> int:
        """Number of CTAs to launch for this N on the given device.

        Used as the fallback when the heuristic doesn't return ``sm_mult``
        (e.g. the Hopper path); the Blackwell path drives ``sm_count``
        directly from ``cfg['sm_mult']``.
        """
        props = torch.cuda.get_device_properties(device)
        base = props.multi_processor_count
        is_blackwell = props.major >= 10
        if is_blackwell and 4 * 1024 < N <= 16 * 1024:
            return base * 2
        # Upstream Quack heuristic (also Blackwell outside the override range):
        if N <= 8192:
            multiplier = (
                16
                if N <= 256
                else 8
                if N <= 1024
                else 4
                if N <= 2048
                else 2
                if N <= 4096
                else 1
            )
            return base * multiplier
        if N <= 16 * 1024:
            return base // 2
        return base * 2

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
        mdO: cute.Tensor,
        mdResO: Optional[cute.Tensor],
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdXSF: Optional[cute.Tensor],
        mdW: Optional[cute.Tensor],
        mdRes: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        gain_center: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                *(
                    t.element_type.width
                    for t in [mX, mW, mdO, mdResO, mdX, mdRes]
                    if t is not None
                )
            )
        )
        vecsize = self.dx_epilogue_vecsize(self.N, largest_dtype_width)
        if const_expr(self.quant_dx):
            # The MXFP8 lane-group amax assumes vectors of 4 or 8 consecutive
            # columns. The Python wrapper validates this before launch.
            # The python wrapper pre-validates this via dx_epilogue_vecsize.
            assert vecsize in (4, 8), (
                f"MXFP8 dx epilogues require a 4- or 8-element vector size, "
                f"got {vecsize} for N={self.N}"
            )
        if const_expr(self.quant_dx):
            assert mdXSF is not None and mdXSF.element_type == Uint8
            assert mdX.element_type == Uint8
            # The epilogue hoists the scale-lane predicate out of the vector
            # loop, which requires the per-vector column stride to preserve
            # col % 32 alignment.
            assert (self._threads_per_row_val * vecsize) % _MX_SF_VEC_SIZE == 0
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mW = (
            layout_utils.expand(mW, dim=0, size=tiler_mn[0])
            if const_expr(mW is not None)
            else None
        )
        # Build TMA atoms for X and dO when the TMA path is active. Each
        # atom carries the per-stage smem layout; the TMA descriptor encodes
        # the gmem extent so OOB lanes are zero-filled at runtime.
        if const_expr(self.USE_TMA):
            tma_op = cpasync.CopyBulkTensorTileG2SOp()
            tma_atom_X, mX_tma = self._make_row_tma_atom(tma_op, mX, tiler_mn)
            tma_atom_dO, mdO_tma = self._make_row_tma_atom(tma_op, mdO, tiler_mn)
        else:
            tma_atom_X, mX_tma, tma_atom_dO, mdO_tma = None, None, None, None
        num_blocks = self.sm_count
        self.kernel(
            mX,
            mW,
            mdO,
            mdResO,
            mRstd,
            mdX,
            mdXSF,
            mdW,
            mdB,
            mdRes,
            gain_center,
            tma_atom_X,
            mX_tma,
            tma_atom_dO,
            mdO_tma,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901 -- constexpr-specialized load/epilogue paths keep the hot loop inline
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mdResO: Optional[cute.Tensor],
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdXSF: Optional[cute.Tensor],
        mdW: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        mdRes: Optional[cute.Tensor],
        gain_center: Float32,
        tma_atom_X: Optional[cute.CopyAtom],
        mX_tma: Optional[cute.Tensor],
        tma_atom_dO: Optional[cute.CopyAtom],
        mdO_tma: Optional[cute.Tensor],
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx_uniform = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        M = shape[0]
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        # TMA path uses self._tma_stages_val stages; the cp.async fallback
        # keeps the original 2-stage ping-pong. Per-stage smem cost is
        # tiler_mn[0] x tiler_mn[1] x (in+out) bytes.
        n_smem_stages = const_expr(self._tma_stages_val if self.USE_TMA else 2)
        smem_layout = cute.make_ordered_layout(
            (tiler_mn[0], tiler_mn[1], n_smem_stages), order=(1, 0, 2)
        )
        # TMA requires 128-byte aligned smem destination.
        smem_align = const_expr(128 if self.USE_TMA else 16)
        sX = smem.allocate_tensor(
            mX.element_type, smem_layout, byte_alignment=smem_align
        )
        sdO = smem.allocate_tensor(
            mdO.element_type, smem_layout, byte_alignment=smem_align
        )
        # The cluster-exchange barriers live in the reserved smem partition,
        # allocated and initialized by PipelineStasAsync below.
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
            byte_alignment=8,
        )

        thr_copy_X = tiled_copy.get_slice(tidx)

        gX, gdO, gdResO, gdX, gdRes, cX = [
            cute.local_tile(mT, tiler_mn, (None, cluster_y)) if mT is not None else None
            for mT in (mX, mdO, mdResO, mdX, mdRes, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if mW is not None else None
        gdW, gdB = [
            cute.local_tile(mT, (1, tiler_mn[1]), (bidx_start, cluster_y))
            if const_expr(mT is not None)
            else None
            for mT in (mdW, mdB)
        ]

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgdO = thr_copy_X.partition_S(gdO)
        tXsdO = thr_copy_X.partition_D(sdO)
        tXgdX = thr_copy_X.partition_D(gdX)
        if const_expr(mdResO is not None):
            tXgdResO = thr_copy_X.partition_S(gdResO)
        if const_expr(mdRes is not None):
            tXgdRes = thr_copy_X.partition_D(gdRes)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]

        tXrX, tXrdO = [
            cute.make_fragment_like(thr[None, None, None, 0]) for thr in (tXgX, tXgdO)
        ]
        # The MXFP8 epilogue streams qdata words straight to gmem, so it needs
        # no dx register fragment.
        tXrdX = (
            None
            if const_expr(self.quant_dx)
            else cute.make_fragment_like(tXgdX[None, None, None, 0])
        )
        tXrdResO = None
        if const_expr(mdResO is not None):
            tXrdResO = cute.make_fragment_like(tXgdResO[None, None, None, 0])
        tXrdRes = None
        if const_expr(mdRes is not None):
            tXrdRes = cute.make_fragment_like(tXgdRes[None, None, None, 0])

        # This doesn't change across iterations
        tXpX = (
            None
            if is_even_N
            else copy_utils.predicate_k(
                thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1]
            )
        )
        # Each copy will use the same number of elements as X
        copy = partial(copy_utils.copy, pred=tXpX)

        tXgdW, tXrdW = None, None
        tXgdB, tXrdB = None, None
        if const_expr(mdW is not None):
            tXgdW = thr_copy_X.partition_S(gdW)
            # Always compute partial weight gradients in fp32
            tXrdW = cute.make_fragment_like(tXgdW, Float32)
        if const_expr(mdB is not None):
            tXgdB = thr_copy_X.partition_S(gdB)
            # Always compute partial bias gradients in fp32
            tXrdB = cute.make_fragment_like(tXgdB, Float32)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

        USE_TMA = const_expr(self.USE_TMA)

        if const_expr(USE_TMA):
            # PipelineTmaAsync needs num_stages * 2 mbarriers (full + empty).
            tma_mbar_ptr = smem.allocate_array(Int64, num_elems=n_smem_stages * 2)

        # Reduction-buffer exchange across the cluster (see PipelineStasAsync):
        # this CTA is both producer and consumer of the per-stage slots. Arrive
        # now; the matching cluster_wait comes after the weight load below so
        # the cluster sync overlaps it.
        pipeline_reduce = None
        if const_expr(self.cluster_n > 1):
            pipeline_reduce = PipelineStasAsync.create(
                num_stages=self.stage,
                num_warps=num_warps,
                cluster_n=self.cluster_n,
                defer_sync=True,
            )
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()

        tma_bytes_x = const_expr(tiler_mn[0] * tiler_mn[1] * mX.element_type.width // 8)
        tma_bytes_do = const_expr(
            tiler_mn[0] * tiler_mn[1] * mdO.element_type.width // 8
        )
        tma_bytes_total = const_expr(tma_bytes_x + tma_bytes_do)

        tXrW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            if const_expr(not is_even_N):
                tXrW.fill(0.0)
            copy(tXgW, tXrW)
            # gain_center is fused at each use in Float32 (below), matching the
            # forward's ``w.to(Float32) + gain_center``. Pre-adding it into the
            # weight registers here would first round ``w + gain_center`` back to
            # the (low-precision) weight dtype, so the backward would multiply by
            # a different effective weight than the forward used and emit a
            # gradient inconsistent with the forward function.

        vecsize_dx = const_expr(cute.size(tXrX, mode=[0]))

        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        # Smem-staging pipeline states, shared by the TMA and cp.async load
        # paths (the cp.async path has no barriers to drive; the states only
        # track the stage index).
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, n_smem_stages
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, n_smem_stages
        )

        # - TMA pipeline setup ----------------------─
        if const_expr(USE_TMA):
            num_threads_total = cute.size(tiled_copy)
            producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
            consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_threads_total
            )
            tma_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=tma_mbar_ptr,
                num_stages=n_smem_stages,
                producer_group=producer_group,
                consumer_group=consumer_group,
                tx_count=tma_bytes_total,
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            # Re-tile gmem for TMA, then partition both gmem/smem so
            # cute.copy(tma_atom, gX_tma[None, idx], sX_tma[None, stage], ...)
            # works. These TMA-partitioned views are only used on the producer
            # side; the consumer (autovec_copy into registers) keeps the
            # tiled_copy partitioning of tXsX/tXsdO from above, since both
            # views alias the same underlying smem allocation.
            gX_tma = cute.local_tile(mX_tma, tiler_mn, (None, cluster_y))
            gdO_tma = cute.local_tile(mdO_tma, tiler_mn, (None, cluster_y))
            tXsX_tma, tXgX_tma = cpasync.tma_partition(
                tma_atom_X,
                0,
                cute.make_layout(1),
                cute.group_modes(sX, 0, 2),
                cute.group_modes(gX_tma, 0, 2),
            )
            tXsdO_tma, tXgdO_tma = cpasync.tma_partition(
                tma_atom_dO,
                0,
                cute.make_layout(1),
                cute.group_modes(sdO, 0, 2),
                cute.group_modes(gdO_tma, 0, 2),
            )

        # - Prefetch the first n_smem_stages - 1 batches --------------------
        # M_ceil is the number of row-tiles after rounding for tiler_mn[0].
        # NB: the prologue's exact shape is register-load-bearing. The
        # register-critical RN-quant epilogue sits on an occupancy cliff, and
        # moving the prologue advance under the warp-0 branch (upstream
        # quack's form) shifts ptxas allocation past 128 regs/thread — a
        # measured 125 -> 140 regs, 4 -> 3 CTAs/SM, 1.46x slowdown at
        # T=128K D=1536. Keep the advance mirrored by all warps.
        M_ceil = cute.ceil_div(M, tiler_mn[0])
        if const_expr(USE_TMA):
            # Warp 0 issues the TMA loads; pipeline.producer_acquire does the
            # arrive_and_expect_tx via elect_one internally. Only the
            # prologue advance is mirrored by all warps — from the main loop
            # on, warp 0 alone maintains producer_state, so other warps'
            # copies are stale and must not be read.
            for prefetch_iter in cutlass.range_constexpr(const_expr(n_smem_stages - 1)):
                init_bidx = bidx_start + prefetch_iter * gdim
                if init_bidx < M_ceil:
                    if warp_idx_uniform == 0:
                        tma_pipeline.producer_acquire(producer_state)
                        pipe_bar = tma_pipeline.producer_get_barrier(producer_state)
                        cute.copy(
                            tma_atom_X,
                            tXgX_tma[None, init_bidx],
                            tXsX_tma[None, producer_state.index],
                            tma_bar_ptr=pipe_bar,
                        )
                        cute.copy(
                            tma_atom_dO,
                            tXgdO_tma[None, init_bidx],
                            tXsdO_tma[None, producer_state.index],
                            tma_bar_ptr=pipe_bar,
                        )
                        tma_pipeline.producer_commit(producer_state)
                    producer_state.advance()
        else:
            # Pre-issue n_smem_stages - 1 prefetches; the bidx loop then keeps
            # exactly n_smem_stages groups in flight via
            # cp_async_wait_group(n_smem_stages - 1).
            for prefetch_iter in cutlass.range_constexpr(const_expr(n_smem_stages - 1)):
                init_bidx = bidx_start + prefetch_iter * gdim
                init_row = tXcX[None, None, None, init_bidx][0][0]
                if init_row < M:
                    copy(
                        tXgX[None, None, None, init_bidx],
                        tXsX[None, None, None, producer_state.index],
                        is_async=True,
                    )
                    copy(
                        tXgdO[None, None, None, init_bidx],
                        tXsdO[None, None, None, producer_state.index],
                        is_async=True,
                    )
                else:
                    if const_expr(tiler_mn[0] > 1):
                        utils.fill_oob(
                            tXsX[None, None, None, producer_state.index],
                            None,
                            fill_value=mX.element_type.zero,
                        )
                        utils.fill_oob(
                            tXsdO[None, None, None, producer_state.index],
                            None,
                            fill_value=mdO.element_type.zero,
                        )
                cute.arch.cp_async_commit_group()
                producer_state.advance()

        if const_expr(mdW is not None):
            tXrdW.fill(0.0)
        if const_expr(mdB is not None):
            tXrdB.fill(0.0)
        # Reduction-slot toggle. A bare Int32, not a PipelineState: the
        # Register-limited MXFP8 epilogue configs measurably lose occupancy
        # to extra loop-carried pipeline-state registers (7% on the T=64K
        # D=2560 BF16-output shape). Only the cluster path needs the state pair —
        # the producer view gates publishing on the empty barriers, the
        # consumer view tracks the full-barrier phase; both advance in
        # lockstep, one stage per row iteration.
        red_slot = Int32(0)
        if const_expr(self.cluster_n > 1):
            prod_state_reduce = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stage
            )
            cons_state_reduce = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.stage
            )
        next_wave_work_id = (n_smem_stages - 1) * gdim
        next_wave_row_id = next_wave_work_id * tiler_mn[0]
        # --- Main persistent loop ---
        # Compute body is structurally the same as upstream RMSNormBackward;
        # the load side is either the TMA pipeline (preferred) or the cp.async
        # staging selected by USE_TMA.
        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]
            ahead_bidx = bidx + next_wave_work_id
            if const_expr(USE_TMA):
                # Warp 0 prefetches into the stage we're about to free; skip
                # past the last tile this CTA owns (no consumer would drain it).
                if warp_idx_uniform == 0:
                    if ahead_bidx < M_ceil:
                        tma_pipeline.producer_acquire(producer_state)
                        pipe_bar = tma_pipeline.producer_get_barrier(producer_state)
                        cute.copy(
                            tma_atom_X,
                            tXgX_tma[None, ahead_bidx],
                            tXsX_tma[None, producer_state.index],
                            tma_bar_ptr=pipe_bar,
                        )
                        cute.copy(
                            tma_atom_dO,
                            tXgdO_tma[None, ahead_bidx],
                            tXsdO_tma[None, producer_state.index],
                            tma_bar_ptr=pipe_bar,
                        )
                        tma_pipeline.producer_commit(producer_state)
                        producer_state.advance()
            else:
                if row + next_wave_row_id < M:
                    copy(
                        tXgX[None, None, None, ahead_bidx],
                        tXsX[None, None, None, producer_state.index],
                        is_async=True,
                    )
                    copy(
                        tXgdO[None, None, None, ahead_bidx],
                        tXsdO[None, None, None, producer_state.index],
                        is_async=True,
                    )
                else:
                    if const_expr(tiler_mn[0] > 1):
                        utils.fill_oob(
                            tXsX[None, None, None, producer_state.index],
                            None,
                            fill_value=mX.element_type.zero,
                        )
                        utils.fill_oob(
                            tXsdO[None, None, None, producer_state.index],
                            None,
                            fill_value=mdO.element_type.zero,
                        )
                cute.arch.cp_async_commit_group()
                producer_state.advance()
            rstd = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                rstd = mRstd[row]
            if const_expr(mdResO is not None):
                if row < M or tiler_mn[0] == 1:
                    copy(tXgdResO[None, None, None, bidx], tXrdResO)
                elif tiler_mn[0] > 1:
                    tXrdResO.fill(0.0)
            if const_expr(USE_TMA):
                tma_pipeline.consumer_wait(consumer_state)
            else:
                cute.arch.cp_async_wait_group(const_expr(n_smem_stages - 1))
            smem_stage = consumer_state.index
            cute.autovec_copy(tXsX[None, None, None, smem_stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, smem_stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)
            wdy = dout
            if const_expr(self.input_scale and mW is not None):
                x_hat = x * (tXrW.load().to(Float32) + gain_center) * rstd
            else:
                x_hat = x * rstd
                if const_expr(mW is not None):
                    wdy *= tXrW.load().to(Float32) + gain_center
            if const_expr(self.cluster_n > 1):
                pipeline_reduce.producer_acquire(prod_state_reduce)
            red_mbar = (
                pipeline_reduce.producer_get_barrier(prod_state_reduce)
                if const_expr(self.cluster_n > 1)
                else None
            )
            mean_xhat_wdy = (
                row_reduce(
                    _fma_dot_f32(x_hat, wdy),
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, red_slot],
                    red_mbar,
                    phase=(
                        cons_state_reduce.phase
                        if const_expr(self.cluster_n > 1)
                        else None
                    ),
                    init_val=0.0,
                )
                / shape[1]
            )

            if const_expr(self.cluster_n > 1):
                # The STAS stores travel the async proxy; fence before
                # releasing so peers' subsequent writes to the slot cannot
                # pass our reads.
                cute.arch.fence_view_async_shared()
                pipeline_reduce.consumer_release(cons_state_reduce)

            if const_expr(self.reload_wdy == "smem"):
                cute.autovec_copy(tXsdO[None, None, None, smem_stage], tXrdO)
                dout = tXrdO.load().to(cute.Float32)
                wdy = dout
                if const_expr(mW is not None and not self.input_scale):
                    wdy *= tXrW.load().to(Float32) + gain_center

            if const_expr(self.reload_x == "smem"):
                cute.autovec_copy(tXsX[None, None, None, smem_stage], tXrX)
                x = tXrX.load().to(cute.Float32)
                x_hat = x * rstd
                if const_expr(self.input_scale and mW is not None):
                    x_hat *= tXrW.load().to(Float32) + gain_center

            dx_inner = (wdy - x_hat * mean_xhat_wdy) * rstd
            if const_expr(self.input_scale and mW is not None):
                dx = dx_inner * (tXrW.load().to(Float32) + gain_center)
            else:
                dx = dx_inner
            if const_expr(mdResO is not None):
                dx += tXrdResO.load().to(cute.Float32)
            # The quantized epilogue is a long instruction sequence; accumulate
            # dw/db first so dout and x_hat retire before the epilogue instead
            # of staying live across it (which pushes the kernel into register
            # spills). Bitwise identical: the per-accumulator FP order is
            # unchanged. The plain-RN path keeps its original order.
            dwdb_accumulated = const_expr(self.quant_dx)
            if const_expr(dwdb_accumulated):
                if const_expr(mdW is not None):
                    tXrdW.store(
                        tXrdW.load()
                        + self._dw_row_contrib(
                            self.input_scale, dx_inner, x, dout, x_hat
                        )
                    )
                if const_expr(mdB is not None):
                    tXrdB.store(tXrdB.load() + dout)
            if const_expr(self.quant_dx):
                tXrdXf = cute.make_fragment_like(tXrX, Float32)
                tXrdXf.store(dx)
                col_base_q = tXcX[None, None, None, bidx][0][1]
                q_ptr = cute.recast_ptr(
                    mdX.iterator + cute.crd2idx((row, col_base_q), mdX.layout),
                    dtype=Uint32,
                )
                _quantize_frag_to_mxfp8(
                    tXrdXf,
                    q_ptr,
                    mdXSF,
                    tXcX[None, None, None, bidx],
                    row,
                    row < M,
                    shape[1],
                    vecsize_dx,
                    const_expr(threads_per_row * vecsize_dx),
                )
            else:
                tXrdX.store(dx.to(tXrdX.element_type))
                if row < M or tiler_mn[0] == 1:
                    copy(tXrdX, tXgdX[None, None, None, bidx])
            if const_expr(mdRes is not None):
                tXrdRes.store(dx.to(tXrdRes.element_type))
                if row < M or tiler_mn[0] == 1:
                    copy(tXrdRes, tXgdRes[None, None, None, bidx])
            if const_expr(mdW is not None and not dwdb_accumulated):
                tXrdW.store(
                    tXrdW.load()
                    + self._dw_row_contrib(self.input_scale, dx_inner, x, dout, x_hat)
                )
            if const_expr(mdB is not None and not dwdb_accumulated):
                tXrdB.store(tXrdB.load() + dout)

            # Release the smem stage back to the producer. All threads
            # arrive (consumer_group sized to num_threads_total).
            if const_expr(USE_TMA):
                tma_pipeline.sync_object_empty.arrive(
                    consumer_state.index, tma_pipeline.consumer_mask
                )
            consumer_state.advance()
            if const_expr(self.cluster_n > 1):
                prod_state_reduce.advance()
                cons_state_reduce.advance()
            red_slot ^= 1

        # No post-loop drain needed: producer skips the last prefetch (no
        # consumer would drain it), so producer/consumer states stay in sync.

        if const_expr(tiler_mn[0] > 1):
            if const_expr(mdW is not None):
                # reduction of dw_partial within the same threadblock
                sdW = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdW = thr_copy_X.partition_D(sdW)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdW, tXsdW)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdW_other = cute.make_fragment_like(tXrdW)
                        tXsdW_other = cute.make_tensor(
                            tXsdW.iterator + i * sdW.stride[0], tXsdW.layout
                        )
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    copy(tXrdW, tXgdW)
                cute.arch.barrier()
            if const_expr(mdB is not None):
                sdB = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdB = thr_copy_X.partition_D(sdB)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdB, tXsdB)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdB_other = cute.make_fragment_like(tXrdB)
                        tXsdB_other = cute.make_tensor(
                            tXsdB.iterator + i * sdB.stride[0], tXsdB.layout
                        )
                        cute.autovec_copy(tXsdB_other, tXrdB_other)
                        tXrdB.store(tXrdB.load() + tXrdB_other.load())
                    copy(tXrdB, tXgdB)
        else:
            # dw is already in fp32, so we can directly copy to global memory
            if const_expr(mdW is not None):
                copy(tXrdW, tXgdW)
            if const_expr(mdB is not None):
                copy(tXrdB, tXgdB)

        if const_expr(self.cluster_n > 1):
            # Drain ALL stages before CTA exit (see producer_tail): remote
            # empty arrives to different barriers are unordered, so draining
            # only the last-used stage would race an in-flight earlier arrive.
            pipeline_reduce.producer_tail(prod_state_reduce)
