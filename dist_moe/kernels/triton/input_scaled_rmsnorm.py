# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused learned input-scale RMSNorm and top-k reduction for BF16 routes."""

import triton
import triton.language as tl

from ..._activation_buffer_planner_kernel import MEMORY_ALIGNMENT_TL_CONSTEXPR


@triton.jit
def _saved_routes(SavedBytes, SavedOffset):
    """Return the BF16 route rows saved at ``SavedOffset`` bytes in the buffer."""
    saved = (SavedBytes + tl.load(SavedOffset)).to(tl.pointer_type(tl.bfloat16))
    align: tl.constexpr = MEMORY_ALIGNMENT_TL_CONSTEXPR // 2
    return tl.multiple_of(saved, align)


@triton.jit
def route_forward(
    Raw,
    Gain,
    Scores,
    ExpertIds,
    Output,
    SavedBytes,
    SavedOffset,
    SkipSave,
    D: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    GAIN_CENTER: tl.constexpr,
):
    """Normalize one token's routes with learned input scale and reduce them.

    When ``SavedBytes`` is given and ``SkipSave`` is zero, the valid route rows
    are also copied into the activation buffer for backward.
    """
    token = tl.program_id(0).to(tl.int64)
    routes = token * K + tl.arange(0, K)
    cols = tl.arange(0, D)
    offsets = routes[:, None] * D + cols[None, :]
    valid = tl.load(ExpertIds + routes) >= 0
    raw = tl.load(Raw + offsets, valid[:, None], 0)
    if SavedBytes is not None:
        if tl.load(SkipSave) == 0:
            tl.store(
                _saved_routes(SavedBytes, SavedOffset) + offsets, raw, valid[:, None]
            )
    raw = raw.to(tl.float32)
    gain = tl.load(Gain + cols).to(tl.float32) + GAIN_CENTER
    scaled = raw * gain[None, :]
    rstd = tl.rsqrt(tl.sum(scaled * scaled, 1) * (1.0 / D) + EPS)
    normalized = (scaled * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    score = tl.load(Scores + routes, valid, 0).to(tl.float32)
    result = tl.sum(tl.where(valid[:, None], normalized * score[:, None], 0.0), 0)
    tl.store(Output + token * D + cols, result)


@triton.jit(do_not_specialize=["num_routes"])
def route_backward(
    Raw,
    SavedBytes,
    SavedOffset,
    NeedReplay,
    Upstream,
    Gain,
    Scores,
    ExpertIds,
    PublishedGradient,
    ScoreGradient,
    GainPartials,
    num_routes,
    D: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    EPS: tl.constexpr,
    GAIN_CENTER: tl.constexpr,
):
    """Reverse the normalized routes; program p owns blocks p, p + P, ...

    Each program accumulates its gain gradient in registers and writes one
    FP32 partial row, so the gain reduction order depends only on the grid.
    ``GainPartials`` is ``None`` for a frozen gain.
    """
    program = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, D)
    gain = tl.load(Gain + cols).to(tl.float32) + GAIN_CENTER
    raw_base = Raw
    if tl.load(NeedReplay) == 0:
        raw_base = _saved_routes(SavedBytes, SavedOffset)
    dgain = tl.zeros((D,), tl.float32)
    for block in range(program, tl.cdiv(num_routes, ROWS), num_programs):
        rows = (block * ROWS + tl.arange(0, ROWS)).to(tl.int64)
        offsets = rows[:, None] * D + cols[None, :]
        present = rows < num_routes
        valid = tl.load(ExpertIds + rows, present, -1) >= 0
        raw = tl.load(raw_base + offsets, valid[:, None], 0).to(tl.float32)
        dy = tl.load(
            Upstream + (rows[:, None] // K) * D + cols[None, :], valid[:, None], 0
        ).to(tl.float32)
        score = tl.load(Scores + rows, valid, 0).to(tl.float32)

        scaled = raw * gain[None, :]
        rstd = tl.rsqrt(tl.sum(scaled * scaled, 1) * (1.0 / D) + EPS)
        unit = scaled * rstd[:, None]
        rounded_unit = unit.to(tl.bfloat16).to(tl.float32)
        dscore = tl.sum(rounded_unit * dy, 1)

        dunit = (dy * score[:, None]).to(tl.bfloat16).to(tl.float32)
        projection = tl.sum(dunit * unit, 1)
        dscaled = (dunit - (unit * (1.0 / D)) * projection[:, None]) * rstd[:, None]
        draw = dscaled * gain[None, :]
        if GainPartials is not None:
            dgain += tl.sum(tl.where(valid[:, None], dscaled * raw, 0.0), 0)

        tl.store(ScoreGradient + rows, tl.where(valid, dscore, 0.0), present)
        # PublishedGradient may alias Raw. Finish this block's loads first.
        tl.debug_barrier()
        tl.store(
            PublishedGradient + offsets,
            tl.where(valid[:, None], draw, 0.0),
            present[:, None],
        )
    if GainPartials is not None:
        tl.store(GainPartials + program * D + cols, dgain)


@triton.jit(do_not_specialize=["num_partials"])
def reduce_gain(
    GainPartials,
    GainGradient,
    num_partials,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Sum the per-program gain partials in a fixed order."""
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    total = tl.zeros((BLOCK_C,), tl.float32)
    for start in range(0, num_partials, BLOCK_P):
        rows = start + tl.arange(0, BLOCK_P)
        partials = tl.load(
            GainPartials + rows[:, None] * D + cols[None, :],
            rows[:, None] < num_partials,
            0.0,
        )
        total += tl.sum(partials, 0)
    tl.store(GainGradient + cols, total)
