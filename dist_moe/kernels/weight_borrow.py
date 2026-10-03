# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stream borrowed expert weights through the chunked-mega forward kernel.

Inference-prefill expert replication: appended groups ``g >= num_local`` hold
foreign experts whose weights live in the owner's publish window, not in this
rank's weight tables. The dispatch producer warps drain a small queue of
weight-fetch work items — pulling each borrowed slot's rows over NVLink into
a local slot buffer — before they start token work, releasing per-(slot,
GEMM) done counters. The weight (A-operand) TMA warp gates each appended
group on those counters; local groups keep their ungated loads. Everything
here is the same handshake tokens use: local counters, gpu scope, writer-side
async-proxy fence before the release.

Window visibility needs no flag: the owner published *all* of its experts
before the layer's EP barrier (see ``interfaces/expert_borrow/
inference_window.py``), so a borrower may pull at any time.

The slot buffer mirrors the window's segment layout for ``max_slots`` rows:
per-group device tensormaps point the appended groups' A/SFA descriptors at
its rows, so no contiguity with the local weight tables is needed.

Counters ride the activation done-counter region (the ring counters set the
precedent): one work counter plus two done counters per slot, zeroed by the
prepare kernel's existing sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .dispatch_quant import (
    _dispatch_quant_fetch_group_tile_id,
    _wait_counter_at_least,
)
from .params import ceil_div, CuteParamsBase

# One work-item copies this many bytes; must divide 16 and comfortably
# oversubscribe the producer groups on a single expert row.
WEIGHT_BORROW_CHUNK_BYTES: int = 1 << 17

# Counters per borrowed slot: FC13 (w13 + its scales) and FC2 (w2 + scales).
WEIGHT_BORROW_COUNTERS_PER_SLOT: int = 2


@dataclass
class WeightBorrowArgs(CuteParamsBase):
    """Runtime inputs of the in-kernel weight fetch, all device-resident.

    Byte geometry is per segment: (window offset, slot-buffer offset,
    per-expert row bytes) for w13, w13 scales, w2, w2 scales in that order.
    Offsets address region starts; row ``r`` of a region sits at
    ``offset + r * row_bytes``.
    """

    window_ptrs: cute.Tensor  # [ep_size] Int64 peer window bases (one parity)
    src_rank: cute.Tensor  # [slots] Int32, -1 = unused slot
    src_slot: cute.Tensor  # [slots] Int32, row in the owner's window
    slot_buf_base: cutlass.Int64
    w13_win_off: cutlass.Int64
    w13_slot_off: cutlass.Int64
    w13_row_bytes: cutlass.Int32
    s13_win_off: cutlass.Int64
    s13_slot_off: cutlass.Int64
    s13_row_bytes: cutlass.Int32
    w2_win_off: cutlass.Int64
    w2_slot_off: cutlass.Int64
    w2_row_bytes: cutlass.Int32
    s2_win_off: cutlass.Int64
    s2_slot_off: cutlass.Int64
    s2_row_bytes: cutlass.Int32


# Host-boundary scalar halves, as plain tuples of cutlass values like
# BlockscaledPointerStrideArgs — CuteParamsBase dataclasses only implement the
# trace-time protocol and segfault the executor's runtime arg packing.
#
# Scalars order: (slot_buf_base, then per segment in w13 / w13_scales / w2 /
# w2_scales order: window offset Int64, slot-buffer offset Int64, per-expert
# row bytes Int32).
WeightBorrowScalars = tuple[
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int32,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int32,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int32,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int32,
]
# Prepare order: per GEMM (0 = FC13 w13+scales, 1 = FC2 w2+scales):
# (A slot-buffer base, A row bytes, SFA slot-buffer base, SFA row bytes).
WeightBorrowPrepare = tuple[
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
    cutlass.Int64,
]


@cute.jit
def _weight_borrow_chunk_counts(
    args: WeightBorrowArgs,
    CHUNK_BYTES: cutlass.Constexpr[int],
):
    """Per-segment work-item counts; identical on every thread that calls."""
    c_w13 = ceil_div(args.w13_row_bytes, cutlass.Int32(CHUNK_BYTES))
    c_s13 = ceil_div(args.s13_row_bytes, cutlass.Int32(CHUNK_BYTES))
    c_w2 = ceil_div(args.w2_row_bytes, cutlass.Int32(CHUNK_BYTES))
    c_s2 = ceil_div(args.s2_row_bytes, cutlass.Int32(CHUNK_BYTES))
    return c_w13, c_s13, c_w2, c_s2


@cute.jit
def _weight_borrow_copy_chunk(
    src_base: cutlass.Int64,
    dst_base: cutlass.Int64,
    chunk_bytes: cutlass.Int32,
    lane: cutlass.Int32,
    GROUP_THREADS: cutlass.Constexpr[int],
    CHUNK_BYTES: cutlass.Constexpr[int],
) -> None:
    """Copy up to ``chunk_bytes`` with the whole producer group, 16 B atoms.

    Window rows and slot-buffer rows are 16 B-aligned by construction (the
    inference window asserts it), so whole atoms never straddle a row end.
    """
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Int32,
        num_bits_per_copy=128,
    )
    src_ptr = cute.make_ptr(
        cutlass.Int32, src_base, cute.AddressSpace.gmem, assumed_align=16
    )
    dst_ptr = cute.make_ptr(
        cutlass.Int32, dst_base, cute.AddressSpace.gmem, assumed_align=16
    )
    max_words: cutlass.Constexpr[int] = CHUNK_BYTES // 4
    src_atoms = cute.tiled_divide(
        cute.make_tensor(src_ptr, cute.make_layout((max_words,), stride=(1,))),
        (4,),
    )
    dst_atoms = cute.tiled_divide(
        cute.make_tensor(dst_ptr, cute.make_layout((max_words,), stride=(1,))),
        (4,),
    )
    # One work item owns exactly CHUNK_BYTES, so cap the mask at the chunk
    # (``chunk_bytes`` is everything left in the row, not just this chunk) and
    # round the trip count UP. Truncating instead would silently drop the tail
    # of every full chunk whenever GROUP_THREADS does not divide
    # CHUNK_BYTES/16 -- e.g. 3 warps per group leaves 512 B uncopied.
    atoms_per_chunk: cutlass.Constexpr[int] = CHUNK_BYTES // 16
    chunk_atoms = cutlass.min(
        ceil_div(chunk_bytes, cutlass.Int32(16)), cutlass.Int32(atoms_per_chunk)
    )
    iters: cutlass.Constexpr[int] = -(-atoms_per_chunk // GROUP_THREADS)
    for i in cutlass.range(iters, unroll=4):
        atom_idx = i * cutlass.Int32(GROUP_THREADS) + lane
        if atom_idx < chunk_atoms:
            frag = cute.make_rmem_tensor(4, cutlass.Int32)
            cute.copy(copy_atom, src_atoms[(None, atom_idx)], frag)
            cute.copy(copy_atom, frag, dst_atoms[(None, atom_idx)])


@cute.jit
def _drain_weight_fetch_items(
    args: WeightBorrowArgs,
    mCounters: cute.Tensor,  # the activation done-counter tensor
    work_offset: cutlass.Int32,
    done_offset: cutlass.Int32,
    tile_smem_ptr: cute.Pointer,
    quant_group: cutlass.Int32,
    quant_group_lane: cutlass.Int32,
    SLOTS: cutlass.Constexpr[int],
    CHUNK_BYTES: cutlass.Constexpr[int],
    GROUP_THREADS: cutlass.Constexpr[int],
    SYNC_BAR: cutlass.Constexpr[int],
) -> None:
    """Pull every borrowed slot's rows into the slot buffer, then return.

    Runs on the dispatch producer warps before their token loop, against a
    dedicated work counter so the token-work decode is untouched. Items are
    ``slot-major``: all of a slot's FC13 bytes, then its FC2 bytes, so the
    first borrowed group's gate opens earliest. Unused slots (src_rank < 0)
    skip the copy but still release their counters, keeping the gate's
    arithmetic unconditional. When *every* slot is unused (a balanced
    layer), the queue is bypassed wholesale: the gate only runs for an
    appended group that routed tokens, which implies an active slot, so no
    consumer can wait on the unreleased counters — and the balanced fast
    path pays no queue atomics, fences, or barriers.
    """
    any_active = cutlass.Boolean(False)
    for s in cutlass.range_constexpr(SLOTS):
        if args.src_rank[s] >= cutlass.Int32(0):
            any_active = cutlass.Boolean(True)
    if any_active:
        c_w13, c_s13, c_w2, c_s2 = _weight_borrow_chunk_counts(args, CHUNK_BYTES)
        n_fc13 = c_w13 + c_s13
        n_fc2 = c_w2 + c_s2
        per_slot = n_fc13 + n_fc2
        total = cutlass.Int32(SLOTS) * per_slot

        mWork = cute.make_tensor(
            mCounters.iterator + work_offset, cute.make_layout((1,), stride=(1,))
        )
        item = _dispatch_quant_fetch_group_tile_id(
            mWork,
            tile_smem_ptr,
            quant_group,
            quant_group_lane,
            DISPATCH_QUANT_SYNC_BAR=SYNC_BAR,
            DISPATCH_QUANT_GROUP_THREADS=GROUP_THREADS,
        )
        while item < total:
            slot = item // per_slot
            rem = item - slot * per_slot
            rank = args.src_rank[slot]
            if rank >= cutlass.Int32(0):
                # Segment select: (window offset, slot offset, row bytes, chunk).
                win_off = args.w13_win_off
                slot_off = args.w13_slot_off
                row_bytes = args.w13_row_bytes
                chunk = rem
                if rem >= c_w13:
                    win_off = args.s13_win_off
                    slot_off = args.s13_slot_off
                    row_bytes = args.s13_row_bytes
                    chunk = rem - c_w13
                if rem >= n_fc13:
                    win_off = args.w2_win_off
                    slot_off = args.w2_slot_off
                    row_bytes = args.w2_row_bytes
                    chunk = rem - n_fc13
                if rem >= n_fc13 + c_w2:
                    win_off = args.s2_win_off
                    slot_off = args.s2_slot_off
                    row_bytes = args.s2_row_bytes
                    chunk = rem - n_fc13 - c_w2
                byte_start = chunk * cutlass.Int32(CHUNK_BYTES)
                src_base = (
                    args.window_ptrs[rank]
                    + win_off
                    + cutlass.Int64(args.src_slot[slot]) * cutlass.Int64(row_bytes)
                    + cutlass.Int64(byte_start)
                )
                dst_base = (
                    args.slot_buf_base
                    + slot_off
                    + cutlass.Int64(slot) * cutlass.Int64(row_bytes)
                    + cutlass.Int64(byte_start)
                )
                _weight_borrow_copy_chunk(
                    src_base,
                    dst_base,
                    row_bytes - byte_start,
                    quant_group_lane,
                    GROUP_THREADS=GROUP_THREADS,
                    CHUNK_BYTES=CHUNK_BYTES,
                )

            # The TMA warp consumes the slot buffer through the async proxy,
            # so publish there before the release — same ordering as the
            # dispatch quant signal, and the same class of race its missing
            # writer-side fence once caused.
            cute.arch.fence_proxy("async.global")
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier(
                barrier_id=SYNC_BAR + quant_group,
                number_of_threads=GROUP_THREADS,
            )
            if quant_group_lane == cutlass.Int32(0):
                cls = cutlass.Int32(0)
                if rem >= n_fc13:
                    cls = cutlass.Int32(1)
                counter_ptr = cute.recast_ptr(
                    mCounters.iterator
                    + done_offset
                    + slot * cutlass.Int32(WEIGHT_BORROW_COUNTERS_PER_SLOT)
                    + cls,
                    dtype=cutlass.Uint32,
                )
                cute.arch.atomic_add(
                    counter_ptr,
                    cutlass.Uint32(1),
                    sem="release",
                    scope="gpu",
                )
            item = _dispatch_quant_fetch_group_tile_id(
                mWork,
                tile_smem_ptr,
                quant_group,
                quant_group_lane,
                DISPATCH_QUANT_SYNC_BAR=SYNC_BAR,
                DISPATCH_QUANT_GROUP_THREADS=GROUP_THREADS,
            )


@cute.jit
def _weight_borrow_gate(
    args: WeightBorrowArgs,
    mCounters: cute.Tensor,
    done_offset: cutlass.Int32,
    g: cutlass.Int32,
    num_local: cutlass.Int32,
    CHUNK_BYTES: cutlass.Constexpr[int],
) -> None:
    """Warp-uniform wait until group ``g``'s borrowed weights are resident.

    Called by the weight (A-operand) TMA warp at group entry, before any of
    the group's descriptors are used. Local groups never reach this. Both
    GEMMs' counters are waited together: the visitor runs a group's FC13 and
    FC2 tiles back to back, and w2 lands long before FC2 needs it, so
    splitting the wait buys nothing.
    """
    c_w13, c_s13, c_w2, c_s2 = _weight_borrow_chunk_counts(args, CHUNK_BYTES)
    slot = g - num_local
    base = cute.recast_ptr(
        mCounters.iterator
        + done_offset
        + slot * cutlass.Int32(WEIGHT_BORROW_COUNTERS_PER_SLOT),
        dtype=cutlass.Uint32,
    )
    _wait_counter_at_least(base, cutlass.Uint32(c_w13 + c_s13))
    _wait_counter_at_least(base + 1, cutlass.Uint32(c_w2 + c_s2))


# --------------------------------------------------------------------------
# Host-side assembly
# --------------------------------------------------------------------------


def weight_borrow_counter_size(slots: int) -> int:
    """int32 slots this feature appends to the activation counter region."""
    return 1 + slots * WEIGHT_BORROW_COUNTERS_PER_SLOT if slots > 0 else 0


@dataclass(frozen=True)
class WeightBorrowHostArgs:
    """Everything the host wrapper needs to arm the fused weight fetch.

    ``segments`` maps segment name -> (window offset, slot-buffer offset,
    per-expert row bytes); names follow the inference window: ``w13``,
    ``w13_scales``, ``w2``, ``w2_scales``. The slot buffer must outlive the
    launch and use the same 16 B row / 128 B region alignment as the window.
    """

    window_ptrs: torch.Tensor  # [ep_size] int64, parity-selected
    src_rank: torch.Tensor  # [slots] int32
    src_slot: torch.Tensor  # [slots] int32
    slot_buffer: torch.Tensor  # flat uint8
    segments: dict[str, tuple[int, int, int]]
    slots: int

    def cute_tensors(self, *, enable_tvm_ffi: bool) -> tuple:
        """(window ptrs, src rank, src slot) as launch-boundary cute tensors."""
        return (
            from_dlpack(
                self.window_ptrs, assumed_align=8, enable_tvm_ffi=enable_tvm_ffi
            ),
            from_dlpack(self.src_rank, assumed_align=4, enable_tvm_ffi=enable_tvm_ffi),
            from_dlpack(self.src_slot, assumed_align=4, enable_tvm_ffi=enable_tvm_ffi),
        )

    def _segment_triples(self) -> tuple:
        # Missing scale segments (the native BF16 tables) contribute zero
        # rows: zero chunk counts and zero gate targets fall out downstream.
        zero = (0, 0, 0)
        return (
            self.segments["w13"],
            self.segments.get("w13_scales", zero),
            self.segments["w2"],
            self.segments.get("w2_scales", zero),
        )

    def scalar_args(self) -> tuple:
        """The WeightBorrowScalars host-boundary tuple."""
        w13, s13, w2, s2 = self._segment_triples()
        base = self.slot_buffer.data_ptr()
        return (
            cutlass.Int64(base),
            cutlass.Int64(w13[0]),
            cutlass.Int64(w13[1]),
            cutlass.Int32(w13[2]),
            cutlass.Int64(s13[0]),
            cutlass.Int64(s13[1]),
            cutlass.Int32(s13[2]),
            cutlass.Int64(w2[0]),
            cutlass.Int64(w2[1]),
            cutlass.Int32(w2[2]),
            cutlass.Int64(s2[0]),
            cutlass.Int64(s2[1]),
            cutlass.Int32(s2[2]),
        )

    def launch_args(self, *, enable_tvm_ffi: bool) -> tuple:
        """The five trailing launch arguments, compile- and run-time alike.

        Order matches the blockscaled kernel entry: window ptrs, src rank,
        src slot, scalar bundle, prepare bundle.
        """
        w13, s13, w2, s2 = self._segment_triples()
        base = self.slot_buffer.data_ptr()
        prepare = (
            cutlass.Int64(base + w13[1]),
            cutlass.Int64(w13[2]),
            cutlass.Int64(base + s13[1]),
            cutlass.Int64(s13[2]),
            cutlass.Int64(base + w2[1]),
            cutlass.Int64(w2[2]),
            cutlass.Int64(base + s2[1]),
            cutlass.Int64(s2[2]),
        )
        return (
            *self.cute_tensors(enable_tvm_ffi=enable_tvm_ffi),
            self.scalar_args(),
            prepare,
        )


def allocate_slot_buffer(
    segments: dict[str, tuple[int, int, int]], slots: int, device: torch.device
) -> tuple[torch.Tensor, dict[str, tuple[int, int, int]]]:
    """A slot buffer mirroring the window's segment layout for ``slots`` rows.

    Returns the buffer and the segment map with slot-buffer offsets filled
    (region offsets re-derived at 128 B alignment for the smaller row count).
    """
    out: dict[str, tuple[int, int, int]] = {}
    at = 0
    for name in ("w13", "w13_scales", "w2", "w2_scales"):
        win_off, _, row_bytes = segments.get(name, (0, 0, 0))
        assert row_bytes % 16 == 0, f"{name} rows must be 16-byte aligned"
        at = -(-at // 128) * 128
        out[name] = (win_off, at, row_bytes)
        at += slots * row_bytes
    return torch.empty(at, dtype=torch.uint8, device=device), out
