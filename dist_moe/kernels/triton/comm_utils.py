# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Reference: https://github.com/yifuwang/symm-mem-recipes

import triton
import triton.language as tl


@triton.jit
def get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_flat_tid():
    tid_x, tid_y, tid_z = get_tid()
    ntid_x, ntid_y, _ = get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def get_flat_bid():
    return (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )


@triton.jit
def sync_threads():
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def send_signal(addrs, sem: tl.constexpr):
    if sem == "relaxed":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.relaxed.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    elif sem == "acq_rel":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        raise RuntimeError(f"Unrecognized sem: {sem}")


@triton.jit
def wait_ge(addr, target):
    """Acquire-poll int64 until ``*addr >= target``.

    Used for monotonic head/tail signaling where counters
    only increase.  The target is a runtime int64 value (not constexpr).

    All reads are LOCAL (no NVLink load)::

        Sender polls:   wait_ge(local_head_ptr, step - NUM_SLOTS + 1)
        Receiver polls: wait_ge(local_tail_ptr, step + 1)
    """
    tl.inline_asm_elementwise(
        """
        {
            .reg .u64   %tmp64_<1>;
            .reg .pred  %p<1>;

            wait_ge:
                ld.global.acquire.sys.b64 %tmp64_0, [$1];
                setp.lt.u64 %p0, %tmp64_0, $2;
                @%p0 bra wait_ge;
        }
        """,
        "=r, l, l",
        [addr, target],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def fence_and_remote_store_i64(addr, val):
    """System fence + remote int64 store for monotonic counter advancement.

    Ensures all prior writes (staging buffer data) are ordered before the
    counter increment.  The store uses system scope so the remote rank's
    acquire-load will observe the data.

    Used to advance head/tail counters::

        Sender:   fence_and_remote_store_i64(remote_tail_ptr, step + 1)
        Receiver: fence_and_remote_store_i64(remote_head_ptr, step + 1)

    """
    tl.inline_asm_elementwise(
        """
        {
            fence.acq_rel.sys;
            st.global.relaxed.sys.b64 [$1], $2;
            mov.u32 $0, 0;
        }
        """,
        "=r, l, l",
        [addr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def multimem_red_add1_release_u32(addr):
    """Multicast arrival: adds 1 to the u32 at ``addr`` on EVERY rank of the
    multicast group with a single instruction.

    ``addr`` must be a multicast VA (``cuMulticastMap``) — the NVLink/NVSwitch
    fabric applies the reduction to each subscribed rank's local copy.
    Release ordering at system scope makes this rank's prior global writes
    visible to any peer whose acquire load observes the increment. u32 is the
    width NCCL's LSA barrier uses for its multimem arrivals; pair with a
    wrapping poll (:func:`wait_wrap_ge_u32`) since the counter may wrap.
    """
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %val<1>;

            mov.u32 %val0, 1;
            multimem.red.release.sys.global.add.u32 [$1], %val0;
            mov.u32 $0, 0;
        }
        """,
        "=r, l",
        [addr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def wait_wrap_ge_u32(addr, target):
    """Acquire-poll a u32 until it reaches ``target`` in wrapping (modular)
    order: satisfied when ``(*addr - target) mod 2^32 <= 2^31 - 1``.

    The NCCL-LSA-style poll for monotonically increasing u32 counters that
    are allowed to wrap: sequential advances are unbounded, and the window
    tolerates up to ``2^31`` increments in flight past ``target``. The read
    is LOCAL (no NVLink load); ``target`` is a u32 (int32 bit pattern) and
    may be a runtime value or a constexpr.
    """
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<2>;
            .reg .pred  %p<1>;

            wait_wrap_ge:
                ld.global.acquire.sys.b32 %tmp32_0, [$1];
                sub.u32 %tmp32_1, %tmp32_0, $2;
                setp.gt.u32 %p0, %tmp32_1, 0x7fffffff;
                @%p0 bra wait_wrap_ge;
            mov.u32 $0, 0;
        }
        """,
        "=r, l, r",
        [addr, target],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def red_sub_relaxed_sys_u32(addr, val):
    """Fire-and-forget atomic subtract of ``val`` from the u32 at ``addr``
    (wrapping), at system scope with relaxed ordering.

    System scope is required when the word is concurrently updated by
    fabric-delivered ``multimem.red`` arrivals from peers: a narrower scope
    is not morally strong w.r.t. those accesses and forfeits atomicity.
    Relaxed suffices for pure counting — the subtract publishes no data.
    """
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %val<1>;

            neg.s32 %val0, $2;
            red.relaxed.sys.global.add.u32 [$1], %val0;
            mov.u32 $0, 0;
        }
        """,
        "=r, l, r",
        [addr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def tma_bulk_wait():
    """Drain the TMA async bulk copy pipeline.

    Ensures all pending TMA descriptor stores (cp.async.bulk) are complete
    and globally visible.  Must be called before signaling the receiver
    when using TMA for staging buffer writes, because bar.sync does NOT
    drain the TMA pipeline — TMA is an asynchronous proxy separate from
    the SM execution pipeline.
    """
    tl.inline_asm_elementwise(
        """
        {
            cp.async.bulk.commit_group;
            cp.async.bulk.wait_group 0;
            mov.u32 $0, 0;
        }
        """,
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def wait_signal(addrs, sem: tl.constexpr):
    if sem == "relaxed":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    atom.global.sys.relaxed.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    elif sem == "acq_rel":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    atom.global.sys.acquire.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        raise RuntimeError(f"Unrecognized sem: {sem}")
