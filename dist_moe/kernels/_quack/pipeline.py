# Copyright (c) 2025-2026, Tri Dao.
# Licensed under the Apache License, Version 2.0.
# Modified by Meta Platforms, Inc.

from dataclasses import dataclass
from typing import Optional

import cutlass.cute as cute
from cutlass import Boolean, Int32
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import dsl_user_op, if_generate
from cutlass.pipeline import (
    Agent,
    agent_sync,
    alloc_reserved_mbarrier,
    CooperativeGroup,
    PipelineAsync as PipelineAsyncOg,
    PipelineOp,
    PipelineState,
    PipelineUserType,
)


class PipelineStateWAdvance(PipelineState):
    @dsl_user_op
    def advance_iters(self, num_iterations: Int32, *, loc=None, ip=None):
        self._count += Int32(num_iterations)
        new_index = self._index + Int32(num_iterations)
        # How many times did we cross the stages boundary
        num_crossings = new_index // self.stages
        self._phase ^= num_crossings
        self._index = new_index % self.stages

    # This can be overridden by derived classes
    def __new_from_mlir_values__(self, values):
        return PipelineStateWAdvance(
            self.stages, Int32(values[0]), Int32(values[1]), Int32(values[2])
        )


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        return PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(1))
    elif type is PipelineUserType.Consumer:
        return PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0))
    else:
        assert False, (
            "Error: invalid PipelineUserType specified for make_pipeline_state."
        )


@dataclass(frozen=True)
class PipelineStasAsync(PipelineAsyncOg):
    """Pipeline for a symmetric all-to-all cluster exchange via st.async (STAS).

    Models the reduction-buffer exchange of persistent cluster kernels (e.g.
    RMSNorm/LayerNorm bwd): every CTA is simultaneously producer and consumer.
    Each stage, every warp's lanes < cluster_n STAS values into every peer
    CTA's buffer(s), crediting the peer's transaction (full) barrier; each CTA
    then reads its own buffer(s) and multicast-releases every peer's empty
    barrier.

    Differences from the stock split-agent pipelines:

    - The full barrier is a transaction barrier armed and credited inside the
      data path (quack.reduce.cluster_reduce issues arrive_and_expect_tx and
      the STAS stores, and waits on it with ``state.phase``), so
      producer_commit is a no-op and producer_get_barrier hands the raw
      barrier to the STAS issuer.
    - Sync slots are decoupled from data slots: a stage owns ONE (full, empty)
      barrier pair, which may guard several reduction-buffer slots — the
      issuer arms the barrier with the combined byte count and one wait covers
      them all (LayerNorm bwd reduces sum(x_hat * wdy) and sum(wdy) per stage
      through one barrier this way). Buffer-slot indexing is the kernel's
      concern; the pipeline only tracks stages and phases.
    - consumer_release arrives with one lane per warp on each peer CTA; the
      empty barriers are initialized with num_warps * cluster_n arrives to
      match. The caller must fence the async proxy (which the STAS writes
      travel) before releasing — proxy ordering is the data path's concern,
      like the tx arming.

    This is close to PipelineTmaAsync with the transaction arming moved into
    the data path and point-to-point signaling replaced by all-to-all.
    """

    cluster_n: int = 1

    @staticmethod
    def create(
        *,
        num_stages: int,
        num_warps: int,
        cluster_n: int,
        barrier_storage: Optional[Pointer] = None,
        defer_sync: bool = False,
    ) -> "PipelineStasAsync":
        """Allocates num_stages (full, empty) barrier pairs, from the reserved smem
        partition when ``barrier_storage`` is None. With ``defer_sync=True`` the
        caller must fence the barrier init and arrive + wait on the cluster
        barrier itself before first use (lets the cluster sync overlap other
        setup work)."""
        if barrier_storage is None:
            barrier_storage = alloc_reserved_mbarrier(num_stages)
        # The full barrier's single arrive is the STAS issuer's
        # arrive_and_expect_tx; the pipeline itself never arrives on it.
        producer = (PipelineOp.AsyncThread, CooperativeGroup(Agent.Thread, 1))
        consumer = (
            PipelineOp.AsyncThread,
            CooperativeGroup(Agent.Thread, num_warps * cluster_n),
        )
        sync_object_full = PipelineAsyncOg._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer
        )
        sync_object_empty = PipelineAsyncOg._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        if not defer_sync:
            cute.arch.mbarrier_init_fence()
            agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)
        return PipelineStasAsync(
            sync_object_full,
            sync_object_empty,
            num_stages,
            None,
            None,
            cluster_n,
        )

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        """Wait until every peer has released this stage. Arming the transaction
        barrier is left to the STAS issuer (see class docstring)."""
        self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip)

    @dsl_user_op
    def producer_commit(self, state: PipelineState, *, loc=None, ip=None):
        """No-op: the STAS transactions complete the full barrier."""
        pass

    def producer_get_barrier(self, state: PipelineState) -> Pointer:
        """Full (transaction) barrier of the stage, for arrive_and_expect_tx + STAS."""
        return self.sync_object_full.get_barrier(state.index)

    @dsl_user_op
    def consumer_wait(
        self,
        state: PipelineState,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc=None,
        ip=None,
    ):
        """Wait for the STAS transactions of this stage. Unused when the STAS
        issuer performs the wait itself (cluster_reduce does)."""
        self.sync_object_full.wait(state.index, state.phase, loc=loc, ip=ip)

    @dsl_user_op
    def consumer_release(self, state: PipelineState, *, loc=None, ip=None):
        # One lane per warp arrives on each peer (cheaper than all-lane arrives;
        # empty arrive counts match). The caller must fence the async proxy first.
        cute.arch.sync_warp()
        lane_idx = cute.arch.lane_idx()
        if_generate(
            lane_idx < self.cluster_n,
            lambda: self.sync_object_empty.arrive(
                state.index, lane_idx, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_tail(self, state: PipelineState, *, loc=None, ip=None):
        """Prevent the CTA from exiting (invalidating its smem barriers for peer
        STAS) while peers may still be arriving. ``state`` points at the next
        useful stage; wait on ALL stages, like PipelineAsync.producer_tail —
        the remote empty arrives are .release.cta scope, so a cluster observer
        gets no ordering between arrives to different barriers and draining
        only the last-used stage would race a still-in-flight earlier arrive."""
        for _ in range(self.num_stages):
            self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip)
            state.advance(loc=loc, ip=ip)
