# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Symmetric-memory buffers for distributed MoE communication."""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed.distributed_c10d as c10d

# Match PyTorch symmetric memory's default peer/channel signal-pad contract.
_SYMMETRIC_MEMORY_MAX_PEERS = 32
_SYMMETRIC_MEMORY_SIGNAL_CHANNELS = 72
_SIGNAL_WORD_SIZE_BYTES = 4
_SIGNAL_PAD_SIZE_BYTES = (
    _SYMMETRIC_MEMORY_MAX_PEERS
    * _SYMMETRIC_MEMORY_SIGNAL_CHANNELS
    * _SIGNAL_WORD_SIZE_BYTES
)
_MULTIMEM_BARRIER_NUM_CHANNELS = 32

# Keep expert IDs cache-line aligned after the fixed collective-shape header.
_ROUTING_HEADER_SIZE_BYTES = 128
_ROUTING_HEADER_INT16_ELEMENTS = _ROUTING_HEADER_SIZE_BYTES // torch.int16.itemsize

logger = logging.getLogger(__name__)


def is_fake_process_group(group: dist.ProcessGroup) -> bool:
    """Return whether ``group`` uses PyTorch's fake distributed backend.

    Args:
        group: Process group to inspect.

    Returns:
        ``True`` when the process-group backend is ``fake``.
    """
    return dist.get_backend(group) == "fake"


class _FakeSymmetricMemory:
    """Local allocation modeling a multi-rank symmetric-memory handle.

    Fake process groups execute one virtual rank. Every peer pointer therefore
    aliases the rank-local payload or signal pad. Peer-scatter destinations are
    initialized before use so absent virtual writers contribute deterministic
    zeros. This preserves distributed shapes, pointer-array sizes, graph
    topology, local memory accounting, and stable local numerics; it does not
    model communication progress or true multi-rank numerical equivalence.
    """

    def __init__(
        self,
        payload: torch.Tensor,
        *,
        rank: int,
        world_size: int,
    ) -> None:
        """Build a virtual peer mapping over one local allocation.

        Args:
            payload: Local symmetric-memory payload tensor.
            rank: Virtual rank represented by this process.
            world_size: Number of virtual ranks in the fake process group.
        """
        self._payload = payload.view(torch.uint8).flatten()
        self._signal_pad = torch.zeros(
            _SIGNAL_PAD_SIZE_BYTES,
            dtype=torch.uint8,
            device=self._payload.device,
        )
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = self._payload.numel()
        self.buffer_ptrs = [self._payload.data_ptr()] * world_size
        self.signal_pad_ptrs = [self._signal_pad.data_ptr()] * world_size

        # Native symmetric memory owns device pointer arrays in addition to the
        # arrays exposed by SymmetricMemoryBuffer. Keep fake allocation
        # accounting equivalent to one real rank.
        self._buffer_ptrs_dev = torch.tensor(
            self.buffer_ptrs,
            dtype=torch.int64,
            device=self._payload.device,
        )
        self._signal_pad_ptrs_dev = torch.tensor(
            self.signal_pad_ptrs,
            dtype=torch.int64,
            device=self._payload.device,
        )

    def _validate_rank(self, rank: int) -> None:
        """Validate a virtual peer rank.

        Args:
            rank: Virtual peer rank.

        Raises:
            ValueError: If ``rank`` is outside the fake process group.
        """
        if not 0 <= rank < self.world_size:
            raise ValueError(
                f"fake symmetric-memory rank {rank} is outside [0, {self.world_size})"
            )

    def get_buffer(
        self,
        rank: int,
        sizes: Sequence[int],
        dtype: torch.dtype,
        storage_offset: int = 0,
    ) -> torch.Tensor:
        """Return a typed view of a virtual peer's symmetric payload.

        Args:
            rank: Virtual peer rank.
            sizes: Requested tensor shape.
            dtype: Requested tensor dtype.
            storage_offset: Element offset into the typed payload.

        Returns:
            Tensor view backed by the fake symmetric allocation.

        Raises:
            ValueError: If the rank or requested payload view is invalid.
        """
        self._validate_rank(rank)
        numel = math.prod(sizes)
        start = storage_offset * dtype.itemsize
        end = start + numel * dtype.itemsize
        if end > self.buffer_size:
            raise ValueError(
                f"requested {end} payload bytes from a {self.buffer_size}-byte buffer"
            )
        return self._payload[start:end].view(dtype).view(tuple(sizes))

    def get_signal_pad(
        self,
        rank: int,
        sizes: tuple[int, ...] = (),
        dtype: torch.dtype | None = None,
        storage_offset: int = 0,
    ) -> torch.Tensor:
        """Return a typed view of the modeled signal pad.

        Args:
            rank: Virtual peer rank.
            sizes: Requested tensor shape. An empty shape selects the full pad.
            dtype: Requested dtype, defaulting to ``torch.uint32``.
            storage_offset: Element offset into the typed signal pad.

        Returns:
            Tensor view backed by the signal-pad allocation.

        Raises:
            ValueError: If the requested view is invalid.
        """
        self._validate_rank(rank)
        dtype = torch.uint32 if dtype is None else dtype
        if not sizes:
            sizes = (_SIGNAL_PAD_SIZE_BYTES // dtype.itemsize,)
        numel = math.prod(sizes)
        start = storage_offset * dtype.itemsize
        end = start + numel * dtype.itemsize
        if end > _SIGNAL_PAD_SIZE_BYTES:
            raise ValueError("requested signal-pad view exceeds the modeled pad")
        return self._signal_pad[start:end].view(dtype).view(sizes)

    def barrier(self, channel: int = 0, timeout_ms: int = 0) -> None:
        """Model a virtual process-group barrier as a no-op.

        Args:
            channel: Signal channel, unused by fake execution.
            timeout_ms: Timeout in milliseconds, unused by fake execution.
        """
        del channel, timeout_ms

    def put_signal(
        self,
        dst_rank: int,
        channel: int = 0,
        timeout_ms: int = 0,
    ) -> None:
        """Model a signal send as a no-op.

        Args:
            dst_rank: Virtual destination rank.
            channel: Signal channel, unused by fake execution.
            timeout_ms: Timeout in milliseconds, unused by fake execution.

        Raises:
            ValueError: If ``dst_rank`` is outside the fake process group.
        """
        del channel, timeout_ms
        self._validate_rank(dst_rank)

    def wait_signal(
        self,
        src_rank: int,
        channel: int = 0,
        timeout_ms: int = 0,
    ) -> None:
        """Model a signal wait as a no-op.

        Args:
            src_rank: Virtual source rank.
            channel: Signal channel, unused by fake execution.
            timeout_ms: Timeout in milliseconds, unused by fake execution.

        Raises:
            ValueError: If ``src_rank`` is outside the fake process group.
        """
        del channel, timeout_ms
        self._validate_rank(src_rank)


def _allocate_handle(
    num_bytes: int,
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    emulate_peer_buffers: bool,
) -> Any:
    """Allocate and rendezvous a symmetric-memory payload.

    Args:
        num_bytes: Symmetric payload size in bytes.
        group: Expert-parallel process group.
        device: CUDA allocation device.
        emulate_peer_buffers: Whether to alias every virtual peer to one local
            allocation instead of rendezvousing native symmetric memory.

    Returns:
        A native or fake symmetric-memory handle.
    """
    if emulate_peer_buffers:
        payload = symm_mem.empty(
            num_bytes,
            dtype=torch.uint8,
            device=device,
        )
        return _FakeSymmetricMemory(
            payload,
            rank=dist.get_rank(group),
            world_size=dist.get_world_size(group),
        )

    allocation = symm_mem.empty(
        num_bytes,
        dtype=torch.uint8,
        device=device,
    )
    return symm_mem.rendezvous(allocation, group=group)


@dataclasses.dataclass(frozen=True)
class _MultimemBarrierWorkspace:
    """Immutable state for the self-resetting NVLS barrier.

    Args:
        flags: Symmetric array containing one arrival count per channel.
        handle: Symmetric-memory handle that owns the multicast mapping.
        multicast_ptr: Multicast virtual address of ``flags``.
        local_ptr: This rank's unicast virtual address of ``flags``.
        num_channels: Number of independent barrier channels.
    """

    flags: torch.Tensor
    handle: Any
    multicast_ptr: int
    local_ptr: int
    num_channels: int


_MULTIMEM_BARRIER_WORKSPACES: dict[dist.ProcessGroup, _MultimemBarrierWorkspace] = {}
_MULTIMEM_BARRIER_SUPPORTED: dict[dist.ProcessGroup, bool] = {}
_MULTIMEM_BARRIER_VOTE_SEQUENCE: dict[str, int] = {}


def _store_min_vote(group: dist.ProcessGroup, key: str, vote: int) -> int:
    """Return the minimum rank vote without initializing an NCCL communicator.

    Args:
        group: Process group whose ranks participate in the vote.
        key: Store key unique to this initialization attempt.
        vote: Local zero-or-one vote.

    Returns:
        Minimum vote across the process group.
    """
    store = c10d._get_process_group_store(group)
    rank = dist.get_rank(group)
    store.set(f"{key}/r{rank}", str(vote))
    votes = [vote]
    for peer in range(dist.get_world_size(group)):
        if peer != rank:
            votes.append(int(store.get(f"{key}/r{peer}")))
    return min(votes)


def initialize_multimem_barrier_workspace(
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    emulate_peer_buffers: bool | None = None,
) -> None:
    """Collectively initialize the optional NVLS barrier workspace.

    Initialization is eager because allocation, rendezvous, and the
    rank-consistent availability vote are illegal during CUDA graph capture.
    Lack of multicast support is cached and transparently selects the unicast
    signal-pad barrier at execution time.

    Args:
        group: Expert-parallel process group.
        device: CUDA device on which to allocate the flag array.
        emulate_peer_buffers: Whether communication buffers are locally
            emulated and therefore cannot use multicast memory. ``None``
            detects PyTorch's built-in FakePG.
    """
    if group in _MULTIMEM_BARRIER_WORKSPACES or group in _MULTIMEM_BARRIER_SUPPORTED:
        return
    if emulate_peer_buffers is None:
        emulate_peer_buffers = is_fake_process_group(group)
    if emulate_peer_buffers:
        _MULTIMEM_BARRIER_SUPPORTED[group] = False
        return

    flags = None
    init_error: Exception | None = None
    try:
        flags = symm_mem.empty(
            _MULTIMEM_BARRIER_NUM_CHANNELS,
            dtype=torch.int32,
            device=device,
        )
        flags.zero_()
    except Exception as error:  # All ranks vote before selecting the fallback.
        init_error = error

    group_name = group.group_name
    sequence = _MULTIMEM_BARRIER_VOTE_SEQUENCE.get(group_name, 0) + 1
    _MULTIMEM_BARRIER_VOTE_SEQUENCE[group_name] = sequence
    vote_prefix = f"dist_moe_multimem_barrier/{group_name}/{sequence}"
    if _store_min_vote(group, f"{vote_prefix}/alloc", int(init_error is None)) == 0:
        logger.warning(
            "DistMoE NVLS barrier allocation failed%s; using the unicast barrier",
            f": {init_error}" if init_error is not None else " on a peer rank",
        )
        _MULTIMEM_BARRIER_SUPPORTED[group] = False
        return

    assert flags is not None
    handle = None
    has_multicast = 0
    try:
        handle = symm_mem.rendezvous(flags, group=group)
        has_multicast = int(getattr(handle, "multicast_ptr", 0) != 0)
    except Exception as error:  # All ranks vote before selecting the fallback.
        init_error = error

    # No rank may publish its first arrival before every peer has zeroed flags.
    torch.cuda.current_stream(device).synchronize()
    if _store_min_vote(group, f"{vote_prefix}/multicast", has_multicast) == 0:
        if init_error is not None:
            logger.warning(
                "DistMoE NVLS barrier rendezvous failed: %s; using the unicast barrier",
                init_error,
            )
        _MULTIMEM_BARRIER_SUPPORTED[group] = False
        return

    assert handle is not None
    _MULTIMEM_BARRIER_WORKSPACES[group] = _MultimemBarrierWorkspace(
        flags=flags,
        handle=handle,
        multicast_ptr=handle.multicast_ptr,
        local_ptr=flags.data_ptr(),
        num_channels=_MULTIMEM_BARRIER_NUM_CHANNELS,
    )
    _MULTIMEM_BARRIER_SUPPORTED[group] = True


def get_multimem_barrier_workspace(
    group: dist.ProcessGroup,
) -> _MultimemBarrierWorkspace | None:
    """Return the eagerly initialized NVLS workspace without allocating.

    Args:
        group: Expert-parallel process group.

    Returns:
        Cached workspace, or ``None`` when NVLS is unavailable.
    """
    return _MULTIMEM_BARRIER_WORKSPACES.get(group)


@dataclasses.dataclass(frozen=True)
class SymmetricMemoryBuffer:
    """Own one symmetric payload and its graph-visible peer pointer arrays.

    ``create()`` performs the collective rendezvous for a real process group or
    creates local peer aliases for FakePG. The handle owns payload and signal
    storage; the pointer tensors keep peer addressing explicit to Triton and
    CuTe launches. Instances are mutable communication state and must not be
    shared by overlapping Dist-MoE invocations.
    """

    # Kernel launchers access this field directly using the handle's native name.
    hdl: Any
    buffer_ptrs_tensor: torch.Tensor
    signal_pad_ptrs_tensor: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    group: dist.ProcessGroup

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        group: dist.ProcessGroup,
        device: torch.device,
        emulate_peer_buffers: bool | None = None,
    ) -> SymmetricMemoryBuffer:
        """Create a symmetric-memory buffer.

        Args:
            shape: Logical payload shape.
            dtype: Logical payload dtype.
            group: Expert-parallel process group.
            device: CUDA allocation device.
            emulate_peer_buffers: Whether virtual peer views should alias one
                local allocation. ``None`` detects PyTorch's built-in FakePG.

        Returns:
            Allocated and rendezvoused symmetric-memory buffer.
        """
        if emulate_peer_buffers is None:
            emulate_peer_buffers = is_fake_process_group(group)
        num_bytes = math.prod(shape) * dtype.itemsize
        handle = _allocate_handle(
            num_bytes,
            group,
            device,
            emulate_peer_buffers=emulate_peer_buffers,
        )
        return cls(
            hdl=handle,
            buffer_ptrs_tensor=torch.tensor(
                handle.buffer_ptrs,
                dtype=torch.int64,
                device=device,
            ),
            signal_pad_ptrs_tensor=torch.tensor(
                handle.signal_pad_ptrs,
                dtype=torch.int64,
                device=device,
            ),
            shape=shape,
            dtype=dtype,
            group=group,
        )

    def local(self) -> torch.Tensor:
        """Return the local rank's typed payload view.

        Returns:
            Tensor view with this buffer's logical shape and dtype.
        """
        return self.hdl.get_buffer(
            self.hdl.rank,
            self.shape,
            self.dtype,
        )


def is_fake_symmetric_memory(buffer: SymmetricMemoryBuffer) -> bool:
    """Return whether a buffer uses a fake symmetric-memory handle.

    Args:
        buffer: Symmetric-memory buffer to inspect.

    Returns:
        ``True`` for a fake symmetric-memory buffer.
    """
    return isinstance(buffer.hdl, _FakeSymmetricMemory)


def _initialize_fake_peer_scatter_output(buffer: SymmetricMemoryBuffer) -> None:
    """Initialize rows whose virtual FakePG peers have no physical writer.

    FakePG aliases every logical peer pointer to one local allocation. A peer
    scatter therefore writes only rows owned by the executing virtual rank,
    while rows owned by other virtual ranks otherwise retain stale payloads.
    Native symmetric memory has one physical writer for every routed row and
    requires no initialization.

    Args:
        buffer: Symmetric-memory destination of a peer-scatter operation.
    """
    if is_fake_symmetric_memory(buffer):
        buffer.local().zero_()


@dataclasses.dataclass(frozen=True, init=False)
class _CommunicationBuffers:
    """Symmetric communication buffers owned by one execution context.

    All expert-parallel ranks create matching buffers collectively and in the
    same order. A buffer set is mutable communication state for one active
    context and must not be used by overlapping calls.

    Args:
        routing: Symmetric int16 storage containing a fixed routing header
            followed by ``[num_local_input_tokens, top_k]`` expert IDs.
        dispatch: Symmetric BF16 source-activation and DGRAD storage with
            capacity ``[num_local_input_tokens, top_k, hidden_dim]``.
        combine: Symmetric BF16 route-output and route-gradient storage with
            the same ``[num_local_input_tokens, top_k, hidden_dim]`` capacity.
    """

    routing: SymmetricMemoryBuffer
    dispatch: SymmetricMemoryBuffer
    combine: SymmetricMemoryBuffer

    def __init__(self) -> None:
        """Reject construction outside the collective buffer factory.

        Raises:
            TypeError: Always. Use :meth:`create` to establish valid symmetric
                memory and peer-pointer state.
        """
        raise TypeError("use _CommunicationBuffers.create()")

    @classmethod
    def _from_buffers(
        cls,
        *,
        routing: SymmetricMemoryBuffer,
        dispatch: SymmetricMemoryBuffer,
        combine: SymmetricMemoryBuffer,
    ) -> _CommunicationBuffers:
        """Construct a validated collection from factory-created buffers.

        Args:
            routing: Initialized routing-ID symmetric buffer.
            dispatch: Initialized dispatch symmetric buffer.
            combine: Initialized combine symmetric buffer.

        Returns:
            A buffer owner whose three allocations share one EP group.
        """
        buffers = object.__new__(cls)
        object.__setattr__(buffers, "routing", routing)
        object.__setattr__(buffers, "dispatch", dispatch)
        object.__setattr__(buffers, "combine", combine)
        return buffers

    @classmethod
    def create(
        cls,
        *,
        num_local_input_tokens: int,
        hidden_dim: int,
        top_k: int,
        group: dist.ProcessGroup,
        device: torch.device,
        emulate_peer_buffers: bool | None = None,
    ) -> _CommunicationBuffers:
        """Create all communication buffers required by distributed MoE.

        Args:
            num_local_input_tokens: Fixed physical input tokens on every rank.
            hidden_dim: Model hidden dimension.
            top_k: Number of selected experts per token.
            group: Expert-parallel process group.
            device: CUDA allocation device.
            emulate_peer_buffers: Whether to model all peers with local
                buffers. This validates shapes and memory without simulating
                distributed communication or numerics. ``None`` detects
                PyTorch's built-in FakePG.

        Returns:
            Routing, dispatch, and combine symmetric-memory buffers.
        """
        if emulate_peer_buffers is None:
            emulate_peer_buffers = is_fake_process_group(group)
        routing = SymmetricMemoryBuffer.create(
            (_ROUTING_HEADER_INT16_ELEMENTS + num_local_input_tokens * top_k,),
            torch.int16,
            group,
            device,
            emulate_peer_buffers=emulate_peer_buffers,
        )
        _routing_token_count_view(routing, routing.hdl.rank).fill_(
            num_local_input_tokens
        )
        return cls._from_buffers(
            routing=routing,
            dispatch=SymmetricMemoryBuffer.create(
                (num_local_input_tokens, top_k, hidden_dim),
                torch.bfloat16,
                group,
                device,
                emulate_peer_buffers=emulate_peer_buffers,
            ),
            combine=SymmetricMemoryBuffer.create(
                (num_local_input_tokens, top_k, hidden_dim),
                torch.bfloat16,
                group,
                device,
                emulate_peer_buffers=emulate_peer_buffers,
            ),
        )


def _routing_token_count_view(
    routing: SymmetricMemoryBuffer,
    rank: int,
) -> torch.Tensor:
    """Return one peer's fixed local-token-count header view.

    Args:
        routing: Routing symmetric allocation.
        rank: Expert-parallel rank whose header is requested.

    Returns:
        Stable device int32 tensor with shape ``[1]``.
    """
    return routing.hdl.get_buffer(rank, (1,), torch.int32)


def _routing_ids_view(
    routing: SymmetricMemoryBuffer,
    rank: int,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Return one peer's expert-ID payload after the routing header.

    Args:
        routing: Routing symmetric allocation.
        rank: Expert-parallel rank whose expert IDs are requested.
        shape: Fixed ``[T, K]`` expert-ID shape.

    Returns:
        Stable device int16 tensor backed by the routing allocation.
    """
    return routing.hdl.get_buffer(
        rank,
        shape,
        torch.int16,
        storage_offset=_ROUTING_HEADER_INT16_ELEMENTS,
    )
