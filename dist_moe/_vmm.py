# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA Virtual Memory Management storage for distributed MoE.

Provides a contiguous virtual address space mapped as a user-configured
sequence of device and host sections, e.g.:
  [device(2MB) | host(1MB) | device(4MB)]

The public distributed-MoE API fixes the physical layout to device activations,
host overflow scratch, and device hot scratch. The lower-level section model
in this module remains private so allocation, prefetch, and cleanup follow the
same path.

Uses CUDA driver VMM APIs (cuMemAddressReserve, cuMemCreate, cuMemMap, etc.)
so that a single contiguous VA range has sections backed by different physical
memory types.

Requirements:
  - CUDA 12.2+ (for CU_MEM_LOCATION_TYPE_HOST; use HOST_NUMA on older CUDA)
  - cuda-python (cuda.bindings.driver)
  - PyTorch with CUDA
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import threading
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List

import torch
from cuda.bindings import driver as drv

logger = logging.getLogger(__name__)

# -- C callback signatures for PyTorch custom allocator ----------------------
ALLOC_FN = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
)
FREE_FN = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
)


def _check(err: Any, msg: str = "", *, driver: Any | None = None) -> None:
    """Raise an actionable exception for a CUDA Driver API failure.

    Args:
        err: CUDA result value or a tuple whose first item is the result.
        msg: Additional operation context.
        driver: CUDA driver module that produced the result.

    Raises:
        RuntimeError: If the CUDA result is not successful.
    """
    driver = drv if driver is None else driver
    if isinstance(err, tuple):
        err = err[0]
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err}. {msg}")


def _cleanup_succeeded(
    operation: Callable[[], object],
    message: str,
    *,
    driver: Any,
    errors: list[str],
) -> bool:
    """Run one cleanup operation and retain any failure for aggregation.

    Args:
        operation: Zero-argument CUDA Driver cleanup call.
        message: Context appended to a failed driver result.
        driver: CUDA driver module that owns the result enum.
        errors: Destination for formatted cleanup failures.

    Returns:
        ``True`` when the cleanup operation succeeded.
    """
    try:
        result = operation()
        _check(result, message, driver=driver)
    except Exception as error:
        errors.append(str(error))
        return False
    return True


# -- Section configuration --------------------------------------------------


class Location(Enum):
    """Physical backing location for one VMM section."""

    DEVICE = "device"
    HOST = "host"


@dataclass
class SectionSpec:
    """Describe one physical section of a VMM region.

    Args:
        size: Requested byte count, rounded to CUDA allocation granularity.
        location: Physical memory location used to back the section.
    """

    size: int  # requested size in bytes (will be rounded up to granularity)
    location: Location  # where the physical memory resides


def get_vmm_allocation_granularity(device_ordinal: int) -> int:
    """Return the recommended CUDA VMM granularity for a device.

    Args:
        device_ordinal: CUDA device whose physical mappings will be created.

    Returns:
        Required section alignment in bytes.
    """
    dev_prop = drv.CUmemAllocationProp()
    dev_prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    dev_prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    dev_prop.location.id = device_ordinal
    err, granularity = drv.cuMemGetAllocationGranularity(
        dev_prop,
        drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    )
    _check(err)
    return int(granularity)


# -- VMM Region --------------------------------------------------------------


class VMMRegion:
    """Own a contiguous virtual range with device and host backing.

    Each section is backed by separate physical memory created via cuMemCreate.
    The entire range is granted GPU read/write access (host sections are
    reached over PCIe / NVLink).

    Args:
        sections: Ordered physical-memory layout.
        device_ordinal: CUDA device granted access to the complete range.
    """

    def __init__(
        self,
        sections: List[SectionSpec],
        device_ordinal: int = 0,
    ) -> None:
        """Reserve and map one contiguous mixed-backing virtual range.

        Args:
            sections: Ordered physical-memory layout.
            device_ordinal: CUDA device granted access to the complete range.

        Raises:
            ValueError: If ``sections`` is empty.
            RuntimeError: If a CUDA Driver VMM operation fails.
        """
        self.device_ordinal = device_ordinal
        self._handles: list[object | None] = []
        self._mapped_section_indices: list[int] = []
        self.va_base = 0
        self.total_size = 0

        if not sections:
            raise ValueError("sections must be non-empty")

        # -- 1. Query allocation granularity -----------------------------
        dev_prop = drv.CUmemAllocationProp()
        dev_prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        dev_prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        dev_prop.location.id = device_ordinal
        granularity = get_vmm_allocation_granularity(device_ordinal)
        # Public so consumers can re-align caller-provided sizes for an
        # equality comparison against this region's actual section sizes.
        self.granularity = granularity

        # Round each section size up to granularity
        def _align(sz: int) -> int:
            """Round one section to CUDA allocation granularity.

            Args:
                sz: Requested section size.

            Returns:
                Granularity-aligned section size.
            """
            return (sz + granularity - 1) // granularity * granularity

        self.section_sizes = [_align(s.size) for s in sections]
        self.section_locations = [s.location for s in sections]
        self.total_size = sum(self.section_sizes)

        # Cumulative offsets for each section within the VA range
        self.section_offsets: list[int] = []
        offset = 0
        for sz in self.section_sizes:
            self.section_offsets.append(offset)
            offset += sz

        # -- 2. Reserve contiguous virtual address range -----------------
        err, va_base = drv.cuMemAddressReserve(self.total_size, granularity, 0, 0)
        _check(err)
        self.va_base = int(va_base)

        # -- 3. Create physical memory handles per section ---------------
        # On Grace-Blackwell (GB200/GB300), CU_MEM_LOCATION_TYPE_HOST with
        # location.id=0 returns CUDA_ERROR_INVALID_VALUE.  Use HOST_NUMA
        # with the correct NUMA node for the GPU instead, falling back to
        # generic HOST only when NUMA detection is unavailable.
        host_prop = drv.CUmemAllocationProp()
        host_prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        host_prop.allocFlags.compressionType = 0

        try:
            numa_err, numa_id = drv.cuDeviceGetAttribute(
                drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                device_ordinal,
            )
            host_numa_type = getattr(
                drv.CUmemLocationType,
                "CU_MEM_LOCATION_TYPE_HOST_NUMA",
                None,
            )
            if (
                numa_err == drv.CUresult.CUDA_SUCCESS
                and int(numa_id) >= 0
                and host_numa_type is not None
            ):
                host_prop.location.type = host_numa_type
                host_prop.location.id = int(numa_id)
                logger.info(
                    "VMM: using HOST_NUMA with NUMA node %d for device %d",
                    int(numa_id),
                    device_ordinal,
                )
            else:
                host_prop.location.type = (
                    drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
                )
                host_prop.location.id = 0
        except Exception:
            # NUMA detection not available; fall back to generic HOST.
            host_prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST
            host_prop.location.id = 0

        for i, (sz, loc) in enumerate(zip(self.section_sizes, self.section_locations)):
            prop = dev_prop if loc == Location.DEVICE else host_prop
            err, handle = drv.cuMemCreate(sz, prop, 0)
            self._check_construction(
                err,
                f"cuMemCreate failed for section {i} "
                f"(location={loc.value}, size={sz}, "
                f"{sz / (1024**3):.2f} GiB)",
            )
            self._handles.append(handle)

        # -- 4. Map physical memory into the VA range --------------------
        for index, (handle, sec_offset, sz) in enumerate(
            zip(self._handles, self.section_offsets, self.section_sizes)
        ):
            assert handle is not None
            (err,) = drv.cuMemMap(self.va_base + sec_offset, sz, 0, handle, 0)
            self._check_construction(err)
            self._mapped_section_indices.append(index)

        # -- 5. Grant GPU read/write access to the full range -----------
        access_desc = drv.CUmemAccessDesc()
        access_desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = device_ordinal
        access_desc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        (err,) = drv.cuMemSetAccess(self.va_base, self.total_size, [access_desc], 1)
        self._check_construction(err)

    def _check_construction(self, err: Any, msg: str = "") -> None:
        """Release partial state before propagating a construction failure.

        Args:
            err: CUDA result returned by the construction operation.
            msg: Additional operation context.

        Raises:
            RuntimeError: If the CUDA operation failed.
        """
        try:
            _check(err, msg)
        except Exception as construction_error:
            try:
                self.cleanup()
            except Exception as cleanup_error:
                construction_error.add_note(
                    f"VMM partial-construction cleanup also failed: {cleanup_error}"
                )
            raise

    # -- Section address helpers -----------------------------------------
    def section_ptr(self, index: int) -> int:
        """Return the start address of a physical section.

        Args:
            index: Section index in the configured layout.

        Returns:
            CUDA virtual address of the section.
        """
        return self.va_base + self.section_offsets[index]

    @property
    def num_sections(self) -> int:
        """Return the number of physical sections in the range.

        Returns:
            Number of mapped physical sections.
        """
        return len(self.section_sizes)

    def reserved_bytes(self, location: Location) -> int:
        """Return physical bytes reserved at one backing location.

        Args:
            location: Device or host backing to total.

        Returns:
            Sum of the mapped section sizes at ``location``.
        """
        return sum(
            size
            for size, section_location in zip(
                self.section_sizes, self.section_locations
            )
            if section_location == location
        )

    # -- Cleanup ---------------------------------------------------------
    def cleanup(self) -> None:
        """Release all VMM resources (unmap, release handles, free VA).

        Every driver result is checked, while independent release operations
        continue after a failure. Resources whose release failed remain owned
        by this object so a later call can retry them. During interpreter
        shutdown, when ``drv`` may already be unavailable, the process owns the
        remaining cleanup. Idempotent after successful release.

        Raises:
            RuntimeError: If any CUDA driver cleanup operation fails.
        """
        # Capture module references locally. During interpreter shutdown
        # the global names can be rebound to None before __del__ runs,
        # and ``drv.cuMemUnmap`` would then raise AttributeError.
        _drv = drv
        if _drv is None:
            return

        errors: list[str] = []
        failed_mappings: list[int] = []
        for index in reversed(self._mapped_section_indices):
            sec_offset = self.section_offsets[index]
            sz = self.section_sizes[index]
            if not _cleanup_succeeded(
                lambda sec_offset=sec_offset, sz=sz: _drv.cuMemUnmap(
                    self.va_base + sec_offset, sz
                ),
                f"cuMemUnmap failed for section {index}",
                driver=_drv,
                errors=errors,
            ):
                failed_mappings.append(index)
        self._mapped_section_indices = list(reversed(failed_mappings))

        mapped_indices = set(self._mapped_section_indices)
        for index in reversed(range(len(self._handles))):
            handle = self._handles[index]
            if handle is None or index in mapped_indices:
                continue
            if _cleanup_succeeded(
                lambda handle=handle: _drv.cuMemRelease(handle),
                f"cuMemRelease failed for handle {index}",
                driver=_drv,
                errors=errors,
            ):
                self._handles[index] = None

        if (
            self.va_base
            and not self._mapped_section_indices
            and _cleanup_succeeded(
                lambda: _drv.cuMemAddressFree(self.va_base, self.total_size),
                "cuMemAddressFree failed",
                driver=_drv,
                errors=errors,
            )
        ):
            self.va_base = 0

        if errors:
            raise RuntimeError("VMM cleanup failed: " + "; ".join(errors))
        if not any(handle is not None for handle in self._handles):
            self._handles.clear()

    def __del__(self) -> None:
        """Best-effort cleanup when explicit ownership was not released."""
        # __del__ may run during interpreter shutdown after this module's
        # globals (drv, logger) have been cleared. Suppress any exception
        # so we don't surface a noisy "Exception ignored in: __del__" trace.
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        """Return the physical layout and virtual address for diagnostics.

        Returns:
            Human-readable VMM range description.
        """
        parts = []
        for i, (sz, loc) in enumerate(zip(self.section_sizes, self.section_locations)):
            parts.append(f"{loc.value}({sz})@{hex(self.section_ptr(i))}")
        layout = ", ".join(parts)
        return (
            f"VMMRegion(total={self.total_size}, "
            f"va={hex(self.va_base)}, "
            f"layout=[{layout}])"
        )


# -- VMM-backed custom allocator for PyTorch ---------------------------------
#
# ctypes CFUNCTYPE callbacks MUST be module-level functions on this platform;
# closures / bound methods segfault when called from C.  We use module-level
# state that the callbacks read/write.

_vmm_regions: Dict[int, VMMRegion] = {}
_vmm_section_specs: List[SectionSpec] = []
_vmm_device_reserved_bytes = 0
_vmm_host_reserved_bytes = 0
_vmm_config_lock = threading.Lock()


def _track_region(region: VMMRegion) -> None:
    """Register one live VMM region and update physical-byte accounting.

    Args:
        region: Newly allocated VMM region.
    """
    global _vmm_device_reserved_bytes, _vmm_host_reserved_bytes

    _vmm_regions[region.va_base] = region
    _vmm_device_reserved_bytes += region.reserved_bytes(Location.DEVICE)
    _vmm_host_reserved_bytes += region.reserved_bytes(Location.HOST)


def _untrack_region(regions: Dict[int, VMMRegion], ptr: int) -> VMMRegion | None:
    """Remove one region by virtual address and update byte accounting.

    Args:
        regions: Live-region mapping to update.
        ptr: Base virtual address to remove.

    Returns:
        Removed region, or ``None`` when ``ptr`` is unknown.
    """
    global _vmm_device_reserved_bytes, _vmm_host_reserved_bytes

    region = regions.pop(ptr, None)
    if region is not None:
        _vmm_device_reserved_bytes -= region.reserved_bytes(Location.DEVICE)
        _vmm_host_reserved_bytes -= region.reserved_bytes(Location.HOST)
    return region


# Holds the real exception from the most recent failed _vmm_alloc. The C
# allocator callback cannot raise across the C boundary (it must return 0,
# which PyTorch turns into a generic, misleading CUDA OOM), so we stash the
# true cause here for the caller to surface. See get_last_alloc_error().
_vmm_last_alloc_errors: Dict[int, BaseException] = {}
_vmm_last_free_errors: Dict[int, BaseException] = {}

# -- Prefetch state ---------------------------------------------------------
_prefetch_lock = threading.Lock()
_prefetches: Dict[int, _VMMRegionPrefetch] = {}
_configured_prefetch: _VMMRegionPrefetch | None = None


def _allocation_specs(size: int, specs: List[SectionSpec]) -> List[SectionSpec]:
    """Return the section layout for one allocator request.

    Args:
        size: Allocation size requested by PyTorch.
        specs: Configured physical sections.

    Returns:
        Sections extended to cover an allocator size-class adjustment.

    Raises:
        ValueError: If the allocator request cannot contain the configured layout.
    """
    runtime_specs = list(specs)
    configured_total = sum(section.size for section in runtime_specs)
    if size < configured_total:
        raise ValueError(
            f"Allocation size {size} is smaller than configured "
            f"section total {configured_total} "
            f"(sections: {[section.size for section in runtime_specs]})"
        )
    if size > configured_total:
        extra = size - configured_total
        runtime_specs[-1] = SectionSpec(
            size=runtime_specs[-1].size + extra,
            location=runtime_specs[-1].location,
        )
    return runtime_specs


def _unregister_prefetch(prefetch: _VMMRegionPrefetch) -> None:
    """Remove a prefetch from the device registry if it still owns the slot.

    Args:
        prefetch: Exact owner to remove.
    """
    with _prefetch_lock:
        if _prefetches.get(prefetch.device_ordinal) is prefetch:
            _prefetches.pop(prefetch.device_ordinal)


class _VMMRegionPrefetch:
    """Own one asynchronous VMM construction until consumption or release.

    A prefetch has a single terminal owner: ``consume()`` transfers its region
    to a PyTorch memory pool, while ``close()`` releases an unconsumed region.
    The registry permits one pending prefetch per CUDA device so later context
    construction cannot observe an ambiguous virtual-address owner.
    """

    def __init__(self, sections: List[SectionSpec], device_ordinal: int) -> None:
        """Capture the exact layout and prepare its background worker.

        Args:
            sections: Physical layout for the future allocator request.
            device_ordinal: CUDA device for the prefetched range.
        """
        self.device_ordinal = int(device_ordinal)
        self._sections = [
            SectionSpec(size=section.size, location=section.location)
            for section in sections
        ]
        self._lock = threading.Lock()
        self._region: VMMRegion | None = None
        self._error: Exception | None = None
        self._state = "pending"
        self._thread = threading.Thread(
            target=self._worker,
            name=f"vmm-prefetch-{self.device_ordinal}",
            daemon=True,
        )

    def start(self) -> None:
        """Start constructing the region on the background thread."""
        self._thread.start()

    def _worker(self) -> None:
        """Construct and retain the region until its owner consumes or closes it."""
        try:
            region = VMMRegion(self._sections, device_ordinal=self.device_ordinal)
        except Exception as error:
            with self._lock:
                self._error = error
                self._state = "failed"
            logger.error("[vmm prefetch] failed:\n%s", traceback.format_exc())
            return
        with self._lock:
            self._region = region
            self._state = "ready"
        logger.info(
            "[vmm prefetch] built region (%d bytes, %.2f GiB) for device %d",
            region.total_size,
            region.total_size / (1024**3),
            self.device_ordinal,
        )

    @property
    def consumed(self) -> bool:
        """Return whether ownership moved to a PyTorch memory pool.

        Returns:
            Whether the prefetched range has been consumed.
        """
        with self._lock:
            return self._state == "consumed"

    @property
    def closed(self) -> bool:
        """Return whether the unconsumed region was released.

        Returns:
            Whether the prefetch was closed before consumption.
        """
        with self._lock:
            return self._state == "closed"

    def consume(
        self,
        *,
        size: int,
        device: int,
        specs: List[SectionSpec],
    ) -> VMMRegion:
        """Wait for and transfer an exactly matching prefetched region.

        Args:
            size: Allocation size requested by PyTorch.
            device: CUDA device ordinal for the allocation.
            specs: Physical sections configured for the allocation.

        Returns:
            The prefetched region whose ownership transfers to the allocator.

        Raises:
            RuntimeError: If construction failed or the handle is not consumable.
            ValueError: If the device or physical layout does not match.
        """
        self._thread.join()
        try:
            runtime_specs = _allocation_specs(size, specs)
        except Exception:
            self.close()
            raise
        with self._lock:
            if self._state == "failed":
                assert self._error is not None
                raise RuntimeError("DistMoE VMM prefetch failed") from self._error
            if self._state == "closed":
                raise RuntimeError("DistMoE VMM prefetch is closed")
            if self._state == "consumed":
                raise RuntimeError("DistMoE VMM prefetch was already consumed")
            if self._state != "ready" or self._region is None:
                raise RuntimeError(
                    f"DistMoE VMM prefetch has invalid state {self._state!r}"
                )
            region = self._region
            expected_sizes = [
                (
                    (section.size + region.granularity - 1)
                    // region.granularity
                    * region.granularity
                )
                for section in runtime_specs
            ]
            expected_locations = [section.location for section in runtime_specs]
            mismatch = (
                region.device_ordinal != int(device)
                or region.section_sizes != expected_sizes
                or region.section_locations != expected_locations
            )
            if mismatch:
                self._state = "closing"
            else:
                self._region = None
                self._state = "consumed"
        if mismatch:
            try:
                region.cleanup()
            except BaseException:
                with self._lock:
                    self._state = "ready"
                raise
            with self._lock:
                self._region = None
                self._state = "closed"
            _unregister_prefetch(self)
            raise ValueError(
                "Prefetched DistMoE VMM layout does not match the allocator "
                "request: "
                f"prefetched=(device={region.device_ordinal}, "
                f"sizes={region.section_sizes}, "
                f"locations={region.section_locations}), "
                f"requested=(device={device}, sizes={expected_sizes}, "
                f"locations={expected_locations})"
            )
        _unregister_prefetch(self)
        return region

    def close(self) -> None:
        """Wait for and release the region unless it was already consumed."""
        self._thread.join()
        with self._lock:
            if self._state in ("closed", "consumed"):
                region = None
            elif self._state == "failed":
                region = None
                self._state = "closed"
            else:
                region = self._region
                self._state = "closing"
        if region is not None:
            try:
                region.cleanup()
            except BaseException:
                with self._lock:
                    self._state = "ready"
                raise
            with self._lock:
                self._region = None
                self._state = "closed"
        _unregister_prefetch(self)


def _vmm_alloc(
    size: int,
    device: int,
    stream: int,
    _drv: Any = drv,
) -> int:
    """Create a VMM region for a PyTorch allocator callback.

    Args:
        size: Allocation size requested by the PyTorch caching allocator.
        device: CUDA device ordinal.
        stream: CUDA stream pointer supplied by the allocator contract.
        _drv: Captured CUDA driver module kept alive during shutdown.

    Returns:
        CUDA virtual address, or zero when allocation fails.
    """
    try:
        device = int(device)
        _vmm_last_alloc_errors.pop(device, None)
        _vmm_last_free_errors.pop(device, None)
        if _configured_prefetch is not None:
            prebuilt = _configured_prefetch.consume(
                size=size,
                device=device,
                specs=_vmm_section_specs,
            )
            ptr = prebuilt.va_base
            _track_region(prebuilt)
            logger.info(
                "[vmm_alloc] Reused prefetched region (%d bytes, %.2f GiB)",
                size,
                size / (1024**3),
            )
            return ptr

        specs = _allocation_specs(size, _vmm_section_specs)
        logger.warning(
            "[vmm_alloc] Allocating %d bytes (%.2f GiB) on device %d with %d sections: %s",
            size,
            size / (1024**3),
            device,
            len(specs),
            [
                (s.location.value, s.size, f"{s.size / (1024**3):.2f} GiB")
                for s in specs
            ],
        )
        region = VMMRegion(specs, device_ordinal=int(device))
        ptr = region.va_base
        _track_region(region)
        logger.warning("[vmm_alloc] Success: %d bytes -> %s", size, region)
        return ptr
    except Exception as e:
        if _configured_prefetch is not None:
            _configured_prefetch.close()
        # Stash the true cause for the caller (see get_last_alloc_error): the
        # OOM PyTorch raises after we return 0 is misleading.
        _vmm_last_alloc_errors[int(device)] = e
        logger.error(
            "[vmm_alloc] FAILED to allocate %d bytes (%.2f GiB) via VMM. "
            " PyTorch allocator will fail with generic error message "
            " as a result of this failure. You can ignore free memory reported "
            " by PyTorch as it is irrelevant "
            "Exception:\n%s",
            size,
            size / (1024**3),
            traceback.format_exc(),
        )
        return 0


def _vmm_free(
    ptr: int,
    size: int,
    device: int,
    stream: int,
    _drv: Any = drv,
) -> None:
    """Custom free callback: tear down the VMM region.

    Best-effort during interpreter shutdown: the C caching allocator can
    invoke this callback after Python module globals (logger, _vmm_regions)
    have been torn down, so each access is guarded.

    Args:
        ptr: CUDA virtual address returned by :func:`_vmm_alloc`.
        size: Allocation size supplied by PyTorch.
        device: CUDA device ordinal.
        stream: CUDA stream pointer supplied by the allocator contract.
        _drv: Captured CUDA driver module kept alive during shutdown.
    """
    try:
        regions = _vmm_regions
        device = int(device)
        _vmm_last_free_errors.pop(device, None)
        region = regions.get(ptr) if regions is not None else None
        if region is not None:
            description = repr(region)
            try:
                region.cleanup()
            except Exception as error:
                _vmm_last_free_errors[device] = error
                if logger is not None and traceback is not None:
                    logger.error(
                        "[vmm_free] FAILED to free %s; retaining ownership for "
                        "diagnosis or retry:\n%s",
                        description,
                        traceback.format_exc(),
                    )
                return
            _untrack_region(regions, ptr)
            if logger is not None:
                logger.warning("[vmm_free] Freed %s", description)
        elif logger is not None:
            logger.warning("[vmm_free] unknown ptr %s", hex(ptr))
    except Exception:
        if logger is not None and traceback is not None:
            try:
                logger.error("[vmm_free] exception:\n%s", traceback.format_exc())
            except Exception:
                pass


# Must keep these alive for the lifetime of the allocator
_c_vmm_alloc = None
_c_vmm_free = None
_allocator = None


def make_vmm_pool(sections: List[SectionSpec]) -> torch.cuda.MemPool:
    """Build a PyTorch memory pool for an exact physical layout.

    ``sections`` defines the physical memory layout.  Example::

        make_vmm_pool([
            SectionSpec(size=4*MB, location=Location.DEVICE),
            SectionSpec(size=2*MB, location=Location.HOST),
            SectionSpec(size=4*MB, location=Location.DEVICE),
        ])

    At allocation time the callback checks that the requested ``size``
    equals ``sum(s.size for s in sections)``.

    Active regions are tracked in the module-level ``_vmm_regions`` dict.

    Args:
        sections: Ordered, non-empty physical sections.

    Returns:
        Isolated PyTorch CUDA memory pool backed by the VMM callbacks.

    Raises:
        ValueError: If ``sections`` is empty.
    """
    global _c_vmm_alloc, _c_vmm_free, _allocator, _vmm_section_specs

    if not sections:
        raise ValueError("sections must be non-empty")

    _vmm_section_specs = list(sections)

    # All pools use the same module-level callbacks. Constructing the callback
    # objects once keeps their C function pointers alive until every device's
    # pool has released its allocation.
    if _allocator is None:
        _c_vmm_alloc = ALLOC_FN(_vmm_alloc)
        _c_vmm_free = FREE_FN(_vmm_free)
        alloc_ptr = ctypes.cast(_c_vmm_alloc, ctypes.c_void_p).value
        free_ptr = ctypes.cast(_c_vmm_free, ctypes.c_void_p).value
        _allocator = torch._C._cuda_customAllocator(alloc_ptr, free_ptr)
    return torch.cuda.MemPool(_allocator)


@contextlib.contextmanager
def configured_vmm_pool(
    sections: List[SectionSpec],
    *,
    prefetch: _VMMRegionPrefetch | None = None,
) -> Iterator[torch.cuda.MemPool]:
    """Serialize one pool configuration and its initial allocation.

    The allocator callback receives only an allocation size and device. This
    lock keeps the module-level section specification stable from pool creation
    through the single raw-tensor allocation, while allocations on different
    devices can remain live afterward.

    Args:
        sections: Ordered physical layout for the next allocation.
        prefetch: Optional exact prefetch handle for the next allocation.

    Yields:
        Configured PyTorch CUDA memory pool.
    """
    global _configured_prefetch

    with _vmm_config_lock:
        _configured_prefetch = prefetch
        try:
            yield make_vmm_pool(sections)
        finally:
            _configured_prefetch = None
            if prefetch is not None and not prefetch.consumed:
                prefetch.close()


def get_active_regions(device_ordinal: int | None = None) -> Dict[int, VMMRegion]:
    """Return active CUDA virtual ranges for diagnostics and tests.

    Args:
        device_ordinal: Optional CUDA device used to filter active ranges.

    Returns:
        Mapping from base virtual address to owning VMM region.
    """
    if device_ordinal is None:
        return dict(_vmm_regions)
    return {
        ptr: region
        for ptr, region in _vmm_regions.items()
        if region.device_ordinal == device_ordinal
    }


def correct_vmm_memory_stats(
    *,
    active_bytes: int,
    allocated_bytes: int,
    reserved_bytes: int,
) -> tuple[int, int, int, int, int]:
    """Remove host-backed VMM bytes from PyTorch's CUDA memory totals.

    This correction assumes the VMM regions and their host/device layout remain
    unchanged between ``torch.cuda.reset_peak_memory_stats()`` and this call.
    DistMoE satisfies this assumption by allocating one VMM buffer before training
    and keeping it alive with fixed residency for the entire training loop. Supporting
    dynamic VMM regions would require analyzing memory snapshot of all blocks which leads
    to much larger runtime overhead.

    Args:
        active_bytes: PyTorch-reported active CUDA bytes.
        allocated_bytes: PyTorch-reported allocated CUDA bytes.
        reserved_bytes: PyTorch-reported reserved CUDA bytes.

    Returns:
        Corrected active, allocated, and reserved bytes followed by host- and
        device-backed DistMoE VMM bytes.
    """
    return (
        active_bytes - _vmm_host_reserved_bytes,
        allocated_bytes - _vmm_host_reserved_bytes,
        reserved_bytes - _vmm_host_reserved_bytes,
        _vmm_host_reserved_bytes,
        _vmm_device_reserved_bytes,
    )


def get_last_alloc_error(device_ordinal: int | None = None) -> BaseException | None:
    """Return the real exception from the most recent failed _vmm_alloc, if any.

    Cleared at the start of every _vmm_alloc, so a non-None value means the
    last allocation attempt failed. Callers that catch the (misleading) CUDA
    OutOfMemoryError PyTorch raises after a failed VMM alloc should consult
    this to surface the true cause (e.g. host-NUMA cuMemCreate exhaustion).

    Args:
        device_ordinal: Optional CUDA device whose failure should be returned.

    Returns:
        Most recent allocator failure, or ``None`` when no failure is recorded.
    """
    if device_ordinal is not None:
        return _vmm_last_alloc_errors.get(device_ordinal)
    if not _vmm_last_alloc_errors:
        return None
    latest_device = next(reversed(_vmm_last_alloc_errors))
    return _vmm_last_alloc_errors[latest_device]


def get_last_free_error(device_ordinal: int | None = None) -> BaseException | None:
    """Return the most recent failed VMM free callback, if any.

    Args:
        device_ordinal: Optional CUDA device whose failure should be returned.

    Returns:
        Most recent free failure, or ``None`` when no failure is recorded.
    """
    if device_ordinal is not None:
        return _vmm_last_free_errors.get(device_ordinal)
    if not _vmm_last_free_errors:
        return None
    latest_device = next(reversed(_vmm_last_free_errors))
    return _vmm_last_free_errors[latest_device]


def retry_failed_vmm_free(device_ordinal: int) -> None:
    """Retry regions retained after a failed allocator free callback.

    Args:
        device_ordinal: CUDA device whose failed free should be retried.

    Raises:
        RuntimeError: If no failed callback is recorded or cleanup still fails.
    """
    previous_error = _vmm_last_free_errors.get(device_ordinal)
    if previous_error is None:
        raise RuntimeError(
            f"CUDA device {device_ordinal} has no failed DistMoE VMM free to retry"
        )
    regions = get_active_regions(device_ordinal)
    if not regions:
        raise RuntimeError(
            f"CUDA device {device_ordinal} recorded a failed DistMoE VMM free "
            "without a retained region"
        ) from previous_error

    errors: list[Exception] = []
    for ptr, region in regions.items():
        description = repr(region)
        try:
            region.cleanup()
        except Exception as error:
            errors.append(error)
            logger.error(
                "[vmm_free] retry FAILED for %s; retaining ownership:\n%s",
                description,
                traceback.format_exc(),
            )
        else:
            _untrack_region(_vmm_regions, ptr)
            logger.warning("[vmm_free] Freed %s after retry", description)

    if errors:
        retry_error = RuntimeError(
            "DistMoE VMM free retry failed: "
            + "; ".join(str(error) for error in errors)
        )
        _vmm_last_free_errors[device_ordinal] = retry_error
        raise retry_error from errors[0]
    _vmm_last_free_errors.pop(device_ordinal, None)


def prefetch_vmm_region(
    sections: List[SectionSpec], device_ordinal: int = 0
) -> _VMMRegionPrefetch:
    """Build a VMMRegion in a daemon thread to hide allocation latency.

    The returned handle must be passed to the matching allocation or explicitly
    closed. An explicit prefetch never falls back to synchronous construction.

    The sections passed here must match what ``make_vmm_pool`` will later be
    configured with after allocation-granularity and allocator-size-class
    adjustments.

    Args:
        sections: Physical layout that the subsequent allocation will request.
        device_ordinal: CUDA device for the prefetched range.

    Returns:
        Single-use owner of the asynchronous construction.

    Raises:
        ValueError: If ``sections`` is empty.
        RuntimeError: If the device already has an unconsumed prefetch.
    """
    if not sections:
        raise ValueError("sections must be non-empty")

    with _prefetch_lock:
        if device_ordinal in _prefetches:
            raise RuntimeError(
                f"CUDA device {device_ordinal} already has an unconsumed "
                "DistMoE VMM prefetch"
            )
        prefetch = _VMMRegionPrefetch(sections, device_ordinal)
        _prefetches[device_ordinal] = prefetch
    try:
        prefetch.start()
    except BaseException:
        _unregister_prefetch(prefetch)
        raise
    return prefetch


def reset_state() -> None:
    """Clean up active and prefetched regions for test teardown."""
    global \
        _c_vmm_alloc, \
        _c_vmm_free, \
        _allocator, \
        _vmm_section_specs, \
        _configured_prefetch

    with _prefetch_lock:
        prefetches = list(_prefetches.values())
        _prefetches.clear()
    for prefetch in prefetches:
        try:
            prefetch.close()
        except Exception:
            pass

    for ptr, region in list(_vmm_regions.items()):
        try:
            region.cleanup()
        except Exception:
            logger.error(
                "Failed to clean up VMM region %s during reset:\n%s",
                hex(ptr),
                traceback.format_exc(),
            )
        else:
            _untrack_region(_vmm_regions, ptr)

    _vmm_section_specs = []
    _configured_prefetch = None
    _c_vmm_alloc = None
    _c_vmm_free = None
    _allocator = None
    _vmm_last_alloc_errors.clear()
    _vmm_last_free_errors.clear()
