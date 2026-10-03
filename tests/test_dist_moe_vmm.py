# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for distributed MoE memory planning and VMM host scratch."""

import dataclasses
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import dist_moe
import dist_moe._vmm as vmm_impl
import dist_moe.api as api_impl
import pytest
import torch
import torch.distributed as dist
import torch.testing._internal.distributed.fake_pg  # noqa: F401
from dist_moe._vmm import (
    get_active_regions,
    get_last_alloc_error,
    Location,
    prefetch_vmm_region,
    reset_state,
    SectionSpec,
    VMMRegion,
)


def _is_blackwell() -> bool:
    """Return whether the current process can execute SM100 CuTe kernels."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _config(**overrides: Any) -> dist_moe.Config:
    """Build a small, exactly calculable distributed MoE configuration.

    Args:
        **overrides: Dataclass fields that replace the test defaults.

    Returns:
        Valid public configuration suitable for memory planning.
    """
    values: dict[str, Any] = {
        "num_local_input_tokens": 4,
        "hidden_dim": 128,
        "intermediate_dim": 128,
        "top_k": 2,
        "num_experts": 4,
        "max_moe_layers_per_activation_slot": 2,
        "device_scratch_capacity_factor": 1.0,
        "vmm": dist_moe.VmmConfig(total_scratch_capacity_factor=2.0),
    }
    values.update(overrides)
    return dist_moe.Config(**values)


class _FakeRegion:
    """In-memory stand-in for VMM ownership tests."""

    instances: list["_FakeRegion"] = []

    def __init__(
        self,
        sections: list[SectionSpec],
        device_ordinal: int = 0,
    ) -> None:
        """Record one construction without calling the CUDA driver.

        Args:
            sections: Physical layout for the fake region.
            device_ordinal: CUDA device recorded on the fake region.
        """
        self.device_ordinal = device_ordinal
        self.granularity = 4096
        self.section_sizes = [section.size for section in sections]
        self.section_locations = [section.location for section in sections]
        self.total_size = sum(self.section_sizes)
        self.va_base = 0x200000 + len(self.instances) * 0x100000
        self.cleanup_count = 0
        self.instances.append(self)

    def reserved_bytes(self, location: Location) -> int:
        """Return bytes assigned to one physical location."""
        return sum(
            size
            for size, section_location in zip(
                self.section_sizes, self.section_locations
            )
            if section_location == location
        )

    def cleanup(self) -> None:
        """Record release of this fake region."""
        self.cleanup_count += 1


def _mock_vmm_driver(num_sections: int) -> SimpleNamespace:
    """Build a successful in-memory CUDA driver with observable calls.

    Args:
        num_sections: Number of distinct physical allocation handles.

    Returns:
        Mock driver namespace whose operations expose call history.
    """
    success = 0
    failure = 1
    return SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=success),
        CUmemAllocationProp=lambda: SimpleNamespace(
            location=SimpleNamespace(),
            allocFlags=SimpleNamespace(),
        ),
        CUmemAccessDesc=lambda: SimpleNamespace(location=SimpleNamespace()),
        CUmemAllocationType=SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED=1),
        CUmemLocationType=SimpleNamespace(
            CU_MEM_LOCATION_TYPE_DEVICE=1,
            CU_MEM_LOCATION_TYPE_HOST=2,
        ),
        CUmemAllocationGranularity_flags=SimpleNamespace(
            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED=1,
        ),
        CUmemAccess_flags=SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=1),
        CUdevice_attribute=SimpleNamespace(CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID=1),
        cuMemGetAllocationGranularity=mock.Mock(return_value=(success, 4096)),
        cuMemAddressReserve=mock.Mock(return_value=(success, 0x100000)),
        cuDeviceGetAttribute=mock.Mock(return_value=(failure, -1)),
        cuMemCreate=mock.Mock(
            side_effect=[(success, 11 + index) for index in range(num_sections)]
        ),
        cuMemMap=mock.Mock(return_value=(success,)),
        cuMemSetAccess=mock.Mock(return_value=(success,)),
        cuMemUnmap=mock.Mock(return_value=(success,)),
        cuMemRelease=mock.Mock(return_value=(success,)),
        cuMemAddressFree=mock.Mock(return_value=(success,)),
    )


class DistMoeMemoryPlanTest(unittest.TestCase):
    """CPU tests for the interpretable memory-planning API."""

    def test_derives_original_planner_sizes(self) -> None:
        """Require every BF16 plan component to match the reference formulas.

        Exact receive rows and activation, device/host scratch, and total bytes
        protect the public planner contract from silent accounting drift.
        """
        plan = dist_moe.plan_memory(_config(), ep_size=2)

        self.assertEqual(plan.balanced_recv_rows, 8)
        self.assertEqual(plan.device_scratch_capacity_rows, 8)
        self.assertEqual(plan.total_scratch_capacity_rows, 16)
        self.assertEqual(plan.saved_input_bytes_per_layer, 1024)
        self.assertEqual(plan.device_scratch_bytes, 12288)
        self.assertEqual(plan.host_scratch_bytes, 12288)
        self.assertEqual(plan.minimum_activation_slot_bytes, 2048)
        self.assertEqual(plan.balanced_full_save_activation_slot_bytes, 16384)
        self.assertEqual(plan.maximum_useful_activation_slot_bytes, 28672)
        self.assertIsNone(plan.activation_slot_capacity_factor)
        self.assertEqual(plan.total_activation_bytes, 2048)
        self.assertEqual(plan.total_device_buffer_bytes, 14336)
        self.assertEqual(plan.activation_slot_bytes, 2048)
        self.assertEqual(plan.total_virtual_bytes, 26624)

    def test_prefetch_policy_does_not_change_the_memory_plan(self) -> None:
        """VMM prefetch controls initialization order, never buffer geometry."""
        prefetched = _config(
            vmm=dist_moe.VmmConfig(
                total_scratch_capacity_factor=2.0,
                prefetch=True,
            )
        )
        assert prefetched.vmm is not None
        synchronous = dataclasses.replace(
            prefetched,
            vmm=dataclasses.replace(prefetched.vmm, prefetch=False),
        )

        self.assertEqual(
            dist_moe.plan_memory(prefetched, ep_size=2),
            dist_moe.plan_memory(synchronous, ep_size=2),
        )

    def test_documented_large_model_memory_plans(self) -> None:
        """Keep documented large-model memory totals equal to planner output.

        BF16 and MXFP8 minimum, exact, and balanced-full-save budgets are checked
        so the memory-planner guide remains executable guidance.
        """
        base = dist_moe.Config(
            num_local_input_tokens=4096,
            hidden_dim=4096,
            intermediate_dim=14336,
            top_k=8,
            num_experts=128,
            max_moe_layers_per_activation_slot=32,
            device_scratch_capacity_factor=1.5,
            num_activation_slots=1,
            vmm=dist_moe.VmmConfig(total_scratch_capacity_factor=16.0),
        )
        expected = {
            "bf16": (
                (1_073_741_824, 7_449_083_904, 72_007_811_072, 8_522_825_728),
                (17_179_869_184, 7_449_083_904, 72_007_811_072, 24_628_953_088),
                (77_309_411_328, 7_449_083_904, 72_007_811_072, 84_758_495_232),
            ),
            "mxfp8": (
                (1_073_741_824, 8_561_811_456, 81_282_465_792, 9_635_553_280),
                (17_179_869_184, 8_561_811_456, 81_282_465_792, 25_741_680_640),
                (90_839_973_888, 8_561_811_456, 81_282_465_792, 99_401_785_344),
            ),
        }

        for name, block_scaled in (
            ("bf16", None),
            ("mxfp8", dist_moe.BlockScaledConfig()),
        ):
            with self.subTest(precision=name):
                config = dataclasses.replace(base, block_scaled=block_scaled)
                minimum = dist_moe.plan_memory(config, ep_size=16)
                middle = dist_moe.plan_memory(
                    dataclasses.replace(
                        config,
                        activation_slot_bytes=16 * 1024**3,
                    ),
                    ep_size=16,
                )
                balanced = dist_moe.plan_memory(
                    dataclasses.replace(
                        config,
                        activation_slot_capacity_factor=1.0,
                    ),
                    ep_size=16,
                )
                actual = tuple(
                    (
                        plan.total_activation_bytes,
                        plan.device_scratch_bytes,
                        plan.host_scratch_bytes,
                        plan.total_device_buffer_bytes,
                    )
                    for plan in (minimum, middle, balanced)
                )
                self.assertEqual(actual, expected[name])

    def test_activation_slot_capacity_modes_are_independent_of_scratch(self) -> None:
        """Resolve exact and balanced policies without coupling them to scratch."""
        minimum = dist_moe.plan_memory(_config(vmm=None), ep_size=2)
        exact = dist_moe.plan_memory(
            _config(
                vmm=None,
                activation_slot_bytes=minimum.minimum_activation_slot_bytes + 1,
                num_activation_slots=3,
            ),
            ep_size=2,
        )
        factor_plans = {
            factor: dist_moe.plan_memory(
                _config(activation_slot_capacity_factor=factor), ep_size=2
            )
            for factor in (0.0, 1.0, 1.5)
        }
        factor = factor_plans[1.5]
        larger_scratch = dist_moe.plan_memory(
            _config(
                activation_slot_capacity_factor=1.5,
                device_scratch_capacity_factor=2.0,
                vmm=dist_moe.VmmConfig(total_scratch_capacity_factor=2.0),
            ),
            ep_size=2,
        )

        self.assertEqual(exact.activation_slot_bytes, 2176)
        self.assertEqual(exact.total_activation_bytes, 3 * 2176)
        optional_balanced_bytes = (
            minimum.balanced_full_save_activation_slot_bytes
            - minimum.minimum_activation_slot_bytes
        )
        self.assertEqual(
            factor_plans[0.0].activation_slot_bytes,
            minimum.minimum_activation_slot_bytes,
        )
        self.assertEqual(
            factor_plans[1.0].activation_slot_bytes,
            minimum.balanced_full_save_activation_slot_bytes,
        )
        expected_factor_bytes = minimum.minimum_activation_slot_bytes + int(
            1.5 * optional_balanced_bytes
        )
        self.assertEqual(factor.activation_slot_bytes, expected_factor_bytes)
        self.assertEqual(
            factor.activation_slot_bytes,
            larger_scratch.activation_slot_bytes,
        )
        self.assertNotEqual(
            factor.device_scratch_bytes,
            larger_scratch.device_scratch_bytes,
        )

        mxfp8 = dist_moe.plan_memory(
            _config(
                num_local_input_tokens=256,
                activation_slot_capacity_factor=1.0,
                block_scaled=dist_moe.BlockScaledConfig(),
            ),
            ep_size=2,
        )
        self.assertEqual(
            mxfp8.activation_slot_bytes,
            mxfp8.balanced_full_save_activation_slot_bytes,
        )
        mxfp8_larger = dist_moe.plan_memory(
            dataclasses.replace(
                _config(
                    num_local_input_tokens=256,
                    block_scaled=dist_moe.BlockScaledConfig(),
                ),
                activation_slot_capacity_factor=1.5,
            ),
            ep_size=2,
        )
        self.assertEqual(
            mxfp8_larger.activation_slot_bytes,
            mxfp8_larger.minimum_activation_slot_bytes
            + int(
                1.5
                * (
                    mxfp8_larger.balanced_full_save_activation_slot_bytes
                    - mxfp8_larger.minimum_activation_slot_bytes
                )
            ),
        )

    def test_rejects_budget_below_all_recompute_minimum(self) -> None:
        """A too-small activation slot reports its model-derived minimum."""
        config = _config(activation_slot_bytes=1024)
        with self.assertRaisesRegex(ValueError, "minimum required 2048"):
            dist_moe.plan_memory(config, ep_size=2)

    def test_host_overflow_must_extend_device_capacity(self) -> None:
        """Combined VMM capacity cannot be smaller than device capacity."""
        config = _config(
            device_scratch_capacity_factor=2.0,
            vmm=dist_moe.VmmConfig(total_scratch_capacity_factor=1.0),
        )
        with self.assertRaisesRegex(ValueError, "must cover at least"):
            dist_moe.plan_memory(config, ep_size=2)

    def test_explain_reports_physical_sections_and_budget_bounds(self) -> None:
        """The log representation makes every memory choice inspectable."""
        explanation = dist_moe.plan_memory(_config(), ep_size=2).explain()
        self.assertIn("activation slots", explanation)
        self.assertIn("device scratch", explanation)
        self.assertIn("host overflow scratch", explanation)
        self.assertIn("minimum", explanation)
        self.assertIn("maximum useful", explanation)

    def test_explain_identifies_scratch_only_inference(self) -> None:
        """Inference plans distinguish lower scratch from saved activations."""
        plan = dist_moe.plan_memory(_config(inference=True), ep_size=2)
        explanation = plan.explain()

        self.assertEqual(plan.minimum_activation_slot_bytes, 0)
        self.assertEqual(plan.balanced_full_save_activation_slot_bytes, 0)
        self.assertEqual(plan.maximum_useful_activation_slot_bytes, 0)
        self.assertEqual(plan.total_activation_bytes, 0)
        self.assertEqual(plan.total_device_buffer_bytes, plan.device_scratch_bytes)
        self.assertIn("activation slots: 0 slot(s)", explanation)
        self.assertIn("scratch-only inference; no saved activations", explanation)
        self.assertNotIn("across", explanation)

    def test_plans_blockscaled_activation_and_host_scratch(self) -> None:
        """MXFP8 uses the same activation buffer and VMM overflow accounting."""
        config = _config(
            num_local_input_tokens=256,
            block_scaled=dist_moe.BlockScaledConfig(
                format=dist_moe.BlockScaledFormat.MXFP8_E4M3
            ),
        )
        plan = dist_moe.plan_memory(config, ep_size=2)
        self.assertEqual(plan.device_scratch_capacity_rows, 640)
        self.assertEqual(plan.total_scratch_capacity_rows, 1152)
        self.assertGreater(plan.total_activation_bytes, 0)
        self.assertGreater(plan.host_scratch_bytes, 0)


class VmmOwnershipTest(unittest.TestCase):
    """CPU-only tests for VMM construction and prefetch ownership."""

    def setUp(self) -> None:
        """Reset fake-region observations before each test."""
        _FakeRegion.instances.clear()

    def tearDown(self) -> None:
        """Release module state after each ownership test."""
        reset_state()
        api_impl._VMM_ACTIVE_DEVICES.clear()

    def test_context_prefetches_before_communication_and_closes_on_failure(
        self,
    ) -> None:
        """Context-owned prefetch overlaps setup and cannot leak on failure."""
        events: list[str] = []
        prefetch = mock.Mock(
            consumed=False,
        )
        prefetch.close.side_effect = [RuntimeError("cleanup failed"), None]
        plan = SimpleNamespace(
            uses_host_scratch=True,
            vmm_device_prefix_bytes=4096,
            vmm_host_section_bytes=4096,
            vmm_device_scratch_section_bytes=4096,
            explain=lambda: "test VMM plan",
        )

        def fail_communication(*_args: Any) -> None:
            """Record ordering before simulating collective setup failure."""
            events.append("communication")
            raise RuntimeError("communication failed")

        def start_vmm_prefetch(**_kwargs: Any) -> mock.Mock:
            """Record prefetch ordering and return its observable owner."""
            self.assertIn(0, api_impl._VMM_ACTIVE_DEVICES)
            events.append("prefetch")
            return prefetch

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=1),
            mock.patch.object(
                api_impl,
                "_resolve_cuda_device",
                return_value=torch.device("cuda", 0),
            ),
            mock.patch.object(api_impl, "plan_memory", return_value=plan),
            mock.patch.object(api_impl, "is_fake_process_group", return_value=False),
            mock.patch.object(
                api_impl,
                "_prefetch_vmm_from_sections",
                side_effect=start_vmm_prefetch,
            ) as start_prefetch,
            mock.patch.object(
                api_impl,
                "_create_comm_buffers",
                side_effect=fail_communication,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "communication failed"):
                dist_moe.create_context(group=mock.Mock(), config=_config())

        start_prefetch.assert_called_once()
        self.assertEqual(events, ["prefetch", "communication"])
        self.assertEqual(prefetch.close.call_count, 2)
        self.assertNotIn(0, api_impl._VMM_ACTIVE_DEVICES)

    def test_context_retains_reservation_when_prefetch_cleanup_fails(self) -> None:
        """A persistent prefetch cleanup failure must block device reuse."""
        prefetch = mock.Mock(consumed=False)
        prefetch.close.side_effect = RuntimeError("cleanup failed")
        plan = SimpleNamespace(
            uses_host_scratch=True,
            vmm_device_prefix_bytes=4096,
            vmm_host_section_bytes=4096,
            vmm_device_scratch_section_bytes=4096,
            explain=lambda: "test VMM plan",
        )

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=1),
            mock.patch.object(
                api_impl,
                "_resolve_cuda_device",
                return_value=torch.device("cuda", 0),
            ),
            mock.patch.object(api_impl, "plan_memory", return_value=plan),
            mock.patch.object(api_impl, "is_fake_process_group", return_value=False),
            mock.patch.object(
                api_impl,
                "_prefetch_vmm_from_sections",
                return_value=prefetch,
            ),
            mock.patch.object(
                api_impl,
                "_create_comm_buffers",
                side_effect=RuntimeError("communication failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "cleanup retry also failed"),
        ):
            dist_moe.create_context(group=mock.Mock(), config=_config())

        self.assertEqual(prefetch.close.call_count, 2)
        self.assertIn(0, api_impl._VMM_ACTIVE_DEVICES)

    def test_context_can_disable_vmm_prefetch(self) -> None:
        """The serial policy reaches communication without starting prefetch."""
        plan = SimpleNamespace(
            uses_host_scratch=True,
            vmm_device_prefix_bytes=4096,
            vmm_host_section_bytes=4096,
            vmm_device_scratch_section_bytes=4096,
            explain=lambda: "test VMM plan",
        )
        config = _config(
            vmm=dist_moe.VmmConfig(
                total_scratch_capacity_factor=2.0,
                prefetch=False,
            )
        )

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=1),
            mock.patch.object(
                api_impl,
                "_resolve_cuda_device",
                return_value=torch.device("cuda", 0),
            ),
            mock.patch.object(api_impl, "plan_memory", return_value=plan),
            mock.patch.object(api_impl, "is_fake_process_group", return_value=False),
            mock.patch.object(
                api_impl,
                "_prefetch_vmm_from_sections",
            ) as start_prefetch,
            mock.patch.object(
                api_impl,
                "_create_comm_buffers",
                side_effect=RuntimeError("communication failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "communication failed"):
                dist_moe.create_context(group=mock.Mock(), config=config)

        start_prefetch.assert_not_called()
        self.assertNotIn(0, api_impl._VMM_ACTIVE_DEVICES)

    def test_context_releases_consumed_vmm_after_construction_failure(self) -> None:
        """A post-allocation failure tears down VMM before releasing its guard."""
        prefetch = mock.Mock(consumed=True)
        plan = SimpleNamespace(
            uses_host_scratch=True,
            vmm_device_prefix_bytes=4096,
            vmm_host_section_bytes=4096,
            vmm_device_scratch_section_bytes=4096,
            total_virtual_bytes=12288,
            total_activation_bytes=4096,
            device_scratch_bytes=4096,
            explain=lambda: "test VMM plan",
        )

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
            mock.patch.object(dist, "get_world_size", return_value=1),
            mock.patch.object(
                api_impl,
                "_resolve_cuda_device",
                return_value=torch.device("cuda", 0),
            ),
            mock.patch.object(api_impl, "plan_memory", return_value=plan),
            mock.patch.object(api_impl, "is_fake_process_group", return_value=False),
            mock.patch.object(
                api_impl,
                "_prefetch_vmm_from_sections",
                return_value=prefetch,
            ),
            mock.patch.object(
                api_impl, "_create_comm_buffers", return_value=mock.Mock()
            ),
            mock.patch.object(
                api_impl,
                "_allocate_vmm_buffer",
                return_value=(mock.Mock(), mock.Mock()),
            ),
            mock.patch(
                "dist_moe._activation_buffer.ActivationBuffer.create_from_buffer",
                side_effect=RuntimeError("activation buffer failed"),
            ),
            mock.patch.object(vmm_impl, "get_last_free_error", return_value=None),
            mock.patch.object(vmm_impl, "get_active_regions", return_value={}),
            self.assertRaisesRegex(RuntimeError, "activation buffer failed"),
        ):
            dist_moe.create_context(group=mock.Mock(), config=_config())

        prefetch.close.assert_called_once_with()
        empty_cache.assert_called_once_with()
        self.assertNotIn(0, api_impl._VMM_ACTIVE_DEVICES)

    def test_partial_map_failure_unwinds_completed_mappings(self) -> None:
        """A failed map unmaps its completed prefix before releasing handles."""
        success = 0
        failure = 1
        driver = _mock_vmm_driver(num_sections=3)
        driver.cuMemMap.side_effect = [(success,), (failure,)]
        sections = [
            SectionSpec(size=4096, location=Location.DEVICE),
            SectionSpec(size=4096, location=Location.DEVICE),
            SectionSpec(size=4096, location=Location.DEVICE),
        ]

        with mock.patch.object(vmm_impl, "drv", driver):
            with self.assertRaisesRegex(RuntimeError, "CUDA driver error"):
                VMMRegion(sections, device_ordinal=0)

        self.assertEqual(
            driver.cuMemUnmap.call_args_list,
            [mock.call(0x100000, 4096)],
        )
        self.assertEqual(
            driver.cuMemRelease.call_args_list,
            [mock.call(13), mock.call(12), mock.call(11)],
        )
        driver.cuMemAddressFree.assert_called_once_with(0x100000, 3 * 4096)

    def test_cleanup_preserves_mapped_handle_and_va_for_retry(self) -> None:
        """A failed unmap retains its associated handle and virtual range."""
        success = 0
        failure = 1
        driver = _mock_vmm_driver(num_sections=2)
        sections = [
            SectionSpec(size=4096, location=Location.DEVICE),
            SectionSpec(size=4096, location=Location.HOST),
        ]

        with mock.patch.object(vmm_impl, "drv", driver):
            region = VMMRegion(sections, device_ordinal=0)
            driver.cuMemUnmap.side_effect = [(failure,), (success,)]

            with self.assertRaisesRegex(RuntimeError, "cuMemUnmap"):
                region.cleanup()

            self.assertEqual(region._mapped_section_indices, [1])
            self.assertEqual(region._handles, [None, 12])
            self.assertEqual(region.va_base, 0x100000)
            self.assertEqual(driver.cuMemUnmap.call_count, 2)
            driver.cuMemRelease.assert_called_once_with(11)
            driver.cuMemAddressFree.assert_not_called()

            driver.cuMemUnmap.side_effect = None
            driver.cuMemUnmap.return_value = (success,)
            region.cleanup()

        self.assertEqual(region._mapped_section_indices, [])
        self.assertEqual(region._handles, [])
        self.assertEqual(region.va_base, 0)

    def test_cleanup_checks_release_and_address_results_without_compacting(
        self,
    ) -> None:
        """Failed handle and VA releases retain exact retry state."""
        success = 0
        failure = 1
        driver = _mock_vmm_driver(num_sections=2)
        sections = [
            SectionSpec(size=4096, location=Location.DEVICE),
            SectionSpec(size=4096, location=Location.HOST),
        ]

        with mock.patch.object(vmm_impl, "drv", driver):
            region = VMMRegion(sections, device_ordinal=0)
            driver.cuMemRelease.side_effect = [(failure,), (success,)]
            driver.cuMemAddressFree.return_value = (failure,)

            with self.assertRaisesRegex(
                RuntimeError,
                "cuMemRelease.*cuMemAddressFree",
            ):
                region.cleanup()

            self.assertEqual(region._mapped_section_indices, [])
            self.assertEqual(region._handles, [None, 12])
            self.assertEqual(region.va_base, 0x100000)
            self.assertEqual(driver.cuMemUnmap.call_count, 2)
            self.assertEqual(
                driver.cuMemRelease.call_args_list,
                [mock.call(12), mock.call(11)],
            )
            driver.cuMemAddressFree.assert_called_once_with(0x100000, 8192)

            driver.cuMemRelease.side_effect = None
            driver.cuMemRelease.return_value = (success,)
            driver.cuMemAddressFree.return_value = (success,)
            region.cleanup()

        self.assertEqual(region._handles, [])
        self.assertEqual(region.va_base, 0)

    def test_construction_error_keeps_cleanup_failure_as_context(self) -> None:
        """Partial cleanup does not replace the root construction error."""
        failure = 1
        driver = _mock_vmm_driver(num_sections=1)
        region = object.__new__(VMMRegion)
        region.device_ordinal = 0
        region._handles = [11]
        region._mapped_section_indices = [0]
        region.va_base = 0x100000
        region.total_size = 4096
        region.section_offsets = [0]
        region.section_sizes = [4096]
        region.section_locations = [Location.DEVICE]
        driver.cuMemUnmap.return_value = (failure,)

        with mock.patch.object(vmm_impl, "drv", driver):
            with self.assertRaisesRegex(RuntimeError, "construction failed") as caught:
                region._check_construction(failure, "construction failed")
            self.assertTrue(
                any(
                    "partial-construction cleanup also failed" in note
                    for note in caught.exception.__notes__
                )
            )
            driver.cuMemUnmap.return_value = (0,)
            region.cleanup()

    def test_free_callback_untracks_only_after_cleanup_succeeds(self) -> None:
        """A callback failure remains tracked and is observable for retry."""
        region = _FakeRegion(
            [SectionSpec(size=4096, location=Location.DEVICE)],
            device_ordinal=0,
        )
        region.cleanup = mock.Mock(side_effect=RuntimeError("unmap failed"))
        vmm_impl._track_region(region)

        vmm_impl._vmm_free(region.va_base, region.total_size, 0, 0)

        self.assertIs(get_active_regions(0)[region.va_base], region)
        self.assertRegex(
            str(vmm_impl.get_last_free_error(0)),
            "unmap failed",
        )

        region.cleanup.side_effect = None
        vmm_impl._vmm_free(region.va_base, region.total_size, 0, 0)

        self.assertEqual(get_active_regions(0), {})
        self.assertIsNone(vmm_impl.get_last_free_error(0))

    def test_prefetch_is_single_owner_and_consumed_exactly_once(self) -> None:
        """The configured callback consumes its exact prefetch handle once."""
        sections = [SectionSpec(size=4096, location=Location.DEVICE)]
        with mock.patch.object(vmm_impl, "VMMRegion", _FakeRegion):
            prefetch = prefetch_vmm_region(sections, device_ordinal=0)
            with (
                mock.patch.object(vmm_impl, "_vmm_section_specs", sections),
                mock.patch.object(vmm_impl, "_configured_prefetch", prefetch),
            ):
                ptr = vmm_impl._vmm_alloc(4096, 0, 0)

        self.assertEqual(ptr, _FakeRegion.instances[0].va_base)
        self.assertTrue(prefetch.consumed)
        self.assertEqual(len(_FakeRegion.instances), 1)
        vmm_impl._vmm_free(ptr, 4096, 0, 0)

    def test_prefetch_close_retains_ownership_when_cleanup_fails(self) -> None:
        """A failed close remains retryable and blocks a second owner."""
        sections = [SectionSpec(size=4096, location=Location.DEVICE)]
        with mock.patch.object(vmm_impl, "VMMRegion", _FakeRegion):
            prefetch = prefetch_vmm_region(sections, device_ordinal=0)
            region = _FakeRegion.instances[0]
            region.cleanup = mock.Mock(side_effect=RuntimeError("unmap failed"))

            with self.assertRaisesRegex(RuntimeError, "unmap failed"):
                prefetch.close()
            self.assertFalse(prefetch.closed)
            with self.assertRaisesRegex(RuntimeError, "unconsumed"):
                prefetch_vmm_region(sections, device_ordinal=0)

            region.cleanup.side_effect = None
            prefetch.close()

        self.assertTrue(prefetch.closed)

    def test_explicit_prefetch_mismatch_does_not_fall_back(self) -> None:
        """A layout mismatch fails instead of constructing a second region."""
        sections = [SectionSpec(size=4096, location=Location.DEVICE)]
        with mock.patch.object(vmm_impl, "VMMRegion", _FakeRegion):
            prefetch = prefetch_vmm_region(sections, device_ordinal=0)
            with (
                mock.patch.object(vmm_impl, "_vmm_section_specs", sections),
                mock.patch.object(vmm_impl, "_configured_prefetch", prefetch),
            ):
                ptr = vmm_impl._vmm_alloc(8192, 0, 0)

        self.assertEqual(ptr, 0)
        self.assertTrue(prefetch.closed)
        self.assertFalse(prefetch.consumed)
        self.assertIsInstance(get_last_alloc_error(0), ValueError)
        self.assertEqual(len(_FakeRegion.instances), 1)

    def test_explicit_prefetch_failure_does_not_retry_synchronously(self) -> None:
        """A failed background construction remains the allocator failure."""
        sections = [SectionSpec(size=4096, location=Location.DEVICE)]
        constructor = mock.Mock(side_effect=RuntimeError("prefetch failed"))
        with mock.patch.object(vmm_impl, "VMMRegion", constructor):
            prefetch = prefetch_vmm_region(sections, device_ordinal=0)
            with (
                mock.patch.object(vmm_impl, "_vmm_section_specs", sections),
                mock.patch.object(vmm_impl, "_configured_prefetch", prefetch),
            ):
                ptr = vmm_impl._vmm_alloc(4096, 0, 0)

        self.assertEqual(ptr, 0)
        self.assertTrue(prefetch.closed)
        constructor.assert_called_once_with(sections, device_ordinal=0)
        error = get_last_alloc_error(0)
        self.assertIsInstance(error, RuntimeError)
        self.assertIsInstance(error.__cause__, RuntimeError)

    def test_context_releases_pool_before_emptying_allocator_cache(self) -> None:
        """The private pool becomes freeable before cached regions are drained."""
        activation_buffer = SimpleNamespace(
            buffer=SimpleNamespace(device=torch.device("cuda", 0))
        )
        context = dist_moe.Context._create(
            group=mock.Mock(),
            buffers=mock.Mock(),
            activation_buffer=activation_buffer,
            config=_config(),
            memory_plan=mock.Mock(),
            context_id="123",
            vmm_pool=object(),
        )
        api_impl._VMM_ACTIVE_DEVICES.add(0)
        events = []

        def assert_pool_is_released() -> None:
            self.assertIsNone(context._vmm_pool)
            self.assertEqual(events, ["collect"])

        with (
            mock.patch.object(torch.cuda, "synchronize"),
            mock.patch.object(
                torch,
                "empty",
                return_value=SimpleNamespace(device=torch.device("cuda", 0)),
            ),
            mock.patch.object(
                torch.cuda,
                "empty_cache",
                side_effect=assert_pool_is_released,
            ) as empty_cache,
            mock.patch.object(
                api_impl.gc,
                "collect",
                side_effect=lambda: events.append("collect"),
            ) as collect,
        ):
            context.close()

        collect.assert_called_once_with()
        empty_cache.assert_called_once_with()
        self.assertNotIn(0, api_impl._VMM_ACTIVE_DEVICES)
        self.assertTrue(context._closed)

    def test_context_close_retries_callback_failure_before_releasing_guard(
        self,
    ) -> None:
        """A transient callback failure is retried before device reuse."""
        device = torch.device("cuda", 0)
        context = dist_moe.Context._create(
            group=mock.Mock(),
            buffers=mock.Mock(),
            activation_buffer=SimpleNamespace(buffer=SimpleNamespace(device=device)),
            config=_config(),
            memory_plan=mock.Mock(),
            context_id="124",
            vmm_pool=object(),
        )
        region = _FakeRegion(
            [SectionSpec(size=4096, location=Location.DEVICE)],
            device_ordinal=device.index,
        )
        region.cleanup = mock.Mock(
            side_effect=[RuntimeError("cuMemUnmap failed"), None]
        )
        vmm_impl._track_region(region)
        api_impl._VMM_ACTIVE_DEVICES.add(device.index)

        def invoke_free_callback() -> None:
            vmm_impl._vmm_free(region.va_base, region.total_size, device.index, 0)

        with (
            mock.patch.object(torch.cuda, "synchronize"),
            mock.patch.object(
                torch,
                "empty",
                return_value=SimpleNamespace(device=device),
            ),
            mock.patch.object(
                torch.cuda, "empty_cache", side_effect=invoke_free_callback
            ),
            mock.patch.object(api_impl.gc, "collect"),
        ):
            context.close()

        self.assertEqual(region.cleanup.call_count, 2)
        self.assertEqual(get_active_regions(device.index), {})
        self.assertNotIn(device.index, api_impl._VMM_ACTIVE_DEVICES)
        self.assertTrue(context._closed)

    def test_context_close_retains_guard_when_cleanup_retry_fails(self) -> None:
        """Persistent cleanup failure blocks another VMM context on the device."""
        device = torch.device("cuda", 0)
        context = dist_moe.Context._create(
            group=mock.Mock(),
            buffers=mock.Mock(),
            activation_buffer=SimpleNamespace(buffer=SimpleNamespace(device=device)),
            config=_config(),
            memory_plan=mock.Mock(),
            context_id="125",
            vmm_pool=object(),
        )
        region = _FakeRegion(
            [SectionSpec(size=4096, location=Location.DEVICE)],
            device_ordinal=device.index,
        )
        region.cleanup = mock.Mock(side_effect=RuntimeError("cuMemUnmap failed"))
        vmm_impl._track_region(region)
        api_impl._VMM_ACTIVE_DEVICES.add(device.index)

        def invoke_free_callback() -> None:
            vmm_impl._vmm_free(region.va_base, region.total_size, device.index, 0)

        with (
            mock.patch.object(torch.cuda, "synchronize"),
            mock.patch.object(
                torch,
                "empty",
                return_value=SimpleNamespace(device=device),
            ),
            mock.patch.object(
                torch.cuda, "empty_cache", side_effect=invoke_free_callback
            ),
            mock.patch.object(api_impl.gc, "collect"),
            self.assertRaisesRegex(RuntimeError, "cleanup retry also failed"),
        ):
            context.close()

        self.assertIs(get_active_regions(device.index)[region.va_base], region)
        self.assertIn(device.index, api_impl._VMM_ACTIVE_DEVICES)
        self.assertTrue(context._closed)

        region.cleanup.side_effect = None
        with (
            mock.patch.object(torch.cuda, "empty_cache"),
            mock.patch.object(api_impl.gc, "collect"),
        ):
            context.close()
        self.assertEqual(get_active_regions(device.index), {})
        self.assertNotIn(device.index, api_impl._VMM_ACTIVE_DEVICES)


@pytest.mark.gpus_needed_1
@pytest.mark.gb10x
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DistMoeVmmAllocationTest(unittest.TestCase):
    """CUDA allocation tests for the host-backed VMM-backed activation buffer."""

    def tearDown(self) -> None:
        """Release VMM state and the fake process group after each test."""
        if dist.is_initialized():
            dist.destroy_process_group()
        reset_state()

    def test_prefetched_context_uses_host_overflow_and_releases_it(self) -> None:
        """BF16 context-owned prefetch builds and releases the VMM-backed activation buffer."""
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        config = _config(
            num_local_input_tokens=256,
            hidden_dim=128,
            intermediate_dim=8192,
            top_k=1,
            num_experts=1,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=0.5,
            vmm=dist_moe.VmmConfig(total_scratch_capacity_factor=1.0),
        )
        context = dist_moe.create_context(
            group=dist.group.WORLD,
            config=config,
            device=device,
        )
        self.assertTrue(context.memory_plan.uses_host_scratch)
        self.assertEqual(
            context.activation_buffer.device_scratch_size,
            context.memory_plan.vmm_device_scratch_section_bytes,
        )
        self.assertEqual(
            context.activation_buffer.host_scratch_size,
            context.memory_plan.vmm_host_section_bytes,
        )
        regions = get_active_regions()
        self.assertEqual(len(regions), 1)
        region = next(iter(regions.values()))
        self.assertEqual(
            region.section_locations,
            [Location.DEVICE, Location.HOST, Location.DEVICE],
        )
        self.assertEqual(
            region.section_sizes,
            [
                context.memory_plan.vmm_device_prefix_bytes,
                context.memory_plan.vmm_host_section_bytes,
                context.memory_plan.vmm_device_scratch_section_bytes,
            ],
        )
        self.assertEqual(
            context.activation_buffer.activation_slot_bytes,
            context.memory_plan.activation_slot_bytes,
        )
        self.assertGreater(context.memory_plan.vmm_padding_bytes, 0)
        with self.assertRaisesRegex(RuntimeError, "already owns a live"):
            dist_moe.create_context(
                group=dist.group.WORLD,
                config=config,
                device=device,
            )

        def execute(current: dist_moe.Context) -> None:
            """Run one BF16 forward/backward and require host-scratch use."""
            torch.manual_seed(19)
            x_TD = torch.randn(
                config.num_local_input_tokens,
                config.hidden_dim,
                dtype=torch.bfloat16,
                device=device,
                requires_grad=True,
            )
            topk_expert_ids_TK = torch.zeros(
                config.num_local_input_tokens,
                config.top_k,
                dtype=torch.int64,
                device=device,
            )
            topk_scores_TK = torch.ones(
                config.num_local_input_tokens,
                config.top_k,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            w13_EFD = torch.randn(
                config.num_experts,
                2 * config.intermediate_dim,
                config.hidden_dim,
                dtype=torch.bfloat16,
                device=device,
                requires_grad=True,
            )
            w2_EDF = torch.randn(
                config.num_experts,
                config.hidden_dim,
                config.intermediate_dim,
                dtype=torch.bfloat16,
                device=device,
                requires_grad=True,
            )
            output_TD = dist_moe.routed_experts(
                x_TD,
                topk_expert_ids_TK,
                topk_scores_TK,
                w13_EFD,
                w2_EDF,
                current,
            )
            output_TD.float().sum().backward()
            self.assertIsNotNone(x_TD.grad)
            overflow = current.activation_buffer.check_overflow()
            self.assertIsNotNone(overflow.host_scratch_used)
            self.assertGreater(overflow.host_scratch_used, 0)

        execute(context)
        context.close()
        self.assertEqual(get_active_regions(), {})

        recreated = dist_moe.create_context(
            group=dist.group.WORLD,
            config=config,
            device=device,
        )
        try:
            execute(recreated)
        finally:
            recreated.close()
        self.assertEqual(get_active_regions(), {})

    def test_blockscaled_synchronous_vmm_uses_host_overflow_and_releases_it(
        self,
    ) -> None:
        """MXFP8 synchronous setup builds the same device-host-device buffer."""
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        config = _config(
            num_local_input_tokens=256,
            hidden_dim=128,
            intermediate_dim=128,
            top_k=1,
            num_experts=1,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=0.5,
            activation_slot_bytes=16 * 1024 * 1024,
            vmm=dist_moe.VmmConfig(
                total_scratch_capacity_factor=1.0,
                prefetch=False,
            ),
            block_scaled=dist_moe.BlockScaledConfig(),
        )
        context = dist_moe.create_context(
            group=dist.group.WORLD,
            config=config,
            device=device,
        )
        self.assertTrue(context.memory_plan.uses_host_scratch)
        self.assertEqual(
            context.activation_buffer.device_scratch_size,
            context.memory_plan.vmm_device_scratch_section_bytes,
        )
        self.assertEqual(
            context.activation_buffer.host_scratch_size,
            context.memory_plan.vmm_host_section_bytes,
        )
        regions = get_active_regions()
        self.assertEqual(len(regions), 1)
        region = next(iter(regions.values()))
        self.assertEqual(
            region.section_locations,
            [Location.DEVICE, Location.HOST, Location.DEVICE],
        )
        self.assertEqual(
            region.section_sizes,
            [
                context.memory_plan.vmm_device_prefix_bytes,
                context.memory_plan.vmm_host_section_bytes,
                context.memory_plan.vmm_device_scratch_section_bytes,
            ],
        )
        self.assertEqual(
            context.activation_buffer.activation_slot_bytes,
            context.memory_plan.activation_slot_bytes,
        )
        self.assertGreater(context.memory_plan.vmm_padding_bytes, 0)
        with self.assertRaisesRegex(RuntimeError, "already owns a live"):
            dist_moe.create_context(
                group=dist.group.WORLD,
                config=config,
                device=device,
            )

        context.close()
        self.assertEqual(get_active_regions(), {})

    @unittest.skipUnless(_is_blackwell(), "Blackwell GPU required")
    def test_blockscaled_execution_spills_into_host_scratch(self) -> None:
        """Require MXFP8 forward/backward to use mapped host overflow scratch.

        A positive device counter proves the execution reached host-backed VMM
        rather than merely constructing the virtual address range.
        """
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        # Keep the planner's 128-row device minimum while routing eight times
        # that many rows so execution must cross into the host-backed section.
        num_tokens = 1024
        config = _config(
            num_local_input_tokens=num_tokens,
            hidden_dim=128,
            intermediate_dim=8192,
            top_k=1,
            num_experts=1,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=0.125,
            vmm=dist_moe.VmmConfig(total_scratch_capacity_factor=1.0),
            block_scaled=dist_moe.BlockScaledConfig(),
        )
        context = dist_moe.create_context(
            group=dist.group.WORLD,
            config=config,
            device=device,
        )
        try:
            x = torch.randn(
                num_tokens,
                config.hidden_dim,
                dtype=torch.bfloat16,
                device=device,
                requires_grad=True,
            )
            scores = torch.ones(
                num_tokens,
                1,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            ids = torch.zeros(num_tokens, 1, dtype=torch.int64, device=device)
            w13 = torch.randn(
                1,
                2 * config.intermediate_dim,
                config.hidden_dim,
                dtype=torch.bfloat16,
                device=device,
                requires_grad=True,
            )
            w2 = torch.randn(
                1,
                config.hidden_dim,
                config.intermediate_dim,
                dtype=torch.bfloat16,
                device=device,
                requires_grad=True,
            )
            activation_buffer = context.activation_buffer
            assert activation_buffer is not None
            slot_state = tuple(
                tensor.clone()
                for tensor in (
                    activation_buffer.buffer_offsets,
                    activation_buffer.saved_activation_bytes_per_rank,
                    activation_buffer.moe_layer_id,
                )
            )
            with torch.no_grad():
                dist_moe.routed_experts(x, ids, scores, w13, w2, context)
            for actual, expected in zip(
                (
                    activation_buffer.buffer_offsets,
                    activation_buffer.saved_activation_bytes_per_rank,
                    activation_buffer.moe_layer_id,
                ),
                slot_state,
                strict=True,
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            forward_only_overflow = activation_buffer.check_overflow()
            self.assertIsNotNone(forward_only_overflow.host_scratch_used)
            self.assertGreater(forward_only_overflow.host_scratch_used, 0)

            dist_moe.routed_experts(x, ids, scores, w13, w2, context).sum().backward()
            overflow = activation_buffer.check_overflow()
            self.assertIsNotNone(overflow.host_scratch_used)
            self.assertGreater(overflow.host_scratch_used, 0)
        finally:
            context.close()

    @unittest.skipUnless(_is_blackwell(), "Blackwell GPU required")
    def test_nvfp4_inference_spills_into_host_scratch(self) -> None:
        """Require prepared NVFP4 inference to use mapped host overflow scratch.

        A positive device counter proves inference exercised the host-backed
        section rather than only validating its allocation metadata.
        """
        dist.init_process_group(
            backend="fake",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        num_tokens = 256
        policy = dist_moe.BlockScaledConfig(
            format=dist_moe.BlockScaledFormat.NVFP4,
        )
        config = _config(
            num_local_input_tokens=num_tokens,
            hidden_dim=256,
            intermediate_dim=8192,
            top_k=1,
            num_experts=1,
            max_moe_layers_per_activation_slot=1,
            num_activation_slots=0,
            device_scratch_capacity_factor=0.5,
            vmm=dist_moe.VmmConfig(total_scratch_capacity_factor=1.0),
            block_scaled=policy,
            inference=True,
        )
        context = dist_moe.create_context(
            group=dist.group.WORLD,
            config=config,
            device=device,
        )
        try:
            x_TD = torch.randn(
                num_tokens,
                config.hidden_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            topk_expert_ids_TK = torch.zeros(
                num_tokens,
                1,
                dtype=torch.int64,
                device=device,
            )
            topk_scores_TK = torch.ones(
                num_tokens,
                1,
                dtype=torch.float32,
                device=device,
            )
            w13_EFD = torch.randn(
                1,
                2 * config.intermediate_dim,
                config.hidden_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            w2_EDF = torch.randn(
                1,
                config.hidden_dim,
                config.intermediate_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            prepared_w13 = dist_moe.prepare_block_scaled_weight(
                w13_EFD,
                policy,
                inference=True,
            )
            prepared_w2 = dist_moe.prepare_block_scaled_weight(
                w2_EDF,
                policy,
                inference=True,
            )
            with torch.no_grad():
                dist_moe.routed_experts(
                    x_TD,
                    topk_expert_ids_TK,
                    topk_scores_TK,
                    prepared_w13,
                    prepared_w2,
                    context,
                )
            overflow = context.activation_buffer.check_overflow()
            self.assertIsNotNone(overflow.host_scratch_used)
            self.assertGreater(overflow.host_scratch_used, 0)
        finally:
            context.close()
