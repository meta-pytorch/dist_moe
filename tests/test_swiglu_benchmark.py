# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the matched activation benchmark's policy."""

import unittest

import torch
from benchmarks.swiglu import (
    ACTIVATIONS,
    GRADIENT_INPUTS,
    make_config,
    parse_args,
    snapshot_gradients,
)


class SwigluBenchmarkTest(unittest.TestCase):
    """Keep timing windows and activation comparisons reproducible."""

    def test_rejects_empty_windows(self) -> None:
        """Do not report throughput from zero work or an uncompiled first step."""
        for field in ("warmup", "windows", "steps-per-window"):
            with self.subTest(field=field), self.assertRaises(SystemExit):
                parse_args([f"--{field}", "0"])

    def test_activations_share_memory_and_gradient_policy(self) -> None:
        """Only activation arithmetic changes within one matched comparison."""
        args = parse_args([])
        for activation in ACTIVATIONS:
            with self.subTest(activation=activation):
                config = make_config(args, activation)
                self.assertEqual(config.activation_slot_capacity_factor, 1.0)
                self.assertEqual(config.device_scratch_capacity_factor, 1.0)
                self.assertEqual(config.num_activation_slots, 1)
                self.assertEqual(config.wgrad_dtype, torch.bfloat16)
                self.assertIsNone(config.vmm)
                self.assertEqual(config.situ_glu_gate_beta, 4.0)
                self.assertEqual(config.situ_glu_up_beta, 25.0)

    def test_gradient_snapshots_are_independent(self) -> None:
        """CPU snapshots preserve exact values without aliasing live gradients."""
        inputs = tuple(torch.zeros(2, requires_grad=True) for _ in range(6))
        for index, _ in GRADIENT_INPUTS:
            inputs[index].grad = torch.full_like(inputs[index], float(index + 1))
        snapshots = snapshot_gradients(inputs)
        for (index, _), snapshot in zip(GRADIENT_INPUTS, snapshots, strict=True):
            torch.testing.assert_close(snapshot, inputs[index].grad, rtol=0, atol=0)
            inputs[index].grad.zero_()
            torch.testing.assert_close(
                snapshot, torch.full_like(snapshot, float(index + 1)), rtol=0, atol=0
            )

    def test_missing_gradients_identify_the_required_input(self) -> None:
        """Every missing input, score, or weight gradient gets a clear error."""
        for missing_index, name in GRADIENT_INPUTS:
            with self.subTest(input=name):
                inputs = tuple(torch.zeros(2, requires_grad=True) for _ in range(6))
                for index, _ in GRADIENT_INPUTS:
                    if index != missing_index:
                        inputs[index].grad = torch.ones_like(inputs[index])
                with self.assertRaisesRegex(
                    RuntimeError, rf"{name}\.grad \(input {missing_index}\)"
                ):
                    snapshot_gradients(inputs)

    def test_fast_math_is_explicit_and_mxfp8_only(self) -> None:
        """Do not silently change BF16 precision to match a fused backend."""
        with self.assertRaises(SystemExit):
            parse_args(["--fast-math", "--pipeline", "bf16"])
        for pipeline in ("staged", "mega"):
            args = parse_args(["--pipeline", pipeline, "--fast-math"])
            for activation in ACTIVATIONS:
                config = make_config(args, activation)
                self.assertEqual(config.block_scaled.pipeline, pipeline)
                self.assertTrue(config.block_scaled.fast_math)
