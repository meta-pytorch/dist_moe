# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the matched activation benchmark's policy."""

import unittest

import dist_moe
import torch
from benchmarks.swiglu import (
    ACTIVATIONS,
    GRADIENT_INPUTS,
    make_config,
    NVFP4_MEGA,
    NVFP4_PIPELINES,
    NVFP4_STAGED,
    parse_args,
    PIPELINES,
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
        for pipeline in PIPELINES:
            args = parse_args(["--pipeline", pipeline])
            activations = (
                ACTIVATIONS if args.activation == "all" else (args.activation,)
            )
            for activation in activations:
                with self.subTest(pipeline=pipeline, activation=activation):
                    config = make_config(args, activation)
                    self.assertEqual(config.activation_slot_capacity_factor, 1.0)
                    self.assertEqual(config.device_scratch_capacity_factor, 1.0)
                    self.assertEqual(config.num_activation_slots, 1)
                    self.assertIs(config.wgrad_dtype, torch.bfloat16)
                    self.assertIsNone(config.vmm)
                    self.assertEqual(config.situ_glu_beta, 4.0)
                    self.assertEqual(config.situ_glu_linear_beta, 25.0)

    def test_pipeline_selects_bf16_or_nvfp4_config(self) -> None:
        """Map each CLI pipeline to the intended public compute policy."""
        for pipeline in PIPELINES:
            with self.subTest(pipeline=pipeline):
                args = parse_args(["--pipeline", pipeline])
                config = make_config(args, "swiglu")
                if pipeline == "bf16":
                    self.assertIsNone(config.block_scaled)
                else:
                    assert config.block_scaled is not None
                    self.assertIs(
                        config.block_scaled.format,
                        dist_moe.BlockScaledFormat.NVFP4,
                    )
                    self.assertEqual(
                        config.block_scaled.pipeline,
                        pipeline.removeprefix("nvfp4-"),
                    )

    def test_pipelines_share_the_default_shape(self) -> None:
        """Keep BF16 and NVFP4 benchmark geometry directly comparable."""
        bf16 = parse_args([])
        self.assertEqual(
            (bf16.hidden, bf16.intermediate, bf16.experts), (3584, 3072, 8)
        )
        for pipeline in NVFP4_PIPELINES:
            nvfp4 = parse_args(["--pipeline", pipeline])
            self.assertEqual(
                (nvfp4.hidden, nvfp4.intermediate, nvfp4.experts),
                (3584, 3072, 8),
            )

    def test_activation_defaults_match_pipeline_lifecycles(self) -> None:
        """Run BF16's matrix together and isolate each NVFP4 context."""
        self.assertEqual(parse_args([]).activation, "all")
        args = parse_args(
            [
                "--pipeline",
                NVFP4_STAGED,
                "--activation",
                "swiglu_clamped",
            ]
        )
        for pipeline in NVFP4_PIPELINES:
            self.assertEqual(
                parse_args(["--pipeline", pipeline]).activation,
                "swiglu",
            )
        self.assertEqual(args.activation, "swiglu_clamped")

    def test_rejects_explicit_nvfp4_situ_glu(self) -> None:
        """Reject a BF16-only activation before allocating distributed state."""
        for pipeline in NVFP4_PIPELINES:
            with self.subTest(pipeline=pipeline), self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--pipeline",
                        pipeline,
                        "--activation",
                        "situ_glu",
                    ]
                )

    def test_rejects_multiple_nvfp4_contexts_in_one_process(self) -> None:
        """Require isolated NVFP4 contexts before persistent kernels launch."""
        for pipeline in (NVFP4_STAGED, NVFP4_MEGA):
            with self.subTest(pipeline=pipeline), self.assertRaises(SystemExit):
                parse_args(["--pipeline", pipeline, "--activation", "all"])

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
