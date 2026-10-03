"""Run: torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 examples/block_scaled_inference.py."""

import dist_moe
import torch
from _common import close, create_config, initialize_inputs

inputs = initialize_inputs(requires_grad=False)
context = None
try:
    for block_format in (
        dist_moe.BlockScaledFormat.MXFP8_E4M3,
        dist_moe.BlockScaledFormat.NVFP4,
    ):
        policy = dist_moe.BlockScaledConfig(format=block_format, pipeline="staged")
        config = create_config(inputs, inference=True, block_scaled=policy)
        context = dist_moe.create_context(
            group=inputs.group,
            config=config,
            device=inputs.device,
        )
        prepared_w13 = dist_moe.prepare_block_scaled_weight(
            inputs.w13_EFD,
            policy,
            inference=True,
        )
        prepared_w2 = dist_moe.prepare_block_scaled_weight(
            inputs.w2_EDF,
            policy,
            inference=True,
        )
        with torch.no_grad():
            dist_moe.routed_experts(
                inputs.x_TD,
                inputs.topk_expert_ids_TK,
                inputs.topk_scores_TK,
                prepared_w13,
                prepared_w2,
                context,
            )
        torch.cuda.synchronize()
        completed_context = context
        context = None
        close(completed_context, destroy_group=False)
finally:
    close(context)
