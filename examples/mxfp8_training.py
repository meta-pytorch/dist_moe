"""Run: torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 examples/mxfp8_training.py."""

import dist_moe
import torch
from _common import close, create_config, initialize_inputs

inputs = initialize_inputs(requires_grad=True)
context = None
try:
    block_scaled = dist_moe.BlockScaledConfig(
        format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
        pipeline="staged",
    )
    config = create_config(inputs, block_scaled=block_scaled)
    context = dist_moe.create_context(
        group=inputs.group,
        config=config,
        device=inputs.device,
    )
    output_TD = dist_moe.routed_experts(
        inputs.x_TD,
        inputs.topk_expert_ids_TK,
        inputs.topk_scores_TK,
        inputs.w13_EFD,
        inputs.w2_EDF,
        context,
    )
    output_TD.float().sum().backward()
    with torch.no_grad():
        dist_moe.routed_experts(
            inputs.x_TD,
            inputs.topk_expert_ids_TK,
            inputs.topk_scores_TK,
            inputs.w13_EFD,
            inputs.w2_EDF,
            context,
        )
    torch.cuda.synchronize()
finally:
    close(context)
