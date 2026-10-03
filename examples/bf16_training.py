"""Run: torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 examples/bf16_training.py."""

import dist_moe
import torch
from _common import close, create_config, initialize_inputs

inputs = initialize_inputs(requires_grad=True)
context = None
try:
    config = create_config(
        inputs,
        activation="swiglu_clamped",
        max_moe_layers_per_activation_slot=2,
        num_activation_slots=2,
    )
    context = dist_moe.create_context(
        group=inputs.group,
        config=config,
        device=inputs.device,
    )
    context.select_activation_slot(activation_slot=1, num_moe_layers_in_slot=1)
    clip_stats_out_3 = torch.empty(3, dtype=torch.float32, device=inputs.device)
    options = dist_moe.ExecutionOptions(
        experts_output_postprocess=dist_moe.RMSNormPostprocess(
            eps=1e-6,
            norm_output_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            require_bitwise=False,
            recompute_rstd=True,
        ),
        inplace_wgrad_accum=True,
        swiglu_clip_stats_out_3=clip_stats_out_3,
        swiglu_clip_limit=7.0,
    )
    for _ in range(2):
        output_TD = dist_moe.routed_experts(
            inputs.x_TD,
            inputs.topk_expert_ids_TK,
            inputs.topk_scores_TK,
            inputs.w13_EFD,
            inputs.w2_EDF,
            context,
            options=options,
        )
        output_TD.float().sum().backward()
    torch.cuda.synchronize()
finally:
    close(context)
