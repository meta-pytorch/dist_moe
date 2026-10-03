"""Run: python examples/memory_planning.py."""

import dist_moe

config = dist_moe.Config(
    num_local_input_tokens=4,
    hidden_dim=128,
    intermediate_dim=128,
    top_k=2,
    num_experts=4,
    max_moe_layers_per_activation_slot=2,
    activation_slot_capacity_factor=1.0,
    device_scratch_capacity_factor=1.0,
    num_activation_slots=1,
    vmm=dist_moe.VmmConfig(total_scratch_capacity_factor=2.0),
)
print(dist_moe.plan_memory(config, ep_size=2).explain())
