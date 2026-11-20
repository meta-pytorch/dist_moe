# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Physical storage bounds for the source-selected Mega inference layouts."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from .formats import _kernel_block_scaled_format
from .kernels.blockscaled_config import auto_blockscaled_config
from .kernels.blockscaled_grouped_gemm import MXFP8_E4M3, NVFP4
from .kernels.chunked_mega_blockscaled_grouped_gemm import _chunked_mega_operand_layout
from .kernels.inference_config import _blockscaled_inference_config

if TYPE_CHECKING:
    from .api import Config


def _inference_operand_bytes(
    config: Config, *, ep_size: int, capacity_rows: int, num_sms: int
) -> int:
    """Bound all source-selected layouts for the context's permitted local T.

    Args:
        config: Inference-only Mega configuration with fixed weight dimensions.
        ep_size: Expert-parallel rank count.
        capacity_rows: Resolved context receive bound, including row padding.
        num_sms: Actual launch SM count, needed for descriptors and tile policy.

    Returns:
        Maximum physical byte requirement of its admitted inference layouts.

    Notes:
        Decode policy depends on ceil(T * top_k / local_experts), bounded by
        256. Larger hints use the source's nonpaged 128/256-row layouts, whose
        storage is monotone in receive capacity. No CUDA allocation or routing
        data read occurs here. This is a storage envelope, not tile tuning.
    """
    policy = config.block_scaled
    assert config.inference and policy is not None and policy.pipeline == "mega"
    format = {
        "mxfp8_e4m3": MXFP8_E4M3,
        "nvfp4": NVFP4,
    }[_kernel_block_scaled_format(policy.format).value]
    groups = config.num_experts // ep_size
    storage_rows = (capacity_rows + 127) // 128 * 128
    if policy.weight_only:
        from .kernels.w4a16_mega import w4a16_mega_operand_bytes

        return w4a16_mega_operand_bytes(
            num_rows=storage_rows,
            num_experts=groups,
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
        )
    explicit = (
        None
        if policy.kernel_config is None
        else policy.kernel_config._as_kernel_kwargs()
    )
    largest_hint = (
        config.max_num_local_input_tokens * config.top_k + groups - 1
    ) // groups
    # One representative of each attainable hint is sufficient for Mega:
    # unlike staged decode, its policy has no independent raw-T threshold.
    token_counts = {
        (hint - 1) * groups // config.top_k + 1
        for hint in range(1, min(largest_hint, 256) + 1)
        if (((hint - 1) * groups // config.top_k + 1) * config.top_k + groups - 1)
        // groups
        == hint
    }
    token_counts.add(config.max_num_local_input_tokens)
    layouts: dict[tuple[int, bool, bool], int] = {}
    for tokens, interleaved_fc13 in product(sorted(token_counts), (False, True)):
        selected = _blockscaled_inference_config(
            format=format,
            num_tokens=tokens,
            topk=config.top_k,
            num_local_experts=groups,
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            num_sms=num_sms,
            routing_capacity=capacity_rows,
            mega=True,
            interleaved_fc13=interleaved_fc13,
            config=explicit,
        )
        if selected is None:
            selected = auto_blockscaled_config(
                GM=capacity_rows,
                G=groups,
                N=2 * config.intermediate_dim,
                K=config.hidden_dim,
                num_sms=num_sms,
                format=format,
            )
        key = (
            int(selected["BLOCK_SIZE_N"]),
            bool(selected.get("COMPACT_DECODE_ROWS", False)),
            interleaved_fc13,
        )
        if key not in layouts:
            layouts[key] = _chunked_mega_operand_layout(
                rows=storage_rows,
                groups=groups,
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                format=format,
                config=selected,
                num_sms=num_sms,
                interleaved_fc13=interleaved_fc13,
            ).required_bytes
    return max(layouts.values())
