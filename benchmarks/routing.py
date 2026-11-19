# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Exact routed-expert quotas with finite-Zipf or per-layer rank loads."""

from __future__ import annotations

import bisect
import math
import random

from .config import Case, PACKING_PROFILES


def apportion(total: int, weights: list[float]) -> list[int]:
    """Allocate integer assignments by largest remainder, preserving their sum."""
    if total < 0 or not weights or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("total and weights must be finite and nonnegative")
    scale = sum(weights)
    if scale == 0:
        if total:
            raise ValueError("positive total requires positive weight")
        return [0] * len(weights)
    values = [total * w / scale for w in weights]
    result = [math.floor(v) for v in values]
    order = sorted(
        range(len(weights)), key=lambda i: (values[i] - result[i], -i), reverse=True
    )
    for i in order[: total - sum(result)]:
        result[i] += 1
    return result


def rank_quotas(case: Case, layer: int = 0) -> tuple[list[int], float | None]:
    """Return exact per-rank routes and the calibrated finite-Zipf exponent.

    Packing profiles describe rank zero, including underloaded layers. Other
    ranks receive the remaining routes equally. Zipf targets instead describe
    maximum/mean rank load and calibrate the finite distribution directly.
    """
    if not 0 <= layer < case.layers:
        raise ValueError("layer index is outside the case")
    balanced = case.shape.tokens * case.shape.top_k
    total = case.ep * balanced
    if case.profile is not None:
        hot = round(PACKING_PROFILES[case.profile][layer] * balanced)
        return [hot, *apportion(total - hot, [1.0] * (case.ep - 1))], None
    if case.imbalance == 1:
        return [balanced] * case.ep, 0.0
    hot = round(case.imbalance * balanced)
    if hot == total:
        return [total, *([0] * (case.ep - 1))], None
    low, high = 0.0, 64.0
    for _ in range(80):
        exponent = (low + high) / 2
        weights = [(i + 1) ** (-exponent) for i in range(case.ep)]
        if case.ep / sum(weights) < case.imbalance:
            low = exponent
        else:
            high = exponent
    exponent = (low + high) / 2
    others = [(i + 1) ** (-exponent) for i in range(1, case.ep)]
    return [hot, *apportion(total - hot, others)], exponent


def expert_quotas(case: Case, layer: int = 0) -> list[int]:
    """Balance integer expert counts within each rank without changing rank load."""
    ranks, _ = rank_quotas(case, layer)
    local_experts = case.shape.experts // case.ep
    result = []
    for count in ranks:
        result.extend(apportion(count, [1.0] * local_experts))
    tokens = case.ep * case.shape.tokens
    if max(result) > tokens:
        raise ValueError("expert quota cannot be realized with distinct token experts")
    return result


def local_routes(case: Case, rank: int, layer: int = 0) -> list[list[int]]:
    """Construct only one rank's routes with exact global expert quotas.

    1. Expert runs partition K columns of global token indices. Each run has
       length at most the global token count, so it cannot repeat an expert
       for the same token, even across a column boundary.
    2. A shared affine token permutation mixes source ranks without changing
       those quotas. Inverting it allows local construction without a global
       token-by-expert matrix or communication.
    """
    if not 0 <= rank < case.ep:
        raise ValueError("rank is outside the expert-parallel group")
    counts = expert_quotas(case, layer)
    rng = random.Random(case.seed + layer)
    experts = list(range(case.shape.experts))
    rng.shuffle(experts)
    ends = []
    total = 0
    for expert in experts:
        total += counts[expert]
        ends.append(total)
    tokens = case.ep * case.shape.tokens
    multiplier = rng.randrange(1, tokens) if tokens > 1 else 1
    while math.gcd(multiplier, tokens) != 1:
        multiplier = multiplier % tokens + 1
    offset = rng.randrange(tokens)
    inverse = pow(multiplier, -1, tokens) if tokens > 1 else 1
    result = []
    for local_token in range(case.shape.tokens):
        token = rank * case.shape.tokens + local_token
        original = (inverse * (token - offset)) % tokens
        row = [
            experts[bisect.bisect_right(ends, original + k * tokens)]
            for k in range(case.shape.top_k)
        ]
        if len(set(row)) != case.shape.top_k:
            raise RuntimeError(f"duplicate expert for global token {token}: {row}")
        result.append(row)
    return result


def routing_summary(case: Case, rank: int, layer: int = 0) -> dict:
    """Report generated local traffic and the exact predicted global counts."""
    counts, exponent = rank_quotas(case, layer)
    routes = local_routes(case, rank, layer)
    local_experts = case.shape.experts // case.ep
    off_rank = sum(expert // local_experts != rank for row in routes for expert in row)
    balanced = case.shape.tokens * case.shape.top_k
    return {
        "rank_counts": counts,
        "expert_counts": expert_quotas(case, layer),
        "observed_max_mean": max(counts) / balanced,
        "tracked_rank_factor": counts[0] / balanced,
        "zipf_exponent": exponent,
        "off_rank_routes": off_rank,
        "bf16_equivalent_forward_payload_bytes": off_rank * case.shape.hidden * 2 * 2,
    }


def padded_rank_rows(case: Case, *, alignment: int) -> list[list[int]]:
    """Return each layer's receive rows after independently padding experts."""
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    local = case.shape.experts // case.ep
    result = []
    for layer in range(case.layers):
        counts = expert_quotas(case, layer)
        padded = [math.ceil(n / alignment) * alignment for n in counts]
        result.append(
            [sum(padded[r * local : (r + 1) * local]) for r in range(case.ep)]
        )
    return result


def mok_memory_settings(case: Case) -> dict:
    """Map the saved-state treatment to native MoK rows without kernel changes.

    Expert scheduler rows are padded to256. The native4096-row minibatch is
    retained; packing rounds factor2.5 up to a whole minibatch. Other cases
    preserve the published131072-row macrobatch.
    """
    balanced = case.shape.tokens * case.shape.top_k
    requested = math.ceil(2.5 * balanced) if case.profile else 131072
    macrobatch = math.ceil(requested / 4096) * 4096
    rows = padded_rank_rows(case, alignment=256)
    factor = max(2, math.ceil(case.ep * 0.5), math.ceil(max(map(max, rows)) / balanced))
    return {
        "minibatch_size": 4096,
        "requested_saved_rows": requested,
        "macrobatch_size": macrobatch,
        "effective_activation_factor": macrobatch / balanced,
        "schedule_capacity_multiplier": factor / case.ep,
        "schedule_capacity_rows": factor * balanced,
        "padded_receive_rows": rows,
    }
