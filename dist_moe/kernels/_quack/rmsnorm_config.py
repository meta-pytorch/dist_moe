# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# Licensed under the Apache License, Version 2.0.
# Modified by Meta Platforms, Inc.

"""Launch configuration required by the vendored RMSNorm forward kernel."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class RmsNormFwdConfig:
    num_threads: int
    threads_per_row: int
    cluster_n: int
    # None = compute once into registers; "smem" / "gmem" = reload x (and
    # residual) before the post-reduction epilogue.
    reload_from: Optional[str]
    # Defer the weight/bias load until after the row reduction.
    delay_w_load: bool = False

    @classmethod
    def from_analytical_heuristic(
        cls,
        N: int,
        dtype_width: int,
        arch_major: Optional[int] = None,
        is_layernorm: bool = False,
    ) -> "RmsNormFwdConfig":
        """Pick a launch config from the hand-tuned analytical heuristic.

        ``arch_major`` defaults to the current device's capability. The same
        ladder is used for Hopper, Blackwell, and SM12x today; a future
        ``_for_blackwell_fwd`` factory can be added and dispatched on
        ``arch_major >= 10``. For autotuning, use :func:`get_all_fwd_configs`.
        """
        if arch_major is None:
            arch_major = _detect_arch_major()
        return _for_hopper_fwd(N, dtype_width, arch_major, is_layernorm)


def _for_hopper_fwd(
    N: int, dtype_width: int, arch_major: int, is_layernorm: bool
) -> RmsNormFwdConfig:
    num_threads = 128 if N <= 16 * 1024 else 256

    threads_per_row = 256
    for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
        if N <= limit:
            threads_per_row = threads
            break

    if arch_major < 9:
        cluster_n = 1
    else:
        max_cluster = 8 if arch_major == 12 else 16
        # cluster_n=4 is faster than cluster_n=2 for N=64k; cluster_n=8 is
        # faster for N=128k.
        if arch_major == 12 and dtype_width >= 32:
            # SM12x 99 KB SMEM: fp32 needs tighter clustering (conservative for residual case)
            thresholds = [(8 * 1024, 1), (16 * 1024, 2), (32 * 1024, 4), (64 * 1024, 8)]
        elif dtype_width == 16:
            thresholds = [
                (16 * 1024, 1),
                (32 * 1024, 2),
                (64 * 1024, 4),
                (128 * 1024, 8),
            ]
        elif is_layernorm:
            # fp32 layernorm: bump cluster earlier than fp16/bf16. The 2-pass path's
            # single-CTA tile is bandwidth-limited at N=16k/32k; cluster_n=2 splits
            # the row across two CTAs and recovers ~3-14% at those sizes.
            thresholds = [
                (8 * 1024, 1),
                (64 * 1024, 2),
                (128 * 1024, 4),
                (256 * 1024, 8),
            ]
        else:
            # fp32 rmsnorm (1-pass) is already saturated at cluster_n=1 for N<=32k;
            # bumping to cluster_n=2 there regresses ~3%.
            thresholds = [
                (32 * 1024, 1),
                (64 * 1024, 2),
                (128 * 1024, 4),
                (256 * 1024, 8),
            ]
        cluster_n = max_cluster
        for limit, cluster in thresholds:
            if N <= limit:
                cluster_n = cluster
                break

    reload_threshold = 16 * 1024 if is_layernorm else 8 * 1024
    return RmsNormFwdConfig(
        num_threads=num_threads,
        threads_per_row=threads_per_row,
        cluster_n=cluster_n,
        reload_from=None if N <= reload_threshold else "smem",
        delay_w_load=False,
    )


def _detect_arch_major() -> int:
    """Return the major device capability of the current CUDA device.

    Honors the ``QUACK_ARCH`` override (via ``get_device_capacity``) so
    GPU-blind processes — compile-pool workers, CPU-only boxes — never
    initialize CUDA here. This function runs at ``import quack`` time (the
    module-level ``get_all_fwd_configs()`` call in ``rmsnorm.py``), so a raw
    ``torch.cuda.current_device()`` would create a CUDA context on import,
    which both slows imports and poisons forked children (torch's
    "Cannot re-initialize CUDA in forked subprocess" guard).

    Falls back to 0 (no-cluster, no-TMA) when CUDA is unavailable so the
    autotune search space stays well-defined for CPU-only imports.
    """
    import os

    if os.environ.get("QUACK_ARCH") is None and not torch.cuda.is_available():
        return 0
    from .cute_dsl_utils import get_device_capacity

    return get_device_capacity()[0]
