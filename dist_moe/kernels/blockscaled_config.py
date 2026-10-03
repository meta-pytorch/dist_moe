# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Format-aware block-scaled DistMoE configuration layer.

Operand format specs, the literal production config tables, the decode /
training / auto config pickers, and the scheduler policy - everything that
maps (format, shape, regime) to a launch config, split out of the
``blockscaled_grouped_gemm`` launch module. ``config.py`` stays the
cutlass-free geometry layer; this module owns the config policy that needs
``BlockScaledFormatSpec`` (and therefore cutlass).
"""

import cutlass
import torch

from . import sm103_blockscaled_helpers as sm103
from .blockscaled_grouped_gemm_kernel import (
    _format_has_fp4_b,
    BlockScaledProductionConfig,
    MXFP8_E4M3,
)
from .config import (
    _FPROP,
    _WGRAD,
    BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT,
    DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    derive_blockscaled_grouped_gemm_config,
    derive_blockscaled_inference_geometry,
    EPILOGUE_SUBTILE_AUTO,
    initialize_blockscaled_inference_pipeline,
    STATIC_FPROP_SCHEDULER_SHAPES,
)
from .params import (
    _torch_dtype,
    BlockScaledFormatSpec,
)

# ---------------------------------------------------------------------------
# Block-scaled operand format specs.
# ---------------------------------------------------------------------------


_MIXED_FORMAT_CACHE: dict[tuple[str, str], BlockScaledFormatSpec] = {}


MXFP8_E5M2 = BlockScaledFormatSpec(
    name="mxfp8_e5m2",
    a_dtype=cutlass.Float8E5M2,
    b_dtype=cutlass.Float8E5M2,
    sf_dtype=cutlass.Float8E8M0FNU,
    sf_vec_size=32,
)
NVFP4 = BlockScaledFormatSpec(
    name="nvfp4",
    a_dtype=cutlass.Float4E2M1FN,
    b_dtype=cutlass.Float4E2M1FN,
    sf_dtype=cutlass.Float8E4M3FN,
    sf_vec_size=16,
)
MXFP4 = BlockScaledFormatSpec(
    name="mxfp4",
    a_dtype=cutlass.Float4E2M1FN,
    b_dtype=cutlass.Float4E2M1FN,
    sf_dtype=cutlass.Float8E8M0FNU,
    sf_vec_size=32,
)

_BLOCKSCALED_FORMAT_SPECS = {
    format.name: format for format in (MXFP8_E4M3, MXFP8_E5M2, NVFP4, MXFP4)
}


def _as_blockscaled_format_spec(
    format: BlockScaledFormatSpec | str,
) -> BlockScaledFormatSpec:
    if isinstance(format, BlockScaledFormatSpec):
        return format
    try:
        return _BLOCKSCALED_FORMAT_SPECS[str(format)]
    except KeyError as error:
        raise ValueError(f"unsupported block-scaled format: {format}") from error


def _format_has_fp4_a(format: BlockScaledFormatSpec) -> bool:
    return format.a_dtype == cutlass.Float4E2M1FN


def _format_uses_fp4(format: BlockScaledFormatSpec) -> bool:
    return _format_has_fp4_a(format) or _format_has_fp4_b(format)


def _resolve_weight_format(
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None,
) -> BlockScaledFormatSpec:
    return format if weight_format is None else weight_format


def make_mixed_blockscaled_format(
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
) -> BlockScaledFormatSpec:
    """Return a transient kernel spec with activation A and weight B dtypes."""
    weight_format = _resolve_weight_format(format, weight_format)
    if format is weight_format:
        return format
    if (
        format.sf_dtype != weight_format.sf_dtype
        or format.sf_vec_size != weight_format.sf_vec_size
    ):
        raise ValueError(
            "mixed block-scaled GEMM requires matching scale dtype and vec size; "
            f"got {format.name} scales and {weight_format.name} scales"
        )
    key = (format.name, weight_format.name)
    mixed = _MIXED_FORMAT_CACHE.get(key)
    if mixed is None:
        mixed = BlockScaledFormatSpec(
            name=f"{format.name};weight_dtype:{weight_format.name}",
            a_dtype=format.a_dtype,
            b_dtype=weight_format.b_dtype,
            sf_dtype=format.sf_dtype,
            sf_vec_size=format.sf_vec_size,
        )
        _MIXED_FORMAT_CACHE[key] = mixed
    return mixed


def _cutlass_dtype_from_torch(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float32:
        return cutlass.Float32
    raise NotImplementedError(f"Unsupported output dtype {dtype}")


# ---------------------------------------------------------------------------
# Config universe + auto-picker.
# ---------------------------------------------------------------------------
# ``build_blockscaled_config`` is kept for ad hoc sweeps; the auto-picker
# below is intentionally narrower and only returns one of the literal
# production configs.


def build_blockscaled_config(
    *,
    num_ctas: int,
    block_n: int,
    swap_ab: bool = False,
    overlapping_accum: bool = False,
    epilogue_subtile: int | None = None,
    block_m: int | None = None,
    num_mmas: int = 1,
    kloop_unroll: int = 2,
    static_scheduler: bool = False,
    format: BlockScaledFormatSpec | str = MXFP8_E4M3,
) -> dict:
    """Build the FINAL block-scaled grouped-GEMM cfg dict — the kernel
    JIT consumes this directly, no further patching at call time.

    Format-driven invariants: FP8 uses BLOCK_K=128 + 1-byte operands;
    FP4 uses BLOCK_K=256 + packed FP4 (BN=256 dispatched as two 128-wide
    MMA-N atoms). ``epilogue_subtile=None`` falls back to
    ``sm100_utils.compute_epilogue_tile_shape``.

    Constraints: ``block_m // num_mmas`` in {64, 128} (1cta) or
    {128, 256} (2cta); ``num_mmas`` is fixed at 1. ``overlapping_accum``
    is rejected for NVFP4 + BN=128 (the seam collides).
    """
    if num_mmas != 1:
        raise NotImplementedError(
            "num_mmas > 1 is unsupported: the SFA TMA atom would need "
            "per-atom V-map alignment the cutlass MMA helper does not "
            "expose."
        )
    if block_m is None:
        block_m = 256 if num_ctas == 2 else 128
    format = _as_blockscaled_format_spec(format)
    fmt_name = format.name
    is_fp4 = _format_uses_fp4(format)
    is_nvfp4 = fmt_name == "nvfp4"
    if is_fp4:
        # NVFP4 + BN=128: num_sf == BN, so both OV acc stages collide.
        # BN=256 keeps a 256-wide acc tile (split into two MMA-N atoms),
        # so the seam is non-degenerate there.
        if is_nvfp4 and overlapping_accum and block_n <= 128:
            raise ValueError(
                "NVFP4 + BN=128 + overlapping_accum=True is degenerate "
                "(num_sf == BN → both stages collide on the same TMEM "
                "region). Pass overlapping_accum=False for NVFP4."
            )
        a_dtype = _torch_dtype(format.a_dtype)
        b_dtype = _torch_dtype(format.b_dtype)
        block_k = 256
        sf_vec_size = format.sf_vec_size
    else:
        a_dtype = _torch_dtype(format.a_dtype)
        b_dtype = _torch_dtype(format.b_dtype)
        block_k = 128
        sf_vec_size = format.sf_vec_size
    cfg = derive_blockscaled_grouped_gemm_config(
        num_ctas=num_ctas,
        num_mmas=num_mmas,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        sf_vec_size=sf_vec_size,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_stage_dtype=torch.bfloat16,
        epilogue_subtile=epilogue_subtile,
        overlapping_accum=overlapping_accum,
        swap_ab=swap_ab,
        kloop_unroll=kloop_unroll,
    )
    cfg["STATIC_SCHEDULER"] = static_scheduler
    if is_nvfp4:
        cfg = dict(cfg)
        if num_ctas == 2 and block_n >= 256 and overlapping_accum:
            # The derived epi=2 / smem=5 over-allocates by one 4 KiB
            # SMEM page on SM100a. Cap to 4 unless the caller asked for
            # epi=4 (which is the literal NVFP4 compute config below).
            smem_cap = 5 if cfg.get("EPILOGUE_SUBTILE") == 4 else 4
            cfg["NUM_SMEM_BUFFERS"] = min(cfg["NUM_SMEM_BUFFERS"], smem_cap)
        else:
            cfg["NUM_SMEM_BUFFERS"] = min(cfg["NUM_SMEM_BUFFERS"], 6)
    return cfg


_MXFP8_FORMAT_NAMES = frozenset({"mxfp8_e4m3", "mxfp8_e5m2"})
_FP4_FORMAT_NAMES = frozenset({"nvfp4", "mxfp4"})
# These T6 prefill contractions are measured overrides of the generic
# compute-tier config; adjacent and unmeasured shapes remain on that config.
_NVFP4_T6_PREFILL_CONTRACTIONS = frozenset(
    {
        (6144, 12288),
        (12288, 3072),
    }
)
_MXFP4_T6_FC2_PREFILL_CONTRACTION = (12288, 3072)

# Literal production configs (one per format-family x regime) so the
# auto-picker hot path doesn't depend on a runtime tuning search.
#
# Compute configs default to the dynamic scheduler; memory and fallback configs
# retain their independently tuned static defaults.
_MXFP8_COMPUTE_CONFIG: dict[str, int | bool] = {
    "NUM_CTAS": 2,
    "NUM_MMAS": 1,
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 128,
    "NUM_SMEM_BUFFERS": 5,
    "NUM_C_STAGES": 3,
    "NUM_TMEM_BUFFERS": 2,
    "NUM_TILE_BUFFERS": 3,
    "EPILOGUE_SUBTILE": 2,
    "OVERLAPPING_ACCUM": True,
    "SWAP_AB": True,
    "KLOOP_UNROLL": 1,
    "NUM_WARPS": 8,
    "STATIC_SCHEDULER": False,
}
_MXFP8_MEMORY_CONFIG: dict[str, int | bool] = {
    "NUM_CTAS": 2,
    "NUM_MMAS": 1,
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "NUM_SMEM_BUFFERS": 7,
    "NUM_C_STAGES": 3,
    "NUM_TMEM_BUFFERS": 2,
    "NUM_TILE_BUFFERS": 3,
    "EPILOGUE_SUBTILE": 1,
    "OVERLAPPING_ACCUM": False,
    "SWAP_AB": True,
    "KLOOP_UNROLL": 2,
    "NUM_WARPS": 8,
    "STATIC_SCHEDULER": True,
}
_MXFP8_MEGA_DECODE_CONFIG: dict[str, int | bool] = {
    **_MXFP8_MEMORY_CONFIG,
    "KLOOP_UNROLL": 1,
    "MEGA_NUM_SMEM_BUFFERS": 8,
    "NUM_TILE_BUFFERS": 4,
    "NUM_TMEM_BUFFERS": 3,
    "PIPELINE_CHUNK_ROWS": 128,
    "PIPELINE_LEAD_CHUNKS": 1,
    "USE_DEVICE_TENSORMAPS": False,
}
_MXFP8_FALLBACK_CONFIG: dict[str, int | bool] = {
    "NUM_CTAS": 1,
    "NUM_MMAS": 1,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 128,
    "NUM_SMEM_BUFFERS": 3,
    "NUM_C_STAGES": 4,
    "NUM_TMEM_BUFFERS": 2,
    "NUM_TILE_BUFFERS": 3,
    "EPILOGUE_SUBTILE": 2,
    "OVERLAPPING_ACCUM": True,
    "SWAP_AB": False,
    "KLOOP_UNROLL": 1,
    "NUM_WARPS": 8,
    "STATIC_SCHEDULER": True,
}
_FP4_COMPUTE_CONFIG: dict[str, int | bool] = {
    "NUM_CTAS": 2,
    "NUM_MMAS": 1,
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 256,
    "NUM_SMEM_BUFFERS": 5,
    "NUM_C_STAGES": 2,
    "NUM_TMEM_BUFFERS": 2,
    "NUM_TILE_BUFFERS": 3,
    "EPILOGUE_SUBTILE": 2,
    "OVERLAPPING_ACCUM": True,
    "SWAP_AB": True,
    "KLOOP_UNROLL": 1,
    "NUM_WARPS": 8,
    "STATIC_SCHEDULER": False,
}
_SM103_NVFP4_COMPUTE_CONFIG: dict[str, int | bool] = {
    **_FP4_COMPUTE_CONFIG,
    "BLOCK_SIZE_K": sm103.SM103_TILE_K,
    "NUM_SMEM_BUFFERS": sm103.SM103_AB_PIPELINE_STAGES,
    "USE_SM103_ULTRA": True,
}
_MXFP4_T6_FC2_PREFILL_CONFIG: dict[str, int | bool] = {
    **_FP4_COMPUTE_CONFIG,
    "NUM_CTAS": 1,
    "BLOCK_SIZE_M": 128,
    "NUM_SMEM_BUFFERS": 4,
    "STATIC_SCHEDULER": True,
    "PREFETCH_FIRST_SCATTER_PTR": True,
}
_FP4_MEMORY_CONFIG: dict[str, int | bool] = {
    "NUM_CTAS": 2,
    "NUM_MMAS": 1,
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 256,
    "NUM_SMEM_BUFFERS": 6,
    "NUM_C_STAGES": 4,
    "NUM_TMEM_BUFFERS": 2,
    "NUM_TILE_BUFFERS": 3,
    "EPILOGUE_SUBTILE": 1,
    "OVERLAPPING_ACCUM": False,
    "SWAP_AB": True,
    "KLOOP_UNROLL": 2,
    "NUM_WARPS": 8,
    "STATIC_SCHEDULER": True,
}
_MXFP4_MEGA_DECODE_CONFIG: dict[str, int | bool] = {
    **_FP4_MEMORY_CONFIG,
    "KLOOP_UNROLL": 1,
    "MEGA_NUM_SMEM_BUFFERS": 11,
    "NUM_TILE_BUFFERS": 4,
    "NUM_TMEM_BUFFERS": 3,
    "PIPELINE_CHUNK_ROWS": 128,
    "PIPELINE_LEAD_CHUNKS": 1,
    "USE_DEVICE_TENSORMAPS": False,
}
_NVFP4_MEGA_DECODE_CONFIG: dict[str, int | bool] = {
    **_FP4_MEMORY_CONFIG,
    "KLOOP_UNROLL": 1,
    "MEGA_NUM_SMEM_BUFFERS": 7,
    "NUM_TILE_BUFFERS": 4,
    "NUM_TMEM_BUFFERS": 3,
    "PIPELINE_CHUNK_ROWS": 128,
    "PIPELINE_LEAD_CHUNKS": 1,
    "USE_DEVICE_TENSORMAPS": False,
}
_FP4_FALLBACK_CONFIG: dict[str, int | bool] = {
    # FP4 1cta fallback: BN=128 + no-OV. Avoids both the FP4 MMA atom-N
    # split and the 4.4.2 1cta+BN=256+OV cross-stage release bug.
    # NVFP4 + BN=128 forbids OV by spec (SF seam collides).
    # Followup: re-enable BN=256+OV once 4.4.2's cross-stage release
    # converges for this geometry.
    "NUM_CTAS": 1,
    "NUM_MMAS": 1,
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 256,
    "NUM_SMEM_BUFFERS": 5,
    "NUM_C_STAGES": 5,
    "NUM_TMEM_BUFFERS": 2,
    "NUM_TILE_BUFFERS": 3,
    "EPILOGUE_SUBTILE": 1,
    "OVERLAPPING_ACCUM": False,
    "SWAP_AB": False,
    "KLOOP_UNROLL": 2,
    "NUM_WARPS": 8,
    "STATIC_SCHEDULER": True,
}


PRODUCTION_BLOCKSCALED_CONFIGS: tuple[BlockScaledProductionConfig, ...] = (
    BlockScaledProductionConfig(
        label="fp8_compute_2cta_bm256_bn256_bk128_swap1_ov1_epi2_smem5_cst3_ku1_w8_static0",
        format_names=_MXFP8_FORMAT_NAMES,
        config=_MXFP8_COMPUTE_CONFIG,
    ),
    BlockScaledProductionConfig(
        label="fp8_fallback_1cta_bm128_bn256_bk128_swap0_ov1_epi2_smem3_cst4_ku1_w8_static1",
        format_names=_MXFP8_FORMAT_NAMES,
        config=_MXFP8_FALLBACK_CONFIG,
    ),
    BlockScaledProductionConfig(
        label="fp8_memory_2cta_bm256_bn128_bk128_swap1_ov0_epi1_smem7_cst3_ku2_w8_static1",
        format_names=_MXFP8_FORMAT_NAMES,
        config=_MXFP8_MEMORY_CONFIG,
    ),
    BlockScaledProductionConfig(
        label="fp4_compute_2cta_bm256_bn256_bk256_swap1_ov1_epi2_smem5_cst2_ku1_w8_static0",
        format_names=_FP4_FORMAT_NAMES,
        config=_FP4_COMPUTE_CONFIG,
    ),
    BlockScaledProductionConfig(
        label="fp4_sm103_compute_2cta_bm256_bn256_bk768_swap1_ov1_epi2_smem5_cst2_ku1_w8_static0",
        format_names=frozenset({"nvfp4", "mxfp4"}),
        config=_SM103_NVFP4_COMPUTE_CONFIG,
        required_compute_capability=sm103.SM103_COMPUTE_CAPABILITY,
        directions=frozenset({"fprop", "dgrad"}),
    ),
    BlockScaledProductionConfig(
        label="fp4_fallback_1cta_bm128_bn128_bk256_swap0_ov0_epi1_smem5_cst5_ku2_w8_static1",
        format_names=_FP4_FORMAT_NAMES,
        config=_FP4_FALLBACK_CONFIG,
    ),
    BlockScaledProductionConfig(
        label="fp4_memory_2cta_bm256_bn128_bk256_swap1_ov0_epi1_smem6_cst4_ku2_w8_static1",
        format_names=_FP4_FORMAT_NAMES,
        config=_FP4_MEMORY_CONFIG,
    ),
)


def _copy_config(cfg: dict[str, int | bool]) -> dict:
    return dict(cfg)


def _is_a8w4_format(
    dtype: BlockScaledFormatSpec,
    weight_dtype: BlockScaledFormatSpec | None,
) -> bool:
    weight_dtype = _resolve_weight_format(dtype, weight_dtype)
    return dtype is MXFP8_E4M3 and weight_dtype is MXFP4


def _require_supported_weight_dtype(
    dtype: BlockScaledFormatSpec,
    weight_dtype: BlockScaledFormatSpec | None,
) -> BlockScaledFormatSpec:
    weight_dtype = _resolve_weight_format(dtype, weight_dtype)
    _require_decode_dtype(dtype)
    _require_decode_dtype(weight_dtype)
    if dtype is weight_dtype:
        return weight_dtype
    if not _is_a8w4_format(dtype, weight_dtype):
        raise NotImplementedError(
            "mixed block-scaled decode currently supports only "
            "dtype:mxfp8_e4m3;weight_dtype:mxfp4"
        )
    make_mixed_blockscaled_format(dtype, weight_dtype)
    return weight_dtype


def _retune_config_for_formats(
    config: dict[str, int | bool],
    *,
    dtype: BlockScaledFormatSpec,
    weight_dtype: BlockScaledFormatSpec | None = None,
    c_stage_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, int | bool]:
    weight_dtype = _resolve_weight_format(dtype, weight_dtype)
    if dtype is weight_dtype:
        return config
    combined = make_mixed_blockscaled_format(dtype, weight_dtype)
    retuned = derive_blockscaled_grouped_gemm_config(
        num_ctas=int(config["NUM_CTAS"]),
        block_m=int(config["BLOCK_SIZE_M"]),
        block_n=int(config["BLOCK_SIZE_N"]),
        num_mmas=int(config["NUM_MMAS"]),
        block_k=int(config["BLOCK_SIZE_K"]),
        sf_vec_size=combined.sf_vec_size,
        a_dtype=_torch_dtype(combined.a_dtype),
        b_dtype=_torch_dtype(combined.b_dtype),
        c_stage_dtype=c_stage_dtype,
        epilogue_subtile=(
            None
            if int(config.get("EPILOGUE_SUBTILE", EPILOGUE_SUBTILE_AUTO))
            == EPILOGUE_SUBTILE_AUTO
            else int(config["EPILOGUE_SUBTILE"])
        ),
        overlapping_accum=bool(config.get("OVERLAPPING_ACCUM", False)),
        swap_ab=bool(config.get("SWAP_AB", False)),
        kloop_unroll=int(config.get("KLOOP_UNROLL", 2)),
    )
    for key in (
        "STATIC_SCHEDULER",
        "NUM_WARPS",
        "MEGA_NUM_SMEM_BUFFERS",
        "PIPELINE_CHUNK_ROWS",
        "PIPELINE_LEAD_CHUNKS",
        "USE_DEVICE_TENSORMAPS",
        "PREFETCH_FIRST_SCATTER_PTR",
    ):
        if key in config:
            retuned[key] = config[key]
    return retuned


def _require_decode_dtype(dtype: BlockScaledFormatSpec) -> None:
    dtype_name = getattr(dtype, "name", "").lower()
    if dtype_name not in _MXFP8_FORMAT_NAMES and dtype_name not in _FP4_FORMAT_NAMES:
        raise ValueError(
            "blockscaled decode configs support only MXFP8 and FP4 dtypes, "
            f"got {getattr(dtype, 'name', dtype)!r}"
        )


def blockscaled_mega_decode_config(
    dtype: BlockScaledFormatSpec,
    *,
    weight_dtype: BlockScaledFormatSpec | None = None,
    num_tokens_per_expert: int | None = None,
    hidden_dim: int | None = None,
    intermediate_dim: int | None = None,
    num_local_experts: int | None = None,
    num_sms: int | None = None,
) -> dict[str, int | bool]:
    weight_dtype = _require_supported_weight_dtype(dtype, weight_dtype)
    config_family = dtype
    family_name = getattr(config_family, "name", "").lower()
    if family_name in _MXFP8_FORMAT_NAMES:
        config = _copy_config(_MXFP8_MEGA_DECODE_CONFIG)
    elif family_name == NVFP4.name:
        config = _copy_config(_NVFP4_MEGA_DECODE_CONFIG)
    else:
        config = _copy_config(_MXFP4_MEGA_DECODE_CONFIG)
    mixed_operand_widths = dtype.a_dtype.width != weight_dtype.b_dtype.width
    max_token_tile = None
    if num_tokens_per_expert is not None and mixed_operand_widths:
        max_token_tile = max(
            32,
            256 * weight_dtype.b_dtype.width // dtype.a_dtype.width,
        )
    geometry_applied = (
        num_tokens_per_expert is not None and 0 < num_tokens_per_expert <= 256
    )
    config = derive_blockscaled_inference_geometry(
        config,
        format_name=family_name,
        staged=False,
        num_tokens_per_expert=num_tokens_per_expert,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_local_experts=num_local_experts,
        num_sms=num_sms,
        retune_config=lambda candidate: _retune_config_for_formats(
            candidate,
            dtype=dtype,
            weight_dtype=weight_dtype,
        ),
        prefer_deeper_pipeline=mixed_operand_widths,
        pipeline_depth_widths=(
            (dtype.a_dtype.width, weight_dtype.b_dtype.width)
            if mixed_operand_widths
            else None
        ),
        max_token_tile=max_token_tile,
        use_mxfp4_mega_tile_buckets=dtype is MXFP4 and weight_dtype is dtype,
    )
    config = _retune_config_for_formats(config, dtype=dtype, weight_dtype=weight_dtype)
    if geometry_applied and not mixed_operand_widths:
        initialize_blockscaled_inference_pipeline(config)
    return config


def blockscaled_staged_decode_config(
    dtype: BlockScaledFormatSpec,
    *,
    weight_dtype: BlockScaledFormatSpec | None = None,
    num_tokens_per_expert: int | None = None,
    hidden_dim: int | None = None,
    intermediate_dim: int | None = None,
    num_local_experts: int | None = None,
    num_sms: int | None = None,
) -> dict[str, int | bool]:
    weight_dtype = _require_supported_weight_dtype(dtype, weight_dtype)
    config_family = dtype
    family_name = getattr(config_family, "name", "").lower()
    if family_name in _MXFP8_FORMAT_NAMES:
        config = _copy_config(_MXFP8_MEMORY_CONFIG)
    else:
        config = _copy_config(_FP4_MEMORY_CONFIG)
    seed_geometry = {
        key: config[key]
        for key in (
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "NUM_MMAS",
            "EPILOGUE_SUBTILE",
        )
    }
    geometry_applied = (
        num_tokens_per_expert is not None and 0 < num_tokens_per_expert <= 256
    )
    mixed_operand_widths = dtype.a_dtype.width != weight_dtype.b_dtype.width
    use_wide_k = (
        mixed_operand_widths
        and num_tokens_per_expert is not None
        and num_local_experts is not None
        and num_sms is not None
        and num_tokens_per_expert * num_local_experts
        <= num_sms * dtype.a_dtype.width // weight_dtype.b_dtype.width
    )
    if use_wide_k:
        config["BLOCK_SIZE_K"] = (
            int(config["BLOCK_SIZE_K"])
            * dtype.a_dtype.width
            // weight_dtype.b_dtype.width
        )
    config = derive_blockscaled_inference_geometry(
        config,
        format_name=family_name,
        staged=True,
        num_tokens_per_expert=num_tokens_per_expert,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_local_experts=num_local_experts,
        num_sms=num_sms,
        retune_config=lambda candidate: _retune_config_for_formats(
            candidate,
            dtype=dtype,
            weight_dtype=weight_dtype,
        ),
        prefer_deeper_pipeline=mixed_operand_widths,
        pipeline_depth_widths=(
            (dtype.a_dtype.width, weight_dtype.b_dtype.width) if use_wide_k else None
        ),
    )
    config = _retune_config_for_formats(config, dtype=dtype, weight_dtype=weight_dtype)
    geometry_changed = any(config[key] != value for key, value in seed_geometry.items())
    # Mixed-width retuning derives complete pipeline depths for the final
    # geometry; same-width templates retain depths that must be reinitialized.
    if (
        dtype.a_dtype.width == weight_dtype.b_dtype.width
        and geometry_applied
        and geometry_changed
    ):
        initialize_blockscaled_inference_pipeline(config)
    return config


def blockscaled_training_config(
    dtype: BlockScaledFormatSpec,
    *,
    wgrad: bool = False,
) -> dict[str, int | bool]:
    """Return the pinned compute-bound config for one training operation.

    FPROP/DGRAD and WGRAD share the measured schedule today. ``wgrad`` keeps
    the operation-specific API available for an independently tuned schedule.
    """
    dtype_name = getattr(dtype, "name", "").lower()
    if dtype_name in _MXFP8_FORMAT_NAMES:
        config = _MXFP8_COMPUTE_CONFIG
    elif dtype_name in _FP4_FORMAT_NAMES:
        config = _FP4_COMPUTE_CONFIG
    else:
        raise ValueError(f"unsupported block-scaled format {dtype_name!r}")
    return _copy_config(config)


def _host_descriptor_required_mn_multiples(cfg: dict, G: int) -> tuple[int, int]:
    if G == 1:
        return 1, 1
    block_m = int(cfg["BLOCK_SIZE_M"])
    block_n = int(cfg["BLOCK_SIZE_N"])
    if cfg.get("SWAP_AB", False):
        return block_n, block_m
    return block_m, block_n


def _host_tensormaps_available(
    cfg: dict,
    *,
    problem_type: int,
    m_multiple_of: int,
    N: int,
    G: int,
) -> bool:
    if problem_type == _WGRAD:
        return False
    m_multiple, n_multiple = _host_descriptor_required_mn_multiples(cfg, G)
    return m_multiple_of % m_multiple == 0 and N % n_multiple == 0


def _require_config_compatible(
    cfg: dict,
    *,
    m_multiple_of: int,
    N: int,
    G: int,
) -> dict:
    if m_multiple_of % BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT != 0:
        raise ValueError(
            "m_multiple_of must be divisible by "
            f"{BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT}, got {m_multiple_of}"
        )
    return cfg


def _is_config_compatible(cfg: dict, *, m_multiple_of: int, N: int, G: int) -> bool:
    return m_multiple_of % BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT == 0


def _pick_literal_config(
    *,
    compute_config: dict[str, int | bool],
    fallback_config: dict[str, int | bool],
    memory_config: dict[str, int | bool],
    memory_bound: bool,
    m_multiple_of: int,
    N: int,
    G: int,
) -> dict:
    # Memory-bound path still falls back through the compute-tier configs
    # if the memory config's BLOCK_N/BLOCK_M doesn't tile N (e.g. G>1,
    # N=384 with memory_config's BLOCK_N=256). Order: memory -> compute
    # -> fallback (base-aligned).
    if memory_bound:
        candidates = (memory_config, compute_config, fallback_config)
    else:
        candidates = (compute_config, fallback_config)
    for candidate in candidates:
        cfg = _copy_config(candidate)
        if _is_config_compatible(cfg, m_multiple_of=m_multiple_of, N=N, G=G):
            return cfg
    return _require_config_compatible(
        _copy_config(candidates[0]), m_multiple_of=m_multiple_of, N=N, G=G
    )


def auto_blockscaled_config(
    *,
    GM: int,
    G: int,
    N: int,
    K: int,
    num_sms: int,
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    weight_format: BlockScaledFormatSpec | None = None,
    m_multiple_of: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    static_scheduler: bool | None = None,
    problem_type: int = _FPROP,
    enable_sm103_ultra: bool = False,
) -> dict:
    """Pick a production config (compute / base-aligned fallback / memory)
    for this shape + format. ``GM // G`` chooses the compute-vs-memory
    regime; ``m_multiple_of`` only enforces the grouped-GEMM base row-alignment
    contract because the launcher can use device tensormaps when a preferred
    config needs stricter host-descriptor M alignment. ``static_scheduler``
    overrides the selected config's default. Only FPROP gates on
    ``memory_bound``: DGRAD/WGRAD always pick the compute-tier config.
    For WGRAD the ``GM`` here is the output-rows axis
    (``a_outer = dy.shape[0] = N_out``), not tokens, so the
    ``GM // G < 256`` proxy mismeasures; for DGRAD the bwd-pass shapes
    have the same compute density as fwd but are reached only via
    grad-of-output paths that are uniformly compute-bound.
    """
    # ``format`` may be the kernel-local ``BlockScaledFormatSpec`` or
    # the public block-scaled format enum.
    weight_format = _resolve_weight_format(format, weight_format)
    fmt_name = getattr(format, "name", "").lower()
    a8w4 = _is_a8w4_format(format, weight_format)
    if format is not weight_format and not a8w4:
        raise NotImplementedError(
            "mixed block-scaled grouped GEMM currently supports only "
            "dtype:mxfp8_e4m3;weight_dtype:mxfp4"
        )
    family_name = fmt_name
    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    if m_multiple_of % BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT != 0:
        raise ValueError(
            "m_multiple_of must be divisible by "
            f"{BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT}, got {m_multiple_of}"
        )
    m_per_group_estimate = max(1, GM // G)
    memory_bound = problem_type == _FPROP and m_per_group_estimate < 256

    if family_name in _MXFP8_FORMAT_NAMES:
        cfg = _pick_literal_config(
            compute_config=_MXFP8_COMPUTE_CONFIG,
            fallback_config=_MXFP8_FALLBACK_CONFIG,
            memory_config=_MXFP8_MEMORY_CONFIG,
            memory_bound=memory_bound,
            m_multiple_of=m_multiple_of,
            N=N,
            G=G,
        )
    elif family_name in _FP4_FORMAT_NAMES:
        compute_config = (
            _SM103_NVFP4_COMPUTE_CONFIG
            if (
                enable_sm103_ultra
                and problem_type != _WGRAD
                and K % sm103.SM103_TILE_K == 0
            )
            else _FP4_COMPUTE_CONFIG
        )
        cfg = _pick_literal_config(
            compute_config=compute_config,
            fallback_config=_FP4_FALLBACK_CONFIG,
            memory_config=_FP4_MEMORY_CONFIG,
            memory_bound=memory_bound,
            m_multiple_of=m_multiple_of,
            N=N,
            G=G,
        )
        # These schedules were qualified only at NTPE=8192. Keep the equality
        # exact instead of extrapolating from a routing-average estimate.
        if (
            family_name == "mxfp4"
            and problem_type == _FPROP
            and m_per_group_estimate == 8192
            and (N, K) == _MXFP4_T6_FC2_PREFILL_CONTRACTION
            and not cfg.get("USE_SM103_ULTRA", False)
        ):
            cfg = _copy_config(_MXFP4_T6_FC2_PREFILL_CONFIG)
        elif (
            family_name == "nvfp4"
            and problem_type == _FPROP
            and m_per_group_estimate == 8192
            and (N, K) in _NVFP4_T6_PREFILL_CONTRACTIONS
            and not cfg.get("USE_SM103_ULTRA", False)
        ):
            cfg["NUM_SMEM_BUFFERS"] = 4
    else:
        raise ValueError(
            "No block-scaled production config among the literal "
            f"configs for format={fmt_name!r}, weight_format={family_name!r}, "
            f"G={G}, GM={GM}, N={N}, K={K}, "
            f"num_sms={num_sms}, m_per_group_estimate={m_per_group_estimate}, "
            f"m_multiple_of={m_multiple_of}."
        )
    cfg = _retune_config_for_formats(cfg, dtype=format, weight_dtype=weight_format)
    if static_scheduler is not None:
        cfg["STATIC_SCHEDULER"] = static_scheduler
    return cfg


def inference_fprop_static_scheduler(
    *,
    inference_mode: bool,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None,
    N: int,
    K: int,
    problem_type: int,
) -> bool | None:
    if not inference_mode or problem_type != _FPROP:
        return None
    format_name = getattr(format, "name", "").lower()
    if format_name == "mxfp4" or _is_a8w4_format(format, weight_format):
        return True if (N, K) in STATIC_FPROP_SCHEDULER_SHAPES else None
    return None
