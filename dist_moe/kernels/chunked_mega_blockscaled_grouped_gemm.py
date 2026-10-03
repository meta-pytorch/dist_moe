# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chunk-pipelined forward MegaMoE kernel experiments."""

import dataclasses
import logging
import math

import cutlass
import torch

from ..formats import (
    canonical_swiglu_clamp,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)
from . import (
    mega_blockscaled_grouped_gemm as base,
    sm103_blockscaled_helpers as sm103,
    weight_borrow as wborrow,
)
from ._environment import num_sms_per_device
from .activation_buffer_kernel import (
    _dispatch_quant_source_dtype_from_torch,
)
from .blockscaled_grouped_gemm import (
    _config_cache_key,
    _format_uses_fp4,
    _is_a8w4_format,
    _parse_blockscaled_problem,
    _resolve_output_tensor,
    _resolve_weight_format,
    _set_swiglu_clamp_config,
    _torch_layout_signature,
    _validate_blockscaled_launch_inputs,
    auto_blockscaled_config,
    blockscaled_training_config,
    BlockScaledFormatSpec,
    make_mixed_blockscaled_format,
    MXFP4,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
)
from .blockscaled_quantize import get_nvfp4_recip_lut
from .chunked_mega_blockscaled_grouped_gemm_kernel import (
    _chunked_mega_producer_topology,
    _MAX_PIPELINE_CHUNK_ROWS,
    _PIPELINE_CHUNK_ROW_ALIGNMENT,
    _ReleaseOnlyChunkedMegaBlockScaledGroupedGemmKernel,
    _validate_pipeline_chunk_rows,
    ChunkedMegaBlockScaledGroupedGemmKernel,
)
from .config import (
    BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES,
    blockscaled_epilogue_subtile_divisor,
    DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    derive_blockscaled_epilogue_tile_n,
    derive_blockscaled_num_tmem_buffers,
    EPILOGUE_SUBTILE_AUTO,
    mixed_blockscaled_tile_prefers_deep_pipeline,
    SM100_TMEM_COLS,
    SMEM_LAUNCH_SAFETY_MARGIN_BYTES,
    uses_paged_blockscaled_scale_rows,
)
from .dist_blockscaled_grouped_gemm import (
    _dispatch_scale_copy_atom_aligned,
    _logical_dim_size,
    _maybe_slice_output,
    _use_nvfp4_high_throughput_producers,
    _validate_global_scale_inv,
    _validate_precleared_counter_storage,
)
from .grouped_gemm import _FPROP
from .swiglu_epilogue import (
    MEGA_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD,
    STAGED_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD,
)
from .tile_scheduler import CHUNKED_MEGA_WORK_INFO_FIELDS

__all__ = [
    "ChunkedMegaBlockScaledGroupedGemmKernel",
    "allocate_decode_counter_storage",
    "chunked_mega_blockscaled_grouped_gemm_fprop_swiglu_fwd",
    "derive_chunked_mega_pipeline_config",
]

_PIPELINE_COUNTER_SIZE = 1
# Dummy row extent for unrequested (never-written) wgrad column storages:
# one routing block, which satisfies the FP4 row-packing and per-sf_vec_size
# scale-shape constraints of every format.
_DUMMY_COL_ROWS = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
_MX_PREFILL_VECTOR_COPY_MIN_SCALE_STRIP_BYTES = 256
_PIPELINE_COUNTER_HEADER_SIZE = 3
_SMEM_ALIGNMENT_BYTES = 1024
_TENSORMAP_ALIGNMENT_BYTES = 128
_MEGA_TENSORMAP_BYTES = 10 * 128
_BF16_BYTES = 2

logger: logging.Logger = logging.getLogger(__name__)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _derive_pipeline_chunk_rows(
    *,
    rows: int,
    groups: int,
    activation_block: int,
) -> int:
    chunk_alignment = math.lcm(_PIPELINE_CHUNK_ROW_ALIGNMENT, activation_block)
    max_chunk_rows = _MAX_PIPELINE_CHUNK_ROWS // chunk_alignment * chunk_alignment
    rows_per_expert = (rows + groups - 1) // groups
    return min(
        max_chunk_rows,
        max(chunk_alignment, _align_up(rows_per_expert, chunk_alignment)),
    )


def _derive_mixed_pipeline_chunk_rows(
    *,
    rows: int,
    groups: int,
    activation_block: int,
    feature_width: int,
    feature_block: int,
    num_sms: int,
    num_ctas: int,
    activation_width: int,
    weight_width: int,
) -> int:
    chunk_rows = _derive_pipeline_chunk_rows(
        rows=rows,
        groups=groups,
        activation_block=activation_block,
    )
    width_ratio = max(1, activation_width // weight_width)
    feature_tiles = (feature_width + feature_block - 1) // feature_block
    effective_ctas_per_activation_tile = feature_tiles * num_ctas * width_ratio
    activation_tiles_per_frontier = (
        num_sms + effective_ctas_per_activation_tile - 1
    ) // effective_ctas_per_activation_tile
    chunk_alignment = math.lcm(_PIPELINE_CHUNK_ROW_ALIGNMENT, activation_block)
    frontier_rows = _align_up(
        activation_tiles_per_frontier * activation_block,
        chunk_alignment,
    )
    return min(chunk_rows, max(chunk_alignment, frontier_rows))


def _estimate_chunked_mega_smem_bytes(
    config: dict[str, int | bool],
    *,
    format: BlockScaledFormatSpec,
    groups: int,
    dispatch_quant_groups: int,
    nvfp4_row_reduction_slots: int,
    use_global_scale_inv: bool,
) -> int:
    block_m = int(config["BLOCK_SIZE_M"])
    block_n = int(config["BLOCK_SIZE_N"])
    block_k = int(config["BLOCK_SIZE_K"])
    num_ctas = int(config["NUM_CTAS"])
    num_smem_buffers = int(config["NUM_SMEM_BUFFERS"])
    num_tmem_buffers = int(config["NUM_TMEM_BUFFERS"])
    num_tile_buffers = int(config["NUM_TILE_BUFFERS"])
    num_c_stages = int(config["NUM_C_STAGES"])
    epilogue_subtile = int(config["EPILOGUE_SUBTILE"])

    needs_unpack_tma = format.a_dtype.width != format.b_dtype.width
    a_smem_width = (
        8 if needs_unpack_tma and format.a_dtype.width < 8 else format.a_dtype.width
    )
    b_smem_width = (
        8 if needs_unpack_tma and format.b_dtype.width < 8 else format.b_dtype.width
    )
    a_stage_bytes = block_m // num_ctas * block_k * a_smem_width // 8
    b_stage_bytes = block_n // num_ctas * block_k * b_smem_width // 8
    sf_k_blocks = block_k // format.sf_vec_size
    sfa_stage_bytes = block_m // num_ctas * sf_k_blocks
    sfb_stage_bytes = _align_up(block_n, 128) * sf_k_blocks
    if epilogue_subtile == 0:
        epi_n = derive_blockscaled_epilogue_tile_n(
            block_m=block_m,
            block_n=block_n,
            num_ctas=num_ctas,
            c_stage_dtype=torch.bfloat16,
        )
    else:
        epilogue_divisor = blockscaled_epilogue_subtile_divisor(
            epilogue_subtile=epilogue_subtile,
            c_width=cutlass.BFloat16.width,
            a_width=format.a_dtype.width,
            full_width=bool(config.get("EPILOGUE_TILE_FULL_WIDTH", False)),
        )
        if block_n % epilogue_divisor != 0:
            raise ValueError(
                f"BLOCK_SIZE_N={block_n} must be divisible by the Mega epilogue "
                f"divisor {epilogue_divisor}"
            )
        epi_n = block_n // epilogue_divisor
    epi_m = min(block_m // num_ctas, 128)
    c_smem_skew = 8 if num_tmem_buffers == 1 else 0
    c_smem_elems = (
        (epi_m + c_smem_skew) * epi_n
        if bool(config.get("SWAP_AB", False))
        else epi_m * (epi_n + c_smem_skew)
    )
    c_smem_bytes = c_smem_elems * _BF16_BYTES * num_c_stages

    offset = 0
    for region_bytes in (
        c_smem_bytes,
        a_stage_bytes * num_smem_buffers,
        b_stage_bytes * num_smem_buffers,
        sfa_stage_bytes * num_smem_buffers,
        sfb_stage_bytes * num_smem_buffers,
    ):
        offset = _align_up(offset, _SMEM_ALIGNMENT_BYTES) + region_bytes

    offset += 4 * num_tile_buffers
    offset += 4 * dispatch_quant_groups
    offset += 8 * dispatch_quant_groups * format.sf_vec_size
    offset += 8 * max(epi_m, epi_n)
    if use_global_scale_inv:
        offset += 4 * block_n
        offset += 4 * nvfp4_row_reduction_slots * dispatch_quant_groups
    tensormap_bytes = max(
        _MEGA_TENSORMAP_BYTES,
        8 * ((groups + CHUNKED_MEGA_WORK_INFO_FIELDS * num_tile_buffers + 1) // 2),
    )
    offset = _align_up(offset, _TENSORMAP_ALIGNMENT_BYTES) + tensormap_bytes
    barrier_count = (
        2 * num_smem_buffers
        + 2 * num_tmem_buffers
        + 2 * num_tile_buffers
        + (2 if num_ctas == 2 else 0)
        + (1 if num_ctas == 2 else 0)
        + (1 if bool(config.get("OVERLAPPING_ACCUM", False)) else 0)
    )
    offset += barrier_count * 8 + 4
    return _align_up(offset, _SMEM_ALIGNMENT_BYTES)


def derive_chunked_mega_pipeline_config(
    config: dict[str, int | bool],
    *,
    format: BlockScaledFormatSpec,
    groups: int,
    dispatch_quant_groups: int,
    nvfp4_row_reduction_slots: int,
    use_global_scale_inv: bool,
) -> dict[str, int | bool]:
    """Derive the deepest legal Mega accumulator and operand pipelines."""
    derived = dict(config)
    max_tmem_buffers = max(
        1,
        SM100_TMEM_COLS // (int(derived["BLOCK_SIZE_N"]) * int(derived["NUM_MMAS"])),
    )
    num_tmem_buffers = derive_blockscaled_num_tmem_buffers(
        block_m=int(derived["BLOCK_SIZE_M"]),
        block_n=int(derived["BLOCK_SIZE_N"]),
        num_mmas=int(derived["NUM_MMAS"]),
        overlapping_accum=bool(derived.get("OVERLAPPING_ACCUM", False)),
        max_tmem_buffers=max_tmem_buffers,
    )
    derived["NUM_TMEM_BUFFERS"] = num_tmem_buffers
    derived["NUM_TILE_BUFFERS"] = num_tmem_buffers + 1
    derived["NUM_C_STAGES"] = 1

    budget = BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES - SMEM_LAUNCH_SAFETY_MARGIN_BYTES
    for num_smem_buffers in range(32, 1, -1):
        derived["NUM_SMEM_BUFFERS"] = num_smem_buffers
        if (
            _estimate_chunked_mega_smem_bytes(
                derived,
                format=format,
                groups=groups,
                dispatch_quant_groups=dispatch_quant_groups,
                nvfp4_row_reduction_slots=nvfp4_row_reduction_slots,
                use_global_scale_inv=use_global_scale_inv,
            )
            <= budget
        ):
            derived["MEGA_NUM_SMEM_BUFFERS"] = num_smem_buffers
            return derived
    raise AssertionError(
        "no chunked MegaMoE SMEM stage count fits "
        f"BLOCK_M={derived['BLOCK_SIZE_M']}, "
        f"BLOCK_N={derived['BLOCK_SIZE_N']}, "
        f"BLOCK_K={derived['BLOCK_SIZE_K']}"
    )


def _verify_chunked_mega_smem_estimate(
    kernel: "ChunkedMegaBlockScaledGroupedGemmKernel",
    config: dict[str, int | bool],
    *,
    format: BlockScaledFormatSpec,
    groups: int,
    use_global_scale_inv: bool,
) -> None:
    if getattr(kernel, "_smem_estimate_verified", False) or not hasattr(
        kernel, "shared_storage"
    ):
        return
    estimated = _estimate_chunked_mega_smem_bytes(
        config,
        format=format,
        groups=groups,
        dispatch_quant_groups=kernel.DISPATCH_QUANT_GROUPS,
        nvfp4_row_reduction_slots=kernel.NVFP4_ROW_REDUCTION_SLOTS,
        use_global_scale_inv=use_global_scale_inv,
    )
    actual = kernel.shared_storage.size_in_bytes()
    if estimated < actual:
        raise AssertionError(
            "chunked MegaMoE SMEM estimate undercounts kernel storage: "
            f"estimated={estimated}, actual={actual}, config={config}"
        )
    budget = BLACKWELL_DYNAMIC_SMEM_BUDGET_BYTES - SMEM_LAUNCH_SAFETY_MARGIN_BYTES
    if actual > budget:
        raise AssertionError(
            "chunked MegaMoE kernel storage exceeds the launch-safe budget: "
            f"actual={actual}, budget={budget}, config={config}"
        )
    if estimated > actual:
        logger.warning(
            "Chunked MegaMoE SMEM estimate is conservative: "
            f"estimated={estimated}, actual={actual}, config={config}"
        )
    kernel._smem_estimate_verified = True


_CHUNKED_KERNEL_CACHE: dict[tuple, ChunkedMegaBlockScaledGroupedGemmKernel] = {}


def _get_chunked_kernel(
    *,
    config: dict,
    format,
    world_size: int,
    swiglu_fast_math: bool,
    chunk_rows: int,
    quantize_dispatch: bool,
    swiglu_row_quant_only: bool,
    dispatch_col_quant: bool,
    source_dtype: type[cutlass.Numeric],
    swiglu_k: int,
    nvfp4_high_throughput: bool,
    nvfp4_wide_row_quant: bool,
    vector_dispatch_scale_copy: bool,
    mx_narrow_copy_work: bool,
    wide_gather: bool,
    interleaved_fc13: bool,
    activation_ring_fc2_done_offset: int = 0,
    activation_ring_fc13_done_offset: int = 0,
    swiglu_clamp: tuple[bool, float, float] = (
        False,
        SWIGLU_CLAMP_ALPHA_DEFAULT,
        SWIGLU_CLAMP_LIMIT_DEFAULT,
    ),
    activation_ring_h1_done_offset: int = 0,
    weight_borrow_slots: int = 0,
    weight_borrow_work_offset: int = 0,
    weight_borrow_done_offset: int = 0,
) -> ChunkedMegaBlockScaledGroupedGemmKernel:
    if format is NVFP4 and nvfp4_high_throughput and not wide_gather:
        raise ValueError(
            "NVFP4 high-throughput producers require the wide-gather topology"
        )
    key = (
        _config_cache_key(config),
        format.name,
        world_size,
        swiglu_fast_math,
        swiglu_clamp,
        chunk_rows,
        quantize_dispatch,
        swiglu_row_quant_only,
        dispatch_col_quant,
        source_dtype,
        swiglu_k,
        nvfp4_high_throughput,
        nvfp4_wide_row_quant,
        vector_dispatch_scale_copy,
        mx_narrow_copy_work,
        wide_gather,
        interleaved_fc13,
        # The ring done-counter offsets are shape-dependent trace-time
        # constexprs baked into the compiled kernel; keying them here keeps
        # concurrent first-use compiles of different shapes from sharing a
        # mutable kernel object with the wrong offsets.
        activation_ring_fc2_done_offset,
        activation_ring_fc13_done_offset,
        activation_ring_h1_done_offset,
        weight_borrow_slots,
        weight_borrow_work_offset,
        weight_borrow_done_offset,
    )
    kernel = _CHUNKED_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel_config = dict(config)
        if swiglu_fast_math:
            kernel_config["COMBINE_SWIGLU_FAST_MATH"] = True
        _set_swiglu_clamp_config(kernel_config, *swiglu_clamp)
        if swiglu_row_quant_only:
            kernel_config["COMBINE_SWIGLU_ROW_QUANT_ONLY"] = True
        kernel_cls = (
            _ReleaseOnlyChunkedMegaBlockScaledGroupedGemmKernel
            if nvfp4_wide_row_quant
            else ChunkedMegaBlockScaledGroupedGemmKernel
        )
        kernel = kernel_cls.from_config(
            kernel_config,
            mode=kernel_cls.MEGA_FORWARD_MODE,
            format=format,
            force_n_major=False,
            num_n_clusters=1,
            world_size=world_size,
            dispatch_source_dtype=source_dtype,
            combine_swiglu_k=swiglu_k,
            chunk_rows=chunk_rows,
            wide_gather=wide_gather,
            interleaved_fc13=interleaved_fc13,
        )
        kernel.QUANTIZE_DISPATCH = quantize_dispatch
        kernel.DISPATCH_COL_QUANT = dispatch_col_quant
        kernel.ACTIVATION_RING_FC2_DONE_OFFSET = activation_ring_fc2_done_offset
        kernel.ACTIVATION_RING_FC13_DONE_OFFSET = activation_ring_fc13_done_offset
        kernel.ACTIVATION_RING_H1_DONE_OFFSET = activation_ring_h1_done_offset
        kernel.WEIGHT_BORROW_SLOTS = weight_borrow_slots
        kernel.WEIGHT_BORROW_WORK_OFFSET = weight_borrow_work_offset
        kernel.WEIGHT_BORROW_DONE_OFFSET = weight_borrow_done_offset
        if format is NVFP4:
            kernel._configure_nvfp4_producer_topology(
                nvfp4_high_throughput,
                nvfp4_wide_row_quant,
            )
        if vector_dispatch_scale_copy:
            # Enable-only: the NVFP4 wide-gather topology already turns this on
            # and sizes its warp count for it, so assigning the narrow-path
            # policy unconditionally would disable it for wide-gather prefill.
            kernel.VECTOR_DISPATCH_SCALE_COPY = True
            if format is NVFP4:
                # Measured for the T6 narrow shape; wider work items halve
                # decode copy parallelism for the MX formats (T5 66 -> 78 us).
                kernel.FORWARD_COPY_COL_TILES_PER_WORK = 8
        if mx_narrow_copy_work:
            # Decode routes so few rows that the copy producer needs more work
            # items, not bigger ones: T5 65.4 -> 59.2 us, T6 86.3 -> 80.6 us.
            # Prefill measured 2.4% slower with this, so it stays decode-only.
            kernel.FORWARD_COPY_COL_TILES_PER_WORK = 2
        _CHUNKED_KERNEL_CACHE[key] = kernel
    return kernel


def _use_nvfp4_high_throughput_topology(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> bool:
    # The square decode shape regressed with the extra producer warps; its
    # measured winner is the dedicated wide-row quant topology below.
    return _use_nvfp4_high_throughput_producers(
        format=format,
        rows=rows,
        num_sms=num_sms,
        hidden_dim=hidden_dim,
        mega=True,
    ) and (hidden_dim, intermediate_dim) != (4096, 4096)


def _use_nvfp4_wide_row_quant(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> bool:
    return (
        format is NVFP4
        and rows < num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
        and (hidden_dim, intermediate_dim) == (4096, 4096)
    )


def _use_vector_dispatch_scale_copy(
    *,
    format: BlockScaledFormatSpec,
    hidden_dim: int,
    intermediate_dim: int,
    wide_gather: bool,
    rows: int,
    num_sms: int,
) -> bool:
    # The blockscaled-dispatch copy producer moves scale factors one Uint8 at
    # a time unless this is set, and each byte is a dependent peer-NVLink
    # load+store -- at decode that latency chain dominates the whole fused
    # kernel (T5 EP4-proxy MXFP4 88.2 -> 65.9 us, T6 107.0 -> 86.8 us with
    # 4-byte copies). NVFP4 keeps its measured policy (wide-gather bundles the
    # flag; the (12288, 3072) narrow shape enables it explicitly). Other
    # formats vectorize whenever a whole 4-column scale atom exists, except at
    # prefill row counts with short per-row scale strips, where the wider
    # copies serialize behind the gather producer instead: with 128-byte
    # strips (D=4096) T5 MXFP4 at 65,536 tokens/rank measures 13,824 -> 14,093
    # us vectorized, while 384-byte strips (D=12288) keep large vectorized
    # wins through 64K tokens (T6 A8W4 at 32,768: 28,333 -> 22,909 us).
    if format is NVFP4:
        return (hidden_dim, intermediate_dim) == (12288, 3072) and not wide_gather
    if not _dispatch_scale_copy_atom_aligned(format=format, hidden_dim=hidden_dim):
        return False
    return (
        rows < num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
        or hidden_dim // format.sf_vec_size
        >= _MX_PREFILL_VECTOR_COPY_MIN_SCALE_STRIP_BYTES
    )


def _use_wide_gather_topology(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
    interleaved_fc13: bool = False,
) -> bool:
    # Wide gather is valid for every shape below. T5 square and T6 contraction
    # set the measured threshold; other MXFP4 contractions use it as a
    # performance heuristic rather than a capability requirement.
    if format is NVFP4:
        high_throughput = _use_nvfp4_high_throughput_topology(
            format=format,
            rows=rows,
            num_sms=num_sms,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )
        return high_throughput or (hidden_dim == 4096 and intermediate_dim == 4096)
    if rows < num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF:
        return False
    if (
        format is MXFP8_E4M3
        and not interleaved_fc13
        and hidden_dim >= 2 * intermediate_dim
    ):
        # MXFP8 contraction-shape training forward is gather-throughput-bound:
        # the gathered operand scales with hidden_dim while FC13 work scales
        # with intermediate_dim. At 12288/3072, the 8-warp gather reduced an
        # EP64 Mega fprop from 4158 to 3493 us on GB300 at E256/NTPE8192. The
        # interleaved inference-prefill path keeps its separately tuned narrow
        # topology.
        return True
    return format is MXFP4 and hidden_dim >= intermediate_dim


def _use_two_slot_schedule_ring(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
) -> bool:
    # Two slots improved square MXFP4 prefill but reduced overlap elsewhere.
    prefill = rows >= num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
    return (
        format is MXFP4 and prefill and hidden_dim == 4096 and intermediate_dim == 4096
    )


def _use_t6_mxfp8_prefill_lead(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
    interleaved_fc13: bool,
) -> bool:
    # The 12K-to-3K MXFP8 prefill shape benefits from leading FC13 publication.
    return (
        format is MXFP8_E4M3
        and interleaved_fc13
        and rows >= num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
        and hidden_dim == 12288
        and intermediate_dim == 3072
    )


def _use_smaller_swiglu_epilogue(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    interleaved_fc13: bool,
) -> bool:
    # Measured T5/T6 FP4 and MXFP8 prefill variants reduce epilogue register
    # pressure; other MXFP8 decode shapes retain the wider fragment.
    if not interleaved_fc13:
        return False
    if format is MXFP4:
        return True
    return format is MXFP8_E4M3 and (
        rows >= num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
    )


def _use_staged_swiglu_fragment(
    *,
    format: BlockScaledFormatSpec,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
    interleaved_fc13: bool,
) -> bool:
    # The square T5 MXFP8 prefill shape gains from staging a wider fragment;
    # the T6 contraction and decode rows measured slower with it.
    return (
        format is MXFP8_E4M3
        and interleaved_fc13
        and rows >= num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
        and hidden_dim == 4096
        and intermediate_dim == 4096
    )


# Measured crossover for the mixed-width (A8W4) chunked mega on the wide-
# intermediate shapes (hidden <= 2*intermediate, e.g. the T5 square): the
# stage-all schedule loses to the wave schedule once the batch reaches a few
# row waves -- EP4 100-iter CUDA-graph means T5 2,048 tok 1,017.8 -> 904.9 us
# (-11.1%) and 1,024 tok -5.5%; EP8 confirms -8.6% / -8.3% on the same T5
# cells. The contraction shapes do NOT join the narrowing: T7 measured only
# noise-level gains at EP4 and +4.1% at EP8 T=2,048 under the wave schedule,
# and the pure formats measured +10.5% / +22.5% on their cells, so both keep
# the 128-row/SM stage-all band. True decode stays staged everywhere.
_STAGE_ALL_MIXED_WIDTH_MIN_ROWS_PER_SM: int = 32


def _stage_all_fc13(
    *,
    interleaved_fc13: bool,
    rows: int,
    num_sms: int,
    hidden_dim: int,
    intermediate_dim: int,
    mixed_weight_format: bool = False,
) -> bool:
    # Avoid parking persistent clusters on FC2 dependencies when FC2 has a
    # large task frontier or FC13 publishes activations from its epilogue.
    narrow_stage_all = mixed_weight_format and hidden_dim <= 2 * intermediate_dim
    decode_rows_per_sm = (
        _STAGE_ALL_MIXED_WIDTH_MIN_ROWS_PER_SM
        if narrow_stage_all
        else DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
    )
    decode = rows < num_sms * decode_rows_per_sm
    if decode:
        return interleaved_fc13 or hidden_dim >= 2 * intermediate_dim
    # Interleaved MXFP8-activation prefill up to a few waves gains from
    # staging on local EP4 (T6 32768 rows/rank -14.5%, A8W4 -10.0%), but the
    # same schedule measured 5-16% slower against the ambient-drift control on
    # T6 at EP8-EP32 for exactly the staged tiers, so prefill keeps the wave
    # schedule until a scale-aware policy is measured.
    return False


def _mega_fp4_mma_atom_n(
    *,
    config: dict,
    format: BlockScaledFormatSpec,
    hidden_dim: int,
    intermediate_dim: int,
    mixed_operand_widths: bool,
) -> int:
    block_n = int(config["BLOCK_SIZE_N"])
    if (
        mixed_operand_widths
        or format is NVFP4
        or (format is MXFP4 and (hidden_dim, intermediate_dim) == (12288, 3072))
    ):
        return block_n
    preferred_atom_n = min(128, block_n)
    return preferred_atom_n if block_n % preferred_atom_n == 0 else block_n


def _fc2_tiles_per_schedule_item(*, max_tiles: int, row_waves: int) -> int:
    candidate = min(max_tiles, max(1, row_waves))
    return 1 << (candidate.bit_length() - 1)


def _mega_row_scale_rows(*, rows: int, groups: int, block_size_n: int) -> int:
    if uses_paged_blockscaled_scale_rows(block_size_n):
        scale_page_rows = ((block_size_n + 127) // 128) * 128
        scale_rows = (rows + block_size_n - 1) // block_size_n * scale_page_rows
    else:
        scale_page_rows = 0
        scale_rows = rows
    if scale_page_rows:
        scale_rows = max(scale_rows, groups * scale_page_rows)
    return scale_rows


def _allocate_pipeline_workspace(
    *,
    num_clusters: int,
    num_ctas: int,
    device: torch.device,
    input_plan,
    activation_plan,
    pipeline_counter_size: int,
    min_tensormap_rows: int,
    activation_ring_counter_size: int = 0,
):
    counter_splits = _pipeline_counter_storage_splits(
        input_done_counter_size=input_plan.done_counter_size,
        activation_done_counter_size=activation_plan.done_counter_size,
        pipeline_counter_size=pipeline_counter_size,
        activation_ring_counter_size=activation_ring_counter_size,
    )
    counter_storage = torch.empty(
        sum(counter_splits),
        dtype=torch.int32,
        device=device,
    )
    (
        counter,
        dispatch_work,
        dispatch_done,
        activation_work,
        activation_done,
        _pipeline_padding,
        pipeline_counters,
    ) = torch.split(
        counter_storage,
        counter_splits,
    )
    grid_size = max(num_clusters * num_ctas, min_tensormap_rows)
    tensormaps = torch.empty(
        (grid_size, input_plan.tensormap_descriptor_count, 16),
        dtype=torch.int64,
        device=device,
    )
    zero_count = sum(counter_splits[1:])
    workspace = (
        counter,
        tensormaps,
        dispatch_work,
        dispatch_done,
        activation_work,
        activation_done,
        zero_count,
    )
    return workspace, pipeline_counters


def _allocate_zeroed_pipeline_workspace(
    *,
    num_clusters: int,
    num_ctas: int,
    device: torch.device,
    input_plan,
    activation_plan,
    pipeline_counter_size: int,
    min_tensormap_rows: int,
    counter_storage: torch.Tensor | None = None,
    activation_ring_counter_size: int = 0,
):
    counter_splits = _pipeline_counter_storage_splits(
        input_done_counter_size=input_plan.done_counter_size,
        activation_done_counter_size=activation_plan.done_counter_size,
        pipeline_counter_size=pipeline_counter_size,
        activation_ring_counter_size=activation_ring_counter_size,
    )
    required_counter_count = sum(counter_splits)
    if counter_storage is None:
        resolved_counter_storage = torch.zeros(
            required_counter_count,
            dtype=torch.int32,
            device=device,
        )
    else:
        _validate_precleared_counter_storage(
            counter_storage,
            required_counter_count=required_counter_count,
            device=device,
        )
        resolved_counter_storage = counter_storage[:required_counter_count]
    (
        counter,
        dispatch_work,
        dispatch_done,
        activation_work,
        activation_done,
        _pipeline_padding,
        pipeline_counters,
    ) = torch.split(
        resolved_counter_storage,
        counter_splits,
    )
    grid_size = max(num_clusters * num_ctas, min_tensormap_rows)
    tensormaps = torch.empty(
        (grid_size, input_plan.tensormap_descriptor_count, 16),
        dtype=torch.int64,
        device=device,
    )
    workspace = (
        counter,
        tensormaps,
        dispatch_work,
        dispatch_done,
        activation_work,
        activation_done,
        0,
    )
    return workspace, pipeline_counters


def _pipeline_counter_storage_splits(
    *,
    input_done_counter_size: int,
    activation_done_counter_size: int,
    pipeline_counter_size: int,
    activation_ring_counter_size: int = 0,
) -> tuple[int, int, int, int, int, int, int]:
    # The ring's per-schedule-chunk FC2 consumption counters extend the
    # activation done-counter region so the prepare kernel's zeroing sweep
    # and the kernel's done-counter operand cover them without new plumbing.
    activation_slot_bytes = activation_done_counter_size + activation_ring_counter_size
    pipeline_padding = (
        _PIPELINE_COUNTER_HEADER_SIZE + input_done_counter_size + activation_slot_bytes
    ) % 2
    return (
        1,
        1,
        input_done_counter_size,
        1,
        activation_slot_bytes,
        pipeline_padding,
        pipeline_counter_size,
    )


def allocate_decode_counter_storage(
    *,
    rows: int,
    num_groups: int,
    hidden_dim: int,
    format,
    device: torch.device,
) -> torch.Tensor:
    """Allocate counters cleared by the decode routing preparation launch."""
    input_counter_sizes = base._mega_dispatch_done_counter_sizes(
        rows=rows,
        dim=hidden_dim,
        num_groups=num_groups,
        format=format,
    )
    activation_counter_sizes = base._mega_forward_done_counter_sizes(
        rows=rows,
        format=format,
    )
    counter_splits = _pipeline_counter_storage_splits(
        input_done_counter_size=sum(input_counter_sizes),
        activation_done_counter_size=sum(activation_counter_sizes),
        pipeline_counter_size=_PIPELINE_COUNTER_SIZE,
    )
    return torch.empty(sum(counter_splits), dtype=torch.int32, device=device)


def _validate_interleaved_feature_tile(
    config: dict[str, int | bool],
    *,
    interleaved_fc13: bool,
) -> None:
    local_mma_m = int(config["BLOCK_SIZE_M"]) // (
        int(config["NUM_CTAS"]) * int(config["NUM_MMAS"])
    )
    if interleaved_fc13 and local_mma_m != 128:
        raise base._MegaFusedUnsupportedError(
            "interleaved blockscaled FC13 requires a 128-column local MMA tile"
        )


def chunked_mega_blockscaled_grouped_gemm_fprop_swiglu_fwd(  # noqa: C901
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    num_out_tokens: int,
    symm_mem_buffer,
    *,
    num_output_tokens: int | None = None,
    format=MXFP8_E4M3,
    weight_format: BlockScaledFormatSpec | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    chunk_rows: int | None = 512,
    return_x_wgrad_quant: bool = False,
    blockscaled_dispatch: bool = False,
    return_h2_wgrad_quant: bool = True,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    counter_storage: torch.Tensor | None = None,
    w13_global_scale_inv: torch.Tensor | None = None,
    w2_global_scale_inv: torch.Tensor | None = None,
    interleaved_fc13: bool = False,
    return_row_quant: bool = False,
    derive_pipeline_depth: bool = False,
    activation_ring_chunks: int | None = None,
    weight_borrow=None,
):
    """Run fused dispatch, FC13, SwiGLU, FC2, and combine.

    With `activation_buffer`, activation-backed return tensors are shape-only
    aliases at buffer offset zero; consume results through `activation_offsets`.

    `return_row_quant` appends the row-quantized `x_gathered` and `h2` pairs
    this kernel fed to the two GEMMs. Both are produced unconditionally; the
    flag only surfaces them, so a BF16 backward can dequantize the exact A
    operands the forward used.
    """
    if format not in (MXFP8_E4M3, MXFP8_E5M2, MXFP4, NVFP4):
        raise base._MegaFusedUnsupportedError(
            "chunked MegaMoE requires a supported block-scaled format"
        )
    weight_format = _resolve_weight_format(format, weight_format)
    if format is not weight_format and not _is_a8w4_format(format, weight_format):
        raise base._MegaFusedUnsupportedError(
            "chunked MegaMoE mixed block-scaled inference supports only "
            "MXFP8_E4M3 activations with MXFP4 weights"
        )
    kernel_format = make_mixed_blockscaled_format(format, weight_format)
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise base._MegaFusedUnsupportedError(
            "chunked MegaMoE requires BF16 or FP16 output"
        )
    if interleaved_fc13:
        if weight_format not in (MXFP8_E4M3, MXFP4, NVFP4):
            raise base._MegaFusedUnsupportedError(
                "interleaved FC13 supports MXFP8 E4M3, MXFP4, and NVFP4 MegaMoE"
            )
        if return_x_wgrad_quant:
            raise base._MegaFusedUnsupportedError(
                "interleaved FC13 is forward-only inference"
            )
        if return_h2_wgrad_quant:
            raise base._MegaFusedUnsupportedError(
                "interleaved FC13 supports row-quantized inference only"
            )
        if return_row_quant:
            raise base._MegaFusedUnsupportedError(
                "interleaved FC13 returns row-quantized h2 as its primary output; "
                "return_row_quant=True is redundant and unsupported"
            )
    if chunk_rows is not None:
        _validate_pipeline_chunk_rows(chunk_rows)
    if w13.dim() != 3 or w2.dim() != 3:
        raise ValueError("w13 and w2 must be 3D")
    groups, doubled_intermediate_dim, hidden_dim_storage = w13.shape
    hidden_dim = _logical_dim_size(hidden_dim_storage, weight_format)
    if doubled_intermediate_dim % 2 != 0:
        raise ValueError("w13 output dimension must be even")
    intermediate_dim = doubled_intermediate_dim // 2
    w2_groups, w2_hidden_dim, w2_intermediate_dim_storage = w2.shape
    w2_intermediate_dim = _logical_dim_size(w2_intermediate_dim_storage, weight_format)
    if (w2_groups, w2_hidden_dim, w2_intermediate_dim) != (
        groups,
        hidden_dim,
        intermediate_dim,
    ):
        raise ValueError(
            "w2 must have shape "
            f"{(groups, hidden_dim, intermediate_dim)}, got logical shape "
            f"{(w2_groups, w2_hidden_dim, w2_intermediate_dim)}"
        )
    # Expert-borrow widens the group count: the trailing slots' weights are
    # streamed in-kernel from peer publish windows, so the local tables stay
    # `num_local` rows while split_sizes, the plans, the tensormap table and
    # the compiled G all cover `num_local + slots` groups.
    num_local_groups = groups
    borrow_slots = 0 if weight_borrow is None else int(weight_borrow.slots)
    if borrow_slots:
        if format is NVFP4:
            raise base._MegaFusedUnsupportedError(
                "expert-borrow weight streaming does not support NVFP4 yet "
                "(per-expert global scales are not transported)"
            )
        groups = num_local_groups + borrow_slots
    rows = int(num_out_tokens)
    if gather_ptrs.dtype != torch.int64 or gather_ptrs.device != w13.device:
        raise ValueError("gather_ptrs must be int64 on the weight device")
    if scatter_ptrs.dtype != torch.int64 or scatter_ptrs.device != w13.device:
        raise ValueError("scatter_ptrs must be int64 on the weight device")
    if gather_ptrs.numel() < rows or scatter_ptrs.numel() < rows:
        raise ValueError(
            "route pointer tensors have fewer than num_out_tokens entries: "
            f"gather={gather_ptrs.numel()}, scatter={scatter_ptrs.numel()}, "
            f"num_out_tokens={rows}"
        )
    if rows % 128 != 0 or hidden_dim % 128 != 0 or intermediate_dim % 128 != 0:
        raise base._MegaFusedUnsupportedError(
            "chunked MegaMoE dimensions must be multiples of 128"
        )
    use_global_scale_inv = format is NVFP4
    if use_global_scale_inv:
        if hidden_dim % 256 != 0 or intermediate_dim % 256 != 0:
            raise base._MegaFusedUnsupportedError(
                "NVFP4 chunked MegaMoE dimensions must be multiples of 256"
            )
        if hidden_dim > 16384 or intermediate_dim > 8192:
            raise base._MegaFusedUnsupportedError(
                "NVFP4 chunked MegaMoE supports hidden_dim <= 16384 and "
                "intermediate_dim <= 8192"
            )
        for name, scale in (
            ("w13_global_scale_inv", w13_global_scale_inv),
            ("w2_global_scale_inv", w2_global_scale_inv),
        ):
            _validate_global_scale_inv(
                scale,
                size=groups,
                device=w13.device,
                name=name,
            )
    elif w13_global_scale_inv is not None or w2_global_scale_inv is not None:
        raise ValueError("global scale inverses are only supported for NVFP4")
    if split_sizes.dtype != torch.int32:
        split_sizes = split_sizes.to(torch.int32)
    if split_sizes.numel() != groups:
        raise ValueError("split_sizes length must match the local expert count")
    if num_sms is None:
        num_sms = num_sms_per_device()
    # ``MEGA_USE_SM103_ULTRA`` and ``MEGA_GLOBAL_LEAD_CHUNKS`` are overlay
    # requests, not full configs: peel them off before deciding between the
    # caller's config and the auto config so e.g.
    # ``config={"MEGA_GLOBAL_LEAD_CHUNKS": 8}`` composes with auto.
    mega_sm103_ultra = False
    mega_global_lead: int | None = None
    if config is not None:
        config = dict(config)
        mega_sm103_ultra = bool(config.pop("MEGA_USE_SM103_ULTRA", False))
        mega_global_lead = config.pop("MEGA_GLOBAL_LEAD_CHUNKS", None)
        if not config:
            config = None
    auto_config = config is None
    explicit_training_config = (
        config is not None
        and weight_format is format
        and config == blockscaled_training_config(format)
    )
    mixed_operand_widths = kernel_format.a_dtype.width != kernel_format.b_dtype.width
    if config is None:
        config = dict(
            auto_blockscaled_config(
                GM=rows,
                G=groups,
                N=doubled_intermediate_dim,
                K=hidden_dim,
                num_sms=num_sms,
                format=format,
                weight_format=weight_format,
                m_multiple_of=m_multiple_of,
                problem_type=_FPROP,
            )
        )
    else:
        config = dict(config)
    auto_decode_config = "MEGA_NUM_SMEM_BUFFERS" in config
    inference_derived_config = auto_config or auto_decode_config
    # Only the decode presets in `blockscaled_config.py` pass this key, and they
    # pass False; every other caller relies on the default.
    use_device_tensormaps = bool(config.pop("USE_DEVICE_TENSORMAPS", True))
    if mega_sm103_ultra:
        ultra_unsupported = None
        if torch.cuda.get_device_capability(w13.device) != (10, 3):
            ultra_unsupported = "SM103 ultra requires compute capability 10.3"
        elif format not in (NVFP4, MXFP4) or weight_format not in (None, format):
            ultra_unsupported = (
                "SM103 ultra chunked-mega supports only same-width NVFP4 or MXFP4"
            )
        elif not interleaved_fc13:
            ultra_unsupported = (
                "SM103 ultra chunked-mega requires the interleaved FC13 epilogue"
            )
        elif (
            hidden_dim % sm103.SM103_TILE_K != 0
            or intermediate_dim % sm103.SM103_TILE_K != 0
        ):
            ultra_unsupported = (
                "SM103 ultra chunked-mega requires hidden and intermediate "
                f"dims divisible by {sm103.SM103_TILE_K}; got "
                f"{hidden_dim}x{intermediate_dim}"
            )
        elif bool(config.get("STAGE_ALL_FC13", False)) or rows < num_sms * 128:
            # Decode / stage-all geometry keeps every chunk live; the ultra
            # pipeline is prefill-only for now.
            ultra_unsupported = "SM103 ultra chunked-mega is prefill-only"
        if ultra_unsupported is not None:
            raise NotImplementedError(ultra_unsupported)
        else:
            config["BLOCK_SIZE_K"] = sm103.SM103_TILE_K
            config["NUM_SMEM_BUFFERS"] = sm103.SM103_AB_PIPELINE_STAGES
            # The ultra k-tile body is ~3x larger (8 K=96 MMAs + segmented
            # waits); unroll 2 regresses the MMA warp.
            config["KLOOP_UNROLL"] = 1
            config["USE_SM103_ULTRA"] = True
    auto_pipeline_depth = mixed_operand_widths or (
        derive_pipeline_depth and auto_decode_config
    )
    # Normalized up front: the explicit ring request participates in the
    # mixed-width auto-config below (the ring requires device tensormaps).
    if activation_ring_chunks is None:
        activation_ring_chunks = int(config.get("ACTIVATION_RING_CHUNKS", 0))
    else:
        activation_ring_chunks = int(activation_ring_chunks)
    if activation_ring_chunks < 0:
        raise ValueError(
            f"activation_ring_chunks must be >= 0; got {activation_ring_chunks}"
        )
    if inference_derived_config and mixed_operand_widths:
        config["PIPELINE_LEAD_CHUNKS"] = 1
        row_wave_size = num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
        row_waves = (rows + row_wave_size - 1) // row_wave_size
        stage_all_fc13 = _stage_all_fc13(
            interleaved_fc13=interleaved_fc13,
            rows=rows,
            num_sms=num_sms,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            mixed_weight_format=True,
        )
        width_ratio = kernel_format.a_dtype.width // kernel_format.b_dtype.width
        config["STAGE_ALL_FC13"] = stage_all_fc13
        config["FC2_TILES_PER_SCHEDULE_ITEM"] = (
            1
            if stage_all_fc13
            else _fc2_tiles_per_schedule_item(
                max_tiles=2 * width_ratio,
                row_waves=row_waves,
            )
        )
        if not stage_all_fc13:
            config["PIPELINE_LEAD_CHUNKS"] = width_ratio
        wider_block_k = int(config["BLOCK_SIZE_K"]) * (
            kernel_format.a_dtype.width // kernel_format.b_dtype.width
        )
        if (
            stage_all_fc13
            and mixed_blockscaled_tile_prefers_deep_pipeline(
                a_width=kernel_format.a_dtype.width,
                b_width=kernel_format.b_dtype.width,
                block_n=int(config["BLOCK_SIZE_N"]),
                block_k=int(config["BLOCK_SIZE_K"]),
            )
            and hidden_dim % wider_block_k == 0
            and intermediate_dim % wider_block_k == 0
        ):
            config["BLOCK_SIZE_K"] = wider_block_k
        if config["FC2_TILES_PER_SCHEDULE_ITEM"] == 4:
            max_block_n = SM100_TMEM_COLS // int(config["NUM_CTAS"])
            config["BLOCK_SIZE_N"] = min(
                int(config["BLOCK_SIZE_N"]) * width_ratio,
                max_block_n,
            )
            config["MMA_ATOM_N"] = int(config["BLOCK_SIZE_N"])
            config["OVERLAPPING_ACCUM"] = True
        config["SWIGLU_VALUES_PER_THREAD"] = MEGA_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD
        # The heuristic reserves device tensormaps for the stage-all
        # geometry, but the activation ring requires them (ring-extent
        # descriptors) and only engages on the non-stage-all interleaved
        # schedule — without the override an explicit mixed-width (A8W4)
        # ring request would always fail the ring validation below.
        #
        # Only ever force them on, never off. Off returns all NaN on the
        # non-stage-all A8W4 configs: BLOCK_SIZE_N 256 would need 256-row
        # alignment for a host descriptor, but rows align to 128, so a host
        # descriptor cannot express the layout. `_host_tensormaps_available`
        # in `blockscaled_config.py` encodes that rule; the sibling launchers
        # consult it and this one does not.
        if activation_ring_chunks > 0 or (
            stage_all_fc13 and int(config["BLOCK_SIZE_N"]) >= 128
        ):
            use_device_tensormaps = True
    if _format_uses_fp4(weight_format):
        config.setdefault(
            "MMA_ATOM_N",
            _mega_fp4_mma_atom_n(
                config=config,
                format=weight_format,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                mixed_operand_widths=mixed_operand_widths,
            ),
        )
    if auto_config and _use_two_slot_schedule_ring(
        format=format,
        rows=rows,
        num_sms=num_sms,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    ):
        config["NUM_TILE_BUFFERS"] = 2
        if interleaved_fc13:
            config["PIPELINE_LEAD_CHUNKS"] = 8
    if auto_config and _use_t6_mxfp8_prefill_lead(
        format=format,
        rows=rows,
        num_sms=num_sms,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        interleaved_fc13=interleaved_fc13,
    ):
        config["PIPELINE_LEAD_CHUNKS"] = 4
    use_staged_swiglu_fragment = (
        auto_config
        and not mixed_operand_widths
        and _use_staged_swiglu_fragment(
            format=format,
            rows=rows,
            num_sms=num_sms,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            interleaved_fc13=interleaved_fc13,
        )
    )
    if use_staged_swiglu_fragment:
        config["PIPELINE_LEAD_CHUNKS"] = 4
    if format == MXFP4 and not interleaved_fc13:
        # Four FC13 chunks hide MXFP4 forward-quantization latency before FC2.
        config.setdefault("PIPELINE_LEAD_CHUNKS", 4)
    stage_all_fc13 = bool(
        config.setdefault(
            "STAGE_ALL_FC13",
            _stage_all_fc13(
                interleaved_fc13=interleaved_fc13,
                rows=rows,
                num_sms=num_sms,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                mixed_weight_format=mixed_operand_widths,
            ),
        )
    )
    if "FC2_TILES_PER_SCHEDULE_ITEM" not in config:
        config["FC2_TILES_PER_SCHEDULE_ITEM"] = (
            1 if stage_all_fc13 else 4 if auto_decode_config else 2
        )
    if borrow_slots:
        if config.get("USE_SM103_ULTRA"):
            raise base._MegaFusedUnsupportedError(
                "expert-borrow weight streaming is untested with SM103 ultra"
            )
        # The borrowed groups' A/SFA descriptors point at the slot buffer,
        # which only the device tensormap path can express.
        use_device_tensormaps = True
    nvfp4_high_throughput = _use_nvfp4_high_throughput_topology(
        format=format,
        rows=rows,
        num_sms=num_sms,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )
    wide_gather = _use_wide_gather_topology(
        format=format,
        rows=rows,
        num_sms=num_sms,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        interleaved_fc13=interleaved_fc13,
    )
    # This experimental override intentionally bypasses the validated heuristic.
    wide_gather = bool(config.pop("MEGA_WIDE_GATHER", wide_gather))
    nvfp4_wide_row_quant = _use_nvfp4_wide_row_quant(
        format=format,
        rows=rows,
        num_sms=num_sms,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )
    if interleaved_fc13 and activation_buffer is not None:
        if not use_device_tensormaps:
            raise base._MegaFusedUnsupportedError(
                "interleaved FC13 with an activation buffer requires device tensormaps"
            )
        if format is NVFP4 and not nvfp4_high_throughput:
            raise base._MegaFusedUnsupportedError(
                "interleaved NVFP4 with an activation buffer requires the "
                "high-throughput producer topology; this shape selects the "
                "narrow topology, which does not support buffer outputs yet"
            )
    config["STATIC_SCHEDULER"] = False
    config["NUM_CTAS"] = 2
    config["SWAP_AB"] = True
    config["COMBINE_FULL_HIDDEN_TILES"] = hidden_dim % config["BLOCK_SIZE_M"] == 0
    if bool(config.get("USE_SM103_ULTRA", False)):
        # The ultra pipeline depth is fixed by the segmented K=768 schedule;
        # NVFP4's mega default only worked here by coincidence (5 == the
        # ultra stage count) and MXFP4's (6) does not.
        config["NUM_SMEM_BUFFERS"] = sm103.SM103_AB_PIPELINE_STAGES
    else:
        default_smem_buffers = 5 if format is NVFP4 else 6
        config["NUM_SMEM_BUFFERS"] = int(
            config.get("MEGA_NUM_SMEM_BUFFERS", default_smem_buffers)
        )
    config["NUM_C_STAGES"] = 1
    if use_staged_swiglu_fragment:
        config["SWIGLU_VALUES_PER_THREAD"] = STAGED_BLOCKSCALED_SWIGLU_VALUES_PER_THREAD
    use_smaller_swiglu_epilogue = _use_smaller_swiglu_epilogue(
        format=format,
        rows=rows,
        num_sms=num_sms,
        interleaved_fc13=interleaved_fc13,
    )
    if config["BLOCK_SIZE_N"] >= 96 and config["BLOCK_SIZE_N"] % 128 != 0:
        config["EPILOGUE_SUBTILE"] = EPILOGUE_SUBTILE_AUTO
    else:
        epilogue_subtiles = max(
            1,
            config["BLOCK_SIZE_N"]
            // (128 if format is NVFP4 else 64 if use_smaller_swiglu_epilogue else 32),
        )
        config["EPILOGUE_SUBTILE"] = epilogue_subtiles & -epilogue_subtiles
        # Mega has SMEM headroom, so spend it on the full-width epilogue tile:
        # fewer passes and wider tcgen05.ld/stmatrix atoms.
        # MXFP4 at BN=32 is where the bytes_scale narrowing bites hardest: the
        # tile lands on an 8-column epilogue subtile, the narrowest the encoding
        # can express, which forces .x1 tcgen05.ld and .x2 stmatrix. Mega has the
        # SMEM headroom to pay for the full-width tile there, worth 3.7-4.5% at
        # 8-32 tokens/expert.
        #
        # Restricted to BN=32 because that is the only geometry measured, and
        # because wider tiles overrun SMEM: BN=128 at >=2048 tokens/rank asks for
        # 235520 bytes against the 232448-byte sm_103a limit. The mega estimator
        # does *not* predict that -- it reports 189440 for the same config -- so
        # the depth search cannot be relied on to back the tile off. Do not widen
        # this gate without both a measurement and an estimator that tracks the
        # real allocation.
        #
        # Measured neutral (<=0.7%, inside this bench's noise floor) for MXFP8 and
        # mixed A8W4, and 2-4% *slower* for NVFP4 whose C-stage growth costs a
        # mainloop stage. The staged path cannot afford it at all: it allocates
        # 280576 bytes against the same limit.
        config["EPILOGUE_TILE_FULL_WIDTH"] = (
            format is MXFP4 and int(config["BLOCK_SIZE_N"]) == 32
        )
    if auto_pipeline_depth:
        dispatch_quant_warps = 8 if wide_gather else 4
        (
            _,
            dispatch_quant_groups,
            nvfp4_row_reduction_slots,
        ) = _chunked_mega_producer_topology(
            format=format,
            dispatch_quant_warps=dispatch_quant_warps,
            nvfp4_high_throughput=nvfp4_high_throughput,
            nvfp4_wide_row_quant=nvfp4_wide_row_quant,
        )
        config = derive_chunked_mega_pipeline_config(
            config,
            format=kernel_format,
            groups=groups,
            dispatch_quant_groups=dispatch_quant_groups,
            nvfp4_row_reduction_slots=nvfp4_row_reduction_slots,
            use_global_scale_inv=use_global_scale_inv,
        )
    block_size_n = int(config["BLOCK_SIZE_N"])
    if chunk_rows is None:
        if mixed_operand_widths:
            chunk_rows = _derive_mixed_pipeline_chunk_rows(
                rows=rows,
                groups=groups,
                activation_block=block_size_n,
                feature_width=doubled_intermediate_dim,
                feature_block=int(config["BLOCK_SIZE_M"]),
                num_sms=num_sms,
                num_ctas=int(config["NUM_CTAS"]),
                activation_width=kernel_format.a_dtype.width,
                weight_width=kernel_format.b_dtype.width,
            )
        else:
            chunk_rows = _derive_pipeline_chunk_rows(
                rows=rows,
                groups=groups,
                activation_block=block_size_n,
            )
    _validate_pipeline_chunk_rows(chunk_rows)
    # The cross-group FC13->FC2 lead window is requested through the
    # `GLOBAL_LEAD_CHUNKS` config key or, composing with the auto config,
    # the peeled `MEGA_GLOBAL_LEAD_CHUNKS` overlay. Precedence: the overlay
    # wins over an explicit config key, which wins over the default policy
    # below (pass 0 to force the legacy per-group schedule). Trace-time;
    # keyed into the kernel and compile caches through the config dict.
    if mega_global_lead is not None:
        config["GLOBAL_LEAD_CHUNKS"] = int(mega_global_lead)
    elif "GLOBAL_LEAD_CHUNKS" not in config and interleaved_fc13 and not stage_all_fc13:
        # Default-on policy for interleaved prefill, validated at EP4/EP8/
        # EP32 (2026-08-30): W=8 wins every measured tier x format cell up
        # to 8 chunks/group and is >= parity at 8; above that the legacy
        # schedule already interleaves within groups, and 64K-token points
        # measured noise-to-slightly-negative, so keep legacy there. When
        # the activation ring is on, the window must stay strictly inside
        # the ring extent for the slot-reuse handshake to stay
        # deadlock-free.
        avg_chunks_per_group = rows / max(groups, 1) / chunk_rows
        if avg_chunks_per_group <= 8:
            default_lead = 8
            if activation_ring_chunks > 0:
                default_lead = min(default_lead, activation_ring_chunks - 1)
            if default_lead > 0:
                config["GLOBAL_LEAD_CHUNKS"] = default_lead
    if stage_all_fc13:
        # Stage-all configs (decode) subsume any lead window; the knob
        # targets the interleaved schedules, so drop it rather than reject a
        # config reused across regimes.
        config.pop("GLOBAL_LEAD_CHUNKS", None)
    if activation_ring_chunks > 0:
        ring_unsupported = None
        if not interleaved_fc13:
            ring_unsupported = (
                "the activation ring requires the interleaved (swizzled) FC13 "
                "epilogue: h2 must be produced in-epilogue"
            )
        elif format not in (MXFP8_E4M3, MXFP4, NVFP4) or (
            weight_format is not format and not _is_a8w4_format(format, weight_format)
        ):
            ring_unsupported = (
                "the activation ring supports same-width MXFP8_E4M3/MXFP4/"
                "NVFP4 and mixed A8W4 interleaved forwards"
            )
        elif stage_all_fc13:
            ring_unsupported = (
                "the activation ring requires the interleaved FC13/FC2 "
                "schedule; STAGE_ALL_FC13 keeps every chunk live at once"
            )
        elif not use_device_tensormaps:
            ring_unsupported = "the activation ring requires device tensormaps"
        elif activation_buffer is not None:
            ring_unsupported = (
                "the activation ring is not supported with an activation "
                "buffer: the buffer plan is capacity-shaped and does not "
                "model ring-extent slots"
            )
        if ring_unsupported is not None:
            # A positive ring request is always explicit (kwarg or config
            # key), so an ineligible path is an error, not a silent skip.
            raise base._MegaFusedUnsupportedError(ring_unsupported)
        elif activation_ring_chunks * chunk_rows >= rows:
            # A window covering every chunk is a full-extent buffer with ring
            # protocol overhead on top, so it normalizes to ring-off (waits,
            # credits, and the ring cache-modifier choice all disengage).
            # This is a per-rank decision: `rows` is the rank-local routed
            # row count, so a fixed request can engage on only the deeper
            # ranks. Logged rather than raised because that is expected.
            logger.debug(
                "activation ring request W=%d covers all %d rows "
                "(chunk_rows=%d); running with full-extent buffers, ring off",
                activation_ring_chunks,
                rows,
                chunk_rows,
            )
            activation_ring_chunks = 0
    if activation_ring_chunks:
        config["ACTIVATION_RING_CHUNKS"] = activation_ring_chunks
    else:
        config.pop("ACTIVATION_RING_CHUNKS", None)
    h2_alloc_rows = (
        activation_ring_chunks * chunk_rows if activation_ring_chunks else rows
    )
    row_scale_rows = _mega_row_scale_rows(
        rows=rows,
        groups=groups,
        block_size_n=block_size_n,
    )
    h2_row_scale_rows = (
        _mega_row_scale_rows(
            rows=h2_alloc_rows,
            groups=1,
            block_size_n=block_size_n,
        )
        if activation_ring_chunks
        else row_scale_rows
    )
    # The gathered x operand shares the h2 extents exactly: ring-off they are
    # both full extent, ring-on both shrink to the same W-chunk window.
    x_alloc_rows = h2_alloc_rows
    x_row_scale_rows = h2_row_scale_rows
    # One counter per schedule chunk for each of the three consumption
    # regions: FC2 (h2 ring), FC13 (x ring), and the NVFP4 row-quant
    # producer's `h1_done` (interleaved_h2 staging ring). Ceil-per-group
    # summed is bounded by the floor total plus one partial chunk per group.
    activation_ring_chunk_slots = rows // chunk_rows + groups
    activation_ring_counter_size = (
        3 * activation_ring_chunk_slots if activation_ring_chunks else 0
    )
    # The weight-borrow counters extend the activation slot past the ring
    # counters, so the prepare kernel's zeroing sweep and the kernel's
    # done-counter operand cover them without new plumbing.
    weight_borrow_counter_size = wborrow.weight_borrow_counter_size(borrow_slots)
    base._validate_mega_padded_scale_capacity(
        split_sizes,
        scale_rows=row_scale_rows,
        block_size_n=block_size_n,
    )
    _validate_interleaved_feature_tile(
        config,
        interleaved_fc13=interleaved_fc13,
    )
    if chunk_rows % config["BLOCK_SIZE_N"] != 0:
        raise ValueError(
            f"chunk_rows={chunk_rows} must be divisible by the activation "
            f"tile size BLOCK_SIZE_N={config['BLOCK_SIZE_N']}"
        )

    num_clusters = max(1, num_sms // config["NUM_CTAS"])
    input_plan = base._make_mega_dispatch_plan(
        split_sizes,
        rows=rows,
        dim=hidden_dim,
        format=format,
        config=config,
        require_full_act_tiles=False,
    )
    h2_plan = base._make_mega_forward_h2_plan(
        split_sizes,
        rows=rows,
        dim=intermediate_dim,
        format=format,
    )
    # The wgrad column-quantized copies exist only for callers that save them
    # for backward. Unrequested ones (every forward-only launch, interleaved
    # or not) compile the stores out of the producers, so their storages
    # shrink to dummies. Buffer-backed storages are zero-cost placeholders
    # and keep the full extent.
    x_col_rows = (
        None
        if return_x_wgrad_quant or activation_buffer is not None
        else _DUMMY_COL_ROWS
    )
    h2_col_rows = (
        None
        if return_h2_wgrad_quant or activation_buffer is not None
        else _DUMMY_COL_ROWS
    )
    (
        x_row_q,
        x_row_scale,
        x_col_quant,
        x_col_q_storage,
        x_col_scale_storage,
    ) = base._mega_bprop_quantized_operands(
        rows=x_alloc_rows,
        dim=hidden_dim,
        format=format,
        device=w13.device,
        activation_buffer=activation_buffer,
        row_scale_rows=x_row_scale_rows,
        col_rows=x_col_rows,
    )
    x_row_global_scale_inv = (
        torch.empty((rows,), dtype=torch.float32, device=w13.device)
        if use_global_scale_inv
        else None
    )
    (
        h2_row_q,
        h2_row_scale,
        h2_col_quant,
        h2_col_q_storage,
        h2_col_scale_storage,
    ) = base._mega_bprop_quantized_operands(
        rows=h2_alloc_rows,
        dim=intermediate_dim,
        format=format,
        device=w13.device,
        activation_buffer=activation_buffer,
        row_scale_rows=h2_row_scale_rows,
        col_rows=h2_col_rows,
    )
    h2_row_global_scale_inv = (
        torch.empty((rows,), dtype=torch.float32, device=w13.device)
        if use_global_scale_inv
        else None
    )
    split_size_multiple_of = (
        int(config["BLOCK_SIZE_N"])
        if bool(config["SWAP_AB"]) and int(config["BLOCK_SIZE_N"]) % 128 != 0
        else 128
    )
    split_size_alignment = split_size_multiple_of
    _validate_blockscaled_launch_inputs(
        a=x_row_q,
        b=w13,
        sfa=x_row_scale,
        sfb=w13_scale,
        split_sizes=split_sizes,
        format=format,
        weight_format=weight_format,
        split_size_multiple_of=split_size_multiple_of,
        split_size_alignment=split_size_alignment,
    )
    fc13_problem = _parse_blockscaled_problem(
        a=x_row_q,
        b=w13,
        format=format,
        weight_format=weight_format,
        split_sizes=split_sizes,
        contraction_axes=(1, 2),
        problem_type=_FPROP,
    )
    if borrow_slots:
        # The parser reads G off the local table; the compiled group count
        # covers the borrowed tail whose descriptors the prepare kernel
        # points at the slot buffer.
        fc13_problem = dataclasses.replace(fc13_problem, G=groups)
    if interleaved_fc13:
        # FC13 feeds row quantization directly; C is only a dynamic descriptor.
        h1 = torch.empty(
            (1, doubled_intermediate_dim), dtype=out_dtype, device=w13.device
        )
        # The MX formats quantize the SwiGLU output inside the FC13 epilogue
        # (their block scales are tile-local), so dense h2 is never
        # materialized. NVFP4's per-token global scale needs the full-row
        # amax, which one epilogue tile cannot see: the epilogue stages dense
        # h2 here and the NVFP4 row-quant producers re-read it to reduce and
        # quantize.
        interleaved_h2 = None
        if format is NVFP4:
            if activation_buffer is not None:
                # Buffer mode: the staging is a plan slot; the kernel rebases
                # both combine-source descriptors onto its offset. The ring is
                # rejected with a buffer, so the extent stays at full rows.
                interleaved_h2 = base._activation_buffer_placeholder(
                    activation_buffer,
                    (h2_alloc_rows, intermediate_dim),
                    out_dtype,
                )
            else:
                interleaved_h2 = torch.empty(
                    (h2_alloc_rows, intermediate_dim),
                    dtype=out_dtype,
                    device=w13.device,
                )
    else:
        h1_placeholder = (
            None
            if activation_buffer is None
            else base._activation_buffer_placeholder(
                activation_buffer,
                (rows, doubled_intermediate_dim),
                out_dtype,
            )
        )
        h1, _ = _resolve_output_tensor(
            c=h1_placeholder,
            c_shape=fc13_problem.c_shape,
            out_dtype=out_dtype,
            device=w13.device,
            problem_type=_FPROP,
            output_accum=False,
        )
        interleaved_h2 = None
    _validate_blockscaled_launch_inputs(
        a=h2_row_q,
        b=w2,
        sfa=h2_row_scale,
        sfb=w2_scale,
        split_sizes=split_sizes,
        format=format,
        weight_format=weight_format,
        split_size_multiple_of=split_size_multiple_of,
        split_size_alignment=split_size_alignment,
    )
    fc2_problem = _parse_blockscaled_problem(
        a=h2_row_q,
        b=w2,
        format=format,
        weight_format=weight_format,
        split_sizes=split_sizes,
        contraction_axes=(1, 2),
        problem_type=_FPROP,
    )
    if borrow_slots:
        fc2_problem = dataclasses.replace(fc2_problem, G=groups)
    # FC2 publishes through the combine scatter pointers and its C tensormap
    # is never stored through — only the FC13 consumer carries a C descriptor
    # — so C is a dynamic descriptor on every chunked path. Same dummy-extent
    # pattern as the interleaved `h1` above.
    h3_placeholder = torch.empty((1, hidden_dim), dtype=out_dtype, device=w13.device)
    pipeline_counter_size = _PIPELINE_COUNTER_SIZE
    if use_device_tensormaps:
        workspace, pipeline_counters = _allocate_pipeline_workspace(
            num_clusters=num_clusters,
            num_ctas=config["NUM_CTAS"],
            device=w13.device,
            input_plan=input_plan,
            activation_plan=h2_plan,
            pipeline_counter_size=pipeline_counter_size,
            min_tensormap_rows=groups,
            activation_ring_counter_size=(
                activation_ring_counter_size + weight_borrow_counter_size
            ),
        )
    else:
        workspace, pipeline_counters = _allocate_zeroed_pipeline_workspace(
            num_clusters=num_clusters,
            num_ctas=config["NUM_CTAS"],
            device=w13.device,
            input_plan=input_plan,
            activation_plan=h2_plan,
            pipeline_counter_size=pipeline_counter_size,
            min_tensormap_rows=groups,
            counter_storage=counter_storage,
            activation_ring_counter_size=(
                activation_ring_counter_size + weight_borrow_counter_size
            ),
        )
    world_size = symm_mem_buffer.hdl.world_size
    vector_dispatch_scale_copy = _use_vector_dispatch_scale_copy(
        format=format,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        wide_gather=wide_gather,
        rows=rows,
        num_sms=num_sms,
    )
    # Three consecutive ring done-counter regions (fc2, fc13, h1), mirroring
    # `activation_ring_counter_size = 3 * activation_ring_chunk_slots`.
    ring_done_base = h2_plan.done_counter_size if activation_ring_chunks else 0
    ring_done_stride = activation_ring_chunk_slots if activation_ring_chunks else 0
    # The weight-borrow work counter and per-slot done counters follow the
    # ring regions at the tail of the activation counter region.
    weight_borrow_work_offset = (
        h2_plan.done_counter_size + activation_ring_counter_size if borrow_slots else 0
    )
    weight_borrow_done_offset = weight_borrow_work_offset + 1 if borrow_slots else 0
    swiglu_clamp = canonical_swiglu_clamp(swiglu_clamped, swiglu_alpha, swiglu_limit)
    kernel = _get_chunked_kernel(
        config=config,
        format=kernel_format,
        world_size=world_size,
        swiglu_fast_math=swiglu_fast_math,
        swiglu_clamp=swiglu_clamp,
        chunk_rows=chunk_rows,
        quantize_dispatch=not blockscaled_dispatch,
        swiglu_row_quant_only=not return_h2_wgrad_quant,
        dispatch_col_quant=return_x_wgrad_quant,
        source_dtype=_dispatch_quant_source_dtype_from_torch(out_dtype),
        swiglu_k=intermediate_dim,
        nvfp4_high_throughput=nvfp4_high_throughput,
        nvfp4_wide_row_quant=nvfp4_wide_row_quant,
        vector_dispatch_scale_copy=vector_dispatch_scale_copy,
        # The narrower work items were measured with the vector scale copy; a
        # scalar-copy decode keeps the default work-item width.
        mx_narrow_copy_work=(
            vector_dispatch_scale_copy
            and format is not NVFP4
            and rows < num_sms * DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
        ),
        wide_gather=wide_gather,
        interleaved_fc13=interleaved_fc13,
        activation_ring_fc2_done_offset=ring_done_base,
        activation_ring_fc13_done_offset=ring_done_base + ring_done_stride,
        activation_ring_h1_done_offset=ring_done_base + 2 * ring_done_stride,
        weight_borrow_slots=borrow_slots,
        weight_borrow_work_offset=weight_borrow_work_offset,
        weight_borrow_done_offset=weight_borrow_done_offset,
    )
    # h2 row-quant producer sources: NVFP4 interleaved re-reads the staged
    # dense h2 (written through the uint32 view); the non-interleaved path
    # reads the real h1 gate/up halves. MX interleaved already quantized in
    # the epilogue, so its h1-dummy views are descriptor-only and never read.
    combine_sources = (
        (interleaved_h2.view(torch.uint32), interleaved_h2)
        if interleaved_h2 is not None
        else (h1[:, :intermediate_dim], h1[:, intermediate_dim:])
    )
    (
        compile_args,
        runtime_args,
        fc13_elem_sizes,
        fc2_elem_sizes,
        placeholders,
        use_activation_buffer,
        use_conditional_execution,
        condition_owner,
    ) = base._make_mega_two_gemm_kernel_args(
        dgrad_tensors=(x_row_q, w13, h1, x_row_scale, w13_scale),
        wgrad_tensors=(h2_row_q, w2, h3_placeholder, h2_row_scale, w2_scale),
        split_sizes=split_sizes,
        workspace=workspace,
        route_ptrs=gather_ptrs[:rows],
        combine_route_ptrs=scatter_ptrs[:rows],
        combine_sources=combine_sources,
        col_q_storage=x_col_q_storage,
        col_scale_storage=x_col_scale_storage,
        activation_quant=(
            h3_placeholder.view(torch.uint32),
            h2_row_q,
            h2_row_scale,
            h2_col_q_storage,
            h2_col_scale_storage,
            h2_plan,
        ),
        dgrad_problem=fc13_problem,
        wgrad_problem=fc2_problem,
        dgrad_out_dtype=out_dtype,
        wgrad_out_dtype=out_dtype,
        format=format,
        dgrad_weight_format=weight_format,
        wgrad_weight_format=weight_format,
        output_accum=False,
        symm_mem_buffer=symm_mem_buffer,
        num_clusters=num_clusters,
        plan=input_plan,
        kernel_mnk=(rows, intermediate_dim, hidden_dim),
        use_device_tensormaps=use_device_tensormaps,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        dispatch_row_global_scale_inv=(
            x_row_global_scale_inv if use_global_scale_inv else None
        ),
        activation_row_global_scale_inv=(
            h2_row_global_scale_inv if use_global_scale_inv else None
        ),
        dgrad_b_global_scale_inv=w13_global_scale_inv,
        wgrad_b_global_scale_inv=w2_global_scale_inv,
        nvfp4_recip_lut=(
            get_nvfp4_recip_lut(w13.device) if use_global_scale_inv else None
        ),
    )
    cache_key = (
        _config_cache_key(config),
        kernel_format.name,
        world_size,
        chunk_rows,
        *(_torch_layout_signature(tensor) for tensor in placeholders),
        rows,
        _torch_layout_signature(h2_row_q),
        _torch_layout_signature(w2),
        _torch_layout_signature(h3_placeholder),
        fc13_problem.G,
        fc13_problem.N,
        fc13_problem.K,
        fc2_problem.N,
        fc2_problem.K,
        fc13_elem_sizes,
        fc2_elem_sizes,
        swiglu_fast_math,
        swiglu_clamp,
        blockscaled_dispatch,
        nvfp4_high_throughput,
        wide_gather,
        return_x_wgrad_quant,
        return_h2_wgrad_quant,
        interleaved_fc13,
        use_device_tensormaps,
        use_activation_buffer,
        use_conditional_execution,
        "chunked_mega_forward",
    )
    if borrow_slots:
        wb_launch = weight_borrow.launch_args(
            enable_tvm_ffi=_format_uses_fp4(kernel_format)
        )
        compile_args = (*compile_args, *wb_launch)
        runtime_args = (*runtime_args, *wb_launch)
        cache_key = (
            *cache_key,
            "weight_borrow",
            borrow_slots,
            weight_borrow_work_offset,
            weight_borrow_done_offset,
        )
    t6_mxfp4_training = (
        explicit_training_config
        and format is MXFP4
        and hidden_dim == 12288
        and intermediate_dim == 3072
    )
    compiler_opt_level = config.get("CUTE_DSL_OPT_LEVEL")
    compile_options = (
        f"--opt-level={compiler_opt_level}"
        if compiler_opt_level is not None
        else "--opt-level=1"
        if format is NVFP4 or t6_mxfp4_training
        else None
    )
    cache_key = (*cache_key, compile_options)
    compiled = base._compile_or_get(
        cache_key,
        kernel,
        *compile_args,
        name_prefix=("_cute_chunked_mega_blockscaled_grouped_gemm_fprop_swiglu_fwd"),
        options=compile_options,
    )
    if auto_pipeline_depth:
        _verify_chunked_mega_smem_estimate(
            kernel,
            config,
            format=kernel_format,
            groups=groups,
            use_global_scale_inv=use_global_scale_inv,
        )
    compiled(*runtime_args)
    if condition_owner is not None:
        condition_owner.record_stream(torch.cuda.current_stream(condition_owner.device))
    h1_result = None if interleaved_fc13 else h1
    result = (
        _maybe_slice_output(symm_mem_buffer.local(), num_output_tokens),
        h1_result,
        h2_col_quant if return_h2_wgrad_quant else None,
    )
    if return_x_wgrad_quant:
        result = (*result, x_col_quant)
    if return_row_quant:
        # NVFP4 rescales each token before block quantization, so its per-token
        # inverse global scale belongs to the bundle; without it a
        # dequantization of the pair is off by that factor.
        if use_global_scale_inv:
            result = (
                *result,
                (x_row_q, x_row_scale, x_row_global_scale_inv),
                (h2_row_q, h2_row_scale, h2_row_global_scale_inv),
            )
        else:
            result = (*result, (x_row_q, x_row_scale), (h2_row_q, h2_row_scale))
    return result
