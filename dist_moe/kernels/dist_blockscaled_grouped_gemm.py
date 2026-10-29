# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed block-scaled grouped GEMM helpers for dist_moe.

Dispatch publishes the local high-precision activation to symmetric memory,
uses the fused zerocopy-gather quantization producer to materialize the
block-scaled A operand, then runs the CuTe block-scaled grouped GEMM.
Combine wraps the CuTe block-scaled grouped GEMM mainloop and scatters the
bf16 output rows from the epilogue warpgroup through the zerocopy pointer
table.

The launchers do not synchronize expert-parallel peers; callers own the
required barriers and pre-stage every symmetric-memory payload.
"""

import logging

import cutlass
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

from ..formats import (
    SITU_GLU_GATE_BETA_DEFAULT,
    SITU_GLU_UP_BETA_DEFAULT,
    SWIGLU_CLAMP_ALPHA_DEFAULT,
    SWIGLU_CLAMP_LIMIT_DEFAULT,
)
from ._environment import num_sms_per_device
from .activation_buffer import (
    _qdata_words,
    _row_quant_tuple,
    activation_buffer_placeholder as _activation_buffer_placeholder,
    ACTIVATION_OFFSET_COUNT,
)
from .activation_buffer_kernel import (
    _column_quantized_output,
    _column_quantized_scale_shape,
    _column_quantized_storage_shape,
    _compact_scale_storage_rows,
    _dispatch_quant_source_dtype_from_torch,
    _dispatch_quantized_operand_placeholder,
    _empty_dispatch_quantized_operand,
    _empty_qdata,
    _prepare_activation_buffer_launch_args,
)
from .blockscaled_grouped_gemm import (
    _allocate_workspace,
    _compile_or_get,
    _config_cache_key,
    _format_uses_fp4,
    _make_launch_tensor_bundle,
    _make_pointer_stride_args,
    _parse_blockscaled_problem,
    _resolve_output_tensor,
    _set_situ_glu_config,
    _set_swiglu_clamp_config,
    _torch_dtype,
    _torch_layout_signature,
    _validate_2d_operand_layout,
    _validate_3d_operand_layout,
    _validate_blockscaled_launch_inputs,
    blockscaled_mega_decode_config,
    blockscaled_staged_decode_config,
    blockscaled_training_config,
    BlockScaledFormatSpec,
    make_mixed_blockscaled_format,
    MXFP4,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
)
from .blockscaled_quantize import get_nvfp4_recip_lut
from .config import (
    BLOCKSCALED_DISPATCH_DIM_ALIGNMENT,
    DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
)
from .dist_blockscaled_config import (
    _blockscaled_split_size_alignment,
    _can_use_fused_combine_swiglu_quant,
    _can_use_fused_dispatch_quant,
    _dispatch_scale_copy_atom_aligned,
    _make_dist_combine_config,
    _select_blockscaled_config,
    _staged_fp4_mma_atom_n,
    _staged_use_device_tensormaps,
    _use_nvfp4_high_throughput_producers,
    _use_t6_mxfp8_prefill_fc13_unroll,
    _uses_derived_staged_pipeline,
    _validate_swap_ab_row_multiple,
    _verify_staged_smem_estimate,
    derive_staged_blockscaled_pipeline_config as derive_staged_blockscaled_pipeline_config,
)
from .dist_blockscaled_grouped_gemm_kernel import (
    _FP4_FORMAT_NAMES,
    DistBlockScaledGroupedGemmKernel,
)
from .grouped_gemm import _DGRAD, _FPROP

logger: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "MXFP4",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "NVFP4",
    "blockscaled_mega_decode_config",
    "blockscaled_staged_decode_config",
    "blockscaled_training_config",
    "derive_staged_blockscaled_pipeline_config",
    "dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine",
    "dist_blockscaled_grouped_gemm_dgrad_dispatch",
    "dist_blockscaled_grouped_gemm_fprop_dispatch",
    "dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine",
]

_DEFAULT_M_MULTIPLE_OF: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
_MXFP8_FORMAT_NAMES: frozenset[str] = frozenset({"mxfp8_e4m3", "mxfp8_e5m2"})
_INT32_MAX = 2**31 - 1


def _flatten_last_dim(t: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    if t.shape[-1] != dim:
        raise ValueError(f"{name}.shape[-1] must be {dim}, got {t.shape[-1]}")
    try:
        return t.view(-1, dim)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} must be viewable as a 2D tensor with trailing dim {dim}; "
            f"got shape={tuple(t.shape)}, stride={tuple(t.stride())}"
        ) from exc


def _logical_dim_size(storage_size: int, format: BlockScaledFormatSpec) -> int:
    return storage_size * 2 if format.name in _FP4_FORMAT_NAMES else storage_size


def _allocate_dispatch_counter_workspace(
    num_clusters: int,
    NUM_CTAS: int,
    device: torch.device,
    *,
    done_counter_size: int,
    min_rows: int = 0,
    zero_in_prepare: bool = False,
    counter_storage: torch.Tensor | None = None,
):
    grid_size = max(num_clusters * NUM_CTAS, min_rows)
    counter_splits = _dispatch_counter_storage_splits(done_counter_size)
    required_counter_count = sum(counter_splits)
    use_precleared_storage = counter_storage is not None
    if counter_storage is None:
        counter_storage_fn = torch.empty if zero_in_prepare else torch.zeros
        resolved_counter_storage = counter_storage_fn(
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
    counter, dispatch_quant_work_counter, dispatch_quant_done_counter = torch.split(
        resolved_counter_storage,
        counter_splits,
    )
    tensormaps = torch.empty((grid_size, 5, 16), dtype=torch.int64, device=device)
    extra_counter_zero_count = (
        sum(counter_splits[1:]) if zero_in_prepare and not use_precleared_storage else 0
    )
    return (
        counter,
        tensormaps,
        dispatch_quant_work_counter,
        dispatch_quant_done_counter,
        extra_counter_zero_count,
    )


def _dispatch_counter_storage_splits(done_counter_size: int) -> tuple[int, int, int]:
    return (1, 1, done_counter_size)


def _validate_precleared_counter_storage(
    counter_storage: torch.Tensor,
    *,
    required_counter_count: int,
    device: torch.device,
) -> None:
    if (
        counter_storage.dtype != torch.int32
        or counter_storage.device != device
        or not counter_storage.is_contiguous()
        or counter_storage.numel() < required_counter_count
    ):
        raise ValueError(
            "counter_storage must be a contiguous int32 tensor on the launch "
            f"device with at least {required_counter_count} elements"
        )


def allocate_decode_counter_storage(
    *,
    rows: int,
    format: BlockScaledFormatSpec,
    device: torch.device,
) -> torch.Tensor:
    """Allocate counters that the decode routing kernel clears before FPROP."""
    done_counter_size = max(1, rows // format.sf_vec_size)
    return torch.empty(
        sum(_dispatch_counter_storage_splits(done_counter_size)),
        dtype=torch.int32,
        device=device,
    )


def _q_words_tensor(words: torch.Tensor, *, wide: bool, format: BlockScaledFormatSpec):
    tensor = from_dlpack(
        words.detach(), assumed_align=16, enable_tvm_ffi=_format_uses_fp4(format)
    )
    # Quant producers index q-words with Int32 rows; past 2**31 words the row
    # stride must be dynamic (Int64) so row * stride cannot wrap. Smaller buffers
    # keep the static layout, which is cheaper in the register-bound fused kernels.
    return tensor.mark_layout_dynamic(leading_dim=1) if wide else tensor


def _maybe_slice_output(output: torch.Tensor, num_output_tokens: int | None):
    if num_output_tokens is None:
        return output
    if output.shape[0] < num_output_tokens:
        raise ValueError(
            f"output has only {output.shape[0]} rows, "
            f"cannot slice to num_output_tokens={num_output_tokens}"
        )
    return output[:num_output_tokens]


def _resolve_combine_descriptor_tensor(
    *,
    c: torch.Tensor | None,
    c_shape: tuple[int, ...],
    out_dtype: torch.dtype | None,
    device: torch.device,
    problem_type: int,
    config: dict,
) -> tuple[torch.Tensor, torch.dtype]:
    """Return the C tensor used only for CuTe epilogue descriptor metadata.

    Fused COMBINE scatters directly to peer buffers and never stores to C.
    The tensor is still needed to build the C-layout/TMA objects that drive
    accumulator partitioning, so the default path allocates one row tile of
    descriptor scratch instead of a full GM x N output.
    """
    if c is not None:
        return _resolve_output_tensor(
            c=c,
            c_shape=c_shape,
            out_dtype=out_dtype,
            device=device,
            problem_type=problem_type,
            output_accum=False,
        )
    if len(c_shape) != 2:
        raise ValueError(f"fused COMBINE expects a 2D C shape, got {c_shape}")
    if out_dtype is None:
        out_dtype = torch.bfloat16
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError(f"out_dtype={out_dtype} not supported.")
    if out_dtype == torch.float32:
        raise NotImplementedError("float32 output is only supported for WGRAD.")

    row_tile = int(
        config["BLOCK_SIZE_N"]
        if config.get("SWAP_AB", False)
        else config["BLOCK_SIZE_M"]
    )
    scratch_rows = max(1, min(int(c_shape[0]), row_tile))
    return torch.empty(
        (scratch_rows, int(c_shape[1])),
        dtype=out_dtype,
        device=device,
    ), out_dtype


_DIST_BLOCKSCALED_KERNEL_CACHE: dict[tuple, DistBlockScaledGroupedGemmKernel] = {}


def _assert_split_sizes_fit_capacity(
    split_sizes: torch.Tensor,
    *,
    rows: int,
    multiple_of: int,
) -> None:
    if split_sizes.is_cuda:
        # Routing produces capacity-bounded, aligned CUDA splits; avoid adding
        # device assertion kernels to the captured steady-state graph.
        return
    if bool((split_sizes < 0).any()):
        raise RuntimeError("split sizes must be nonnegative")
    if bool((split_sizes % multiple_of != 0).any()):
        raise RuntimeError("split sizes must satisfy the grouped-GEMM row alignment")
    if int(split_sizes.sum(dtype=torch.int64)) > rows:
        raise RuntimeError("split sizes exceed the row capacity")


def _validate_global_scale_inv(
    scale: torch.Tensor | None,
    *,
    size: int,
    device: torch.device,
    name: str,
) -> None:
    if (
        scale is None
        or scale.shape != (size,)
        or scale.dtype != torch.float32
        or scale.device != device
        or not scale.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous float32 with shape ({size},) on {device}"
        )


def _get_dist_blockscaled_kernel(
    *,
    config: dict,
    problem_type: int,
    mode: int,
    format: BlockScaledFormatSpec,
    force_n_major: bool,
    num_n_clusters: int,
    world_size: int,
    dispatch_source_dtype: type[cutlass.Numeric] | None = None,
    blockscaled_dispatch: bool = False,
    nvfp4_high_throughput: bool = False,
    combine_swiglu_k: int | None = None,
    interleaved_fc13: bool = False,
    vector_dispatch_scale_copy: bool = False,
    schedule_rotation: bool = True,
) -> DistBlockScaledGroupedGemmKernel:
    nvfp4_high_throughput = nvfp4_high_throughput and (
        blockscaled_dispatch
        or mode == DistBlockScaledGroupedGemmKernel.COMBINE_SWIGLU_FWD_MODE
    )
    key = (
        _config_cache_key(config),
        problem_type,
        mode,
        format.name,
        force_n_major,
        num_n_clusters,
        world_size,
        dispatch_source_dtype,
        blockscaled_dispatch,
        nvfp4_high_throughput,
        combine_swiglu_k,
        interleaved_fc13,
        vector_dispatch_scale_copy,
        schedule_rotation,
    )
    inst = _DIST_BLOCKSCALED_KERNEL_CACHE.get(key)
    if inst is None:
        inst = DistBlockScaledGroupedGemmKernel.from_config(
            config,
            mode=mode,
            problem_type=problem_type,
            format=format,
            force_n_major=force_n_major,
            num_n_clusters=num_n_clusters,
            world_size=world_size,
            dispatch_source_dtype=dispatch_source_dtype,
            blockscaled_dispatch=blockscaled_dispatch,
            nvfp4_high_throughput=nvfp4_high_throughput,
            combine_swiglu_k=combine_swiglu_k,
            interleaved_fc13=interleaved_fc13,
            schedule_rotation=schedule_rotation,
        )
        if vector_dispatch_scale_copy:
            inst.VECTOR_DISPATCH_SCALE_COPY = True
        _DIST_BLOCKSCALED_KERNEL_CACHE[key] = inst
    return inst


def _fused_dist_blockscaled_grouped_gemm_combine_impl(  # noqa: C901
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
    problem_type: int,
    contraction_axes: tuple[int, int],
    c: torch.Tensor | None,
    out_dtype: torch.dtype,
    num_sms: int | None,
    config: dict | None,
    m_multiple_of: int,
    num_output_tokens: int | None,
    combine_swiglu_x: torch.Tensor | None = None,
    combine_swiglu_y: torch.Tensor | None = None,
    combine_swiglu_col_q_storage: torch.Tensor | None = None,
    combine_swiglu_col_scale: torch.Tensor | None = None,
    combine_swiglu_bwd_dz: torch.Tensor | None = None,
    combine_swiglu_bwd_h1: torch.Tensor | None = None,
    combine_swiglu_fast_math: bool = False,
    combine_swiglu_clamped: bool = False,
    combine_swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    combine_swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    combine_situ_glu: bool = False,
    combine_situ_glu_gate_beta: float = SITU_GLU_GATE_BETA_DEFAULT,
    combine_situ_glu_up_beta: float = SITU_GLU_UP_BETA_DEFAULT,
    combine_swiglu_row_quant_only: bool = False,
    combine_precomputed_swiglu: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    counter_storage: torch.Tensor | None = None,
    a_global_scale_inv: torch.Tensor | None = None,
    b_global_scale_inv: torch.Tensor | None = None,
    derive_pipeline_depth: bool = False,
    pre_shuffled_rows: bool = False,
) -> torch.Tensor:
    weight_format = format if weight_format is None else weight_format
    kernel_format = make_mixed_blockscaled_format(format, weight_format)
    if scatter_ptrs.dtype != torch.int64:
        raise TypeError("scatter_ptrs must be int64")
    if not scatter_ptrs.is_cuda or scatter_ptrs.device != a.device:
        raise ValueError("scatter_ptrs must be on a.device")
    use_combine_swiglu_fwd = (
        combine_swiglu_x is not None or combine_swiglu_y is not None
    )
    use_combine_swiglu_bwd = (
        combine_swiglu_bwd_dz is not None or combine_swiglu_bwd_h1 is not None
    )
    if use_combine_swiglu_fwd and use_combine_swiglu_bwd:
        raise ValueError("combine swiglu fwd and bwd fusions are mutually exclusive")
    use_combine_swiglu = use_combine_swiglu_fwd or use_combine_swiglu_bwd
    if use_combine_swiglu_fwd:
        if combine_swiglu_x is None or combine_swiglu_y is None:
            raise ValueError(
                "combine_swiglu_x and combine_swiglu_y must be provided together"
            )
        if not combine_swiglu_row_quant_only and (
            combine_swiglu_col_q_storage is None or combine_swiglu_col_scale is None
        ):
            raise ValueError(
                "combine swiglu fusion requires column quant output storage"
            )
        if combine_swiglu_x.shape != combine_swiglu_y.shape:
            raise ValueError(
                "combine_swiglu_x and combine_swiglu_y must have the same shape; "
                f"got {tuple(combine_swiglu_x.shape)} and {tuple(combine_swiglu_y.shape)}"
            )
        if combine_swiglu_x.dtype != combine_swiglu_y.dtype:
            raise TypeError(
                "combine_swiglu_x and combine_swiglu_y must have the same dtype"
            )
        if combine_swiglu_x.device != a.device or combine_swiglu_y.device != a.device:
            raise ValueError("combine_swiglu_x/y must be on a.device")
        if combine_precomputed_swiglu and format is not NVFP4:
            raise NotImplementedError(
                "precomputed SwiGLU combine currently supports NVFP4 only"
            )
    elif combine_precomputed_swiglu:
        raise ValueError("precomputed SwiGLU requires the fused forward combine")
    if use_combine_swiglu_bwd:
        if combine_swiglu_bwd_dz is None or combine_swiglu_bwd_h1 is None:
            raise ValueError(
                "combine_swiglu_bwd_dz and combine_swiglu_bwd_h1 must be provided together"
            )
        if combine_swiglu_row_quant_only:
            raise ValueError("combine swiglu bwd fusion always produces column quant")
        if combine_swiglu_col_q_storage is None or combine_swiglu_col_scale is None:
            raise ValueError(
                "combine swiglu bwd fusion requires dxy column output storage"
            )
        if combine_swiglu_bwd_dz.dtype != combine_swiglu_bwd_h1.dtype:
            raise TypeError(
                "combine_swiglu_bwd_dz and combine_swiglu_bwd_h1 must have the same dtype"
            )
        if (
            combine_swiglu_bwd_dz.device != a.device
            or combine_swiglu_bwd_h1.device != a.device
        ):
            raise ValueError("combine_swiglu_bwd_dz/h1 must be on a.device")

    _validate_blockscaled_launch_inputs(
        a=a,
        b=b,
        sfa=sfa,
        sfb=sfb,
        split_sizes=split_sizes,
        format=format,
        weight_format=weight_format,
        split_size_multiple_of=m_multiple_of,
        split_size_alignment=_blockscaled_split_size_alignment(m_multiple_of),
    )
    problem = _parse_blockscaled_problem(
        a=a,
        b=b,
        format=format,
        weight_format=weight_format,
        split_sizes=split_sizes,
        contraction_axes=contraction_axes,
        problem_type=problem_type,
    )
    if use_combine_swiglu:
        _assert_split_sizes_fit_capacity(
            split_sizes,
            rows=problem.GM,
            multiple_of=m_multiple_of,
        )
    if use_combine_swiglu_fwd:
        assert combine_swiglu_x is not None
        assert combine_swiglu_y is not None
        if combine_swiglu_x.shape != (problem.GM, problem.K):
            raise ValueError(
                "combine_swiglu_x/y must have shape (GM, K) for FC2 input; "
                f"expected {(problem.GM, problem.K)}, got {tuple(combine_swiglu_x.shape)}"
            )
        if not combine_swiglu_row_quant_only:
            assert combine_swiglu_col_q_storage is not None
            assert combine_swiglu_col_scale is not None
            expected_col_q_shape = _column_quantized_storage_shape(
                problem.GM, problem.K, format
            )
            if combine_swiglu_col_q_storage.shape != expected_col_q_shape:
                raise ValueError(
                    "combine swiglu column qdata storage must have shape (GM, K); "
                    f"expected {expected_col_q_shape}, got {tuple(combine_swiglu_col_q_storage.shape)}"
                )
            expected_col_scale_shape = _column_quantized_scale_shape(
                problem.GM, problem.K, format
            )
            if combine_swiglu_col_scale.shape != expected_col_scale_shape:
                raise ValueError(
                    "combine swiglu column scale storage has wrong shape; "
                    f"expected {expected_col_scale_shape}, got {tuple(combine_swiglu_col_scale.shape)}"
                )
    if use_combine_swiglu_bwd:
        assert combine_swiglu_bwd_dz is not None
        assert combine_swiglu_bwd_h1 is not None
        assert combine_swiglu_col_q_storage is not None
        assert combine_swiglu_col_scale is not None
        if problem_type != _DGRAD:
            raise ValueError("combine swiglu bwd fusion only supports DGRAD combine")
        if problem.K % 2 != 0:
            raise ValueError(
                f"combine swiglu bwd requires even DXY dim, got {problem.K}"
            )
        source_K = problem.K // 2
        if combine_swiglu_bwd_dz.shape != (problem.GM, source_K):
            raise ValueError(
                "combine_swiglu_bwd_dz must have shape (GM, K/2); "
                f"expected {(problem.GM, source_K)}, got {tuple(combine_swiglu_bwd_dz.shape)}"
            )
        if combine_swiglu_bwd_h1.shape != (problem.GM, problem.K):
            raise ValueError(
                "combine_swiglu_bwd_h1 must have shape (GM, K); "
                f"expected {(problem.GM, problem.K)}, got {tuple(combine_swiglu_bwd_h1.shape)}"
            )
        expected_col_q_shape = _column_quantized_storage_shape(
            problem.GM, problem.K, format
        )
        if combine_swiglu_col_q_storage.shape != expected_col_q_shape:
            raise ValueError(
                "combine swiglu bwd DXY column qdata storage must have shape (GM, K); "
                f"expected {expected_col_q_shape}, got {tuple(combine_swiglu_col_q_storage.shape)}"
            )
        expected_dxy_col_scale_shape = _column_quantized_scale_shape(
            problem.GM, problem.K, format
        )
        if combine_swiglu_col_scale.shape != expected_dxy_col_scale_shape:
            raise ValueError(
                "combine swiglu bwd DXY column scale storage has wrong shape; "
                f"expected {expected_dxy_col_scale_shape}, got {tuple(combine_swiglu_col_scale.shape)}"
            )
    use_global_scale_inv = (
        a_global_scale_inv is not None or b_global_scale_inv is not None
    )
    if use_global_scale_inv:
        if format is not NVFP4 or problem_type != _FPROP:
            raise ValueError("epilogue global scaling requires NVFP4 FPROP")
        _validate_global_scale_inv(
            a_global_scale_inv,
            size=problem.GM,
            device=a.device,
            name="a_global_scale_inv",
        )
        _validate_global_scale_inv(
            b_global_scale_inv,
            size=problem.G,
            device=a.device,
            name="b_global_scale_inv",
        )
    if scatter_ptrs.numel() < problem.GM:
        raise ValueError(
            f"scatter_ptrs has {scatter_ptrs.numel()} entries, "
            f"but GEMM output has {problem.GM} rows"
        )
    if problem.N % 8 != 0:
        raise ValueError(
            f"fused blockscaled COMBINE requires output dim {problem.N} "
            "to be divisible by 8 for 16-byte vector scatter"
        )
    if split_sizes.dtype != torch.int32:
        split_sizes = split_sizes.to(torch.int32)
    if num_sms is None:
        num_sms = num_sms_per_device()

    auto_selected_config = config is None
    if auto_selected_config:
        config = _select_blockscaled_config(
            problem=problem,
            num_sms=num_sms,
            format=format,
            weight_format=weight_format,
            m_multiple_of=m_multiple_of,
            problem_type=problem_type,
            activation_buffer=activation_buffer,
            derive_pipeline_depth=derive_pipeline_depth,
        )
    c_stage_dtype = c.dtype if c is not None else (out_dtype or torch.bfloat16)
    cfg = _make_dist_combine_config(
        config=config,
        format=kernel_format,
        auto_selected=auto_selected_config,
        c_stage_dtype=c_stage_dtype,
        num_c_stages=DistBlockScaledGroupedGemmKernel.COMBINE_C_STAGES,
    )
    _validate_swap_ab_row_multiple(cfg, m_multiple_of)
    auto_pipeline_depth = derive_pipeline_depth and _uses_derived_staged_pipeline(
        cfg,
        format=format,
        weight_format=weight_format,
    )
    if _format_uses_fp4(weight_format):
        cfg.setdefault(
            "MMA_ATOM_N",
            _staged_fp4_mma_atom_n(
                config=cfg,
                format=weight_format,
                n=problem.N,
                k=problem.K,
                mixed_operand_widths=format is not weight_format,
            ),
        )
    if use_combine_swiglu and combine_swiglu_fast_math:
        cfg["COMBINE_SWIGLU_FAST_MATH"] = True
    if use_combine_swiglu:
        _set_swiglu_clamp_config(
            cfg, combine_swiglu_clamped, combine_swiglu_alpha, combine_swiglu_limit
        )
        _set_situ_glu_config(
            cfg,
            combine_situ_glu,
            combine_situ_glu_gate_beta,
            combine_situ_glu_up_beta,
        )
    if use_combine_swiglu_fwd and combine_swiglu_row_quant_only:
        cfg["COMBINE_SWIGLU_ROW_QUANT_ONLY"] = True
    if combine_precomputed_swiglu:
        cfg["COMBINE_PRECOMPUTED_SWIGLU"] = True
    c, out_dtype = _resolve_combine_descriptor_tensor(
        c=c,
        c_shape=problem.c_shape,
        out_dtype=out_dtype,
        device=a.device,
        problem_type=problem_type,
        config=cfg,
    )
    use_device_tensormaps = _staged_use_device_tensormaps(
        cfg,
        activation_buffer=activation_buffer,
        problem_type=problem_type,
        m_multiple_of=m_multiple_of,
        n=problem.N,
        groups=problem.G,
    )
    num_clusters = max(1, num_sms // cfg["NUM_CTAS"])

    local_rank = symm_mem_buffer.hdl.rank
    world_size = symm_mem_buffer.hdl.world_size
    force_n_major = False
    num_n_clusters = 1
    if use_combine_swiglu_bwd:
        mode = DistBlockScaledGroupedGemmKernel.COMBINE_SWIGLU_BWD_MODE
        assert combine_swiglu_bwd_dz is not None
        dispatch_source_dtype = _dispatch_quant_source_dtype_from_torch(
            combine_swiglu_bwd_dz.dtype
        )
    elif use_combine_swiglu_fwd:
        mode = DistBlockScaledGroupedGemmKernel.COMBINE_SWIGLU_FWD_MODE
        assert combine_swiglu_x is not None
        dispatch_source_dtype = _dispatch_quant_source_dtype_from_torch(
            combine_swiglu_x.dtype
        )
    else:
        mode = DistBlockScaledGroupedGemmKernel.COMBINE_MODE
        dispatch_source_dtype = None
    nvfp4_high_throughput = _use_nvfp4_high_throughput_producers(
        format=format,
        rows=problem.GM,
        num_sms=num_sms,
        hidden_dim=problem.N,
    )
    combine_swiglu_k = (
        int(combine_swiglu_x.shape[-1]) if use_combine_swiglu_fwd else None
    )
    if auto_pipeline_depth:
        combine_c_stages = (
            1 if format == MXFP8_E4M3 and weight_format == MXFP4 else None
        )
        cfg = derive_staged_blockscaled_pipeline_config(
            cfg,
            format=kernel_format,
            activation_format=format,
            weight_format=weight_format,
            num_c_stages=combine_c_stages,
            groups=problem.G,
            mode=mode,
            blockscaled_dispatch=False,
            nvfp4_high_throughput=nvfp4_high_throughput,
            use_global_scale_inv=use_global_scale_inv,
            combine_swiglu_k=combine_swiglu_k,
            c_stage_dtype=c_stage_dtype,
        )
    kernel = _get_dist_blockscaled_kernel(
        config=cfg,
        problem_type=problem_type,
        mode=mode,
        format=kernel_format,
        force_n_major=force_n_major,
        num_n_clusters=num_n_clusters,
        world_size=world_size,
        dispatch_source_dtype=dispatch_source_dtype,
        nvfp4_high_throughput=nvfp4_high_throughput,
        combine_swiglu_k=combine_swiglu_k,
        schedule_rotation=not pre_shuffled_rows,
    )

    # Quant producers derive their schedule from the runtime split tensor.
    dispatch_quant_total_tiles = 0
    dispatch_quant_counter_zero_count = 0
    if use_combine_swiglu:
        dispatch_quant_done_counter_size = max(1, problem.GM // format.sf_vec_size)
        (
            counter,
            tensormaps,
            dispatch_quant_work_counter,
            dispatch_quant_done_counter,
            dispatch_quant_counter_zero_count,
        ) = _allocate_dispatch_counter_workspace(
            num_clusters,
            cfg["NUM_CTAS"],
            a.device,
            done_counter_size=dispatch_quant_done_counter_size,
            min_rows=problem.G if use_device_tensormaps else 0,
            zero_in_prepare=use_device_tensormaps,
            counter_storage=counter_storage,
        )
    else:
        counter, tensormaps = _allocate_workspace(
            num_clusters,
            cfg["NUM_CTAS"],
            a.device,
            static_scheduler=bool(cfg.get("STATIC_SCHEDULER", False)),
            min_rows=problem.G if use_device_tensormaps else 0,
        )
    launch_tensors = _make_launch_tensor_bundle(
        a=a,
        b=b,
        c=c,
        sfa=sfa,
        sfb=sfb,
        split_sizes=split_sizes,
        counter=counter,
        tensormaps=tensormaps,
        format=format,
        weight_format=weight_format,
        out_dtype=out_dtype,
    )
    dlpack_kwargs = {"enable_tvm_ffi": _format_uses_fp4(kernel_format)}
    scatter_cute = from_dlpack(
        scatter_ptrs[: problem.GM].detach(), assumed_align=8, **dlpack_kwargs
    )
    if use_combine_swiglu:
        if use_combine_swiglu_bwd:
            assert combine_swiglu_bwd_dz is not None
            assert combine_swiglu_bwd_h1 is not None
            combine_swiglu_producer_x = combine_swiglu_bwd_dz
            combine_swiglu_producer_y = combine_swiglu_bwd_h1
        else:
            assert combine_swiglu_x is not None
            assert combine_swiglu_y is not None
            combine_swiglu_producer_x = combine_swiglu_x
            combine_swiglu_producer_y = combine_swiglu_y
        combine_swiglu_x_compile = from_dlpack(
            combine_swiglu_producer_x.detach(),
            assumed_align=16,
            **dlpack_kwargs,
        )
        combine_swiglu_y_compile = from_dlpack(
            combine_swiglu_producer_y.detach(),
            assumed_align=16,
            **dlpack_kwargs,
        )
        combine_swiglu_x_runtime = combine_swiglu_x_compile
        combine_swiglu_y_runtime = combine_swiglu_y_compile
        dispatch_quant_work_compile = from_dlpack(
            dispatch_quant_work_counter,
            assumed_align=4,
            **dlpack_kwargs,
        )
        dispatch_quant_done_compile = from_dlpack(
            dispatch_quant_done_counter,
            assumed_align=4,
            **dlpack_kwargs,
        )
        dispatch_quant_work_runtime = dispatch_quant_work_compile
        dispatch_quant_done_runtime = dispatch_quant_done_compile
        col_q_bytes = (
            0
            if combine_swiglu_col_q_storage is None
            else combine_swiglu_col_q_storage.nbytes
        )
        wide_q_words = max(a.nbytes, col_q_bytes) // 4 > _INT32_MAX
        row_q_words_compile = _q_words_tensor(
            a.view(torch.uint32), wide=wide_q_words, format=kernel_format
        )
        row_scale_compile = from_dlpack(
            sfa.view(torch.uint8).detach(),
            assumed_align=16,
            **dlpack_kwargs,
        )
        if a_global_scale_inv is not None:
            row_global_scale_inv_compile = from_dlpack(
                a_global_scale_inv.detach(),
                assumed_align=16,
                **dlpack_kwargs,
            )
        else:
            row_global_scale_inv_compile = row_scale_compile
        if use_combine_swiglu_fwd and combine_swiglu_row_quant_only:
            col_q_words_compile = row_q_words_compile
            col_scale_compile = row_scale_compile
        else:
            assert combine_swiglu_col_q_storage is not None
            assert combine_swiglu_col_scale is not None
            col_q_words_compile = _q_words_tensor(
                _qdata_words(combine_swiglu_col_q_storage),
                wide=wide_q_words,
                format=kernel_format,
            )
            col_scale_compile = from_dlpack(
                combine_swiglu_col_scale.view(-1).view(torch.uint8).detach(),
                assumed_align=16,
                **dlpack_kwargs,
            )
        row_q_words_runtime = row_q_words_compile
        row_scale_runtime = row_scale_compile
        row_global_scale_inv_runtime = row_global_scale_inv_compile
        col_q_words_runtime = col_q_words_compile
        col_scale_runtime = col_scale_compile
    else:
        wide_q_words = False
        combine_swiglu_x_compile = launch_tensors.compile_tensors[5]
        combine_swiglu_y_compile = launch_tensors.compile_tensors[5]
        combine_swiglu_x_runtime = launch_tensors.runtime_tensors[5]
        combine_swiglu_y_runtime = launch_tensors.runtime_tensors[5]
        dispatch_quant_work_compile = launch_tensors.compile_tensors[6]
        dispatch_quant_done_compile = launch_tensors.compile_tensors[6]
        dispatch_quant_work_runtime = launch_tensors.runtime_tensors[6]
        dispatch_quant_done_runtime = launch_tensors.runtime_tensors[6]
        row_q_words_compile = launch_tensors.compile_tensors[6]
        row_scale_compile = launch_tensors.compile_tensors[6]
        row_q_words_runtime = launch_tensors.runtime_tensors[6]
        row_scale_runtime = launch_tensors.runtime_tensors[6]
        if a_global_scale_inv is not None:
            row_global_scale_inv_compile = from_dlpack(
                a_global_scale_inv.detach(),
                assumed_align=16,
                **dlpack_kwargs,
            )
            row_global_scale_inv_runtime = row_global_scale_inv_compile
        else:
            row_global_scale_inv_compile = row_scale_compile
            row_global_scale_inv_runtime = row_scale_runtime
        col_q_words_compile = row_q_words_compile
        col_scale_compile = row_scale_compile
        col_q_words_runtime = row_q_words_runtime
        col_scale_runtime = row_scale_runtime
    pointer_stride_args, elem_sizes = _make_pointer_stride_args(
        a=a,
        b=b,
        c=c,
        sfa=sfa,
        sfb=sfb,
        format=format,
        weight_format=weight_format,
        N=problem.N,
        K=problem.K,
    )
    elem_a, elem_b, elem_c = elem_sizes
    (
        activation_buffer_base_ptr,
        activation_offsets_cute,
        use_activation_buffer,
        condition_cute,
        use_conditional_execution,
        condition_owner,
        activation_buffer_size_bytes,
    ) = _prepare_activation_buffer_launch_args(
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        fallback_offsets=tensormaps,
        expected_offset_count=ACTIVATION_OFFSET_COUNT,
        enable_tvm_ffi=_format_uses_fp4(kernel_format),
    )
    stream = cutlass_torch.current_stream()
    b_global_scale_inv_ptr = (
        0 if b_global_scale_inv is None else b_global_scale_inv.data_ptr()
    )
    nvfp4_recip_lut = (
        get_nvfp4_recip_lut(a.device)
        if format is NVFP4 and use_combine_swiglu_fwd
        else None
    )
    nvfp4_recip_lut_ptr = 0 if nvfp4_recip_lut is None else nvfp4_recip_lut.data_ptr()
    compile_args = (
        launch_tensors.compile_tensors
        + (scatter_cute,)
        + (
            combine_swiglu_x_compile,
            combine_swiglu_y_compile,
        )
        + (
            dispatch_quant_work_compile,
            dispatch_quant_done_compile,
            row_q_words_compile,
            row_scale_compile,
            row_global_scale_inv_compile,
            col_q_words_compile,
            col_scale_compile,
        )
        + (
            pointer_stride_args,
            elem_sizes,
            problem.G,
            problem.GM,
            problem.N,
            problem.K,
            local_rank,
            num_clusters,
            dispatch_quant_total_tiles,
            dispatch_quant_counter_zero_count,
            use_device_tensormaps,
            activation_buffer_base_ptr,
            cutlass.Int64(activation_buffer_size_bytes),
            activation_offsets_cute,
            use_activation_buffer,
            condition_cute,
            use_conditional_execution,
            cutlass.Int64(b_global_scale_inv_ptr),
            use_global_scale_inv,
            cutlass.Int64(nvfp4_recip_lut_ptr),
            stream,
        )
    )
    runtime_args = (
        launch_tensors.runtime_tensors
        + (scatter_cute,)
        + (
            combine_swiglu_x_runtime,
            combine_swiglu_y_runtime,
        )
        + (
            dispatch_quant_work_runtime,
            dispatch_quant_done_runtime,
            row_q_words_runtime,
            row_scale_runtime,
            row_global_scale_inv_runtime,
            col_q_words_runtime,
            col_scale_runtime,
        )
        + (
            pointer_stride_args,
            problem.GM,
            problem.N,
            problem.K,
            local_rank,
            num_clusters,
            dispatch_quant_total_tiles,
            dispatch_quant_counter_zero_count,
            activation_buffer_base_ptr,
            cutlass.Int64(activation_buffer_size_bytes),
            activation_offsets_cute,
            condition_cute,
            cutlass.Int64(b_global_scale_inv_ptr),
            cutlass.Int64(nvfp4_recip_lut_ptr),
            stream,
        )
    )
    a_placeholder, b_placeholder, c_placeholder = launch_tensors.placeholders
    if use_combine_swiglu_bwd:
        combine_cache_tag = "dist_combine_swiglu_bwd"
    elif combine_precomputed_swiglu:
        combine_cache_tag = "dist_combine_precomputed_swiglu_row_quant_only"
    elif use_combine_swiglu_fwd and combine_swiglu_row_quant_only:
        combine_cache_tag = "dist_combine_swiglu_row_quant_only_16x4"
    elif use_combine_swiglu_fwd:
        combine_cache_tag = "dist_combine_swiglu"
    else:
        combine_cache_tag = "dist_combine"
    combine_swiglu_x_for_layout = (
        combine_swiglu_bwd_dz if use_combine_swiglu_bwd else combine_swiglu_x
    )
    combine_swiglu_y_for_layout = (
        combine_swiglu_bwd_h1 if use_combine_swiglu_bwd else combine_swiglu_y
    )
    combine_swiglu_x_layout = (
        _torch_layout_signature(combine_swiglu_x_for_layout)
        if combine_swiglu_x_for_layout is not None
        else None
    )
    combine_swiglu_y_layout = (
        _torch_layout_signature(combine_swiglu_y_for_layout)
        if combine_swiglu_y_for_layout is not None
        else None
    )
    routed_rows_cache_key = None if use_activation_buffer else problem.GM
    # Packed FP4 column-q storage is (K, M / 2), so its uint32 leading stride
    # depends on M and cannot be shared across row capacities.
    fp4_rows_cache_key = problem.GM if format.name in _FP4_FORMAT_NAMES else None
    cache_key = (
        _config_cache_key(cfg),
        problem_type,
        mode,
        kernel_format.name,
        force_n_major,
        num_n_clusters,
        world_size,
        _torch_layout_signature(a_placeholder),
        _torch_layout_signature(b_placeholder),
        _torch_layout_signature(c_placeholder),
        routed_rows_cache_key,
        problem.G,
        problem.N,
        problem.K,
        elem_a,
        elem_b,
        elem_c,
        use_device_tensormaps,
        use_activation_buffer,
        use_conditional_execution,
        use_global_scale_inv,
        combine_cache_tag,
        dispatch_source_dtype,
        nvfp4_high_throughput,
        combine_swiglu_x_layout,
        combine_swiglu_y_layout,
        pre_shuffled_rows,
        fp4_rows_cache_key,
        wide_q_words,
    )
    _PROBLEM_NAMES = {
        _FPROP: "fprop",
        _DGRAD: "dgrad",
    }
    if use_combine_swiglu_bwd:
        name_suffix = "swiglu_bwd_combine"
    elif use_combine_swiglu_fwd:
        name_suffix = "swiglu_fwd_combine"
    else:
        name_suffix = "combine"
    name_prefix = f"_cute_dist_blockscaled_grouped_gemm_{_PROBLEM_NAMES[problem_type]}_{name_suffix}"
    compile_options = None
    if use_combine_swiglu_bwd:
        compile_options = "--opt-level=1 --ptxas-options='--Ofast-compile=min'"
        cache_key = cache_key + (compile_options,)
    compiled = _compile_or_get(
        cache_key,
        kernel,
        *compile_args,
        name_prefix=name_prefix,
        options=compile_options,
    )
    if derive_pipeline_depth:
        _verify_staged_smem_estimate(
            kernel,
            cfg,
            format=kernel_format,
            groups=problem.G,
            mode=mode,
            blockscaled_dispatch=False,
            nvfp4_high_throughput=nvfp4_high_throughput,
            use_global_scale_inv=use_global_scale_inv,
            combine_swiglu_k=combine_swiglu_k,
            c_stage_dtype=c_stage_dtype,
        )
    compiled(*runtime_args)
    if condition_owner is not None:
        condition_owner.record_stream(torch.cuda.current_stream(condition_owner.device))
    return _maybe_slice_output(symm_mem_buffer.local(), num_output_tokens)


def _fused_dist_blockscaled_grouped_gemm_dispatch_quant_impl(  # noqa: C901
    *,
    b: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
    problem_type: int,
    contraction_axes: tuple[int, int],
    c: torch.Tensor | None,
    out_dtype: torch.dtype,
    dim: int,
    dtype: torch.dtype,
    num_sms: int | None,
    config: dict | None,
    m_multiple_of: int,
    return_wgrad_quant: bool,
    blockscaled_dispatch: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    counter_storage: torch.Tensor | None = None,
    b_global_scale_inv: torch.Tensor | None = None,
    interleaved_fc13: bool = False,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    derive_pipeline_depth: bool = False,
    pre_shuffled_rows: bool = False,
) -> (
    tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor] | None,
    ]
    | tuple[
        None,
        tuple[torch.Tensor, torch.Tensor],
        None,
        tuple[torch.Tensor, torch.Tensor],
    ]
):
    weight_format = format if weight_format is None else weight_format
    kernel_format = make_mixed_blockscaled_format(format, weight_format)
    if gather_ptrs.dtype != torch.int64:
        raise TypeError("gather_ptrs must be int64")
    if not gather_ptrs.is_cuda or gather_ptrs.device != b.device:
        raise ValueError("gather_ptrs must be on b.device")
    rows = int(gather_ptrs.numel()) if num_out_tokens is None else int(num_out_tokens)
    if gather_ptrs.numel() < rows:
        raise ValueError(
            "gather_ptrs length must be at least num_out_tokens; "
            f"got len={gather_ptrs.numel()} and num_out_tokens={rows}"
        )
    if interleaved_fc13:
        if weight_format not in (MXFP8_E4M3, MXFP4, NVFP4):
            raise NotImplementedError(
                "fused interleaved FC13 dispatch supports MXFP8 E4M3, MXFP4, and NVFP4"
            )
        if problem_type != _FPROP or not blockscaled_dispatch:
            raise NotImplementedError(
                "interleaved FC13 dispatch is forward-only inference"
            )
        if activation_buffer is not None:
            raise NotImplementedError(
                "interleaved FC13 dispatch does not support activation-buffer outputs"
            )
        if return_wgrad_quant:
            raise NotImplementedError(
                "interleaved FC13 dispatch does not produce backward quantization"
            )
        if c is not None:
            raise ValueError("interleaved FC13 dispatch owns its output storage")
    if not _can_use_fused_dispatch_quant(
        rows=rows,
        dim=dim,
        dtype=dtype,
        format=format,
        blockscaled_dispatch=blockscaled_dispatch,
        m_multiple_of=m_multiple_of,
    ):
        raise NotImplementedError(
            "fused blockscaled DISPATCH quant currently supports compatible CuTe "
            "block-scaled formats with BF16/FP16 inputs, rows aligned to "
            f"m_multiple_of={m_multiple_of}, and dim aligned to "
            f"{BLOCKSCALED_DISPATCH_DIM_ALIGNMENT}"
        )

    _assert_split_sizes_fit_capacity(
        split_sizes,
        rows=rows,
        multiple_of=m_multiple_of,
    )
    # Quant producers derive their schedule from the runtime split tensor.
    dispatch_quant_total_tiles = 0

    a, sfa = _dispatch_quantized_operand_placeholder(
        rows=rows,
        scale_rows=_compact_scale_storage_rows(rows, m_multiple_of),
        dim=dim,
        format=format,
        device=b.device,
        activation_buffer=activation_buffer,
    )
    _validate_2d_operand_layout(
        "DISPATCH a",
        a,
        format,
        contraction_axis=contraction_axes[0],
    )
    _validate_3d_operand_layout(
        "DISPATCH b",
        b,
        weight_format,
        contraction_axis=contraction_axes[1],
    )
    _validate_blockscaled_launch_inputs(
        a=a,
        b=b,
        sfa=sfa,
        sfb=sfb,
        split_sizes=split_sizes,
        format=format,
        weight_format=weight_format,
        split_size_multiple_of=m_multiple_of,
        split_size_alignment=_blockscaled_split_size_alignment(m_multiple_of),
    )
    problem = _parse_blockscaled_problem(
        a=a,
        b=b,
        format=format,
        weight_format=weight_format,
        split_sizes=split_sizes,
        contraction_axes=contraction_axes,
        problem_type=problem_type,
    )
    if interleaved_fc13 and (problem.N % 2 != 0 or (problem.N // 2) % 128 != 0):
        raise NotImplementedError(
            "interleaved FC13 dispatch requires intermediate_dim divisible by 128"
        )
    use_global_scale_inv = b_global_scale_inv is not None
    if use_global_scale_inv:
        if format is not NVFP4 or problem_type != _FPROP or not blockscaled_dispatch:
            raise ValueError(
                "dynamic global scaling requires NVFP4 FPROP blockscaled dispatch"
            )
        _validate_global_scale_inv(
            b_global_scale_inv,
            size=problem.G,
            device=b.device,
            name="b_global_scale_inv",
        )
        a_global_scale_inv = torch.empty(
            (problem.GM,), dtype=torch.float32, device=b.device
        )
    else:
        a_global_scale_inv = None
    if activation_buffer is not None and c is not None:
        raise ValueError("c must be None when activation_buffer is provided")
    if activation_buffer is not None and c is None:
        c = _activation_buffer_placeholder(
            activation_buffer,
            problem.c_shape,
            out_dtype,
        )
    if interleaved_fc13:
        c = torch.empty((1, problem.N), dtype=out_dtype, device=b.device)
    else:
        c, out_dtype = _resolve_output_tensor(
            c=c,
            c_shape=problem.c_shape,
            out_dtype=out_dtype,
            device=b.device,
            problem_type=problem_type,
            output_accum=False,
        )
    if interleaved_fc13:
        if format is NVFP4:
            col_q_storage = torch.empty(
                (rows, problem.N // 2), dtype=out_dtype, device=b.device
            )
            col_scale = torch.empty(
                0,
                dtype=_torch_dtype(format.sf_dtype),
                device=b.device,
            )
        else:
            col_q_storage, col_scale = _empty_dispatch_quantized_operand(
                rows=rows,
                scale_rows=_compact_scale_storage_rows(rows, m_multiple_of),
                dim=problem.N // 2,
                format=format,
                device=b.device,
            )
    elif activation_buffer is None:
        col_q_storage = _empty_qdata(
            _column_quantized_storage_shape(rows, dim, format),
            format,
            b.device,
        )
        col_scale = torch.empty(
            _column_quantized_scale_shape(rows, dim, format),
            dtype=_torch_dtype(format.sf_dtype),
            device=b.device,
        )
    else:
        col_q_storage = _activation_buffer_placeholder(
            activation_buffer,
            _column_quantized_storage_shape(rows, dim, format),
            _torch_dtype(format.a_dtype),
        )
        col_scale = _activation_buffer_placeholder(
            activation_buffer,
            _column_quantized_scale_shape(rows, dim, format),
            _torch_dtype(format.sf_dtype),
        )
    col_quant = None
    if return_wgrad_quant:
        col_quant = (
            _column_quantized_output(col_q_storage, format),
            col_scale.view(-1),
        )
    interleaved_output = (
        (
            col_q_storage,
            None if format is NVFP4 else col_scale,
            None,
        )
        if interleaved_fc13
        else None
    )
    if rows == 0:
        if activation_buffer is None:
            col_q_storage.view(torch.uint8).zero_()
            col_scale.zero_()
        # Placeholders alias byte zero of the shared buffer, so clearing them
        # would clobber other operands. Consumers ignore their contents at zero rows.
        if interleaved_fc13:
            return (
                None,
                _row_quant_tuple(a, sfa, a_global_scale_inv),
                col_quant,
                interleaved_output,
            )
        return c, _row_quant_tuple(a, sfa, a_global_scale_inv), col_quant
    if split_sizes.dtype != torch.int32:
        split_sizes = split_sizes.to(torch.int32)
    if num_sms is None:
        num_sms = num_sms_per_device()

    auto_selected_config = config is None
    if auto_selected_config:
        config = _select_blockscaled_config(
            problem=problem,
            num_sms=num_sms,
            format=format,
            weight_format=weight_format,
            m_multiple_of=m_multiple_of,
            problem_type=problem_type,
            activation_buffer=activation_buffer,
            derive_pipeline_depth=derive_pipeline_depth,
        )
    explicit_training_config = (
        not auto_selected_config
        and weight_format is format
        and config == blockscaled_training_config(format)
    )
    cfg = dict(config)
    _validate_swap_ab_row_multiple(cfg, m_multiple_of)
    auto_pipeline_depth = derive_pipeline_depth and _uses_derived_staged_pipeline(
        cfg,
        format=format,
        weight_format=weight_format,
    )
    if _format_uses_fp4(weight_format):
        cfg.setdefault(
            "MMA_ATOM_N",
            _staged_fp4_mma_atom_n(
                config=cfg,
                format=weight_format,
                n=problem.N,
                k=problem.K,
                mixed_operand_widths=format is not weight_format,
            ),
        )
    if auto_selected_config and _use_t6_mxfp8_prefill_fc13_unroll(
        format=format,
        weight_format=weight_format,
        rows=rows,
        num_sms=num_sms,
        n=problem.N,
        k=problem.K,
        interleaved_fc13=interleaved_fc13,
    ):
        cfg["KLOOP_UNROLL"] = 2
    if interleaved_fc13 and swiglu_fast_math:
        cfg["COMBINE_SWIGLU_FAST_MATH"] = True
    if interleaved_fc13:
        _set_swiglu_clamp_config(cfg, swiglu_clamped, swiglu_alpha, swiglu_limit)
    if interleaved_fc13:
        if cfg.get("SWAP_AB", False):
            epilogue_cols = cfg["BLOCK_SIZE_M"] // (cfg["NUM_CTAS"] * cfg["NUM_MMAS"])
        else:
            width_scale = max(1, out_dtype.itemsize * 8 // format.a_dtype.width)
            epilogue_cols = cfg["BLOCK_SIZE_N"] // (
                cfg["EPILOGUE_SUBTILE"] * width_scale
            )
        if epilogue_cols not in (64, 128):
            raise NotImplementedError(
                "interleaved FC13 dispatch requires a 64- or 128-column "
                f"epilogue tile; got {epilogue_cols}"
            )
    use_device_tensormaps = _staged_use_device_tensormaps(
        cfg,
        activation_buffer=activation_buffer,
        problem_type=problem_type,
        m_multiple_of=m_multiple_of,
        n=problem.N,
        groups=problem.G,
    )
    num_clusters = max(1, num_sms // cfg["NUM_CTAS"])

    local_rank = symm_mem_buffer.hdl.rank
    world_size = symm_mem_buffer.hdl.world_size
    force_n_major = True
    num_n_clusters = 1
    dispatch_quant_source_dtype = _dispatch_quant_source_dtype_from_torch(dtype)
    nvfp4_high_throughput = _use_nvfp4_high_throughput_producers(
        format=format,
        rows=rows,
        num_sms=num_sms,
        hidden_dim=problem.K,
    )
    if auto_pipeline_depth:
        cfg = derive_staged_blockscaled_pipeline_config(
            cfg,
            format=kernel_format,
            activation_format=format,
            weight_format=weight_format,
            groups=problem.G,
            mode=DistBlockScaledGroupedGemmKernel.DISPATCH_MODE,
            blockscaled_dispatch=blockscaled_dispatch,
            nvfp4_high_throughput=nvfp4_high_throughput,
            use_global_scale_inv=use_global_scale_inv,
            combine_swiglu_k=None,
            c_stage_dtype=out_dtype,
        )
    kernel = _get_dist_blockscaled_kernel(
        config=cfg,
        problem_type=problem_type,
        mode=DistBlockScaledGroupedGemmKernel.DISPATCH_MODE,
        format=kernel_format,
        force_n_major=force_n_major,
        num_n_clusters=num_n_clusters,
        world_size=world_size,
        dispatch_source_dtype=dispatch_quant_source_dtype,
        blockscaled_dispatch=blockscaled_dispatch,
        nvfp4_high_throughput=nvfp4_high_throughput,
        interleaved_fc13=interleaved_fc13,
        # See _use_vector_dispatch_scale_copy: the byte-granular scale copy is
        # the decode latency bottleneck for the MX formats. NVFP4 keeps its
        # separately tuned staged policy.
        vector_dispatch_scale_copy=(
            blockscaled_dispatch
            and format is not NVFP4
            and _dispatch_scale_copy_atom_aligned(format=format, hidden_dim=problem.K)
        ),
        schedule_rotation=not pre_shuffled_rows,
    )

    dispatch_quant_done_counter_size = max(1, rows // format.sf_vec_size)
    (
        counter,
        tensormaps,
        dispatch_quant_work_counter,
        dispatch_quant_done_counter,
        dispatch_quant_counter_zero_count,
    ) = _allocate_dispatch_counter_workspace(
        num_clusters,
        cfg["NUM_CTAS"],
        b.device,
        done_counter_size=dispatch_quant_done_counter_size,
        min_rows=problem.G if use_device_tensormaps else 0,
        zero_in_prepare=use_device_tensormaps,
        counter_storage=counter_storage,
    )
    launch_tensors = _make_launch_tensor_bundle(
        a=a,
        b=b,
        c=c,
        sfa=sfa,
        sfb=sfb,
        split_sizes=split_sizes,
        counter=counter,
        tensormaps=tensormaps,
        format=format,
        weight_format=weight_format,
        out_dtype=out_dtype,
    )
    dlpack_kwargs = {"enable_tvm_ffi": _format_uses_fp4(kernel_format)}
    gather_cute = from_dlpack(
        gather_ptrs[:rows].detach(), assumed_align=8, **dlpack_kwargs
    )
    dispatch_quant_work_cute = from_dlpack(
        dispatch_quant_work_counter, assumed_align=4, **dlpack_kwargs
    )
    dispatch_quant_done_cute = from_dlpack(
        dispatch_quant_done_counter, assumed_align=4, **dlpack_kwargs
    )
    row_q_words = a.view(torch.uint32)
    row_scale_bytes = sfa.view(torch.uint8)
    row_q_words_cute = from_dlpack(
        row_q_words.detach(), assumed_align=16, **dlpack_kwargs
    )
    row_scale_cute = from_dlpack(
        row_scale_bytes.detach(), assumed_align=16, **dlpack_kwargs
    )
    if a_global_scale_inv is not None:
        row_global_scale_inv_cute = from_dlpack(
            a_global_scale_inv,
            assumed_align=16,
            **dlpack_kwargs,
        )
    else:
        row_global_scale_inv_cute = launch_tensors.compile_tensors[6]
    pointer_stride_args, elem_sizes = _make_pointer_stride_args(
        a=a,
        b=b,
        c=c,
        sfa=sfa,
        sfb=sfb,
        format=format,
        weight_format=weight_format,
        N=problem.N,
        K=problem.K,
    )
    elem_a, elem_b, elem_c = elem_sizes
    col_q_words = _qdata_words(col_q_storage)
    col_scale_bytes = col_scale.view(-1).view(torch.uint8)
    col_q_words_cute = from_dlpack(
        col_q_words.detach(),
        assumed_align=16,
        **dlpack_kwargs,
    )
    col_scale_cute = from_dlpack(
        col_scale_bytes.detach(),
        assumed_align=16,
        **dlpack_kwargs,
    )
    compile_dispatch_col_q_words = col_q_words_cute
    compile_dispatch_col_scale = col_scale_cute
    runtime_dispatch_col_q_words = col_q_words_cute
    runtime_dispatch_col_scale = col_scale_cute
    (
        activation_buffer_base_ptr,
        activation_offsets_cute,
        use_activation_buffer,
        condition_cute,
        use_conditional_execution,
        condition_owner,
        activation_buffer_size_bytes,
    ) = _prepare_activation_buffer_launch_args(
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        fallback_offsets=tensormaps,
        expected_offset_count=ACTIVATION_OFFSET_COUNT,
        enable_tvm_ffi=_format_uses_fp4(kernel_format),
    )
    stream = cutlass_torch.current_stream()
    b_global_scale_inv_ptr = (
        0 if b_global_scale_inv is None else b_global_scale_inv.data_ptr()
    )
    compile_combine_swiglu_x_dummy = launch_tensors.compile_tensors[0]
    compile_combine_swiglu_y_dummy = launch_tensors.compile_tensors[0]
    runtime_combine_swiglu_x_dummy = launch_tensors.runtime_tensors[0]
    runtime_combine_swiglu_y_dummy = launch_tensors.runtime_tensors[0]
    compile_args = (
        launch_tensors.compile_tensors
        + (
            gather_cute,
            compile_combine_swiglu_x_dummy,
            compile_combine_swiglu_y_dummy,
            dispatch_quant_work_cute,
            dispatch_quant_done_cute,
            row_q_words_cute,
            row_scale_cute,
            row_global_scale_inv_cute,
            compile_dispatch_col_q_words,
            compile_dispatch_col_scale,
        )
        + (
            pointer_stride_args,
            elem_sizes,
            problem.G,
            problem.GM,
            problem.N,
            problem.K,
            local_rank,
            num_clusters,
            dispatch_quant_total_tiles,
            dispatch_quant_counter_zero_count,
            use_device_tensormaps,
            activation_buffer_base_ptr,
            cutlass.Int64(activation_buffer_size_bytes),
            activation_offsets_cute,
            use_activation_buffer,
            condition_cute,
            use_conditional_execution,
            cutlass.Int64(b_global_scale_inv_ptr),
            use_global_scale_inv,
            cutlass.Int64(0),
            stream,
        )
    )
    runtime_args = (
        launch_tensors.runtime_tensors
        + (
            gather_cute,
            runtime_combine_swiglu_x_dummy,
            runtime_combine_swiglu_y_dummy,
            dispatch_quant_work_cute,
            dispatch_quant_done_cute,
            row_q_words_cute,
            row_scale_cute,
            row_global_scale_inv_cute,
            runtime_dispatch_col_q_words,
            runtime_dispatch_col_scale,
        )
        + (
            pointer_stride_args,
            problem.GM,
            problem.N,
            problem.K,
            local_rank,
            num_clusters,
            dispatch_quant_total_tiles,
            dispatch_quant_counter_zero_count,
            activation_buffer_base_ptr,
            cutlass.Int64(activation_buffer_size_bytes),
            activation_offsets_cute,
            condition_cute,
            cutlass.Int64(b_global_scale_inv_ptr),
            cutlass.Int64(0),
            stream,
        )
    )
    a_placeholder, b_placeholder, c_placeholder = launch_tensors.placeholders
    routed_rows_cache_key = None if use_activation_buffer else rows
    # Packed FP4 column-q storage is (K, M / 2), so its uint32 leading stride
    # depends on M and cannot be shared across row capacities.
    fp4_rows_cache_key = problem.GM if format.name in _FP4_FORMAT_NAMES else None
    cache_key = (
        _config_cache_key(cfg),
        problem_type,
        DistBlockScaledGroupedGemmKernel.DISPATCH_MODE,
        kernel_format.name,
        force_n_major,
        num_n_clusters,
        world_size,
        _torch_layout_signature(a_placeholder),
        _torch_layout_signature(b_placeholder),
        _torch_layout_signature(c_placeholder),
        routed_rows_cache_key,
        problem.G,
        problem.N,
        problem.K,
        elem_a,
        elem_b,
        elem_c,
        dtype,
        blockscaled_dispatch,
        nvfp4_high_throughput,
        use_device_tensormaps,
        use_activation_buffer,
        use_conditional_execution,
        use_global_scale_inv,
        interleaved_fc13,
        "dist_dispatch_quant",
        fp4_rows_cache_key,
        pre_shuffled_rows,
    )
    _PROBLEM_NAMES = {
        _FPROP: "fprop",
        _DGRAD: "dgrad",
    }
    name_prefix = (
        f"_cute_dist_blockscaled_grouped_gemm_{_PROBLEM_NAMES[problem_type]}_dispatch"
    )
    compile_options = (
        "--opt-level=1" if explicit_training_config and format is MXFP4 else None
    )
    cache_key = (*cache_key, compile_options)
    compiled = _compile_or_get(
        cache_key,
        kernel,
        *compile_args,
        name_prefix=name_prefix,
        options=compile_options,
    )
    if derive_pipeline_depth:
        _verify_staged_smem_estimate(
            kernel,
            cfg,
            format=kernel_format,
            groups=problem.G,
            mode=DistBlockScaledGroupedGemmKernel.DISPATCH_MODE,
            blockscaled_dispatch=blockscaled_dispatch,
            nvfp4_high_throughput=nvfp4_high_throughput,
            use_global_scale_inv=use_global_scale_inv,
            combine_swiglu_k=None,
            c_stage_dtype=out_dtype,
        )
    compiled(*runtime_args)
    if condition_owner is not None:
        condition_owner.record_stream(torch.cuda.current_stream(condition_owner.device))

    if interleaved_fc13:
        return (
            None,
            _row_quant_tuple(a, sfa, a_global_scale_inv),
            col_quant,
            interleaved_output,
        )
    return c, _row_quant_tuple(a, sfa, a_global_scale_inv), col_quant


def dist_blockscaled_grouped_gemm_fprop_dispatch(
    x: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    topk: int | None = None,
    y: torch.Tensor | None = None,
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    weight_format: BlockScaledFormatSpec | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = _DEFAULT_M_MULTIPLE_OF,
    return_wgrad_quant: bool = False,
    blockscaled_dispatch: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    counter_storage: torch.Tensor | None = None,
    b_global_scale_inv: torch.Tensor | None = None,
    interleaved_fc13: bool = False,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    derive_pipeline_depth: bool = False,
    pre_shuffled_rows: bool = False,
):
    """Gather+quantize A over NVLink, then run block-scaled FPROP GEMM.

    With `activation_buffer`, activation-backed return tensors are shape-only
    aliases at buffer offset zero; consume results through `activation_offsets`.
    """
    del topk
    if blockscaled_dispatch and return_wgrad_quant:
        raise ValueError(
            "blockscaled dispatch does not produce column-quantized activations"
        )
    if gather_ptrs.dtype != torch.int64:
        raise TypeError("gather_ptrs must be int64")
    weight_format = format if weight_format is None else weight_format
    G, N, K_storage = w.shape
    del G, N
    K = _logical_dim_size(K_storage, weight_format)
    x_2d = _flatten_last_dim(x, K, "x")
    rows = int(gather_ptrs.numel()) if num_out_tokens is None else int(num_out_tokens)
    if _can_use_fused_dispatch_quant(
        rows=rows,
        dim=K,
        dtype=x_2d.dtype,
        format=format,
        blockscaled_dispatch=blockscaled_dispatch,
        m_multiple_of=m_multiple_of,
    ):
        result = _fused_dist_blockscaled_grouped_gemm_dispatch_quant_impl(
            b=w,
            sfb=sfb,
            split_sizes=split_sizes,
            gather_ptrs=gather_ptrs,
            num_out_tokens=num_out_tokens,
            symm_mem_buffer=symm_mem_buffer,
            format=format,
            weight_format=weight_format,
            problem_type=_FPROP,
            contraction_axes=(1, 2),
            c=y,
            out_dtype=out_dtype,
            dim=K,
            dtype=x_2d.dtype,
            num_sms=num_sms,
            config=config,
            m_multiple_of=m_multiple_of,
            return_wgrad_quant=return_wgrad_quant,
            blockscaled_dispatch=blockscaled_dispatch,
            activation_buffer=activation_buffer,
            activation_offsets=activation_offsets,
            conditional_execution=conditional_execution,
            counter_storage=counter_storage,
            b_global_scale_inv=b_global_scale_inv,
            interleaved_fc13=interleaved_fc13,
            swiglu_fast_math=swiglu_fast_math,
            swiglu_clamped=swiglu_clamped,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            derive_pipeline_depth=derive_pipeline_depth,
            pre_shuffled_rows=pre_shuffled_rows,
        )
        if interleaved_fc13:
            return result
        y, row_quant, x_col_quant = result
        if return_wgrad_quant:
            return y, row_quant, x_col_quant
        return y, row_quant

    raise NotImplementedError(
        "fused blockscaled FPROP DISPATCH currently supports compatible CuTe "
        "block-scaled formats with BF16/FP16 inputs, rows aligned to "
        f"m_multiple_of={m_multiple_of}, and dim aligned to "
        f"{BLOCKSCALED_DISPATCH_DIM_ALIGNMENT}"
    )


def dist_blockscaled_grouped_gemm_dgrad_dispatch(
    dy: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    dx: torch.Tensor | None = None,
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = _DEFAULT_M_MULTIPLE_OF,
    return_wgrad_quant: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    pre_shuffled_rows: bool = False,
):
    """Gather+quantize dy over NVLink, then run block-scaled DGRAD GEMM.

    With `activation_buffer`, activation-backed return tensors are shape-only
    aliases at buffer offset zero; consume results through `activation_offsets`.
    """
    if gather_ptrs.dtype != torch.int64:
        raise TypeError("gather_ptrs must be int64")
    G, N_storage, K = w.shape
    del G, K
    N = _logical_dim_size(N_storage, format)
    dy_2d = _flatten_last_dim(dy, N, "dy")
    rows = int(gather_ptrs.numel()) if num_out_tokens is None else int(num_out_tokens)
    if _can_use_fused_dispatch_quant(
        rows=rows,
        dim=N,
        dtype=dy_2d.dtype,
        format=format,
        m_multiple_of=m_multiple_of,
    ):
        dx, (dy_q, sfa), dy_col_quant = (
            _fused_dist_blockscaled_grouped_gemm_dispatch_quant_impl(
                b=w.transpose(1, 2),
                sfb=sfb,
                split_sizes=split_sizes,
                gather_ptrs=gather_ptrs,
                num_out_tokens=num_out_tokens,
                symm_mem_buffer=symm_mem_buffer,
                format=format,
                problem_type=_DGRAD,
                contraction_axes=(1, 2),
                c=dx,
                out_dtype=out_dtype,
                dim=N,
                dtype=dy_2d.dtype,
                num_sms=num_sms,
                config=config,
                m_multiple_of=m_multiple_of,
                return_wgrad_quant=return_wgrad_quant,
                activation_buffer=activation_buffer,
                activation_offsets=activation_offsets,
                conditional_execution=conditional_execution,
                pre_shuffled_rows=pre_shuffled_rows,
            )
        )
        if return_wgrad_quant:
            return dx, (dy_q, sfa), dy_col_quant
        return dx, (dy_q, sfa)

    raise NotImplementedError(
        "fused blockscaled DGRAD DISPATCH currently supports compatible CuTe "
        "block-scaled formats with BF16/FP16 inputs, rows aligned to "
        f"m_multiple_of={m_multiple_of}, and dim aligned to "
        f"{BLOCKSCALED_DISPATCH_DIM_ALIGNMENT}"
    )


def dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine(  # noqa: C901
    h1: torch.Tensor | None,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    y: torch.Tensor | None = None,
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    weight_format: BlockScaledFormatSpec | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = _DEFAULT_M_MULTIPLE_OF,
    num_output_tokens: int | None = None,
    return_wgrad_quant: bool = False,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    situ_glu: bool = False,
    situ_glu_gate_beta: float = SITU_GLU_GATE_BETA_DEFAULT,
    situ_glu_up_beta: float = SITU_GLU_UP_BETA_DEFAULT,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    num_recv_tokens: int | None = None,
    counter_storage: torch.Tensor | None = None,
    b_global_scale_inv: torch.Tensor | None = None,
    h2_quant: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]
    | None = None,
    precomputed_swiglu: bool = False,
    return_row_quant: bool = False,
    derive_pipeline_depth: bool = False,
    pre_shuffled_rows: bool = False,
):
    """Run SwiGLU fwd + activation quant + FC2 FPROP GEMM + peer scatter.

    Activation-buffer launches address the input through `activation_offsets`
    and use `h1.dtype` to interpret its bytes; `h1` storage is consumed only by
    direct-tensor launches.
    Activation-backed return tensors are shape-only aliases at buffer offset
    zero; consume results through `activation_offsets`.

    `return_row_quant` additionally returns the row-quantized `h2` this kernel
    fed to the GEMM. It is produced unconditionally; the flag only surfaces it,
    so a BF16 backward can dequantize the exact FC2 A operand the forward used.
    """
    weight_format = format if weight_format is None else weight_format
    if h2_quant is not None:
        if precomputed_swiglu:
            raise ValueError("h2_quant and precomputed_swiglu are mutually exclusive")
        if h1 is not None:
            raise ValueError("h1 and h2_quant are mutually exclusive")
        if return_wgrad_quant:
            raise NotImplementedError(
                "prequantized h2 combine is forward-only inference"
            )
        if return_row_quant:
            raise NotImplementedError(
                "prequantized h2 combine already owns its row quant"
            )
        if activation_buffer is not None:
            raise NotImplementedError(
                "prequantized h2 combine does not support activation-buffer inputs"
            )
        h2_q, h2_scale, h2_global_scale_inv = h2_quant
        return _fused_dist_blockscaled_grouped_gemm_combine_impl(
            a=h2_q,
            b=w,
            sfa=h2_scale,
            sfb=sfb,
            split_sizes=split_sizes,
            scatter_ptrs=scatter_ptrs,
            symm_mem_buffer=symm_mem_buffer,
            format=format,
            weight_format=weight_format,
            problem_type=_FPROP,
            contraction_axes=(1, 2),
            c=y,
            out_dtype=out_dtype,
            num_sms=num_sms,
            config=config,
            m_multiple_of=m_multiple_of,
            num_output_tokens=num_output_tokens,
            conditional_execution=conditional_execution,
            counter_storage=counter_storage,
            a_global_scale_inv=h2_global_scale_inv,
            b_global_scale_inv=b_global_scale_inv,
            derive_pipeline_depth=derive_pipeline_depth,
            pre_shuffled_rows=pre_shuffled_rows,
        )
    if h1 is None:
        raise ValueError("h1 is required when h2_quant is not provided")
    G, N, K_storage = w.shape
    del G, N
    K = _logical_dim_size(K_storage, weight_format)
    h1_dim = K if precomputed_swiglu else 2 * K
    if activation_buffer is None:
        h1_2d = _flatten_last_dim(h1, h1_dim, "h1")
    else:
        if num_recv_tokens is None:
            raise ValueError(
                "activation_buffer SwiGLU combine requires num_recv_tokens capacity"
            )
        h1_2d = _activation_buffer_placeholder(
            activation_buffer,
            (num_recv_tokens, h1_dim),
            h1.dtype,
        )
    if h1_2d.device != w.device:
        raise ValueError("h1 and w must be on the same device")
    rows = int(h1_2d.shape[0])
    if format is NVFP4 and return_wgrad_quant:
        raise ValueError(
            "NVFP4 SwiGLU combine does not produce column-quantized activations"
        )
    if not _can_use_fused_combine_swiglu_quant(
        rows=rows,
        dim=K,
        dtype=h1_2d.dtype,
        format=format,
        allow_nvfp4=True,
    ):
        raise NotImplementedError(
            "fused blockscaled COMBINE SwiGLU currently supports compatible CuTe "
            "block-scaled formats with "
            "BF16/FP16 inputs with rows and dim multiples of 128"
        )
    a, sfa = _dispatch_quantized_operand_placeholder(
        rows=rows,
        scale_rows=_compact_scale_storage_rows(rows, m_multiple_of),
        dim=K,
        format=format,
        device=w.device,
        activation_buffer=activation_buffer,
    )
    a_global_scale_inv = None
    if format is NVFP4:
        if b_global_scale_inv is None:
            raise ValueError("NVFP4 SwiGLU combine requires a weight global scale")
        if activation_buffer is None:
            a_global_scale_inv = torch.empty(
                rows,
                dtype=torch.float32,
                device=w.device,
            )
        else:
            a_global_scale_inv = _activation_buffer_placeholder(
                activation_buffer,
                (rows,),
                torch.float32,
            )
    col_q_storage = None
    col_scale = None
    if return_wgrad_quant:
        if activation_buffer is None:
            col_q_storage = _empty_qdata(
                _column_quantized_storage_shape(rows, K, format),
                format,
                w.device,
            )
            col_scale = torch.empty(
                _column_quantized_scale_shape(rows, K, format),
                dtype=_torch_dtype(format.sf_dtype),
                device=w.device,
            )
        else:
            col_q_storage = _activation_buffer_placeholder(
                activation_buffer,
                _column_quantized_storage_shape(rows, K, format),
                _torch_dtype(format.a_dtype),
            )
            col_scale = _activation_buffer_placeholder(
                activation_buffer,
                _column_quantized_scale_shape(rows, K, format),
                _torch_dtype(format.sf_dtype),
            )
    if precomputed_swiglu:
        if format is not NVFP4:
            raise NotImplementedError(
                "precomputed SwiGLU combine currently supports NVFP4 only"
            )
        h1_gate = h1_2d
        h1_up = h1_2d
    else:
        h1_gate = h1_2d[:, :K]
        h1_up = h1_2d[:, K:]
    _validate_2d_operand_layout(
        "COMBINE_SWIGLU a",
        a,
        format,
        contraction_axis=1,
    )
    _validate_3d_operand_layout(
        "COMBINE_SWIGLU w",
        w,
        weight_format,
        contraction_axis=2,
    )
    out = _fused_dist_blockscaled_grouped_gemm_combine_impl(
        a=a,
        b=w,
        sfa=sfa,
        sfb=sfb,
        split_sizes=split_sizes,
        scatter_ptrs=scatter_ptrs,
        symm_mem_buffer=symm_mem_buffer,
        format=format,
        weight_format=weight_format,
        problem_type=_FPROP,
        contraction_axes=(1, 2),
        c=y,
        out_dtype=out_dtype,
        num_sms=num_sms,
        config=config,
        m_multiple_of=m_multiple_of,
        num_output_tokens=num_output_tokens,
        combine_swiglu_x=h1_gate,
        combine_swiglu_y=h1_up,
        combine_swiglu_col_q_storage=col_q_storage,
        combine_swiglu_col_scale=col_scale,
        combine_swiglu_fast_math=swiglu_fast_math,
        combine_swiglu_clamped=swiglu_clamped,
        combine_swiglu_alpha=swiglu_alpha,
        combine_swiglu_limit=swiglu_limit,
        combine_situ_glu=situ_glu,
        combine_situ_glu_gate_beta=situ_glu_gate_beta,
        combine_situ_glu_up_beta=situ_glu_up_beta,
        combine_swiglu_row_quant_only=not return_wgrad_quant,
        combine_precomputed_swiglu=precomputed_swiglu,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        counter_storage=counter_storage,
        a_global_scale_inv=a_global_scale_inv,
        b_global_scale_inv=b_global_scale_inv,
        derive_pipeline_depth=derive_pipeline_depth,
        pre_shuffled_rows=pre_shuffled_rows,
    )
    if return_wgrad_quant:
        assert col_q_storage is not None
        assert col_scale is not None
        col_quant = (
            _column_quantized_output(col_q_storage, format),
            col_scale.view(-1),
        )
        if return_row_quant:
            return out, col_quant, _row_quant_tuple(a, sfa, a_global_scale_inv)
        return out, col_quant
    if return_row_quant:
        return out, _row_quant_tuple(a, sfa, a_global_scale_inv)
    return out


def dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine(
    grad_h2: torch.Tensor,
    h1: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    dx: torch.Tensor | None = None,
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = _DEFAULT_M_MULTIPLE_OF,
    num_output_tokens: int | None = None,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    situ_glu: bool = False,
    situ_glu_gate_beta: float = SITU_GLU_GATE_BETA_DEFAULT,
    situ_glu_up_beta: float = SITU_GLU_UP_BETA_DEFAULT,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    num_recv_tokens: int | None = None,
    pre_shuffled_rows: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Run SwiGLU bwd quant + W13 DGRAD GEMM + peer scatter.

    Activation-buffer launches address activation operands through
    `activation_offsets` and use `grad_h2.dtype` and `h1.dtype` to interpret
    their bytes; operand storage is consumed only by direct-tensor launches.
    Activation-backed return tensors are shape-only aliases at buffer offset
    zero; consume results through `activation_offsets`.
    """
    if activation_buffer is not None:
        if w.shape[1] % 2 != 0:
            raise ValueError("W13 output dimension must be 2 * intermediate_dim")
        # Buffer placeholders derive K from the W13 (G, 2*K, N) layout.
        K = int(w.shape[1] // 2)
    else:
        K = int(grad_h2.shape[-1])
    dxy_dim = 2 * K
    if activation_buffer is None:
        grad_h2_2d = _flatten_last_dim(grad_h2, K, "grad_h2")
        h1_2d = _flatten_last_dim(h1, dxy_dim, "h1")
    else:
        if num_recv_tokens is None:
            raise ValueError(
                "activation_buffer SwiGLU backward requires num_recv_tokens capacity"
            )
        grad_h2_2d = _activation_buffer_placeholder(
            activation_buffer, (num_recv_tokens, K), grad_h2.dtype
        )
        h1_2d = _activation_buffer_placeholder(
            activation_buffer, (num_recv_tokens, dxy_dim), h1.dtype
        )
    if grad_h2_2d.shape[0] != h1_2d.shape[0]:
        raise ValueError(
            "grad_h2 and h1 must flatten to the same row count; "
            f"got {grad_h2_2d.shape[0]} and {h1_2d.shape[0]}"
        )
    if grad_h2_2d.device != w.device or h1_2d.device != w.device:
        raise ValueError("grad_h2, h1, and w must be on the same device")
    rows = int(grad_h2_2d.shape[0])
    if not _can_use_fused_combine_swiglu_quant(
        rows=rows,
        dim=K,
        dtype=grad_h2_2d.dtype,
        format=format,
    ):
        raise NotImplementedError(
            "fused blockscaled DGRAD SwiGLU-bwd combine currently supports cute "
            "block-scaled formats with BF16/FP16 inputs and rows and dim multiples "
            "of 128"
        )
    dy, sfa = _dispatch_quantized_operand_placeholder(
        rows=rows,
        dim=dxy_dim,
        format=format,
        device=w.device,
        activation_buffer=activation_buffer,
    )
    if activation_buffer is None:
        dxy_col_q_storage = _empty_qdata(
            _column_quantized_storage_shape(rows, dxy_dim, format),
            format,
            w.device,
        )
        dxy_col_scale = torch.empty(
            _column_quantized_scale_shape(rows, dxy_dim, format),
            dtype=_torch_dtype(format.sf_dtype),
            device=w.device,
        )
    else:
        dxy_col_q_storage = _activation_buffer_placeholder(
            activation_buffer,
            _column_quantized_storage_shape(rows, dxy_dim, format),
            _torch_dtype(format.a_dtype),
        )
        dxy_col_scale = _activation_buffer_placeholder(
            activation_buffer,
            _column_quantized_scale_shape(rows, dxy_dim, format),
            _torch_dtype(format.sf_dtype),
        )
    _validate_2d_operand_layout(
        "DGRAD_SWIGLU_BWD dy",
        dy,
        format,
        contraction_axis=1,
    )
    _validate_3d_operand_layout("DGRAD_SWIGLU_BWD w", w, format, contraction_axis=1)
    out = _fused_dist_blockscaled_grouped_gemm_combine_impl(
        a=dy,
        b=w.transpose(1, 2),
        sfa=sfa,
        sfb=sfb,
        split_sizes=split_sizes,
        scatter_ptrs=scatter_ptrs,
        symm_mem_buffer=symm_mem_buffer,
        format=format,
        problem_type=_DGRAD,
        contraction_axes=(1, 2),
        c=dx,
        out_dtype=out_dtype,
        num_sms=num_sms,
        config=config,
        m_multiple_of=m_multiple_of,
        num_output_tokens=num_output_tokens,
        combine_swiglu_col_q_storage=dxy_col_q_storage,
        combine_swiglu_col_scale=dxy_col_scale,
        combine_swiglu_bwd_dz=grad_h2_2d,
        combine_swiglu_bwd_h1=h1_2d,
        combine_swiglu_fast_math=swiglu_fast_math,
        combine_swiglu_clamped=swiglu_clamped,
        combine_swiglu_alpha=swiglu_alpha,
        combine_swiglu_limit=swiglu_limit,
        combine_situ_glu=situ_glu,
        combine_situ_glu_gate_beta=situ_glu_gate_beta,
        combine_situ_glu_up_beta=situ_glu_up_beta,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
        conditional_execution=conditional_execution,
        pre_shuffled_rows=pre_shuffled_rows,
    )
    return out, (
        _column_quantized_output(dxy_col_q_storage, format),
        dxy_col_scale.view(-1),
    )
