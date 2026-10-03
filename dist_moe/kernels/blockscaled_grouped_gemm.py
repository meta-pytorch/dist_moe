# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL block-scaled grouped GEMM (MXFP8 / NVFP4 / MXFP4) for dist_moe.

This persistent, warp-specialized SM100 kernel extends the BF16/FP16 grouped
GEMM with two block-scaled requirements: (1) the MMA path uses Blackwell's
tcgen05 variant and reads SFA/SFB from TMEM; (2) the TMA producer loads SFA/SFB
before A/B onto the same combined ``ab_full_mbar``.

Per-CTA warp layout (8 total): 4 epilog + 1 MMA + 1 TMA + 2 idle.

Glossary:

* **BM / BN / BK** — ``BLOCK_SIZE_M / N / K`` in the cfg dict.
* **OV** (``OVERLAPPING_ACCUM``) — seam-sharing trick that recovers a
  second TMEM acc stage at BN=256 by overlapping the two stages' last SF
  cols. Coordinated through ``cross_seam_mbar``.
* **SWAP_AB** — swap MMA-A / MMA-B operand roles so the 2cta cluster
  multicast lands on W. BLOCK_M then tiles W's N-axis; BLOCK_N tiles
  activations' M-axis.
* **SFA / SFB** — per-operand scale factors, 1 byte per ``sf_vec_size``
  operand elements.
"""

import threading

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

from ..formats import canonical_swiglu_clamp
from . import sm103_blockscaled_helpers as sm103
from ._environment import num_sms_per_device
from ._kernel_name_prefix import scoped_kernel_name_prefixes
from .activation_buffer import (
    GEMM_OPERAND_OFFSET_COUNT,
    validate_activation_buffer_launch,
)

# Re-exported for the many existing importers of this module; the
# format-aware configuration layer itself lives in blockscaled_config.
from .blockscaled_config import (  # noqa: F401
    _as_blockscaled_format_spec,
    _BLOCKSCALED_FORMAT_SPECS,
    _copy_config,
    _cutlass_dtype_from_torch,
    _format_has_fp4_a,
    _format_uses_fp4,
    _FP4_COMPUTE_CONFIG,
    _FP4_FALLBACK_CONFIG,
    _FP4_FORMAT_NAMES,
    _FP4_MEMORY_CONFIG,
    _host_descriptor_required_mn_multiples,
    _host_tensormaps_available,
    _is_a8w4_format,
    _is_config_compatible,
    _MIXED_FORMAT_CACHE,
    _MXFP4_MEGA_DECODE_CONFIG,
    _MXFP4_T6_FC2_PREFILL_CONFIG,
    _MXFP4_T6_FC2_PREFILL_CONTRACTION,
    _MXFP8_COMPUTE_CONFIG,
    _MXFP8_FALLBACK_CONFIG,
    _MXFP8_FORMAT_NAMES,
    _MXFP8_MEGA_DECODE_CONFIG,
    _MXFP8_MEMORY_CONFIG,
    _NVFP4_MEGA_DECODE_CONFIG,
    _NVFP4_T6_PREFILL_CONTRACTIONS,
    _pick_literal_config,
    _require_config_compatible,
    _require_decode_dtype,
    _require_supported_weight_dtype,
    _resolve_weight_format,
    _retune_config_for_formats,
    auto_blockscaled_config,
    blockscaled_mega_decode_config,
    blockscaled_staged_decode_config,
    blockscaled_training_config,
    build_blockscaled_config,
    inference_fprop_static_scheduler,
    make_mixed_blockscaled_format,
    MXFP4,
    MXFP8_E5M2,
    NVFP4,
    PRODUCTION_BLOCKSCALED_CONFIGS,
)
from .blockscaled_grouped_gemm_kernel import (  # noqa: F401
    _format_has_fp4_b,
    _LaunchTensorBundle,
    _ParsedGemmProblem,
    BlockScaledFormatSpec,
    BlockScaledGroupedGemmKernel,
    BlockScaledProductionConfig,
    MXFP8_E4M3,
)
from .config import (  # noqa: F401
    _MXFP4_STAGED_COMPACT_MIN_FEATURE_WAVES,
    BLOCKSCALED_GROUPED_GEMM_M_ALIGNMENT,
    DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    derive_blockscaled_grouped_gemm_config,
    derive_blockscaled_inference_geometry,
    EPILOGUE_SUBTILE_AUTO,
    initialize_blockscaled_inference_pipeline,
    STATIC_FPROP_SCHEDULER_SHAPES,
)

# Reuse the bf16 grouped_gemm helpers so both kernels stay on the same
# scheduler + mbarrier protocol.
from .grouped_gemm import (
    _DGRAD,
    _FPROP,
    _WGRAD,
)
from .params import (
    _torch_dtype,
    BlockscaledPointerStrideArgs,
)

# ---------------------------------------------------------------------------
# Kernel cache + compile helpers.
# ---------------------------------------------------------------------------


_KERNEL_CACHE: dict = {}
_COMPILED_CACHE: dict = {}
_COMPILE_LOCK = threading.Lock()


def _config_cache_key(cfg: dict) -> tuple:
    """Hashable signature of a cfg dict for cache lookups."""
    return tuple(sorted(cfg.items()))


def _set_swiglu_clamp_config(
    cfg: dict, clamped: bool, alpha: float, limit: float
) -> None:
    """Write the clamp triple into `cfg` when clamped.

    Plain SwiGLU leaves the keys unset so its cache key does not change.
    """
    clamped, alpha, limit = canonical_swiglu_clamp(clamped, alpha, limit)
    if clamped:
        cfg["COMBINE_SWIGLU_CLAMPED"] = True
        cfg["COMBINE_SWIGLU_ALPHA"] = alpha
        cfg["COMBINE_SWIGLU_LIMIT"] = limit


def _get_kernel(
    config: dict,
    problem_type: int,
    format: BlockScaledFormatSpec,
    force_n_major: bool,
    num_n_clusters: int,
    world_size: int,
) -> "BlockScaledGroupedGemmKernel":
    """Cached kernel lookup. The cfg is trusted to be format-final
    (FP4 paths must go through ``build_blockscaled_config(format=...)``)."""
    key = (
        _config_cache_key(config),
        problem_type,
        format.name,
        force_n_major,
        num_n_clusters,
        world_size,
    )
    inst = _KERNEL_CACHE.get(key)
    if inst is None:
        inst = BlockScaledGroupedGemmKernel.from_config(
            config,
            problem_type=problem_type,
            format=format,
            force_n_major=force_n_major,
            num_n_clusters=num_n_clusters,
            world_size=world_size,
        )
        _KERNEL_CACHE[key] = inst
    return inst


def _torch_layout_signature(t: torch.Tensor):
    return (str(t.dtype), tuple(int(s == 1) for s in t.stride()))


def _compile_or_get(
    cache_key: tuple,
    kernel,
    *args,
    name_prefix: str | None = None,
    **kwargs,
):
    compiled = _COMPILED_CACHE.get(cache_key)
    if compiled is not None:
        return compiled
    with _COMPILE_LOCK:
        compiled = _COMPILED_CACHE.get(cache_key)
        if compiled is not None:
            return compiled
        if _format_uses_fp4(kernel.format):
            options = kwargs.get("options") or ""
            if "--enable-tvm-ffi" not in options:
                kwargs["options"] = f"{options} --enable-tvm-ffi".strip()
        if name_prefix is not None:
            kernel_cls = type(kernel)
            prepare_name_prefix = name_prefix.replace(
                "grouped_gemm_", "grouped_gemm_prepare_", 1
            )
            # Kernel descriptors are class attributes and may be inherited;
            # set them immediately before compile so the baked trace name is
            # correct even when subclasses share the descriptor object.
            prefixes = tuple(
                (getattr(kernel_cls, attr), prefix)
                for attr, prefix in (
                    ("kernel", name_prefix),
                    ("combine_kernel", name_prefix),
                    ("dispatch_kernel", name_prefix),
                    ("dispatch_bprop_kernel", name_prefix),
                    ("prepare_kernel", prepare_name_prefix),
                    ("prepare_dispatch_bprop_kernel", prepare_name_prefix),
                )
                if hasattr(kernel_cls, attr)
            )
        else:
            prefixes = ()
        with scoped_kernel_name_prefixes(prefixes):
            compiled = cute.compile(kernel, *args, **kwargs)
        _COMPILED_CACHE[cache_key] = compiled
    return compiled


def _logical_shape_and_stride(
    operand: torch.Tensor,
    format: BlockScaledFormatSpec,
    *,
    matrix_axes: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = [int(dim) for dim in operand.shape]
    stride = tuple(int(s) for s in operand.stride())
    if _format_uses_fp4(format):
        for axis in matrix_axes:
            if stride[axis] == 1:
                shape[axis] *= 2
                break
    return tuple(shape), stride


def _logical_stride(
    operand: torch.Tensor,
    format: BlockScaledFormatSpec,
    axis: int,
) -> int:
    stride = int(operand.stride(axis))
    if _format_uses_fp4(format) and stride != 1:
        return stride * 2
    return stride


def _parse_2d_operand_shapes(
    name: str,
    operand: torch.Tensor,
    format: BlockScaledFormatSpec,
    *,
    contraction_axis: int,
) -> tuple[int, int]:
    if operand.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(operand.shape)}")
    if contraction_axis not in (0, 1):
        raise ValueError(
            f"{name} contraction_axis must be 0 or 1, got {contraction_axis}"
        )
    shape, _ = _logical_shape_and_stride(operand, format, matrix_axes=(0, 1))
    outer_axis = 1 - contraction_axis
    return shape[outer_axis], shape[contraction_axis]


def _parse_3d_operand_shapes(
    name: str,
    operand: torch.Tensor,
    format: BlockScaledFormatSpec,
    *,
    contraction_axis: int,
) -> tuple[int, int, int]:
    if operand.dim() != 3:
        raise ValueError(f"{name} must be 3D, got shape={tuple(operand.shape)}")
    if contraction_axis not in (1, 2):
        raise ValueError(
            f"{name} contraction_axis must be 1 or 2, got {contraction_axis}"
        )
    shape, _ = _logical_shape_and_stride(operand, format, matrix_axes=(1, 2))
    outer_axis = 1 if contraction_axis == 2 else 2
    return shape[0], shape[outer_axis], shape[contraction_axis]


def _allocate_workspace(
    num_clusters: int,
    NUM_CTAS: int,
    device: torch.device,
    *,
    static_scheduler: bool = False,
    min_rows: int = 0,
    zero_in_prepare: bool = False,
):
    """Persistent counter + per-CTA tensormap workspace (5 descriptors:
    A, B, SFA, SFB, C — two more than the bf16 kernel)."""
    grid_size = max(num_clusters * NUM_CTAS, min_rows)
    counter = (
        torch.empty(1, dtype=torch.int32, device=device)
        if static_scheduler or zero_in_prepare
        else torch.zeros(1, dtype=torch.int32, device=device)
    )
    tensormaps = torch.empty((grid_size, 5, 16), dtype=torch.int64, device=device)
    return counter, tensormaps


def _validate_blockscaled_launch_inputs(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
    split_size_multiple_of: int,
    split_size_alignment: int = 128,
) -> None:
    weight_format = _resolve_weight_format(format, weight_format)
    if format not in (MXFP8_E4M3, MXFP8_E5M2, MXFP4, NVFP4):
        raise ValueError(
            f"format must be MXFP8_E4M3, MXFP8_E5M2, MXFP4, NVFP4 (got {format}); "
        )
    if weight_format not in (MXFP8_E4M3, MXFP8_E5M2, MXFP4, NVFP4):
        raise ValueError(
            "weight_format must be MXFP8_E4M3, MXFP8_E5M2, MXFP4, NVFP4 "
            f"(got {weight_format}); "
        )
    make_mixed_blockscaled_format(format, weight_format)
    if split_size_alignment <= 0 or split_size_alignment % 32 != 0:
        raise ValueError(
            "split_size_alignment must be a positive multiple of 32, "
            f"got {split_size_alignment}."
        )
    if split_size_multiple_of % split_size_alignment != 0:
        raise ValueError(
            "split_size_multiple_of must be divisible by "
            f"{split_size_alignment}, "
            f"got {split_size_multiple_of}."
        )

    for name, t in (
        ("a", a),
        ("b", b),
        ("sfa", sfa),
        ("sfb", sfb),
        ("split_sizes", split_sizes),
    ):
        if not t.is_cuda:
            raise ValueError(f"{name} must reside on CUDA device, got {t.device}")
        if t.device != a.device:
            raise ValueError(
                f"all inputs must share a.device={a.device}; {name}.device={t.device}"
            )

    a_dtype = _torch_dtype(format.a_dtype)
    b_dtype = _torch_dtype(weight_format.b_dtype)
    if a.dtype != a_dtype or b.dtype != b_dtype:
        raise ValueError(
            f"A/B dtype mismatch for format={format.name}: expected "
            f"a.dtype={a_dtype}, b.dtype={b_dtype}; "
            f"got a.dtype={a.dtype}, b.dtype={b.dtype}"
        )
    sf_dtype = _torch_dtype(format.sf_dtype)
    weight_sf_dtype = _torch_dtype(weight_format.sf_dtype)
    if sfa.dtype != sf_dtype or sfb.dtype != weight_sf_dtype:
        raise ValueError(
            f"SFA/SFB dtype mismatch for format={format.name}, "
            f"weight_format={weight_format.name}: expected "
            f"sfa.dtype={sf_dtype}, sfb.dtype={weight_sf_dtype}; "
            f"got sfa.dtype={sfa.dtype}, sfb.dtype={sfb.dtype}"
        )


def _validate_nvfp4_epilogue_scales(
    *,
    a_global_scale_inv: torch.Tensor | None,
    b_global_scale_inv: torch.Tensor | None,
    format: BlockScaledFormatSpec,
    problem: _ParsedGemmProblem,
    problem_type: int,
    device: torch.device,
) -> None:
    if a_global_scale_inv is None and b_global_scale_inv is None:
        return
    if format != NVFP4:
        raise ValueError("A/B global epilogue scales are supported only for NVFP4")
    if problem_type == _WGRAD:
        raise NotImplementedError(
            "per-row/per-group epilogue scaling is not valid for WGRAD"
        )
    for name, scale, expected_size in (
        ("a_global_scale_inv", a_global_scale_inv, problem.GM),
        ("b_global_scale_inv", b_global_scale_inv, problem.G),
    ):
        if scale is None:
            continue
        if scale.shape != (expected_size,) or scale.dtype != torch.float32:
            raise ValueError(
                f"NVFP4 {name} must be float32 shape ({expected_size},), got "
                f"{tuple(scale.shape)} {scale.dtype}"
            )
        if scale.device != device:
            raise ValueError(f"NVFP4 {name} must be on {device}, got {scale.device}")
        if not scale.is_contiguous():
            raise ValueError(f"NVFP4 {name} must be contiguous")


def _parse_blockscaled_problem(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
    split_sizes: torch.Tensor,
    contraction_axes: tuple[int, int],
    problem_type: int,
) -> _ParsedGemmProblem:
    weight_format = _resolve_weight_format(format, weight_format)
    assert len(contraction_axes) == 2, (
        f"contraction_axes must be a 2-tuple, got {contraction_axes}"
    )
    a_outer, K_a = _parse_2d_operand_shapes(
        "a",
        a,
        format,
        contraction_axis=contraction_axes[0],
    )

    if b.dim() == 3:
        if problem_type == _WGRAD:
            raise ValueError("WGRAD expects a 2D B operand")
        G, N, K_b = _parse_3d_operand_shapes(
            "b",
            b,
            weight_format,
            contraction_axis=contraction_axes[1],
        )
        c_shape = (a_outer, N)
    elif b.dim() == 2:
        if problem_type != _WGRAD:
            raise ValueError("2D B operands are only supported for WGRAD")
        N, K_b = _parse_2d_operand_shapes(
            "b",
            b,
            weight_format,
            contraction_axis=contraction_axes[1],
        )
        G = int(split_sizes.shape[0])
        c_shape = (G, a_outer, N)
    else:
        raise ValueError(f"b must be 2D or 3D, got shape={tuple(b.shape)}")

    if K_a != K_b:
        raise ValueError(f"K_a={K_a} != K_b={K_b}")
    return _ParsedGemmProblem(G=G, GM=a_outer, N=N, K=K_a, c_shape=c_shape)


def _resolve_output_tensor(
    *,
    c: torch.Tensor | None,
    c_shape: tuple[int, ...],
    out_dtype: torch.dtype | None,
    device: torch.device,
    problem_type: int,
    output_accum: bool,
) -> tuple[torch.Tensor, torch.dtype]:
    if c is not None:
        if out_dtype is not None and c.dtype != out_dtype:
            raise ValueError(f"c.dtype={c.dtype} does not match out_dtype={out_dtype}")
        out_dtype = c.dtype
    elif output_accum:
        raise ValueError("output_accum=True requires an existing output tensor")
    elif out_dtype is None:
        out_dtype = torch.bfloat16

    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError(f"out_dtype={out_dtype} not supported.")
    if out_dtype == torch.float32 and problem_type != _WGRAD:
        raise NotImplementedError("float32 output is only supported for WGRAD.")
    if c is None:
        c = torch.empty(c_shape, dtype=out_dtype, device=device)
    elif tuple(c.shape) != c_shape:
        raise ValueError(f"c.shape={tuple(c.shape)} does not match expected {c_shape}")
    return c, out_dtype


def _make_operand_compile_tensor(
    dtype: type[cutlass.Numeric],
    *,
    leading_dim: int,
) -> object:
    shape = [
        cute.sym_int(divisibility=1),
        cute.sym_int(divisibility=1),
        cute.sym_int(divisibility=1),
    ]
    stride = [
        cute.sym_int64(divisibility=2 if dtype == cutlass.Float4E2M1FN else 1),
        cute.sym_int64(divisibility=2 if dtype == cutlass.Float4E2M1FN else 1),
        cute.sym_int64(divisibility=2 if dtype == cutlass.Float4E2M1FN else 1),
    ]
    stride[leading_dim] = 1
    if dtype == cutlass.Float4E2M1FN:
        shape[leading_dim] = cute.sym_int(divisibility=2)
    return cute.runtime.make_fake_tensor(
        dtype,
        tuple(shape),
        stride=tuple(stride),
        assumed_align=16,
    )


def _make_blockscaled_compile_tensors(
    *,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec,
    out_dtype: torch.dtype,
    a_leading_dim: int,
    b_leading_dim: int,
    split_cute,
    counter_cute,
    tensormaps_cute,
) -> tuple:
    a_compile = _make_operand_compile_tensor(
        format.a_dtype,
        leading_dim=a_leading_dim,
    )
    b_compile = _make_operand_compile_tensor(
        weight_format.b_dtype,
        leading_dim=b_leading_dim,
    )
    combined = make_mixed_blockscaled_format(format, weight_format)
    sf_dtype_cute = combined.sf_dtype
    c_compile = cute.runtime.make_fake_tensor(
        _cutlass_dtype_from_torch(out_dtype),
        (cute.sym_int(), cute.sym_int(), cute.sym_int()),
        stride=(cute.sym_int64(), 1, cute.sym_int64()),
        assumed_align=16,
    )
    sfa_compile = cute.runtime.make_fake_tensor(
        sf_dtype_cute,
        (cute.sym_int(),),
        stride=(1,),
        assumed_align=16,
    )
    sfb_compile = cute.runtime.make_fake_tensor(
        sf_dtype_cute,
        (cute.sym_int(),),
        stride=(1,),
        assumed_align=16,
    )
    return (
        a_compile,
        b_compile,
        c_compile,
        sfa_compile,
        sfb_compile,
        split_cute,
        counter_cute,
        tensormaps_cute,
    )


def _make_launch_tensor_bundle(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    counter: torch.Tensor,
    tensormaps: torch.Tensor,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
    out_dtype: torch.dtype,
) -> _LaunchTensorBundle:
    weight_format = _resolve_weight_format(format, weight_format)
    is_fp4_op = _format_uses_fp4(format) or _format_uses_fp4(weight_format)
    a_placeholder = a.detach().unsqueeze(-1)
    b_placeholder = (
        b.detach().permute(1, 2, 0) if b.dim() == 3 else b.detach().unsqueeze(-1)
    )
    c_placeholder = (
        c.detach().permute(1, 2, 0) if c.dim() == 3 else c.detach().unsqueeze(-1)
    )

    # Auto-detect the stride-1 axis so callers can pass strided views (e.g.
    # WGRAD's ``dy.t()`` / ``x.t()`` without a materialized ``.contiguous()``
    # copy). FPROP/DGRAD operands are contiguous K-major → leading_dim=1;
    # WGRAD's transposed view is M-major → leading_dim=0.
    a_leading_dim = next(i for i, s in enumerate(a_placeholder.stride()) if s == 1)
    b_leading_dim = next(i for i, s in enumerate(b_placeholder.stride()) if s == 1)
    c_leading_dim = next(i for i, s in enumerate(c_placeholder.stride()) if s == 1)
    dlpack_kwargs = {"enable_tvm_ffi": is_fp4_op}
    a_cute = from_dlpack(
        a_placeholder, assumed_align=16, **dlpack_kwargs
    ).mark_layout_dynamic(leading_dim=a_leading_dim)
    b_cute = from_dlpack(
        b_placeholder, assumed_align=16, **dlpack_kwargs
    ).mark_layout_dynamic(leading_dim=b_leading_dim)
    c_cute = from_dlpack(
        c_placeholder, assumed_align=16, **dlpack_kwargs
    ).mark_layout_dynamic(leading_dim=c_leading_dim)
    sfa_cute = from_dlpack(sfa.detach(), assumed_align=16, **dlpack_kwargs)
    sfb_cute = from_dlpack(sfb.detach(), assumed_align=16, **dlpack_kwargs)
    split_cute = from_dlpack(split_sizes.detach(), assumed_align=4, **dlpack_kwargs)
    counter_cute = from_dlpack(counter, assumed_align=4, **dlpack_kwargs)
    tensormaps_cute = from_dlpack(tensormaps, assumed_align=8, **dlpack_kwargs)
    runtime_tensors = (
        a_cute,
        b_cute,
        c_cute,
        sfa_cute,
        sfb_cute,
        split_cute,
        counter_cute,
        tensormaps_cute,
    )
    compile_tensors = (
        _make_blockscaled_compile_tensors(
            format=format,
            weight_format=weight_format,
            out_dtype=out_dtype,
            a_leading_dim=a_leading_dim,
            b_leading_dim=b_leading_dim,
            split_cute=split_cute,
            counter_cute=counter_cute,
            tensormaps_cute=tensormaps_cute,
        )
        if is_fp4_op
        else runtime_tensors
    )
    return _LaunchTensorBundle(
        placeholders=(a_placeholder, b_placeholder, c_placeholder),
        compile_tensors=compile_tensors,
        runtime_tensors=runtime_tensors,
    )


def _make_pointer_stride_args(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
    N: int,
    K: int,
) -> tuple[BlockscaledPointerStrideArgs, tuple[int, int, int]]:
    weight_format = _resolve_weight_format(format, weight_format)
    make_mixed_blockscaled_format(format, weight_format)
    elem_sizes = (a.element_size(), b.element_size(), c.element_size())
    sf_cols = K // format.sf_vec_size
    a_strides = (
        cutlass.Int32(_logical_stride(a, format, 0)),
        cutlass.Int32(_logical_stride(a, format, 1)),
        cutlass.Int64(0),
    )

    if b.dim() == 3:
        b_s0 = _logical_stride(b, weight_format, 1)
        b_s1 = _logical_stride(b, weight_format, 2)
        b_group_stride = b.stride(0)
    else:
        b_s0 = _logical_stride(b, weight_format, 0)
        b_s1 = _logical_stride(b, weight_format, 1)
        b_group_stride = 0

    if c.dim() == 3:
        c_s0, c_s1, c_group_stride = c.stride(1), c.stride(2), c.stride(0)
    else:
        c_s0, c_s1, c_group_stride = c.stride(0), c.stride(1), 0

    return (
        (
            (
                cutlass.Int64(a.data_ptr()),
                cutlass.Int64(b.data_ptr()),
                cutlass.Int64(c.data_ptr()),
                cutlass.Int64(sfa.data_ptr()),
                cutlass.Int64(sfb.data_ptr()),
            ),
            (
                a_strides,
                (
                    cutlass.Int32(b_s0),
                    cutlass.Int32(b_s1),
                    cutlass.Int64(b_group_stride),
                ),
                (
                    cutlass.Int32(c_s0),
                    cutlass.Int32(c_s1),
                    cutlass.Int64(c_group_stride),
                ),
            ),
            (cutlass.Int32(sf_cols), cutlass.Int32(N * sf_cols)),
        ),
        elem_sizes,
    )


def _blockscaled_grouped_gemm(  # noqa: C901
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    format: BlockScaledFormatSpec,
    weight_format: BlockScaledFormatSpec | None = None,
    split_sizes: torch.Tensor,
    split_size_multiple_of: int,
    contraction_axes: tuple[int, int],
    problem_type: int = _FPROP,
    out_dtype: torch.dtype | None = None,
    num_sms: int | None = None,
    config: dict | None = None,
    output_accum: bool = False,
    use_device_tensormaps: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    a_global_scale_inv: torch.Tensor | None = None,
    b_global_scale_inv: torch.Tensor | None = None,
) -> torch.Tensor:
    weight_format = _resolve_weight_format(format, weight_format)
    kernel_format = make_mixed_blockscaled_format(format, weight_format)
    use_activation_buffer = activation_buffer is not None
    if use_activation_buffer:
        if activation_offsets is None:
            raise ValueError(
                "activation_buffer requires an activation_offsets tensor with "
                f"{GEMM_OPERAND_OFFSET_COUNT} elements"
            )
        validate_activation_buffer_launch(
            activation_buffer=activation_buffer,
            activation_offsets=activation_offsets,
            expected_offset_count=GEMM_OPERAND_OFFSET_COUNT,
            device=a.device,
        )
        use_device_tensormaps = True
    elif activation_offsets is not None:
        raise ValueError("activation_offsets requires activation_buffer")
    _validate_blockscaled_launch_inputs(
        a=a,
        b=b,
        sfa=sfa,
        sfb=sfb,
        split_sizes=split_sizes,
        format=format,
        weight_format=weight_format,
        split_size_multiple_of=split_size_multiple_of,
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
    _validate_nvfp4_epilogue_scales(
        a_global_scale_inv=a_global_scale_inv,
        b_global_scale_inv=b_global_scale_inv,
        format=format,
        problem=problem,
        problem_type=problem_type,
        device=a.device,
    )
    c, out_dtype = _resolve_output_tensor(
        c=c,
        c_shape=problem.c_shape,
        out_dtype=out_dtype,
        device=a.device,
        problem_type=problem_type,
        output_accum=output_accum,
    )
    if split_sizes.dtype != torch.int32:
        split_sizes = split_sizes.to(torch.int32)
    if num_sms is None:
        num_sms = num_sms_per_device()

    if config is None:
        config = auto_blockscaled_config(
            GM=problem.GM,
            G=problem.G,
            N=problem.N,
            K=problem.K,
            num_sms=num_sms,
            format=format,
            weight_format=weight_format,
            m_multiple_of=split_size_multiple_of,
            problem_type=problem_type,
            enable_sm103_ultra=(
                torch.cuda.get_device_capability(a.device)
                == sm103.SM103_COMPUTE_CAPABILITY
            ),
        )
    cfg = config
    if cfg.get("USE_SM103_ULTRA", False):
        capability = torch.cuda.get_device_capability(a.device)
        if capability != sm103.SM103_COMPUTE_CAPABILITY:
            raise ValueError(
                f"SM103 ultra MMA requires compute capability 10.3, got {capability}"
            )
        if format not in (NVFP4, MXFP4):
            raise ValueError("SM103 ultra MMA supports only NVFP4 or MXFP4")
        if problem_type == _WGRAD:
            raise ValueError("SM103 ultra MMA does not support WGRAD")
    if not use_device_tensormaps and not _host_tensormaps_available(
        cfg,
        problem_type=problem_type,
        m_multiple_of=split_size_multiple_of,
        N=problem.N,
        G=problem.G,
    ):
        use_device_tensormaps = True
    num_clusters = max(1, num_sms // cfg["NUM_CTAS"])

    counter, tensormaps = _allocate_workspace(
        num_clusters,
        cfg["NUM_CTAS"],
        a.device,
        static_scheduler=bool(cfg.get("STATIC_SCHEDULER", False)),
        min_rows=problem.G if use_device_tensormaps else 0,
        zero_in_prepare=use_device_tensormaps,
    )

    # Single-rank defaults; dist-MoE tile-coord branches compile-elide.
    force_n_major = False
    num_n_clusters = 1
    local_rank = 0
    world_size = 1
    kernel = _get_kernel(
        config=cfg,
        problem_type=problem_type,
        format=kernel_format,
        force_n_major=force_n_major,
        num_n_clusters=num_n_clusters,
        world_size=world_size,
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
    if use_activation_buffer:
        assert activation_buffer is not None and activation_offsets is not None
        activation_buffer_base_ptr = activation_buffer.data_ptr()
        activation_buffer_size_bytes = activation_buffer.nbytes
        activation_offsets_tensor = activation_offsets
    else:
        activation_buffer_base_ptr = 0
        activation_buffer_size_bytes = 0
        # The disabled offset path is compile-elided; reuse existing launch storage.
        assert tensormaps.numel() >= GEMM_OPERAND_OFFSET_COUNT, (
            "tensormap buffer too small for offset aliasing"
        )
        activation_offsets_tensor = tensormaps.view(-1).narrow(
            0, 0, GEMM_OPERAND_OFFSET_COUNT
        )
    activation_offsets_cute = from_dlpack(
        activation_offsets_tensor,
        assumed_align=8,
        enable_tvm_ffi=_format_uses_fp4(kernel_format),
    )
    use_a_global_scale_inv = a_global_scale_inv is not None
    use_b_global_scale_inv = b_global_scale_inv is not None
    a_global_scale_inv_ptr = (
        0 if a_global_scale_inv is None else a_global_scale_inv.data_ptr()
    )
    b_global_scale_inv_ptr = (
        0 if b_global_scale_inv is None else b_global_scale_inv.data_ptr()
    )
    stream = cutlass_torch.current_stream()
    compile_args = launch_tensors.compile_tensors + (
        pointer_stride_args,
        elem_sizes,
        problem.G,
        problem.GM,
        problem.N,
        problem.K,
        local_rank,
        num_clusters,
        output_accum,
        use_device_tensormaps,
        activation_buffer_base_ptr,
        cutlass.Int64(activation_buffer_size_bytes),
        activation_offsets_cute,
        use_activation_buffer,
        cutlass.Int64(a_global_scale_inv_ptr),
        cutlass.Int64(b_global_scale_inv_ptr),
        use_a_global_scale_inv,
        use_b_global_scale_inv,
        stream,
    )
    runtime_args = launch_tensors.runtime_tensors + (
        pointer_stride_args,
        problem.GM,
        problem.N,
        problem.K,
        local_rank,
        num_clusters,
        activation_buffer_base_ptr,
        cutlass.Int64(activation_buffer_size_bytes),
        activation_offsets_cute,
        cutlass.Int64(a_global_scale_inv_ptr),
        cutlass.Int64(b_global_scale_inv_ptr),
        stream,
    )
    a_placeholder, b_placeholder, c_placeholder = launch_tensors.placeholders
    cache_key = (
        _config_cache_key(cfg),
        problem_type,
        kernel_format.name,
        force_n_major,
        num_n_clusters,
        world_size,
        _torch_layout_signature(a_placeholder),
        _torch_layout_signature(b_placeholder),
        _torch_layout_signature(c_placeholder),
        problem.G,
        problem.N,
        problem.K,
        elem_a,
        elem_b,
        elem_c,
        output_accum,
        use_device_tensormaps,
        use_activation_buffer,
        use_a_global_scale_inv,
        use_b_global_scale_inv,
    )
    _PROBLEM_NAMES = {
        _FPROP: "fprop",
        _DGRAD: "dgrad",
        _WGRAD: "wgrad",
    }
    name_prefix = f"_cute_blockscaled_grouped_gemm_{_PROBLEM_NAMES[problem_type]}"
    compiled = _compile_or_get(
        cache_key, kernel, *compile_args, name_prefix=name_prefix
    )
    compiled(*runtime_args)
    return c


def _validate_2d_operand_layout(
    name: str,
    operand: torch.Tensor,
    format: BlockScaledFormatSpec,
    *,
    contraction_axis: int,
) -> None:
    if operand.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(operand.shape)}")

    contraction_axis = contraction_axis % operand.dim()
    if contraction_axis not in (0, 1):
        raise ValueError(
            f"{name} contraction_axis must be one of the matrix axes 0 or 1, "
            f"got {contraction_axis}."
        )
    if _format_uses_fp4(format):
        if operand.stride(contraction_axis) != 1:
            raise ValueError(
                f"{name} FP4 operand must be K-major for its GEMM role: expected "
                f"stride({contraction_axis}) == 1, got shape={tuple(operand.shape)}, "
                f"stride={tuple(operand.stride())}."
            )
    elif operand.stride(0) != 1 and operand.stride(1) != 1:
        raise ValueError(
            f"{name} FP8 operand must have stride(0) == 1 or stride(1) == 1; "
            f"got shape={tuple(operand.shape)}, stride={tuple(operand.stride())}."
        )


def _validate_3d_operand_layout(
    name: str,
    operand: torch.Tensor,
    format: BlockScaledFormatSpec,
    *,
    contraction_axis: int,
) -> None:
    if operand.dim() != 3:
        raise ValueError(f"{name} must be 3D, got shape={tuple(operand.shape)}")

    contraction_axis = contraction_axis % operand.dim()
    if contraction_axis not in (1, 2):
        raise ValueError(
            f"{name} contraction_axis must be one of the matrix axes 1 or 2, "
            f"got {contraction_axis}."
        )
    matrix_axes = (1, 2)
    if _format_uses_fp4(format):
        if operand.stride(contraction_axis) != 1:
            raise ValueError(
                f"{name} FP4 operand must be K-major for its GEMM role: expected "
                f"stride({contraction_axis}) == 1, got shape={tuple(operand.shape)}, "
                f"stride={tuple(operand.stride())}."
            )
    elif all(operand.stride(axis) != 1 for axis in matrix_axes):
        raise ValueError(
            f"{name} FP8 operand must have stride(1) == 1 or stride(2) == 1; "
            f"got shape={tuple(operand.shape)}, stride={tuple(operand.stride())}."
        )


def blockscaled_grouped_gemm_fprop(
    x: torch.Tensor,  # [GM, K]  FP8/FP4
    w: torch.Tensor,  # [G, N, K] FP8/FP4 (K-major)
    sfa: torch.Tensor,  # [GM, K // sf_vec_size] atom-tiled
    sfb: torch.Tensor,  # [G, N, K // sf_vec_size] atom-tiled
    split_sizes: torch.Tensor,  # [G]
    *,
    y: torch.Tensor | None = None,  # [GM, N]
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    weight_format: BlockScaledFormatSpec | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    use_device_tensormaps: bool = False,
    a_global_scale_inv: torch.Tensor | None = None,
    b_global_scale_inv: torch.Tensor | None = None,
) -> torch.Tensor:
    """FPROP: ``y[g] = x[g] @ w[g].T`` with block scales."""
    weight_format = _resolve_weight_format(format, weight_format)
    _validate_2d_operand_layout(
        "FPROP x",
        x,
        format,
        contraction_axis=1,
    )
    _validate_3d_operand_layout(
        "FPROP w",
        w,
        weight_format,
        contraction_axis=2,
    )
    return _blockscaled_grouped_gemm(
        a=x,
        b=w,
        sfa=sfa,
        sfb=sfb,
        split_sizes=split_sizes,
        split_size_multiple_of=m_multiple_of,
        c=y,
        format=format,
        weight_format=weight_format,
        out_dtype=out_dtype,
        num_sms=num_sms,
        config=config,
        contraction_axes=(1, 2),
        use_device_tensormaps=use_device_tensormaps,
        a_global_scale_inv=a_global_scale_inv,
        b_global_scale_inv=b_global_scale_inv,
    )


def blockscaled_grouped_gemm_dgrad(
    dy: torch.Tensor,  # [GM, N] FP8/FP4
    w: torch.Tensor,  # [G, N, K] logical weights
    sfa: torch.Tensor,  # [GM, N // sf_vec_size] atom-tiled
    sfb: torch.Tensor,  # [G, K, N // sf_vec_size] atom-tiled
    split_sizes: torch.Tensor,  # [G]
    *,
    dx: torch.Tensor | None = None,  # [GM, K]
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    use_device_tensormaps: bool = False,
    a_global_scale_inv: torch.Tensor | None = None,
    b_global_scale_inv: torch.Tensor | None = None,
) -> torch.Tensor:
    """DGRAD: ``dx[g] = dy[g] @ w[g]`` with block scales."""
    _validate_2d_operand_layout(
        "DGRAD dy",
        dy,
        format,
        contraction_axis=1,
    )
    _validate_3d_operand_layout(
        "DGRAD w",
        w,
        format,
        contraction_axis=1,
    )
    return _blockscaled_grouped_gemm(
        a=dy,
        b=w.transpose(1, 2),
        sfa=sfa,
        sfb=sfb,
        split_sizes=split_sizes,
        split_size_multiple_of=m_multiple_of,
        c=dx,
        format=format,
        problem_type=_DGRAD,
        out_dtype=out_dtype,
        num_sms=num_sms,
        config=config,
        contraction_axes=(1, 2),
        use_device_tensormaps=use_device_tensormaps,
        a_global_scale_inv=a_global_scale_inv,
        b_global_scale_inv=b_global_scale_inv,
    )


def blockscaled_grouped_gemm_wgrad(
    dy: torch.Tensor,  # [N, GM] FP8 or [N, GM/2] packed FP4
    x: torch.Tensor,  # [K, GM] FP8 or [K, GM/2] packed FP4
    sfa: torch.Tensor,  # flat per-g [N, m_g // sf_vec_size] atom-tiled
    sfb: torch.Tensor,  # flat per-g [K, m_g // sf_vec_size] atom-tiled
    split_sizes: torch.Tensor,  # [G]
    *,
    dw: torch.Tensor | None = None,  # [G, N, K]
    format: BlockScaledFormatSpec = MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    output_accum: bool = False,
    m_multiple_of: int = DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """WGRAD: ``dw[g] = dy[g].T @ x[g]`` with block scales.

    ``dy`` and ``x`` are already laid out with the token dimension as
    the contraction axis. The public interface prepares that layout once
    outside the timed call.
    """
    _validate_2d_operand_layout(
        "WGRAD dy.T",
        dy,
        format,
        contraction_axis=1,
    )
    _validate_2d_operand_layout(
        "WGRAD x.T",
        x,
        format,
        contraction_axis=1,
    )
    return _blockscaled_grouped_gemm(
        a=dy,
        b=x,
        sfa=sfa,
        sfb=sfb,
        split_sizes=split_sizes,
        split_size_multiple_of=m_multiple_of,
        c=dw,
        format=format,
        problem_type=_WGRAD,
        out_dtype=out_dtype,
        num_sms=num_sms,
        config=config,
        contraction_axes=(1, 1),
        output_accum=output_accum,
        use_device_tensormaps=True,
        activation_buffer=activation_buffer,
        activation_offsets=activation_offsets,
    )
