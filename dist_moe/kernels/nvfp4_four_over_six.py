# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Four Over Six NVFP4 quantization of grouped expert weights.

Weight-only NVFP4 (W4A16) quantizes with Four Over Six (arXiv 2512.02010) in
Transformer Engine's NVFP4 arithmetic, configured as the W4A16 RL recipe of
radixark/miles does: each 1x16 block maps its largest magnitude to 4 or to 6,
whichever gives the lower squared error (candidate errors in FP16), with E4M3
block scales up to 448 and one FP32 global scale per expert,
``encode = 2688 / amax``.

The block arithmetic is miles' fused QDQ kernel (``_miles_nvfp4_qdq.py``).
These launches run it over a ``[G, N, K]`` expert stack in one grid of
``[G, ctas_per_expert]`` CTAs, each expert with its own global scale:

* ``four_over_six_qdq`` writes ``code * (block_scale * amax / 2688)`` in the
  input dtype, the fake-quantized weights of training;
* ``four_over_six_quantize`` writes the packed E2M1 codes and E4M3 scales that
  ``four_over_six_qdq`` dequantizes, the operands of inference.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from ._miles_nvfp4_qdq import (
    _4OVER6_BLOCKS_PER_SM,
    _4OVER6_THREADS,
    _block_amax,
    _dequantize_store,
    _fdiv_rn,
    _four_over_six_quantize,
    _FP4_BLOCK_SIZE,
    _get_ptr,
    _global_encode_scale,
    _input_values,
    _load_v4_u32,
    _NVFP4QDQSpecialization,
    _validate_input,
    NVFP4QDQConfig,
    NVFP4QDQErrorMode,
)

# Largest E4M3 block scale and E2M1 value; ``encode = 2688 / amax``.
_E4M3_MAX = 448
_E2M1_MAX = 6.0
# miles' W4A16 recipe: Four Over Six on every block, squared error in FP16.
_FOUR_OVER_SIX = NVFP4QDQConfig(
    use_4over6=True,
    e4m3_max=_E4M3_MAX,
    error_mode=NVFP4QDQErrorMode.MSE,
    error_use_fast_math=True,
)
# Resident CTAs per SM the grid is sized for (128 threads each).
_GRID_CTAS_PER_SM = 64


def four_over_six_global_scales(
    weight_GNK: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return each expert's amax, global encode scale and decode scale.

    ``encode = 2688 / amax`` (FP32 division) and ``decode = 1 / encode``, with
    an all-zero or overflowing expert falling back to one, as the kernels
    compute them.

    Args:
        weight_GNK: BF16 or FP16 expert weights ``[G, N, K]``.

    Returns:
        FP32 ``amax``, ``encode`` and ``decode``, each ``[G]``.
    """
    amax_G = torch.linalg.vector_norm(
        weight_GNK.flatten(1), ord=float("inf"), dim=-1, dtype=torch.float32
    )
    fp32_max = torch.finfo(torch.float32).max
    encode_G = (torch.full_like(amax_G, _E2M1_MAX * _E4M3_MAX) / amax_G).clamp(
        max=fp32_max
    )
    encode_G = torch.where((amax_G == 0) | (encode_G == 0), 1.0, encode_G)
    return amax_G, encode_G, 1.0 / encode_G


@dsl_user_op
def _store_v2_u32(base_ptr: Int64, v0: Uint32, v1: Uint32, *, loc=None, ip=None):
    """Store two 32-bit words to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_u8(base_ptr: Int64, value: Uint32, *, loc=None, ip=None):
    """Store the low byte of ``value`` to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u8 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _quantize_block(
    input_tensor: cute.Tensor,
    offset: Int32,
    amax: Float32,
    global_encode_scale: Float32,
    global_decode_scale: Float32,
    is_bfloat16: bool,
) -> tuple[Uint32, Uint32, Uint32]:
    """Load the 1x16 block at element ``offset`` and quantize it as miles does.

    Returns:
        The E4M3 scale bits and the block's eight packed code bytes as two
        words.
    """
    w0, w1, w2, w3 = _load_v4_u32(_get_ptr(input_tensor, offset))
    w4, w5, w6, w7 = _load_v4_u32(_get_ptr(input_tensor, offset + Int32(8)))
    words = (w0, w1, w2, w3, w4, w5, w6, w7)
    return _four_over_six_quantize(
        _input_values(words, is_bfloat16),
        _block_amax(words, is_bfloat16),
        amax,
        global_encode_scale,
        global_decode_scale,
        _FOUR_OVER_SIX,
    )


class _GroupedKernel:
    """Grid ``[G, ctas_per_expert]``; one thread quantizes one 1x16 block."""

    threads = _4OVER6_THREADS

    def __init__(self, is_bfloat16: bool) -> None:
        """Kernel for BF16 (``is_bfloat16``) or FP16 weights."""
        self.is_bfloat16 = is_bfloat16


class _GroupedQDQKernel(_GroupedKernel):
    """Writes each block's dequantized values in the input dtype."""

    @staticmethod
    def output_fakes(element_type) -> list:  # noqa: ANN001
        """Compile-time stand-ins of the output tensors."""
        return [
            cute.runtime.make_fake_compact_tensor(
                element_type, (cute.sym_int(),), assumed_align=16
            )
        ]

    @cute.jit
    def __call__(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        global_amaxes: cute.Tensor,
        blocks_per_expert: Int32,
        ctas_per_expert: Int32,
        num_experts: Int32,
        stream,  # noqa: ANN001
    ) -> None:
        """Launch one grid over all experts."""
        self.kernel(
            input_tensor, output_tensor, global_amaxes, blocks_per_expert
        ).launch(
            grid=[num_experts, ctas_per_expert, 1],
            block=[self.threads, 1, 1],
            max_number_threads=[self.threads, 1, 1],
            min_blocks_per_mp=_4OVER6_BLOCKS_PER_SM,
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        global_amaxes: cute.Tensor,
        blocks_per_expert: Int32,
    ) -> None:
        """Quantize and dequantize grid-stride blocks of expert ``blockIdx.x``."""
        thread_idx, _, _ = cute.arch.thread_idx()
        expert, cta_idx, _ = cute.arch.block_idx()
        _, ctas_per_expert, _ = cute.arch.grid_dim()

        amax = Float32(global_amaxes[expert])
        global_encode_scale = _global_encode_scale(amax, _E4M3_MAX)
        global_decode_scale = _fdiv_rn(Float32(1.0), global_encode_scale)
        expert_block = expert * blocks_per_expert
        block = cta_idx * Int32(self.threads) + thread_idx
        stride = ctas_per_expert * Int32(self.threads)
        while block < blocks_per_expert:
            offset = (expert_block + block) * Int32(_FP4_BLOCK_SIZE)
            scale, lo, hi = _quantize_block(
                input_tensor,
                offset,
                amax,
                global_encode_scale,
                global_decode_scale,
                self.is_bfloat16,
            )
            _dequantize_store(
                output_tensor, offset, lo, hi, scale, amax, _E4M3_MAX, self.is_bfloat16
            )
            block = block + stride


class _GroupedQuantizeKernel(_GroupedKernel):
    """Writes each block's packed E2M1 codes (8 bytes) and E4M3 scale (1 byte)."""

    @staticmethod
    def output_fakes(element_type) -> list:  # noqa: ANN001
        """Compile-time stand-ins of the output tensors."""
        del element_type
        return [
            cute.runtime.make_fake_compact_tensor(
                cutlass.Uint8, (cute.sym_int(),), assumed_align=8
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Uint8, (cute.sym_int(),), assumed_align=1
            ),
        ]

    @cute.jit
    def __call__(
        self,
        input_tensor: cute.Tensor,
        codes_tensor: cute.Tensor,
        scale_tensor: cute.Tensor,
        global_amaxes: cute.Tensor,
        blocks_per_expert: Int32,
        ctas_per_expert: Int32,
        num_experts: Int32,
        stream,  # noqa: ANN001
    ) -> None:
        """Launch one grid over all experts."""
        self.kernel(
            input_tensor, codes_tensor, scale_tensor, global_amaxes, blocks_per_expert
        ).launch(
            grid=[num_experts, ctas_per_expert, 1],
            block=[self.threads, 1, 1],
            max_number_threads=[self.threads, 1, 1],
            min_blocks_per_mp=_4OVER6_BLOCKS_PER_SM,
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_tensor: cute.Tensor,
        codes_tensor: cute.Tensor,
        scale_tensor: cute.Tensor,
        global_amaxes: cute.Tensor,
        blocks_per_expert: Int32,
    ) -> None:
        """Quantize grid-stride blocks of expert ``blockIdx.x``; store codes and scales."""
        thread_idx, _, _ = cute.arch.thread_idx()
        expert, cta_idx, _ = cute.arch.block_idx()
        _, ctas_per_expert, _ = cute.arch.grid_dim()

        amax = Float32(global_amaxes[expert])
        global_encode_scale = _global_encode_scale(amax, _E4M3_MAX)
        global_decode_scale = _fdiv_rn(Float32(1.0), global_encode_scale)
        expert_block = expert * blocks_per_expert
        block = cta_idx * Int32(self.threads) + thread_idx
        stride = ctas_per_expert * Int32(self.threads)
        while block < blocks_per_expert:
            global_block = expert_block + block
            scale, lo, hi = _quantize_block(
                input_tensor,
                global_block * Int32(_FP4_BLOCK_SIZE),
                amax,
                global_encode_scale,
                global_decode_scale,
                self.is_bfloat16,
            )
            _store_v2_u32(
                _get_ptr(codes_tensor, global_block * Int32(_FP4_BLOCK_SIZE // 2)),
                lo,
                hi,
            )
            _store_u8(_get_ptr(scale_tensor, global_block), scale)
            block = block + stride


_COMPILED: dict[tuple[Any, ...], _NVFP4QDQSpecialization] = {}


def _compiled(
    kernel_cls: type[_GroupedKernel],
    dtype: torch.dtype,
    capability: tuple[int, int],
) -> _NVFP4QDQSpecialization:
    """Compile (once per kernel, dtype and architecture) a grouped launch.

    Raises:
        RuntimeError: On a first use during CUDA graph capture.
    """
    key = (kernel_cls, capability, dtype)
    specialization = _COMPILED.get(key)
    if specialization is not None:
        return specialization
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Warm up the Four Over Six NVFP4 kernels before CUDA graph capture."
        )
    kernel = kernel_cls(dtype == torch.bfloat16)
    element_type = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    compiled = cute.compile(
        kernel,
        cute.runtime.make_fake_compact_tensor(
            element_type, (cute.sym_int(),), assumed_align=16
        ),
        *kernel.output_fakes(element_type),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(),), assumed_align=4
        ),
        Int32(1),
        Int32(1),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    specialization = _NVFP4QDQSpecialization(
        launch=compiled, threads=kernel.threads, grid_blocks_per_sm=_GRID_CTAS_PER_SM
    )
    _COMPILED[key] = specialization
    return specialization


def _check_output(
    out: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    weight: torch.Tensor,
    alignment: int,
) -> None:
    """Validate one caller-provided output tensor.

    Raises:
        ValueError: If its shape, dtype, device, contiguity or alignment is wrong.
    """
    if tuple(out.shape) != shape or out.dtype != dtype or out.device != weight.device:
        raise ValueError(
            f"expected a {dtype} output of shape {shape} on {weight.device}, got "
            f"{out.dtype} {tuple(out.shape)} on {out.device}"
        )
    if not out.is_contiguous() or out.data_ptr() % alignment:
        raise ValueError(
            f"Four Over Six outputs must be contiguous and {alignment}-byte aligned"
        )


def _launch(
    kernel_cls: type[_GroupedKernel],
    weight_GNK: torch.Tensor,
    amax_G: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    """Launch one grouped kernel over a contiguous ``[G, N, K]`` stack.

    Raises:
        ValueError: If the weights are not rank 3 or fail miles' input contract
            (contiguous BF16/FP16 on an SM10x GPU, K a multiple of 16).
    """
    if weight_GNK.ndim != 3:
        raise ValueError(
            "Four Over Six quantization requires grouped [G, N, K] weights, got "
            f"shape {tuple(weight_GNK.shape)}"
        )
    device_index, capability, multiprocessors, num_elements = _validate_input(
        weight_GNK.view(-1, weight_GNK.shape[-1]), amax_G[:1]
    )
    with torch.cuda.device(device_index):
        specialization = _compiled(kernel_cls, weight_GNK.dtype, capability)
        num_experts = weight_GNK.shape[0]
        blocks_per_expert = num_elements // num_experts // _FP4_BLOCK_SIZE
        total_ctas = multiprocessors * specialization.grid_blocks_per_sm
        ctas_per_expert = min(
            (blocks_per_expert + specialization.threads - 1) // specialization.threads,
            max(1, (total_ctas + num_experts - 1) // num_experts),
        )
        specialization.launch(
            weight_GNK.detach().view(-1),
            *(out.view(-1) for out in outputs),
            amax_G,
            blocks_per_expert,
            ctas_per_expert,
            num_experts,
        )


def four_over_six_qdq(
    weight_GNK: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Quantize and dequantize expert weights with Four Over Six.

    Args:
        weight_GNK: Contiguous CUDA BF16 or FP16 expert weights ``[G, N, K]``,
            K a multiple of 16.
        out: Optional contiguous, 16-byte aligned output like ``weight_GNK``.

    Returns:
        ``code * (block_scale * amax / 2688)`` per element, in the input dtype.

    Raises:
        ValueError: If the weights or ``out`` violate the contract above.
    """
    if out is None:
        out = torch.empty_like(weight_GNK, memory_format=torch.contiguous_format)
    _check_output(out, tuple(weight_GNK.shape), weight_GNK.dtype, weight_GNK, 16)
    amax_G, _, _ = four_over_six_global_scales(weight_GNK)
    _launch(_GroupedQDQKernel, weight_GNK, amax_G, (out,))
    return out


def four_over_six_quantize(
    weight_GNK: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize expert weights with Four Over Six.

    Args:
        weight_GNK: Contiguous CUDA BF16 or FP16 expert weights ``[G, N, K]``,
            K a multiple of 16.

    Returns:
        uint8 codes ``[G, N, K / 2]`` (two E2M1 codes per byte, the even column
        in the low nibble) and uint8 E4M3 scale bits ``[G, N, K / 16]`` in
        natural order: the operands ``four_over_six_qdq`` dequantizes. Then
        the FP32 global encode and decode scales ``[G]``.

    Raises:
        ValueError: If the weights violate the contract above.
    """
    num_experts, num_rows, num_cols = weight_GNK.shape
    codes_GNQ = weight_GNK.new_empty(
        (num_experts, num_rows, num_cols // 2), dtype=torch.uint8
    )
    scale_GNB = weight_GNK.new_empty(
        (num_experts, num_rows, num_cols // _FP4_BLOCK_SIZE), dtype=torch.uint8
    )
    amax_G, encode_G, decode_G = four_over_six_global_scales(weight_GNK)
    _launch(_GroupedQuantizeKernel, weight_GNK, amax_G, (codes_GNQ, scale_GNB))
    return codes_GNQ, scale_GNB, encode_G, decode_G
