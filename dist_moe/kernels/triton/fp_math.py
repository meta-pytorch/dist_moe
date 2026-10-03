# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PTX Floating-Point Math Operations via Inline Assembly.

This module provides Triton inline assembly wrappers for all 10 PTX
floating-point math instructions with different precision/performance
trade-offs.

Division modes (div):
- div.approx.f32: Fast approximate divide
- div.full.f32: Full-range approximate divide
- div.rnd.f32: IEEE 754 compliant rounding for f32
- div.rn.f64: IEEE 754 compliant round-to-nearest division for f64

Reciprocal modes (rcp):
- rcp.approx.f32: Fast approximate reciprocal
- rcp.rnd.f32: IEEE 754 compliant rounding for f32

FMA (fused multiply-add):
- fma.rn.f32: IEEE 754 compliant fused multiply-add for f32


Square root (sqrt):
- sqrt.approx{.ftz}.f32: Fast approximate sqrt for f32
- sqrt.rnd{.ftz}.f32: IEEE 754 compliant rounding for f32
- sqrt.rnd.f64: IEEE 754 compliant rounding for f64

Reciprocal square root (rsqrt):
- rsqrt.approx{.ftz}.f32: Fast approximate rsqrt for f32
- rsqrt.approx.f64: Fast approximate rsqrt for f64

Sine (sin):
- sin.approx{.ftz}.f32: Fast approximate sine for f32

Cosine (cos):
- cos.approx{.ftz}.f32: Fast approximate cosine for f32

Log base 2 (lg2):
- lg2.approx{.ftz}.f32: Fast approximate log2 for f32

Exponential base 2 (ex2):
- ex2.approx{.ftz}.f32: Fast approximate exp2 for f32

Hyperbolic tangent (tanh):
- tanh.approx.f32: Fast approximate tanh for f32 (sm_75+, no .ftz support)

Rounding modes (.rnd):
- .rn: Round to nearest even
- .rz: Round toward zero
- .rm: Round toward negative infinity (floor)
- .rp: Round toward positive infinity (ceil)

Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sqrt
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sin
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-cos
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-lg2
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2
Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

# =============================================================================
# NaN-propagating reductions
# =============================================================================


@triton.jit
def max_propagate_nan(a, b):
    """`tl.maximum` with NaN propagation, usable as a `tl.reduce` combine fn.

    maxNum-based `tl.max` / `tl.maximum` silently DROP NaN, so amax-based
    quantizers that must poison NaN inputs need a propagating max — the
    resulting scale goes NaN and the block dequantizes non-finite instead of
    encoding the NaN as a plausible finite code. Lowers to `max.NaN.f32`
    (same throughput as `max.f32`), unlike the `tl.where(x != x, inf, |x|)`
    substitution idiom inlined in `quant/blockscaled_quantize.py`, which
    costs an extra compare+select per element (~21% on the NVFP4 transport
    quantizer). The CuTe twin is `abs_max_nan_f32` in
    `cute/activation/swiglu_fwbw.py`.
    """
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


# =============================================================================
# Fast Approximate Division (div.approx.f32)
# =============================================================================


@triton.jit
def div_approx_f32(a, b):
    """Fast approximate division for f32.

    Uses PTX: div.approx.f32 d, a, b;

    This is the fastest but least accurate division mode.
    Subnormal inputs and results are flushed to zero.
    """
    return tl.inline_asm_elementwise(
        asm="div.approx.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_approx_ftz_f32(a, b):
    """Fast approximate division for f32 with flush-to-zero.

    Uses PTX: div.approx.ftz.f32 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.approx.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# Full-Range Approximate Division (div.full.f32)
# =============================================================================


@triton.jit
def div_full_f32(a, b):
    """Full-range approximate division for f32.

    Uses PTX: div.full.f32 d, a, b;

    More accurate than div.approx, handles larger range of values.
    """
    return tl.inline_asm_elementwise(
        asm="div.full.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_full_ftz_f32(a, b):
    """Full-range approximate division for f32 with flush-to-zero.

    Uses PTX: div.full.ftz.f32 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.full.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# IEEE 754 Compliant Division - f32 (div.rnd.f32)
# =============================================================================


@triton.jit
def div_rn_f32(a, b):
    """IEEE 754 compliant division for f32, round to nearest even.

    Uses PTX: div.rn.f32 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rn_f64(a, b):
    """IEEE 754 compliant division for f64, round to nearest even.

    Uses PTX: div.rn.f64 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.rn.f64 $0, $1, $2;",
        constraints="=d,d,d",
        args=[a, b],
        dtype=tl.float64,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rn_ftz_f32(a, b):
    """IEEE 754 compliant division for f32, round to nearest even, flush-to-zero.

    Uses PTX: div.rn.ftz.f32 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.rn.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rz_f32(a, b):
    """IEEE 754 compliant division for f32, round toward zero.

    Uses PTX: div.rz.f32 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.rz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rz_ftz_f32(a, b):
    """IEEE 754 compliant division for f32, round toward zero, flush-to-zero.

    Uses PTX: div.rz.ftz.f32 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.rz.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rm_f32(a, b):
    """IEEE 754 compliant division for f32, round toward negative infinity.

    Uses PTX: div.rm.f32 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.rm.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rm_ftz_f32(a, b):
    """IEEE 754 compliant division for f32, round toward -inf, flush-to-zero.

    Uses PTX: div.rm.ftz.f32 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.rm.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rp_f32(a, b):
    """IEEE 754 compliant division for f32, round toward positive infinity.

    Uses PTX: div.rp.f32 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.rp.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def div_rp_ftz_f32(a, b):
    """IEEE 754 compliant division for f32, round toward +inf, flush-to-zero.

    Uses PTX: div.rp.ftz.f32 d, a, b;
    """
    return tl.inline_asm_elementwise(
        asm="div.rp.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# Fast Approximate Reciprocal (rcp.approx.f32)
# =============================================================================


@triton.jit
def rcp_approx_f32(a):
    """Fast approximate reciprocal for f32.

    Uses PTX: rcp.approx.f32 d, a;

    This is the fastest but least accurate reciprocal mode.
    """
    return tl.inline_asm_elementwise(
        asm="rcp.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rcp_approx_ftz_f32(a):
    """Fast approximate reciprocal for f32 with flush-to-zero.

    Uses PTX: rcp.approx.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# IEEE 754 Compliant Reciprocal - f32 (rcp.rnd.f32)
# =============================================================================


@triton.jit
def rcp_rn_f32(a):
    """IEEE 754 compliant reciprocal for f32, round to nearest even.

    Uses PTX: rcp.rn.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rcp.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rcp_rn_ftz_f32(a):
    """IEEE 754 compliant reciprocal for f32, round to nearest even, flush-to-zero.

    Uses PTX: rcp.rn.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rcp.rn.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rcp_rz_f32(a):
    """IEEE 754 compliant reciprocal for f32, round toward zero.

    Uses PTX: rcp.rz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rcp.rz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rcp_rz_ftz_f32(a):
    """IEEE 754 compliant reciprocal for f32, round toward zero, flush-to-zero.

    Uses PTX: rcp.rz.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rcp.rz.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rcp_rm_f32(a):
    """IEEE 754 compliant reciprocal for f32, round toward negative infinity.

    Uses PTX: rcp.rm.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rcp.rm.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rcp_rm_ftz_f32(a):
    """IEEE 754 compliant reciprocal for f32, round toward -inf, flush-to-zero.

    Uses PTX: rcp.rm.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rcp.rm.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rcp_rp_f32(a):
    """IEEE 754 compliant reciprocal for f32, round toward positive infinity.

    Uses PTX: rcp.rp.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rcp.rp.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rcp_rp_ftz_f32(a):
    """IEEE 754 compliant reciprocal for f32, round toward +inf, flush-to-zero.

    Uses PTX: rcp.rp.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rcp.rp.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# Square Root (sqrt)
#
# sqrt.approx{.ftz}.f32: Fast approximate, f32 only
# sqrt.rnd{.ftz}.f32: IEEE 754 compliant rounding, f32
# sqrt.rnd.f64: IEEE 754 compliant rounding, f64
#
# Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sqrt
# =============================================================================


@triton.jit
def sqrt_approx_f32(a):
    """Fast approximate square root for f32.

    Uses PTX: sqrt.approx.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_approx_ftz_f32(a):
    """Fast approximate square root for f32 with flush-to-zero.

    Uses PTX: sqrt.approx.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rn_f32(a):
    """IEEE 754 compliant square root for f32, round to nearest even.

    Uses PTX: sqrt.rn.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rn_ftz_f32(a):
    """IEEE 754 compliant square root for f32, round to nearest even, flush-to-zero.

    Uses PTX: sqrt.rn.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rn.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rz_f32(a):
    """IEEE 754 compliant square root for f32, round toward zero.

    Uses PTX: sqrt.rz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rz_ftz_f32(a):
    """IEEE 754 compliant square root for f32, round toward zero, flush-to-zero.

    Uses PTX: sqrt.rz.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rz.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rm_f32(a):
    """IEEE 754 compliant square root for f32, round toward negative infinity.

    Uses PTX: sqrt.rm.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rm.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rm_ftz_f32(a):
    """IEEE 754 compliant square root for f32, round toward -inf, flush-to-zero.

    Uses PTX: sqrt.rm.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rm.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rp_f32(a):
    """IEEE 754 compliant square root for f32, round toward positive infinity.

    Uses PTX: sqrt.rp.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rp.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rp_ftz_f32(a):
    """IEEE 754 compliant square root for f32, round toward +inf, flush-to-zero.

    Uses PTX: sqrt.rp.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rp.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rn_f64(a):
    """IEEE 754 compliant square root for f64, round to nearest even.

    Uses PTX: sqrt.rn.f64 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rn.f64 $0, $1;",
        constraints="=d,d",
        args=[a],
        dtype=tl.float64,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rz_f64(a):
    """IEEE 754 compliant square root for f64, round toward zero.

    Uses PTX: sqrt.rz.f64 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rz.f64 $0, $1;",
        constraints="=d,d",
        args=[a],
        dtype=tl.float64,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rm_f64(a):
    """IEEE 754 compliant square root for f64, round toward negative infinity.

    Uses PTX: sqrt.rm.f64 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rm.f64 $0, $1;",
        constraints="=d,d",
        args=[a],
        dtype=tl.float64,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sqrt_rp_f64(a):
    """IEEE 754 compliant square root for f64, round toward positive infinity.

    Uses PTX: sqrt.rp.f64 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sqrt.rp.f64 $0, $1;",
        constraints="=d,d",
        args=[a],
        dtype=tl.float64,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# IEEE 754 Compliant Fused Multiply-Add (fma.rnd.f32)
# =============================================================================


@triton.jit
def fma_rn_f32(a, b, c):
    """IEEE 754 compliant fused multiply-add for f32, round to nearest even.

    Uses PTX: fma.rn.f32 d, a, b, c;

    Computes d = a * b + c with a single rounding at the end.
    The intermediate product a * b is computed in infinite precision.
    """
    return tl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def fma_rn_ftz_f32(a, b, c):
    """IEEE 754 compliant fused multiply-add for f32, round to nearest even, flush-to-zero.

    Uses PTX: fma.rn.ftz.f32 d, a, b, c;
    """
    return tl.inline_asm_elementwise(
        asm="fma.rn.ftz.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# Fast Approximate Reciprocal Square Root (rsqrt.approx)
#
# Note: rsqrt.approx.ftz.f64 does not exist in the PTX ISA; the .ftz modifier
# is only valid for .f32 operands.
#
# Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt
# =============================================================================


@triton.jit
def rsqrt_approx_f32(a):
    """Fast approximate reciprocal square root for f32.

    Uses PTX: rsqrt.approx.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rsqrt.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rsqrt_approx_ftz_f32(a):
    """Fast approximate reciprocal square root for f32 with flush-to-zero.

    Uses PTX: rsqrt.approx.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rsqrt.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def rsqrt_approx_f64(a):
    """Fast approximate reciprocal square root for f64.

    Uses PTX: rsqrt.approx.f64 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="rsqrt.approx.f64 $0, $1;",
        constraints="=d,d",
        args=[a],
        dtype=tl.float64,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# Sine (sin) — approx only, f32 only
#
# Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sin
# =============================================================================


@triton.jit
def sin_approx_f32(a):
    """Fast approximate sine for f32.

    Uses PTX: sin.approx.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sin.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sin_approx_ftz_f32(a):
    """Fast approximate sine for f32 with flush-to-zero.

    Uses PTX: sin.approx.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="sin.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# Cosine (cos) — approx only, f32 only
#
# Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-cos
# =============================================================================


@triton.jit
def cos_approx_f32(a):
    """Fast approximate cosine for f32.

    Uses PTX: cos.approx.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="cos.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def cos_approx_ftz_f32(a):
    """Fast approximate cosine for f32 with flush-to-zero.

    Uses PTX: cos.approx.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="cos.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# Log Base 2 (lg2) — approx only, f32 only
#
# Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-lg2
# =============================================================================


@triton.jit
def lg2_approx_f32(a):
    """Fast approximate log base 2 for f32.

    Uses PTX: lg2.approx.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="lg2.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def lg2_approx_ftz_f32(a):
    """Fast approximate log base 2 for f32 with flush-to-zero.

    Uses PTX: lg2.approx.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="lg2.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# Exponential Base 2 (ex2) — approx only, f32 only
#
# Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2
# =============================================================================


@triton.jit
def ex2_approx_f32(a):
    """Fast approximate exponential base 2 for f32.

    Uses PTX: ex2.approx.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="ex2.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def ex2_approx_ftz_f32(a):
    """Fast approximate exponential base 2 for f32 with flush-to-zero.

    Uses PTX: ex2.approx.ftz.f32 d, a;
    """
    return tl.inline_asm_elementwise(
        asm="ex2.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# Hyperbolic Tangent (tanh) — approx only, f32 only, sm_75+
#
# Note: tanh does not support the .ftz modifier in the PTX ISA.
#
# Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-tanh
# =============================================================================


@triton.jit
def tanh_approx_f32(a):
    """Fast approximate hyperbolic tangent for f32.

    Uses PTX: tanh.approx.f32 d, a;

    Requires sm_75 or higher.
    """
    return tl.inline_asm_elementwise(
        asm="tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


# =============================================================================
# Sigmoid Function (IEEE 754 Compliant)
# =============================================================================


@triton.jit
def sigmoid(x, FAST_MATH: tl.constexpr = False):
    """Sigmoid function with default IEEE-oriented math and optional fast math.

    Computes: sigmoid(x) = 1 / (1 + exp(-x))

    This implementation uses libdevice.exp for numerical stability instead of
    tl.sigmoid which uses tl.exp (ex2.approx without proper range reduction).

    Benefits of libdevice.exp:
    - Proper handling of denormalized numbers
    - Overflow/underflow saturation
    - Extended range via exponent extraction
    - Calls ex2.approx with pre-processed inputs using range reduction
      and polynomial approximation

    Uses rcp_rn_f32 for IEEE-754 compliant reciprocal (round to nearest even).

    Args:
        x: Input tensor (Triton tensor). Will be cast to float32 internally
           because libdevice.exp only accepts float32 inputs.

    Returns:
        Sigmoid of x in float32.

    Note:
        This function achieves numeric parity with PyTorch and TransformerEngine (TE).
    """
    x = x.to(tl.float32)
    if FAST_MATH:
        return rcp_approx_ftz_f32(1.0 + tl.exp(-x))
    return rcp_rn_f32(1.0 + libdevice.exp(-x))
