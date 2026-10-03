# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for generating dynamic Triton kernel repr names with dtype info.

Provides a factory function `make_dtype_repr` that creates repr callables for
`@triton.jit(repr=...)`. The generated repr appends pointer parameter dtype
suffixes to the kernel name, making it easy to identify kernel specializations
in profilers and debuggers.

Example:
    repr=make_dtype_repr("_triton_rmsnorm_fwd", ["ptr_x", "ptr_y"])
    # produces: "_triton_rmsnorm_fwd_x_bf16_y_fp32" at runtime
"""

from typing import Callable, Sequence, Union


def _clean_param_name(name: str) -> str:
    """Strip ptr_ prefix or _ptr suffix from parameter name."""
    if name.startswith("ptr_"):
        return name[4:]
    if name.endswith("_ptr"):
        return name[:-4]
    return name


def _resolve_dtype_str(signature_dtype: str, fallback_dtype_str: str | None) -> str:
    """Resolve a dtype string from a signature entry, with optional fallback."""
    dtype_str = signature_dtype
    if dtype_str.startswith("*"):
        dtype_str = dtype_str[1:]
    if dtype_str == "i64" and fallback_dtype_str is not None:
        dtype_str = fallback_dtype_str
    return dtype_str


def _get_fallback_dtype(specialization, dtype_constexpr: str | None) -> str | None:
    """Extract fallback dtype string from a constexpr parameter."""
    if dtype_constexpr is None:
        return None
    constants = specialization.constants
    if dtype_constexpr in constants:
        return constants[dtype_constexpr].name
    return None


def make_dtype_repr(
    base_name: Union[str, Callable],
    ptr_params: Sequence[str],
    dtype_constexpr: str | None = None,
) -> Callable:
    """Factory creating a repr function that appends pointer dtype suffixes.

    Args:
        base_name: Either a string kernel name or a callable(specialization) -> str
            for kernels that need dynamic base names (e.g., rope's fwd/bwd).
        ptr_params: List of pointer parameter names whose dtypes to include.
        dtype_constexpr: Optional name of a tl.constexpr parameter (e.g., "DTYPE")
            to use as fallback when a pointer param is an int64 offset tensor
            (activation buffer mode) instead of a typed pointer.

    Returns:
        A callable suitable for `@triton.jit(repr=...)`.

    Example:
        >>> repr_fn = make_dtype_repr("_triton_rmsnorm_fwd", ["ptr_x", "ptr_y"])
        # At runtime with bf16 inputs and fp32 outputs:
        # repr_fn(specialization) -> "_triton_rmsnorm_fwd_x_bf16_y_fp32"

        >>> repr_fn = make_dtype_repr("_triton_kernel", ["dy_ptr", "dx_ptr"], dtype_constexpr="DTYPE")
        # When pointers are int64 offsets (activation buffer mode), falls back to DTYPE:
        # repr_fn(specialization) -> "_triton_kernel_dy_bf16_dx_bf16"
    """

    def _repr(specialization) -> str:
        name = base_name(specialization) if callable(base_name) else base_name
        fallback = _get_fallback_dtype(specialization, dtype_constexpr)

        parts = []
        for param in ptr_params:
            if param not in specialization.signature:
                continue
            dtype_str = _resolve_dtype_str(specialization.signature[param], fallback)
            parts.append(f"{_clean_param_name(param)}_{dtype_str}")

        if parts:
            name = name + "_" + "_".join(parts)
        return name

    return _repr
