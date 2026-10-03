# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# Licensed under the Apache License, Version 2.0.
# Modified by Meta Platforms, Inc.

import cutlass.cute as cute
from cutlass import Int32


def expand(a: cute.Tensor, dim: int, size: Int32 | int) -> cute.Tensor:
    shape = (*a.shape[:dim], size, *a.shape[dim:])
    stride = (*a.layout.stride[:dim], 0, *a.layout.stride[dim:])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))
