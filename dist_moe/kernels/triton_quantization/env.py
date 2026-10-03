# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Environment definitions required by the Triton quantizer."""

import triton

TRITON_VERSION: tuple[int, ...] = tuple(
    int(v) for v in triton.__version__.split(".")[:2]
)


_IS_CUDA: bool = False


def is_cuda() -> bool:
    """Return the fixed backend predicate used by the retained quantizer.

    The public NVFP4 path uses the identity producer, so it deliberately does
    not initialize the optional TMA producer path.

    Returns:
        ``False`` for the supported identity-producer path.
    """
    return _IS_CUDA
