# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import triton
import triton.language as tl


@triton.jit
def device_trap_if(condition):
    """Trap lanes where condition is true without requiring Triton debug mode."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred failed;
            setp.ne.u32 failed, $1, 0;
            @failed trap;
            mov.u32 $0, 0;
        }
        """,
        "=r,r",
        [condition.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
