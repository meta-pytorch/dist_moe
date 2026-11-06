# BF16 and NVFP4 Training Kernel Flow

This document compares the BF16 training path with the implemented NVFP4
training recipe. Each chart aligns corresponding BF16 and NVFP4 phases
horizontally:

1. staged NVFP4 forward and its backward;
2. Mega NVFP4 forward and its backward.

NVFP4 uses real FP4 tensor-core GEMMs in forward. Its backward regenerates the
quantized forward operands, exactly dequantizes them to BF16, and then uses the
shared BF16 DGRAD and WGRAD kernels. No gradient GEMM has an FP4 input.

## Notation and storage

```text
T   local input tokens
K   routes per token
M   planned received-row capacity; only the routed prefix is valid
E   local experts
P   expert-parallel ranks
D   hidden dimension
F   intermediate dimension
S   router-score dtype: BF16 or FP32
WG  configured WGRAD output dtype: BF16 or FP32

[P]    persistent activation-slot storage retained for backward
[S]    reusable activation-buffer scratch
[SYM]  peer-addressable symmetric HBM
[OUT]  returned autograd output

q(a_bf16)       operation that quantizes a BF16 tensor
dq(a_nvfp4)     operation that exactly dequantizes an NVFP4 bundle
a_nvfp4         complete NVFP4 tensor bundle: qdata and every required scale
a_q_dq_bf16     BF16 tensor produced by dq(a_nvfp4)

pad128(n) = ceil(n / 128) * 128
pad4(n)   = ceil(n / 4) * 4
```

Every NVFP4 name in the charts denotes the complete bundle:

| Tensor | Logical shape | Complete physical bundle |
| --- | --- | --- |
| `x_nvfp4` | `[T,D]` | E2M1x2 qdata `[T,D/2]`; natural E4M3 block scale `[T,D/16]`; FP32 inverse global scale `[T]` |
| `x_gathered_nvfp4` | `[M,D]` | pointer-selected rows of `x_nvfp4`; each row carries qdata `[D/2]`, block scale `[D/16]`, and one FP32 inverse global scale |
| `h2_nvfp4` | `[M,F]` | E2M1x2 qdata `[M,F/2]`; blocked E4M3 scale `[pad128(M),pad4(F/16)]`; FP32 inverse global scale `[M]` |
| `w13_nvfp4` | `[E,2F,D]` | E2M1x2 qdata `[E,2F,D/2]`; blocked E4M3 scale `[E*pad128(2F),pad4(D/16)]`; FP32 global scale and inverse `[E]` |
| `w2_nvfp4` | `[E,D,F]` | E2M1x2 qdata `[E,D,F/2]`; blocked E4M3 scale `[E*pad128(D),pad4(F/16)]`; FP32 global scale and inverse `[E]` |

The charts show the default score multiplication and top-k reduction
postprocess. A configured postprocess replaces those boxes while preserving
the expert-kernel and activation-save boundaries.

## Chart 1: staged execution

### Forward: BF16 versus staged NVFP4

```text
BF16 FORWARD                                                                      || NVFP4 STAGED FORWARD
==================================================================================++==================================================================================
                                                                                  ||
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| INPUTS                                                                        | || | INPUTS                                                                        |
|   x_bf16             BF16        [T,D]                                        | || |   x_bf16             BF16        [T,D]                                        |
|   w13_bf16           BF16        [E,2F,D]                                     | || |   w13_bf16           BF16        [E,2F,D]                                     |
|   w2_bf16            BF16        [E,D,F]                                      | || |   w2_bf16            BF16        [E,D,F]                                      |
|   topk_ids           INT32/INT64 [T,K]                                        | || |   topk_ids           INT32/INT64 [T,K]                                        |
|   topk_scores        S           [T,K]                                        | || |   topk_scores        S           [T,K]                                        |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| NO PREPARATION KERNEL                                                         | || | KERNEL GROUP: prepare dynamic NVFP4 weights                                   |
| INPUTS                                                                        | || | OP                                                                            |
|   w13_bf16 BF16 [E,2F,D]; w2_bf16 BF16 [E,D,F]                                | || |   w13_nvfp4 = q(w13_bf16); w2_nvfp4 = q(w2_bf16)                              |
| OUTPUTS                                                                       | || | INPUTS                                                                        |
|   the same native BF16 weight operands                                        | || |   w13_bf16 BF16 [E,2F,D]; w2_bf16 BF16 [E,D,F]                                |
| NOTE: BF16 kernels consume the caller weights directly                        | || | OUTPUTS                                                                       |
|                                                                               | || |   w13_nvfp4 NVFP4 bundle logical [E,2F,D], transient                          |
|                                                                               | || |   w2_nvfp4  NVFP4 bundle logical [E,D,F], transient                           |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL GROUP: copy_routing_and_dispatch                                       | || | KERNEL GROUP: copy routing + _stage_blockscaled_dispatch                      |
| INPUTS                                                                        | || | OP                                                                            |
|   x_bf16 BF16 [T,D]; topk_ids INT32/INT64 [T,K]                               | || |   x_nvfp4 = q(x_bf16); publish the complete row bundle                        |
| OUTPUTS                                                                       | || | INPUTS                                                                        |
|   dispatch_x_bf16 BF16 [T,D] in [SYM] dispatch HBM                            | || |   x_bf16 BF16 [T,D]; topk_ids INT32/INT64 [T,K]                               |
|   routing_ids INT16 [T,K]; routing_token_count INT32 [1]                      | || | OUTPUTS                                                                       |
|                                                                               | || |   x_nvfp4 NVFP4 bundle logical [T,D] in [SYM] dispatch HBM                    |
|                                                                               | || |   routing_ids INT16 [T,K]; routing_token_count INT32 [1]                      |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| DEVICE SYNC: expert-parallel dispatch barrier                                 | || | DEVICE SYNC: expert-parallel dispatch barrier                                 |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   BF16 dispatch and routing publication state                                 | || |   packed dispatch and routing publication state                               |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   peer-visible BF16 rows and route IDs; no tensor output                      | || |   peer-visible x_nvfp4 rows and route IDs; no tensor output                   |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL GROUP: dist_dispatch_routing                                           | || | KERNEL GROUP: dist_dispatch_routing                                           |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   topk_ids INT32/INT64 [T,K]; peer BF16 row bases [P]                         | || |   topk_ids INT32/INT64 [T,K]; peer packed-row bases [P]                       |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   expert_counts INT32 [E]; recv_by_rank INT32 [P]; recv_count INT32 [1]       | || |   expert_counts INT32 [E]; recv_by_rank INT32 [P]; recv_count INT32 [1]       |
|   fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs INT64 [M] each               | || |   fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs INT64 [M] each               |
| SAVE: counts and pointer tables cross autograd                                | || | SAVE: counts and pointer tables cross autograd                                |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: get_forward_plan                                                      | || | KERNEL: get_forward_plan                                                      |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   recv_count INT32 [1]; recv_by_rank INT32 [P]; slot_id INT64 [1]             | || |   recv_count INT32 [1]; recv_by_rank INT32 [P]; slot_id INT64 [1]             |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   need_recompute BOOL [1]; forward_state INT64 [6]                            | || |   need_recompute BOOL [1]; recompute_condition INT32 [1]                      |
|   recv_snapshot INT32 [P]                                                     | || |   forward_offsets INT64 [9]; activation_offsets INT64 [18]                    |
| SAVE: decision, offsets, physical slot ID, and snapshot                       | || |   recv_snapshot INT32 [P]                                                     |
|                                                                               | || | SAVE: decision, offsets, physical slot ID, predicates, and snapshot           |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: copy_dispatch_to_activation (device-predicated)                       | || | 2 PREDICATE-SELECTED KERNELS: save the chosen X payload                       |
| INPUTS                                                                        | || | OPS                                                                           |
|   dispatch_x_bf16 BF16 [T,D]; need_recompute BOOL [1]                         | || |   RECOMPUTE: x_saved_bf16 = copy(x_bf16)                                      |
| OUTPUTS                                                                       | || |   FULL-SAVE: x_q_dq_bf16 = dq(x_gathered_nvfp4)                               |
|   RECOMPUTE: x_saved_bf16 BF16 [T,D] in [P]                                   | || | INPUTS                                                                        |
|   FULL-SAVE: no activation write                                              | || |   x_bf16 BF16 [T,D]; x_gathered_nvfp4 bundle logical [M,D]                    |
|                                                                               | || | OUTPUTS                                                                       |
|                                                                               | || |   RECOMPUTE: x_saved_bf16 BF16 [T,D] in [P]                                   |
|                                                                               | || |   FULL-SAVE: x_q_dq_bf16 BF16 [M,D] in [P]                                    |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: BF16 distributed W13 FPROP dispatch                                   | || | KERNEL: staged NVFP4 W13 FPROP dispatch                                       |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   dispatch_x_bf16 BF16 [T,D], via fwd_gather_ptrs INT64 [M]                   | || |   x_gathered_nvfp4 NVFP4 bundle logical [M,D], pointer-selected               |
|   w13_bf16 BF16 [E,2F,D]; expert_counts INT32 [E]                             | || |   w13_nvfp4 NVFP4 bundle logical [E,2F,D]                                     |
| OUTPUTS                                                                       | || |   expert_counts INT32 [E]; fwd_gather_ptrs INT64 [M]                          |
|   x_gathered_bf16 BF16 [M,D] in [P] FULL-SAVE, otherwise [S]                  | || | OUTPUTS                                                                       |
|   h1_bf16 BF16 [M,2F] in [P] FULL-SAVE, otherwise [S]                         | || |   h1_bf16 BF16 [M,2F] in [P] FULL-SAVE, otherwise [S]                         |
|   eligible native epilogue also emits h2_bf16 BF16 [M,F] in [S]               | || | NOTE: FP4 inputs; FP32 accumulation; BF16 H1 boundary                         |
| NOTE: BF16 inputs; FP32 accumulation                                          | || |                                                                               |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL(S): SwiGLU and BF16 W2 FPROP combine                                   | || | ONE KERNEL: fused q(SwiGLU) + NVFP4 W2 FPROP + peer combine                   |
| INPUTS                                                                        | || | OPS                                                                           |
|   h1_bf16 BF16 [M,2F]; w2_bf16 BF16 [E,D,F]                                   | || |   h2_nvfp4 = q(swiglu(h1_bf16))                                               |
|   expert_counts INT32 [E]; scatter_ptrs INT64 [M]                             | || |   h3_route_bf16 = nvfp4_gemm(h2_nvfp4, w2_nvfp4)                              |
| OUTPUTS                                                                       | || | INPUTS                                                                        |
|   h2_bf16 BF16 [M,F] in [S]                                                   | || |   h1_bf16 BF16 [M,2F]; w2_nvfp4 bundle logical [E,D,F]                        |
|   h3_route_bf16 BF16 [T*K,D] in peer [SYM] combine HBM                        | || |   expert_counts INT32 [E]; scatter_ptrs INT64 [M]                             |
| NOTE: standalone swiglu_fwd runs only when W13 did not emit H2;               | || | OUTPUTS                                                                       |
|       W2 combine is a separate kernel                                         | || |   h2_nvfp4 bundle logical [M,F], transient [S]                                |
|                                                                               | || |   h3_route_bf16 BF16 [T*K,D] in peer [SYM] combine HBM                        |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| DEVICE SYNC: combine-buffer barrier                                           | || | DEVICE SYNC: combine-buffer barrier                                           |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   h3_route_bf16 publication state                                             | || |   h3_route_bf16 publication state                                             |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   peer-visible BF16 [T*K,D]; no new tensor                                    | || |   peer-visible BF16 [T*K,D]; no new tensor                                    |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL GROUP: default expert postprocess forward                              | || | KERNEL GROUP: default expert postprocess forward                              |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   h3_route_bf16 BF16 [T*K,D]; topk_scores S [T,K]                             | || |   h3_route_bf16 BF16 [T*K,D]; topk_scores S [T,K]                             |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   y_bf16 BF16 [T,D] [OUT]                                                     | || |   y_bf16 BF16 [T,D] [OUT]                                                     |
|   FULL-SAVE: h3_saved_bf16 BF16 [T*K,D] in [P]                                | || |   FULL-SAVE: h3_saved_bf16 BF16 [T*K,D] in [P]                                |
|   RECOMPUTE: no H3 save                                                       | || |   RECOMPUTE: no H3 save                                                       |
| NOTE: score products and K reduction accumulate in FP32                       | || | NOTE: score products and K reduction accumulate in FP32                       |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| AUTOGRAD SAVE: BF16                                                           | || | AUTOGRAD SAVE: NVFP4 STAGED                                                   |
| FULL-SAVE [P]                                                                 | || | FULL-SAVE [P]                                                                 |
|   x_gathered_bf16 BF16 [M,D]                                                  | || |   x_q_dq_bf16 BF16 [M,D] = dq(x_gathered_nvfp4)                               |
|   h1_bf16 BF16 [M,2F]                                                         | || |   h1_bf16 BF16 [M,2F]                                                         |
|   h3_saved_bf16 BF16 [T*K,D]                                                  | || |   h3_saved_bf16 BF16 [T*K,D]                                                  |
| RECOMPUTE [P]: x_saved_bf16 BF16 [T,D]                                        | || | RECOMPUTE [P]: x_saved_bf16 BF16 [T,D]                                        |
| BOTH: weights, IDs/scores, counts, pointers, planner/slot/postprocess state   | || | BOTH: native weights, IDs/scores, counts, pointers, planner state             |
| NOT SAVED: h2_bf16                                                            | || | NOT SAVED: x_nvfp4, h2_nvfp4, w13_nvfp4, w2_nvfp4, or their scales            |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
```

### Backward: BF16 versus staged-NVFP4 training

```text
BF16 BACKWARD                                                                     || NVFP4 STAGED-FORWARD BACKWARD
==================================================================================++==================================================================================
                                                                                  || staged NVFP4 reconstruction + shared BF16 gradient kernels
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_y_bf16 BF16 [T,D]                                                      | || |   grad_y_bf16 BF16 [T,D]                                                      |
|   BF16 saved state from the forward chart                                     | || |   NVFP4 staged saved state from the forward chart                             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| NO WEIGHT RECONSTRUCTION KERNEL                                               | || | KERNEL GROUP: regenerate dynamic NVFP4 weights                                |
| INPUTS                                                                        | || | OP                                                                            |
|   saved w13_bf16 BF16 [E,2F,D]; w2_bf16 BF16 [E,D,F]                          | || |   w13_nvfp4 = q(w13_bf16); w2_nvfp4 = q(w2_bf16)                              |
| OUTPUTS                                                                       | || | INPUTS                                                                        |
|   the same native BF16 gradient operands                                      | || |   native w13_bf16 BF16 [E,2F,D]; w2_bf16 BF16 [E,D,F]                         |
|                                                                               | || | OUTPUTS                                                                       |
|                                                                               | || |   w13_nvfp4 bundle logical [E,2F,D]; w2_nvfp4 bundle logical [E,D,F]          |
|                                                                               | || |                                                                               |
|                                                                               | || | 2 KERNELS: exact dequantization                                               |
|                                                                               | || | INPUTS                                                                        |
|                                                                               | || |   w13_nvfp4 bundle logical [E,2F,D]; w2_nvfp4 bundle logical [E,D,F]          |
|                                                                               | || | OUTPUTS                                                                       |
|                                                                               | || |   w13_q_dq_bf16 = dq(w13_nvfp4) -> BF16 [E,2F,D] in [S]                       |
|                                                                               | || |   w2_q_dq_bf16 = dq(w2_nvfp4)   -> BF16 [E,D,F] in [S]                        |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: get_backward_plan                                                     | || | KERNEL: get_backward_plan                                                     |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   need_recompute BOOL [1]; forward_state INT64 [6]                            | || |   need_recompute BOOL [1]; forward_offsets INT64 [9]                          |
|   recv_count INT32 [1]; recv_snapshot INT32 [P]                               | || |   activation_offsets INT64 [18]; recv_count INT32 [1]; snapshot INT32 [P]     |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   backward_offsets INT64 [6]                                                  | || |   backward_offsets INT64 [16]                                                 |
|   matching forward slot released in mutable planner state                     | || |   backward_activation_offsets INT64 [54]                                      |
|                                                                               | || |   matching forward slot released in mutable planner state                     |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL 1 + SYNC: conditional local-X republish                                | || | KERNELS 1-2 + SYNC: conditional local-X republish                             |
| INPUTS                                                                        | || | OPS                                                                           |
|   x_saved_bf16 BF16 [T,D]; need_recompute BOOL [1]                            | || |   local_x_bf16 = conditional_load(x_saved_bf16)                               |
| OUTPUTS                                                                       | || |   x_nvfp4 = q(local_x_bf16) via _stage_blockscaled_dispatch                   |
|   dispatch_x_bf16 BF16 [T,D] in [SYM], only for RECOMPUTE                     | || | INPUTS                                                                        |
|   peer-visible rows after conditional dispatch barrier                        | || |   x_saved_bf16 BF16 [T,D]; need_recompute BOOL [1]                            |
|                                                                               | || | OUTPUTS                                                                       |
|                                                                               | || |   x_nvfp4 NVFP4 bundle logical [T,D] in [SYM], transient                      |
|                                                                               | || |   peer-visible packed rows after conditional dispatch barrier                 |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL 2: conditional BF16 W13 FPROP dispatch                                 | || | KERNELS 3-4: conditional staged NVFP4 W13 + exact X decode                    |
| INPUTS                                                                        | || | OPS                                                                           |
|   dispatch_x_bf16 BF16 [T,D]; w13_bf16 BF16 [E,2F,D]                          | || |   h1_recomputed_bf16 = nvfp4_gemm(x_gathered_nvfp4, w13_nvfp4)                |
|   fwd_gather_ptrs INT64 [M]; expert_counts INT32 [E]                          | || |   x_q_dq_bf16 = dq(x_gathered_nvfp4)                                          |
| OUTPUTS                                                                       | || | INPUTS                                                                        |
|   x_gathered_bf16 BF16 [M,D] in [S], only for RECOMPUTE                       | || |   x_gathered_nvfp4 bundle logical [M,D]; w13_nvfp4 logical [E,2F,D]           |
|   h1_recomputed_bf16 BF16 [M,2F] in [S], only for RECOMPUTE                   | || |   fwd_gather_ptrs INT64 [M]; expert_counts INT32 [E]                          |
| SELECT                                                                        | || | OUTPUTS                                                                       |
|   FULL-SAVE uses x_gathered_bf16 and h1_bf16 from [P]                         | || |   h1_recomputed_bf16 BF16 [M,2F] in [S], only for RECOMPUTE                   |
|   RECOMPUTE uses the outputs above                                            | || |   x_q_dq_bf16 BF16 [M,D] in [S], only for RECOMPUTE                           |
|                                                                               | || | SELECT: FULL-SAVE uses H1/X from [P]; RECOMPUTE uses outputs above            |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNELS 3-4: rebuild H2; conditionally rebuild H3                             | || | KERNEL + 2 DECODER LAUNCHES: rebuild exact H2 boundary and H3                 |
| OPS                                                                           | || | OPS                                                                           |
|   h2_bf16 = swiglu(h1_for_bwd_bf16)  [always runs]                            | || |   h2_nvfp4 = q(swiglu(h1_for_bwd_bf16))                                       |
|   h3_candidate = bf16_gemm(h2_bf16, w2_bf16) [conditional]                    | || |   h3_candidate = nvfp4_gemm(h2_nvfp4, w2_nvfp4)                               |
| INPUTS                                                                        | || |   h2_q_dq_bf16 = dq(h2_nvfp4)                                                 |
|   h1_for_bwd_bf16 BF16 [M,2F]; w2_bf16 BF16 [E,D,F]                           | || | INPUTS                                                                        |
| OUTPUTS                                                                       | || |   h1_for_bwd_bf16 BF16 [M,2F]; w2_nvfp4 bundle logical [E,D,F]                |
|   h2_bf16 BF16 [M,F] in [S]                                                   | || | OUTPUTS                                                                       |
|   h3_candidate_bf16 BF16 [T*K,D] in [SYM], only for RECOMPUTE                 | || |   h2_nvfp4 bundle logical [M,F] in [S]                                        |
| NOTE: FULL-SAVE consumes h3_saved_bf16 instead of the candidate               | || |   h2_q_dq_bf16 BF16 [M,F] in [S]                                              |
|                                                                               | || |   h3_candidate_bf16 BF16 [T*K,D] in [SYM]                                     |
|                                                                               | || | NOTE: fused H2/W2 kernel always runs; paired decoders keep graph stable       |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| DEVICE SYNC: conditional combine barrier                                      | || | DEVICE SYNC: unconditional adapter combine barrier                            |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   h3_candidate publication; need_recompute BOOL [1]                           | || |   h3_candidate publication state                                              |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   peer-visible candidate for RECOMPUTE; no tensor output                      | || |   peer-visible candidate; no tensor output                                    |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL GROUP: default expert postprocess backward                             | || | KERNEL GROUP: default expert postprocess backward                             |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_y_bf16 BF16 [T,D]; h3_for_bwd_bf16 BF16 [T*K,D]                        | || |   grad_y_bf16 BF16 [T,D]; h3_for_bwd_bf16 BF16 [T*K,D]                        |
|   topk_scores S [T,K]                                                         | || |   topk_scores S [T,K]                                                         |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_h3_route_bf16 BF16 [T*K,D] in local [SYM] combine HBM                  | || |   grad_h3_route_bf16 BF16 [T*K,D] in local [SYM] combine HBM                  |
|   grad_scores S [T,K] [OUT]                                                   | || |   grad_scores S [T,K] [OUT]                                                   |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| DEVICE SYNC: combine-buffer barrier                                           | || | DEVICE SYNC: combine-buffer barrier                                           |
| INPUTS: grad_h3_route_bf16 publication state                                  | || | INPUTS: grad_h3_route_bf16 publication state                                  |
| OUTPUTS: peer-visible BF16 [T*K,D]; no new tensor                             | || | OUTPUTS: peer-visible BF16 [T*K,D]; no new tensor                             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: BF16 distributed W2 DGRAD dispatch                                    | || | KERNEL: BF16 distributed W2 DGRAD dispatch                                    |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_h3_route_bf16 BF16 [T*K,D], via scatter_ptrs INT64 [M]                 | || |   grad_h3_route_bf16 BF16 [T*K,D], via scatter_ptrs INT64 [M]                 |
|   w2_bf16 BF16 [E,D,F]; expert_counts INT32 [E]                               | || |   w2_q_dq_bf16 BF16 [E,D,F]; expert_counts INT32 [E]                          |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_h3_gathered_bf16 BF16 [M,D] in [S]                                     | || |   grad_h3_gathered_bf16 BF16 [M,D] in [S]                                     |
|   grad_h2_bf16 BF16 [M,F] in [S]                                              | || |   grad_h2_bf16 BF16 [M,F] in [S]                                              |
| NOTE: BF16 inputs; FP32 accumulation                                          | || | NOTE: BF16 inputs; FP32 accumulation; no FP4 gradient operand                 |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: BF16 grouped W2 WGRAD                                                 | || | KERNEL: BF16 grouped W2 WGRAD                                                 |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_h3_gathered_bf16 BF16 [M,D]; h2_bf16 BF16 [M,F]                        | || |   grad_h3_gathered_bf16 BF16 [M,D]; h2_q_dq_bf16 BF16 [M,F]                   |
|   expert_counts INT32 [E]                                                     | || |   expert_counts INT32 [E]                                                     |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_w2 WG [E,D,F] [OUT]                                                    | || |   grad_w2 WG [E,D,F] [OUT]                                                    |
| NOTE: BF16 inputs; FP32 accumulation                                          | || | NOTE: BF16 inputs; FP32 accumulation                                          |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: fused SwiGLU backward + BF16 W13 DGRAD + peer combine                 | || | KERNEL: fused SwiGLU backward + BF16 W13 DGRAD + peer combine                 |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_h2_bf16 BF16 [M,F]; h1_for_bwd_bf16 BF16 [M,2F]                        | || |   grad_h2_bf16 BF16 [M,F]; h1_for_bwd_bf16 BF16 [M,2F]                        |
|   w13_bf16 BF16 [E,2F,D]; bwd_gather_ptrs INT64 [M]                           | || |   w13_q_dq_bf16 BF16 [E,2F,D]; bwd_gather_ptrs INT64 [M]                      |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_h1_bf16 BF16 [M,2F] in [S]                                             | || |   grad_h1_bf16 BF16 [M,2F] in [S]                                             |
|   grad_x_route_bf16 BF16 [T*K,D] in peer [SYM] dispatch HBM                   | || |   grad_x_route_bf16 BF16 [T*K,D] in peer [SYM] dispatch HBM                   |
| NOTE: unsupported widths use separate SwiGLU-bwd and W13-DGRAD kernels        | || | NOTE: unsupported widths use the same two-kernel fallback                     |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: BF16 grouped W13 WGRAD                                                | || | KERNEL: BF16 grouped W13 WGRAD                                                |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_h1_bf16 BF16 [M,2F]; x_gathered_bf16 BF16 [M,D]                        | || |   grad_h1_bf16 BF16 [M,2F]; x_q_dq_bf16 BF16 [M,D]                            |
|   expert_counts INT32 [E]                                                     | || |   expert_counts INT32 [E]                                                     |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_w13 WG [E,2F,D] [OUT]                                                  | || |   grad_w13 WG [E,2F,D] [OUT]                                                  |
| NOTE: BF16 inputs; FP32 accumulation                                          | || | NOTE: BF16 inputs; FP32 accumulation                                          |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| DEVICE SYNC: dispatch-buffer barrier                                          | || | DEVICE SYNC: dispatch-buffer barrier                                          |
| INPUTS: grad_x_route_bf16 publication state                                   | || | INPUTS: grad_x_route_bf16 publication state                                   |
| OUTPUTS: peer-visible BF16 [T*K,D]; no new tensor                             | || | OUTPUTS: peer-visible BF16 [T*K,D]; no new tensor                             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: reduce_from_topk                                                      | || | KERNEL: reduce_from_topk                                                      |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_x_route_bf16 BF16 [T*K,D]; topk_ids INT32/INT64 [T,K]                  | || |   grad_x_route_bf16 BF16 [T*K,D]; topk_ids INT32/INT64 [T,K]                  |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_x_bf16 BF16 [T,D] [OUT]                                                | || |   grad_x_bf16 BF16 [T,D] [OUT]                                                |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_x_bf16 BF16 [T,D]; grad_scores S [T,K]                                 | || |   grad_x_bf16 BF16 [T,D]; grad_scores S [T,K]                                 |
|   grad_w13 WG [E,2F,D]; grad_w2 WG [E,D,F]                                    | || |   grad_w13 WG [E,2F,D]; grad_w2 WG [E,D,F]                                    |
| GRADIENT GEMM OPERANDS                                                        | || | GRADIENT GEMM OPERANDS                                                        |
|   W2: grad_h3_gathered_bf16, w2_bf16/h2_bf16                                  | || |   W2: grad_h3_gathered_bf16, w2_q_dq_bf16/h2_q_dq_bf16                        |
|   W13: grad_h1_bf16, w13_bf16/x_gathered_bf16                                 | || |   W13: grad_h1_bf16, w13_q_dq_bf16/x_q_dq_bf16                                |
|   all inputs BF16; accumulation FP32                                          | || |   all inputs BF16; accumulation FP32; no native FP4 backward GEMM             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
```

## Chart 2: Mega execution

### Forward: BF16 versus Mega NVFP4

```text
BF16 FORWARD                                                                      || NVFP4 MEGA FORWARD
==================================================================================++==================================================================================
                                                                                  ||
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| INPUTS                                                                        | || | INPUTS                                                                        |
|   x_bf16 BF16 [T,D]; w13_bf16 BF16 [E,2F,D]                                   | || |   x_bf16 BF16 [T,D]; w13_bf16 BF16 [E,2F,D]                                   |
|   w2_bf16 BF16 [E,D,F]                                                        | || |   w2_bf16 BF16 [E,D,F]                                                        |
|   topk_ids INT32/INT64 [T,K]; topk_scores S [T,K]                             | || |   topk_ids INT32/INT64 [T,K]; topk_scores S [T,K]                             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| NO PREPARATION KERNEL                                                         | || | KERNEL GROUP: prepare dynamic NVFP4 weights                                   |
| INPUTS                                                                        | || | OP                                                                            |
|   native w13_bf16 BF16 [E,2F,D]; w2_bf16 BF16 [E,D,F]                         | || |   w13_nvfp4 = q(w13_bf16); w2_nvfp4 = q(w2_bf16)                              |
| OUTPUTS                                                                       | || | INPUTS                                                                        |
|   the same native BF16 weight operands                                        | || |   native BF16 weights with the shapes at chart entry                          |
|                                                                               | || | OUTPUTS                                                                       |
|                                                                               | || |   w13_nvfp4 bundle logical [E,2F,D], transient                                |
|                                                                               | || |   w2_nvfp4 bundle logical [E,D,F], transient                                  |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL GROUP: copy_routing_and_dispatch                                       | || | KERNEL GROUP: copy routing + _stage_blockscaled_dispatch                      |
| INPUTS                                                                        | || | OP: x_nvfp4 = q(x_bf16); publish the complete row bundle                      |
|   x_bf16 BF16 [T,D]; topk_ids INT32/INT64 [T,K]                               | || | INPUTS                                                                        |
| OUTPUTS                                                                       | || |   x_bf16 BF16 [T,D]; topk_ids INT32/INT64 [T,K]                               |
|   dispatch_x_bf16 BF16 [T,D] in [SYM]                                         | || | OUTPUTS                                                                       |
|   routing_ids INT16 [T,K]; routing_token_count INT32 [1]                      | || |   x_nvfp4 bundle logical [T,D] in [SYM] dispatch HBM                          |
|                                                                               | || |   routing_ids INT16 [T,K]; routing_token_count INT32 [1]                      |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| DEVICE SYNC: expert-parallel dispatch barrier                                 | || | DEVICE SYNC: expert-parallel dispatch barrier                                 |
| INPUTS: BF16 dispatch/routing publication state                               | || | INPUTS: packed dispatch/routing publication state                             |
| OUTPUTS: peer-visible BF16 rows and IDs; no tensor output                     | || | OUTPUTS: peer-visible packed rows and IDs; no tensor output                   |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL GROUP: dist_dispatch_routing                                           | || | KERNEL GROUP: dist_dispatch_routing                                           |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   topk_ids INT32/INT64 [T,K]; peer BF16 row bases [P]                         | || |   topk_ids INT32/INT64 [T,K]; peer packed-row bases [P]                       |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   expert_counts INT32 [E]; recv_by_rank INT32 [P]; recv_count INT32 [1]       | || |   expert_counts INT32 [E]; recv_by_rank INT32 [P]; recv_count INT32 [1]       |
|   fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs INT64 [M] each               | || |   fwd_gather_ptrs, bwd_gather_ptrs, scatter_ptrs INT64 [M] each               |
| SAVE: counts and pointer tables cross autograd                                | || | SAVE: counts and pointer tables cross autograd                                |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: get_forward_plan                                                      | || | KERNEL: get_forward_plan                                                      |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   recv_count INT32 [1]; recv_by_rank INT32 [P]; slot_id INT64 [1]             | || |   recv_count INT32 [1]; recv_by_rank INT32 [P]; slot_id INT64 [1]             |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   need_recompute BOOL [1]; forward_state INT64 [6]                            | || |   need_recompute BOOL [1]; recompute_condition INT32 [1]                      |
|   recv_snapshot INT32 [P]                                                     | || |   forward_offsets INT64 [9]; activation_offsets INT64 [18]                    |
| SAVE: decision, offsets, physical slot ID, and snapshot                       | || |   recv_snapshot INT32 [P]                                                     |
|                                                                               | || | SAVE: decision, offsets, physical slot ID, predicates, and snapshot           |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: copy_dispatch_to_activation (device-predicated)                       | || | 2 PREDICATE-SELECTED KERNELS: save the chosen X payload                       |
| INPUTS                                                                        | || | OPS                                                                           |
|   dispatch_x_bf16 BF16 [T,D]; need_recompute BOOL [1]                         | || |   RECOMPUTE: x_saved_bf16 = copy(x_bf16)                                      |
| OUTPUTS                                                                       | || |   FULL-SAVE: x_q_dq_bf16 = dq(x_gathered_nvfp4)                               |
|   RECOMPUTE: x_saved_bf16 BF16 [T,D] in [P]                                   | || | INPUTS                                                                        |
|   FULL-SAVE: no activation write                                              | || |   x_bf16 BF16 [T,D]; x_gathered_nvfp4 bundle logical [M,D]                    |
|                                                                               | || | OUTPUTS                                                                       |
|                                                                               | || |   RECOMPUTE: x_saved_bf16 BF16 [T,D] in [P]                                   |
|                                                                               | || |   FULL-SAVE: x_q_dq_bf16 BF16 [M,D] in [P]                                    |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNELS: BF16 W13 dispatch, SwiGLU, and W2 combine                            | || | ONE KERNEL: chunked Mega NVFP4 forward                                        |
| INPUTS                                                                        | || | OPS                                                                           |
|   dispatch_x_bf16 BF16 [T,D]; w13_bf16 BF16 [E,2F,D]                          | || |   h1_bf16 = nvfp4_gemm(x_gathered_nvfp4, w13_nvfp4)                           |
|   w2_bf16 BF16 [E,D,F]; expert_counts INT32 [E]                               | || |   h2_nvfp4 = q(swiglu(h1_bf16))                                               |
|   fwd_gather_ptrs, scatter_ptrs INT64 [M]                                     | || |   h3_route_bf16 = nvfp4_gemm(h2_nvfp4, w2_nvfp4)                              |
| OUTPUTS                                                                       | || | INPUTS                                                                        |
|   x_gathered_bf16 BF16 [M,D] in [P] or [S]                                    | || |   x_gathered_nvfp4 bundle logical [M,D], pointer-selected                     |
|   h1_bf16 BF16 [M,2F] in [P] or [S]                                           | || |   w13_nvfp4 logical [E,2F,D]; w2_nvfp4 logical [E,D,F]                        |
|   h2_bf16 BF16 [M,F] in [S]                                                   | || |   expert_counts INT32 [E]; gather/scatter pointers INT64 [M]                  |
|   h3_route_bf16 BF16 [T*K,D] in peer [SYM] combine HBM                        | || | OUTPUTS                                                                       |
| NOTE: W13 may fuse SwiGLU; W2 combine remains a separate kernel               | || |   h1_bf16 BF16 [M,2F] in [P] FULL-SAVE, otherwise [S]                         |
|                                                                               | || |   h2_nvfp4 bundle logical [M,F], transient                                    |
|                                                                               | || |   h3_route_bf16 BF16 [T*K,D] in peer [SYM] combine HBM                        |
|                                                                               | || | NOTE: W13, q(SwiGLU), W2, and peer combine pipeline in one launch             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| DEVICE SYNC: combine-buffer barrier                                           | || | DEVICE SYNC: combine-buffer barrier                                           |
| INPUTS: h3_route_bf16 publication state                                       | || | INPUTS: h3_route_bf16 publication state                                       |
| OUTPUTS: peer-visible BF16 [T*K,D]; no new tensor                             | || | OUTPUTS: peer-visible BF16 [T*K,D]; no new tensor                             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL GROUP: default expert postprocess forward                              | || | KERNEL GROUP: default expert postprocess forward                              |
| INPUTS: h3_route_bf16 BF16 [T*K,D]; topk_scores S [T,K]                       | || | INPUTS: h3_route_bf16 BF16 [T*K,D]; topk_scores S [T,K]                       |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   y_bf16 BF16 [T,D] [OUT]                                                     | || |   y_bf16 BF16 [T,D] [OUT]                                                     |
|   FULL-SAVE: h3_saved_bf16 BF16 [T*K,D] in [P]                                | || |   FULL-SAVE: h3_saved_bf16 BF16 [T*K,D] in [P]                                |
|   RECOMPUTE: no H3 save                                                       | || |   RECOMPUTE: no H3 save                                                       |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| AUTOGRAD SAVE: BF16                                                           | || | AUTOGRAD SAVE: NVFP4 MEGA                                                     |
| FULL-SAVE [P]                                                                 | || | FULL-SAVE [P]                                                                 |
|   x_gathered_bf16 BF16 [M,D]                                                  | || |   x_q_dq_bf16 BF16 [M,D] = dq(x_gathered_nvfp4)                               |
|   h1_bf16 BF16 [M,2F]                                                         | || |   h1_bf16 BF16 [M,2F]                                                         |
|   h3_saved_bf16 BF16 [T*K,D]                                                  | || |   h3_saved_bf16 BF16 [T*K,D]                                                  |
| RECOMPUTE [P]: x_saved_bf16 BF16 [T,D]                                        | || | RECOMPUTE [P]: x_saved_bf16 BF16 [T,D]                                        |
| BOTH: weights, IDs/scores, counts, pointers, planner/slot/postprocess state   | || | BOTH: native weights, IDs/scores, counts, pointers, planner state             |
| NOT SAVED: h2_bf16                                                            | || | NOT SAVED: x_nvfp4, h2_nvfp4, w13_nvfp4, w2_nvfp4, or their scales            |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
```

### Backward: BF16 versus Mega-NVFP4 training

Mega changes only the original NVFP4 forward launch. The NVFP4 backward below
still uses the staged reconstruction adapter before entering the shared BF16
gradient kernels.

```text
BF16 BACKWARD                                                                     || NVFP4 MEGA-FORWARD BACKWARD
==================================================================================++==================================================================================
                                                                                  || staged NVFP4 reconstruction + shared BF16 gradient kernels
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_y_bf16 BF16 [T,D]                                                      | || |   grad_y_bf16 BF16 [T,D]                                                      |
|   BF16 saved state from the forward chart                                     | || |   NVFP4 Mega saved state from the forward chart                               |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| NO WEIGHT RECONSTRUCTION KERNEL                                               | || | KERNEL GROUP: regenerate dynamic NVFP4 weights                                |
| INPUTS                                                                        | || | OP                                                                            |
|   saved w13_bf16 BF16 [E,2F,D]; w2_bf16 BF16 [E,D,F]                          | || |   w13_nvfp4 = q(w13_bf16); w2_nvfp4 = q(w2_bf16)                              |
| OUTPUTS                                                                       | || | INPUTS                                                                        |
|   the same native BF16 gradient operands                                      | || |   native w13_bf16 BF16 [E,2F,D]; w2_bf16 BF16 [E,D,F]                         |
|                                                                               | || | OUTPUTS                                                                       |
|                                                                               | || |   w13_nvfp4 bundle logical [E,2F,D]; w2_nvfp4 bundle logical [E,D,F]          |
|                                                                               | || | 2 KERNELS: exact dequantization                                               |
|                                                                               | || | INPUTS                                                                        |
|                                                                               | || |   w13_nvfp4 bundle logical [E,2F,D]; w2_nvfp4 bundle logical [E,D,F]          |
|                                                                               | || | OUTPUTS                                                                       |
|                                                                               | || |   w13_q_dq_bf16 = dq(w13_nvfp4) -> BF16 [E,2F,D] in [S]                       |
|                                                                               | || |   w2_q_dq_bf16 = dq(w2_nvfp4)   -> BF16 [E,D,F] in [S]                        |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: get_backward_plan                                                     | || | KERNEL: get_backward_plan                                                     |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   need_recompute BOOL [1]; forward_state INT64 [6]                            | || |   need_recompute BOOL [1]; forward_offsets INT64 [9]                          |
|   recv_count INT32 [1]; recv_snapshot INT32 [P]                               | || |   activation_offsets INT64 [18]; recv_count INT32 [1]; snapshot INT32 [P]     |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   backward_offsets INT64 [6]                                                  | || |   backward_offsets INT64 [16]                                                 |
|   matching forward slot released in mutable planner state                     | || |   backward_activation_offsets INT64 [54]                                      |
|                                                                               | || |   matching forward slot released in mutable planner state                     |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL 1 + SYNC: conditional local-X republish                                | || | KERNELS 1-2 + SYNC: conditional local-X republish                             |
| INPUTS                                                                        | || | OPS                                                                           |
|   x_saved_bf16 BF16 [T,D]; need_recompute BOOL [1]                            | || |   local_x_bf16 = conditional_load(x_saved_bf16)                               |
| OUTPUTS                                                                       | || |   x_nvfp4 = q(local_x_bf16) via _stage_blockscaled_dispatch                   |
|   dispatch_x_bf16 BF16 [T,D] in [SYM], only for RECOMPUTE                     | || | INPUTS                                                                        |
|   peer-visible rows after conditional dispatch barrier                        | || |   x_saved_bf16 BF16 [T,D]; need_recompute BOOL [1]                            |
|                                                                               | || | OUTPUTS                                                                       |
|                                                                               | || |   x_nvfp4 NVFP4 bundle logical [T,D] in [SYM], transient                      |
|                                                                               | || |   peer-visible packed rows after conditional dispatch barrier                 |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL 2: conditional BF16 W13 FPROP dispatch                                 | || | KERNELS 3-4: conditional staged NVFP4 W13 + exact X decode                    |
| INPUTS                                                                        | || | OPS                                                                           |
|   dispatch_x_bf16 BF16 [T,D]; w13_bf16 BF16 [E,2F,D]                          | || |   h1_recomputed_bf16 = nvfp4_gemm(x_gathered_nvfp4, w13_nvfp4)                |
|   fwd_gather_ptrs INT64 [M]; expert_counts INT32 [E]                          | || |   x_q_dq_bf16 = dq(x_gathered_nvfp4)                                          |
| OUTPUTS                                                                       | || | INPUTS                                                                        |
|   x_gathered_bf16 BF16 [M,D] in [S], only for RECOMPUTE                       | || |   x_gathered_nvfp4 bundle logical [M,D]; w13_nvfp4 logical [E,2F,D]           |
|   h1_recomputed_bf16 BF16 [M,2F] in [S], only for RECOMPUTE                   | || |   fwd_gather_ptrs INT64 [M]; expert_counts INT32 [E]                          |
| SELECT                                                                        | || | OUTPUTS                                                                       |
|   FULL-SAVE uses x_gathered_bf16 and h1_bf16 from [P]                         | || |   h1_recomputed_bf16 BF16 [M,2F] in [S], only for RECOMPUTE                   |
|   RECOMPUTE uses the outputs above                                            | || |   x_q_dq_bf16 BF16 [M,D] in [S], only for RECOMPUTE                           |
|                                                                               | || | SELECT: FULL-SAVE uses H1/X from [P]; RECOMPUTE uses outputs above            |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNELS 3-4: rebuild H2; conditionally rebuild H3                             | || | KERNEL + 2 DECODER LAUNCHES: staged H2/H3 reconstruction                      |
| OPS                                                                           | || | OPS                                                                           |
|   h2_bf16 = swiglu(h1_for_bwd_bf16)  [always runs]                            | || |   h2_nvfp4 = q(swiglu(h1_for_bwd_bf16))                                       |
|   h3_candidate = bf16_gemm(h2_bf16, w2_bf16) [conditional]                    | || |   h3_candidate = nvfp4_gemm(h2_nvfp4, w2_nvfp4)                               |
| INPUTS                                                                        | || |   h2_q_dq_bf16 = dq(h2_nvfp4)                                                 |
|   h1_for_bwd_bf16 BF16 [M,2F]; w2_bf16 BF16 [E,D,F]                           | || | INPUTS                                                                        |
| OUTPUTS                                                                       | || |   h1_for_bwd_bf16 BF16 [M,2F]; w2_nvfp4 bundle logical [E,D,F]                |
|   h2_bf16 BF16 [M,F] in [S]                                                   | || | OUTPUTS                                                                       |
|   h3_candidate_bf16 BF16 [T*K,D] in [SYM], only for RECOMPUTE                 | || |   h2_nvfp4 bundle logical [M,F] in [S]                                        |
| NOTE: FULL-SAVE consumes h3_saved_bf16 instead of the candidate               | || |   h2_q_dq_bf16 BF16 [M,F] in [S]                                              |
|                                                                               | || |   h3_candidate_bf16 BF16 [T*K,D] in [SYM]                                     |
|                                                                               | || | NOTE: reconstruction is staged even though the original forward was Mega      |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| DEVICE SYNC: conditional combine barrier                                      | || | DEVICE SYNC: unconditional adapter combine barrier                            |
| INPUTS: h3_candidate publication; need_recompute BOOL [1]                     | || | INPUTS: h3_candidate publication state                                        |
| OUTPUTS: peer-visible candidate for RECOMPUTE; no tensor output               | || | OUTPUTS: peer-visible candidate; no tensor output                             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL GROUP: default expert postprocess backward                             | || | KERNEL GROUP: default expert postprocess backward                             |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_y_bf16 BF16 [T,D]; h3_for_bwd_bf16 BF16 [T*K,D]                        | || |   grad_y_bf16 BF16 [T,D]; h3_for_bwd_bf16 BF16 [T*K,D]                        |
|   topk_scores S [T,K]                                                         | || |   topk_scores S [T,K]                                                         |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_h3_route_bf16 BF16 [T*K,D] in local [SYM] combine HBM                  | || |   grad_h3_route_bf16 BF16 [T*K,D] in local [SYM] combine HBM                  |
|   grad_scores S [T,K] [OUT]                                                   | || |   grad_scores S [T,K] [OUT]                                                   |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| DEVICE SYNC: combine-buffer barrier                                           | || | DEVICE SYNC: combine-buffer barrier                                           |
| INPUTS: grad_h3_route_bf16 publication state                                  | || | INPUTS: grad_h3_route_bf16 publication state                                  |
| OUTPUTS: peer-visible BF16 [T*K,D]; no new tensor                             | || | OUTPUTS: peer-visible BF16 [T*K,D]; no new tensor                             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: BF16 distributed W2 DGRAD dispatch                                    | || | KERNEL: BF16 distributed W2 DGRAD dispatch                                    |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_h3_route_bf16 BF16 [T*K,D], via scatter_ptrs INT64 [M]                 | || |   grad_h3_route_bf16 BF16 [T*K,D], via scatter_ptrs INT64 [M]                 |
|   w2_bf16 BF16 [E,D,F]; expert_counts INT32 [E]                               | || |   w2_q_dq_bf16 BF16 [E,D,F]; expert_counts INT32 [E]                          |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_h3_gathered_bf16 BF16 [M,D] in [S]                                     | || |   grad_h3_gathered_bf16 BF16 [M,D] in [S]                                     |
|   grad_h2_bf16 BF16 [M,F] in [S]                                              | || |   grad_h2_bf16 BF16 [M,F] in [S]                                              |
| NOTE: BF16 inputs; FP32 accumulation                                          | || | NOTE: BF16 inputs; FP32 accumulation; no FP4 gradient operand                 |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: BF16 grouped W2 WGRAD                                                 | || | KERNEL: BF16 grouped W2 WGRAD                                                 |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_h3_gathered_bf16 BF16 [M,D]; h2_bf16 BF16 [M,F]                        | || |   grad_h3_gathered_bf16 BF16 [M,D]; h2_q_dq_bf16 BF16 [M,F]                   |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_w2 WG [E,D,F] [OUT]                                                    | || |   grad_w2 WG [E,D,F] [OUT]                                                    |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: fused SwiGLU backward + BF16 W13 DGRAD + peer combine                 | || | KERNEL: fused SwiGLU backward + BF16 W13 DGRAD + peer combine                 |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_h2_bf16 BF16 [M,F]; h1_for_bwd_bf16 BF16 [M,2F]                        | || |   grad_h2_bf16 BF16 [M,F]; h1_for_bwd_bf16 BF16 [M,2F]                        |
|   w13_bf16 BF16 [E,2F,D]; bwd_gather_ptrs INT64 [M]                           | || |   w13_q_dq_bf16 BF16 [E,2F,D]; bwd_gather_ptrs INT64 [M]                      |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_h1_bf16 BF16 [M,2F] in [S]                                             | || |   grad_h1_bf16 BF16 [M,2F] in [S]                                             |
|   grad_x_route_bf16 BF16 [T*K,D] in peer [SYM] dispatch HBM                   | || |   grad_x_route_bf16 BF16 [T*K,D] in peer [SYM] dispatch HBM                   |
| NOTE: unsupported widths use separate SwiGLU-bwd and W13-DGRAD kernels        | || | NOTE: unsupported widths use the same two-kernel fallback                     |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: BF16 grouped W13 WGRAD                                                | || | KERNEL: BF16 grouped W13 WGRAD                                                |
| INPUTS                                                                        | || | INPUTS                                                                        |
|   grad_h1_bf16 BF16 [M,2F]; x_gathered_bf16 BF16 [M,D]                        | || |   grad_h1_bf16 BF16 [M,2F]; x_q_dq_bf16 BF16 [M,D]                            |
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_w13 WG [E,2F,D] [OUT]                                                  | || |   grad_w13 WG [E,2F,D] [OUT]                                                  |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| DEVICE SYNC: dispatch-buffer barrier                                          | || | DEVICE SYNC: dispatch-buffer barrier                                          |
| INPUTS: grad_x_route_bf16 publication state                                   | || | INPUTS: grad_x_route_bf16 publication state                                   |
| OUTPUTS: peer-visible BF16 [T*K,D]; no new tensor                             | || | OUTPUTS: peer-visible BF16 [T*K,D]; no new tensor                             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| KERNEL: reduce_from_topk                                                      | || | KERNEL: reduce_from_topk                                                      |
| INPUTS: grad_x_route_bf16 BF16 [T*K,D]; topk_ids INT32/INT64 [T,K]            | || | INPUTS: grad_x_route_bf16 BF16 [T*K,D]; topk_ids INT32/INT64 [T,K]            |
| OUTPUTS: grad_x_bf16 BF16 [T,D] [OUT]                                         | || | OUTPUTS: grad_x_bf16 BF16 [T,D] [OUT]                                         |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
                                        |                                         ||                                         |
                                        v                                         ||                                         v
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
| OUTPUTS                                                                       | || | OUTPUTS                                                                       |
|   grad_x_bf16 BF16 [T,D]; grad_scores S [T,K]                                 | || |   grad_x_bf16 BF16 [T,D]; grad_scores S [T,K]                                 |
|   grad_w13 WG [E,2F,D]; grad_w2 WG [E,D,F]                                    | || |   grad_w13 WG [E,2F,D]; grad_w2 WG [E,D,F]                                    |
| GRADIENT GEMM OPERANDS                                                        | || | GRADIENT GEMM OPERANDS                                                        |
|   W2: grad_h3_gathered_bf16, w2_bf16/h2_bf16                                  | || |   W2: grad_h3_gathered_bf16, w2_q_dq_bf16/h2_q_dq_bf16                        |
|   W13: grad_h1_bf16, w13_bf16/x_gathered_bf16                                 | || |   W13: grad_h1_bf16, w13_q_dq_bf16/x_q_dq_bf16                                |
|   all inputs BF16; accumulation FP32                                          | || |   all inputs BF16; accumulation FP32; no native FP4 backward GEMM             |
+-------------------------------------------------------------------------------+ || +-------------------------------------------------------------------------------+
```
