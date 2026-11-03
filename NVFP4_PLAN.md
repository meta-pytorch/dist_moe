# NVFP4 Forward and Backward

The graphs compare the BF16 baseline with the implemented NVFP4 training
path. Corresponding components are aligned horizontally.

Legend:

```text
[P] persistent activation slot
[S] reusable scratch
[SYM] peer-addressable symmetric HBM
Q(A) packed NVFP4
A_bw = DQ(Q(A))
```

## Forward

```text
BF16 FORWARD                                                             | IMPLEMENTED NVFP4 TRAINING FORWARD
-------------------------------------------------------------------------+--------------------------------------------------------------------------------
                                                                         | BF16 W13/W2
                                                                         |      │
                                                                         |      │ dynamic per-expert NVFP4 quantization
                                                                         |      ▼
                                                                         | Q(W13), Q(W2) + scales [temporary; not saved]
                                                                         |
Local X BF16 + expert IDs                                                | Local X BF16 + expert IDs
          │                                                              |      │
          │ publish                                                      |      │ per-token NVFP4 quantization and publication
          ▼                                                              |      ▼
[SYM] BF16 X + routing IDs                                               | [SYM] Q(X) + block scales + token outer scale + routing IDs
          │                                                              |      │
          ├──────────────→ routing counts and gather/scatter pointers    |      ├──────────────→ routing counts and gather/scatter pointers
          │                                  │                           |      │                                  │
          │                                  ▼                           |      │                                  ▼
          │                           device forward planner             |      │                           hybrid device forward planner
          │                           ┌───────┴────────┐                 |      │                           ┌───────┴────────┐
          │                           │                │                 |      │                           │                │
          │                       full-save        recompute             |      │                       full-save        recompute
          │                           │                │                 |      │                           │                │
          │                           │         save local X [P]         |      │                           │         save original local BF16 X [P]
          ▼                           │                                  |      ▼                           │
                                                                         | gather exact peer-addressed Q(X) │
                                                                         |      │                           │
                                                                         |      └── DQ ──→ X_bw [P]         │ no X_bw write in forward
                                                                         |                                  │
                                                                         |      ▼                           ▼
┌─────────────────────┐              │                                   | ┌──────────────────────────┐
│ Gather + W13 GEMM   │              │                                   | │ NVFP4 W13                │
│                     │              │                                   | │                          │
│ X_gathered BF16     │──────────────┴──→ [P] or [S]                     | │ Q(X) × Q(W13)            │
│ H1 BF16             │─────────────────→ [P] or [S]                     | │ → BF16 H1 boundary       │──────────────→ H1 [P] or [S]
└──────────┬──────────┘                                                  | └────────────┬─────────────┘
           │                                                             |              │
           ▼                                                             |              ▼
       SwiGLU                                                            |           SwiGLU
           │                                                             |              │
           ▼                                                             |              ▼
       H2 BF16 ──────────────────────────→ [S] always                    |             H2
           │                                                             |              │ per-route NVFP4 quantization
           ▼                                                             |              ▼
                                                                         |           Q(H2) + block scales + token outer scale [S]
                                                                         |              │
                                                                         |              ▼
┌─────────────────────┐                                                  | ┌──────────────────────────┐
│ W2 GEMM             │                                                  | │ NVFP4 W2                 │
│ + peer scatter      │                                                  | │ Q(H2) × Q(W2)            │
└──────────┬──────────┘                                                  | │ → H3 BF16 + peer scatter │
           │                                                             | └────────────┬─────────────┘
           ▼                                                             |              ▼
[SYM] H3 route outputs                                                   | [SYM] H3 route outputs
           │                                                             |              │
           ├── full-save ─────────────────→ H3 [P]                       |              ├── full-save ─────────────────→ H3 [P]
           │                                                             |              │
           ▼                                                             |              ▼
score × H3 + top-k reduction                                             | score × H3 + top-k reduction
           │                                                             |              │
           ▼                                                             |              ▼
       Y BF16                                                            |           Y BF16
                                                                         |
Full-save: [X_gathered, H1, H3]                                          | Full-save: [X_bw = DQ(Q(X_gathered)), H1, H3]
Recompute: [original local X]                                            | Recompute: [original local BF16 X]
                                                                         |
                                                                         | Fixed 21-tensor state: 10 empty weight slots, expert row counts,
                                                                         | forward/backward gather and scatter pointers, the per-rank row-count
                                                                         | snapshot, local row count, need_recompute, planner/descriptor offsets,
                                                                         | the device recompute predicate, and postprocessing state.
                                                                         | BF16 weights remain explicit registered-op inputs.
                                                                         |
H2 is never persistent.                                                  | H2, Q(H2), Q(X), and Q(W) are never persistent.
```

## Backward

```text
BF16 BACKWARD                                                            | IMPLEMENTED NVFP4 TRAINING BACKWARD
-------------------------------------------------------------------------+--------------------------------------------------------------------------------
                                                                         | BF16 W13                            BF16 W2
                                                                         |     │                                   │
                                                                         |     │ quantize again                    │ quantize again
                                                                         |     ▼                                   ▼
                                                                         | Q(W13) + scales                     Q(W2) + scales
                                                                         |     │                                   │
                                                                         |     │ exact dequantize                  │ exact dequantize
                                                                         |     ▼                                   ▼
                                                                         | W13_bw = DQ(Q(W13))                 W2_bw = DQ(Q(W2))
                                                                         |
Saved routing + offsets + planner predicate                              | Saved routing + offsets + planner predicate
                      │                                                  |                       │
                      ▼                                                  |                       ▼
              device backward planner                                    |               device backward planner
                      │                                                  |                       │
          ┌───────────┴────────────┐                                     |           ┌───────────┴────────────┐
          │                        │                                     |           │                        │
      full-save                 recompute                                |       full-save                 recompute
          │                        │                                     |           │                        │
 load X_gathered, H1, H3      saved local X                              |  load X_bw, H1, H3          saved local BF16 X
          │                    → republish                               |           │                    → quantize/publish Q(X)
          │                    → gather + W13                            |           │                    → staged NVFP4 W13
          │                    → recreate X_gathered, H1                 |           │                    → recreate H1
          │                    → SwiGLU + W2                             |           │                    → DQ gathered Q(X) as X_bw
          │                    → recreate H3                             |           └───────────┬────────────┘
          └───────────┬────────────┘                                     |                       │
                      │                                                  |                       ▼
                      ▼                                                  |                     H1
                    H1                                                   |                       │
                      │                                                  |                       │ staged fused SwiGLU/Q(H2)/W2 rerun
                      │ SwiGLU, always rerun                             |                       │ (used for both planner decisions and
                      ▼                                                  |                       │  for Mega or Staged original forward)
                    H2 [S]                                               |                       ▼
                                                                         |          Q(H2) + scales [S], recomputed H3
                                                                         |                       │
                                                                         |                       │ exact dequantize Q(H2)
                                                                         |                       ▼
                                                                         |       H2_bw = DQ(Q(SwiGLU(H1))) [S]
                                                                         |
                                                                         | H3 = saved H3 for full-save, recomputed H3 for recompute
                                                                         |
grad_Y + scores + H3                                                     | grad_Y + scores + H3
           │                                                             |            │
           ▼                                                             |            ▼
  score/reduction backward                                               |   shared BF16 score/reduction backward
       │              │                                                  |        │              │
       │              └────────────────────────────→ grad_scores         |        │              └────────────────────────────→ grad_scores
       ▼                                                                 |        ▼
    grad_H3                                                              |     grad_H3
       │                                                                 |        │
       ├── W2 DGRAD with W2 ───────────────────────→ grad_H2             |        ├── BF16 W2 DGRAD with W2_bw ───────────────→ grad_H2
       │                                                                 |        │
       └── W2 WGRAD with H2 ───────────────────────→ grad_W2             |        └── BF16 W2 WGRAD with H2_bw ───────────────→ grad_W2
                                                                         |
grad_H2 + H1                                                             | grad_H2 + H1
       │                                                                 |        │
       ▼                                                                 |        ▼
SwiGLU backward                                                          | shared BF16 SwiGLU backward
       │                                                                 |        │
       ▼                                                                 |        ▼
    grad_H1                                                              |     grad_H1
       │                                                                 |        │
       ├── W13 DGRAD with W13 ─────────────────────→ route grad_X        |        ├── BF16 W13 DGRAD with W13_bw ─────────────→ route grad_X
       │                                                    │            |        │                                                      │
       │                                             peer scatter        |        │                                               peer scatter
       │                                             + top-k sum         |        │                                               + top-k sum
       │                                                    │            |        │                                                      │
       │                                                    ▼            |        │                                                      ▼
       │                                                 grad_X          |        │                                                   grad_X
       │                                                                 |        │
       └── W13 WGRAD with X_gathered ───────────────→ grad_W13           |        └── BF16 W13 WGRAD with X_bw ─────────────────→ grad_W13
                                                                         |
              planner rewind                                             |               shared BF16 planner rewind
```

At the shared BF16 gradient-math boundary, NVFP4 supplies these reconstructed
operands:

```text
BF16:   X_gathered, H2, W13, W2
NVFP4:  DQ(Q(X_gathered)),
        DQ(Q(SwiGLU(H1))),
        DQ(Q(W13)),
        DQ(Q(W2))
```
