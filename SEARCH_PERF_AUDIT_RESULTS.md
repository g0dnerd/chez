# Search performance audit — results

Measurement-driven audit of the `searchParallel` / NNUE inner loop for self-play
throughput (nodes/s), no playing-strength change. All kept changes are
**bit-identical** (verified by the reproducible `bench` node signature: identical
total node count ⇒ identical search tree ⇒ identical eval).

## Methodology
- **Profile:** `perf record -e task-clock` (frame pointers, ReleaseFast `x86_64_v3`),
  20k samples, `bench --depth 12 --threads 1 --nnue`. Deterministic enough for a
  relative hotspot breakdown.
- **Timing:** interleaved A/B harness (`scripts/nps_ab_awk.sh`), pinned to one core,
  ≥8 reps, verifies the node signature is identical between A and B. Noise floor
  CoV ≈ 0.2–0.4%; ≳1% is a real signal.
- **Hardware:** measured on the **fleet** host (AMD EPYC 4245P, Zen 4 — the real
  paid target) and an AMD Ryzen 7700X bench host (Zen 4 proxy). The 7700X turned
  out to *understate* the win; the fleet is the number that matters.

## Profiled hotspot breakdown (depth 12, 1T, NNUE)
| Bucket | % self | Notes |
|---|---|---|
| **NNUE eval** | **~55%** | `evaluateRawFromAccumulator` 43.5% (≈30% `vpmaddwd` MACs / ≈28%-of-fn horizontal reductions), `refreshPerspective` 4.5% (king-move refresh, inherent to HalfKP), accumulator delta `subVec`+`addVec`+memcpy ≈6.4% |
| Movegen | ~15% | `movesForPiece` 4.5%, `legalMoves`/`legalCaptures` 6%, `isSquareAttackedBy` 2.4% |
| TT probe | 8.9% | 94% on a single load = cache-miss stall (16 MB TT spills L2/L3) |
| Move ordering | ~7.7% | `scoreAll` (MVV-LVA + history/cont-hist) 4.2%, `pickNext` 3.5% |
| Search logic | ~6.4% | `negamax` 5.0%, `quiescence` 1.4% |
| make/unmake | ~5.1% | incl. one cached `isSquareAttackedBy` for check detection |

**Key structural finding:** the loop is **eval-bound**. The scalar work (movegen,
TT, ordering, make/unmake) runs on different execution resources and *overlaps*
with the dominant NNUE vector work, so shaving scalar cost rarely moves wall-clock.
Only reducing the NNUE eval itself helps — and within the eval, **memory traffic**
is reclaimable while **compute is already near its floor** on this hardware.

## Kept optimizations (all bit-identical, measured nodes/s)
| # | Change | File | Fleet (EPYC) | 7700X |
|---|---|---|---|---|
| 1 | **Fused accumulator delta** — gather the ≤1 add / ≤2 sub feature rows and apply them in **one pass** (`child = parent + add − sub0 − sub1`) instead of a 256-wide memcpy + up to four separate in-place add/sub passes that each re-load and re-store the whole accumulator. | `nnue.zig` `applyDeltaPerspective` | **+3.65%** | +2.41% |
| 2 | **Trust cached `state.in_check`** in `legalMoves`/`legalCaptures`/`hasAnyLegalMove` instead of recomputing a full attack scan (`makeMove`/`fromFen` already maintain it). | `movegen.zig` | bundled | ~0% |
| 3 | **Prefetch the TT bucket** as soon as the hash is known in `negamax`, ahead of the repetition/material checks, to hide part of the probe miss. | `search.zig` | bundled | ~0% |
| | **Cumulative (1+2+3)** | | **+4.09%** | +2.41% |

2+3 together add **+0.48%** on the fleet (tight, every rep positive) and ≈0% on
the 7700X. The prefetch is expected to help **more in production** (128 MB TT,
multi-threaded) where more probes miss to memory; kept because it is zero-risk.
The single big win (#1) is purely a memory-traffic reduction, which is why the
EPYC — more memory-sensitive than the 7700X — gains more from it.

## Rejected (measured, reverted — useful negative results)
| Attempt | Result | Why it failed |
|---|---|---|
| FC horizontal-reduction combine (`vphaddd` tree over 4 outputs) | **−2.9%** | `vphaddd` is a slow 3-µop insn on Zen 4, and grouping outputs **broke the ILP** the per-output reductions had (they overlap the next output's `vpmaddwd`). The profile's "28% on the reduce line" overstated the *reclaimable* cost. |
| Force-unroll the FC output loop (`inline for`) | **−7.3%** | 32× full unroll blows the µop cache / bloats code. |
| 512-bit AVX-512 width (`-Dmarch=avx512`) | **−0.3%** | Zen 4 (both 7700X and the EPYC fleet) double-pumps 512-bit ops → no throughput gain from width. |
| AVX-VNNI `vpdpbusd` (u8×i8 FC dot product) | **−26%** | LLVM only emits `vpdpbusd` when each dot product is an **isolated function**; forcing that with `never_inline` costs 64 calls/eval, far more than VNNI saves. Inlined, LLVM falls back to `vpmaddwd`. Not viable for these tiny per-output dot products in Zig/LLVM 0.16 without inline asm (which would break the kore `@Vector` portability convention). |

## Bottom line
- **+4.09% nodes/s on the fleet, bit-identical, zero strength risk** — concentrated
  in change #1 (fused accumulator delta).
- The NNUE eval *compute* (FC `vpmaddwd` MACs + horizontal reductions) is at its
  practical floor on Zen 4: reductions overlap MACs, 512-bit double-pumps, and VNNI
  is inaccessible to the inliner. Further eval speedups would require an
  architecture change (smaller/king-bucketed net, retrain) — out of scope here.
- Untested lever for later: production self-play uses a 128 MB TT and many threads;
  the TT prefetch (#3) and a TT-size sweep for cache locality (strength-neutral to
  measure, but tree-changing → needs SPRT) are the most promising remaining avenues.
