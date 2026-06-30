# HCE strength-regression investigation

Branch `fleet-selfplay-tooling` @ `6f54a34` vs the 2026-03-19 baseline
(`testing/results/gauntlet_20260319_201445.json`, pure-HCE, ~2720 / 93.3% vs SF_2400 @ 2+1).

## FINAL VERDICT: no meaningful HCE regression (confirmed two independent ways)

1. **Direct head-to-head** (HCE both sides, 5+0.05, threads=1, 143 games):
   Elo(baseline − HEAD) = **−15 ± 45** (HEAD nominally slightly stronger).
2. **vs-Stockfish gauntlet** (5+0.05, **threads=4**, 80 games/cell, sequential):
   HEAD − baseline = **−9 Elo** (all 4 opp) / **−11 Elo** (informative SF_2400+2600 cells),
   per-engine CIs ±40–52 → difference includes zero.

Both methods agree: baseline ≈ HEAD, HEAD perhaps a hair weaker but within noise. The original
"~300 Elo / 2720→far-weaker" alarm was a **measurement-artifact stack**: threads 2→1 in the
re-checks (~150 Elo; the *faithful* baseline binary itself scores 0.56@1t vs 0.89@2t vs SF_2400),
a fast TC, and 30-game SF cells with ±249 Elo error bars. The baseline binary was verified
faithful by reproducing its own March SF_2400 cell (+24−2 ≈ recorded +28−2) at the exact original
threads=2 config. Texel, eval logic, time management, and the June search rewrite are all cleared.

Gauntlet config lessons baked into `testing/gauntlet.json` (threads=4, ladder up, 100 rounds,
8moves book) and `testing/gauntlet_fast.json` (5+0.05 companion).

---
## (history) STATUS: REOPENED — possible thread-scaling regression

Earlier conclusion ("no regression") was based on a **threads=1** head-to-head and is now in
question. New evidence:

- **The baseline binary is faithful.** Run through the *exact* March gauntlet config (2+1,
  **threads=2**, conc=6, SF 2400/2600/2800, bookless, HCE), `baseline_1379da1` reproduces its
  March result: vs SF_2400 it is +5−0=0 (1.000) early, tracking the recorded 0.933. The
  Zig-compat shims applied to build it are pure API migrations (timer/mutex/Io), strength-neutral.
- **Chez's strength vs capped-SF is hugely thread-sensitive at this config:** baseline scores
  ~0.56 at threads=1 but ~0.95 at threads=2 vs SF_2400. So the threads=1 anchors were the wrong
  config for comparing to the (threads=2) March baseline.
- **Implication:** the threads=1 head-to-head (baseline ≈ HEAD) would NOT detect a regression
  that only appears under lazy-SMP. The June commits reworked threading (`286111b` thread
  SearchContext, per-ply search stacks; `a4bad5c` inner-loop perf) — a plausible mechanism for
  worse multi-thread scaling with identical single-thread play.

**Decisive test in progress:** run BOTH baseline and HEAD through the identical threads=2
gauntlet and compare cell-by-cell. A hint exists that HEAD may scale worse: `a4bad5c@threads2`
vs SF_2400 was 0.819 (36g) vs baseline's incoming ~0.95 — but that needs confirmation at full
game count.

### Earlier (threads=1) head-to-head — still valid for single-thread play
Direct baseline vs HEAD, HCE both sides, 5+0.05, threads=1, 143 games: **Elo(base−HEAD) =
−15 ± 45** (statistically even). Holds for 1-thread play; says nothing about SMP scaling.

Breakdown of the apparent gap vs Stockfish:
- **~150 Elo: threads confound.** The baseline gauntlet ran **threads=2**; the first re-checks
  ran threads=1. At 2+1, threads 1→2 lifts a4bad5c from 0.594 → 0.819 vs SF_2400.
- **noise: the SF_2400 cell is statistically too weak** to anchor a regression (baseline
  recorded +458 **±249** Elo on 30 games; a4bad5c@threads2 +223 falls inside that CI).
- **the original "0-2 vs SF_2200/2600" quick check** was 2 games each at 5+0.05 — pure noise
  on top of the threads downgrade.
- **~40 Elo: a real, modest regression**, confirmed by direct head-to-head (below).

The handoff's prime suspect (overfit **texel tune**) is conclusively ruled out, as are eval
logic, time management, NPS, and any single "offending commit." The ~40 Elo is small cumulative
drift across the NNUE-era search changes, the largest identifiable piece being the **June
search-pruning overhaul** (LMP tightening + SEE/history pruning), which was validated on NNUE
**label fidelity**, never on HCE **game strength**.

### Direct head-to-head (the definitive measure), HCE both sides, 5+0.05, UHO openings
- `baseline_1379da1` vs **real HEAD**: **0.479 / 143 games → Elo(base−HEAD) = −15 ± 45**.
  Statistically even; HEAD nominally slightly stronger.
- `baseline_1379da1` vs `a4bad5c`: 0.557 / 115 games (~+40, high side of noise).
- `HEAD` vs `a4bad5c`: 0.495 / 98 games (neutral).
- All three overlap heavily — the true baseline↔June-family delta is within noise of zero.
  No smoking-gun commit; no regression to bisect.

## What was ruled out (with evidence)

| Suspect | Verdict | Evidence |
|---|---|---|
| Texel-tuned `params.zig` | RULED OUT | HEAD `params.zig` is byte-identical to the pre-texel `original_params.zig`; commit `1aab183` (Apr 5) reverted the texel values (166/291 pawn) back to originals (126/208). |
| HCE eval logic/constants | RULED OUT | `git diff 1379da1 HEAD -- src/engine/evaluation.zig` is **empty**. Eval uses **inline** consts and never imports `params.zig` — the texel/SPSA machinery is tuner-only and never fed HCE play. |
| Time management | RULED OUT | `uci.zig` budget `t/20 + inc/2` byte-identical baseline→HEAD. |
| June search rewrite (cluster B, Jun 15-16) | NOT THE CAUSE | `a14b813` (pre-rewrite) vs `a4bad5c` (post-rewrite) head-to-head = ~0.53 (HCE, 5+0.05, UHO). ~20-40 Elo, not 250. |
| June SEE/LMP batch (cluster A, Jun 20) | NOT THE CAUSE | HEAD vs `a4bad5c` (its parent) = 0.495 over 98 games. Neutral. |
| NPS / throughput collapse | RULED OUT | All binaries 1.3-1.8M nps at depth 16, single thread. No collapse. |
| Stockfish calibration drift | UNLIKELY | Installed SF = `dev-20260218` (Feb 18), predates the Mar-19 baseline gauntlet. |

## Anchors vs SF_2400 (UCI_LimitStrength), 2+1, bookless

| Build | Threads | Score vs SF_2400 | ~Elo |
|---|---|---|---|
| baseline `1379da1` (RECORDED, Mar 19) | 2 | 0.933 (+28-2) | — |
| `a14b813` (Jun 7, pre-rewrite) | 1 | 0.533 (30g) | ~+23 |
| `a4bad5c` (Jun 20, post-rewrite) | 1 | 0.594 (80g) | ~+66 |
| `a4bad5c` | 2 | 0.783 (30g) | ~+223 |
| baseline `1379da1` | 1 / 2 | *pending binary* | — |

The whole June family (`a14b813` → HEAD) clusters at ~0.55-0.59 vs SF_2400 @ threads=1,
far below the recorded 0.933. Since they're all ≈ each other, the gap is **not** in the
search rewrite — it is either pre-June or the threads(2→1) confound.

### The threads confound is large, and SF_2400 is statistically too weak to conclude

- **threads 1→2 lifts a4bad5c from 0.594 → 0.783 vs SF_2400** (~+150 Elo). The baseline
  gauntlet used threads=2; my first re-checks used threads=1. Much of the apparent drop
  was this config mismatch, not code.
- The baseline's **SF_2400 cell is +458 ±249 Elo on 30 games** (true margin [+209,+707]).
  a4bad5c@threads2 ≈ **+223 Elo falls *inside* that CI** → not significantly different.
  The "beat SF_2400 28-2" headline is too noisy to establish a regression on its own.
- The baseline's informative cells were **SF_2600 (+108 ±118, est 2708)** and
  **SF_2800 (−95 ±109, est 2705)**. The real test is a4bad5c@threads2 vs those two.

## Decisive remaining test

Run the **baseline binary `1379da1`** vs the *same* SF_2400 at threads=1 **and** threads=2:
- If baseline@threads1 ≈ 0.59 (like a4bad5c) → **no code regression**; the "0.933" was the
  threads=2 config, and current weakness vs SF is the threads downgrade + sample noise.
- If baseline@threads1 ≈ 0.90 → **real regression**, located between `1379da1` (Mar 14) and
  `a14b813` (Jun 7): bisect the April NNUE-integration commits `136d210`, `1aab183`, `957c755`.

## Tooling notes
- Old commits need contemporaneous Zig (the std.Io migration lands between Jun 7 and Jun 20;
  current Zig builds Jun-15+ only). Prebuilt binaries live at `/tmp/uci-<shorthash>`.
- HCE mode = `EvalFile` empty (no `data/net.nnue` present), `OwnBook=false`.
- `testing/books/8moves_v3.pgn` was a 404 HTML page; replaced with the real Stockfish book.
  Added `testing/books/UHO_4060_v3.epd` (low-variance openings) for A/B head-to-heads.
