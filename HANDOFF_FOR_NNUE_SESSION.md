# Handoff → NNUE session (definitive facts only)

From an HCE strength-regression investigation, 2026-06-22. Only confirmed results below.

## Directly relevant to tuning
- **`params.zig` (the texel/SPSA tuner state) does NOT feed the HCE eval.**
  `src/engine/evaluation.zig` uses its own inline `const` values and never imports
  `params.zig`. `evaluation.zig` is byte-identical between the March baseline (`1379da1`)
  and HEAD (`6f54a34`). → SPSA/texel runs do not change actual playing strength unless their
  output is hand-copied into `evaluation.zig`. Verify this wiring before trusting any tune.
- **HCE eval (`--eval none`) is unchanged baseline→HEAD and at full strength.** It is a valid,
  intact reference/bootstrap. No HCE regression exists (see below).

## Measurement facts (use these to design NNUE gauntlets)
- **Chez strength vs capped-SF is hugely thread-sensitive.** Same baseline binary, 2+1, vs
  SF_2400: **0.56 @ threads=1 → 0.89 @ threads=2**. → Always compare at identical thread
  counts; never read across different `Threads` settings.
- **Capped Stockfish (`UCI_LimitStrength`+`UCI_Elo`) is a noisy, miscalibrated yardstick.**
  It reads high vs CCRL, saturates (baseline beat SF_2400 +28−2 = uninformative, ±249 Elo on
  30 games), and the informative band shifts with TC. → For A/B decisions use **direct
  engine-vs-engine head-to-head** (lowest variance), not absolute SF rating estimates.
- **Absolute Elo is not comparable across time controls.** 5+0.05 ≠ 2+1: faster TC = weaker
  play and the useful opponent band moves down. Pick one TC and stay on it.
- **Installed Stockfish = `dev-20260218`** (predates the March baseline run, so no SF
  calibration drift between then and now).
- **Quick 30-game SF cells carry ~±250 Elo.** Need ~100+ games/cell for ~±50 Elo.

## The regression scare was a measurement artifact, not code
The "HCE dropped ~300 Elo" alarm was: threads 2→1 in the re-checks (~150 Elo) + fast TC +
tiny-sample SF noise. Confirmed no regression by two completed independent methods (direct
head-to-head: baseline−HEAD = −15 ± 45 Elo; fast SF gauntlet: ~−10 Elo) and a 2+1 gauntlet
in progress (completed cells tied). Texel/eval/time-management/June-search-rewrite all cleared.

## Reusable tooling left in repo (untracked)
- `scripts/remote_gauntlet.sh` + `scripts/remote_gauntlet_run.sh` — provision a fresh host
  (copies prebuilt stockfish+fastchess, no install/zig needed) and run a capped-SF gauntlet.
- `testing/gauntlet.json` (2+1, threads=4, 100 rounds, 8moves book) and
  `testing/gauntlet_fast.json` (5+0.05 quick A/B). `testing/books/` now has the real
  `8moves_v3.pgn` (was a 404 stub) + `UHO_4060_v3.epd`.
- HCE mode = empty `EvalFile` (no `data/net.nnue` present) + `OwnBook=false`.
