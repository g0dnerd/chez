# Texel Tuning — Design Notes

## Overview

Chez uses a Texel-style tuner to optimize evaluation parameters by minimizing the
mean squared error between the engine's static evaluation (mapped through a sigmoid
to [0,1]) and position labels from a dataset. The implementation lives in
`src/tune.zig` and `src/tuner/`.

## Algorithm

SPSA rather than the more common analytical gradient approach.
The eval function is linear in its features, so
analytical gradients are feasible and would converge faster per-iteration. SPSA was
chosen for **implementation simplicity**: it requires only two MSE evaluations per
iteration regardless of parameter count, and avoids the need to derive and maintain
gradient extraction code for every eval term. This is a deliberate tradeoff — the
tuner is meant to be re-run as eval terms are added or modified, so minimizing
implementation burden was prioritized over convergence speed.

### SPSA Details

- **Perturbation**: Rademacher ±1 delta vector (all params perturbed simultaneously)
- **Batch size**: 16,384 positions sampled with replacement per iteration
- **Schedule**: Standard SPSA gains sequences — `a_t = a/(t+A)^alpha`, `c_t = c/t^gamma`
- **Calibration pass**: Estimates a good `a` value by running dry iterations and
  measuring average |g_hat|, targeting a first-step size of 2.0 float units. The
  calibrated value is auto-applied unless `--a` is explicitly provided.
- **Float-space optimization**: All SPSA work happens in f64 space (`ParamsF64`) to
  avoid i16 quantization noise. Rounding to i16 only happens at checkpoint writes and
  final output.
- **Frozen parameters**: King piece value, pawn PST rows 0/7, passed pawn bonus
  ranks 0/7, and 46 unreachable mobility slots (knight 9-27, bishop 14-27, rook 15-27)
  are snapped back to defaults after every gradient step.
- **Early stopping**: Ring buffer tracks full-MSE every 100 iterations; terminates when
  relative improvement over a 5,000-iteration window drops below 1e-6.

### Perturbation Scaling

Per-parameter perturbation scaling via comptime `c_scales` array: piece values 5×,
passed pawn bonuses 2×, PSTs 0.5×, mobility/scalars 1×. `c_scales` appears in both
perturbation construction and the gradient denominator (dividing by `c_scales[i]`)
so that the gradient estimate is unbiased regardless of perturbation scale.

## Parameter Space

**1,055 parameters** total:

- 6 piece values (12 floats: mg+eg) — king frozen at 20000/20000
- 8 passed pawn bonuses (16 floats) — rank 0 and 7 frozen at 0
- 4×28 mobility bonuses (224 floats) — unreachable slots frozen at 0
- 17 named Score scalars (34 floats) — bishop pair, rook bonuses, pawn structure, etc.
- 1 king_proximity_passer (i16, 1 float)
- 6×64 piece-square tables (768 floats) — pawn rows 0 and 7 frozen at 0

## Eval Parity

The evaluation function is generic over the Score type (`Score(i16)` vs `Score(f64)`),
so both the production eval path and the tuner's `evaluateWithParamsF64` share
identical logic.

## Dataset

### Pipeline

Positions are generated through a multi-stage pipeline:

```
PGN games ──► extract_positions.py ──► candidate FENs
                                           │
                                           ▼
                                   quiet-filter (Zig)
                                           │
                                           ▼
                                      quiet FENs
                                           │
                                           ▼
                                   label_positions.py
                                     (Stockfish d15)
                                           │
                                           ▼
                                  dataset.epd (final)
```

1. **Extraction** (`scripts/extract_positions.py`): Extracts FENs from PGN files.
   Skips opening (first 16 plies), late moves (ply > 400), positions in check,
   post-capture positions, and sparse endgames (< 6 pieces). Samples 1 position
   every 4 plies to reduce inter-position correlation.

2. **Quiet filtering** (`zig-out/bin/quiet-filter`): Runs Chez's static eval and
   quiescence search on each position; discards positions where
   `|eval - qsearch| >= threshold` (default 100 internal units). This ensures the
   dataset only contains positions where the static eval is meaningful.

3. **Labeling** (`scripts/label_positions.py`): Runs Stockfish at depth 15 on each
   quiet position. Outputs centipawn evaluations from white's perspective in EPD
   format (`ce` opcode). Discards positions with |eval| > 1000cp. Uses multiprocessing
   with per-position time limits to prevent hanging.

### Current Dataset

- **Source**: Lichess Elite Database (2500+ vs 2300+ rated players)
- **Size**: ~5M positions (~4,700 positions per parameter)
- **Labels**: Stockfish depth-15 centipawn evaluations, converted to [0,1] via
  sigmoid `1 / (1 + exp(-cp / 400))` at parse time
- **Side-to-move normalization**: Results are flipped when black is to move so the
  eval and label share the same perspective

### Supported EPD Formats

The dataset parser (`src/tuner/dataset.zig`) auto-detects two formats per line:

- `c9 "<result>"` — game outcome (1-0, 1/2-1/2, 0-1)
- `ce "<centipawns>"` — Stockfish centipawn eval from white's perspective

## K (Sigmoid Scaling)

K is tuned once via ternary search over [0.5, 3.0] before SPSA begins, then frozen.
The sigmoid formula is `1 / (1 + exp(-K * eval / 400))`. With centipawn labels,
K finds the scaling between Chez's internal units and Stockfish's centipawn scale.

## Architecture

```
src/tune.zig           Main binary: CLI parsing, K-tuning, calibration, orchestration
src/tuner/
├── dataset.zig        EPD parser (c9 game-outcome and ce centipawn formats)
├── mse.zig            Parallel MSE computation (full-dataset and batch/indexed)
├── spsa.zig           SPSA optimizer with frozen param support, checkpoint writes
└── codegen.zig        Regenerates params.zig by splicing the default_params block

src/quiet_filter.zig   Binary: filters FENs for quiet positions using qsearch
src/engine/
├── search.zig         quiescenceEval() public wrapper for standalone qsearch
├── params.zig         Params (i16) and ParamsF64 (f64) with flat-array serialization
├── score.zig          Generic Score(T) supporting i16 and f64
└── evaluation.zig     Generic evaluate function used by both production and tuner

scripts/
├── extract_positions.py   PGN → candidate FENs (python-chess)
└── label_positions.py     Stockfish depth-15 labeling → EPD with ce annotations
```

## Usage

```bash
# Generate dataset
uv run python scripts/extract_positions.py data/pgn/*.pgn > data/candidates.fen
zig-out/bin/quiet-filter --input data/candidates.fen > data/quiet.fen
uv run python scripts/label_positions.py --input data/quiet.fen --output data/dataset.epd

# Run tuner
zig build tune -- --dataset data/dataset.epd
zig build tune -- --dataset data/dataset.epd --k-only          # just tune K
zig build tune -- --dataset data/dataset.epd --skip-calibrate   # skip a calibration
```

Output is written to `src/engine/params.zig` by default (atomic write + zig fmt).
Checkpoints are written every 1,000 iterations.

## Potential Improvements

1. **Staged tuning** — tune material + simple terms first with PSTs frozen, then
   tune PSTs with material frozen; reduces effective dimensionality
2. **Analytical gradients** — would converge dramatically faster since the eval is
   linear in features, but requires maintaining gradient code for every eval term
3. **Larger batch sizes** — may reduce per-iteration noise
