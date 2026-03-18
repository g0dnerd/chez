# Texel Tuning — Design Notes & Status

## Overview

Chez uses a Texel-style tuner to optimize evaluation parameters by minimizing the
mean squared error between the engine's static evaluation (mapped through a sigmoid
to [0,1]) and game outcomes from a labeled position dataset. The implementation lives
in `src/tune.zig` and `src/tuner/`.

## Algorithm

**SPSA (Simultaneous Perturbation Stochastic Approximation)** rather than the more
common analytical gradient approach. The eval function is linear in its features, so
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
- **Calibration pass**: Estimates a good `a` value by running ~50 dry iterations and
  measuring average |g_hat|, targeting a first-step size of 2.0 float units. The
  calibrated value is auto-applied unless `--a` is explicitly provided.
- **Float-space optimization**: All SPSA work happens in f64 space (`ParamsF64`) to
  avoid i16 quantization noise. Rounding to i16 only happens at checkpoint writes and
  final output.

## Parameter Space

**1,055 parameters** total:

- 6 piece values (12 floats: mg+eg) — king frozen at 20000/20000
- 8 passed pawn bonuses (16 floats) — rank 0 and 7 frozen at 0
- 4×28 mobility bonuses (224 floats)
- 17 named Score scalars (34 floats) — bishop pair, rook bonuses, pawn structure, etc.
- 1 king_proximity_passer (i16, 1 float)
- 6×64 piece-square tables (768 floats) — pawn rows 0 and 7 frozen at 0

### Dead mobility parameters (resolved)

The mobility arrays are fixed at [4][28] but each piece type has a different maximum
move count (knight=8, bishop=13, rook=14, queen=27). The 46 unreachable Score slots
(92 float entries) — knight indices 9-27, bishop 14-27, rook 15-27 — are now frozen
at 0 in the SPSA frozen-entries list (~8.7% of parameter count). Previously these
were random-walking, wasting gradient budget and injecting noise.

## Eval Parity

The evaluation function was made generic over the Score type (`Score(i16)` vs
`Score(f64)`), so both the production eval path and the tuner's
`evaluateWithParamsF64` share identical logic. This eliminates the risk of
optimizing against a subtly different objective.

## Dataset

- **Source**: Zurichess quiet-labeled EPD dataset
- **Size**: ~1.4M positions (~1,327 positions per parameter — on the low side;
  5,000+ per parameter is preferable)
- **Labels**: Game outcomes (1-0, 1/2-1/2, 0-1), not position-level adjudications.
  Game-outcome labels are inherently noisy since a +3 eval position might be drawn
  due to later blunders.
- **Quiet filtering**: No additional validation beyond what the dataset provides.
  Non-quiet positions (captures/checks in progress) distort the loss surface because
  the static eval doesn't account for hanging pieces.
- **Side-to-move normalization**: Results are flipped when black is to move so the
  eval and label share the same perspective.

### Dataset concerns

- The 7M-position Zurichess dataset was not available; the 1.4M dataset is small
  relative to the parameter count.
- Game-outcome labels add noise vs. position-level adjudications from a strong engine.
- No independent quiet-position validation — relying on the dataset being pre-filtered.

## K (Sigmoid Scaling)

K is tuned once via ternary search over [0.5, 3.0] before SPSA begins, then frozen.
Converged to **K ≈ 1.4**. This compensates for the engine's inflated internal scale
(pawn ≈ 243mg/220eg rather than the traditional ~100 centipawns). The sigmoid
formula is `1 / (1 + exp(-K * eval / 400))`.

## Results

- **MSE behavior**: Plateaus within the first ~50k iterations of a 500k-iteration run.
  Early stopping detects this and terminates when relative improvement over a
  5,000-iteration window drops below 1e-6.
- **Early plateau causes** (likely contributing factors):
  1. SPSA's inherent inefficiency with 1,055 simultaneous parameters
  2. Dataset size (1.4M) may be insufficient for the parameter count
  3. Game-outcome label noise setting a MSE floor
  4. PSTs (768/1055 = 73% of params) are noisy to tune per-square since each square
     appears in only a fraction of positions

## Perturbation Scaling

Per-parameter perturbation scaling is applied via a comptime `c_scales` array that
multiplies the base `c_t` perturbation. Piece values (large magnitude) get 5× scaling,
passed pawn bonuses get 2×, PSTs (small magnitude) get 0.5×, and mobility/scalars
stay at 1×. This controls gradient estimate quality without affecting step size.

**Important**: `c_scales` must only appear in the perturbation construction, not in
the gradient denominator. Dividing the gradient by `c_scales[i]` inversely scales the
step size — PSTs (0.5×) would get 2× larger steps and piece values (5×) would get 5×
smaller steps, causing PSTs to random-walk to extreme values. The calibration pass in
`tune.zig` uses `c_scales` in its perturbation to match the SPSA run.

## Potential Improvements

In rough priority order:

1. **Larger/better dataset** — find or generate a 5M+ position dataset; consider
   generating positions from self-play with quiescence search to ensure quiet positions
2. **Staged tuning** — tune material + simple terms first with PSTs frozen, then
   tune PSTs with material frozen; reduces effective dimensionality dramatically
3. **Position-level labels** — use a strong engine to adjudicate positions rather than
   game outcomes, for a cleaner training signal
4. **Larger batch sizes** — may help with noisy game-outcome labels
5. **Analytical gradients** — would converge dramatically faster since the eval is
   linear in features, but requires maintaining gradient code for every eval term

### Already implemented

- ~~Freeze dead mobility slots~~ — 46 Score slots (92 floats) now frozen
- ~~Per-parameter perturbation sizes~~ — group-based `c_scales` array (perturbation-only)
- ~~Early stopping~~ — ring buffer detects MSE plateau, auto-terminates
- ~~Auto-apply calibration~~ — calibrated `a` applied unless `--a` is set

## Architecture

```
src/tune.zig           Main binary: CLI parsing, K-tuning, calibration, orchestration
src/tuner/
├── dataset.zig        EPD parser (Zurichess c9 format), side-to-move normalization
├── mse.zig            Parallel MSE computation (full-dataset and batch/indexed)
├── spsa.zig           SPSA optimizer with frozen param support, checkpoint writes
└── codegen.zig        Regenerates params.zig by splicing the default_params block
src/engine/
├── params.zig         Params (i16) and ParamsF64 (f64) with flat-array serialization
├── score.zig          Generic Score(T) supporting i16 and f64
└── evaluation.zig     Generic evaluate function used by both production and tuner
```

## Usage

```bash
zig build tune -- --dataset path/to/positions.epd
zig build tune -- --dataset positions.epd --k 1.4 --skip-calibrate --iterations 100000
zig build tune -- --dataset positions.epd --k-only   # just tune K
```

Output is written to `src/engine/params.zig` by default (atomic write + zig fmt).
Checkpoints are written every 1,000 iterations.
