# NNUE Pipeline Audit

## Context

net_v2 (trained Jun 7) wins ~7% against HCE. All subsequent nets (v3-v5) trained with the current worktree changes are drastically weaker (0 wins vs HCE). A thorough code review of the export/inference pipeline confirmed all weight layout math is correct — the regression is NOT a transposition or indexing bug. This audit should investigate the full selfplay → training → quantization → play pipeline for correctness issues and improvement opportunities.

## What Has Been Verified Correct

These areas have been manually traced and cross-verified:

1. **GPU weight layout**: `kore.ml.Linear` stores `[in_features, out_features]`. `kore.ml.SparseLinear` (FT) also stores `[in_features, out_features]`.
2. **Export buf indexing**: `buf[i * out + j]` correctly reads weight(input=i, output=j) from the GPU buffer.
3. **Export struct storage**: `net.fc1_weights[j][i]` correctly stores in output-major `[out][in]` for `linearForward_i8`.
4. **v1 load (transpose)**: Correctly transposes `[in][out]` on disk → `[out][in]` in struct.
5. **v2 load (memcpy)**: Correctly copies `[out][in]` on disk → `[out][in]` in struct.
6. **linearForward_i8**: Computes `output[j] = bias[j] + dot(input, weight[j*in..])` — correct for output-major weights.
7. **shiftClippedRelu_i8**: Equivalent to old `divTrunc(..., 64) + clippedRelu` for the [0,127] clamp range.
8. **Quantization scale chain**: FT×127→i16, FC×64→i8, bias×(127×64)→i32, output÷(127×64)→cp.
9. **Parameter ordering**: model.parameters() returns ft.weight, ft.bias, fc1.weight, fc1.bias, fc2.weight, fc2.bias, out.weight, out.bias — matches export assumptions.
10. **MseLoss gradient**: Sets grad directly on the predicted tensor; `graph.backward(output)` replays tape in reverse using this grad. Correct.
11. **Training matmul**: `graph.matmul(input[batch,in], weight[in,out]) → output[batch,out]` — standard convention.

## What Changed Between v2 and v3-v5

All changes are visible via `git diff HEAD` on the relevant files.

### Training hyperparameters (train_nnue.zig)
| Parameter | v2 (committed HEAD) | v3-v5 (working tree) |
|-----------|---------------------|----------------------|
| default_epochs | 100 | 50 |
| default_lr | 0.001 | 0.005 |
| default_lambda | 1.0 | 0.75 |
| weight_decay | 0 (not supported) | 0.03 |
| default_lr_min | 0.0001 | 0.0001 |

Note: CLI args can override defaults. The handoff doc says v2 was trained with `lr=0.005, epochs=50`, suggesting CLI overrides were used. If so, the only training diffs are weight_decay (0→0.03) and lambda (1.0→0.75).

### Training data
| Net | Data file | Records |
|-----|-----------|---------|
| v2 | data/selfplay_data_merged.bin | 29,547,019 |
| v3-v5 | data/selfplay_final.bin | 40,273,506 |

Both datasets were generated using HCE (selfplay passes `network=null` to search).

### Adam optimizer (kore Adam.zig)
Added decoupled weight decay: `param *= (1 - lr * weight_decay)` before Adam update. Applied to ALL parameters including the sparse FT.

## Comparison Test Results (v2 vs v5)

A comparison test (currently in `nnue.zig`, test "compare v2 vs v5") produced:

```
FC1: v2 5906/16384 nonzero, abssum=51574 | v5 6849/16384 nonzero, abssum=52640
FC2: v2 360/1024 nonzero, abssum=26034  | v5 350/1024 nonzero, abssum=10721
FT:  v2 7.7M/10.5M nonzero, absmax=120 | v5 8.9M/10.5M nonzero, absmax=257

Output weights (all saturated ±127/-128):
  v2: 21 positive, 9 negative, 2 zero, bias=+80782
  v5: 19 positive, 6 negative, 7 zero, bias=-269719

Eval comparison (quantized cp, before nnue_scale):
  startpos:       v2=+19, v5=-26
  white up queen: v2=+51, v5=+4
  white dn queen: v2=-8,  v5=-45
  Ruy Lopez:      v2=+13, v5=-28
```

**Key observation**: v5's large negative output bias (-269719 / 8128 = -33cp) shifts all evals negative. Combined with fewer non-zero output weights, v5's maximum positive eval is only ~5cp (10cp after nnue_scale=2). This severely limits the engine's ability to play.

## Audit Checklist

### 1. Selfplay Data Generation (`src/selfplay.zig`)

- [ ] **Score perspective**: The search returns scores from STM perspective. Verify the score stored in the record correctly represents STM advantage. Check both the search result handling and the bufferPosition logic.
- [ ] **WDL encoding**: Game outcome 0=white_wins, 1=black_wins, 2=draw. Verify this is correct throughout (selfplay writer → dataloader reader → target computation).
- [ ] **Position filtering**: Positions are skipped when `in_check`, `ply < 16`, `score > 3000`, `ply % 4 != 0`. Are these filters appropriate for NNUE training? Some engines skip captures and checks differently.
- [ ] **Adjudication**: Games are adjudicated after 5 consecutive moves where |score| > 1500. Verify the winning side tracking is correct. Does the adjudication WDL label match what would actually happen?
- [ ] **Random opening**: 8 random plies before depth search. Is this sufficient diversity? Could systematic biases in the opening phase affect training data quality?
- [ ] **Search depth**: Default depth 8, single-threaded, no NNUE. For HCE-scored training data, is depth 8 sufficient for the training signal? Higher depth gives more accurate scores but takes longer.
- [ ] **Data volume and diversity**: 29-40M records. Is this enough for a 10.5M parameter network? Check if the position distribution covers enough of the feature space.

### 2. Data Encoding (`src/selfplay/serde.zig`)

- [ ] **Round-trip correctness**: Unit tests exist. Verify edge cases: en passant encoding (file+1 offset), castling rights, halfmove clock (7 bits → max 127).
- [ ] **Bit-level fidelity**: The Huffman encoding for pieces — verify the decode matches the encode for all piece types. Check the pawn code (1 bit), knight (2), bishop (3), rook (4), queen (4).

### 3. DataLoader (`src/trainer/dataloader.zig`)

- [ ] **Score perspective alignment**: Score is i16 from STM perspective. The dataloader computes `sigmoid(score / 400)`. This should produce the win probability from STM's perspective.
- [ ] **WDL flip for black**: `wdl_value = if (stm == black) 1.0 - wdl_raw else wdl_raw`. The raw WDL has 0=white_wins (1.0), 1=black_wins (0.0). When STM is black: white_wins should be 0.0 (STM loses), so `1.0 - 1.0 = 0.0`. Correct? Trace all cases.
- [ ] **Feature extraction perspective**: `activeFeatures(&state, stm)` for STM, `activeFeatures(&state, opp)` for OPP. Verify this matches the inference code's perspective handling.
- [ ] **Target blend**: `lambda * score_sigmoid + (1 - lambda) * wdl_value`. With lambda=0.75, the target blends 75% score + 25% outcome. Is the sigmoid scale (1/400) calibrated for HCE score ranges?

### 4. Training Model (`src/trainer/model.zig`)

- [ ] **FT weight sharing**: The same FT is used for both STM and OPP perspectives (different indices, same weights). This is correct for HalfKP but verify `setIndices` doesn't leak state between the two forward calls.
- [ ] **Concat order**: `concat(stm_relu, opp_relu)` → STM first [0..256), OPP second [256..512). Does this match inference?
- [ ] **ClippedReLU max_val**: Training uses `ClippedReLU(1.0)` (clips to [0,1]). Inference CReLU clips to [0,127] (i16→i8 with scale 127). The quantization factor of 127 should account for this difference. Verify.

### 5. GPU Training Pipeline (`src/train_nnue.zig`)

- [ ] **Weight decay on sparse FT**: Adam applies `param *= (1 - lr * wd)` to ALL parameters every step. For the FT with 40,960 input features and ~30 active per sample, inactive feature rows get decayed but never receive gradient updates in a given step. Over many steps, rarely-used features could be pushed toward zero. Investigate: what fraction of the 40,960 feature indices actually appear in the training data?
- [ ] **Lambda sensitivity**: With lambda=1.0, training is purely score-based (no game outcome noise). With lambda=0.75, 25% of the target comes from win/draw/loss. If the game outcomes are noisy (e.g., games decided by adjudication or time), this noise enters the training signal. Check if lambda=0.75 hurts.
- [ ] **Learning rate**: 0.005 is relatively high for Adam. Combined with cosine decay and weight decay, check if the final weights are in a good range for i8 quantization.
- [ ] **Gradient clipping**: `clipGradNorm` at 1.0. Is this appropriate? Too aggressive clipping can slow convergence.

### 6. Quantized Export (`src/trainer/export.zig`)

- [ ] **Saturation check**: After quantization, check what fraction of FC/output weights saturate at ±127 (i8) or ±128. High saturation means loss of relative weight information. The v2 and v5 comparison shows ALL output weights are saturated. Consider whether the quantization scale (64) is appropriate, or if the float weights have grown too large.
- [ ] **Dynamic range**: The quantized net's eval range is very narrow (v2: ~[-8, 52] cp, v5: ~[-45, 5] cp). For competitive play, NNUE evals should have ranges of ±1000+ cp. This could be a quantization issue or a training issue (weights too small).

### 7. Inference (`src/engine/nnue.zig`)

- [ ] **Incremental accumulator**: `recordMove` and `applyDeltaPerspective` handle king moves, captures, en passant, castling, promotions. Verify each case adds/removes the correct features. The debug assertion (every 1024th eval compares incremental vs full recomputation) helps, but only fires in Debug builds.
- [ ] **Perspective ordering in eval**: `evaluateRawFromAccumulator` concatenates `acc.values[stm]` first, `acc.values[opp]` second. Must match training concat order.

### 8. Search Integration (`src/engine/search.zig`)

- [ ] **NNUE scale**: `nnue_scale = 2` doubles NNUE cp values to approximate HCE scale. With v5's narrow range, the scaled range is [-90, 10]. This interacts poorly with pruning margins (futility at 300/600, RFP at 80*depth). The engine will almost never trigger RFP (needs eval - 80*depth ≥ beta, hard when eval max is +10), and futility pruning may incorrectly prune at shallow depths.
- [ ] **Interaction with TT**: TT scores are clamped to i16. NNUE scores (after scale) are tiny, so this shouldn't overflow, but verify.

## Recommended Experiments

### Quick isolation tests (priority order)

1. **Retrain with v2 hyperparams on v5 data**: `--data data/selfplay_final.bin --epochs 50 --lr 0.005` with lambda=1.0 and weight_decay=0 (revert train_nnue.zig changes, keep export/inference changes). If the resulting net plays well, the issue is purely hyperparams.

2. **Retrain with v5 hyperparams on v2 data**: `--data data/selfplay_data_merged.bin --epochs 50 --lr 0.005` with current working tree code. If the net plays badly, the issue is hyperparams not data.

3. **Weight decay = 0 only**: Revert only weight_decay to 0, keep lambda=0.75. Tests whether weight decay alone causes the regression.

4. **Lambda = 1.0 only**: Keep weight_decay=0.03 but revert lambda to 1.0. Tests whether WDL blending causes the regression.

### Deeper investigations

5. **FT feature coverage**: Count unique feature indices across the training data. If many of the 40,960 features are never or rarely seen, weight decay will zero them out, and the FT will have dead zones.

6. **Float vs quantized eval comparison**: After training, compare the float model's eval (via GPU forward pass) with the quantized .nnue eval on the same positions. If the float model discriminates well but the quantized model doesn't, the issue is quantization granularity.

7. **Output weight analysis**: Why are ALL output weights saturated at ±127? This happens when float weight magnitudes exceed 127/64 ≈ 2.0. Consider:
   - Reducing the output weight quantization scale (e.g., 32 instead of 64)
   - Adding L2 regularization specifically on output weights
   - Checking if the float output weights really need to be that large

8. **Data quality deep dive**: Sample 1000 records from each dataset, decode positions, evaluate with HCE, and compare the stored scores vs fresh HCE evals. Check for systematic differences (score perspective, encoding errors, stale TT values leaking into scores).

## Files Reference

| File | Role |
|------|------|
| `src/selfplay.zig` | Self-play data generation |
| `src/selfplay/serde.zig` | Position encoding/decoding |
| `src/trainer/dataloader.zig` | Training data batching and target computation |
| `src/trainer/model.zig` | GPU training model (SparseLinear FT + Sequential dense) |
| `src/trainer/export.zig` | GPU float → quantized .nnue export |
| `src/train_nnue.zig` | Training entry point, hyperparameters |
| `src/engine/nnue.zig` | Network struct, load, inference, accumulator stack |
| `kore/src/ml/src/Adam.zig` | Adam optimizer with weight_decay |
| `kore/src/ml/src/Linear.zig` | Dense linear layer (weight shape: [in, out]) |
| `kore/src/ml/src/SparseLinear.zig` | Sparse FT layer (weight shape: [in, out]) |
| `kore/src/ml/src/cpu/quantized.zig` | linearForward_i8, shiftClippedRelu_i8, dotProduct_i8 |
| `kore/src/ml/src/cpu/ops.zig` | clippedRelu_i16, addVec_i16, subVec_i16 |

## Diagnostic Commands

```bash
# Run v2 vs v5 comparison test
zig build test -Dtest_filter="compare v2 vs v5" --summary all

# Run layer-by-layer diagnostic on a net (edit path in nnue.zig:805)
zig build test -Dtest_filter="diagnose" --summary all

# Quick elo test
fastchess \
  -engine cmd=./zig-out/bin/uci name=test proto=uci "option.EvalFile=data/net_v5.nnue" "option.Threads=4" "option.OwnBook=false" \
  -engine cmd=./zig-out/bin/uci name=hce proto=uci "option.Threads=4" "option.OwnBook=false" \
  -each tc=1+0.01 restart=on timemargin=300 \
  -rounds 20 -repeat -recover -concurrency 4 \
  -draw movenumber=40 movecount=10 score=5 \
  -resign movecount=5 score=1000 \
  -output format=cutechess

# Train with specific settings
./zig-out/bin/train_nnue --data data/selfplay_data_merged.bin --export data/test.nnue --epochs 50 --lr 0.005
```

## Elo Results for Reference

| Net | Training | vs v2 | vs HCE |
|-----|----------|-------|--------|
| v2 | merged.bin, wd=0, lambda=1.0 | — | wins ~7% |
| v3 | final.bin, wd=0.1 (FT-only?) | 0-35-5 | 0-40-0 |
| v4 | final.bin, wd=0.1 global | 5-15-20 (-89 Elo) | 1-35-4 (-436 Elo) |
| v5 | final.bin, wd=0.03 global | 1-36-3 (-470 Elo) | 0-40-0 |
