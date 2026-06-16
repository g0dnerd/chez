# NNUE Debug Handoff

## Problem Statement

net_v2 (trained Jun 7, format v1) plays well — wins ~7% of games against HCE. All nets trained since (v3-v5) with the current code are drastically weaker (0-40 vs HCE). The code changed between v2's export and the later nets. The goal is to find and fix the regression.

## Timeline of Changes

1. **v2 exported (Jun 7)** — OLD code: `matmul`-based inference, `fc1_weights: [fc1_in][fc1_out]i8` (input-major struct), no weight decay. Trained on `data/selfplay_data_merged.bin` (29.5M records), lr=0.005, epochs=50. Val_loss=0.019.

2. **User changed inference code (after Jun 7, uncommitted):**
   - Replaced `quantized.matmul` with `quantized.linearForward_i8` + `quantized.shiftClippedRelu_i8` + `quantized.dotProduct_i8` in `evaluateRawFromAccumulator`
   - Transposed FC weight arrays: `fc1_weights: [fc1_out][fc1_in]i8` (output-major)
   - Added new functions to `kore/src/ml/src/cpu/quantized.zig`
   - User reports v2 still plays well after this change (loaded via v1 transpose path)

3. **This session's changes:**
   - Added weight_decay to kore's Adam optimizer
   - Added v1/v2 format versioning to `loadFromBytes` (v1 transposes FC weights on load, v2 memcpys)
   - Fixed export to write `net.fc1_weights[j][i] = quantizeI8(buf[i * fc1_out + j])` (was `[i][j]` which overflowed the new struct layout)
   - Bumped format_version to 2
   - Various training hyperparameter changes

4. **Nets trained this session (all play terribly vs v2 and HCE):**
   - v3: FT-only wd=0.1, 80 epochs, 40M records. Val_loss=0.018.
   - v4: global wd=0.1, 50 epochs, 40M records. Val_loss=0.044.
   - v5: global wd=0.03, 50 epochs, 40M records. Val_loss=0.025.

## Key Files and Their Roles

| File | Role |
|------|------|
| `src/engine/nnue.zig` | Network struct, loadFromBytes (v1/v2 paths), evaluateRawFromAccumulator (inference) |
| `src/trainer/export.zig` | GPU float model → quantized .nnue file (`exportNnue`) |
| `src/trainer/model.zig` | GPU training model (NnueModel, SparseLinear FT + Sequential dense) |
| `src/train_nnue.zig` | Training entry point, hyperparameters, validation |
| `kore/src/ml/src/cpu/quantized.zig` | `linearForward_i8`, `shiftClippedRelu_i8`, `dotProduct_i8`, old `matmul` |
| `kore/src/ml/src/Adam.zig` | Adam optimizer with weight_decay support |

## Weight Layout Chain (Critical)

The GPU model's `ml.Linear(in=512, out=32)` stores weights as a flat buffer. The key question is what layout.

**Old matmul** expected input-major: `b[k * N + j]` where K=in, N=out → `[in, out]`  
**New linearForward_i8** expects output-major: `weight[j * in_features + k]` → `[out, in]`

The export reads the GPU buffer as: `buf[i * fc1_out + j]` (treating it as `[fc1_in, fc1_out]` = input-major).

### v2 path (works):
1. Export: `net.fc1_weights[i][j] = buf[i*32+j]` into old struct `[512][32]` → input-major on disk
2. Load: v1 path transposes `[in][out]` → `[out][in]` in new struct
3. Inference: `linearForward_i8` reads output-major → correct

### v5 path (broken):
1. Export: `net.fc1_weights[j][i] = buf[i*32+j]` into new struct `[32][512]`
2. `writeToWriter` dumps raw struct bytes (output-major) to disk
3. Load: v2 path memcpys directly → output-major in struct
4. Inference: `linearForward_i8` reads output-major → should be correct IF step 1 is correct

### The suspect: export buf indexing

`buf[i * fc1_out + j]` interprets the GPU buffer as `[fc1_in, fc1_out]`. If the GPU's `ml.Linear` actually stores weights as `[out_features, in_features]` (standard convention), then this indexing is WRONG — it reads transposed values. For v2, the wrong read + v1 load transpose accidentally cancel out (two transposes = identity). For v5, the wrong read is stored directly without correction.

**To verify:** check kore's `ml.Linear` weight tensor shape. If it stores `[out, in]`, the export buf indexing is the root cause. The fix would be `buf[j * fc1_in + i]`.

## What to Investigate

1. **GPU weight layout**: What shape does `kore.ml.Linear.init(allocator, ctx, 512, 32, seed)` use for its weight tensor? Check `kore/src/ml/src/Linear.zig` or run a test that downloads the weight buffer and checks dimensions.

2. **Export buf indexing correctness**: Is `buf[i * fc1_out + j]` correct for the GPU's layout? If the GPU uses `[out, in]`, it should be `buf[j * fc1_in + i]`.

3. **v2 vs v5 byte-level comparison**: Load both nets, extract the FC1 weight matrices, check if one is the transpose of the other. If v5's FC1 weights are the transpose of v2's, the export indexing bug is confirmed.

4. **Quick validation**: Temporarily revert inference to the old `matmul`-based code and test v5. If v5 suddenly works, the issue is in linearForward_i8 or the weight layout interaction, not in training.

## Available Test Data

| File | Records | Source |
|------|---------|--------|
| `data/selfplay_data_merged.bin` | 29,547,019 | v2's exact training data |
| `data/selfplay_final.bin` | 40,273,506 | v3-v5 training data |
| `data/net_v2.nnue` | format v1 | The "good" net |
| `data/net_v5.nnue` | format v2 | Latest "bad" net |

Data distributions are virtually identical (checked); the regression is not from data quality.

## Diagnostic Commands

```bash
# Run diagnostic test on a net (edit path in nnue.zig:805 first)
zig build test -Dtest_filter="diagnose" --summary all

# Quick elo test (same binary, different UCI options)
fastchess \
  -engine cmd=./zig-out/bin/uci name=test proto=uci "option.EvalFile=data/net_v5.nnue" "option.Threads=4" "option.OwnBook=false" \
  -engine cmd=./zig-out/bin/uci name=hce proto=uci "option.Threads=4" "option.OwnBook=false" \
  -each tc=1+0.01 restart=on timemargin=300 \
  -rounds 20 -repeat -recover -concurrency 4 \
  -draw movenumber=40 movecount=10 score=5 \
  -resign movecount=5 score=1000 \
  -output format=cutechess

# Train a net
./zig-out/bin/train_nnue --data data/selfplay_data_merged.bin --export data/test.nnue --epochs 50 --lr 0.005
```

## Elo Results for Reference

| Net | vs v2 | vs HCE |
|-----|-------|--------|
| v2 | — | wins ~7% |
| v3 | 0-35-5 | 0-40-0 |
| v4 | 5-15-20 (-89 Elo) | 1-35-4 (-436 Elo) |
| v5 | 1-36-3 (-470 Elo) | 0-40-0 |
