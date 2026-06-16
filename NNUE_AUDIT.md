# NNUE Implementation Audit

## Scope

Full audit of quantized inference (`nnue.zig`), GPU training (`trainer/model.zig`), quantized export (`trainer/export.zig`), data loading (`trainer/dataloader.zig`), self-play data generation (`selfplay.zig`, `selfplay/serde.zig`), training entry point (`train_nnue.zig`), and search integration (`search.zig`). kore SIMD primitives treated as trusted.

## Summary

The implementation is architecturally correct. Weight layout, quantization math, feature indexing, perspective handling, and the incremental accumulator stack all check out. No smoking-gun correctness bug was found that would explain wrong evaluations from a trained net.

The most likely cause of poor eval quality is the combination of a very flat loss landscape (sigmoid at 1/400 scale produces tiny gradients) and training hyperparameters that may be insufficient to overcome it. See the debugging guide at the end.

---

## Verified Correct

These areas were audited and confirmed correct:

- **Feature index formula**: `king_sq * 640 + (rel_color * 5 + piece_type) * 64 + piece_sq`, max index = 40959 < 40960. Same function (`activeFeatures`) used in both training dataloader and inference.
- **Black perspective flipping**: All squares XOR 56 (rank mirror) applied consistently to both king square and piece square.
- **Relative color**: `piece_color XOR perspective` correctly gives 0=friendly, 1=enemy.
- **Weight layout**: kore `Linear` and `SparseLinear` both store weights as `[in_features, out_features]` row-major. Export code downloads in this layout. Inference `quantized.matmul(M, K, N, A, B)` computes `A[M,K] @ B[K,N]`, which matches.
- **Quantization scales**: FT×127→i16, hidden weights×64→i8, hidden biases×(127×64)→i32, output÷(127×64)→centipawns. Math traces through correctly: float value × combined scale = quantized value at each layer.
- **Perspective concat order**: STM first in both `graph.concat(stm_relu, opp_relu)` (training) and `concat[0..256] = stm, concat[256..512] = opp` (inference). `graph.concat` confirmed to produce `[batch, 512]` (feature concat, not batch concat).
- **SparseLinear backward pass**: Graph tape captures index buffer references at forward time. When `setIndices` is called a second time for OTM, the tape still holds the STM buffer reference. Backward uses captured indices correctly.
- **Loss function**: `MseLoss` with `use_sigmoid=true` computes `MSE(sigmoid(output * scale), target)`. Gradient correctly propagates through sigmoid via chain rule.
- **Incremental accumulator updates**: Delta computation correctly handles all move types (quiet, capture, en passant, promotion, castling, promotion+capture). King moves trigger full recompute for the moved side's perspective. Lazy evaluation recursively walks to nearest computed ancestor. Debug assertion validates incremental vs non-incremental in debug builds.
- **Castling integration**: `castle_data[color][side]` where side=0 is kingside, side=1 is queenside. `UndoInfo.castling_side` set correctly in `State.makeMove`. `applyDeltaPerspective` correctly moves rook features.
- **En passant**: `UndoInfo.captured_square` set to the captured pawn's actual square (not `move.end`). `applyDeltaPerspective` removes the captured piece from the correct square.
- **Root accumulator**: Refreshed once in `workerThread` before iterative deepening begins.
- **Ply overflow**: `recordMove`/`recordNullMove` silently return when `child_ply > max_stack_ply`. `evaluateLazy` falls back to non-incremental `evaluate`. Graceful degradation.
- **Serde round-trip**: Position encoding/decoding tested with multiple FENs.

---

## Findings

### H1: NNUE-to-HCE scale factor is a constant `* 2` — all pruning margins are miscalibrated

**File**: `src/engine/search.zig:411`
**Severity**: HIGH

```zig
if (self.network) |net| return nnue.evaluateLazy(state, net, acc_stack, ply) * 2;
```

NNUE `evaluate()` returns centipawns (pawn ≈ 100). The `* 2` makes pawn ≈ 200 to roughly match the HCE internal scale. But all pruning/reduction thresholds are hard-coded for HCE:

| Mechanism | Constant | HCE-calibrated | NNUE-adjusted |
|-----------|----------|----------------|---------------|
| Reverse futility | `80 * depth` | Yes | Probably too tight |
| Futility margins | `{0, 300, 600}` | Yes | Might over-prune |
| Delta pruning | `200` | Yes | Might over-prune |
| LMP thresholds | `{5, 6, 9, 14}` | Yes | Unaffected (move count) |

**Impact**: Incorrect pruning can make the search either too aggressive (misses tactics) or too conservative (wastes time). The crude `* 2` factor won't match HCE scale across the full evaluation range — NNUE and HCE have different piece-value curves, pawn structure weights, etc.

**Fix**: Either (a) tune all margins specifically for NNUE via SPSA/tuning, or (b) derive a per-position scaling factor. For a quick fix, the `* 2` constant could be tuned via the existing SPSA infrastructure as a single parameter.

---

### H2: Sigmoid loss landscape at scale 1/400 produces very small gradients

**Severity**: HIGH (training quality, not code bug)

The loss is `MSE(sigmoid(output / 400), target)`. The gradient w.r.t. the model output contains a `sigmoid'(output/400) / 400` factor. For typical centipawn values (output ≈ ±200), this gives:

```
sigmoid'(200/400) / 400 = sigmoid'(0.5) / 400 = 0.235 / 400 ≈ 0.00059
```

This is a very small multiplier on the gradient. Adam compensates somewhat via per-parameter adaptive learning rates, but the initial learning signal is weak.

**Impact**: The model may need many more epochs or a higher base learning rate than the current defaults (100 epochs, lr=0.001) to converge. A poorly converged model would produce near-zero evaluations for all positions because the output weights remain near their initialization values.

**Recommended actions**:
- Try lr=0.01 or even lr=0.1 for the first 10 epochs to escape the flat region
- Increase total epochs to 300+
- Monitor weight norms per layer during training — if output layer weights aren't growing, the model isn't learning
- Consider training with `lambda=0.5` to add WDL signal (see M1)

---

### M1: Default `lambda=1.0` discards game-outcome signal entirely

**File**: `src/train_nnue.zig:28`
**Severity**: MEDIUM

```zig
const default_lambda: f32 = 1.0;
```

With `lambda=1.0`, the target is `1.0 * sigmoid(score/400) + 0.0 * wdl_value` — purely the engine's own evaluation, no game outcome. The model learns to mimic HCE.

**Impact**: The model can't learn anything the HCE doesn't already know. Game outcomes provide a ground-truth signal about position quality that self-play scores lack.

**Fix**: Use `lambda=0.5` or `lambda=0.75` to blend score and outcome signals. Stockfish's trainer uses ~0.8 for later generations.

---

### M2: Non-Linux `Network.load` path reuses I/O buffer as data buffer

**File**: `src/engine/nnue.zig:225-229`
**Severity**: MEDIUM (only affects non-Linux platforms)

```zig
var buf = try allocator.alloc(u8, len);
defer allocator.free(buf);
var reader = file.reader(io, &buf);
_ = try reader.interface.take(len);
return loadFromBytes(allocator, buf[0..len]);
```

The same `buf` is used as both the reader's internal staging buffer and the destination for file data. The return value of `take(len)` (a slice into the reader's buffer) is discarded, and `buf[0..len]` is read directly. This works today because the reader fills from position 0 for a fresh read, but it's fragile:
- Depends on reader implementation details
- Will silently produce corrupt data if the reader's buffering strategy changes

**Fix**: Allocate two separate buffers, or use the returned slice from `take()`:
```zig
var read_buf: [4096]u8 = undefined;
var reader = file.reader(io, &read_buf);
const data = try reader.interface.take(len);
return loadFromBytes(allocator, data);
```

Or simply use mmap on all POSIX platforms, not just Linux.

---

### M3: No float-vs-quantized validation after export

**Severity**: MEDIUM

There is no automated check that the quantized `.nnue` file produces similar evaluations to the float GPU model. A silent export bug (e.g., wrong scale, transposed weights) would only be caught by noticing wrong evals at runtime.

**Fix**: After export, run 10-20 positions through both the float model (on GPU) and the quantized inference path, and assert the centipawn values agree within a tolerance (±5 cp). Add this as a post-export validation step in `train_nnue.zig`.

---

### M4: `evaluateLazy` debug assertion runs full feature extraction every call in debug builds

**File**: `src/engine/nnue.zig:344-350`
**Severity**: MEDIUM (debug-build performance only)

```zig
if (@import("builtin").mode == .Debug) {
    var fresh: Accumulator = undefined;
    refreshAccumulator(state, net, &fresh);
    // ... compare all values ...
}
```

This runs full non-incremental feature extraction on every `evaluateLazy` call in debug builds, which makes debug-mode search ~100× slower.

**Impact**: Debug builds become impractically slow for any meaningful testing. This is fine as a correctness safeguard, but consider making it configurable or sampling (e.g., check every 1000th call).

---

### L1: `StackDelta.promotion_piece` defaults to 0 (pawn) for non-promotions

**File**: `src/engine/nnue.zig:502`
**Severity**: LOW

```zig
.promotion_piece = if (undo.promotion_piece) |pp| pp else 0,
```

When `was_promotion` is false, `promotion_piece` is set to 0 (pawn type). This is harmless because `applyDeltaPerspective` checks `was_promotion` before using the field. But it's misleading — a sentinel value or leaving it as `undefined` would be clearer.

---

### L2: Self-play uses HCE, preventing iterative NNUE improvement (Phase 6 gap)

**File**: `src/selfplay.zig:115-122`
**Severity**: LOW (known Phase 6 item, but worth noting for completeness)

```zig
const search_res = (try engine.search.searchWithHistory(
    &self.state, self.depth, 1, &self.history, self.ttable,
    null,  // no network — always HCE
)) orelse return error.SearchFailed;
```

The self-play generator always uses HCE for search. Phase 6 requires adding `--nnue` support to the self-play binary so that training iterations can use the NNUE-powered engine to generate better data.

---

### L3: No weight decay / L2 regularization in training

**File**: `src/train_nnue.zig:88`
**Severity**: LOW

The Adam optimizer is used without weight decay. For NNUE training with large sparse feature transformers, weight decay helps prevent overfitting and keeps quantized weights in a reasonable range.

**Fix**: Use AdamW with weight_decay=1e-4 or similar (if kore supports it), or add explicit L2 penalty to the loss.

---

### P1: Quiescence search does non-incremental NNUE evaluation

**Severity**: PERFORMANCE

The quiescence search correctly uses incremental accumulator updates via `recordMove` + `evaluateLazy`. This is already implemented. However, quiescence generates the most `evalPosition` calls, so any overhead in the incremental path is amplified here.

**Note**: This is already handled correctly. No action needed.

---

### P2: `AccumulatorStack` is ~19KB per thread

**Severity**: PERFORMANCE (minor)

```
(max_stack_ply + 1) * (Accumulator + StackDelta)
= 97 * (2 * 256 * 2 + 2 + padding + delta fields)
≈ 97 * 196 ≈ 19KB
```

With 16 threads, this is ~304KB of stack space. Acceptable for desktop, but worth noting for memory-constrained environments.

---

## Debugging Guide: Why Trained Net Evaluations Look Wrong

Since no correctness bug was found, here's a systematic approach to diagnose the issue:

### Step 1: Check if the model actually learned

Run the `diagnose trained net` test and look at:
- **FT weight statistics**: If `ft_nonzero` is near 0 or `ft_max` is near 0, the FT weights haven't moved from initialization. The model didn't learn.
- **Accumulator range**: If all values are negative (killed by CReLU) or all at 0, the activations are dead.
- **FC1/FC2 after CReLU**: If `nonzero` counts are 0, the hidden layers are dead (all neurons killed by CReLU).
- **Output weights**: If all near 0, the output layer didn't learn.

### Step 2: Compare float vs quantized

Add a validation step after export:
```zig
// After exportNnue(), load the .nnue file back and compare
const qnet = try Network.load(io, allocator, export_path);
const state = State.defaultPosition();
const q_score = nnue.evaluate(&state, qnet);
// Compare with float model output (run forward on GPU)
```

If float model gives reasonable values (e.g., ±50 for equal position) but quantized gives 0, the issue is in export quantization. If both give near-0, the model hasn't converged.

### Step 3: Check training convergence

Plot the training loss and validation loss across epochs. If the loss plateaus early at a high value, the learning rate is too low or the sigmoid scaling is flattening gradients too much.

**Quick experiment**: Try training with `--lr 0.01 --epochs 300 --lambda 0.75`. The higher LR should overcome the flat sigmoid landscape, and lambda < 1 adds game-outcome signal.

### Step 4: Check output scale

The float model should output values roughly in the centipawn range (±100 for a pawn, ±300 for a minor piece). If the model outputs values in the range ±1, the sigmoid(output/400) is always ≈ 0.5 and the model hasn't differentiated positions.

Add a print statement after training to check:
```zig
// Evaluate a few positions with the float model
const output = try model.forward(batch, &graph);
// Download and print the raw output values
```

### Step 5: Verify self-play data quality

Check the distribution of scores in the training data:
```bash
# Quick histogram of scores in training data
python3 -c "
import struct, sys, collections
data = open(sys.argv[1], 'rb').read()
scores = []
for i in range(0, len(data), 35):
    score = struct.unpack_from('<h', data, i + 32)[0]
    scores.append(score)
print(f'N={len(scores)}, mean={sum(scores)/len(scores):.1f}, '
      f'min={min(scores)}, max={max(scores)}, '
      f'near_zero={sum(1 for s in scores if -10 < s < 10)}/{len(scores)}')
" data/selfplay.bin
```

If most scores are near 0 or the distribution is heavily skewed, the training data might not have enough signal.
