# NNUE Implementation Plan

## Architecture Summary

```
HalfKP(256→32→32→1)
├── Input: 40960 sparse features per perspective (64 king sq × 10 piece types × 64 piece sq)
├── Accumulator: 40960→256 per perspective (i16 quantized at inference)
├── Concat: white_acc ++ black_acc = 512 (side-to-move perspective first)
├── FC1: 512→32 + ClippedReLU
├── FC2: 32→32 + ClippedReLU
├── Output: 32→1 (centipawns)
└── Total params: ~10.5M (~20MB quantized .nnue file)
```

## Quantization Scheme

Scale factors: FT = 127 (fills i16 range relative to ClippedReLU [0,127]), hidden = 64.

| Layer         | Weights | Biases | Activations                         | Float→Quant                   |
| ------------- | ------- | ------ | ----------------------------------- | ----------------------------- |
| Accumulator   | i16     | i16    | i16 → clamp [0, 127] → i8           | ×127                          |
| FC1 (512→32)  | i8      | i32    | i8×i8→i32, ÷64, clamp [0, 127] → i8 | weights ×64, biases ×(127×64) |
| FC2 (32→32)   | i8      | i32    | i8×i8→i32, ÷64, clamp [0, 127] → i8 | weights ×64, biases ×(127×64) |
| Output (32→1) | i8      | i32    | i8×i8→i32, ÷(127×64) → centipawns   | weights ×64, biases ×(127×64) |

## .nnue File Format

```
Offset      Size          Field
0           4             Magic: "CHEZ"
4           4             Version: u32 (1)
8           4             Architecture hash: u32 (0x484B5031 "HKP1")
12          4             Reserved: u32 (0)
--- Feature Transformer (accumulator) ---
16          512           FT biases: [256]i16
528         20,971,520    FT weights: [40960][256]i16
--- FC1 (512→32) ---
20,972,048  16,384        FC1 weights: [512][32]i8
20,988,432  128           FC1 biases: [32]i32
--- FC2 (32→32) ---
20,988,560  1,024         FC2 weights: [32][32]i8
20,989,584  128           FC2 biases: [32]i32
--- Output (32→1) ---
20,989,712  32            Output weights: [32]i8
20,989,744  4             Output bias: [1]i32
```

Total file size: 20,989,748 bytes (little-endian throughout).

## HalfKP Feature Indexing

Implemented in `nnue.activeFeatures()`. Needed here for Phase 5 incremental update logic.

```
For perspective color c:
  own_king_sq = king square (flipped if c == black)
  For each non-king piece:
    relative_color = piece_color XOR c  (0 = friendly, 1 = enemy)
    piece_type = 0..4 (pawn..queen, no king)
    piece_sq = square (flipped if c == black)
    index = own_king_sq * 640 + (relative_color * 5 + piece_type) * 64 + piece_sq

Square flipping for black perspective: sq XOR 56 (rank mirror)
Active features per position: ≤30 (number of non-king pieces)
```

---

## Completed Phases

- **Phase 0** ✅ — Self-play data generator (`src/selfplay.zig`, `src/selfplay/serde.zig`). 35-byte records, CLI: `selfplay --depth 8 --num_games 1000 --num_threads 4 > data.bin`
- **Phase 1** ✅ — Data structures, feature extraction, file I/O (`src/engine/nnue.zig`, re-exported via `engine.nnue`)
- **Phase 2** ✅ — Non-incremental quantized inference (`nnue.evaluate(state, net) -> i32`). Uses `kore.ml.cpu` SIMD ops.

---

## Remaining Phases

### Phase 3: GPU Training (using kore.ml)

**Replaces the previously planned CPU trainer. Uses kore.ml for all generic ML operations (Tensor, GPU ops, autograd, optimizer). This phase covers the NNUE-specific wiring.**

**New files:**

- `src/train_nnue.zig` — training binary entry point
- `src/trainer/dataloader.zig` — binary data loading, position decoding, feature extraction
- `src/trainer/model.zig` — NNUE model topology using kore.ml layers
- `src/trainer/export.zig` — float32 → quantized .nnue export

**Model topology (in `model.zig`):**

The NNUE forward pass is not a simple sequential chain — it has a shared-weight feature transformer applied to two perspectives, then a perspective-aware concat. The model composes kore.ml primitives manually:

```
// Shared feature transformer (SparseLinear, same weights for both perspectives)
ft.setIndices(stm_indices, stm_num_active, batch_size)
stm_acc = ft.forward(dummy, graph)            // SparseLinear(40960, 256)
ft.setIndices(opp_indices, opp_num_active, batch_size)
opp_acc = ft.forward(dummy, graph)            // same weights, different indices
stm_relu = crelu.forward(stm_acc, graph)      // ClippedReLU(max=1.0)
opp_relu = crelu.forward(opp_acc, graph)
combined = graph.concat(stm_relu, opp_relu)   // [batch, 512], STM first

// Dense layers (Sequential)
output = dense.forward(combined, graph)        // Linear(512,32) → CReLU → Linear(32,32) → CReLU → Linear(32,1)
```

**Note on buffer lifetimes:** Both sets of index buffers (STM and OTM) are captured by value in the tape. The data loader must keep all per-batch GPU index buffers alive until `graph.backward()` completes and `graph.reset()` is called.

Side-to-move ordering is handled by the data loader: it always provides STM features first, OTM features second. No conditional logic in the model.

**Data loader (`dataloader.zig`):**

Single-threaded synchronous pipeline:

1. Memory-map the `.bin` file (fixed 35-byte records, total_records = file_size / 35)
2. Partition: first 98% = training, last 2% = validation
3. Shuffle training record indices at epoch start
4. Per batch of 16384 records:
   - `serde.decodePosition(buf[0..32])` → `State`
   - `nnue.activeFeatures(&state, stm)` → STM feature indices
   - `nnue.activeFeatures(&state, opp)` → OTM feature indices
   - Read i16 score at byte 32, u8 WDL at byte 34
   - Compute blended target: `label = λ × sigmoid(score × K/400) + (1-λ) × wdl_value`
   - WDL mapping: 0 (white wins) → 1.0, 1 (black wins) → 0.0, 2 (draw) → 0.5; flip if STM is black
   - Score is already STM perspective (no flip needed)
5. Pack feature index lists and targets into kore.ml tensors, upload to GPU

**Loss function:**

Blended-target MSE with sigmoid pre-transform, using `kore.ml.MseLoss`:

```
loss = (1/N) × Σ (sigmoid(output × K/400) - label)²
```

- K ≈ 111 (Stockfish-style sigmoid scaling)
- λ = 1.0 initially (pure search score targets), tune blending later

**Training hyperparameters:**

| Parameter         | Value                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------- |
| Batch size        | 16384                                                                                    |
| Optimizer         | Adam (lr=0.001, β1=0.9, β2=0.999, ε=1e-8)                                                |
| LR schedule       | Linear warmup 1 epoch, cosine decay to 0.0001 (manual: mutate `adam.config.lr` per step) |
| Gradient clipping | Global norm, max_norm=1.0 (via `graph.clipGradNorm`, GPU-accelerated)                    |
| Epochs            | ~100                                                                                     |
| Dataset           | 50M positions                                                                            |
| Validation        | Last 2% of .bin file                                                                     |
| Checkpoints       | Every 5 epochs (KTML format via `kore.ml.serialize`)                                     |
| Val loss logging  | Every epoch                                                                              |

**Weight initialization:**

- Feature transformer: Kaiming uniform scaled by `1/√30` (avg active features)
- Dense layers: standard Kaiming uniform
- All biases: zero

**Quantization export (`export.zig`):**

After training, convert float32 network to .nnue:

1. FT weights: `round(w × 127)` → i16
2. FT biases: `round(b × 127)` → i16
3. Hidden weights: `round(w × 64)` → i8
4. Hidden biases: `round(b × 127 × 64)` → i32
5. Output weights: `round(w × 64)` → i8
6. Output bias: `round(b × 127 × 64)` → i32
7. Write to .nnue format via `Network.writeToWriter()`
8. Validate: load quantized .nnue, run inference on validation set, compare against float32 — error should be < 2 centipawns average

**Checkpoint resume:**

On save, set `metadata.adam_step = adam.step_count`. On load, restore `adam.step_count = metadata.adam_step` to preserve bias correction state.

**Training binary CLI (`train_nnue.zig`):**

```
train_nnue --data data.bin --epochs 100 --batch_size 16384 --lr 0.001 --lambda 1.0
           [--checkpoint resume.ktml] [--export output.nnue]
```

**build.zig changes:** Add `train_nnue` executable, link `chez` module + `kore.ml` module + system GL/EGL libs.

---

### Phase 4: Search Integration & UCI

**Modifications to `search.zig`:**

- `SharedSearchState` gains `network: ?*const nnue.Network`
- `ThreadContext` gains an `nnue.Accumulator`
- Replace `evaluation.evaluate(state)` with `nnue.evaluate(state, network)` when net loaded
- Fall back to HCE when `network == null`

**Modifications to `uci.zig`:**

- New UCI option: `EvalFile` (path to `.nnue` file)
- Load network via `nnue.Network.load(io, allocator, path)` at startup or when option changes

**Modifications to `tui.zig`:**

- CLI arg `--nnue <path>`
- Fall back to HCE if no net specified

**Modifications to `bench.zig`:**

- Support `--nnue <path>` for benchmarking with NNUE eval
- Compare NPS: HCE vs NNUE

---

### Phase 5: Incremental Accumulator Updates

**Accumulator stack:**

- `[max_ply]Accumulator` array in `ThreadContext`
- On `makeMove`: copy parent accumulator, then apply delta based on move type:
  - **Quiet move:** remove `ft_weights[old_index]`, add `ft_weights[new_index]` — 2 row ops per perspective
  - **Capture:** also remove captured piece's row — 3 ops per perspective
  - **Promotion:** remove pawn row, add promoted piece row
  - **Castling:** also move rook feature rows
- On `unmakeMove`: decrement ply index (parent accumulator intact)

**Feature index delta computation** (derived from HalfKP formula above):

- Moving piece changes `piece_sq` component: old index and new index differ only in the last 6 bits
- Captured piece: compute its full index and subtract its row
- King moves change `own_king_sq` which affects ALL feature indices for that perspective → full recomputation required
- Opposite perspective: king didn't move, so incremental update still works

**Lazy evaluation:**

- Mark accumulators as "dirty" (`computed = false`), only compute when `evaluate()` is called
- Many nodes pruned without eval (null move, LMR fail-low, etc.)

**Validation:** Debug builds run both full recomputation and incremental path, assert identical results.

---

### Phase 6: Training Loop & Iteration

1. **Generate** 50M positions via self-play at depth 8
2. **Train** NNUE for ~100 epochs using GPU trainer (Phase 3)
3. **Elo test** against HCE baseline using `elo_test.py`
4. **Iterate:** generate more data with NNUE-powered engine, retrain on combined data
5. Scale up: deeper self-play (depth 10-12), more positions (100M+), tune hyperparameters

---

## File Summary

| File                         | Status   | Phase | Purpose                                |
| ---------------------------- | -------- | ----- | -------------------------------------- |
| `src/train_nnue.zig`         | New      | 3     | Training binary entry point            |
| `src/trainer/model.zig`      | New      | 3     | NNUE topology using kore.ml layers     |
| `src/trainer/dataloader.zig` | New      | 3     | Binary data loading + feature extract  |
| `src/trainer/export.zig`     | New      | 3     | Float32 → quantized .nnue export       |
| `build.zig`                  | Modify   | 3+    | New executables and module imports     |
| `src/engine/search.zig`      | Modify   | 4-5   | NNUE eval calls, accumulator stack     |
| `src/uci.zig`                | Modify   | 4     | EvalFile UCI option, network loading   |
| `src/tui.zig`                | Modify   | 4     | `--nnue` CLI arg                       |
| `src/bench.zig`              | Modify   | 4     | NNUE benchmark support                 |

## Implementation Order

```
Phase 0-2 ✅ →  Phase 3  →  Phase 4  →  Phase 5
(done)          (training)   (integrate)  (incremental)
                    ↓
               Phase 6 (iterate)
```

Phase 3 (training) can start immediately — kore.ml has all required ops (SparseLinear, concat with autograd, MseLoss with sigmoid, Adam, serialization). Phase 5 (incremental) delivers the biggest speed gain for search.
