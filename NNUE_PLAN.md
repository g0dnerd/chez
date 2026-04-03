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

| Layer | Weights | Biases | Activations | Float→Quant |
|-------|---------|--------|-------------|-------------|
| Accumulator | i16 | i16 | i16 → clamp [0, 127] → u8 | ×127 |
| FC1 (512→32) | i8 | i32 | i8×u8→i32, ÷64, clamp [0, 127] → u8 | weights ×64, biases ×(127×64) |
| FC2 (32→32) | i8 | i32 | i8×u8→i32, ÷64, clamp [0, 127] → u8 | weights ×64, biases ×(127×64) |
| Output (32→1) | i8 | i32 | i8×u8→i32, ÷(127×64) → centipawns | weights ×64, biases ×(127×64) |

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

## Phases

### Phase 0: Self-Play Data Generator ✅

**Files:** `src/selfplay.zig`, `src/selfplay/serde.zig`

**Binary record format (35 bytes, fixed-size):**

```
[32 bytes] Position (Huffman-coded, zero-padded)
[2 bytes]  Score (i16, little-endian, side-to-move perspective)
[1 byte]   WDL outcome (u8: 0=white wins, 1=black wins, 2=draw)
```

**CLI:** `selfplay --depth 8 --num_games 1000 --num_threads 4 > data.bin`

**Decoding:** `serde.decodePosition(buf: []u8) -> !engine.State` reconstructs a full `State` from a 32-byte buffer.

---

### Phase 1: NNUE Data Structures & Feature Extraction ✅

**File:** `src/engine/nnue.zig` (re-exported via `engine.nnue`)

**Available API for subsequent phases:**

```zig
// Feature extraction
nnue.activeFeatures(state: *const State, perspective: Color) -> FeatureList
// FeatureList { features: [30]u32, len: u8 }

// Network (heap-allocated, ~20MB)
Network.loadFromBytes(allocator, data: []const u8) -> !*Network
Network.load(io: std.Io, allocator, path: []const u8) -> !*Network  // mmap on Linux
Network.writeToWriter(self, w: *std.Io.Writer) -> !void
Network.deinit(self, allocator) -> void

// Accumulator
Accumulator.empty  // zero-initialized, computed=false

// Constants
nnue.num_features       // 40960
nnue.ft_out             // 256
nnue.fc1_in / fc1_out   // 512, 32
nnue.fc2_in / fc2_out   // 32, 32
nnue.expected_file_size // 20,989,748
```

---

### Phase 2: NNUE Inference (Non-Incremental)

**In `src/engine/nnue.zig`:**

**Forward pass (quantized i16/i8), using `kore.ml.cpu` for SIMD primitives:**

1. Compute active features for both perspectives via `activeFeatures()`
2. Accumulator = `ft_biases` + sum of `ft_weights[idx]` for each active feature (i16, via `kore.ml.cpu.ops`)
3. ClippedReLU: `clamp(x, 0, 127)` → cast to u8
4. Concatenate accumulators: side-to-move perspective first → `[512]u8`
5. FC1: i8 weights × u8 activations → i32 (via `kore.ml.cpu.quantized`), add i32 bias, ÷64, ClippedReLU → u8
6. FC2: same
7. Output: i8 weights × u8 activations → i32, add bias, ÷(127×64) → centipawns

**New function:**

```zig
pub fn evaluate(state: *const State, net: *const Network) i32
```

Keep HCE (`evaluation.evaluate()`) as fallback when no network is loaded.

---

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
stm_acc = ft.forward(stm_features, tape)     // SparseLinear(40960, 256)
opp_acc = ft.forward(opp_features, tape)     // same layer, weight sharing via tape
stm_relu = crelu.forward(stm_acc, tape)      // ClippedReLU(max=1.0)
opp_relu = crelu.forward(opp_acc, tape)
combined = concat(stm_relu, opp_relu, tape)  // [batch, 512], STM first

// Dense layers (Sequential)
output = dense.forward(combined, tape)        // Linear(512,32) → CReLU → Linear(32,32) → CReLU → Linear(32,1)
```

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

| Parameter | Value |
|-----------|-------|
| Batch size | 16384 |
| Optimizer | Adam (lr=0.001, β1=0.9, β2=0.999, ε=1e-8) |
| LR schedule | Linear warmup 1 epoch, cosine decay to 0.0001 |
| Gradient clipping | Global norm, max_norm=1.0 |
| Epochs | ~100 |
| Dataset | 50M positions |
| Validation | Last 2% of .bin file |
| Checkpoints | Every 5 epochs (kore.ml checkpoint format) |
| Val loss logging | Every epoch |

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

| File | Status | Phase | Purpose |
|------|--------|-------|---------|
| `src/selfplay.zig` | **Done** | 0 | Self-play data generation binary |
| `src/selfplay/serde.zig` | **Done** | 0 | Position serialization (Huffman encode/decode) |
| `src/engine/nnue.zig` | **Done** | 1 | Data structures, feature extraction, file I/O |
| `src/engine/engine.zig` | **Done** | 1 | Re-exports nnue module |
| `src/engine/nnue.zig` | **Modify** | 2 | Add `evaluate()` forward pass (uses kore.ml.cpu) |
| `src/train_nnue.zig` | New | 3 | Training binary entry point |
| `src/trainer/model.zig` | New | 3 | NNUE topology using kore.ml layers |
| `src/trainer/dataloader.zig` | New | 3 | Binary data loading + feature extraction |
| `src/trainer/export.zig` | New | 3 | Float32 → quantized .nnue export |
| `build.zig` | Modify | 3+ | New executables and module imports |
| `src/engine/search.zig` | Modify | 4-5 | NNUE eval calls, accumulator stack |
| `src/uci.zig` | Modify | 4 | EvalFile UCI option, network loading |
| `src/tui.zig` | Modify | 4 | `--nnue` CLI arg |
| `src/bench.zig` | Modify | 4 | NNUE benchmark support |

## Implementation Order

```
Phase 0 ✅ →  Phase 1 ✅ →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5
(data)        (structures)   (inference)  (training)   (integrate)  (incremental)
                                              ↓
                                         Phase 6 (iterate)
```

Phase 2 (inference) can start immediately. Phase 3 (training) depends on kore.ml being ready through its Phase 5. Phase 5 (incremental) delivers the biggest speed gain for search.
