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

## HalfKP Feature Indexing

Needed for Phase 5 incremental update logic.

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

- **Phase 0** ✅ — Self-play data generator (`src/selfplay.zig`, `src/selfplay/serde.zig`)
- **Phase 1** ✅ — Data structures, feature extraction, file I/O (`src/engine/nnue.zig`)
- **Phase 2** ✅ — Non-incremental quantized inference (`nnue.evaluate(state, net) -> i32`)
- **Phase 3** ✅ — GPU training (`src/train_nnue.zig`, `src/trainer/{model,dataloader,export}.zig`)
- **Phase 4** ✅ — Search integration & UCI/CLI wiring
  - `search.zig`: `SharedSearchState.network: ?*const nnue.Network`; `evaluate()` dispatches to `nnue.evaluate(state, net) * 2` when loaded (×2 to roughly match HCE scale), falls back to HCE otherwise. `searchParallel`/`searchWithHistory` accept a `network` parameter.
  - `uci.zig`: `EvalFile` UCI option loads via `nnue.Network.load(io, allocator, path)`; passes network into search.
  - `tui.zig`: `--nnue <path>` CLI arg loads network at startup, threads it through to search.
  - `bench.zig`: `--nnue <path>` CLI arg; benchmark reports eval label (HCE vs NNUE) and exercises `nnue.evaluate` in the micro-eval loop.
  - Note: `ThreadContext` does not yet carry an `nnue.Accumulator` — evaluation is fully non-incremental; that work is deferred to Phase 5.

---

## Remaining Phases

### Phase 5: Incremental Accumulator Updates

**Accumulator stack:**

- `[max_ply]Accumulator` array in `ThreadContext`
- On `makeMove`: copy parent accumulator, then apply delta based on move type:
  - **Quiet move:** remove `ft_weights[old_index]`, add `ft_weights[new_index]` — 2 row ops per perspective
  - **Capture:** also remove captured piece's row — 3 ops per perspective
  - **Promotion:** remove pawn row, add promoted piece row
  - **Castling:** also move rook feature rows
- On `unmakeMove`: decrement ply index (parent accumulator intact)

**Feature index delta computation:**

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
2. **Train** NNUE for ~100 epochs using GPU trainer
3. **Elo test** against HCE baseline using `elo_test.py`
4. **Iterate:** generate more data with NNUE-powered engine, retrain on combined data
5. Scale up: deeper self-play (depth 10-12), more positions (100M+), tune hyperparameters

---

## File Summary

| File                 | Phase | Purpose                          |
| -------------------- | ----- | -------------------------------- |
| `src/engine/search.zig` | 4-5   | NNUE eval calls, accumulator stack |
| `src/uci.zig`        | 4     | EvalFile UCI option              |
| `src/tui.zig`        | 4     | `--nnue` CLI arg                 |
| `src/bench.zig`      | 4     | NNUE benchmark support           |

## Implementation Order

```
Phase 0-4 ✅ →  Phase 5  →  Phase 6
(done)          (incremental) (iterate)
```
