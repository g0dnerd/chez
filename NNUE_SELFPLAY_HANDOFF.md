# NNUE Self-play Speed/Quality Handoff

## Goal for this session
Find a self-play data-generation setup that is **both fast and produces a strong net**.
We fixed a data-*quality* regression but in doing so lost too much *speed*: on the
user's DGX Spark, throughput fell from **~90 pos/s → ~20 pos/s**, which is unacceptable.
Your job: recover speed (~90 pos/s class) **without** reintroducing the quality regression,
and **validate by training a net and Elo-testing it** — not by distribution/val-loss alone.

## Hard-won rule
**Val loss and label distribution do NOT predict playing strength.** Every conclusion this
session that relied on them was wrong. The ONLY reliable signal is: train a net, Elo-test it
vs the prior best (`net_v7_lambda075.nnue`) with `testing/elo_test.py --mode quick`. Budget
~45 min/train + ~3 min/match per experiment.

## Background: the v8 regression (solved)
- **v7** (`data/net_v7_lambda075.nnue`, the current best ≈ HCE strength): trained on
  `data/selfplay_final.bin` — HCE eval, depth 8, **no** adjudication/filter/node-cap (Jun 14).
  Score range ±9943, p99≈4723, draws 39%.
- **v8** (`data/net_v8_mixed_lambda075.nnue`): trained on `data/selfplay_nnue_mixed.bin` —
  NNUE+HCE eval, depth 9/10, generated AFTER commit `aaa5058 "speed up selfplay"` which added
  **adjudication (1000cp), score_filter (3000), node cap**. Score range capped ±3000, p99≈1000,
  draws 25%. **v8 was −205 to −478 Elo vs v7** (catastrophic).
- **Root cause (confirmed by timeline + data shape + code):** the "speed up" gutted data quality.
  `adjudication_threshold=1000cp` abandoned games once a side was +10 pawns → the net never
  learned the +10→+25 pawn range; `score_filter=3000` dropped the rest. Decisive positions were
  never recorded and **cannot be recovered by re-labeling** — they don't exist on disk.

### What was RULED OUT (don't re-investigate)
All training-side knobs were tested on the v8 data; none recovered it (still −230 to −478):
- weight decay (wd=0 revived a collapsed dense head but Elo unchanged — dense collapse was a red herring)
- sigmoid_k recalibration (D=235 made it **worse**, −436)
- λ=0 pure-WDL (worse, −478; also proves the score is net-helpful, i.e. **no score sign/perspective bug**)
- Pipeline is correct: net loads & evals sanely, parallel dataloader is deterministic
  (batch-0 loss identical across runs), split-optimizer checkpoint bug fixed (commit 438594e).

## THE OPEN PROBLEM: speed vs quality
The quality fix and speed are **orthogonal** — the old config conflated them:
- **Quality** comes from *keeping decisive positions*: high `--score_filter` (10000) and high
  `--adjudication_threshold` (2500). This is the real fix.
- **Speed** comes from *search effort per move*: `--depth` and `--nodes` (node cap).

The current defaults I set are quality-correct but **too slow** because the node cap is too
generous (`nodes=200000` ≈ near-uncapped at depth 10 → ~20 pos/s on DGX).

**Leading hypothesis (test first):** drop the node cap hard. Standard SF-style data gen uses
**fixed ~5k–25k nodes/move**, which is far faster than depth-10 and still gives good labels.
A tight node cap (e.g. 5000–10000) **with** the good `score_filter`/`adjudication` settings should
restore ~90 pos/s AND keep quality — because speed (nodes) and the decisive-position retention
(filter/adjudication) are independent. This is the most promising path; validate it by training.

### Suggested experiment plan
1. Bisect throughput on the DGX: measure pos/s for `--nodes` ∈ {5000, 10000, 25000} at a fixed
   `--depth` (e.g. 12 ceiling) with `--adjudication_threshold 2500 --score_filter 10000 --eval <best net>`.
   Pick the fastest that hits the speed target.
2. Generate a SMALL dataset (~2–5M positions) at the chosen config; **fit the sigmoid divisor**
   to its score scale (see snippet below — the hardcoded 1/400 mis-trains on a different scale),
   then train a net and Elo-test vs v7.
3. If it matches/beats v7 → scale up via the fleet. If not, raise nodes/depth and retest.
   Also worth trying: adding a *quiet* filter (skip positions whose best move is a capture / where
   side is in check — only `in_check` is filtered today) which can improve label quality cheaply.

## Tooling / state
**Committed:**
- `438594e` parallel dataloader (~1.35× faster training) + split-optimizer checkpoint fix (kore branch `fix-split-optimizer-checkpoint`, commit 21e326d).
- `e138b56` `nnue-inspect`: `zig build nnue-inspect -- --net X [--compare Y]` — per-layer weight
  stats, quantization saturation, accumulator CReLU occupancy, eval spread. Great for debugging nets.
- `46e67cf` selfplay flags `--score_filter --adjudication_threshold --adjudication_count`;
  defaults raised to 10000 / 2500 / 4 (was 3000 / 1000 / 4).

**Uncommitted (decide whether to keep):**
- `src/train_nnue.zig` + `src/trainer/dataloader.zig`: `--weight_decay` and `--sigmoid_divisor`
  flags (the sigmoid divisor should be data-calibrated; these are useful, just not yet committed).
- `scripts/fleet_selfplay.sh` (default `--eval` → `net_v7_lambda075.nnue`) and
  `scripts/shard_selfplay.sh` (`nodes 0 → 200000`, depth 10). **The 200000 node cap is the likely
  speed culprit — reduce it.**

## Key commands
```bash
# generate (local shard): tune --nodes for speed, keep filters high
./zig-out/bin/selfplay --depth 12 --nodes 8000 --eval data/net_v7_lambda075.nnue \
    --num_games 50000 --num_threads <cores> --score_filter 10000 --adjudication_threshold 2500 > out.bin
# DGX Spark builds need -Dgb10=true; selfplay is CPU-bound (Grace cores are the limit)

# train (after fitting sigmoid divisor D to the new data's score scale)
./zig-out/bin/train_nnue --data out.bin --export data/net_v9.nnue \
    --epochs 50 --lr 0.005 --lambda 0.75 --sigmoid_divisor <D> --checkpoint_interval 10

# Elo-test (THE signal that matters)
uv run python testing/elo_test.py --mode quick \
  --current ./zig-out/bin/uci  --current-name v9 --uci0 EvalFile=data/net_v9.nnue \
  --baseline ./zig-out/bin/uci --baseline-name v7 --uci1 EvalFile=data/net_v7_lambda075.nnue

# inspect a net
zig build nnue-inspect -- --net data/net_v9.nnue --compare data/net_v7_lambda075.nnue
```

### Fit the sigmoid divisor D to a dataset (records are 35B: 32 pos + i16 score@32 + wdl@34)
```python
import struct, math
data=open('out.bin','rb').read(35*400000); m=len(data)//35
pairs=[(struct.unpack_from('<h',data,i*35+32)[0],
        1.0 if data[i*35+34]==0 else 0.0 if data[i*35+34]==1 else 0.5) for i in range(m)]
bestD=min(range(50,1201,5), key=lambda D: sum((1/(1+math.exp(-s/D))-o)**2 for s,o in pairs))
print("optimal sigmoid_divisor:", bestD)   # v7 data fit ~645; v8 (bad) ~235; use the new data's value
```

## Files
- Gen: `src/selfplay.zig` (filters/adjudication/`RecordCfg`), `scripts/{shard,cloud,fleet}_selfplay.sh`.
- Train: `src/train_nnue.zig`, `src/trainer/dataloader.zig`, `src/trainer/export.zig`.
- Net format/inference: `src/engine/nnue.zig`. Quant: FT scale 127, hidden 64, /8128 → cp.
- Data: `data/selfplay_final.bin` (v7's, good), `data/selfplay_nnue_mixed.bin` (v8's, do-not-reuse).
- v8 experiment nets (all regressions, for reference): `net_v8_mixed_lambda075`, `_wd0`, `_k235`, `_lambda0`.
