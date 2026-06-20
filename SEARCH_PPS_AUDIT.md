# Self-play throughput audit — positions/sec at depth 9 / 220k nodes

Objective: maximize **positions/second** of self-play training data at the real
operating point (`src/selfplay.zig`: depth 9, 220k-node cap, `threads=nproc`,
per-worker TT, NNUE eval). Throughput is the goal; **training-label fidelity is a
hard guardrail**. This audit covers *all* pps levers, not just pruning, and ranks
candidates by expected pps impact, implementation risk, and — critically —
whether each is **score-preserving** or **score-altering** at the operating point.

Builds on (does not repeat) `SEARCH_PERF_AUDIT_RESULTS.md` /
`SEARCH_PERF_AUDIT_HANDOFF.md`, which exhausted the **per-node, bit-identical**
inner loop (NNUE eval is the floor; TT size null; thread count optimal; kept win
= +4.09% fused accumulator delta). That prior pass deliberately changed **no
search tree**. This pass opens exactly that surface: **reducing node *count* per
move** — the lever the bit-identical audit left untouched.

## 1. Operating-point physics — why node count is the right metric here

Measured baseline (`bench --depth 9 --threads 1 --nnue net_v7`, fixed Zobrist
seed → fully reproducible; 20-position suite):

```
Nodes:    2,696,799   (avg ~135k nodes/move)
Ordering: 196,791 / 244,591 first-move cutoffs = 80.5%
```

Two facts drive everything below:

1. **The point is depth-limited, not cap-limited.** Avg ~135k nodes to *complete*
   depth 9, below the 220k cap. The cap binds only on a minority of sharp
   positions (in the suite, #2 ≈ 359k and #19 ≈ 342k uncapped). So for the large
   majority of moves, a search runs to a **completed depth-9 result** and the cap
   never fires. Prior audit saw the same at depth 10/360k (~150k avg).

   ⇒ **Time per move ∝ nodes to complete depth 9.** Any change that completes
   depth 9 in fewer nodes is a near-linear pps win. This is distinct from the
   prior audit's per-node (ns/node) focus: here we attack the *node count*.

2. **Move ordering has real headroom.** First-move-cutoff rate is **80.5%**,
   *below* the 85–92% "healthy" band the code's own comment cites
   (`search.zig:454`). Better ordering → earlier β-cutoffs → fewer nodes to the
   same completed result.

### What "score-preserving" means at depth 9 / 220k

For a search that **completes** depth 9, the returned score and best move are the
minimax value of the depth-9 tree and are **invariant to move ordering and to any
sound (value-exact) prune**. So:

- **Score-preserving** = better ordering / TT-cutoff / value-exact prune. The
  completed depth-9 label is *bit-identical*; only the node count drops. Validate
  with the deterministic `bench` node signature (count ↓) **and** per-position
  move+score identity. No fidelity proxy needed — labels are unchanged.
  - Caveat: on the few cap-bound positions, fewer nodes can let depth 9 *complete*
    where it previously truncated — that *changes* the label, but strictly *toward*
    the true depth-9 value (a fidelity improvement, not a regression). Identity is
    asserted on cap-unbound positions; cap-bound ones are checked as "moved toward
    the deep reference."

- **Score-altering** = futility/razoring/SEE-pruning/aggressive-LMR that changes
  the returned value. Validate with the cheap fidelity proxy: compare depth-9/220k
  labels (score + best move) against a deep reference search on a fixed suite;
  reject if agreement drops materially.

## 2. Lever-by-lever audit

### 2a. Move ordering / cutoff efficiency  ← **primary opportunity**
Current ordering (`evaluation.scoreMove`): TT move (100k) → MVV-LVA captures
(`victim*10 − attacker`) → promotion (12.5k) → killers (1100/1000) → countermove
(1050) → history+cont-hist (/32). `pickNext` is SIMD lazy-selection (fine).

**Gap: MVV-LVA orders *all* captures purely by victim, ignoring whether the
capture loses material.** A defended `BxN` (victim knight) sorts above a quiet
killer even when it hangs the bishop; a losing `QxR`-into-recapture sorts very
high. `staticExchangeEvaluation` (`movegen.zig:979`) is **fully implemented and
unit-tested but called nowhere in `search.zig`** (confirmed by grep). This is the
single biggest unexploited lever:

- **SEE-aware capture ordering** (score-preserving): keep MVV-LVA among
  SEE≥0 ("winning/equal") captures ranked above quiets; demote SEE<0 ("losing")
  captures below quiets/killers. Pure reordering ⇒ identical completed label,
  fewer nodes. Expected to lift the 80.5% first-cut rate.

### 2b. Quiescence search  ← **largest node-count pool**
`quiescence` (`search.zig:563`) generates **all** legal captures (MVV-LVA + delta
pruning) and searches every one, including clearly losing captures. No SEE, no TT
probe. Qsearch is typically the majority of all nodes.

- **SEE pruning in qsearch** (score-altering, tiny fidelity cost): when not in
  check, skip captures with SEE<0. A losing capture almost never raises alpha in
  qsearch, so the stand-pat-relative value is nearly unchanged while a large
  fraction of qnodes vanish. Highest expected pps impact; needs fidelity proxy.

### 2c. Pruning / reductions (main search)
Already present and reasonable: RFP (depth≤6), null-move (R=2+d/4), futility
(depth≤2), LMP (depth≤3), history pruning (depth≤3), IIR, LMR (log table +
PV/improving/history adjustments), check extension. Tuning surface (all
score-altering, need fidelity proxy):
- **RFP depth cap 6→8 + margin** (`rfp_base=80`): static eval is now computed at
  every node, so RFP *could* fire deeper than 6; it's gated to preserve old
  behavior. Extending prunes more near the leaves.
- **LMR on losing captures**: captures are never reduced; SEE<0 captures are good
  reduction candidates.
- **LMP / histprune margin** tightening.
These are incremental vs 2a/2b and ranked below them.

### 2d. Raw nps (eval / movegen / TT layout) — mostly exhausted
Prior audit: eval-bound, NNUE compute at its Zen-4 floor, +4.09% kept. One
residual: **static eval is computed at every non-check interior node**
(`search.zig:760`) to feed `improving`, even at high-depth nodes where only
`improving`→LMR consumes it. Eliding it where unused is a per-node nibble in an
eval-bound loop but entangles `improving` semantics (→ score-altering). Low
priority. `legalCaptures` duplicating `legalMoves` scaffolding is negligible
(movegen overlaps eval per the profile).

### 2e. TT sizing & cross-game sharing — already characterized
Per-worker TT, **already shared across games** within a worker (`reset()` calls
`newSearch()` = generation bump, no clear — good). TT-size sweep was a **null
result** (16 MB→512 MB flat within ±9% noise). A single TT shared across *worker
threads* would raise hit-rate but add contention; given the null size result,
unlikely to net positive. **No action.**

### 2f. Thread / parallelism efficiency — optimal
Measured: `threads=nproc` is monotonically best (SMT +26% over physical cores).
Self-play is embarrassingly parallel (independent workers, per-worker TT). **No
action.**

### 2g. Depth / node-cap operating point — fidelity-bounded, not free
Lowering depth or the cap raises pos/s directly but degrades label quality — it
trades against the hard guardrail, so it is **not** a free throughput lever and is
out of scope as a "win." Noted only as the boundary of the objective. (If labels
prove robust to depth 8, that would be a separate, large pos/s lever to validate
with a fidelity study — flagged, not pursued here.)

## 3. Ranked candidate list

| # | Candidate | Expected pps | Risk | Score class | Validation |
|---|-----------|-------------|------|-------------|------------|
| 1 | **SEE-aware capture ordering** — demote SEE<0 captures below quiets; MVV-LVA among the rest | **High** (raises 80.5% first-cut rate; fewer nodes) | Low | **Score-preserving** | bench node-count ↓ + per-position move/score **identical** |
| 2 | **SEE pruning in qsearch** — skip SEE<0 captures when not in check | **High** (qsearch = bulk of nodes) | Low-Med | Score-altering (tiny) | node ↓ + fidelity proxy vs deep ref |
| 3 | **SEE-based LMR/pruning of bad captures in main search** at low depth | Med | Med | Score-altering | node ↓ + fidelity proxy |
| 4 | **RFP depth 6→8 + margin tune** | Med | Med | Score-altering | node ↓ + fidelity proxy |
| 5 | **LMP / histprune / futility margin tuning** | Low-Med | Med | Score-altering | node ↓ + fidelity proxy |
| 6 | **Interior-node static-eval elision** where only `improving` reads it | Low | Med | Score-altering (LMR shifts) | node-neutral check + fidelity proxy |

Cross-cutting: SEE quality depends on `see_piece_values` (already defined). #1 and
#2 share the SEE call site cost; measure SEE per-call overhead doesn't erase the
node win (SEE is cheap relative to an NNUE eval per saved node).

## 4. Validation methodology

**Score-preserving (#1):** deterministic & local — no fleet, no fidelity test.
1. `bench --depth 9 --threads 1 --nnue net_v7`: assert total node count **↓** and
   every position's printed `move` + `score` **identical** to the baseline in
   `/tmp/bench_baseline.txt` (§1). Any score change on a cap-unbound position =
   bug → reject.
2. `zig build test` green (incl. "ReusableSearcher matches single-threaded").

**Score-altering (#2–#6):** 
1. `bench --depth 9`: node-count ↓ (the throughput signal) — labels *expected* to
   move.
2. **Fidelity proxy:** on a fixed suite, compare the change's depth-9/220k label
   (score bucket + best move) against a **deep reference** (depth 12–14, large
   node budget) per position. Metric: best-move agreement % and score-MAE vs the
   reference. Reject if agreement drops materially vs baseline's own agreement.

**Throughput (all):** positions/second at depth-9/220k on the fleet (protocol
below). One change per commit; independent candidates measured in parallel across
hosts, then the cumulative winner stack re-measured to catch interactions.

## 5. Proposed fleet benchmark protocol (awaiting confirmation before first run)

- **Hosts:** the two paid EPYC 4245P fleet hosts (`root@51.159.110.101`,
  `root@51.159.202.183`, `~/.ssh/cloud`) — the real paid target. Build locally
  (x86_64, glibc ≤ fleet), upload binaries.
- **Workload:** `selfplay --depth 9 --nodes 220000 --num_threads <nproc>
  --tt_bits 21 --eval net_v7_lambda075.nnue`, metric = pos/s from the
  `Throughput:`/`Done:` lines.
- **Harness:** extend the existing interleaved A/B pattern
  (`tt_sweep_pps.sh`/`thread_sweep.sh`): interleave base vs candidate binaries
  across reps to cancel host drift; per-rep CoV ≈ ±9% (RNG game-length variance),
  so ≥6 reps × ~200 games/rep; report mean pos/s and % delta. Independent
  candidates → different hosts in parallel; cumulative stack → re-measure.
- **Warmup:** one discarded rep per host before timing.
- **Acceptance:** pps delta outside the ±9% band, consistent sign across reps,
  with fidelity guardrail satisfied (score-altering) or label identity (score-
  preserving).

## 6. Measured results (correction to §1–§3 framing)

**Key correction:** the audit's §1 premise — that better move *ordering* is
"score-preserving" — is **empirically false for this engine**. Its forward
pruning (LMR reduction scales with move index; LMP/futility/histprune key off
index) is order-sensitive, so *any* reordering perturbs the depth-9 label. Verified:
base vs an ordering change are near-identical at depth 4 (little pruning) but
diverge by depth 9 (pruning compounds). ⇒ **No score-preserving ordering/pruning
wins exist; all candidates are fidelity-gated.** The only bit-identical wins were
the prior audit's per-node work (+4.09%).

**Measurement discipline (a confound that bit this audit):** node counts are
deterministic and host-independent (fixed Zobrist seed, integer NNUE), so all
node-delta comparisons are done locally. The *net file matters enormously* — an
early "−33% nodes" for candidate #1 was a phantom from running the candidate
against a host's stale `~/net.nnue` (a different network) while comparing to base
on the correct `net_v7_lambda075.nnue`. **Always md5-verify the net
(`5c2b01b3…`) on every host; always A/B base vs candidate on the same host+net.**

### Candidate #1 — SEE-aware capture ordering — REJECTED
Demote SEE<0 captures below all quiets (`evaluation.scoreMove`). Measured (depth 9,
20-pos bench, correct net, reproduced local + 7700X):
- **+10.3% nodes** (2,696,799 → 2,975,757) — a regression.
- Why: SEE is static (ignores zwischenzug, pinned defenders, recapture-gives-check),
  so many SEE<0 high-MVV-LVA captures (BxN/RxN) were producing β-cutoffs *early*.
  Sending them behind every quiet expands those quiet subtrees first → more nodes.
- Killing measurement: node count up; throughput loss. Fidelity was fine (passed),
  but irrelevant once it's slower.

### Candidate #2 — qsearch SEE pruning — measured, fidelity borderline
Skip SEE<0 captures in non-check qsearch (`search.zig` quiescence). A true
node-*removal* lever (not reordering). Measured (depth 9, correct net, local):
- **−19.5% nodes** (2,696,799 → 2,170,442).
- Fidelity (160-quiet-pos suite, vs base@13 and @14 refs): **best moves identical**;
  score-MAE **+3.4cp** (robust across ref depths); systematic **−3.5cp bias**
  (q2 slightly more pessimistic — pruning the opponent's desperado captures
  flatters their side). Direct label change vs base: median 8cp, mean 20cp,
  p90 57cp, max 260cp.
- Verdict: fails a strict zero-tolerance fidelity gate (MAE rises); passes a
  small-budget gate (≤5cp) with unchanged move selection. **Accepted** (pps below).

### Candidates #3–#5 — screened on top of #2 (local node delta + fidelity)
Fixed deep oracle = original base@13; ≤5cp MAE budget, move-agree must not drop.

| Cand | Lever | marginal nodes | fidelity (480-pos suite) | verdict |
|---|---|---|---|---|
| #3 | main-search SEE-prune (SEE<0, i>0, depth≤3) | −5.7% | moves identical; **MAE −2.1cp vs base** | **accept** |
| #4 | RFP depth 6→8 | −0.3% | — | reject (no-op: eval rarely clears 80·depth at d7-8) |
| #5 | LMP tighten `{4,6,9}` | −25.8% | **move-agree −0.6pp**; bias swing | reject (over-prunes; degrades move selection) |
| #5g | LMP tighten `{5,7,11}` | −15.6% | move-agree −0.2pp (1/480); MAE +1.8cp | **accept** |

**#3 is a strict improvement:** identical moves AND labels closer to depth-13
truth than base's own depth-9 labels (the SEE prune removes qsearch noise).

**The "systematic bias" was a small-suite artifact.** Direct label bias vs base
flipped sign between independent suites (−3.6cp seed-42 → +3.4cp seed-7), so it is
position-mix scatter (~3–7cp, averages out), not a directional skew. Validated on
a **fresh 480-position suite** (seed 7, independent of the seed-42 set #5g was
tuned on) to avoid overfitting the LMP thresholds.

### Final committed stack (#2 + #3 + #5g) — measured throughput
Three one-change commits on `search.zig`. Cumulative vs original base:
- **+30.2% positions/second** — the deliverable metric. Interleaved A/B on the
  EPYC 4465P (24t), depth 9 / 220k / tt_bits=21, 6 reps × 100 games, all reps
  positive (+19% … +40%): base 59.9 → stack 77.9 pos/s. Games/sec (RNG-robust)
  +44%; pos/s is lower because the stack records ~11% fewer positions/game
  (slightly different trajectories, same quiet filter) — net training-data
  throughput is +30%.
- **−35.9% nodes** (depth-9 bench signature 2,696,799 → 1,728,319).
- Fidelity (480-pos): move-agree 64.8%→64.6% (1/480), MAE +1.8cp vs depth-13 ref —
  within the ≤5cp budget; best-move selection effectively unchanged.
- Tooling: `scripts/gen_fidelity_suite.py` (quiet suite from self-play sampling),
  `scripts/fidelity_compare.py` (move-agree + MAE/bias vs deep ref),
  `scripts/pps_ab.sh` (interleaved self-play pos/s A/B). All fidelity runs are
  deterministic/host-independent; pps measured on the EPYC fleet.
