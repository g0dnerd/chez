#!/usr/bin/env python3
"""Fidelity proxy: judge depth-9 labels of one or more candidate engines against
a deep reference (base engine at --ref-depth) on a fixed quiet suite.

For each FEN:
  ref   = reference engine at ref-depth  (trusted "deep" label: move + cp)
  label = each engine at label-depth      (the self-play operating point)
Metrics per engine, vs ref: best-move agreement % and score MAE (cp, clamped to
the self-play label domain +/-3000). A candidate passes if its agreement does
not drop and its MAE does not rise vs the baseline engine's own numbers.

Usage:
  uv run python scripts/fidelity_compare.py --suite fidelity_suite.fen \
      --net data/net_v7_lambda075.nnue --ref /tmp/base-uci \
      --label base=/tmp/base-uci --label see=/tmp/see-uci \
      --ref-depth 13 --label-depth 9 --jobs 6
"""
import argparse
import concurrent.futures as cf

import chess
import chess.engine

CP_CLAMP = 3000


def _engine(path, net):
    e = chess.engine.SimpleEngine.popen_uci(path)
    e.configure({"EvalFile": net, "Threads": 1})
    return e


def _probe(eng, board, depth):
    info = eng.analyse(board, chess.engine.Limit(depth=depth))
    cp = info["score"].pov(board.turn).score(mate_score=100000)
    pv = info.get("pv")
    mv = pv[0].uci() if pv else None
    return mv, cp


def _worker(args):
    ref_path, label_specs, net, ref_depth, label_depth, fens = args
    ref_eng = _engine(ref_path, net)
    label_engs = {name: _engine(path, net) for name, path in label_specs}
    out = []
    for fen in fens:
        board = chess.Board(fen)
        ref_mv, ref_cp = _probe(ref_eng, board, ref_depth)
        row = {"ref_mv": ref_mv, "ref_cp": ref_cp, "labels": {}}
        for name, _ in label_specs:
            mv, cp = _probe(label_engs[name], board, label_depth)
            row["labels"][name] = (mv, cp)
        out.append(row)
    ref_eng.quit()
    for e in label_engs.values():
        e.quit()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--label", action="append", required=True,
                    help="name=path; repeatable")
    ap.add_argument("--ref-depth", type=int, default=13)
    ap.add_argument("--label-depth", type=int, default=9)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    label_specs = [tuple(s.split("=", 1)) for s in args.label]
    with open(args.suite) as f:
        fens = [ln.strip() for ln in f if ln.strip()]

    # Shard FENs across workers.
    shards = [fens[i::args.jobs] for i in range(args.jobs)]
    tasks = [(args.ref, label_specs, args.net, args.ref_depth,
              args.label_depth, sh) for sh in shards if sh]

    rows = []
    with cf.ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        for res in ex.map(_worker, tasks):
            rows.extend(res)

    def clamp(x):
        return max(-CP_CLAMP, min(CP_CLAMP, x))

    names = [n for n, _ in label_specs]
    base_name = names[0]
    agree = {n: 0 for n in names}
    abs_err = {n: 0 for n in names}   # MAE vs ref
    sgn_err = {n: 0 for n in names}   # signed mean error vs ref (bias)
    n = len(rows)
    for row in rows:
        rc = clamp(row["ref_cp"])
        for name in names:
            mv, cp = row["labels"][name]
            if mv == row["ref_mv"]:
                agree[name] += 1
            cc = clamp(cp)
            abs_err[name] += abs(cc - rc)
            sgn_err[name] += (cc - rc)

    print(f"# suite={args.suite} n={n} ref={args.ref}@{args.ref_depth} "
          f"label_depth={args.label_depth}")
    print(f"{'engine':<10} {'move-agree%':>12} {'MAE-vs-ref':>11} {'bias-vs-ref':>12}")
    for name in names:
        ap_ = 100.0 * agree[name] / n
        mae = abs_err[name] / n
        bias = sgn_err[name] / n
        tag = ""
        if name != base_name:
            ba = 100.0 * agree[base_name] / n
            bm = abs_err[base_name] / n
            tag = "  PASS" if (ap_ >= ba and mae <= bm) else "  FAIL"
            tag += f" (agree {ap_-ba:+.1f}pp, MAE {mae-bm:+.1f}cp)"
        print(f"{name:<10} {ap_:>11.1f}% {mae:>10.1f} {bias:>+11.1f}{tag}")

    # Direct candidate-vs-base label change (how much the actual labels move).
    print("\n# direct label change vs base (same position, depth-9 score)")
    print(f"{'engine':<10} {'bias':>7} {'MAE':>7} {'p50|d|':>7} {'p90|d|':>7} "
          f"{'max|d|':>7} {'<=5cp':>7} {'<=25cp':>7}")
    for name in names[1:]:
        diffs = [clamp(row["labels"][name][1]) - clamp(row["labels"][base_name][1])
                 for row in rows]
        ad = sorted(abs(d) for d in diffs)
        def pct(q):
            return ad[min(len(ad) - 1, int(q * len(ad)))]
        bias = sum(diffs) / n
        mae = sum(ad) / n
        le5 = 100.0 * sum(1 for d in ad if d <= 5) / n
        le25 = 100.0 * sum(1 for d in ad if d <= 25) / n
        print(f"{name:<10} {bias:>+7.1f} {mae:>7.1f} {pct(0.50):>7.0f} "
              f"{pct(0.90):>7.0f} {ad[-1]:>7.0f} {le5:>6.0f}% {le25:>6.0f}%")


if __name__ == "__main__":
    main()
