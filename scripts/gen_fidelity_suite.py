#!/usr/bin/env python3
"""Generate a quiet position suite representative of self-play labeling.

Plays self-play-style games with the engine (8 random opening plies, then
engine moves both sides) and samples positions with the SAME quiet filter as
src/selfplay.zig shouldRecord(): ply >= 16, not in check, |cp| < 3000, every
4th ply. Writes one FEN per line to the output file. Deterministic (seeded).

Usage:
  uv run python scripts/gen_fidelity_suite.py --engine /tmp/base-uci \
      --net data/net_v7_lambda075.nnue --out fidelity_suite.fen \
      --positions 160 --gen-depth 8
"""
import argparse
import random
import sys

import chess
import chess.engine

# Mirror selfplay.zig constants.
RANDOM_PLIES = 8
SKIP_PLIES = 16
SAMPLE_INTERVAL = 4
SCORE_FILTER = 3000
MAX_GAME_PLIES = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, default=160)
    ap.add_argument("--gen-depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-plies", type=int, default=80)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    eng.configure({"EvalFile": args.net, "Threads": 1})

    fens = []
    seen = set()
    games = 0
    while len(fens) < args.positions:
        board = chess.Board()
        ply = 0
        # Random opening.
        for _ in range(RANDOM_PLIES):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            ply += 1
        # Engine self-play with quiet sampling.
        while ply < args.max_plies and not board.is_game_over():
            res = eng.play(board, chess.engine.Limit(depth=args.gen_depth),
                           info=chess.engine.INFO_SCORE)
            cp = None
            if "score" in res.info:
                cp = res.info["score"].pov(board.turn).score(mate_score=100000)
            # shouldRecord: sample BEFORE making the move (the labeled position).
            if (ply >= SKIP_PLIES and not board.is_check()
                    and ply % SAMPLE_INTERVAL == 0
                    and cp is not None and abs(cp) < SCORE_FILTER):
                fen = board.fen()
                if fen not in seen:
                    seen.add(fen)
                    fens.append(fen)
                    if len(fens) >= args.positions:
                        break
            if res.move is None:
                break
            board.push(res.move)
            ply += 1
        games += 1
        print(f"\rgames={games} positions={len(fens)}", end="", file=sys.stderr)
    eng.quit()
    print(file=sys.stderr)

    with open(args.out, "w") as f:
        for fen in fens[:args.positions]:
            f.write(fen + "\n")
    print(f"wrote {min(len(fens), args.positions)} FENs to {args.out}")


if __name__ == "__main__":
    main()
