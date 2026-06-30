#!/usr/bin/env python3
"""Build a blended selfplay opening book (one FEN per line).

Mixes the off-startpos balanced UHO suite with near-startpos lines produced by
short random walks from the initial position, so the next selfplay dataset
regains coverage of the standard opening region (which a UHO-only book skips,
since those positions start around move ~6-9).

The output is a plain FEN-per-line file consumed directly by selfplay's existing
`--openings` flag -- no engine change. To use it for the next generation run,
add to the launch command, e.g.:

    scripts/fleet_selfplay.sh start ... -- --openings data/selfplay_book_blended.fen

Usage:
    uv run python scripts/build_selfplay_book.py --out /tmp/book.fen
    uv run python scripts/build_selfplay_book.py --startpos-fraction 0.33 --walk-max 6

Composition: keeps all base lines and appends round(base * f/(1-f)) near-startpos
lines so near/(near+base) = f. Final list is shuffled (seeded) so startpos lines
are not clustered.
"""

import argparse
import random
import sys
from collections import Counter

import chess


def read_base(path):
    """Read full-FEN lines from the base book, skipping blank lines."""
    lines = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line:
                lines.append(line)
    return lines


def random_walk_fen(walk_min, walk_max):
    """Play a uniform-random number of legal plies from startpos, return (fen, n).

    Breaks early if a terminal position (no legal moves) is reached; n is the
    actual number of plies played (n == 0 yields the bare startpos FEN).
    """
    n = random.randint(walk_min, walk_max)
    board = chess.Board()
    played = 0
    for _ in range(n):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
        played += 1
    return board.fen(), played


def main():
    parser = argparse.ArgumentParser(
        description="Build a blended selfplay opening book (FEN per line)"
    )
    parser.add_argument(
        "--base",
        default="testing/books/UHO_4060_v3.epd",
        help="Base off-startpos book, one full FEN per line "
        "(default: testing/books/UHO_4060_v3.epd)",
    )
    parser.add_argument(
        "--out",
        default="data/selfplay_book_blended.fen",
        help="Output FEN file (default: data/selfplay_book_blended.fen)",
    )
    parser.add_argument(
        "--startpos-fraction",
        type=float,
        default=0.33,
        help="Fraction of the final book that is near-startpos lines (default: 0.33)",
    )
    parser.add_argument(
        "--walk-min",
        type=int,
        default=0,
        help="Minimum random-walk plies (0 = bare startpos) (default: 0)",
    )
    parser.add_argument(
        "--walk-max",
        type=int,
        default=6,
        help="Maximum random-walk plies (default: 6)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=0,
        help="Optional cap on base lines used (0 = use all)",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate near-startpos lines (default: off, keep duplicates)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)"
    )
    args = parser.parse_args()

    f = args.startpos_fraction
    if not (0.0 <= f < 1.0):
        parser.error("--startpos-fraction must be in [0, 1)")
    if args.walk_min < 0 or args.walk_max < args.walk_min:
        parser.error("require 0 <= --walk-min <= --walk-max")

    random.seed(args.seed)

    base = read_base(args.base)
    if args.size > 0 and args.size < len(base):
        base = random.sample(base, args.size)
    base_count = len(base)
    if base_count == 0:
        parser.error(f"no FEN lines read from base book: {args.base}")

    near_count = round(base_count * f / (1.0 - f)) if f > 0.0 else 0

    near = []
    walk_hist = Counter()
    for _ in range(near_count):
        fen, n = random_walk_fen(args.walk_min, args.walk_max)
        near.append(fen)
        walk_hist[n] += 1

    distinct_near = len(set(near))
    if args.dedup:
        near = list(dict.fromkeys(near))

    book = base + near
    random.shuffle(book)

    with open(args.out, "w") as out:
        out.write("\n".join(book))
        out.write("\n")

    bare_count = walk_hist.get(0, 0)

    print(f"Base lines:        {base_count}", file=sys.stderr)
    print(
        f"Near lines:        {len(near)} (requested {near_count}, "
        f"distinct {distinct_near}{', deduped' if args.dedup else ''})",
        file=sys.stderr,
    )
    print(f"Bare-startpos:     {bare_count}", file=sys.stderr)
    print(f"Final total:       {len(book)}", file=sys.stderr)
    print(
        f"Near fraction:     {len(near) / len(book):.3f} (target {f})",
        file=sys.stderr,
    )
    print("Walk-length histogram (plies played):", file=sys.stderr)
    for n in range(args.walk_min, args.walk_max + 1):
        c = walk_hist.get(n, 0)
        bar = "#" * (c * 40 // near_count) if near_count else ""
        print(f"  {n:2d}: {c:8d} {bar}", file=sys.stderr)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
