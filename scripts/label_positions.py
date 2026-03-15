#!/usr/bin/env python3
"""Label positions with Stockfish centipawn evaluations.

Reads FEN positions (one per line) and runs Stockfish at a configurable depth,
outputting EPD lines with centipawn annotations.

Usage:
    python scripts/label_positions.py --input quiet.fen --output dataset.epd
    python scripts/label_positions.py --input quiet.fen --depth 15 --workers 8

Output format (EPD with ce opcode):
    rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - ce "45";

The centipawn value is always from white's perspective.
Positions with |eval| > 1000cp are discarded (decided games).
Mate scores are capped at +/-10000cp.
"""

import argparse
import sys
import os
from multiprocessing import Pool
from functools import partial

import chess
import chess.engine


# Sentinel for "no result" — filtered or errored positions
SKIP = None


def evaluate_batch(fens, stockfish_path, depth, max_eval_cp, time_limit):
    """Evaluate a batch of FENs using a single Stockfish instance.

    Returns a list of (fen, centipawn) tuples, or SKIP for filtered positions.
    Uses both depth and time limits to prevent hanging on difficult positions.
    Restarts the engine after timeouts since it may be left in a bad state.
    """
    results = []
    engine = None

    def start_engine():
        try:
            return chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception as e:
            print(f"Error starting Stockfish: {e}", file=sys.stderr)
            return None

    engine = start_engine()
    if engine is None:
        return [SKIP] * len(fens)

    try:
        for fen in fens:
            try:
                board = chess.Board(fen)
                limit = chess.engine.Limit(depth=depth, time=time_limit)
                info = engine.analyse(board, limit)
                score = info["score"].white()

                if score.is_mate():
                    mate_in = score.mate()
                    cp = 10000 if mate_in > 0 else -10000
                else:
                    cp = score.score()

                # Discard positions with extreme evaluations
                if abs(cp) > max_eval_cp:
                    results.append(SKIP)
                else:
                    results.append((fen, cp))
            except chess.engine.EngineTerminatedError:
                results.append(SKIP)
                engine = start_engine()
                if engine is None:
                    results.extend([SKIP] * (len(fens) - len(results)))
                    return results
            except Exception:
                results.append(SKIP)
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

    return results


def fen_to_epd_base(fen):
    """Convert a full FEN to EPD base (first 4 fields only)."""
    parts = fen.split()
    return " ".join(parts[:4])


def main():
    parser = argparse.ArgumentParser(description="Label positions with Stockfish evaluations")
    parser.add_argument("--input", required=True, help="Input FEN file (one FEN per line)")
    parser.add_argument("--output", default=None, help="Output EPD file (default: stdout)")
    parser.add_argument("--stockfish", default="stockfish",
                        help="Path to Stockfish binary (default: 'stockfish' in PATH)")
    parser.add_argument("--depth", type=int, default=15,
                        help="Stockfish search depth (default: 15)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel Stockfish instances (default: CPU count)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Positions per Stockfish batch (default: 256)")
    parser.add_argument("--max-eval", type=int, default=1000,
                        help="Discard positions with |eval| > this (cp, default: 1000)")
    parser.add_argument("--time-limit", type=float, default=10.0,
                        help="Per-position time limit in seconds (default: 10)")
    parser.add_argument("--max-positions", type=int, default=0,
                        help="Stop after labeling this many positions (0 = unlimited)")
    parser.add_argument("--dedup", action="store_true",
                        help="Deduplicate positions by EPD base (first 4 FEN fields)")
    args = parser.parse_args()

    workers = args.workers or os.cpu_count() or 1

    # Read all FENs
    print(f"Reading FENs from {args.input}...", file=sys.stderr)
    with open(args.input) as f:
        fens = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(fens)} FENs", file=sys.stderr)

    if args.max_positions > 0:
        fens = fens[:args.max_positions * 2]  # overshoot to account for filtering

    # Split into batches
    batches = [fens[i:i + args.batch_size] for i in range(0, len(fens), args.batch_size)]
    print(f"Processing {len(batches)} batches with {workers} workers at depth {args.depth}...",
          file=sys.stderr)

    eval_fn = partial(evaluate_batch,
                      stockfish_path=args.stockfish,
                      depth=args.depth,
                      max_eval_cp=args.max_eval,
                      time_limit=args.time_limit)

    out = open(args.output, "w") if args.output else sys.stdout
    seen = set() if args.dedup else None
    total = 0
    kept = 0
    filtered_eval = 0
    duplicates = 0

    try:
        with Pool(workers) as pool:
            for batch_results in pool.imap(eval_fn, batches):
                for result in batch_results:
                    total += 1
                    if result is SKIP:
                        filtered_eval += 1
                        continue

                    fen, cp = result

                    if seen is not None:
                        epd_base = fen_to_epd_base(fen)
                        if epd_base in seen:
                            duplicates += 1
                            continue
                        seen.add(epd_base)

                    # Output EPD format with ce annotation
                    epd_base = fen_to_epd_base(fen)
                    out.write(f'{epd_base} ce "{cp}";\n')
                    kept += 1

                    if args.max_positions > 0 and kept >= args.max_positions:
                        break

                if kept % 10000 < args.batch_size:
                    print(f"  processed={total} kept={kept} filtered={filtered_eval} dupes={duplicates}",
                          file=sys.stderr)

                if args.max_positions > 0 and kept >= args.max_positions:
                    break

    finally:
        if args.output:
            out.close()

    print(f"Done. total={total} kept={kept} filtered_eval={filtered_eval} duplicates={duplicates}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
