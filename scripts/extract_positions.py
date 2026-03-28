#!/usr/bin/env python3
"""Extract candidate positions from PGN files for Texel tuning.

Reads PGN games and outputs EPD lines with game-outcome (c9) annotations,
ready for direct use by the tuner without a separate SF labeling step.

Usage:
    python scripts/extract_positions.py data/pgn/*.pgn > dataset.epd
    python scripts/extract_positions.py --skip-plies 16 --sample-interval 4 game.pgn

Output format (EPD with c9 opcode):
    rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - c9 "1-0";

Filtering rules:
  - Skip games without a decisive or drawn result
  - Skip first N plies (default 16 = 8 full moves): opening book territory
  - Skip positions after ply 400 (move 200): rare, noisy
  - Sample 1 position every M plies (default 4): reduces inter-position correlation
  - Exclude positions where side to move is in check
  - Exclude positions where the last move was a capture
  - Exclude positions with fewer than 6 pieces total
"""

import argparse
import random
import sys

import chess
import chess.pgn


def fen_to_epd_base(fen):
    """Convert a full FEN to EPD base (first 4 fields only)."""
    parts = fen.split()
    return " ".join(parts[:4])


def extract_from_game(game, result_str, skip_plies=16, max_ply=400, sample_interval=4):
    """Yield (epd_base, result_str) tuples from a single game."""
    board = game.board()
    ply = 0
    last_was_capture = False

    for move in game.mainline_moves():
        is_capture = board.is_capture(move)
        board.push(move)
        ply += 1

        # Skip opening
        if ply <= skip_plies:
            last_was_capture = is_capture
            continue

        # Skip late endgame noise
        if ply > max_ply:
            break

        # Sample every N plies (with random offset per game for diversity)
        if ply % sample_interval != 0:
            last_was_capture = is_capture
            continue

        # Skip if the last move was a capture (likely tactical aftermath)
        if is_capture or last_was_capture:
            last_was_capture = is_capture
            continue

        # Skip positions where side to move is in check
        if board.is_check():
            last_was_capture = is_capture
            continue

        # Skip positions with fewer than 6 pieces (tablebase territory)
        if len(board.piece_map()) < 6:
            last_was_capture = is_capture
            continue

        yield fen_to_epd_base(board.fen()), result_str
        last_was_capture = is_capture


def main():
    parser = argparse.ArgumentParser(description="Extract positions from PGN files")
    parser.add_argument("pgn_files", nargs="+", help="PGN file(s) to process")
    parser.add_argument(
        "--skip-plies",
        type=int,
        default=16,
        help="Skip first N plies (default: 16 = 8 full moves)",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=400,
        help="Skip positions after this ply (default: 400)",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=4,
        help="Sample 1 position every N plies (default: 4)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=0,
        help="Stop after this many positions (0 = unlimited)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    total_games = 0
    skipped_no_result = 0
    total_positions = 0
    valid_results = {"1-0", "0-1", "1/2-1/2"}

    for pgn_path in args.pgn_files:
        print(f"Processing {pgn_path}...", file=sys.stderr)
        with open(pgn_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                total_games += 1
                result_str = game.headers.get("Result", "*")
                if result_str not in valid_results:
                    skipped_no_result += 1
                    continue

                for epd_base, result in extract_from_game(
                    game,
                    result_str,
                    skip_plies=args.skip_plies,
                    max_ply=args.max_ply,
                    sample_interval=args.sample_interval,
                ):
                    print(f'{epd_base} c9 "{result}";')
                    total_positions += 1

                    if args.max_positions > 0 and total_positions >= args.max_positions:
                        print(
                            f"Reached max positions ({args.max_positions})",
                            file=sys.stderr,
                        )
                        print(
                            f"Total: {total_games} games ({skipped_no_result} skipped), {total_positions} positions",
                            file=sys.stderr,
                        )
                        return

                if total_games % 10000 == 0:
                    print(
                        f"  {total_games} games ({skipped_no_result} skipped), {total_positions} positions so far...",
                        file=sys.stderr,
                    )

    print(
        f"Total: {total_games} games ({skipped_no_result} skipped), {total_positions} positions",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
