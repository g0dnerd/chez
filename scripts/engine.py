#!/usr/bin/env python3
"""
Neural network chess engine for Chez.

Reads positions from stdin, outputs best moves to stdout.
Designed to be spawned as a subprocess by the Zig TUI.

Protocol:
  Input:  FEN string (one per line)
  Output: UCI move (e.g., "e2e4", "e7e8q")

Usage:
  # Standalone test
  echo "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" | uv run python scripts/engine.py

  # With checkpoint
  uv run python scripts/engine.py --checkpoint models/iter_0100.pt
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from alphazero.bindings import State, CMove
from alphazero.network import create_network, get_device
from alphazero.mcts import MCTS, MCTSConfig
from alphazero.encoding import StateEncoder


def move_to_uci(move: CMove) -> str:
    """Convert CMove to UCI string."""
    files = "abcdefgh"
    start_file = move.start % 8
    start_rank = move.start // 8
    end_file = move.end % 8
    end_rank = move.end // 8

    uci = f"{files[start_file]}{start_rank + 1}{files[end_file]}{end_rank + 1}"

    if move.promotion_piece:
        uci += "nbrq"[move.promotion_piece - 1]

    return uci


def main():
    parser = argparse.ArgumentParser(description="Neural network chess engine")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: latest in models/)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
        help="MCTS simulations per move (default: 400)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Move selection temperature (0 = greedy, default: 0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info to stderr",
    )
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Find latest checkpoint in models/
        models_dir = Path("models")
        if not models_dir.exists():
            print("ERROR: No models/ directory found", file=sys.stderr)
            sys.exit(1)

        checkpoints = sorted(models_dir.glob("iter_*.pt"))
        if not checkpoints:
            print("ERROR: No checkpoints found in models/", file=sys.stderr)
            sys.exit(1)

        checkpoint_path = checkpoints[-1]

    if args.debug:
        print(f"Loading checkpoint: {checkpoint_path}", file=sys.stderr)

    # Load checkpoint
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract network config from checkpoint
    config = checkpoint.get("config")
    if config is not None:
        num_filters = config.num_filters
        num_blocks = config.num_blocks
    else:
        # Fallback for old checkpoints
        num_filters = checkpoint.get("num_filters", 128)
        num_blocks = checkpoint.get("num_blocks", 6)

    if args.debug:
        print(f"Network: {num_filters} filters, {num_blocks} blocks", file=sys.stderr)
        print(f"Device: {device}", file=sys.stderr)

    # Create network and load weights
    network = create_network(num_filters, num_blocks, device=device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()

    # Create MCTS
    mcts_config = MCTSConfig(
        num_simulations=args.simulations,
        batch_size=16,
    )
    mcts = MCTS(network, mcts_config)

    if args.debug:
        print(f"MCTS: {args.simulations} simulations", file=sys.stderr)
        print("Ready", file=sys.stderr)

    # Main loop: read FEN, output move
    encoder = StateEncoder()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if line.lower() in ("quit", "exit"):
            break

        try:
            # Parse FEN
            state = State.from_fen(line)

            # Reset encoder for new game position
            # (we don't have history context from subprocess)
            encoder.reset()

            # Search
            policy, value = mcts.search(state, encoder=encoder)

            # Select move (greedy by default for play)
            move = mcts.select_move(policy, state, temperature=args.temperature)

            # Output UCI move
            uci = move_to_uci(move)
            print(uci, flush=True)

            if args.debug:
                print(f"FEN: {line}", file=sys.stderr)
                print(f"Move: {uci}, Value: {value:.3f}", file=sys.stderr)

        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            print("0000", flush=True)  # Null move on error


if __name__ == "__main__":
    main()
