#!/usr/bin/env python3
"""
Train AlphaZero on Chez chess engine.

Usage:
    # Quick test (few minutes)
    uv run python scripts/train.py --config small --iterations 5

    # Day-long training run (~1200 ELO target)
    uv run python scripts/train.py --config medium --iterations 100

    # Resume from checkpoint
    uv run python scripts/train.py --config medium --iterations 50 --checkpoint models/iter_0050.pt
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphazero.train import Trainer, TrainConfig
from alphazero.network import get_device


def main():
    parser = argparse.ArgumentParser(
        description="Train AlphaZero",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  %(prog)s --config small --iterations 5

  # Overnight training (~12h, ~50 iterations)
  %(prog)s --config medium --iterations 50

  # Full day training (~24h, ~100 iterations)
  %(prog)s --config medium --iterations 100

  # Resume from checkpoint
  %(prog)s --checkpoint models/iter_0050.pt --iterations 50
        """,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--config",
        choices=["small", "medium", "large"],
        default="medium",
        help="Configuration preset (default: medium)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models",
        help="Directory for checkpoints (default: models)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N iterations (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    args = parser.parse_args()

    # Create config based on preset
    if args.config == "small":
        # Quick test config (~2-3 min per iteration)
        config = TrainConfig(
            num_filters=64,
            num_blocks=4,
            mcts_simulations=50,
            mcts_batch_size=8,
            games_per_iteration=20,
            training_steps_per_iteration=200,
            batch_size=128,
            max_moves=150,
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
        )
    elif args.config == "medium":
        # Day-long training config (~13-14 min per iteration)
        # Target: ~1200 ELO after 100 iterations
        config = TrainConfig(
            num_filters=128,
            num_blocks=6,
            mcts_simulations=800,
            mcts_batch_size=16,
            games_per_iteration=100,
            training_steps_per_iteration=1000,
            batch_size=512,
            max_moves=200,
            temperature_moves=30,
            replay_buffer_size=200_000,
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
        )
    else:  # large
        # Extended training config
        config = TrainConfig(
            num_filters=256,
            num_blocks=10,
            mcts_simulations=400,
            mcts_batch_size=16,
            games_per_iteration=200,
            training_steps_per_iteration=2000,
            batch_size=512,
            max_moves=300,
            temperature_moves=30,
            replay_buffer_size=500_000,
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
        )

    print("=" * 60)
    print("AlphaZero Training")
    print("=" * 60)
    print(f"Config:      {args.config}")
    print(f"Device:      {get_device()}")
    print(f"Iterations:  {args.iterations}")
    print(f"Network:     {config.num_filters} filters, {config.num_blocks} blocks")
    print(
        f"MCTS:        {config.mcts_simulations} sims, batch_size={config.mcts_batch_size}"
    )
    print(
        f"Self-play:   {config.games_per_iteration} games/iter, max {config.max_moves} moves"
    )
    print(
        f"Training:    {config.training_steps_per_iteration} steps/iter, batch={config.batch_size}"
    )
    print(
        f"Checkpoints: {config.checkpoint_dir}/ (every {config.save_every} iterations)"
    )
    print("=" * 60)
    print()

    # Create trainer
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        print()

    # Estimate time
    if args.config == "small":
        est_time = args.iterations * 2.5  # ~2.5 min per iteration
    elif args.config == "medium":
        est_time = args.iterations * 13  # ~13 min per iteration
    else:
        est_time = args.iterations * 25  # ~25 min per iteration (estimate)

    if est_time < 60:
        print(f"Estimated time: ~{est_time:.0f} minutes")
    else:
        print(f"Estimated time: ~{est_time / 60:.1f} hours")
    print()

    # Train
    try:
        trainer.train(args.iterations, verbose=not args.quiet)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Total iterations: {trainer.iteration}")
    print(f"Total games:      {trainer.total_games}")
    print(f"Total positions:  {len(trainer.buffer)}")
    print(f"Total steps:      {trainer.total_steps}")
    print(f"Latest checkpoint: {config.checkpoint_dir}/iter_{trainer.iteration:04d}.pt")


if __name__ == "__main__":
    main()
