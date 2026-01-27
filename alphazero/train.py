"""
Training loop for AlphaZero.

Alternates between self-play game generation and network training.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch

# import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# from torch.optim.lr_scheduler import CosineAnnealingLR

# from .network import AlphaZeroNetwork,
from .network import create_network, get_device, count_parameters
from .mcts import MCTSConfig
from .selfplay import generate_games
from .replay_buffer import ReplayBuffer


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Network architecture
    num_filters: int = 128
    num_blocks: int = 6

    # MCTS
    mcts_simulations: int = 200  # Higher = better quality training data
    mcts_batch_size: int = 16  # Leaves to evaluate in parallel (GPU batching)

    # Self-play
    games_per_iteration: int = 100
    temperature_moves: int = 30  # Use temperature=1 for first N moves
    max_moves: int = 200  # Cap game length to speed up iteration

    # Training
    batch_size: int = 512
    learning_rate: float = 0.001  # Adam-appropriate LR
    weight_decay: float = 1e-4
    training_steps_per_iteration: int = 1000

    # Buffer
    replay_buffer_size: int = 200_000

    # Checkpointing
    checkpoint_dir: str = "models"
    save_every: int = 1  # Save every N iterations


class Trainer:
    """
    AlphaZero training manager.

    Handles self-play, training, and checkpointing.
    """

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        self.device = get_device()

        # Create network
        self.network = create_network(
            num_filters=self.config.num_filters,
            num_blocks=self.config.num_blocks,
            device=self.device,
        )
        print(f"Network parameters: {count_parameters(self.network):,}")

        # Create optimizer
        self.optimizer = Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Create replay buffer
        self.buffer = ReplayBuffer(capacity=self.config.replay_buffer_size)

        # MCTS config
        self.mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            batch_size=self.config.mcts_batch_size,
        )

        # Training state
        self.iteration = 0
        self.total_games = 0
        self.total_steps = 0

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_iteration(self, verbose: bool = True) -> dict:
        """
        Run one training iteration (self-play + training).

        Returns:
            Dictionary of training statistics
        """
        self.iteration += 1
        stats = {"iteration": self.iteration}

        # Self-play phase
        if verbose:
            print(f"\n=== Iteration {self.iteration}: Self-play ===")

        self.network.eval()
        games = generate_games(
            self.network,
            num_games=self.config.games_per_iteration,
            mcts_config=self.mcts_config,
            max_moves=self.config.max_moves,
            temperature_moves=self.config.temperature_moves,
            verbose=verbose,
        )
        self.buffer.add_games(games)
        self.total_games += len(games)

        total_moves = sum(len(g) for g in games)
        stats["games"] = len(games)
        stats["total_moves"] = total_moves
        stats["buffer_size"] = len(self.buffer)

        if verbose:
            print(f"Generated {len(games)} games, {total_moves} positions")
            print(f"Buffer size: {len(self.buffer)}")

        # Training phase
        if verbose:
            print(f"\n=== Iteration {self.iteration}: Training ===")

        self.network.train()
        policy_losses = []
        value_losses = []
        total_losses = []

        for step in range(self.config.training_steps_per_iteration):
            loss_dict = self._train_step()
            policy_losses.append(loss_dict["policy_loss"])
            value_losses.append(loss_dict["value_loss"])
            total_losses.append(loss_dict["total_loss"])
            self.total_steps += 1

            if verbose and (step + 1) % 100 == 0:
                avg_policy = sum(policy_losses[-100:]) / 100
                avg_value = sum(value_losses[-100:]) / 100
                avg_total = sum(total_losses[-100:]) / 100
                print(
                    f"Step {step + 1}: loss={avg_total:.4f} (policy={avg_policy:.4f}, value={avg_value:.4f})"
                )

        stats["avg_policy_loss"] = sum(policy_losses) / len(policy_losses)
        stats["avg_value_loss"] = sum(value_losses) / len(value_losses)
        stats["avg_total_loss"] = sum(total_losses) / len(total_losses)

        # Save checkpoint
        if self.iteration % self.config.save_every == 0:
            self.save_checkpoint()

        return stats

    def _train_step(self) -> dict:
        """Run a single training step."""
        # Sample batch
        states, policies, values = self.buffer.sample(self.config.batch_size)

        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)
        policies_t = torch.from_numpy(policies).to(self.device)
        values_t = torch.from_numpy(values).to(self.device).unsqueeze(1)

        # Forward pass
        policy_logits, value_pred = self.network(states_t)

        # Policy loss: cross-entropy with soft targets (MCTS visit distribution)
        # Formula: -sum(target * log_softmax(logits)) / batch_size
        # This is equivalent to KL divergence up to a constant (target entropy)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.sum(policies_t * log_probs) / policies_t.size(0)

        # Value loss: MSE
        value_loss = F.mse_loss(value_pred, values_t)

        # Total loss
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def train(self, num_iterations: int, verbose: bool = True) -> None:
        """
        Run full training loop.

        Args:
            num_iterations: Number of iterations to run
            verbose: Print progress
        """
        for _ in range(num_iterations):
            stats = self.train_iteration(verbose=verbose)
            if verbose:
                print(f"\nIteration {stats['iteration']} complete:")
                print(f"  Policy loss: {stats['avg_policy_loss']:.4f}")
                print(f"  Value loss: {stats['avg_value_loss']:.4f}")

    def save_checkpoint(self, path: str | None = None) -> str:
        """
        Save training checkpoint.

        Args:
            path: Optional custom path (default: checkpoint_dir/iter_N.pt)

        Returns:
            Path where checkpoint was saved
        """
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir, f"iter_{self.iteration:04d}.pt"
            )

        torch.save(
            {
                "iteration": self.iteration,
                "total_games": self.total_games,
                "total_steps": self.total_steps,
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

        print(f"Saved checkpoint: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.iteration = checkpoint["iteration"]
        self.total_games = checkpoint["total_games"]
        self.total_steps = checkpoint["total_steps"]
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded checkpoint: {path}")
        print(f"  Iteration: {self.iteration}")
        print(f"  Total games: {self.total_games}")
        print(f"  Total steps: {self.total_steps}")


if __name__ == "__main__":
    print("Testing training loop...")

    # Small config for quick test
    config = TrainConfig(
        num_filters=32,
        num_blocks=2,
        mcts_simulations=10,
        games_per_iteration=2,
        training_steps_per_iteration=10,
        batch_size=16,
        checkpoint_dir="models/test",
    )

    trainer = Trainer(config)
    print(f"Device: {trainer.device}")

    # Run one iteration
    stats = trainer.train_iteration(verbose=True)
    print(f"\nStats: {stats}")

    # Test checkpoint save/load
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pt")
        trainer.save_checkpoint(path)

        trainer2 = Trainer(config)
        trainer2.load_checkpoint(path)
        assert trainer2.iteration == trainer.iteration

    print("\nAll training tests passed!")
