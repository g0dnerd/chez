"""
Configuration and hyperparameters for AlphaZero training.
"""

from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""

    num_filters: int = 128  # Filters in conv layers (64-256 typical)
    num_blocks: int = 6  # Residual blocks (4-20 typical)


@dataclass
class MCTSConfig:
    """MCTS hyperparameters."""

    num_simulations: int = 200  # Simulations per move (higher = better quality)
    c_puct: float = 1.5  # Exploration constant
    dirichlet_alpha: float = 0.3  # Noise parameter (0.3 for chess)
    dirichlet_epsilon: float = 0.25  # Noise weight at root
    batch_size: int = 32  # Leaves to evaluate in parallel (GPU batching)


@dataclass
class SelfPlayConfig:
    """Self-play configuration."""

    games_per_iteration: int = 100  # Games to generate per iteration
    temperature_moves: int = 30  # Moves with temperature > 0
    max_moves: int = 200  # Max moves before draw


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    training_steps_per_iteration: int = 1000
    replay_buffer_size: int = 200_000


@dataclass
class AlphaZeroConfig:
    """Combined configuration for full training run."""

    network: NetworkConfig
    mcts: MCTSConfig
    selfplay: SelfPlayConfig
    training: TrainingConfig
    checkpoint_dir: str = "models"
    save_every: int = 1

    @classmethod
    def small(cls) -> "AlphaZeroConfig":
        """Small configuration for testing/debugging."""
        return cls(
            network=NetworkConfig(num_filters=64, num_blocks=4),
            mcts=MCTSConfig(num_simulations=50, batch_size=8),
            selfplay=SelfPlayConfig(games_per_iteration=20, max_moves=150),
            training=TrainingConfig(batch_size=128, training_steps_per_iteration=200),
        )

    @classmethod
    def medium(cls) -> "AlphaZeroConfig":
        """Medium configuration for day-long training runs (~1200 ELO target)."""
        return cls(
            network=NetworkConfig(num_filters=128, num_blocks=6),
            mcts=MCTSConfig(num_simulations=200, batch_size=16),
            selfplay=SelfPlayConfig(games_per_iteration=100, max_moves=200),
            training=TrainingConfig(
                batch_size=512,
                training_steps_per_iteration=1000,
                replay_buffer_size=200_000,
            ),
        )

    @classmethod
    def large(cls) -> "AlphaZeroConfig":
        """Large configuration for extended training runs."""
        return cls(
            network=NetworkConfig(num_filters=256, num_blocks=10),
            mcts=MCTSConfig(num_simulations=400, batch_size=16),
            selfplay=SelfPlayConfig(games_per_iteration=200, max_moves=300),
            training=TrainingConfig(
                batch_size=512,
                training_steps_per_iteration=2000,
                replay_buffer_size=500_000,
            ),
        )
