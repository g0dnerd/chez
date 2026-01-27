"""
Replay buffer for storing self-play training data.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from .selfplay import GameRecord
from .encoding import TOTAL_PLANES
from .policy import POLICY_SIZE


@dataclass
class TrainingExample:
    """Single training example."""
    state: NDArray[np.float32]   # (119, 8, 8)
    policy: NDArray[np.float32]  # (4672,)
    value: float                  # [-1, 1]


class ReplayBuffer:
    """
    Replay buffer storing training examples from self-play games.

    Stores individual (state, policy, value) tuples with a maximum capacity.
    Oldest examples are discarded when capacity is exceeded.
    """

    def __init__(self, capacity: int = 100_000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of examples to store
        """
        self.capacity = capacity
        self._buffer: deque[TrainingExample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add_game(self, game: GameRecord) -> None:
        """
        Add all positions from a game to the buffer.

        Args:
            game: Completed game record
        """
        for state, policy, result in game:
            example = TrainingExample(state=state, policy=policy, value=result)
            self._buffer.append(example)

    def add_games(self, games: list[GameRecord]) -> None:
        """
        Add multiple games to the buffer.

        Args:
            games: List of game records
        """
        for game in games:
            self.add_game(game)

    def sample(self, batch_size: int) -> tuple[NDArray, NDArray, NDArray]:
        """
        Sample a random batch from the buffer.

        Args:
            batch_size: Number of examples to sample

        Returns:
            states: (batch, 119, 8, 8)
            policies: (batch, 4672)
            values: (batch,)
        """
        if batch_size > len(self._buffer):
            batch_size = len(self._buffer)

        examples = random.sample(list(self._buffer), batch_size)

        states = np.stack([e.state for e in examples])
        policies = np.stack([e.policy for e in examples])
        values = np.array([e.value for e in examples], dtype=np.float32)

        return states, policies, values

    def clear(self) -> None:
        """Clear all examples from the buffer."""
        self._buffer.clear()

    def save(self, path: str) -> None:
        """
        Save buffer to disk.

        Args:
            path: File path (should end in .npz)
        """
        states = np.stack([e.state for e in self._buffer])
        policies = np.stack([e.policy for e in self._buffer])
        values = np.array([e.value for e in self._buffer], dtype=np.float32)

        np.savez_compressed(path, states=states, policies=policies, values=values)

    def load(self, path: str) -> None:
        """
        Load buffer from disk.

        Args:
            path: File path to load from
        """
        data = np.load(path)
        states = data["states"]
        policies = data["policies"]
        values = data["values"]

        self._buffer.clear()
        for i in range(len(states)):
            example = TrainingExample(
                state=states[i],
                policy=policies[i],
                value=values[i]
            )
            self._buffer.append(example)


if __name__ == "__main__":
    print("Testing replay buffer...")

    # Create dummy game data
    from .encoding import TOTAL_PLANES
    from .policy import POLICY_SIZE

    dummy_games = []
    for _ in range(3):
        n_moves = random.randint(10, 20)
        game = GameRecord(
            states=[np.random.randn(TOTAL_PLANES, 8, 8).astype(np.float32) for _ in range(n_moves)],
            policies=[np.random.rand(POLICY_SIZE).astype(np.float32) for _ in range(n_moves)],
            results=[random.choice([-1.0, 0.0, 1.0])] * n_moves
        )
        dummy_games.append(game)

    # Test buffer operations
    buffer = ReplayBuffer(capacity=1000)
    print(f"Empty buffer size: {len(buffer)}")

    buffer.add_games(dummy_games)
    total_positions = sum(len(g) for g in dummy_games)
    print(f"After adding {len(dummy_games)} games: {len(buffer)} positions")
    assert len(buffer) == total_positions

    # Test sampling
    states, policies, values = buffer.sample(8)
    print(f"Sample shapes: states={states.shape}, policies={policies.shape}, values={values.shape}")
    assert states.shape == (8, TOTAL_PLANES, 8, 8)
    assert policies.shape == (8, POLICY_SIZE)
    assert values.shape == (8,)

    # Test save/load
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "buffer.npz")
        buffer.save(path)
        print(f"Saved buffer to {path}")

        buffer2 = ReplayBuffer()
        buffer2.load(path)
        print(f"Loaded buffer: {len(buffer2)} positions")
        assert len(buffer2) == len(buffer)

    print("\nAll replay buffer tests passed!")
