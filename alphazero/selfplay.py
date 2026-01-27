"""
Self-play game generation for AlphaZero training.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from .bindings import State, GameResult
from .encoding import StateEncoder, TOTAL_PLANES
from .mcts import MCTS, MCTSConfig
from .network import AlphaZeroNetwork
from .policy import POLICY_SIZE


@dataclass
class GameRecord:
    """Record of a single self-play game."""

    # List of (encoded_state, mcts_policy, result) tuples
    # Result is from the perspective of the player to move at that state
    states: list[NDArray[np.float32]]
    policies: list[NDArray[np.float32]]
    results: list[float]

    def __len__(self) -> int:
        return len(self.states)

    def __iter__(
        self,
    ) -> Iterator[tuple[NDArray[np.float32], NDArray[np.float32], float]]:
        for s, p, r in zip(self.states, self.policies, self.results):
            yield s, p, r


def play_game(
    network: AlphaZeroNetwork,
    mcts_config: MCTSConfig | None = None,
    temperature_moves: int = 30,
    max_moves: int = 512,
) -> GameRecord:
    """
    Play a single self-play game.

    Args:
        network: Neural network for MCTS evaluation
        mcts_config: MCTS configuration
        temperature_moves: Number of moves to use temperature > 0
        max_moves: Maximum moves before declaring draw

    Returns:
        GameRecord with all states, policies, and results
    """
    mcts = MCTS(network, mcts_config)
    encoder = StateEncoder()

    state = State.default()
    states: list[NDArray[np.float32]] = []
    policies: list[NDArray[np.float32]] = []
    players: list[int] = []  # 0 = white, 1 = black

    move_count = 0
    while move_count < max_moves:
        # Check for game over
        result = state.game_result()
        if result != GameResult.ONGOING:
            break

        # Encode current state
        encoded = encoder.encode(state)
        states.append(encoded)
        players.append(state.to_move())

        # Run MCTS (pass encoder so MCTS has access to game history)
        policy, _ = mcts.search(state, encoder=encoder)
        policies.append(policy)

        # Select move (with temperature for first N moves)
        temperature = 1.0 if move_count < temperature_moves else 0.0
        move = mcts.select_move(policy, state, temperature)

        # Make move
        state.make_move(move)
        encoder.push(state)
        move_count += 1

    # Determine game result
    result = state.game_result()
    if result == GameResult.DRAW or move_count >= max_moves:
        game_value = 0.0
    elif result == GameResult.WHITE_WINS:
        game_value = 1.0  # White won
    else:
        game_value = -1.0  # Black won (white lost)

    # Assign results from each player's perspective
    results: list[float] = []
    for player in players:
        if player == 0:  # White
            results.append(game_value)
        else:  # Black
            results.append(-game_value)

    return GameRecord(states=states, policies=policies, results=results)


def generate_games(
    network: AlphaZeroNetwork,
    num_games: int,
    mcts_config: MCTSConfig | None = None,
    max_moves: int = 200,
    temperature_moves: int = 30,
    verbose: bool = True,
) -> list[GameRecord]:
    """
    Generate multiple self-play games.

    Args:
        network: Neural network for MCTS
        num_games: Number of games to generate
        mcts_config: MCTS configuration
        max_moves: Maximum moves per game before declaring draw
        temperature_moves: Number of moves to use temperature > 0
        verbose: Print progress

    Returns:
        List of GameRecords
    """
    games: list[GameRecord] = []

    for i in range(num_games):
        game = play_game(network, mcts_config, temperature_moves=temperature_moves, max_moves=max_moves)
        games.append(game)

        if verbose:
            result_str = {0.0: "draw", 1.0: "white wins", -1.0: "black wins"}
            final_result = game.results[0] if game.results else 0.0
            winner = result_str.get(final_result, "unknown")
            print(f"Game {i + 1}/{num_games}: {len(game)} moves, {winner}")

    return games


if __name__ == "__main__":
    print("Testing self-play...")

    from .network import create_network, get_device

    device = get_device()
    print(f"Using device: {device}")

    # Small network and few simulations for quick test
    network = create_network(num_filters=32, num_blocks=2, device=device)
    config = MCTSConfig(num_simulations=10)

    # Play a single game
    print("\nPlaying test game...")
    game = play_game(network, config)
    print(f"Game length: {len(game)} moves")
    print(
        f"Final result (white perspective): {game.results[0] if game.results else 'N/A'}"
    )

    # Verify data shapes
    if len(game) > 0:
        state, policy, result = next(iter(game))
        print(f"State shape: {state.shape}")
        print(f"Policy shape: {policy.shape}")
        print(f"Policy sum: {policy.sum():.4f}")
        assert state.shape == (TOTAL_PLANES, 8, 8)
        assert policy.shape == (POLICY_SIZE,)

    print("\nAll self-play tests passed!")
