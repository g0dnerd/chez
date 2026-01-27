"""
Monte Carlo Tree Search for AlphaZero.

Uses the neural network for position evaluation instead of random rollouts.
Supports batched inference for efficient GPU utilization.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from .bindings import State, CMove, GameResult
from .encoding import StateEncoder, TOTAL_PLANES
from .network import AlphaZeroNetwork
from .policy import move_to_policy_index, POLICY_SIZE


@dataclass
class MCTSConfig:
    """MCTS hyperparameters."""
    num_simulations: int = 100
    c_puct: float = 1.5  # Exploration constant
    dirichlet_alpha: float = 0.3  # Noise for root exploration
    dirichlet_epsilon: float = 0.25  # Weight of noise at root
    temperature: float = 1.0  # For move selection (0 = greedy)
    batch_size: int = 8  # Number of leaves to evaluate in parallel


@dataclass
class Node:
    """MCTS tree node."""
    state: State
    parent: Optional["Node"] = None
    move: Optional[CMove] = None  # Move that led to this node
    children: dict[int, "Node"] = field(default_factory=dict)  # policy_idx -> Node

    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # Policy prior from network

    # Virtual loss for parallel MCTS (temporary penalty during batch selection)
    virtual_loss: int = 0

    # Cached values
    _legal_moves: Optional[list[CMove]] = field(default=None, repr=False)
    _is_terminal: Optional[bool] = field(default=None, repr=False)
    _terminal_value: Optional[float] = field(default=None, repr=False)

    @property
    def value(self) -> float:
        """Average value (Q) of this node, including virtual loss."""
        total_visits = self.visit_count + self.virtual_loss
        if total_visits == 0:
            return 0.0
        # Virtual losses count as losses (-1)
        return (self.value_sum - self.virtual_loss) / total_visits

    @property
    def total_visits(self) -> int:
        """Visit count including virtual losses."""
        return self.visit_count + self.virtual_loss

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded (children created)."""
        return len(self.children) > 0

    def legal_moves(self) -> list[CMove]:
        """Get legal moves (cached)."""
        if self._legal_moves is None:
            self._legal_moves = list(self.state.legal_moves())
        return self._legal_moves

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        if self._is_terminal is None:
            result = self.state.game_result()
            self._is_terminal = result != GameResult.ONGOING
            if self._is_terminal:
                # Cache terminal value from current player's perspective
                if result == GameResult.DRAW:
                    self._terminal_value = 0.0
                else:
                    # Win for white (1) or black (2)
                    winner = result - 1  # 0 = white, 1 = black
                    current = self.state.to_move()
                    self._terminal_value = 1.0 if winner == current else -1.0
        return self._is_terminal

    def terminal_value(self) -> float:
        """Get terminal value (only valid if is_terminal)."""
        assert self._terminal_value is not None
        return self._terminal_value


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Supports batched inference for efficient GPU utilization. Multiple leaves
    are collected using virtual loss, then evaluated in a single forward pass.

    Usage:
        mcts = MCTS(network, config)
        policy, value = mcts.search(state)
        move = mcts.select_move(policy, temperature=1.0)
    """

    def __init__(self, network: AlphaZeroNetwork, config: MCTSConfig | None = None):
        self.network = network
        self.config = config or MCTSConfig()
        self.device = next(network.parameters()).device

    def search(
        self,
        state: State,
        encoder: StateEncoder | None = None
    ) -> tuple[NDArray[np.float32], float]:
        """
        Run MCTS from the given state.

        Args:
            state: Current game state
            encoder: Optional encoder with game history for proper position encoding.
                     If None, positions are encoded without history context.

        Returns:
            policy: Visit count distribution over moves, shape (4672,)
            value: Estimated value of the position
        """
        # Store encoder for use in expansion (contains pre-root game history)
        self._game_encoder = encoder

        # Create root node
        root = Node(state=state.clone())

        # Expand root (single evaluation)
        self._expand_single(root)

        # Add Dirichlet noise to root priors for exploration
        self._add_dirichlet_noise(root)

        # Run simulations in batches
        batch_size = self.config.batch_size
        remaining_sims = self.config.num_simulations

        while remaining_sims > 0:
            # Determine batch size for this iteration
            current_batch = min(batch_size, remaining_sims)
            remaining_sims -= current_batch

            # Collect leaves using virtual loss
            leaves_and_paths = []
            terminal_results = []

            for _ in range(current_batch):
                node = root
                search_path = [node]

                # Selection: traverse tree to leaf, applying virtual loss
                while node.is_expanded and not node.is_terminal():
                    node = self._select_child(node)
                    node.virtual_loss += 1
                    search_path.append(node)

                if node.is_terminal():
                    # Terminal node - backup immediately
                    terminal_results.append((search_path, node.terminal_value()))
                else:
                    leaves_and_paths.append((node, search_path))

            # Batch evaluate non-terminal leaves
            if leaves_and_paths:
                self._expand_batch(leaves_and_paths)

            # Backup terminal results
            for search_path, value in terminal_results:
                self._backup(search_path, value)

        # Return policy (visit counts) and value
        policy = self._get_policy(root)
        value = root.value

        return policy, value

    def select_move(
        self,
        policy: NDArray[np.float32],
        state: State,
        temperature: float | None = None
    ) -> CMove:
        """
        Select a move based on the MCTS policy.

        Args:
            policy: Visit count distribution from search()
            state: Current game state (for move validation)
            temperature: Sampling temperature (0 = greedy, None = use config)

        Returns:
            Selected move
        """
        if temperature is None:
            temperature = self.config.temperature

        flip = state.to_move() == 1

        # Get legal move indices
        legal_moves = list(state.legal_moves())
        legal_indices = [move_to_policy_index(m, flip) for m in legal_moves]

        # Get policy values for legal moves
        legal_probs = policy[legal_indices]

        if temperature == 0:
            # Greedy selection
            best_idx = np.argmax(legal_probs)
        else:
            # Sample with temperature
            scaled = legal_probs ** (1.0 / temperature)
            probs = scaled / scaled.sum()
            best_idx = np.random.choice(len(legal_moves), p=probs)

        return legal_moves[best_idx]

    def _encode_node(self, node: Node) -> NDArray[np.float32]:
        """
        Encode a node's state with proper history.

        Returns:
            Encoded planes of shape (TOTAL_PLANES, 8, 8)
        """
        # Build history by walking up the tree from this node to root
        path_states: list[State] = []
        current = node.parent
        while current is not None:
            path_states.append(current.state)
            current = current.parent

        # Create encoder with proper history
        encoder = StateEncoder()

        # First, copy the game history from before MCTS search started
        if self._game_encoder is not None:
            for hist_state, _ in reversed(self._game_encoder._history):
                encoder.push(hist_state)

        # Then add the in-tree path (root to parent of current node)
        for state in reversed(path_states):
            encoder.push(state)

        # Encode the current node's state with full history
        return encoder.encode(node.state)

    def _expand_single(self, node: Node) -> float:
        """
        Expand a single node: evaluate with network, create children.

        Returns the value estimate for this position.
        """
        planes = self._encode_node(node)

        # Evaluate with network
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        policy_probs, value = self.network.predict(x)
        policy_probs = policy_probs[0].cpu().numpy()
        value = value[0, 0].item()

        # Create children for legal moves
        self._create_children(node, policy_probs)

        return value

    def _expand_batch(self, leaves_and_paths: list[tuple[Node, list[Node]]]) -> None:
        """
        Expand multiple nodes with batched network evaluation.

        Args:
            leaves_and_paths: List of (leaf_node, search_path) tuples
        """
        if not leaves_and_paths:
            return

        # Encode all leaves
        batch_planes = np.zeros(
            (len(leaves_and_paths), TOTAL_PLANES, 8, 8),
            dtype=np.float32
        )
        for i, (node, _) in enumerate(leaves_and_paths):
            batch_planes[i] = self._encode_node(node)

        # Batch evaluate with network
        x = torch.from_numpy(batch_planes).to(self.device)
        policy_probs_batch, values_batch = self.network.predict(x)
        policy_probs_batch = policy_probs_batch.cpu().numpy()
        values_batch = values_batch.cpu().numpy()

        # Expand each node and backup
        for i, (node, search_path) in enumerate(leaves_and_paths):
            policy_probs = policy_probs_batch[i]
            value = values_batch[i, 0]

            # Create children
            self._create_children(node, policy_probs)

            # Remove virtual loss and backup
            self._backup(search_path, value)

    def _create_children(self, node: Node, policy_probs: NDArray[np.float32]) -> None:
        """Create child nodes for all legal moves."""
        flip = node.state.to_move() == 1
        for move in node.legal_moves():
            idx = move_to_policy_index(move, flip)
            prior = policy_probs[idx]

            # Create child state
            child_state = node.state.clone()
            child_state.make_move(move)

            child = Node(
                state=child_state,
                parent=node,
                move=move,
                prior=prior
            )
            node.children[idx] = child

    def _select_child(self, node: Node) -> Node:
        """Select best child using PUCT formula with virtual loss."""
        best_score = -float("inf")
        best_child = None

        # Use total visits (including virtual) for exploration calculation
        sqrt_parent_visits = math.sqrt(node.total_visits)

        for child in node.children.values():
            # PUCT score: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
            # Virtual loss makes Q more negative and N_child larger,
            # discouraging selection of nodes being evaluated in parallel
            q_value = -child.value  # Negate because child is opponent's perspective
            exploration = (
                self.config.c_puct
                * child.prior
                * sqrt_parent_visits
                / (1 + child.total_visits)
            )
            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child

    def _backup(self, search_path: list[Node], value: float) -> None:
        """Backup value through the search path and remove virtual loss."""
        for node in reversed(search_path):
            # Remove virtual loss if present (skip root which has no virtual loss)
            if node.virtual_loss > 0:
                node.virtual_loss -= 1

            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent's perspective

    def _add_dirichlet_noise(self, node: Node) -> None:
        """Add Dirichlet noise to root priors for exploration."""
        if not node.children:
            return

        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(node.children)
        )
        eps = self.config.dirichlet_epsilon

        for i, child in enumerate(node.children.values()):
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _get_policy(self, root: Node) -> NDArray[np.float32]:
        """
        Get policy (visit count distribution) from root.

        Returns:
            Policy array of shape (4672,) with visit counts normalized
        """
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)

        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits == 0:
            return policy

        for idx, child in root.children.items():
            policy[idx] = child.visit_count / total_visits

        return policy


if __name__ == "__main__":
    print("Testing MCTS with batched inference...")

    from .network import create_network, get_device
    import time

    device = get_device()
    print(f"Using device: {device}")

    # Create small network for testing
    network = create_network(num_filters=32, num_blocks=2, device=device)

    # Test with different batch sizes
    for batch_size in [1, 4, 8, 16]:
        config = MCTSConfig(num_simulations=64, batch_size=batch_size)
        mcts = MCTS(network, config)

        state = State.default()

        # Warm up
        mcts.search(state)

        # Time multiple searches
        start = time.perf_counter()
        n_searches = 5
        for _ in range(n_searches):
            policy, value = mcts.search(state)
        elapsed = time.perf_counter() - start

        print(f"Batch size {batch_size:2d}: {elapsed/n_searches:.3f}s per search")

    # Verify correctness
    print("\nVerifying correctness...")
    config = MCTSConfig(num_simulations=50, batch_size=8)
    mcts = MCTS(network, config)

    state = State.default()
    policy, value = mcts.search(state)

    print(f"Root value: {value:.4f}")
    print(f"Policy sum: {policy.sum():.4f}")
    print(f"Non-zero policy entries: {(policy > 0).sum()}")

    # Select move
    move = mcts.select_move(policy, state, temperature=1.0)
    print(f"Selected move: {move}")

    # Verify move is legal
    legal = list(state.legal_moves())
    assert any(m.start == move.start and m.end == move.end for m in legal)

    # Play a few moves with history
    print("\nPlaying a short game with encoder history...")
    from .encoding import StateEncoder

    encoder = StateEncoder()
    state = State.default()
    for i in range(6):
        policy, value = mcts.search(state, encoder=encoder)
        move = mcts.select_move(policy, state, temperature=0.5)
        print(f"Move {i+1}: {move} (value: {value:.3f})")
        state.make_move(move)
        encoder.push(state)

        if state.is_game_over():
            print("Game over!")
            break

    print("\nAll MCTS tests passed!")
