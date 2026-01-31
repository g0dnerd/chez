"""
State encoding for neural network input.

AlphaZero-style encoding: 8x8x119 planes
- Planes 0-111: 8 timesteps × 14 planes (12 piece planes + 2 repetition planes)
- Planes 112-118: 7 constant planes (color, move count, castling, halfmove)

Board is always oriented from current player's perspective (flipped if black to move).
"""

from ctypes import c_void_p, c_float, POINTER
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .bindings import _lib, State

# Number of historical positions to include (including current)
HISTORY_LENGTH = 8

# Planes per position: 6 P1 pieces + 6 P2 pieces + 2 repetition = 14
PLANES_PER_POSITION = 14
PIECE_PLANES_PER_POSITION = 12
REPETITION_PLANES = 2

# Constant planes: color, move count, 4 castling, halfmove = 7
META_PLANES = 7

# Total planes
TOTAL_PLANES = HISTORY_LENGTH * PLANES_PER_POSITION + META_PLANES  # 119

# Set up FFI for encoding functions
_lib.chez_encode_position.argtypes = [c_void_p, POINTER(c_float)]
_lib.chez_encode_position.restype = None

_lib.chez_encode_meta.argtypes = [c_void_p, POINTER(c_float)]
_lib.chez_encode_meta.restype = None


def encode_position(state: State, buffer: NDArray[np.float32]) -> None:
    """
    Encode a single position to 12 planes (pieces only).

    Args:
        state: Game state to encode
        buffer: Pre-allocated numpy array of shape (12, 8, 8) or (768,), will be modified
    """
    assert buffer.dtype == np.float32
    assert buffer.size == PIECE_PLANES_PER_POSITION * 64
    _lib.chez_encode_position(state._ptr, buffer.ctypes.data_as(POINTER(c_float)))


def encode_meta(state: State, buffer: NDArray[np.float32]) -> None:
    """
    Encode constant/meta planes (7 planes).

    Args:
        state: Game state to encode
        buffer: Pre-allocated numpy array of shape (7, 8, 8) or (448,), will be modified
    """
    assert buffer.dtype == np.float32
    assert buffer.size == META_PLANES * 64
    _lib.chez_encode_meta(state._ptr, buffer.ctypes.data_as(POINTER(c_float)))


def count_repetitions(current_hash: int, history: Sequence[int]) -> int:
    """Count how many times current position appears in history."""
    return sum(1 for h in history if h == current_hash)


class StateEncoder:
    """
    Encodes game states for neural network input.

    Maintains position history for proper encoding. Call `push` after each move.
    Caches encoded planes for efficient MCTS encoding.

    Example:
        encoder = StateEncoder()
        state = State.default()

        # Initial encoding
        planes = encoder.encode(state)

        # After making a move
        state.make_move("e2e4")
        encoder.push(state)
        planes = encoder.encode(state)
    """

    def __init__(self):
        # History of (State, hash, encoded_planes) tuples
        # Most recent first. encoded_planes is (12, 8, 8) piece planes only.
        self._history: list[tuple[State, int, NDArray[np.float32]]] = []

    def reset(self) -> None:
        """Clear history (call when starting a new game)."""
        self._history.clear()

    def push(self, state: State) -> None:
        """
        Add current position to history (call after each move).

        Encodes immediately and caches for efficient later use.

        Args:
            state: Current state (will be cloned for history)
        """
        # Encode position immediately
        piece_planes = np.zeros((PIECE_PLANES_PER_POSITION, 8, 8), dtype=np.float32)
        encode_position(state, piece_planes.reshape(-1))

        self._history.insert(0, (state.clone(), state.hash(), piece_planes))

        # Keep only HISTORY_LENGTH - 1 previous positions
        # (current position is passed to encode() separately)
        if len(self._history) >= HISTORY_LENGTH:
            while len(self._history) >= HISTORY_LENGTH:
                self._history.pop()
                # State.__del__ handles cleanup

    def encode(self, state: State) -> NDArray[np.float32]:
        """
        Encode current state with history into 119 planes.

        Args:
            state: Current game state

        Returns:
            numpy array of shape (119, 8, 8)
        """
        planes = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)

        # Collect all hashes for repetition counting
        current_hash = state.hash()
        all_hashes = [h for _, h, _ in self._history]

        # Encode current position (timestep 0)
        piece_buffer = planes[0:PIECE_PLANES_PER_POSITION].reshape(-1)
        encode_position(state, piece_buffer)

        # Repetition planes for current position
        rep_count = count_repetitions(current_hash, all_hashes)
        if rep_count >= 1:
            planes[12] = 1.0
        if rep_count >= 2:
            planes[13] = 1.0

        # Encode historical positions (timesteps 1-7) using cached planes
        for t, (_, hist_hash, cached_planes) in enumerate(
            self._history[: HISTORY_LENGTH - 1], start=1
        ):
            base = t * PLANES_PER_POSITION

            # Copy cached piece planes
            planes[base : base + PIECE_PLANES_PER_POSITION] = cached_planes

            # Repetition planes (count positions older than this one)
            older_hashes = [h for _, h, _ in self._history[t:]]
            rep_count = count_repetitions(hist_hash, older_hashes)
            if rep_count >= 1:
                planes[base + 12] = 1.0
            if rep_count >= 2:
                planes[base + 13] = 1.0

        # Encode meta planes (based on current state)
        meta_buffer = planes[HISTORY_LENGTH * PLANES_PER_POSITION :].reshape(-1)
        encode_meta(state, meta_buffer)

        return planes

    def get_history_hashes(self) -> list[int]:
        """Get list of hashes from history (most recent first)."""
        return [h for _, h, _ in self._history]

    def get_history_planes(self) -> list[NDArray[np.float32]]:
        """Get list of cached piece planes from history (most recent first)."""
        return [p for _, _, p in self._history]


class MCTSEncoder:
    """
    Efficient encoder for MCTS tree traversal.

    Uses a stack-based approach that can push/pop positions during make/unmake
    traversal, avoiding redundant re-encoding of the path to root.

    Example:
        # Create from game encoder's history
        mcts_enc = MCTSEncoder.from_game_encoder(game_encoder)

        # During tree traversal
        mcts_enc.push(state)  # After make_move
        planes = mcts_enc.encode(state)
        mcts_enc.pop()  # Before unmake_move
    """

    def __init__(self):
        # Base history from game (fixed during search)
        self._base_hashes: list[int] = []
        self._base_planes: list[NDArray[np.float32]] = []

        # Stack for MCTS traversal (push/pop during make/unmake)
        self._stack_hashes: list[int] = []
        self._stack_planes: list[NDArray[np.float32]] = []

    @classmethod
    def from_game_encoder(cls, encoder: StateEncoder | None) -> "MCTSEncoder":
        """Create MCTSEncoder initialized with game history."""
        mcts_enc = cls()
        if encoder is not None:
            mcts_enc._base_hashes = encoder.get_history_hashes()
            mcts_enc._base_planes = encoder.get_history_planes()
        return mcts_enc

    def push(self, state: State) -> None:
        """
        Push current position onto the MCTS stack.

        Call this after applying a move during tree traversal.
        """
        piece_planes = np.zeros((PIECE_PLANES_PER_POSITION, 8, 8), dtype=np.float32)
        encode_position(state, piece_planes.reshape(-1))
        self._stack_hashes.append(state.hash())
        self._stack_planes.append(piece_planes)

    def pop(self) -> None:
        """
        Pop the last position from the MCTS stack.

        Call this before unmaking a move during tree traversal.
        """
        if self._stack_hashes:
            self._stack_hashes.pop()
            self._stack_planes.pop()

    def clear_stack(self) -> None:
        """Clear the MCTS traversal stack (but keep base history)."""
        self._stack_hashes.clear()
        self._stack_planes.clear()

    def encode(self, state: State) -> NDArray[np.float32]:
        """
        Encode the current state using combined history.

        The history is: MCTS stack (most recent) + base history (older).
        """
        planes = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)

        # Combined history: stack (most recent first) + base
        combined_hashes = list(reversed(self._stack_hashes)) + self._base_hashes
        combined_planes = list(reversed(self._stack_planes)) + self._base_planes

        # Encode current position (timestep 0)
        current_hash = state.hash()
        piece_buffer = planes[0:PIECE_PLANES_PER_POSITION].reshape(-1)
        encode_position(state, piece_buffer)

        # Repetition planes for current position
        rep_count = count_repetitions(current_hash, combined_hashes)
        if rep_count >= 1:
            planes[12] = 1.0
        if rep_count >= 2:
            planes[13] = 1.0

        # Encode historical positions (timesteps 1-7)
        for t in range(1, HISTORY_LENGTH):
            idx = t - 1  # Index into combined history
            if idx >= len(combined_planes):
                break

            base = t * PLANES_PER_POSITION
            planes[base : base + PIECE_PLANES_PER_POSITION] = combined_planes[idx]

            # Repetition planes
            older_hashes = (
                combined_hashes[idx + 1 :] if idx + 1 < len(combined_hashes) else []
            )
            rep_count = count_repetitions(combined_hashes[idx], older_hashes)
            if rep_count >= 1:
                planes[base + 12] = 1.0
            if rep_count >= 2:
                planes[base + 13] = 1.0

        # Encode meta planes
        meta_buffer = planes[HISTORY_LENGTH * PLANES_PER_POSITION :].reshape(-1)
        encode_meta(state, meta_buffer)

        return planes


def encode_single(state: State) -> NDArray[np.float32]:
    """
    Convenience function to encode a single position without history.

    Useful for testing. For actual training, use StateEncoder.

    Args:
        state: Game state to encode

    Returns:
        numpy array of shape (119, 8, 8)
    """
    encoder = StateEncoder()
    return encoder.encode(state)


if __name__ == "__main__":
    # Quick test
    print("Testing encoding...")

    state = State.default()
    encoder = StateEncoder()

    # Encode starting position
    planes = encoder.encode(state)
    print(f"Encoded shape: {planes.shape}")
    assert planes.shape == (119, 8, 8)

    # Check that white pawns are on rank 2 (indices 8-15 in plane 0)
    pawn_plane = planes[0]  # P1 pawns (white)
    assert pawn_plane[1, :].sum() == 8, "Expected 8 white pawns on rank 2"

    # Check that black pawns are on rank 7 from white's perspective
    opp_pawn_plane = planes[6]  # P2 pawns (black)
    assert opp_pawn_plane[6, :].sum() == 8, "Expected 8 black pawns on rank 7"

    # Play some moves
    state.make_move("e2e4")
    encoder.push(state)
    planes = encoder.encode(state)

    # Now it's black's turn - board should be flipped
    # Black's pawns (P1 from black's perspective) should be on rank 2 of the encoded board
    pawn_plane = planes[0]  # P1 pawns (now black)
    assert pawn_plane[1, :].sum() == 8, "Expected 8 black pawns on rank 2 (flipped)"

    state.make_move("e7e5")
    encoder.push(state)

    state.make_move("g1f3")
    encoder.push(state)

    planes = encoder.encode(state)

    print("Basic encoding tests passed!")

    # Test MCTSEncoder
    print("\nTesting MCTSEncoder...")

    mcts_enc = MCTSEncoder.from_game_encoder(encoder)

    # Simulate MCTS traversal
    state2 = state.clone()
    state2.make_move("b8c6")
    mcts_enc.push(state2)

    planes_mcts = mcts_enc.encode(state2)
    assert planes_mcts.shape == (119, 8, 8)

    # Pop and verify we can re-encode
    mcts_enc.pop()
    state2.make_move("f1b5")
    mcts_enc.push(state2)
    planes_mcts2 = mcts_enc.encode(state2)
    assert planes_mcts2.shape == (119, 8, 8)

    print("MCTSEncoder tests passed!")

    # Test that we handle all 8 timesteps
    encoder2 = StateEncoder()
    state2 = State.default()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
    for m in moves:
        state2.make_move(m)
        encoder2.push(state2)

    planes2 = encoder2.encode(state2)
    print(f"After {len(moves)} moves, encoded shape: {planes2.shape}")

    # Verify history is being used (piece planes should differ between timesteps)
    t0_pieces = planes2[0:12].sum()
    t1_pieces = planes2[14:26].sum()
    assert t0_pieces > 0 and t1_pieces > 0, "History should have piece data"

    print("\nAll encoding tests passed!")
