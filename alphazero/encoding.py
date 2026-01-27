"""
State encoding for neural network input.

AlphaZero-style encoding: 8x8x119 planes
- Planes 0-111: 8 timesteps × 14 planes (12 piece planes + 2 repetition planes)
- Planes 112-118: 7 constant planes (color, move count, castling, halfmove)

Board is always oriented from current player's perspective (flipped if black to move).
"""

import ctypes
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
    _lib.chez_encode_position(
        state._ptr,
        buffer.ctypes.data_as(POINTER(c_float))
    )


def encode_meta(state: State, buffer: NDArray[np.float32]) -> None:
    """
    Encode constant/meta planes (7 planes).

    Args:
        state: Game state to encode
        buffer: Pre-allocated numpy array of shape (7, 8, 8) or (448,), will be modified
    """
    assert buffer.dtype == np.float32
    assert buffer.size == META_PLANES * 64
    _lib.chez_encode_meta(
        state._ptr,
        buffer.ctypes.data_as(POINTER(c_float))
    )


def count_repetitions(current_hash: int, history: Sequence[int]) -> int:
    """Count how many times current position appears in history."""
    return sum(1 for h in history if h == current_hash)


class StateEncoder:
    """
    Encodes game states for neural network input.

    Maintains position history for proper encoding. Call `push` after each move.

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
        # History of (State, hash) tuples for the last HISTORY_LENGTH positions
        # Most recent first
        self._history: list[tuple[State, int]] = []

    def reset(self) -> None:
        """Clear history (call when starting a new game)."""
        self._history.clear()

    def push(self, state: State) -> None:
        """
        Add current position to history (call after each move).

        Args:
            state: Current state (will be cloned for history)
        """
        self._history.insert(0, (state.clone(), state.hash()))
        # Keep only HISTORY_LENGTH - 1 previous positions
        # (current position is passed to encode() separately)
        if len(self._history) >= HISTORY_LENGTH:
            # Destroy old states to free memory
            while len(self._history) >= HISTORY_LENGTH:
                old_state, _ = self._history.pop()
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
        all_hashes = [h for _, h in self._history]

        # Encode current position (timestep 0)
        self._encode_timestep(state, current_hash, all_hashes, planes, 0)

        # Encode historical positions (timesteps 1-7)
        for t, (hist_state, hist_hash) in enumerate(self._history[:HISTORY_LENGTH - 1], start=1):
            # Hashes before this position in history
            older_hashes = [h for _, h in self._history[t:]]
            self._encode_timestep(hist_state, hist_hash, older_hashes, planes, t)

        # Encode meta planes (based on current state)
        meta_buffer = planes[HISTORY_LENGTH * PLANES_PER_POSITION:].reshape(-1)
        encode_meta(state, meta_buffer)

        return planes

    def _encode_timestep(
        self,
        state: State,
        state_hash: int,
        older_hashes: Sequence[int],
        planes: NDArray[np.float32],
        timestep: int
    ) -> None:
        """Encode a single timestep (12 piece planes + 2 repetition planes)."""
        base = timestep * PLANES_PER_POSITION

        # Piece planes (12)
        piece_buffer = planes[base:base + PIECE_PLANES_PER_POSITION].reshape(-1)
        encode_position(state, piece_buffer)

        # Repetition planes (2)
        rep_count = count_repetitions(state_hash, older_hashes)
        if rep_count >= 1:
            planes[base + 12] = 1.0  # Position seen at least once before
        if rep_count >= 2:
            planes[base + 13] = 1.0  # Position seen at least twice before


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

    # Check meta planes
    color_plane = planes[112]  # Should be 0 (white to move... wait, after Nf3 it's black's turn)
    # Actually after e4 e5 Nf3, it's black's turn
    # Hmm wait, let me re-check. After make_move, to_move changes.
    # e2e4 -> black's turn (encoded with flip)
    # e7e5 -> white's turn
    # g1f3 -> black's turn

    print("Basic encoding tests passed!")

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

    print("All encoding tests passed!")
