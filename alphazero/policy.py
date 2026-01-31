"""
Policy encoding/decoding for neural network output.

AlphaZero-style policy: 8x8x73 planes (4672 total moves)
- Planes 0-55: Queen-like moves (8 directions × 7 distances)
- Planes 56-63: Knight moves (8 L-shaped jumps)
- Planes 64-72: Underpromotions (3 directions × 3 pieces)

The policy is always from the current player's perspective (board flipped if black).
"""

import numpy as np
from numpy.typing import NDArray

from .bindings import State, CMove

# Policy dimensions
POLICY_PLANES = 73
POLICY_SIZE = 64 * POLICY_PLANES  # 4672

# Direction vectors (file_delta, rank_delta) for queen-like moves
# Order: N, NE, E, SE, S, SW, W, NW
QUEEN_DIRECTIONS = [
    (0, 1),  # N
    (1, 1),  # NE
    (1, 0),  # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1),  # SW
    (-1, 0),  # W
    (-1, 1),  # NW
]

# Knight move offsets (file_delta, rank_delta)
# Consistent ordering for planes 56-63
KNIGHT_MOVES = [
    (1, 2),  # NNE
    (2, 1),  # ENE
    (2, -1),  # ESE
    (1, -2),  # SSE
    (-1, -2),  # SSW
    (-2, -1),  # WSW
    (-2, 1),  # WNW
    (-1, 2),  # NNW
]

# Underpromotion directions (file_delta) - rank is always +1 (forward)
# From the player's perspective: left capture, straight, right capture
UNDERPROMO_DIRECTIONS = [-1, 0, 1]  # Left, straight, right

# Underpromotion pieces: knight=1, bishop=2, rook=3 (queen=4 uses regular move)
UNDERPROMO_PIECES = [1, 2, 3]


def _flip_square(sq: int) -> int:
    """Flip square vertically (for black's perspective)."""
    return sq ^ 56


def _square_to_coords(sq: int) -> tuple[int, int]:
    """Convert square index to (file, rank)."""
    return sq % 8, sq // 8


def _coords_to_square(file: int, rank: int) -> int:
    """Convert (file, rank) to square index."""
    return file + rank * 8


def move_to_policy_index(move: CMove, flip: bool) -> int:
    """
    Convert a move to its policy index (0-4671).

    Args:
        move: The move to encode
        flip: True if encoding for black (flip coordinates)

    Returns:
        Policy index in range [0, 4672)
    """
    start = move.start
    end = move.end
    promo = move.promotion_piece

    # Flip coordinates if black to move
    if flip:
        start = _flip_square(start)
        end = _flip_square(end)

    start_file, start_rank = _square_to_coords(start)
    end_file, end_rank = _square_to_coords(end)

    file_delta = end_file - start_file
    rank_delta = end_rank - start_rank

    # Check for underpromotion (promo piece is knight, bishop, or rook)
    if promo in (1, 2, 3):
        # Underpromotion: planes 64-72
        # Direction: -1 (left), 0 (straight), 1 (right)
        dir_idx = UNDERPROMO_DIRECTIONS.index(file_delta)
        piece_idx = UNDERPROMO_PIECES.index(promo)
        plane = 64 + dir_idx * 3 + piece_idx
        return start * POLICY_PLANES + plane

    # Check for knight move
    if (file_delta, rank_delta) in KNIGHT_MOVES:
        knight_idx = KNIGHT_MOVES.index((file_delta, rank_delta))
        plane = 56 + knight_idx
        return start * POLICY_PLANES + plane

    # Queen-like move (including pawn pushes and queen promotions)
    # Normalize to unit direction
    if file_delta != 0:
        file_dir = file_delta // abs(file_delta)
    else:
        file_dir = 0
    if rank_delta != 0:
        rank_dir = rank_delta // abs(rank_delta)
    else:
        rank_dir = 0

    direction = (file_dir, rank_dir)
    dir_idx = QUEEN_DIRECTIONS.index(direction)
    distance = max(abs(file_delta), abs(rank_delta))

    plane = dir_idx * 7 + (distance - 1)
    return start * POLICY_PLANES + plane


def policy_index_to_move(index: int, flip: bool) -> CMove:
    """
    Convert a policy index back to a move.

    Args:
        index: Policy index in range [0, 4672)
        flip: True if decoding for black (flip coordinates back)

    Returns:
        CMove struct
    """
    start = index // POLICY_PLANES
    plane = index % POLICY_PLANES

    start_file, start_rank = _square_to_coords(start)

    if plane < 56:
        # Queen-like move
        dir_idx = plane // 7
        distance = (plane % 7) + 1
        file_dir, rank_dir = QUEEN_DIRECTIONS[dir_idx]
        end_file = start_file + file_dir * distance
        end_rank = start_rank + rank_dir * distance
        end = _coords_to_square(end_file, end_rank)
        promo = 0

        # Check if this is a pawn promotion (pawn reaching back rank)
        # We can't know for sure without the board, but queen promo uses this encoding
        if end_rank == 7 and rank_dir == 1:
            # Could be queen promotion - caller needs to verify with board state
            promo = 4  # Queen

    elif plane < 64:
        # Knight move
        knight_idx = plane - 56
        file_delta, rank_delta = KNIGHT_MOVES[knight_idx]
        end_file = start_file + file_delta
        end_rank = start_rank + rank_delta
        end = _coords_to_square(end_file, end_rank)
        promo = 0

    else:
        # Underpromotion
        underpromo_idx = plane - 64
        dir_idx = underpromo_idx // 3
        piece_idx = underpromo_idx % 3
        file_delta = UNDERPROMO_DIRECTIONS[dir_idx]
        end_file = start_file + file_delta
        end_rank = start_rank + 1  # Always forward
        end = _coords_to_square(end_file, end_rank)
        promo = UNDERPROMO_PIECES[piece_idx]

    # Flip back if needed
    if flip:
        start = _flip_square(start)
        end = _flip_square(end)

    return CMove(start, end, promo)


def get_legal_move_mask(state: State) -> NDArray[np.float32]:
    """
    Get a mask of legal moves in policy space.

    Args:
        state: Current game state

    Returns:
        Boolean array of shape (4672,) where True = legal move
    """
    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    flip = state.to_move() == 1  # Black to move

    moves = state.legal_moves()
    for move in moves:
        idx = move_to_policy_index(move, flip)
        mask[idx] = 1.0

    return mask


def mask_illegal_moves(
    policy_logits: NDArray[np.float32], state: State
) -> NDArray[np.float32]:
    """
    Mask illegal moves by setting their logits to -inf.

    Args:
        policy_logits: Raw policy output from network, shape (4672,) or (8, 8, 73)
        state: Current game state

    Returns:
        Masked logits with same shape as input
    """
    flat = policy_logits.reshape(-1)
    mask = get_legal_move_mask(state)

    # Set illegal moves to -inf
    masked = np.where(mask > 0, flat, -np.inf)
    return masked.reshape(policy_logits.shape)


def policy_to_moves(
    policy: NDArray[np.float32], state: State
) -> list[tuple[CMove, float]]:
    """
    Convert policy probabilities to a list of (move, probability) pairs.

    Only includes legal moves.

    Args:
        policy: Policy probabilities, shape (4672,) or (8, 8, 73)
        state: Current game state

    Returns:
        List of (CMove, probability) tuples, sorted by probability descending
    """
    flat = policy.reshape(-1)
    flip = state.to_move() == 1

    moves = state.legal_moves()
    result = []

    for move in moves:
        idx = move_to_policy_index(move, flip)
        prob = flat[idx]
        result.append((move, float(prob)))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def sample_move(
    policy_logits: NDArray[np.float32], state: State, temperature: float = 1.0
) -> CMove:
    """
    Sample a move from the policy distribution.

    Args:
        policy_logits: Raw policy output from network
        state: Current game state
        temperature: Temperature for sampling (0 = greedy, higher = more random)

    Returns:
        Sampled move
    """
    masked = mask_illegal_moves(policy_logits, state)
    flat = masked.reshape(-1)

    if temperature == 0:
        # Greedy selection
        idx = int(np.argmax(flat))
    else:
        # Apply temperature and softmax
        scaled = flat / temperature
        # Subtract max for numerical stability
        scaled = scaled - np.max(scaled)
        exp_logits = np.exp(scaled)
        probs = exp_logits / np.sum(exp_logits)

        # Sample
        idx = np.random.choice(len(probs), p=probs)

    flip = state.to_move() == 1
    return policy_index_to_move(idx, flip)


if __name__ == "__main__":
    print("Testing policy encoding...")

    # Test basic move encoding/decoding roundtrip
    state = State.default()

    moves = state.legal_moves()
    print(f"Starting position has {len(moves)} legal moves")

    flip = state.to_move() == 1  # False for white

    # Test roundtrip for all legal moves
    for move in moves:
        idx = move_to_policy_index(move, flip)
        decoded = policy_index_to_move(idx, flip)

        # For non-promotions, should match exactly
        if move.promotion_piece == 0:
            assert move.start == decoded.start, f"Start mismatch: {move} vs {decoded}"
            assert move.end == decoded.end, f"End mismatch: {move} vs {decoded}"

    print("Roundtrip test passed for starting position")

    # Test with some specific moves
    test_cases = [
        ("e2e4", False),  # Pawn push
        ("g1f3", False),  # Knight move
        ("e2e3", False),  # Short pawn push
    ]

    for uci, flip in test_cases:
        from .bindings import _parse_move

        move = _parse_move(uci)
        idx = move_to_policy_index(move, flip)
        decoded = policy_index_to_move(idx, flip)
        print(f"{uci}: index={idx}, decoded={decoded}")
        assert move.start == decoded.start
        assert move.end == decoded.end

    # Test promotion encoding
    promo_state = State.from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
    promo_moves = promo_state.legal_moves()

    print(f"\nPromotion position has {len(promo_moves)} legal moves:")
    for move in promo_moves:
        idx = move_to_policy_index(move, False)
        plane = idx % 73
        print(f"  {move} -> index={idx}, plane={plane}")

    # Test mask generation
    mask = get_legal_move_mask(state)
    assert mask.sum() == len(moves), f"Mask sum {mask.sum()} != {len(moves)} moves"
    print(f"\nMask has {int(mask.sum())} legal moves")

    # Test with black to move
    state.make_move("e2e4")
    moves_black = state.legal_moves()
    mask_black = get_legal_move_mask(state)
    assert mask_black.sum() == len(moves_black)
    print(
        f"After e4, black has {len(moves_black)} legal moves, mask sum={int(mask_black.sum())}"
    )

    print("\nAll policy tests passed!")
