"""
ctypes bindings to libchez.so for legal move generation and game state management.
"""

import ctypes
from ctypes import c_uint8, c_uint64, c_int32, c_void_p, c_char_p, POINTER, Structure
from pathlib import Path
from typing import Iterator

# Find and load the shared library
_lib_path = Path(__file__).parent.parent / "zig-out" / "lib" / "libchez.so"
if not _lib_path.exists():
    raise RuntimeError(
        f"libchez.so not found at {_lib_path}. Run 'zig build' first."
    )
_lib = ctypes.CDLL(str(_lib_path))


class CMove(Structure):
    """C-compatible move structure."""
    _fields_ = [
        ("start", c_uint8),
        ("end", c_uint8),
        ("promotion_piece", c_uint8),  # 0=none, 1=N, 2=B, 3=R, 4=Q
    ]

    def __repr__(self) -> str:
        promo = ""
        if self.promotion_piece:
            promo = "nbrq"[self.promotion_piece - 1]
        return f"{_square_name(self.start)}{_square_name(self.end)}{promo}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CMove):
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and self.promotion_piece == other.promotion_piece
        )

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.promotion_piece))


class CMoveList(Structure):
    """C-compatible move list structure."""
    _fields_ = [
        ("moves", CMove * 256),
        ("len", c_uint8),
    ]

    def __iter__(self) -> Iterator[CMove]:
        for i in range(self.len):
            yield self.moves[i]

    def __len__(self) -> int:
        return self.len


# Function signatures
_lib.chez_create_default.argtypes = []
_lib.chez_create_default.restype = c_void_p

_lib.chez_create_fen.argtypes = [c_char_p]
_lib.chez_create_fen.restype = c_void_p

_lib.chez_destroy.argtypes = [c_void_p]
_lib.chez_destroy.restype = None

_lib.chez_clone.argtypes = [c_void_p]
_lib.chez_clone.restype = c_void_p

_lib.chez_legal_moves.argtypes = [c_void_p, POINTER(CMoveList)]
_lib.chez_legal_moves.restype = None

_lib.chez_make_move.argtypes = [c_void_p, POINTER(CMove)]
_lib.chez_make_move.restype = None

_lib.chez_game_result.argtypes = [c_void_p]
_lib.chez_game_result.restype = c_int32

_lib.chez_hash.argtypes = [c_void_p]
_lib.chez_hash.restype = c_uint64

_lib.chez_to_move.argtypes = [c_void_p]
_lib.chez_to_move.restype = c_uint8


def _square_name(sq: int) -> str:
    """Convert square index (0-63) to algebraic notation (a1-h8)."""
    file = sq % 8
    rank = sq // 8
    return chr(ord("a") + file) + str(rank + 1)


def _parse_square(name: str) -> int:
    """Convert algebraic notation (a1-h8) to square index (0-63)."""
    file = ord(name[0]) - ord("a")
    rank = int(name[1]) - 1
    return file + rank * 8


def _parse_move(uci: str) -> CMove:
    """Parse UCI move string (e.g., 'e2e4', 'e7e8q') to CMove."""
    start = _parse_square(uci[0:2])
    end = _parse_square(uci[2:4])
    promo = 0
    if len(uci) == 5:
        promo = "nbrq".index(uci[4].lower()) + 1
    return CMove(start, end, promo)


class GameResult:
    """Game result constants."""
    ONGOING = 0
    WHITE_WINS = 1
    BLACK_WINS = 2
    DRAW = 3


class State:
    """
    Python wrapper for the Chez game state.

    Uses opaque handle to Zig State struct. Memory is managed automatically.
    """

    __slots__ = ("_ptr",)

    def __init__(self, ptr: c_void_p):
        """Initialize with an opaque pointer. Use class methods to create."""
        if not ptr:
            raise ValueError("Null state pointer")
        self._ptr = ptr

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.chez_destroy(self._ptr)

    @classmethod
    def default(cls) -> "State":
        """Create a new game at the starting position."""
        ptr = _lib.chez_create_default()
        if not ptr:
            raise MemoryError("Failed to allocate State")
        return cls(ptr)

    @classmethod
    def from_fen(cls, fen: str) -> "State":
        """Create a new game from a FEN string."""
        ptr = _lib.chez_create_fen(fen.encode("utf-8"))
        if not ptr:
            raise ValueError(f"Invalid FEN: {fen}")
        return cls(ptr)

    def clone(self) -> "State":
        """Create a deep copy of this state."""
        ptr = _lib.chez_clone(self._ptr)
        if not ptr:
            raise MemoryError("Failed to clone State")
        return State(ptr)

    def legal_moves(self) -> CMoveList:
        """Get all legal moves in the current position."""
        moves = CMoveList()
        _lib.chez_legal_moves(self._ptr, ctypes.byref(moves))
        return moves

    def make_move(self, move: CMove | str) -> None:
        """Apply a move to the state. Accepts CMove or UCI string."""
        if isinstance(move, str):
            move = _parse_move(move)
        _lib.chez_make_move(self._ptr, ctypes.byref(move))

    def game_result(self) -> int:
        """
        Get the game result.

        Returns:
            GameResult.ONGOING (0): Game in progress
            GameResult.WHITE_WINS (1): White won by checkmate
            GameResult.BLACK_WINS (2): Black won by checkmate
            GameResult.DRAW (3): Draw (stalemate, 50-move, repetition)
        """
        return _lib.chez_game_result(self._ptr)

    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self.game_result() != GameResult.ONGOING

    def hash(self) -> int:
        """Get the Zobrist hash of the current position."""
        return _lib.chez_hash(self._ptr)

    def to_move(self) -> int:
        """Get whose turn it is (0=white, 1=black)."""
        return _lib.chez_to_move(self._ptr)

    @property
    def white_to_move(self) -> bool:
        """True if it's white's turn."""
        return self.to_move() == 0


def play_random_game(max_moves: int = 500) -> tuple[int, int]:
    """
    Play a random game and return (result, num_moves).

    Useful for testing the bindings.
    """
    import random

    state = State.default()
    num_moves = 0

    while num_moves < max_moves:
        result = state.game_result()
        if result != GameResult.ONGOING:
            return result, num_moves

        moves = state.legal_moves()
        if len(moves) == 0:
            break

        move = random.choice(list(moves))
        state.make_move(move)
        num_moves += 1

    return state.game_result(), num_moves


if __name__ == "__main__":
    # Quick test
    print("Testing bindings...")

    state = State.default()
    moves = state.legal_moves()
    print(f"Starting position has {len(moves)} legal moves")
    assert len(moves) == 20, f"Expected 20, got {len(moves)}"

    # Play e4
    state.make_move("e2e4")
    moves = state.legal_moves()
    print(f"After e4: {len(moves)} legal moves")
    assert len(moves) == 20, f"Expected 20, got {len(moves)}"

    # Test clone
    state2 = state.clone()
    assert state.hash() == state2.hash()

    # Test FEN
    state3 = State.from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert state.hash() == state3.hash()

    # Play a random game
    result, num_moves = play_random_game()
    result_names = ["ongoing", "white wins", "black wins", "draw"]
    print(f"Random game: {result_names[result]} after {num_moves} moves")

    print("All tests passed!")
