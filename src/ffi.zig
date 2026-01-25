const game = @import("game.zig");
const movegen = @import("movegen.zig");
const search = @import("search.zig");
const State = @import("State.zig");

// Game state in a C-friendly representation.
// Bitboards become u64s, any non-twos-complement integer types (`u1`, `u6`) are represented
// as their closest twos-complement neighbor.
pub const CState = extern struct {
    pieces: [6]u64, // Bits per piece type
    colors: [2]u64, // Bits per color
    to_move: u8,
    castling_rights: u8,
    en_passant: i8,
    halfmove_clock: u16,
    fullmove_clock: u16,
    in_check: i8,
};

pub const CMove = extern struct {
    start: u8,
    end: u8,
    promotion_piece: u8, // 0 = none, 1 = N, 2 = B, 3 = R, 4 = Q
};

pub const CMoveList = extern struct {
    moves: [256]CMove,
    len: u8,
};

// Initialize the given state to the default position
export fn chez_init_default(c_state: *CState) void {
    State.defaultPosition().toCState(c_state);
}

// Write all legal moves for the state into the move list
export fn chez_legal_moves(c_state: *const CState, c_moves: *CMoveList) void {
    const state: State = .initCState(c_state);
    const moves = movegen.legalMoves(&state, state.to_move);
    return moves.toCMoves(c_moves);
}

// Apply a move to the state
export fn chez_make_move(c_state: *CState, c_move: *const CMove) void {
    const move: game.Move = .initCMove(c_move);
    var state: State = .initCState(c_state);

    // TODO: Maybe remove the additional overhead from tracking undo info
    // since we don't need it without outside of search.
    _ = state.makeMove(move, state.to_move, state.mailbox[move.start].?);

    // Persist changes to C state
    state.toCState(c_state);
}

// Return the game result for the state:
//   0 = ongoing
//   1 = white wins
//   2 = black wins
//   3 = draw
export fn chez_game_result(c_state: *const CState) i32 {
    const state: State = .initCState(c_state);
    const res = search.isGameOverWithHistory(&state, null) orelse return 0;
    return switch (res) {
        .checkmate => |c| @as(i32, c) + 1,
        else => 3,
    };
}

// Return the Zobrist hash for the state
export fn chez_hash(c_state: *const CState) u64 {
    return State.initCState(c_state).zobrist_hash;
}
