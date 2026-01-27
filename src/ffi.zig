const std = @import("std");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const search = @import("search.zig");
const State = @import("State.zig");

const allocator = std.heap.page_allocator;

pub const CMove = extern struct {
    start: u8,
    end: u8,
    promotion_piece: u8, // 0 = none, 1 = N, 2 = B, 3 = R, 4 = Q
};

pub const CMoveList = extern struct {
    moves: [256]CMove,
    len: u8,
};

// Create a new game state initialized to the default position.
// Returns an opaque handle to the state. Caller must call chez_destroy() when done.
export fn chez_create_default() ?*State {
    const state = allocator.create(State) catch return null;
    state.* = State.defaultPosition();
    return state;
}

// Create a new game state from a FEN string.
// Returns null if the FEN is invalid. Caller must call chez_destroy() when done.
export fn chez_create_fen(fen: [*:0]const u8) ?*State {
    const state = allocator.create(State) catch return null;
    const fen_slice = std.mem.span(fen);
    state.* = State.fromFen(fen_slice) catch {
        allocator.destroy(state);
        return null;
    };
    return state;
}

// Destroy a game state and free its memory.
export fn chez_destroy(state: *State) void {
    allocator.destroy(state);
}

// Create a copy of the game state.
// Returns an opaque handle to the new state. Caller must call chez_destroy() when done.
export fn chez_clone(state: *const State) ?*State {
    const new_state = allocator.create(State) catch return null;
    new_state.* = state.*;
    return new_state;
}

// Write all legal moves for the state into the move list.
export fn chez_legal_moves(state: *const State, c_moves: *CMoveList) void {
    var moves = movegen.legalMoves(state, state.to_move);
    moves.toCMoves(c_moves);
}

// Apply a move to the state.
export fn chez_make_move(state: *State, c_move: *const CMove) void {
    const move: game.Move = .initCMove(c_move);
    _ = state.makeMove(move, state.to_move, state.mailbox[move.start].?);
}

// Return the game result for the state:
//   0 = ongoing
//   1 = white wins
//   2 = black wins
//   3 = draw
export fn chez_game_result(state: *const State) i32 {
    const res = search.isGameOverWithHistory(state, null) orelse return 0;
    return switch (res) {
        .checkmate => |c| @as(i32, c) + 1,
        else => 3,
    };
}

// Return the Zobrist hash for the state.
export fn chez_hash(state: *const State) u64 {
    return state.zobrist_hash;
}

// Return whose turn it is (0 = white, 1 = black).
export fn chez_to_move(state: *const State) u8 {
    return state.to_move;
}

// ============================================================================
// Neural Network Encoding
// ============================================================================

const Colors = game.Colors;
const Castling = game.Castling;

// Encode position to 12 planes (6 P1 pieces + 6 P2 pieces).
// P1 = current player to move, P2 = opponent.
// Board is flipped if black to move (current player's pieces always at bottom).
// Buffer must be pre-zeroed, size = 12 * 64 = 768 floats.
export fn chez_encode_position(state: *const State, buffer: [*]f32) void {
    const flip = state.to_move == Colors.black;
    const p1 = state.to_move;
    const p2 = ~p1;

    // P1 pieces (planes 0-5: pawn, knight, bishop, rook, queen, king)
    inline for (0..6) |piece| {
        const bb = state.pieces[piece].bitAnd(state.colors[p1]);
        bitboardToPlane(bb.bits, buffer + piece * 64, flip);
    }

    // P2 pieces (planes 6-11)
    inline for (0..6) |piece| {
        const bb = state.pieces[piece].bitAnd(state.colors[p2]);
        bitboardToPlane(bb.bits, buffer + (6 + piece) * 64, flip);
    }
}

// Encode constant/meta planes (7 planes: color, move count, 4x castling, halfmove).
// Buffer must be pre-zeroed, size = 7 * 64 = 448 floats.
export fn chez_encode_meta(state: *const State, buffer: [*]f32) void {
    const p1 = state.to_move;
    const p2 = ~p1;

    // Plane 0: Color (all 1s if black to move)
    if (state.to_move == Colors.black) {
        fillPlane(buffer, 1.0);
    }

    // Plane 1: Move count (normalized by 200 to keep roughly in [0, 1])
    const move_count_normalized: f32 = @as(f32, @floatFromInt(state.fullmove_clock)) / 200.0;
    fillPlane(buffer + 64, move_count_normalized);

    // Planes 2-3: P1 castling rights (kingside, queenside)
    const p1_kingside = if (p1 == Colors.white)
        state.castling_rights & Castling.white_kingside != 0
    else
        state.castling_rights & Castling.black_kingside != 0;
    const p1_queenside = if (p1 == Colors.white)
        state.castling_rights & Castling.white_queenside != 0
    else
        state.castling_rights & Castling.black_queenside != 0;

    if (p1_kingside) fillPlane(buffer + 2 * 64, 1.0);
    if (p1_queenside) fillPlane(buffer + 3 * 64, 1.0);

    // Planes 4-5: P2 castling rights (kingside, queenside)
    const p2_kingside = if (p2 == Colors.white)
        state.castling_rights & Castling.white_kingside != 0
    else
        state.castling_rights & Castling.black_kingside != 0;
    const p2_queenside = if (p2 == Colors.white)
        state.castling_rights & Castling.white_queenside != 0
    else
        state.castling_rights & Castling.black_queenside != 0;

    if (p2_kingside) fillPlane(buffer + 4 * 64, 1.0);
    if (p2_queenside) fillPlane(buffer + 5 * 64, 1.0);

    // Plane 6: No-progress count (halfmove_clock / 100)
    const no_progress: f32 = @as(f32, @floatFromInt(state.halfmove_clock)) / 100.0;
    fillPlane(buffer + 6 * 64, no_progress);
}

fn bitboardToPlane(bits: u64, plane: [*]f32, flip: bool) void {
    var b = bits;
    while (b != 0) {
        const sq: u6 = @intCast(@ctz(b));
        // XOR with 56 flips the rank (0-7 <-> 56-63, 8-15 <-> 48-55, etc.)
        const out_sq: usize = if (flip) sq ^ 56 else sq;
        plane[out_sq] = 1.0;
        b &= b - 1;
    }
}

fn fillPlane(plane: [*]f32, value: f32) void {
    for (0..64) |i| {
        plane[i] = value;
    }
}
