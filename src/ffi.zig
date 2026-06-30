const std = @import("std");
const allocator = std.heap.smp_allocator;

const chez = @import("chez.zig");
const engine = chez.engine;
const Colors = engine.Colors;
const Move = engine.Move;
const Bitboard = engine.Bitboard;
const State = engine.State;
const piece = engine.piece;
const castling = engine.castling;

pub const CMove = extern struct {
    start: u8,
    end: u8,
    promotion_piece: u8, // 0 = none, 1 = N, 2 = B, 3 = R, 4 = Q
};

pub const CMoveList = extern struct {
    moves: [256]CMove,
    len: u8,
};

// Undo information for make/unmake pattern.
// All optional values use 0xFF as sentinel for "none".
pub const CUndoInfo = extern struct {
    captured_piece: u8, // 0xFF = none
    captured_square: u8,
    castling_rights: u8,
    en_passant: u8, // 0xFF = none
    halfmove_clock: u16,
    in_check: u8, // 0xFF = none
    zobrist_hash: u64,
    flags: u8, // bit0: was_promotion, bit1: was_castling, bit2: castling_side
    promotion_piece: u8, // 0xFF = none

    const flag_was_promotion: u8 = 1 << 0;
    const flag_was_castling: u8 = 1 << 1;
    const flag_castling_side: u8 = 1 << 2;

    fn fromUndo(undo: State.UndoInfo) CUndoInfo {
        var flags: u8 = 0;
        if (undo.was_promotion) flags |= flag_was_promotion;
        if (undo.was_castling) flags |= flag_was_castling;
        if (undo.castling_side == 1) flags |= flag_castling_side;

        return .{
            .captured_piece = if (undo.captured_piece) |p| p else 0xFF,
            .captured_square = undo.captured_square,
            .castling_rights = undo.castling_rights,
            .en_passant = if (undo.en_passant) |ep| ep else 0xFF,
            .halfmove_clock = undo.halfmove_clock,
            .in_check = if (undo.in_check) |c| c else 0xFF,
            .zobrist_hash = undo.zobrist_hash,
            .flags = flags,
            .promotion_piece = if (undo.promotion_piece) |p| p else 0xFF,
        };
    }

    fn toUndo(self: CUndoInfo) State.UndoInfo {
        return .{
            .captured_piece = if (self.captured_piece == 0xFF) null else @intCast(self.captured_piece),
            .captured_square = @intCast(self.captured_square),
            .castling_rights = @intCast(self.castling_rights),
            .en_passant = if (self.en_passant == 0xFF) null else @intCast(self.en_passant),
            .halfmove_clock = self.halfmove_clock,
            .in_check = if (self.in_check == 0xFF) null else @intCast(self.in_check),
            .zobrist_hash = self.zobrist_hash,
            .was_promotion = self.flags & flag_was_promotion != 0,
            .promotion_piece = if (self.promotion_piece == 0xFF) null else @intCast(self.promotion_piece),
            .was_castling = self.flags & flag_was_castling != 0,
            .castling_side = @intFromBool(self.flags & flag_castling_side != 0),
        };
    }
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
    var moves = chez.engine.movegen.legalMoves(state, state.to_move);
    moves.toCMoves(c_moves);
}

export fn chez_make_move(state: *State, c_move: *const CMove) void {
    const move: Move = .initCMove(c_move);
    _ = state.makeMove(move, state.to_move, state.mailbox[move.start].?);
}

// Apply a move and return undo information for later unmake.
export fn chez_make_move_with_undo(state: *State, c_move: *const CMove, c_undo: *CUndoInfo) void {
    const move: Move = .initCMove(c_move);
    const undo = state.makeMove(move, state.to_move, state.mailbox[move.start].?);
    c_undo.* = CUndoInfo.fromUndo(undo);
}

// Unmake a move, restoring the previous state.
// The color and piece parameters are derived from the current state and move.
export fn chez_unmake_move(state: *State, c_move: *const CMove, c_undo: *const CUndoInfo) void {
    const move: Move = .initCMove(c_move);
    const undo = c_undo.toUndo();
    // After makeMove, to_move was flipped. The color that made the move is the opponent of current.
    const color = ~state.to_move;
    // The piece that moved is now at the end square (unless it was a promotion)
    const p = if (undo.was_promotion) piece.pawn else state.mailbox[move.end].?;
    state.unmakeMove(move, color, p, undo);
}

// Return the piece at a square (0-5 for pieces, 0xFF for empty).
export fn chez_piece_at(state: *const State, square: u8) u8 {
    return if (state.pieceAt(@intCast(square))) |p| p else 0xFF;
}

// Return the game result for the state:
//   0 = ongoing
//   1 = white wins
//   2 = black wins
//   3 = draw
export fn chez_game_result(state: *const State) i32 {
    const res = chez.engine.search.isGameOverWithHistory(state, null) orelse return 0;
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

// Encode position to 12 planes (6 P1 pieces + 6 P2 pieces).
// P1 = current player to move, P2 = opponent.
// Board is flipped if black to move (current player's pieces always at bottom).
// Buffer must be pre-zeroed, size = 12 * 64 = 768 floats.
export fn chez_encode_position(state: *const State, buffer: [*]f32) void {
    const flip = state.to_move == Colors.black;
    const p1 = state.to_move;
    const p2 = ~p1;

    // P1 pieces (planes 0-5: pawn, knight, bishop, rook, queen, king)
    inline for (0..6) |p| {
        const bb = state.pieces[p].bitAnd(Bitboard, state.colors[p1]);
        bitboardToPlane(bb.bits, buffer + p * 64, flip);
    }

    // P2 pieces (planes 6-11)
    inline for (0..6) |p| {
        const bb = state.pieces[p].bitAnd(Bitboard, state.colors[p2]);
        bitboardToPlane(bb.bits, buffer + (6 + p) * 64, flip);
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
        state.castling_rights & castling.white_kingside != 0
    else
        state.castling_rights & castling.black_kingside != 0;
    const p1_queenside = if (p1 == Colors.white)
        state.castling_rights & castling.white_queenside != 0
    else
        state.castling_rights & castling.black_queenside != 0;

    if (p1_kingside) fillPlane(buffer + 2 * 64, 1.0);
    if (p1_queenside) fillPlane(buffer + 3 * 64, 1.0);

    // Planes 4-5: P2 castling rights (kingside, queenside)
    const p2_kingside = if (p2 == Colors.white)
        state.castling_rights & castling.white_kingside != 0
    else
        state.castling_rights & castling.black_kingside != 0;
    const p2_queenside = if (p2 == Colors.white)
        state.castling_rights & castling.white_queenside != 0
    else
        state.castling_rights & castling.black_queenside != 0;

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
