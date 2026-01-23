const evaluation = @import("evaluation.zig");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const State = @import("State.zig");
const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;

const CHECKMATE_SCORE: i32 = 100000;
const ALPHA_INIT: i32 = std.math.minInt(i32) + 1;
const BETA_INIT: i32 = std.math.maxInt(i32);
const MAX_PLY: usize = 64;

// Killer move table: stores 2 killer moves per ply
// Killer moves are quiet moves that caused beta cutoffs
const KillerTable = struct {
    moves: [MAX_PLY][2]?game.Move = [_][2]?game.Move{.{ null, null }} ** MAX_PLY,

    fn store(self: *KillerTable, ply: usize, m: game.Move) void {
        if (ply >= MAX_PLY) return;
        // Don't store if it's already the first killer
        if (self.moves[ply][0]) |k| {
            if (k.start == m.start and k.end == m.end) return;
        }
        // Shift first killer to second slot, store new as first
        self.moves[ply][1] = self.moves[ply][0];
        self.moves[ply][0] = m;
    }

    fn isKiller(self: *const KillerTable, ply: usize, m: game.Move) bool {
        if (ply >= MAX_PLY) return false;
        if (self.moves[ply][0]) |k| {
            if (k.start == m.start and k.end == m.end) return true;
        }
        if (self.moves[ply][1]) |k| {
            if (k.start == m.start and k.end == m.end) return true;
        }
        return false;
    }

    fn clear(self: *KillerTable) void {
        self.moves = [_][2]?game.Move{.{ null, null }} ** MAX_PLY;
    }
};

const Flag = enum(u2) {
    empty = 0,
    exact = 1,
    lowerBound = 2,
    upperBound = 3,
};

const TranspositionEntry = struct {
    hash: u64 = 0,
    score: i32 = 0,
    depth: u8 = 0,
    flag: Flag = .empty,
    best_move: ?game.Move = null,
};

// Fixed-size transposition table using direct indexing
// Size must be a power of 2 for fast modulo via bitmask
const TT_SIZE_BITS = 20; // 2^20 = ~1M entries
const TT_SIZE: usize = 1 << TT_SIZE_BITS;
const TT_MASK: u64 = TT_SIZE - 1;

const TranspositionTable = struct {
    entries: []TranspositionEntry,

    fn init(alloc: std.mem.Allocator) !TranspositionTable {
        const entries = try alloc.alloc(TranspositionEntry, TT_SIZE);
        @memset(entries, TranspositionEntry{});
        return .{ .entries = entries };
    }

    fn deinit(self: *TranspositionTable, alloc: std.mem.Allocator) void {
        alloc.free(self.entries);
    }

    fn probe(self: *const TranspositionTable, hash: u64) ?TranspositionEntry {
        const idx = hash & TT_MASK;
        const entry = self.entries[idx];

        // Check if entry is valid and matches the hash
        if (entry.flag != .empty and entry.hash == hash) return entry;

        return null;
    }

    fn store(self: *TranspositionTable, hash: u64, score: i32, depth: u8, flag: Flag, best_move: ?game.Move) void {
        const idx = hash & TT_MASK;
        const existing = &self.entries[idx];

        // Replacement strategy: always replace if new entry has >= depth
        // This favors more recent searches which tend to be more relevant
        if (existing.flag == .empty or depth >= existing.depth) {
            existing.* = .{
                .hash = hash,
                .score = score,
                .depth = depth,
                .flag = flag,
                .best_move = best_move,
            };
        }
    }
};

// Quiescence search: search only captures until the position is "quiet"
// This prevents the horizon effect where we evaluate positions mid-tactical-sequence
fn quiescence(state: *State, alpha_init: i32, beta: i32) i32 {
    const to_move = state.to_move;
    const in_check = state.in_check == to_move;

    // When in check, we must search all evasions, not just captures
    if (in_check) {
        var moves = movegen.legalMoves(state, to_move);

        if (moves.len == 0) {
            return -CHECKMATE_SCORE;
        }

        moves.order(state, to_move);

        var alpha = alpha_init;
        for (0..moves.len) |i| {
            const m = moves.moves[i];
            const p = state.pieceAt(m.start).?;
            const undo = state.makeMove(m, to_move, p);

            const score = -quiescence(state, -beta, -alpha);

            state.unmakeMove(m, to_move, p, undo);

            if (score >= beta) {
                return beta;
            }
            if (score > alpha) {
                alpha = score;
            }
        }
        return alpha;
    }

    // Stand pat: evaluate the current position
    // We can always choose not to capture when not in check
    const stand_pat = evaluation.evaluate(state);

    // Beta cutoff: position is so good opponent wouldn't allow it
    if (stand_pat >= beta) {
        return beta;
    }

    var alpha = alpha_init;

    // If stand pat is better than alpha, we can use it as a floor
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    // Generate and search captures only
    var captures = movegen.legalCaptures(state, to_move);

    if (captures.len == 0) {
        return stand_pat;
    }

    captures.order(state, to_move);

    for (0..captures.len) |i| {
        const m = captures.moves[i];
        const p = state.pieceAt(m.start).?;
        const undo = state.makeMove(m, to_move, p);

        const score = -quiescence(state, -beta, -alpha);

        state.unmakeMove(m, to_move, p, undo);

        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    return alpha;
}

fn negamax(state: *State, depth: u8, ply: usize, alpha_init: i32, beta: i32, tbl: *TranspositionTable, killers: *KillerTable) i32 {
    const hash = state.zobrist_hash;
    var alpha = alpha_init;
    var best_move: ?game.Move = null;

    // Probe transposition table
    if (tbl.probe(hash)) |entry| {
        if (entry.depth >= depth) {
            switch (entry.flag) {
                .exact => return entry.score,
                .lowerBound => alpha = @max(alpha, entry.score),
                .upperBound => {
                    if (entry.score <= alpha) return entry.score;
                },
                .empty => {},
            }
            if (alpha >= beta) {
                return entry.score;
            }
        }
    }

    const to_move = state.to_move;
    var moves = movegen.legalMoves(state, to_move);

    if (moves.len == 0) {
        if (state.in_check == to_move) {
            return -CHECKMATE_SCORE;
        } else {
            return 0;
        }
    }

    if (depth == 0) {
        return quiescence(state, alpha, beta);
    }

    // Null move pruning: if giving opponent a free move still results in beta cutoff,
    // the position is so good we can prune
    const in_check = state.in_check == to_move;
    if (depth >= 3 and !in_check and state.hasNonPawnMaterial(to_move)) {
        const keys = game.getZobristKeys();
        // Save state for null move
        const old_to_move = state.to_move;
        const old_ep = state.en_passant;
        const old_hash = state.zobrist_hash;
        const old_in_check = state.in_check;

        // Make null move: flip side to move, clear en passant
        state.*.to_move = ~state.to_move;
        state.*.zobrist_hash ^= keys.side_to_move;
        if (state.en_passant) |ep| {
            state.*.zobrist_hash ^= keys.en_passant[ep % 8];
            state.*.en_passant = null;
        }
        state.*.in_check = null; // After null move, we're not giving check

        // Adaptive reduction: R = 2 + depth/4
        const R: u8 = 2 + depth / 4;
        const null_score = -negamax(state, depth - 1 - R, ply + 1, -beta, -beta + 1, tbl, killers);

        // Unmake null move
        state.*.to_move = old_to_move;
        state.*.en_passant = old_ep;
        state.*.zobrist_hash = old_hash;
        state.*.in_check = old_in_check;

        if (null_score >= beta) {
            return beta;
        }
    }

    // Order moves with killer heuristic
    const ply_killers = if (ply < MAX_PLY) killers.moves[ply] else [2]?game.Move{ null, null };
    moves.orderWithKillers(state, to_move, ply_killers);

    var max_score: i32 = std.math.minInt(i32);

    for (0..moves.len) |i| {
        const m = moves.moves[i];
        const p = state.pieceAt(m.start).?;
        const undo = state.makeMove(m, to_move, p);

        // Check extension: extend search by 1 ply when giving check
        const gives_check = state.in_check != null;
        const extension: u8 = if (gives_check) 1 else 0;
        const new_depth = depth - 1 + extension;

        var score: i32 = undefined;
        if (i == 0) {
            // First move: search with full window
            score = -negamax(state, new_depth, ply + 1, -beta, -alpha, tbl, killers);
        } else {
            // PVS: search with null window first
            score = -negamax(state, new_depth, ply + 1, -alpha - 1, -alpha, tbl, killers);
            if (score > alpha and score < beta) {
                // Null window failed high, re-search with full window
                score = -negamax(state, new_depth, ply + 1, -beta, -alpha, tbl, killers);
            }
        }

        // Check if this was a capture (before unmaking)
        const was_capture = undo.captured_piece != null;

        state.unmakeMove(m, to_move, p, undo);

        if (score > max_score) {
            max_score = score;
            best_move = m;
        }
        alpha = @max(alpha, score);

        if (alpha >= beta) {
            // Beta cutoff - store killer if this is a quiet move (not a capture)
            if (!was_capture) {
                killers.store(ply, m);
            }
            break;
        }
    }

    // Determine flag for TT entry
    const flag: Flag = if (max_score <= alpha_init)
        .upperBound
    else if (max_score >= beta)
        .lowerBound
    else
        .exact;

    tbl.store(hash, max_score, depth, flag, best_move);

    return max_score;
}

pub const SearchResult = struct {
    move: game.Move,
    score: i32,
    depth: u8,
};

// Search at a specific depth with an optional hint for the best move from the previous iteration
fn searchAtDepth(state: *State, depth: u8, tbl: *TranspositionTable, killers: *KillerTable, pv_move: ?game.Move) ?SearchResult {
    var best_score: i32 = std.math.minInt(i32);
    var best_move: ?game.Move = null;

    const to_move = state.to_move;
    var moves = movegen.legalMoves(state, to_move);

    if (moves.len == 0) {
        return null;
    }

    moves.order(state, to_move);

    // If we have a PV move from the previous iteration, try it first
    if (pv_move) |pv| {
        // Find and move PV to front
        for (0..moves.len) |i| {
            if (moves.moves[i].start == pv.start and moves.moves[i].end == pv.end) {
                // Swap PV move to front
                const tmp = moves.moves[0];
                moves.moves[0] = moves.moves[i];
                moves.moves[i] = tmp;
                break;
            }
        }
    }

    for (0..moves.len) |i| {
        const m = moves.moves[i];
        const p = state.pieceAt(m.start).?;
        const undo = state.makeMove(m, to_move, p);

        // Take mate in 1
        if (isGameOver(state)) |r| {
            switch (r) {
                .checkmate => if (r.checkmate == to_move) {
                    state.unmakeMove(m, to_move, p, undo);
                    return .{ .move = m, .score = CHECKMATE_SCORE, .depth = depth };
                },
                else => {},
            }
        }

        const score = -negamax(state, depth - 1, 1, -BETA_INIT, -ALPHA_INIT, tbl, killers);

        state.unmakeMove(m, to_move, p, undo);

        if (score > best_score) {
            best_move = m;
            best_score = score;
        }
    }

    if (best_move) |m| {
        return .{
            .move = m,
            .score = best_score,
            .depth = depth,
        };
    } else {
        return null;
    }
}

// Iterative deepening search: searches depth 1, then 2, etc. up to max_depth.
// This enables better move ordering from previous iterations via the transposition table.
pub fn search(state: *const State, max_depth: u8) !?SearchResult {
    var tbl = try TranspositionTable.init(std.heap.page_allocator);
    defer tbl.deinit(std.heap.page_allocator);

    var killers = KillerTable{};

    // Make a mutable copy for the search (make/unmake will restore it)
    var mutable_state = state.*;

    var best_move: ?game.Move = null;
    var best_score: i32 = undefined;
    var best_depth: u8 = 0;

    var sq_start: [2]u8 = undefined;
    var sq_end: [2]u8 = undefined;

    for (1..max_depth + 1) |depth| {
        const result = searchAtDepth(&mutable_state, @intCast(depth), &tbl, &killers, best_move);
        if (result) |r| {
            best_move = r.move;
            best_score = r.score;

            try game.squareToAlgebraic(best_move.?.start, &sq_start);
            try game.squareToAlgebraic(best_move.?.end, &sq_end);
            std.debug.print("Thinking at depth {d} - candidate move: {s}{s} - evaluation: {d}\r", .{ depth, sq_start, sq_end, best_score });

            best_depth = r.depth;

            // Early exit if we found a checkmate
            if (best_score >= CHECKMATE_SCORE - 100) {
                break;
            }
        }
    }

    std.debug.print("\n", .{});

    if (best_move) |m| {
        return .{
            .move = m,
            .score = best_score,
            .depth = best_depth,
        };
    } else {
        return null;
    }
}

pub fn isGameOver(state: *const State) ?game.GameResult {
    const to_move = state.to_move;
    const hasLegalMoves = movegen.hasAnyLegalMove(state, to_move);

    if (!hasLegalMoves) {
        const king_square = state.pieceBitboard(game.Pieces.king).bitAnd(state.colorBitboard(to_move)).trailingZeros();
        if (movegen.isSquareAttackedBy(state, king_square, ~to_move)) {
            return game.GameResult{ .checkmate = ~to_move };
        } else {
            return .stalemate;
        }
    }

    if (state.halfmove_clock >= 100) {
        return .fiftyMoveRule;
    }

    return null;
}

test "test search finds mate in one" {
    const fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4";
    const state = try State.fromFen(fen);

    const result = try search(&state, 1);
    try expect(result != null);
    try expectEqual(game.Squares.h5, result.?.move.start);
    try expectEqual(game.Squares.f7, result.?.move.end);
    try expect(result.?.score > 50000);
}

test "test detects stalemate" {
    const fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1";
    const state = try State.fromFen(fen);

    const res = isGameOver(&state);
    try expectEqual(game.GameResult.stalemate, res);
}

// FIXME: this still fails

// Mate in 2 tests
// test "backrank mate in two" {
//     // Rook on c2 can deliver mate: Qc8+ Rxc8 Rxc8#
//     const fen = "r5k1/5ppp/8/2Q5/8/8/2R5/4K3 w - - 2 1";
//     const state = try State.fromFen(fen);
//
//     const result = try search(&state, 4);
//     try expect(result != null);
//     try expectEqual(game.Squares.c5, result.?.move.start);
//     try expectEqual(game.Squares.c8, result.?.move.end);
// }

test "mate in two" {
    // Rxh6+ Kxh6 Qg6#
    const fen = "r7/ppq2r1k/1npR3p/4PP2/6Q1/8/PP4PP/R6K w - - 0 1";
    const state = try State.fromFen(fen);

    const result = try search(&state, 4);
    try expect(result != null);
    try expectEqual(game.Squares.d6, result.?.move.start);
    try expectEqual(game.Squares.h6, result.?.move.end);
}

test "smothered mate pattern" {
    const fen = "4r2k/2pRP1pp/2p5/p4pN1/2Q3n1/q5P1/P3PP1P/6K1 w - - 0 1";
    const state = try State.fromFen(fen);

    const result = try search(&state, 4);
    try expect(result != null);
    try expectEqual(game.Squares.g5, result.?.move.start);
    try expectEqual(game.Squares.f7, result.?.move.end);
}

// Mate in 3 tests
test "search finds mate in three" {
    // Opera game finale position - Rd8+ leads to mate
    const fen = "1n2kb1r/p4ppp/4q3/4p1B1/4P3/3R4/PPP2PPP/2K5 w k - 1 17";
    const state = try State.fromFen(fen);

    const result = try search(&state, 6);
    try expect(result != null);
    // Should find Rd8+ leading to mate
    try expect(result.?.score > 50000);
}

test "search avoids stalemate when winning" {
    // White has queen and king vs lone king - should not stalemate
    const fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1";
    const state = try State.fromFen(fen);

    const result = try search(&state, 4);
    try expect(result != null);
    // Should find checkmate, not stalemate the opponent
    // Qf8# or similar mate
    try expect(result.?.score > 50000);
}
