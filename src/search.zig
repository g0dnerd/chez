const evaluation = @import("evaluation.zig");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const State = @import("State.zig");
const std = @import("std");

const CheckmateScore: i32 = 100000;

const Flag = enum {
    exact,
    lowerBound,
    upperBound,
};

const TranspositionEntry = struct {
    depth: u8,
    score: i32,
    flag: Flag,
};

const TranspositionTable = struct {
    tbl: std.AutoHashMap(u64, TranspositionEntry),
    hits: std.atomic.Value(u32),
    misses: std.atomic.Value(u32),

    fn init(alloc: std.mem.Allocator) TranspositionTable {
        return .{ .hits = .init(0), .misses = .init(0), .tbl = std.AutoHashMap(u64, TranspositionEntry).init(alloc) };
    }

    fn deinit(self: TranspositionTable) void {
        var tbl = self.tbl;
        tbl.deinit();
    }

    fn getEntry(self: *TranspositionTable, hash: u64) ?TranspositionEntry {
        if (self.tbl.get(hash)) |entry| {
            _ = self.*.hits.fetchAdd(1, .release);
            return entry;
        } else {
            _ = self.*.misses.fetchAdd(1, .release);
            return null;
        }
    }

    fn storeEntry(self: *TranspositionTable, hash: u64, entry: TranspositionEntry) !void {
        const old_entry = self.tbl.getPtr(hash);
        if (old_entry) |e| {
            if (entry.depth > e.depth) {
                e.* = entry;
                return;
            }
        }
        try self.*.tbl.put(hash, entry);
    }
};

fn negamax(state: *const State, depth: u8, alpha: *i32, beta: *i32, tbl: *TranspositionTable) !i32 {
    const hash = state.hash();

    if (tbl.getEntry(hash)) |entry| {
        if (entry.depth >= depth) {
            switch (entry.flag) {
                .exact => return entry.score,
                .lowerBound => alpha.* = @max(alpha.*, entry.score),
                .upperBound => beta.* = @min(beta.*, entry.score),
            }
            if (alpha.* >= beta.*) {
                return entry.score;
            }
        }
    }

    const to_move = state.to_move;
    var moves = try movegen.legalMoves(std.heap.page_allocator, state, to_move);
    defer moves.deinit(std.heap.page_allocator);

    if (moves.items.len == 0) {
        if (state.in_check == to_move) {
            return -CheckmateScore;
        } else {
            return 0;
        }
    }

    if (depth == 0) {
        return evaluation.evaluate(state);
    }

    evaluation.orderMoves(state, moves.items, to_move);

    var max_score: i32 = std.math.minInt(i32);

    for (moves.items) |m| {
        const p = state.pieceAt(m.start).?;
        var new_state = state.*;
        new_state.makeMove(m, to_move, p);
        new_state.last_move = m;

        var new_alpha = -(beta.*);
        var new_beta = -(alpha.*);
        const score = -try negamax(&new_state, depth - 1, &new_alpha, &new_beta, tbl);
        max_score = @max(score, max_score);
        alpha.* = @max(alpha.*, score);

        if (alpha.* >= beta.*) break;
    }

    const flag: Flag = if (max_score <= alpha.*)
        .upperBound
    else if (max_score >= beta.*)
        .lowerBound
    else
        .exact;

    try tbl.storeEntry(hash, .{ .depth = depth, .score = max_score, .flag = flag });

    return max_score;
}

pub fn search(state: *const State, depth: u8) !?struct { move: game.Move, score: i32 } {
    var tbl = TranspositionTable.init(std.heap.page_allocator);
    defer tbl.deinit();

    const to_move = state.to_move;
    var moves = try movegen.legalMoves(std.heap.page_allocator, state, to_move);
    defer moves.deinit(std.heap.page_allocator);

    if (moves.items.len == 0) {
        return null;
    }

    evaluation.orderMoves(state, moves.items, to_move);

    var best_score: i32 = std.math.minInt(i32);
    var best_move: ?game.Move = null;

    for (moves.items) |m| {
        const p = state.pieceAt(m.start).?;
        var new_state = state.*;
        new_state.makeMove(m, to_move, p);

        // Take mate in 1
        if (try isGameOver(&new_state)) |r| {
            switch (r) {
                .checkmate => if (r.checkmate == to_move) {
                    return .{ .move = m, .score = CheckmateScore };
                },
                else => {},
            }
        }

        var alpha: i32 = std.math.minInt(i32) + 1;
        var beta: i32 = std.math.maxInt(i32);

        const score = -try negamax(&new_state, depth - 1, &alpha, &beta, &tbl);
        if (score > best_score) {
            best_move = m;
            best_score = score;
        }
    }

    if (best_move) |m| {
        return .{
            .move = m,
            .score = best_score,
        };
    } else {
        return null;
    }
}

pub fn isGameOver(state: *const State) !?game.GameResult {
    const to_move = state.to_move;
    var moves = try movegen.legalMoves(std.heap.page_allocator, state, to_move);
    defer moves.deinit(std.heap.page_allocator);

    if (moves.items.len == 0) {
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
    try std.testing.expect(result != null);
    try std.testing.expectEqual(game.Squares.h5, result.?.move.start);
    try std.testing.expectEqual(game.Squares.f7, result.?.move.end);
    try std.testing.expect(result.?.score > 50000);
}

test "test detects stalemate" {
    const fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1";
    const state = try State.fromFen(fen);

    const res = try isGameOver(&state);
    try std.testing.expectEqual(game.GameResult.stalemate, res);
}
