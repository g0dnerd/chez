const std = @import("std");

const ffi = @import("../ffi.zig");
const precompute = @import("../precompute.zig");

pub const Bitboard = @import("Bitboard.zig");
pub const State = @import("State.zig");
pub const castling = @import("castling.zig");
pub const evaluation = @import("evaluation.zig");
pub const movegen = @import("movegen.zig");
pub const piece = @import("piece.zig");
pub const square = @import("square.zig");
pub const search = @import("search.zig");

pub const MagicTableEntry = struct {
    magic: u64,
    mask: u64,
    shift: u6,
    offset: u32,

    pub fn magicTableIndex(self: *const MagicTableEntry, blockers: Bitboard) usize {
        const blockers_masked = blockers.bits & self.mask;
        const hash = blockers_masked *% self.magic;
        const idx: usize = @intCast(hash >> self.shift);
        return @as(usize, self.offset) + idx;
    }
};

pub const Move = struct {
    start: square.Square,
    end: square.Square,
    promotion_piece: ?piece.Piece = null,

    pub fn initCMove(c: *const ffi.CMove) Move {
        const promotion_piece: ?piece.Piece = if (c.promotion_piece == 0)
            null
        else
            @as(piece.Piece, @intCast(c.promotion_piece));

        return .{
            .start = @intCast(c.start),
            .end = @intCast(c.end),
            .promotion_piece = promotion_piece,
        };
    }

    pub fn toCMove(self: *const Move, c_move: *ffi.CMove) void {
        const c_promotion_piece: u8 = if (self.promotion_piece) |p|
            @as(u8, @intCast(p))
        else
            0;

        c_move.*.start = self.start;
        c_move.*.end = self.end;
        c_move.*.promotion_piece = c_promotion_piece;
    }

    pub fn format(self: Move, w: *std.Io.Writer) !void {
        var end_buf: [2]u8 = undefined;
        var start_buf: [2]u8 = undefined;
        square.toAlgebraic(self.start, &start_buf) catch {};
        square.toAlgebraic(self.end, &end_buf) catch {};
        if (self.promotion_piece) |p| {
            try w.print("{s}{s}{c}", .{ start_buf, end_buf, piece.pieceLetter(p) });
        } else {
            try w.print("{s}{s}", .{ start_buf, end_buf });
        }
        try w.flush();
    }

    pub fn eql(self: Move, other: Move) bool {
        return self.start == other.start and self.end == other.end and self.promotion_piece == other.promotion_piece;
    }
};

pub const Direction = enum {
    horizontal,
    vertical,
    diagonal,
    antiDiagonal,
    none,

    pub fn fromSquares(from: square.Square, to: square.Square) Direction {
        const from_file = from % 8;
        const from_rank = from / 8;
        const to_file = to % 8;
        const to_rank = to / 8;

        if (from_rank == to_rank) {
            return Direction.horizontal;
        } else if (from_file == to_file) {
            return Direction.vertical;
        } else if (square.absDiff(from_rank, to_rank) == square.absDiff(from_file, to_file)) {
            const rank_diff = @as(i8, to_rank) - @as(i8, from_rank);
            const file_diff = @as(i8, to_file) - @as(i8, from_file);

            if (std.math.sign(rank_diff) == std.math.sign(file_diff)) {
                return Direction.diagonal;
            } else {
                return Direction.antiDiagonal;
            }
        } else {
            return Direction.none;
        }
    }
};

pub const GameResult = union(enum) {
    checkmate: Color,
    stalemate,
    fiftyMoveRule,
    threefoldRepetition,
};

pub const Color = u1;
pub const Colors = struct {
    pub const white: Color = 0;
    pub const black: Color = 1;
};

test {
    std.testing.refAllDecls(@This());
}
