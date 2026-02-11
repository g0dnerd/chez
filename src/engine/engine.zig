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

pub const Move = packed struct(u16) {
    start: square.Square,
    end: square.Square,
    is_promotion: bool = false,
    promotion_piece: piece.Piece = undefined,

    pub fn initCMove(c: *const ffi.CMove) Move {
        var is_promotion = false;
        var promotion_piece: piece.Piece = undefined;

        if (c.promotion_piece != 0) {
            promotion_piece = @as(piece.Piece, @intCast(c.promotion_piece));
            is_promotion = true;
        }

        return .{
            .start = @intCast(c.start),
            .end = @intCast(c.end),
            .promotion_piece = promotion_piece,
            .is_promotion = is_promotion,
        };
    }

    pub fn toCMove(self: *const Move, c_move: *ffi.CMove) void {
        const c_promotion_piece: u8 = if (self.is_promotion)
            @as(u8, @intCast(self.promotion_piece))
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
        if (self.is_promotion) {
            try w.print("{s}{s}{c}", .{ start_buf, end_buf, piece.pieceLetter(self.promotion_piece) });
        } else {
            try w.print("{s}{s}", .{ start_buf, end_buf });
        }
        try w.flush();
    }

    pub fn eql(self: Move, other: Move) bool {
        return self.start == other.start and
            self.end == other.end and
            self.promotion_piece == other.promotion_piece and
            self.is_promotion == other.is_promotion;
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
    insufficientMaterial,
};

pub const Color = u1;
pub const Colors = struct {
    pub const white: Color = 0;
    pub const black: Color = 1;
};

test {
    std.testing.refAllDecls(@This());
}
