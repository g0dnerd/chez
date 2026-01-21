const std = @import("std");
const Bitboard = @import("Bitboard.zig");

pub const Move = struct {
    start: Squares.Square,
    end: Squares.Square,
};

pub const GameResult = union(enum) {
    checkmate: Colors.Color,
    stalemate,
    fiftyMoveRule,
};

pub const Squares = struct {
    pub const Square = u6;

    pub const a1: Square = 0;
    pub const b1: Square = 1;
    pub const c1: Square = 2;
    pub const d1: Square = 3;
    pub const e1: Square = 4;
    pub const f1: Square = 5;
    pub const g1: Square = 6;
    pub const h1: Square = 7;
    pub const a2: Square = 8;
    pub const b2: Square = 9;
    pub const c2: Square = 10;
    pub const d2: Square = 11;
    pub const e2: Square = 12;
    pub const f2: Square = 13;
    pub const g2: Square = 14;
    pub const h2: Square = 15;
    pub const a3: Square = 16;
    pub const b3: Square = 17;
    pub const c3: Square = 18;
    pub const d3: Square = 19;
    pub const e3: Square = 20;
    pub const f3: Square = 21;
    pub const g3: Square = 22;
    pub const h3: Square = 23;
    pub const a4: Square = 24;
    pub const b4: Square = 25;
    pub const c4: Square = 26;
    pub const d4: Square = 27;
    pub const e4: Square = 28;
    pub const f4: Square = 29;
    pub const g4: Square = 30;
    pub const h4: Square = 31;
    pub const a5: Square = 32;
    pub const b5: Square = 33;
    pub const c5: Square = 34;
    pub const d5: Square = 35;
    pub const e5: Square = 36;
    pub const f5: Square = 37;
    pub const g5: Square = 38;
    pub const h5: Square = 39;
    pub const a6: Square = 40;
    pub const b6: Square = 41;
    pub const c6: Square = 42;
    pub const d6: Square = 43;
    pub const e6: Square = 44;
    pub const f6: Square = 45;
    pub const g6: Square = 46;
    pub const h6: Square = 47;
    pub const a7: Square = 48;
    pub const b7: Square = 49;
    pub const c7: Square = 50;
    pub const d7: Square = 51;
    pub const e7: Square = 52;
    pub const f7: Square = 53;
    pub const g7: Square = 54;
    pub const h7: Square = 55;
    pub const a8: Square = 56;
    pub const b8: Square = 57;
    pub const c8: Square = 58;
    pub const d8: Square = 59;
    pub const e8: Square = 60;
    pub const f8: Square = 61;
    pub const g8: Square = 62;
    pub const h8: Square = 63;
};

pub fn trySquareOffset(s: Squares.Square, dx: i3, dy: i3) ?Squares.Square {
    const file: i6 = @intCast(s % 8);
    const rank: i6 = @intCast(s / 8);
    const new_file = file + dx;
    const new_rank = rank + dy;

    if (new_file >= 0 and new_file < 8 and new_rank >= 0 and new_rank < 8) {
        const new_rank_u: u6 = @intCast(new_rank);
        const new_file_u: u6 = @intCast(new_file);
        return @intCast(new_rank_u * 8 + new_file_u);
    } else {
        return null;
    }
}

pub fn absDiff(lhs: Squares.Square, rhs: Squares.Square) Squares.Square {
    return @max(lhs, rhs) - @min(lhs, rhs);
}

pub const Pieces = struct {
    pub const Piece = u3;

    pub const pawn: Piece = 0;
    pub const knight: Piece = 1;
    pub const bishop: Piece = 2;
    pub const rook: Piece = 3;
    pub const queen: Piece = 4;
    pub const king: Piece = 5;
};

pub const Slider = struct {
    pub const SliderDirections = [4][2]i2;
    pub const RookDirections: SliderDirections = .{
        .{ 0, 1 },
        .{ 1, 0 },
        .{ 0, -1 },
        .{ -1, 0 },
    };
    pub const BishopDirections: SliderDirections = .{
        .{ 1, 1 },
        .{ 1, -1 },
        .{ -1, 1 },
        .{ -1, -1 },
    };
};

pub const Colors = struct {
    pub const Color = u1;

    pub const white: Color = 0;
    pub const black: Color = 1;
};

pub const Castling = struct {
    pub const CastlingRights = u4;

    pub const NoLegal: CastlingRights = 0;
    pub const WhiteKingside: CastlingRights = 1;
    pub const WhiteQueenside: CastlingRights = 2;
    pub const BlackQueenside: CastlingRights = 4;
    pub const BlackKingside: CastlingRights = 8;
    pub const BothKingsides: CastlingRights = WhiteKingside | BlackKingside;
    pub const BothQueensides: CastlingRights = WhiteQueenside | BlackQueenside;
    pub const WhiteCastling: CastlingRights = WhiteKingside | WhiteQueenside;
    pub const BlackCastling: CastlingRights = BlackKingside | BlackQueenside;
    pub const AllLegal: CastlingRights = WhiteCastling | BlackCastling;
};

pub const Direction = enum {
    horizontal,
    vertical,
    diagonal,
    antiDiagonal,
    none,

    pub fn fromSquares(from: Squares.Square, to: Squares.Square) Direction {
        const from_file = from % 8;
        const from_rank = from / 8;
        const to_file = to % 8;
        const to_rank = to / 8;

        if (from_rank == to_rank) {
            return Direction.horizontal;
        } else if (from_file == to_file) {
            return Direction.vertical;
        } else if (absDiff(from_rank, to_rank) == absDiff(from_file, to_file)) {
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

pub fn betweenSquares(from: Squares.Square, to: Squares.Square) Bitboard {
    if (from == to) {
        return Bitboard.empty();
    }

    const direction = Direction.fromSquares(from, to);
    if (direction == .none) {
        return Bitboard.empty();
    }

    const ray = rayBetweenInclusive(from, to, direction);
    return ray.bitAnd(Bitboard.fromSquare(from).not()).bitAnd(Bitboard.fromSquare(to).not());
}

pub fn rayBetweenInclusive(from: Squares.Square, to: Squares.Square, d: Direction) Bitboard {
    const from_file = from % 8;
    const from_rank = from / 8;
    const to_file = to % 8;
    const to_rank = to / 8;

    var ray = Bitboard.empty();

    switch (d) {
        .horizontal => {
            const min_file = @min(from_file, to_file);
            const max_file = @max(from_file, to_file);
            for (min_file..max_file + 1) |f| {
                const f_u6: u6 = @intCast(f);
                ray.bitOrAssign(from_rank * 8 + f_u6);
            }
        },
        .vertical => {
            const min_rank = @min(from_rank, to_rank);
            const max_rank = @max(from_rank, to_rank);
            for (min_rank..max_rank + 1) |r| {
                const r_u6: u6 = @intCast(r);
                ray.bitOrAssign(r_u6 * 8 + from_file);
            }
        },
        .diagonal, .antiDiagonal => {
            const rank_step: i8 = blk: {
                if (to_rank > from_rank) {
                    break :blk 1;
                } else {
                    break :blk -1;
                }
            };
            const file_step: i8 = blk: {
                if (to_file > from_file) {
                    break :blk 1;
                } else {
                    break :blk -1;
                }
            };

            var r = @as(i8, from_rank);
            var f = @as(i8, from_file);

            while (true) {
                const s: Squares.Square = @intCast(r * 8 + f);
                ray.bitOrAssign(s);
                if (r == @as(i8, to_rank) and f == @as(i8, to_file)) {
                    break;
                }
                r += rank_step;
                f += file_step;
            }
        },
        .none => {},
    }

    return ray;
}

var keys_once = std.once(initZobristKeys);
var keys_storage: [6][64]u64 = undefined;

fn initZobristKeys() void {
    var seed: u64 = undefined;
    std.posix.getrandom(std.mem.asBytes(&seed)) catch @panic("getrandom failed");
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();

    var dupe_keys = std.AutoHashMap(u64, void).init(std.heap.page_allocator);
    defer dupe_keys.deinit();

    for (0..6) |piece_type| {
        for (0..64) |square| {
            const key = random.int(u64);
            std.debug.assert(!dupe_keys.contains(key));
            dupe_keys.put(key, {}) catch @panic("OOM");
            keys_storage[piece_type][square] = key;
        }
    }
}

pub fn getZobristKeys() *const [6][64]u64 {
    keys_once.call();
    return &keys_storage;
}

pub const PieceRepr = [2][6]u8{
    [_]u8{ 'P', 'N', 'B', 'R', 'Q', 'K' },
    [_]u8{ 'p', 'n', 'b', 'r', 'q', 'k' },
};

test "test try square offset" {
    try std.testing.expectEqual(trySquareOffset(Squares.a1, -1, 0), null);
}
