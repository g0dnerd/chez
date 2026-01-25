const std = @import("std");
const ffi = @import("ffi.zig");
const Bitboard = @import("Bitboard.zig");

pub const Move = struct {
    start: Squares.Square,
    end: Squares.Square,
    promotion_piece: ?Pieces.Piece = null,

    pub fn initCMove(c: *const ffi.CMove) Move {
        const promotion_piece: ?Pieces.Piece = if (c.promotion_piece == 0)
            null
        else
            @as(Pieces.Piece, @intCast(c.promotion_piece));

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
};

pub const GameResult = union(enum) {
    checkmate: Colors.Color,
    stalemate,
    fiftyMoveRule,
    threefoldRepetition,
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
    const diff = @as(i8, @intCast(lhs)) - @as(i8, @intCast(rhs));
    return @intCast(@abs(diff));
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
    pub const rook_directions: SliderDirections = .{
        .{ 0, 1 },
        .{ 1, 0 },
        .{ 0, -1 },
        .{ -1, 0 },
    };
    pub const bishop_directions: SliderDirections = .{
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

    pub const no_legal: CastlingRights = 0;
    pub const white_kingside: CastlingRights = 1;
    pub const white_queenside: CastlingRights = 2;
    pub const black_queenside: CastlingRights = 4;
    pub const black_kingside: CastlingRights = 8;
    pub const both_kingsides: CastlingRights = white_kingside | black_kingside;
    pub const both_queensides: CastlingRights = white_queenside | black_queenside;
    pub const white_castling: CastlingRights = white_kingside | white_queenside;
    pub const black_castling: CastlingRights = black_kingside | black_queenside;
    pub const all_legal: CastlingRights = white_castling | black_castling;
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
        return Bitboard.empty;
    }

    const direction = Direction.fromSquares(from, to);
    if (direction == .none) {
        return Bitboard.empty;
    }

    const ray = rayBetweenInclusive(from, to, direction);
    return ray.bitAnd(Bitboard.fromSquare(from).not()).bitAnd(Bitboard.fromSquare(to).not());
}

pub fn rayBetweenInclusive(from: Squares.Square, to: Squares.Square, d: Direction) Bitboard {
    const from_file = from % 8;
    const from_rank = from / 8;
    const to_file = to % 8;
    const to_rank = to / 8;

    var ray = Bitboard.empty;

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
            const rank_step: i8 = if (to_rank > from_rank)
                1
            else
                -1;

            const file_step: i8 = if (to_file > from_file)
                1
            else
                -1;

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

pub const ZobristKeys = struct {
    pieces: [2][6][64]u64, // [color][piece_type][square]
    side_to_move: u64, // XOR when black to move
    castling: [16]u64, // One key per castling rights combination
    en_passant: [8]u64, // One key per file (only file matters for en passant)
};

var keys_once = std.once(initZobristKeys);
var keys_storage: ZobristKeys = undefined;

fn initZobristKeys() void {
    var seed: u64 = undefined;
    _ = std.os.linux.getrandom(std.mem.asBytes(&seed), 1, 0); // catch @panic("getrandom failed");
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();

    // Piece-square keys for each color
    for (0..2) |color| {
        for (0..6) |piece_type| {
            for (0..64) |square| {
                keys_storage.pieces[color][piece_type][square] = random.int(u64);
            }
        }
    }

    // Side to move key
    keys_storage.side_to_move = random.int(u64);

    // Castling rights keys
    for (0..16) |rights| {
        keys_storage.castling[rights] = random.int(u64);
    }

    // En passant file keys
    for (0..8) |file| {
        keys_storage.en_passant[file] = random.int(u64);
    }
}

pub fn getZobristKeys() *const ZobristKeys {
    keys_once.call();
    return &keys_storage;
}

pub const piece_repr = [2][6][]const u8{
    [_][]const u8{ "\u{265F}", "\u{265E}", "\u{265D}", "\u{265C}", "\u{265B}", "\u{265A}" },
    [_][]const u8{ "\u{2659}", "\u{2658}", "\u{2657}", "\u{2656}", "\u{2655}", "\u{2654}" },
};

pub const piece_repr_symbol = [6][]const u8{ "󰡙", "󰡘", "󰡜", "󰡛", "󰡚", "󰡗" };

pub const CastleData = struct {
    king_end: Squares.Square,
    rook_from: Squares.Square,
    rook_to: Squares.Square,
    rights_bit: Castling.CastlingRights,
};

pub const castle_data: [2][2]CastleData = .{
    // White
    .{
        .{ .king_end = Squares.g1, .rook_from = Squares.h1, .rook_to = Squares.f1, .rights_bit = Castling.white_kingside },
        .{ .king_end = Squares.c1, .rook_from = Squares.a1, .rook_to = Squares.d1, .rights_bit = Castling.white_queenside },
    },
    // Black
    .{
        .{ .king_end = Squares.g8, .rook_from = Squares.h8, .rook_to = Squares.f8, .rights_bit = Castling.black_kingside },
        .{ .king_end = Squares.c8, .rook_from = Squares.a8, .rook_to = Squares.d8, .rights_bit = Castling.black_queenside },
    },
};

pub const king_castling_mask: [2]Castling.CastlingRights = .{
    ~Castling.white_castling,
    ~Castling.black_castling,
};

pub const rook_castling_mask: [64]Castling.CastlingRights = blk: {
    var mask: [64]Castling.CastlingRights = @splat(Castling.all_legal);

    mask[Squares.a1] = ~Castling.white_queenside;
    mask[Squares.h1] = ~Castling.white_kingside;
    mask[Squares.a8] = ~Castling.black_queenside;
    mask[Squares.h8] = ~Castling.black_kingside;

    break :blk mask;
};

pub fn squareToAlgebraic(square: Squares.Square, buf: []u8) !void {
    const file: u8 = 'a' + @as(u8, square) % 8;
    const rank: u8 = '1' + @as(u8, square) / 8;
    _ = try std.fmt.bufPrint(buf, "{c}{c}", .{ file, rank });
}

test "test try square offset" {
    try std.testing.expectEqual(trySquareOffset(Squares.a1, -1, 0), null);
}
