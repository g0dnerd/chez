const std = @import("std");

pub const Bitboard = @import("Bitboard.zig");
pub const State = @import("State.zig");
pub const evaluation = @import("evaluation.zig");
pub const game = @import("game.zig");
pub const movegen = @import("movegen.zig");
pub const precompute = @import("precompute.zig");
pub const search = @import("search.zig");

pub const GameResult = game.GameResult;
pub const Squares = game.Squares;
pub const Square = Squares.Square;
pub const Pieces = game.Pieces;
pub const Piece = Pieces.Piece;
pub const Castling = game.Castling;
pub const CastlingRights = Castling.CastlingRights;
pub const Colors = game.Colors;
pub const Color = Colors.Color;
pub const Move = game.Move;
pub const MoveList = movegen.MoveList;

pub const legalMoves = movegen.legalMoves;

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

test {
    std.testing.refAllDecls(@This());
}
