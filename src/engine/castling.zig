const square = @import("square.zig");
const Square = square.Square;

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

pub const CastleData = struct {
    king_end: Square,
    rook_from: Square,
    rook_to: Square,
    rights_bit: CastlingRights,
};

pub const castle_data: [2][2]CastleData = .{
    // White
    .{
        .{
            .king_end = square.g1,
            .rook_from = square.h1,
            .rook_to = square.f1,
            .rights_bit = white_kingside,
        },
        .{
            .king_end = square.c1,
            .rook_from = square.a1,
            .rook_to = square.d1,
            .rights_bit = white_queenside,
        },
    },
    // Black
    .{
        .{
            .king_end = square.g8,
            .rook_from = square.h8,
            .rook_to = square.f8,
            .rights_bit = black_kingside,
        },
        .{
            .king_end = square.c8,
            .rook_from = square.a8,
            .rook_to = square.d8,
            .rights_bit = black_queenside,
        },
    },
};

pub const king_castling_mask: [2]CastlingRights = .{
    ~white_castling,
    ~black_castling,
};

pub const rook_castling_mask: [64]CastlingRights = blk: {
    var mask: [64]CastlingRights = @splat(all_legal);

    mask[square.a1] = ~white_queenside;
    mask[square.h1] = ~white_kingside;
    mask[square.a8] = ~black_queenside;
    mask[square.h8] = ~black_kingside;

    break :blk mask;
};
