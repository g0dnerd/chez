const std = @import("std");
const Bitboard = @import("Bitboard.zig");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const Castling = game.Castling;
const Colors = game.Colors;
const Color = Colors.Color;
const Pieces = game.Pieces;
const Piece = Pieces.Piece;
const Squares = game.Squares;
const Square = Squares.Square;
const expectEqual = std.testing.expectEqual;

pub const State = @This();

pieces: [6]Bitboard,
colors: [2]Bitboard,
to_move: Color,
castling_rights: Castling.CastlingRights,
en_passant: ?Square = null,
in_check: ?Color = null,
halfmove_clock: u16 = 0,
fullmove_clock: u16 = 1,

pub fn defaultPosition() State {
    const pawns = Bitboard{ .bits = 0xff00000000ff00 };
    const knights = Bitboard{ .bits = 0x4200000000000042 };
    const bishops = Bitboard{ .bits = 0x2400000000000024 };
    const rooks = Bitboard{ .bits = 0x8100000000000081 };
    const queens = Bitboard{ .bits = 0x800000000000008 };
    const kings = Bitboard{ .bits = 0x1000000000000010 };

    const white_pieces = Bitboard{ .bits = 0xffff };
    const black_pieces = Bitboard{ .bits = 0xffff000000000000 };

    return State{
        .colors = .{ white_pieces, black_pieces },
        .pieces = .{ pawns, knights, bishops, rooks, queens, kings },
        .to_move = Colors.white,
        .castling_rights = Castling.AllLegal
    };
}

pub fn fromFen(fen: []const u8) !State {
    const FenState = enum {
        placement,
        color,
        castling,
        enPassant,
        halfmove,
        fullmove,
    };
    var state: FenState = .placement;
    var current_square: Square = 56;

    var pawns = Bitboard.empty();
    var knights = Bitboard.empty();
    var bishops = Bitboard.empty();
    var rooks = Bitboard.empty();
    var queens = Bitboard.empty();
    var kings = Bitboard.empty();

    var white_pieces = Bitboard.empty();
    var black_pieces = Bitboard.empty();
    var to_move: ?Color = null;
    var en_passant: ?Square = null;
    var en_passant_file: ?Square = null;
    var castling_rights = Castling.NoLegal;
    var halfmove_clock: ?u16 = null;

    var halfmove_start: usize = 0;
    var fullmove_start: usize = 0;

    for (fen, 0..) |c, i| {
        switch (state) {
            .placement => {
                switch (c) {
                    'p' => {
                        pawns.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'P' => {
                        pawns.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'n' => {
                        knights.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'N' => {
                        knights.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'b' => {
                        bishops.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'B' => {
                        bishops.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'r' => {
                        rooks.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'R' => {
                        rooks.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'q' => {
                        queens.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'Q' => {
                        queens.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'k' => {
                        kings.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'K' => {
                        kings.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    '0', '1', '2', '3', '4', '5', '6', '7', '8' => {
                        const current_rank = current_square / 8;
                        const amount: u6 = @intCast(try std.fmt.charToDigit(c, 10));
                        var new_square = current_square + amount;
                        const new_file = new_square % 8;
                        if (new_file == 0) {
                            new_square -= 1;
                        }
                        const new_rank = new_square / 8;

                        if (new_rank < current_rank) return error.TooManySquares;

                        current_square = new_square;
                        continue;
                    },
                    '/' => {
                        if (current_square % 8 == 7) {
                            current_square -= 15;
                            continue;
                        } else {
                            return error.UnexpectedSlash;
                        }
                    },
                    ' ' => {
                        if (current_square == 7) {
                            state = .color;
                        } else {
                            return error.UnexpectedWhitespace;
                        }
                    },
                    else => return error.InvalidCharacter,
                }
            },
            .color => {
                switch (c) {
                    'w' => to_move = Colors.white,
                    'b' => to_move = Colors.black,
                    ' ' => {
                        if (to_move == null) {
                            return error.UnexpectedWhitespace;
                        } else {
                            state = .castling;
                        }
                    },
                    else => return error.InvalidCharacter,
                }
            },
            .castling => {
                switch (c) {
                    '-' => {},
                    'k' => castling_rights |= Castling.BlackKingside,
                    'K' => castling_rights |= Castling.WhiteKingside,
                    'q' => castling_rights |= Castling.BlackQueenside,
                    'Q' => castling_rights |= Castling.WhiteQueenside,
                    ' ' => state = .enPassant,
                    else => return error.InvalidCharacter,
                }
            },
            .enPassant => {
                switch (c) {
                    '-' => {},
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h' => en_passant_file = @intCast(@as(u8, c) - 'a'),
                    '0', '1', '2', '3', '4', '5', '6', '7', '8' => {
                        if (en_passant_file) |f| {
                            const rank: u6 = @intCast(try std.fmt.charToDigit(c, 10));
                            const ep: Square = rank * 8 + f;
                            if (ep > 63) {
                                return error.InvalidEnpassantSquare;
                            }
                            en_passant = ep;
                        } else {
                            return error.InvalidEnpassantSquare;
                        }
                    },
                    ' ' => {
                        halfmove_start = i + 1;
                        state = .halfmove;
                    },
                    else => return error.InvalidCharacter,
                }
            },
            .halfmove => {
                switch (c) {
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' => continue,
                    ' ' => {
                        halfmove_clock = std.fmt.parseInt(u16, fen[halfmove_start..i], 10) catch return error.InvalidHalfmoveClock;
                        fullmove_start = i + 1;
                        state = .fullmove;
                    },
                    else => return error.InvalidCharacter,
                }
            },
            .fullmove => {
                switch (c) {
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' => continue,
                    else => return error.InvalidCharacter,
                }
            },
        }
    }

    const fullmove_clock = std.fmt.parseInt(u16, fen[fullmove_start..], 10) catch return error.InvalidFullmoveClock;

    var res = State{
        .pieces = .{ pawns, knights, bishops, rooks, queens, kings },
        .colors = .{ white_pieces, black_pieces },
        .to_move = to_move.?,
        .castling_rights = castling_rights,
        .en_passant = en_passant,
        .in_check = null,
        .halfmove_clock = halfmove_clock.?,
        .fullmove_clock = fullmove_clock,
    };

    const pieces = res.colorBitboard(res.to_move);
    const king_mask = res.pieceBitboard(Pieces.king).bitAnd(pieces);
    const king_square = king_mask.trailingZeros();
    if (movegen.isSquareAttackedBy(&res, king_square, ~res.to_move)) {
        res.in_check = res.to_move;
    }

    return res;
}

pub fn colorBitboard(self: *const State, c: Color) Bitboard {
    return self.colors[c];
}

pub fn pieceBitboard(self: *const State, p: Piece) Bitboard {
    return self.pieces[p];
}

pub fn allPieces(self: *const State) Bitboard {
    return self.colors[0].bitOr(self.colors[1]);
}

pub fn pieceAt(self: *const State, s: Square) ?Piece {
    var piece_idx: Piece = 0;
    while (piece_idx < self.pieces.len) {
        defer piece_idx += 1;
        const piece_bb = self.pieces[piece_idx];
        if (piece_bb.contains(s)) {
            return piece_idx;
        }
    }
    return null;
}

pub fn colorAt(self: *const State, s: Square) ?Color {
    if (self.colors[0].contains(s)) {
        return Colors.white;
    } else if (self.colors[1].contains(s)) {
        return Colors.black;
    } else {
        return null;
    }
}

pub fn isSquareEmpty(self: *const State, s: Square) bool {
    return !self.allPieces().contains(s);
}

pub fn makeMove(self: *State, m: game.Move, c: Color, p: Piece) void {
    const start = m.start;
    const end = m.end;

    var is_promotion = false;
    const piece_to_capture = self.pieceAt(end);
    var en_passant_target: ?Square = null;

    self.*.halfmove_clock += 1;

    switch (p) {
        Pieces.pawn => {
            self.*.halfmove_clock = 0;
            const start_rank = start / 8;
            const end_rank = end / 8;

            switch (c) {
                Colors.white => {
                    if (start_rank == 1 and end_rank == 3) {
                        self.*.en_passant = end - 8;
                    } else if (self.en_passant == end) {
                        self.en_passant = null;
                        en_passant_target = end - 8;
                    } else {
                        self.en_passant = null;
                        if (end_rank == 7) is_promotion = true;
                    }
                },
                Colors.black => {
                    if (start_rank == 6 and end_rank == 4) {
                        self.*.en_passant = end + 8;
                    } else if (self.en_passant == end) {
                        self.en_passant = null;
                        en_passant_target = end + 8;
                    } else {
                        self.en_passant = null;
                        if (end_rank == 0) is_promotion = true;
                    }
                },
            }
        },
        Pieces.king => {
            switch (c) {
                Colors.white => {
                    if (game.absDiff(start, end) == 2) {
                        switch (end) {
                            Squares.g1 => {
                                if (self.castling_rights & Castling.WhiteKingside != 0) {
                                    self.*.pieces[Pieces.rook].bitXorAssign(Squares.h1);
                                    self.*.pieces[Pieces.rook].bitOrAssign(Squares.f1);
                                    self.*.colors[c].bitXorAssign(Squares.h1);
                                    self.*.colors[c].bitOrAssign(Squares.f1);
                                }
                            },
                            Squares.c1 => {
                                if (self.castling_rights & Castling.WhiteQueenside != 0) {
                                    self.*.pieces[Pieces.rook].bitXorAssign(Squares.a1);
                                    self.*.pieces[Pieces.rook].bitOrAssign(Squares.c1);
                                    self.*.colors[c].bitXorAssign(Squares.a1);
                                    self.*.colors[c].bitOrAssign(Squares.c1);
                                }
                            },
                            else => unreachable,
                        }
                    }
                    self.*.castling_rights &= ~Castling.WhiteCastling;
                },
                Colors.black => {
                    if (game.absDiff(start, end) == 2) {
                        switch (end) {
                            Squares.g8 => {
                                if (self.castling_rights & Castling.BlackKingside != 0) {
                                    self.*.pieces[Pieces.rook].bitXorAssign(Squares.h8);
                                    self.*.pieces[Pieces.rook].bitOrAssign(Squares.f8);
                                    self.*.colors[c].bitXorAssign(Squares.h8);
                                    self.*.colors[c].bitOrAssign(Squares.f8);
                                }
                            },
                            Squares.c8 => {
                                if (self.castling_rights & Castling.BlackQueenside != 0) {
                                    self.*.pieces[Pieces.rook].bitXorAssign(Squares.a8);
                                    self.*.pieces[Pieces.rook].bitOrAssign(Squares.c8);
                                    self.*.colors[c].bitXorAssign(Squares.a8);
                                    self.*.colors[c].bitOrAssign(Squares.c8);
                                }
                            },
                            else => unreachable,
                        }
                    }
                    self.*.castling_rights &= ~Castling.BlackCastling;
                },
            }
            self.en_passant = null;
        },
        Pieces.rook => {
            switch (c) {
                Colors.white => {
                    switch (start) {
                        Squares.a1 => self.*.castling_rights &= ~Castling.WhiteQueenside,
                        Squares.h1 => self.*.castling_rights &= ~Castling.WhiteKingside,
                        else => {},
                    }
                },
                Colors.black => {
                    switch (start) {
                        Squares.a8 => self.*.castling_rights &= ~Castling.BlackQueenside,
                        Squares.h8 => self.*.castling_rights &= ~Castling.BlackKingside,
                        else => {},
                    }
                },
            }
            self.en_passant = null;
        },
        else => self.en_passant = null,
    }

    if (piece_to_capture) |x| {
        self.*.halfmove_clock = 0;
        self.*.pieces[x].bitXorAssign(end);
        self.*.colors[~c].bitXorAssign(end);
    }

    if (en_passant_target) |t| {
        self.*.halfmove_clock = 0;
        self.*.pieces[Pieces.pawn].bitXorAssign(t);
        self.*.colors[~c].bitXorAssign(t);
    }

    // Actually move the piece
    self.*.pieces[p].bitXorAssign(start);
    self.*.colors[c].bitXorAssign(start);
    self.*.pieces[p].bitOrAssign(end);
    self.*.colors[c].bitOrAssign(end);

    if (is_promotion) {
        self.*.pieces[Pieces.pawn].bitXorAssign(end);
        self.*.pieces[Pieces.queen].bitOrAssign(end);
    }

    if (self.to_move == Colors.black) {
        self.*.fullmove_clock += 1;
    }

    self.*.to_move = ~self.to_move;
}

test "piece at sanity" {
    const state = State.defaultPosition();
    const p_pawn = state.pieceAt(12);
    try expectEqual(p_pawn, 0);

    const p_king = state.pieceAt(60);
    try expectEqual(p_king, 5);

    const no_piece = state.pieceAt(20);
    try expectEqual(no_piece, null);
}

test "color at sanity" {
    const state = State.defaultPosition();
    const c_white = state.colorAt(12);
    try expectEqual(c_white, 0);

    const c_black = state.colorAt(60);
    try expectEqual(c_black, 1);

    const no_color = state.colorAt(20);
    try expectEqual(no_color, null);
}

test "all pieces sanity" {
    const state = State.defaultPosition();
    const all_pieces = state.allPieces();

    try expectEqual(all_pieces.bits, 0xffff00000000ffff);
}

test "test castling rights removal" {
    var state = State.defaultPosition();
    state.makeMove(game.Move{ .start = Squares.e2, .end = Squares.e4 }, Colors.white, Pieces.pawn);
    state.makeMove(game.Move{ .start = Squares.e7, .end = Squares.e5 }, Colors.black, Pieces.pawn);
    state.makeMove(game.Move{ .start = Squares.g1, .end = Squares.f3 }, Colors.white, Pieces.knight);
    state.makeMove(game.Move{ .start = Squares.d8, .end = Squares.e7 }, Colors.black, Pieces.queen);
    state.makeMove(game.Move{ .start = Squares.f1, .end = Squares.e2 }, Colors.white, Pieces.bishop);
    state.makeMove(game.Move{ .start = Squares.e7, .end = Squares.d8 }, Colors.black, Pieces.queen);
    state.makeMove(game.Move{ .start = Squares.e1, .end = Squares.f1 }, Colors.white, Pieces.king);
    try expectEqual(Castling.AllLegal ^ Castling.WhiteCastling, state.castling_rights);

    state = State.defaultPosition();
    state.makeMove(game.Move{ .start = Squares.a2, .end = Squares.a3 }, Colors.white, Pieces.pawn);
    state.makeMove(game.Move{ .start = Squares.a7, .end = Squares.a6 }, Colors.black, Pieces.pawn);
    state.makeMove(game.Move{ .start = Squares.a1, .end = Squares.a2 }, Colors.white, Pieces.rook);
    try expectEqual(Castling.AllLegal ^ Castling.WhiteQueenside, state.castling_rights);
}

test "test fen from default" {
    const starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const state = try State.fromFen(starting_fen);
    const default_state = State.defaultPosition();

    try expectEqual(default_state.pieces, state.pieces);
    try expectEqual(default_state.colors, state.colors);
    try expectEqual(default_state.to_move, state.to_move);
    try expectEqual(default_state.castling_rights, state.castling_rights);
    try expectEqual(default_state.en_passant, state.en_passant);
    try expectEqual(default_state.in_check, state.in_check);
    try expectEqual(default_state.halfmove_clock, state.halfmove_clock);
    try expectEqual(default_state.fullmove_clock, state.fullmove_clock);
}

test "test fen from e4 c5 nf3" {
    const fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";
    const state = try State.fromFen(fen);

    const pawns = state.pieceBitboard(Pieces.pawn);
    const knights = state.pieceBitboard(Pieces.knight);
    const bishops = state.pieceBitboard(Pieces.bishop);
    const rooks = state.pieceBitboard(Pieces.rook);
    const queens = state.pieceBitboard(Pieces.queen);
    const kings = state.pieceBitboard(Pieces.king);

    try expectEqual(Bitboard{ .bits = 0xfb00041000ef00}, pawns);
    try expectEqual(Bitboard{ .bits = 0x4200000000200002}, knights);
    try expectEqual(Bitboard{ .bits = 0x2400000000000024}, bishops);
    try expectEqual(Bitboard{ .bits = 0x8100000000000081}, rooks);
    try expectEqual(Bitboard{ .bits = 0x800000000000008}, queens);
    try expectEqual(Bitboard{ .bits = 0x1000000000000010}, kings);

    const white_pieces = state.colorBitboard(Colors.white);
    const black_pieces = state.colorBitboard(Colors.black);

    try expectEqual(Bitboard{ .bits = 0x1020efbf}, white_pieces);
    try expectEqual(Bitboard{ .bits = 0xfffb000400000000}, black_pieces);

    try expectEqual(Colors.black, state.to_move);
    try expectEqual(Castling.AllLegal, state.castling_rights);
    try expectEqual(null, state.en_passant);
    try expectEqual(1, state.halfmove_clock);
    try expectEqual(2, state.fullmove_clock);
}
