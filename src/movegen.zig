const std = @import("std");
const Bitboard = @import("Bitboard.zig");
const magics = @import("magics.zig");
const moves = @import("moves.zig");
const game = @import("game.zig");
const precompute = @import("precompute.zig");
const State = @import("State.zig");
const Colors = game.Colors;
const Color = Colors.Color;
const Direction = game.Direction;
const magicTableIndex = precompute.magicTableIndex;
const Pieces = game.Pieces;
const Piece = Pieces.Piece;
const Squares = game.Squares;
const Square = Squares.Square;

pub const PawnAttacks = [2][8]Bitboard{ [_]Bitboard{
    Bitboard{ .bits = 0x200 },
    Bitboard{ .bits = 0x500 },
    Bitboard{ .bits = 0xA00 },
    Bitboard{ .bits = 0x1400 },
    Bitboard{ .bits = 0x2800 },
    Bitboard{ .bits = 0x5000 },
    Bitboard{ .bits = 0xA000 },
    Bitboard{ .bits = 0x4000 },
}, [_]Bitboard{
    Bitboard{ .bits = 0x2 },
    Bitboard{ .bits = 0x5 },
    Bitboard{ .bits = 0xA },
    Bitboard{ .bits = 0x14 },
    Bitboard{ .bits = 0x28 },
    Bitboard{ .bits = 0x50 },
    Bitboard{ .bits = 0xa0 },
    Bitboard{ .bits = 0x40 },
} };

pub const KnightMoves = [64]Bitboard{
    Bitboard{ .bits = 0x20400 },
    Bitboard{ .bits = 0x50800 },
    Bitboard{ .bits = 0xa1100 },
    Bitboard{ .bits = 0x142200 },
    Bitboard{ .bits = 0x284400 },
    Bitboard{ .bits = 0x508800 },
    Bitboard{ .bits = 0xa01000 },
    Bitboard{ .bits = 0x402000 },
    Bitboard{ .bits = 0x2040004 },
    Bitboard{ .bits = 0x5080008 },
    Bitboard{ .bits = 0xa110011 },
    Bitboard{ .bits = 0x14220022 },
    Bitboard{ .bits = 0x28440044 },
    Bitboard{ .bits = 0x50880088 },
    Bitboard{ .bits = 0xa0100010 },
    Bitboard{ .bits = 0x40200020 },
    Bitboard{ .bits = 0x204000402 },
    Bitboard{ .bits = 0x508000805 },
    Bitboard{ .bits = 0xa1100110a },
    Bitboard{ .bits = 0x1422002214 },
    Bitboard{ .bits = 0x2844004428 },
    Bitboard{ .bits = 0x5088008850 },
    Bitboard{ .bits = 0xa0100010a0 },
    Bitboard{ .bits = 0x4020002040 },
    Bitboard{ .bits = 0x20400040200 },
    Bitboard{ .bits = 0x50800080500 },
    Bitboard{ .bits = 0xa1100110a00 },
    Bitboard{ .bits = 0x142200221400 },
    Bitboard{ .bits = 0x284400442800 },
    Bitboard{ .bits = 0x508800885000 },
    Bitboard{ .bits = 0xa0100010a000 },
    Bitboard{ .bits = 0x402000204000 },
    Bitboard{ .bits = 0x2040004020000 },
    Bitboard{ .bits = 0x5080008050000 },
    Bitboard{ .bits = 0xa1100110a0000 },
    Bitboard{ .bits = 0x14220022140000 },
    Bitboard{ .bits = 0x28440044280000 },
    Bitboard{ .bits = 0x50880088500000 },
    Bitboard{ .bits = 0xa0100010a00000 },
    Bitboard{ .bits = 0x40200020400000 },
    Bitboard{ .bits = 0x204000402000000 },
    Bitboard{ .bits = 0x508000805000000 },
    Bitboard{ .bits = 0xa1100110a000000 },
    Bitboard{ .bits = 0x1422002214000000 },
    Bitboard{ .bits = 0x2844004428000000 },
    Bitboard{ .bits = 0x5088008850000000 },
    Bitboard{ .bits = 0xa0100010a0000000 },
    Bitboard{ .bits = 0x4020002040000000 },
    Bitboard{ .bits = 0x400040200000000 },
    Bitboard{ .bits = 0x800080500000000 },
    Bitboard{ .bits = 0x1100110a00000000 },
    Bitboard{ .bits = 0x2200221400000000 },
    Bitboard{ .bits = 0x4400442800000000 },
    Bitboard{ .bits = 0x8800885000000000 },
    Bitboard{ .bits = 0x100010a000000000 },
    Bitboard{ .bits = 0x2000204000000000 },
    Bitboard{ .bits = 0x4020000000000 },
    Bitboard{ .bits = 0x8050000000000 },
    Bitboard{ .bits = 0x110a0000000000 },
    Bitboard{ .bits = 0x22140000000000 },
    Bitboard{ .bits = 0x44280000000000 },
    Bitboard{ .bits = 0x88500000000000 },
    Bitboard{ .bits = 0x10a00000000000 },
    Bitboard{ .bits = 0x20400000000000 },
};

pub fn knightMoves(s: Square) Bitboard {
    return KnightMoves[s];
}

pub fn pawnAttacks(s: Square, c: Color) Bitboard {
    const file = @as(usize, s % 8);
    const rank = @as(usize, s / 8);
    if ((c == Colors.white and rank == 7) or (c == Colors.black and rank == 0)) {
        return Bitboard.empty();
    }

    const rank_idx: u6 = switch (c) {
        Colors.white => @intCast(rank),
        Colors.black => @intCast(rank - 1),
    };

    return PawnAttacks[c][file].shl(8 * rank_idx);
}

// Possible pawn moves that do not check positional legality (e.g. whether or not your king would
// be left in check after making a move).
pub fn pawnMoves(state: *const State, s: Square, c: Color) Bitboard {
    var ret = Bitboard.empty();

    const direction: i3 = switch (c) {
        Colors.white => 1,
        Colors.black => -1,
    };

    // Check if the square one ahead is within bounds
    var offs = game.trySquareOffset(s, 0, direction);
    if (offs != null and state.isSquareEmpty(offs.?)) {
        ret.bitOrAssign(offs.?);
        const rank = s / 8;
        if ((rank == 1 and c == Colors.white) or (rank == 6 and c == Colors.black)) {
            const two_ahead: u6 = @intCast(@as(i8, s) + 16 * @as(i8, direction));
            if (state.isSquareEmpty(two_ahead)) {
                ret.bitOrAssign(two_ahead);
            }
        }
    }

    // Check for captures
    offs = game.trySquareOffset(s, -1, direction);
    if (offs != null and (!state.isSquareEmpty(offs.?) or state.en_passant == offs)) {
        ret.bitOrAssign(offs.?);
    }
    offs = game.trySquareOffset(s, 1, direction);
    if (offs != null and (!state.isSquareEmpty(offs.?) or state.en_passant == offs)) {
        ret.bitOrAssign(offs.?);
    }

    return ret;
}

fn blockersFromState(state: *const State, s: Square, p: Piece) Bitboard {
    const blockers = switch (p) {
        Pieces.rook => Bitboard{ .bits = magics.RookMagics[s].mask },
        Pieces.bishop => Bitboard{ .bits = magics.BishopMagics[s].mask },
        Pieces.queen => Bitboard{ .bits = magics.RookMagics[s].mask | magics.BishopMagics[s].mask },
        else => unreachable,
    };
    return blockers.bitAnd(state.allPieces());
}

pub fn sliderMoves(state: *const State, s: Square, p: Piece) Bitboard {
    const blockers = blockersFromState(state, s, p);

    return blk: switch (p) {
        Pieces.rook => break :blk Bitboard{ .bits = moves.RookMoves[magicTableIndex(&magics.RookMagics[s], &blockers)] },
        Pieces.bishop => break :blk Bitboard{ .bits = moves.BishopMoves[magicTableIndex(&magics.BishopMagics[s], &blockers)] },
        Pieces.queen => {
            const rookMoves = Bitboard{ .bits = moves.RookMoves[magicTableIndex(&magics.RookMagics[s], &blockers)] };
            const bishopMoves = Bitboard{ .bits = moves.BishopMoves[magicTableIndex(&magics.BishopMagics[s], &blockers)] };
            break :blk rookMoves.bitOr(bishopMoves);
        },
        else => unreachable,
    };
}

pub fn kingMoves(state: *const State, s: Square, c: Color) Bitboard {
    var ret = Bitboard.empty();
    for (game.Slider.RookDirections) |d| {
        const dx = d[0];
        const dy = d[1];
        const offs = game.trySquareOffset(s, dx, dy);
        if (offs != null) {
            ret.bitOrAssign(offs.?);
        }
    }
    for (game.Slider.BishopDirections) |d| {
        const dx = d[0];
        const dy = d[1];
        const offs = game.trySquareOffset(s, dx, dy);
        if (offs != null) {
            ret.bitOrAssign(offs.?);
        }
    }

    if (state.in_check == null) {
        const castling_rights = state.castling_rights;
        switch (c) {
            Colors.white => {
                if (castling_rights & game.Castling.WhiteKingside != 0 and state.isSquareEmpty(Squares.f1) and state.isSquareEmpty(Squares.g1) and state.colorBitboard(Colors.white).contains(Squares.h1) and state.pieceBitboard(Pieces.rook).contains(Squares.h1)) {
                    ret.bitOrAssign(Squares.g1);
                }
                if (castling_rights & game.Castling.WhiteQueenside != 0 and state.isSquareEmpty(Squares.b1) and state.isSquareEmpty(Squares.c1) and state.isSquareEmpty(Squares.d1) and state.colorBitboard(Colors.white).contains(Squares.a1) and state.pieceBitboard(Pieces.rook).contains(Squares.a1)) {
                    ret.bitOrAssign(Squares.c1);
                }
            },
            Colors.black => {
                if (castling_rights & game.Castling.BlackKingside != 0 and state.isSquareEmpty(Squares.f8) and state.isSquareEmpty(Squares.g8) and state.colorBitboard(Colors.black).contains(Squares.h1) and state.pieceBitboard(Pieces.rook).contains(Squares.h1)) {
                    ret.bitOrAssign(Squares.g8);
                }
                if (castling_rights & game.Castling.BlackQueenside != 0 and state.isSquareEmpty(Squares.b8) and state.isSquareEmpty(Squares.c8) and state.isSquareEmpty(Squares.d8) and state.colorBitboard(Colors.black).contains(Squares.a1) and state.pieceBitboard(Pieces.rook).contains(Squares.a1)) {
                    ret.bitOrAssign(Squares.c8);
                }
            },
        }
    }

    var opp_king_mask = state.pieceBitboard(Pieces.king).bitAnd(state.colorBitboard(~c));
    const opp_king_square = opp_king_mask.trailingZeros();
    for (game.Slider.RookDirections) |d| {
        const dx = d[0];
        const dy = d[1];
        const offs = game.trySquareOffset(opp_king_square, dx, dy);
        if (offs != null) {
            opp_king_mask.bitOrAssign(offs.?);
        }
    }
    for (game.Slider.BishopDirections) |d| {
        const dx = d[0];
        const dy = d[1];
        const offs = game.trySquareOffset(opp_king_square, dx, dy);
        if (offs != null) {
            opp_king_mask.bitOrAssign(offs.?);
        }
    }

    return ret.bitAnd(opp_king_mask.not());
}

pub fn pseudolegalForPiece(state: *const State, s: Square, c: Color, p: Piece) Bitboard {
    return switch (p) {
        // Keep only pawn attacks that point at an opposing piece
        Pieces.pawn => pawnAttacks(s, c).bitAnd(state.colorBitboard(~c)).bitOr(pawnMoves(state, s, c)),
        Pieces.knight => knightMoves(s),
        Pieces.bishop, Pieces.rook, Pieces.queen => sliderMoves(state, s, p),
        Pieces.king => kingMoves(state, s, c),
        else => unreachable,
    };
}

pub fn isSquareAttackedBy(state: *const State, s: Square, by_color: Color) bool {
    const attackers = state.colorBitboard(by_color);

    const pawn_attackers = pawnAttacks(s, ~by_color).bitAnd(state.pieceBitboard(Pieces.pawn)).bitAnd(attackers);
    if (!pawn_attackers.isEmpty()) return true;

    const knight_attackers = knightMoves(s).bitAnd(state.pieceBitboard(Pieces.knight)).bitAnd(attackers);
    if (!knight_attackers.isEmpty()) return true;

    const king_square = state.pieceBitboard(Pieces.king).bitAnd(attackers).trailingZeros();
    if (game.absDiff(s, king_square) <= 1) {
        const file_diff = game.absDiff(s % 8, king_square % 8);
        const rank_diff = game.absDiff(s / 8, king_square / 8);
        if (file_diff <= 1 and rank_diff <= 1) return true;
    }

    // Check slider attacks (bishops, rooks, queens)
    const bishop_attacks = sliderMoves(state, s, Pieces.bishop);
    const bishop_attackers = bishop_attacks.bitAnd((state.pieceBitboard(Pieces.bishop).bitOr(state.pieceBitboard(Pieces.queen)))).bitAnd(attackers);
    if (!bishop_attackers.isEmpty()) return true;

    const rook_attacks = sliderMoves(state, s, Pieces.rook);
    const rook_attackers = rook_attacks.bitAnd((state.pieceBitboard(Pieces.rook).bitOr(state.pieceBitboard(Pieces.queen)))).bitAnd(attackers);
    if (!rook_attackers.isEmpty()) return true;

    return false;
}

pub fn movesForPiece(state: *const State, s: Square, c: Color, p: Piece) Bitboard {
    const ret = pseudolegalForPiece(state, s, c, p);
    return ret.bitAnd(state.colorBitboard(c).not());
}

pub fn legalMoves(alloc: std.mem.Allocator, state: *const State, c: Color) !std.ArrayList(game.Move) {
    var ret = std.ArrayList(game.Move).empty;

    var pieces = state.colorBitboard(c);
    const king_mask = state.pieceBitboard(Pieces.king).bitAnd(pieces);
    const king_square = king_mask.trailingZeros();
    const in_check = isSquareAttackedBy(state, king_square, ~c);

    var piece_iter = pieces.iter();
    while (piece_iter.next()) |s| {
        const p = state.pieceAt(s) orelse unreachable;
        var piece_moves = movesForPiece(state, s, c, p);

        var piece_move_iter = piece_moves.iter();
        while (piece_move_iter.next()) |end| {
            const candidate_move = game.Move{ .start = s, .end = end };
            if (in_check) {
                var tmp_state = state.*;
                tmp_state.makeMove(candidate_move, c, p);

                const new_king_square = blk: {
                    if (p == Pieces.king) {
                        break :blk end;
                    } else {
                        break :blk king_square;
                    }
                };

                if (!isSquareAttackedBy(&tmp_state, new_king_square, ~c)) {
                    try ret.append(alloc, candidate_move);
                }
            } else {
                if (isLegalMove(state, candidate_move, c, p, king_square)) {
                    try ret.append(alloc, candidate_move);
                }
            }
        }
    }

    return ret;
}

fn isLegalMove(state: *const State, m: game.Move, c: Color, p: Piece, king_square: Square) bool {
    // King moves: check if destination is attacked
    if (p == Pieces.king) {
        if (game.absDiff(m.start, m.end) == 2) {
            const intermediate = (m.start + m.end) / 2;
            if (isSquareAttackedBy(state, intermediate, ~c)) {
                return false;
            }
        }
        return !isSquareAttackedBy(state, m.end, ~c);
    }

    const pin_ray = pinRay(state, m.start, king_square, c);
    if (!pin_ray.isEmpty() and !pin_ray.contains(m.end)) {
        return false;
    }

    if (p == Pieces.pawn and state.en_passant == m.end) {
        return !enPassantExposesKing(state, m, c, king_square);
    }

    return true;
}

// Check if there's a slider attacking both the king and this piece
// If so, return the ray between them
fn pinRay(state: *const State, s: Square, king_square: Square, c: Color) Bitboard {
    if (s == king_square) {
        return Bitboard.empty();
    }

    const direction = Direction.fromSquares(king_square, s);

    if (direction == Direction.none) {
        return Bitboard.empty();
    }

    const between = game.betweenSquares(king_square, s);
    if (!(between.bitAnd(state.allPieces())).isEmpty()) {
        return Bitboard.empty();
    }

    const king_file = king_square % 8;
    const king_rank = king_square / 8;
    const piece_file = s % 8;
    const piece_rank = s / 8;

    const file_step: i8 = switch (std.math.order(piece_file, king_file)) {
        .gt => 1,
        .lt => -1,
        .eq => 0,
    };

    const rank_step: i8 = switch (std.math.order(piece_rank, king_rank)) {
        .gt => 1,
        .lt => -1,
        .eq => 0,
    };

    // Step from piece_square in the direction away from king to find next piece
    var current_rank = @as(i8, piece_rank) + rank_step;
    var current_file = @as(i8, piece_file) + file_step;

    var next_piece_square: ?Square = null;

    while (current_rank >= 0 and current_rank < 8 and current_file >= 0 and current_file < 8) {
        const sq: Square = @intCast(current_rank * 8 + current_file);
        if (state.allPieces().contains(sq)) {
            next_piece_square = sq;
            break;
        }
        current_rank += rank_step;
        current_file += file_step;
    }

    // Check if that piece is an enemy slider of the right type
    if (next_piece_square) |sq| {
        const enemy_sliders = switch (direction) {
            .horizontal, .vertical => state.pieceBitboard(Pieces.rook).bitOr(state.pieceBitboard(Pieces.queen)).bitAnd(state.colorBitboard(~c)),
            .diagonal, .antiDiagonal => state.pieceBitboard(Pieces.bishop).bitOr(state.pieceBitboard(Pieces.queen)).bitAnd(state.colorBitboard(~c)),
            .none => return Bitboard.empty(),
        };

        if (enemy_sliders.contains(sq)) {
            return game.rayBetweenInclusive(king_square, sq, direction);
        }
    }

    return Bitboard.empty();
}

fn enPassantExposesKing(state: *const State, m: game.Move, c: Color, king_square: Square) bool {
    const king_rank = king_square / 8;
    const move_rank = m.start / 8;

    // Only matters if king and moving pawn are on the same rank
    if (king_rank != move_rank) {
        return false;
    }

    const captured_pawn_square = blk: {
        if (c == Colors.white) {
            break :blk m.end - 8;
        } else {
            break :blk m.end + 8;
        }
    };

    // Temporarily remove both pawns and check for attacks
    const all_pieces = state.allPieces().bitAnd(Bitboard.fromSquare(m.start).not()).bitAnd(Bitboard.fromSquare(captured_pawn_square).not());

    // Check for enemy rooks/queens on the same rank
    const enemy_pieces = state.colorBitboard(~c);
    var enemy_rooks_queens = state.pieceBitboard(Pieces.rook).bitOr(state.pieceBitboard(Pieces.queen)).bitAnd(enemy_pieces);

    // Check horizontal attacks on the king's rank
    var rk_iter = enemy_rooks_queens.iter();
    while (rk_iter.next()) |sq| {
        if (sq / 8 == king_rank) {
            // Check if there's a clear path between attacker and king
            const between = game.betweenSquares(sq, king_square);
            if (between.bitAnd(all_pieces).isEmpty()) {
                return true; // Exposed to check
            }
        }
    }

    return false;
}

test "test get pawn attacks" {
    const atx_white = pawnAttacks(Squares.e2, Colors.white);
    try std.testing.expectEqual(atx_white.bits, 0x280000);

    const atx_black = pawnAttacks(Squares.e7, Colors.black);
    try std.testing.expectEqual(atx_black.bits, 0x280000000000);
}

test "test pawn moves" {
    const state = State.defaultPosition();

    const moves_e2 = pawnMoves(&state, Squares.e2, Colors.white);
    try std.testing.expectEqual(moves_e2.bits, 0x10100000);

    const moves_e7 = pawnMoves(&state, Squares.e7, Colors.black);
    try std.testing.expectEqual(moves_e7.bits, 0x101000000000);
}

test "test slider moves" {
    const state = State.defaultPosition();

    const moves_bishop_c1 = sliderMoves(&state, Squares.c1, Pieces.bishop);
    try std.testing.expectEqual(2560, moves_bishop_c1.bits);

    const moves_rook_h8 = sliderMoves(&state, Squares.h8, Pieces.rook);
    try std.testing.expectEqual(0x4080000000000000, moves_rook_h8.bits);
}

test "test legal moves from default" {
    const state = State.defaultPosition();
    const alloc = std.heap.page_allocator;

    var actual = try legalMoves(alloc, &state, state.to_move);
    defer actual.deinit(alloc);

    try std.testing.expectEqual(20, actual.items.len);
}

test "test legal moves no discovered check" {
    const fen = "rnbqk1nr/pppp1ppp/8/4p3/1b2P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
    const alloc = std.heap.page_allocator;
    const state = try State.fromFen(fen);
    const actual = try legalMoves(alloc, &state, Colors.white);

    for (actual.items) |m| {
        try std.testing.expect(m.start != Squares.d2);
    }
}

test "test legal moves cannot move into check" {
    const fen = "rn1qkbnr/ppp1pppp/8/3p4/3PP1b1/8/PPP2PPP/RNBQKBNR w KQkq - 1 3";
    const alloc = std.heap.page_allocator;
    const state = try State.fromFen(fen);
    const actual = try legalMoves(alloc, &state, Colors.white);

    for (actual.items) |m| {
        try std.testing.expect(!(m.start == Squares.e1 and m.end == Squares.e2));
    }
}
