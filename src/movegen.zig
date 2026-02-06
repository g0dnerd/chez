const std = @import("std");
const ffi = @import("ffi.zig");
const Bitboard = @import("Bitboard.zig");
const evaluation = @import("evaluation.zig");
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

pub const pawn_attack_mask = [2][8]Bitboard{ [_]Bitboard{
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

pub const knight_move_mask = [64]Bitboard{
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

pub const king_move_mask = [64]Bitboard{
    Bitboard{ .bits = 0x302 },
    Bitboard{ .bits = 0x705 },
    Bitboard{ .bits = 0xE0A },
    Bitboard{ .bits = 0x1C14 },
    Bitboard{ .bits = 0x3828 },
    Bitboard{ .bits = 0x7050 },
    Bitboard{ .bits = 0xE0A0 },
    Bitboard{ .bits = 0xC040 },
    Bitboard{ .bits = 0x30203 },
    Bitboard{ .bits = 0x70507 },
    Bitboard{ .bits = 0xE0A0E },
    Bitboard{ .bits = 0x1C141C },
    Bitboard{ .bits = 0x382838 },
    Bitboard{ .bits = 0x705070 },
    Bitboard{ .bits = 0xE0A0E0 },
    Bitboard{ .bits = 0xC040C0 },
    Bitboard{ .bits = 0x3020300 },
    Bitboard{ .bits = 0x7050700 },
    Bitboard{ .bits = 0xE0A0E00 },
    Bitboard{ .bits = 0x1C141C00 },
    Bitboard{ .bits = 0x38283800 },
    Bitboard{ .bits = 0x70507000 },
    Bitboard{ .bits = 0xE0A0E000 },
    Bitboard{ .bits = 0xC040C000 },
    Bitboard{ .bits = 0x302030000 },
    Bitboard{ .bits = 0x705070000 },
    Bitboard{ .bits = 0xE0A0E0000 },
    Bitboard{ .bits = 0x1C141C0000 },
    Bitboard{ .bits = 0x3828380000 },
    Bitboard{ .bits = 0x7050700000 },
    Bitboard{ .bits = 0xE0A0E00000 },
    Bitboard{ .bits = 0xC040C00000 },
    Bitboard{ .bits = 0x30203000000 },
    Bitboard{ .bits = 0x70507000000 },
    Bitboard{ .bits = 0xE0A0E000000 },
    Bitboard{ .bits = 0x1C141C000000 },
    Bitboard{ .bits = 0x382838000000 },
    Bitboard{ .bits = 0x705070000000 },
    Bitboard{ .bits = 0xE0A0E0000000 },
    Bitboard{ .bits = 0xC040C0000000 },
    Bitboard{ .bits = 0x3020300000000 },
    Bitboard{ .bits = 0x7050700000000 },
    Bitboard{ .bits = 0xE0A0E00000000 },
    Bitboard{ .bits = 0x1C141C00000000 },
    Bitboard{ .bits = 0x38283800000000 },
    Bitboard{ .bits = 0x70507000000000 },
    Bitboard{ .bits = 0xE0A0E000000000 },
    Bitboard{ .bits = 0xC040C000000000 },
    Bitboard{ .bits = 0x302030000000000 },
    Bitboard{ .bits = 0x705070000000000 },
    Bitboard{ .bits = 0xE0A0E0000000000 },
    Bitboard{ .bits = 0x1C141C0000000000 },
    Bitboard{ .bits = 0x3828380000000000 },
    Bitboard{ .bits = 0x7050700000000000 },
    Bitboard{ .bits = 0xE0A0E00000000000 },
    Bitboard{ .bits = 0xC040C00000000000 },
    Bitboard{ .bits = 0x203000000000000 },
    Bitboard{ .bits = 0x507000000000000 },
    Bitboard{ .bits = 0xA0E000000000000 },
    Bitboard{ .bits = 0x141C000000000000 },
    Bitboard{ .bits = 0x2838000000000000 },
    Bitboard{ .bits = 0x5070000000000000 },
    Bitboard{ .bits = 0xA0E0000000000000 },
    Bitboard{ .bits = 0x40C0000000000000 },
};

pub const MoveList = struct {
    moves: [256]game.Move = undefined,
    scores: [256]i32 = undefined,
    len: u8 = 0,

    pub fn append(self: *MoveList, m: game.Move) void {
        self.moves[self.len] = m;
        self.len += 1;
    }

    pub const SortCtx = struct {
        state: *const State,
        color: Color,
        killers: [2]?game.Move,
        history: ?*const evaluation.HistoryTable,
        countermove: ?game.Move = null,
    };

    // Pre-compute scores for all moves (one scoreMove call per move).
    pub fn scoreAll(self: *MoveList, ctx: *const SortCtx) void {
        for (0..self.len) |i| {
            self.scores[i] = evaluation.scoreMove(ctx, self.moves[i]);
        }
    }

    // Incremental selection: find the best-scored move from index..len,
    // swap it to position index. Used instead of a full sort so only
    // the moves actually examined get ordered (alpha-beta cuts early).
    pub fn pickNext(self: *MoveList, index: usize) game.Move {
        var best_idx = index;
        var best_score = self.scores[index];
        for (index + 1..self.len) |i| {
            if (self.scores[i] > best_score) {
                best_score = self.scores[i];
                best_idx = i;
            }
        }
        if (best_idx != index) {
            const tmp_move = self.moves[index];
            self.moves[index] = self.moves[best_idx];
            self.moves[best_idx] = tmp_move;
            self.scores[index] = self.scores[best_idx];
            self.scores[best_idx] = best_score;
        }
        return self.moves[index];
    }

    pub fn toCMoves(self: *const MoveList, c_moves: *ffi.CMoveList) void {
        for (0..self.len) |i| {
            self.moves[i].toCMove(&c_moves.moves[i]);
        }
        c_moves.len = self.len;
    }
};

pub fn pawnAttacks(s: Square, c: Color) Bitboard {
    const rank = s / 8;
    if (rank == State.pawn_promo_rank[c]) return Bitboard.empty;

    const file = s % 8;

    const rank_idx = switch (c) {
        Colors.white => rank,
        Colors.black => rank - 1,
    };

    return pawn_attack_mask[c][file].shl(8 * rank_idx);
}

// Possible pawn moves that do not check positional legality (e.g. whether or not your king would
// be left in check after making a move).
pub fn pawnMoves(state: *const State, s: Square, c: Color) Bitboard {
    var ret = Bitboard.empty;

    const direction: i3 = switch (c) {
        Colors.white => 1,
        Colors.black => -1,
    };

    // Check if the square one ahead is within bounds
    var offs = game.trySquareOffset(s, 0, direction);
    if (offs) |o| {
        if (state.isSquareEmpty(o)) {
            ret.bitOrAssign(o);
            const rank = s / 8;
            if ((rank == 1 and c == Colors.white) or (rank == 6 and c == Colors.black)) {
                const two_ahead: u6 = @intCast(@as(i8, s) + 16 * @as(i8, direction));
                if (state.isSquareEmpty(two_ahead)) {
                    ret.bitOrAssign(two_ahead);
                }
            }
        }
    }

    // Check for captures
    offs = game.trySquareOffset(s, -1, direction);
    if (offs) |o| {
        if (!state.isSquareEmpty(o) or state.en_passant == offs) {
            ret.bitOrAssign(o);
        }
    }
    offs = game.trySquareOffset(s, 1, direction);
    if (offs) |o| {
        if (!state.isSquareEmpty(o) or state.en_passant == offs) {
            ret.bitOrAssign(o);
        }
    }

    return ret;
}

pub fn sliderMoves(state: *const State, s: Square, p: Piece) Bitboard {
    // Pass all_pieces directly - magicTableIndex applies the entry's mask internally,
    // so pre-masking in a separate blockersFromState was redundant.
    const all = state.all_pieces;

    return blk: switch (p) {
        Pieces.rook => break :blk Bitboard{ .bits = moves.rook_moves[magicTableIndex(&magics.rook_magics[s], all)] },
        Pieces.bishop => break :blk Bitboard{ .bits = moves.bishop_moves[magicTableIndex(&magics.bishop_magics[s], all)] },
        Pieces.queen => {
            const rookMoves = Bitboard{ .bits = moves.rook_moves[magicTableIndex(&magics.rook_magics[s], all)] };
            const bishopMoves = Bitboard{ .bits = moves.bishop_moves[magicTableIndex(&magics.bishop_magics[s], all)] };
            break :blk rookMoves.bitOr(bishopMoves);
        },
        else => unreachable,
    };
}

pub fn kingMoves(state: *const State, s: Square, c: Color) Bitboard {
    var ret = king_move_mask[s];

    if (state.in_check == null) {
        const castling_rights = state.castling_rights;
        switch (c) {
            Colors.white => {
                if (castling_rights & game.Castling.white_kingside != 0 and state.isSquareEmpty(Squares.f1) and state.isSquareEmpty(Squares.g1) and state.colorBitboard(Colors.white).contains(Squares.h1) and state.pieceBitboard(Pieces.rook).contains(Squares.h1)) {
                    ret.bitOrAssign(Squares.g1);
                }
                if (castling_rights & game.Castling.white_queenside != 0 and state.isSquareEmpty(Squares.b1) and state.isSquareEmpty(Squares.c1) and state.isSquareEmpty(Squares.d1) and state.colorBitboard(Colors.white).contains(Squares.a1) and state.pieceBitboard(Pieces.rook).contains(Squares.a1)) {
                    ret.bitOrAssign(Squares.c1);
                }
            },
            Colors.black => {
                if (castling_rights & game.Castling.black_kingside != 0 and state.isSquareEmpty(Squares.f8) and state.isSquareEmpty(Squares.g8) and state.colorBitboard(Colors.black).contains(Squares.h8) and state.pieceBitboard(Pieces.rook).contains(Squares.h8)) {
                    ret.bitOrAssign(Squares.g8);
                }
                if (castling_rights & game.Castling.black_queenside != 0 and state.isSquareEmpty(Squares.b8) and state.isSquareEmpty(Squares.c8) and state.isSquareEmpty(Squares.d8) and state.colorBitboard(Colors.black).contains(Squares.a8) and state.pieceBitboard(Pieces.rook).contains(Squares.a8)) {
                    ret.bitOrAssign(Squares.c8);
                }
            },
        }
    }

    var opp_king_mask = state.pieceBitboard(Pieces.king).bitAnd(state.colorBitboard(~c));
    const opp_king_square = opp_king_mask.trailingZeros();
    opp_king_mask.bitOrAssign(king_move_mask[opp_king_square]);

    return ret.bitAnd(opp_king_mask.not());
}

pub fn pseudolegalForPiece(state: *const State, s: Square, c: Color, p: Piece) Bitboard {
    return switch (p) {
        // Keep only pawn attacks that point at an opposing piece
        Pieces.pawn => pawnAttacks(s, c).bitAnd(state.colorBitboard(~c)).bitOr(pawnMoves(state, s, c)),
        Pieces.knight => knight_move_mask[s],
        Pieces.bishop, Pieces.rook, Pieces.queen => sliderMoves(state, s, p),
        Pieces.king => kingMoves(state, s, c),
        else => unreachable,
    };
}

pub fn isSquareAttackedBy(state: *const State, s: Square, by_color: Color) bool {
    const attackers = state.colorBitboard(by_color);

    const pawn_attackers = pawnAttacks(s, ~by_color).bitAnd(state.pieceBitboard(Pieces.pawn)).bitAnd(attackers);
    if (!pawn_attackers.isEmpty()) return true;

    const knight_attackers = knight_move_mask[s].bitAnd(state.pieceBitboard(Pieces.knight)).bitAnd(attackers);
    if (!knight_attackers.isEmpty()) return true;

    const king_square = state.pieceBitboard(Pieces.king).bitAnd(attackers).trailingZeros();
    if (king_move_mask[king_square].contains(s)) return true;

    // Check slider attacks (bishops, rooks, queens)
    const queens = state.pieceBitboard(Pieces.queen);

    const bishop_attacks = sliderMoves(state, s, Pieces.bishop);
    const bishop_attackers = bishop_attacks.bitAnd(state.pieceBitboard(Pieces.bishop).bitOr(queens)).bitAnd(attackers);
    if (!bishop_attackers.isEmpty()) return true;

    const rook_attacks = sliderMoves(state, s, Pieces.rook);
    const rook_attackers = rook_attacks.bitAnd(state.pieceBitboard(Pieces.rook).bitOr(queens)).bitAnd(attackers);
    if (!rook_attackers.isEmpty()) return true;

    return false;
}

pub fn movesForPiece(state: *const State, s: Square, c: Color, p: Piece) Bitboard {
    const ret = pseudolegalForPiece(state, s, c, p);
    return ret.bitAnd(state.colorBitboard(c).not());
}

pub fn hasAnyLegalMove(state: *const State, c: Color) bool {
    var pieces = state.colorBitboard(c);
    const king_mask = state.pieceBitboard(Pieces.king).bitAnd(pieces);
    const king_square = king_mask.trailingZeros();
    const in_check = isSquareAttackedBy(state, king_square, ~c);

    while (pieces.next()) |s| {
        const p = state.pieceAt(s).?;
        var piece_moves = movesForPiece(state, s, c, p);

        while (piece_moves.next()) |end| {
            const candidate_move = game.Move{ .start = s, .end = end };
            if (in_check) {
                var tmp_state = state.*;
                _ = tmp_state.makeMove(candidate_move, c, p);

                const new_king_square = if (p == Pieces.king)
                    end
                else
                    king_square;

                if (!isSquareAttackedBy(&tmp_state, new_king_square, ~c)) {
                    return true;
                }
            } else {
                if (isLegalMove(state, candidate_move, c, p, king_square)) {
                    return true;
                }
            }
        }
    }

    return false;
}

pub fn legalMoves(state: *const State, c: Color) MoveList {
    var ret = MoveList{};

    var pieces = state.colorBitboard(c);
    const king_mask = state.pieceBitboard(Pieces.king).bitAnd(pieces);
    const king_square = king_mask.trailingZeros();
    const in_check = isSquareAttackedBy(state, king_square, ~c);

    while (pieces.next()) |s| {
        const p = state.pieceAt(s) orelse unreachable;
        var piece_moves = movesForPiece(state, s, c, p);

        while (piece_moves.next()) |end| {
            var candidate_move = game.Move{ .start = s, .end = end };

            if (in_check) {
                var tmp_state = state.*;
                _ = tmp_state.makeMove(candidate_move, c, p);

                const new_king_square = if (p == Pieces.king)
                    end
                else
                    king_square;

                if (!isSquareAttackedBy(&tmp_state, new_king_square, ~c)) {
                    if (p == Pieces.pawn and end / 8 == State.pawn_promo_rank[c]) {
                        candidate_move.promotion_piece = Pieces.queen;
                        ret.append(candidate_move);
                        inline for (1..4) |promotion_target| {
                            ret.append(.{ .start = s, .end = end, .promotion_piece = @as(Piece, promotion_target) });
                        }
                        continue;
                    }
                    ret.append(candidate_move);
                }
            } else {
                if (isLegalMove(state, candidate_move, c, p, king_square)) {
                    if (p == Pieces.pawn and end / 8 == State.pawn_promo_rank[c]) {
                        candidate_move.promotion_piece = Pieces.queen;
                        ret.append(candidate_move);
                        inline for (1..4) |promotion_target| {
                            ret.append(.{ .start = s, .end = end, .promotion_piece = @as(Piece, promotion_target) });
                        }
                        continue;
                    }
                    ret.append(candidate_move);
                }
            }
        }
    }

    return ret;
}

/// Generate only legal captures and promotions (for quiescence search)
pub fn legalCaptures(state: *const State, c: Color) MoveList {
    var ret = MoveList{};

    var pieces = state.colorBitboard(c);
    const king_mask = state.pieceBitboard(Pieces.king).bitAnd(pieces);
    const king_square = king_mask.trailingZeros();
    const in_check = isSquareAttackedBy(state, king_square, ~c);
    const enemy_pieces = state.colorBitboard(~c);

    while (pieces.next()) |s| {
        const p = state.pieceAt(s) orelse unreachable;
        var piece_moves = movesForPiece(state, s, c, p);

        // Filter to only captures (landing on enemy piece) or promotions
        const promotion_rank: u6 = if (c == Colors.white) 7 else 0;

        while (piece_moves.next()) |end| {
            const is_capture = enemy_pieces.contains(end) or (p == Pieces.pawn and state.en_passant == end);
            const is_promotion = p == Pieces.pawn and (end / 8) == promotion_rank;

            if (!is_capture and !is_promotion) continue;

            var candidate_move = game.Move{ .start = s, .end = end };
            if (in_check) {
                var tmp_state = state.*;
                _ = tmp_state.makeMove(candidate_move, c, p);

                const new_king_square = if (p == Pieces.king)
                    end
                else
                    king_square;

                if (!isSquareAttackedBy(&tmp_state, new_king_square, ~c)) {
                    if (p == Pieces.pawn and end / 8 == State.pawn_promo_rank[c]) {
                        candidate_move.promotion_piece = Pieces.queen;
                        ret.append(candidate_move);
                        inline for (1..4) |promotion_target| {
                            ret.append(.{ .start = s, .end = end, .promotion_piece = @as(Piece, promotion_target) });
                        }
                        continue;
                    }
                    ret.append(candidate_move);
                }
            } else {
                if (isLegalMove(state, candidate_move, c, p, king_square)) {
                    if (p == Pieces.pawn and end / 8 == State.pawn_promo_rank[c]) {
                        candidate_move.promotion_piece = Pieces.queen;
                        ret.append(candidate_move);
                        inline for (1..4) |promotion_target| {
                            ret.append(.{ .start = s, .end = end, .promotion_piece = @as(Piece, promotion_target) });
                        }
                        continue;
                    }
                    ret.append(candidate_move);
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
            const intermediate: Square = @intCast((@as(u8, m.start) + @as(u8, m.end)) / @as(u8, 2));
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
        return Bitboard.empty;
    }

    const direction = Direction.fromSquares(king_square, s);

    if (direction == Direction.none) {
        return Bitboard.empty;
    }

    const between = game.betweenSquares(king_square, s);
    if (!(between.bitAnd(state.all_pieces)).isEmpty()) {
        return Bitboard.empty;
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
        if (state.all_pieces.contains(sq)) {
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
            .none => return Bitboard.empty,
        };

        if (enemy_sliders.contains(sq)) {
            return game.rayBetweenInclusive(king_square, sq, direction);
        }
    }

    return Bitboard.empty;
}

fn enPassantExposesKing(state: *const State, m: game.Move, c: Color, king_square: Square) bool {
    const king_rank = king_square / 8;
    const move_rank = m.start / 8;

    // Only matters if king and moving pawn are on the same rank
    if (king_rank != move_rank) {
        return false;
    }

    const captured_pawn_square = if (c == Colors.white)
        m.end - 8
    else
        m.end + 8;

    // Temporarily remove both pawns and check for attacks
    const all_pieces = state.all_pieces.bitAnd(Bitboard.fromSquare(m.start).not()).bitAnd(Bitboard.fromSquare(captured_pawn_square).not());

    // Check for enemy rooks/queens on the same rank
    const enemy_pieces = state.colorBitboard(~c);
    var enemy_rooks_queens = state.pieceBitboard(Pieces.rook).bitOr(state.pieceBitboard(Pieces.queen)).bitAnd(enemy_pieces);

    // Check horizontal attacks on the king's rank
    while (enemy_rooks_queens.next()) |sq| {
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
    const actual = legalMoves(&state, state.to_move);
    try std.testing.expectEqual(20, actual.len);
}

test "test legal moves no discovered check" {
    const fen = "rnbqk1nr/pppp1ppp/8/4p3/1b2P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    for (0..actual.len) |i| {
        const m = actual.moves[i];
        try std.testing.expect(m.start != Squares.d2);
    }
}

test "test legal moves cannot move into check" {
    const fen = "rn1qkbnr/ppp1pppp/8/3p4/3PP1b1/8/PPP2PPP/RNBQKBNR w KQkq - 1 3";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    for (0..actual.len) |i| {
        const m = actual.moves[i];
        try std.testing.expect(!(m.start == Squares.e1 and m.end == Squares.e2));
    }
}

// Knight move tests
test "knight moves from center" {
    const fen = "8/8/8/8/4N3/8/8/4K2k w - - 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    // Knight on e4 should have 8 possible squares: d2, f2, c3, g3, c5, g5, d6, f6
    var knight_moves: u8 = 0;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e4) {
            knight_moves += 1;
        }
    }
    try std.testing.expectEqual(@as(u8, 8), knight_moves);
}

test "knight moves blocked by own pieces" {
    const fen = "8/8/8/8/8/5P1P/8/4K1Nk w - - 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    // Knight on g1 with pawns on f3 and h3 - should have fewer moves
    var knight_moves: u8 = 0;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.g1) {
            knight_moves += 1;
            // Verify f3 and h3 are NOT in the move list
            try std.testing.expect(actual.moves[i].end != Squares.f3);
            try std.testing.expect(actual.moves[i].end != Squares.h3);
        }
    }
    // g1 knight normally has 3 moves (e2, f3, h3), but f3 and h3 blocked = 1 move (e2)
    try std.testing.expectEqual(@as(u8, 1), knight_moves);
}

// Castling legality tests
test "white kingside castling legal" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    var found_kingside = false;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e1 and actual.moves[i].end == Squares.g1) {
            found_kingside = true;
            break;
        }
    }
    try std.testing.expect(found_kingside);
}

test "white queenside castling legal" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    var found_queenside = false;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e1 and actual.moves[i].end == Squares.c1) {
            found_queenside = true;
            break;
        }
    }
    try std.testing.expect(found_queenside);
}

test "black castling legal" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.black);

    var found_kingside = false;
    var found_queenside = false;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e8 and actual.moves[i].end == Squares.g8) {
            found_kingside = true;
        }
        if (actual.moves[i].start == Squares.e8 and actual.moves[i].end == Squares.c8) {
            found_queenside = true;
        }
    }
    try std.testing.expect(found_kingside);
    try std.testing.expect(found_queenside);
}

test "castling blocked by piece" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3KB1R w KQkq - 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    // Bishop on f1 blocks kingside castling
    var found_kingside = false;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e1 and actual.moves[i].end == Squares.g1) {
            found_kingside = true;
            break;
        }
    }
    try std.testing.expect(!found_kingside);
}

test "castling through attacked square illegal" {
    const fen = "1k3r2/8/8/8/8/8/8/R3K2R w KQ - 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    // Rook on f8 attacks f1 - kingside castling should be illegal
    var found_kingside = false;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e1 and actual.moves[i].end == Squares.g1) {
            found_kingside = true;
            break;
        }
    }
    try std.testing.expect(!found_kingside);
}

test "castling while in check illegal" {
    const fen = "r3k2r/pppppppp/8/8/4q3/8/PPPP1PPP/R3K2R w KQkq - 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    // Queen on e4 gives check - no castling allowed
    var found_kingside = false;
    var found_queenside = false;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e1 and actual.moves[i].end == Squares.g1) {
            found_kingside = true;
        }
        if (actual.moves[i].start == Squares.e1 and actual.moves[i].end == Squares.c1) {
            found_queenside = true;
        }
    }
    try std.testing.expect(!found_kingside);
    try std.testing.expect(!found_queenside);
}

// Promotion tests
test "pawn promotion moves generated" {
    const fen = "8/4P3/8/8/8/8/8/4K2k w - - 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    // Pawn on e7 can promote to e8
    var found_promotion = false;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e7 and actual.moves[i].end == Squares.e8) {
            found_promotion = true;
            break;
        }
    }
    try std.testing.expect(found_promotion);
}

test "pawn promotion capture" {
    const fen = "3r1r2/4P3/8/8/8/8/8/4K2k w - - 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    // Pawn on e7 can capture d8 and f8 with promotion
    var found_d8 = false;
    var found_f8 = false;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e7 and actual.moves[i].end == Squares.d8) {
            found_d8 = true;
        }
        if (actual.moves[i].start == Squares.e7 and actual.moves[i].end == Squares.f8) {
            found_f8 = true;
        }
    }
    try std.testing.expect(found_d8);
    try std.testing.expect(found_f8);
}

// En passant tests
test "en passant capture generated" {
    const fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    // e5 pawn can capture d6 en passant
    var found_en_passant = false;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e5 and actual.moves[i].end == Squares.d6) {
            found_en_passant = true;
            break;
        }
    }
    try std.testing.expect(found_en_passant);
}

test "en passant illegal when pinned" {
    const fen = "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    // Horizontal pin: king on a5, pawn on e5, opponent pawn on d5, rook on h5
    // En passant would expose king to rook attack
    var found_en_passant = false;
    for (0..actual.len) |i| {
        if (actual.moves[i].start == Squares.e5 and actual.moves[i].end == Squares.d6) {
            found_en_passant = true;
            break;
        }
    }
    try std.testing.expect(!found_en_passant);
}

// Static Exchange Evaluation (SEE)
// Returns the material gain/loss from a capture sequence on a square
// Positive = winning exchange, negative = losing exchange
pub const see_piece_values = [6]i32{ 100, 320, 330, 500, 900, 20000 };

pub fn staticExchangeEvaluation(state: *const State, m: game.Move) i32 {
    const target_sq = m.end;
    const attacker_sq = m.start;
    const attacker_color = state.colorAt(attacker_sq) orelse return 0;
    const attacker_piece = state.pieceAt(attacker_sq) orelse return 0;

    // Get the initial captured piece value
    var gain: [32]i32 = undefined;
    var depth: usize = 0;

    // Handle initial capture (including en passant)
    const initial_victim = if (state.pieceAt(target_sq)) |p|
        see_piece_values[p]
    else if (attacker_piece == Pieces.pawn and state.en_passant == target_sq)
        see_piece_values[Pieces.pawn]
    else
        return 0; // No capture

    gain[depth] = initial_victim;

    // Track occupied squares
    var occupied = state.all_pieces;
    occupied.bitAndAssign(Bitboard.fromSquare(attacker_sq).not());

    // For en passant, also remove the captured pawn
    if (attacker_piece == Pieces.pawn and state.en_passant == target_sq) {
        const captured_pawn_sq: Square = if (attacker_color == Colors.white)
            target_sq - 8
        else
            target_sq + 8;
        occupied.bitAndAssign(Bitboard.fromSquare(captured_pawn_sq).not());
    }

    // Track the value of the piece on the target square
    var piece_on_target = attacker_piece;

    // Handle promotion - attacker becomes queen
    if (attacker_piece == Pieces.pawn) {
        const promo_rank: u6 = if (attacker_color == Colors.white) 7 else 0;
        if (target_sq / 8 == promo_rank) {
            piece_on_target = Pieces.queen;
            gain[depth] += see_piece_values[Pieces.queen] - see_piece_values[Pieces.pawn];
        }
    }

    var side_to_move = ~attacker_color;

    // Simulate the exchange
    while (depth < 31) {
        // Find the least valuable attacker for side_to_move
        const next_attacker = getLeastValuableAttacker(state, target_sq, side_to_move, occupied);
        if (next_attacker == null) break;

        depth += 1;

        // Negamax: gain from this capture is victim value minus what opponent can gain
        gain[depth] = see_piece_values[piece_on_target] - gain[depth - 1];

        const next_sq = next_attacker.?.sq;
        const next_piece = next_attacker.?.piece;

        // Remove attacker from occupied
        occupied.bitAndAssign(Bitboard.fromSquare(next_sq).not());

        // Update piece on target
        piece_on_target = next_piece;

        // Handle promotion
        if (next_piece == Pieces.pawn) {
            const promo_rank: u6 = if (side_to_move == Colors.white) 7 else 0;
            if (target_sq / 8 == promo_rank) {
                piece_on_target = Pieces.queen;
                gain[depth] += see_piece_values[Pieces.queen] - see_piece_values[Pieces.pawn];
            }
        }

        side_to_move = ~side_to_move;
    }

    // Minimax the gain array (from the end, each side chooses optimally)
    while (depth > 0) {
        depth -= 1;
        gain[depth] = -@max(-gain[depth], gain[depth + 1]);
    }

    return gain[0];
}

const AttackerInfo = struct {
    sq: Square,
    piece: Piece,
};

fn getLeastValuableAttacker(state: *const State, target_sq: Square, color: Color, occupied: Bitboard) ?AttackerInfo {
    const color_pieces = state.colorBitboard(color).bitAnd(occupied);

    // Check pawns first (least valuable)
    var pawn_attackers = pawnAttacks(target_sq, ~color).bitAnd(state.pieceBitboard(Pieces.pawn)).bitAnd(color_pieces);
    if (pawn_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = Pieces.pawn };
    }

    // Knights
    var knight_attackers = knight_move_mask[target_sq].bitAnd(state.pieceBitboard(Pieces.knight)).bitAnd(color_pieces);
    if (knight_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = Pieces.knight };
    }

    // Bishops (and diagonal queens)
    const bishop_attacks = sliderMovesWithOccupancy(target_sq, Pieces.bishop, occupied);
    var bishop_attackers = bishop_attacks.bitAnd(state.pieceBitboard(Pieces.bishop)).bitAnd(color_pieces);
    if (bishop_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = Pieces.bishop };
    }

    // Rooks (and orthogonal queens)
    const rook_attacks = sliderMovesWithOccupancy(target_sq, Pieces.rook, occupied);
    var rook_attackers = rook_attacks.bitAnd(state.pieceBitboard(Pieces.rook)).bitAnd(color_pieces);
    if (rook_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = Pieces.rook };
    }

    // Queens (check both diagonal and orthogonal)
    const queen_attacks = bishop_attacks.bitOr(rook_attacks);
    var queen_attackers = queen_attacks.bitAnd(state.pieceBitboard(Pieces.queen)).bitAnd(color_pieces);
    if (queen_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = Pieces.queen };
    }

    // King (only if no other attackers - king captures last)
    var king_attackers = king_move_mask[target_sq].bitAnd(state.pieceBitboard(Pieces.king)).bitAnd(color_pieces);
    if (king_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = Pieces.king };
    }

    return null;
}

// Slider moves with custom occupancy (for SEE x-ray attacks)
fn sliderMovesWithOccupancy(s: Square, p: Piece, occupied: Bitboard) Bitboard {
    return switch (p) {
        Pieces.rook => Bitboard{ .bits = moves.rook_moves[magicTableIndex(&magics.rook_magics[s], occupied)] },
        Pieces.bishop => Bitboard{ .bits = moves.bishop_moves[magicTableIndex(&magics.bishop_magics[s], occupied)] },
        else => unreachable,
    };
}

test "SEE winning capture" {
    // White pawn captures black pawn - positive exchange
    const fen = "8/8/8/3p4/4P3/8/8/4K2k w - - 0 1";
    const state = try State.fromFen(fen);
    const m = game.Move{ .start = Squares.e4, .end = Squares.d5 };
    const score = staticExchangeEvaluation(&state, m);
    try std.testing.expectEqual(@as(i32, 100), score); // Win a pawn
}

test "SEE queen takes defended pawn" {
    // Queen takes pawn defended by pawn - bad capture
    const fen = "8/8/3p4/2p5/1Q6/8/8/4K2k w - - 0 1";
    const state = try State.fromFen(fen);
    const m = game.Move{ .start = Squares.b4, .end = Squares.c5 };
    const score = staticExchangeEvaluation(&state, m);
    try std.testing.expect(score < 0); // Queen for pawn is bad
}

test "SEE x-ray attack" {
    // Rook takes rook, but there's another rook behind
    const fen = "3r3r/8/8/8/8/8/3R4/3R1K1k w - - 0 1";
    const state = try State.fromFen(fen);
    const m = game.Move{ .start = Squares.d1, .end = Squares.d8 };
    const score = staticExchangeEvaluation(&state, m);
    try std.testing.expectEqual(@as(i32, 500), score); // Win the rook (x-ray from a1 rook)
}
