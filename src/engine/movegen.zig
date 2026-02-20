const std = @import("std");

const ffi = @import("../ffi.zig");
const magics = @import("generated/magics.zig");
const moves = @import("generated/moves.zig");
const Bitboard = @import("Bitboard.zig");
const engine = @import("engine.zig");
const Color = engine.Color;
const Colors = engine.Colors;
const Move = engine.Move;
const State = @import("State.zig");
const castling = @import("castling.zig");
const piece = @import("piece.zig");
const Piece = piece.Piece;
const square = @import("square.zig");
const Square = square.Square;

pub const pawn_attack_mask = [2][8]u64{ [_]u64{
    0x200,
    0x500,
    0xA00,
    0x1400,
    0x2800,
    0x5000,
    0xA000,
    0x4000,
}, [_]u64{
    0x2,
    0x5,
    0xA,
    0x14,
    0x28,
    0x50,
    0xa0,
    0x40,
} };

pub const knight_move_mask = [64]u64{
    0x20400,
    0x50800,
    0xa1100,
    0x142200,
    0x284400,
    0x508800,
    0xa01000,
    0x402000,
    0x2040004,
    0x5080008,
    0xa110011,
    0x14220022,
    0x28440044,
    0x50880088,
    0xa0100010,
    0x40200020,
    0x204000402,
    0x508000805,
    0xa1100110a,
    0x1422002214,
    0x2844004428,
    0x5088008850,
    0xa0100010a0,
    0x4020002040,
    0x20400040200,
    0x50800080500,
    0xa1100110a00,
    0x142200221400,
    0x284400442800,
    0x508800885000,
    0xa0100010a000,
    0x402000204000,
    0x2040004020000,
    0x5080008050000,
    0xa1100110a0000,
    0x14220022140000,
    0x28440044280000,
    0x50880088500000,
    0xa0100010a00000,
    0x40200020400000,
    0x204000402000000,
    0x508000805000000,
    0xa1100110a000000,
    0x1422002214000000,
    0x2844004428000000,
    0x5088008850000000,
    0xa0100010a0000000,
    0x4020002040000000,
    0x400040200000000,
    0x800080500000000,
    0x1100110a00000000,
    0x2200221400000000,
    0x4400442800000000,
    0x8800885000000000,
    0x100010a000000000,
    0x2000204000000000,
    0x4020000000000,
    0x8050000000000,
    0x110a0000000000,
    0x22140000000000,
    0x44280000000000,
    0x88500000000000,
    0x10a00000000000,
    0x20400000000000,
};

pub const king_move_mask = [64]u64{
    0x302,
    0x705,
    0xE0A,
    0x1C14,
    0x3828,
    0x7050,
    0xE0A0,
    0xC040,
    0x30203,
    0x70507,
    0xE0A0E,
    0x1C141C,
    0x382838,
    0x705070,
    0xE0A0E0,
    0xC040C0,
    0x3020300,
    0x7050700,
    0xE0A0E00,
    0x1C141C00,
    0x38283800,
    0x70507000,
    0xE0A0E000,
    0xC040C000,
    0x302030000,
    0x705070000,
    0xE0A0E0000,
    0x1C141C0000,
    0x3828380000,
    0x7050700000,
    0xE0A0E00000,
    0xC040C00000,
    0x30203000000,
    0x70507000000,
    0xE0A0E000000,
    0x1C141C000000,
    0x382838000000,
    0x705070000000,
    0xE0A0E0000000,
    0xC040C0000000,
    0x3020300000000,
    0x7050700000000,
    0xE0A0E00000000,
    0x1C141C00000000,
    0x38283800000000,
    0x70507000000000,
    0xE0A0E000000000,
    0xC040C000000000,
    0x302030000000000,
    0x705070000000000,
    0xE0A0E0000000000,
    0x1C141C0000000000,
    0x3828380000000000,
    0x7050700000000000,
    0xE0A0E00000000000,
    0xC040C00000000000,
    0x203000000000000,
    0x507000000000000,
    0xA0E000000000000,
    0x141C000000000000,
    0x2838000000000000,
    0x5070000000000000,
    0xA0E0000000000000,
    0x40C0000000000000,
};

pub const MoveList = struct {
    moves: [256]engine.Move = undefined,
    scores: [256]i32 align(32) = undefined,
    len: u8 = 0,

    pub fn append(self: *MoveList, m: Move) void {
        self.moves[self.len] = m;
        self.len += 1;
    }

    pub const SortCtx = struct {
        state: *const State,
        color: Color,
        killers: [2]?Move,
        history: ?*const engine.evaluation.HistoryTable,
        countermove: ?Move = null,
        tt_move: ?Move = null,
    };

    // Pre-compute scores for all moves (one scoreMove call per move).
    pub fn scoreAll(self: *MoveList, ctx: *const SortCtx) void {
        for (0..self.len) |i| {
            self.scores[i] = engine.evaluation.scoreMove(ctx, self.moves[i]);
        }
    }

    // Incremental selection: find the best-scored move from index..len,
    // swap it to position index. Used instead of a full sort so only
    // the moves actually examined get ordered (alpha-beta cuts early).
    // Uses @Vector(8, i32) SIMD reduction to scan scores 8 at a time.
    pub fn pickNext(self: *MoveList, index: usize) Move {
        var best_idx = index;
        var best_score = self.scores[index];

        const start = index + 1;
        const end: usize = self.len;
        if (start < end) {
            const count = end - start;
            const simd_len = 8;
            const simd_count = count / simd_len;
            var i = start;

            for (0..simd_count) |_| {
                const chunk: @Vector(8, i32) = self.scores[i..][0..8].*;
                const max_val = @reduce(.Max, chunk);
                if (max_val > best_score) {
                    for (0..8) |j| {
                        if (self.scores[i + j] > best_score) {
                            best_score = self.scores[i + j];
                            best_idx = i + j;
                        }
                    }
                }
                i += simd_len;
            }

            // Scalar tail
            while (i < end) : (i += 1) {
                if (self.scores[i] > best_score) {
                    best_score = self.scores[i];
                    best_idx = i;
                }
            }
        }

        if (best_idx != index) {
            const tmp_move = self.moves[index];
            const tmp_score = self.scores[index];
            self.moves[index] = self.moves[best_idx];
            self.moves[best_idx] = tmp_move;
            self.scores[index] = self.scores[best_idx];
            self.scores[best_idx] = tmp_score;
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

pub fn pawnAttacks(s: Square, c: Color) u64 {
    const rank = s / 8;
    if (rank == State.pawn_promo_rank[c]) {
        return 0;
    }

    const file = s % 8;

    const rank_idx = switch (c) {
        Colors.white => rank,
        Colors.black => rank - 1,
    };

    return pawn_attack_mask[c][file] << (8 * rank_idx);
}

// Possible pawn moves that do not check positional legality (e.g. whether or not your king would
// be left in check after making a move).
pub fn pawnMoves(state: *const State, s: Square, c: Color) u64 {
    var ret: u64 = 0;

    const direction: i3 = switch (c) {
        Colors.white => 1,
        Colors.black => -1,
    };

    // Check if the square one ahead is within bounds
    var offs = square.trySquareOffset(s, 0, direction);
    if (offs) |o| {
        if (state.isSquareEmpty(o)) {
            ret |= (@as(u64, 1) << o);
            const rank = s / 8;
            if ((rank == 1 and c == Colors.white) or (rank == 6 and c == Colors.black)) {
                const two_ahead: u6 = @intCast(@as(i8, s) + 16 * @as(i8, direction));
                if (state.isSquareEmpty(two_ahead)) {
                    ret |= (@as(u64, 1) << two_ahead);
                }
            }
        }
    }

    // Check for captures
    offs = square.trySquareOffset(s, -1, direction);
    if (offs) |o| {
        if (!state.isSquareEmpty(o) or state.en_passant == offs) {
            ret |= (@as(u64, 1) << o);
        }
    }
    offs = square.trySquareOffset(s, 1, direction);
    if (offs) |o| {
        if (!state.isSquareEmpty(o) or state.en_passant == offs) {
            ret |= (@as(u64, 1) << o);
        }
    }

    return ret;
}

pub fn sliderMoves(state: *const State, s: Square, p: Piece) u64 {
    // Pass all_pieces directly - magicTableIndex applies the entry's mask internally,
    // so pre-masking in a separate blockersFromState was redundant.
    const all = state.all_pieces;

    return blk: switch (p) {
        piece.rook => break :blk moves.rook_moves[magics.rook_magics[s].magicTableIndex(all)],
        piece.bishop => break :blk moves.bishop_moves[magics.bishop_magics[s].magicTableIndex(all)],
        piece.queen => {
            const rookMoves = moves.rook_moves[magics.rook_magics[s].magicTableIndex(all)];
            const bishopMoves = moves.bishop_moves[magics.bishop_magics[s].magicTableIndex(all)];
            break :blk rookMoves | bishopMoves;
        },
        else => unreachable,
    };
}

pub fn kingMoves(state: *const State, s: Square, c: Color) u64 {
    var ret = king_move_mask[s];

    if (state.in_check == null) {
        const castling_rights = state.castling_rights;
        switch (c) {
            Colors.white => {
                if (castling_rights & castling.white_kingside != 0 and
                    state.isSquareEmpty(square.f1) and
                    state.isSquareEmpty(square.g1) and
                    state.colorBitboard(Colors.white).contains(square.h1) and
                    state.pieceBitboard(piece.rook).contains(square.h1))
                {
                    ret |= (@as(u64, 1) << square.g1);
                }
                if (castling_rights & castling.white_queenside != 0 and
                    state.isSquareEmpty(square.b1) and
                    state.isSquareEmpty(square.c1) and
                    state.isSquareEmpty(square.d1) and
                    state.colorBitboard(Colors.white).contains(square.a1) and
                    state.pieceBitboard(piece.rook).contains(square.a1))
                {
                    ret |= (@as(u64, 1) << square.c1);
                }
            },
            Colors.black => {
                if (castling_rights & castling.black_kingside != 0 and state.isSquareEmpty(square.f8) and state.isSquareEmpty(square.g8) and state.colorBitboard(Colors.black).contains(square.h8) and state.pieceBitboard(piece.rook).contains(square.h8)) {
                    ret |= (@as(u64, 1) << square.g8);
                }
                if (castling_rights & castling.black_queenside != 0 and state.isSquareEmpty(square.b8) and state.isSquareEmpty(square.c8) and state.isSquareEmpty(square.d8) and state.colorBitboard(Colors.black).contains(square.a8) and state.pieceBitboard(piece.rook).contains(square.a8)) {
                    ret |= (@as(u64, 1) << square.c8);
                }
            },
        }
    }

    var opp_king_mask = state.pieceBitboard(piece.king).bits & state.colorBitboard(~c).bits;
    const opp_king_square: Square = @intCast(@ctz(opp_king_mask));
    opp_king_mask |= king_move_mask[opp_king_square];

    return ret & ~opp_king_mask;
}

pub fn pseudolegalForPiece(state: *const State, s: Square, c: Color, p: Piece) u64 {
    return switch (p) {
        // Keep only pawn attacks that point at an opposing piece
        piece.pawn => pawnAttacks(s, c) & state.colorBitboard(~c).bits | pawnMoves(state, s, c),
        piece.knight => knight_move_mask[s],
        piece.bishop, piece.rook, piece.queen => sliderMoves(state, s, p),
        piece.king => kingMoves(state, s, c),
        else => unreachable,
    };
}

pub fn isSquareAttackedBy(state: *const State, s: Square, by_color: Color) bool {
    const attackers = state.colorBitboard(by_color).bits;

    const pawn_attackers = pawnAttacks(s, ~by_color) & state.pieceBitboard(piece.pawn).bits & attackers;
    if (pawn_attackers != 0) return true;

    const knight_attackers = knight_move_mask[s] & state.pieceBitboard(piece.knight).bits & attackers;
    if (knight_attackers != 0) return true;

    const king_square = state.pieceBitboard(piece.king).bitAnd(u64, attackers).trailingZeros();
    if (Bitboard.contains_u64(king_move_mask[king_square], s)) return true;

    // Check slider attacks (bishops, rooks, queens)
    const queens = state.pieceBitboard(piece.queen).bits;

    const bishop_attacks = sliderMoves(state, s, piece.bishop);
    const bishop_attackers = bishop_attacks & (state.pieceBitboard(piece.bishop).bits | queens) & attackers;
    if (bishop_attackers != 0) return true;

    const rook_attacks = sliderMoves(state, s, piece.rook);
    const rook_attackers = rook_attacks & (state.pieceBitboard(piece.rook).bits | queens) & attackers;
    if (rook_attackers != 0) return true;

    return false;
}

pub fn movesForPiece(state: *const State, s: Square, c: Color, p: Piece) u64 {
    return pseudolegalForPiece(state, s, c, p) & ~state.colorBitboard(c).bits;
}

pub fn hasAnyLegalMove(state: *const State, c: Color) bool {
    var pieces = state.colorBitboard(c);
    const king_mask = state.pieceBitboard(piece.king).bitAnd(Bitboard, pieces);
    const king_square = king_mask.trailingZeros();
    const in_check = isSquareAttackedBy(state, king_square, ~c);

    if (in_check) {
        // Single mutable copy, reused via make/unmake for all candidate moves
        var mutable = state.*;
        while (pieces.next()) |s| {
            const p = state.mailbox[s].?;
            var piece_moves = Bitboard{ .bits = movesForPiece(state, s, c, p) };

            while (piece_moves.next()) |end| {
                const candidate_move = Move{ .start = s, .end = end };
                const undo = mutable.makeMoveNoCheck(candidate_move, c, p);

                const new_king_square = if (p == piece.king) end else king_square;
                const legal = !isSquareAttackedBy(&mutable, new_king_square, ~c);

                mutable.unmakeMove(candidate_move, c, p, undo);

                if (legal) return true;
            }
        }
    } else {
        while (pieces.next()) |s| {
            const p = state.mailbox[s].?;
            var piece_moves = Bitboard{ .bits = movesForPiece(state, s, c, p) };

            while (piece_moves.next()) |end| {
                const candidate_move = Move{ .start = s, .end = end };
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
    const king_mask = state.pieceBitboard(piece.king).bitAnd(Bitboard, pieces);
    const king_square = king_mask.trailingZeros();
    const in_check = isSquareAttackedBy(state, king_square, ~c);

    if (in_check) {
        // Single mutable copy, reused via make/unmake for all candidate moves
        var mutable = state.*;
        while (pieces.next()) |s| {
            const p = state.mailbox[s] orelse unreachable;
            var piece_moves = Bitboard{ .bits = movesForPiece(state, s, c, p) };

            while (piece_moves.next()) |end| {
                var candidate_move = Move{ .start = s, .end = end };
                const undo = mutable.makeMoveNoCheck(candidate_move, c, p);

                const new_king_square = if (p == piece.king) end else king_square;
                const legal = !isSquareAttackedBy(&mutable, new_king_square, ~c);

                mutable.unmakeMove(candidate_move, c, p, undo);

                if (legal) {
                    if (p == piece.pawn and end / 8 == State.pawn_promo_rank[c]) {
                        candidate_move.promotion_piece = piece.queen;
                        candidate_move.is_promotion = true;
                        ret.append(candidate_move);
                        inline for (1..4) |promotion_target| {
                            ret.append(.{ .start = s, .end = end, .is_promotion = true, .promotion_piece = @as(Piece, promotion_target) });
                        }
                        continue;
                    }
                    ret.append(candidate_move);
                }
            }
        }
    } else {
        while (pieces.next()) |s| {
            const p = state.mailbox[s] orelse unreachable;
            var piece_moves = Bitboard{ .bits = movesForPiece(state, s, c, p) };

            while (piece_moves.next()) |end| {
                var candidate_move = Move{ .start = s, .end = end };
                if (isLegalMove(state, candidate_move, c, p, king_square)) {
                    if (p == piece.pawn and end / 8 == State.pawn_promo_rank[c]) {
                        candidate_move.promotion_piece = piece.queen;
                        candidate_move.is_promotion = true;
                        ret.append(candidate_move);
                        inline for (1..4) |promotion_target| {
                            ret.append(.{ .start = s, .end = end, .is_promotion = true, .promotion_piece = @as(Piece, promotion_target) });
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

// Generate only legal captures and promotions (for quiescence search)
pub fn legalCaptures(state: *const State, c: Color) MoveList {
    var ret = MoveList{};

    var pieces = state.colorBitboard(c);
    const king_mask = state.pieceBitboard(piece.king).bitAnd(Bitboard, pieces);
    const king_square = king_mask.trailingZeros();
    const in_check = isSquareAttackedBy(state, king_square, ~c);
    const enemy_pieces = state.colorBitboard(~c);

    if (in_check) {
        // Single mutable copy, reused via make/unmake for all candidate moves
        var mutable = state.*;
        while (pieces.next()) |s| {
            const p = state.mailbox[s] orelse unreachable;
            var piece_moves = Bitboard{ .bits = movesForPiece(state, s, c, p) };
            const promotion_rank: u6 = if (c == Colors.white) 7 else 0;

            while (piece_moves.next()) |end| {
                const is_capture = enemy_pieces.contains(end) or (p == piece.pawn and state.en_passant == end);
                const is_promo = p == piece.pawn and (end / 8) == promotion_rank;

                if (!is_capture and !is_promo) continue;

                var candidate_move = Move{ .start = s, .end = end };
                const undo = mutable.makeMoveNoCheck(candidate_move, c, p);

                const new_king_square = if (p == piece.king) end else king_square;
                const legal = !isSquareAttackedBy(&mutable, new_king_square, ~c);

                mutable.unmakeMove(candidate_move, c, p, undo);

                if (legal) {
                    if (is_promo) {
                        candidate_move.promotion_piece = piece.queen;
                        candidate_move.is_promotion = true;
                        ret.append(candidate_move);
                        inline for (1..4) |promotion_target| {
                            ret.append(.{ .start = s, .end = end, .is_promotion = true, .promotion_piece = @as(Piece, promotion_target) });
                        }
                        continue;
                    }
                    ret.append(candidate_move);
                }
            }
        }
    } else {
        while (pieces.next()) |s| {
            const p = state.mailbox[s] orelse unreachable;
            var piece_moves = Bitboard{ .bits = movesForPiece(state, s, c, p) };
            const promotion_rank: u6 = if (c == Colors.white) 7 else 0;

            while (piece_moves.next()) |end| {
                const is_capture = enemy_pieces.contains(end) or (p == piece.pawn and state.en_passant == end);
                const is_promo = p == piece.pawn and (end / 8) == promotion_rank;

                if (!is_capture and !is_promo) continue;

                var candidate_move = Move{ .start = s, .end = end };
                if (isLegalMove(state, candidate_move, c, p, king_square)) {
                    if (is_promo) {
                        candidate_move.promotion_piece = piece.queen;
                        candidate_move.is_promotion = true;
                        ret.append(candidate_move);
                        inline for (1..4) |promotion_target| {
                            ret.append(.{ .start = s, .end = end, .is_promotion = true, .promotion_piece = @as(Piece, promotion_target) });
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

fn isLegalMove(state: *const State, m: Move, c: Color, p: Piece, king_square: Square) bool {
    // King moves: check if destination is attacked
    if (p == piece.king) {
        if (square.absDiff(m.start, m.end) == 2) {
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

    if (p == piece.pawn and state.en_passant == m.end) {
        return !enPassantExposesKing(state, m, c, king_square);
    }

    return true;
}

// Check if there's a slider pinning this piece to the king.
// If so, return the ray from king to pinner (inclusive) — the piece may only move along it.
fn pinRay(state: *const State, s: Square, king_square: Square, c: Color) Bitboard {
    if (s == king_square) return Bitboard.empty;

    const direction = engine.Direction.fromSquares(king_square, s);
    const away_dir = square.toRayDirection(direction, king_square, s) orelse return Bitboard.empty;
    const toward_dir = away_dir.opposite();

    const away_idx = @intFromEnum(away_dir);
    const toward_idx = @intFromEnum(toward_dir);

    // Squares strictly between king and piece (intersection of opposing rays)
    const between_bits = square.ray_attacks[away_idx][king_square] &
        square.ray_attacks[toward_idx][s];
    if (between_bits & state.all_pieces.bits != 0) return Bitboard.empty;

    // First occupied square beyond s, away from king
    const beyond_occupied = square.ray_attacks[away_idx][s] & state.all_pieces.bits;
    if (beyond_occupied == 0) return Bitboard.empty;

    const pinner_sq: Square = if (away_dir.isPositive())
        @intCast(@ctz(beyond_occupied))
    else
        @intCast(63 - @as(u7, @clz(beyond_occupied)));

    // Check if pinner is an enemy slider of the correct type
    const enemy_sliders = switch (direction) {
        .horizontal, .vertical => state.pieceBitboard(piece.rook).bitOr(Bitboard, state.pieceBitboard(piece.queen)).bitAnd(Bitboard, state.colorBitboard(~c)),
        .diagonal, .antiDiagonal => state.pieceBitboard(piece.bishop).bitOr(Bitboard, state.pieceBitboard(piece.queen)).bitAnd(Bitboard, state.colorBitboard(~c)),
        .none => unreachable,
    };

    if (enemy_sliders.contains(pinner_sq)) {
        // Ray from king to pinner inclusive, computed from the ray table
        const king_ray = square.ray_attacks[away_idx][king_square];
        const beyond_pinner = square.ray_attacks[away_idx][pinner_sq];
        return Bitboard{ .bits = (king_ray & ~beyond_pinner) | (@as(u64, 1) << king_square) };
    }

    return Bitboard.empty;
}

fn enPassantExposesKing(state: *const State, m: Move, c: Color, king_square: Square) bool {
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
    const all_pieces = state.all_pieces.bitAnd(Bitboard, Bitboard.initSquare(m.start).not()).bitAnd(Bitboard, Bitboard.initSquare(captured_pawn_square).not());

    // Check for enemy rooks/queens on the same rank
    const enemy_pieces = state.colorBitboard(~c);
    var enemy_rooks_queens = state.pieceBitboard(piece.rook).bitOr(Bitboard, state.pieceBitboard(piece.queen)).bitAnd(Bitboard, enemy_pieces);

    // Check horizontal attacks on the king's rank
    while (enemy_rooks_queens.next()) |sq| {
        if (sq / 8 == king_rank) {
            // Check if there's a clear path between attacker and king
            const between = square.betweenSquares(sq, king_square);
            if (between.bitAnd(Bitboard, all_pieces).isEmpty()) {
                return true; // Exposed to check
            }
        }
    }

    return false;
}

test "test get pawn attacks" {
    const atx_white = pawnAttacks(square.e2, Colors.white);
    try std.testing.expectEqual(atx_white, 0x280000);

    const atx_black = pawnAttacks(square.e7, Colors.black);
    try std.testing.expectEqual(atx_black, 0x280000000000);
}

test "test pawn moves" {
    const state = State.defaultPosition();

    const moves_e2 = pawnMoves(&state, square.e2, Colors.white);
    try std.testing.expectEqual(moves_e2, 0x10100000);

    const moves_e7 = pawnMoves(&state, square.e7, Colors.black);
    try std.testing.expectEqual(moves_e7, 0x101000000000);
}

test "test slider moves" {
    const state = State.defaultPosition();

    const moves_bishop_c1 = sliderMoves(&state, square.c1, piece.bishop);
    try std.testing.expectEqual(2560, moves_bishop_c1);

    const moves_rook_h8 = sliderMoves(&state, square.h8, piece.rook);
    try std.testing.expectEqual(0x4080000000000000, moves_rook_h8);
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
        try std.testing.expect(m.start != square.d2);
    }
}

test "test legal moves cannot move into check" {
    const fen = "rn1qkbnr/ppp1pppp/8/3p4/3PP1b1/8/PPP2PPP/RNBQKBNR w KQkq - 1 3";
    const state = try State.fromFen(fen);
    const actual = legalMoves(&state, Colors.white);

    for (0..actual.len) |i| {
        const m = actual.moves[i];
        try std.testing.expect(!(m.start == square.e1 and m.end == square.e2));
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
        if (actual.moves[i].start == square.e4) {
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
        if (actual.moves[i].start == square.g1) {
            knight_moves += 1;
            // Verify f3 and h3 are NOT in the move list
            try std.testing.expect(actual.moves[i].end != square.f3);
            try std.testing.expect(actual.moves[i].end != square.h3);
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
        if (actual.moves[i].start == square.e1 and actual.moves[i].end == square.g1) {
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
        if (actual.moves[i].start == square.e1 and actual.moves[i].end == square.c1) {
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
        if (actual.moves[i].start == square.e8 and actual.moves[i].end == square.g8) {
            found_kingside = true;
        }
        if (actual.moves[i].start == square.e8 and actual.moves[i].end == square.c8) {
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
        if (actual.moves[i].start == square.e1 and actual.moves[i].end == square.g1) {
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
        if (actual.moves[i].start == square.e1 and actual.moves[i].end == square.g1) {
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
        if (actual.moves[i].start == square.e1 and actual.moves[i].end == square.g1) {
            found_kingside = true;
        }
        if (actual.moves[i].start == square.e1 and actual.moves[i].end == square.c1) {
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
        if (actual.moves[i].start == square.e7 and actual.moves[i].end == square.e8) {
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
        if (actual.moves[i].start == square.e7 and actual.moves[i].end == square.d8) {
            found_d8 = true;
        }
        if (actual.moves[i].start == square.e7 and actual.moves[i].end == square.f8) {
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
        if (actual.moves[i].start == square.e5 and actual.moves[i].end == square.d6) {
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
        if (actual.moves[i].start == square.e5 and actual.moves[i].end == square.d6) {
            found_en_passant = true;
            break;
        }
    }
    try std.testing.expect(!found_en_passant);
}

// Static Exchange Evaluation (SEE)
// Returns the material gain/loss from a capture sequence on a square
// Positive = winning exchange, negative = losing exchange
pub const see_piece_values = [6]i32{ 126, 781, 825, 1276, 2538, 20000 };

pub fn staticExchangeEvaluation(state: *const State, m: Move) i32 {
    const target_sq = m.end;
    const attacker_sq = m.start;
    const attacker_color = state.colorAt(attacker_sq) orelse return 0;
    const attacker_piece = state.mailbox[attacker_sq] orelse return 0;

    // Get the initial captured piece value
    var gain: [32]i32 = undefined;
    var depth: usize = 0;

    // Handle initial capture (including en passant)
    const initial_victim = if (state.mailbox[target_sq]) |p|
        see_piece_values[p]
    else if (attacker_piece == piece.pawn and state.en_passant == target_sq)
        see_piece_values[piece.pawn]
    else
        return 0; // No capture

    gain[depth] = initial_victim;

    // Track occupied squares
    var occupied = state.all_pieces;
    occupied.bits &= ~(@as(u64, 1) << attacker_sq);

    // For en passant, also remove the captured pawn
    if (attacker_piece == piece.pawn and state.en_passant == target_sq) {
        const captured_pawn_sq: Square = if (attacker_color == Colors.white)
            target_sq - 8
        else
            target_sq + 8;
        occupied.bits &= ~(@as(u64, 1) << captured_pawn_sq);
    }

    // Track the value of the piece on the target square
    var piece_on_target = attacker_piece;

    // Handle promotion - attacker becomes queen
    if (attacker_piece == piece.pawn) {
        const promo_rank: u6 = if (attacker_color == Colors.white) 7 else 0;
        if (target_sq / 8 == promo_rank) {
            piece_on_target = piece.queen;
            gain[depth] += see_piece_values[piece.queen] - see_piece_values[piece.pawn];
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
        occupied.bits &= ~(@as(u64, 1) << next_sq);

        // Update piece on target
        piece_on_target = next_piece;

        // Handle promotion
        if (next_piece == piece.pawn) {
            const promo_rank: u6 = if (side_to_move == Colors.white) 7 else 0;
            if (target_sq / 8 == promo_rank) {
                piece_on_target = piece.queen;
                gain[depth] += see_piece_values[piece.queen] - see_piece_values[piece.pawn];
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
    const color_pieces = state.colorBitboard(color).bits & occupied.bits;

    // Check pawns first (least valuable)
    var pawn_attackers = Bitboard{ .bits = pawnAttacks(target_sq, ~color) & state.pieceBitboard(piece.pawn).bits & color_pieces };
    if (pawn_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = piece.pawn };
    }

    // Knights
    var knight_attackers = Bitboard{ .bits = knight_move_mask[target_sq] & state.pieceBitboard(piece.knight).bits & color_pieces };
    if (knight_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = piece.knight };
    }

    // Bishops (and diagonal queens)
    const bishop_attacks = sliderMovesWithOccupancy(target_sq, piece.bishop, occupied);
    var bishop_attackers = Bitboard{ .bits = bishop_attacks & state.pieceBitboard(piece.bishop).bits & color_pieces };
    if (bishop_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = piece.bishop };
    }

    // Rooks (and orthogonal queens)
    const rook_attacks = sliderMovesWithOccupancy(target_sq, piece.rook, occupied);
    var rook_attackers = Bitboard{ .bits = rook_attacks & state.pieceBitboard(piece.rook).bits & color_pieces };
    if (rook_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = piece.rook };
    }

    // Queens (check both diagonal and orthogonal)
    const queen_attacks = bishop_attacks | rook_attacks;
    var queen_attackers = Bitboard{ .bits = queen_attacks & state.pieceBitboard(piece.queen).bits & color_pieces };
    if (queen_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = piece.queen };
    }

    // King (only if no other attackers - king captures last)
    var king_attackers = Bitboard{ .bits = king_move_mask[target_sq] & state.pieceBitboard(piece.king).bits & color_pieces };
    if (king_attackers.next()) |sq| {
        return .{ .sq = sq, .piece = piece.king };
    }

    return null;
}

// Slider moves with custom occupancy (for SEE x-ray attacks)
pub fn sliderMovesWithOccupancy(s: Square, p: Piece, occupied: Bitboard) u64 {
    return switch (p) {
        piece.rook => moves.rook_moves[magics.rook_magics[s].magicTableIndex(occupied)],
        piece.bishop => moves.bishop_moves[magics.bishop_magics[s].magicTableIndex(occupied)],
        else => unreachable,
    };
}

test "SEE winning capture" {
    // White pawn captures black pawn - positive exchange
    const fen = "8/8/8/3p4/4P3/8/8/4K2k w - - 0 1";
    const state = try State.fromFen(fen);
    const m = Move{ .start = square.e4, .end = square.d5 };
    const score = staticExchangeEvaluation(&state, m);
    try std.testing.expectEqual(@as(i32, 126), score); // Win a pawn
}

test "SEE queen takes defended pawn" {
    // Queen takes pawn defended by pawn - bad capture
    const fen = "8/8/3p4/2p5/1Q6/8/8/4K2k w - - 0 1";
    const state = try State.fromFen(fen);
    const m = Move{ .start = square.b4, .end = square.c5 };
    const score = staticExchangeEvaluation(&state, m);
    try std.testing.expect(score < 0); // Queen for pawn is bad
}

test "SEE x-ray attack" {
    // Rook takes rook, but there's another rook behind
    const fen = "3r3r/8/8/8/8/8/3R4/3R1K1k w - - 0 1";
    const state = try State.fromFen(fen);
    const m = Move{ .start = square.d1, .end = square.d8 };
    const score = staticExchangeEvaluation(&state, m);
    try std.testing.expectEqual(@as(i32, 1276), score); // Win the rook (x-ray from a1 rook)
}
