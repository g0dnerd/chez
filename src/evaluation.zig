const std = @import("std");
const State = @import("State.zig");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const Colors = game.Colors;
const Color = Colors.Color;
const Pieces = game.Pieces;

const PIECE_VALUES = [6]i32{ 100, 305, 333, 563, 950, 20000 };
const POSITIONAL_SCORES = [6][64]i32{
    // Pawns
    [_]i32{
        0,  0,  0,  0,  0, 0, 0, 0, 5,  10, 10, -20, -20, 10, 10, 5,  5,  -5, -10, 0,  0,  -10, -5, 5,  0,  0,
        0,  20, 20, 0,  0, 0, 5, 5, 10, 25, 25, 10,  5,   5,  10, 10, 20, 30, 30,  20, 10, 10,  50, 50, 50, 50,
        50, 50, 50, 50, 0, 0, 0, 0, 0,  0,  0,  0,
    },
    // Knights
    [_]i32{
        -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0,   5,   5,   0,   -20, -40, -30, 5,   10,  15,  15,
        10,  5,   -30, -30, 0,   15,  20,  20,  15,  0,   -30, -30, 5,   15,  20,  20,  15,  5,   -30, -30, 0,
        10,  15,  15,  10,  0,   -30, -40, -20, 0,   0,   0,   0,   -20, -40, -50, -40, -30, -30, -30, -30, -40,
        -50,
    },
    // Bishops
    [_]i32{
        -20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0,   0,   0, 0,   5,   -10, -10, 10,  10,  10,  10,
        10,  10,  -10, -10, 0,   10,  10,  10,  10,  0, -10, -10, 5, 5,   10,  10,  5,   5,   -10, -10, 0,
        5,   10,  10,  5,   0,   -10, -10, 0,   0,   0, 0,   0,   0, -10, -20, -10, -10, -10, -10, -10, -10,
        -20,
    },
    // Rooks
    [_]i32{
        0, 0, 0,  5,  5, 0, 0, 0, -5, 0, 0,  0,  0, 0, 0, -5, -5, 0, 0,  0, 0,  0,  0,  -5, -5, 0,  0, 0, 0,
        0, 0, -5, -5, 0, 0, 0, 0, 0,  0, -5, -5, 0, 0, 0, 0,  0,  0, -5, 5, 10, 10, 10, 10, 10, 10, 5, 0, 0,
        0, 0, 0,  0,  0, 0,
    },
    // Queens
    [_]i32{
        -20, -10, -10, -5, -5, -10, -10, -20, -10, 0,   5,   0,   0,   0,  0,  -10, -10, 5,   5, 5, 5, 5, 0,
        -10, 0,   0,   5,  5,  5,   5,   0,   -10, -5,  0,   5,   5,   5,  5,  0,   -10, -10, 0, 5, 5, 5, 5,
        0,   -10, -10, 0,  0,  0,   0,   0,   0,   -10, -20, -10, -10, -5, -5, -10, -10, -20,
    },
    // Kings
    [_]i32{
        20,  30,  10,  0,   0,   10,  30,  20,  20,  20,  0,   0,   0,   0,   20,  20,  -10, -20, -20, -20, -20, -20,
        -20, -10, -20, -30, -30, -40, -40, -30, -30, -20, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50,
        -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
    },
};

const KING_ENDGAME = [64]i32{
    -50, -30, -30, -30, -30, -30, -30, -50, -30, -10, 0,   0,   0,   0,   -10, -30, -30, 0,   20,  30,  30, 20,
    0,   -30, -30, 0,   30,  40,  40,  30,  0,   -30, -30, 0,   30,  40,  40,  30,  0,   -30, -30, 0,   20, 30,
    30,  20,  0,   -30, -30, -10, 0,   0,   0,   0,   -10, -30, -50, -30, -30, -30, -30, -30, -30, -50,
};

fn materialCount(state: *const State, c: Color) i32 {
    const pieces = state.colorBitboard(c);
    const pawns = state.pieceBitboard(Pieces.pawn).bitAnd(pieces);
    const knights = state.pieceBitboard(Pieces.knight).bitAnd(pieces);
    const bishops = state.pieceBitboard(Pieces.bishop).bitAnd(pieces);
    const rooks = state.pieceBitboard(Pieces.rook).bitAnd(pieces);
    const queens = state.pieceBitboard(Pieces.queen).bitAnd(pieces);

    var p: i32 = @intCast(pawns.popCount());
    p *= 100;
    var n: i32 = @intCast(knights.popCount());
    n *= 305;
    var b: i32 = @intCast(bishops.popCount());
    b *= 333;
    var r: i32 = @intCast(rooks.popCount());
    r *= 563;
    var q: i32 = @intCast(queens.popCount());
    q *= 950;

    return p + n + b + r + q;
}

fn positionalScore(state: *const State, c: Color, is_endgame: bool) i32 {
    var score: i32 = 0;

    var piece: Pieces.Piece = 0;
    while (piece < 6) {
        var pieces = state.colorBitboard(c).bitAnd(state.pieceBitboard(piece));
        defer piece += 1;

        while (pieces.next()) |s| {
            const sq = blk: {
                if (c == Colors.black) {
                    const rank = s / 8;
                    const file = s % 8;
                    break :blk (7 - rank) * 8 + file;
                } else {
                    break :blk s;
                }
            };

            if (piece == Pieces.king and is_endgame) {
                score += KING_ENDGAME[sq];
            } else {
                score += POSITIONAL_SCORES[piece][sq];
            }
        }
    }
    return score;
}

fn pawnStructureScore(state: *const State, c: Color) i32 {
    var score: i32 = 0;
    var pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(c));

    while (pawns.next()) |s| {
        const file = s % 8;
        const rank = s / 8;

        // Passed pawn bonus
        const ahead_mask = blk: {
            if (c == Colors.white) {
                if (rank < 7) {
                    const lhs: u64 = @as(u64, 0xFF) << ((rank + 1) * 8);
                    const rhs: u64 = rhs: {
                        if (file > 0) {
                            break :rhs @as(u64, 0x0101010101010101) << (file - 1);
                        } else {
                            break :rhs 0;
                        }
                    };
                    const inner: u64 = inner: {
                        if (file < 7) {
                            break :inner @as(u64, 0x0101010101010101) << file | @as(u64, 0x0101010101010101) << (file + 1);
                        } else {
                            break :inner @as(u64, 0x0101010101010101) << file | 0;
                        }
                    };
                    break :blk lhs & (rhs | inner);
                } else {
                    break :blk 0;
                }
            } else if (rank > 0) {
                const lhs: u64 = (@as(u64, 1) << (rank * 8)) - 1;
                const rhs: u64 = rhs: {
                    if (file > 0) {
                        break :rhs @as(u64, 0x0101010101010101) << (file - 1);
                    } else {
                        break :rhs 0;
                    }
                };
                const inner: u64 = inner: {
                    if (file < 7) {
                        break :inner @as(u64, 0x0101010101010101) << file | @as(u64, 0x0101010101010101) << (file + 1);
                    } else {
                        break :inner @as(u64, 0x0101010101010101) << file | 0;
                    }
                };
                break :blk lhs & (rhs | inner);
            } else {
                break :blk 0;
            }
        };

        const opp_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(~c));
        if ((opp_pawns.bits & ahead_mask) == 0) {
            const passed_rank: u6 = if (c == Colors.white)
                rank
            else
                7 - rank;

            score += 10 + @as(i32, passed_rank) * 10;
        }

        // Doubled pawn penalty
        const file_mask = @as(u64, 0x0101010101010101) << file;
        if (@popCount(pawns.bits & file_mask) > 1) {
            score -= 10;
        }
    }

    return score;
}

fn mobilityScore(state: *const State, c: Color) i32 {
    var score: i32 = 0;
    const pieces = state.colorBitboard(c);

    var knights = state.pieceBitboard(Pieces.knight).bitAnd(pieces);
    while (knights.next()) |s| {
        const moves = movegen.KNIGHT_MOVES[s].bitAnd(pieces.not());
        const numMoves: i32 = @intCast(moves.popCount());
        score += numMoves;
    }

    var bishops = state.pieceBitboard(Pieces.bishop).bitAnd(pieces);
    while (bishops.next()) |s| {
        const moves = movegen.sliderMoves(state, s, Pieces.bishop).bitAnd(pieces.not());
        const numMoves: i32 = @intCast(moves.popCount());
        score += numMoves;
    }

    var rooks = state.pieceBitboard(Pieces.rook).bitAnd(pieces);
    while (rooks.next()) |s| {
        const moves = movegen.sliderMoves(state, s, Pieces.rook).bitAnd(pieces.not());
        const numMoves: i32 = @intCast(moves.popCount());
        score += numMoves;
    }

    return @divTrunc(score, 4);
}

pub fn evaluate(state: *const State) i32 {
    const to_move = state.to_move;
    const opp = ~to_move;

    const our_material = materialCount(state, to_move);
    const opp_material = materialCount(state, opp);
    const total_material = our_material + opp_material;
    const is_endgame = total_material < 2500;

    const our_mobility = mobilityScore(state, to_move);
    const opp_mobility = mobilityScore(state, ~to_move);
    const our_pawn_structure = pawnStructureScore(state, to_move);
    const opp_pawn_structure = pawnStructureScore(state, ~to_move);
    const our_position = positionalScore(state, to_move, is_endgame);
    const opp_position = positionalScore(state, opp, is_endgame);

    return our_material + our_position + our_pawn_structure + our_mobility - opp_material - opp_position - opp_pawn_structure - opp_mobility;
}

pub fn scoreMove(ctx: *const movegen.MoveList.SortCtx, m: game.Move) i32 {
    var score: i32 = 0;

    // MVV-LVA for captures
    if (ctx.state.pieceAt(m.end)) |captured_piece| {
        const attacker_piece = ctx.state.pieceAt(m.start).?;
        score += PIECE_VALUES[captured_piece] * 10 - PIECE_VALUES[attacker_piece];
    }

    // Promotion bonus
    const end_rank = m.end / 8;
    if (ctx.state.pieceAt(m.start) == game.Pieces.pawn and
        ((end_rank == 7 and ctx.color == Colors.white) or (end_rank == 0 and ctx.color == Colors.black)))
    {
        score += 5000;
    }

    // Killer move bonus (below captures, above quiet moves)
    if (ctx.killers[0]) |k| {
        if (k.start == m.start and k.end == m.end) {
            score += 900;
        }
    }
    if (ctx.killers[1]) |k| {
        if (k.start == m.start and k.end == m.end) {
            score += 800;
        }
    }

    return score;
}
