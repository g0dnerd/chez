const std = @import("std");
const State = @import("State.zig");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const Colors = game.Colors;
const Color = Colors.Color;
const Pieces = game.Pieces;

const PIECE_VALUES = [6]i32{ 100, 305, 333, 563, 950, 20000 };

// Phase weights for tapered evaluation (max phase = 24)
const PHASE_WEIGHTS = [6]i32{ 0, 1, 1, 2, 4, 0 }; // pawn, knight, bishop, rook, queen, king
const MAX_PHASE: i32 = 24;

// File masks for rook on open file detection
const FILE_MASKS: [8]u64 = blk: {
    var masks: [8]u64 = undefined;
    for (0..8) |file| {
        masks[file] = @as(u64, 0x0101010101010101) << @intCast(file);
    }
    break :blk masks;
};

// Adjacent file masks for isolated pawn detection
const ADJACENT_FILES: [8]u64 = blk: {
    var masks: [8]u64 = undefined;
    for (0..8) |file| {
        var mask: u64 = 0;
        if (file > 0) mask |= @as(u64, 0x0101010101010101) << @intCast(file - 1);
        if (file < 7) mask |= @as(u64, 0x0101010101010101) << @intCast(file + 1);
        masks[file] = mask;
    }
    break :blk masks;
};

// Mobility weights per piece type (centipawns per move)
const MOBILITY_WEIGHTS = [6]i32{ 0, 2, 3, 2, 0, 0 }; // knight=2, bishop=3, rook=2
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

fn computePhase(state: *const State) i32 {
    var phase: i32 = 0;
    for ([_]Pieces.Piece{ Pieces.knight, Pieces.bishop, Pieces.rook, Pieces.queen }) |piece| {
        const count: i32 = @intCast(state.pieceBitboard(piece).popCount());
        phase += count * PHASE_WEIGHTS[piece];
    }
    return @min(phase, MAX_PHASE);
}

fn bishopPairBonus(state: *const State, c: Color) i32 {
    const bishops = state.pieceBitboard(Pieces.bishop).bitAnd(state.colorBitboard(c));
    if (bishops.popCount() >= 2) {
        return 50;
    }
    return 0;
}

fn rookOnOpenFile(state: *const State, c: Color) i32 {
    var score: i32 = 0;
    const our_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(c));
    const opp_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(~c));
    var rooks = state.pieceBitboard(Pieces.rook).bitAnd(state.colorBitboard(c));

    while (rooks.next()) |s| {
        const file: u3 = @intCast(s % 8);
        const file_mask = FILE_MASKS[file];

        const has_our_pawn = (our_pawns.bits & file_mask) != 0;
        const has_opp_pawn = (opp_pawns.bits & file_mask) != 0;

        if (!has_our_pawn and !has_opp_pawn) {
            score += 25; // Open file
        } else if (!has_our_pawn and has_opp_pawn) {
            score += 15; // Semi-open file
        }
    }
    return score;
}

fn kingSafety(state: *const State, c: Color, phase: i32) i32 {
    // Only evaluate king safety in middlegame (phase > 12 means enough pieces)
    if (phase <= 12) return 0;

    const king_bb = state.pieceBitboard(Pieces.king).bitAnd(state.colorBitboard(c));
    const king_sq = @ctz(king_bb.bits);
    const king_file: u3 = @intCast(king_sq % 8);
    const king_rank: u3 = @intCast(king_sq / 8);

    // Check if king is on back ranks (castled position)
    const on_back_ranks = if (c == Colors.white) king_rank <= 1 else king_rank >= 6;
    if (!on_back_ranks) return 0;

    var score: i32 = 0;
    const our_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(c));

    // Check pawn shield in front of king (3 files: king file and adjacent)
    const shield_rank: u6 = if (c == Colors.white) king_rank + 1 else king_rank - 1;

    // Check each file in shield zone
    const min_file: u3 = if (king_file > 0) king_file - 1 else 0;
    const max_file: u3 = if (king_file < 7) king_file + 1 else 7;

    var file: u4 = min_file;
    while (file <= max_file) : (file += 1) {
        const shield_sq: u6 = @as(u6, @as(u3, @intCast(file))) + @as(u6, shield_rank) * 8;
        const shield_mask: u64 = @as(u64, 1) << shield_sq;
        if ((our_pawns.bits & shield_mask) != 0) {
            score += 15; // Pawn in shield position
        } else {
            score -= 10; // Missing shield pawn
        }
    }

    return score;
}

fn knightOutposts(state: *const State, c: Color) i32 {
    var score: i32 = 0;
    const our_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(c));
    const opp_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(~c));
    var knights = state.pieceBitboard(Pieces.knight).bitAnd(state.colorBitboard(c));

    while (knights.next()) |s| {
        const file: u3 = @intCast(s % 8);
        const rank: u3 = @intCast(s / 8);

        // Check if knight is in opponent's half (outpost territory)
        const in_outpost_zone = if (c == Colors.white) rank >= 4 else rank <= 3;
        if (!in_outpost_zone) continue;

        // Check if enemy pawns can attack this square
        // Enemy pawns would need to be on adjacent files, behind the knight
        const can_be_attacked = blk: {
            if (ADJACENT_FILES[file] == 0) break :blk false;
            const adjacent_file_mask = ADJACENT_FILES[file];

            // For white, enemy pawns attacking from above means they're on higher ranks
            // For black, enemy pawns attacking from below means they're on lower ranks
            const rank_u6: u6 = rank;
            const attack_ranks: u64 = if (c == Colors.white)
                // Enemy pawns must be on ranks above to attack down
                if (rank < 7) @as(u64, 0xFFFFFFFFFFFFFFFF) << ((rank_u6 + 1) * 8) else 0
            else
                // Enemy pawns must be on ranks below to attack up
                if (rank > 0) (@as(u64, 1) << (rank_u6 * 8)) - 1 else 0;

            break :blk (opp_pawns.bits & adjacent_file_mask & attack_ranks) != 0;
        };

        if (!can_be_attacked) {
            // Check if defended by our pawn
            const defended_by_pawn = blk: {
                if (file == 0 or file == 7) {
                    const def_file: u3 = if (file == 0) 1 else 6;
                    const def_rank: u3 = if (c == Colors.white) rank - 1 else rank + 1;
                    if ((c == Colors.white and rank == 0) or (c == Colors.black and rank == 7)) break :blk false;
                    const def_sq: u6 = @as(u6, def_file) + @as(u6, def_rank) * 8;
                    break :blk (our_pawns.bits & (@as(u64, 1) << def_sq)) != 0;
                }
                const def_rank: u3 = if (c == Colors.white) rank - 1 else rank + 1;
                if ((c == Colors.white and rank == 0) or (c == Colors.black and rank == 7)) break :blk false;
                const left_sq: u6 = @as(u6, file - 1) + @as(u6, def_rank) * 8;
                const right_sq: u6 = @as(u6, file + 1) + @as(u6, def_rank) * 8;
                const left_mask: u64 = @as(u64, 1) << left_sq;
                const right_mask: u64 = @as(u64, 1) << right_sq;
                break :blk (our_pawns.bits & (left_mask | right_mask)) != 0;
            };

            if (defended_by_pawn) {
                score += 25; // Defended outpost
            } else {
                score += 10; // Undefended outpost
            }
        }
    }

    return score;
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
    const all_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(c));
    var pawns = all_pawns;

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

        // Isolated pawn penalty
        if ((all_pawns.bits & ADJACENT_FILES[file]) == 0) {
            score -= 15;
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
        score += numMoves * MOBILITY_WEIGHTS[Pieces.knight];
    }

    var bishops = state.pieceBitboard(Pieces.bishop).bitAnd(pieces);
    while (bishops.next()) |s| {
        const moves = movegen.sliderMoves(state, s, Pieces.bishop).bitAnd(pieces.not());
        const numMoves: i32 = @intCast(moves.popCount());
        score += numMoves * MOBILITY_WEIGHTS[Pieces.bishop];
    }

    var rooks = state.pieceBitboard(Pieces.rook).bitAnd(pieces);
    while (rooks.next()) |s| {
        const moves = movegen.sliderMoves(state, s, Pieces.rook).bitAnd(pieces.not());
        const numMoves: i32 = @intCast(moves.popCount());
        score += numMoves * MOBILITY_WEIGHTS[Pieces.rook];
    }

    return score;
}

pub fn evaluate(state: *const State) i32 {
    const to_move = state.to_move;
    const opp = ~to_move;

    // Compute game phase for tapered evaluation
    const phase = computePhase(state);
    const is_endgame = phase <= 12;

    // Material
    const our_material = materialCount(state, to_move);
    const opp_material = materialCount(state, opp);

    // Positional scores (PST)
    const our_position = positionalScore(state, to_move, is_endgame);
    const opp_position = positionalScore(state, opp, is_endgame);

    // Pawn structure
    const our_pawn_structure = pawnStructureScore(state, to_move);
    const opp_pawn_structure = pawnStructureScore(state, opp);

    // Mobility
    const our_mobility = mobilityScore(state, to_move);
    const opp_mobility = mobilityScore(state, opp);

    // Bishop pair bonus
    const our_bishop_pair = bishopPairBonus(state, to_move);
    const opp_bishop_pair = bishopPairBonus(state, opp);

    // Rook on open/semi-open file
    const our_rook_file = rookOnOpenFile(state, to_move);
    const opp_rook_file = rookOnOpenFile(state, opp);

    // King safety (only in middlegame)
    const our_king_safety = kingSafety(state, to_move, phase);
    const opp_king_safety = kingSafety(state, opp, phase);

    // Knight outposts
    const our_outposts = knightOutposts(state, to_move);
    const opp_outposts = knightOutposts(state, opp);

    const our_score = our_material + our_position + our_pawn_structure + our_mobility +
        our_bishop_pair + our_rook_file + our_king_safety + our_outposts;
    const opp_score = opp_material + opp_position + opp_pawn_structure + opp_mobility +
        opp_bishop_pair + opp_rook_file + opp_king_safety + opp_outposts;

    return our_score - opp_score;
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
