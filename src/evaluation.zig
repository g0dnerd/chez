const std = @import("std");

const chez = @import("chez.zig");
const State = chez.State;
// const game = @import("game.zig");
// const movegen = @import("movegen.zig");
const Colors = chez.Colors;
const Color = chez.Color;
const Pieces = chez.Pieces;
const Move = chez.Move;
const MoveList = chez.MoveList;

// Phase weights for tapered evaluation (max phase = 24)
const phase_weights = [6]i32{ 0, 1, 1, 2, 4, 0 }; // pawn, knight, bishop, rook, queen, king
const max_phase: i32 = 24;

// Packed score holding both middlegame and endgame values.
// Allows evaluating once and interpolating at the end based on game phase.
pub const Score = packed struct {
    mg: i16,
    eg: i16,

    pub const zero = Score{ .mg = 0, .eg = 0 };

    pub fn init(mg: i32, eg: i32) Score {
        return .{ .mg = @intCast(mg), .eg = @intCast(eg) };
    }

    pub fn add(self: Score, other: Score) Score {
        return .{ .mg = self.mg + other.mg, .eg = self.eg + other.eg };
    }

    pub fn sub(self: Score, other: Score) Score {
        return .{ .mg = self.mg - other.mg, .eg = self.eg - other.eg };
    }

    pub fn mul(self: Score, n: i32) Score {
        return .{
            .mg = @intCast(self.mg * @as(i16, @intCast(n))),
            .eg = @intCast(self.eg * @as(i16, @intCast(n))),
        };
    }

    pub fn neg(self: Score) Score {
        return .{ .mg = -self.mg, .eg = -self.eg };
    }

    // Interpolate between MG and EG based on phase (0 = endgame, 24 = opening)
    pub fn taper(self: Score, phase: i32) i32 {
        return @divTrunc(@as(i32, self.mg) * phase + @as(i32, self.eg) * (max_phase - phase), max_phase);
    }
};

// Piece values: (middlegame, endgame)
pub const piece_values = [6]Score{
    Score.init(100, 120), // pawn - more valuable in endgame
    Score.init(305, 290), // knight - slightly weaker in endgame
    Score.init(333, 350), // bishop - stronger in endgame
    Score.init(563, 575), // rook - slightly stronger in endgame
    Score.init(950, 1000), // queen
    Score.init(20000, 20000), // king
};

// For MVV-LVA move ordering (uses middlegame values)
pub const piece_values_mg = [6]i32{ 100, 305, 333, 563, 950, 20000 };

// Passed pawn bonus by rank (from pawn's perspective, rank 1-6 relevant)
const passed_pawn_bonus = [8]Score{
    Score.init(0, 0), // rank 0 (impossible for white)
    Score.init(5, 10), // rank 1
    Score.init(10, 20), // rank 2
    Score.init(20, 40), // rank 3
    Score.init(35, 70), // rank 4
    Score.init(60, 120), // rank 5
    Score.init(100, 200), // rank 6
    Score.init(0, 0), // rank 7 (promoted)
};

// Mobility bonus per move (middlegame, endgame)
const mobility_bonus = [6]Score{
    Score.init(0, 0), // pawn
    Score.init(4, 4), // knight
    Score.init(5, 5), // bishop
    Score.init(2, 4), // rook - mobility matters more in endgame
    Score.init(1, 2), // queen
    Score.init(0, 0), // king
};

// Bonus/penalty constants
const bishop_pair = Score.init(30, 50); // More valuable in endgame
const rook_open_file = Score.init(25, 15);
const rook_semi_open = Score.init(15, 10);
const rook_on_seventh = Score.init(20, 40); // Much stronger in endgame
const isolated_pawn = Score.init(-15, -20); // Worse in endgame
const doubled_pawn = Score.init(-10, -20); // Worse in endgame
const connected_pawn = Score.init(7, 10); // Pawns side-by-side or on adjacent files
const protected_passed_pawn = Score.init(15, 30); // Passed pawn defended by another pawn
const blocked_passed_pawn = Score.init(-10, -20); // Passed pawn blocked by a piece
const knight_outpost_defended = Score.init(25, 15); // Less relevant in endgame
const knight_outpost_undefended = Score.init(10, 5);
const pawn_shield = Score.init(15, 0); // Only matters in middlegame
const pawn_shield_missing = Score.init(-10, 0);

// File masks for rook on open file detection
const file_masks: [8]u64 = blk: {
    var masks: [8]u64 = undefined;
    for (0..8) |file| {
        masks[file] = @as(u64, 0x0101010101010101) << @intCast(file);
    }
    break :blk masks;
};

// Adjacent file masks for isolated pawn detection
const adjacent_files: [8]u64 = blk: {
    var masks: [8]u64 = undefined;
    for (0..8) |file| {
        var mask: u64 = 0;
        if (file > 0) mask |= @as(u64, 0x0101010101010101) << @intCast(file - 1);
        if (file < 7) mask |= @as(u64, 0x0101010101010101) << @intCast(file + 1);
        masks[file] = mask;
    }
    break :blk masks;
};

// Piece-square tables: [piece][square] -> Score(mg, eg)
pub const pst = [6][64]Score{
    // Pawns
    blk: {
        const mg = [64]i16{
            0,  0,  0,   0,   0,   0,   0,  0,
            5,  10, 10,  -20, -20, 10,  10, 5,
            5,  -5, -10, 0,   0,   -10, -5, 5,
            0,  0,  0,   20,  20,  0,   0,  0,
            5,  5,  10,  25,  25,  10,  5,  5,
            10, 10, 20,  30,  30,  20,  10, 10,
            50, 50, 50,  50,  50,  50,  50, 50,
            0,  0,  0,   0,   0,   0,   0,  0,
        };
        const eg = [64]i16{
            0,  0,  0,  0,  0,  0,  0,  0,
            10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10,
            20, 20, 20, 20, 20, 20, 20, 20,
            30, 30, 30, 30, 30, 30, 30, 30,
            50, 50, 50, 50, 50, 50, 50, 50,
            80, 80, 80, 80, 80, 80, 80, 80,
            0,  0,  0,  0,  0,  0,  0,  0,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score{ .mg = mg[i], .eg = eg[i] };
        }
        break :blk result;
    },
    // Knights
    blk: {
        const mg = [64]i16{
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0,   5,   5,   0,   -20, -40,
            -30, 5,   10,  15,  15,  10,  5,   -30,
            -30, 0,   15,  20,  20,  15,  0,   -30,
            -30, 5,   15,  20,  20,  15,  5,   -30,
            -30, 0,   10,  15,  15,  10,  0,   -30,
            -40, -20, 0,   0,   0,   0,   -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50,
        };
        const eg = [64]i16{
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0,   0,   0,   0,   -20, -40,
            -30, 0,   10,  15,  15,  10,  0,   -30,
            -30, 5,   15,  20,  20,  15,  5,   -30,
            -30, 5,   15,  20,  20,  15,  5,   -30,
            -30, 0,   10,  15,  15,  10,  0,   -30,
            -40, -20, 0,   0,   0,   0,   -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score{ .mg = mg[i], .eg = eg[i] };
        }
        break :blk result;
    },
    // Bishops
    blk: {
        const mg = [64]i16{
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5,   0,   0,   0,   0,   5,   -10,
            -10, 10,  10,  10,  10,  10,  10,  -10,
            -10, 0,   10,  10,  10,  10,  0,   -10,
            -10, 5,   5,   10,  10,  5,   5,   -10,
            -10, 0,   5,   10,  10,  5,   0,   -10,
            -10, 0,   0,   0,   0,   0,   0,   -10,
            -20, -10, -10, -10, -10, -10, -10, -20,
        };
        const eg = [64]i16{
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 0,   0,   0,   0,   0,   0,   -10,
            -10, 0,   5,   10,  10,  5,   0,   -10,
            -10, 0,   10,  15,  15,  10,  0,   -10,
            -10, 0,   10,  15,  15,  10,  0,   -10,
            -10, 0,   5,   10,  10,  5,   0,   -10,
            -10, 0,   0,   0,   0,   0,   0,   -10,
            -20, -10, -10, -10, -10, -10, -10, -20,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score{ .mg = mg[i], .eg = eg[i] };
        }
        break :blk result;
    },
    // Rooks
    blk: {
        const mg = [64]i16{
            0,  0,  0,  5,  5,  0,  0,  0,
            -5, 0,  0,  0,  0,  0,  0,  -5,
            -5, 0,  0,  0,  0,  0,  0,  -5,
            -5, 0,  0,  0,  0,  0,  0,  -5,
            -5, 0,  0,  0,  0,  0,  0,  -5,
            -5, 0,  0,  0,  0,  0,  0,  -5,
            5,  10, 10, 10, 10, 10, 10, 5,
            0,  0,  0,  0,  0,  0,  0,  0,
        };
        const eg = [64]i16{
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score{ .mg = mg[i], .eg = eg[i] };
        }
        break :blk result;
    },
    // Queens
    blk: {
        const mg = [64]i16{
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0,   5,   0,  0,  0,   0,   -10,
            -10, 5,   5,   5,  5,  5,   0,   -10,
            0,   0,   5,   5,  5,  5,   0,   -5,
            -5,  0,   5,   5,  5,  5,   0,   -5,
            -10, 0,   5,   5,  5,  5,   0,   -10,
            -10, 0,   0,   0,  0,  0,   0,   -10,
            -20, -10, -10, -5, -5, -10, -10, -20,
        };
        const eg = [64]i16{
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0,   0,   0,  0,  0,   0,   -10,
            -10, 0,   5,   5,  5,  5,   0,   -10,
            -5,  0,   5,   10, 10, 5,   0,   -5,
            -5,  0,   5,   10, 10, 5,   0,   -5,
            -10, 0,   5,   5,  5,  5,   0,   -10,
            -10, 0,   0,   0,  0,  0,   0,   -10,
            -20, -10, -10, -5, -5, -10, -10, -20,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score{ .mg = mg[i], .eg = eg[i] };
        }
        break :blk result;
    },
    // Kings
    blk: {
        // Middlegame: stay safe on the side
        const mg = [64]i16{
            20,  30,  10,  0,   0,   10,  30,  20,
            20,  20,  0,   0,   0,   0,   20,  20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
        };
        // Endgame: centralize the king
        const eg = [64]i16{
            -50, -30, -30, -30, -30, -30, -30, -50,
            -30, -10, 0,   0,   0,   0,   -10, -30,
            -30, 0,   20,  30,  30,  20,  0,   -30,
            -30, 0,   30,  40,  40,  30,  0,   -30,
            -30, 0,   30,  40,  40,  30,  0,   -30,
            -30, 0,   20,  30,  30,  20,  0,   -30,
            -30, -10, 0,   0,   0,   0,   -10, -30,
            -50, -30, -30, -30, -30, -30, -30, -50,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score{ .mg = mg[i], .eg = eg[i] };
        }
        break :blk result;
    },
};

fn computePhase(state: *const State) i32 {
    var phase: i32 = 0;
    for ([_]Pieces.Piece{ Pieces.knight, Pieces.bishop, Pieces.rook, Pieces.queen }) |piece| {
        const count: i32 = @intCast(state.pieceBitboard(piece).popCount());
        phase += count * phase_weights[piece];
    }
    return @min(phase, max_phase);
}

fn materialScore(state: *const State, c: Color) Score {
    var score = Score.zero;
    const pieces = state.colorBitboard(c);

    inline for (0..5) |piece_idx| {
        const piece: Pieces.Piece = @intCast(piece_idx);
        const count: i32 = @intCast(state.pieceBitboard(piece).bitAnd(pieces).popCount());
        score = score.add(piece_values[piece].mul(count));
    }

    return score;
}

fn positionalScore(state: *const State, c: Color) Score {
    var score = Score.zero;

    var piece: Pieces.Piece = 0;
    while (piece < 6) {
        var pieces = state.colorBitboard(c).bitAnd(state.pieceBitboard(piece));
        defer piece += 1;

        while (pieces.next()) |s| {
            const rank = s / 8;
            const file = s % 8;
            // Mirror square for black (flip rank)
            const sq: u6 = if (c == Colors.black)
                @intCast((7 - rank) * 8 + file)
            else
                s;

            score = score.add(pst[piece][sq]);
        }
    }
    return score;
}

fn pawnStructureScore(state: *const State, c: Color) Score {
    var score = Score.zero;
    const all_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(c));
    const opp_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(~c));
    const occupied = state.all_pieces.bits;
    var pawns = all_pawns;

    while (pawns.next()) |s| {
        const file: u6 = s % 8;
        const rank: u6 = s / 8;

        // Connected pawn bonus: pawn on adjacent file, same rank or one rank behind
        const connected = blk: {
            var mask: u64 = 0;
            // Same rank, adjacent files
            if (file > 0) mask |= @as(u64, 1) << (s - 1);
            if (file < 7) mask |= @as(u64, 1) << (s + 1);
            // One rank behind (supporting), adjacent files
            if (c == Colors.white and rank > 0) {
                if (file > 0) mask |= @as(u64, 1) << (s - 9);
                if (file < 7) mask |= @as(u64, 1) << (s - 7);
            } else if (c == Colors.black and rank < 7) {
                if (file > 0) mask |= @as(u64, 1) << (s + 7);
                if (file < 7) mask |= @as(u64, 1) << (s + 9);
            }
            break :blk (all_pawns.bits & mask) != 0;
        };
        if (connected) {
            score = score.add(connected_pawn);
        }

        // Passed pawn detection
        const ahead_mask = computePassedPawnMask(c, file, rank);

        if ((opp_pawns.bits & ahead_mask) == 0) {
            // It's a passed pawn - use rank from pawn's perspective
            const passed_rank: usize = if (c == Colors.white) rank else 7 - rank;
            score = score.add(passed_pawn_bonus[passed_rank]);

            // Protected passed pawn: defended by another pawn
            const protected = blk: {
                var def_mask: u64 = 0;
                if (c == Colors.white and rank > 0) {
                    if (file > 0) def_mask |= @as(u64, 1) << (s - 9);
                    if (file < 7) def_mask |= @as(u64, 1) << (s - 7);
                } else if (c == Colors.black and rank < 7) {
                    if (file > 0) def_mask |= @as(u64, 1) << (s + 7);
                    if (file < 7) def_mask |= @as(u64, 1) << (s + 9);
                }
                break :blk (all_pawns.bits & def_mask) != 0;
            };
            if (protected) {
                score = score.add(protected_passed_pawn);
            }

            // Blocked passed pawn: piece directly in front
            const blocked = blk: {
                if (c == Colors.white and rank < 7) {
                    break :blk (occupied & (@as(u64, 1) << (s + 8))) != 0;
                } else if (c == Colors.black and rank > 0) {
                    break :blk (occupied & (@as(u64, 1) << (s - 8))) != 0;
                }
                break :blk false;
            };
            if (blocked) {
                score = score.add(blocked_passed_pawn);
            }
        }

        // Doubled pawn penalty
        const file_mask = file_masks[file];
        if (@popCount(all_pawns.bits & file_mask) > 1) {
            score = score.add(doubled_pawn);
        }

        // Isolated pawn penalty
        if ((all_pawns.bits & adjacent_files[file]) == 0) {
            score = score.add(isolated_pawn);
        }
    }

    return score;
}

fn computePassedPawnMask(c: Color, file: u6, rank: u6) u64 {
    if (c == Colors.white) {
        if (rank >= 7) return 0;
        // Squares ahead on same file and adjacent files
        const ranks_ahead: u64 = @as(u64, 0xFFFFFFFFFFFFFFFF) << ((rank + 1) * 8);
        var file_mask: u64 = file_masks[file];
        if (file > 0) file_mask |= file_masks[file - 1];
        if (file < 7) file_mask |= file_masks[file + 1];
        return ranks_ahead & file_mask;
    } else {
        if (rank == 0) return 0;
        // Squares ahead (lower ranks for black)
        const ranks_ahead: u64 = (@as(u64, 1) << (rank * 8)) - 1;
        var file_mask: u64 = file_masks[file];
        if (file > 0) file_mask |= file_masks[file - 1];
        if (file < 7) file_mask |= file_masks[file + 1];
        return ranks_ahead & file_mask;
    }
}

fn mobilityScore(state: *const State, c: Color) Score {
    var score = Score.zero;
    const our_pieces = state.colorBitboard(c);

    // Knights
    var knights = state.pieceBitboard(Pieces.knight).bitAnd(our_pieces);
    while (knights.next()) |s| {
        const moves = chez.movegen.knight_move_mask[s].bitAnd(our_pieces.not());
        const count: i32 = @intCast(moves.popCount());
        score = score.add(mobility_bonus[Pieces.knight].mul(count));
    }

    // Bishops
    var bishops = state.pieceBitboard(Pieces.bishop).bitAnd(our_pieces);
    while (bishops.next()) |s| {
        const moves = chez.movegen.sliderMoves(state, s, Pieces.bishop).bitAnd(our_pieces.not());
        const count: i32 = @intCast(moves.popCount());
        score = score.add(mobility_bonus[Pieces.bishop].mul(count));
    }

    // Rooks
    var rooks = state.pieceBitboard(Pieces.rook).bitAnd(our_pieces);
    while (rooks.next()) |s| {
        const moves = chez.movegen.sliderMoves(state, s, Pieces.rook).bitAnd(our_pieces.not());
        const count: i32 = @intCast(moves.popCount());
        score = score.add(mobility_bonus[Pieces.rook].mul(count));
    }

    return score;
}

fn bishopPairBonus(state: *const State, c: Color) Score {
    const bishops = state.pieceBitboard(Pieces.bishop).bitAnd(state.colorBitboard(c));
    if (bishops.popCount() >= 2) {
        return bishop_pair;
    }
    return Score.zero;
}

fn rookBonus(state: *const State, c: Color) Score {
    var score = Score.zero;
    const our_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(c));
    const opp_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(~c));
    var rooks = state.pieceBitboard(Pieces.rook).bitAnd(state.colorBitboard(c));

    const seventh_rank: u6 = if (c == Colors.white) 6 else 1;

    while (rooks.next()) |s| {
        const file: usize = s % 8;
        const rank: u6 = @intCast(s / 8);
        const file_mask = file_masks[file];

        // Open/semi-open file
        const has_our_pawn = (our_pawns.bits & file_mask) != 0;
        const has_opp_pawn = (opp_pawns.bits & file_mask) != 0;

        if (!has_our_pawn and !has_opp_pawn) {
            score = score.add(rook_open_file);
        } else if (!has_our_pawn and has_opp_pawn) {
            score = score.add(rook_semi_open);
        }

        // Rook on 7th rank
        if (rank == seventh_rank) {
            score = score.add(rook_on_seventh);
        }
    }
    return score;
}

fn kingSafetyScore(state: *const State, c: Color) Score {
    var score = Score.zero;

    const king_bb = state.pieceBitboard(Pieces.king).bitAnd(state.colorBitboard(c));
    const king_sq = @ctz(king_bb.bits);
    const king_file: u3 = @intCast(king_sq % 8);
    const king_rank: u3 = @intCast(king_sq / 8);

    // Check if king is on back ranks (castled position)
    const on_back_ranks = if (c == Colors.white) king_rank <= 1 else king_rank >= 6;
    if (!on_back_ranks) return Score.zero;

    const our_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(c));

    // Check pawn shield in front of king
    const shield_rank: u6 = if (c == Colors.white) @as(u6, king_rank) + 1 else @as(u6, king_rank) - 1;

    const min_file: u3 = if (king_file > 0) king_file - 1 else 0;
    const max_file: u3 = if (king_file < 7) king_file + 1 else 7;

    var file: u4 = min_file;
    while (file <= max_file) : (file += 1) {
        const shield_sq: u6 = @as(u6, @as(u3, @intCast(file))) + shield_rank * 8;
        const shield_mask: u64 = @as(u64, 1) << shield_sq;
        if ((our_pawns.bits & shield_mask) != 0) {
            score = score.add(pawn_shield);
        } else {
            score = score.add(pawn_shield_missing);
        }
    }

    return score;
}

fn knightOutposts(state: *const State, c: Color) Score {
    var score = Score.zero;
    const our_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(c));
    const opp_pawns = state.pieceBitboard(Pieces.pawn).bitAnd(state.colorBitboard(~c));
    var knights = state.pieceBitboard(Pieces.knight).bitAnd(state.colorBitboard(c));

    while (knights.next()) |s| {
        const file: u3 = @intCast(s % 8);
        const rank: u3 = @intCast(s / 8);

        // Check if knight is in opponent's half
        const in_outpost_zone = if (c == Colors.white) rank >= 4 else rank <= 3;
        if (!in_outpost_zone) continue;

        // Check if enemy pawns can attack this square
        const can_be_attacked = blk: {
            if (adjacent_files[file] == 0) break :blk false;
            const adjacent_file_mask = adjacent_files[file];

            const rank_u6: u6 = rank;
            const attack_ranks: u64 = if (c == Colors.white)
                if (rank < 7) @as(u64, 0xFFFFFFFFFFFFFFFF) << ((rank_u6 + 1) * 8) else 0
            else if (rank > 0) (@as(u64, 1) << (rank_u6 * 8)) - 1 else 0;

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
                score = score.add(knight_outpost_defended);
            } else {
                score = score.add(knight_outpost_undefended);
            }
        }
    }

    return score;
}

pub fn evaluate(state: *const State) i32 {
    const to_move = state.to_move;
    const opp = ~to_move;

    // Compute game phase for tapered evaluation
    const phase = computePhase(state);

    // Accumulate all scores
    var our_score = Score.zero;
    var opp_score = Score.zero;

    // Material
    our_score = our_score.add(materialScore(state, to_move));
    opp_score = opp_score.add(materialScore(state, opp));

    // Positional (PST)
    our_score = our_score.add(positionalScore(state, to_move));
    opp_score = opp_score.add(positionalScore(state, opp));

    // Pawn structure
    our_score = our_score.add(pawnStructureScore(state, to_move));
    opp_score = opp_score.add(pawnStructureScore(state, opp));

    // Mobility
    our_score = our_score.add(mobilityScore(state, to_move));
    opp_score = opp_score.add(mobilityScore(state, opp));

    // Bishop pair
    our_score = our_score.add(bishopPairBonus(state, to_move));
    opp_score = opp_score.add(bishopPairBonus(state, opp));

    // Rook bonuses (open file, 7th rank)
    our_score = our_score.add(rookBonus(state, to_move));
    opp_score = opp_score.add(rookBonus(state, opp));

    // King safety (pawn shield) - already tapered via Score
    our_score = our_score.add(kingSafetyScore(state, to_move));
    opp_score = opp_score.add(kingSafetyScore(state, opp));

    // Knight outposts
    our_score = our_score.add(knightOutposts(state, to_move));
    opp_score = opp_score.add(knightOutposts(state, opp));

    // Compute final tapered score
    const total = our_score.sub(opp_score);
    return total.taper(phase);
}

// History heuristic table: [color][from_square][to_square] -> score
// Tracks which quiet moves have caused beta cutoffs
pub const HistoryTable = struct {
    table: [2][64][64]i32 = [_][64][64]i32{[_][64]i32{[_]i32{0} ** 64} ** 64} ** 2,

    pub fn get(self: *const HistoryTable, color: Color, from: u6, to: u6) i32 {
        return self.table[color][from][to];
    }

    pub fn update(self: *HistoryTable, color: Color, from: u6, to: u6, depth: u8) void {
        // Bonus proportional to depth squared (deeper cutoffs are more valuable)
        const bonus: i32 = @as(i32, depth) * @as(i32, depth);
        self.table[color][from][to] += bonus;
        // Prevent overflow - cap at reasonable value
        if (self.table[color][from][to] > 10000) {
            self.table[color][from][to] = 10000;
        }
    }

    pub fn clear(self: *HistoryTable) void {
        self.table = [_][64][64]i32{[_][64]i32{[_]i32{0} ** 64} ** 64} ** 2;
    }
};

pub fn scoreMove(ctx: *const MoveList.SortCtx, m: Move) i32 {
    var score: i32 = 0;

    // MVV-LVA for captures
    if (ctx.state.pieceAt(m.end)) |captured_piece| {
        const attacker_piece = ctx.state.pieceAt(m.start).?;
        score += piece_values_mg[captured_piece] * 10 - piece_values_mg[attacker_piece];
    }

    // Promotion bonus
    const end_rank = m.end / 8;
    if (ctx.state.pieceAt(m.start) == Pieces.pawn and
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

    // History heuristic for quiet moves (non-captures, non-promotions)
    if (ctx.state.pieceAt(m.end) == null and ctx.history != null) {
        const is_promotion = ctx.state.pieceAt(m.start) == Pieces.pawn and
            ((end_rank == 7 and ctx.color == Colors.white) or (end_rank == 0 and ctx.color == Colors.black));
        if (!is_promotion) {
            score += @divTrunc(ctx.history.?.get(ctx.color, m.start, m.end), 10);
        }
    }

    return score;
}
