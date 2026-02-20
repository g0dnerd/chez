const std = @import("std");

const Bitboard = @import("Bitboard.zig");
const State = @import("State.zig");
const engine = @import("engine.zig");
const Move = engine.Move;
const movegen = @import("movegen.zig");
const MoveList = movegen.MoveList;
const Color = engine.Color;
const Colors = engine.Colors;
const piece = @import("piece.zig");
const square = @import("square.zig");

const FILE_A: u64 = 0x0101010101010101;
const FILE_H: u64 = 0x8080808080808080;
const LOW_RANKS_WHITE: u64 = 0x0000000000FFFF00; // ranks 2-3
const LOW_RANKS_BLACK: u64 = 0x00FFFF0000000000; // ranks 6-7

const PieceAttacks = struct {
    knight: u64 = 0,
    bishop: u64 = 0,
    rook: u64 = 0,
};

// Phase weights for tapered evaluation
const phase_weights = [6]i32{ 0, 1, 1, 2, 4, 0 }; // pawn, knight, bishop, rook, queen, king
const max_phase_mg: i32 = 24;

// Packed score holding both middlegame and endgame values.
// Allows evaluating once and interpolating at the end based on game phase.
pub const Score = struct {
    v: @Vector(2, i16),

    pub const zero = Score{ .v = @splat(0) };

    pub fn init(mg: i16, eg: i16) Score {
        return .{ .v = .{ mg, eg } };
    }

    pub fn add(self: Score, other: Score) Score {
        return .{ .v = self.v + other.v };
    }

    pub fn sub(self: Score, other: Score) Score {
        return .{ .v = self.v - other.v };
    }

    pub fn mul(self: Score, n: i32) Score {
        const factor: @Vector(2, i16) = @splat(@intCast(n));
        return .{ .v = self.v * factor };
    }

    pub fn neg(self: Score) Score {
        return .{ .v = -self.v };
    }

    // Interpolate between MG and EG based on phase (0 = endgame, 24 = opening)
    pub fn taper(self: Score, phase: i32) i32 {
        const mg: i32 = self.v[0];
        const eg: i32 = self.v[1];
        return @divTrunc(mg * phase + eg * (max_phase_mg - phase), max_phase_mg);
    }

    pub fn midgame(self: Score) i16 {
        return self.v[0];
    }

    pub fn endgame(self: Score) i16 {
        return self.v[1];
    }
};

pub fn toCentipawns(val: i32) f32 {
    const val_f: f32 = @floatFromInt(val);
    return val_f / @as(f32, @floatFromInt(piece_values[0].endgame()));
}

pub const piece_values = [6]Score{
    Score.init(126, 208), // pawn
    Score.init(781, 854), // knight
    Score.init(825, 915), // bishop
    Score.init(1276, 1380), // rook
    Score.init(2538, 2682), // queen
    Score.init(20000, 20000), // king
};

// For MVV-LVA move ordering (uses middlegame values)
pub const piece_values_mg = [6]i32{ 126, 781, 825, 1276, 2538, 20000 };

// Passed pawn bonus by rank (from pawn's perspective, rank 1-6 relevant)
const passed_pawn_bonus = [8]Score{
    Score.init(0, 0), // rank 0
    Score.init(9, 28), // rank 1
    Score.init(15, 31), // rank 2
    Score.init(17, 39), // rank 3
    Score.init(64, 70), // rank 4
    Score.init(171, 177), // rank 5
    Score.init(277, 260), // rank 6
    Score.init(0, 0), // rank 7
};

// Mobility bonus per move (middlegame, endgame)
const mobility_bonus = [4][28]Score{
    // Knights
    [_]Score{
        Score.init(-62, -81),
        Score.init(-53, -56),
        Score.init(-12, -31),
        Score.init(-4, -16),
        Score.init(3, 5),
        Score.init(13, 11),
        Score.init(22, 17),
        Score.init(28, 20),
        Score.init(33, 25),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
    },
    // Bishops
    [_]Score{
        Score.init(-48, -59),
        Score.init(-20, -23),
        Score.init(16, -3),
        Score.init(26, 13),
        Score.init(38, 24),
        Score.init(51, 42),
        Score.init(55, 54),
        Score.init(63, 57),
        Score.init(63, 65),
        Score.init(68, 73),
        Score.init(81, 78),
        Score.init(81, 86),
        Score.init(91, 88),
        Score.init(98, 97),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
    },
    // Rooks
    [_]Score{
        Score.init(-60, -78),
        Score.init(-20, -17),
        Score.init(2, 23),
        Score.init(3, 39),
        Score.init(3, 70),
        Score.init(11, 99),
        Score.init(22, 103),
        Score.init(31, 121),
        Score.init(40, 134),
        Score.init(40, 139),
        Score.init(41, 158),
        Score.init(48, 164),
        Score.init(57, 168),
        Score.init(57, 169),
        Score.init(62, 172),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
        Score.init(0, 0),
    },
    // Queens
    [_]Score{
        Score.init(-30, -48),
        Score.init(-12, -30),
        Score.init(-8, -7),
        Score.init(-9, 19),
        Score.init(20, 40),
        Score.init(23, 55),
        Score.init(23, 59),
        Score.init(35, 75),
        Score.init(38, 78),
        Score.init(53, 96),
        Score.init(64, 96),
        Score.init(65, 100),
        Score.init(65, 121),
        Score.init(66, 127),
        Score.init(67, 131),
        Score.init(67, 133),
        Score.init(72, 136),
        Score.init(72, 141),
        Score.init(77, 147),
        Score.init(79, 150),
        Score.init(93, 151),
        Score.init(108, 168),
        Score.init(108, 168),
        Score.init(108, 171),
        Score.init(110, 182),
        Score.init(114, 182),
        Score.init(114, 192),
        Score.init(116, 218),
    },
};

// Bonus/penalty constants
const bishop_pair = Score.init(30, 50);
const rook_open_file = Score.init(48, 27);
const rook_semi_open = Score.init(19, 7);
const rook_on_seventh = Score.init(20, 40);
const isolated_pawn = Score.init(-15, -20);
const doubled_pawn = Score.init(-10, -20);
const backward_pawn = Score.init(-9, -22);
const connected_pawn = Score.init(7, 10);
const protected_passed_pawn = Score.init(15, 30);
const blocked_passed_pawn = Score.init(-10, -20);
const knight_outpost_defended = Score.init(56, 34);
const bishop_outpost_defended = Score.init(31, 23);
const pawn_shield = Score.init(15, 0);
const pawn_shield_missing = Score.init(-10, 0);
const tempo = Score.init(28, 28);

const promotion_bonus: i32 = 12500;

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
// Stockfish classical (pre-NNUE) values. Non-pawn tables mirrored from half-tables
// using edge_distance (A<>H, B<>G, C<>F, D<>E). Pawn table is asymmetric (full 8 files).
pub const pst = [6][64]Score{
    // Pawns (from Stockfish PBonus, full 8-file asymmetric table)
    blk: {
        const mg = [64]i16{
            0,  0,   0,   0,   0,  0,   0,   0,
            3,  3,   10,  19,  16, 19,  7,   -5,
            -9, -15, 11,  15,  32, 22,  5,   -22,
            -4, -23, 6,   20,  40, 17,  4,   -8,
            13, 0,   -13, 1,   11, -2,  -13, 5,
            5,  -12, -7,  22,  -8, -5,  -15, -8,
            -7, 7,   -3,  -13, 5,  -16, 10,  -8,
            0,  0,   0,   0,   0,  0,   0,   0,
        };
        const eg = [64]i16{
            0,   0,   0,   0,  0,   0,   0,   0,
            -10, -6,  10,  0,  14,  7,   -5,  -19,
            -10, -10, -10, 4,  4,   3,   -6,  -4,
            6,   -2,  -8,  -4, -13, -12, -10, -9,
            10,  5,   4,   -5, -5,  -5,  14,  9,
            28,  20,  21,  28, 30,  7,   6,   13,
            0,   -11, 12,  21, 25,  19,  4,   7,
            0,   0,   0,   0,  0,   0,   0,   0,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score.init(mg[i], eg[i]);
        }
        break :blk result;
    },
    // Knights (mirrored: col0=A/H, col1=B/G, col2=C/F, col3=D/E)
    blk: {
        const mg = [64]i16{
            -175, -92, -74, -73, -73, -74, -92, -175,
            -77,  -41, -27, -15, -15, -27, -41, -77,
            -61,  -17, 6,   12,  12,  6,   -17, -61,
            -35,  8,   40,  49,  49,  40,  8,   -35,
            -34,  13,  44,  51,  51,  44,  13,  -34,
            -9,   22,  58,  53,  53,  58,  22,  -9,
            -67,  -27, 4,   37,  37,  4,   -27, -67,
            -201, -83, -56, -26, -26, -56, -83, -201,
        };
        const eg = [64]i16{
            -96,  -65, -49, -21, -21, -49, -65, -96,
            -67,  -54, -18, 8,   8,   -18, -54, -67,
            -40,  -27, -8,  29,  29,  -8,  -27, -40,
            -35,  -2,  13,  28,  28,  13,  -2,  -35,
            -45,  -16, 9,   39,  39,  9,   -16, -45,
            -51,  -44, -16, 17,  17,  -16, -44, -51,
            -69,  -50, -51, 12,  12,  -51, -50, -69,
            -100, -88, -56, -17, -17, -56, -88, -100,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score.init(mg[i], eg[i]);
        }
        break :blk result;
    },
    // Bishops (mirrored: col0=A/H, col1=B/G, col2=C/F, col3=D/E)
    blk: {
        const mg = [64]i16{
            -53, -5,  -8,  -23, -23, -8,  -5,  -53,
            -15, 8,   19,  4,   4,   19,  8,   -15,
            -7,  21,  -5,  17,  17,  -5,  21,  -7,
            -5,  11,  25,  39,  39,  25,  11,  -5,
            -12, 29,  22,  31,  31,  22,  29,  -12,
            -16, 6,   1,   11,  11,  1,   6,   -16,
            -17, -14, 5,   0,   0,   5,   -14, -17,
            -48, 1,   -14, -23, -23, -14, 1,   -48,
        };
        const eg = [64]i16{
            -57, -30, -37, -12, -12, -37, -30, -57,
            -37, -13, -17, 1,   1,   -17, -13, -37,
            -16, -1,  -2,  10,  10,  -2,  -1,  -16,
            -20, -6,  0,   17,  17,  0,   -6,  -20,
            -17, -1,  -14, 15,  15,  -14, -1,  -17,
            -30, 6,   4,   6,   6,   4,   6,   -30,
            -31, -20, -1,  1,   1,   -1,  -20, -31,
            -46, -42, -37, -24, -24, -37, -42, -46,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score.init(mg[i], eg[i]);
        }
        break :blk result;
    },
    // Rooks (mirrored: col0=A/H, col1=B/G, col2=C/F, col3=D/E)
    blk: {
        const mg = [64]i16{
            -31, -20, -14, -5, -5, -14, -20, -31,
            -21, -13, -8,  6,  6,  -8,  -13, -21,
            -25, -11, -1,  3,  3,  -1,  -11, -25,
            -13, -5,  -4,  -6, -6, -4,  -5,  -13,
            -27, -15, -4,  3,  3,  -4,  -15, -27,
            -22, -2,  6,   12, 12, 6,   -2,  -22,
            -2,  12,  16,  18, 18, 16,  12,  -2,
            -17, -19, -1,  9,  9,  -1,  -19, -17,
        };
        const eg = [64]i16{
            -9,  -13, -10, -9, -9, -10, -13, -9,
            -12, -9,  -1,  -2, -2, -1,  -9,  -12,
            6,   -8,  -2,  -6, -6, -2,  -8,  6,
            -6,  1,   -9,  7,  7,  -9,  1,   -6,
            -5,  8,   7,   -6, -6, 7,   8,   -5,
            6,   1,   -7,  10, 10, -7,  1,   6,
            4,   5,   20,  -5, -5, 20,  5,   4,
            18,  0,   19,  13, 13, 19,  0,   18,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score.init(mg[i], eg[i]);
        }
        break :blk result;
    },
    // Queens (mirrored: col0=A/H, col1=B/G, col2=C/F, col3=D/E)
    blk: {
        const mg = [64]i16{
            3,  -5, -5, 4,  4,  -5, -5, 3,
            -3, 5,  8,  12, 12, 8,  5,  -3,
            -3, 6,  13, 7,  7,  13, 6,  -3,
            4,  5,  9,  8,  8,  9,  5,  4,
            0,  14, 12, 5,  5,  12, 14, 0,
            -4, 10, 6,  8,  8,  6,  10, -4,
            -5, 6,  10, 8,  8,  10, 6,  -5,
            -2, -2, 1,  -2, -2, 1,  -2, -2,
        };
        const eg = [64]i16{
            -69, -57, -47, -26, -26, -47, -57, -69,
            -55, -31, -22, -4,  -4,  -22, -31, -55,
            -39, -18, -9,  3,   3,   -9,  -18, -39,
            -23, -3,  13,  24,  24,  13,  -3,  -23,
            -29, -6,  9,   21,  21,  9,   -6,  -29,
            -38, -18, -12, 1,   1,   -12, -18, -38,
            -50, -27, -24, -8,  -8,  -24, -27, -50,
            -75, -52, -43, -36, -36, -43, -52, -75,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score.init(mg[i], eg[i]);
        }
        break :blk result;
    },
    // Kings (mirrored: col0=A/H, col1=B/G, col2=C/F, col3=D/E)
    blk: {
        const mg = [64]i16{
            271, 327, 271, 198, 198, 271, 327, 271,
            278, 303, 234, 179, 179, 234, 303, 278,
            195, 258, 169, 120, 120, 169, 258, 195,
            164, 190, 138, 98,  98,  138, 190, 164,
            154, 179, 105, 70,  70,  105, 179, 154,
            123, 145, 81,  31,  31,  81,  145, 123,
            88,  120, 65,  33,  33,  65,  120, 88,
            59,  89,  45,  -1,  -1,  45,  89,  59,
        };
        const eg = [64]i16{
            1,   45,  85,  76,  76,  85,  45,  1,
            53,  100, 133, 135, 135, 133, 100, 53,
            88,  130, 169, 175, 175, 169, 130, 88,
            103, 156, 172, 172, 172, 172, 156, 103,
            96,  166, 199, 199, 199, 199, 166, 96,
            92,  172, 184, 191, 191, 184, 172, 92,
            47,  121, 116, 131, 131, 116, 121, 47,
            11,  59,  73,  78,  78,  73,  59,  11,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score.init(mg[i], eg[i]);
        }
        break :blk result;
    },
};

fn computePassedPawnMask(c: Color, file: u6, rank: u6) u64 {
    if (c == Colors.white) {
        if (rank >= 7) return 0;
        const ranks_ahead: u64 = @as(u64, 0xFFFFFFFFFFFFFFFF) << ((rank + 1) * 8);
        var file_mask: u64 = file_masks[file];
        if (file > 0) file_mask |= file_masks[file - 1];
        if (file < 7) file_mask |= file_masks[file + 1];
        return ranks_ahead & file_mask;
    } else {
        if (rank == 0) return 0;
        const ranks_ahead: u64 = (@as(u64, 1) << (rank * 8)) - 1;
        var file_mask: u64 = file_masks[file];
        if (file > 0) file_mask |= file_masks[file - 1];
        if (file < 7) file_mask |= file_masks[file + 1];
        return ranks_ahead & file_mask;
    }
}

// Check if a square is an outpost: in opponent's half and no enemy pawns
// on adjacent files that could attack it from ahead.
fn isOutpost(c: Color, file: u6, rank: u6, opp_pawns_bb: Bitboard) bool {
    const in_zone = if (c == Colors.white) rank >= 4 else rank <= 3;
    if (!in_zone) return false;
    if (adjacent_files[file] == 0) return true;
    const adjacent_file_mask = adjacent_files[file];
    const attack_ranks: u64 = if (c == Colors.white)
        if (rank < 7)
            @as(u64, 0xFFFFFFFFFFFFFFFF) << ((rank + 1) * 8)
        else
            0
    else if (rank > 0)
        (@as(u64, 1) << (rank * 8)) - 1
    else
        0;

    return (opp_pawns_bb.bits & adjacent_file_mask & attack_ranks) == 0;
}

// Check if a square is defended by a friendly pawn (on diagonal behind).
fn isDefendedByPawn(c: Color, file: u6, rank: u6, our_pawns_bb: Bitboard) bool {
    if ((c == Colors.white and rank == 0) or (c == Colors.black and rank == 7)) return false;
    const def_rank: u6 = if (c == Colors.white) rank - 1 else rank + 1;
    var mask: u64 = 0;
    if (file > 0) mask |= @as(u64, 1) << (@as(u6, file - 1) + def_rank * 8);
    if (file < 7) mask |= @as(u64, 1) << (@as(u6, file + 1) + def_rank * 8);
    return (our_pawns_bb.bits & mask) != 0;
}

// Compute all pawn attacks for a color using bulk bit shifts.
fn pawnAttacksBB(pawns: u64, c: Color) u64 {
    if (c == Colors.white) {
        return ((pawns << 7) & ~FILE_H) | ((pawns << 9) & ~FILE_A);
    } else {
        return ((pawns >> 7) & ~FILE_A) | ((pawns >> 9) & ~FILE_H);
    }
}

// Compute the set of our pieces pinned against our king by enemy sliders.
fn blockersForKing(state: *const State, c: Color) u64 {
    const our_bb = state.colorBitboard(c).bits;
    const opp_bb = state.colorBitboard(~c).bits;
    const king_bb = state.pieceBitboard(piece.king).bits & our_bb;
    const king_sq: u6 = @intCast(@ctz(king_bb));
    const all = state.all_pieces.bits;

    const enemy_rq = (state.pieceBitboard(piece.rook).bits | state.pieceBitboard(piece.queen).bits) & opp_bb;
    const enemy_bq = (state.pieceBitboard(piece.bishop).bits | state.pieceBitboard(piece.queen).bits) & opp_bb;

    // Find snipers: enemy sliders that lie on a ray from the king
    var snipers: u64 = 0;
    // Rook-like rays: N(0), E(2), S(4), W(6)
    snipers |= (square.ray_attacks[0][king_sq] | square.ray_attacks[2][king_sq] |
        square.ray_attacks[4][king_sq] | square.ray_attacks[6][king_sq]) & enemy_rq;
    // Bishop-like rays: NE(1), SE(3), SW(5), NW(7)
    snipers |= (square.ray_attacks[1][king_sq] | square.ray_attacks[3][king_sq] |
        square.ray_attacks[5][king_sq] | square.ray_attacks[7][king_sq]) & enemy_bq;

    var blockers: u64 = 0;
    var sniper_bb = snipers;
    while (sniper_bb != 0) {
        const sniper_sq: u6 = @intCast(@ctz(sniper_bb));
        sniper_bb &= sniper_bb - 1;

        // Compute between(king, sniper) using intersection of opposing rays
        const direction = engine.Direction.fromSquares(king_sq, sniper_sq);
        const away_dir = square.toRayDirection(direction, king_sq, sniper_sq) orelse continue;
        const toward_dir = away_dir.opposite();
        const between = square.ray_attacks[@intFromEnum(away_dir)][king_sq] &
            square.ray_attacks[@intFromEnum(toward_dir)][sniper_sq];
        const between_occ = between & all;
        // Exactly one piece in between => it's a blocker (pinned piece)
        if (between_occ != 0 and (between_occ & (between_occ - 1)) == 0) {
            blockers |= between_occ & our_bb;
        }
    }
    return blockers;
}

// Compute the mobility area mask: squares valid for mobility counting.
// Excludes: enemy-pawn-attacked squares, our blocked/low-rank pawns,
// our king, our queens, and our pinned pieces.
fn computeMobilityArea(state: *const State, c: Color, enemy_pawn_attacks: u64, blockers: u64) u64 {
    const our_bb = state.colorBitboard(c).bits;
    const our_pawns = state.pieceBitboard(piece.pawn).bits & our_bb;
    const our_kings = state.pieceBitboard(piece.king).bits & our_bb;
    const our_queens = state.pieceBitboard(piece.queen).bits & our_bb;

    // Pawns that are blocked (piece directly ahead) or on low ranks
    const low_ranks = if (c == Colors.white) LOW_RANKS_WHITE else LOW_RANKS_BLACK;
    const shift_back: u64 = if (c == Colors.white) state.all_pieces.bits >> 8 else state.all_pieces.bits << 8;
    const blocked_or_low = our_pawns & (shift_back | low_ranks);

    return ~(enemy_pawn_attacks | blocked_or_low | our_kings | our_queens | blockers);
}

// Single-pass evaluation for one color. Iterates each piece type once,
// accumulating material, PST, mobility, and structural scores together.
// Returns the total Score, phase accumulator, and accumulated piece attacks.
fn evaluateColor(
    state: *const State,
    c: Color,
    our_pieces: u64,
    our_pawns_bb: Bitboard,
    opp_pawns_bb: Bitboard,
    mobility_area: u64,
) struct { score: Score, phase: i32, attacks: PieceAttacks } {
    var score = Score.zero;
    var phase: i32 = 0;
    var attacks = PieceAttacks{};
    const occupied = state.all_pieces.bits;

    // X-ray occupancy: bishops see through own queens, rooks see through own rooks+queens
    const our_queens_bb = state.pieceBitboard(piece.queen).bits & our_pieces;
    const occ_without_our_queens = Bitboard{ .bits = occupied ^ our_queens_bb };
    const our_rq = (state.pieceBitboard(piece.rook).bits | our_queens_bb) & our_pieces;
    const occ_without_our_rq = Bitboard{ .bits = occupied ^ our_rq };

    // --- Pawns: material + PST + pawn structure ---
    {
        var pawns = our_pawns_bb;
        const pawn_count: i32 = @intCast(pawns.popCount());
        score = score.add(piece_values[piece.pawn].mul(pawn_count));

        // Doubled pawn penalty: apply once per extra pawn on each file
        for (0..8) |file| {
            const count: i32 = @intCast(@popCount(our_pawns_bb.bits & file_masks[file]));
            if (count > 1) {
                score = score.add(doubled_pawn.mul(count - 1));
            }
        }

        while (pawns.next()) |s| {
            const file: u6 = s % 8;
            const rank: u6 = s / 8;

            // PST
            const sq: u6 = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            score = score.add(pst[piece.pawn][sq]);

            // Connected pawn
            const connected = blk: {
                var mask: u64 = 0;
                if (file > 0) mask |= @as(u64, 1) << (s - 1);
                if (file < 7) mask |= @as(u64, 1) << (s + 1);
                if (c == Colors.white and rank > 0) {
                    if (file > 0) mask |= @as(u64, 1) << (s - 9);
                    if (file < 7) mask |= @as(u64, 1) << (s - 7);
                } else if (c == Colors.black and rank < 7) {
                    if (file > 0) mask |= @as(u64, 1) << (s + 7);
                    if (file < 7) mask |= @as(u64, 1) << (s + 9);
                }
                break :blk (our_pawns_bb.bits & mask) != 0;
            };
            if (connected) {
                score = score.add(connected_pawn);
            }

            // Passed pawn
            const ahead_mask = computePassedPawnMask(c, file, rank);
            if ((opp_pawns_bb.bits & ahead_mask) == 0) {
                const passed_rank: usize = if (c == Colors.white) rank else 7 - rank;
                score = score.add(passed_pawn_bonus[passed_rank]);

                // Protected passed pawn
                const is_protected = blk: {
                    var def_mask: u64 = 0;
                    if (c == Colors.white and rank > 0) {
                        if (file > 0) def_mask |= @as(u64, 1) << (s - 9);
                        if (file < 7) def_mask |= @as(u64, 1) << (s - 7);
                    } else if (c == Colors.black and rank < 7) {
                        if (file > 0) def_mask |= @as(u64, 1) << (s + 7);
                        if (file < 7) def_mask |= @as(u64, 1) << (s + 9);
                    }
                    break :blk (our_pawns_bb.bits & def_mask) != 0;
                };
                if (is_protected) {
                    score = score.add(protected_passed_pawn);
                }

                // Blocked passed pawn
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

            // Isolated pawn
            if ((our_pawns_bb.bits & adjacent_files[file]) == 0) {
                score = score.add(isolated_pawn);
            } else {
                // Backward pawn: no friendly pawns on adjacent files at same rank or behind,
                // and stop square is attacked by enemy pawn
                const behind_mask = blk: {
                    const adj = adjacent_files[file];
                    const at_or_behind: u64 = if (c == Colors.white)
                        (@as(u64, 1) << ((rank + 1) * 8)) - 1
                    else
                        @as(u64, 0xFFFFFFFFFFFFFFFF) << (rank * 8);
                    break :blk adj & at_or_behind;
                };
                if ((our_pawns_bb.bits & behind_mask) == 0) {
                    // Check if stop square is attacked by enemy pawn
                    const stop_sq: u6 = if (c == Colors.white) s + 8 else s - 8;
                    const stop_file: u6 = stop_sq % 8;
                    const stop_rank: u6 = stop_sq / 8;
                    const stop_attacked = blk: {
                        // Enemy pawn attacks the stop square from adjacent files, one rank beyond
                        const atk_rank: u6 = if (c == Colors.white) stop_rank + 1 else if (stop_rank > 0) stop_rank - 1 else break :blk false;
                        if (c == Colors.black and stop_rank == 0) break :blk false;
                        if (c == Colors.white and atk_rank > 7) break :blk false;
                        var atk_mask: u64 = 0;
                        if (stop_file > 0) atk_mask |= @as(u64, 1) << (@as(u6, stop_file - 1) + atk_rank * 8);
                        if (stop_file < 7) atk_mask |= @as(u64, 1) << (@as(u6, stop_file + 1) + atk_rank * 8);
                        break :blk (opp_pawns_bb.bits & atk_mask) != 0;
                    };
                    if (stop_attacked) {
                        score = score.add(backward_pawn);
                    }
                }
            }
        }
    }

    // --- Knights: material + PST + mobility + outposts ---
    {
        var knights = state.pieceBitboard(piece.knight).bitAnd(u64, our_pieces);
        const knight_count: i32 = @intCast(knights.popCount());
        score = score.add(piece_values[piece.knight].mul(knight_count));
        phase += knight_count * phase_weights[piece.knight];

        while (knights.next()) |s| {
            const file: u3 = @intCast(s % 8);
            const rank: u3 = @intCast(s / 8);

            // PST
            const sq: u6 = if (c == Colors.black) @intCast((@as(u6, 7) - rank) * 8 + file) else s;
            score = score.add(pst[piece.knight][sq]);

            // Mobility (using mobility area instead of ~our_pieces)
            const knight_atk = movegen.knight_move_mask[s];
            attacks.knight |= knight_atk;
            const mob = knight_atk & mobility_area;
            const move_count: usize = @intCast(@popCount(mob));
            score = score.add(mobility_bonus[0][move_count]);

            // Outpost check (only defended outposts rewarded)
            if (isOutpost(c, file, rank, opp_pawns_bb) and
                isDefendedByPawn(c, file, rank, our_pawns_bb))
            {
                score = score.add(knight_outpost_defended);
            }
        }
    }

    // --- Bishops: material + PST + mobility + bishop pair + outposts ---
    {
        var bishops = state.pieceBitboard(piece.bishop).bitAnd(u64, our_pieces);
        const bishop_count: i32 = @intCast(bishops.popCount());
        score = score.add(piece_values[piece.bishop].mul(bishop_count));
        phase += bishop_count * phase_weights[piece.bishop];

        if (bishop_count >= 2) {
            score = score.add(bishop_pair);
        }

        while (bishops.next()) |s| {
            const rank: u6 = s / 8;
            const file: u6 = s % 8;

            // PST
            const sq: u6 = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            score = score.add(pst[piece.bishop][sq]);

            // Mobility (x-ray through own queens, using mobility area)
            const bishop_atk = movegen.sliderMovesWithOccupancy(s, piece.bishop, occ_without_our_queens);
            attacks.bishop |= bishop_atk;
            const mob = bishop_atk & mobility_area;
            const move_count: usize = @intCast(@popCount(mob));
            score = score.add(mobility_bonus[1][move_count]);

            // Bishop outpost (defended only)
            if (isOutpost(c, file, rank, opp_pawns_bb) and
                isDefendedByPawn(c, file, rank, our_pawns_bb))
            {
                score = score.add(bishop_outpost_defended);
            }
        }
    }

    // --- Rooks: material + PST + mobility + open file + 7th rank ---
    {
        var rooks = Bitboard{ .bits = state.pieceBitboard(piece.rook).bits & our_pieces };
        const rook_count: i32 = @intCast(rooks.popCount());
        score = score.add(piece_values[piece.rook].mul(rook_count));
        phase += rook_count * phase_weights[piece.rook];

        const seventh_rank: u6 = if (c == Colors.white) 6 else 1;

        while (rooks.next()) |s| {
            const file: u6 = s % 8;
            const rank: u6 = s / 8;

            // PST
            const sq: u6 = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            score = score.add(pst[piece.rook][sq]);

            // Mobility (x-ray through own rooks+queens, using mobility area)
            const rook_atk = movegen.sliderMovesWithOccupancy(s, piece.rook, occ_without_our_rq);
            attacks.rook |= rook_atk;
            const mob = rook_atk & mobility_area;
            const move_count: usize = @intCast(@popCount(mob));
            score = score.add(mobility_bonus[2][move_count]);

            // Open/semi-open file
            const fmask = file_masks[file];
            const has_our_pawn = (our_pawns_bb.bits & fmask) != 0;
            const has_opp_pawn = (opp_pawns_bb.bits & fmask) != 0;
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
    }

    // --- Queens: material + PST (mobility deferred to evaluateQueenMobility) ---
    {
        var queens = state.pieceBitboard(piece.queen).bitAnd(u64, our_pieces);
        const queen_count: i32 = @intCast(queens.popCount());
        score = score.add(piece_values[piece.queen].mul(queen_count));
        phase += queen_count * phase_weights[4];

        while (queens.next()) |s| {
            const rank: u6 = s / 8;
            const file: u6 = s % 8;
            const sq: u6 = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            score = score.add(pst[piece.queen][sq]);
        }
    }

    // --- King: PST + king safety ---
    {
        const king_bb = state.pieceBitboard(piece.king).bitAnd(u64, our_pieces);
        const king_sq: u6 = @intCast(@ctz(king_bb.bits));
        const king_file: u3 = @intCast(king_sq % 8);
        const king_rank: u3 = @intCast(king_sq / 8);

        // PST
        const pst_sq: u6 = if (c == Colors.black) @intCast((@as(u6, 7) - king_rank) * 8 + king_file) else king_sq;
        score = score.add(pst[piece.king][pst_sq]);

        // King safety: pawn shield (only when king on back ranks)
        const on_back_ranks = if (c == Colors.white) king_rank <= 1 else king_rank >= 6;
        if (on_back_ranks) {
            const shield_rank: u6 = if (c == Colors.white) @as(u6, king_rank) + 1 else @as(u6, king_rank) - 1;
            const min_file: u3 = if (king_file > 0) king_file - 1 else 0;
            const max_file: u3 = if (king_file < 7) king_file + 1 else 7;

            var file: u4 = min_file;
            while (file <= max_file) : (file += 1) {
                const shield_sq: u6 = @as(u6, @as(u3, @intCast(file))) + shield_rank * 8;
                const shield_mask: u64 = @as(u64, 1) << shield_sq;
                if ((our_pawns_bb.bits & shield_mask) != 0) {
                    score = score.add(pawn_shield);
                } else {
                    score = score.add(pawn_shield_missing);
                }
            }
        }
    }

    return .{ .score = score, .phase = phase, .attacks = attacks };
}

// Evaluate queen mobility separately, after both colors' piece attacks are known.
// Queen mobility excludes squares defended by enemy minor pieces and rooks.
fn evaluateQueenMobility(
    state: *const State,
    our_pieces: u64,
    mobility_area: u64,
    enemy_attacks: PieceAttacks,
) Score {
    var score = Score.zero;
    var queens = state.pieceBitboard(piece.queen).bitAnd(u64, our_pieces);
    const enemy_minor_rook = enemy_attacks.knight | enemy_attacks.bishop | enemy_attacks.rook;

    while (queens.next()) |s| {
        const bishop_moves = movegen.sliderMoves(state, s, piece.bishop);
        const rook_moves = movegen.sliderMoves(state, s, piece.rook);
        const queen_atk = bishop_moves | rook_moves;
        const mob = queen_atk & mobility_area & ~enemy_minor_rook;
        const move_count: usize = @intCast(@popCount(mob));
        score = score.add(mobility_bonus[3][move_count]);
    }
    return score;
}

pub fn evaluate(state: *const State) i32 {
    const to_move = state.to_move;
    const opp = ~to_move;

    // Extract bitboards once for both colors
    const our_pieces = state.colorBitboard(to_move).bits;
    const opp_pieces = state.colorBitboard(opp).bits;
    const pawn_bb = state.pieceBitboard(piece.pawn);
    const our_pawns = pawn_bb.bitAnd(u64, our_pieces);
    const opp_pawns = pawn_bb.bitAnd(u64, opp_pieces);

    // Pre-compute mobility prerequisites
    const our_pawn_atk = pawnAttacksBB(our_pawns.bits, to_move);
    const opp_pawn_atk = pawnAttacksBB(opp_pawns.bits, opp);
    const our_blockers = blockersForKing(state, to_move);
    const opp_blockers = blockersForKing(state, opp);
    const our_mob_area = computeMobilityArea(state, to_move, opp_pawn_atk, our_blockers);
    const opp_mob_area = computeMobilityArea(state, opp, our_pawn_atk, opp_blockers);

    // Evaluate both colors (accumulates attack maps, defers queen mobility)
    const our = evaluateColor(state, to_move, our_pieces, our_pawns, opp_pawns, our_mob_area);
    const their = evaluateColor(state, opp, opp_pieces, opp_pawns, our_pawns, opp_mob_area);

    // Queen mobility using opponent's accumulated attack maps
    const our_q = evaluateQueenMobility(state, our_pieces, our_mob_area, their.attacks);
    const their_q = evaluateQueenMobility(state, opp_pieces, opp_mob_area, our.attacks);

    const phase = @min(our.phase + their.phase, max_phase_mg);
    const total = our.score.add(our_q).sub(their.score).sub(their_q).add(tempo);
    return total.taper(phase);
}

pub const EvalTrace = struct {
    material: [2]Score,
    pst: [2]Score,
    pawn_structure: [2]Score,
    passed_pawns: [2]Score,
    outposts: [2]Score,
    bishop_pair: [2]Score,
    rook_bonuses: [2]Score,
    mobility: [2]Score,
    king_safety: [2]Score,
    tempo_score: Score,
    phase: i32,
    total: i32,

    pub fn dump(self: *const EvalTrace, writer: *std.Io.Writer) !void {
        try writer.print("              | White MG   EG  | Black MG   EG  |\n", .{});
        try writer.print("--------------+----------------+----------------+\n", .{});
        try printRow(writer, "Material     ", self.material);
        try printRow(writer, "PST          ", self.pst);
        try printRow(writer, "Pawn struct  ", self.pawn_structure);
        try printRow(writer, "Passed pawns ", self.passed_pawns);
        try printRow(writer, "Outposts     ", self.outposts);
        try printRow(writer, "Bishop pair  ", self.bishop_pair);
        try printRow(writer, "Rook bonuses ", self.rook_bonuses);
        try printRow(writer, "Mobility     ", self.mobility);
        try printRow(writer, "King safety  ", self.king_safety);
        try writer.print("Tempo         | {d:>6}  {d:>6} |                |\n", .{ self.tempo_score.midgame(), self.tempo_score.endgame() });
        try writer.print("--------------+----------------+----------------+\n", .{});
        try writer.print("Phase: {d}/24\n", .{self.phase});
        const cp_total: f32 = @as(f32, @floatFromInt(self.total)) / @as(f32, @floatFromInt(piece_values[0].endgame()));
        const sign: []const u8 = if (cp_total < 0) "-" else if (cp_total > 0) "+" else "±";
        try writer.print("Total: {s}{d:.2} ({d})\n", .{ sign, @abs(cp_total), self.total });
    }

    fn printRow(writer: *std.Io.Writer, label: []const u8, scores: [2]Score) !void {
        try writer.print("{s} | {d:>6}  {d:>6} | {d:>6}  {d:>6} |\n", .{
            label,
            scores[0].midgame(),
            scores[0].endgame(),
            scores[1].midgame(),
            scores[1].endgame(),
        });
    }
};

fn evaluateColorTrace(
    state: *const State,
    c: Color,
    our_pieces: u64,
    our_pawns_bb: Bitboard,
    opp_pawns_bb: Bitboard,
    mobility_area: u64,
) struct {
    material: Score,
    pst_score: Score,
    pawn_structure: Score,
    passed_pawns_score: Score,
    outposts_score: Score,
    bishop_pair_score: Score,
    rook_bonuses_score: Score,
    mobility_score: Score,
    king_safety_score: Score,
    phase: i32,
    attacks: PieceAttacks,
} {
    var material_score = Score.zero;
    var pst_score = Score.zero;
    var pawn_structure_score = Score.zero;
    var passed_pawns_score = Score.zero;
    var outposts_score = Score.zero;
    var bishop_pair_score = Score.zero;
    var rook_bonuses_score = Score.zero;
    var mobility_score = Score.zero;
    var king_safety_score = Score.zero;
    var phase: i32 = 0;
    var attacks = PieceAttacks{};
    const occupied = state.all_pieces.bits;

    // X-ray occupancy
    const our_queens_bb = state.pieceBitboard(piece.queen).bits & our_pieces;
    const occ_without_our_queens = Bitboard{ .bits = occupied ^ our_queens_bb };
    const our_rq = (state.pieceBitboard(piece.rook).bits | our_queens_bb) & our_pieces;
    const occ_without_our_rq = Bitboard{ .bits = occupied ^ our_rq };

    // --- Pawns ---
    {
        var pawns = our_pawns_bb;
        const pawn_count: i32 = @intCast(pawns.popCount());
        material_score = material_score.add(piece_values[piece.pawn].mul(pawn_count));

        for (0..8) |file| {
            const count: i32 = @intCast(@popCount(our_pawns_bb.bits & file_masks[file]));
            if (count > 1) {
                pawn_structure_score = pawn_structure_score.add(doubled_pawn.mul(count - 1));
            }
        }

        while (pawns.next()) |s| {
            const file: u6 = s % 8;
            const rank: u6 = s / 8;
            const sq: u6 = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            pst_score = pst_score.add(pst[piece.pawn][sq]);

            const connected = blk: {
                var mask: u64 = 0;
                if (file > 0) mask |= @as(u64, 1) << (s - 1);
                if (file < 7) mask |= @as(u64, 1) << (s + 1);
                if (c == Colors.white and rank > 0) {
                    if (file > 0) mask |= @as(u64, 1) << (s - 9);
                    if (file < 7) mask |= @as(u64, 1) << (s - 7);
                } else if (c == Colors.black and rank < 7) {
                    if (file > 0) mask |= @as(u64, 1) << (s + 7);
                    if (file < 7) mask |= @as(u64, 1) << (s + 9);
                }
                break :blk (our_pawns_bb.bits & mask) != 0;
            };
            if (connected) {
                pawn_structure_score = pawn_structure_score.add(connected_pawn);
            }

            const ahead_mask = computePassedPawnMask(c, file, rank);
            if ((opp_pawns_bb.bits & ahead_mask) == 0) {
                const passed_rank: usize = if (c == Colors.white) rank else 7 - rank;
                passed_pawns_score = passed_pawns_score.add(passed_pawn_bonus[passed_rank]);

                const is_protected = blk: {
                    var def_mask: u64 = 0;
                    if (c == Colors.white and rank > 0) {
                        if (file > 0) def_mask |= @as(u64, 1) << (s - 9);
                        if (file < 7) def_mask |= @as(u64, 1) << (s - 7);
                    } else if (c == Colors.black and rank < 7) {
                        if (file > 0) def_mask |= @as(u64, 1) << (s + 7);
                        if (file < 7) def_mask |= @as(u64, 1) << (s + 9);
                    }
                    break :blk (our_pawns_bb.bits & def_mask) != 0;
                };
                if (is_protected) {
                    passed_pawns_score = passed_pawns_score.add(protected_passed_pawn);
                }

                const blocked = blk: {
                    if (c == Colors.white and rank < 7) {
                        break :blk (occupied & (@as(u64, 1) << (s + 8))) != 0;
                    } else if (c == Colors.black and rank > 0) {
                        break :blk (occupied & (@as(u64, 1) << (s - 8))) != 0;
                    }
                    break :blk false;
                };
                if (blocked) {
                    passed_pawns_score = passed_pawns_score.add(blocked_passed_pawn);
                }
            }

            if ((our_pawns_bb.bits & adjacent_files[file]) == 0) {
                pawn_structure_score = pawn_structure_score.add(isolated_pawn);
            } else {
                // Backward pawn detection
                const behind_mask = blk: {
                    const adj = adjacent_files[file];
                    const at_or_behind: u64 = if (c == Colors.white)
                        (@as(u64, 1) << ((rank + 1) * 8)) - 1
                    else
                        @as(u64, 0xFFFFFFFFFFFFFFFF) << (rank * 8);
                    break :blk adj & at_or_behind;
                };
                if ((our_pawns_bb.bits & behind_mask) == 0) {
                    const stop_sq: u6 = if (c == Colors.white) s + 8 else s - 8;
                    const stop_file: u6 = stop_sq % 8;
                    const stop_rank: u6 = stop_sq / 8;
                    const stop_attacked = blk: {
                        const atk_rank: u6 = if (c == Colors.white) stop_rank + 1 else if (stop_rank > 0) stop_rank - 1 else break :blk false;
                        if (c == Colors.black and stop_rank == 0) break :blk false;
                        if (c == Colors.white and atk_rank > 7) break :blk false;
                        var atk_mask: u64 = 0;
                        if (stop_file > 0) atk_mask |= @as(u64, 1) << (@as(u6, stop_file - 1) + atk_rank * 8);
                        if (stop_file < 7) atk_mask |= @as(u64, 1) << (@as(u6, stop_file + 1) + atk_rank * 8);
                        break :blk (opp_pawns_bb.bits & atk_mask) != 0;
                    };
                    if (stop_attacked) {
                        pawn_structure_score = pawn_structure_score.add(backward_pawn);
                    }
                }
            }
        }
    }

    // --- Knights ---
    {
        var knights = state.pieceBitboard(piece.knight).bitAnd(u64, our_pieces);
        const knight_count: i32 = @intCast(knights.popCount());
        material_score = material_score.add(piece_values[piece.knight].mul(knight_count));
        phase += knight_count * phase_weights[piece.knight];

        while (knights.next()) |s| {
            const file: u3 = @intCast(s % 8);
            const rank: u3 = @intCast(s / 8);
            const sq: u6 = if (c == Colors.black) @intCast((@as(u6, 7) - rank) * 8 + file) else s;
            pst_score = pst_score.add(pst[piece.knight][sq]);

            const knight_atk = movegen.knight_move_mask[s];
            attacks.knight |= knight_atk;
            const mob = knight_atk & mobility_area;
            const move_count: usize = @intCast(@popCount(mob));
            mobility_score = mobility_score.add(mobility_bonus[0][move_count]);

            if (isOutpost(c, file, rank, opp_pawns_bb) and
                isDefendedByPawn(c, file, rank, our_pawns_bb))
            {
                outposts_score = outposts_score.add(knight_outpost_defended);
            }
        }
    }

    // --- Bishops ---
    {
        var bishops = state.pieceBitboard(piece.bishop).bitAnd(u64, our_pieces);
        const bishop_count: i32 = @intCast(bishops.popCount());
        material_score = material_score.add(piece_values[piece.bishop].mul(bishop_count));
        phase += bishop_count * phase_weights[piece.bishop];

        if (bishop_count >= 2) {
            bishop_pair_score = bishop_pair_score.add(bishop_pair);
        }

        while (bishops.next()) |s| {
            const rank: u6 = s / 8;
            const file: u6 = s % 8;
            const sq: u6 = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            pst_score = pst_score.add(pst[piece.bishop][sq]);

            const bishop_atk = movegen.sliderMovesWithOccupancy(s, piece.bishop, occ_without_our_queens);
            attacks.bishop |= bishop_atk;
            const mob = bishop_atk & mobility_area;
            const move_count: usize = @intCast(@popCount(mob));
            mobility_score = mobility_score.add(mobility_bonus[1][move_count]);

            // Bishop outpost (defended only)
            if (isOutpost(c, file, rank, opp_pawns_bb) and
                isDefendedByPawn(c, file, rank, our_pawns_bb))
            {
                outposts_score = outposts_score.add(bishop_outpost_defended);
            }
        }
    }

    // --- Rooks ---
    {
        var rooks = state.pieceBitboard(piece.rook).bitAnd(u64, our_pieces);
        const rook_count: i32 = @intCast(rooks.popCount());
        material_score = material_score.add(piece_values[piece.rook].mul(rook_count));
        phase += rook_count * phase_weights[piece.rook];

        const seventh_rank: u6 = if (c == Colors.white) 6 else 1;

        while (rooks.next()) |s| {
            const file: u6 = s % 8;
            const rank: u6 = s / 8;
            const sq: u6 = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            pst_score = pst_score.add(pst[piece.rook][sq]);

            const rook_atk = movegen.sliderMovesWithOccupancy(s, piece.rook, occ_without_our_rq);
            attacks.rook |= rook_atk;
            const mob = rook_atk & mobility_area;
            const move_count: usize = @intCast(@popCount(mob));
            mobility_score = mobility_score.add(mobility_bonus[2][move_count]);

            const fmask = file_masks[file];
            const has_our_pawn = (our_pawns_bb.bits & fmask) != 0;
            const has_opp_pawn = (opp_pawns_bb.bits & fmask) != 0;
            if (!has_our_pawn and !has_opp_pawn) {
                rook_bonuses_score = rook_bonuses_score.add(rook_open_file);
            } else if (!has_our_pawn and has_opp_pawn) {
                rook_bonuses_score = rook_bonuses_score.add(rook_semi_open);
            }

            if (rank == seventh_rank) {
                rook_bonuses_score = rook_bonuses_score.add(rook_on_seventh);
            }
        }
    }

    // --- Queens (material + PST only, mobility deferred) ---
    {
        var queens = state.pieceBitboard(piece.queen).bitAnd(u64, our_pieces);
        const queen_count: i32 = @intCast(queens.popCount());
        material_score = material_score.add(piece_values[piece.queen].mul(queen_count));
        phase += queen_count * phase_weights[piece.queen];

        while (queens.next()) |s| {
            const rank: u6 = s / 8;
            const file: u6 = s % 8;
            const sq: u6 = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            pst_score = pst_score.add(pst[piece.queen][sq]);
        }
    }

    // --- King ---
    {
        const king_bb = state.pieceBitboard(piece.king).bitAnd(u64, our_pieces);
        const king_sq: u6 = @intCast(@ctz(king_bb.bits));
        const king_file: u3 = @intCast(king_sq % 8);
        const king_rank: u3 = @intCast(king_sq / 8);
        const pst_sq: u6 = if (c == Colors.black) @intCast((@as(u6, 7) - king_rank) * 8 + king_file) else king_sq;
        pst_score = pst_score.add(pst[piece.king][pst_sq]);

        const on_back_ranks = if (c == Colors.white) king_rank <= 1 else king_rank >= 6;
        if (on_back_ranks) {
            const shield_rank: u6 = if (c == Colors.white) @as(u6, king_rank) + 1 else @as(u6, king_rank) - 1;
            const min_file: u3 = if (king_file > 0) king_file - 1 else 0;
            const max_file: u3 = if (king_file < 7) king_file + 1 else 7;

            var f: u4 = min_file;
            while (f <= max_file) : (f += 1) {
                const shield_sq: u6 = @as(u6, @as(u3, @intCast(f))) + shield_rank * 8;
                const shield_mask: u64 = @as(u64, 1) << shield_sq;
                if ((our_pawns_bb.bits & shield_mask) != 0) {
                    king_safety_score = king_safety_score.add(pawn_shield);
                } else {
                    king_safety_score = king_safety_score.add(pawn_shield_missing);
                }
            }
        }
    }

    return .{
        .material = material_score,
        .pst_score = pst_score,
        .pawn_structure = pawn_structure_score,
        .passed_pawns_score = passed_pawns_score,
        .outposts_score = outposts_score,
        .bishop_pair_score = bishop_pair_score,
        .rook_bonuses_score = rook_bonuses_score,
        .mobility_score = mobility_score,
        .king_safety_score = king_safety_score,
        .phase = phase,
        .attacks = attacks,
    };
}

fn evaluateQueenMobilityTrace(
    state: *const State,
    our_pieces: u64,
    mobility_area: u64,
    enemy_attacks: PieceAttacks,
) Score {
    var score = Score.zero;
    var queens = state.pieceBitboard(piece.queen).bitAnd(u64, our_pieces);
    const enemy_minor_rook = enemy_attacks.knight | enemy_attacks.bishop | enemy_attacks.rook;

    while (queens.next()) |s| {
        const bishop_moves = movegen.sliderMoves(state, s, piece.bishop);
        const rook_moves = movegen.sliderMoves(state, s, piece.rook);
        const queen_atk = bishop_moves | rook_moves;
        const mob = queen_atk & mobility_area & ~enemy_minor_rook;
        const move_count: usize = @intCast(@popCount(mob));
        score = score.add(mobility_bonus[3][move_count]);
    }
    return score;
}

pub fn evaluateTrace(state: *const State) EvalTrace {
    const to_move = state.to_move;
    const opp = ~to_move;

    const our_pieces = state.colorBitboard(to_move).bits;
    const opp_pieces = state.colorBitboard(opp).bits;
    const pawn_bb = state.pieceBitboard(piece.pawn);
    const our_pawns = pawn_bb.bitAnd(u64, our_pieces);
    const opp_pawns = pawn_bb.bitAnd(u64, opp_pieces);

    // Pre-compute mobility prerequisites
    const our_pawn_atk = pawnAttacksBB(our_pawns.bits, to_move);
    const opp_pawn_atk = pawnAttacksBB(opp_pawns.bits, opp);
    const our_blockers = blockersForKing(state, to_move);
    const opp_blockers = blockersForKing(state, opp);
    const our_mob_area = computeMobilityArea(state, to_move, opp_pawn_atk, our_blockers);
    const opp_mob_area = computeMobilityArea(state, opp, our_pawn_atk, opp_blockers);

    // White is index 0, black is index 1 regardless of side to move
    const white_idx: usize = if (to_move == Colors.white) 0 else 1;
    const black_idx: usize = 1 - white_idx;

    const our = evaluateColorTrace(state, to_move, our_pieces, our_pawns, opp_pawns, our_mob_area);
    const their = evaluateColorTrace(state, opp, opp_pieces, opp_pawns, our_pawns, opp_mob_area);

    // Queen mobility using opponent's accumulated attack maps
    const our_q = evaluateQueenMobilityTrace(state, our_pieces, our_mob_area, their.attacks);
    const their_q = evaluateQueenMobilityTrace(state, opp_pieces, opp_mob_area, our.attacks);

    var trace: EvalTrace = undefined;
    trace.material[white_idx] = our.material;
    trace.material[black_idx] = their.material;
    trace.pst[white_idx] = our.pst_score;
    trace.pst[black_idx] = their.pst_score;
    trace.pawn_structure[white_idx] = our.pawn_structure;
    trace.pawn_structure[black_idx] = their.pawn_structure;
    trace.passed_pawns[white_idx] = our.passed_pawns_score;
    trace.passed_pawns[black_idx] = their.passed_pawns_score;
    trace.outposts[white_idx] = our.outposts_score;
    trace.outposts[black_idx] = their.outposts_score;
    trace.bishop_pair[white_idx] = our.bishop_pair_score;
    trace.bishop_pair[black_idx] = their.bishop_pair_score;
    trace.rook_bonuses[white_idx] = our.rook_bonuses_score;
    trace.rook_bonuses[black_idx] = their.rook_bonuses_score;
    trace.mobility[white_idx] = our.mobility_score.add(our_q);
    trace.mobility[black_idx] = their.mobility_score.add(their_q);
    trace.king_safety[white_idx] = our.king_safety_score;
    trace.king_safety[black_idx] = their.king_safety_score;

    trace.tempo_score = tempo;
    trace.phase = @min(our.phase + their.phase, max_phase_mg);
    const total = our.material.add(our.pst_score).add(our.pawn_structure).add(our.passed_pawns_score)
        .add(our.outposts_score).add(our.bishop_pair_score).add(our.rook_bonuses_score)
        .add(our.mobility_score).add(our_q).add(our.king_safety_score)
        .sub(their.material).sub(their.pst_score).sub(their.pawn_structure).sub(their.passed_pawns_score)
        .sub(their.outposts_score).sub(their.bishop_pair_score).sub(their.rook_bonuses_score)
        .sub(their.mobility_score).sub(their_q).sub(their.king_safety_score)
        .add(tempo);
    trace.total = total.taper(trace.phase);

    return trace;
}

// History heuristic table: [color][from_square][to_square] -> score
// Tracks which quiet moves have caused beta cutoffs
pub const HistoryTable = struct {
    table: [2][64][64]i32 = [_][64][64]i32{[_][64]i32{[_]i32{0} ** 64} ** 64} ** 2,

    pub fn get(self: *const HistoryTable, color: Color, from: u6, to: u6) i32 {
        return self.table[color][from][to];
    }

    const max_history: i32 = 16384;

    pub fn update(self: *HistoryTable, color: Color, from: u6, to: u6, bonus: i32) void {
        const entry = &self.table[color][from][to];
        // Gravity formula: bonus is damped as value approaches max_history
        // This provides natural aging — large values get smaller effective bonuses
        entry.* += bonus - @divTrunc(entry.* * @as(i32, @intCast(@abs(bonus))), max_history);
    }

    pub fn clear(self: *HistoryTable) void {
        self.table = [_][64][64]i32{[_][64]i32{[_]i32{0} ** 64} ** 64} ** 2;
    }
};

pub fn scoreMove(ctx: *const MoveList.SortCtx, m: Move) i32 {
    // TT move gets maximum priority
    if (ctx.tt_move) |tt| {
        if (tt.eql(m)) {
            return 100_000;
        }
    }

    var score: i32 = 0;
    const p = ctx.state.mailbox[m.start].?;

    // MVV-LVA for captures
    if (ctx.state.mailbox[m.end]) |captured_piece| {
        const attacker_piece = ctx.state.mailbox[m.start].?;
        score += piece_values_mg[captured_piece] * 10 - piece_values_mg[attacker_piece];
    } else {
        // Check for en-passant
        if (square.absDiff(m.start, m.end) % 8 != 0) {
            score += piece_values_mg[piece.pawn] * 10 - piece_values_mg[piece.pawn];
        }
    }

    // Promotion bonus
    const end_rank = m.end / 8;
    if (p == piece.pawn and ((end_rank == 7 and ctx.color == Colors.white) or
        (end_rank == 0 and ctx.color == Colors.black)))
    {
        score += promotion_bonus;
    }

    // Killer move bonus (below captures, above quiet moves)
    if (ctx.killers[0]) |k| {
        if (k.start == m.start and k.end == m.end) {
            score += 1100;
        }
    }
    if (ctx.killers[1]) |k| {
        if (k.start == m.start and k.end == m.end) {
            score += 1000;
        }
    }

    // Countermove bonus (between killers and history)
    if (ctx.countermove) |cm| {
        if (cm.start == m.start and cm.end == m.end) {
            score += 1050;
        }
    }

    // History heuristic for quiet moves (non-captures, non-promotions)
    if (ctx.state.mailbox[m.end] == null and ctx.history != null) {
        const is_promotion = ctx.state.mailbox[m.start] == piece.pawn and
            ((end_rank == 7 and ctx.color == Colors.white) or (end_rank == 0 and ctx.color == Colors.black));
        if (!is_promotion) {
            score += @divTrunc(ctx.history.?.get(ctx.color, m.start, m.end), 32);
        }
    }

    return score;
}
