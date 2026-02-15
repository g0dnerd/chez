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

// Phase weights for tapered evaluation
const phase_weights = [6]i32{ 0, 1, 1, 2, 4, 0 }; // pawn, knight, bishop, rook, queen, king
const max_phase_mg: i32 = 24;

// Packed score holding both middlegame and endgame values.
// Allows evaluating once and interpolating at the end based on game phase.
pub const Score = packed struct {
    midgame: i16,
    endgame: i16,

    pub const zero = Score{ .midgame = 0, .endgame = 0 };

    pub fn add(self: Score, other: Score) Score {
        return .{ .midgame = self.midgame + other.midgame, .endgame = self.endgame + other.endgame };
    }

    pub fn sub(self: Score, other: Score) Score {
        return .{ .midgame = self.midgame - other.midgame, .endgame = self.endgame - other.endgame };
    }

    pub fn mul(self: Score, n: i32) Score {
        return .{
            .midgame = @intCast(self.midgame * @as(i16, @intCast(n))),
            .endgame = @intCast(self.endgame * @as(i16, @intCast(n))),
        };
    }

    pub fn neg(self: Score) Score {
        return .{ .midgame = -self.midgame, .endgame = -self.endgame };
    }

    // Interpolate between MG and EG based on phase (0 = endgame, 24 = opening)
    pub fn taper(self: Score, phase: i32) i32 {
        return @divTrunc(@as(i32, self.midgame) * phase + @as(i32, self.endgame) * (max_phase_mg - phase), max_phase_mg);
    }
};

pub const piece_values = [6]Score{
    Score{ .midgame = 124, .endgame = 206 }, // pawn
    Score{ .midgame = 781, .endgame = 854 }, // knight
    Score{ .midgame = 825, .endgame = 915 }, // bishop
    Score{ .midgame = 1276, .endgame = 1380 }, // rook
    Score{ .midgame = 2538, .endgame = 2682 }, // queen
    Score{ .midgame = 20000, .endgame = 20000 }, // king
};

// For MVV-LVA move ordering (uses middlegame values)
pub const piece_values_mg = [6]i32{ 100, 305, 333, 563, 950, 20000 };

// Passed pawn bonus by rank (from pawn's perspective, rank 1-6 relevant)
const passed_pawn_bonus = [8]Score{
    Score{ .midgame = 0, .endgame = 0 }, // rank 0 (impossible for white)
    Score{ .midgame = 5, .endgame = 10 }, // rank 1
    Score{ .midgame = 10, .endgame = 20 }, // rank 2
    Score{ .midgame = 20, .endgame = 40 }, // rank 3
    Score{ .midgame = 35, .endgame = 70 }, // rank 4
    Score{ .midgame = 60, .endgame = 120 }, // rank 5
    Score{ .midgame = 100, .endgame = 200 }, // rank 6
    Score{ .midgame = 0, .endgame = 0 }, // rank 7 (promoted)
};

// Mobility bonus per move (middlegame, endgame)
const mobility_bonus = [4][28]Score{
    // Knights
    [_]Score{
        Score{ .midgame = -62, .endgame = -81 },
        Score{ .midgame = -53, .endgame = -56 },
        Score{ .midgame = -12, .endgame = -31 },
        Score{ .midgame = -4, .endgame = -16 },
        Score{ .midgame = 3, .endgame = 5 },
        Score{ .midgame = 13, .endgame = 11 },
        Score{ .midgame = 22, .endgame = 17 },
        Score{ .midgame = 28, .endgame = 20 },
        Score{ .midgame = 33, .endgame = 25 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
    },
    // Bishops
    [_]Score{
        Score{ .midgame = -48, .endgame = -59 },
        Score{ .midgame = -20, .endgame = -23 },
        Score{ .midgame = 16, .endgame = -3 },
        Score{ .midgame = 26, .endgame = 13 },
        Score{ .midgame = 38, .endgame = 24 },
        Score{ .midgame = 51, .endgame = 42 },
        Score{ .midgame = 55, .endgame = 54 },
        Score{ .midgame = 63, .endgame = 57 },
        Score{ .midgame = 63, .endgame = 65 },
        Score{ .midgame = 68, .endgame = 73 },
        Score{ .midgame = 81, .endgame = 78 },
        Score{ .midgame = 81, .endgame = 86 },
        Score{ .midgame = 91, .endgame = 88 },
        Score{ .midgame = 98, .endgame = 97 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
    },
    // Rooks
    [_]Score{
        Score{ .midgame = -60, .endgame = -78 },
        Score{ .midgame = -20, .endgame = -17 },
        Score{ .midgame = 2, .endgame = 23 },
        Score{ .midgame = 3, .endgame = 39 },
        Score{ .midgame = 3, .endgame = 70 },
        Score{ .midgame = 11, .endgame = 99 },
        Score{ .midgame = 22, .endgame = 103 },
        Score{ .midgame = 31, .endgame = 121 },
        Score{ .midgame = 40, .endgame = 134 },
        Score{ .midgame = 40, .endgame = 139 },
        Score{ .midgame = 41, .endgame = 158 },
        Score{ .midgame = 48, .endgame = 164 },
        Score{ .midgame = 57, .endgame = 168 },
        Score{ .midgame = 57, .endgame = 169 },
        Score{ .midgame = 62, .endgame = 172 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
        Score{ .midgame = 0, .endgame = 0 },
    },
    // Queens
    [_]Score{
        Score{ .midgame = -30, .endgame = -48 },
        Score{ .midgame = -12, .endgame = -30 },
        Score{ .midgame = -8, .endgame = -7 },
        Score{ .midgame = -9, .endgame = 19 },
        Score{ .midgame = 20, .endgame = 40 },
        Score{ .midgame = 23, .endgame = 55 },
        Score{ .midgame = 23, .endgame = 59 },
        Score{ .midgame = 35, .endgame = 75 },
        Score{ .midgame = 38, .endgame = 78 },
        Score{ .midgame = 53, .endgame = 96 },
        Score{ .midgame = 64, .endgame = 96 },
        Score{ .midgame = 65, .endgame = 100 },
        Score{ .midgame = 65, .endgame = 121 },
        Score{ .midgame = 66, .endgame = 127 },
        Score{ .midgame = 67, .endgame = 131 },
        Score{ .midgame = 67, .endgame = 133 },
        Score{ .midgame = 72, .endgame = 136 },
        Score{ .midgame = 72, .endgame = 141 },
        Score{ .midgame = 77, .endgame = 147 },
        Score{ .midgame = 79, .endgame = 150 },
        Score{ .midgame = 93, .endgame = 151 },
        Score{ .midgame = 108, .endgame = 168 },
        Score{ .midgame = 108, .endgame = 168 },
        Score{ .midgame = 108, .endgame = 171 },
        Score{ .midgame = 110, .endgame = 182 },
        Score{ .midgame = 114, .endgame = 182 },
        Score{ .midgame = 114, .endgame = 192 },
        Score{ .midgame = 116, .endgame = 218 },
    },
};

// Bonus/penalty constants
const bishop_pair = Score{ .midgame = 30, .endgame = 50 };
const rook_open_file = Score{ .midgame = 25, .endgame = 15 };
const rook_semi_open = Score{ .midgame = 15, .endgame = 10 };
const rook_on_seventh = Score{ .midgame = 20, .endgame = 40 };
const isolated_pawn = Score{ .midgame = -15, .endgame = -20 };
const doubled_pawn = Score{ .midgame = -10, .endgame = -20 };
const connected_pawn = Score{ .midgame = 7, .endgame = 10 };
const protected_passed_pawn = Score{ .midgame = 15, .endgame = 30 };
const blocked_passed_pawn = Score{ .midgame = -10, .endgame = -20 };
const knight_outpost_defended = Score{ .midgame = 25, .endgame = 15 };
const knight_outpost_undefended = Score{ .midgame = 10, .endgame = 5 };
const pawn_shield = Score{ .midgame = 15, .endgame = 0 };
const pawn_shield_missing = Score{ .midgame = -10, .endgame = 0 };

const promotion_bonus: i32 = 5000;

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
            result[i] = Score{ .midgame = mg[i], .endgame = eg[i] };
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
            result[i] = Score{ .midgame = mg[i], .endgame = eg[i] };
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
            result[i] = Score{ .midgame = mg[i], .endgame = eg[i] };
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
        // Endgame: rooks benefit from centralization and active positions
        const eg = [64]i16{
            0, 0,  0,  0,  0,  0,  0,  0,
            0, 0,  0,  0,  0,  0,  0,  0,
            0, 5,  5,  5,  5,  5,  5,  0,
            0, 5,  10, 10, 10, 10, 5,  0,
            0, 5,  10, 10, 10, 10, 5,  0,
            0, 5,  5,  5,  5,  5,  5,  0,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 5,  5,  5,  5,  5,  5,  0,
        };
        var result: [64]Score = undefined;
        for (0..64) |i| {
            result[i] = Score{ .midgame = mg[i], .endgame = eg[i] };
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
            result[i] = Score{ .midgame = mg[i], .endgame = eg[i] };
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
            result[i] = Score{ .midgame = mg[i], .endgame = eg[i] };
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

// Single-pass evaluation for one color. Iterates each piece type once,
// accumulating material, PST, mobility, and structural scores together.
// Returns the total Score and adds to the phase accumulator.
fn evaluateColor(
    state: *const State,
    c: Color,
    our_pieces: u64,
    our_pieces_not: u64,
    our_pawns_bb: Bitboard,
    opp_pawns_bb: Bitboard,
) struct { score: Score, phase: i32 } {
    var score = Score.zero;
    var phase: i32 = 0;
    const occupied = state.all_pieces.bits;

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

            // Mobility
            const moves = movegen.knight_move_mask[s] & our_pieces_not;
            const move_count: i32 = @intCast(@popCount(moves));
            score = score.add(mobility_bonus[0][@as(usize, @intCast(move_count))]);

            // Outpost check
            const in_outpost_zone = if (c == Colors.white) rank >= 4 else rank <= 3;
            if (in_outpost_zone) {
                const can_be_attacked = blk: {
                    if (adjacent_files[file] == 0) break :blk false;
                    const adjacent_file_mask = adjacent_files[file];
                    const rank_u6: u6 = rank;
                    const attack_ranks: u64 = if (c == Colors.white)
                        if (rank < 7) @as(u64, 0xFFFFFFFFFFFFFFFF) << ((rank_u6 + 1) * 8) else 0
                    else if (rank > 0) (@as(u64, 1) << (rank_u6 * 8)) - 1 else 0;
                    break :blk (opp_pawns_bb.bits & adjacent_file_mask & attack_ranks) != 0;
                };

                if (!can_be_attacked) {
                    const defended_by_pawn = blk: {
                        if (file == 0 or file == 7) {
                            const def_file: u3 = if (file == 0) 1 else 6;
                            const def_rank: u3 = if (c == Colors.white) rank - 1 else rank + 1;
                            if ((c == Colors.white and rank == 0) or (c == Colors.black and rank == 7)) break :blk false;
                            const def_sq: u6 = @as(u6, def_file) + @as(u6, def_rank) * 8;
                            break :blk (our_pawns_bb.bits & (@as(u64, 1) << def_sq)) != 0;
                        }
                        const def_rank: u3 = if (c == Colors.white) rank - 1 else rank + 1;
                        if ((c == Colors.white and rank == 0) or (c == Colors.black and rank == 7)) break :blk false;
                        const left_sq: u6 = @as(u6, file - 1) + @as(u6, def_rank) * 8;
                        const right_sq: u6 = @as(u6, file + 1) + @as(u6, def_rank) * 8;
                        const left_mask: u64 = @as(u64, 1) << left_sq;
                        const right_mask: u64 = @as(u64, 1) << right_sq;
                        break :blk (our_pawns_bb.bits & (left_mask | right_mask)) != 0;
                    };

                    if (defended_by_pawn) {
                        score = score.add(knight_outpost_defended);
                    } else {
                        score = score.add(knight_outpost_undefended);
                    }
                }
            }
        }
    }

    // --- Bishops: material + PST + mobility + bishop pair ---
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

            // Mobility
            const moves = movegen.sliderMoves(state, s, piece.bishop) & our_pieces_not;
            const move_count: i32 = @intCast(@popCount(moves));
            score = score.add(mobility_bonus[1][@as(usize, @intCast(move_count))]);
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

            // Mobility
            const moves = movegen.sliderMoves(state, s, piece.rook) & our_pieces_not;
            const move_count: i32 = @intCast(@popCount(moves));
            score = score.add(mobility_bonus[2][@as(usize, @intCast(move_count))]);

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

    // --- Queens: material + PST + mobility ---
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

            // Mobility (queen moves as bishop + rook)
            const bishop_moves = movegen.sliderMoves(state, s, piece.bishop) & our_pieces_not;
            const rook_moves = movegen.sliderMoves(state, s, piece.rook) & our_pieces_not;
            const move_count: i32 = @intCast(@popCount(bishop_moves | rook_moves));
            score = score.add(mobility_bonus[3][@as(usize, @intCast(move_count))]);
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

    return .{ .score = score, .phase = phase };
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

    // Single-pass evaluation for each color
    const our = evaluateColor(state, to_move, our_pieces, ~our_pieces, our_pawns, opp_pawns);
    const their = evaluateColor(state, opp, opp_pieces, ~opp_pieces, opp_pawns, our_pawns);

    const phase = @min(our.phase + their.phase, max_phase_mg);
    const total = our.score.sub(their.score);
    return total.taper(phase);
}

pub const EvalTrace = struct {
    material: [2]Score,
    pst: [2]Score,
    pawn_structure: [2]Score,
    passed_pawns: [2]Score,
    knight_outposts: [2]Score,
    bishop_pair: [2]Score,
    rook_bonuses: [2]Score,
    mobility: [2]Score,
    king_safety: [2]Score,
    phase: i32,
    total: i32,

    pub fn dump(self: *const EvalTrace, writer: *std.Io.Writer) !void {
        try writer.print("              | White MG  EG | Black MG  EG |\n", .{});
        try writer.print("--------------+--------------+--------------+\n", .{});
        try printRow(writer, "Material     ", self.material);
        try printRow(writer, "PST          ", self.pst);
        try printRow(writer, "Pawn struct  ", self.pawn_structure);
        try printRow(writer, "Passed pawns ", self.passed_pawns);
        try printRow(writer, "Kt outposts  ", self.knight_outposts);
        try printRow(writer, "Bishop pair  ", self.bishop_pair);
        try printRow(writer, "Rook bonuses ", self.rook_bonuses);
        try printRow(writer, "Mobility     ", self.mobility);
        try printRow(writer, "King safety  ", self.king_safety);
        try writer.print("--------------+--------------+--------------+\n", .{});
        try writer.print("Phase: {d}/24\n", .{self.phase});
        try writer.print("Total: {d}\n", .{self.total});
    }

    fn printRow(writer: *std.Io.Writer, label: []const u8, scores: [2]Score) !void {
        try writer.print("{s} | {d:>5}  {d:>5} | {d:>5}  {d:>5} |\n", .{
            label,
            scores[0].midgame,
            scores[0].endgame,
            scores[1].midgame,
            scores[1].endgame,
        });
    }
};

fn evaluateColorTrace(
    state: *const State,
    c: Color,
    our_pieces: u64,
    our_pieces_not: u64,
    our_pawns_bb: Bitboard,
    opp_pawns_bb: Bitboard,
) struct {
    material: Score,
    pst_score: Score,
    pawn_structure: Score,
    passed_pawns_score: Score,
    knight_outposts_score: Score,
    bishop_pair_score: Score,
    rook_bonuses_score: Score,
    mobility_score: Score,
    king_safety_score: Score,
    phase: i32,
} {
    var material_score = Score.zero;
    var pst_score = Score.zero;
    var pawn_structure_score = Score.zero;
    var passed_pawns_score = Score.zero;
    var knight_outposts_score = Score.zero;
    var bishop_pair_score = Score.zero;
    var rook_bonuses_score = Score.zero;
    var mobility_score = Score.zero;
    var king_safety_score = Score.zero;
    var phase: i32 = 0;
    const occupied = state.all_pieces.bits;

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

            const moves = movegen.knight_move_mask[s] & our_pieces_not;
            const move_count: i32 = @intCast(@popCount(moves));
            mobility_score = mobility_score.add(mobility_bonus[0][move_count]);

            const in_outpost_zone = if (c == Colors.white) rank >= 4 else rank <= 3;
            if (in_outpost_zone) {
                const can_be_attacked = blk: {
                    if (adjacent_files[file] == 0) break :blk false;
                    const adjacent_file_mask = adjacent_files[file];
                    const rank_u6: u6 = rank;
                    const attack_ranks: u64 = if (c == Colors.white)
                        if (rank < 7) @as(u64, 0xFFFFFFFFFFFFFFFF) << ((rank_u6 + 1) * 8) else 0
                    else if (rank > 0) (@as(u64, 1) << (rank_u6 * 8)) - 1 else 0;
                    break :blk (opp_pawns_bb.bits & adjacent_file_mask & attack_ranks) != 0;
                };

                if (!can_be_attacked) {
                    const defended_by_pawn = blk: {
                        if (file == 0 or file == 7) {
                            const def_file: u3 = if (file == 0) 1 else 6;
                            const def_rank: u3 = if (c == Colors.white) rank - 1 else rank + 1;
                            if ((c == Colors.white and rank == 0) or (c == Colors.black and rank == 7)) break :blk false;
                            const def_sq: u6 = @as(u6, def_file) + @as(u6, def_rank) * 8;
                            break :blk (our_pawns_bb.bits & (@as(u64, 1) << def_sq)) != 0;
                        }
                        const def_rank: u3 = if (c == Colors.white) rank - 1 else rank + 1;
                        if ((c == Colors.white and rank == 0) or (c == Colors.black and rank == 7)) break :blk false;
                        const left_sq: u6 = @as(u6, file - 1) + @as(u6, def_rank) * 8;
                        const right_sq: u6 = @as(u6, file + 1) + @as(u6, def_rank) * 8;
                        const left_mask: u64 = @as(u64, 1) << left_sq;
                        const right_mask: u64 = @as(u64, 1) << right_sq;
                        break :blk (our_pawns_bb.bits & (left_mask | right_mask)) != 0;
                    };

                    if (defended_by_pawn) {
                        knight_outposts_score = knight_outposts_score.add(knight_outpost_defended);
                    } else {
                        knight_outposts_score = knight_outposts_score.add(knight_outpost_undefended);
                    }
                }
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

            const moves = movegen.sliderMoves(state, s, piece.bishop) & our_pieces_not;
            const move_count: i32 = @intCast(@popCount(moves));
            mobility_score = mobility_score.add(mobility_bonus[1][move_count]);
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

            const moves = movegen.sliderMoves(state, s, piece.rook) & our_pieces_not;
            const move_count: i32 = @intCast(@popCount(moves));
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

    // --- Queens ---
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

            const bishop_moves = movegen.sliderMoves(state, s, piece.bishop) & our_pieces_not;
            const rook_moves = movegen.sliderMoves(state, s, piece.rook) & our_pieces_not;
            const move_count: i32 = @intCast(@popCount(bishop_moves | rook_moves));
            mobility_score = mobility_score.add(mobility_bonus[3][move_count]);
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
        .knight_outposts_score = knight_outposts_score,
        .bishop_pair_score = bishop_pair_score,
        .rook_bonuses_score = rook_bonuses_score,
        .mobility_score = mobility_score,
        .king_safety_score = king_safety_score,
        .phase = phase,
    };
}

pub fn evaluateTrace(state: *const State) EvalTrace {
    const to_move = state.to_move;
    const opp = ~to_move;

    const our_pieces = state.colorBitboard(to_move).bits;
    const opp_pieces = state.colorBitboard(opp).bits;
    const pawn_bb = state.pieceBitboard(piece.pawn);
    const our_pawns = pawn_bb.bitAnd(u64, our_pieces);
    const opp_pawns = pawn_bb.bitAnd(u64, opp_pieces);

    // White is index 0, black is index 1 regardless of side to move
    const white_idx: usize = if (to_move == Colors.white) 0 else 1;
    const black_idx: usize = 1 - white_idx;

    const our = evaluateColorTrace(state, to_move, our_pieces, ~our_pieces, our_pawns, opp_pawns);
    const their = evaluateColorTrace(state, opp, opp_pieces, ~opp_pieces, opp_pawns, our_pawns);

    var trace: EvalTrace = undefined;
    trace.material[white_idx] = our.material;
    trace.material[black_idx] = their.material;
    trace.pst[white_idx] = our.pst_score;
    trace.pst[black_idx] = their.pst_score;
    trace.pawn_structure[white_idx] = our.pawn_structure;
    trace.pawn_structure[black_idx] = their.pawn_structure;
    trace.passed_pawns[white_idx] = our.passed_pawns_score;
    trace.passed_pawns[black_idx] = their.passed_pawns_score;
    trace.knight_outposts[white_idx] = our.knight_outposts_score;
    trace.knight_outposts[black_idx] = their.knight_outposts_score;
    trace.bishop_pair[white_idx] = our.bishop_pair_score;
    trace.bishop_pair[black_idx] = their.bishop_pair_score;
    trace.rook_bonuses[white_idx] = our.rook_bonuses_score;
    trace.rook_bonuses[black_idx] = their.rook_bonuses_score;
    trace.mobility[white_idx] = our.mobility_score;
    trace.mobility[black_idx] = their.mobility_score;
    trace.king_safety[white_idx] = our.king_safety_score;
    trace.king_safety[black_idx] = their.king_safety_score;

    trace.phase = @min(our.phase + their.phase, max_phase_mg);
    const total = our.material.add(our.pst_score).add(our.pawn_structure).add(our.passed_pawns_score)
        .add(our.knight_outposts_score).add(our.bishop_pair_score).add(our.rook_bonuses_score)
        .add(our.mobility_score).add(our.king_safety_score)
        .sub(their.material).sub(their.pst_score).sub(their.pawn_structure).sub(their.passed_pawns_score)
        .sub(their.knight_outposts_score).sub(their.bishop_pair_score).sub(their.rook_bonuses_score)
        .sub(their.mobility_score).sub(their.king_safety_score);
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
    var score: i32 = 0;
    const p = ctx.state.mailbox[m.start].?;

    // MVV-LVA for captures
    if (ctx.state.mailbox[m.end]) |captured_piece| {
        const attacker_piece = ctx.state.mailbox[m.start].?;
        score += piece_values_mg[captured_piece] * 10 - piece_values_mg[attacker_piece];
    } else {
        // Check for en-passant
        const ep_square: u6 = @intCast(@as(u6, m.start) + State.pawn_ep_offset[ctx.state.to_move]);
        if (m.end == ep_square) {
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
            score += 900;
        }
    }
    if (ctx.killers[1]) |k| {
        if (k.start == m.start and k.end == m.end) {
            score += 800;
        }
    }

    // Countermove bonus (between killers and history)
    if (ctx.countermove) |cm| {
        if (cm.start == m.start and cm.end == m.end) {
            score += 850;
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
