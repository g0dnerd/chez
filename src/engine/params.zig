const std = @import("std");

const score_mod = @import("score.zig");
pub const Score = score_mod.Score(i16);
pub const ScoreF64 = score_mod.Score(f64);

// Layout (declaration order):
//   piece_values            [6]Score      ->  12
//   passed_pawn_bonus       [8]Score      ->  16
//   mobility_bonus          [4][28]Score  -> 224
//   bishop_pair .. tempo    17×Score      ->  34
//   king_proximity_passer   i16           ->   1
//   pst                     [6][64]Score  -> 768
//                                     total 1055
pub const PARAM_COUNT: usize = 1055;

pub const Params = struct {
    piece_values: [6]Score, // king (idx 5) is frozen at 20000/20000
    passed_pawn_bonus: [8]Score, // ranks 0 and 7 are frozen at 0/0
    mobility_bonus: [4][28]Score,
    bishop_pair: Score,
    rook_open_file: Score,
    rook_semi_open: Score,
    rook_on_seventh: Score,
    isolated_pawn: Score,
    doubled_pawn: Score,
    backward_pawn: Score,
    connected_pawn: Score,
    protected_passed_pawn: Score,
    blocked_passed_pawn: Score,
    rook_behind_passer: Score,
    free_passed_pawn: Score,
    knight_outpost_defended: Score,
    bishop_outpost_defended: Score,
    pawn_shield: Score,
    pawn_shield_missing: Score,
    tempo: Score,
    king_proximity_passer: i16,
    pst: [6][64]Score, // pawn rank-0/7 rows frozen at 0/0

    // Serialise to a flat f64 slice. Used to initialise the SPSA float vector
    // and to read back tuned values at checkpoint/output time.
    pub fn toFloats(self: *const Params, out: []f64) void {
        std.debug.assert(out.len >= PARAM_COUNT);
        var i: usize = 0;

        for (self.piece_values) |s| {
            out[i] = @floatFromInt(s.midgame());
            i += 1;
            out[i] = @floatFromInt(s.endgame());
            i += 1;
        }
        for (self.passed_pawn_bonus) |s| {
            out[i] = @floatFromInt(s.midgame());
            i += 1;
            out[i] = @floatFromInt(s.endgame());
            i += 1;
        }
        for (0..4) |pi| {
            for (0..28) |mi| {
                out[i] = @floatFromInt(self.mobility_bonus[pi][mi].midgame());
                i += 1;
                out[i] = @floatFromInt(self.mobility_bonus[pi][mi].endgame());
                i += 1;
            }
        }
        // 17 named Score scalars in declaration order
        for ([_]Score{
            self.bishop_pair,
            self.rook_open_file,
            self.rook_semi_open,
            self.rook_on_seventh,
            self.isolated_pawn,
            self.doubled_pawn,
            self.backward_pawn,
            self.connected_pawn,
            self.protected_passed_pawn,
            self.blocked_passed_pawn,
            self.rook_behind_passer,
            self.free_passed_pawn,
            self.knight_outpost_defended,
            self.bishop_outpost_defended,
            self.pawn_shield,
            self.pawn_shield_missing,
            self.tempo,
        }) |s| {
            out[i] = @floatFromInt(s.midgame());
            i += 1;
            out[i] = @floatFromInt(s.endgame());
            i += 1;
        }
        out[i] = @floatFromInt(self.king_proximity_passer);
        i += 1;
        for (0..6) |pi| {
            for (0..64) |sq| {
                out[i] = @floatFromInt(self.pst[pi][sq].midgame());
                i += 1;
                out[i] = @floatFromInt(self.pst[pi][sq].endgame());
                i += 1;
            }
        }
        std.debug.assert(i == PARAM_COUNT);
    }

    // Deserialise from f64 slice, rounding each value to nearest i16 (clamped).
    // Only called at checkpoint writes and final codegen: never during SPSA.
    pub fn fromFloats(floats: []const f64) Params {
        std.debug.assert(floats.len >= PARAM_COUNT);
        var p: Params = undefined;
        var i: usize = 0;

        for (0..6) |pi| {
            p.piece_values[pi] = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
            i += 2;
        }
        for (0..8) |ri| {
            p.passed_pawn_bonus[ri] = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
            i += 2;
        }
        for (0..4) |pi| {
            for (0..28) |mi| {
                p.mobility_bonus[pi][mi] = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
                i += 2;
            }
        }
        p.bishop_pair = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.rook_open_file = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.rook_semi_open = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.rook_on_seventh = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.isolated_pawn = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.doubled_pawn = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.backward_pawn = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.connected_pawn = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.protected_passed_pawn = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.blocked_passed_pawn = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.rook_behind_passer = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.free_passed_pawn = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.knight_outpost_defended = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.bishop_outpost_defended = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.pawn_shield = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.pawn_shield_missing = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.tempo = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
        i += 2;
        p.king_proximity_passer = clampI16(floats[i]);
        i += 1;
        for (0..6) |pi| {
            for (0..64) |sq| {
                p.pst[pi][sq] = Score.init(clampI16(floats[i]), clampI16(floats[i + 1]));
                i += 2;
            }
        }
        std.debug.assert(i == PARAM_COUNT);
        return p;
    }
};

// Floating-point mirror of Params, used exclusively in the SPSA
// hot loop so that gradient estimation never sees i16 quantisation noise.
pub const ParamsF64 = struct {
    piece_values: [6]ScoreF64,
    passed_pawn_bonus: [8]ScoreF64,
    mobility_bonus: [4][28]ScoreF64,
    bishop_pair: ScoreF64,
    rook_open_file: ScoreF64,
    rook_semi_open: ScoreF64,
    rook_on_seventh: ScoreF64,
    isolated_pawn: ScoreF64,
    doubled_pawn: ScoreF64,
    backward_pawn: ScoreF64,
    connected_pawn: ScoreF64,
    protected_passed_pawn: ScoreF64,
    blocked_passed_pawn: ScoreF64,
    rook_behind_passer: ScoreF64,
    free_passed_pawn: ScoreF64,
    knight_outpost_defended: ScoreF64,
    bishop_outpost_defended: ScoreF64,
    pawn_shield: ScoreF64,
    pawn_shield_missing: ScoreF64,
    tempo: ScoreF64,
    king_proximity_passer: f64,
    pst: [6][64]ScoreF64,

    // Deserialise from f64 slice without rounding. Used in the SPSA hot loop.
    pub fn fromFloats(floats: []const f64) ParamsF64 {
        std.debug.assert(floats.len >= PARAM_COUNT);
        var p: ParamsF64 = undefined;
        var i: usize = 0;

        for (0..6) |pi| {
            p.piece_values[pi] = ScoreF64.init(floats[i], floats[i + 1]);
            i += 2;
        }
        for (0..8) |ri| {
            p.passed_pawn_bonus[ri] = ScoreF64.init(floats[i], floats[i + 1]);
            i += 2;
        }
        for (0..4) |pi| {
            for (0..28) |mi| {
                p.mobility_bonus[pi][mi] = ScoreF64.init(floats[i], floats[i + 1]);
                i += 2;
            }
        }
        p.bishop_pair = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.rook_open_file = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.rook_semi_open = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.rook_on_seventh = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.isolated_pawn = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.doubled_pawn = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.backward_pawn = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.connected_pawn = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.protected_passed_pawn = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.blocked_passed_pawn = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.rook_behind_passer = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.free_passed_pawn = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.knight_outpost_defended = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.bishop_outpost_defended = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.pawn_shield = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.pawn_shield_missing = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.tempo = ScoreF64.init(floats[i], floats[i + 1]);
        i += 2;
        p.king_proximity_passer = floats[i];
        i += 1;
        for (0..6) |pi| {
            for (0..64) |sq| {
                p.pst[pi][sq] = ScoreF64.init(floats[i], floats[i + 1]);
                i += 2;
            }
        }
        std.debug.assert(i == PARAM_COUNT);
        return p;
    }

    // Round each f64 field to nearest i16. Used to produce the final Params
    // for checkpoint writes and codegen output.
    pub fn toParams(self: *const ParamsF64) Params {
        var p: Params = undefined;

        for (0..6) |pi| {
            p.piece_values[pi] = Score.init(
                clampI16(self.piece_values[pi].midgame()),
                clampI16(self.piece_values[pi].endgame()),
            );
        }
        for (0..8) |ri| {
            p.passed_pawn_bonus[ri] = Score.init(
                clampI16(self.passed_pawn_bonus[ri].midgame()),
                clampI16(self.passed_pawn_bonus[ri].endgame()),
            );
        }
        for (0..4) |pi| {
            for (0..28) |mi| {
                p.mobility_bonus[pi][mi] = Score.init(
                    clampI16(self.mobility_bonus[pi][mi].midgame()),
                    clampI16(self.mobility_bonus[pi][mi].endgame()),
                );
            }
        }
        p.bishop_pair = Score.init(clampI16(self.bishop_pair.midgame()), clampI16(self.bishop_pair.endgame()));
        p.rook_open_file = Score.init(clampI16(self.rook_open_file.midgame()), clampI16(self.rook_open_file.endgame()));
        p.rook_semi_open = Score.init(clampI16(self.rook_semi_open.midgame()), clampI16(self.rook_semi_open.endgame()));
        p.rook_on_seventh = Score.init(clampI16(self.rook_on_seventh.midgame()), clampI16(self.rook_on_seventh.endgame()));
        p.isolated_pawn = Score.init(clampI16(self.isolated_pawn.midgame()), clampI16(self.isolated_pawn.endgame()));
        p.doubled_pawn = Score.init(clampI16(self.doubled_pawn.midgame()), clampI16(self.doubled_pawn.endgame()));
        p.backward_pawn = Score.init(clampI16(self.backward_pawn.midgame()), clampI16(self.backward_pawn.endgame()));
        p.connected_pawn = Score.init(clampI16(self.connected_pawn.midgame()), clampI16(self.connected_pawn.endgame()));
        p.protected_passed_pawn = Score.init(
            clampI16(self.protected_passed_pawn.midgame()),
            clampI16(self.protected_passed_pawn.endgame()),
        );
        p.blocked_passed_pawn = Score.init(clampI16(self.blocked_passed_pawn.midgame()), clampI16(self.blocked_passed_pawn.endgame()));
        p.rook_behind_passer = Score.init(clampI16(self.rook_behind_passer.midgame()), clampI16(self.rook_behind_passer.endgame()));
        p.free_passed_pawn = Score.init(clampI16(self.free_passed_pawn.midgame()), clampI16(self.free_passed_pawn.endgame()));
        p.knight_outpost_defended = Score.init(
            clampI16(self.knight_outpost_defended.midgame()),
            clampI16(self.knight_outpost_defended.endgame()),
        );
        p.bishop_outpost_defended = Score.init(
            clampI16(self.bishop_outpost_defended.midgame()),
            clampI16(self.bishop_outpost_defended.endgame()),
        );
        p.pawn_shield = Score.init(clampI16(self.pawn_shield.midgame()), clampI16(self.pawn_shield.endgame()));
        p.pawn_shield_missing = Score.init(
            clampI16(self.pawn_shield_missing.midgame()),
            clampI16(self.pawn_shield_missing.endgame()),
        );
        p.tempo = Score.init(clampI16(self.tempo.midgame()), clampI16(self.tempo.endgame()));
        p.king_proximity_passer = clampI16(self.king_proximity_passer);
        for (0..6) |pi| {
            for (0..64) |sq| {
                p.pst[pi][sq] = Score.init(
                    clampI16(self.pst[pi][sq].midgame()),
                    clampI16(self.pst[pi][sq].endgame()),
                );
            }
        }
        return p;
    }
};

pub const default_params: Params = .{
    .piece_values = .{
        Score.init(166, 291), // pawn
        Score.init(793, 906), // knight
        Score.init(787, 855), // bishop
        Score.init(1117, 1378), // rook
        Score.init(2538, 2685), // queen
        Score.init(20000, 20000), // king (frozen)
    },
    .passed_pawn_bonus = .{
        Score.init(0, 0), // rank 0 (frozen)
        Score.init(0, -25), // rank 1
        Score.init(-40, -29), // rank 2
        Score.init(-13, 38), // rank 3
        Score.init(59, 92), // rank 4
        Score.init(174, 197), // rank 5
        Score.init(303, 332), // rank 6
        Score.init(0, 0), // rank 7 (frozen)
    },
    .mobility_bonus = .{
        // Knights (max 8)
        .{
            Score.init(-86, -71),
            Score.init(-83, -49),
            Score.init(-26, -41),
            Score.init(-6, -15),
            Score.init(8, 16),
            Score.init(28, 36),
            Score.init(46, 56),
            Score.init(59, 53),
            Score.init(61, 31),
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
        // Bishops (max 13)
        .{
            Score.init(-100, -76),
            Score.init(-46, -48),
            Score.init(-6, -47),
            Score.init(13, -21),
            Score.init(41, 15),
            Score.init(56, 48),
            Score.init(73, 57),
            Score.init(77, 78),
            Score.init(69, 106),
            Score.init(99, 83),
            Score.init(102, 80),
            Score.init(69, 108),
            Score.init(60, 128),
            Score.init(95, 106),
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
        // Rooks (max 14)
        .{
            Score.init(-52, -93),
            Score.init(-52, -18),
            Score.init(-19, 16),
            Score.init(-19, 15),
            Score.init(-27, 51),
            Score.init(-15, 91),
            Score.init(-14, 113),
            Score.init(0, 129),
            Score.init(23, 144),
            Score.init(45, 162),
            Score.init(49, 185),
            Score.init(74, 175),
            Score.init(62, 196),
            Score.init(49, 198),
            Score.init(65, 187),
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
        // Queens (max 27)
        .{
            Score.init(-1, -59),
            Score.init(-32, -21),
            Score.init(1, -30),
            Score.init(-28, 7),
            Score.init(0, 0),
            Score.init(16, 52),
            Score.init(24, 61),
            Score.init(33, 37),
            Score.init(40, 68),
            Score.init(51, 81),
            Score.init(59, 104),
            Score.init(71, 99),
            Score.init(72, 137),
            Score.init(72, 114),
            Score.init(70, 146),
            Score.init(63, 148),
            Score.init(64, 170),
            Score.init(58, 161),
            Score.init(89, 145),
            Score.init(67, 169),
            Score.init(87, 181),
            Score.init(119, 136),
            Score.init(102, 187),
            Score.init(89, 188),
            Score.init(96, 174),
            Score.init(119, 180),
            Score.init(111, 191),
            Score.init(109, 209),
        },
    },
    .bishop_pair = Score.init(82, 179),
    .rook_open_file = Score.init(45, 25),
    .rook_semi_open = Score.init(19, 29),
    .rook_on_seventh = Score.init(6, 94),
    .isolated_pawn = Score.init(-35, -36),
    .doubled_pawn = Score.init(-16, -42),
    .backward_pawn = Score.init(-36, -35),
    .connected_pawn = Score.init(13, 17),
    .protected_passed_pawn = Score.init(36, 6),
    .blocked_passed_pawn = Score.init(-46, -71),
    .rook_behind_passer = Score.init(-8, 38),
    .free_passed_pawn = Score.init(-24, 6),
    .knight_outpost_defended = Score.init(48, 72),
    .bishop_outpost_defended = Score.init(76, 1),
    .pawn_shield = Score.init(42, -44),
    .pawn_shield_missing = Score.init(4, -30),
    .tempo = Score.init(56, 23),
    .king_proximity_passer = 20,
    .pst = .{
        // Pawns (rank 0 and 7 frozen at 0)
        .{
            Score.init(0,0), Score.init(0,0), Score.init(0,0), Score.init(0,0), Score.init(0,0), Score.init(0,0), Score.init(0,0), Score.init(0,0),
            Score.init(-28,-28), Score.init(-70,6), Score.init(-32,-8), Score.init(15,-5), Score.init(-9,34), Score.init(20,59), Score.init(-12,-1), Score.init(-41,-40),
            Score.init(-42,-22), Score.init(-46,-4), Score.init(-50,7), Score.init(-13,6), Score.init(22,18), Score.init(5,13), Score.init(3,-17), Score.init(-22,-29),
            Score.init(-37,12), Score.init(-65,17), Score.init(-17,-23), Score.init(-24,-25), Score.init(-13,-13), Score.init(17,13), Score.init(-19,-17), Score.init(-16,-16),
            Score.init(-18,41), Score.init(-38,16), Score.init(-39,26), Score.init(-26,-18), Score.init(35,-29), Score.init(31,17), Score.init(-11,27), Score.init(-12,71),
            Score.init(6,7), Score.init(-16,40), Score.init(-44,23), Score.init(22,-24), Score.init(34,0), Score.init(37,101), Score.init(-54,25), Score.init(6,30),
            Score.init(5,40), Score.init(60,-18), Score.init(32,-45), Score.init(-66,39), Score.init(2,35), Score.init(20,-6), Score.init(-41,6), Score.init(46,-20),
            Score.init(0,0), Score.init(0,0), Score.init(0,0), Score.init(0,0), Score.init(0,0), Score.init(0,0), Score.init(0,0), Score.init(0,0),
        },
        // Knights
        .{
            Score.init(-122,-99), Score.init(-40,-94), Score.init(-3,-60), Score.init(-61,-20), Score.init(-62,0), Score.init(-89,-137), Score.init(-81,-69), Score.init(-144,-184),
            Score.init(-117,-36), Score.init(-89,-65), Score.init(-26,23), Score.init(-29,-45), Score.init(-19,-2), Score.init(-29,-63), Score.init(-30,-52), Score.init(-35,-51),
            Score.init(-43,-69), Score.init(-37,-37), Score.init(-49,12), Score.init(6,45), Score.init(15,-6), Score.init(-26,-72), Score.init(2,-75), Score.init(-48,-19),
            Score.init(-23,-45), Score.init(-4,0), Score.init(17,28), Score.init(14,3), Score.init(54,11), Score.init(96,12), Score.init(48,-36), Score.init(-16,18),
            Score.init(20,9), Score.init(-19,38), Score.init(67,1), Score.init(67,18), Score.init(41,39), Score.init(54,6), Score.init(52,10), Score.init(-54,-1),
            Score.init(-14,-97), Score.init(12,-20), Score.init(125,-25), Score.init(76,6), Score.init(120,-16), Score.init(29,22), Score.init(44,-11), Score.init(52,-64),
            Score.init(-110,-39), Score.init(30,-59), Score.init(-29,-67), Score.init(41,16), Score.init(36,109), Score.init(-24,-112), Score.init(38,-139), Score.init(-118,-21),
            Score.init(-248,-70), Score.init(-75,-90), Score.init(-73,-76), Score.init(19,-30), Score.init(-21,-50), Score.init(-130,9), Score.init(-65,-105), Score.init(-180,-83),
        },
        // Bishops
        .{
            Score.init(-89,-76), Score.init(22,-33), Score.init(12,-15), Score.init(-100,18), Score.init(18,-28), Score.init(14,-3), Score.init(27,-62), Score.init(-73,-88),
            Score.init(31,-100), Score.init(-11,-9), Score.init(53,10), Score.init(-33,-14), Score.init(1,10), Score.init(3,16), Score.init(41,-2), Score.init(-30,-62),
            Score.init(-6,15), Score.init(76,19), Score.init(26,-14), Score.init(12,16), Score.init(16,19), Score.init(-23,4), Score.init(33,-24), Score.init(39,-26),
            Score.init(-37,21), Score.init(40,-8), Score.init(11,24), Score.init(69,1), Score.init(64,5), Score.init(5,-7), Score.init(5,6), Score.init(2,38),
            Score.init(-65,3), Score.init(40,-10), Score.init(99,-53), Score.init(-20,70), Score.init(19,12), Score.init(28,18), Score.init(34,13), Score.init(62,27),
            Score.init(-16,-117), Score.init(36,11), Score.init(-29,56), Score.init(39,-21), Score.init(-10,0), Score.init(-24,6), Score.init(-1,79), Score.init(1,-33),
            Score.init(56,-29), Score.init(32,56), Score.init(37,17), Score.init(11,4), Score.init(-35,-68), Score.init(27,17), Score.init(-20,-12), Score.init(-33,-14),
            Score.init(-33,-86), Score.init(34,-36), Score.init(-50,-63), Score.init(5,23), Score.init(-105,26), Score.init(52,7), Score.init(39,-3), Score.init(-47,-6),
        },
        // Rooks
        .{
            Score.init(-59,-45), Score.init(-63,-43), Score.init(-50,-58), Score.init(-63,-54), Score.init(-44,-47), Score.init(-45,-20), Score.init(-8,-27), Score.init(-48,26),
            Score.init(-42,-31), Score.init(-53,32), Score.init(-21,-12), Score.init(-17,-50), Score.init(-36,-41), Score.init(-11,50), Score.init(-24,-37), Score.init(-27,-35),
            Score.init(-24,22), Score.init(-7,-48), Score.init(-22,18), Score.init(-25,12), Score.init(1,-40), Score.init(14,-26), Score.init(8,-35), Score.init(7,12),
            Score.init(-12,-17), Score.init(-35,3), Score.init(-46,30), Score.init(35,-45), Score.init(16,16), Score.init(32,-54), Score.init(-42,-35), Score.init(-27,-14),
            Score.init(62,5), Score.init(1,35), Score.init(2,66), Score.init(-23,42), Score.init(36,25), Score.init(-24,-2), Score.init(-18,51), Score.init(-16,68),
            Score.init(53,43), Score.init(7,43), Score.init(60,-25), Score.init(-48,19), Score.init(-1,37), Score.init(5,56), Score.init(54,-3), Score.init(60,36),
            Score.init(12,18), Score.init(84,20), Score.init(-9,18), Score.init(19,-37), Score.init(15,-14), Score.init(52,-8), Score.init(-8,-41), Score.init(9,-44),
            Score.init(-12,25), Score.init(-15,50), Score.init(-2,34), Score.init(6,101), Score.init(17,2), Score.init(1,25), Score.init(-44,-60), Score.init(-30,72),
        },
        // Queens
        .{
            Score.init(12,-22), Score.init(-50,-25), Score.init(-18,-65), Score.init(-55,-42), Score.init(21,-92), Score.init(-82,-108), Score.init(6,-32), Score.init(52,-87),
            Score.init(17,-66), Score.init(-83,-25), Score.init(-35,-91), Score.init(-38,3), Score.init(-45,-2), Score.init(4,-41), Score.init(68,-57), Score.init(-34,-14),
            Score.init(-12,-29), Score.init(-40,-20), Score.init(-7,-46), Score.init(-29,-104), Score.init(-12,-44), Score.init(-10,70), Score.init(3,-78), Score.init(-22,-119),
            Score.init(-19,-89), Score.init(52,-39), Score.init(-7,11), Score.init(19,-5), Score.init(-46,32), Score.init(26,2), Score.init(35,-24), Score.init(-3,-13),
            Score.init(2,-17), Score.init(-3,7), Score.init(14,-66), Score.init(-14,27), Score.init(36,15), Score.init(44,-46), Score.init(62,-17), Score.init(23,-63),
            Score.init(-25,-37), Score.init(42,-23), Score.init(35,45), Score.init(30,-63), Score.init(8,-40), Score.init(55,6), Score.init(32,83), Score.init(39,-31),
            Score.init(-17,-43), Score.init(-24,42), Score.init(-13,-32), Score.init(-41,-1), Score.init(72,-22), Score.init(-10,-2), Score.init(29,6), Score.init(65,-31),
            Score.init(-28,-37), Score.init(-24,-107), Score.init(26,-13), Score.init(-21,34), Score.init(2,-31), Score.init(-17,-89), Score.init(85,-93), Score.init(-45,-15),
        },
        // Kings
        .{
            Score.init(180,46), Score.init(344,53), Score.init(325,152), Score.init(151,102), Score.init(295,34), Score.init(192,79), Score.init(299,64), Score.init(240,17),
            Score.init(272,106), Score.init(307,97), Score.init(181,186), Score.init(210,128), Score.init(167,164), Score.init(227,113), Score.init(284,124), Score.init(260,44),
            Score.init(181,73), Score.init(234,135), Score.init(153,131), Score.init(100,142), Score.init(153,133), Score.init(175,114), Score.init(219,85), Score.init(150,74),
            Score.init(156,123), Score.init(186,192), Score.init(210,193), Score.init(108,145), Score.init(43,191), Score.init(105,181), Score.init(182,85), Score.init(162,86),
            Score.init(147,160), Score.init(200,216), Score.init(100,213), Score.init(170,218), Score.init(123,175), Score.init(136,188), Score.init(181,201), Score.init(165,101),
            Score.init(90,115), Score.init(152,187), Score.init(117,201), Score.init(49,257), Score.init(107,222), Score.init(52,253), Score.init(118,176), Score.init(138,61),
            Score.init(122,59), Score.init(147,158), Score.init(39,210), Score.init(91,57), Score.init(16,64), Score.init(97,120), Score.init(110,139), Score.init(85,89),
            Score.init(92,9), Score.init(129,35), Score.init(11,67), Score.init(-70,83), Score.init(1,38), Score.init(31,29), Score.init(85,103), Score.init(106,-2),
        },
    },
};



// ==============================================================================
// Internal helpers
// ==============================================================================

// Round a f64 to the nearest i16, clamping to [minInt(i16), maxInt(i16)].
fn clampI16(v: f64) i16 {
    const rounded = @round(v);
    const min: f64 = @floatFromInt(std.math.minInt(i16));
    const max: f64 = @floatFromInt(std.math.maxInt(i16));
    return @intFromFloat(std.math.clamp(rounded, min, max));
}

// ==============================================================================
// Tests
// ==============================================================================

test "params round-trip: toFloats then fromFloats recovers default_params exactly" {
    var buf: [PARAM_COUNT]f64 = undefined;
    default_params.toFloats(&buf);
    const recovered = Params.fromFloats(&buf);

    // Re-serialise the recovered params and compare buffers element by element.
    // Since default_params contains only integer-valued i16 fields, the round-
    // trip through f64 and back must be bit-exact.
    var buf2: [PARAM_COUNT]f64 = undefined;
    recovered.toFloats(&buf2);

    for (0..PARAM_COUNT) |i| {
        try std.testing.expectEqual(buf[i], buf2[i]);
    }
}

test "paramsf64 fromFloats/toParams round-trips default_params exactly" {
    var buf: [PARAM_COUNT]f64 = undefined;
    default_params.toFloats(&buf);

    const p_f64 = ParamsF64.fromFloats(&buf);
    const recovered = p_f64.toParams();

    var buf2: [PARAM_COUNT]f64 = undefined;
    recovered.toFloats(&buf2);

    for (0..PARAM_COUNT) |i| {
        try std.testing.expectEqual(buf[i], buf2[i]);
    }
}
