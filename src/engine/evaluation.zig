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
const Square = square.Square;
const score_mod = @import("score.zig");
const params_mod = @import("params.zig");
pub const Score = score_mod.Score(i16);
pub const Params = params_mod.Params;
const max_phase_mg = score_mod.max_phase_mg;

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

pub fn toCentipawns(val: i32) f32 {
    const val_f: f32 = @floatFromInt(val);
    return val_f / @as(f32, @floatFromInt(params_mod.default_params.piece_values[0].endgame()));
}

// ==============================================================================
// Backward-compatible re-exports derived from default_params at comptime.
// search.zig uses piece_values_mg for MVV-LVA; tui.zig and other callers
// reference pst/piece_values directly.
// ==============================================================================

pub const pst = params_mod.default_params.pst;
pub const piece_values = params_mod.default_params.piece_values;

// For MVV-LVA move ordering (uses middlegame values)
pub const piece_values_mg = blk: {
    const vals = params_mod.default_params.piece_values;
    break :blk [6]i32{
        vals[0].midgame(), vals[1].midgame(), vals[2].midgame(),
        vals[3].midgame(), vals[4].midgame(), vals[5].midgame(),
    };
};

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

fn computePassedPawnMask(c: Color, file: Square, rank: Square) u64 {
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
fn isOutpost(c: Color, file: Square, rank: Square, opp_pawns_bb: Bitboard) bool {
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
fn isDefendedByPawn(c: Color, file: Square, rank: Square, our_pawns_bb: Bitboard) bool {
    if ((c == Colors.white and rank == 0) or (c == Colors.black and rank == 7)) return false;
    const def_rank: Square = if (c == Colors.white) rank - 1 else rank + 1;
    var mask: u64 = 0;
    if (file > 0) mask |= @as(u64, 1) << (@as(Square, file - 1) + def_rank * 8);
    if (file < 7) mask |= @as(u64, 1) << (@as(Square, file + 1) + def_rank * 8);
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
    const king_sq: Square = @intCast(@ctz(king_bb));
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
        const sniper_sq: Square = @intCast(@ctz(sniper_bb));
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

// ==============================================================================
// Generic evaluation core
// ==============================================================================

// Single-pass evaluation for one color. Iterates each piece type once,
// accumulating material, PST, mobility, and structural scores together.
// Returns the total score, phase accumulator, and accumulated piece attacks.
// `p` is anytype — either *const Params (i16) or *const ParamsF64 (f64).
fn evaluateColorGeneric(
    state: *const State,
    c: Color,
    our_pieces: u64,
    our_pawns_bb: Bitboard,
    opp_pawns_bb: Bitboard,
    mobility_area: u64,
    p: anytype,
) struct { score: @TypeOf(p.piece_values[0]), phase: i32, attacks: PieceAttacks } {
    const ScoreT = @TypeOf(p.piece_values[0]);
    var score = ScoreT.zero;
    var phase: i32 = 0;
    var attacks = PieceAttacks{};
    const occupied = state.all_pieces.bits;

    // Precompute for passed pawn evaluation
    const opp_pieces_bb = occupied ^ our_pieces;
    const our_rooks = state.pieceBitboard(piece.rook).bits & our_pieces;
    const opp_king_sq: Square = @intCast(@ctz(state.pieceBitboard(piece.king).bits & opp_pieces_bb));
    const opp_pawn_atk = pawnAttacksBB(opp_pawns_bb.bits, ~c);

    // X-ray occupancy: bishops see through own queens, rooks see through own rooks+queens
    const our_queens_bb = state.pieceBitboard(piece.queen).bits & our_pieces;
    const occ_without_our_queens = Bitboard{ .bits = occupied ^ our_queens_bb };
    const our_rq = (state.pieceBitboard(piece.rook).bits | our_queens_bb) & our_pieces;
    const occ_without_our_rq = Bitboard{ .bits = occupied ^ our_rq };

    // --- Pawns: material + PST + pawn structure ---
    {
        var pawns = our_pawns_bb;
        const pawn_count: i32 = @intCast(pawns.popCount());
        score = score.add(p.piece_values[piece.pawn].mul(pawn_count));

        // Doubled pawn penalty: apply once per extra pawn on each file
        for (0..8) |file| {
            const count: i32 = @intCast(@popCount(our_pawns_bb.bits & file_masks[file]));
            if (count > 1) {
                score = score.add(p.doubled_pawn.mul(count - 1));
            }
        }

        while (pawns.next()) |s| {
            const file: Square = s % 8;
            const rank: Square = s / 8;

            // PST
            const sq: Square = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            score = score.add(p.pst[piece.pawn][sq]);

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
                score = score.add(p.connected_pawn);
            }

            // Passed pawn
            const ahead_mask = computePassedPawnMask(c, file, rank);
            if ((opp_pawns_bb.bits & ahead_mask) == 0) {
                const passed_rank = if (c == Colors.white) rank else 7 - rank;
                score = score.add(p.passed_pawn_bonus[passed_rank]);

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
                    score = score.add(p.protected_passed_pawn);
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
                    score = score.add(p.blocked_passed_pawn);
                }

                // Rook behind passed pawn
                {
                    const ranks_behind: u64 = if (c == Colors.white)
                        (@as(u64, 1) << (rank * 8)) - 1
                    else
                        @as(u64, 0xFFFFFFFFFFFFFFFF) << ((rank + 1) * 8);
                    if ((our_rooks & file_masks[file] & ranks_behind) != 0) {
                        score = score.add(p.rook_behind_passer);
                    }
                }

                // Enemy king distance to passed pawn (endgame bonus).
                // king_proximity_passer is i16 in Params and f64 in ParamsF64;
                // dist is always i16, so we cast to match the score component type.
                {
                    const king_file: Square = opp_king_sq % 8;
                    const king_rank: Square = opp_king_sq / 8;
                    const df: Square = if (file > king_file) file - king_file else king_file - file;
                    const dr: Square = if (rank > king_rank) rank - king_rank else king_rank - rank;
                    const dist: i16 = @intCast(@max(df, dr));
                    const kp_eg: @TypeOf(p.king_proximity_passer) = switch (@TypeOf(p.king_proximity_passer)) {
                        i16 => dist * p.king_proximity_passer,
                        f64 => @as(f64, @floatFromInt(dist)) * p.king_proximity_passer,
                        else => @compileError("unexpected king_proximity_passer type"),
                    };
                    score = score.add(ScoreT.init(0, kp_eg));
                }

                // Free passed pawn (advance square not occupied or attacked by enemy pawns)
                if (!blocked) {
                    const ahead_sq: u64 = if (c == Colors.white and rank < 7)
                        @as(u64, 1) << (s + 8)
                    else if (c == Colors.black and rank > 0)
                        @as(u64, 1) << (s - 8)
                    else
                        0;
                    if (ahead_sq != 0 and (opp_pawn_atk & ahead_sq) == 0) {
                        score = score.add(p.free_passed_pawn);
                    }
                }
            }

            // Isolated pawn
            if ((our_pawns_bb.bits & adjacent_files[file]) == 0) {
                score = score.add(p.isolated_pawn);
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
                    const stop_sq: Square = if (c == Colors.white) s + 8 else s - 8;
                    const stop_file: Square = stop_sq % 8;
                    const stop_rank: Square = stop_sq / 8;
                    const stop_attacked = blk: {
                        // Enemy pawn attacks the stop square from adjacent files, one rank beyond
                        const atk_rank: Square = if (c == Colors.white) stop_rank + 1 else if (stop_rank > 0) stop_rank - 1 else break :blk false;
                        if (c == Colors.black and stop_rank == 0) break :blk false;
                        if (c == Colors.white and atk_rank > 7) break :blk false;
                        var atk_mask: u64 = 0;
                        if (stop_file > 0) atk_mask |= @as(u64, 1) << (@as(Square, stop_file - 1) + atk_rank * 8);
                        if (stop_file < 7) atk_mask |= @as(u64, 1) << (@as(Square, stop_file + 1) + atk_rank * 8);
                        break :blk (opp_pawns_bb.bits & atk_mask) != 0;
                    };
                    if (stop_attacked) {
                        score = score.add(p.backward_pawn);
                    }
                }
            }
        }
    }

    // --- Knights: material + PST + mobility + outposts ---
    {
        var knights = state.pieceBitboard(piece.knight).bitAnd(u64, our_pieces);
        const knight_count: i32 = @intCast(knights.popCount());
        score = score.add(p.piece_values[piece.knight].mul(knight_count));
        phase += knight_count * phase_weights[piece.knight];

        while (knights.next()) |s| {
            const file: u3 = @intCast(s % 8);
            const rank: u3 = @intCast(s / 8);

            // PST
            const sq: Square = if (c == Colors.black) @intCast((@as(Square, 7) - rank) * 8 + file) else s;
            score = score.add(p.pst[piece.knight][sq]);

            // Mobility (using mobility area instead of ~our_pieces)
            const knight_atk = movegen.knight_move_mask[s];
            attacks.knight |= knight_atk;
            const mob = knight_atk & mobility_area;
            const move_count = @popCount(mob);
            score = score.add(p.mobility_bonus[0][move_count]);

            // Outpost check (only defended outposts rewarded)
            if (isOutpost(c, file, rank, opp_pawns_bb) and
                isDefendedByPawn(c, file, rank, our_pawns_bb))
            {
                score = score.add(p.knight_outpost_defended);
            }
        }
    }

    // --- Bishops: material + PST + mobility + bishop pair + outposts ---
    {
        var bishops = state.pieceBitboard(piece.bishop).bitAnd(u64, our_pieces);
        const bishop_count: i32 = @intCast(bishops.popCount());
        score = score.add(p.piece_values[piece.bishop].mul(bishop_count));
        phase += bishop_count * phase_weights[piece.bishop];

        if (bishop_count >= 2) {
            score = score.add(p.bishop_pair);
        }

        while (bishops.next()) |s| {
            const rank: Square = s / 8;
            const file: Square = s % 8;

            // PST
            const sq: Square = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            score = score.add(p.pst[piece.bishop][sq]);

            // Mobility (x-ray through own queens, using mobility area)
            const bishop_atk = movegen.sliderMovesWithOccupancy(s, piece.bishop, occ_without_our_queens);
            attacks.bishop |= bishop_atk;
            const mob = bishop_atk & mobility_area;
            const move_count = @popCount(mob);
            score = score.add(p.mobility_bonus[1][move_count]);

            // Bishop outpost (defended only)
            if (isOutpost(c, file, rank, opp_pawns_bb) and
                isDefendedByPawn(c, file, rank, our_pawns_bb))
            {
                score = score.add(p.bishop_outpost_defended);
            }
        }
    }

    // --- Rooks: material + PST + mobility + open file + 7th rank ---
    {
        var rooks = Bitboard{ .bits = state.pieceBitboard(piece.rook).bits & our_pieces };
        const rook_count: i32 = @intCast(rooks.popCount());
        score = score.add(p.piece_values[piece.rook].mul(rook_count));
        phase += rook_count * phase_weights[piece.rook];

        const seventh_rank: Square = if (c == Colors.white) 6 else 1;

        while (rooks.next()) |s| {
            const file: Square = s % 8;
            const rank: Square = s / 8;

            // PST
            const sq: Square = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            score = score.add(p.pst[piece.rook][sq]);

            // Mobility (x-ray through own rooks+queens, using mobility area)
            const rook_atk = movegen.sliderMovesWithOccupancy(s, piece.rook, occ_without_our_rq);
            attacks.rook |= rook_atk;
            const mob = rook_atk & mobility_area;
            const move_count = @popCount(mob);
            score = score.add(p.mobility_bonus[2][move_count]);

            // Open/semi-open file
            const fmask = file_masks[file];
            const has_our_pawn = (our_pawns_bb.bits & fmask) != 0;
            const has_opp_pawn = (opp_pawns_bb.bits & fmask) != 0;
            if (!has_our_pawn and !has_opp_pawn) {
                score = score.add(p.rook_open_file);
            } else if (!has_our_pawn and has_opp_pawn) {
                score = score.add(p.rook_semi_open);
            }

            // Rook on 7th rank
            if (rank == seventh_rank) {
                score = score.add(p.rook_on_seventh);
            }
        }
    }

    // --- Queens: material + PST (mobility deferred to evaluateQueenMobilityGeneric) ---
    {
        var queens = state.pieceBitboard(piece.queen).bitAnd(u64, our_pieces);
        const queen_count: i32 = @intCast(queens.popCount());
        score = score.add(p.piece_values[piece.queen].mul(queen_count));
        phase += queen_count * phase_weights[4];

        while (queens.next()) |s| {
            const rank: Square = s / 8;
            const file: Square = s % 8;
            const sq: Square = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            score = score.add(p.pst[piece.queen][sq]);
        }
    }

    // --- King: PST + king safety ---
    {
        const king_bb = state.pieceBitboard(piece.king).bitAnd(u64, our_pieces);
        const king_sq: Square = @intCast(@ctz(king_bb.bits));
        const king_file: u3 = @intCast(king_sq % 8);
        const king_rank: u3 = @intCast(king_sq / 8);

        // PST
        const pst_sq: Square = if (c == Colors.black) @intCast((@as(Square, 7) - king_rank) * 8 + king_file) else king_sq;
        score = score.add(p.pst[piece.king][pst_sq]);

        // King safety: pawn shield (only when king on back ranks)
        const on_back_ranks = if (c == Colors.white) king_rank <= 1 else king_rank >= 6;
        if (on_back_ranks) {
            const shield_rank: Square = if (c == Colors.white) @as(Square, king_rank) + 1 else @as(Square, king_rank) - 1;
            const min_file: u3 = if (king_file > 0) king_file - 1 else 0;
            const max_file: u3 = if (king_file < 7) king_file + 1 else 7;

            var file: u4 = min_file;
            while (file <= max_file) : (file += 1) {
                const shield_sq: Square = @as(Square, @as(u3, @intCast(file))) + shield_rank * 8;
                const shield_mask: u64 = @as(u64, 1) << shield_sq;
                if ((our_pawns_bb.bits & shield_mask) != 0) {
                    score = score.add(p.pawn_shield);
                } else {
                    score = score.add(p.pawn_shield_missing);
                }
            }
        }
    }

    return .{ .score = score, .phase = phase, .attacks = attacks };
}

// Evaluate queen mobility separately, after both colors' piece attacks are known.
// Queen mobility excludes squares defended by enemy minor pieces and rooks.
// `p` is anytype — either *const Params (i16) or *const ParamsF64 (f64).
fn evaluateQueenMobilityGeneric(
    state: *const State,
    our_pieces: u64,
    mobility_area: u64,
    enemy_attacks: PieceAttacks,
    p: anytype,
) @TypeOf(p.piece_values[0]) {
    const ScoreT = @TypeOf(p.piece_values[0]);
    var score = ScoreT.zero;
    var queens = state.pieceBitboard(piece.queen).bitAnd(u64, our_pieces);
    const enemy_minor_rook = enemy_attacks.knight | enemy_attacks.bishop | enemy_attacks.rook;

    while (queens.next()) |s| {
        const bishop_moves = movegen.sliderMoves(state, s, piece.bishop);
        const rook_moves = movegen.sliderMoves(state, s, piece.rook);
        const queen_atk = bishop_moves | rook_moves;
        const mob = queen_atk & mobility_area & ~enemy_minor_rook;
        const move_count = @popCount(mob);
        score = score.add(p.mobility_bonus[3][move_count]);
    }
    return score;
}

// Generic implementation shared by evaluate(), evaluateWithParams(), and
// evaluateWithParamsF64(). `p` is anytype (either *const Params or *const
// ParamsF64). The return type is i32 for Params and f64 for ParamsF64,
// inferred from Score(T).taper()'s return type.
fn evaluateGeneric(state: *const State, p: anytype) @TypeOf(p.piece_values[0].taper(0)) {
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
    const our = evaluateColorGeneric(state, to_move, our_pieces, our_pawns, opp_pawns, our_mob_area, p);
    const their = evaluateColorGeneric(state, opp, opp_pieces, opp_pawns, our_pawns, opp_mob_area, p);

    // Queen mobility using opponent's accumulated attack maps
    const our_q = evaluateQueenMobilityGeneric(state, our_pieces, our_mob_area, their.attacks, p);
    const their_q = evaluateQueenMobilityGeneric(state, opp_pieces, opp_mob_area, our.attacks, p);

    const phase = @min(our.phase + their.phase, max_phase_mg);
    const total = our.score.add(our_q).sub(their.score).sub(their_q).add(p.tempo);
    return total.taper(phase);
}

// Production entry point. Passes default_params as a comptime-constant address;
// LLVM constant-propagates field accesses under ReleaseFast, giving identical
// performance to the original named constants.
pub fn evaluate(state: *const State) i32 {
    return evaluateGeneric(state, &params_mod.default_params);
}

// Called by the tuner with i16 params (production type). Because the
// production call passes a comptime-constant address, LLVM inlines field
// accesses identically to the original named constants.
pub fn evaluateWithParams(state: *const State, p: *const Params) i32 {
    return evaluateGeneric(state, p);
}

// Called by the tuner's MSE computation with f64 params. Returns f64 so that
// MSE accumulation stays in floating-point and never discards gradient signal
// through integer truncation.
pub fn evaluateWithParamsF64(state: *const State, p: *const params_mod.ParamsF64) f64 {
    return evaluateGeneric(state, p);
}

// ==============================================================================
// Trace evaluation (debug-only, not hot path)
// ==============================================================================

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
        const cp_total: f32 = @as(f32, @floatFromInt(self.total)) / @as(f32, @floatFromInt(params_mod.default_params.piece_values[0].endgame()));
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

// Trace variant of evaluateColorGeneric. Not parametric — always references
// default_params directly, since the trace is a debug tool and does not need
// to reflect in-flight tuning state.
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
    const dp = &params_mod.default_params;
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

    // Precompute for passed pawn evaluation
    const opp_pieces_bb = occupied ^ our_pieces;
    const our_rooks = state.pieceBitboard(piece.rook).bits & our_pieces;
    const opp_king_sq: Square = @intCast(@ctz(state.pieceBitboard(piece.king).bits & opp_pieces_bb));
    const opp_pawn_atk = pawnAttacksBB(opp_pawns_bb.bits, ~c);

    // X-ray occupancy
    const our_queens_bb = state.pieceBitboard(piece.queen).bits & our_pieces;
    const occ_without_our_queens = Bitboard{ .bits = occupied ^ our_queens_bb };
    const our_rq = (state.pieceBitboard(piece.rook).bits | our_queens_bb) & our_pieces;
    const occ_without_our_rq = Bitboard{ .bits = occupied ^ our_rq };

    // --- Pawns ---
    {
        var pawns = our_pawns_bb;
        const pawn_count: i32 = @intCast(pawns.popCount());
        material_score = material_score.add(dp.piece_values[piece.pawn].mul(pawn_count));

        for (0..8) |file| {
            const count: i32 = @intCast(@popCount(our_pawns_bb.bits & file_masks[file]));
            if (count > 1) {
                pawn_structure_score = pawn_structure_score.add(dp.doubled_pawn.mul(count - 1));
            }
        }

        while (pawns.next()) |s| {
            const file: Square = s % 8;
            const rank: Square = s / 8;
            const sq: Square = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            pst_score = pst_score.add(dp.pst[piece.pawn][sq]);

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
                pawn_structure_score = pawn_structure_score.add(dp.connected_pawn);
            }

            const ahead_mask = computePassedPawnMask(c, file, rank);
            if ((opp_pawns_bb.bits & ahead_mask) == 0) {
                const passed_rank = if (c == Colors.white) rank else 7 - rank;
                passed_pawns_score = passed_pawns_score.add(dp.passed_pawn_bonus[passed_rank]);

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
                    passed_pawns_score = passed_pawns_score.add(dp.protected_passed_pawn);
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
                    passed_pawns_score = passed_pawns_score.add(dp.blocked_passed_pawn);
                }

                // Rook behind passed pawn
                {
                    const ranks_behind: u64 = if (c == Colors.white)
                        (@as(u64, 1) << (rank * 8)) - 1
                    else
                        @as(u64, 0xFFFFFFFFFFFFFFFF) << ((rank + 1) * 8);
                    if ((our_rooks & file_masks[file] & ranks_behind) != 0) {
                        passed_pawns_score = passed_pawns_score.add(dp.rook_behind_passer);
                    }
                }

                // Enemy king distance to passed pawn (endgame bonus)
                {
                    const king_file: Square = opp_king_sq % 8;
                    const king_rank: Square = opp_king_sq / 8;
                    const df: Square = if (file > king_file) file - king_file else king_file - file;
                    const dr: Square = if (rank > king_rank) rank - king_rank else king_rank - rank;
                    const dist: i16 = @intCast(@max(df, dr));
                    passed_pawns_score = passed_pawns_score.add(Score.init(0, dist * dp.king_proximity_passer));
                }

                // Free passed pawn (advance square not occupied or attacked by enemy pawns)
                if (!blocked) {
                    const ahead_sq: u64 = if (c == Colors.white and rank < 7)
                        @as(u64, 1) << (s + 8)
                    else if (c == Colors.black and rank > 0)
                        @as(u64, 1) << (s - 8)
                    else
                        0;
                    if (ahead_sq != 0 and (opp_pawn_atk & ahead_sq) == 0) {
                        passed_pawns_score = passed_pawns_score.add(dp.free_passed_pawn);
                    }
                }
            }

            if ((our_pawns_bb.bits & adjacent_files[file]) == 0) {
                pawn_structure_score = pawn_structure_score.add(dp.isolated_pawn);
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
                    const stop_sq: Square = if (c == Colors.white) s + 8 else s - 8;
                    const stop_file: Square = stop_sq % 8;
                    const stop_rank: Square = stop_sq / 8;
                    const stop_attacked = blk: {
                        const atk_rank: Square = if (c == Colors.white) stop_rank + 1 else if (stop_rank > 0) stop_rank - 1 else break :blk false;
                        if (c == Colors.black and stop_rank == 0) break :blk false;
                        if (c == Colors.white and atk_rank > 7) break :blk false;
                        var atk_mask: u64 = 0;
                        if (stop_file > 0) atk_mask |= @as(u64, 1) << (@as(Square, stop_file - 1) + atk_rank * 8);
                        if (stop_file < 7) atk_mask |= @as(u64, 1) << (@as(Square, stop_file + 1) + atk_rank * 8);
                        break :blk (opp_pawns_bb.bits & atk_mask) != 0;
                    };
                    if (stop_attacked) {
                        pawn_structure_score = pawn_structure_score.add(dp.backward_pawn);
                    }
                }
            }
        }
    }

    // --- Knights ---
    {
        var knights = state.pieceBitboard(piece.knight).bitAnd(u64, our_pieces);
        const knight_count: i32 = @intCast(knights.popCount());
        material_score = material_score.add(dp.piece_values[piece.knight].mul(knight_count));
        phase += knight_count * phase_weights[piece.knight];

        while (knights.next()) |s| {
            const file: u3 = @intCast(s % 8);
            const rank: u3 = @intCast(s / 8);
            const sq: Square = if (c == Colors.black) @intCast((@as(Square, 7) - rank) * 8 + file) else s;
            pst_score = pst_score.add(dp.pst[piece.knight][sq]);

            const knight_atk = movegen.knight_move_mask[s];
            attacks.knight |= knight_atk;
            const mob = knight_atk & mobility_area;
            const move_count = @popCount(mob);
            mobility_score = mobility_score.add(dp.mobility_bonus[0][move_count]);

            if (isOutpost(c, file, rank, opp_pawns_bb) and
                isDefendedByPawn(c, file, rank, our_pawns_bb))
            {
                outposts_score = outposts_score.add(dp.knight_outpost_defended);
            }
        }
    }

    // --- Bishops ---
    {
        var bishops = state.pieceBitboard(piece.bishop).bitAnd(u64, our_pieces);
        const bishop_count: i32 = @intCast(bishops.popCount());
        material_score = material_score.add(dp.piece_values[piece.bishop].mul(bishop_count));
        phase += bishop_count * phase_weights[piece.bishop];

        if (bishop_count >= 2) {
            bishop_pair_score = bishop_pair_score.add(dp.bishop_pair);
        }

        while (bishops.next()) |s| {
            const rank: Square = s / 8;
            const file: Square = s % 8;
            const sq: Square = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            pst_score = pst_score.add(dp.pst[piece.bishop][sq]);

            const bishop_atk = movegen.sliderMovesWithOccupancy(s, piece.bishop, occ_without_our_queens);
            attacks.bishop |= bishop_atk;
            const mob = bishop_atk & mobility_area;
            const move_count = @popCount(mob);
            mobility_score = mobility_score.add(dp.mobility_bonus[1][move_count]);

            // Bishop outpost (defended only)
            if (isOutpost(c, file, rank, opp_pawns_bb) and
                isDefendedByPawn(c, file, rank, our_pawns_bb))
            {
                outposts_score = outposts_score.add(dp.bishop_outpost_defended);
            }
        }
    }

    // --- Rooks ---
    {
        var rooks = state.pieceBitboard(piece.rook).bitAnd(u64, our_pieces);
        const rook_count: i32 = @intCast(rooks.popCount());
        material_score = material_score.add(dp.piece_values[piece.rook].mul(rook_count));
        phase += rook_count * phase_weights[piece.rook];

        const seventh_rank: Square = if (c == Colors.white) 6 else 1;

        while (rooks.next()) |s| {
            const file: Square = s % 8;
            const rank: Square = s / 8;
            const sq: Square = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            pst_score = pst_score.add(dp.pst[piece.rook][sq]);

            const rook_atk = movegen.sliderMovesWithOccupancy(s, piece.rook, occ_without_our_rq);
            attacks.rook |= rook_atk;
            const mob = rook_atk & mobility_area;
            const move_count = @popCount(mob);
            mobility_score = mobility_score.add(dp.mobility_bonus[2][move_count]);

            const fmask = file_masks[file];
            const has_our_pawn = (our_pawns_bb.bits & fmask) != 0;
            const has_opp_pawn = (opp_pawns_bb.bits & fmask) != 0;
            if (!has_our_pawn and !has_opp_pawn) {
                rook_bonuses_score = rook_bonuses_score.add(dp.rook_open_file);
            } else if (!has_our_pawn and has_opp_pawn) {
                rook_bonuses_score = rook_bonuses_score.add(dp.rook_semi_open);
            }

            if (rank == seventh_rank) {
                rook_bonuses_score = rook_bonuses_score.add(dp.rook_on_seventh);
            }
        }
    }

    // --- Queens (material + PST only, mobility deferred) ---
    {
        var queens = state.pieceBitboard(piece.queen).bitAnd(u64, our_pieces);
        const queen_count: i32 = @intCast(queens.popCount());
        material_score = material_score.add(dp.piece_values[piece.queen].mul(queen_count));
        phase += queen_count * phase_weights[piece.queen];

        while (queens.next()) |s| {
            const rank: Square = s / 8;
            const file: Square = s % 8;
            const sq: Square = if (c == Colors.black) @intCast((7 - rank) * 8 + file) else s;
            pst_score = pst_score.add(dp.pst[piece.queen][sq]);
        }
    }

    // --- King ---
    {
        const king_bb = state.pieceBitboard(piece.king).bitAnd(u64, our_pieces);
        const king_sq: Square = @intCast(@ctz(king_bb.bits));
        const king_file: u3 = @intCast(king_sq % 8);
        const king_rank: u3 = @intCast(king_sq / 8);
        const pst_sq: Square = if (c == Colors.black) @intCast((@as(Square, 7) - king_rank) * 8 + king_file) else king_sq;
        pst_score = pst_score.add(dp.pst[piece.king][pst_sq]);

        const on_back_ranks = if (c == Colors.white) king_rank <= 1 else king_rank >= 6;
        if (on_back_ranks) {
            const shield_rank: Square = if (c == Colors.white) @as(Square, king_rank) + 1 else @as(Square, king_rank) - 1;
            const min_file: u3 = if (king_file > 0) king_file - 1 else 0;
            const max_file: u3 = if (king_file < 7) king_file + 1 else 7;

            var f: u4 = min_file;
            while (f <= max_file) : (f += 1) {
                const shield_sq: Square = @as(Square, @as(u3, @intCast(f))) + shield_rank * 8;
                const shield_mask: u64 = @as(u64, 1) << shield_sq;
                if ((our_pawns_bb.bits & shield_mask) != 0) {
                    king_safety_score = king_safety_score.add(dp.pawn_shield);
                } else {
                    king_safety_score = king_safety_score.add(dp.pawn_shield_missing);
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
        const move_count = @popCount(mob);
        score = score.add(params_mod.default_params.mobility_bonus[3][move_count]);
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

    const our = evaluateColorTrace(state, to_move, our_pieces, our_pawns, opp_pawns, our_mob_area);
    const their = evaluateColorTrace(state, opp, opp_pieces, opp_pawns, our_pawns, opp_mob_area);

    // Queen mobility using opponent's accumulated attack maps
    const our_q = evaluateQueenMobilityTrace(state, our_pieces, our_mob_area, their.attacks);
    const their_q = evaluateQueenMobilityTrace(state, opp_pieces, opp_mob_area, our.attacks);

    var trace: EvalTrace = undefined;
    trace.material[to_move] = our.material;
    trace.material[~to_move] = their.material;
    trace.pst[to_move] = our.pst_score;
    trace.pst[~to_move] = their.pst_score;
    trace.pawn_structure[to_move] = our.pawn_structure;
    trace.passed_pawns[to_move] = our.passed_pawns_score;
    trace.pawn_structure[~to_move] = their.pawn_structure;
    trace.passed_pawns[~to_move] = their.passed_pawns_score;
    trace.outposts[to_move] = our.outposts_score;
    trace.outposts[~to_move] = their.outposts_score;
    trace.bishop_pair[to_move] = our.bishop_pair_score;
    trace.bishop_pair[~to_move] = their.bishop_pair_score;
    trace.rook_bonuses[to_move] = our.rook_bonuses_score;
    trace.rook_bonuses[~to_move] = their.rook_bonuses_score;
    trace.mobility[to_move] = our.mobility_score.add(our_q);
    trace.mobility[~to_move] = their.mobility_score.add(their_q);
    trace.king_safety[to_move] = our.king_safety_score;
    trace.king_safety[~to_move] = their.king_safety_score;

    trace.tempo_score = params_mod.default_params.tempo;
    trace.phase = @min(our.phase + their.phase, max_phase_mg);
    const total = our.material
        .add(our.pst_score)
        .add(our.pawn_structure)
        .add(our.passed_pawns_score)
        .add(our.outposts_score)
        .add(our.bishop_pair_score)
        .add(our.rook_bonuses_score)
        .add(our.mobility_score)
        .add(our_q)
        .add(our.king_safety_score)
        .sub(their.material)
        .sub(their.pst_score)
        .sub(their.pawn_structure)
        .sub(their.passed_pawns_score)
        .sub(their.outposts_score)
        .sub(their.bishop_pair_score)
        .sub(their.rook_bonuses_score)
        .sub(their.mobility_score)
        .sub(their_q)
        .sub(their.king_safety_score)
        .add(params_mod.default_params.tempo);
    trace.total = total.taper(trace.phase);

    return trace;
}

// ==============================================================================
// Move ordering
// ==============================================================================

// History heuristic table: [color][from_square][to_square] -> score
// Tracks which quiet moves have caused beta cutoffs
pub const HistoryTable = struct {
    table: [2][64][64]i32 = [_][64][64]i32{
        [_][64]i32{
            [_]i32{0} ** 64,
        } ** 64,
    } ** 2,

    pub fn get(self: *const HistoryTable, color: Color, from: Square, to: Square) i32 {
        return self.table[color][from][to];
    }

    const max_history: i32 = 16384;

    pub fn update(self: *HistoryTable, color: Color, from: Square, to: Square, bonus: i32) void {
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
