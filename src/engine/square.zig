const std = @import("std");
const Bitboard = @import("Bitboard.zig");
const Direction = @import("engine.zig").Direction;

pub const Square = u6;

pub const a1: Square = 0;
pub const b1: Square = 1;
pub const c1: Square = 2;
pub const d1: Square = 3;
pub const e1: Square = 4;
pub const f1: Square = 5;
pub const g1: Square = 6;
pub const h1: Square = 7;
pub const a2: Square = 8;
pub const b2: Square = 9;
pub const c2: Square = 10;
pub const d2: Square = 11;
pub const e2: Square = 12;
pub const f2: Square = 13;
pub const g2: Square = 14;
pub const h2: Square = 15;
pub const a3: Square = 16;
pub const b3: Square = 17;
pub const c3: Square = 18;
pub const d3: Square = 19;
pub const e3: Square = 20;
pub const f3: Square = 21;
pub const g3: Square = 22;
pub const h3: Square = 23;
pub const a4: Square = 24;
pub const b4: Square = 25;
pub const c4: Square = 26;
pub const d4: Square = 27;
pub const e4: Square = 28;
pub const f4: Square = 29;
pub const g4: Square = 30;
pub const h4: Square = 31;
pub const a5: Square = 32;
pub const b5: Square = 33;
pub const c5: Square = 34;
pub const d5: Square = 35;
pub const e5: Square = 36;
pub const f5: Square = 37;
pub const g5: Square = 38;
pub const h5: Square = 39;
pub const a6: Square = 40;
pub const b6: Square = 41;
pub const c6: Square = 42;
pub const d6: Square = 43;
pub const e6: Square = 44;
pub const f6: Square = 45;
pub const g6: Square = 46;
pub const h6: Square = 47;
pub const a7: Square = 48;
pub const b7: Square = 49;
pub const c7: Square = 50;
pub const d7: Square = 51;
pub const e7: Square = 52;
pub const f7: Square = 53;
pub const g7: Square = 54;
pub const h7: Square = 55;
pub const a8: Square = 56;
pub const b8: Square = 57;
pub const c8: Square = 58;
pub const d8: Square = 59;
pub const e8: Square = 60;
pub const f8: Square = 61;
pub const g8: Square = 62;
pub const h8: Square = 63;

pub fn toAlgebraic(square: Square, buf: []u8) !void {
    const file: u8 = 'a' + @as(u8, square) % 8;
    const rank: u8 = '1' + @as(u8, square) / 8;
    _ = try std.fmt.bufPrint(buf, "{c}{c}", .{ file, rank });
}

pub fn algebraicToSquare(s: []const u8) ?Square {
    if (s.len != 2) {
        return null;
    }

    const file = s[0];
    const rank = s[1];

    if (!(file >= 'a' and file <= 'h') or !(rank >= '1' and rank <= '8')) {
        return null;
    }

    const file_idx = file - 'a';
    const rank_idx = rank - '1';

    return @intCast(rank_idx * 8 + file_idx);
}

pub fn trySquareOffset(s: Square, dx: i3, dy: i3) ?Square {
    const file: i6 = @intCast(s % 8);
    const rank: i6 = @intCast(s / 8);
    const new_file = file + dx;
    const new_rank = rank + dy;

    if (new_file >= 0 and new_file < 8 and new_rank >= 0 and new_rank < 8) {
        const new_rank_u: u6 = @intCast(new_rank);
        const new_file_u: u6 = @intCast(new_file);
        return @intCast(new_rank_u * 8 + new_file_u);
    } else {
        return null;
    }
}

test "test try square offset" {
    try std.testing.expectEqual(trySquareOffset(a1, -1, 0), null);
}

pub fn absDiff(lhs: Square, rhs: Square) Square {
    const diff = @as(i8, @intCast(lhs)) - @as(i8, @intCast(rhs));
    return @intCast(@abs(diff));
}

// 8 cardinal/diagonal ray directions
pub const RayDirection = enum(u3) {
    N = 0,
    NE = 1,
    E = 2,
    SE = 3,
    S = 4,
    SW = 5,
    W = 6,
    NW = 7,

    pub fn opposite(self: RayDirection) RayDirection {
        return @enumFromInt(@as(u3, @intFromEnum(self)) ^ 4);
    }

    // Positive directions have increasing square indices (use @ctz for closest blocker).
    // Negative directions have decreasing indices (use @clz).
    pub fn isPositive(self: RayDirection) bool {
        return switch (self) {
            .N, .NE, .E, .NW => true,
            .S, .SE, .SW, .W => false,
        };
    }
};

// Map a Direction (4-way) plus relative square order to a RayDirection (8-way)
pub fn toRayDirection(direction: Direction, from: Square, to: Square) ?RayDirection {
    return switch (direction) {
        .vertical => if (to > from) RayDirection.N else RayDirection.S,
        .horizontal => if (to > from) RayDirection.E else RayDirection.W,
        .diagonal => if (to > from) RayDirection.NE else RayDirection.SW,
        .antiDiagonal => if (to > from) RayDirection.NW else RayDirection.SE,
        .none => null,
    };
}

// Precomputed ray attacks: ray_attacks[direction][square]
// Each entry is the set of squares from `square` (exclusive) extending to the board edge.
pub const ray_attacks = computeRayAttacks();

fn computeRayAttacks() [8][64]u64 {
    @setEvalBranchQuota(10000);
    var rays: [8][64]u64 = undefined;
    const deltas = [8][2]i8{
        .{ 1, 0 },   // N
        .{ 1, 1 },   // NE
        .{ 0, 1 },   // E
        .{ -1, 1 },  // SE
        .{ -1, 0 },  // S
        .{ -1, -1 }, // SW
        .{ 0, -1 },  // W
        .{ 1, -1 },  // NW
    };

    for (0..64) |sq| {
        const file: i8 = @intCast(sq % 8);
        const rank: i8 = @intCast(sq / 8);

        for (0..8) |dir| {
            var bb: u64 = 0;
            var r = rank + deltas[dir][0];
            var f = file + deltas[dir][1];
            while (r >= 0 and r < 8 and f >= 0 and f < 8) {
                bb |= @as(u64, 1) << @intCast(r * 8 + f);
                r += deltas[dir][0];
                f += deltas[dir][1];
            }
            rays[dir][sq] = bb;
        }
    }

    return rays;
}

pub fn betweenSquares(from: Square, to: Square) Bitboard {
    if (from == to) {
        return Bitboard.empty;
    }

    const direction = Direction.fromSquares(from, to);
    if (direction == .none) {
        return Bitboard.empty;
    }

    const ray = rayBetweenInclusive(from, to, direction);
    return ray.bitAnd(Bitboard.initSquare(from).not()).bitAnd(Bitboard.initSquare(to).not());
}

pub fn rayBetweenInclusive(from: Square, to: Square, d: Direction) Bitboard {
    const from_file = from % 8;
    const from_rank = from / 8;
    const to_file = to % 8;
    const to_rank = to / 8;

    var ray = Bitboard.empty;

    switch (d) {
        .horizontal => {
            const min_file = @min(from_file, to_file);
            const max_file = @max(from_file, to_file);
            for (min_file..max_file + 1) |f| {
                const f_u6: u6 = @intCast(f);
                ray.bitOrAssign(from_rank * 8 + f_u6);
            }
        },
        .vertical => {
            const min_rank = @min(from_rank, to_rank);
            const max_rank = @max(from_rank, to_rank);
            for (min_rank..max_rank + 1) |r| {
                const r_u6: u6 = @intCast(r);
                ray.bitOrAssign(r_u6 * 8 + from_file);
            }
        },
        .diagonal, .antiDiagonal => {
            const rank_step: i8 = if (to_rank > from_rank)
                1
            else
                -1;

            const file_step: i8 = if (to_file > from_file)
                1
            else
                -1;

            var r = @as(i8, from_rank);
            var f = @as(i8, from_file);

            while (true) {
                const s: Square = @intCast(r * 8 + f);
                ray.bitOrAssign(s);
                if (r == @as(i8, to_rank) and f == @as(i8, to_file)) {
                    break;
                }
                r += rank_step;
                f += file_step;
            }
        },
        .none => {},
    }

    return ray;
}
