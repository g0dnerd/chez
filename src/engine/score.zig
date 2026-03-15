const std = @import("std");

// Generic packed score interface holding both middlegame and endgame values.
// Allows evaluating once and interpolating at the end based on game phase.
pub fn Score(comptime T: type) type {
    std.debug.assert(T == i16 or T == f64);

    return struct {
        const Self = @This();

        v: @Vector(2, T),

        pub const zero = Self{ .v = @splat(0) };

        pub fn init(mg: T, eg: T) Self {
            return .{ .v = .{ mg, eg } };
        }

        pub fn add(self: Self, other: Self) Self {
            return .{ .v = self.v + other.v };
        }

        pub fn sub(self: Self, other: Self) Self {
            return .{ .v = self.v - other.v };
        }

        pub fn neg(self: Self) Self {
            return .{ .v = -self.v };
        }

        pub fn mul(self: Self, n: i32) Self {
            const factor: T = switch (T) {
                i16 => @intCast(n),
                f64 => @floatFromInt(n),
                else => unreachable,
            };
            const vec_factor: @Vector(2, T) = @splat(factor);
            return .{ .v = self.v * vec_factor };
        }

        pub fn taper(self: Self, phase: i32) i32 {
            const mg: i32 = switch (T) {
                i16 => self.v[0],
                f64 => @intFromFloat(self.v[0]),
                else => unreachable,
            };
            const eg: i32 = switch (T) {
                i16 => self.v[1],
                f64 => @intFromFloat(self.v[1]),
                else => unreachable,
            };
            return @divTrunc(mg * phase + eg * (max_phase_mg - phase), max_phase_mg);
        }

        pub fn midgame(self: Self) T {
            return self.v[0];
        }

        pub fn endgame(self: Self) T {
            return self.v[1];
        }
    };
}

pub const max_phase_mg: i32 = 24;
