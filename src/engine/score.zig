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

        // Returns i32 when T=i16, f64 when T=f64. The f64 path is used by
        // evaluateWithParamsF64 so that MSE accumulation never truncates gradient
        // signal through integer rounding mid-run.
        pub fn taper(self: Self, phase: i32) if (T == f64) f64 else i32 {
            if (T == f64) {
                const mg: f64 = self.v[0];
                const eg: f64 = self.v[1];
                const p: f64 = @floatFromInt(phase);
                const mp: f64 = @floatFromInt(max_phase_mg);
                return (mg * p + eg * (mp - p)) / mp;
            } else {
                const mg: i32 = self.v[0];
                const eg: i32 = self.v[1];
                return @divTrunc(mg * phase + eg * (max_phase_mg - phase), max_phase_mg);
            }
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
