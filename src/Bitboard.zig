const std = @import("std");
const expectEqual = std.testing.expectEqual;

const chez = @import("chez.zig");
const Squares = chez.Squares;
const Square = chez.Square;

pub const Bitboard = @This();
bits: u64 = 0,

pub const empty = Bitboard{ .bits = 0 };

pub fn fromSquare(s: Square) Bitboard {
    return .{ .bits = @as(u64, 1) << s };
}

pub fn contains(self: *const Bitboard, s: Square) bool {
    return self.bits & @as(u64, 1) << s != 0;
}

pub fn isEmpty(self: *const Bitboard) bool {
    return self.bits == 0;
}

pub fn popCount(self: *const Bitboard) u32 {
    return @popCount(self.bits);
}

pub fn trailingZeros(self: *const Bitboard) Square {
    return @intCast(@ctz(self.bits));
}

fn clearLsb(self: *Bitboard) void {
    self.*.bits &= self.bits - 1;
}

pub fn colorflip(self: *const Bitboard) Bitboard {
    var flipped = Bitboard.empty;
    flipped.bits = @byteSwap(self.bits);
    return flipped;
}

pub fn next(self: *Bitboard) ?Square {
    if (self.isEmpty()) {
        return null;
    }
    const ret = self.trailingZeros();
    self.clearLsb();
    return ret;
}

pub fn bitOr(self: *const Bitboard, rhs: anytype) Bitboard {
    return switch (@TypeOf(rhs)) {
        Square => Bitboard{ .bits = self.bits | @as(u64, 1) << rhs },
        u64 => Bitboard{ .bits = self.bits | rhs },
        Bitboard => Bitboard{ .bits = self.bits | rhs.bits },
        else => unreachable,
    };
}

pub fn bitAnd(self: *const Bitboard, rhs: anytype) Bitboard {
    return switch (@TypeOf(rhs)) {
        Square => Bitboard{ .bits = self.bits & @as(u64, 1) << rhs },
        u64 => Bitboard{ .bits = self.bits & rhs },
        Bitboard => Bitboard{ .bits = self.bits & rhs.bits },
        else => unreachable,
    };
}

pub fn bitXor(self: *const Bitboard, rhs: anytype) Bitboard {
    return switch (@TypeOf(rhs)) {
        Square => Bitboard{ .bits = self.bits ^ @as(u64, 1) << rhs },
        u64 => Bitboard{ .bits = self.bits ^ rhs },
        Bitboard => Bitboard{ .bits = self.bits ^ rhs.bits },
        else => unreachable,
    };
}

pub fn bitOrAssign(self: *Bitboard, rhs: anytype) void {
    switch (@TypeOf(rhs)) {
        Square => self.bits |= @as(u64, 1) << rhs,
        u64 => self.bits |= rhs,
        Bitboard => self.bits |= rhs.bits,
        else => unreachable,
    }
}

pub fn bitAndAssign(self: *Bitboard, rhs: anytype) void {
    switch (@TypeOf(rhs)) {
        Square => self.bits &= @as(u64, 1) << rhs,
        u64 => self.bits &= rhs,
        Bitboard => self.bits &= rhs.bits,
        else => unreachable,
    }
}

pub fn bitXorAssign(self: *Bitboard, rhs: anytype) void {
    switch (@TypeOf(rhs)) {
        Square => self.bits ^= @as(u64, 1) << rhs,
        u64 => self.bits ^= rhs,
        Bitboard => self.bits ^= rhs.bits,
        else => unreachable,
    }
}

pub fn shl(self: *const Bitboard, rhs: anytype) Bitboard {
    switch (@TypeOf(rhs)) {
        Square, u64 => return Bitboard{ .bits = self.bits << rhs },
        Bitboard => return Bitboard{ .bits = self.bits << rhs.bits },
        else => unreachable,
    }
}

pub fn shr(self: *const Bitboard, rhs: anytype) Bitboard {
    switch (@TypeOf(rhs)) {
        Square, u64 => return Bitboard{ .bits = self.bits >> rhs },
        Bitboard => return Bitboard{ .bits = self.bits >> rhs.bits },
        else => unreachable,
    }
}

pub fn not(self: *const Bitboard) Bitboard {
    return Bitboard{ .bits = ~self.bits };
}

test "bitboard contains sanity" {
    const bb = Bitboard.fromSquare(Squares.e2);
    try std.testing.expect(bb.contains(Squares.e2));
}

test "bitboard square bitwise sanity" {
    var bb1 = Bitboard.empty;
    bb1.bitOrAssign(Squares.e2);
    const bb1_from_square = Bitboard.fromSquare(Squares.e2);
    try expectEqual(bb1.bits, 4096);
    try expectEqual(bb1, bb1_from_square);

    var bb2 = Bitboard.empty;
    bb2.bitAndAssign(Squares.e2);
    const bb2_from_square = Bitboard.empty;
    try expectEqual(bb2.bits, 0);
    try expectEqual(bb2, bb2_from_square);

    var bb3 = Bitboard.fromSquare(Squares.d2);
    bb3.bitOrAssign(Squares.e2);
    bb3.bitXorAssign(Squares.d2);
    const bb3_from_square = Bitboard.fromSquare(Squares.e2);
    try expectEqual(bb3.bits, 4096);
    try expectEqual(bb3, bb3_from_square);
}

test "bitboard colorflip sanity" {
    const bb = Bitboard{ .bits = 0xaa55 };
    const flipped = bb.colorflip();
    try expectEqual(flipped.bits, 0x55aa000000000000);
}
