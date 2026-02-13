const std = @import("std");
const expectEqual = std.testing.expectEqual;

const square = @import("square.zig");
const Square = square.Square;

pub const Bitboard = @This();

bits: u64 = 0,

pub const empty = Bitboard{ .bits = 0 };

pub fn initSquare(s: Square) Bitboard {
    return .{ .bits = @as(u64, 1) << s };
}

pub fn contains(self: Bitboard, s: Square) bool {
    return self.bits & @as(u64, 1) << s != 0;
}

pub fn contains_u64(self: u64, s: Square) bool {
    return self & @as(u64, 1) << s != 0;
}

pub fn isEmpty(self: Bitboard) bool {
    return self.bits == 0;
}

pub fn popCount(self: Bitboard) u32 {
    return @popCount(self.bits);
}

pub fn trailingZeros(self: Bitboard) Square {
    return @intCast(@ctz(self.bits));
}

fn clearLsb(self: *Bitboard) void {
    self.*.bits &= self.bits - 1;
}

pub fn colorflip(self: Bitboard) Bitboard {
    var flipped = empty;
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

pub fn bitOr(self: Bitboard, T: type, rhs: T) Bitboard {
    return switch (T) {
        Square => Bitboard{ .bits = self.bits | @as(u64, 1) << rhs },
        u64 => Bitboard{ .bits = self.bits | rhs },
        Bitboard => Bitboard{ .bits = self.bits | rhs.bits },
        else => unreachable,
    };
}

pub fn bitAnd(self: Bitboard, T: type, rhs: T) Bitboard {
    return switch (T) {
        Square => Bitboard{ .bits = self.bits & @as(u64, 1) << rhs },
        u64 => Bitboard{ .bits = self.bits & rhs },
        Bitboard => Bitboard{ .bits = self.bits & rhs.bits },
        else => unreachable,
    };
}

pub fn bitXor(self: Bitboard, T: type, rhs: T) Bitboard {
    return switch (T) {
        Square => Bitboard{ .bits = self.bits ^ @as(u64, 1) << rhs },
        u64 => Bitboard{ .bits = self.bits ^ rhs },
        Bitboard => Bitboard{ .bits = self.bits ^ rhs.bits },
        else => unreachable,
    };
}

pub fn bitOrAssign(self: *Bitboard, T: type, rhs: T) void {
    switch (T) {
        Square => self.bits |= @as(u64, 1) << rhs,
        u64 => self.bits |= rhs,
        Bitboard => self.bits |= rhs.bits,
        else => unreachable,
    }
}

pub fn bitAndAssign(self: *Bitboard, T: type, rhs: T) void {
    switch (T) {
        Square => self.bits &= @as(u64, 1) << rhs,
        u64 => self.bits &= rhs,
        Bitboard => self.bits &= rhs.bits,
        else => unreachable,
    }
}

pub fn bitXorAssign(self: *Bitboard, T: type, rhs: T) void {
    switch (T) {
        Square => self.bits ^= @as(u64, 1) << rhs,
        u64 => self.bits ^= rhs,
        Bitboard => self.bits ^= rhs.bits,
        else => unreachable,
    }
}

pub fn not(self: Bitboard) Bitboard {
    return Bitboard{ .bits = ~self.bits };
}

test "bitboard contains sanity" {
    const bb = Bitboard.initSquare(square.e2);
    try std.testing.expect(bb.contains(square.e2));
}

test "bitboard square bitwise sanity" {
    var bb1 = Bitboard.empty;
    bb1.bitOrAssign(square.e2);
    const bb1_from_square = Bitboard.initSquare(square.e2);
    try expectEqual(bb1.bits, 4096);
    try expectEqual(bb1, bb1_from_square);

    var bb2 = Bitboard.empty;
    bb2.bitAndAssign(square.e2);
    const bb2_from_square = Bitboard.empty;
    try expectEqual(bb2.bits, 0);
    try expectEqual(bb2, bb2_from_square);

    var bb3 = Bitboard.initSquare(square.d2);
    bb3.bitOrAssign(square.e2);
    bb3.bitXorAssign(square.d2);
    const bb3_from_square = Bitboard.initSquare(square.e2);
    try expectEqual(bb3.bits, 4096);
    try expectEqual(bb3, bb3_from_square);
}

test "bitboard colorflip sanity" {
    const bb = Bitboard{ .bits = 0xaa55 };
    const flipped = bb.colorflip();
    try expectEqual(flipped.bits, 0x55aa000000000000);
}
