// Syzygy tablebase WDL probing (pure Zig, WASM-safe).
//
// This module holds a RAW, primitive-typed function pointer to the actual C
// prober (Fathom) plus all of the State->args marshalling and WDL mapping. The
// pointer's type mentions only u64/u32/bool, so it is structurally identical
// across module boundaries -- the uci/selfplay executables install &fathom.probeRaw
// into it from their main(), where `engine.tablebase` is the same instance search
// reads. The engine module itself imports no C, so the WASM target stays C-free.

const std = @import("std");
const State = @import("State.zig");
const Colors = @import("engine.zig").Colors;

pub const Wdl = enum { loss, draw, win };

// Fathom's stable result codes (mirrors tbprobe.h; no C header needed here).
const TB_LOSS: u32 = 0;
const TB_BLESSED_LOSS: u32 = 1;
const TB_DRAW: u32 = 2;
const TB_CURSED_WIN: u32 = 3;
const TB_WIN: u32 = 4;
const TB_FAILED: u32 = 0xFFFFFFFF;

// Args order: white, black, kings, queens, rooks, bishops, knights, pawns, ep, turn.
pub var raw_probe_fn: ?*const fn (u64, u64, u64, u64, u64, u64, u64, u64, u32, bool) u32 = null;

// Largest supported man-count (TB_LARGEST from Fathom). 0 until tables are loaded.
pub var largest: u32 = 0;

pub fn available() bool {
    return raw_probe_fn != null and largest >= 3;
}

// Probe WDL for `state`. Returns null when no tables are loaded, the position is
// out of range (too many pieces or castling rights present), or the probe fails.
pub fn probeWdl(state: *const State) ?Wdl {
    const f = raw_probe_fn orelse return null;
    // TB positions have no castling rights (CastlingRights is a u4 bitset).
    if (state.castling_rights != 0) return null;
    if (state.all_pieces.popCount() > largest) return null;
    // Fathom: ep square, or 0 for none. Square a1 (0) can never be an ep target.
    const ep: u32 = if (state.en_passant) |sq| sq else 0;
    const r = f(
        state.colors[0].bits, // white
        state.colors[1].bits, // black
        state.pieces[5].bits, // kings
        state.pieces[4].bits, // queens
        state.pieces[3].bits, // rooks
        state.pieces[2].bits, // bishops
        state.pieces[1].bits, // knights
        state.pieces[0].bits, // pawns
        ep,
        state.to_move == Colors.white, // turn: true = white
    );
    return switch (r) {
        TB_LOSS => .loss,
        TB_WIN => .win,
        // 50-move-correct collapse: cursed/blessed results are draws under the
        // 50-move rule, which we always respect (we never pass rule50 > 0).
        TB_DRAW, TB_CURSED_WIN, TB_BLESSED_LOSS => .draw,
        else => null, // TB_FAILED or any unexpected code
    };
}

// Test-only fake prober: records the last call's args and returns a configurable
// code. Container-level decls are lazily analyzed, so this is only pulled into
// test builds. Tests run serially and each restores the globals via defer.
const mock = struct {
    const Args = struct {
        white: u64,
        black: u64,
        kings: u64,
        queens: u64,
        rooks: u64,
        bishops: u64,
        knights: u64,
        pawns: u64,
        ep: u32,
        turn: bool,
    };
    var last: ?Args = null;
    var ret: u32 = TB_DRAW;

    fn probe(white: u64, black: u64, kings: u64, queens: u64, rooks: u64, bishops: u64, knights: u64, pawns: u64, ep: u32, turn: bool) u32 {
        last = .{
            .white = white,
            .black = black,
            .kings = kings,
            .queens = queens,
            .rooks = rooks,
            .bishops = bishops,
            .knights = knights,
            .pawns = pawns,
            .ep = ep,
            .turn = turn,
        };
        return ret;
    }
};

test "no tablebase loaded: available() false and probeWdl returns null" {
    // Default global state: no prober wired in. (Other tests restore to this.)
    try std.testing.expect(raw_probe_fn == null);
    try std.testing.expect(!available());

    const state = State.defaultPosition();
    try std.testing.expect(probeWdl(&state) == null);
}

test "available() requires a prober and largest >= 3" {
    defer {
        raw_probe_fn = null;
        largest = 0;
    }

    raw_probe_fn = null;
    largest = 5;
    try std.testing.expect(!available()); // no prober

    raw_probe_fn = &mock.probe;
    largest = 0;
    try std.testing.expect(!available()); // tables found none
    largest = 2;
    try std.testing.expect(!available()); // below 3-man

    largest = 3;
    try std.testing.expect(available());
    largest = 5;
    try std.testing.expect(available());
}

test "probeWdl maps Fathom result codes" {
    defer {
        raw_probe_fn = null;
        largest = 0;
    }
    raw_probe_fn = &mock.probe;
    largest = 5;
    const state = try State.fromFen("8/8/8/4k3/8/8/8/3QK3 w - - 0 1");

    mock.ret = TB_WIN;
    try std.testing.expectEqual(Wdl.win, probeWdl(&state).?);
    mock.ret = TB_LOSS;
    try std.testing.expectEqual(Wdl.loss, probeWdl(&state).?);
    // Cursed/blessed collapse to draw (50-move-correct); plain draw too.
    mock.ret = TB_DRAW;
    try std.testing.expectEqual(Wdl.draw, probeWdl(&state).?);
    mock.ret = TB_CURSED_WIN;
    try std.testing.expectEqual(Wdl.draw, probeWdl(&state).?);
    mock.ret = TB_BLESSED_LOSS;
    try std.testing.expectEqual(Wdl.draw, probeWdl(&state).?);
    // Probe failure and any unexpected code -> null.
    mock.ret = TB_FAILED;
    try std.testing.expect(probeWdl(&state) == null);
    mock.ret = 99;
    try std.testing.expect(probeWdl(&state) == null);
}

test "probeWdl marshals State fields in Fathom arg order" {
    defer {
        raw_probe_fn = null;
        largest = 0;
    }
    raw_probe_fn = &mock.probe;
    largest = 8;
    mock.ret = TB_DRAW;

    // KQ vs k, white to move, no castling, no en passant.
    mock.last = null;
    const wtm = try State.fromFen("8/8/8/4k3/8/8/8/3QK3 w - - 0 1");
    _ = probeWdl(&wtm);
    const a = mock.last.?;
    try std.testing.expectEqual(wtm.colors[0].bits, a.white);
    try std.testing.expectEqual(wtm.colors[1].bits, a.black);
    try std.testing.expectEqual(wtm.pieces[5].bits, a.kings);
    try std.testing.expectEqual(wtm.pieces[4].bits, a.queens);
    try std.testing.expectEqual(wtm.pieces[3].bits, a.rooks);
    try std.testing.expectEqual(wtm.pieces[2].bits, a.bishops);
    try std.testing.expectEqual(wtm.pieces[1].bits, a.knights);
    try std.testing.expectEqual(wtm.pieces[0].bits, a.pawns);
    try std.testing.expectEqual(@as(u32, 0), a.ep);
    try std.testing.expectEqual(true, a.turn); // turn: true = white

    // Same position, black to move: turn flips to false.
    mock.last = null;
    const btm = try State.fromFen("8/8/8/4k3/8/8/8/3QK3 b - - 0 1");
    _ = probeWdl(&btm);
    try std.testing.expectEqual(false, mock.last.?.turn);
}

test "probeWdl forwards the en passant square" {
    defer {
        raw_probe_fn = null;
        largest = 0;
    }
    raw_probe_fn = &mock.probe;
    largest = 8;
    mock.ret = TB_DRAW;
    mock.last = null;

    // Black just played c7-c5; white Pd5 can capture ep on c6.
    const state = try State.fromFen("7k/8/8/2pP4/8/8/8/K7 w - c6 0 1");
    try std.testing.expect(state.en_passant != null);
    _ = probeWdl(&state);
    try std.testing.expectEqual(@as(u32, state.en_passant.?), mock.last.?.ep);
}

test "probeWdl gates: castling rights and piece count disable probing" {
    defer {
        raw_probe_fn = null;
        largest = 0;
    }
    raw_probe_fn = &mock.probe;
    mock.ret = TB_WIN;

    // Castling rights present: probe is skipped even within the man-count limit.
    largest = 32;
    mock.last = null;
    const castled = try State.fromFen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    try std.testing.expect(probeWdl(&castled) == null);
    try std.testing.expect(mock.last == null); // prober not called

    // More men than `largest` (and no castling): probe is skipped.
    largest = 3;
    mock.last = null;
    const four_men = try State.fromFen("8/8/8/3k4/8/3K4/3Q4/3R4 w - - 0 1");
    try std.testing.expect(probeWdl(&four_men) == null);
    try std.testing.expect(mock.last == null);
}
