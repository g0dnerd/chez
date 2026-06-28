// Temp: pilot-validation stats over a selfplay
// .bin dataset. Decodes each 35-byte record with the real serde decoder and
// reports WDL balance, score distribution, and a piece-count histogram (used as
// a late-game-tail proxy, since the record format doesn't store game ply).
const builtin = @import("builtin");
const std = @import("std");
const chez = @import("chez");
const kore = @import("kore");
const engine = chez.engine;
const serde = @import("selfplay/serde.zig");

const record_size = 35;

pub fn main(init: std.process.Init.Minimal) !void {
    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .environ = .empty });
    const io = threaded.io();

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
    const out: *std.Io.Writer = &stdout_writer.interface;

    var args = if (builtin.os.tag == .windows)
        try init.args.iterateAllocator(std.heap.page_allocator)
    else
        init.args.iterate();
    _ = args.next(); // exe name
    const path = args.next() orelse return error.NoPathGiven;

    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const len: usize = @intCast(stat.size);
    if (len < record_size) return error.TooSmall;

    const ptr = try std.posix.mmap(
        null,
        len,
        std.os.linux.PROT{ .READ = true },
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(@alignCast(ptr));
    const data = ptr[0..len];

    const n = len / record_size;
    var piece_hist = [_]u64{0} ** 33;
    var wdl = [_]u64{0} ** 3;
    var score_sum: i64 = 0;
    var score_min: i32 = 100000;
    var score_max: i32 = -100000;
    var near0: u64 = 0;
    var decoded: u64 = 0;

    var posbuf: [32]u8 = undefined;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const rec = data[i * record_size ..][0..record_size];
        @memcpy(&posbuf, rec[0..32]);
        const state = serde.decodePosition(&posbuf) catch continue;
        decoded += 1;
        const pc: usize = @intCast(state.all_pieces.popCount());
        if (pc <= 32) piece_hist[pc] += 1;
        const score: i32 = std.mem.readInt(i16, rec[32..34], .little);
        score_sum += score;
        if (score < score_min) score_min = score;
        if (score > score_max) score_max = score;
        if (@abs(score) < 10) near0 += 1;
        const w = rec[34];
        if (w < 3) wdl[w] += 1;
    }

    const fn_ = @as(f64, @floatFromInt(n));
    try out.print("file: {s}\nrecords: {d}  decoded: {d}\n", .{ path, n, decoded });
    try out.print("WDL[0,1,2]: {d} {d} {d}  frac {d:.3}/{d:.3}/{d:.3}\n", .{
        wdl[0],                                  wdl[1], wdl[2],
        @as(f64, @floatFromInt(wdl[0])) / fn_,   @as(f64, @floatFromInt(wdl[1])) / fn_,
        @as(f64, @floatFromInt(wdl[2])) / fn_,
    });
    try out.print("score: mean {d:.1}  min {d}  max {d}  near0frac {d:.3}\n", .{
        @as(f64, @floatFromInt(score_sum)) / fn_, score_min, score_max,
        @as(f64, @floatFromInt(near0)) / fn_,
    });
    try out.print("piece-count histogram:\n", .{});
    var pcid: usize = 2;
    while (pcid <= 32) : (pcid += 1) {
        if (piece_hist[pcid] == 0) continue;
        try out.print("  {d:>2}: {d:>10}  {d:.4}\n", .{
            pcid, piece_hist[pcid], @as(f64, @floatFromInt(piece_hist[pcid])) / fn_,
        });
    }
    var le12: u64 = 0;
    var le10: u64 = 0;
    var le8: u64 = 0;
    var le7: u64 = 0;
    pcid = 2;
    while (pcid <= 32) : (pcid += 1) {
        if (pcid <= 12) le12 += piece_hist[pcid];
        if (pcid <= 10) le10 += piece_hist[pcid];
        if (pcid <= 8) le8 += piece_hist[pcid];
        if (pcid <= 7) le7 += piece_hist[pcid];
    }
    try out.print("endgame tail: <=12pc {d:.4}  <=10pc {d:.4}  <=8pc {d:.4}  <=7pc {d:.4}\n", .{
        @as(f64, @floatFromInt(le12)) / fn_, @as(f64, @floatFromInt(le10)) / fn_,
        @as(f64, @floatFromInt(le8)) / fn_,  @as(f64, @floatFromInt(le7)) / fn_,
    });
    try out.flush();
}
