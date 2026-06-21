// Reusable debugger for trained .nnue nets. Loads via the real loader and
// reports per-layer weight statistics, quantization saturation, accumulator
// CReLU occupancy, and eval spread on probe positions. Optionally diffs two
// nets side by side (e.g. a regressed net vs a known-good one).
//
//   zig build nnue-inspect -- --net data/net_v8.nnue
//   zig build nnue-inspect -- --net data/net_v8.nnue --compare data/net_v7.nnue
const builtin = @import("builtin");
const std = @import("std");
const kore = @import("kore");
const chez = @import("chez");
const engine = chez.engine;
const nnue = engine.nnue;
const State = engine.State;

const Args = struct {
    net: []const u8,
    compare: ?[]const u8,
};

const LayerStats = struct {
    n: usize,
    nonzero: usize,
    min: i64,
    max: i64,
    max_abs: u64,
    mean: f64,
    std: f64,
    sat: usize, // count at +/- saturation limit (quantization clipping)
    sat_limit: i64,

    fn of(comptime T: type, data: []const T) LayerStats {
        const lim: i64 = std.math.maxInt(T);
        var sum: f64 = 0;
        var sumsq: f64 = 0;
        var mn: i64 = std.math.maxInt(i64);
        var mx: i64 = std.math.minInt(i64);
        var nz: usize = 0;
        var sat: usize = 0;
        var max_abs: u64 = 0;
        for (data) |v| {
            const x: i64 = v;
            const xf: f64 = @floatFromInt(x);
            sum += xf;
            sumsq += xf * xf;
            if (x != 0) nz += 1;
            if (x < mn) mn = x;
            if (x > mx) mx = x;
            const a: u64 = @abs(x);
            if (a > max_abs) max_abs = a;
            if (x >= lim or x <= -lim) sat += 1;
        }
        const nf: f64 = @floatFromInt(data.len);
        const mean = sum / nf;
        const variance = @max(0.0, sumsq / nf - mean * mean);
        return .{
            .n = data.len,
            .nonzero = nz,
            .min = mn,
            .max = mx,
            .max_abs = max_abs,
            .mean = mean,
            .std = @sqrt(variance),
            .sat = sat,
            .sat_limit = lim,
        };
    }

    fn print(self: LayerStats, w: *std.Io.Writer, name: []const u8) !void {
        const pct_nz = 100.0 * @as(f64, @floatFromInt(self.nonzero)) / @as(f64, @floatFromInt(self.n));
        const pct_sat = 100.0 * @as(f64, @floatFromInt(self.sat)) / @as(f64, @floatFromInt(self.n));
        try w.print(
            "  {s:<14} n={d:<8} mean={d:>8.3} std={d:>8.3} min={d:>6} max={d:>6} max_abs={d:>6} nz={d:>5.1}% sat(±{d})={d:>5.2}%\n",
            .{ name, self.n, self.mean, self.std, self.min, self.max, self.max_abs, pct_nz, self.sat_limit, pct_sat },
        );
    }
};

fn flat(comptime T: type, ptr: anytype, n: usize) []const T {
    return @as([*]const T, @ptrCast(ptr))[0..n];
}

const probe_fens = [_]?[]const u8{
    null, // startpos
    "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkb1r/pppppppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
};

fn report(io: std.Io, allocator: std.mem.Allocator, w: *std.Io.Writer, path: []const u8) !void {
    const net = try nnue.Network.load(io, allocator, path);
    defer net.deinit(allocator);

    try w.print("\n=== {s} ===\n", .{path});
    try w.print("Layer weight statistics:\n", .{});
    try (LayerStats.of(i16, net.ft_biases[0..])).print(w, "ft_biases");
    try (LayerStats.of(i16, flat(i16, &net.ft_weights[0][0], nnue.num_features * nnue.ft_out))).print(w, "ft_weights");
    try (LayerStats.of(i8, flat(i8, &net.fc1_weights[0][0], nnue.fc1_out * nnue.fc1_in))).print(w, "fc1_weights");
    try (LayerStats.of(i32, net.fc1_biases[0..])).print(w, "fc1_biases");
    try (LayerStats.of(i8, flat(i8, &net.fc2_weights[0][0], nnue.fc2_out * nnue.fc2_in))).print(w, "fc2_weights");
    try (LayerStats.of(i32, net.fc2_biases[0..])).print(w, "fc2_biases");
    try (LayerStats.of(i16, net.output_weights[0..])).print(w, "output_weights");
    try w.print("  output_bias    = {d}\n", .{net.output_bias});

    // Accumulator CReLU occupancy across probe positions. The activation clamps
    // to [0,127]; lots of values <0 or >127 means saturated/dead neurons.
    var acc: nnue.Accumulator = undefined;
    var total: usize = 0;
    var in_range: usize = 0;
    var clipped_hi: usize = 0;
    var clipped_lo: usize = 0;
    var acc_min: i64 = std.math.maxInt(i64);
    var acc_max: i64 = std.math.minInt(i64);
    for (probe_fens) |maybe_fen| {
        const state = if (maybe_fen) |f| (State.fromFen(f) catch continue) else State.defaultPosition();
        nnue.refreshAccumulator(&state, net, &acc);
        for (acc.values) |perspective| {
            for (perspective) |v| {
                total += 1;
                const x: i64 = v;
                if (x < acc_min) acc_min = x;
                if (x > acc_max) acc_max = x;
                if (x < 0) clipped_lo += 1 else if (x > 127) clipped_hi += 1 else in_range += 1;
            }
        }
    }
    const tf: f64 = @floatFromInt(total);
    try w.print("Accumulator CReLU occupancy (over {d} probe positions, both perspectives):\n", .{probe_fens.len});
    try w.print("  range=[{d},{d}]  in[0,127]={d:.1}%  clipped<0={d:.1}%  clipped>127={d:.1}%\n", .{
        acc_min,                                                              acc_max,
        100.0 * @as(f64, @floatFromInt(in_range)) / tf,                       100.0 * @as(f64, @floatFromInt(clipped_lo)) / tf,
        100.0 * @as(f64, @floatFromInt(clipped_hi)) / tf,
    });

    // Eval spread on probe positions.
    try w.print("Evals (cp):\n", .{});
    var first: ?i32 = null;
    var differ = false;
    for (probe_fens) |maybe_fen| {
        const state = if (maybe_fen) |f| (State.fromFen(f) catch continue) else State.defaultPosition();
        const cp = nnue.evaluate(&state, net);
        if (first) |fv| {
            if (cp != fv) differ = true;
        } else first = cp;
        try w.print("  {s:<70} {d:>6}\n", .{ maybe_fen orelse "startpos", cp });
    }
    if (!differ) try w.print("  WARNING: all probe positions scored identically (net may be dead)\n", .{});
}

pub fn main(init: std.process.Init.Minimal) !void {
    const allocator = std.heap.page_allocator;
    const arg_parser = try kore.args.declarative.Parser(Args);
    var args_iter = if (builtin.os.tag == .windows)
        try init.args.iterateAllocator(allocator)
    else
        init.args.iterate();
    const args = try arg_parser.parse(&args_iter);

    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    var out_buf: [8192]u8 = undefined;
    var out_writer = std.Io.File.stdout().writer(io, &out_buf);
    const w: *std.Io.Writer = &out_writer.interface;

    try report(io, allocator, w, args.net);
    if (args.compare) |other| try report(io, allocator, w, other);
    try w.flush();
}
