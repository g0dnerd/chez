// Reusable debugger for trained .nnue nets. Reports per-layer weight statistics,
// quantization saturation, accumulator CReLU occupancy, and eval spread on probe
// positions. Optionally diffs two nets side by side (e.g. a new net vs the
// current champion).
//
//   zig build nnue-inspect -- --net data/net_v13_screlu.nnue
//   zig build nnue-inspect -- --net data/net_v13_screlu.nnue --compare data/net_v11_80M_wd0.03.nnue
//
// Unlike the engine (which loads a single format on purpose), this diagnostic
// tool accepts the current format AND the prior FT-512 single-head format
// (v4/HKP2, e.g. the v11 nets) via loadAnyFormat below, so older nets stay
// inspectable. Pre-FT512 (256-wide) nets are a different width and cannot be
// loaded by this binary.
//
// Caveat: eval-spread uses the engine forward pass, which is now SCReLU. Weight
// stats and accumulator occupancy are accurate for any loadable net, but the
// eval-spread numbers are only meaningful for current-format (SCReLU) nets — for
// a v4/v11 (CReLU-trained) net they reflect SCReLU applied to CReLU weights.
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
    // Optional file of positions (one FEN or EPD per line) for per-neuron FT
    // occupancy. Without it, occupancy is reported over the built-in probes only.
    positions: ?[]const u8,
    // Cap on positions to stream from --positions (default 100k).
    limit: ?usize,
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

// Inspector-only loader. The engine's Network.loadFromBytes is intentionally
// single-format (current bucketed v6/HKP4). This tool additionally reads two
// older formats:
//   - v5/HKP3: byte-identical layout to v6 (the v5→v6 change only swapped the FT
//     activation CReLU→SCReLU, an inference-time difference); we reuse the engine
//     loader by patching the version/arch header into a copy.
//   - v4/HKP2 (the v11 champion nets): prior FT-512 single-head format, parsed
//     into the current Network struct by broadcasting the one output head across
//     all buckets (faithful, since v11 used one head for every position). The FT
//     and FC sections are byte-identical to v5/v6 (same widths); only the output
//     section differs. This duplication is deliberate and lives only here.
const v4_arch_hash: u32 = 0x48_4B_50_32; // "HKP2"
const v5_arch_hash: u32 = 0x48_4B_50_33; // "HKP3"

fn loadAnyFormat(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !*nnue.Network {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    const stat = try file.stat(io);
    const len: usize = @intCast(stat.size);
    defer file.close(io);

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

    if (len < nnue.header_size) return error.InvalidFileSize;
    if (!std.mem.eql(u8, data[0..4], &nnue.magic_bytes)) return error.InvalidMagic;

    const ver = std.mem.readInt(u32, data[4..8], .little);
    if (ver == nnue.format_version) return nnue.Network.loadFromBytes(allocator, data);
    if (ver == 5) return loadV5Net(allocator, data);
    if (ver == 4) return loadV4Net(allocator, data);
    return error.UnsupportedVersion;
}

// Parse a v5/HKP3 net. Its on-disk layout is byte-identical to the current
// v6/HKP4 format, so we patch the version/arch header bytes into a copy and reuse
// the engine loader rather than duplicating the section parse here.
fn loadV5Net(allocator: std.mem.Allocator, data: []const u8) !*nnue.Network {
    const arch = std.mem.readInt(u32, data[8..12], .little);
    if (arch != v5_arch_hash) return error.ArchitectureMismatch;

    const buf = try allocator.alloc(u8, data.len);
    defer allocator.free(buf);
    @memcpy(buf, data);
    std.mem.writeInt(u32, buf[4..8], nnue.format_version, .little);
    std.mem.writeInt(u32, buf[8..12], nnue.arch_hash, .little);
    return nnue.Network.loadFromBytes(allocator, buf);
}

// Parse a v4/HKP2 (FT-512, single output head) net into the current Network.
fn loadV4Net(allocator: std.mem.Allocator, data: []const u8) !*nnue.Network {
    const ft_biases_bytes = nnue.ft_out * @sizeOf(i16);
    const ft_weights_bytes = nnue.num_features * nnue.ft_out * @sizeOf(i16);
    const fc1_weights_bytes = nnue.fc1_in * nnue.fc1_out * @sizeOf(i8);
    const fc1_biases_bytes = nnue.fc1_out * @sizeOf(i32);
    const fc2_weights_bytes = nnue.fc2_in * nnue.fc2_out * @sizeOf(i8);
    const fc2_biases_bytes = nnue.fc2_out * @sizeOf(i32);
    const v4_out_weights_bytes = nnue.fc2_out * @sizeOf(i16); // single head
    const v4_out_bias_bytes = @sizeOf(i32);
    const v4_size = nnue.header_size + ft_biases_bytes + ft_weights_bytes +
        fc1_weights_bytes + fc1_biases_bytes + fc2_weights_bytes + fc2_biases_bytes +
        v4_out_weights_bytes + v4_out_bias_bytes;

    if (data.len < v4_size) return error.InvalidFileSize;
    const arch = std.mem.readInt(u32, data[8..12], .little);
    if (arch != v4_arch_hash) return error.ArchitectureMismatch;

    const net = try allocator.create(nnue.Network);
    errdefer allocator.destroy(net);

    var off: usize = nnue.header_size;
    @memcpy(std.mem.asBytes(&net.ft_biases), data[off..][0..ft_biases_bytes]);
    off += ft_biases_bytes;
    @memcpy(std.mem.asBytes(&net.ft_weights), data[off..][0..ft_weights_bytes]);
    off += ft_weights_bytes;
    @memcpy(std.mem.asBytes(&net.fc1_weights), data[off..][0..fc1_weights_bytes]);
    off += fc1_weights_bytes;
    @memcpy(std.mem.asBytes(&net.fc1_biases), data[off..][0..fc1_biases_bytes]);
    off += fc1_biases_bytes;
    @memcpy(std.mem.asBytes(&net.fc2_weights), data[off..][0..fc2_weights_bytes]);
    off += fc2_weights_bytes;
    @memcpy(std.mem.asBytes(&net.fc2_biases), data[off..][0..fc2_biases_bytes]);
    off += fc2_biases_bytes;

    // Single output head → broadcast across all buckets (read element-wise to
    // avoid any alignment assumptions on the mapped buffer).
    var head_weights: [nnue.fc2_out]i16 = undefined;
    for (0..nnue.fc2_out) |i| {
        head_weights[i] = std.mem.readInt(i16, data[off + i * 2 ..][0..2], .little);
    }
    off += v4_out_weights_bytes;
    const head_bias = std.mem.readInt(i32, data[off..][0..4], .little);

    for (0..nnue.num_output_buckets) |b| {
        net.output_weights[b] = head_weights;
        net.output_bias[b] = head_bias;
    }
    return net;
}

// Per-output-bucket breakdown. Each of the 8 heads is a [fc2_out]i16 weight row
// plus an i32 bias, selected at inference by piece count. Trained v5 nets should
// differentiate the heads (so endgames and openings get distinct output scaling);
// if every head is identical the bucketing learned nothing. NOTE: v4/v11 nets are
// single-head and get broadcast across all buckets by loadV4Net, so "identical"
// is expected for them, not a defect.
fn perBucketReport(w: *std.Io.Writer, net: *const nnue.Network) !void {
    try w.print("Output buckets ({d} heads, selected by piece count = clamp((popcount-1)/4, 0, 7)):\n", .{nnue.num_output_buckets});
    try w.print("  {s:>6}  {s:>10}  {s:>9}  {s:>8}  {s:>6}  {s:>6}  {s:>6}\n", .{ "bucket", "bias", "w_mean", "w_std", "w_min", "w_max", "nz" });
    var same_as_0: usize = 0; // among buckets 1..N-1
    for (0..nnue.num_output_buckets) |b| {
        const row = net.output_weights[b][0..];
        const s = LayerStats.of(i16, row);
        const pct_nz = 100.0 * @as(f64, @floatFromInt(s.nonzero)) / @as(f64, @floatFromInt(s.n));
        try w.print("  {d:>6}  {d:>10}  {d:>9.3}  {d:>8.3}  {d:>6}  {d:>6}  {d:>5.1}%\n", .{ b, net.output_bias[b], s.mean, s.std, s.min, s.max, pct_nz });
        if (b > 0 and net.output_bias[b] == net.output_bias[0] and std.mem.eql(i16, row, net.output_weights[0][0..])) {
            same_as_0 += 1;
        }
    }
    if (same_as_0 == nnue.num_output_buckets - 1) {
        try w.print("  NOTE: all {d} heads are identical (single-head v4 net, or bucketing collapsed in training)\n", .{nnue.num_output_buckets});
    } else {
        try w.print("  heads differing from bucket 0: {d}/{d}\n", .{ nnue.num_output_buckets - 1 - same_as_0, nnue.num_output_buckets - 1 });
    }
}

// Extract the 4-field FEN core (board, stm, castling, ep) from a FEN or EPD line
// into buf, dropping move counters and EPD operations. Returns null if malformed.
fn extractFen(line: []const u8, buf: []u8) ?[]const u8 {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    var n: usize = 0;
    var len: usize = 0;
    while (it.next()) |tok| {
        if (n >= 4) break;
        if (n > 0) {
            if (len >= buf.len) return null;
            buf[len] = ' ';
            len += 1;
        }
        if (len + tok.len > buf.len) return null;
        @memcpy(buf[len..][0..tok.len], tok);
        len += tok.len;
        n += 1;
    }
    if (n < 4) return null;
    return buf[0..len];
}

// Stream positions from `data` (one FEN/EPD per line), refresh the accumulator
// for each, and aggregate per-FT-neuron activation statistics. A neuron "fires"
// when its raw accumulator value is > 0 (CReLU passes it); it saturates high when
// > 127. Reports how many neurons are dead/rare/always-saturated plus a fire-rate
// histogram — distinguishing globally-dead capacity from sparse position-specific
// coding, which the few built-in probes cannot.
fn neuronOccupancy(w: *std.Io.Writer, net: *const nnue.Network, data: []const u8, limit: usize) !void {
    var fire = [_]u64{0} ** nnue.ft_out;
    var high = [_]u64{0} ** nnue.ft_out;

    var acc: nnue.Accumulator = undefined;
    var positions: usize = 0;
    var skipped: usize = 0;

    var lines = std.mem.tokenizeScalar(u8, data, '\n');
    while (lines.next()) |raw_line| {
        if (positions >= limit) break;
        const line = std.mem.trim(u8, raw_line, " \r\t");
        if (line.len == 0) continue;
        var fen_buf: [128]u8 = undefined;
        const fen = extractFen(line, &fen_buf) orelse {
            skipped += 1;
            continue;
        };
        const state = State.fromFen(fen) catch {
            skipped += 1;
            continue;
        };
        nnue.refreshAccumulator(&state, net, &acc);
        for (acc.values) |perspective| {
            for (perspective, 0..) |v, j| {
                if (v > 0) fire[j] += 1;
                if (v > 127) high[j] += 1;
            }
        }
        positions += 1;
    }

    if (positions == 0) {
        try w.print("Per-neuron FT occupancy: no parseable positions found\n", .{});
        return;
    }

    const samples: u64 = @as(u64, positions) * 2; // both perspectives per position
    var dead: usize = 0;
    var rare: usize = 0;
    var always_high: usize = 0;
    var bands = [_]usize{0} ** 7;
    var rate_sum: f64 = 0;
    for (0..nnue.ft_out) |j| {
        const rate = @as(f64, @floatFromInt(fire[j])) / @as(f64, @floatFromInt(samples));
        rate_sum += rate;
        if (fire[j] == 0) dead += 1;
        if (rate < 0.01) rare += 1;
        if (high[j] == samples) always_high += 1;
        const band: usize = if (rate == 0)
            0
        else if (rate <= 0.01)
            1
        else if (rate <= 0.10)
            2
        else if (rate <= 0.25)
            3
        else if (rate <= 0.50)
            4
        else if (rate <= 0.75)
            5
        else
            6;
        bands[band] += 1;
    }

    try w.print("Per-neuron FT occupancy over {d} positions ({d} samples/neuron, both perspectives", .{ positions, samples });
    if (skipped > 0) try w.print(", {d} lines skipped", .{skipped});
    try w.print("):\n", .{});
    try w.print("  dead (never fire > 0):        {d:>4} / {d}\n", .{ dead, nnue.ft_out });
    try w.print("  rarely fire (< 1% of pos):    {d:>4} / {d}\n", .{ rare, nnue.ft_out });
    try w.print("  always saturate high (>127):  {d:>4} / {d}\n", .{ always_high, nnue.ft_out });
    try w.print("  mean per-neuron fire-rate:    {d:.1}%\n", .{100.0 * rate_sum / @as(f64, nnue.ft_out)});

    const labels = [_][]const u8{ "      0%", "  (0,1%]", " (1,10%]", "(10,25%]", "(25,50%]", "(50,75%]", "(75,100%]" };
    var maxb: usize = 1;
    for (bands) |c| {
        if (c > maxb) maxb = c;
    }
    try w.print("  fire-rate histogram (neurons per band):\n", .{});
    for (labels, bands) |lab, c| {
        const barlen = (c * 40) / maxb;
        var bar: [40]u8 = undefined;
        for (0..barlen) |k| bar[k] = '#';
        try w.print("    {s:<9} {s:<40} {d}\n", .{ lab, bar[0..barlen], c });
    }
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

fn report(io: std.Io, allocator: std.mem.Allocator, w: *std.Io.Writer, path: []const u8, positions: ?[]const u8, limit: usize) !void {
    const net = try loadAnyFormat(io, allocator, path);
    defer net.deinit(allocator);

    try w.print("\n=== {s} ===\n", .{path});
    try w.print("Layer weight statistics:\n", .{});
    try (LayerStats.of(i16, net.ft_biases[0..])).print(w, "ft_biases");
    try (LayerStats.of(i16, flat(i16, &net.ft_weights[0][0], nnue.num_features * nnue.ft_out))).print(w, "ft_weights");
    try (LayerStats.of(i8, flat(i8, &net.fc1_weights[0][0], nnue.fc1_out * nnue.fc1_in))).print(w, "fc1_weights");
    try (LayerStats.of(i32, net.fc1_biases[0..])).print(w, "fc1_biases");
    try (LayerStats.of(i8, flat(i8, &net.fc2_weights[0][0], nnue.fc2_out * nnue.fc2_in))).print(w, "fc2_weights");
    try (LayerStats.of(i32, net.fc2_biases[0..])).print(w, "fc2_biases");
    try (LayerStats.of(i16, flat(i16, &net.output_weights[0][0], nnue.num_output_buckets * nnue.fc2_out))).print(w, "output_weights");
    try w.print("  output_bias    = {any}\n", .{net.output_bias});

    try perBucketReport(w, net);

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

    // Per-neuron occupancy over a large position file (optional).
    if (positions) |pos_path| {
        if (std.Io.Dir.cwd().openFile(io, pos_path, .{})) |file| {
            defer file.close(io);
            const stat = try file.stat(io);
            const len: usize = @intCast(stat.size);
            if (len > 0) {
                const ptr = try std.posix.mmap(
                    null,
                    len,
                    std.os.linux.PROT{ .READ = true },
                    .{ .TYPE = .SHARED },
                    file.handle,
                    0,
                );
                defer std.posix.munmap(@alignCast(ptr));
                try neuronOccupancy(w, net, ptr[0..len], limit);
            }
        } else |e| {
            try w.print("Per-neuron FT occupancy: could not open '{s}': {s}\n", .{ pos_path, @errorName(e) });
        }
    }

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

    const limit = args.limit orelse 100_000;
    try report(io, allocator, w, args.net, args.positions, limit);
    if (args.compare) |other| try report(io, allocator, w, other, args.positions, limit);
    try w.flush();
}
