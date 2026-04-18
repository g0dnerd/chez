const std = @import("std");
const chez = @import("chez.zig");
const nnue = chez.engine.nnue;
const search = chez.engine.search;
const State = chez.engine.State;

const positions = [_][]const u8{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 6 5",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QR1K1 w - - 4 8",
    "rnbqr1k1/pp3pbp/3p1np1/2pP4/4P3/2N2N2/PP2BPPP/R1BQK2R w KQ - 1 9",
    "2rr2k1/pp3ppp/2n1bn2/4N3/1bP5/1P3NP1/PB2PPBP/3RR1K1 w - - 3 16",
    "r1bq1rk1/pp2ppbp/2n2np1/2pp4/8/2NPBNP1/PPP1PPBP/R2Q1RK1 w - - 0 8",
    "r2qkb1r/ppp2ppp/2n1bn2/3pp3/4P3/1BN2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5",
    "1r4k1/3b1ppp/1q2p3/3pP3/pp1N4/4QN2/PP3PPP/1R4K1 w - - 0 24",
    "8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 w - - 2 47",
    "8/1p4k1/p7/5R2/8/1P2r3/P5K1/8 w - - 0 40",
    "8/5pk1/7p/3p1R2/p1p5/P1P2P2/1P3K2/8 w - - 1 42",
    "3r2k1/pp3ppp/4b3/8/3N4/P4P2/1PP3PP/3R2K1 w - - 0 22",
    "r2qkb1r/pp2nppp/2n1p3/3pPb2/3P4/2N2N2/PPP1BPPP/R1BQ1RK1 w kq - 2 8",
    "2kr3r/pp3ppp/2nbbn2/3p4/3P4/2NBBN2/PP3PPP/2KR3R w - - 8 14",
};

const depth_default: u8 = 16;
const threads_default: usize = 1;

pub fn main(init: std.process.Init.Minimal) !void {
    var depth: u8 = depth_default;
    var threads: usize = threads_default;
    var nnue_path: ?[]const u8 = null;

    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .environ = .empty });
    const io = threaded.io();
    var stdout_buffer: [512]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    var args = try init.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip(); // program name

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--depth")) {
            if (args.next()) |d| {
                depth = std.fmt.parseInt(u8, d, 10) catch depth_default;
            }
        } else if (std.mem.eql(u8, arg, "--threads")) {
            if (args.next()) |t| {
                threads = std.fmt.parseInt(usize, t, 10) catch threads_default;
            }
        } else if (std.mem.eql(u8, arg, "--nnue")) {
            nnue_path = args.next();
        }
    }

    var network: ?*nnue.Network = null;
    defer if (network) |n| n.deinit(std.heap.page_allocator);

    if (nnue_path) |path| {
        network = nnue.Network.load(io, std.heap.page_allocator, path) catch |err| blk: {
            try stdout.print("Failed to load NNUE file: {}\n", .{err});
            try stdout.flush();
            break :blk null;
        };
    }

    const eval_label: []const u8 = if (network != null) "NNUE" else "HCE";
    try stdout.print("Bench: {d} positions, depth {d}, {d} thread(s), eval={s}\n", .{ positions.len, depth, threads, eval_label });

    const clock = std.Io.Clock.awake;

    // Micro-bench the evaluation function directly.
    {
        const micro_state = State.fromFen(positions[0]) catch unreachable;
        const iters: usize = 50_000;
        var sink: i64 = 0;
        const eval_start = std.Io.Timestamp.now(io, clock);
        if (network) |net| {
            for (0..iters) |_| sink +|= nnue.evaluate(&micro_state, net);
        } else {
            for (0..iters) |_| sink +|= chez.engine.evaluation.evaluate(&micro_state);
        }
        const eval_ns = std.Io.Timestamp.now(io, clock).nanoseconds - eval_start.nanoseconds;
        const per_eval_ns = @divTrunc(eval_ns, @as(i128, @intCast(iters)));
        const evals_per_sec = if (eval_ns > 0) @divTrunc(@as(i128, @intCast(iters)) * 1_000_000_000, eval_ns) else 0;
        try stdout.print("Eval microbench: {d} iters, {d}ns/eval, {d} evals/sec (sink={d})\n", .{ iters, per_eval_ns, evals_per_sec, sink });
        try stdout.flush();
    }

    var tbl = try search.TranspositionTable.init(std.heap.page_allocator);
    defer tbl.deinit();

    var start = std.Io.Timestamp.now(io, clock);

    try stdout.flush();
    for (positions, 1..) |fen, i| {
        const state = State.fromFen(fen) catch {
            try stdout.print("  #{d}: INVALID FEN\n", .{i});
            continue;
        };

        const result = try search.searchParallel(&state, depth, threads, null, &tbl, .{}, network);
        if (result) |r| {
            var start_buf: [2]u8 = undefined;
            var end_buf: [2]u8 = undefined;
            try chez.engine.square.toAlgebraic(r.move.start, &start_buf);
            try chez.engine.square.toAlgebraic(r.move.end, &end_buf);
            const cp_score = chez.engine.evaluation.toCentipawns(r.score);
            try stdout.print("  #{d}: {s}{s} score={d:.2} depth={d}\n", .{ i, start_buf, end_buf, cp_score, r.depth });
            try stdout.flush();
        } else {
            try stdout.print("  #{d}: no result\n", .{i});
            try stdout.flush();
        }
        tbl.newSearch();
    }

    const elapsed = start.untilNow(io, clock);
    const elapsed_ms = elapsed.toMilliseconds();
    const elapsed_s: f64 = @as(f64, @floatFromInt(elapsed_ms)) / @as(f64, @floatFromInt(std.time.ms_per_s));

    try stdout.print("\n===========================\n", .{});
    try stdout.print("Total time: {d}ms ({d:.2}s)\n", .{ elapsed_ms, elapsed_s });
    try stdout.print("Positions:  {d}\n", .{positions.len});
    try stdout.print("Depth:      {d}\n", .{depth});
    try stdout.print("Threads:    {d}\n", .{threads});
    try stdout.print("===========================\n", .{});
    try stdout.flush();
}
