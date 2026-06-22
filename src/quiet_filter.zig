// Reads FEN positions from an input file and outputs only "quiet" positions
// where the static eval and quiescence eval agree within a threshold.
//
// Usage:
//   zig-out/bin/quiet-filter --input_path positions.fen [--threshold 100]
//
// Quiet criterion: |evaluate(pos) - quiescenceEval(pos)| < threshold
// Positions where captures/tactics change the eval significantly are discarded.

const builtin = @import("builtin");
const std = @import("std");
const chez = @import("chez");
const kore = @import("kore");

const State = chez.engine.State;
const evaluation = chez.engine.evaluation;
const search = chez.engine.search;

const default_threshold: i32 = 100;

const Args = struct {
    input_path: []const u8,
    threshold: ?u32,
};

pub fn main(init: std.process.Init.Minimal) !void {
    const arg_parser = try kore.args.declarative.Parser(Args);

    var args_iter = if (builtin.os.tag == .windows)
        try init.args.iterateAllocator(std.heap.page_allocator)
    else
        init.args.iterate();
    const parsed_args = try arg_parser.parse(&args_iter);

    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .environ = .empty });
    const io = threaded.io();

    var stdout_buffer: [65536]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    var stderr_buffer: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr: *std.Io.Writer = &stderr_writer.interface;

    const path = parsed_args.input_path;
    const threshold = parsed_args.threshold orelse default_threshold;

    // Open input file
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| {
        try stderr.print("error: cannot open '{s}': {}\n", .{ path, err });
        return err;
    };
    defer file.close(io);

    var reader_buf: [65536]u8 = undefined;
    var reader = file.reader(io, &reader_buf);

    var total: usize = 0;
    var kept: usize = 0;
    var parse_errors: usize = 0;

    while (true) {
        const line_raw = reader.interface.takeDelimiterExclusive('\n') catch break;
        reader.interface.toss(1);

        const line = std.mem.trim(u8, line_raw, &std.ascii.whitespace);
        if (line.len == 0) continue;

        total += 1;

        const state = State.fromFen(line) catch {
            parse_errors += 1;
            continue;
        };

        const static_eval = evaluation.evaluate(&state);
        const qsearch_eval = search.quiescenceEval(&state);

        const diff = if (static_eval > qsearch_eval)
            static_eval - qsearch_eval
        else
            qsearch_eval - static_eval;

        if (diff < threshold) {
            try stdout.print("{s}\n", .{line});
            kept += 1;
        }

        const pct =
            if (total > 0)
                @as(f64, @floatFromInt(kept)) / @as(f64, @floatFromInt(total)) * 100.0
            else
                0.0;
        if (total % 100_000 == 0) {
            try stderr.print("quiet-filter: processed {d}, kept {d} ({d:.1}%)\n", .{
                total,
                kept,
                pct,
            });
        }
    }

    try stdout_writer.flush();

    try stderr.print("quiet-filter: done. total={d} kept={d} filtered={d} parse_errors={d}\n", .{
        total,
        kept,
        total - kept - parse_errors,
        parse_errors,
    });
}
