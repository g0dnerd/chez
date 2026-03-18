// src/quiet_filter.zig
//
// Reads FEN positions from an input file and outputs only "quiet" positions
// where the static eval and quiescence eval agree within a threshold.
//
// Usage:
//   zig-out/bin/quiet-filter --input positions.fen [--threshold 100]
//
// Quiet criterion: |evaluate(pos) - quiescenceEval(pos)| < threshold
// Positions where captures/tactics change the eval significantly are discarded.

const std = @import("std");
const chez = @import("chez");
const State = chez.engine.State;
const evaluation = chez.engine.evaluation;
const search = chez.engine.search;

const default_threshold: i32 = 100;

pub fn main(init: std.process.Init.Minimal) !void {
    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .environ = .empty });
    const io = threaded.io();
    var stdout_buffer: [65536]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    var stderr_buffer: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr: *std.Io.Writer = &stderr_writer.interface;

    // Parse CLI args
    var input_path: ?[]const u8 = null;
    var threshold: i32 = default_threshold;

    var args = try init.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip(); // program name

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--input")) {
            input_path = args.next() orelse {
                stderr.print("error: --input requires a path argument\n", .{}) catch {};
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--threshold")) {
            const val = args.next() orelse {
                stderr.print("error: --threshold requires a numeric argument\n", .{}) catch {};
                std.process.exit(1);
            };
            threshold = std.fmt.parseInt(i32, val, 10) catch {
                stderr.print("error: invalid threshold: {s}\n", .{val}) catch {};
                std.process.exit(1);
            };
        }
    }

    const path = input_path orelse {
        stderr.print("usage: quiet-filter --input <fen-file> [--threshold <int>]\n", .{}) catch {};
        std.process.exit(1);
    };

    stderr.print("quiet-filter: threshold={d}, input={s}\n", .{ threshold, path }) catch {};

    // Open input file
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| {
        stderr.print("error: cannot open '{s}': {}\n", .{ path, err }) catch {};
        std.process.exit(1);
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
            stdout.print("{s}\n", .{line}) catch {};
            kept += 1;
        }

        if (total % 100_000 == 0) {
            stderr.print("quiet-filter: processed {d}, kept {d} ({d:.1}%)\n", .{
                total,
                kept,
                if (total > 0) @as(f64, @floatFromInt(kept)) / @as(f64, @floatFromInt(total)) * 100.0 else 0.0,
            }) catch {};
        }
    }

    // Flush stdout
    stdout_writer.flush() catch {};

    stderr.print("quiet-filter: done. total={d} kept={d} filtered={d} parse_errors={d}\n", .{
        total,
        kept,
        total - kept - parse_errors,
        parse_errors,
    }) catch {};
}
