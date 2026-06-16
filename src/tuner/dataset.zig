// Loads EPD positions into a flat Position slice for Texel tuning.
//
// Supports two EPD annotation formats:
//
// 1. Game outcome (Zurichess quiet-labeled):
//    rnbqkb1r/... w KQkq - 2 3 c9 "1/2-1/2";
//    Results: "1-0" = white wins, "1/2-1/2" = draw, "0-1" = black wins.
//
// 2. Centipawn evaluation (Stockfish-labeled):
//    rnbqkb1r/... w KQkq - ce "150";
//    The integer is Stockfish's eval from white's perspective.
//    Converted to [0,1] via sigmoid: 1 / (1 + exp(-cp / 400)).
//
// Both formats are auto-detected per line. Results are stored in side-to-move
// perspective: if black to move, result = 1.0 - raw.

const std = @import("std");
const chez = @import("chez");
const State = chez.engine.State;
const Colors = chez.engine.Colors;

pub const Position = struct {
    state: State,
    result: f32, // side-to-move perspective: 1.0=win, 0.5=draw, 0.0=loss
};

pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    max_positions: usize,
) ![]Position {
    // Open the dataset file. The path is resolved relative to the current
    // working directory (i.e., the project root when invoked via `zig build`).
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    var positions: std.ArrayListUnmanaged(Position) = .{ .items = &.{}, .capacity = 0 };
    try positions.ensureTotalCapacity(allocator, @min(max_positions, 1_000_000));
    errdefer positions.deinit(allocator);

    // Use a 64 KiB reader buffer so the kernel can deliver large reads; the
    // tuner dataset may be several hundred MB.
    var reader_buf: [65536]u8 = undefined;
    var reader = file.reader(io, &reader_buf);

    var lines_parsed: usize = 0;
    var lines_skipped: usize = 0;

    // Read line by line until EOF or position cap.
    while (positions.items.len < max_positions) {
        // takeDelimiterExclusive returns the data before the delimiter and
        // leaves the reader positioned at the delimiter itself.
        const line_raw = reader.interface.takeDelimiterExclusive('\n') catch break;
        reader.interface.toss(1); // discard the '\n'

        const line = std.mem.trim(u8, line_raw, &std.ascii.whitespace);
        if (line.len == 0) continue;

        const pos = parseLine(line) catch {
            lines_skipped += 1;
            if (lines_skipped <= 10) {
                std.debug.print("dataset: skipping malformed line {d}: {s}\n", .{ lines_parsed + lines_skipped, line[0..@min(line.len, 80)] });
            }
            continue;
        };

        try positions.append(allocator, pos);
        lines_parsed += 1;
    }

    std.debug.print("dataset: loaded {d} positions ({d} skipped)\n", .{ positions.items.len, lines_skipped });
    return positions.toOwnedSlice(allocator);
}

// Parses one EPD line into a Position. Auto-detects c9 (game outcome) vs ce
// (centipawn eval) annotation format.
fn parseLine(line: []const u8) !Position {
    const c9_marker = " c9 \"";
    const ce_marker = " ce \"";

    if (std.mem.indexOf(u8, line, ce_marker)) |ce_pos| {
        return parseCeLine(line, ce_pos, ce_marker.len);
    } else if (std.mem.indexOf(u8, line, c9_marker)) |c9_pos| {
        return parseC9Line(line, c9_pos, c9_marker.len);
    } else {
        return error.MissingAnnotation;
    }
}

// Parse game-outcome format: <FEN> c9 "<result>";
fn parseC9Line(line: []const u8, marker_pos: usize, marker_len: usize) !Position {
    const fen_str = std.mem.trimEnd(u8, line[0..marker_pos], " \t");
    if (fen_str.len == 0) return error.EmptyFen;

    const after = line[marker_pos + marker_len ..];
    const quote_end = std.mem.indexOfScalar(u8, after, '"') orelse return error.UnclosedQuote;
    const result_str = after[0..quote_end];

    const raw: f32 = if (std.mem.eql(u8, result_str, "1-0"))
        1.0
    else if (std.mem.eql(u8, result_str, "1/2-1/2"))
        0.5
    else if (std.mem.eql(u8, result_str, "0-1"))
        0.0
    else
        return error.UnknownResult;

    const state = try State.fromFen(fen_str);
    const result: f32 = if (state.to_move == Colors.black) 1.0 - raw else raw;
    return Position{ .state = state, .result = result };
}

// Parse centipawn-eval format: <FEN> ce "<centipawns>";
// Converts centipawn value to [0,1] via sigmoid: 1 / (1 + exp(-cp / 400))
fn parseCeLine(line: []const u8, marker_pos: usize, marker_len: usize) !Position {
    const fen_str = std.mem.trimEnd(u8, line[0..marker_pos], " \t");
    if (fen_str.len == 0) return error.EmptyFen;

    const after = line[marker_pos + marker_len ..];
    const quote_end = std.mem.indexOfScalar(u8, after, '"') orelse return error.UnclosedQuote;
    const cp_str = after[0..quote_end];

    const cp = std.fmt.parseInt(i32, cp_str, 10) catch return error.InvalidCentipawn;

    // Sigmoid mapping: 1 / (1 + exp(-cp / 400))
    const raw: f32 = 1.0 / (1.0 + @exp(-@as(f32, @floatFromInt(cp)) / 400.0));

    const state = try State.fromFen(fen_str);
    // ce values are from white's perspective; convert to side-to-move
    const result: f32 = if (state.to_move == Colors.black) 1.0 - raw else raw;
    return Position{ .state = state, .result = result };
}
