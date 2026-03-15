// src/tuner/dataset.zig
//
// Loads quiet-labeled EPD positions into a flat Position slice for Texel tuning.
//
// Expected line format (Zurichess quiet-labeled EPD):
//
//   rnbqkb1r/... w KQkq - 2 3 c9 "0.5";
//
// The FEN is everything before the `c9` annotation.
// Results: "1.0" = white wins, "0.5" = draw, "0.0" = black wins.
// Stored in side-to-move perspective: if black to move, result = 1.0 - raw.

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

// ==============================================================================
// Line parsing
// ==============================================================================

// Parses one EPD line into a Position.
//
// Expected format: <FEN> c9 "<result>";
// The FEN may contain 4 or 6 space-separated fields (EPD or full FEN).
fn parseLine(line: []const u8) !Position {
    // Locate the c9 annotation to split FEN from result.
    const c9_marker = " c9 \"";
    const c9_pos = std.mem.indexOf(u8, line, c9_marker) orelse return error.MissingC9;

    const fen_str = std.mem.trimEnd(u8, line[0..c9_pos], " \t");
    if (fen_str.len == 0) return error.EmptyFen;

    // Extract the quoted result value after "c9 \"".
    const after_c9 = line[c9_pos + c9_marker.len ..];
    const quote_end = std.mem.indexOfScalar(u8, after_c9, '"') orelse return error.UnclosedQuote;
    const result_str = after_c9[0..quote_end];

    // Parse the raw result: "1.0" = white wins, "0.5" = draw, "0.0" = black wins.
    const raw: f32 = if (std.mem.eql(u8, result_str, "1-0"))
        1.0
    else if (std.mem.eql(u8, result_str, "1/2-1/2"))
        0.5
    else if (std.mem.eql(u8, result_str, "0-1"))
        0.0
    else
        return error.UnknownResult;

    // Parse the FEN into a State, fail hard on invalid FEN.
    const state = try State.fromFen(fen_str);

    // Convert to side-to-move perspective.
    const result: f32 = if (state.to_move == Colors.black) 1.0 - raw else raw;

    return Position{ .state = state, .result = result };
}
