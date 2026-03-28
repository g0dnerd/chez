// Regenerates src/engine/params.zig from a tuned Params value.
//
// Reads the existing params.zig, splices out the `pub const default_params`
// block, emits a freshly generated block with the tuned i16 values,
// and writes the result to a .tmp file before atomically renaming.
// This keeps the Params/ParamsF64 struct definitions and helper functions
// verbatim, so the file remains self-contained and hand-editable after tuning.

const std = @import("std");
const chez = @import("chez");

const params_mod = chez.engine.params;
const Params = params_mod.Params;
const Score = params_mod.Score;

const PARAM_COUNT = params_mod.PARAM_COUNT;

// Marker that starts the block we replace.
const start_marker = "pub const default_params: Params = .{";

pub fn write(
    io: std.Io,
    allocator: std.mem.Allocator,
    p: *const Params,
    output_path: []const u8,
) !void {
    const source = try readFile(io, allocator, output_path);
    defer allocator.free(source);

    const block_start = std.mem.indexOf(u8, source, start_marker) orelse
        return error.DefaultParamsMarkerNotFound;

    const block_end = findBlockEnd(source, block_start) orelse
        return error.UnmatchedBraces;

    const tmp_path = try std.mem.concat(allocator, u8, &.{ output_path, ".tmp" });
    defer allocator.free(tmp_path);

    const out_file = try std.Io.Dir.cwd().createFile(io, tmp_path, .{});

    var writer_buf: [131072]u8 = undefined;
    var writer = out_file.writer(io, &writer_buf);
    const w: *std.Io.Writer = &writer.interface;

    try w.writeAll(source[0..block_start]);

    try emitDefaultParams(w, p);

    try w.writeAll(source[block_end..]);
    try w.flush();

    out_file.close(io);

    const cwd = std.Io.Dir.cwd();
    try std.Io.Dir.rename(cwd, tmp_path, cwd, output_path, io);
}

fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const stat = try file.stat(io);
    const len: usize = @intCast(stat.size);
    const buf = try allocator.alloc(u8, len);
    errdefer allocator.free(buf);

    // Read the entire file using posix read in a loop.
    var total_read: usize = 0;
    while (total_read < len) {
        const n = try std.posix.read(file.handle, buf[total_read..]);
        if (n == 0) break;
        total_read += n;
    }

    return buf[0..total_read];
}

// Returns the index just after the `;` that closes the block (so the suffix
// starts there). Returns null if the braces are unmatched.
fn findBlockEnd(source: []const u8, block_start: usize) ?usize {
    var i = block_start;

    // Advance to the first `{` which opens the struct literal.
    while (i < source.len and source[i] != '{') : (i += 1) {}
    if (i >= source.len) return null;

    var depth: usize = 1;
    i += 1; // consume the opening `{`

    while (i < source.len) : (i += 1) {
        switch (source[i]) {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if (depth == 0) {
                    // Consume '}', then skip whitespace and the mandatory ';'.
                    i += 1;
                    while (i < source.len and source[i] != ';') : (i += 1) {}
                    if (i >= source.len) return null;
                    return i + 1; // include the ';'
                }
            },
            else => {},
        }
    }

    return null;
}

fn emitDefaultParams(w: *std.Io.Writer, p: *const Params) !void {
    try w.writeAll("pub const default_params: Params = .{\n");

    // ── piece_values ────────────────────────────────────────────────────────
    try w.writeAll("    .piece_values = .{\n");
    const piece_names = [6][]const u8{ "pawn", "knight", "bishop", "rook", "queen", "king (frozen)" };
    for (p.piece_values, 0..) |s, i| {
        try w.print("        Score.init({d}, {d}), // {s}\n", .{ s.midgame(), s.endgame(), piece_names[i] });
    }
    try w.writeAll("    },\n");

    // ── passed_pawn_bonus ────────────────────────────────────────────────────
    try w.writeAll("    .passed_pawn_bonus = .{\n");
    const ppb_comments = [8][]const u8{
        "rank 0 (frozen)",
        "rank 1",
        "rank 2",
        "rank 3",
        "rank 4",
        "rank 5",
        "rank 6",
        "rank 7 (frozen)",
    };
    for (p.passed_pawn_bonus, 0..) |s, i| {
        try w.print("        Score.init({d}, {d}), // {s}\n", .{ s.midgame(), s.endgame(), ppb_comments[i] });
    }
    try w.writeAll("    },\n");

    // ── mobility_bonus ───────────────────────────────────────────────────────
    try w.writeAll("    .mobility_bonus = .{\n");
    const mob_names = [4][]const u8{ "Knights (max 8)", "Bishops (max 13)", "Rooks (max 14)", "Queens (max 27)" };
    for (p.mobility_bonus, 0..) |piece_mob, pi| {
        try w.print("        // {s}\n", .{mob_names[pi]});
        try w.writeAll("        .{\n");
        for (piece_mob) |s| {
            try w.print("            Score.init({d}, {d}),\n", .{ s.midgame(), s.endgame() });
        }
        try w.writeAll("        },\n");
    }
    try w.writeAll("    },\n");

    // ── 17 named Score scalars ───────────────────────────────────────────────
    try emitScore(w, "bishop_pair", p.bishop_pair);
    try emitScore(w, "rook_open_file", p.rook_open_file);
    try emitScore(w, "rook_semi_open", p.rook_semi_open);
    try emitScore(w, "rook_on_seventh", p.rook_on_seventh);
    try emitScore(w, "isolated_pawn", p.isolated_pawn);
    try emitScore(w, "doubled_pawn", p.doubled_pawn);
    try emitScore(w, "backward_pawn", p.backward_pawn);
    try emitScore(w, "connected_pawn", p.connected_pawn);
    try emitScore(w, "protected_passed_pawn", p.protected_passed_pawn);
    try emitScore(w, "blocked_passed_pawn", p.blocked_passed_pawn);
    try emitScore(w, "rook_behind_passer", p.rook_behind_passer);
    try emitScore(w, "free_passed_pawn", p.free_passed_pawn);
    try emitScore(w, "knight_outpost_defended", p.knight_outpost_defended);
    try emitScore(w, "bishop_outpost_defended", p.bishop_outpost_defended);
    try emitScore(w, "pawn_shield", p.pawn_shield);
    try emitScore(w, "pawn_shield_missing", p.pawn_shield_missing);
    try emitScore(w, "tempo", p.tempo);

    // ── king_proximity_passer ────────────────────────────────────────────────
    try w.print("    .king_proximity_passer = {d},\n", .{p.king_proximity_passer});

    // ── pst ─────────────────────────────────────────────────────────────────
    // Emit as a direct [6][64]Score array literal with 8 entries per line
    // (one rank per line). zig fmt will reflow if needed.
    try w.writeAll("    .pst = .{\n");
    const pst_piece_names = [6][]const u8{
        "Pawns (rank 0 and 7 frozen at 0)",
        "Knights",
        "Bishops",
        "Rooks",
        "Queens",
        "Kings",
    };
    for (p.pst, 0..) |piece_pst, pi| {
        try w.print("        // {s}\n", .{pst_piece_names[pi]});
        try w.writeAll("        .{\n");
        for (0..8) |rank| {
            try w.writeAll("            ");
            for (0..8) |file| {
                const sq = rank * 8 + file;
                const s = piece_pst[sq];
                try w.print("Score.init({d},{d}),", .{ s.midgame(), s.endgame() });
                if (file < 7) try w.writeByte(' ');
            }
            try w.writeByte('\n');
        }
        try w.writeAll("        },\n");
    }
    try w.writeAll("    },\n");

    try w.writeAll("};\n");
}

fn emitScore(w: *std.Io.Writer, name: []const u8, s: Score) !void {
    try w.print("    .{s} = Score.init({d}, {d}),\n", .{ name, s.midgame(), s.endgame() });
}
