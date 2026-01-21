const std = @import("std");
const Bitboard = @import("Bitboard.zig");
const game = @import("game.zig");
const Slider = game.Slider;
const SliderDirections = Slider.SliderDirections;
const Squares = game.Squares;
const Square = Squares.Square;

pub const MagicTableEntry = struct { magic: u64, mask: u64, shift: u6, offset: u32 };

pub const MagicEntry = struct { magic: u64, mask: Bitboard, shift: u6 };

fn blockersForSquare(s: Square, directions: *const SliderDirections) Bitboard {
    var blockers = Bitboard.empty();

    for (directions) |d| {
        const dx = d[0];
        const dy = d[1];

        var ray = s;
        while (true) {
            const offs = game.trySquareOffset(ray, @as(i3, dx), @as(i3, dy)) orelse break;
            blockers.bitOrAssign(ray);
            ray = offs;
        }
    }

    return blockers.bitAnd(Bitboard.fromSquare(s).not());
}

fn sliderMoves(s: Square, blockers: *const Bitboard, directions: *const SliderDirections) Bitboard {
    var moves = Bitboard.empty();

    for (directions) |d| {
        const dx = d[0];
        const dy = d[1];
        var ray = s;

        while (!blockers.contains(ray)) {
            const offs = game.trySquareOffset(ray, @as(i3, dx), @as(i3, dy)) orelse break;
            ray = offs;
            moves.bitOrAssign(ray);
        }
    }

    return moves;
}

fn magicIndex(entry: *const MagicEntry, blockers: *const Bitboard) usize {
    const blockers_masked = blockers.bitAnd(entry.mask);
    const hash = blockers_masked.bits *% entry.magic;
    return @intCast(hash >> entry.shift);
}

pub fn magicTableIndex(entry: *const MagicTableEntry, blockers: *const Bitboard) usize {
    const blockers_masked = blockers.bits & entry.mask;
    const hash = blockers_masked *% entry.magic;
    const idx: usize = @intCast(hash >> entry.shift);
    return @as(usize, entry.offset) + idx;
}

fn attemptMagics(alloc: std.mem.Allocator, directions: *const SliderDirections, s: Square, entry: *const MagicEntry) ![]Bitboard {
    const shift: u6 = @intCast(@as(u8, 64) - @as(u8, entry.shift));

    var tbl = try alloc.alloc(Bitboard, @as(u64, 1) << shift);
    var blockers = Bitboard.empty();

    while (true) {
        const moves = sliderMoves(s, &blockers, directions);
        const potential_entry = &tbl[magicIndex(entry, &blockers)];

        if (potential_entry.isEmpty()) {
            potential_entry.* = moves;
        } else if (potential_entry.bits != moves.bits) {
            return error.TableFillError;
        }

        blockers.bits = (blockers.bits -% entry.mask.bits) & entry.mask.bits;
        if (blockers.isEmpty()) {
            break;
        }
    }

    return tbl;
}

fn computeMagics(alloc: std.mem.Allocator, directions: *const SliderDirections, s: Square, shift_amt: u6, rng: *std.Random.DefaultPrng) struct { entry: MagicEntry, magics: []Bitboard } {
    const blockers = blockersForSquare(s, directions);
    const shift: u6 = @intCast(@as(u8, 64) - @as(u8, shift_amt));

    while (true) {
        const magic = rng.next() & rng.next() & rng.next();

        const entry = MagicEntry{ .magic = magic, .mask = blockers, .shift = shift };

        const magics = attemptMagics(alloc, directions, s, &entry) catch continue;
        return .{ .entry = entry, .magics = magics };
    }
}

fn precomputeMagics(alloc: std.mem.Allocator, rng: *std.Random.DefaultPrng) !void {
    const out_f = try std.fs.cwd().createFile("src/magics.zig", .{});

    var writer_buf: [8192]u8 = undefined;
    var writer = out_f.writer(&writer_buf);

    try writer.interface.writeAll("const precompute = @import(\"precompute.zig\");\n");
    try writer.interface.writeAll("const MagicTableEntry = precompute.MagicTableEntry;\n\n");
    try writer.interface.flush();

    const piece_names = [2][]const u8{ "Rook", "Bishop" };
    const sliders = [2]*const SliderDirections{ &Slider.RookDirections, &Slider.BishopDirections };
    for (0..2) |i| {
        const piece_name = piece_names[i];
        const slider = sliders[i];

        try writer.interface.print("pub const {s}Magics = [64]MagicTableEntry {{\n", .{piece_name});
        try writer.interface.flush();
        var tbl_len: usize = 0;

        var s: Square = 0;
        while (true) {
            const num_blockers: u6 = @intCast(blockersForSquare(s, slider).popCount());
            const magic = computeMagics(alloc, slider, s, num_blockers, rng);
            defer alloc.free(magic.magics);
            const entry = magic.entry;
            const magics = magic.magics;

            try writer.interface.print("    MagicTableEntry {{ .mask = 0x{x:0>16}, .magic = 0x{x:0>16}, .shift = {d}, .offset = {d} }},\n", .{ entry.mask.bits, entry.magic, entry.shift, tbl_len });
            try writer.interface.flush();
            tbl_len += magics.len;

            if (s == 63) break;
            s += 1;
        }

        try writer.interface.writeAll("};\n");
        try writer.interface.print("pub const {s}TableSize: usize = {d};\n", .{ piece_name, tbl_len });
        try writer.interface.flush();
    }
    try writer.interface.flush();
}

fn makeMoveTable(alloc: std.mem.Allocator, size: usize, directions: *const SliderDirections, magics: *const [64]MagicTableEntry) ![]Bitboard {
    var tbl = try alloc.alloc(Bitboard, size);

    for (magics, 0..) |entry, s| {
        const mask = Bitboard{ .bits = entry.mask };
        var blockers = Bitboard.empty();

        while (true) {
            const s_u6: u6 = @intCast(s);
            const moves = sliderMoves(s_u6, &blockers, directions);
            tbl[magicTableIndex(&entry, &blockers)] = moves;

            blockers.bits = (blockers.bits -% mask.bits) & mask.bits;
            if (blockers.isEmpty()) {
                break;
            }
        }
    }
    return tbl;
}

pub fn writeMoveTable(piece_name: []const u8, tbl: *const []Bitboard, writer: *std.Io.Writer) !void {
    try writer.print("pub const {s}Moves = [{d}]u64 {{\n", .{ piece_name, tbl.len });
    for (tbl.*) |entry| {
        try writer.print("    0x{x:0>16},\n", .{entry.bits});
        try writer.flush();
    }
    try writer.writeAll("};\n");
    try writer.flush();
}

pub fn writeMagics(piece_name: []const u8, magics: *const [64]MagicTableEntry, writer: *std.Io.Writer) !void {
    try writer.writeByte('\n');
    try writer.print("pub const {s}Magics = [64]MagicTableEntry {{\n", .{piece_name});
    try writer.flush();

    for (magics) |entry| {
        try writer.writeAll("    MagicTableEntry {\n");
        try writer.print("        .mask = 0x{x:0>16},\n", .{entry.mask});
        try writer.print("        .magic = 0x{x:0>16},\n", .{entry.magic});
        try writer.print("        .shift = {d},\n", .{entry.shift});
        try writer.print("        .offset = {d},\n", .{entry.offset});
        try writer.writeAll("    },\n");
        try writer.flush();
    }

    try writer.writeAll("};\n");
    try writer.flush();
}

pub fn main() !void {
    // const time: u128 = @bitCast(std.time.nanoTimestamp());
    // const seed: u64 = @truncate(time);
    // var rng = std.Random.DefaultPrng.init(seed);
    const alloc = std.heap.page_allocator;

    // try precomputeMagics(alloc, &rng);

    const magics = @import("magics.zig");

    var out_f = try std.fs.cwd().createFile("src/moves.zig", .{});
    var writer_buf: [8192]u8 = undefined;
    var writer = out_f.writer(&writer_buf);

    const rook_tbl = try makeMoveTable(alloc, magics.RookTableSize, &Slider.RookDirections, &magics.RookMagics);
    const bishop_tbl = try makeMoveTable(alloc, magics.BishopTableSize, &Slider.BishopDirections, &magics.BishopMagics);
    defer alloc.free(rook_tbl);
    defer alloc.free(bishop_tbl);

    try writer.interface.writeAll("const Bitboard = @import(\"Bitboard.zig\");\n");
    try writer.interface.writeAll("const precompute = @import(\"precompute.zig\");\n");
    try writer.interface.writeAll("const MagicTableEntry = precompute.MagicTableEntry;\n");

    try writeMagics("Rook", &magics.RookMagics, &writer.interface);
    try writeMagics("Bishop", &magics.BishopMagics, &writer.interface);
    try writeMoveTable("Rook", &rook_tbl, &writer.interface);
    try writeMoveTable("Bishop", &bishop_tbl, &writer.interface);
}

test "test blockers for square" {
    const blockers = blockersForSquare(Squares.e2, &Slider.RookDirections);
    try std.testing.expectEqual(blockers.bits, 0x10101010106e00);
}
