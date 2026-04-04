const std = @import("std");
const engine = @import("engine.zig");
const piece = @import("piece.zig");
const square = @import("square.zig");
const Bitboard = @import("Bitboard.zig");
const State = @import("State.zig");

const kore = @import("kore");
const ops = kore.ml.cpu.ops;
const quantized = kore.ml.cpu.quantized;

const Color = engine.Color;
const Colors = engine.Colors;
const Square = square.Square;

// HalfKP feature indexing
// Per perspective: 64 king squares × 10 piece buckets (2 colors × 5 types) × 64 piece squares
pub const num_king_squares = 64;
pub const num_piece_types = 5; // pawn, knight, bishop, rook, queen (no king)
pub const num_piece_colors = 2; // friendly, enemy
pub const num_piece_squares = 64;
pub const pieces_per_king = num_piece_colors * num_piece_types * num_piece_squares; // 640
pub const num_features = num_king_squares * pieces_per_king; // 40960

// Network layer dimensions
pub const ft_out = 256;
pub const fc1_in = ft_out * 2; // 512 (white ++ black accumulators)
pub const fc1_out = 32;
pub const fc2_in = fc1_out;
pub const fc2_out = 32;

pub const max_active_features = 30;

// .nnue file format constants
pub const magic_bytes = [4]u8{ 'C', 'H', 'E', 'Z' };
pub const format_version: u32 = 1;
pub const arch_hash: u32 = 0x48_4B_50_31; // "HKP1"
pub const header_size = 16;

// Per-section byte sizes (packed, no alignment padding)
const ft_biases_bytes = ft_out * @sizeOf(i16);
const ft_weights_bytes = num_features * ft_out * @sizeOf(i16);
const fc1_weights_bytes = fc1_in * fc1_out * @sizeOf(i8);
const fc1_biases_bytes = fc1_out * @sizeOf(i32);
const fc2_weights_bytes = fc2_in * fc2_out * @sizeOf(i8);
const fc2_biases_bytes = fc2_out * @sizeOf(i32);
const output_weights_bytes = fc2_out * @sizeOf(i8);
const output_bias_bytes = @sizeOf(i32);

pub const data_bytes = ft_biases_bytes + ft_weights_bytes +
    fc1_weights_bytes + fc1_biases_bytes +
    fc2_weights_bytes + fc2_biases_bytes +
    output_weights_bytes + output_bias_bytes;
pub const expected_file_size = header_size + data_bytes;

// Data structures
pub const Accumulator = struct {
    values: [2][ft_out]i16, // [0] = white, [1] = black perspective
    computed: bool,

    pub const empty: Accumulator = .{
        .values = .{ .{0} ** ft_out, .{0} ** ft_out },
        .computed = false,
    };
};

pub const Network = struct {
    ft_biases: [ft_out]i16,
    ft_weights: [num_features][ft_out]i16,
    fc1_weights: [fc1_in][fc1_out]i8,
    fc1_biases: [fc1_out]i32,
    fc2_weights: [fc2_in][fc2_out]i8,
    fc2_biases: [fc2_out]i32,
    output_weights: [fc2_out]i8,
    output_bias: i32,

    // Parse a Network from raw file bytes (little-endian).
    // Caller owns the returned pointer and must call deinit().
    pub fn loadFromBytes(allocator: std.mem.Allocator, data: []const u8) !*Network {
        if (data.len < expected_file_size) return error.InvalidFileSize;

        if (!std.mem.eql(u8, data[0..4], &magic_bytes)) return error.InvalidMagic;

        const ver = std.mem.readInt(u32, data[4..8], .little);
        if (ver != format_version) return error.UnsupportedVersion;

        const arch = std.mem.readInt(u32, data[8..12], .little);
        if (arch != arch_hash) return error.ArchitectureMismatch;

        const net = try allocator.create(Network);
        errdefer allocator.destroy(net);

        var off: usize = header_size;

        @memcpy(
            std.mem.asBytes(&net.ft_biases),
            data[off..][0..ft_biases_bytes],
        );
        off += ft_biases_bytes;

        @memcpy(
            std.mem.asBytes(&net.ft_weights),
            data[off..][0..ft_weights_bytes],
        );
        off += ft_weights_bytes;

        @memcpy(
            std.mem.asBytes(&net.fc1_weights),
            data[off..][0..fc1_weights_bytes],
        );
        off += fc1_weights_bytes;

        @memcpy(
            std.mem.asBytes(&net.fc1_biases),
            data[off..][0..fc1_biases_bytes],
        );
        off += fc1_biases_bytes;

        @memcpy(
            std.mem.asBytes(&net.fc2_weights),
            data[off..][0..fc2_weights_bytes],
        );
        off += fc2_weights_bytes;

        @memcpy(
            std.mem.asBytes(&net.fc2_biases),
            data[off..][0..fc2_biases_bytes],
        );
        off += fc2_biases_bytes;

        @memcpy(
            std.mem.asBytes(&net.output_weights),
            data[off..][0..output_weights_bytes],
        );
        off += output_weights_bytes;

        net.output_bias = std.mem.readInt(i32, data[off..][0..4], .little);

        return net;
    }

    // Load a .nnue file from disk.
    pub fn load(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !*Network {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        const stat = try file.stat(io);
        const len: usize = @intCast(stat.size);
        defer file.close(io);

        if (@import("builtin").target.os.tag == .linux) {
            const ptr = try std.posix.mmap(
                null,
                len,
                std.os.linux.PROT{ .READ = true },
                .{ .TYPE = .SHARED },
                file.handle,
                0,
            );
            defer std.posix.munmap(@alignCast(ptr));
            return loadFromBytes(allocator, ptr[0..len]);
        } else {
            var buf = try allocator.alloc(u8, len);
            defer allocator.free(buf);
            var reader = file.reader(io, &buf);
            _ = try reader.interface.take(len);
            return loadFromBytes(allocator, buf[0..len]);
        }
    }

    // Serialize to a writer in .nnue format (little-endian).
    pub fn writeToWriter(self: *const Network, w: *std.Io.Writer) !void {
        // Header
        try w.writeAll(&magic_bytes);
        try w.writeInt(u32, format_version, .little);
        try w.writeInt(u32, arch_hash, .little);
        try w.writeInt(u32, 0, .little); // reserved

        // Feature transformer
        try w.writeAll(std.mem.asBytes(&self.ft_biases));
        try w.writeAll(std.mem.asBytes(&self.ft_weights));

        // FC1
        try w.writeAll(std.mem.asBytes(&self.fc1_weights));
        try w.writeAll(std.mem.asBytes(&self.fc1_biases));

        // FC2
        try w.writeAll(std.mem.asBytes(&self.fc2_weights));
        try w.writeAll(std.mem.asBytes(&self.fc2_biases));

        // Output
        try w.writeAll(std.mem.asBytes(&self.output_weights));
        try w.writeInt(i32, self.output_bias, .little);
    }

    pub fn deinit(self: *Network, allocator: std.mem.Allocator) void {
        allocator.destroy(self);
    }
};

// Feature extraction
// Flip square vertically (rank mirror) for black perspective.
fn flipSquare(sq: Square) Square {
    return sq ^ 56;
}

pub const FeatureList = struct {
    features: [max_active_features]u32,
    len: u8,
};

// Compute active HalfKP feature indices for a position from the given perspective.
//
// Index formula per non-king piece:
//   king_sq * 640 + (relative_color * 5 + piece_type) * 64 + piece_sq
//
// For the black perspective, all squares are rank-mirrored (XOR 56).
pub fn activeFeatures(state: *const State, perspective: Color) FeatureList {
    var result = FeatureList{
        .features = undefined,
        .len = 0,
    };

    // King square for this perspective
    const king_bb = state.pieces[piece.king].bitAnd(Bitboard, state.colors[perspective]);
    const raw_king_sq = king_bb.trailingZeros();
    const king_sq: u32 = if (perspective == Colors.black)
        flipSquare(raw_king_sq)
    else
        raw_king_sq;

    // Iterate all non-king piece types: pawn(0) .. queen(4)
    inline for (0..5) |pt| {
        for (0..2) |col| {
            const c: Color = @intCast(col);
            var bb = state.pieces[pt].bitAnd(Bitboard, state.colors[c]);

            while (bb.next()) |sq| {
                const rel_color: u32 = @as(u32, c) ^ @as(u32, perspective);
                const psq: u32 = if (perspective == Colors.black)
                    flipSquare(sq)
                else
                    sq;

                const index = king_sq * pieces_per_king +
                    (rel_color * num_piece_types + @as(u32, pt)) * num_piece_squares +
                    psq;

                result.features[result.len] = index;
                result.len += 1;
            }
        }
    }

    return result;
}

// ==============================================================================
// NNUE Inference (Non-Incremental)
// ==============================================================================
//
// Quantized forward pass: accumulator → ClippedReLU → FC1 → FC2 → output.
// All arithmetic uses integer types to match the .nnue quantization scheme:
//   - Feature transformer: i16 weights/biases, i16 accumulator
//   - Hidden layers: i8 weights, i32 biases, i8 activations (after CReLU)
//   - Output: i8 weights, i32 bias, result in centipawns

// Compute the NNUE evaluation for a position (non-incremental).
// Returns a score in centipawns from the side-to-move's perspective.
pub fn evaluate(state: *const State, net: *const Network) i32 {
    const stm = state.to_move;
    const opp: Color = @intCast(~@as(u1, @intCast(stm)));

    const stm_features = activeFeatures(state, stm);
    const opp_features = activeFeatures(state, opp);

    // Accumulator: start from FT biases, add weight rows for each active feature.
    var stm_acc: [ft_out]i16 = net.ft_biases;
    for (stm_features.features[0..stm_features.len]) |idx| {
        ops.addVec_i16(ft_out, &stm_acc, &net.ft_weights[idx]);
    }

    var opp_acc: [ft_out]i16 = net.ft_biases;
    for (opp_features.features[0..opp_features.len]) |idx| {
        ops.addVec_i16(ft_out, &opp_acc, &net.ft_weights[idx]);
    }

    // ClippedReLU: clamp to [0, 127], truncate to i8.
    const stm_relu = ops.clippedRelu_i16(ft_out, &stm_acc);
    const opp_relu = ops.clippedRelu_i16(ft_out, &opp_acc);

    // Concatenate perspectives: side-to-move first → [512]i8
    var concat: [fc1_in]i8 = undefined;
    @memcpy(concat[0..ft_out], &stm_relu);
    @memcpy(concat[ft_out..fc1_in], &opp_relu);

    // FC1: [1, 512] @ [512, 32] → [32]i32, add bias, ÷64, ClippedReLU → i8
    const fc1_raw = quantized.matmul(i8, 1, fc1_in, fc1_out, &concat, @ptrCast(&net.fc1_weights));
    var fc1_scaled: [fc1_out]i16 = undefined;
    for (0..fc1_out) |i| {
        fc1_scaled[i] = @intCast(std.math.clamp(@divTrunc(fc1_raw[i] + net.fc1_biases[i], 64), -32768, 32767));
    }
    const fc1_act = ops.clippedRelu_i16(fc1_out, &fc1_scaled);

    // FC2: [1, 32] @ [32, 32] → [32]i32, add bias, ÷64, ClippedReLU → i8
    const fc2_raw = quantized.matmul(i8, 1, fc2_in, fc2_out, &fc1_act, @ptrCast(&net.fc2_weights));
    var fc2_scaled: [fc2_out]i16 = undefined;
    for (0..fc2_out) |i| {
        fc2_scaled[i] = @intCast(std.math.clamp(@divTrunc(fc2_raw[i] + net.fc2_biases[i], 64), -32768, 32767));
    }
    const fc2_act = ops.clippedRelu_i16(fc2_out, &fc2_scaled);

    // Output: dot product of i8 weights × i8 activations + i32 bias, ÷(127×64) → centipawns
    var output: i32 = net.output_bias;
    for (0..fc2_out) |i| {
        output += @as(i32, fc2_act[i]) * @as(i32, net.output_weights[i]);
    }
    return @divTrunc(output, 127 * 64);
}

test "feature index bounds" {
    // Maximum possible index: king_sq=63, rel_color=1, piece_type=4, piece_sq=63
    const max_index = 63 * pieces_per_king + (1 * num_piece_types + 4) * num_piece_squares + 63;
    try std.testing.expect(max_index < num_features);
    try std.testing.expectEqual(num_features, 40960);
    try std.testing.expectEqual(pieces_per_king, 640);
}

test "active features starting position" {
    const state = State.defaultPosition();
    const white_features = activeFeatures(&state, Colors.white);
    const black_features = activeFeatures(&state, Colors.black);

    // 30 non-king pieces in starting position (16 pawns + 4 knights + 4 bishops + 4 rooks + 2 queens)
    try std.testing.expectEqual(@as(u8, 30), white_features.len);
    try std.testing.expectEqual(@as(u8, 30), black_features.len);

    // All feature indices must be in [0, num_features)
    for (white_features.features[0..white_features.len]) |idx| {
        try std.testing.expect(idx < num_features);
    }
    for (black_features.features[0..black_features.len]) |idx| {
        try std.testing.expect(idx < num_features);
    }

    // No duplicate features within a perspective
    for (0..white_features.len) |i| {
        for (i + 1..white_features.len) |j| {
            try std.testing.expect(white_features.features[i] != white_features.features[j]);
        }
    }
}

test "active features specific index" {
    // In the starting position, white perspective:
    // White king on e1 (sq 4), white pawn on a2 (sq 8):
    //   relative_color = 0 (friendly), piece_type = 0 (pawn)
    //   index = 4 * 640 + (0 * 5 + 0) * 64 + 8 = 2560 + 8 = 2568
    const state = State.defaultPosition();
    const features = activeFeatures(&state, Colors.white);

    var found = false;
    for (features.features[0..features.len]) |idx| {
        if (idx == 2568) found = true;
    }
    try std.testing.expect(found);
}

test "active features black perspective mirror" {
    // Black perspective: black king on e8 (sq 60), flipped = 60^56 = 4
    // Black pawn on a7 (sq 48), flipped = 48^56 = 8, relative_color = 0 (friendly), piece_type = 0
    // index = 4 * 640 + (0 * 5 + 0) * 64 + 8 = 2568
    // Same as white's pawn feature due to symmetry of the starting position.
    const state = State.defaultPosition();
    const features = activeFeatures(&state, Colors.black);

    var found = false;
    for (features.features[0..features.len]) |idx| {
        if (idx == 2568) found = true;
    }
    try std.testing.expect(found);
}

test "active features after move" {
    // After 1. e4, white has a pawn on e4 (sq 28) instead of e2 (sq 12)
    // Feature for that pawn from white perspective:
    //   king_sq = 4 (e1), relative_color = 0, piece_type = 0, piece_sq = 28
    //   index = 4 * 640 + 0 + 28 = 2588
    var state = State.defaultPosition();
    const move = engine.Move{ .start = square.e2, .end = square.e4 };
    _ = state.makeMove(move, state.to_move, engine.piece.pawn);

    const features = activeFeatures(&state, Colors.white);

    // Should NOT contain old pawn feature (e2 = sq 12 → index 2572)
    // Should contain new pawn feature (e4 = sq 28 → index 2588)
    var found_old = false;
    var found_new = false;
    for (features.features[0..features.len]) |idx| {
        if (idx == 2572) found_old = true;
        if (idx == 2588) found_new = true;
    }
    try std.testing.expect(!found_old);
    try std.testing.expect(found_new);
}

test "network round trip" {
    const allocator = std.testing.allocator;

    // Build a small test buffer with known values
    var buf: [expected_file_size]u8 = .{0} ** expected_file_size;

    // Write header
    @memcpy(buf[0..4], &magic_bytes);
    std.mem.writeInt(u32, buf[4..8], format_version, .little);
    std.mem.writeInt(u32, buf[8..12], arch_hash, .little);
    std.mem.writeInt(u32, buf[12..16], 0, .little);

    // Set a known FT bias value at index 0
    std.mem.writeInt(i16, buf[header_size..][0..2], 42, .little);

    // Set a known output bias
    const output_bias_off = header_size + data_bytes - output_bias_bytes;
    std.mem.writeInt(i32, buf[output_bias_off..][0..4], -123, .little);

    const net = try Network.loadFromBytes(allocator, &buf);
    defer net.deinit(allocator);

    try std.testing.expectEqual(@as(i16, 42), net.ft_biases[0]);
    try std.testing.expectEqual(@as(i16, 0), net.ft_biases[1]);
    try std.testing.expectEqual(@as(i32, -123), net.output_bias);
}

test "loadFromBytes rejects bad magic" {
    var buf: [expected_file_size]u8 = .{0} ** expected_file_size;
    @memcpy(buf[0..4], "NOPE");
    std.mem.writeInt(u32, buf[4..8], format_version, .little);
    std.mem.writeInt(u32, buf[8..12], arch_hash, .little);

    const result = Network.loadFromBytes(std.testing.allocator, &buf);
    try std.testing.expectError(error.InvalidMagic, result);
}

test "loadFromBytes rejects wrong version" {
    var buf: [expected_file_size]u8 = .{0} ** expected_file_size;
    @memcpy(buf[0..4], &magic_bytes);
    std.mem.writeInt(u32, buf[4..8], 99, .little);
    std.mem.writeInt(u32, buf[8..12], arch_hash, .little);

    const result = Network.loadFromBytes(std.testing.allocator, &buf);
    try std.testing.expectError(error.UnsupportedVersion, result);
}

test "loadFromBytes rejects short file" {
    var buf: [15]u8 = .{0} ** 15;
    const result = Network.loadFromBytes(std.testing.allocator, &buf);
    try std.testing.expectError(error.InvalidFileSize, result);
}

test "expected file size" {
    try std.testing.expectEqual(
        @as(usize, 16 + 512 + 20_971_520 + 16_384 + 128 + 1_024 + 128 + 32 + 4),
        expected_file_size,
    );
}

test "accumulator empty" {
    const acc = Accumulator.empty;
    try std.testing.expect(!acc.computed);
    for (acc.values[0]) |v| try std.testing.expectEqual(@as(i16, 0), v);
    for (acc.values[1]) |v| try std.testing.expectEqual(@as(i16, 0), v);
}

// ==============================================================================
// Evaluate tests
// ==============================================================================

// Helper: create a zero-initialized Network on the heap for testing.
// All weights/biases are 0, so evaluate() should return 0 for any position.
fn createZeroNetwork(allocator: std.mem.Allocator) !*Network {
    var buf: [expected_file_size]u8 = .{0} ** expected_file_size;
    @memcpy(buf[0..4], &magic_bytes);
    std.mem.writeInt(u32, buf[4..8], format_version, .little);
    std.mem.writeInt(u32, buf[8..12], arch_hash, .little);
    return Network.loadFromBytes(allocator, &buf);
}

test "evaluate zero network returns zero" {
    const allocator = std.testing.allocator;
    const net = try createZeroNetwork(allocator);
    defer net.deinit(allocator);

    const state = State.defaultPosition();
    const score = evaluate(&state, net);
    try std.testing.expectEqual(@as(i32, 0), score);
}

test "evaluate output bias only" {
    // With all weights zero, only the output bias contributes.
    // Result = output_bias ÷ (127 × 64)
    const allocator = std.testing.allocator;
    const net = try createZeroNetwork(allocator);
    defer net.deinit(allocator);

    net.output_bias = 127 * 64; // Should produce exactly 1 centipawn
    try std.testing.expectEqual(@as(i32, 1), evaluate(&State.defaultPosition(), net));

    net.output_bias = -(127 * 64); // Should produce exactly -1 centipawn
    try std.testing.expectEqual(@as(i32, -1), evaluate(&State.defaultPosition(), net));

    net.output_bias = 127 * 64 * 100; // 100 centipawns
    try std.testing.expectEqual(@as(i32, 100), evaluate(&State.defaultPosition(), net));
}

test "evaluate symmetric position gives same magnitude for both sides" {
    // In the starting position, white and black have identical piece arrangements
    // (after perspective flipping). The network should give the same evaluation
    // regardless of who is to move, since the position is symmetric.
    const allocator = std.testing.allocator;
    const net = try createZeroNetwork(allocator);
    defer net.deinit(allocator);

    // Set a uniform FT bias so the accumulator has non-zero values after CReLU
    for (&net.ft_biases) |*b| b.* = 50;

    const state_white = State.defaultPosition();
    const score_white = evaluate(&state_white, net);

    // Flip side to move
    var state_black = State.defaultPosition();
    state_black.to_move = @intCast(Colors.black);
    const score_black = evaluate(&state_black, net);

    // Symmetric position → same score for both perspectives
    try std.testing.expectEqual(score_white, score_black);
}
