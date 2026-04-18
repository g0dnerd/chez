const std = @import("std");
const engine = @import("engine.zig");
const piece = @import("piece.zig");
const square = @import("square.zig");
const Bitboard = @import("Bitboard.zig");
const State = @import("State.zig");
const castling = @import("castling.zig");

const kore = @import("kore");
const ops = kore.ml.cpu.ops;
const quantized = kore.ml.cpu.quantized;

const Color = engine.Color;
const Colors = engine.Colors;
const Move = engine.Move;
const Square = square.Square;
const Piece = piece.Piece;

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
    computed: [2]bool, // per-perspective dirty flags

    pub const empty: Accumulator = .{
        .values = .{ .{0} ** ft_out, .{0} ** ft_out },
        .computed = .{ false, false },
    };
};

// Maximum search depth supported by the incremental accumulator stack.
// Quiescence can extend slightly past the regular max_ply, so allow some headroom.
pub const max_stack_ply: usize = 96;

// Information needed to apply (or recompute) the accumulator transition from
// the parent ply to the child ply.
pub const StackDelta = struct {
    is_null: bool,
    move: Move,
    color: Color,
    piece: Piece,
    captured_piece: ?Piece,
    captured_square: Square,
    was_promotion: bool,
    promotion_piece: Piece,
    was_castling: bool,
    castling_side: Color,
    // King squares as observed in the parent (white perspective and black perspective).
    // Used to compute the parent's feature indices when applying the delta.
    parent_white_king_sq: Square,
    parent_black_king_sq: Square,

    pub const empty: StackDelta = .{
        .is_null = false,
        .move = .{ .start = 0, .end = 0 },
        .color = 0,
        .piece = 0,
        .captured_piece = null,
        .captured_square = 0,
        .was_promotion = false,
        .promotion_piece = 0,
        .was_castling = false,
        .castling_side = 0,
        .parent_white_king_sq = 0,
        .parent_black_king_sq = 0,
    };
};

// Per-thread stack of accumulators indexed by search ply. Indexed [0..max_stack_ply].
// `accs[0]` is the root accumulator, refreshed once at search start.
// Each `deltas[ply]` describes the move (or null move) that produced `accs[ply]`
// from `accs[ply - 1]`.
pub const AccumulatorStack = struct {
    accs: [max_stack_ply + 1]Accumulator,
    deltas: [max_stack_ply + 1]StackDelta,

    pub fn init() AccumulatorStack {
        return .{
            .accs = [_]Accumulator{Accumulator.empty} ** (max_stack_ply + 1),
            .deltas = [_]StackDelta{StackDelta.empty} ** (max_stack_ply + 1),
        };
    }

    // Mark the given ply (and everything above it) as dirty. Called when the
    // root state changes (e.g., between iterative-deepening iterations on a
    // shared stack — currently we always rebuild the root explicitly).
    pub fn invalidate(self: *AccumulatorStack) void {
        for (&self.accs) |*acc| acc.computed = .{ false, false };
    }
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

// Quantized forward pass: accumulator → ClippedReLU → FC1 → FC2 → output.
// All arithmetic uses integer types to match the .nnue quantization scheme:
//   - Feature transformer: i16 weights/biases, i16 accumulator
//   - Hidden layers: i8 weights, i32 biases, i8 activations (after CReLU)
//   - Output: i8 weights, i32 bias, result in centipawns

// Compute the NNUE evaluation for a position (non-incremental).
// Returns a score in centipawns from the side-to-move's perspective.
pub fn evaluate(state: *const State, net: *const Network) i32 {
    var acc: Accumulator = undefined;
    refreshAccumulator(state, net, &acc);
    return @divTrunc(evaluateRawFromAccumulator(state, net, &acc), 127 * 64);
}

// Stack-aware evaluate: computes (lazily) accumulators along the stack chain
// and returns the centipawn score. Falls back to a fresh recomputation if the
// search has gone deeper than the stack supports.
pub fn evaluateLazy(state: *const State, net: *const Network, stack: *AccumulatorStack, ply: usize) i32 {
    if (ply > max_stack_ply) return evaluate(state, net);
    ensureComputed(stack, ply, Colors.white, net);
    ensureComputed(stack, ply, Colors.black, net);

    if (@import("builtin").mode == .Debug) {
        var fresh: Accumulator = undefined;
        refreshAccumulator(state, net, &fresh);
        for (0..2) |c_idx| {
            for (0..ft_out) |i| {
                std.debug.assert(fresh.values[c_idx][i] == stack.accs[ply].values[c_idx][i]);
            }
        }
    }

    return @divTrunc(evaluateRawFromAccumulator(state, net, &stack.accs[ply]), 127 * 64);
}

// Compute the FT feature index for a single piece on a given square, viewed
// from the given perspective. `own_king_for_indexing` must already be flipped
// when perspective == black.
fn computeFeatureIndex(
    own_king_for_indexing: u32,
    piece_color: Color,
    perspective: Color,
    piece_type: Piece,
    sq: Square,
) u32 {
    const rel_color: u32 = @as(u32, piece_color) ^ @as(u32, perspective);
    const psq: u32 = if (perspective == Colors.black) flipSquare(sq) else sq;
    return own_king_for_indexing * pieces_per_king +
        (rel_color * num_piece_types + @as(u32, piece_type)) * num_piece_squares +
        psq;
}

// Refresh a single perspective from scratch.
fn refreshPerspective(state: *const State, perspective: Color, net: *const Network, out: *[ft_out]i16) void {
    out.* = net.ft_biases;

    const king_bb = state.pieces[piece.king].bitAnd(Bitboard, state.colors[perspective]);
    const raw_king_sq = king_bb.trailingZeros();
    const king_sq: u32 = if (perspective == Colors.black) flipSquare(raw_king_sq) else raw_king_sq;

    inline for (0..5) |pt| {
        for (0..2) |col| {
            const c: Color = @intCast(col);
            var bb = state.pieces[pt].bitAnd(Bitboard, state.colors[c]);
            while (bb.next()) |sq| {
                const idx = computeFeatureIndex(king_sq, c, perspective, @as(Piece, @intCast(pt)), sq);
                ops.addVec_i16(ft_out, out, &net.ft_weights[idx]);
            }
        }
    }
}

// Refresh both perspectives from scratch.
pub fn refreshAccumulator(state: *const State, net: *const Network, acc: *Accumulator) void {
    refreshPerspective(state, Colors.white, net, &acc.values[Colors.white]);
    refreshPerspective(state, Colors.black, net, &acc.values[Colors.black]);
    acc.computed = .{ true, true };
}

// Apply a single delta to compute child[perspective] from parent[perspective].
fn applyDeltaPerspective(
    parent: *const [ft_out]i16,
    child: *[ft_out]i16,
    perspective: Color,
    delta: *const StackDelta,
    net: *const Network,
) void {
    @memcpy(child, parent);
    if (delta.is_null) return;

    const own_king_raw: Square = if (perspective == Colors.white)
        delta.parent_white_king_sq
    else
        delta.parent_black_king_sq;
    const own_king: u32 = if (perspective == Colors.black) flipSquare(own_king_raw) else own_king_raw;

    // 1. Remove moved piece's parent feature (kings are not features).
    if (delta.piece != piece.king) {
        const old_idx = computeFeatureIndex(own_king, delta.color, perspective, delta.piece, delta.move.start);
        ops.subVec_i16(ft_out, child, &net.ft_weights[old_idx]);
    }

    // 2. Add moved piece's child feature (using promoted type if applicable).
    if (delta.piece != piece.king) {
        const new_piece: Piece = if (delta.was_promotion) delta.promotion_piece else delta.piece;
        const new_idx = computeFeatureIndex(own_king, delta.color, perspective, new_piece, delta.move.end);
        ops.addVec_i16(ft_out, child, &net.ft_weights[new_idx]);
    }

    // 3. Captured piece (opposite color, may be off m.end for en passant).
    if (delta.captured_piece) |cp| {
        const cap_color: Color = @intCast(~@as(u1, @intCast(delta.color)));
        const cap_idx = computeFeatureIndex(own_king, cap_color, perspective, cp, delta.captured_square);
        ops.subVec_i16(ft_out, child, &net.ft_weights[cap_idx]);
    }

    // 4. Castling rook.
    if (delta.was_castling) {
        const data = castling.castle_data[delta.color][delta.castling_side];
        const old_rook = computeFeatureIndex(own_king, delta.color, perspective, piece.rook, data.rook_from);
        const new_rook = computeFeatureIndex(own_king, delta.color, perspective, piece.rook, data.rook_to);
        ops.subVec_i16(ft_out, child, &net.ft_weights[old_rook]);
        ops.addVec_i16(ft_out, child, &net.ft_weights[new_rook]);
    }
}

// Recursively walk back to the nearest computed ancestor, then apply deltas
// forward to compute `stack.accs[ply].values[perspective]`.
fn ensureComputed(stack: *AccumulatorStack, ply: usize, perspective: Color, net: *const Network) void {
    if (stack.accs[ply].computed[perspective]) return;
    // Root must be primed before search; missing root recomputation is a bug.
    std.debug.assert(ply > 0);

    ensureComputed(stack, ply - 1, perspective, net);
    applyDeltaPerspective(
        &stack.accs[ply - 1].values[perspective],
        &stack.accs[ply].values[perspective],
        perspective,
        &stack.deltas[ply],
        net,
    );
    stack.accs[ply].computed[perspective] = true;
}

// Record a real move: populate the delta at `child_ply` and decide what to do
// with the accumulator slots. King moves of color C force a full recompute of
// C's perspective (own king square changes, all features re-index); the other
// perspective is left dirty and computed lazily.
//
// `state_after` is the position after the move has been applied.
pub fn recordMove(
    stack: *AccumulatorStack,
    child_ply: usize,
    state_after: *const State,
    net: *const Network,
    m: Move,
    c: Color,
    p: Piece,
    undo: *const State.UndoInfo,
) void {
    if (child_ply > max_stack_ply) return;

    // Derive parent king squares from the after-state.
    const cur_white_king: Square = state_after.pieces[piece.king]
        .bitAnd(Bitboard, state_after.colors[Colors.white]).trailingZeros();
    const cur_black_king: Square = state_after.pieces[piece.king]
        .bitAnd(Bitboard, state_after.colors[Colors.black]).trailingZeros();

    var parent_white_king = cur_white_king;
    var parent_black_king = cur_black_king;
    if (p == piece.king) {
        if (c == Colors.white) parent_white_king = m.start else parent_black_king = m.start;
    }

    stack.deltas[child_ply] = .{
        .is_null = false,
        .move = m,
        .color = c,
        .piece = p,
        .captured_piece = undo.captured_piece,
        .captured_square = undo.captured_square,
        .was_promotion = undo.was_promotion,
        .promotion_piece = if (undo.promotion_piece) |pp| pp else 0,
        .was_castling = undo.was_castling,
        .castling_side = undo.castling_side,
        .parent_white_king_sq = parent_white_king,
        .parent_black_king_sq = parent_black_king,
    };

    // Mark child dirty by default.
    stack.accs[child_ply].computed = .{ false, false };

    // King moves: refresh moved-side's perspective eagerly (we have the state).
    // The other perspective stays lazy.
    if (p == piece.king) {
        refreshPerspective(state_after, c, net, &stack.accs[child_ply].values[c]);
        stack.accs[child_ply].computed[c] = true;
    }
}

// Record a null move: the position's pieces don't change, so the accumulator
// at the child ply equals the parent's. Mark dirty with an is_null delta so
// the lazy walk just memcpys.
pub fn recordNullMove(stack: *AccumulatorStack, child_ply: usize) void {
    if (child_ply > max_stack_ply) return;
    stack.deltas[child_ply] = StackDelta.empty;
    stack.deltas[child_ply].is_null = true;
    stack.accs[child_ply].computed = .{ false, false };
}

fn evaluateRawFromAccumulator(state: *const State, net: *const Network, acc: *const Accumulator) i32 {
    const stm = state.to_move;
    const opp: Color = @intCast(~@as(u1, @intCast(stm)));

    // ClippedReLU: clamp to [0, 127], truncate to i8.
    const stm_relu = ops.clippedRelu_i16(ft_out, &acc.values[stm]);
    const opp_relu = ops.clippedRelu_i16(ft_out, &acc.values[opp]);

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
    return output;
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
    try std.testing.expect(!acc.computed[0]);
    try std.testing.expect(!acc.computed[1]);
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

test "diagnose trained net" {
    const allocator = std.testing.allocator;

    // Load the trained net — adjust path if needed
    var single_threaded: std.Io.Threaded = .init_single_threaded;
    const io = single_threaded.io();
    const net = Network.load(io, allocator, "data/net_k2_30M.nnue") catch |err| {
        std.debug.print("Could not load net: {}\n", .{err});
        return err;
    };

    defer net.deinit(allocator);

    // 1. Weight statistics — are weights alive or dead?
    var ft_nonzero: usize = 0;
    var ft_max: i16 = 0;
    for (net.ft_weights) |row| {
        for (row) |w| {
            if (w != 0) ft_nonzero += 1;
            if (w > ft_max) ft_max = w;
            if (-w > ft_max) ft_max = -w;
        }
    }
    var fc1_nonzero: usize = 0;
    for (net.fc1_weights) |row| for (row) |w| {
        if (w != 0) fc1_nonzero += 1;
    };
    var out_nonzero: usize = 0;
    for (net.output_weights) |w| {
        if (w != 0) out_nonzero += 1;
    }
    std.debug.print("\n=== Weight statistics ===\n", .{});
    std.debug.print("FT: {d}/{d} nonzero, max abs = {d}\n", .{ ft_nonzero, num_features * ft_out, ft_max });
    std.debug.print("FC1: {d}/{d} nonzero\n", .{ fc1_nonzero, fc1_in * fc1_out });
    std.debug.print("Output: {d}/{d} nonzero, bias = {d}\n", .{ out_nonzero, fc2_out, net.output_bias });

    // 2. Eval known positions — print both raw and divided values
    const positions = [_]struct { name: []const u8, fen: ?[]const u8 }{
        .{ .name = "Starting position", .fen = null },
        .{ .name = "White up a queen", .fen = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" },
        .{ .name = "White down a queen", .fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1" },
        .{ .name = "Ruy Lopez", .fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3" },
        .{ .name = "White up a rook", .fen = "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1" },
    };

    std.debug.print("\n=== Eval results (raw / 8128 = cp) ===\n", .{});
    for (positions) |pos| {
        const state = if (pos.fen) |fen| State.fromFen(fen) catch continue else State.defaultPosition();
        var diag_acc: Accumulator = undefined;
        refreshAccumulator(&state, net, &diag_acc);
        const raw = evaluateRawFromAccumulator(&state, net, &diag_acc);
        std.debug.print("{s}: raw={d}, cp={d}\n", .{ pos.name, raw, @divTrunc(raw, 127 * 64) });
    }

    // 3. Full layer-by-layer trace through the forward pass
    std.debug.print("\n=== Layer-by-layer trace (startpos, white to move) ===\n", .{});

    const start = State.defaultPosition();
    const stm_features = activeFeatures(&start, Colors.white);
    const opp_features = activeFeatures(&start, Colors.black);

    var stm_acc: [ft_out]i16 = net.ft_biases;
    for (stm_features.features[0..stm_features.len]) |idx| {
        ops.addVec_i16(ft_out, &stm_acc, &net.ft_weights[idx]);
    }
    var opp_acc: [ft_out]i16 = net.ft_biases;
    for (opp_features.features[0..opp_features.len]) |idx| {
        ops.addVec_i16(ft_out, &opp_acc, &net.ft_weights[idx]);
    }

    // Accumulator stats
    var acc_pos: usize = 0;
    var acc_min: i16 = std.math.maxInt(i16);
    var acc_max: i16 = std.math.minInt(i16);
    for (stm_acc) |v| {
        if (v > 0) acc_pos += 1;
        if (v < acc_min) acc_min = v;
        if (v > acc_max) acc_max = v;
    }
    std.debug.print("Accumulator: {d}/{d} positive, range [{d}, {d}]\n", .{ acc_pos, ft_out, acc_min, acc_max });

    // CReLU
    const stm_relu = ops.clippedRelu_i16(ft_out, &stm_acc);
    const opp_relu = ops.clippedRelu_i16(ft_out, &opp_acc);
    var crelu_nonzero: usize = 0;
    var crelu_max: i8 = 0;
    for (stm_relu) |v| {
        if (v != 0) crelu_nonzero += 1;
        if (v > crelu_max) crelu_max = v;
    }
    std.debug.print("After CReLU: {d}/{d} nonzero, max = {d}\n", .{ crelu_nonzero, ft_out, crelu_max });

    // Concat
    var concat: [fc1_in]i8 = undefined;
    @memcpy(concat[0..ft_out], &stm_relu);
    @memcpy(concat[ft_out..fc1_in], &opp_relu);

    // FC1 matmul
    const fc1_raw = quantized.matmul(i8, 1, fc1_in, fc1_out, &concat, @ptrCast(&net.fc1_weights));
    std.debug.print("FC1 raw (before bias): ", .{});
    for (0..@min(8, fc1_out)) |i| std.debug.print("{d} ", .{fc1_raw[i]});
    std.debug.print("...\n", .{});

    std.debug.print("FC1 biases: ", .{});
    for (0..@min(8, fc1_out)) |i| std.debug.print("{d} ", .{net.fc1_biases[i]});
    std.debug.print("...\n", .{});

    // FC1 after bias + /64 + CReLU
    var fc1_scaled: [fc1_out]i16 = undefined;
    for (0..fc1_out) |i| {
        fc1_scaled[i] = @intCast(std.math.clamp(@divTrunc(fc1_raw[i] + net.fc1_biases[i], 64), -32768, 32767));
    }
    std.debug.print("FC1 after bias+/64: ", .{});
    for (0..@min(8, fc1_out)) |i| std.debug.print("{d} ", .{fc1_scaled[i]});
    std.debug.print("...\n", .{});

    const fc1_act = ops.clippedRelu_i16(fc1_out, &fc1_scaled);
    var fc1_nonzero2: usize = 0;
    for (fc1_act) |v| {
        if (v != 0) fc1_nonzero2 += 1;
    }
    std.debug.print("FC1 after CReLU: {d}/{d} nonzero, values: ", .{ fc1_nonzero2, fc1_out });
    for (0..@min(8, fc1_out)) |i| std.debug.print("{d} ", .{fc1_act[i]});
    std.debug.print("...\n", .{});

    // FC2
    const fc2_raw = quantized.matmul(i8, 1, fc2_in, fc2_out, &fc1_act, @ptrCast(&net.fc2_weights));
    var fc2_scaled: [fc2_out]i16 = undefined;
    for (0..fc2_out) |i| {
        fc2_scaled[i] = @intCast(std.math.clamp(@divTrunc(fc2_raw[i] + net.fc2_biases[i], 64), -32768, 32767));
    }
    const fc2_act = ops.clippedRelu_i16(fc2_out, &fc2_scaled);
    var fc2_nonzero: usize = 0;
    for (fc2_act) |v| {
        if (v != 0) fc2_nonzero += 1;
    }
    std.debug.print("FC2 after CReLU: {d}/{d} nonzero, values: ", .{ fc2_nonzero, fc2_out });
    for (0..@min(8, fc2_out)) |i| std.debug.print("{d} ", .{fc2_act[i]});
    std.debug.print("...\n", .{});

    // Output
    var output: i32 = net.output_bias;
    for (0..fc2_out) |i| {
        output += @as(i32, fc2_act[i]) * @as(i32, net.output_weights[i]);
    }
    std.debug.print("Output raw = {d}, /8128 = {d}\n", .{ output, @divTrunc(output, 127 * 64) });
}
