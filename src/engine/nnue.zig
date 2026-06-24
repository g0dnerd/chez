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
pub const ft_out = 512;
pub const fc1_in = ft_out * 2; // 1024 (white ++ black accumulators)
pub const fc1_out = 32;
pub const fc2_in = fc1_out;
pub const fc2_out = 32;

// Piece-count output buckets: bucket = clamp((popcount(all_pieces) - 1) / 4, 0, 7).
pub const num_output_buckets = 8;

pub const max_active_features = 30;

// SIMD width for accumulator (i16) passes.
const ft_vec_len = std.simd.suggestVectorLength(i16) orelse 8;

// .nnue file format constants
pub const magic_bytes = [4]u8{ 'C', 'H', 'E', 'Z' };
pub const format_version: u32 = 6;
pub const arch_hash: u32 = 0x48_4B_50_34; // "HKP4" (HalfKP, FT width 512, 8 output buckets, SCReLU FT activation)
pub const header_size = 16;

// Per-section byte sizes (packed, no alignment padding)
const ft_biases_bytes = ft_out * @sizeOf(i16);
const ft_weights_bytes = num_features * ft_out * @sizeOf(i16);
const fc1_weights_bytes = fc1_in * fc1_out * @sizeOf(i8);
const fc1_biases_bytes = fc1_out * @sizeOf(i32);
const fc2_weights_bytes = fc2_in * fc2_out * @sizeOf(i8);
const fc2_biases_bytes = fc2_out * @sizeOf(i32);
// One output head per piece-count bucket.
const output_weights_bytes = num_output_buckets * fc2_out * @sizeOf(i16);
const output_bias_bytes = num_output_buckets * @sizeOf(i32);

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
    debug_eval_count: u32,

    pub fn init() AccumulatorStack {
        return .{
            .accs = [_]Accumulator{Accumulator.empty} ** (max_stack_ply + 1),
            .deltas = [_]StackDelta{StackDelta.empty} ** (max_stack_ply + 1),
            .debug_eval_count = 0,
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
    // FC weights stored output-major (transposed from file layout) for linearForward_i8.
    fc1_weights: [fc1_out][fc1_in]i8,
    fc1_biases: [fc1_out]i32,
    fc2_weights: [fc2_out][fc2_in]i8,
    fc2_biases: [fc2_out]i32,
    // One output head per piece-count bucket (bucket-major for inference indexing).
    output_weights: [num_output_buckets][fc2_out]i16,
    output_bias: [num_output_buckets]i32,

    // Parse a Network from raw file bytes (little-endian).
    // Caller owns the returned pointer and must call deinit().
    pub fn loadFromBytes(allocator: std.mem.Allocator, data: []const u8) !*Network {
        if (data.len < header_size) return error.InvalidFileSize;

        if (!std.mem.eql(u8, data[0..4], &magic_bytes)) return error.InvalidMagic;

        const ver = std.mem.readInt(u32, data[4..8], .little);
        if (ver != format_version) return error.UnsupportedVersion;

        if (data.len < expected_file_size) return error.InvalidFileSize;

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

        // FC weights are stored output-major [out][in].
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

        // Output heads: [num_output_buckets][fc2_out]i16 weights, then
        // [num_output_buckets]i32 biases (bucket-major).
        @memcpy(
            std.mem.asBytes(&net.output_weights),
            data[off..][0..output_weights_bytes],
        );
        off += output_weights_bytes;

        @memcpy(
            std.mem.asBytes(&net.output_bias),
            data[off..][0..output_bias_bytes],
        );

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

        // FC1 -- output-major [out][in] on disk for v2
        try w.writeAll(std.mem.asBytes(&self.fc1_weights));
        try w.writeAll(std.mem.asBytes(&self.fc1_biases));

        // FC2 -- same
        try w.writeAll(std.mem.asBytes(&self.fc2_weights));
        try w.writeAll(std.mem.asBytes(&self.fc2_biases));

        // Output heads (bucket-major): weights then biases.
        try w.writeAll(std.mem.asBytes(&self.output_weights));
        try w.writeAll(std.mem.asBytes(&self.output_bias));
    }

    pub fn deinit(self: *Network, allocator: std.mem.Allocator) void {
        allocator.destroy(self);
    }
};

// Piece-count output bucket: bucket = clamp((popcount(all_pieces) - 1) / 4, 0, 7).
// Piece count ranges 2..32 (kings included) → buckets 0..7. Must stay identical to
// the training-side bucket in trainer/dataloader.zig.
pub fn outputBucket(state: *const State) usize {
    const piece_count = state.colors[Colors.white].bitOr(Bitboard, state.colors[Colors.black]).popCount();
    return @min(@as(usize, (piece_count - 1) / 4), num_output_buckets - 1);
}

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

// Quantized forward pass: accumulator → SCReLU → FC1 → FC2 → output.
// All arithmetic uses integer types to match the .nnue quantization scheme:
//   - Feature transformer: i16 weights/biases, i16 accumulator
//   - FT activation: SCReLU (squared clipped ReLU), output i8 at scale 127
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
        stack.debug_eval_count +%= 1;
        if (stack.debug_eval_count % 1024 == 0) {
            var fresh: Accumulator = undefined;
            refreshAccumulator(state, net, &fresh);
            for (0..2) |c_idx| {
                for (0..ft_out) |i| {
                    std.debug.assert(fresh.values[c_idx][i] == stack.accs[ply].values[c_idx][i]);
                }
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
    if (delta.is_null) {
        @memcpy(child, parent);
        return;
    }

    const own_king_raw: Square = if (perspective == Colors.white)
        delta.parent_white_king_sq
    else
        delta.parent_black_king_sq;
    const own_king: u32 = if (perspective == Colors.black) flipSquare(own_king_raw) else own_king_raw;

    // Gather the feature rows to add/subtract, then apply them in a single fused
    // pass below. Every legal delta touches at most one "add" and two "sub"
    // features (kings are not features, and castling -- a king move -- cannot
    // also capture), so three optionals cover all cases. The previous code did
    // a 256-wide memcpy followed by up to four separate in-place add/sub passes,
    // each reloading and rewriting the whole accumulator; the fused pass reads
    // parent and each weight row once and writes child once. Result is identical
    // (integer add/sub reassociation).
    var add0: ?*const [ft_out]i16 = null;
    var sub0: ?*const [ft_out]i16 = null;
    var sub1: ?*const [ft_out]i16 = null;

    // Moved piece: remove parent feature, add child feature (promoted if applicable).
    if (delta.piece != piece.king) {
        const old_idx = computeFeatureIndex(own_king, delta.color, perspective, delta.piece, delta.move.start);
        sub0 = &net.ft_weights[old_idx];
        const new_piece: Piece = if (delta.was_promotion) delta.promotion_piece else delta.piece;
        const new_idx = computeFeatureIndex(own_king, delta.color, perspective, new_piece, delta.move.end);
        add0 = &net.ft_weights[new_idx];
    }

    // Captured piece (opposite color, may be off m.end for en passant). sub0 is
    // free iff the mover was a king (king moves skip the block above).
    if (delta.captured_piece) |cp| {
        const cap_color: Color = @intCast(~@as(u1, @intCast(delta.color)));
        const cap_idx = computeFeatureIndex(own_king, cap_color, perspective, cp, delta.captured_square);
        if (sub0 == null) sub0 = &net.ft_weights[cap_idx] else sub1 = &net.ft_weights[cap_idx];
    }

    // Castling rook. Castling is a king move (so the moved-piece block above was
    // skipped) and cannot capture, hence sub0/add0 are still free here -- assert
    // it rather than rely on the invariant silently (a violation would drop a
    // feature and slowly corrupt the accumulator).
    if (delta.was_castling) {
        std.debug.assert(sub0 == null and add0 == null);
        const data = castling.castle_data[delta.color][delta.castling_side];
        const old_rook = computeFeatureIndex(own_king, delta.color, perspective, piece.rook, data.rook_from);
        const new_rook = computeFeatureIndex(own_king, delta.color, perspective, piece.rook, data.rook_to);
        sub0 = &net.ft_weights[old_rook];
        add0 = &net.ft_weights[new_rook];
    }

    // Single fused pass: child = parent + add0 - sub0 - sub1 (absent terms skipped).
    // The optionals are loop-invariant, so the compiler specializes the loop body.
    const L = ft_vec_len;
    var i: usize = 0;
    while (i + L <= ft_out) : (i += L) {
        var v: @Vector(L, i16) = parent[i..][0..L].*;
        if (add0) |a| v += @as(@Vector(L, i16), a[i..][0..L].*);
        if (sub0) |s| v -= @as(@Vector(L, i16), s[i..][0..L].*);
        if (sub1) |s| v -= @as(@Vector(L, i16), s[i..][0..L].*);
        child[i..][0..L].* = v;
    }
    while (i < ft_out) : (i += 1) {
        var x = parent[i];
        if (add0) |a| x += a[i];
        if (sub0) |s| x -= s[i];
        if (sub1) |s| x -= s[i];
        child[i] = x;
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

    // SCReLU on the feature-transformer output: a_i8 = round(clamp(acc,0,127)² / 127),
    // staying at scale 127 so FC1 is unchanged (see squaredClippedRelu_i16). The
    // hidden layers below keep plain shiftClippedRelu.
    const stm_relu = ops.squaredClippedRelu_i16(ft_out, &acc.values[stm]);
    const opp_relu = ops.squaredClippedRelu_i16(ft_out, &acc.values[opp]);

    // Concatenate perspectives: side-to-move first → [512]i8
    var concat: [fc1_in]i8 = undefined;
    @memcpy(concat[0..ft_out], &stm_relu);
    @memcpy(concat[ft_out..fc1_in], &opp_relu);

    // FC1: linearForward_i8 (fused bias + i8 SIMD dot products) → shiftClippedReLU
    const fc1_raw = quantized.linearForward_i8(fc1_in, fc1_out, &concat, @ptrCast(&net.fc1_weights), &net.fc1_biases);
    const fc1_act = quantized.shiftClippedRelu_i8(fc1_out, 6, &fc1_raw);

    // FC2: same fused path
    const fc2_raw = quantized.linearForward_i8(fc2_in, fc2_out, &fc1_act, @ptrCast(&net.fc2_weights), &net.fc2_biases);
    const fc2_act = quantized.shiftClippedRelu_i8(fc2_out, 6, &fc2_raw);

    // Output: select the piece-count bucket head, then i8 activations × i16
    // weights dot product + bias.
    const bucket = outputBucket(state);
    var output: i32 = net.output_bias[bucket];
    for (0..fc2_out) |i| {
        output += @as(i32, fc2_act[i]) * @as(i32, net.output_weights[bucket][i]);
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

    // Set a known output bias (first bucket's bias = start of the bias section).
    const output_bias_off = header_size + data_bytes - output_bias_bytes;
    std.mem.writeInt(i32, buf[output_bias_off..][0..4], -123, .little);

    const net = try Network.loadFromBytes(allocator, &buf);
    defer net.deinit(allocator);

    try std.testing.expectEqual(@as(i16, 42), net.ft_biases[0]);
    try std.testing.expectEqual(@as(i16, 0), net.ft_biases[1]);
    try std.testing.expectEqual(@as(i32, -123), net.output_bias[0]);
}

test "network write/load round trip" {
    // Full serialize→deserialize round trip across every section at the current
    // FT width. Catches layout/size drift after a width or format bump.
    const allocator = std.testing.allocator;
    const net = try createZeroNetwork(allocator);
    defer net.deinit(allocator);

    // Distinctive values spanning the first/last element of every section.
    net.ft_biases[0] = 7;
    net.ft_biases[ft_out - 1] = -9;
    net.ft_weights[0][0] = 11;
    net.ft_weights[num_features - 1][ft_out - 1] = -13;
    net.fc1_weights[0][0] = 1;
    net.fc1_weights[fc1_out - 1][fc1_in - 1] = -2;
    net.fc1_biases[0] = 100;
    net.fc2_weights[0][0] = 3;
    net.fc2_biases[fc2_out - 1] = -50;
    net.output_weights[0][0] = 21;
    net.output_weights[num_output_buckets - 1][fc2_out - 1] = -22;
    net.output_bias[0] = 12345;
    net.output_bias[num_output_buckets - 1] = -6789;

    const buf = try allocator.alloc(u8, expected_file_size);
    defer allocator.free(buf);
    var w = std.Io.Writer.fixed(buf);
    try net.writeToWriter(&w);
    try std.testing.expectEqual(expected_file_size, w.buffered().len);

    const net2 = try Network.loadFromBytes(allocator, w.buffered());
    defer net2.deinit(allocator);

    try std.testing.expectEqual(@as(i16, 7), net2.ft_biases[0]);
    try std.testing.expectEqual(@as(i16, -9), net2.ft_biases[ft_out - 1]);
    try std.testing.expectEqual(@as(i16, 11), net2.ft_weights[0][0]);
    try std.testing.expectEqual(@as(i16, -13), net2.ft_weights[num_features - 1][ft_out - 1]);
    try std.testing.expectEqual(@as(i8, 1), net2.fc1_weights[0][0]);
    try std.testing.expectEqual(@as(i8, -2), net2.fc1_weights[fc1_out - 1][fc1_in - 1]);
    try std.testing.expectEqual(@as(i32, 100), net2.fc1_biases[0]);
    try std.testing.expectEqual(@as(i8, 3), net2.fc2_weights[0][0]);
    try std.testing.expectEqual(@as(i32, -50), net2.fc2_biases[fc2_out - 1]);
    try std.testing.expectEqual(@as(i16, 21), net2.output_weights[0][0]);
    try std.testing.expectEqual(@as(i16, -22), net2.output_weights[num_output_buckets - 1][fc2_out - 1]);
    try std.testing.expectEqual(@as(i32, 12345), net2.output_bias[0]);
    try std.testing.expectEqual(@as(i32, -6789), net2.output_bias[num_output_buckets - 1]);
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
    // header 16 + ft_biases (512×2) + ft_weights (40960×512×2) + fc1_weights
    // (1024×32) + fc1_biases (32×4) + fc2_weights (32×32) + fc2_biases (32×4)
    // + output_weights (8×32×2) + output_bias (8×4)
    try std.testing.expectEqual(
        @as(usize, 16 + 1_024 + 41_943_040 + 32_768 + 128 + 1_024 + 128 + 512 + 32),
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
    // Result = output_bias ÷ (127 × 64). The start position has 32 pieces →
    // bucket (32-1)/4 = 7, so the bias of head 7 is the one that contributes.
    const allocator = std.testing.allocator;
    const net = try createZeroNetwork(allocator);
    defer net.deinit(allocator);

    const start_bucket = outputBucket(&State.defaultPosition());
    try std.testing.expectEqual(@as(usize, 7), start_bucket);

    net.output_bias[start_bucket] = 127 * 64; // Should produce exactly 1 centipawn
    try std.testing.expectEqual(@as(i32, 1), evaluate(&State.defaultPosition(), net));

    net.output_bias[start_bucket] = -(127 * 64); // Should produce exactly -1 centipawn
    try std.testing.expectEqual(@as(i32, -1), evaluate(&State.defaultPosition(), net));

    net.output_bias[start_bucket] = 127 * 64 * 100; // 100 centipawns
    try std.testing.expectEqual(@as(i32, 100), evaluate(&State.defaultPosition(), net));
}

test "evaluate selects output head by piece-count bucket" {
    // With all weights zero, eval = output_bias[bucket] ÷ (127×64). Two positions
    // in different buckets must read different heads.
    const allocator = std.testing.allocator;
    const net = try createZeroNetwork(allocator);
    defer net.deinit(allocator);

    net.output_bias[0] = 127 * 64 * 7; // low-piece head → 7 cp
    net.output_bias[num_output_buckets - 1] = 127 * 64 * 33; // full-board head → 33 cp

    // Start position: 32 pieces → bucket 7.
    try std.testing.expectEqual(@as(usize, num_output_buckets - 1), outputBucket(&State.defaultPosition()));
    try std.testing.expectEqual(@as(i32, 33), evaluate(&State.defaultPosition(), net));

    // K+P vs K: 3 pieces → bucket 0.
    const endgame = try State.fromFen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    try std.testing.expectEqual(@as(usize, 0), outputBucket(&endgame));
    try std.testing.expectEqual(@as(i32, 7), evaluate(&endgame, net));
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
