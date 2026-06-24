const std = @import("std");
const engine = @import("engine.zig");
const Bitboard = @import("Bitboard.zig");
const State = @import("State.zig");
const movegen = @import("movegen.zig");
const nnue = @import("nnue.zig");
const piece = @import("piece.zig");
const square = @import("square.zig");
const clock = @import("clock.zig");

// Max tapered phase (full non-pawn material), matching evaluation.zig.
const max_phase = @import("score.zig").max_phase_mg;

const Atomic = std.atomic.Value;
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const GameResult = engine.GameResult;
const Move = engine.Move;
const MoveList = movegen.MoveList;
const evaluation = engine.evaluation;

pub const SearchParams = struct {
    nnue_scale: i32 = 2,
    // NNUE output scaling (applied only when a network is loaded). Both are
    // percentages so they expose cleanly as integer UCI spin options.
    // Material scaling: compress eval toward material_scale_min% at bare-kings,
    // ramping to 100% at full non-pawn material (phase == max_phase_mg).
    material_scale_min: i32 = 75,
    // 50-move damping: eval is undamped until halfmove_clock reaches
    // fifty_move_start, then ramps down to (100 - fifty_move_damp)% at clock 100.
    fifty_move_start: i32 = 20,
    fifty_move_damp: i32 = 50,
    rfp_base: i32 = 80,
    futility_margin_1: i32 = 300,
    futility_margin_2: i32 = 600,
    delta_margin: i32 = 200,
    // LMR reduction = lmr_base/100 + ln(d)*ln(i) / (lmr_div/100). Stored as
    // hundredths so they can be exposed as integer UCI spin options.
    lmr_base: i32 = 75,
    lmr_div: i32 = 120,
    // LMR history adjustment: reduction -= clamp(combined_history/lmr_hist_div, -2, 2).
    lmr_hist_div: i32 = 8000,
    // History-based pruning: at depth <= histprune_depth, skip late quiet moves
    // whose combined history < -histprune_margin * depth.
    histprune_depth: i32 = 3,
    histprune_margin: i32 = 2000,
    // Internal Iterative Reductions: with no TT move at depth >= iir_min_depth,
    // search one ply shallower.
    iir_min_depth: i32 = 4,
};

// Precompute the LMR reduction table [depth][move_index] from the log formula.
// Runtime (not comptime) so lmr_base/lmr_div stay tunable and we avoid any
// @log-at-comptime dependency.
fn computeLmrTable(lmr_base: i32, lmr_div: i32) [64][64]u8 {
    var table: [64][64]u8 = undefined;
    const base: f64 = @as(f64, @floatFromInt(lmr_base)) / 100.0;
    const div: f64 = @as(f64, @floatFromInt(@max(1, lmr_div))) / 100.0;
    for (0..64) |d| {
        for (0..64) |i| {
            if (d == 0 or i == 0) {
                table[d][i] = 0;
                continue;
            }
            const ld = @log(@as(f64, @floatFromInt(d)));
            const li = @log(@as(f64, @floatFromInt(i)));
            const r = @floor(base + ld * li / div);
            table[d][i] = if (r < 0) 0 else if (r > 32) 32 else @intFromFloat(r);
        }
    }
    return table;
}

const checkmate_score: i32 = 100000;
const max_ply: usize = 64;
const mate_score_threshold: i32 = checkmate_score - max_ply;
const alpha_init: i32 = std.math.minInt(i32) + 1;
const beta_init: i32 = std.math.maxInt(i32);
const max_threads: usize = 16;
const default_threads: usize = 8;
const max_game_length: usize = 1024;

// Position history for threefold repetition detection.
// Stores Zobrist hashes of positions since game start
pub const PositionHistory = struct {
    hashes: [max_game_length]u64 = undefined,
    len: usize = 0,

    pub fn push(self: *PositionHistory, hash: u64) void {
        if (self.len < max_game_length) {
            self.hashes[self.len] = hash;
            self.len += 1;
        }
    }

    pub fn pop(self: *PositionHistory) void {
        if (self.len > 0) {
            self.len -= 1;
        }
    }

    // Check if current position is a repetition.
    // halfmove_clock tells us how far back we need to look since last irreversible move.
    // For search: returns true for 2-fold (implies 3-fold in game context).
    // For game: set require_threefold=true to check for actual threefold
    pub fn isRepetition(self: *const PositionHistory, hash: u64, halfmove_clock: u16, require_threefold: bool) bool {
        if (self.len < 5) return false; // Need at least 5 positions for a repetition

        // Only look back halfmove_clock positions (since last irreversible move)
        // Repetition can only occur with positions that share the same side to move,
        // which means we check every 2 ply (4 half-moves minimum for one repetition)
        // Current position is at index len-1, we check indices len-1-4, len-1-6, etc.
        const max_lookback: usize = @min(@as(usize, halfmove_clock), self.len - 1);
        if (max_lookback < 4) return false;

        var count: u8 = 0;
        const target: u8 = if (require_threefold)
            2
        else
            1;

        // Check every 2 positions (same side to move)
        // i represents how many ply back from current position (len - 1)
        var i: usize = 4;
        while (i <= max_lookback) : (i += 2) {
            const idx = self.len - 1 - i;
            if (self.hashes[idx] == hash) {
                count += 1;
                if (count >= target) return true;
            }
        }
        return false;
    }

    pub fn isThreefold(self: *const PositionHistory, hash: u64, halfmove_clock: u16) bool {
        return self.isRepetition(hash, halfmove_clock, true);
    }

    pub fn isTwofold(self: *const PositionHistory, hash: u64, halfmove_clock: u16) bool {
        return self.isRepetition(hash, halfmove_clock, false);
    }
};

// Killer move table: stores 2 killer moves per ply.
// Killer moves are quiet moves that caused beta cutoffs.
const KillerTable = struct {
    moves: [max_ply][2]?Move = [_][2]?Move{.{ null, null }} ** max_ply,

    fn store(self: *KillerTable, ply: usize, m: Move) void {
        if (ply >= max_ply) return;

        // Don't store if it's already the first killer
        if (self.moves[ply][0]) |k| {
            if (m.eql(k)) return;
        }
        // Shift first killer to second slot, store new as first
        self.moves[ply][1] = self.moves[ply][0];
        self.moves[ply][0] = m;
    }

    fn isKiller(self: *const KillerTable, ply: usize, m: Move) bool {
        if (ply >= max_ply) return false;

        if (self.moves[ply][0]) |k| {
            if (m.eql(k)) return true;
        }
        if (self.moves[ply][1]) |k| {
            if (m.eql(k)) return true;
        }
        return false;
    }

    fn clear(self: *KillerTable) void {
        self.moves = [_][2]?Move{.{ null, null }} ** max_ply;
    }
};

// Countermove table: stores the move that refuted the opponent's previous move.
// Indexed by [from_square][to_square] of the previous move.
pub const CountermoveTable = struct {
    table: [64][64]?Move = [_][64]?Move{[_]?Move{null} ** 64} ** 64,

    pub fn store(self: *CountermoveTable, prev_move: Move, counter: Move) void {
        self.table[prev_move.start][prev_move.end] = counter;
    }

    pub fn get(self: *const CountermoveTable, prev_move: Move) ?Move {
        return self.table[prev_move.start][prev_move.end];
    }

    pub fn isCountermove(self: *const CountermoveTable, prev_move: ?Move, m: Move) bool {
        if (prev_move) |pm| {
            if (self.table[pm.start][pm.end]) |cm| {
                return cm.start == m.start and cm.end == m.end;
            }
        }
        return false;
    }
};

const TranspositionFlag = enum(u2) {
    empty = 0,
    exact = 1,
    lowerBound = 2,
    upperBound = 3,
};

const TranspositionEntry = struct {
    hash: u64 = 0,
    score: i32 = 0,
    depth: u8 = 0,
    flag: TranspositionFlag = .empty,
    best_move: ?Move = null,
};

// Lock-free transposition table entry packed into two 64-bit words
// Word 1: hash XOR data (for validation)
// Word 2: data (score:16, depth:8, flag:2, move_start:6, move_end:6, move_valid:1,
//          is_promotion:1, promotion_piece:3 = 43 bits)
const builtin = @import("builtin");
const is_wasm = builtin.target.cpu.arch == .wasm32;

// For WASM, provide aliased methods to allow the "generic" type to function.
const TTWord = if (is_wasm) struct {
    const Self = @This();

    raw: u64 = 0,

    fn init(v: u64) Self {
        return .{ .raw = v };
    }

    fn load(self: *const Self, _: std.builtin.AtomicOrder) u64 {
        return self.raw;
    }

    fn store(self: *Self, v: u64, _: std.builtin.AtomicOrder) void {
        self.raw = v;
    }
} else Atomic(u64);

// wasm32 has no 64-bit atomics, but the wasm path is single-threaded, so a plain
// counter with the same method surface (load/fetchAdd) is a safe stand-in.
const AtomicCounter = if (is_wasm) struct {
    const Self = @This();

    raw: u64 = 0,

    fn init(v: u64) Self {
        return .{ .raw = v };
    }

    fn load(self: *const Self, _: std.builtin.AtomicOrder) u64 {
        return self.raw;
    }

    fn fetchAdd(self: *Self, v: u64, _: std.builtin.AtomicOrder) u64 {
        const old = self.raw;
        self.raw += v;
        return old;
    }
} else Atomic(u64);

const PackedTTEntry = struct {
    key: TTWord = TTWord.init(0),
    data: TTWord = TTWord.init(0),

    // bits 0-15: score (as u16, offset by 32768 to handle negatives)
    // bits 16-23: depth
    // bits 24-25: flag
    // bits 26-31: move start
    // bits 32-37: move end
    // bit 38: move valid
    // bit 39: is_promotion
    // bits 40-42: promotion_piece
    // bits 43-50: generation
    fn pack(hash: u64, score: i32, depth: u8, flag: TranspositionFlag, best_move: ?Move, generation: u8) struct { key: u64, data: u64 } {
        const clamped_score = std.math.clamp(score, std.math.minInt(i16), std.math.maxInt(i16));
        const score_u: u16 = @bitCast(@as(i16, @intCast(clamped_score)));
        var data: u64 = score_u;
        data |= @as(u64, depth) << 16;
        data |= @as(u64, @intFromEnum(flag)) << 24;
        if (best_move) |m| {
            data |= @as(u64, m.start) << 26;
            data |= @as(u64, m.end) << 32;
            data |= @as(u64, 1) << 38;
            if (m.is_promotion) {
                data |= @as(u64, 1) << 39;
                data |= @as(u64, m.promotion_piece) << 40;
            }
        }
        data |= @as(u64, generation) << 43;

        const key = hash ^ data;
        return .{ .key = key, .data = data };
    }

    fn unpack(key: u64, data: u64, hash: u64) ?TranspositionEntry {
        // Key XOR data should equal original hash
        if ((key ^ data) != hash) return null;

        const score_u: u16 = @truncate(data);
        const score: i32 = @as(i16, @bitCast(score_u));
        const depth: u8 = @truncate(data >> 16);
        const flag: TranspositionFlag = @enumFromInt(@as(u2, @truncate(data >> 24)));
        const move_start: u6 = @truncate(data >> 26);
        const move_end: u6 = @truncate(data >> 32);
        const move_valid: u1 = @truncate(data >> 38);
        const is_promotion: u1 = @truncate(data >> 39);
        const promotion_piece: u3 = @truncate(data >> 40);

        if (flag == .empty) return null;

        return TranspositionEntry{
            .hash = hash,
            .score = score,
            .depth = depth,
            .flag = flag,
            .best_move = if (move_valid == 1) Move{
                .start = move_start,
                .end = move_end,
                .is_promotion = is_promotion == 1,
                .promotion_piece = promotion_piece,
            } else null,
        };
    }
};

// Adjust mate scores for TT storage: convert from ply-relative to root-relative
// A score of checkmate_score - ply means "mate in ply moves from here".
// When storing, we add ply so the stored score is distance from root.
// When retrieving, we subtract ply to get distance from the retrieval node.
fn scoreToTT(score: i32, ply: usize) i32 {
    const p: i32 = @intCast(ply);
    if (score > mate_score_threshold) return score + p;
    if (score < -mate_score_threshold) return score - p;
    return score;
}

fn scoreFromTT(score: i32, ply: usize) i32 {
    const p: i32 = @intCast(ply);
    if (score > mate_score_threshold) return score - p;
    if (score < -mate_score_threshold) return score + p;
    return score;
}

pub const TranspositionTable = struct {
    entries: []PackedTTEntry,
    alloc: std.mem.Allocator,
    generation: u8 = 0,
    bucket_mask: u64,

    // Multi-bucket transposition table: 4 entries per bucket = 1 cache line (64 bytes)
    const bucket_size: usize = 4;
    const default_buckets_bits: u6 = 18;

    pub fn init(alloc: std.mem.Allocator) !TranspositionTable {
        return initSized(alloc, default_buckets_bits);
    }

    // Allocate 2^buckets_bits buckets (bucket_size entries each). Larger tables
    // cut re-search at high depth; self-play uses this to size up.
    pub fn initSized(alloc: std.mem.Allocator, buckets_bits: u6) !TranspositionTable {
        const num_buckets: usize = @as(usize, 1) << @as(u5, @intCast(buckets_bits));
        const entries = try alloc.alloc(PackedTTEntry, num_buckets * bucket_size);
        @memset(entries, PackedTTEntry{});
        return .{ .entries = entries, .alloc = alloc, .bucket_mask = num_buckets - 1 };
    }

    pub fn newSearch(self: *TranspositionTable) void {
        self.generation +%= 1;
    }

    pub fn deinit(self: *TranspositionTable) void {
        self.alloc.free(self.entries);
    }

    // Prefetch the bucket cache line for an upcoming probe. The probe's load is
    // almost pure memory-latency stall (TT spills L2/L3), so issuing this as
    // soon as the hash is known — ahead of the intervening repetition/material
    // checks — hides part of the miss. No semantic effect.
    fn prefetch(self: *const TranspositionTable, hash: u64) void {
        const base: usize = @intCast((hash & self.bucket_mask) * bucket_size);
        @prefetch(&self.entries[base], .{ .rw = .read, .locality = 3, .cache = .data });
    }

    fn probe(self: *TranspositionTable, hash: u64) ?TranspositionEntry {
        const base: usize = @intCast((hash & self.bucket_mask) * bucket_size);

        for (0..bucket_size) |i| {
            const entry = &self.entries[base + i];
            const key = entry.key.load(.monotonic);
            const data = entry.data.load(.monotonic);

            if (PackedTTEntry.unpack(key, data, hash)) |result| {
                return result;
            }
        }
        return null;
    }

    fn store(self: *TranspositionTable, hash: u64, score: i32, depth: u8, flag: TranspositionFlag, best_move: ?Move) void {
        const base: usize = @intCast((hash & self.bucket_mask) * bucket_size);
        const gen = self.generation;

        var victim_idx: usize = base;
        var victim_score: i32 = std.math.maxInt(i32);
        var same_hash_idx: ?usize = null;

        // Scan all 4 entries in the bucket
        for (0..bucket_size) |i| {
            const idx = base + i;
            const entry = &self.entries[idx];
            const old_data = entry.data.load(.monotonic);
            const old_flag: TranspositionFlag = @enumFromInt(@as(u2, @truncate(old_data >> 24)));

            // Empty slot: use immediately
            if (old_flag == .empty) {
                victim_idx = idx;
                victim_score = std.math.minInt(i32);
                break;
            }

            // Check for same hash (XOR verification)
            const old_key = entry.key.load(.monotonic);
            if ((old_key ^ old_data) == hash) {
                same_hash_idx = idx;
                continue;
            }

            // Compute replacement score: depth - 4 * age
            const old_depth: u8 = @truncate(old_data >> 16);
            const old_gen: u8 = @truncate(old_data >> 43);
            const age: i32 = @intCast(gen -% old_gen);
            const rs: i32 = @as(i32, old_depth) - 4 * age;
            if (rs < victim_score) {
                victim_score = rs;
                victim_idx = idx;
            }
        }

        // Same hash: always replace, but preserve old best_move if new entry has none
        if (same_hash_idx) |idx| {
            const actual_move = if (best_move == null) blk: {
                const old_data = self.entries[idx].data.load(.monotonic);
                const move_valid: u1 = @truncate(old_data >> 38);
                if (move_valid == 1) {
                    break :blk @as(?Move, Move{
                        .start = @truncate(old_data >> 26),
                        .end = @truncate(old_data >> 32),
                        .is_promotion = @as(u1, @truncate(old_data >> 39)) == 1,
                        .promotion_piece = @truncate(old_data >> 40),
                    });
                }
                break :blk null;
            } else best_move;

            const p = PackedTTEntry.pack(hash, score, depth, flag, actual_move, gen);
            self.entries[idx].data.store(p.data, .monotonic);
            self.entries[idx].key.store(p.key, .monotonic);
            return;
        }

        // Use victim slot
        const p = PackedTTEntry.pack(hash, score, depth, flag, best_move, gen);
        self.entries[victim_idx].data.store(p.data, .monotonic);
        self.entries[victim_idx].key.store(p.key, .monotonic);
    }
};

pub const InfoCallback = struct {
    context: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, depth: u8, score: i32, nodes: u64, time_ms: u64, pv: []const Move) void,
};

pub const SearchOptions = struct {
    stop: ?*Atomic(bool) = null,
    max_time_ms: ?u64 = null,
    max_nodes: ?u64 = null,
    on_info: ?InfoCallback = null,
    search_params: SearchParams = .{},
};

// Minimal shared state for Lazy SMP - threads run independently
const SharedSearchState = struct {
    stop_flag: Atomic(bool) = Atomic(bool).init(false),
    node_count: AtomicCounter = AtomicCounter.init(0),
    // Move-ordering quality counters: total beta cutoffs and cutoffs on the
    // first move searched. first/total ≈ 85-92% indicates healthy ordering.
    cutoffs: AtomicCounter = AtomicCounter.init(0),
    first_move_cutoffs: AtomicCounter = AtomicCounter.init(0),
    max_depth: u8 = 0,
    // Monotonic nanoseconds at search start, for elapsed-time measurement.
    // See clock.nowNanos (0 on freestanding/wasm, which has no time control).
    start_ns: i96 = 0,
    options: SearchOptions = .{},
    network: ?*const nnue.Network = null,
    search_params: SearchParams = .{},
    // LMR reductions [depth][move_index], filled per search from search_params.
    // Zero until set (qsearch-only shared states never use it).
    lmr_table: [64][64]u8 = [_][64]u8{[_]u8{0} ** 64} ** 64,

    fn futilityMargin(self: *const SharedSearchState, depth: u8) i32 {
        return switch (depth) {
            0 => 0,
            1 => self.search_params.futility_margin_1,
            2 => self.search_params.futility_margin_2,
            else => 0,
        };
    }

    fn evalPosition(
        self: *const SharedSearchState,
        state: *const State,
        acc_stack: *nnue.AccumulatorStack,
        ply: usize,
    ) i32 {
        // NNUE returns centipawns (pawn ≈ 100). HCE uses an internal scale
        // (pawn endgame ≈ 208). Pruning margins in this file (futility_margins,
        // delta_margin, 80*depth in RFP) are tuned for the HCE scale, so scale
        // NNUE up by 2 to keep them approximately calibrated. Fine-tuning is
        // a Phase 6 concern.
        if (self.network) |net| {
            var v = nnue.evaluateLazy(state, net, acc_stack, ply) * self.search_params.nnue_scale;

            // Material scaling: shrink eval as non-pawn material disappears.
            // phase uses the same weights as the tapered eval (N=B=1, R=2, Q=4).
            const phase: i32 = @min(@as(i32, @intCast(state.pieceBitboard(piece.knight).popCount() +
                state.pieceBitboard(piece.bishop).popCount() +
                2 * state.pieceBitboard(piece.rook).popCount() +
                4 * state.pieceBitboard(piece.queen).popCount())), max_phase);
            const min = self.search_params.material_scale_min;
            const material_factor = min + @divTrunc((100 - min) * phase, max_phase);
            v = @divTrunc(v * material_factor, 100);

            // 50-move damping: pull eval toward 0 as the halfmove clock climbs.
            const start = self.search_params.fifty_move_start;
            const hmc: i32 = @min(@as(i32, state.halfmove_clock), 100);
            const over = @max(hmc - start, 0);
            const fifty_factor = 100 - @divTrunc(over * self.search_params.fifty_move_damp, @max(1, 100 - start));
            v = @divTrunc(v * fifty_factor, 100);

            return v;
        }
        return evaluation.evaluate(state);
    }
};

// Check time limit and external stop signal periodically.
// Called every N nodes from within negamax/quiescence to enable mid-depth abort.
fn checkTime(shared: *SharedSearchState) void {
    if (shared.options.stop) |ext_stop| {
        if (ext_stop.load(.monotonic)) {
            shared.stop_flag.store(true, .monotonic);
            return;
        }
    }
    if (shared.options.max_time_ms) |max_ms| {
        const elapsed: u64 = @intCast(@divTrunc(
            clock.nowNanos() - shared.start_ns,
            std.time.ns_per_ms,
        ));
        if (elapsed >= max_ms) {
            shared.stop_flag.store(true, .monotonic);
        }
    }
    if (shared.options.max_nodes) |max_nodes| {
        if (shared.node_count.load(.monotonic) >= max_nodes) {
            shared.stop_flag.store(true, .monotonic);
        }
    }
}

// Sentinel for "no static eval at this ply" (in check, or out of stack range).
const no_eval: i32 = std.math.minInt(i32);

// Per-ply search stack: one entry per ply, shared across a thread's search.
// Holds the data later phases need - static eval (for `improving` and
// improving-aware pruning) and the (color,piece,to) of the move made at this
// ply (for continuation history).
const StackEntry = struct {
    static_eval: i32 = no_eval,
    piece_to: u16 = 0, // (color*6 + piece)*64 + to of the move made here
    moved_valid: bool = false, // false for null move / root
};

// "improving": is the side-to-move's static eval better than two plies ago?
// Later phases prune less when improving. Per the plan: false in check (current
// node has no eval) or at ply < 2; optimistic-true when only the grandparent
// eval is missing.
fn isImproving(stack: []const StackEntry, ply: usize) bool {
    if (ply < 2) return false;
    const cur = stack[ply].static_eval;
    if (cur == no_eval) return false;
    const prev = stack[ply - 2].static_eval;
    if (prev == no_eval) return true;
    return cur > prev;
}

// Per-thread context for search
const ThreadContext = struct {
    state: State,
    killers: KillerTable,
    history: PositionHistory,
    history_table: evaluation.HistoryTable,
    countermoves: CountermoveTable,
    search_stack: [max_ply + 1]StackEntry = [_]StackEntry{.{}} ** (max_ply + 1),
    cont_hist: *evaluation.ContHistTable,
    thread_id: usize,
    tbl: *TranspositionTable,
    shared: *SharedSearchState,
    acc_stack: nnue.AccumulatorStack,
    best_move: ?Move = null,
    best_score: i32 = std.math.minInt(i32) + 1,
    best_depth: u8 = 0,
};

// Quiescence search: search only captures until the position is "quiet"
// This prevents the horizon effect where we evaluate positions mid-tactical-sequence
fn quiescence(
    state: *State,
    ply: usize,
    alpha_initial: i32,
    beta: i32,
    shared: *SharedSearchState,
    acc_stack: *nnue.AccumulatorStack,
) i32 {
    const nodes = shared.node_count.fetchAdd(1, .monotonic);

    if (nodes & 2047 == 0) checkTime(shared);
    if (shared.stop_flag.load(.monotonic)) return 0;

    const to_move = state.to_move;
    const in_check = state.in_check == to_move;

    // When in check, we must search all evasions, not just captures
    if (in_check) {
        var moves = movegen.legalMoves(state, to_move);

        if (moves.len == 0) {
            return -checkmate_score + @as(i32, @intCast(ply));
        }

        const ctx = MoveList.SortCtx{ .state = state, .color = to_move, .killers = .{ null, null }, .history = null };
        moves.scoreAll(&ctx);

        var alpha = alpha_initial;
        for (0..moves.len) |i| {
            const m = moves.pickNext(i);
            const p = state.mailbox[m.start].?;
            const undo = state.makeMove(m, to_move, p);
            if (shared.network) |net| {
                nnue.recordMove(acc_stack, ply + 1, state, net, m, to_move, p, &undo);
            }

            const score = -quiescence(state, ply + 1, -beta, -alpha, shared, acc_stack);

            state.unmakeMove(m, to_move, p, undo);

            if (score >= beta) {
                return beta;
            }
            if (score > alpha) {
                alpha = score;
            }
        }
        return alpha;
    }

    // Stand pat: evaluate the current position
    // We can always choose not to capture when not in check
    const stand_pat = shared.evalPosition(state, acc_stack, ply);

    // Beta cutoff: position is so good opponent wouldn't allow it
    if (stand_pat >= beta) {
        return beta;
    }

    var alpha = alpha_initial;

    // If stand pat is better than alpha, we can use it as a floor
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    // Generate and search captures only
    var captures = movegen.legalCaptures(state, to_move);

    if (captures.len == 0) {
        return stand_pat;
    }

    const cap_ctx = MoveList.SortCtx{
        .state = state,
        .color = to_move,
        .killers = .{ null, null },
        .history = null,
    };
    captures.scoreAll(&cap_ctx);

    for (0..captures.len) |i| {
        const m = captures.pickNext(i);
        const p = state.mailbox[m.start].?;

        // SEE + delta pruning: skip captures that can't help. Not reached when
        // in check (that path is handled above and searches all evasions).
        if (state.mailbox[m.end]) |captured_piece| {
            // SEE pruning: drop captures that lose material outright.
            if (seeLoses(state, m, 0)) continue;
            // Delta pruning: skip captures that can't possibly improve alpha.
            var gain = evaluation.piece_values_mg[captured_piece];
            if (m.is_promotion) {
                gain += evaluation.piece_values_mg[m.promotion_piece] - evaluation.piece_values_mg[piece.pawn];
            }
            if (stand_pat + gain + shared.search_params.delta_margin < alpha) {
                continue;
            }
        }

        const undo = state.makeMove(m, to_move, p);
        if (shared.network) |net| {
            nnue.recordMove(acc_stack, ply + 1, state, net, m, to_move, p, &undo);
        }

        const score = -quiescence(state, ply + 1, -beta, -alpha, shared, acc_stack);

        state.unmakeMove(m, to_move, p, undo);

        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    return alpha;
}

// Late Move Pruning thresholds: at depth d, prune quiet moves after this many moves
const lmp_thresholds = [4]u8{ 4, 5, 7, 11 };

// SEE pruning of losing captures in the main search at shallow depth: at
// depth <= see_prune_depth, skip non-first captures that lose more than
// see_prune_margin SEE units (pawn ~= 126). Tunable for screening.
const see_prune_depth: i32 = 3;
const see_prune_margin: i32 = 0;

// True when capturing move `m` loses material per static exchange evaluation:
// its SEE is below `threshold`. Only a higher-value attacker can lose (victim
// >= attacker => SEE >= 0), so the exchange is evaluated only there. Caller
// guarantees `m` captures a piece sitting on m.end.
fn seeLoses(state: *const State, m: Move, threshold: i32) bool {
    const attacker = state.mailbox[m.start].?;
    const victim = state.mailbox[m.end].?;
    return evaluation.piece_values_mg[attacker] > evaluation.piece_values_mg[victim] and
        movegen.staticExchangeEvaluation(state, m) < threshold;
}

const SearchContext = struct {
    tt: *TranspositionTable,
    killers: *KillerTable,
    history: *PositionHistory,
    history_table: *evaluation.HistoryTable,
    countermoves: *CountermoveTable,
    prev_move: ?Move,
    shared: *SharedSearchState,
    acc_stack: *nnue.AccumulatorStack,
    stack: *[max_ply + 1]StackEntry,
    cont_hist: *evaluation.ContHistTable,
};

fn negamax(
    state: *State,
    depth_param: u8,
    ply: usize,
    alpha_initial: i32,
    beta_param: i32,
    search_ctx: SearchContext,
) i32 {
    const nodes = search_ctx.shared.node_count.fetchAdd(1, .monotonic);
    if (nodes & 2047 == 0) checkTime(search_ctx.shared);
    if (search_ctx.shared.stop_flag.load(.monotonic)) return 0;

    // Mutable so Internal Iterative Reductions can lower it after the TT probe.
    var depth = depth_param;
    const hash = state.zobrist_hash;
    // Kick off the TT bucket fetch now; the repetition/material checks below
    // run while the line travels from L3/memory, hiding part of the probe miss.
    search_ctx.tt.prefetch(hash);
    var alpha = alpha_initial;
    var best_move: ?Move = null;

    // Check for repetition - return draw score (0) if position occurred before
    // We check for twofold since we're in the search tree (implies threefold in game)
    if (ply > 0 and search_ctx.history.isTwofold(hash, state.halfmove_clock)) {
        return 0;
    }

    if (state.hasInsufficientMaterial()) {
        return 0;
    }

    // Probe transposition table
    var beta = beta_param;
    var tt_move: ?Move = null;
    if (search_ctx.tt.probe(hash)) |entry| {
        tt_move = entry.best_move;
        if (entry.depth >= depth) {
            const tt_score = scoreFromTT(entry.score, ply);
            switch (entry.flag) {
                .exact => return tt_score,
                .lowerBound => alpha = @max(alpha, tt_score),
                .upperBound => beta = @min(beta, tt_score),
                .empty => {},
            }
            if (alpha >= beta) {
                return tt_score;
            }
        }
    }

    const to_move = state.to_move;

    // At depth 0, drop into quiescence search immediately.
    // Quiescence handles checkmate detection when in check.
    // This avoids generating a full MoveList at the most numerous nodes.
    if (depth == 0) {
        return quiescence(state, ply, alpha, beta, search_ctx.shared, search_ctx.acc_stack);
    }

    const in_check = state.in_check == to_move;
    const is_pv_node = beta_param - alpha_initial > 1;

    // Compute static eval at every non-check node so `improving` and
    // continuation-aware pruning have it at all depths; record it in the
    // per-ply stack for later phases.
    const static_eval: ?i32 = if (!in_check)
        search_ctx.shared.evalPosition(state, search_ctx.acc_stack, ply)
    else
        null;
    if (ply < max_ply) {
        search_ctx.stack[ply].static_eval = static_eval orelse no_eval;
    }

    // Improving: is our static eval better than two plies ago? Used to reduce
    // less (LMR) when the position is trending our way. Gated on ply < max_ply
    // to match the static_eval write above: beyond it stack[ply] is stale, and
    // at ply == max_ply + 1 it is out of bounds (stack has max_ply + 1 entries).
    const improving = ply < max_ply and isImproving(search_ctx.stack, ply);

    // Reverse futility pruning (static null move pruning):
    // If eval is far above beta, the position is so good we can prune.
    // Gated to depth <= 6 to preserve prior behavior (eval used to be computed
    // only at depths 1-6, so RFP only ever fired there).
    if (!is_pv_node and depth <= 6) {
        if (static_eval) |eval| {
            if (eval - search_ctx.shared.search_params.rfp_base * @as(i32, depth) >= beta) {
                return eval;
            }
        }
    }

    // Internal Iterative Reductions: with no TT move the ordering is unreliable,
    // so search one ply shallower (the reduced search also seeds the TT).
    if (tt_move == null and @as(i32, depth) >= search_ctx.shared.search_params.iir_min_depth) {
        depth -= 1;
    }

    var moves = movegen.legalMoves(state, to_move);

    if (moves.len == 0) {
        if (in_check)
            return -checkmate_score + @as(i32, @intCast(ply))
        else
            return 0;
    }

    // Futility pruning setup: at shallow depths, if static eval is far below alpha,
    // we can skip quiet moves that are unlikely to improve.
    const can_futility_prune = if (static_eval) |eval|
        depth <= 2 and eval + search_ctx.shared.futilityMargin(depth) <= alpha
    else
        false;

    // Null move pruning: if giving opponent a free move still results in beta cutoff,
    // the position is so good we can prune.
    if (depth >= 3 and !in_check and state.hasNonPawnMaterial(to_move)) {
        const keys = State.getZobristKeys();
        // Save state for null move
        const old_to_move = state.to_move;
        const old_ep = state.en_passant;
        const old_hash = state.zobrist_hash;
        const old_in_check = state.in_check;

        // Make null move: flip side to move, clear en passant
        state.*.to_move = ~state.to_move;
        state.*.zobrist_hash ^= keys.side_to_move;
        if (state.en_passant) |ep| {
            state.*.zobrist_hash ^= keys.en_passant[ep % 8];
            state.*.en_passant = null;
        }
        state.*.in_check = null; // After null move, we're not giving check

        if (search_ctx.shared.network) |_| {
            nnue.recordNullMove(search_ctx.acc_stack, ply + 1);
        }

        // Adaptive reduction: R = 2 + depth/4
        const R: u8 = 2 + depth / 4;
        if (ply < max_ply) search_ctx.stack[ply].moved_valid = false;
        var null_ctx = search_ctx;
        null_ctx.prev_move = null;
        const null_score = -negamax(state, depth - 1 - R, ply + 1, -beta, -beta + 1, null_ctx);

        // Unmake null move
        state.*.to_move = old_to_move;
        state.*.en_passant = old_ep;
        state.*.zobrist_hash = old_hash;
        state.*.in_check = old_in_check;

        if (null_score >= beta) {
            return beta;
        }
    }

    // Order moves with killer, countermove, and history heuristics
    const ply_killers = if (ply < max_ply) search_ctx.killers.moves[ply] else [2]?Move{ null, null };
    const countermove = if (search_ctx.prev_move) |pm| search_ctx.countermoves.get(pm) else null;
    // Continuation history: condition on the previous move when it was a real
    // move (not null move / root) and within stack range.
    const prev1_valid = ply >= 1 and (ply - 1) < max_ply and search_ctx.stack[ply - 1].moved_valid;
    const prev1_pt: u16 = if (prev1_valid) search_ctx.stack[ply - 1].piece_to else 0;
    const sort_ctx = MoveList.SortCtx{
        .state = state,
        .color = to_move,
        .killers = ply_killers,
        .history = search_ctx.history_table,
        .countermove = countermove,
        .tt_move = tt_move,
        .cont1 = if (prev1_valid) search_ctx.cont_hist else null,
        .prev1_pt = prev1_pt,
    };
    moves.scoreAll(&sort_ctx);

    var max_score: i32 = std.math.minInt(i32);
    var quiets_tried: [256]struct { start: u6, end: u6, piece_to: u16 } = undefined;
    var num_quiets: usize = 0;

    for (0..moves.len) |i| {
        const m = moves.pickNext(i);
        const p = state.mailbox[m.start].?;
        // Piece-to index of this move, for continuation history / the stack.
        const cur_pt: u16 = (@as(u16, to_move) * 6 + @as(u16, p)) * 64 + @as(u16, m.end);
        // Combined history of this move (butterfly + 1-ply continuation), used
        // by LMR history adjustment. ~0 for captures (not stored there).
        const combined_hist: i32 = search_ctx.history_table.get(to_move, m.start, m.end) +
            (if (prev1_valid) search_ctx.cont_hist.get(prev1_pt, cur_pt) else 0);

        // Check if this is a capture before making the move (for LMR decision)
        const is_capture = state.mailbox[m.end] != null;
        const end_rank = m.end / 8;
        const is_promotion = p == piece.pawn and
            ((end_rank == 7 and to_move == engine.Colors.white) or (end_rank == 0 and to_move == engine.Colors.black));
        const is_killer = search_ctx.killers.isKiller(ply, m);

        // Futility pruning: skip quiet moves at shallow depths when hopeless
        // Note: we need to make the move first to check if it gives check
        const should_futility_prune = can_futility_prune and !is_capture and i > 0 and !is_promotion;

        // Late Move Pruning flag: at shallow depths, consider skipping late quiet moves
        const should_lmp = depth <= 3 and !in_check and i >= lmp_thresholds[depth] and
            !is_capture and !is_promotion and !is_killer;

        // History-based pruning: at shallow depths, skip late quiet moves whose
        // combined history is strongly negative (proven bad).
        const should_histprune = @as(i32, depth) <= search_ctx.shared.search_params.histprune_depth and
            !in_check and i > 0 and !is_capture and !is_promotion and !is_killer and
            combined_hist < -search_ctx.shared.search_params.histprune_margin * @as(i32, depth);

        // SEE pruning: at shallow depth, skip captures that lose material badly.
        // Evaluated on the pre-move state, so we prune before makeMove and skip
        // the accumulator update entirely. Unlike the quiet-move prunes below, a
        // losing capture is dropped even when it would give check.
        if (@as(i32, depth) <= see_prune_depth and is_capture and i > 0 and
            !in_check and !is_promotion and seeLoses(state, m, -see_prune_margin))
        {
            continue;
        }

        const undo = state.makeMove(m, to_move, p);
        if (search_ctx.shared.network) |net| {
            nnue.recordMove(search_ctx.acc_stack, ply + 1, state, net, m, to_move, p, &undo);
        }
        search_ctx.history.push(state.zobrist_hash);

        // Check extension: extend search by 1 ply when giving check
        const gives_check = state.in_check != null;

        // Apply pruning only if the move doesn't give check
        if (!gives_check) {
            if (should_futility_prune) {
                search_ctx.history.pop();
                state.unmakeMove(m, to_move, p, undo);
                continue;
            }
            if (should_lmp) {
                search_ctx.history.pop();
                state.unmakeMove(m, to_move, p, undo);
                continue;
            }
            if (should_histprune) {
                search_ctx.history.pop();
                state.unmakeMove(m, to_move, p, undo);
                continue;
            }
        }

        const extension: u8 = if (gives_check)
            1
        else
            0;
        const new_depth = depth - 1 + extension;

        // Record the move made at this ply (for continuation history, consumed
        // by later phases) and thread the child context via copy-and-modify so
        // future field additions stay one-liners.
        if (ply < max_ply) {
            search_ctx.stack[ply].piece_to = cur_pt;
            search_ctx.stack[ply].moved_valid = true;
        }
        var child_ctx = search_ctx;
        child_ctx.prev_move = m;

        var score: i32 = undefined;
        if (i == 0) {
            // First move: search with full window
            score = -negamax(state, new_depth, ply + 1, -beta, -alpha, child_ctx);
        } else {
            // Late Move Reductions (LMR):
            // Moves ordered later are likely worse, so search with reduced depth first.
            // Only reduce quiet moves at sufficient depth that don't give check.
            var reduction: u8 = 0;
            if (i >= 3 and depth >= 3 and !is_capture and !gives_check and !in_check) {
                // Log-based base reduction, precomputed in shared.lmr_table.
                var r: i32 = search_ctx.shared.lmr_table[@min(depth, 63)][@min(i, 63)];
                // Runtime adjustments: reduce less on PV nodes, when improving,
                // and for moves with good history; more for bad history.
                if (is_pv_node) r -= 1;
                if (improving) r -= 1;
                const hist_div = @max(1, search_ctx.shared.search_params.lmr_hist_div);
                r -= std.math.clamp(@divTrunc(combined_hist, hist_div), -2, 2);
                if (r < 0) r = 0;
                reduction = @intCast(@min(r, 63));
                // Don't reduce into qsearch
                if (reduction >= new_depth) {
                    reduction = if (new_depth > 1) new_depth - 1 else 0;
                }
            }

            // PVS with LMR: search with reduced depth and null window
            score = -negamax(state, new_depth - reduction, ply + 1, -alpha - 1, -alpha, child_ctx);

            // Re-search at full depth if reduced search improved alpha
            if (score > alpha and reduction > 0) {
                score = -negamax(state, new_depth, ply + 1, -alpha - 1, -alpha, child_ctx);
            }

            // Re-search with full window if null window failed high
            if (score > alpha and score < beta) {
                score = -negamax(state, new_depth, ply + 1, -beta, -alpha, child_ctx);
            }
        }

        // Track if it was a capture (for killer move storage)
        const was_capture = is_capture or undo.captured_piece != null; // en passant

        search_ctx.history.pop();
        state.unmakeMove(m, to_move, p, undo);

        if (!was_capture and !is_promotion) {
            quiets_tried[num_quiets] = .{ .start = m.start, .end = m.end, .piece_to = cur_pt };
            num_quiets += 1;
        }

        if (score > max_score) {
            max_score = score;
            best_move = m;
        }
        alpha = @max(alpha, score);

        if (alpha >= beta) {
            // Move-ordering quality: count this cutoff, and whether it came on
            // the first move searched (ideal ordering).
            _ = search_ctx.shared.cutoffs.fetchAdd(1, .monotonic);
            if (i == 0) _ = search_ctx.shared.first_move_cutoffs.fetchAdd(1, .monotonic);

            // Beta cutoff - store killer, countermove, and update history for quiet moves
            if (!was_capture) {
                const bonus: i32 = @as(i32, depth) * @as(i32, depth);
                search_ctx.killers.store(ply, m);
                search_ctx.history_table.update(to_move, m.start, m.end, bonus);
                if (prev1_valid) search_ctx.cont_hist.update(prev1_pt, cur_pt, bonus);
                // Malus: penalize all quiet moves tried before the cutoff move
                // If cutoff move is quiet it's the last entry in quiets_tried: skip it.
                // If cutoff move is a promotion it's not in quiets_tried: penalize all.
                const malus_count = if (!is_promotion) num_quiets - 1 else num_quiets;
                for (0..malus_count) |qi| {
                    search_ctx.history_table.update(to_move, quiets_tried[qi].start, quiets_tried[qi].end, -bonus);
                    if (prev1_valid) search_ctx.cont_hist.update(prev1_pt, quiets_tried[qi].piece_to, -bonus);
                }
                // Store countermove: this move refutes opponent's previous move
                if (search_ctx.prev_move) |pm| {
                    search_ctx.countermoves.store(pm, m);
                }
            }
            break;
        }
    }

    // Determine flag for TT entry
    const flag: TranspositionFlag = if (max_score <= alpha_initial)
        .upperBound
    else if (max_score >= beta)
        .lowerBound
    else
        .exact;

    search_ctx.tt.store(hash, scoreToTT(max_score, ply), depth, flag, best_move);

    return max_score;
}

pub const SearchResult = struct {
    move: Move,
    score: i32,
    depth: u8,
    // Search instrumentation (populated only at the top-level return).
    // Intermediate per-depth results leave these at the defaults.
    nodes: u64 = 0,
    cutoffs: u64 = 0,
    first_move_cutoffs: u64 = 0,
};

// Search at a specific depth with an optional hint for the best move from the previous iteration
// alpha_bound and beta_bound allow aspiration windows when not at full window
fn searchAtDepthWithBounds(
    state: *State,
    depth: u8,
    tbl: *TranspositionTable,
    killers: *KillerTable,
    pv_move: ?Move,
    history: *PositionHistory,
    history_table: *evaluation.HistoryTable,
    countermoves: *CountermoveTable,
    alpha_bound: i32,
    beta_bound: i32,
    shared: *SharedSearchState,
    acc_stack: *nnue.AccumulatorStack,
    stack: *[max_ply + 1]StackEntry,
    cont_hist: *evaluation.ContHistTable,
) ?SearchResult {
    var best_score: i32 = std.math.minInt(i32);
    var best_move: ?Move = null;
    var alpha = alpha_bound;

    const to_move = state.to_move;
    var moves = movegen.legalMoves(state, to_move);

    if (moves.len == 0) {
        return null;
    }

    const root_killers = if (0 < max_ply) killers.moves[0] else [2]?Move{ null, null };
    const root_ctx = MoveList.SortCtx{
        .state = state,
        .color = to_move,
        .killers = root_killers,
        .history = history_table,
    };
    moves.scoreAll(&root_ctx);

    // If we have a PV move from the previous iteration, give it max score
    // so pickNext selects it first
    if (pv_move) |pv| {
        for (0..moves.len) |i| {
            if (moves.moves[i].start == pv.start and moves.moves[i].end == pv.end) {
                moves.scores[i] = std.math.maxInt(i32);
                break;
            }
        }
    }

    for (0..moves.len) |i| {
        if (shared.stop_flag.load(.monotonic)) break;

        const m = moves.pickNext(i);
        const p = state.mailbox[m.start].?;
        const undo = state.makeMove(m, to_move, p);
        if (shared.network) |net| {
            nnue.recordMove(acc_stack, 1, state, net, m, to_move, p, &undo);
        }
        history.push(state.zobrist_hash);

        // Take mate in 1
        if (isGameOver(state)) |r| {
            switch (r) {
                .checkmate => if (r.checkmate == to_move) {
                    history.pop();
                    state.unmakeMove(m, to_move, p, undo);
                    return .{ .move = m, .score = checkmate_score - 1, .depth = depth };
                },
                else => {},
            }
        }

        // Record the root move so 1-ply continuation history is active at ply 1.
        stack[0].piece_to = (@as(u16, to_move) * 6 + @as(u16, p)) * 64 + @as(u16, m.end);
        stack[0].moved_valid = true;

        var score: i32 = undefined;
        const ctx = SearchContext{
            .tt = tbl,
            .killers = killers,
            .history = history,
            .history_table = history_table,
            .countermoves = countermoves,
            .prev_move = m,
            .shared = shared,
            .acc_stack = acc_stack,
            .stack = stack,
            .cont_hist = cont_hist,
        };
        if (i == 0) {
            // First move: full window search
            score = -negamax(state, depth - 1, 1, -beta_bound, -alpha, ctx);
        } else {
            // PVS: null window search first
            score = -negamax(state, depth - 1, 1, -alpha - 1, -alpha, ctx);
            // Re-search with full window if failed high
            if (score > alpha and score < beta_bound) {
                score = -negamax(state, depth - 1, 1, -beta_bound, -alpha, ctx);
            }
        }

        history.pop();
        state.unmakeMove(m, to_move, p, undo);

        if (score > best_score) {
            best_move = m;
            best_score = score;
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta_bound) {
            break; // Beta cutoff
        }
    }

    if (best_move) |m| {
        return .{
            .move = m,
            .score = best_score,
            .depth = depth,
        };
    } else {
        return null;
    }
}

// Search at a specific depth with full window
fn searchAtDepth(
    state: *State,
    depth: u8,
    tbl: *TranspositionTable,
    killers: *KillerTable,
    pv_move: ?Move,
    history: *PositionHistory,
    history_table: *evaluation.HistoryTable,
    countermoves: *CountermoveTable,
    shared: *SharedSearchState,
    acc_stack: *nnue.AccumulatorStack,
    stack: *[max_ply + 1]StackEntry,
    cont_hist: *evaluation.ContHistTable,
) ?SearchResult {
    return searchAtDepthWithBounds(
        state,
        depth,
        tbl,
        killers,
        pv_move,
        history,
        history_table,
        countermoves,
        alpha_init,
        beta_init,
        shared,
        acc_stack,
        stack,
        cont_hist,
    );
}

// Aspiration window initial size (centipawns)
// FIXME: Check/tune this.
const aspiration_window: i32 = 25;

// Extract the principal variation from the transposition table
fn extractPV(root_state: *const State, tbl: *TranspositionTable, buf: []Move) usize {
    var state = root_state.*;
    var count: usize = 0;
    var seen: [32]u64 = undefined;
    while (count < buf.len) {
        // Loop detection
        for (seen[0..count]) |h| if (h == state.zobrist_hash) return count;
        seen[count] = state.zobrist_hash;

        const entry = tbl.probe(state.zobrist_hash) orelse break;
        const best = entry.best_move orelse break;

        // Validate the move is legal
        var moves = movegen.legalMoves(&state, state.to_move);
        var found = false;
        for (0..moves.len) |i| {
            if (moves.moves[i].eql(best)) {
                found = true;
                break;
            }
        }
        if (!found) break;

        buf[count] = best;
        count += 1;
        const color = state.to_move;
        const piece_at = state.mailbox[best.start] orelse break;
        _ = state.makeMove(best, color, piece_at);
    }
    return count;
}

// Worker thread function for Lazy SMP
// Each thread does full iterative deepening independently
// Threads diverge naturally due to TT interactions and timing
fn workerThread(ctx: *ThreadContext) void {
    var pv_move: ?Move = null;
    var prev_score: i32 = 0;

    // Refresh the root accumulator once; subsequent plies update incrementally.
    if (ctx.shared.network) |net| {
        nnue.refreshAccumulator(&ctx.state, net, &ctx.acc_stack.accs[0]);
    }

    // Each thread does iterative deepening up to max_depth
    for (1..ctx.shared.max_depth + 1) |d| {
        if (ctx.shared.stop_flag.load(.monotonic)) break;

        const depth: u8 = @intCast(d);

        // Get PV hint from TT (may have been populated by other threads)
        const tt_move = if (ctx.tbl.probe(ctx.state.zobrist_hash)) |entry| entry.best_move else null;
        const hint = pv_move orelse tt_move;

        var result: ?SearchResult = null;

        // Use aspiration windows after depth 1
        if (depth > 1) {
            var window = aspiration_window;
            var alpha = prev_score - window;
            var beta = prev_score + window;
            var attempts: u8 = 0;

            while (attempts < 3) : (attempts += 1) {
                result = searchAtDepthWithBounds(
                    &ctx.state,
                    depth,
                    ctx.tbl,
                    &ctx.killers,
                    hint,
                    &ctx.history,
                    &ctx.history_table,
                    &ctx.countermoves,
                    alpha,
                    beta,
                    ctx.shared,
                    &ctx.acc_stack,
                    &ctx.search_stack,
                    ctx.cont_hist,
                );

                if (ctx.shared.stop_flag.load(.monotonic)) break;

                if (result) |r| {
                    if (r.score <= alpha) {
                        // Fail low: widen alpha
                        window *= 4;
                        alpha = prev_score - window;
                    } else if (r.score >= beta) {
                        // Fail high: widen beta
                        window *= 4;
                        beta = prev_score + window;
                    } else {
                        // Score within window, we're done
                        break;
                    }
                } else {
                    break;
                }
            }

            // If still failing after 3 attempts, do full window search
            if (!ctx.shared.stop_flag.load(.monotonic)) {
                if (result) |r| {
                    if (r.score <= alpha or r.score >= beta) {
                        result = searchAtDepth(
                            &ctx.state,
                            depth,
                            ctx.tbl,
                            &ctx.killers,
                            hint,
                            &ctx.history,
                            &ctx.history_table,
                            &ctx.countermoves,
                            ctx.shared,
                            &ctx.acc_stack,
                            &ctx.search_stack,
                            ctx.cont_hist,
                        );
                    }
                }
            }
        } else {
            // Depth 1: always use full window
            result = searchAtDepth(
                &ctx.state,
                depth,
                ctx.tbl,
                &ctx.killers,
                hint,
                &ctx.history,
                &ctx.history_table,
                &ctx.countermoves,
                ctx.shared,
                &ctx.acc_stack,
                &ctx.search_stack,
                ctx.cont_hist,
            );
        }

        // Discard results from an aborted depth - keep previous depth's result
        if (ctx.shared.stop_flag.load(.monotonic)) break;

        if (result) |r| {
            ctx.best_move = r.move;
            ctx.best_score = r.score;
            ctx.best_depth = depth;
            pv_move = r.move;
            prev_score = r.score;

            // Emit info from thread 0 only
            if (ctx.thread_id == 0) {
                if (ctx.shared.options.on_info) |cb| {
                    const nodes = ctx.shared.node_count.load(.monotonic);
                    const elapsed_ms: u64 = @intCast(@divTrunc(
                        clock.nowNanos() - ctx.shared.start_ns,
                        std.time.ns_per_ms,
                    ));
                    var pv_buf: [32]Move = undefined;
                    const pv_len = extractPV(&ctx.state, ctx.tbl, &pv_buf);
                    cb.func(cb.context, depth, r.score, nodes, elapsed_ms, pv_buf[0..pv_len]);
                }
            }

            // Early exit if checkmate found
            // FIXME: why -100? Check this
            if (r.score >= checkmate_score - 100) {
                ctx.shared.stop_flag.store(true, .monotonic);
                break;
            }
        }
    }
}

// Parallel search using Lazy SMP
// Spawns multiple threads that each do full iterative deepening
// Threads share TT but run independently (no barriers)
// game_history: optional history from the actual game (for repetition detection across search boundary)
pub fn searchParallel(
    state: *const State,
    max_depth: u8,
    num_threads: usize,
    game_history: ?*const PositionHistory,
    tbl: *TranspositionTable,
    options: SearchOptions,
    network: ?*const nnue.Network,
) !?SearchResult {
    const actual_threads = @min(num_threads, max_threads);

    tbl.newSearch();

    // Per-thread 1-ply continuation history. Heap-allocated because each table
    // is ~2.25 MB, far too large to embed in the stack-resident contexts array.
    // Zeroed per search like the other per-thread move-ordering tables.
    const cont_tables = try std.heap.page_allocator.alloc(evaluation.ContHistTable, actual_threads);
    defer std.heap.page_allocator.free(cont_tables);
    for (cont_tables) |*t| t.clear();

    var shared = SharedSearchState{
        .max_depth = max_depth,
        .start_ns = clock.nowNanos(),
        .options = options,
        .network = network,
        .search_params = options.search_params,
    };
    shared.lmr_table = computeLmrTable(shared.search_params.lmr_base, shared.search_params.lmr_div);

    // Create thread contexts
    var contexts: [max_threads]ThreadContext = undefined;
    for (0..actual_threads) |i| {
        // Initialize history with game history if provided
        var history = PositionHistory{};
        if (game_history) |gh| {
            for (0..gh.len) |j| {
                history.push(gh.hashes[j]);
            }
        } else {
            // Push the root position
            history.push(state.zobrist_hash);
        }

        contexts[i] = ThreadContext{
            .state = state.*,
            .killers = KillerTable{},
            .history = history,
            .history_table = evaluation.HistoryTable{},
            .countermoves = CountermoveTable{},
            .cont_hist = &cont_tables[i],
            .thread_id = i,
            .tbl = tbl,
            .shared = &shared,
            .acc_stack = undefined,
        };
        // Mark every accumulator dirty; the worker refreshes accs[0] before
        // its first eval and lazily fills the rest as deltas accumulate.
        for (&contexts[i].acc_stack.accs) |*acc| acc.computed = .{ false, false };
        contexts[i].acc_stack.debug_eval_count = 0;
    }

    // Spawn worker threads
    var threads: [max_threads]std.Thread = undefined;
    var spawned_threads: usize = 0;
    for (1..actual_threads) |i| {
        threads[i] = std.Thread.spawn(.{}, workerThread, .{&contexts[i]}) catch break;
        spawned_threads = i;
    }

    // Main thread also searches (as thread 0)
    workerThread(&contexts[0]);

    // Signal stop and join all threads
    shared.stop_flag.store(true, .release);
    for (1..spawned_threads + 1) |i| {
        threads[i].join();
    }

    // Collect best result from all threads
    var best_move: ?Move = null;
    var best_score: i32 = std.math.minInt(i32);
    var best_depth: u8 = 0;

    for (0..spawned_threads + 1) |i| {
        if (contexts[i].best_move != null) {
            // Prefer higher depth, then higher score
            if (contexts[i].best_depth > best_depth or
                (contexts[i].best_depth == best_depth and contexts[i].best_score > best_score))
            {
                best_move = contexts[i].best_move;
                best_score = contexts[i].best_score;
                best_depth = contexts[i].best_depth;
            }
        }
    }

    if (best_move) |m| {
        return .{
            .move = m,
            .score = best_score,
            .depth = best_depth,
            .nodes = shared.node_count.load(.monotonic),
            .cutoffs = shared.cutoffs.load(.monotonic),
            .first_move_cutoffs = shared.first_move_cutoffs.load(.monotonic),
        };
    } else {
        return null;
    }
}

// Reusable single-threaded searcher for high-throughput callers (self-play) that
// run one search per move across millions of moves. searchParallel allocates and
// zeroes a 2.25 MB continuation-history table and recomputes the 64x64 LMR table
// (thousands of @log calls) on every call; here both are owned once and reused,
// so the per-move fixed cost drops to clearing already-resident memory. Search
// behavior is identical to searchParallel(state, depth, 1, ...).
pub const ReusableSearcher = struct {
    cont_hist: *evaluation.ContHistTable,
    lmr_table: [64][64]u8,
    alloc: std.mem.Allocator,

    // search_params must match the options.search_params later passed to search()
    // so the precomputed LMR table stays consistent (self-play uses defaults).
    pub fn init(alloc: std.mem.Allocator, search_params: SearchParams) !ReusableSearcher {
        const cont = try alloc.create(evaluation.ContHistTable);
        return .{
            .cont_hist = cont,
            .lmr_table = computeLmrTable(search_params.lmr_base, search_params.lmr_div),
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *ReusableSearcher) void {
        self.alloc.destroy(self.cont_hist);
    }

    pub fn search(
        self: *ReusableSearcher,
        state: *const State,
        max_depth: u8,
        game_history: ?*const PositionHistory,
        tbl: *TranspositionTable,
        options: SearchOptions,
        network: ?*const nnue.Network,
    ) ?SearchResult {
        tbl.newSearch();
        self.cont_hist.clear();

        var shared = SharedSearchState{
            .max_depth = max_depth,
            .start_ns = clock.nowNanos(),
            .options = options,
            .network = network,
            .search_params = options.search_params,
            .lmr_table = self.lmr_table,
        };

        var history = PositionHistory{};
        if (game_history) |gh| {
            for (0..gh.len) |j| history.push(gh.hashes[j]);
        } else {
            history.push(state.zobrist_hash);
        }

        var ctx = ThreadContext{
            .state = state.*,
            .killers = KillerTable{},
            .history = history,
            .history_table = evaluation.HistoryTable{},
            .countermoves = CountermoveTable{},
            .cont_hist = self.cont_hist,
            .thread_id = 0,
            .tbl = tbl,
            .shared = &shared,
            .acc_stack = undefined,
        };
        for (&ctx.acc_stack.accs) |*acc| acc.computed = .{ false, false };
        ctx.acc_stack.debug_eval_count = 0;

        workerThread(&ctx);

        if (ctx.best_move) |m| {
            return .{
                .move = m,
                .score = ctx.best_score,
                .depth = ctx.best_depth,
                .nodes = shared.node_count.load(.monotonic),
                .cutoffs = shared.cutoffs.load(.monotonic),
                .first_move_cutoffs = shared.first_move_cutoffs.load(.monotonic),
            };
        }
        return null;
    }
};

pub fn search(state: *const State, max_depth: u8) !?SearchResult {
    var tbl = try TranspositionTable.init(std.heap.page_allocator);
    return searchParallel(state, max_depth, default_threads, null, &tbl, .{}, null);
}

// Search with game history for repetition detection
pub fn searchWithHistory(
    state: *const State,
    max_depth: u8,
    num_threads: usize,
    history: *const PositionHistory,
    tbl: *TranspositionTable,
    network: ?*const nnue.Network,
) !?SearchResult {
    return searchParallel(state, max_depth, num_threads, history, tbl, .{}, network);
}

// Single-threaded search for testing and debugging.
// `network` is the NNUE net to eval with, or null for HCE (used by the wasm path).
pub fn searchSingleThreaded(state: *const State, max_depth: u8, network: ?*const nnue.Network) !?SearchResult {
    var tbl = try TranspositionTable.init(std.heap.page_allocator);
    defer tbl.deinit();

    var killers = KillerTable{};
    var history = PositionHistory{};
    var history_table = evaluation.HistoryTable{};
    var countermoves = CountermoveTable{};
    history.push(state.zobrist_hash);

    // Make a mutable copy for the search (make/unmake will restore it)
    var mutable_state = state.*;

    var best_move: ?Move = null;
    var best_score: i32 = undefined;
    var best_depth: u8 = 0;

    var shared = SharedSearchState{
        .max_depth = max_depth,
        .start_ns = clock.nowNanos(),
        .network = network,
    };
    shared.lmr_table = computeLmrTable(shared.search_params.lmr_base, shared.search_params.lmr_div);

    var acc_stack: nnue.AccumulatorStack = undefined;
    for (&acc_stack.accs) |*acc| acc.computed = .{ false, false };
    acc_stack.debug_eval_count = 0;

    var search_stack: [max_ply + 1]StackEntry = [_]StackEntry{.{}} ** (max_ply + 1);

    const cont_hist = try std.heap.page_allocator.create(evaluation.ContHistTable);
    defer std.heap.page_allocator.destroy(cont_hist);
    cont_hist.clear();

    for (1..max_depth + 1) |depth| {
        const result = searchAtDepth(
            &mutable_state,
            @intCast(depth),
            &tbl,
            &killers,
            best_move,
            &history,
            &history_table,
            &countermoves,
            &shared,
            &acc_stack,
            &search_stack,
            cont_hist,
        );
        if (result) |r| {
            best_move = r.move;
            best_score = r.score;
            best_depth = r.depth;

            // Early exit if we found a checkmate
            if (best_score >= checkmate_score - 100) {
                break;
            }
        }
    }

    if (best_move) |m| {
        return .{
            .move = m,
            .score = best_score,
            .depth = best_depth,
            .nodes = shared.node_count.load(.monotonic),
            .cutoffs = shared.cutoffs.load(.monotonic),
            .first_move_cutoffs = shared.first_move_cutoffs.load(.monotonic),
        };
    } else {
        return null;
    }
}

// Standalone quiescence evaluation for use outside of search (e.g. quiet filtering).
// Runs qsearch with a full window and no time constraints.
pub fn quiescenceEval(state: *const State) i32 {
    var mutable_state = state.*;
    var shared = SharedSearchState{
        .max_depth = 0,
        .start_ns = clock.nowNanos(),
    };
    var acc_stack: nnue.AccumulatorStack = undefined;
    for (&acc_stack.accs) |*acc| acc.computed = .{ false, false };
    acc_stack.debug_eval_count = 0;
    return quiescence(&mutable_state, 0, -checkmate_score, checkmate_score, &shared, &acc_stack);
}

pub fn isGameOver(state: *const State) ?GameResult {
    return isGameOverWithHistory(state, null);
}

// Check for game over with optional position history for threefold repetition
pub fn isGameOverWithHistory(state: *const State, history: ?*const PositionHistory) ?GameResult {
    const to_move = state.to_move;
    const hasLegalMoves = movegen.hasAnyLegalMove(state, to_move);

    if (!hasLegalMoves) {
        const king_square = state.pieceBitboard(piece.king).bitAnd(Bitboard, state.colorBitboard(to_move)).trailingZeros();
        if (movegen.isSquareAttackedBy(state, king_square, ~to_move)) {
            return GameResult{ .checkmate = ~to_move };
        } else {
            return .stalemate;
        }
    }

    if (state.hasInsufficientMaterial()) {
        return .insufficientMaterial;
    }

    if (state.halfmove_clock >= 100) {
        return .fiftyMoveRule;
    }

    // Check for threefold repetition
    if (history) |h| {
        if (h.isThreefold(state.zobrist_hash, state.halfmove_clock)) {
            return .threefoldRepetition;
        }
    }

    return null;
}

test "test search finds mate in one" {
    const fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4";
    const state = try State.fromFen(fen);

    const result = try search(&state, 1);
    try expect(result != null);
    try expectEqual(square.h5, result.?.move.start);
    try expectEqual(square.f7, result.?.move.end);
    try expect(result.?.score > 50000);
}

test "test detects stalemate" {
    const fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1";
    const state = try State.fromFen(fen);

    const res = isGameOver(&state);
    try expectEqual(GameResult.stalemate, res);
}

// Mate in 2 tests
test "backrank mate in two" {
    // Rook on c2 can deliver mate: Qc8+ Rxc8 Rxc8#
    const fen = "r5k1/5ppp/8/2Q5/8/8/2R5/4K3 w - - 2 1";
    const state = try State.fromFen(fen);

    const result = try search(&state, 4);
    try expect(result != null);
    try expectEqual(square.c5, result.?.move.start);
    try expectEqual(square.c8, result.?.move.end);
}

test "mate in two" {
    // Rxh6+ Kxh6 Qg6#
    const fen = "r7/ppq2r1k/1npR3p/4PP2/6Q1/8/PP4PP/R6K w - - 0 1";
    const state = try State.fromFen(fen);

    const result = try search(&state, 4);
    try expect(result != null);
    try expectEqual(square.d6, result.?.move.start);
    try expectEqual(square.h6, result.?.move.end);
}

test "smothered mate pattern" {
    const fen = "4r2k/2pRP1pp/2p5/p4pN1/2Q3n1/q5P1/P3PP1P/6K1 w - - 0 1";
    const state = try State.fromFen(fen);

    const result = try search(&state, 6);
    try expect(result != null);
    try expectEqual(square.g5, result.?.move.start);
    try expectEqual(square.f7, result.?.move.end);
}

// Mate in 3 tests
test "search finds mate in three" {
    // Opera game finale position - Rd8+ leads to mate
    const fen = "1n2kb1r/p4ppp/4q3/4p1B1/4P3/3R4/PPP2PPP/2K5 w k - 1 17";
    const state = try State.fromFen(fen);

    const result = try search(&state, 6);
    try expect(result != null);
    // Should find Rd8+ leading to mate
    try expect(result.?.score > 50000);
}

test "search avoids stalemate when winning" {
    // White has queen and king vs lone king - should not stalemate
    const fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1";
    const state = try State.fromFen(fen);

    const result = try search(&state, 4);
    try expect(result != null);
    // Should find checkmate, not stalemate the opponent
    // Qf8# or similar mate
    try expect(result.?.score > 50000);
}

// Repetition detection tests
test "twofold repetition detection" {
    var history = PositionHistory{};

    // Push some positions
    history.push(0x1234);
    history.push(0x5678);
    history.push(0x9ABC);
    history.push(0xDEF0);
    history.push(0x1234); // Repetition of first position

    // Check: with halfmove_clock of 5 (all positions since last irreversible move),
    // we should find twofold repetition for 0x1234
    try expect(history.isTwofold(0x1234, 5));

    // Non-repeated position should not be detected
    try expect(!history.isTwofold(0xFFFF, 5));

    // 0x5678 has not repeated, should not be detected
    try expect(!history.isTwofold(0x5678, 5));
}

test "threefold repetition detection" {
    var history = PositionHistory{};

    // Simulate a game with repetitions:
    // For threefold, we need position A to appear 3 times at same-side-to-move positions
    // Same side to move = every 2 ply (indices 0, 2, 4, 6, ...)
    // So we need: A, B, A, B, A, B, A (7 positions, A at indices 0, 2, 4, 6)
    const pos_a: u64 = 0x1111;
    const pos_b: u64 = 0x2222;

    history.push(pos_a); // Position 0: A (1st occurrence)
    history.push(pos_b); // Position 1: B
    history.push(pos_a); // Position 2: A (2nd occurrence)
    history.push(pos_b); // Position 3: B
    history.push(pos_a); // Position 4: A (3rd occurrence)
    history.push(pos_b); // Position 5: B
    history.push(pos_a); // Position 6: A (4th occurrence - current)

    // Current position is at index 6 (A)
    // Checking 4 ply back: index 2 (A) - match! count=1
    // Checking 6 ply back: index 0 (A) - match! count=2 - threefold!
    try expect(history.isThreefold(pos_a, 7));

    // pos_b only appeared at odd indices (1, 3, 5) - wrong side to move
    try expect(!history.isThreefold(pos_b, 7));

    // Twofold should also work
    try expect(history.isTwofold(pos_a, 7));
}

test "repetition respects halfmove clock" {
    var history = PositionHistory{};

    // Push positions
    history.push(0x1111);
    history.push(0x2222);
    history.push(0x3333);
    history.push(0x4444);
    history.push(0x1111); // Repetition at position 4

    // With halfmove_clock = 2, we only look back 2 positions (indices 3, 4)
    // Position 0x1111 at index 0 is too far back
    try expect(!history.isTwofold(0x1111, 2));

    // With halfmove_clock = 5, we look back all 5 positions
    try expect(history.isTwofold(0x1111, 5));
}

test "repetition only checks same side to move" {
    var history = PositionHistory{};

    // In a real game, positions with same side to move are at even intervals
    // Push 6 positions: 0, 1, 2, 3, 4, 5
    history.push(0xAAAA); // Position 0 (white to move)
    history.push(0xBBBB); // Position 1 (black to move)
    history.push(0xCCCC); // Position 2 (white to move)
    history.push(0xDDDD); // Position 3 (black to move)
    history.push(0xAAAA); // Position 4 (white to move) - same as position 0

    // The algorithm checks every 2 positions (4-0=4, 4-2=2, etc.)
    // So it should find 0xAAAA at position 0 when checking from position 4
    try expect(history.isTwofold(0xAAAA, 5));

    // 0xBBBB at position 1 should not match anything checked from position 4
    // (we check positions 2, 0 - not 1, 3)
    try expect(!history.isTwofold(0xBBBB, 5));
}

test "ReusableSearcher matches single-threaded searchParallel" {
    // The self-play searcher must select the same move/score as
    // searchParallel(.., 1, ..) -- it is the same search with the per-move
    // scratch (cont-hist + LMR table) hoisted out of the hot path.
    const fens = [_][]const u8{
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    };
    for (fens) |fen| {
        const state = try State.fromFen(fen);

        var tbl_a = try TranspositionTable.init(std.testing.allocator);
        defer tbl_a.deinit();
        const a = (try searchParallel(&state, 6, 1, null, &tbl_a, .{}, null)).?;

        var searcher = try ReusableSearcher.init(std.testing.allocator, .{});
        defer searcher.deinit();
        var tbl_b = try TranspositionTable.init(std.testing.allocator);
        defer tbl_b.deinit();
        const b = searcher.search(&state, 6, null, &tbl_b, .{}, null).?;

        try expectEqual(a.move.start, b.move.start);
        try expectEqual(a.move.end, b.move.end);
        try expectEqual(a.score, b.score);
    }
}

test "isImproving semantics" {
    var stack: [max_ply + 1]StackEntry = [_]StackEntry{.{}} ** (max_ply + 1);

    // ply < 2 is never improving.
    try expect(!isImproving(&stack, 0));
    try expect(!isImproving(&stack, 1));

    // Current node has no eval (in check) => not improving.
    stack[2].static_eval = 50;
    stack[4].static_eval = no_eval;
    try expect(!isImproving(&stack, 4));

    // Grandparent eval missing => optimistic true.
    stack[2].static_eval = no_eval;
    stack[4].static_eval = 50;
    try expect(isImproving(&stack, 4));

    // Both present: improving iff current > grandparent (equal is not improving).
    stack[2].static_eval = 30;
    stack[4].static_eval = 50;
    try expect(isImproving(&stack, 4));
    stack[4].static_eval = 10;
    try expect(!isImproving(&stack, 4));
    stack[4].static_eval = 30;
    try expect(!isImproving(&stack, 4));
}
