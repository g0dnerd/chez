const evaluation = @import("evaluation.zig");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const State = @import("State.zig");
const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expect = std.testing.expect;
const Atomic = std.atomic.Value;

const CHECKMATE_SCORE: i32 = 100000;
const ALPHA_INIT: i32 = std.math.minInt(i32) + 1;
const BETA_INIT: i32 = std.math.maxInt(i32);
const MAX_PLY: usize = 64;
const MAX_THREADS: usize = 16;
const DEFAULT_THREADS: usize = 8;

// Maximum game length - 1024 half-moves (512 full moves) is very generous
const MAX_GAME_LENGTH: usize = 1024;

// Position history for threefold repetition detection
// Stores Zobrist hashes of positions since game start
pub const PositionHistory = struct {
    hashes: [MAX_GAME_LENGTH]u64 = undefined,
    len: usize = 0,

    pub fn init() PositionHistory {
        return .{};
    }

    pub fn push(self: *PositionHistory, hash: u64) void {
        if (self.len < MAX_GAME_LENGTH) {
            self.hashes[self.len] = hash;
            self.len += 1;
        }
    }

    pub fn pop(self: *PositionHistory) void {
        if (self.len > 0) {
            self.len -= 1;
        }
    }

    // Check if current position is a repetition
    // halfmove_clock tells us how far back we need to look (since last irreversible move)
    // For search: returns true for 2-fold (implies 3-fold in game context)
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
        const target = if (require_threefold) @as(u8, 2) else @as(u8, 1);

        // Check every 2 positions (same side to move)
        // i represents how many ply back from current position (len-1)
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

    // Check for threefold repetition (for game-over detection)
    pub fn isThreefold(self: *const PositionHistory, hash: u64, halfmove_clock: u16) bool {
        return self.isRepetition(hash, halfmove_clock, true);
    }

    // Check for twofold repetition (for search pruning)
    pub fn isTwofold(self: *const PositionHistory, hash: u64, halfmove_clock: u16) bool {
        return self.isRepetition(hash, halfmove_clock, false);
    }
};

// Killer move table: stores 2 killer moves per ply
// Killer moves are quiet moves that caused beta cutoffs
const KillerTable = struct {
    moves: [MAX_PLY][2]?game.Move = [_][2]?game.Move{.{ null, null }} ** MAX_PLY,

    fn store(self: *KillerTable, ply: usize, m: game.Move) void {
        if (ply >= MAX_PLY) return;
        // Don't store if it's already the first killer
        if (self.moves[ply][0]) |k| {
            if (k.start == m.start and k.end == m.end) return;
        }
        // Shift first killer to second slot, store new as first
        self.moves[ply][1] = self.moves[ply][0];
        self.moves[ply][0] = m;
    }

    fn isKiller(self: *const KillerTable, ply: usize, m: game.Move) bool {
        if (ply >= MAX_PLY) return false;
        if (self.moves[ply][0]) |k| {
            if (k.start == m.start and k.end == m.end) return true;
        }
        if (self.moves[ply][1]) |k| {
            if (k.start == m.start and k.end == m.end) return true;
        }
        return false;
    }

    fn clear(self: *KillerTable) void {
        self.moves = [_][2]?game.Move{.{ null, null }} ** MAX_PLY;
    }
};

const Flag = enum(u2) {
    empty = 0,
    exact = 1,
    lowerBound = 2,
    upperBound = 3,
};

const TranspositionEntry = struct {
    hash: u64 = 0,
    score: i32 = 0,
    depth: u8 = 0,
    flag: Flag = .empty,
    best_move: ?game.Move = null,
};

// Fixed-size transposition table using direct indexing
// Size must be a power of 2 for fast modulo via bitmask
const TT_SIZE_BITS = 20; // 2^20 = ~1M entries
const TT_SIZE: usize = 1 << TT_SIZE_BITS;
const TT_MASK: u64 = TT_SIZE - 1;

// Lock-free transposition table entry packed into two 64-bit words
// This allows atomic read/write without locks (Stockfish-style)
// Word 1: hash XOR data (for validation)
// Word 2: data (score:16, depth:8, flag:2, move_start:6, move_end:6, move_valid:1 = 39 bits)
const PackedTTEntry = struct {
    key: Atomic(u64) = Atomic(u64).init(0),
    data: Atomic(u64) = Atomic(u64).init(0),

    fn pack(hash: u64, score: i32, depth: u8, flag: Flag, best_move: ?game.Move) struct { key: u64, data: u64 } {
        // Pack data into 64 bits:
        // bits 0-15: score (as u16, offset by 32768 to handle negatives)
        // bits 16-23: depth
        // bits 24-25: flag
        // bits 26-31: move start
        // bits 32-37: move end
        // bit 38: move valid
        // Clamp score to i16 range to avoid corruption of mate scores
        const clamped_score = std.math.clamp(score, std.math.minInt(i16), std.math.maxInt(i16));
        const score_u: u16 = @bitCast(@as(i16, @intCast(clamped_score)));
        var data: u64 = score_u;
        data |= @as(u64, depth) << 16;
        data |= @as(u64, @intFromEnum(flag)) << 24;
        if (best_move) |m| {
            data |= @as(u64, m.start) << 26;
            data |= @as(u64, m.end) << 32;
            data |= @as(u64, 1) << 38;
        }
        // XOR hash with data for validation
        const key = hash ^ data;
        return .{ .key = key, .data = data };
    }

    fn unpack(key: u64, data: u64, hash: u64) ?TranspositionEntry {
        // Validate: key XOR data should equal original hash
        if ((key ^ data) != hash) return null;

        const score_u: u16 = @truncate(data);
        const score: i32 = @as(i16, @bitCast(score_u));
        const depth: u8 = @truncate(data >> 16);
        const flag: Flag = @enumFromInt(@as(u2, @truncate(data >> 24)));
        const move_start: u6 = @truncate(data >> 26);
        const move_end: u6 = @truncate(data >> 32);
        const move_valid: u1 = @truncate(data >> 38);

        if (flag == .empty) return null;

        return TranspositionEntry{
            .hash = hash,
            .score = score,
            .depth = depth,
            .flag = flag,
            .best_move = if (move_valid == 1) game.Move{ .start = move_start, .end = move_end } else null,
        };
    }
};

const TranspositionTable = struct {
    entries: []PackedTTEntry,
    alloc: std.mem.Allocator,

    fn init(alloc: std.mem.Allocator) !TranspositionTable {
        const entries = try alloc.alloc(PackedTTEntry, TT_SIZE);
        @memset(entries, PackedTTEntry{});
        return .{ .entries = entries, .alloc = alloc };
    }

    fn deinit(self: *TranspositionTable) void {
        self.alloc.free(self.entries);
    }

    fn probe(self: *TranspositionTable, hash: u64) ?TranspositionEntry {
        const idx = hash & TT_MASK;
        const entry = &self.entries[idx];

        // Lock-free read with atomic loads
        const key = entry.key.load(.monotonic);
        const data = entry.data.load(.monotonic);

        return PackedTTEntry.unpack(key, data, hash);
    }

    fn store(self: *TranspositionTable, hash: u64, score: i32, depth: u8, flag: Flag, best_move: ?game.Move) void {
        const idx = hash & TT_MASK;
        const entry = &self.entries[idx];

        // Check replacement policy: only replace if new depth >= existing
        const old_data = entry.data.load(.monotonic);
        const old_depth: u8 = @truncate(old_data >> 16);
        const old_flag: Flag = @enumFromInt(@as(u2, @truncate(old_data >> 24)));

        if (old_flag != .empty and depth < old_depth) return;

        const p = PackedTTEntry.pack(hash, score, depth, flag, best_move);

        // Lock-free write - data races are benign (just cause cache misses)
        entry.data.store(p.data, .monotonic);
        entry.key.store(p.key, .monotonic);
    }
};

// Minimal shared state for Lazy SMP - threads run independently
const SharedSearchState = struct {
    stop_flag: Atomic(bool) = Atomic(bool).init(false),
    max_depth: u8 = 0,
};

// Per-thread context for search
const ThreadContext = struct {
    state: State,
    killers: KillerTable,
    history: PositionHistory,
    thread_id: usize,
    tbl: *TranspositionTable,
    shared: *SharedSearchState,
    // Each thread reports its best result here
    best_move: ?game.Move = null,
    best_score: i32 = std.math.minInt(i32) + 1,
    best_depth: u8 = 0,
};

// Quiescence search: search only captures until the position is "quiet"
// This prevents the horizon effect where we evaluate positions mid-tactical-sequence
fn quiescence(state: *State, alpha_init: i32, beta: i32) i32 {
    const to_move = state.to_move;
    const in_check = state.in_check == to_move;

    // When in check, we must search all evasions, not just captures
    if (in_check) {
        var moves = movegen.legalMoves(state, to_move);

        if (moves.len == 0) {
            return -CHECKMATE_SCORE;
        }

        moves.order(state, to_move);

        var alpha = alpha_init;
        for (0..moves.len) |i| {
            const m = moves.moves[i];
            const p = state.pieceAt(m.start).?;
            const undo = state.makeMove(m, to_move, p);

            const score = -quiescence(state, -beta, -alpha);

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
    const stand_pat = evaluation.evaluate(state);

    // Beta cutoff: position is so good opponent wouldn't allow it
    if (stand_pat >= beta) {
        return beta;
    }

    var alpha = alpha_init;

    // If stand pat is better than alpha, we can use it as a floor
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    // Generate and search captures only
    var captures = movegen.legalCaptures(state, to_move);

    if (captures.len == 0) {
        return stand_pat;
    }

    captures.order(state, to_move);

    for (0..captures.len) |i| {
        const m = captures.moves[i];
        const p = state.pieceAt(m.start).?;
        const undo = state.makeMove(m, to_move, p);

        const score = -quiescence(state, -beta, -alpha);

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

fn negamax(state: *State, depth: u8, ply: usize, alpha_init: i32, beta: i32, tbl: *TranspositionTable, killers: *KillerTable, history: *PositionHistory) i32 {
    const hash = state.zobrist_hash;
    var alpha = alpha_init;
    var best_move: ?game.Move = null;

    // Check for repetition - return draw score (0) if position occurred before
    // We check for twofold since we're in the search tree (implies threefold in game)
    if (ply > 0 and history.isTwofold(hash, state.halfmove_clock)) {
        return 0;
    }

    // Probe transposition table
    if (tbl.probe(hash)) |entry| {
        if (entry.depth >= depth) {
            switch (entry.flag) {
                .exact => return entry.score,
                .lowerBound => alpha = @max(alpha, entry.score),
                .upperBound => {
                    if (entry.score <= alpha) return entry.score;
                },
                .empty => {},
            }
            if (alpha >= beta) {
                return entry.score;
            }
        }
    }

    const to_move = state.to_move;
    var moves = movegen.legalMoves(state, to_move);

    if (moves.len == 0) {
        if (state.in_check == to_move) {
            return -CHECKMATE_SCORE;
        } else {
            return 0;
        }
    }

    if (depth == 0) {
        return quiescence(state, alpha, beta);
    }

    // Null move pruning: if giving opponent a free move still results in beta cutoff,
    // the position is so good we can prune
    const in_check = state.in_check == to_move;
    if (depth >= 3 and !in_check and state.hasNonPawnMaterial(to_move)) {
        const keys = game.getZobristKeys();
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

        // Adaptive reduction: R = 2 + depth/4
        const R: u8 = 2 + depth / 4;
        const null_score = -negamax(state, depth - 1 - R, ply + 1, -beta, -beta + 1, tbl, killers, history);

        // Unmake null move
        state.*.to_move = old_to_move;
        state.*.en_passant = old_ep;
        state.*.zobrist_hash = old_hash;
        state.*.in_check = old_in_check;

        if (null_score >= beta) {
            return beta;
        }
    }

    // Order moves with killer heuristic
    const ply_killers = if (ply < MAX_PLY) killers.moves[ply] else [2]?game.Move{ null, null };
    moves.orderWithKillers(state, to_move, ply_killers);

    var max_score: i32 = std.math.minInt(i32);

    for (0..moves.len) |i| {
        const m = moves.moves[i];
        const p = state.pieceAt(m.start).?;

        // Check if this is a capture before making the move (for LMR decision)
        const is_capture = state.pieceAt(m.end) != null;

        const undo = state.makeMove(m, to_move, p);
        history.push(state.zobrist_hash);

        // Check extension: extend search by 1 ply when giving check
        const gives_check = state.in_check != null;
        const extension: u8 = if (gives_check) 1 else 0;
        const new_depth = depth - 1 + extension;

        var score: i32 = undefined;
        if (i == 0) {
            // First move: search with full window
            score = -negamax(state, new_depth, ply + 1, -beta, -alpha, tbl, killers, history);
        } else {
            // Late Move Reductions (LMR):
            // Moves ordered later are likely worse, so search with reduced depth first.
            // Only reduce quiet moves at sufficient depth that don't give check.
            var reduction: u8 = 0;
            if (i >= 3 and depth >= 3 and !is_capture and !gives_check and !in_check) {
                // Base reduction + increase for later moves and higher depths
                // Formula: 1 + ln(depth) * ln(moveIndex) / 2 (simplified integer version)
                reduction = 1;
                if (i >= 6) reduction += 1;
                if (i >= 12) reduction += 1;
                if (depth >= 6) reduction += 1;
                // Don't reduce into qsearch
                if (reduction >= new_depth) {
                    reduction = if (new_depth > 1) new_depth - 1 else 0;
                }
            }

            // PVS with LMR: search with reduced depth and null window
            score = -negamax(state, new_depth - reduction, ply + 1, -alpha - 1, -alpha, tbl, killers, history);

            // Re-search at full depth if reduced search improved alpha
            if (score > alpha and reduction > 0) {
                score = -negamax(state, new_depth, ply + 1, -alpha - 1, -alpha, tbl, killers, history);
            }

            // Re-search with full window if null window failed high
            if (score > alpha and score < beta) {
                score = -negamax(state, new_depth, ply + 1, -beta, -alpha, tbl, killers, history);
            }
        }

        // Track if it was a capture (for killer move storage)
        const was_capture = is_capture or undo.captured_piece != null; // en passant

        history.pop();
        state.unmakeMove(m, to_move, p, undo);

        if (score > max_score) {
            max_score = score;
            best_move = m;
        }
        alpha = @max(alpha, score);

        if (alpha >= beta) {
            // Beta cutoff - store killer if this is a quiet move (not a capture)
            if (!was_capture) {
                killers.store(ply, m);
            }
            break;
        }
    }

    // Determine flag for TT entry
    const flag: Flag = if (max_score <= alpha_init)
        .upperBound
    else if (max_score >= beta)
        .lowerBound
    else
        .exact;

    tbl.store(hash, max_score, depth, flag, best_move);

    return max_score;
}

pub const SearchResult = struct {
    move: game.Move,
    score: i32,
    depth: u8,
};

// Search at a specific depth with an optional hint for the best move from the previous iteration
fn searchAtDepth(state: *State, depth: u8, tbl: *TranspositionTable, killers: *KillerTable, pv_move: ?game.Move, history: *PositionHistory) ?SearchResult {
    var best_score: i32 = std.math.minInt(i32);
    var best_move: ?game.Move = null;

    const to_move = state.to_move;
    var moves = movegen.legalMoves(state, to_move);

    if (moves.len == 0) {
        return null;
    }

    moves.order(state, to_move);

    // If we have a PV move from the previous iteration, try it first
    if (pv_move) |pv| {
        // Find and move PV to front
        for (0..moves.len) |i| {
            if (moves.moves[i].start == pv.start and moves.moves[i].end == pv.end) {
                // Swap PV move to front
                const tmp = moves.moves[0];
                moves.moves[0] = moves.moves[i];
                moves.moves[i] = tmp;
                break;
            }
        }
    }

    for (0..moves.len) |i| {
        const m = moves.moves[i];
        const p = state.pieceAt(m.start).?;
        const undo = state.makeMove(m, to_move, p);
        history.push(state.zobrist_hash);

        // Take mate in 1
        if (isGameOver(state)) |r| {
            switch (r) {
                .checkmate => if (r.checkmate == to_move) {
                    history.pop();
                    state.unmakeMove(m, to_move, p, undo);
                    return .{ .move = m, .score = CHECKMATE_SCORE, .depth = depth };
                },
                else => {},
            }
        }

        const score = -negamax(state, depth - 1, 1, -BETA_INIT, -ALPHA_INIT, tbl, killers, history);

        history.pop();
        state.unmakeMove(m, to_move, p, undo);

        if (score > best_score) {
            best_move = m;
            best_score = score;
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

// Worker thread function for Lazy SMP
// Each thread does full iterative deepening independently
// Threads diverge naturally due to TT interactions and timing
fn workerThread(ctx: *ThreadContext) void {
    var pv_move: ?game.Move = null;

    // Each thread does iterative deepening up to max_depth
    for (1..ctx.shared.max_depth + 1) |depth_usize| {
        if (ctx.shared.stop_flag.load(.monotonic)) break;

        const depth: u8 = @intCast(depth_usize);

        // Get PV hint from TT (may have been populated by other threads)
        const tt_move = if (ctx.tbl.probe(ctx.state.zobrist_hash)) |entry| entry.best_move else null;
        const hint = pv_move orelse tt_move;

        const result = searchAtDepth(&ctx.state, depth, ctx.tbl, &ctx.killers, hint, &ctx.history);

        if (result) |r| {
            ctx.best_move = r.move;
            ctx.best_score = r.score;
            ctx.best_depth = depth;
            pv_move = r.move;

            // Early exit if checkmate found
            if (r.score >= CHECKMATE_SCORE - 100) {
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
pub fn searchParallel(state: *const State, max_depth: u8, num_threads: usize, game_history: ?*const PositionHistory) !?SearchResult {
    const actual_threads = @min(num_threads, MAX_THREADS);

    var tbl = try TranspositionTable.init(std.heap.page_allocator);
    defer tbl.deinit();

    var shared = SharedSearchState{
        .max_depth = max_depth,
    };

    // Create thread contexts
    var contexts: [MAX_THREADS]ThreadContext = undefined;
    for (0..actual_threads) |i| {
        // Initialize history with game history if provided
        var history = PositionHistory.init();
        if (game_history) |gh| {
            for (0..gh.len) |j| {
                history.push(gh.hashes[j]);
            }
        }
        // Push the root position
        history.push(state.zobrist_hash);

        contexts[i] = ThreadContext{
            .state = state.*,
            .killers = KillerTable{},
            .history = history,
            .thread_id = i,
            .tbl = &tbl,
            .shared = &shared,
        };
    }

    // Spawn worker threads
    var threads: [MAX_THREADS]std.Thread = undefined;
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
    var best_move: ?game.Move = null;
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
        var sq_start: [2]u8 = undefined;
        var sq_end: [2]u8 = undefined;
        game.squareToAlgebraic(m.start, &sq_start) catch {};
        game.squareToAlgebraic(m.end, &sq_end) catch {};
        std.debug.print("Depth {d} ({d} threads) - move: {s}{s} - eval: {d}\n", .{
            best_depth, spawned_threads + 1, sq_start, sq_end, best_score,
        });

        return .{
            .move = m,
            .score = best_score,
            .depth = best_depth,
        };
    } else {
        return null;
    }
}

// Iterative deepening search: searches depth 1, then 2, etc. up to max_depth.
// Uses parallel search with default thread count.
pub fn search(state: *const State, max_depth: u8) !?SearchResult {
    return searchParallel(state, max_depth, DEFAULT_THREADS, null);
}

// Search with explicit thread count
pub fn searchWithThreads(state: *const State, max_depth: u8, num_threads: usize) !?SearchResult {
    return searchParallel(state, max_depth, num_threads, null);
}

// Search with game history for repetition detection
pub fn searchWithHistory(state: *const State, max_depth: u8, num_threads: usize, history: *const PositionHistory) !?SearchResult {
    return searchParallel(state, max_depth, num_threads, history);
}

// Single-threaded search for testing and debugging
pub fn searchSingleThreaded(state: *const State, max_depth: u8) !?SearchResult {
    var tbl = try TranspositionTable.init(std.heap.page_allocator);
    defer tbl.deinit();

    var killers = KillerTable{};
    var history = PositionHistory.init();
    history.push(state.zobrist_hash);

    // Make a mutable copy for the search (make/unmake will restore it)
    var mutable_state = state.*;

    var best_move: ?game.Move = null;
    var best_score: i32 = undefined;
    var best_depth: u8 = 0;

    var sq_start: [2]u8 = undefined;
    var sq_end: [2]u8 = undefined;

    for (1..max_depth + 1) |depth| {
        const result = searchAtDepth(&mutable_state, @intCast(depth), &tbl, &killers, best_move, &history);
        if (result) |r| {
            best_move = r.move;
            best_score = r.score;

            try game.squareToAlgebraic(best_move.?.start, &sq_start);
            try game.squareToAlgebraic(best_move.?.end, &sq_end);
            std.debug.print("Depth {d} - move: {s}{s} - eval: {d}\r", .{ depth, sq_start, sq_end, best_score });

            best_depth = r.depth;

            // Early exit if we found a checkmate
            if (best_score >= CHECKMATE_SCORE - 100) {
                break;
            }
        }
    }

    std.debug.print("\n", .{});

    if (best_move) |m| {
        return .{
            .move = m,
            .score = best_score,
            .depth = best_depth,
        };
    } else {
        return null;
    }
}

pub fn isGameOver(state: *const State) ?game.GameResult {
    return isGameOverWithHistory(state, null);
}

// Check for game over with optional position history for threefold repetition
pub fn isGameOverWithHistory(state: *const State, history: ?*const PositionHistory) ?game.GameResult {
    const to_move = state.to_move;
    const hasLegalMoves = movegen.hasAnyLegalMove(state, to_move);

    if (!hasLegalMoves) {
        const king_square = state.pieceBitboard(game.Pieces.king).bitAnd(state.colorBitboard(to_move)).trailingZeros();
        if (movegen.isSquareAttackedBy(state, king_square, ~to_move)) {
            return game.GameResult{ .checkmate = ~to_move };
        } else {
            return .stalemate;
        }
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
    try expectEqual(game.Squares.h5, result.?.move.start);
    try expectEqual(game.Squares.f7, result.?.move.end);
    try expect(result.?.score > 50000);
}

test "test detects stalemate" {
    const fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1";
    const state = try State.fromFen(fen);

    const res = isGameOver(&state);
    try expectEqual(game.GameResult.stalemate, res);
}

// Mate in 2 tests
test "backrank mate in two" {
    // Rook on c2 can deliver mate: Qc8+ Rxc8 Rxc8#
    const fen = "r5k1/5ppp/8/2Q5/8/8/2R5/4K3 w - - 2 1";
    const state = try State.fromFen(fen);

    const result = try search(&state, 4);
    try expect(result != null);
    try expectEqual(game.Squares.c5, result.?.move.start);
    try expectEqual(game.Squares.c8, result.?.move.end);
}

test "mate in two" {
    // Rxh6+ Kxh6 Qg6#
    const fen = "r7/ppq2r1k/1npR3p/4PP2/6Q1/8/PP4PP/R6K w - - 0 1";
    const state = try State.fromFen(fen);

    const result = try search(&state, 4);
    try expect(result != null);
    try expectEqual(game.Squares.d6, result.?.move.start);
    try expectEqual(game.Squares.h6, result.?.move.end);
}

test "smothered mate pattern" {
    const fen = "4r2k/2pRP1pp/2p5/p4pN1/2Q3n1/q5P1/P3PP1P/6K1 w - - 0 1";
    const state = try State.fromFen(fen);

    const result = try search(&state, 4);
    try expect(result != null);
    try expectEqual(game.Squares.g5, result.?.move.start);
    try expectEqual(game.Squares.f7, result.?.move.end);
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
    var history = PositionHistory.init();

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
    var history = PositionHistory.init();

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
    var history = PositionHistory.init();

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
    var history = PositionHistory.init();

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
