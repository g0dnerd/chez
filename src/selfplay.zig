const builtin = @import("builtin");
const std = @import("std");
const engine = @import("chez").engine;
const fathom = @import("fathom.zig");
const kore = @import("kore");
pub const serde = @import("selfplay/serde.zig");

const position_size = 32;
const record_size = position_size + 2 + 1; // 32 position + 2 score + 1 WDL
const max_game_records = 512;
const default_depth: u8 = 8;
const default_games: usize = 1000;
const default_threads: usize = 4;
// Upper bound on selfplay worker threads (sizes the fixed context/thread arrays).
// High enough to saturate large many-core data-gen boxes.
const max_threads: usize = 256;
const default_random_plies: u16 = 8;
const skip_plies: u16 = 16;
// Defaults restore pre-"speedup" data quality: keep decisive positions
// (high score_filter) and adjudicate only clearly-won games (high threshold).
// Aggressive values compress the eval-label range and weaken the trained net.
const default_score_filter: i32 = 10000;
const default_adjudication_threshold: i32 = 2500;
const default_adjudication_count: u16 = 4;
const sample_interval: u16 = 4;
// Syzygy WDL label magnitude (centipawns). Large enough to saturate the training
// sigmoid for decisive endgames; draws are labeled 0. Recorded via shouldRecordTb,
// which bypasses score_filter so TB-perfect decisive labels are always kept.
const tb_label_cp: i32 = 20000;

// Buckets for the recorded-position ply histogram (ply measured from each game's
// root, i.e. the book position or startpos). Game length is capped at
// max_game_plies (200) plus a few random opening plies, so 256 covers all plies.
const ply_hist_buckets: usize = 256;

// Tunable data-quality knobs, overridable via CLI.
const RecordCfg = struct {
    score_filter: i32 = default_score_filter,
    adjudication_threshold: i32 = default_adjudication_threshold,
    adjudication_count: u16 = default_adjudication_count,
};
// Draw adjudication: balanced eval for many consecutive plies past the opening.
// Trigger deeper (min_ply 120) and require a longer dead-equal run (16 plies) so
// long maneuvering/endgame phases get recorded instead of being cut as draws --
// and fewer winnable-but-near-0 positions are mislabeled draws.
const draw_adjudication_threshold: i32 = 10;
const draw_adjudication_count: u16 = 16;
const draw_adjudication_min_ply: u16 = 120;
// Hard cap on game length so worst-case games can't run to the 50-move rule at full depth.
const max_game_plies: u16 = 200;
// Per-worker TT size: 2^21 buckets = 8M entries (~128 MB). Larger than the engine
// default (2^18) to cut re-search at the deep, long searches self-play runs.
const tt_buckets_bits: u6 = 21;

const GameOutcome = enum(u8) {
    white_wins = 0,
    black_wins = 1,
    draw = 2,
};

const TrainingRecord = struct {
    position: [position_size]u8,
    score: i16,
};

const SelfplayGame = struct {
    const Self = @This();

    state: engine.State,
    rng: std.Random,
    history: engine.search.PositionHistory,
    ttable: *engine.search.TranspositionTable,
    searcher: *engine.search.ReusableSearcher,
    network: ?*const nnue.Network,
    // Balanced opening positions (parsed from a FEN book). Empty => start from
    // the initial position. Each game picks one at random as its root, then plays
    // random_plies random moves on top so games from the same line still diverge
    // (search is deterministic at a fixed node budget).
    book: []const engine.State,
    random_plies: u16,
    depth: u8,
    max_nodes: ?u64,
    cfg: RecordCfg,
    ply: u16,
    records: [max_game_records]TrainingRecord,
    // Game ply at which each buffered record was sampled (parallel to records).
    record_plies: [max_game_records]u16,
    num_records: usize,
    adjudication_consecutive: u16,
    adjudication_winning_side: engine.Color,
    draw_consecutive: u16,

    fn init(rng: std.Random, depth: u8, max_nodes: ?u64, cfg: RecordCfg, ttable: *engine.search.TranspositionTable, searcher: *engine.search.ReusableSearcher, network: ?*const nnue.Network, book: []const engine.State, random_plies: u16) Self {
        return .{
            .state = .defaultPosition(),
            .rng = rng,
            .history = .{},
            .ttable = ttable,
            .searcher = searcher,
            .network = network,
            .book = book,
            .random_plies = random_plies,
            .depth = depth,
            .max_nodes = max_nodes,
            .cfg = cfg,
            .ply = 0,
            .records = undefined,
            .record_plies = undefined,
            .num_records = 0,
            .adjudication_consecutive = 0,
            .adjudication_winning_side = engine.Colors.white,
            .draw_consecutive = 0,
        };
    }

    fn reset(self: *Self) void {
        self.state = if (self.book.len > 0)
            self.book[self.rng.uintLessThan(usize, self.book.len)]
        else
            .defaultPosition();
        self.history = .{};
        self.ply = 0;
        self.num_records = 0;
        self.adjudication_consecutive = 0;
        self.draw_consecutive = 0;
        self.ttable.newSearch();
    }

    fn gameResultToOutcome(res: engine.GameResult) GameOutcome {
        return switch (res) {
            .checkmate => |color| if (color == engine.Colors.white) .white_wins else .black_wins,
            else => .draw,
        };
    }

    // Play random opening moves without recording positions.
    // Returns an outcome if the game ends during the opening.
    fn makeRandomMove(self: *Self) ?GameOutcome {
        const to_move = self.state.to_move;
        const moves = engine.movegen.legalMoves(&self.state, to_move);

        if (moves.len == 0) {
            if (engine.search.isGameOverWithHistory(&self.state, &self.history)) |res| {
                return gameResultToOutcome(res);
            }
            return .draw;
        }

        const random_move_idx = self.rng.uintLessThan(u8, moves.len);
        const random_move = moves.moves[random_move_idx];
        _ = self.state.makeMove(random_move, to_move, self.state.mailbox[random_move.start].?);
        self.history.push(self.state.zobrist_hash);
        self.ply += 1;
        return null;
    }

    fn shouldRecord(self: *const Self, score: i32) bool {
        if (self.ply < skip_plies) return false;
        if (self.state.in_check != null) return false;
        if (score > self.cfg.score_filter or score < -self.cfg.score_filter) return false;
        if (self.ply % sample_interval != 0) return false;
        return true;
    }

    // Record gate for TB-labeled endgame positions. Drops shouldRecord's
    // score_filter (decisive endgames must be kept) and sample_interval (immediate
    // adjudication means at most one TB record per game -- no within-game
    // correlation to thin). Keeps skip_plies and the in-check gate.
    fn shouldRecordTb(self: *const Self) bool {
        if (self.ply < skip_plies) return false;
        if (self.state.in_check != null) return false;
        return true;
    }

    fn bufferPosition(self: *Self, score: i32) !void {
        if (self.num_records >= max_game_records) return;
        var record = &self.records[self.num_records];
        record.position = @splat(0);
        try serde.encodePositionToBuffer(self.state, &record.position);
        record.score = std.math.cast(i16, score) orelse
            if (score > 0) std.math.maxInt(i16) else std.math.minInt(i16);
        self.record_plies[self.num_records] = self.ply;
        self.num_records += 1;
    }

    // Search, optionally record the position, then make the best move.
    // Returns an outcome if the game ends.
    fn makeMoveAtDepth(self: *Self) !?GameOutcome {
        if (engine.search.isGameOverWithHistory(&self.state, &self.history)) |res| {
            return gameResultToOutcome(res);
        }

        const search_res = self.searcher.search(
            &self.state,
            self.depth,
            &self.history,
            self.ttable,
            .{ .max_nodes = self.max_nodes },
            self.network,
        ) orelse return error.SearchFailed;

        const best_move = search_res.move;
        const score = search_res.score;

        // Syzygy WDL: when the position is in TB range, prefer the exact endgame
        // verdict over the search score. Record it (bypassing score_filter so
        // decisive endgames are kept) and adjudicate the game immediately on the
        // exact result -- shorter games + a TB-correct per-game WDL byte.
        // probeWdl returns null when no tables are loaded, so the else branch keeps
        // the existing search-score labeling behavior unchanged.
        if (engine.tablebase.probeWdl(&self.state)) |wdl| {
            if (self.shouldRecordTb()) {
                const tb_score: i32 = switch (wdl) {
                    .win => tb_label_cp,
                    .loss => -tb_label_cp,
                    .draw => 0,
                };
                try self.bufferPosition(tb_score);
            }
            // WDL is from the side to move; map to white's perspective exactly like
            // the eval-based adjudication below.
            return switch (wdl) {
                .win => if (self.state.to_move == engine.Colors.white) .white_wins else .black_wins,
                .loss => if (self.state.to_move == engine.Colors.white) .black_wins else .white_wins,
                .draw => .draw,
            };
        } else if (self.shouldRecord(score)) {
            // Label with the backed-up search score (NNUE-driven when a net is
            // loaded), a stronger target than a static eval of the same position.
            try self.bufferPosition(score);
        }

        const abs_score = @as(i32, @intCast(@abs(score)));
        if (abs_score > self.cfg.adjudication_threshold) {
            const winning_side: engine.Color = if (score > 0) self.state.to_move else ~self.state.to_move;
            if (self.adjudication_consecutive > 0 and winning_side == self.adjudication_winning_side) {
                self.adjudication_consecutive += 1;
            } else {
                self.adjudication_consecutive = 1;
                self.adjudication_winning_side = winning_side;
            }
            if (self.adjudication_consecutive >= self.cfg.adjudication_count) {
                return if (winning_side == engine.Colors.white) .white_wins else .black_wins;
            }
        } else {
            self.adjudication_consecutive = 0;
        }

        // Draw adjudication: a long run of near-zero evals past the opening.
        if (self.ply >= draw_adjudication_min_ply and abs_score <= draw_adjudication_threshold) {
            self.draw_consecutive += 1;
            if (self.draw_consecutive >= draw_adjudication_count) {
                return .draw;
            }
        } else {
            self.draw_consecutive = 0;
        }

        _ = self.state.makeMove(best_move, self.state.to_move, self.state.mailbox[best_move.start].?);
        self.history.push(self.state.zobrist_hash);
        self.ply += 1;

        if (engine.search.isGameOverWithHistory(&self.state, &self.history)) |res| {
            return gameResultToOutcome(res);
        }

        return null;
    }

    // Play one complete game. Returns the outcome and number of recorded positions.
    fn playGame(self: *Self) !struct { outcome: GameOutcome, positions: usize } {
        self.reset();

        // Random opening phase (on top of the book root, if any)
        for (0..self.random_plies) |_| {
            if (self.makeRandomMove()) |outcome| {
                return .{ .outcome = outcome, .positions = self.num_records };
            }
        }

        // Search phase
        while (self.ply < max_game_plies) {
            if (try self.makeMoveAtDepth()) |outcome| {
                return .{ .outcome = outcome, .positions = self.num_records };
            }
        }
        return .{ .outcome = .draw, .positions = self.num_records };
    }

    // Write all buffered records to the output writer.
    // Each record: 32 bytes position + 2 bytes i16 score (little-endian) + 1 byte WDL
    fn writeRecords(self: *const Self, writer: *std.Io.Writer, outcome: GameOutcome) !void {
        const wdl: u8 = @intFromEnum(outcome);
        for (self.records[0..self.num_records]) |record| {
            var buf: [record_size]u8 = undefined;
            @memcpy(buf[0..position_size], &record.position);
            std.mem.writeInt(i16, buf[position_size..][0..2], record.score, .little);
            buf[position_size + 2] = wdl;
            try writer.writeAll(&buf);
        }
    }
};

const WorkerCtx = struct {
    games_per_worker: usize,
    depth: u8,
    max_nodes: ?u64,
    cfg: RecordCfg,
    seed: u64,
    writer: *std.Io.Writer,
    write_mutex: *std.Io.Mutex,
    stderr: *std.Io.Writer,
    stderr_mutex: *std.Io.Mutex,
    io: std.Io,
    total_positions: *std.atomic.Value(usize),
    total_games: *std.atomic.Value(usize),
    network: ?*const nnue.Network,
    book: []const engine.State,
    random_plies: u16,
    // Shared recorded-position ply histogram; each worker merges its local copy in once at the end.
    ply_hist: *[ply_hist_buckets]u64,
    hist_mutex: *std.Io.Mutex,
};

fn workerLoop(ctx: *WorkerCtx) void {
    var ttable = engine.search.TranspositionTable.initSized(std.heap.page_allocator, tt_buckets_bits) catch return;
    defer ttable.deinit();

    // One reusable searcher per worker: its 2.25 MB continuation-history table and
    // LMR table are allocated/computed once here, not per move.
    var searcher = engine.search.ReusableSearcher.init(std.heap.page_allocator, .{}) catch return;
    defer searcher.deinit();

    var rng = std.Random.Pcg.init(ctx.seed);
    var game = SelfplayGame.init(rng.random(), ctx.depth, ctx.max_nodes, ctx.cfg, &ttable, &searcher, ctx.network, ctx.book, ctx.random_plies);

    // Worker-local ply histogram, merged into the shared one once at the end to
    // avoid per-record contention.
    var local_hist = [_]u64{0} ** ply_hist_buckets;

    for (0..ctx.games_per_worker) |_| {
        const result = game.playGame() catch continue;

        if (result.positions > 0) {
            for (game.record_plies[0..result.positions]) |p| {
                local_hist[@min(@as(usize, p), ply_hist_buckets - 1)] += 1;
            }
            // No flush here: the shared writer's buffer auto-drains when full and
            // is flushed once at the end. Flushing per game would serialize a
            // syscall under the write mutex across all workers.
            ctx.write_mutex.lock(ctx.io) catch continue;
            defer ctx.write_mutex.unlock(ctx.io);
            game.writeRecords(ctx.writer, result.outcome) catch return;
        }

        _ = ctx.total_positions.fetchAdd(result.positions, .monotonic);
        const games_done = ctx.total_games.fetchAdd(1, .monotonic) + 1;

        if (games_done % 100 == 0) {
            const positions = ctx.total_positions.load(.monotonic);
            ctx.stderr_mutex.lock(ctx.io) catch continue;
            ctx.stderr.print("\rGames: {d} | Positions: {d}", .{ games_done, positions }) catch {};
            ctx.stderr.flush() catch {};
            ctx.stderr_mutex.unlock(ctx.io);
        }
    }

    ctx.hist_mutex.lock(ctx.io) catch return;
    defer ctx.hist_mutex.unlock(ctx.io);
    for (ctx.ply_hist, local_hist) |*g, l| g.* += l;
}

const nnue = engine.nnue;

const Args = struct {
    depth: ?u8,
    nodes: ?u64,
    num_games: ?usize,
    num_threads: ?usize,
    eval: ?[]const u8,
    score_filter: ?i32,
    adjudication_threshold: ?i32,
    adjudication_count: ?u16,
    openings: ?[]const u8,
    random_plies: ?u16,
    syzygy: ?[]const u8,
};

// Load a balanced opening book: one FEN per line (blank lines and unparseable
// lines are skipped). Two passes so the result is an exact-sized slice.
fn loadOpenings(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]engine.State {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const len: usize = @intCast(stat.size);
    if (len == 0) return error.EmptyOpenings;

    const ptr = try std.posix.mmap(
        null,
        len,
        std.os.linux.PROT{ .READ = true },
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    defer std.posix.munmap(@alignCast(ptr));
    const data = ptr[0..len];

    var count: usize = 0;
    var pass1 = std.mem.tokenizeScalar(u8, data, '\n');
    while (pass1.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \r\t");
        if (line.len == 0) continue;
        _ = engine.State.fromFen(line) catch continue;
        count += 1;
    }
    if (count == 0) return error.EmptyOpenings;

    const book = try allocator.alloc(engine.State, count);
    var i: usize = 0;
    var pass2 = std.mem.tokenizeScalar(u8, data, '\n');
    while (pass2.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \r\t");
        if (line.len == 0) continue;
        book[i] = engine.State.fromFen(line) catch continue;
        i += 1;
    }
    return book[0..i];
}

pub fn main(init: std.process.Init.Minimal) !void {
    const arg_parser = try kore.args.declarative.Parser(Args);

    var args_iter = if (builtin.os.tag == .windows)
        try init.args.iterateAllocator(std.heap.page_allocator)
    else
        init.args.iterate();
    const parsed_args = try arg_parser.parse(&args_iter);

    const depth = parsed_args.depth orelse default_depth;
    const max_nodes = parsed_args.nodes;
    const num_games = parsed_args.num_games orelse default_games;
    const num_threads = parsed_args.num_threads orelse default_threads;
    const random_plies = parsed_args.random_plies orelse default_random_plies;
    const cfg = RecordCfg{
        .score_filter = parsed_args.score_filter orelse default_score_filter,
        .adjudication_threshold = parsed_args.adjudication_threshold orelse default_adjudication_threshold,
        .adjudication_count = parsed_args.adjudication_count orelse default_adjudication_count,
    };

    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .environ = .empty });
    const io = threaded.io();

    var stdout_buf: [65536]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    var stderr_buf: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buf);
    const stderr: *std.Io.Writer = &stderr_writer.interface;

    var write_mutex: std.Io.Mutex = .init;
    var stderr_mutex: std.Io.Mutex = .init;

    // Load NNUE network if specified
    var network: ?*nnue.Network = null;
    if (parsed_args.eval) |eval_path| {
        network = nnue.Network.load(io, std.heap.page_allocator, eval_path) catch |err| blk: {
            try stderr.print("Warning: could not load {s}: {}\n", .{ eval_path, err });
            try stderr.flush();
            break :blk null;
        };
    }
    defer if (network) |n| n.deinit(std.heap.page_allocator);

    // Load the balanced opening book if one was given; otherwise start from the
    // initial position and rely on random_plies alone for diversity.
    var book: []engine.State = &.{};
    if (parsed_args.openings) |openings_path| {
        book = loadOpenings(io, std.heap.page_allocator, openings_path) catch |err| blk: {
            try stderr.print("Warning: could not load openings {s}: {}\n", .{ openings_path, err });
            try stderr.flush();
            break :blk &.{};
        };
    }
    defer if (book.len > 0) std.heap.page_allocator.free(book);

    try stderr.print("Selfplay: {d} games, depth {d}, {d} thread(s)", .{ num_games, depth, num_threads });
    if (max_nodes) |n| try stderr.print(", node cap {d}", .{n});
    if (network != null) try stderr.print(", NNUE eval", .{});
    try stderr.print("\n", .{});
    try stderr.print("Filters: score_filter {d}, adjudication {d}cp x{d} plies\n", .{
        cfg.score_filter, cfg.adjudication_threshold, cfg.adjudication_count,
    });
    if (book.len > 0) {
        try stderr.print("Openings: {d} book positions + {d} random plies\n", .{ book.len, random_plies });
    } else {
        try stderr.print("Openings: startpos + {d} random plies\n", .{random_plies});
    }
    try stderr.print("Record format: {d} bytes (32 pos + 2 score + 1 wdl)\n", .{record_size});
    try stderr.flush();

    // Initialize Syzygy tablebases before spawning workers (tb_init is not
    // thread-safe; probes afterward are concurrent-safe). When loaded, endgame
    // positions get TB-perfect WDL labels + adjudication; otherwise probing is a
    // no-op and selfplay behaves exactly as before.
    if (parsed_args.syzygy) |syzygy_path| {
        if (std.heap.page_allocator.dupeZ(u8, syzygy_path)) |path_z| {
            defer std.heap.page_allocator.free(path_z);
            if (fathom.init(path_z.ptr)) {
                engine.tablebase.raw_probe_fn = &fathom.probeRaw;
                engine.tablebase.largest = fathom.largest;
                try stderr.print("Syzygy: loaded up to {d}-man tables from {s}\n", .{ fathom.largest, syzygy_path });
            } else {
                try stderr.print("Warning: no Syzygy tables found at {s} (probing disabled)\n", .{syzygy_path});
            }
            try stderr.flush();
        } else |err| {
            try stderr.print("Warning: could not allocate Syzygy path: {}\n", .{err});
            try stderr.flush();
        }
    }

    const actual_threads = @min(num_threads, max_threads);
    const games_per_worker = num_games / actual_threads;
    const remainder = num_games % actual_threads;

    var base_seed: u64 = undefined;
    if (builtin.target.os.tag == .linux) {
        _ = std.os.linux.getrandom(std.mem.asBytes(&base_seed), 8, 0);
    } else {
        std.Io.random(io, std.mem.asBytes(&base_seed));
    }

    var shared_positions = std.atomic.Value(usize).init(0);
    var shared_games = std.atomic.Value(usize).init(0);
    var ply_hist = [_]u64{0} ** ply_hist_buckets;
    var hist_mutex: std.Io.Mutex = .init;

    var contexts: [max_threads]WorkerCtx = undefined;
    for (0..actual_threads) |i| {
        contexts[i] = .{
            .games_per_worker = games_per_worker + @as(usize, if (i < remainder) 1 else 0),
            .depth = depth,
            .max_nodes = max_nodes,
            .cfg = cfg,
            .seed = base_seed +% i,
            .writer = stdout,
            .write_mutex = &write_mutex,
            .stderr = stderr,
            .stderr_mutex = &stderr_mutex,
            .io = io,
            .total_positions = &shared_positions,
            .total_games = &shared_games,
            .network = network,
            .book = book,
            .random_plies = random_plies,
            .ply_hist = &ply_hist,
            .hist_mutex = &hist_mutex,
        };
    }

    // Spawn worker threads (thread 0 runs on main)
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;
    for (1..actual_threads) |i| {
        threads[i] = std.Thread.spawn(.{}, workerLoop, .{&contexts[i]}) catch break;
        spawned = i;
    }

    workerLoop(&contexts[0]);

    for (1..spawned + 1) |i| {
        threads[i].join();
    }

    try stdout.flush();

    const total_positions = shared_positions.load(.monotonic);
    const total_games_done = shared_games.load(.monotonic);
    try stderr.print("\rDone: {d} games, {d} positions ({d} bytes)\n", .{
        total_games_done,
        total_positions,
        total_positions * record_size,
    });

    try printPlyHistogram(stderr, &ply_hist);
    try stderr.flush();
}

// Emit the recorded-position ply distribution: per-10-ply bins plus cumulative
// tails at the draw-adjudication-relevant thresholds. Ply is measured from each
// game's root (book position or startpos), so it reflects how deep into the
// played-out game positions are being sampled.
fn printPlyHistogram(w: *std.Io.Writer, hist: *const [ply_hist_buckets]u64) !void {
    var total: u64 = 0;
    var weighted: u64 = 0;
    var max_ply: usize = 0;
    for (hist, 0..) |c, p| {
        total += c;
        weighted += c * p;
        if (c > 0) max_ply = p;
    }
    if (total == 0) return;
    const ftot: f64 = @floatFromInt(total);

    try w.print("\nRecorded-position ply histogram ({d} positions, mean ply {d:.1}, max {d}):\n", .{
        total, @as(f64, @floatFromInt(weighted)) / ftot, max_ply,
    });
    var lo: usize = 0;
    while (lo <= max_ply) : (lo += 10) {
        var bin: u64 = 0;
        var i = lo;
        while (i < lo + 10 and i < ply_hist_buckets) : (i += 1) bin += hist[i];
        if (bin == 0) continue;
        const frac = @as(f64, @floatFromInt(bin)) / ftot;
        const bars = @as(usize, @intFromFloat(frac * 200.0));
        try w.print("  ply {d:>3}-{d:<3} {d:>10}  {d:.4}  ", .{ lo, lo + 9, bin, frac });
        for (0..bars) |_| try w.writeByte('#');
        try w.writeByte('\n');
    }
    for ([_]usize{ 80, 100, 120, 160 }) |thr| {
        var tail: u64 = 0;
        for (hist, 0..) |c, p| {
            if (p >= thr) tail += c;
        }
        try w.print("  ply >= {d:>3}: {d:.4}\n", .{ thr, @as(f64, @floatFromInt(tail)) / ftot });
    }
}

test {
    std.testing.refAllDecls(@This());
}
