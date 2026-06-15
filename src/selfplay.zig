const builtin = @import("builtin");
const std = @import("std");
const engine = @import("chez").engine;
const kore = @import("kore");
pub const serde = @import("selfplay/serde.zig");

const position_size = 32;
const record_size = position_size + 2 + 1; // 32 position + 2 score + 1 WDL
const max_game_records = 512;
const default_depth: u8 = 8;
const default_games: usize = 1000;
const default_threads: usize = 4;
const random_plies: u16 = 8;
const skip_plies: u16 = 16;
const score_filter: i32 = 3000;
const sample_interval: u16 = 4;
const adjudication_threshold: i32 = 1500;
const adjudication_count: u16 = 5;

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
    network: ?*const nnue.Network,
    depth: u8,
    ply: u16,
    records: [max_game_records]TrainingRecord,
    num_records: usize,
    adjudication_consecutive: u16,
    adjudication_winning_side: engine.Color,

    fn init(rng: std.Random, depth: u8, ttable: *engine.search.TranspositionTable, network: ?*const nnue.Network) Self {
        return .{
            .state = .defaultPosition(),
            .rng = rng,
            .history = .{},
            .ttable = ttable,
            .network = network,
            .depth = depth,
            .ply = 0,
            .records = undefined,
            .num_records = 0,
            .adjudication_consecutive = 0,
            .adjudication_winning_side = engine.Colors.white,
        };
    }

    fn reset(self: *Self) void {
        self.state = .defaultPosition();
        self.history = .{};
        self.ply = 0;
        self.num_records = 0;
        self.adjudication_consecutive = 0;
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
        if (score > score_filter or score < -score_filter) return false;
        if (self.ply % sample_interval != 0) return false;
        return true;
    }

    fn bufferPosition(self: *Self, score: i32) !void {
        if (self.num_records >= max_game_records) return;
        var record = &self.records[self.num_records];
        record.position = @splat(0);
        try serde.encodePositionToBuffer(self.state, &record.position);
        record.score = std.math.cast(i16, score) orelse
            if (score > 0) std.math.maxInt(i16) else std.math.minInt(i16);
        self.num_records += 1;
    }

    // Search, optionally record the position, then make the best move.
    // Returns an outcome if the game ends.
    fn makeMoveAtDepth(self: *Self) !?GameOutcome {
        if (engine.search.isGameOverWithHistory(&self.state, &self.history)) |res| {
            return gameResultToOutcome(res);
        }

        const search_res = (try engine.search.searchWithHistory(
            &self.state,
            self.depth,
            1,
            &self.history,
            self.ttable,
            null,
        )) orelse return error.SearchFailed;

        const best_move = search_res.move;
        const score = search_res.score;

        if (self.shouldRecord(score)) {
            const record_score = if (self.network) |net|
                nnue.evaluate(&self.state, net)
            else
                score;
            try self.bufferPosition(record_score);
        }

        const abs_score = @as(i32, @intCast(@abs(score)));
        if (abs_score > adjudication_threshold) {
            const winning_side: engine.Color = if (score > 0) self.state.to_move else ~self.state.to_move;
            if (self.adjudication_consecutive > 0 and winning_side == self.adjudication_winning_side) {
                self.adjudication_consecutive += 1;
            } else {
                self.adjudication_consecutive = 1;
                self.adjudication_winning_side = winning_side;
            }
            if (self.adjudication_consecutive >= adjudication_count) {
                return if (winning_side == engine.Colors.white) .white_wins else .black_wins;
            }
        } else {
            self.adjudication_consecutive = 0;
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

        // Random opening phase
        for (0..random_plies) |_| {
            if (self.makeRandomMove()) |outcome| {
                return .{ .outcome = outcome, .positions = self.num_records };
            }
        }

        // Search phase
        while (true) {
            if (try self.makeMoveAtDepth()) |outcome| {
                return .{ .outcome = outcome, .positions = self.num_records };
            }
        }
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
    seed: u64,
    writer: *std.Io.Writer,
    write_mutex: *std.Io.Mutex,
    stderr: *std.Io.Writer,
    stderr_mutex: *std.Io.Mutex,
    io: std.Io,
    total_positions: *std.atomic.Value(usize),
    total_games: *std.atomic.Value(usize),
    network: ?*const nnue.Network,
};

fn workerLoop(ctx: *WorkerCtx) void {
    var ttable = engine.search.TranspositionTable.init(std.heap.page_allocator) catch return;
    defer ttable.deinit();

    var rng = std.Random.Pcg.init(ctx.seed);
    var game = SelfplayGame.init(rng.random(), ctx.depth, &ttable, ctx.network);

    for (0..ctx.games_per_worker) |_| {
        const result = game.playGame() catch continue;

        if (result.positions > 0) {
            ctx.write_mutex.lock(ctx.io) catch continue;
            defer ctx.write_mutex.unlock(ctx.io);
            game.writeRecords(ctx.writer, result.outcome) catch return;
            ctx.writer.flush() catch return;
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
}

const nnue = engine.nnue;

const Args = struct {
    depth: ?u8,
    num_games: ?usize,
    num_threads: ?usize,
    eval: ?[]const u8,
};

pub fn main(init: std.process.Init.Minimal) !void {
    const arg_parser = try kore.args.declarative.Parser(Args);

    var args_iter = if (builtin.os.tag == .windows)
        try init.args.iterateAllocator(std.heap.page_allocator)
    else
        init.args.iterate();
    const parsed_args = try arg_parser.parse(&args_iter);

    const depth = parsed_args.depth orelse default_depth;
    const num_games = parsed_args.num_games orelse default_games;
    const num_threads = parsed_args.num_threads orelse default_threads;

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

    try stderr.print("Selfplay: {d} games, depth {d}, {d} thread(s)", .{ num_games, depth, num_threads });
    if (network != null) try stderr.print(", NNUE eval", .{});
    try stderr.print("\n", .{});
    try stderr.print("Record format: {d} bytes (32 pos + 2 score + 1 wdl)\n", .{record_size});
    try stderr.flush();

    const actual_threads = @min(num_threads, 16);
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

    var contexts: [16]WorkerCtx = undefined;
    for (0..actual_threads) |i| {
        contexts[i] = .{
            .games_per_worker = games_per_worker + @as(usize, if (i < remainder) 1 else 0),
            .depth = depth,
            .seed = base_seed +% i,
            .writer = stdout,
            .write_mutex = &write_mutex,
            .stderr = stderr,
            .stderr_mutex = &stderr_mutex,
            .io = io,
            .total_positions = &shared_positions,
            .total_games = &shared_games,
            .network = network,
        };
    }

    // Spawn worker threads (thread 0 runs on main)
    var threads: [16]std.Thread = undefined;
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
    try stderr.flush();
}

test {
    std.testing.refAllDecls(@This());
}
