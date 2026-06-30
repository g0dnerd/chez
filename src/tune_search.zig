// In-engine SPSA tuner for SEARCH parameters, with NNUE loaded.
//
// Unlike `tune` (a Texel/MSE eval-param tuner), search params (RFP/futility/
// delta/LMR) have no effect on a static eval, so they can only be tuned against
// a game-playing objective. This binary perturbs SearchParams as theta +/- c*D,
// plays theta+ DIRECTLY against theta- in fast fixed-node self-play games with
// the same net, and SPSA-steps theta toward the winner (fishtest-style).
//
//   zig build tune-search -- --net data/net.nnue --nodes 5000 \
//       --games_per_step 100 --iterations 300 --threads 8
//
// Output is a paste-able SearchParams literal. SPSA self-play only proves a
// RELATIVE gain: validate the result vs an external SF anchor (gauntlet.py)
// before keeping it.

const builtin = @import("builtin");
const std = @import("std");
const engine = @import("chez").engine;
const kore = @import("kore");

const nnue = engine.nnue;
const search = engine.search;
const SearchParams = search.SearchParams;
const ReusableSearcher = search.ReusableSearcher;
const TranspositionTable = search.TranspositionTable;

// Game generation knobs (mirrors selfplay's adjudication so games stay short).
const random_plies: u16 = 8;
const max_game_plies: u16 = 200;
const tt_buckets_bits: u6 = 18; // tiny TT; node-capped searches never need more
const win_adj_threshold: i32 = 2500;
const win_adj_count: u16 = 4;
const draw_adj_threshold: i32 = 10;
const draw_adj_count: u16 = 8;
const draw_adj_min_ply: u16 = 80;
const max_threads: usize = 256;

// SPSA defaults.
const default_nodes: u64 = 5000;
const default_depth: u8 = 16; // ceiling; the node cap governs in practice
const default_pairs: usize = 100; // game-pairs (2 games each) per SPSA step
const default_iterations: usize = 300;
const default_threads: usize = 8;
const default_a: f64 = 0.05;
const default_c: f64 = 0.08;
const default_alpha: f64 = 0.602;
const default_gamma: f64 = 0.101;

// High-impact continuous search params. theta is normalized to [0,1] over
// [min,max]; coarse/low-signal params (histprune_*, iir, nnue_scale) are frozen
// at their SearchParams{} defaults.
const ParamSpec = struct { name: []const u8, min: f64, max: f64, default: f64 };
const tuned = [_]ParamSpec{
    .{ .name = "rfp_base", .min = 20, .max = 200, .default = 80 },
    .{ .name = "futility_margin_1", .min = 50, .max = 800, .default = 300 },
    .{ .name = "futility_margin_2", .min = 100, .max = 1500, .default = 600 },
    .{ .name = "delta_margin", .min = 50, .max = 600, .default = 200 },
    .{ .name = "lmr_base", .min = 0, .max = 300, .default = 75 },
    .{ .name = "lmr_div", .min = 50, .max = 500, .default = 120 },
    .{ .name = "lmr_hist_div", .min = 500, .max = 32000, .default = 8000 },
    // NNUE output scaling. max=100/min=0 respectively let SPSA disable the
    // feature (material_scale_min=100 => no compression, fifty_move_damp=0 => no damp).
    .{ .name = "material_scale_min", .min = 50, .max = 100, .default = 75 },
    .{ .name = "fifty_move_start", .min = 0, .max = 60, .default = 20 },
    .{ .name = "fifty_move_damp", .min = 0, .max = 90, .default = 50 },
};
const N = tuned.len;

const GameOutcome = enum(u8) { white_wins, black_wins, draw };

fn denorm(t: f64, spec: ParamSpec) i32 {
    const clamped = std.math.clamp(t, 0.0, 1.0);
    return @intFromFloat(@round(spec.min + clamped * (spec.max - spec.min)));
}

// Build a full SearchParams from a normalized theta. Frozen fields keep their
// SearchParams{} defaults (which now include nnue_scale = 2).
fn buildParams(theta: [N]f64) SearchParams {
    var p = SearchParams{};
    p.rfp_base = denorm(theta[0], tuned[0]);
    p.futility_margin_1 = denorm(theta[1], tuned[1]);
    p.futility_margin_2 = denorm(theta[2], tuned[2]);
    p.delta_margin = denorm(theta[3], tuned[3]);
    p.lmr_base = denorm(theta[4], tuned[4]);
    p.lmr_div = denorm(theta[5], tuned[5]);
    p.lmr_hist_div = denorm(theta[6], tuned[6]);
    p.material_scale_min = denorm(theta[7], tuned[7]);
    p.fifty_move_start = denorm(theta[8], tuned[8]);
    p.fifty_move_damp = denorm(theta[9], tuned[9]);
    return p;
}

const Opening = struct {
    state: engine.State,
    history: search.PositionHistory,
    ply: u16,
};

// Play random_plies random moves from the start. Returns null if the line
// happens to end the game (caller retries with the next RNG draw).
fn buildOpening(rng: std.Random) ?Opening {
    var state = engine.State.defaultPosition();
    var history = search.PositionHistory{};
    var ply: u16 = 0;
    for (0..random_plies) |_| {
        const to_move = state.to_move;
        const moves = engine.movegen.legalMoves(&state, to_move);
        if (moves.len == 0) return null;
        const mv = moves.moves[rng.uintLessThan(u8, moves.len)];
        _ = state.makeMove(mv, to_move, state.mailbox[mv.start].?);
        history.push(state.zobrist_hash);
        ply += 1;
    }
    if (search.isGameOverWithHistory(&state, &history) != null) return null;
    return .{ .state = state, .history = history, .ply = ply };
}

fn outcomeOf(res: engine.GameResult) GameOutcome {
    return switch (res) {
        .checkmate => |color| if (color == engine.Colors.white) .white_wins else .black_wins,
        else => .draw,
    };
}

// Half-points (win=2, draw=1, loss=0) scored from theta+'s perspective.
fn plusHalf(o: GameOutcome, plus_is_white: bool) u64 {
    return switch (o) {
        .draw => 1,
        .white_wins => if (plus_is_white) 2 else 0,
        .black_wins => if (plus_is_white) 0 else 2,
    };
}

// One game between the theta+ and theta- engines from a fixed opening. Each
// side keeps its own searcher (LMR table baked from its params) and its own TT.
const TunerGame = struct {
    searcher_plus: *ReusableSearcher,
    searcher_minus: *ReusableSearcher,
    params_plus: SearchParams,
    params_minus: SearchParams,
    tt_plus: *TranspositionTable,
    tt_minus: *TranspositionTable,
    network: *const nnue.Network,
    nodes: ?u64,
    depth: u8,

    state: engine.State,
    history: search.PositionHistory,
    ply: u16,
    adj_consec: u16,
    adj_side: engine.Color,
    draw_consec: u16,

    fn play(self: *TunerGame, op: Opening, plus_is_white: bool) GameOutcome {
        self.state = op.state;
        self.history = op.history;
        self.ply = op.ply;
        self.adj_consec = 0;
        self.adj_side = engine.Colors.white;
        self.draw_consec = 0;
        self.tt_plus.newSearch();
        self.tt_minus.newSearch();

        while (self.ply < max_game_plies) {
            if (search.isGameOverWithHistory(&self.state, &self.history)) |res| {
                return outcomeOf(res);
            }

            const plus_to_move = (self.state.to_move == engine.Colors.white) == plus_is_white;
            const searcher = if (plus_to_move) self.searcher_plus else self.searcher_minus;
            const params = if (plus_to_move) self.params_plus else self.params_minus;
            const tt = if (plus_to_move) self.tt_plus else self.tt_minus;

            const res = searcher.search(
                &self.state,
                self.depth,
                &self.history,
                tt,
                .{ .max_nodes = self.nodes, .search_params = params },
                self.network,
            ) orelse return .draw;

            const score = res.score;
            const abs_score = @as(i32, @intCast(@abs(score)));

            // Win adjudication: a side clearly winning for several plies in a row.
            if (abs_score > win_adj_threshold) {
                const winning_side: engine.Color = if (score > 0) self.state.to_move else ~self.state.to_move;
                if (self.adj_consec > 0 and winning_side == self.adj_side) {
                    self.adj_consec += 1;
                } else {
                    self.adj_consec = 1;
                    self.adj_side = winning_side;
                }
                if (self.adj_consec >= win_adj_count) {
                    return if (winning_side == engine.Colors.white) .white_wins else .black_wins;
                }
            } else {
                self.adj_consec = 0;
            }

            // Draw adjudication: a long run of near-zero evals past the opening.
            if (self.ply >= draw_adj_min_ply and abs_score <= draw_adj_threshold) {
                self.draw_consec += 1;
                if (self.draw_consec >= draw_adj_count) return .draw;
            } else {
                self.draw_consec = 0;
            }

            _ = self.state.makeMove(res.move, self.state.to_move, self.state.mailbox[res.move.start].?);
            self.history.push(self.state.zobrist_hash);
            self.ply += 1;
        }
        return .draw;
    }
};

const WorkerCtx = struct {
    pairs: usize,
    params_plus: SearchParams,
    params_minus: SearchParams,
    nodes: ?u64,
    depth: u8,
    seed: u64,
    network: *const nnue.Network,
    plus_halfpoints: *std.atomic.Value(u64),
    games_played: *std.atomic.Value(u64),
};

fn workerLoop(ctx: *WorkerCtx) void {
    var tt_plus = TranspositionTable.initSized(std.heap.page_allocator, tt_buckets_bits) catch return;
    defer tt_plus.deinit();
    var tt_minus = TranspositionTable.initSized(std.heap.page_allocator, tt_buckets_bits) catch return;
    defer tt_minus.deinit();
    var searcher_plus = ReusableSearcher.init(std.heap.page_allocator, ctx.params_plus) catch return;
    defer searcher_plus.deinit();
    var searcher_minus = ReusableSearcher.init(std.heap.page_allocator, ctx.params_minus) catch return;
    defer searcher_minus.deinit();

    var rng = std.Random.Pcg.init(ctx.seed);

    var game = TunerGame{
        .searcher_plus = &searcher_plus,
        .searcher_minus = &searcher_minus,
        .params_plus = ctx.params_plus,
        .params_minus = ctx.params_minus,
        .tt_plus = &tt_plus,
        .tt_minus = &tt_minus,
        .network = ctx.network,
        .nodes = ctx.nodes,
        .depth = ctx.depth,
        .state = undefined,
        .history = undefined,
        .ply = 0,
        .adj_consec = 0,
        .adj_side = engine.Colors.white,
        .draw_consec = 0,
    };

    var local_half: u64 = 0;
    var local_games: u64 = 0;
    var done: usize = 0;
    while (done < ctx.pairs) {
        const op = buildOpening(rng.random()) orelse continue;
        // Paired games: same opening, theta+ plays both colors to cancel bias.
        local_half += plusHalf(game.play(op, true), true);
        local_half += plusHalf(game.play(op, false), false);
        local_games += 2;
        done += 1;
    }

    _ = ctx.plus_halfpoints.fetchAdd(local_half, .monotonic);
    _ = ctx.games_played.fetchAdd(local_games, .monotonic);
}

const IterResult = struct { half: u64, games: u64 };

fn runIteration(
    params_plus: SearchParams,
    params_minus: SearchParams,
    total_pairs: usize,
    num_threads: usize,
    base_seed: u64,
    network: *const nnue.Network,
    nodes: ?u64,
    depth: u8,
) IterResult {
    var plus_half = std.atomic.Value(u64).init(0);
    var games = std.atomic.Value(u64).init(0);

    const per = total_pairs / num_threads;
    const rem = total_pairs % num_threads;

    var contexts: [max_threads]WorkerCtx = undefined;
    for (0..num_threads) |i| {
        contexts[i] = .{
            .pairs = per + @as(usize, if (i < rem) 1 else 0),
            .params_plus = params_plus,
            .params_minus = params_minus,
            .nodes = nodes,
            .depth = depth,
            .seed = base_seed +% (@as(u64, i) *% 0x9E3779B97F4A7C15),
            .network = network,
            .plus_halfpoints = &plus_half,
            .games_played = &games,
        };
    }

    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;
    for (1..num_threads) |i| {
        if (contexts[i].pairs == 0) continue;
        threads[i] = std.Thread.spawn(.{}, workerLoop, .{&contexts[i]}) catch break;
        spawned = i;
    }
    workerLoop(&contexts[0]);
    for (1..spawned + 1) |i| {
        if (contexts[i].pairs == 0) continue;
        threads[i].join();
    }

    return .{ .half = plus_half.load(.monotonic), .games = games.load(.monotonic) };
}

fn formatParams(w: *std.Io.Writer, theta: [N]f64) !void {
    try w.print(
        \\.{{ .nnue_scale = 2, .rfp_base = {d}, .futility_margin_1 = {d}, .futility_margin_2 = {d}, .delta_margin = {d}, .lmr_base = {d}, .lmr_div = {d}, .lmr_hist_div = {d}, .material_scale_min = {d}, .fifty_move_start = {d}, .fifty_move_damp = {d}, .histprune_depth = 3, .histprune_margin = 2000, .iir_min_depth = 4 }}
        \\
    , .{
        denorm(theta[0], tuned[0]),
        denorm(theta[1], tuned[1]),
        denorm(theta[2], tuned[2]),
        denorm(theta[3], tuned[3]),
        denorm(theta[4], tuned[4]),
        denorm(theta[5], tuned[5]),
        denorm(theta[6], tuned[6]),
        denorm(theta[7], tuned[7]),
        denorm(theta[8], tuned[8]),
        denorm(theta[9], tuned[9]),
    });
}

fn writeOutput(io: std.Io, path: []const u8, theta: [N]f64) !void {
    const out_file = try std.Io.Dir.cwd().createFile(io, path, .{});
    var wbuf: [4096]u8 = undefined;
    var writer = out_file.writer(io, &wbuf);
    const w: *std.Io.Writer = &writer.interface;
    try formatParams(w, theta);
    try w.flush();
    out_file.close(io);
}

const Args = struct {
    net: ?[]const u8,
    nodes: ?u64,
    depth: ?u8,
    games_per_step: ?usize,
    iterations: ?usize,
    threads: ?usize,
    seed: ?u64,
    a: ?f64,
    c: ?f64,
    alpha: ?f64,
    gamma: ?f64,
    big_a: ?f64,
    output: ?[]const u8,
    checkpoint_interval: ?usize,
};

pub fn main(init: std.process.Init.Minimal) !void {
    const arg_parser = try kore.args.declarative.Parser(Args);
    var args_iter = if (builtin.os.tag == .windows)
        try init.args.iterateAllocator(std.heap.page_allocator)
    else
        init.args.iterate();
    const parsed_args = try arg_parser.parse(&args_iter);

    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .environ = .empty });
    const io = threaded.io();

    var stderr_buf: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buf);
    const stderr: *std.Io.Writer = &stderr_writer.interface;

    const net_path = parsed_args.net orelse {
        try stderr.print("error: --net <path.nnue> is required (search params are tuned with the net loaded)\n", .{});
        try stderr.flush();
        return;
    };

    const network: *nnue.Network = nnue.Network.load(io, std.heap.page_allocator, net_path) catch |err| {
        try stderr.print("error: could not load {s}: {}\n", .{ net_path, err });
        try stderr.flush();
        return;
    };
    defer network.deinit(std.heap.page_allocator);

    const nodes: ?u64 = parsed_args.nodes orelse default_nodes;
    const depth = parsed_args.depth orelse default_depth;
    const pairs = parsed_args.games_per_step orelse default_pairs;
    const iterations = parsed_args.iterations orelse default_iterations;
    const num_threads = @min(parsed_args.threads orelse default_threads, max_threads);
    const a = parsed_args.a orelse default_a;
    const c = parsed_args.c orelse default_c;
    const alpha = parsed_args.alpha orelse default_alpha;
    const gamma = parsed_args.gamma orelse default_gamma;
    const big_a = parsed_args.big_a orelse @as(f64, @floatFromInt(iterations)) / 10.0;
    const checkpoint_interval = parsed_args.checkpoint_interval orelse 10;

    var seed: u64 = undefined;
    if (parsed_args.seed) |s| {
        seed = s;
    } else if (builtin.target.os.tag == .linux) {
        _ = std.os.linux.getrandom(std.mem.asBytes(&seed), 8, 0);
    } else {
        std.Io.random(io, std.mem.asBytes(&seed));
    }

    try stderr.print("SPSA search-param tuner: net {s}, nodes {?d}, {d} pairs/step ({d} games), {d} iters, {d} threads\n", .{
        net_path, nodes, pairs, pairs * 2, iterations, num_threads,
    });
    try stderr.print("Tuning {d} params: ", .{N});
    for (tuned) |spec| try stderr.print("{s} ", .{spec.name});
    try stderr.print("\nSPSA: a={d:.4} c={d:.4} alpha={d:.3} gamma={d:.3} A={d:.1} seed={d}\n", .{ a, c, alpha, gamma, big_a, seed });
    try stderr.flush();

    // theta starts at the normalized current defaults.
    var theta: [N]f64 = undefined;
    for (0..N) |i| theta[i] = (tuned[i].default - tuned[i].min) / (tuned[i].max - tuned[i].min);

    // Polyak tail-averaging over the back half for a more stable final estimate.
    var avg: [N]f64 = @splat(0);
    var avg_count: f64 = 0;

    var master = std.Random.Pcg.init(seed);
    const mrng = master.random();

    for (1..iterations + 1) |k| {
        const kf: f64 = @floatFromInt(k);
        const ck = c / std.math.pow(f64, kf, gamma);
        const ak = a / std.math.pow(f64, kf + big_a, alpha);

        var delta: [N]f64 = undefined;
        var theta_plus: [N]f64 = undefined;
        var theta_minus: [N]f64 = undefined;
        for (0..N) |i| {
            delta[i] = if (mrng.boolean()) 1.0 else -1.0;
            theta_plus[i] = std.math.clamp(theta[i] + ck * delta[i], 0.0, 1.0);
            theta_minus[i] = std.math.clamp(theta[i] - ck * delta[i], 0.0, 1.0);
        }

        const r = runIteration(
            buildParams(theta_plus),
            buildParams(theta_minus),
            pairs,
            num_threads,
            mrng.int(u64),
            network,
            nodes,
            depth,
        );
        if (r.games == 0) continue;

        // y = theta+'s score share in [0,1]; R in [-1,1] is the gradient signal.
        const y = @as(f64, @floatFromInt(r.half)) / (2.0 * @as(f64, @floatFromInt(r.games)));
        const big_r = 2.0 * (y - 0.5);

        // Gradient ascent toward the stronger perturbation (constants fold into a).
        for (0..N) |i| {
            theta[i] = std.math.clamp(theta[i] + ak * big_r * delta[i] / ck, 0.0, 1.0);
        }

        if (k > iterations / 2) {
            for (0..N) |i| avg[i] += theta[i];
            avg_count += 1;
        }

        if (k % checkpoint_interval == 0 or k == iterations) {
            try stderr.print("[{d}/{d}] theta+ score {d:.3} (R={d:.3})  cur=", .{ k, iterations, y, big_r });
            try formatParams(stderr, theta);
            try stderr.flush();
            if (parsed_args.output) |path| writeOutput(io, path, theta) catch |err| {
                try stderr.print("warning: could not write {s}: {}\n", .{ path, err });
                try stderr.flush();
            };
        }
    }

    try stderr.print("\n=== Final (current theta) ===\n", .{});
    try formatParams(stderr, theta);
    if (avg_count > 0) {
        var avg_theta: [N]f64 = undefined;
        for (0..N) |i| avg_theta[i] = avg[i] / avg_count;
        try stderr.print("=== Final (tail-averaged, recommended) ===\n", .{});
        try formatParams(stderr, avg_theta);
        if (parsed_args.output) |path| try writeOutput(io, path, avg_theta);
    }
    try stderr.print("\nValidate vs an external anchor before keeping:\n", .{});
    try stderr.print("  uv run python testing/gauntlet.py --config testing/gauntlet_sf_2plus1.json \\\n", .{});
    try stderr.print("    --eval {s} --threads 1 --rounds 200   # after pasting params into SearchParams\n", .{net_path});
    try stderr.flush();
}

test {
    std.testing.refAllDecls(@This());
}
