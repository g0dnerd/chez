const std = @import("std");
const chez = @import("chez.zig");
const engine = chez.engine;
const nnue = engine.nnue;
const search = engine.search;
const State = engine.State;
const Move = engine.Move;
const piece = engine.piece;
const squaremod = engine.square;
const movegen = engine.movegen;

const engine_name = "Chez";
const engine_author = "Paul";

const checkmate_score: i32 = 100000;
const mate_score_threshold: i32 = 99900;

// Apply a move (in UCI notation, e.g. "e2e4", "a7a8q") to a state.
// Looks up the move in the legal move list so the engine's Move struct is
// used directly (avoids promotion_piece undefined-value issues).
// Returns true on success.
fn applyMove(state: *State, notation: []const u8) bool {
    const trimmed = std.mem.trim(u8, notation, &std.ascii.whitespace);
    if (trimmed.len < 4) return false;
    const from = squaremod.algebraicToSquare(trimmed[0..2]) orelse return false;
    const to = squaremod.algebraicToSquare(trimmed[2..4]) orelse return false;

    const legal = movegen.legalMoves(state, state.to_move);

    for (0..legal.len) |i| {
        const m = legal.moves[i];
        if (m.start != from or m.end != to) continue;
        if (trimmed.len > 4) {
            if (!m.is_promotion) continue;
            const promo_piece: piece.Piece = switch (trimmed[4]) {
                'q' => piece.queen,
                'r' => piece.rook,
                'b' => piece.bishop,
                'n' => piece.knight,
                else => return false,
            };
            if (m.promotion_piece != promo_piece) continue;
        } else {
            if (m.is_promotion) continue;
        }
        const p = state.mailbox[from] orelse return false;
        _ = state.makeMove(m, state.to_move, p);
        return true;
    }
    return false;
}

// Context passed to the per-depth info callback.
// Both writer and mutex are owned by main(); the callback runs from the
// search thread so the mutex serialises stdout access.
const InfoCtx = struct {
    writer: *std.Io.Writer,
    mutex: *std.Io.Mutex,
    io: std.Io,
};

fn infoCallback(
    ctx_ptr: ?*anyopaque,
    depth: u8,
    score: i32,
    nodes: u64,
    time_ms: u64,
    pv: []const Move,
) void {
    const ctx: *InfoCtx = @ptrCast(@alignCast(ctx_ptr.?));
    ctx.mutex.lock(ctx.io) catch unreachable;
    defer ctx.mutex.unlock(ctx.io);

    if (score >= mate_score_threshold) {
        const plies = checkmate_score - score;
        const full_moves = @divTrunc(plies + 1, 2);
        ctx.writer.print("info depth {d} score mate {d} nodes {d} time {d}", .{ depth, full_moves, nodes, time_ms }) catch return;
    } else if (score <= -mate_score_threshold) {
        const plies = checkmate_score + score;
        const full_moves = @divTrunc(plies + 1, 2);
        ctx.writer.print("info depth {d} score mate -{d} nodes {d} time {d}", .{ depth, full_moves, nodes, time_ms }) catch return;
    } else {
        ctx.writer.print("info depth {d} score cp {d} nodes {d} time {d}", .{ depth, score, nodes, time_ms }) catch return;
    }

    if (pv.len > 0) {
        ctx.writer.writeAll(" pv") catch return;
        for (pv) |m| {
            ctx.writer.writeByte(' ') catch return;
            ctx.writer.print("{f}", .{m}) catch return;
        }
    }
    ctx.writer.writeByte('\n') catch return;
    ctx.writer.flush() catch return;
}

// Arguments owned by main() for the lifetime of a search.
// Passed by pointer to the search thread.
const SearchRunArgs = struct {
    state: State,
    max_depth: u8,
    num_threads: usize,
    history: search.PositionHistory,
    tbl: *search.TranspositionTable,
    options: search.SearchOptions,
    network: ?*const nnue.Network,
    writer: *std.Io.Writer,
    mutex: *std.Io.Mutex,
    io: std.Io,
};

fn runSearch(args: *SearchRunArgs) void {
    const result = search.searchParallel(
        &args.state,
        args.max_depth,
        args.num_threads,
        &args.history,
        args.tbl,
        args.options,
        args.network,
    ) catch null;

    args.mutex.lock(args.io) catch {};
    defer args.mutex.unlock(args.io);

    if (result) |r| {
        args.writer.print("bestmove {f}\n", .{r.move}) catch {};
    } else {
        args.writer.writeAll("bestmove 0000\n") catch {};
    }
    args.writer.flush() catch {};
}

pub fn main() !void {
    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .environ = .empty });
    const io = threaded.io();

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);

    var stdout_mutex = std.Io.Mutex.init;

    var state = State.defaultPosition();
    var history = search.PositionHistory{};
    history.push(state.zobrist_hash);

    var tbl = try search.TranspositionTable.init(std.heap.page_allocator);
    defer tbl.deinit();

    var num_threads: usize = 4;
    var stop_flag = std.atomic.Value(bool).init(false);
    var search_thread: ?std.Thread = null;
    var search_run_args: SearchRunArgs = undefined;

    var own_book = true;
    var opening_book: ?engine.book.Book = null;
    defer if (opening_book) |*b| b.deinit();

    var network: ?*nnue.Network = null;
    defer if (network) |n| n.deinit(std.heap.page_allocator);

    var search_params = search.SearchParams{};

    var info_ctx = InfoCtx{
        .writer = stdout,
        .mutex = &stdout_mutex,
        .io = io,
    };

    while (true) {
        search_thread = null;

        const line_raw = stdin_reader.interface.takeDelimiterExclusive('\n') catch break;
        stdin_reader.interface.toss(1);
        const line = std.mem.trimEnd(u8, line_raw, &std.ascii.whitespace);

        if (std.mem.eql(u8, line, "uci")) {
            try stdout_mutex.lock(io);
            stdout.print("id name {s}\n", .{engine_name}) catch {};
            stdout.print("id author {s}\n", .{engine_author}) catch {};
            stdout.writeAll("option name Threads type spin default 4 min 1 max 16\n") catch {};
            stdout.writeAll("option name OwnBook type check default true\n") catch {};
            stdout.writeAll("option name BookFile type string default /home/paul/projects/chez/testing/books/komodo.bin\n") catch {};
            stdout.writeAll("option name EvalFile type string default <empty>\n") catch {};
            stdout.writeAll("option name NnueScale type spin default 2 min 1 max 10\n") catch {};
            stdout.writeAll("option name RfpBase type spin default 80 min 20 max 200\n") catch {};
            stdout.writeAll("option name FutilityMargin1 type spin default 300 min 50 max 800\n") catch {};
            stdout.writeAll("option name FutilityMargin2 type spin default 600 min 100 max 1500\n") catch {};
            stdout.writeAll("option name DeltaMargin type spin default 200 min 50 max 600\n") catch {};
            stdout.writeAll("option name LmrBase type spin default 75 min 0 max 300\n") catch {};
            stdout.writeAll("option name LmrDiv type spin default 120 min 50 max 500\n") catch {};
            stdout.writeAll("option name LmrHistDiv type spin default 8000 min 500 max 32000\n") catch {};
            stdout.writeAll("option name HistPruneDepth type spin default 3 min 0 max 8\n") catch {};
            stdout.writeAll("option name HistPruneMargin type spin default 2000 min 200 max 12000\n") catch {};
            stdout.writeAll("option name IirMinDepth type spin default 4 min 2 max 12\n") catch {};
            stdout.writeAll("uciok\n") catch {};
            stdout.flush() catch {};
            stdout_mutex.unlock(io);
        } else if (std.mem.eql(u8, line, "isready")) {
            try stdout_mutex.lock(io);
            stdout.writeAll("readyok\n") catch {};
            stdout.flush() catch {};
            stdout_mutex.unlock(io);
        } else if (std.mem.eql(u8, line, "ucinewgame")) {
            if (search_thread) |t| {
                stop_flag.store(true, .release);
                t.join();
                search_thread = null;
            }
            state = State.defaultPosition();
            history = search.PositionHistory{};
            history.push(state.zobrist_hash);
            tbl.newSearch();
        } else if (std.mem.startsWith(u8, line, "setoption ")) {
            const name_prefix = "setoption name ";
            if (line.len <= name_prefix.len) continue;
            const rest = line[name_prefix.len..];
            const value_sep = std.mem.indexOf(u8, rest, " value ");
            const opt_name = if (value_sep) |idx| rest[0..idx] else rest;
            const opt_val = if (value_sep) |idx| rest[idx + " value ".len ..] else "";

            if (std.mem.eql(u8, opt_name, "Threads")) {
                num_threads = std.fmt.parseInt(usize, opt_val, 10) catch num_threads;
            } else if (std.mem.eql(u8, opt_name, "OwnBook")) {
                own_book = std.mem.eql(u8, opt_val, "true");
            } else if (std.mem.eql(u8, opt_name, "BookFile")) {
                if (opt_val.len > 0) {
                    if (opening_book) |*b| b.deinit();
                    opening_book = engine.book.Book.load(io, std.heap.page_allocator, opt_val) catch null;
                }
            } else if (std.mem.eql(u8, opt_name, "EvalFile")) {
                if (opt_val.len > 0) {
                    if (network) |n| n.deinit(std.heap.page_allocator);
                    network = nnue.Network.load(io, std.heap.page_allocator, opt_val) catch null;
                }
            } else if (std.mem.eql(u8, opt_name, "NnueScale")) {
                search_params.nnue_scale = std.fmt.parseInt(i32, opt_val, 10) catch search_params.nnue_scale;
            } else if (std.mem.eql(u8, opt_name, "RfpBase")) {
                search_params.rfp_base = std.fmt.parseInt(i32, opt_val, 10) catch search_params.rfp_base;
            } else if (std.mem.eql(u8, opt_name, "FutilityMargin1")) {
                search_params.futility_margin_1 = std.fmt.parseInt(i32, opt_val, 10) catch search_params.futility_margin_1;
            } else if (std.mem.eql(u8, opt_name, "FutilityMargin2")) {
                search_params.futility_margin_2 = std.fmt.parseInt(i32, opt_val, 10) catch search_params.futility_margin_2;
            } else if (std.mem.eql(u8, opt_name, "DeltaMargin")) {
                search_params.delta_margin = std.fmt.parseInt(i32, opt_val, 10) catch search_params.delta_margin;
            } else if (std.mem.eql(u8, opt_name, "LmrBase")) {
                search_params.lmr_base = std.fmt.parseInt(i32, opt_val, 10) catch search_params.lmr_base;
            } else if (std.mem.eql(u8, opt_name, "LmrDiv")) {
                search_params.lmr_div = std.fmt.parseInt(i32, opt_val, 10) catch search_params.lmr_div;
            } else if (std.mem.eql(u8, opt_name, "LmrHistDiv")) {
                search_params.lmr_hist_div = std.fmt.parseInt(i32, opt_val, 10) catch search_params.lmr_hist_div;
            } else if (std.mem.eql(u8, opt_name, "HistPruneDepth")) {
                search_params.histprune_depth = std.fmt.parseInt(i32, opt_val, 10) catch search_params.histprune_depth;
            } else if (std.mem.eql(u8, opt_name, "HistPruneMargin")) {
                search_params.histprune_margin = std.fmt.parseInt(i32, opt_val, 10) catch search_params.histprune_margin;
            } else if (std.mem.eql(u8, opt_name, "IirMinDepth")) {
                search_params.iir_min_depth = std.fmt.parseInt(i32, opt_val, 10) catch search_params.iir_min_depth;
            }
        } else if (std.mem.startsWith(u8, line, "position")) {
            if (search_thread != null) continue;

            var rest = line["position".len..];

            if (std.mem.startsWith(u8, rest, " startpos")) {
                state = State.defaultPosition();
                history = search.PositionHistory{};
                history.push(state.zobrist_hash);
                rest = rest[" startpos".len..];
            } else if (std.mem.startsWith(u8, rest, " fen ")) {
                rest = rest[" fen ".len..];
                const moves_idx = std.mem.indexOf(u8, rest, " moves");
                const fen = if (moves_idx) |idx| rest[0..idx] else rest;
                state = State.fromFen(fen) catch continue;
                history = search.PositionHistory{};
                history.push(state.zobrist_hash);
                rest = if (moves_idx) |idx| rest[idx..] else "";
            }

            if (std.mem.startsWith(u8, rest, " moves ")) {
                rest = rest[" moves ".len..];
                var moves_it = std.mem.splitScalar(u8, rest, ' ');
                while (moves_it.next()) |move_str| {
                    const ms = std.mem.trim(u8, move_str, &std.ascii.whitespace);
                    if (ms.len == 0) continue;
                    if (applyMove(&state, ms)) {
                        history.push(state.zobrist_hash);
                    }
                }
            }
        } else if (std.mem.startsWith(u8, line, "go")) {
            if (search_thread != null) continue;

            // Probe opening book before search
            if (own_book) {
                if (opening_book) |*b| {
                    if (b.probe(&state)) |book_move| {
                        try stdout_mutex.lock(io);
                        stdout.print("bestmove {f}\n", .{book_move}) catch {};
                        stdout.flush() catch {};
                        stdout_mutex.unlock(io);
                        continue;
                    }
                }
            }

            var max_depth: u8 = 64;
            var max_time_ms: ?u64 = null;
            var wtime: ?i64 = null;
            var btime: ?i64 = null;
            var winc: i64 = 0;
            var binc: i64 = 0;
            var infinite = false;

            var it = std.mem.splitScalar(u8, line, ' ');
            _ = it.next(); // "go"
            while (it.next()) |token| {
                if (std.mem.eql(u8, token, "depth")) {
                    if (it.next()) |v| max_depth = std.fmt.parseInt(u8, v, 10) catch 64;
                } else if (std.mem.eql(u8, token, "movetime")) {
                    if (it.next()) |v| max_time_ms = std.fmt.parseInt(u64, v, 10) catch null;
                } else if (std.mem.eql(u8, token, "infinite")) {
                    infinite = true;
                } else if (std.mem.eql(u8, token, "wtime")) {
                    if (it.next()) |v| wtime = std.fmt.parseInt(i64, v, 10) catch null;
                } else if (std.mem.eql(u8, token, "btime")) {
                    if (it.next()) |v| btime = std.fmt.parseInt(i64, v, 10) catch null;
                } else if (std.mem.eql(u8, token, "winc")) {
                    if (it.next()) |v| winc = std.fmt.parseInt(i64, v, 10) catch 0;
                } else if (std.mem.eql(u8, token, "binc")) {
                    if (it.next()) |v| binc = std.fmt.parseInt(i64, v, 10) catch 0;
                }
            }

            // Calculate time budget from clock if movetime/infinite not specified
            if (!infinite and max_time_ms == null) {
                const our_time = if (state.to_move == engine.Colors.white) wtime else btime;
                const our_inc = if (state.to_move == engine.Colors.white) winc else binc;
                if (our_time) |t| {
                    var budget: i64 = @divTrunc(t, 20) + @divTrunc(our_inc, 2);
                    if (budget > t - 50) budget = t - 50;
                    if (budget < 10) budget = 10;
                    max_time_ms = @intCast(budget);
                }
            }

            stop_flag.store(false, .release);
            search_run_args = .{
                .state = state,
                .max_depth = max_depth,
                .num_threads = num_threads,
                .history = history,
                .tbl = &tbl,
                .options = .{
                    .stop = &stop_flag,
                    .max_time_ms = max_time_ms,
                    .on_info = .{
                        .context = &info_ctx,
                        .func = infoCallback,
                    },
                    .search_params = search_params,
                },
                .network = network,
                .io = io,
                .writer = stdout,
                .mutex = &stdout_mutex,
            };
            search_thread = try std.Thread.spawn(.{}, runSearch, .{&search_run_args});
        } else if (std.mem.eql(u8, line, "stop")) {
            stop_flag.store(true, .release);
            if (search_thread) |t| {
                t.join();
                search_thread = null;
            }
        } else if (std.mem.eql(u8, line, "quit")) {
            stop_flag.store(true, .release);
            if (search_thread) |t| {
                t.join();
            }
            break;
        }
    }

    // EOF or quit: stop any in-flight search before TT is freed.
    stop_flag.store(true, .release);
    if (search_thread) |t| {
        t.join();
    }
}
