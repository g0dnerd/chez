// UCI protocol binary for Chez chess engine.
// Speaks the Universal Chess Interface protocol so the engine can be used
// with any standard chess GUI (Arena, Cutechess, Lichess, etc.).

const std = @import("std");
const chez = @import("chez.zig");
const engine = chez.engine;
const search = engine.search;
const State = engine.State;
const Move = engine.Move;
const piece = engine.piece;
const squaremod = engine.square;
const movegen = engine.movegen;

const engine_name = "Chez";
const engine_author = "Paul";

// Scores above this threshold indicate a forced mate
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

    var legal = movegen.legalMoves(state, state.to_move);
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
    mutex: *std.Thread.Mutex,
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
    ctx.mutex.lock();
    defer ctx.mutex.unlock();

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
    writer: *std.Io.Writer,
    mutex: *std.Thread.Mutex,
};

fn runSearch(args: *SearchRunArgs) void {
    const result = search.searchParallel(
        &args.state,
        args.max_depth,
        args.num_threads,
        &args.history,
        args.tbl,
        args.options,
    ) catch null;

    args.mutex.lock();
    defer args.mutex.unlock();

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

    var stdout_mutex = std.Thread.Mutex{};

    var state = State.defaultPosition();
    var history = search.PositionHistory.init();
    history.push(state.zobrist_hash);

    var tbl = try search.TranspositionTable.init(std.heap.page_allocator);
    defer tbl.deinit();

    var num_threads: usize = 8;
    var stop_flag = std.atomic.Value(bool).init(false);
    var search_thread: ?std.Thread = null;
    var search_run_args: SearchRunArgs = undefined;

    var info_ctx = InfoCtx{
        .writer = stdout,
        .mutex = &stdout_mutex,
    };

    while (true) {
        const line_raw = stdin_reader.interface.takeDelimiterExclusive('\n') catch break;
        stdin_reader.interface.toss(1);
        const line = std.mem.trimEnd(u8, line_raw, &std.ascii.whitespace);

        if (std.mem.eql(u8, line, "uci")) {
            stdout_mutex.lock();
            stdout.print("id name {s}\n", .{engine_name}) catch {};
            stdout.print("id author {s}\n", .{engine_author}) catch {};
            stdout.writeAll("option name Threads type spin default 8 min 1 max 16\n") catch {};
            stdout.writeAll("uciok\n") catch {};
            stdout.flush() catch {};
            stdout_mutex.unlock();
        } else if (std.mem.eql(u8, line, "isready")) {
            stdout_mutex.lock();
            stdout.writeAll("readyok\n") catch {};
            stdout.flush() catch {};
            stdout_mutex.unlock();
        } else if (std.mem.eql(u8, line, "ucinewgame")) {
            if (search_thread) |t| {
                stop_flag.store(true, .release);
                t.join();
                search_thread = null;
            }
            state = State.defaultPosition();
            history = search.PositionHistory.init();
            history.push(state.zobrist_hash);
            tbl.newSearch();
        } else if (std.mem.startsWith(u8, line, "setoption ")) {
            var it = std.mem.splitScalar(u8, line, ' ');
            _ = it.next(); // "setoption"
            _ = it.next(); // "name"
            const opt_name = it.next() orelse continue;
            _ = it.next(); // "value"
            const opt_val = it.next() orelse continue;
            if (std.mem.eql(u8, opt_name, "Threads")) {
                num_threads = std.fmt.parseInt(usize, opt_val, 10) catch num_threads;
            }
        } else if (std.mem.startsWith(u8, line, "position")) {
            if (search_thread != null) continue;

            var rest = line["position".len..];

            if (std.mem.startsWith(u8, rest, " startpos")) {
                state = State.defaultPosition();
                history = search.PositionHistory.init();
                history.push(state.zobrist_hash);
                rest = rest[" startpos".len..];
            } else if (std.mem.startsWith(u8, rest, " fen ")) {
                rest = rest[" fen ".len..];
                const moves_idx = std.mem.indexOf(u8, rest, " moves");
                const fen = if (moves_idx) |idx| rest[0..idx] else rest;
                state = State.fromFen(fen) catch continue;
                history = search.PositionHistory.init();
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
                },
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
