const std = @import("std");
const kore = @import("kore");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const search = @import("search.zig");
const State = @import("State.zig");

fn algebraicToSquare(s: []const u8) ?game.Squares.Square {
    if (s.len != 2) {
        return null;
    }

    const file = s[0];
    const rank = s[1];

    if (!(file >= 'a' and file <= 'h') or !(rank >= '1' and rank <= '8')) {
        return null;
    }

    const file_idx = file - 'a';
    const rank_idx = rank - '1';

    return @intCast(rank_idx * 8 + file_idx);
}

fn parseMove(input: []const u8) ?game.Move {
    const trimmed = std.mem.trimEnd(u8, input, &std.ascii.whitespace);
    if (trimmed.len < 4) return null;

    const start = algebraicToSquare(trimmed[0..2]);
    const end = algebraicToSquare(trimmed[2..4]);

    if (start == null or end == null) {
        return null;
    }

    return .{
        .start = start.?,
        .end = end.?,
    };
}

fn pieceName(p: game.Pieces.Piece) []const u8 {
    return switch (p) {
        game.Pieces.pawn => "pawn",
        game.Pieces.knight => "knight",
        game.Pieces.bishop => "bishop",
        game.Pieces.rook => "rook",
        game.Pieces.queen => "queen",
        game.Pieces.king => "king",
        else => unreachable,
    };
}

fn containsMove(haystack: *const [256]game.Move, needle: *const game.Move) bool {
    for (haystack) |straw| {
        if (straw.start == needle.start and straw.end == needle.end) {
            return true;
        }
    }
    return false;
}
const ns_per_s: f64 = @floatCast(std.time.ns_per_s);

const Args = struct {
    engine_color: ?[]const u8,
    depth: ?u8,
    num_threads: ?usize,
    adaptive_depth: ?bool,
    fen: ?[]const u8,
    nn_engine: ?[]const u8, // Path to NN checkpoint, e.g. "models/iter_0100.pt"
    nn_simulations: ?u32, // MCTS simulations for NN engine
};

// Neural network engine subprocess
const NNEngine = struct {
    process: std.process.Child,
    stdin: std.Io.File,
    stdout: std.Io.File,

    fn init(io: std.Io, checkpoint: []const u8, simulations: u32) !NNEngine {
        var sim_buf: [16]u8 = undefined;
        var child = try std.process.spawn(io, .{
            .argv = &.{
                "/home/paul/.local/bin/uv",
                "run",
                "python",
                "/home/paul/projects/chez/scripts/engine.py",
                "--checkpoint",
                checkpoint,
                "--simulations",
                std.fmt.bufPrint(&sim_buf, "{d}", .{simulations}) catch "400",
                    // "--debug",
            },
            .stdin = .pipe,
            .stdout = .pipe,
        });

        return .{
            .process = child,
            .stdin = child.stdin.?,
            .stdout = child.stdout.?,
        };
    }

    fn deinit(self: *NNEngine, io: std.Io) void {
        self.stdin.close(io);
        _ = self.process.kill(io);
        _ = self.process.wait(io) catch {};
    }

    fn getMove(self: *NNEngine, io: std.Io, state: *State, buf: []u8) !?game.Move {
        // Send FEN to engine (with newline to signal end of input)
        var fen_buf: [128]u8 = undefined;
        const fen_len = try state.toFen(&fen_buf);
        fen_buf[fen_len] = '\n';
        try self.stdin.writeStreamingAll(io, fen_buf[0 .. fen_len + 1]);

        // Read UCI move response
        var stdout = self.stdout.reader(io, buf);
        const move = try stdout.interface.takeDelimiterExclusive('\n');
        const move_str = std.mem.trimEnd(u8, move[0..move.len], &std.ascii.whitespace);

        if (move_str.len < 4) return null;
        return parseMove(move_str);
    }
};

fn writeHeader(stdout: *std.Io.Writer, state: *State, depth: ?u8, num_threads: usize, nn_mode: bool) !void {
    try stdout.writeAll("\x1B[2J\x1B[1;1H"); // ANSI clear screen
    try stdout.writeAll(" === Chez Paul ===\n");
    if (nn_mode) {
        try stdout.print(" Move {d} - Neural Network Engine\n\n", .{state.fullmove_clock});
    } else if (depth) |d| {
        try stdout.print(" Move {d} - Depth {d} - {d} Threads\n\n", .{ state.fullmove_clock, d, num_threads });
    } else {
        try stdout.print(" Move {d} - Adaptive Depth - {d} Threads\n\n", .{ state.fullmove_clock, num_threads });
    }
    try stdout.print("{f}", .{state});
    try stdout.flush();
}

pub fn main(init: std.process.Init.Minimal) !void {
    const arg_parser = try kore.args.declarative.Parser(Args);

    var args_iter = init.args.iterate();
    const parsed_args = try arg_parser.parse(&args_iter);

    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .environ = .empty });
    const io = threaded.io();
    var stdout_buffer: [512]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);
    const stdin: *std.Io.Reader = &stdin_reader.interface;

    try stdout.writeAll("=== Chez Paul ===\n");

    var state = blk: {
        if (parsed_args.fen) |fen| break :blk try State.fromFen(fen) else break :blk State.defaultPosition();
    };

    // Initialize position history for repetition detection
    var history = search.PositionHistory.init();
    history.push(state.zobrist_hash);

    var engine_color = blk: {
        if (parsed_args.engine_color) |c| {
            if (std.mem.eql(u8, c, "white")) break :blk game.Colors.white;
            if (std.mem.eql(u8, c, "black")) break :blk game.Colors.black;
            return error.InvalidColor;
        } else break :blk state.to_move;
    };
    var depth: ?u8 = null;

    if (parsed_args.depth) |d| {
        if (parsed_args.adaptive_depth.?) return error.DepthConflict;
        depth = d;
    }

    const num_threads: usize = parsed_args.num_threads orelse 4;

    // Initialize NN engine if requested
    const nn_mode = parsed_args.nn_engine != null;
    var nn_engine: ?NNEngine = null;
    defer if (nn_engine) |*eng| eng.deinit(io);

    if (parsed_args.nn_engine) |checkpoint| {
        try stdout.writeAll("Starting neural network engine...\n");
        try stdout.flush();
        nn_engine = try NNEngine.init(io, checkpoint, parsed_args.nn_simulations orelse 400);
    }

    while (true) {
        try writeHeader(stdout, &state, depth, num_threads, nn_mode);

        if (search.isGameOverWithHistory(&state, &history)) |res| {
            switch (res) {
                .checkmate => {
                    const winner = switch (res.checkmate) {
                        0 => "White",
                        1 => "Black",
                    };

                    try stdout.print("\n Checkmate! {s} wins!\n", .{winner});
                },
                .stalemate => try stdout.print("\n Stalemate! Draw.\n", .{}),
                .fiftyMoveRule => try stdout.print("\n Draw by 50-move rule.\n", .{}),
                .threefoldRepetition => try stdout.print("\n Draw by threefold repetition.\n", .{}),
            }
            try stdout.flush();
            break;
        }

        const current_color = state.to_move;
        const moves = movegen.legalMoves(&state, current_color);

        if (moves.len == 0) {
            try stdout.writeAll("No legal moves!\n");
            try stdout.flush();
            break;
        }

        if (current_color != engine_color) {
            // Human's turn

            while (true) {
                try stdout.writeAll(" Your turn: ");
                try stdout.flush();

                const move_raw = blk: {
                    while (stdin_reader.interface.takeDelimiterExclusive('\n')) |line| {
                        break :blk line;
                    } else |err| return err;
                };
                stdin.toss(1);

                if (move_raw.len == 0) {
                    engine_color = ~engine_color;
                    break;
                }

                const move = std.mem.trimEnd(u8, move_raw, &std.ascii.whitespace);

                if (std.mem.eql(u8, move, "quit") or std.mem.eql(u8, move, "q")) {
                    try stdout.writeAll(" Thanks for playing!\n");
                    try stdout.flush();
                    return;
                }

                if (parseMove(move)) |user_move| {
                    if (containsMove(&moves.moves, &user_move)) {
                        const piece = state.pieceAt(user_move.start).?;
                        _ = state.makeMove(user_move, ~engine_color, piece);
                        history.push(state.zobrist_hash);
                        try writeHeader(stdout, &state, depth, num_threads, nn_mode);
                        break;
                    } else {
                        try stdout.writeAll(" Illegal move! Try again.\n");
                        try stdout.flush();
                    }
                } else {
                    try stdout.writeAll(" Invalid format! Specify move like 'e2e4'\n");
                    try stdout.flush();
                }
            }
        }

        // Engine's turn
        try stdout.writeAll(" Thinking... ");
        try stdout.flush();

        const start = try std.time.Instant.now();

        var best_move: ?game.Move = null;
        var best_score: f64 = 0.0;

        if (nn_engine) |*eng| {
            // Use neural network engine
            var move_buf: [16]u8 = undefined;
            best_move = try eng.getMove(io, &state, &move_buf);
        } else {
            // Use traditional search
            if (try search.searchWithHistory(&state, depth, num_threads, &history)) |search_res| {
                best_move = search_res.move;
                best_score = search_res.score;
            }
        }

        if (best_move) |move| {
            const end = try std.time.Instant.now();
            const elapsed: f64 = @floatFromInt(end.since(start));
            const piece = state.pieceAt(move.start).?;

            var sq_start: [2]u8 = undefined;
            var sq_end: [2]u8 = undefined;
            try game.squareToAlgebraic(move.start, &sq_start);
            try game.squareToAlgebraic(move.end, &sq_end);

            _ = state.makeMove(move, state.to_move, piece);
            history.push(state.zobrist_hash);

            try stdout.writeByte('\n');
            try writeHeader(stdout, &state, depth, num_threads, nn_mode);
            if (nn_mode) {
                try stdout.print(" Engine moved {s} from {s} to {s} (thought for {d:.2} seconds)\n", .{ pieceName(piece), sq_start, sq_end, elapsed / ns_per_s });
            } else {
                try stdout.print(" Engine moved {s} from {s} to {s} (eval: {d:.2}, thought for {d:.2} seconds)\n", .{ pieceName(piece), sq_start, sq_end, best_score, elapsed / ns_per_s });
            }

            if (search.isGameOverWithHistory(&state, &history)) |res| {
                switch (res) {
                    .checkmate => {
                        const winner = switch (res.checkmate) {
                            0 => "White",
                            1 => "Black",
                        };

                        try stdout.print(" Checkmate! {s} wins!\n", .{winner});
                    },
                    .stalemate => try stdout.print(" Stalemate! Draw.\n", .{}),
                    .fiftyMoveRule => try stdout.print(" Draw by 50-move rule.\n", .{}),
                    .threefoldRepetition => try stdout.print(" Draw by threefold repetition.\n", .{}),
                }
                try stdout.flush();
                break;
            }

            try stdout.writeAll(" Press enter to continue... ");
            try stdout.flush();

            _ = blk: {
                while (stdin_reader.interface.takeDelimiterExclusive('\n')) |line| {
                    break :blk line;
                } else |err| return err;
            };
            stdin.toss(1);
        } else {
            try stdout.writeAll(" Engine has no legal moves!\n");
            try stdout.flush();
            break;
        }
    }
}
