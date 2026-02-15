const builtin = @import("builtin");
const std = @import("std");
const kore = @import("kore");
const chez = @import("chez.zig");
const engine = chez.engine;

fn parseMove(input: []const u8) ?engine.Move {
    const trimmed = std.mem.trimEnd(u8, input, &std.ascii.whitespace);
    if (trimmed.len < 4) return null;

    const start = engine.square.algebraicToSquare(trimmed[0..2]);
    const end = engine.square.algebraicToSquare(trimmed[2..4]);

    var promotion_piece: engine.piece.Piece = undefined;
    var is_promotion = false;
    if (trimmed.len == 5) {
        promotion_piece = switch (trimmed[4]) {
            'n' => engine.piece.knight,
            'b' => engine.piece.bishop,
            'r' => engine.piece.rook,
            'q' => engine.piece.queen,
            else => unreachable,
        };
        is_promotion = true;
    }

    if (start == null or end == null) {
        return null;
    }

    return .{
        .start = start.?,
        .end = end.?,
        .promotion_piece = promotion_piece,
        .is_promotion = is_promotion,
    };
}

fn printLegalMoves(m: engine.movegen.MoveList, w: *std.Io.Writer) !void {
    for (0..m.len) |i| {
        const mv = m.moves[i];
        try w.print("{f}, ", .{mv});
    }
    try w.writeByte('\n');
    try w.flush();
}

fn containsMove(haystack: *const [256]engine.Move, needle: *const engine.Move) bool {
    for (haystack) |straw| {
        if (straw.start == needle.start and straw.end == needle.end and straw.promotion_piece == needle.promotion_piece) {
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

    fn getMove(self: *NNEngine, io: std.Io, state: *engine.State, buf: []u8) !?engine.Move {
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

fn writeHeader(stdout: *std.Io.Writer, state: *engine.State, depth: ?u8, num_threads: usize, nn_mode: bool) !void {
    try stdout.writeAll("\x1B[2J\x1B[1;1H"); // ANSI clear screen
    try stdout.writeAll(" === Chez Paul ===\n");
    if (nn_mode) {
        try stdout.print(" Move {d} - Neural Network Engine\n\n", .{state.fullmove_clock});
    } else if (depth) |d| {
        try stdout.print(" Move {d} - Depth {d} - {d} Threads\n\n", .{ state.fullmove_clock, d, num_threads });
    }
    try stdout.print("{f}", .{state});

    // const eval = engine.evaluation.evaluateTrace(state);
    // try eval.dump(stdout);
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
        if (parsed_args.fen) |fen| break :blk try engine.State.fromFen(fen) else break :blk engine.State.defaultPosition();
    };

    // Initialize position history for repetition detection
    var history = engine.search.PositionHistory.init();
    history.push(state.zobrist_hash);

    var tbl = try engine.search.TranspositionTable.init(std.heap.page_allocator);
    defer tbl.deinit();

    var engine_color = blk: {
        if (parsed_args.engine_color) |c| {
            if (std.mem.eql(u8, c, "white")) break :blk engine.Colors.white;
            if (std.mem.eql(u8, c, "black")) break :blk engine.Colors.black;
            return error.InvalidColor;
        } else break :blk state.to_move;
    };

    const depth: u8 = parsed_args.depth orelse 12;
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

    var undo_info: [2]engine.State.UndoInfo = undefined;
    var undo_moves: [2]engine.Move = undefined;
    var undo_pieces: [2]engine.piece.Piece = undefined;

    outer: while (true) {
        try writeHeader(stdout, &state, depth, num_threads, nn_mode);

        if (engine.search.isGameOverWithHistory(&state, &history)) |res| {
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
                .insufficientMaterial => try stdout.print("\n Draw by insufficient material.\n", .{}),
            }
            try stdout.flush();
            break;
        }

        const current_color = state.to_move;
        var last_move: ?engine.Move = null;
        const moves = engine.movegen.legalMoves(&state, current_color);

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
                } else if (move_raw.len == 1 and move_raw[0] == 'u') {
                    // Undo move
                    state.unmakeMove(undo_moves[0], state.to_move, undo_pieces[0], undo_info[0]);
                    state.unmakeMove(undo_moves[1], ~state.to_move, undo_pieces[1], undo_info[1]);
                    state.to_move = ~state.to_move;
                    undo_moves = undefined;
                    undo_pieces = undefined;
                    undo_info = undefined;
                    continue :outer;
                }

                const move = std.mem.trimEnd(u8, move_raw, &std.ascii.whitespace);

                if (std.mem.eql(u8, move, "quit") or std.mem.eql(u8, move, "q")) {
                    try stdout.writeAll(" Thanks for playing!\n");
                    try stdout.flush();
                    return;
                }

                if (parseMove(move)) |*user_move| {
                    if (containsMove(&moves.moves, user_move)) {
                        last_move = user_move.*;
                        const piece = state.pieceAt(user_move.start).?;
                        const ui = state.makeMove(user_move.*, ~engine_color, piece);
                        undo_info[0] = ui;
                        undo_pieces[0] = piece;
                        undo_moves[0] = user_move.*;
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

        var best_move: ?engine.Move = null;
        var best_score: f64 = 0.0;

        if (nn_engine) |*eng| {
            // Use neural network engine
            var move_buf: [16]u8 = undefined;
            best_move = try eng.getMove(io, &state, &move_buf);
        } else {
            if (try engine.search.searchWithHistory(&state, depth, num_threads, &history, &tbl)) |search_res| {
                // Use traditional search
                best_move = search_res.move;
                best_score = search_res.score;
            }
        }

        if (best_move) |move| {
            const piece = state.mailbox[move.start].?;

            var sq_start: [2]u8 = undefined;
            var sq_end: [2]u8 = undefined;
            try engine.square.toAlgebraic(move.start, &sq_start);
            try engine.square.toAlgebraic(move.end, &sq_end);

            const ui = state.makeMove(move, state.to_move, piece);
            undo_info[1] = ui;
            undo_pieces[1] = piece;
            undo_moves[1] = move;
            history.push(state.zobrist_hash);

            try stdout.writeByte('\n');
            try writeHeader(stdout, &state, depth, num_threads, nn_mode);

            if (engine.search.isGameOverWithHistory(&state, &history)) |res| {
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
                    .insufficientMaterial => try stdout.print(" Draw by insufficient material.\n", .{}),
                }
                try stdout.flush();
                break;
            }
        } else {
            try stdout.writeAll(" Engine has no legal moves!\n");
            try stdout.flush();
            break;
        }
    }
}
