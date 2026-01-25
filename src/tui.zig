const std = @import("std");
const kore = @import("kore");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const search = @import("search.zig");
const State = @import("State.zig");

// fn displayLegalMoves(moves: *const movegen.MoveList, w: *std.Io.Writer) !void {
//     try w.writeAll("Legal moves: ");
//     for (0..moves.len) |i| {
//         const m = moves.moves[i];
//         var sq_start: [2]u8 = undefined;
//         var sq_end: [2]u8 = undefined;
//         try game.squareToAlgebraic(m.start, &sq_start);
//         try game.squareToAlgebraic(m.end, &sq_end);
//
//         try w.print("{s}{s}", .{ sq_start, sq_end });
//         try w.flush();
//         if (i < moves.len - 1) {
//             try w.writeAll(", ");
//             try w.flush();
//         }
//     }
//     try w.writeByte('\n');
//     try w.flush();
// }

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
    fen: ?[]const u8,
};

fn writeHeader(stdout: *std.Io.Writer, state: *State, depth: u8, num_threads: usize) !void {
    try stdout.writeAll("\x1B[2J\x1B[1;1H"); // ANSI clear screen
    try stdout.writeAll(" === Chez Paul ===\n");
    try stdout.print(" Move {d} - Depth {d} - {d} Threads\n\n", .{ state.fullmove_clock, depth, num_threads });
    try stdout.print("{f}", .{state});
    try stdout.flush();
}

pub fn main(init: std.process.Init.Minimal) !void {
    const arg_parser = try kore.args.declarative.Parser(Args);

    var args_iter = init.args.iterate();
    const parsed_args = try arg_parser.parse(&args_iter);

    var threaded: std.Io.Threaded = .init_single_threaded;
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
    const depth: u8 = parsed_args.depth orelse 6;
    const num_threads: usize = parsed_args.num_threads orelse 4;

    while (true) {
        try writeHeader(stdout, &state, depth, num_threads);

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
                        try writeHeader(stdout, &state, depth, num_threads);
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
        if (try search.searchWithHistory(&state, depth, num_threads, &history)) |search_res| {
            const end = try std.time.Instant.now();
            const elapsed: f64 = @floatFromInt(end.since(start));
            const best_move = search_res.move;
            const best_score = search_res.score;
            const piece = state.pieceAt(best_move.start).?;

            var sq_start: [2]u8 = undefined;
            var sq_end: [2]u8 = undefined;
            try game.squareToAlgebraic(best_move.start, &sq_start);
            try game.squareToAlgebraic(best_move.end, &sq_end);

            _ = state.makeMove(best_move, state.to_move, piece);
            history.push(state.zobrist_hash);

            try stdout.writeByte('\n');
            try writeHeader(stdout, &state, depth, num_threads);
            try stdout.print(" Engine moved {s} from {s} to {s} (eval: {d:.2}, thought for {d:.2} seconds)\n", .{ pieceName(piece), sq_start, sq_end, best_score, elapsed / ns_per_s });
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
