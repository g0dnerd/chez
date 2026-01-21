const std = @import("std");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const search = @import("search.zig");
const State = @import("State.zig");

fn squareToAlgebraic(square: game.Squares.Square, buf: []u8) !void {
    const file: u8 = 'a' + @as(u8, square) % 8;
    const rank: u8 = '1' + @as(u8, square) / 8;
    _ = try std.fmt.bufPrint(buf, "{c}{c}", .{ file, rank });
}

fn displayLegalMoves(moves: []game.Move, w: *std.Io.Writer) !void {
    try w.writeAll("Legal moves: ");
    for (moves, 0..) |m, i| {
        var sq_start: [2]u8 = undefined;
        var sq_end: [2]u8 = undefined;
        try squareToAlgebraic(m.start, &sq_start);
        try squareToAlgebraic(m.end, &sq_end);

        try w.print("{s}{s}", .{ sq_start, sq_end });
        try w.flush();
        if (i < moves.len - 1) {
            try w.writeAll(", ");
            try w.flush();
        }
    }
    try w.writeByte('\n');
    try w.flush();
}

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

fn containsMove(haystack: []game.Move, needle: *const game.Move) bool {
    for (haystack) |straw| {
        if (straw.start == needle.start and straw.end == needle.end) {
            return true;
        }
    }
    return false;
}
const ns_per_s: f64 = @floatCast(std.time.ns_per_s);

pub fn main() !void {
    var stdout_buffer: [512]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&stdin_buf);
    const stdin: *std.Io.Reader = &stdin_reader.interface;

    try stdout.writeAll("=== Chez Paul ===\n");
    try stdout.writeAll("Enter FEN or press enter to start from new position\n");
    try stdout.flush();

    const fen_raw = blk: {
        while (stdin.takeDelimiterExclusive('\n')) |line| {
            break :blk line;
        } else |err| {
            return err;
        }
    };
    const fen = std.mem.trimEnd(u8, fen_raw, &std.ascii.whitespace);

    var state = if (fen.len == 0)
        State.defaultPosition()
    else
        try State.fromFen(fen);

    var num_moves: u16 = 1;

    try stdout.writeAll("Enter search engine depth (max 20):\n");
    try stdout.flush();

    const depth_input_raw = try stdin_reader.interface.takeDelimiterExclusive('\n');
    const depth_input = std.mem.trimEnd(u8, depth_input_raw, &std.ascii.whitespace);
    const depth = try std.fmt.parseInt(u8, depth_input, 10);
    std.debug.assert(depth <= 20);

    const engine_color = if (fen.len == 0)
        game.Colors.black
    else
        state.to_move;

    while (true) {
        try stdout.writeAll("\x1B[2J\x1B[1;1H"); // ANSI clear screen
        try stdout.print("Move {d}\n", .{num_moves});
        try stdout.print("{f}\n", .{state});
        try stdout.flush();

        if (try search.isGameOver(&state)) |res| {
            switch (res) {
                .checkmate => {
                    const winner = switch (res.checkmate) {
                        0 => "White",
                        1 => "Black",
                    };

                    try stdout.print("\nCheckmate! {s} wins!\n", .{winner});
                },
                .stalemate => try stdout.print("\nStalemate! Draw.\n", .{}),
                .fiftyMoveRule => try stdout.print("\nDraw by 50-move rule.\n", .{}),
            }
            break;
        }

        const current_color = state.to_move;
        var moves = try movegen.legalMoves(std.heap.page_allocator, &state, current_color);
        defer moves.deinit(std.heap.page_allocator);

        if (moves.items.len == 0) {
            try stdout.writeAll("No legal moves!\n");
            try stdout.flush();
            break;
        }

        if (current_color != engine_color) {
            // Human's turn
            try stdout.writeAll("Your turn\n");

            try displayLegalMoves(moves.items, stdout);

            while (true) {
                try stdout.writeAll("\nEnter move (e.g., e2e4) or 'quit/q': ");
                try stdout.flush();

                const move_raw = blk: {
                    while (stdin_reader.interface.takeDelimiterExclusive('\n')) |line| {
                        break :blk line;
                    } else |err| return err;
                };
                const move = std.mem.trimEnd(u8, move_raw, &std.ascii.whitespace);

                if (std.mem.eql(u8, move, "quit") or std.mem.eql(u8, move, "q")) {
                    try stdout.writeAll("Thanks for playing!\n");
                    try stdout.flush();
                    return;
                }

                if (parseMove(move)) |user_move| {
                    if (containsMove(moves.items, &user_move)) {
                        const piece = state.pieceAt(user_move.start).?;
                        state.makeMove(user_move, ~engine_color, piece);
                        num_moves += 1;

                        var sq_start: [2]u8 = undefined;
                        var sq_end: [2]u8 = undefined;
                        try squareToAlgebraic(user_move.start, &sq_start);
                        try squareToAlgebraic(user_move.end, &sq_end);
                        try stdout.print("\nYou moved {s} from {s} to {s}\n", .{
                            pieceName(piece),
                            sq_start,
                            sq_end,
                        });
                        try stdout.flush();
                        break;
                    } else {
                        try stdout.writeAll("Illegal move! Try again.\n");
                        try stdout.flush();
                    }
                } else {
                    try stdout.writeAll("Invalid format! Specify move like 'e2e4'\n");
                    try stdout.flush();
                }
            }
        } else {
            // Engine's turn
            try stdout.print("Engine thinking (depth {d})...\n", .{depth});
            try stdout.flush();

            const start = try std.time.Instant.now();
            if (try search.search(&state, depth)) |search_res| {
                const end = try std.time.Instant.now();
                const elapsed: f64 = @floatFromInt(end.since(start));
                const best_move = search_res.move;
                const best_score = search_res.score;
                const piece = state.pieceAt(best_move.start).?;

                var sq_start: [2]u8 = undefined;
                var sq_end: [2]u8 = undefined;
                try squareToAlgebraic(best_move.start, &sq_start);
                try squareToAlgebraic(best_move.end, &sq_end);

                state.makeMove(best_move, engine_color, piece);
                num_moves += 1;

                try stdout.writeAll("\x1B[2J\x1B[1;1H"); // ANSI clear screen
                try stdout.print("Move {d}\n", .{num_moves});
                try stdout.print("{f}\n", .{state});
                try stdout.print("Engine moved {s} from {s} to {s} (score: {d:.2}, found in {d:.2} seconds)\n", .{ pieceName(piece), sq_start, sq_end, best_score, elapsed / ns_per_s });
                try stdout.writeAll("Press enter to continue...\n");
                try stdout.flush();

                _ = blk: {
                    while (stdin_reader.interface.takeDelimiterExclusive('\n')) |line| {
                        break :blk line;
                    } else |err| return err;
                };
            } else {
                try stdout.writeAll("Engine has no legal moves!\n");
                try stdout.flush();
                break;
            }
        }
    }
}
