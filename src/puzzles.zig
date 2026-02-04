const std = @import("std");
const chez = @import("chez.zig");

const puzzle_file = @embedFile("puzzles.json");

const Puzzle = struct {
    fen: []const u8,
    moves: [][]const u8,

    pub fn format(self: @This(), w: *std.Io.Writer) !void {
        try w.print("FEN: {s}\n", .{self.fen});
        try w.print("Engine needs to find moves ", .{self.num_moves});
        for (self.moves) |m| {
            try w.print("{s} ", .{m});
        }
        try w.flush();
    }
};

fn parseMove(mv: []const u8) ?chez.game.Move {
    std.debug.assert(mv.len > 3);
    std.debug.assert(mv.len < 6);

    const start = chez.game.algebraicToSquare(mv[0..2]);
    const end = chez.game.algebraicToSquare(mv[2..4]);
    const promotion_piece = blk: {
        if (mv.len == 5) {
            break :blk switch (mv[4]) {
                'n' => chez.game.Pieces.knight,
                'b' => chez.game.Pieces.bishop,
                'r' => chez.game.Pieces.rook,
                'q' => chez.game.Pieces.queen,
                else => unreachable,
            };
        } else {
            break :blk null;
        }
    };

    if (start == null or end == null) {
        return null;
    }

    return .{
        .start = start.?,
        .end = end.?,
        .promotion_piece = promotion_piece,
    };
}

const num_threads: usize = 4;

test "puzzles" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var correct: usize = 0;

    const puzzles: []Puzzle = try std.json.parseFromSliceLeaky([]Puzzle, alloc, puzzle_file, .{});
    pz: for (puzzles, 0..) |p, p_i| {
        var state = try chez.State.fromFen(p.fen);
        var history = chez.search.PositionHistory.init();
        history.push(state.zobrist_hash);

        for (p.moves, 0..) |mv, i| {
            const parsed_move = parseMove(mv) orelse {
                std.log.err("Unable to parse move {s} from puzzle.\n", .{mv});
                return error.InvalidMove;
            };
            const piece = state.mailbox[parsed_move.start].?;

            // Engine's move
            if (i % 2 == 0) {
                if (try chez.search.searchParallel(&state, 11, num_threads, &history)) |res| {
                    const eng_mv = res.move;
                    const is_move_correct = eng_mv.eql(parsed_move);
                    if (!is_move_correct) {
                        std.log.err(
                            "Engine made move {f} instead of {f} for move #{d} in puzzle #{d}",
                            .{ eng_mv, parsed_move, i + 1, p_i + 1 },
                        );
                        continue :pz;
                    }
                    _ = state.makeMove(eng_mv, state.to_move, piece);
                }
            } else {
                _ = state.makeMove(parsed_move, state.to_move, piece);
            }
            history.push(state.zobrist_hash);
        }
        correct += 1;
    }
    std.log.info("Correctly solved {d}/{d} puzzles.", .{ correct, puzzles.len });
    try std.testing.expectEqual(puzzles.len, correct);
}
