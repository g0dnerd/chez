const std = @import("std");
const chez = @import("chez");
const piece = chez.engine.piece;

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

fn parseMove(mv: []const u8) ?chez.engine.Move {
    std.debug.assert(mv.len > 3);
    std.debug.assert(mv.len < 6);

    const start = chez.engine.square.algebraicToSquare(mv[0..2]);
    const end = chez.engine.square.algebraicToSquare(mv[2..4]);
    var promotion_piece: piece.Piece = undefined;
    var is_promotion = false;

    if (mv.len == 5) {
        promotion_piece = switch (mv[4]) {
            'n' => piece.knight,
            'b' => piece.bishop,
            'r' => piece.rook,
            'q' => piece.queen,
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

const num_threads: usize = 4;

test "puzzles" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var correct: usize = 0;

    const puzzles: []Puzzle = try std.json.parseFromSliceLeaky([]Puzzle, alloc, puzzle_file, .{});
    pz: for (puzzles, 0..) |p, p_i| {
        var state = try chez.engine.State.fromFen(p.fen);
        var history = chez.engine.search.PositionHistory.init();
        history.push(state.zobrist_hash);

        var tbl: chez.engine.search.TranspositionTable = try .init(std.heap.page_allocator);

        for (p.moves, 0..) |mv, i| {
            const parsed_move = parseMove(mv) orelse {
                std.log.err("Unable to parse move {s} from puzzle.\n", .{mv});
                return error.InvalidMove;
            };
            const pc = state.mailbox[parsed_move.start].?;

            // Engine's move
            if (i % 2 == 0) {
                if (try chez.engine.search.searchParallel(&state, 11, num_threads, &history, &tbl)) |res| {
                    const eng_mv = res.move;
                    const is_move_correct = eng_mv.eql(parsed_move);
                    if (!is_move_correct) {
                        std.log.err(
                            "Engine made move {f} instead of {f} for move #{d} in puzzle #{d}",
                            .{ eng_mv, parsed_move, i + 1, p_i + 1 },
                        );
                        continue :pz;
                    }
                    _ = state.makeMove(eng_mv, state.to_move, pc);
                }
            } else {
                _ = state.makeMove(parsed_move, state.to_move, pc);
            }
            history.push(state.zobrist_hash);
        }
        correct += 1;
    }
    std.log.info("Correctly solved {d}/{d} puzzles.", .{ correct, puzzles.len });
    try std.testing.expectEqual(puzzles.len, correct);
}
