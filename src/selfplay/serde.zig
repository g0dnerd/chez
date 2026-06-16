const std = @import("std");
const chez = @import("chez");
const kore = @import("kore");
const engine = chez.engine;

const huffman_pawn: u1 = 0b0;
const huffman_knight: u2 = 0b01;
const huffman_bishop: u3 = 0b011;
const huffman_rook: u4 = 0b0111;
const huffman_queen: u4 = 0b1111;

pub fn encodePosition(state: engine.State, writer: *std.Io.Writer) !void {
    var bit_writer = kore.io.BitWriter.init(writer);
    const header: PositionHeader = .fromState(state);

    try header.write(&bit_writer);

    for (state.mailbox, 0..) |p, sq| {
        const square: engine.square.Square = @intCast(sq);

        if (p) |piece| {
            if (square == header.white_king or square == header.black_king) continue;

            try bit_writer.write(1, 1);

            switch (piece) {
                engine.piece.pawn => try bit_writer.write(huffman_pawn, 1),
                engine.piece.knight => try bit_writer.write(huffman_knight, 2),
                engine.piece.bishop => try bit_writer.write(huffman_bishop, 3),
                engine.piece.rook => try bit_writer.write(huffman_rook, 4),
                engine.piece.queen => try bit_writer.write(huffman_queen, 4),
                else => unreachable,
            }

            const color = state.colorAt(square).?;
            try bit_writer.write(color, 1);
        } else {
            try bit_writer.write(0, 1);
        }
    }

    try bit_writer.finish();
}

pub fn encodePositionToBuffer(state: engine.State, buf: []u8) !void {
    var w: std.Io.Writer = .fixed(buf);
    try encodePosition(state, &w);
}

pub fn decodePosition(buf: []u8) !engine.State {
    std.debug.assert(buf.len == 32);

    var r: std.Io.Reader = .fixed(buf);
    var bit_reader = kore.io.BitReader.init(&r);
    const header = try PositionHeader.read(&bit_reader);

    var pawns = engine.Bitboard.empty;
    var knights = engine.Bitboard.empty;
    var bishops = engine.Bitboard.empty;
    var rooks = engine.Bitboard.empty;
    var queens = engine.Bitboard.empty;
    var kings = engine.Bitboard.empty;

    var white_pieces = engine.Bitboard.empty;
    var black_pieces = engine.Bitboard.empty;

    kings.bitOrAssign(engine.square.Square, header.white_king);
    kings.bitOrAssign(engine.square.Square, header.black_king);
    white_pieces.bitOrAssign(engine.square.Square, header.white_king);
    black_pieces.bitOrAssign(engine.square.Square, header.black_king);

    for (0..64) |sq| {
        const square: engine.square.Square = @intCast(sq);

        if (square == header.white_king or square == header.black_king) continue;

        const presence_bit = try bit_reader.read(1);
        if (presence_bit == 0) continue;

        // Read Huffman code bit-by-bit: 0=pawn, 10=knight, 110=bishop, 1110=rook, 1111=queen
        // Read successive 1s until a 0 terminates, or 4th bit is queen.
        var piece_idx: usize = 0;
        for (0..4) |p_idx| {
            const bit = try bit_reader.read(1);
            if (bit == 0 or p_idx == 3) {
                piece_idx = if (bit == 0) p_idx else p_idx + 1;
                break;
            }
        }

        const color_bit = try bit_reader.read(1);
        switch (color_bit) {
            0 => white_pieces.bitOrAssign(engine.square.Square, square),
            1 => black_pieces.bitOrAssign(engine.square.Square, square),
            else => unreachable,
        }

        switch (piece_idx) {
            0 => pawns.bitOrAssign(engine.square.Square, square),
            1 => knights.bitOrAssign(engine.square.Square, square),
            2 => bishops.bitOrAssign(engine.square.Square, square),
            3 => rooks.bitOrAssign(engine.square.Square, square),
            4 => queens.bitOrAssign(engine.square.Square, square),
            else => unreachable,
        }
    }

    const all_pieces = [6]engine.Bitboard{ pawns, knights, bishops, rooks, queens, kings };
    var mailbox: [64]?engine.piece.Piece = @splat(null);
    for (0..6) |piece_idx| {
        var bb = all_pieces[piece_idx];
        while (bb.next()) |s| {
            mailbox[s] = @intCast(piece_idx);
        }
    }

    const en_passant: ?engine.square.Square = if (header.en_passant == 0)
        null
    else
        engine.State.pawn_ep_rank[header.to_move] * 8 + header.en_passant - 1;

    var res = engine.State{
        .pieces = all_pieces,
        .colors = .{ white_pieces, black_pieces },
        .to_move = header.to_move,
        .castling_rights = header.castling,
        .en_passant = en_passant,
        .in_check = null,
        .halfmove_clock = @intCast(header.halfmove_clock),
        .fullmove_clock = 1,
        .all_pieces = white_pieces.bitOr(engine.Bitboard, black_pieces),
        .mailbox = mailbox,
    };

    const pieces = res.colorBitboard(res.to_move);
    const king_mask = res.pieceBitboard(engine.piece.king).bitAnd(engine.Bitboard, pieces);
    const king_square = king_mask.trailingZeros();

    if (engine.movegen.isSquareAttackedBy(&res, king_square, ~res.to_move)) {
        res.in_check = res.to_move;
    }
    res.zobrist_hash = res.computeHash();

    return res;
}

const PositionHeader = packed struct {
    to_move: u1, // 0=white, 1=black
    white_king: engine.square.Square,
    black_king: engine.square.Square,
    castling: engine.castling.CastlingRights,
    en_passant: u4, // 0=none, 1-8=file a-h
    halfmove_clock: u7,

    fn write(self: PositionHeader, w: *kore.io.BitWriter) !void {
        try w.write(@intCast(self.to_move), 1);
        try w.write(@intCast(self.white_king), 6);
        try w.write(@intCast(self.black_king), 6);
        try w.write(@intCast(self.castling), 4);
        try w.write(@intCast(self.en_passant), 4);
        try w.write(@intCast(self.halfmove_clock), 7);
    }

    fn read(r: *kore.io.BitReader) !PositionHeader {
        var ret: PositionHeader = undefined;

        ret.to_move = @intCast(try r.read(1));
        ret.white_king = @intCast(try r.read(6));
        ret.black_king = @intCast(try r.read(6));
        ret.castling = @intCast(try r.read(4));
        ret.en_passant = @intCast(try r.read(4));
        ret.halfmove_clock = @intCast(try r.read(7));

        return ret;
    }

    pub fn fromState(state: engine.State) PositionHeader {
        const kings = state.pieceBitboard(engine.piece.king);
        const white_king = kings
            .bitAnd(engine.Bitboard, state.colorBitboard(engine.Colors.white))
            .trailingZeros();
        const black_king = kings
            .bitAnd(engine.Bitboard, state.colorBitboard(engine.Colors.black))
            .trailingZeros();

        const en_passant: u4 = if (state.en_passant) |ep|
            @as(u4, @intCast(ep % 8)) + 1
        else
            0;

        return .{
            .to_move = state.to_move,
            .white_king = white_king,
            .black_king = black_king,
            .castling = state.castling_rights,
            .en_passant = en_passant,
            .halfmove_clock = @intCast(state.halfmove_clock),
        };
    }
};

const expect = std.testing.expect;

test "serde roundtrip from default" {
    const state = engine.State.defaultPosition();
    var buf = [_]u8{0} ** 32;
    try encodePositionToBuffer(state, &buf);
    const decoded_state = try decodePosition(&buf);

    try expect(state.eql(decoded_state));
}

test "serde roundtrip from FENs" {
    const fens = [_][]const u8{
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    };

    for (fens) |fen| {
        const state = try engine.State.fromFen(fen);
        var buf = [_]u8{0} ** 32;
        try encodePositionToBuffer(state, &buf);
        const decoded_state = try decodePosition(&buf);

        try expect(state.eql(decoded_state));
    }
}
