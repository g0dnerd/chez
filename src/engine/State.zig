const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const Bitboard = @import("Bitboard.zig");
const piece = @import("piece.zig");
const Piece = piece.Piece;
const square = @import("square.zig");
const Square = square.Square;
const castling = @import("castling.zig");
const CastlingRights = castling.CastlingRights;
const engine = @import("engine.zig");
const Colors = engine.Colors;
const Color = engine.Color;
const Move = engine.Move;
const isSquareAttackedBy = @import("movegen.zig").isSquareAttackedBy;

const State = @This();

// Information needed to unmake a move
pub const UndoInfo = struct {
    captured_piece: ?Piece,
    captured_square: Square, // Different from move.end for en passant
    castling_rights: CastlingRights,
    en_passant: ?Square,
    halfmove_clock: u16,
    in_check: ?Color,
    zobrist_hash: u64,
    was_promotion: bool,
    promotion_piece: ?Piece, // Actual promotion piece (not always queen)
    was_castling: bool,
    castling_side: Color, // 0 = kingside, 1 = queenside (only valid if was_castling)
};

pieces: [6]Bitboard,
colors: [2]Bitboard,
to_move: Color,
castling_rights: CastlingRights,
en_passant: ?Square = null,
in_check: ?Color = null,
halfmove_clock: u16 = 0,
fullmove_clock: u16 = 1,
zobrist_hash: u64 = 0,
all_pieces: Bitboard,
mailbox: [64]?Piece,

const pawn_start_rank: [2]Square = .{ 1, 6 };
const pawn_double_rank: [2]Square = .{ 3, 4 };
pub const pawn_promo_rank: [2]Square = .{ 7, 0 };
pub const pawn_ep_offset: [2]i8 = .{ -8, 8 };

pub fn defaultPosition() State {
    const pawns = Bitboard{ .bits = 0xff00000000ff00 };
    const knights = Bitboard{ .bits = 0x4200000000000042 };
    const bishops = Bitboard{ .bits = 0x2400000000000024 };
    const rooks = Bitboard{ .bits = 0x8100000000000081 };
    const queens = Bitboard{ .bits = 0x800000000000008 };
    const kings = Bitboard{ .bits = 0x1000000000000010 };

    const white_pieces = Bitboard{ .bits = 0xffff };
    const black_pieces = Bitboard{ .bits = 0xffff000000000000 };

    const mailbox = [64]?Piece{
        piece.rook,
        piece.knight,
        piece.bishop,
        piece.queen,
        piece.king,
        piece.bishop,
        piece.knight,
        piece.rook,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.pawn,
        piece.rook,
        piece.knight,
        piece.bishop,
        piece.queen,
        piece.king,
        piece.bishop,
        piece.knight,
        piece.rook,
    };

    var state = State{
        .colors = .{ white_pieces, black_pieces },
        .pieces = .{ pawns, knights, bishops, rooks, queens, kings },
        .to_move = Colors.white,
        .castling_rights = castling.all_legal,
        .all_pieces = white_pieces.bitOr(Bitboard, black_pieces),
        .mailbox = mailbox,
    };
    state.zobrist_hash = state.computeHash();
    return state;
}

pub fn fromFen(fen: []const u8) !State {
    const FenState = enum {
        placement,
        color,
        castling,
        enPassant,
        halfmove,
        fullmove,
    };
    var state: FenState = .placement;
    var current_square: Square = 56;

    var pawns = Bitboard.empty;
    var knights = Bitboard.empty;
    var bishops = Bitboard.empty;
    var rooks = Bitboard.empty;
    var queens = Bitboard.empty;
    var kings = Bitboard.empty;

    var white_pieces = Bitboard.empty;
    var black_pieces = Bitboard.empty;
    var to_move: ?Color = null;
    var en_passant: ?Square = null;
    var en_passant_file: ?Square = null;
    var castling_rights = castling.no_legal;
    var halfmove_clock: ?u16 = null;
    var fullmove_clock: ?u16 = null;

    var halfmove_start: usize = 0;
    var fullmove_start: usize = 0;

    loop: for (fen, 0..) |c, i| {
        switch (state) {
            .placement => {
                switch (c) {
                    'p' => {
                        pawns.bitOrAssign(Square, current_square);
                        black_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'P' => {
                        pawns.bitOrAssign(Square, current_square);
                        white_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'n' => {
                        knights.bitOrAssign(Square, current_square);
                        black_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'N' => {
                        knights.bitOrAssign(Square, current_square);
                        white_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'b' => {
                        bishops.bitOrAssign(Square, current_square);
                        black_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'B' => {
                        bishops.bitOrAssign(Square, current_square);
                        white_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'r' => {
                        rooks.bitOrAssign(Square, current_square);
                        black_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'R' => {
                        rooks.bitOrAssign(Square, current_square);
                        white_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'q' => {
                        queens.bitOrAssign(Square, current_square);
                        black_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'Q' => {
                        queens.bitOrAssign(Square, current_square);
                        white_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'k' => {
                        kings.bitOrAssign(Square, current_square);
                        black_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'K' => {
                        kings.bitOrAssign(Square, current_square);
                        white_pieces.bitOrAssign(Square, current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    '1', '2', '3', '4', '5', '6', '7', '8' => {
                        const current_rank = current_square / 8;
                        const amount: u6 = @intCast(try std.fmt.charToDigit(c, 10));
                        var new_square = @as(u8, current_square) + @as(u8, amount);
                        const new_file = new_square % 8;
                        const new_rank = new_square / 8;
                        if (new_file == 0) {
                            new_square -= 1;
                        }

                        if (new_rank < current_rank) return error.TooManySquares;

                        current_square = @intCast(new_square);
                        continue;
                    },
                    '/' => {
                        if (current_square % 8 == 7) {
                            current_square -= 15;
                            continue;
                        } else {
                            return error.UnexpectedSlash;
                        }
                    },
                    ' ' => {
                        if (current_square == 7) {
                            state = .color;
                        } else {
                            return error.UnexpectedWhitespace;
                        }
                    },
                    else => return error.InvalidCharacter,
                }
            },
            .color => {
                switch (c) {
                    'w' => to_move = Colors.white,
                    'b' => to_move = Colors.black,
                    ' ' => {
                        if (to_move == null) {
                            return error.UnexpectedWhitespace;
                        } else {
                            state = .castling;
                        }
                    },
                    else => return error.InvalidCharacter,
                }
            },
            .castling => {
                switch (c) {
                    '-' => {},
                    'k' => castling_rights |= castling.black_kingside,
                    'K' => castling_rights |= castling.white_kingside,
                    'q' => castling_rights |= castling.black_queenside,
                    'Q' => castling_rights |= castling.white_queenside,
                    ' ' => state = .enPassant,
                    else => return error.InvalidCharacter,
                }
            },
            .enPassant => {
                switch (c) {
                    '-' => {},
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h' => en_passant_file = @intCast(@as(u8, c) - 'a'),
                    '1', '2', '3', '4', '5', '6', '7', '8' => {
                        if (en_passant_file) |f| {
                            const rank: u6 = @intCast(try std.fmt.charToDigit(c, 10));
                            const ep: Square = (rank - 1) * 8 + f;
                            if (ep > 63) {
                                return error.InvalidEnpassantSquare;
                            }
                            en_passant = ep;
                        } else {
                            return error.InvalidEnpassantSquare;
                        }
                    },
                    ' ' => {
                        halfmove_start = i + 1;
                        state = .halfmove;
                    },
                    else => return error.InvalidCharacter,
                }
            },
            .halfmove => {
                switch (c) {
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' => continue,
                    ' ' => {
                        halfmove_clock = std.fmt.parseInt(u16, fen[halfmove_start..i], 10) catch return error.InvalidHalfmoveClock;
                        fullmove_start = i + 1;
                        state = .fullmove;
                    },
                    '-' => {
                        halfmove_clock = 0;
                        fullmove_clock = 1;
                        break :loop;
                    },
                    else => return error.InvalidCharacter,
                }
            },
            .fullmove => {
                switch (c) {
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' => continue,
                    else => return error.InvalidCharacter,
                }
            },
        }
    }

    if (fullmove_clock == null) {
        fullmove_clock = std.fmt.parseInt(u16, fen[fullmove_start..], 10) catch return error.InvalidFullmoveClock;
    }

    const all_pieces = [6]Bitboard{ pawns, knights, bishops, rooks, queens, kings };
    var mailbox: [64]?Piece = @splat(null);
    for (0..6) |piece_idx| {
        var bb = all_pieces[piece_idx];
        while (bb.next()) |s| {
            mailbox[s] = @intCast(piece_idx);
        }
    }

    var res = State{
        .pieces = all_pieces,
        .colors = .{ white_pieces, black_pieces },
        .to_move = to_move.?,
        .castling_rights = castling_rights,
        .en_passant = en_passant,
        .in_check = null,
        .halfmove_clock = halfmove_clock.?,
        .fullmove_clock = fullmove_clock.?,
        .all_pieces = white_pieces.bitOr(Bitboard, black_pieces),
        .mailbox = mailbox,
    };

    const pieces = res.colorBitboard(res.to_move);
    const king_mask = res.pieceBitboard(piece.king).bitAnd(Bitboard, pieces);
    const king_square = king_mask.trailingZeros();
    if (isSquareAttackedBy(&res, king_square, ~res.to_move)) {
        res.in_check = res.to_move;
    }

    // Compute initial hash
    res.zobrist_hash = res.computeHash();

    return res;
}

const Layer = enum {
    background,
    foreground,
};

const TerminalColor = enum {
    light,
    dark,
    white,
    black,
};

fn configureColor(writer: *std.Io.Writer, layer: Layer, c: TerminalColor) !void {
    switch (layer) {
        .background => switch (c) {
            .light => try writer.writeAll("\x1B[48;2;160;100;100m"),
            .dark => try writer.writeAll("\x1B[48;2;190;150;140m"),
            else => unreachable,
        },
        .foreground => switch (c) {
            .white => try writer.writeAll("\x1B[38;2;255;255;255m"),
            .black => try writer.writeAll("\x1B[38;2;0;0;0m"),
            .light => try writer.writeAll("\x1B[38;2;160;100;100m"),
            .dark => try writer.writeAll("\x1B[38;2;190;150;140m"),
        },
    }
}

fn resetColor(writer: *std.Io.Writer) !void {
    try writer.writeAll("\x1B[0m");
}

fn printSpacer(writer: *std.Io.Writer) !void {
    try writer.writeAll("▌");
}

pub fn format(self: State, writer: *std.Io.Writer) !void {
    try writer.writeAll("    a  b  c  d  e  f  g  h\n");
    var rank: Square = 7;
    while (true) {
        try writer.print(" {d} ", .{rank + 1});
        var file: u6 = 0;
        while (file < 8) {
            defer file += 1;
            const sq = rank * 8 + file;

            const isDarkSquare = if ((@as(u8, rank) + @as(u8, sq)) % 2 == 1)
                true
            else
                false;

            if (isDarkSquare) {
                try configureColor(writer, .background, .dark);
            } else {
                try configureColor(writer, .background, .light);
            }

            if (self.pieceAt(sq)) |p| {
                if (isDarkSquare) {
                    try configureColor(writer, .foreground, .dark);
                } else {
                    try configureColor(writer, .foreground, .light);
                }
                try printSpacer(writer);

                switch (self.colorAt(sq).?) {
                    Colors.white => try configureColor(writer, .foreground, .white),
                    Colors.black => try configureColor(writer, .foreground, .black),
                }
                try writer.print("{s} ", .{piece.piece_repr_symbol[p]});
            } else {
                try writer.writeAll("   ");
            }
        }
        try resetColor(writer);
        try writer.writeByte('\n');

        if (rank == 0) break;
        rank -= 1;
    }
    try writer.writeByte('\n');
    try writer.flush();
}

// Compute the full Zobrist hash from scratch. Used for initialization.
pub fn computeHash(self: *const State) u64 {
    const keys = State.getZobristKeys();
    var h: u64 = 0;

    // Hash all pieces with their colors
    for (0..2) |color_idx| {
        const color: Color = @intCast(color_idx);
        var color_pieces = self.colorBitboard(color);
        while (color_pieces.next()) |s| {
            const p = self.mailbox[s].?;
            h ^= keys.pieces[color][p][s];
        }
    }

    // Hash side to move (XOR if black to move)
    if (self.to_move == Colors.black) {
        h ^= keys.side_to_move;
    }

    // Hash castling rights
    h ^= keys.castling[self.castling_rights];

    // Hash en passant file (if set)
    if (self.en_passant) |ep| {
        const file = ep % 8;
        h ^= keys.en_passant[file];
    }

    return h;
}

pub inline fn colorBitboard(self: State, c: Color) Bitboard {
    return self.colors[c];
}

pub inline fn pieceBitboard(self: State, p: Piece) Bitboard {
    return self.pieces[p];
}

inline fn allPieces(self: State) Bitboard {
    return self.colors[0].bitOr(Bitboard, self.colors[1]);
}

pub fn pieceAt(self: State, s: Square) ?Piece {
    return self.mailbox[s];
}

pub inline fn colorAt(self: State, s: Square) ?Color {
    if (!self.all_pieces.contains(s)) return null;

    return @intFromBool(self.colors[1].contains(s));
}

pub inline fn isSquareEmpty(self: State, s: Square) bool {
    return !self.all_pieces.contains(s);
}

// Returns true if the given color has any non-pawn material (knights, bishops, rooks, queens)
pub fn hasNonPawnMaterial(self: *const State, c: Color) bool {
    const color_pieces = self.colorBitboard(c);
    const non_pawn_pieces = self.pieceBitboard(piece.knight)
        .bitOr(Bitboard, self.pieceBitboard(piece.bishop))
        .bitOr(Bitboard, self.pieceBitboard(piece.rook))
        .bitOr(Bitboard, self.pieceBitboard(piece.queen));
    return !color_pieces.bitAnd(Bitboard, non_pawn_pieces).isEmpty();
}

// Returns true if neither side has enough material to checkmate.
// Detected cases: K vs K, K+N vs K, K+B vs K, K+B vs K+B (same-color bishops).
pub fn hasInsufficientMaterial(self: *const State) bool {
    if (!self.pieceBitboard(piece.pawn).isEmpty()) return false;
    if (!self.pieceBitboard(piece.rook).isEmpty()) return false;
    if (!self.pieceBitboard(piece.queen).isEmpty()) return false;

    const knights = self.pieceBitboard(piece.knight);
    const bishops = self.pieceBitboard(piece.bishop);
    const knight_count = knights.popCount();
    const bishop_count = bishops.popCount();

    // K vs K
    if (knight_count == 0 and bishop_count == 0) return true;

    // K+N vs K or K+B vs K
    if (knight_count + bishop_count == 1) return true;

    // K+B vs K+B with same-color bishops
    if (knight_count == 0 and bishop_count == 2) {
        const white_bishops = bishops.bitAnd(Bitboard, self.colorBitboard(Colors.white));
        const black_bishops = bishops.bitAnd(Bitboard, self.colorBitboard(Colors.black));
        if (white_bishops.popCount() == 1 and black_bishops.popCount() == 1) {
            const light_squares = Bitboard{ .bits = 0x55AA55AA55AA55AA };
            const w_on_light = !white_bishops.bitAnd(Bitboard, light_squares).isEmpty();
            const b_on_light = !black_bishops.bitAnd(Bitboard, light_squares).isEmpty();
            return w_on_light == b_on_light;
        }
    }

    return false;
}

pub fn makeMove(self: *State, m: Move, c: Color, p: Piece) UndoInfo {
    return self.makeMoveInner(m, c, p, true);
}

// Skip in_check detection — for movegen legality testing where caller
// checks king safety separately. Avoids a redundant isSquareAttackedBy call.
pub fn makeMoveNoCheck(self: *State, m: Move, c: Color, p: Piece) UndoInfo {
    return self.makeMoveInner(m, c, p, false);
}

inline fn makeMoveInner(self: *State, m: Move, c: Color, p: Piece, comptime detect_check: bool) UndoInfo {
    const keys = State.getZobristKeys();
    const start = m.start;
    const end = m.end;

    // Store undo info before modifying state
    var undo = UndoInfo{
        .captured_piece = self.mailbox[end],
        .captured_square = end,
        .castling_rights = self.castling_rights,
        .en_passant = self.en_passant,
        .halfmove_clock = self.halfmove_clock,
        .in_check = self.in_check,
        .zobrist_hash = self.zobrist_hash,
        .was_promotion = false,
        .promotion_piece = null,
        .was_castling = false,
        .castling_side = 0,
    };

    var en_passant_target: ?Square = null;
    var new_en_passant: ?Square = null;

    // Store old castling rights for hash update
    const old_castling = self.castling_rights;

    // XOR out old en passant from hash
    if (self.en_passant) |ep| {
        self.*.zobrist_hash ^= keys.en_passant[ep % 8];
    }

    self.*.halfmove_clock += 1;

    switch (p) {
        piece.pawn => {
            self.*.halfmove_clock = 0;
            const start_rank = start / 8;
            const end_rank = end / 8;

            if (start_rank == pawn_start_rank[c] and end_rank == pawn_double_rank[c]) {
                new_en_passant = @intCast(@as(i8, end) + pawn_ep_offset[c]);
            } else if (self.en_passant == end) {
                en_passant_target = @intCast(@as(i8, end) + pawn_ep_offset[c]);
            }
        },
        piece.king => {
            if (square.absDiff(start, end) == 2) {
                // Determine kingside (0) or queenside (1) based on end file
                const side: Color = @intFromBool(end % 8 < 4); // c-file < e-file
                const data = castling.castle_data[c][side];

                if (self.castling_rights & data.rights_bit != 0) {
                    undo.was_castling = true;
                    undo.castling_side = side;

                    // Move rook
                    self.*.pieces[piece.rook].bitXorAssign(Square, data.rook_from);
                    self.*.colors[c].bitXorAssign(Square, data.rook_from);
                    self.*.mailbox[data.rook_from] = null;
                    self.*.pieces[piece.rook].bitOrAssign(Square, data.rook_to);
                    self.*.colors[c].bitOrAssign(Square, data.rook_to);
                    self.*.mailbox[data.rook_to] = piece.rook;

                    // Update hash
                    self.*.zobrist_hash ^= keys.pieces[c][piece.rook][data.rook_from];
                    self.*.zobrist_hash ^= keys.pieces[c][piece.rook][data.rook_to];
                }
            }
            self.*.castling_rights &= castling.king_castling_mask[c];
        },
        piece.rook => self.*.castling_rights &= castling.rook_castling_mask[start],
        else => {},
    }

    // Update en_passant field
    self.*.en_passant = new_en_passant;

    // XOR in new en passant to hash
    if (new_en_passant) |ep| {
        self.*.zobrist_hash ^= keys.en_passant[ep % 8];
    }

    // Handle capture
    if (undo.captured_piece) |x| {
        self.*.castling_rights &= castling.rook_castling_mask[end];
        self.*.halfmove_clock = 0;
        self.*.pieces[x].bitXorAssign(Square, end);
        self.*.colors[~c].bitXorAssign(Square, end);
        self.*.mailbox[end] = null;
        // XOR out captured piece from hash
        self.*.zobrist_hash ^= keys.pieces[~c][x][end];
    }

    // Handle en passant capture
    if (en_passant_target) |t| {
        undo.captured_piece = piece.pawn;
        undo.captured_square = t;
        self.*.halfmove_clock = 0;
        self.*.pieces[piece.pawn].bitXorAssign(Square, t);
        self.*.colors[~c].bitXorAssign(Square, t);
        self.*.mailbox[t] = null;
        // XOR out captured pawn from hash
        self.*.zobrist_hash ^= keys.pieces[~c][piece.pawn][t];
    }

    // XOR out piece from start square
    self.*.zobrist_hash ^= keys.pieces[c][p][start];

    // Actually move the piece
    self.*.pieces[p].bitXorAssign(Square, start);
    self.*.colors[c].bitXorAssign(Square, start);
    self.*.mailbox[start] = null;
    self.*.pieces[p].bitOrAssign(Square, end);
    self.*.colors[c].bitOrAssign(Square, end);
    self.*.mailbox[end] = p;

    if (m.is_promotion) {
        const promo_target = m.promotion_piece;
        undo.was_promotion = true;
        undo.promotion_piece = promo_target;
        self.*.pieces[piece.pawn].bitXorAssign(Square, end);
        self.*.pieces[promo_target].bitOrAssign(Square, end);

        // XOR in promoted piece at end square (not pawn)
        self.*.zobrist_hash ^= keys.pieces[c][promo_target][end];
        self.*.mailbox[end] = promo_target;
    } else {
        // XOR in piece at end square
        self.*.zobrist_hash ^= keys.pieces[c][p][end];
    }

    // Update castling rights hash if changed
    if (old_castling != self.castling_rights) {
        self.*.zobrist_hash ^= keys.castling[old_castling];
        self.*.zobrist_hash ^= keys.castling[self.castling_rights];
    }

    // Toggle side to move in hash
    self.*.zobrist_hash ^= keys.side_to_move;

    self.*.fullmove_clock += self.to_move;

    self.*.to_move = ~self.to_move;
    self.all_pieces = self.allPieces();

    if (detect_check) {
        // Update in_check for the new side to move
        const new_to_move = self.to_move;
        const king_bb = self.pieceBitboard(piece.king).bitAnd(Bitboard, self.colorBitboard(new_to_move));
        const king_square = king_bb.trailingZeros();
        if (isSquareAttackedBy(self, king_square, ~new_to_move)) {
            self.*.in_check = new_to_move;
        } else {
            self.*.in_check = null;
        }
    }

    return undo;
}

// Unmake a move, restoring the previous state
pub fn unmakeMove(self: *State, m: Move, c: Color, p: Piece, undo: UndoInfo) void {
    const start = m.start;
    const end = m.end;

    // Restore simple fields from undo info
    self.*.castling_rights = undo.castling_rights;
    self.*.en_passant = undo.en_passant;
    self.*.halfmove_clock = undo.halfmove_clock;
    self.*.in_check = undo.in_check;
    self.*.zobrist_hash = undo.zobrist_hash;
    self.*.fullmove_clock -= c; // Undo the increment (only increments when black moves)
    self.*.to_move = c;

    // Handle promotion: piece on end square is promoted piece, but we need to restore pawn
    const actual_piece = if (undo.was_promotion) undo.promotion_piece.? else p;

    // Move piece back from end to start
    self.*.pieces[actual_piece].bitXorAssign(Square, end);
    self.*.colors[c].bitXorAssign(Square, end);
    self.*.mailbox[end] = null;
    self.*.pieces[p].bitOrAssign(Square, start);
    self.*.colors[c].bitOrAssign(Square, start);
    self.*.mailbox[start] = p;

    // Handle castling: unmove the rook
    if (undo.was_castling) {
        const data = castling.castle_data[c][undo.castling_side];
        // Move rook back
        self.*.pieces[piece.rook].bitXorAssign(Square, data.rook_to);
        self.*.colors[c].bitXorAssign(Square, data.rook_to);
        self.*.mailbox[data.rook_to] = null;
        self.*.pieces[piece.rook].bitOrAssign(Square, data.rook_from);
        self.*.colors[c].bitOrAssign(Square, data.rook_from);
        self.*.mailbox[data.rook_from] = piece.rook;
    }

    // Restore captured piece
    if (undo.captured_piece) |captured| {
        const cap_sq = undo.captured_square;
        self.*.pieces[captured].bitOrAssign(Square, cap_sq);
        self.*.colors[~c].bitOrAssign(Square, cap_sq);
        self.*.mailbox[cap_sq] = captured;
    }

    // Update all_pieces
    self.all_pieces = self.allPieces();
}

pub fn toFen(self: *const State, buf: []u8) !u8 {
    var i: u8 = 0;

    var empty_squares: u8 = 0;

    var rank: Square = 7;
    while (true) {
        var file: u6 = 0;
        while (file < 8) {
            defer file += 1;
            const sq = rank * 8 + file;

            if (self.pieceAt(sq)) |p| {
                if (empty_squares > 0) {
                    buf[i] = '0' + empty_squares;
                    empty_squares = 0;
                    i += 1;
                }

                const color = self.colorAt(sq);

                buf[i] = switch (p) {
                    piece.pawn => if (color == Colors.white) 'P' else 'p',
                    piece.knight => if (color == Colors.white) 'N' else 'n',
                    piece.bishop => if (color == Colors.white) 'B' else 'b',
                    piece.rook => if (color == Colors.white) 'R' else 'r',
                    piece.queen => if (color == Colors.white) 'Q' else 'q',
                    piece.king => if (color == Colors.white) 'K' else 'k',
                    else => unreachable,
                };
                i += 1;
            } else {
                empty_squares += 1;
            }
        }

        if (empty_squares > 0) {
            buf[i] = '0' + empty_squares;
            i += 1;
            empty_squares = 0;
        }

        if (rank == 0) break;

        buf[i] = '/';
        i += 1;

        rank -= 1;
    }

    buf[i] = ' ';
    i += 1;

    // Color to move
    buf[i] = if (self.to_move == Colors.white) 'w' else 'b';
    i += 1;
    buf[i] = ' ';
    i += 1;

    if (self.castling_rights == castling.no_legal) {
        buf[i] = '-';
        i += 1;
    }
    if (self.castling_rights & castling.white_kingside != 0) {
        buf[i] = 'K';
        i += 1;
    }
    if (self.castling_rights & castling.white_queenside != 0) {
        buf[i] = 'Q';
        i += 1;
    }
    if (self.castling_rights & castling.black_kingside != 0) {
        buf[i] = 'k';
        i += 1;
    }
    if (self.castling_rights & castling.black_queenside != 0) {
        buf[i] = 'q';
        i += 1;
    }
    buf[i] = ' ';
    i += 1;

    if (self.en_passant) |ep| {
        var square_buf: [2]u8 = undefined;
        try square.toAlgebraic(ep, &square_buf);
        buf[i] = square_buf[0];
        buf[i + 1] = square_buf[1];
        buf[i + 2] = '-';
        i += 3;
    } else {
        buf[i] = '-';
        i += 1;
    }

    buf[i] = ' ';
    i += 1;

    const hm_slice = try std.fmt.bufPrint(buf[i..], "{d}", .{self.halfmove_clock});
    i += @intCast(hm_slice.len);
    buf[i] = ' ';
    i += 1;

    const fm_slice = try std.fmt.bufPrint(buf[i..], "{d}", .{self.fullmove_clock});
    i += @intCast(fm_slice.len);

    return i;
}

pub const ZobristKeys = struct {
    pieces: [2][6][64]u64, // [color][piece_type][square]
    side_to_move: u64, // XOR when black to move
    castling: [16]u64, // One key per castling rights combination
    en_passant: [8]u64, // One key per file (only file matters for en passant)
};

var init_mutex: std.Io.Mutex = .init;
var init_done = false;
var keys_storage: ZobristKeys = undefined;

fn initZobristKeys(io: std.Io) void {
    if (@atomicLoad(bool, &init_done, .monotonic)) return;
    init_mutex.lock(io) catch unreachable;

    const builtin = @import("builtin");
    var seed: u64 = undefined;
    if (builtin.target.os.tag == .freestanding) {
        // Fixed seed for WASM - deterministic behavior
        seed = 0x4d595f5345454421;
    } else if (builtin.target.os.tag == .linux) {
        _ = std.os.linux.getrandom(std.mem.asBytes(&seed), @sizeOf(u64), 0);
    } else {
        std.Io.random(io, std.mem.asBytes(&seed));
    }
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();

    // Piece-square keys for each color
    for (0..2) |color| {
        for (0..6) |piece_type| {
            for (0..64) |sq| {
                keys_storage.pieces[color][piece_type][sq] = random.int(u64);
            }
        }
    }

    // Side to move key
    keys_storage.side_to_move = random.int(u64);

    // Castling rights keys
    for (0..16) |rights| {
        keys_storage.castling[rights] = random.int(u64);
    }

    // En passant file keys
    for (0..8) |file| {
        keys_storage.en_passant[file] = random.int(u64);
    }

    @atomicStore(bool, &init_done, true, .release);
    init_mutex.unlock(io);
}

pub fn getZobristKeys() *const ZobristKeys {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    initZobristKeys(io);
    return &keys_storage;
}

test "piece at sanity" {
    const state = State.defaultPosition();
    const p_pawn = state.pieceAt(12);
    try expectEqual(p_pawn, 0);

    const p_king = state.pieceAt(60);
    try expectEqual(p_king, 5);

    const no_piece = state.pieceAt(20);
    try expectEqual(no_piece, null);
}

test "color at sanity" {
    const state = State.defaultPosition();
    const c_white = state.colorAt(12);
    try expectEqual(c_white, 0);

    const c_black = state.colorAt(60);
    try expectEqual(c_black, 1);

    const no_color = state.colorAt(20);
    try expectEqual(no_color, null);
}

test "all pieces sanity" {
    const state = State.defaultPosition();
    const all_pieces = state.allPieces();

    try expectEqual(all_pieces.bits, 0xffff00000000ffff);
}

test "test castling rights removal" {
    var state = State.defaultPosition();
    _ = state.makeMove(Move{ .start = square.e2, .end = square.e4 }, Colors.white, piece.pawn);
    _ = state.makeMove(Move{ .start = square.e7, .end = square.e5 }, Colors.black, piece.pawn);
    _ = state.makeMove(Move{ .start = square.g1, .end = square.f3 }, Colors.white, piece.knight);
    _ = state.makeMove(Move{ .start = square.d8, .end = square.e7 }, Colors.black, piece.queen);
    _ = state.makeMove(Move{ .start = square.f1, .end = square.e2 }, Colors.white, piece.bishop);
    _ = state.makeMove(Move{ .start = square.e7, .end = square.d8 }, Colors.black, piece.queen);
    _ = state.makeMove(Move{ .start = square.e1, .end = square.f1 }, Colors.white, piece.king);
    try expectEqual(castling.all_legal ^ castling.white_castling, state.castling_rights);

    state = State.defaultPosition();
    _ = state.makeMove(Move{ .start = square.a2, .end = square.a3 }, Colors.white, piece.pawn);
    _ = state.makeMove(Move{ .start = square.a7, .end = square.a6 }, Colors.black, piece.pawn);
    _ = state.makeMove(Move{ .start = square.a1, .end = square.a2 }, Colors.white, piece.rook);
    try expectEqual(castling.all_legal ^ castling.white_queenside, state.castling_rights);
}

test "test fen from default" {
    const starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const state = try State.fromFen(starting_fen);
    const default_state = State.defaultPosition();

    // // try expectEqual(default_state.pieces, state.pieces);
    // // try expectEqual(default_state.colors, state.colors);
    // // try expectEqual(default_state.to_move, state.to_move);
    // // try expectEqual(default_state.castling_rights, state.castling_rights);
    // try expectEqual(default_state.en_passant, state.en_passant);
    try expectEqual(default_state.in_check, state.in_check);
    try expectEqual(default_state.halfmove_clock, state.halfmove_clock);
    try expectEqual(default_state.fullmove_clock, state.fullmove_clock);
}

test "test fen from e4 c5 nf3" {
    const fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";
    const state = try State.fromFen(fen);

    const pawns = state.pieceBitboard(piece.pawn);
    const knights = state.pieceBitboard(piece.knight);
    const bishops = state.pieceBitboard(piece.bishop);
    const rooks = state.pieceBitboard(piece.rook);
    const queens = state.pieceBitboard(piece.queen);
    const kings = state.pieceBitboard(piece.king);

    try expectEqual(Bitboard{ .bits = 0xfb00041000ef00 }, pawns);
    try expectEqual(Bitboard{ .bits = 0x4200000000200002 }, knights);
    try expectEqual(Bitboard{ .bits = 0x2400000000000024 }, bishops);
    try expectEqual(Bitboard{ .bits = 0x8100000000000081 }, rooks);
    try expectEqual(Bitboard{ .bits = 0x800000000000008 }, queens);
    try expectEqual(Bitboard{ .bits = 0x1000000000000010 }, kings);

    const white_pieces = state.colorBitboard(Colors.white);
    const black_pieces = state.colorBitboard(Colors.black);

    try expectEqual(Bitboard{ .bits = 0x1020efbf }, white_pieces);
    try expectEqual(Bitboard{ .bits = 0xfffb000400000000 }, black_pieces);

    try expectEqual(Colors.black, state.to_move);
    try expectEqual(castling.all_legal, state.castling_rights);
    try expectEqual(null, state.en_passant);
    try expectEqual(1, state.halfmove_clock);
    try expectEqual(2, state.fullmove_clock);
}

test "incremental hash matches computed hash" {
    var state = State.defaultPosition();

    // Play some moves and verify hash consistency after each
    _ = state.makeMove(Move{ .start = square.e2, .end = square.e4 }, Colors.white, piece.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(Move{ .start = square.e7, .end = square.e5 }, Colors.black, piece.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(Move{ .start = square.g1, .end = square.f3 }, Colors.white, piece.knight);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(Move{ .start = square.b8, .end = square.c6 }, Colors.black, piece.knight);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    // Test capture
    _ = state.makeMove(Move{ .start = square.f1, .end = square.b5 }, Colors.white, piece.bishop);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(Move{ .start = square.a7, .end = square.a6 }, Colors.black, piece.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(Move{ .start = square.b5, .end = square.c6 }, Colors.white, piece.bishop);
    try expectEqual(state.computeHash(), state.zobrist_hash);
}

test "incremental hash with castling" {
    // Position where white can castle kingside
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    // Castle kingside
    _ = state.makeMove(Move{ .start = square.e1, .end = square.g1 }, Colors.white, piece.king);
    try expectEqual(state.computeHash(), state.zobrist_hash);
}

test "incremental hash with en passant" {
    var state = State.defaultPosition();

    // Set up en passant
    _ = state.makeMove(Move{ .start = square.e2, .end = square.e4 }, Colors.white, piece.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(Move{ .start = square.a7, .end = square.a6 }, Colors.black, piece.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(Move{ .start = square.e4, .end = square.e5 }, Colors.white, piece.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(Move{ .start = square.d7, .end = square.d5 }, Colors.black, piece.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    // En passant capture
    _ = state.makeMove(Move{ .start = square.e5, .end = square.d6 }, Colors.white, piece.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);
}

// Castling rights removal tests - king moves
test "king move removes both castling rights" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move white king e1 to f1
    _ = state.makeMove(Move{ .start = square.e1, .end = square.f1 }, Colors.white, piece.king);

    // Both white castling rights should be removed
    try expectEqual(castling.no_legal, state.castling_rights & castling.white_castling);
    // Black castling rights should remain
    try expectEqual(castling.black_castling, state.castling_rights & castling.black_castling);
}

test "black king move removes black castling" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move black king e8 to f8
    _ = state.makeMove(Move{ .start = square.e8, .end = square.f8 }, Colors.black, piece.king);

    // Both black castling rights should be removed
    try expectEqual(castling.no_legal, state.castling_rights & castling.black_castling);
    // White castling rights should remain
    try expectEqual(castling.white_castling, state.castling_rights & castling.white_castling);
}

// Castling rights removal tests - rook moves
test "kingside rook move removes kingside castling" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move h1 rook to g1
    _ = state.makeMove(Move{ .start = square.h1, .end = square.g1 }, Colors.white, piece.rook);

    // White kingside castling should be removed
    try expectEqual(castling.no_legal, state.castling_rights & castling.white_kingside);
    // White queenside should remain
    try expectEqual(castling.white_queenside, state.castling_rights & castling.white_queenside);
}

test "queenside rook move removes queenside castling" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move a1 rook to b1
    _ = state.makeMove(Move{ .start = square.a1, .end = square.b1 }, Colors.white, piece.rook);

    // White queenside castling should be removed
    try expectEqual(castling.no_legal, state.castling_rights & castling.white_queenside);
    // White kingside should remain
    try expectEqual(castling.white_kingside, state.castling_rights & castling.white_kingside);
}

test "black kingside rook move" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move h8 rook to g8
    _ = state.makeMove(Move{ .start = square.h8, .end = square.g8 }, Colors.black, piece.rook);

    // Black kingside castling should be removed
    try expectEqual(castling.no_legal, state.castling_rights & castling.black_kingside);
    // Black queenside should remain
    try expectEqual(castling.black_queenside, state.castling_rights & castling.black_queenside);
}

test "black queenside rook move" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move a8 rook to b8
    _ = state.makeMove(Move{ .start = square.a8, .end = square.b8 }, Colors.black, piece.rook);

    // Black queenside castling should be removed
    try expectEqual(castling.no_legal, state.castling_rights & castling.black_queenside);
    // Black kingside should remain
    try expectEqual(castling.black_kingside, state.castling_rights & castling.black_kingside);
}

// Rook capture removes opponent castling rights
test "capturing rook removes opponent castling" {
    const fen = "r3k2r/pppppppp/8/7B/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Bishop on h5 captures h8 rook
    _ = state.makeMove(Move{ .start = square.h5, .end = square.h8 }, Colors.white, piece.bishop);

    // Black kingside castling should be removed
    try expectEqual(castling.no_legal, state.castling_rights & castling.black_kingside);
    // Black queenside should remain
    try expectEqual(castling.black_queenside, state.castling_rights & castling.black_queenside);
}

test "capturing white rook removes white castling" {
    const fen = "r3k2r/pppppppp/8/8/8/7b/PPPPPPPP/R3K2R b KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Bishop on h3 captures h1 rook
    _ = state.makeMove(Move{ .start = square.h3, .end = square.h1 }, Colors.black, piece.bishop);

    // White kingside castling should be removed
    try expectEqual(castling.no_legal, state.castling_rights & castling.white_kingside);
    // White queenside should remain
    try expectEqual(castling.white_queenside, state.castling_rights & castling.white_queenside);
}

// Castling execution test
test "castling moves rook correctly" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Castle kingside (O-O)
    _ = state.makeMove(Move{ .start = square.e1, .end = square.g1 }, Colors.white, piece.king);

    // King should be on g1
    try expect(state.pieceAt(square.g1) == piece.king);
    try expect(state.colorAt(square.g1) == Colors.white);

    // Rook should be on f1
    try expect(state.pieceAt(square.f1) == piece.rook);
    try expect(state.colorAt(square.f1) == Colors.white);

    // h1 should be empty
    try expect(state.pieceAt(square.h1) == null);
}

// En passant square setting tests
test "en passant square set after double push" {
    var state = State.defaultPosition();

    // e2-e4 should set en passant square to e3
    _ = state.makeMove(Move{ .start = square.e2, .end = square.e4 }, Colors.white, piece.pawn);

    try expectEqual(square.e3, state.en_passant.?);
}

test "en passant square set for black" {
    var state = State.defaultPosition();

    // White move first
    _ = state.makeMove(Move{ .start = square.e2, .end = square.e3 }, Colors.white, piece.pawn);

    // d7-d5 should set en passant square to d6
    _ = state.makeMove(Move{ .start = square.d7, .end = square.d5 }, Colors.black, piece.pawn);

    try expectEqual(square.d6, state.en_passant.?);
}

// En passant square clearing tests
test "en passant cleared after non-pawn move" {
    var state = State.defaultPosition();

    // e2-e4 sets en passant
    _ = state.makeMove(Move{ .start = square.e2, .end = square.e4 }, Colors.white, piece.pawn);
    try expect(state.en_passant != null);

    // Knight move should clear en passant
    _ = state.makeMove(Move{ .start = square.g8, .end = square.f6 }, Colors.black, piece.knight);

    try expectEqual(@as(?square.Square, null), state.en_passant);
}

test "en passant cleared after single push" {
    var state = State.defaultPosition();

    // e2-e4 sets en passant
    _ = state.makeMove(Move{ .start = square.e2, .end = square.e4 }, Colors.white, piece.pawn);
    try expect(state.en_passant != null);

    // a7-a6 (single push) should clear en passant
    _ = state.makeMove(Move{ .start = square.a7, .end = square.a6 }, Colors.black, piece.pawn);

    try expectEqual(@as(?square.Square, null), state.en_passant);
}

// En passant capture tests
test "en passant capture removes captured pawn" {
    const fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
    var state = try State.fromFen(fen);

    // Verify d5 pawn exists before capture
    try expect(state.pieceAt(square.d5) == piece.pawn);
    try expect(state.colorAt(square.d5) == Colors.black);

    // e5xd6 en passant
    _ = state.makeMove(Move{ .start = square.e5, .end = square.d6 }, Colors.white, piece.pawn);

    // d5 pawn should be removed
    try expect(state.pieceAt(square.d5) == null);

    // White pawn should be on d6
    try expect(state.pieceAt(square.d6) == piece.pawn);
    try expect(state.colorAt(square.d6) == Colors.white);
}

test "black en passant capture" {
    const fen = "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3";
    var state = try State.fromFen(fen);

    // Verify d4 pawn exists before capture
    try expect(state.pieceAt(square.d4) == piece.pawn);
    try expect(state.colorAt(square.d4) == Colors.white);

    // e4xd3 en passant
    _ = state.makeMove(Move{ .start = square.e4, .end = square.d3 }, Colors.black, piece.pawn);

    // d4 pawn should be removed
    try expect(state.pieceAt(square.d4) == null);

    // Black pawn should be on d3
    try expect(state.pieceAt(square.d3) == piece.pawn);
    try expect(state.colorAt(square.d3) == Colors.black);
}

test "to fen sanity" {
    const default_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const default_state = State.defaultPosition();

    var fen_buf: [128]u8 = undefined;
    const fen_len = try default_state.toFen(&fen_buf);

    try expectEqual(default_fen.len, fen_len);
    try expect(std.mem.eql(u8, default_fen, fen_buf[0..fen_len]));
}

test "insufficient material: K vs K" {
    const state = try State.fromFen("8/8/4k3/8/8/3K4/8/8 w - - 0 1");
    try expect(state.hasInsufficientMaterial());
}

test "insufficient material: K+N vs K" {
    const state = try State.fromFen("8/8/4k3/8/8/3K4/8/1N6 w - - 0 1");
    try expect(state.hasInsufficientMaterial());
}

test "insufficient material: K+B vs K" {
    const state = try State.fromFen("8/8/4k3/8/8/3K4/8/5B2 w - - 0 1");
    try expect(state.hasInsufficientMaterial());
}

test "insufficient material: K+B vs K+B same color" {
    // Both bishops on light squares
    const state = try State.fromFen("8/8/4k3/5b2/8/3K4/8/5B2 w - - 0 1");
    try expect(state.hasInsufficientMaterial());
}

test "insufficient material: K+B vs K+B opposite color" {
    // White bishop on light square (f1), black bishop on dark square (e5)
    const state = try State.fromFen("8/8/4k3/4b3/8/3K4/8/5B2 w - - 0 1");
    try expect(!state.hasInsufficientMaterial());
}

test "insufficient material: K+N+N vs K is sufficient" {
    const state = try State.fromFen("8/8/4k3/8/8/3K4/8/NN6 w - - 0 1");
    try expect(!state.hasInsufficientMaterial());
}

test "insufficient material: starting position is sufficient" {
    const state = State.defaultPosition();
    try expect(!state.hasInsufficientMaterial());
}

test "insufficient material: K+R vs K is sufficient" {
    const state = try State.fromFen("8/8/4k3/8/8/3K4/8/R7 w - - 0 1");
    try expect(!state.hasInsufficientMaterial());
}

test "insufficient material: K+P vs K is sufficient" {
    const state = try State.fromFen("8/8/4k3/8/8/3K4/P7/8 w - - 0 1");
    try expect(!state.hasInsufficientMaterial());
}
