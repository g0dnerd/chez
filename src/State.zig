const std = @import("std");
const ffi = @import("ffi.zig");
const Bitboard = @import("Bitboard.zig");
const game = @import("game.zig");
const movegen = @import("movegen.zig");
const Castling = game.Castling;
const Colors = game.Colors;
const Color = Colors.Color;
const Pieces = game.Pieces;
const Piece = Pieces.Piece;
const Squares = game.Squares;
const Square = Squares.Square;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

pub const State = @This();

// Information needed to unmake a move
pub const UndoInfo = struct {
    captured_piece: ?Piece,
    captured_square: Square, // Different from move.end for en passant
    castling_rights: Castling.CastlingRights,
    en_passant: ?Square,
    halfmove_clock: u16,
    in_check: ?Color,
    zobrist_hash: u64,
    was_promotion: bool,
    was_castling: bool,
    castling_side: Color, // 0 = kingside, 1 = queenside (only valid if was_castling)
};

pieces: [6]Bitboard,
colors: [2]Bitboard,
to_move: Color,
castling_rights: Castling.CastlingRights,
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
const pawn_ep_offset: [2]i8 = .{ -8, 8 };

// Initializes a zig State struct from its C counterpart
pub fn initCState(c: *const ffi.CState) State {
    const en_passant: ?Square = if (c.en_passant == -1)
        null
    else
        @as(u6, @intCast(c.en_passant));

    const in_check: ?Color = if (c.in_check == -1)
        null
    else
        @as(u1, @intCast(c.in_check));

    const white_pieces = Bitboard{ .bits = c.colors[0] };
    const black_pieces = Bitboard{ .bits = c.colors[1] };
    const all_pieces = [6]Bitboard{
        .{ .bits = c.pieces[0] },
        .{ .bits = c.pieces[1] },
        .{ .bits = c.pieces[2] },
        .{ .bits = c.pieces[3] },
        .{ .bits = c.pieces[4] },
        .{ .bits = c.pieces[5] },
    };

    var mailbox: [64]?Piece = @splat(null);
    for (0..6) |piece_idx| {
        var bb = all_pieces[piece_idx];
        while (bb.next()) |s| {
            mailbox[s] = @intCast(piece_idx);
        }
    }

    var res: State = .{
        .colors = .{ white_pieces, black_pieces },
        .pieces = all_pieces,
        .to_move = @intCast(c.to_move),
        .castling_rights = @intCast(c.castling_rights),
        .en_passant = en_passant,
        .halfmove_clock = c.halfmove_clock,
        .fullmove_clock = c.fullmove_clock,
        .in_check = in_check,
        .all_pieces = white_pieces.bitOr(black_pieces),
        .mailbox = mailbox,
    };

    res.zobrist_hash = res.computeHash();
    return res;
}

pub fn toCState(self: *const State, c_state: *ffi.CState) void {
    inline for (0..6) |piece_idx| c_state.*.pieces[piece_idx] = self.pieces[piece_idx].bits;
    inline for (0..2) |color_idx| c_state.*.colors[color_idx] = self.colors[color_idx].bits;

    c_state.*.to_move = @intCast(self.to_move);
    c_state.*.castling_rights = @intCast(self.castling_rights);
    c_state.*.en_passant = if (self.en_passant) |ep|
        @intCast(ep)
    else
        -1;
    c_state.*.in_check = if (self.in_check) |c|
        @intCast(c)
    else
        -1;
    c_state.*.halfmove_clock = self.halfmove_clock;
    c_state.*.fullmove_clock = self.fullmove_clock;
}

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
        Pieces.rook,
        Pieces.knight,
        Pieces.bishop,
        Pieces.queen,
        Pieces.king,
        Pieces.bishop,
        Pieces.knight,
        Pieces.rook,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
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
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.pawn,
        Pieces.rook,
        Pieces.knight,
        Pieces.bishop,
        Pieces.queen,
        Pieces.king,
        Pieces.bishop,
        Pieces.knight,
        Pieces.rook,
    };

    var state = State{
        .colors = .{ white_pieces, black_pieces },
        .pieces = .{ pawns, knights, bishops, rooks, queens, kings },
        .to_move = Colors.white,
        .castling_rights = Castling.all_legal,
        .all_pieces = white_pieces.bitOr(black_pieces),
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
    var castling_rights = Castling.no_legal;
    var halfmove_clock: ?u16 = null;

    var halfmove_start: usize = 0;
    var fullmove_start: usize = 0;

    for (fen, 0..) |c, i| {
        switch (state) {
            .placement => {
                switch (c) {
                    'p' => {
                        pawns.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'P' => {
                        pawns.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'n' => {
                        knights.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'N' => {
                        knights.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'b' => {
                        bishops.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'B' => {
                        bishops.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'r' => {
                        rooks.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'R' => {
                        rooks.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'q' => {
                        queens.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'Q' => {
                        queens.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'k' => {
                        kings.bitOrAssign(current_square);
                        black_pieces.bitOrAssign(current_square);
                        if (current_square % 8 < 7) {
                            current_square += 1;
                        }
                    },
                    'K' => {
                        kings.bitOrAssign(current_square);
                        white_pieces.bitOrAssign(current_square);
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
                    'k' => castling_rights |= Castling.black_kingside,
                    'K' => castling_rights |= Castling.white_kingside,
                    'q' => castling_rights |= Castling.black_queenside,
                    'Q' => castling_rights |= Castling.white_queenside,
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

    const fullmove_clock = std.fmt.parseInt(u16, fen[fullmove_start..], 10) catch return error.InvalidFullmoveClock;

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
        .fullmove_clock = fullmove_clock,
        .all_pieces = white_pieces.bitOr(black_pieces),
        .mailbox = mailbox,
    };

    const pieces = res.colorBitboard(res.to_move);
    const king_mask = res.pieceBitboard(Pieces.king).bitAnd(pieces);
    const king_square = king_mask.trailingZeros();
    if (movegen.isSquareAttackedBy(&res, king_square, ~res.to_move)) {
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
    var rank: Squares.Square = 7;
    while (true) {
        try writer.print(" {d} ", .{rank + 1});
        var file: u6 = 0;
        while (file < 8) {
            defer file += 1;
            const square = rank * 8 + file;

            const isDarkSquare = if ((@as(u8, rank) + @as(u8, square)) % 2 == 1)
                true
            else
                false;

            if (isDarkSquare) {
                try configureColor(writer, .background, .dark);
            } else {
                try configureColor(writer, .background, .light);
            }

            if (self.pieceAt(square)) |piece| {
                if (isDarkSquare) {
                    try configureColor(writer, .foreground, .dark);
                } else {
                    try configureColor(writer, .foreground, .light);
                }
                try printSpacer(writer);

                switch (self.colorAt(square).?) {
                    game.Colors.white => try configureColor(writer, .foreground, .white),
                    game.Colors.black => try configureColor(writer, .foreground, .black),
                }
                try writer.print("{s} ", .{game.piece_repr_symbol[piece]});
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
    const keys = game.getZobristKeys();
    var h: u64 = 0;

    // Hash all pieces with their colors
    for (0..2) |color_idx| {
        const color: Color = @intCast(color_idx);
        var color_pieces = self.colorBitboard(color);
        while (color_pieces.next()) |s| {
            const p = self.pieceAt(s).?;
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

pub fn colorBitboard(self: *const State, c: Color) Bitboard {
    return self.colors[c];
}

pub fn pieceBitboard(self: *const State, p: Piece) Bitboard {
    return self.pieces[p];
}

fn allPieces(self: *const State) Bitboard {
    return self.colors[0].bitOr(self.colors[1]);
}

pub fn pieceAt(self: *const State, s: Square) ?Piece {
    return self.mailbox[s];
}

pub fn colorAt(self: *const State, s: Square) ?Color {
    if (!self.all_pieces.contains(s)) return null;

    return @intFromBool(self.colors[1].contains(s));
}

pub fn colorAtUnchecked(self: *const State, s: Square) Color {
    return @intFromBool(self.colors[1].contains(s));
}

pub fn isSquareEmpty(self: *const State, s: Square) bool {
    return !self.all_pieces.contains(s);
}

// Returns true if the given color has any non-pawn material (knights, bishops, rooks, queens)
pub fn hasNonPawnMaterial(self: *const State, c: Color) bool {
    const color_pieces = self.colorBitboard(c);
    const non_pawn_pieces = self.pieceBitboard(Pieces.knight)
        .bitOr(self.pieceBitboard(Pieces.bishop))
        .bitOr(self.pieceBitboard(Pieces.rook))
        .bitOr(self.pieceBitboard(Pieces.queen));
    return !color_pieces.bitAnd(non_pawn_pieces).isEmpty();
}

pub fn makeMove(self: *State, m: game.Move, c: Color, p: Piece) UndoInfo {
    const keys = game.getZobristKeys();
    const start = m.start;
    const end = m.end;

    // Store undo info before modifying state
    var undo = UndoInfo{
        .captured_piece = self.pieceAt(end),
        .captured_square = end,
        .castling_rights = self.castling_rights,
        .en_passant = self.en_passant,
        .halfmove_clock = self.halfmove_clock,
        .in_check = self.in_check,
        .zobrist_hash = self.zobrist_hash,
        .was_promotion = false,
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
        Pieces.pawn => {
            self.*.halfmove_clock = 0;
            const start_rank = start / 8;
            const end_rank = end / 8;

            if (start_rank == pawn_start_rank[c] and end_rank == pawn_double_rank[c]) {
                new_en_passant = @intCast(@as(i8, end) + pawn_ep_offset[c]);
            } else if (self.en_passant == end) {
                en_passant_target = @intCast(@as(i8, end) + pawn_ep_offset[c]);
            }
        },
        Pieces.king => {
            if (game.absDiff(start, end) == 2) {
                // Determine kingside (0) or queenside (1) based on end file
                const side: Color = @intFromBool(end % 8 < 4); // c-file < e-file
                const data = game.castle_data[c][side];

                if (self.castling_rights & data.rights_bit != 0) {
                    undo.was_castling = true;
                    undo.castling_side = side;

                    // Move rook
                    self.*.pieces[Pieces.rook].bitXorAssign(data.rook_from);
                    self.*.colors[c].bitXorAssign(data.rook_from);
                    self.*.mailbox[data.rook_from] = null;
                    self.*.pieces[Pieces.rook].bitOrAssign(data.rook_to);
                    self.*.colors[c].bitOrAssign(data.rook_to);
                    self.*.mailbox[data.rook_to] = Pieces.rook;

                    // Update hash
                    self.*.zobrist_hash ^= keys.pieces[c][Pieces.rook][data.rook_from];
                    self.*.zobrist_hash ^= keys.pieces[c][Pieces.rook][data.rook_to];
                }
            }
            self.*.castling_rights &= game.king_castling_mask[c];
        },
        Pieces.rook => self.*.castling_rights &= game.rook_castling_mask[start],
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
        self.*.castling_rights &= game.rook_castling_mask[end];
        self.*.halfmove_clock = 0;
        self.*.pieces[x].bitXorAssign(end);
        self.*.colors[~c].bitXorAssign(end);
        self.*.mailbox[end] = null;
        // XOR out captured piece from hash
        self.*.zobrist_hash ^= keys.pieces[~c][x][end];
    }

    // Handle en passant capture
    if (en_passant_target) |t| {
        undo.captured_piece = Pieces.pawn;
        undo.captured_square = t;
        self.*.halfmove_clock = 0;
        self.*.pieces[Pieces.pawn].bitXorAssign(t);
        self.*.colors[~c].bitXorAssign(t);
        self.*.mailbox[t] = null;
        // XOR out captured pawn from hash
        self.*.zobrist_hash ^= keys.pieces[~c][Pieces.pawn][t];
    }

    // XOR out piece from start square
    self.*.zobrist_hash ^= keys.pieces[c][p][start];

    // Actually move the piece
    self.*.pieces[p].bitXorAssign(start);
    self.*.colors[c].bitXorAssign(start);
    self.*.mailbox[start] = null;
    self.*.pieces[p].bitOrAssign(end);
    self.*.colors[c].bitOrAssign(end);
    self.*.mailbox[end] = p;

    if (m.promotion_piece) |promo_target| {
        undo.was_promotion = true;
        self.*.pieces[Pieces.pawn].bitXorAssign(end);
        self.*.pieces[promo_target].bitOrAssign(end);
        // XOR in queen at end square (not pawn)
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

    // Update in_check for the new side to move
    const new_to_move = self.to_move;
    const king_bb = self.pieceBitboard(Pieces.king).bitAnd(self.colorBitboard(new_to_move));
    const king_square = king_bb.trailingZeros();
    if (movegen.isSquareAttackedBy(self, king_square, ~new_to_move)) {
        self.*.in_check = new_to_move;
    } else {
        self.*.in_check = null;
    }

    return undo;
}

// Unmake a move, restoring the previous state
pub fn unmakeMove(self: *State, m: game.Move, c: Color, p: Piece, undo: UndoInfo) void {
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

    // Handle promotion: piece on end square is queen, but we need to restore pawn
    const actual_piece = if (undo.was_promotion) Pieces.queen else p;

    // Move piece back from end to start
    self.*.pieces[actual_piece].bitXorAssign(end);
    self.*.colors[c].bitXorAssign(end);
    self.*.mailbox[end] = null;
    self.*.pieces[p].bitOrAssign(start);
    self.*.colors[c].bitOrAssign(start);
    self.*.mailbox[start] = p;

    // Handle castling: unmove the rook
    if (undo.was_castling) {
        const data = game.castle_data[c][undo.castling_side];
        // Move rook back
        self.*.pieces[Pieces.rook].bitXorAssign(data.rook_to);
        self.*.colors[c].bitXorAssign(data.rook_to);
        self.*.mailbox[data.rook_to] = null;
        self.*.pieces[Pieces.rook].bitOrAssign(data.rook_from);
        self.*.colors[c].bitOrAssign(data.rook_from);
        self.*.mailbox[data.rook_from] = Pieces.rook;
    }

    // Restore captured piece
    if (undo.captured_piece) |captured| {
        const cap_sq = undo.captured_square;
        self.*.pieces[captured].bitOrAssign(cap_sq);
        self.*.colors[~c].bitOrAssign(cap_sq);
        self.*.mailbox[cap_sq] = captured;
    }

    // Update all_pieces
    self.all_pieces = self.allPieces();
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
    _ = state.makeMove(game.Move{ .start = Squares.e2, .end = Squares.e4 }, Colors.white, Pieces.pawn);
    _ = state.makeMove(game.Move{ .start = Squares.e7, .end = Squares.e5 }, Colors.black, Pieces.pawn);
    _ = state.makeMove(game.Move{ .start = Squares.g1, .end = Squares.f3 }, Colors.white, Pieces.knight);
    _ = state.makeMove(game.Move{ .start = Squares.d8, .end = Squares.e7 }, Colors.black, Pieces.queen);
    _ = state.makeMove(game.Move{ .start = Squares.f1, .end = Squares.e2 }, Colors.white, Pieces.bishop);
    _ = state.makeMove(game.Move{ .start = Squares.e7, .end = Squares.d8 }, Colors.black, Pieces.queen);
    _ = state.makeMove(game.Move{ .start = Squares.e1, .end = Squares.f1 }, Colors.white, Pieces.king);
    try expectEqual(Castling.all_legal ^ Castling.white_castling, state.castling_rights);

    state = State.defaultPosition();
    _ = state.makeMove(game.Move{ .start = Squares.a2, .end = Squares.a3 }, Colors.white, Pieces.pawn);
    _ = state.makeMove(game.Move{ .start = Squares.a7, .end = Squares.a6 }, Colors.black, Pieces.pawn);
    _ = state.makeMove(game.Move{ .start = Squares.a1, .end = Squares.a2 }, Colors.white, Pieces.rook);
    try expectEqual(Castling.all_legal ^ Castling.white_queenside, state.castling_rights);
}

test "test fen from default" {
    const starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const state = try State.fromFen(starting_fen);
    const default_state = State.defaultPosition();

    try expectEqual(default_state.pieces, state.pieces);
    try expectEqual(default_state.colors, state.colors);
    try expectEqual(default_state.to_move, state.to_move);
    try expectEqual(default_state.castling_rights, state.castling_rights);
    try expectEqual(default_state.en_passant, state.en_passant);
    try expectEqual(default_state.in_check, state.in_check);
    try expectEqual(default_state.halfmove_clock, state.halfmove_clock);
    try expectEqual(default_state.fullmove_clock, state.fullmove_clock);
}

test "test fen from e4 c5 nf3" {
    const fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";
    const state = try State.fromFen(fen);

    const pawns = state.pieceBitboard(Pieces.pawn);
    const knights = state.pieceBitboard(Pieces.knight);
    const bishops = state.pieceBitboard(Pieces.bishop);
    const rooks = state.pieceBitboard(Pieces.rook);
    const queens = state.pieceBitboard(Pieces.queen);
    const kings = state.pieceBitboard(Pieces.king);

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
    try expectEqual(Castling.all_legal, state.castling_rights);
    try expectEqual(null, state.en_passant);
    try expectEqual(1, state.halfmove_clock);
    try expectEqual(2, state.fullmove_clock);
}

test "incremental hash matches computed hash" {
    var state = State.defaultPosition();

    // Play some moves and verify hash consistency after each
    _ = state.makeMove(game.Move{ .start = Squares.e2, .end = Squares.e4 }, Colors.white, Pieces.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(game.Move{ .start = Squares.e7, .end = Squares.e5 }, Colors.black, Pieces.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(game.Move{ .start = Squares.g1, .end = Squares.f3 }, Colors.white, Pieces.knight);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(game.Move{ .start = Squares.b8, .end = Squares.c6 }, Colors.black, Pieces.knight);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    // Test capture
    _ = state.makeMove(game.Move{ .start = Squares.f1, .end = Squares.b5 }, Colors.white, Pieces.bishop);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(game.Move{ .start = Squares.a7, .end = Squares.a6 }, Colors.black, Pieces.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(game.Move{ .start = Squares.b5, .end = Squares.c6 }, Colors.white, Pieces.bishop);
    try expectEqual(state.computeHash(), state.zobrist_hash);
}

test "incremental hash with castling" {
    // Position where white can castle kingside
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    // Castle kingside
    _ = state.makeMove(game.Move{ .start = Squares.e1, .end = Squares.g1 }, Colors.white, Pieces.king);
    try expectEqual(state.computeHash(), state.zobrist_hash);
}

test "incremental hash with en passant" {
    var state = State.defaultPosition();

    // Set up en passant
    _ = state.makeMove(game.Move{ .start = Squares.e2, .end = Squares.e4 }, Colors.white, Pieces.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(game.Move{ .start = Squares.a7, .end = Squares.a6 }, Colors.black, Pieces.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(game.Move{ .start = Squares.e4, .end = Squares.e5 }, Colors.white, Pieces.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    _ = state.makeMove(game.Move{ .start = Squares.d7, .end = Squares.d5 }, Colors.black, Pieces.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);

    // En passant capture
    _ = state.makeMove(game.Move{ .start = Squares.e5, .end = Squares.d6 }, Colors.white, Pieces.pawn);
    try expectEqual(state.computeHash(), state.zobrist_hash);
}

// Castling rights removal tests - king moves
test "king move removes both castling rights" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move white king e1 to f1
    _ = state.makeMove(game.Move{ .start = Squares.e1, .end = Squares.f1 }, Colors.white, Pieces.king);

    // Both white castling rights should be removed
    try expectEqual(Castling.no_legal, state.castling_rights & Castling.white_castling);
    // Black castling rights should remain
    try expectEqual(Castling.black_castling, state.castling_rights & Castling.black_castling);
}

test "black king move removes black castling" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move black king e8 to f8
    _ = state.makeMove(game.Move{ .start = Squares.e8, .end = Squares.f8 }, Colors.black, Pieces.king);

    // Both black castling rights should be removed
    try expectEqual(Castling.no_legal, state.castling_rights & Castling.black_castling);
    // White castling rights should remain
    try expectEqual(Castling.white_castling, state.castling_rights & Castling.white_castling);
}

// Castling rights removal tests - rook moves
test "kingside rook move removes kingside castling" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move h1 rook to g1
    _ = state.makeMove(game.Move{ .start = Squares.h1, .end = Squares.g1 }, Colors.white, Pieces.rook);

    // White kingside castling should be removed
    try expectEqual(Castling.no_legal, state.castling_rights & Castling.white_kingside);
    // White queenside should remain
    try expectEqual(Castling.white_queenside, state.castling_rights & Castling.white_queenside);
}

test "queenside rook move removes queenside castling" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move a1 rook to b1
    _ = state.makeMove(game.Move{ .start = Squares.a1, .end = Squares.b1 }, Colors.white, Pieces.rook);

    // White queenside castling should be removed
    try expectEqual(Castling.no_legal, state.castling_rights & Castling.white_queenside);
    // White kingside should remain
    try expectEqual(Castling.white_kingside, state.castling_rights & Castling.white_kingside);
}

test "black kingside rook move" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move h8 rook to g8
    _ = state.makeMove(game.Move{ .start = Squares.h8, .end = Squares.g8 }, Colors.black, Pieces.rook);

    // Black kingside castling should be removed
    try expectEqual(Castling.no_legal, state.castling_rights & Castling.black_kingside);
    // Black queenside should remain
    try expectEqual(Castling.black_queenside, state.castling_rights & Castling.black_queenside);
}

test "black queenside rook move" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Move a8 rook to b8
    _ = state.makeMove(game.Move{ .start = Squares.a8, .end = Squares.b8 }, Colors.black, Pieces.rook);

    // Black queenside castling should be removed
    try expectEqual(Castling.no_legal, state.castling_rights & Castling.black_queenside);
    // Black kingside should remain
    try expectEqual(Castling.black_kingside, state.castling_rights & Castling.black_kingside);
}

// Rook capture removes opponent castling rights
test "capturing rook removes opponent castling" {
    const fen = "r3k2r/pppppppp/8/7B/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Bishop on h5 captures h8 rook
    _ = state.makeMove(game.Move{ .start = Squares.h5, .end = Squares.h8 }, Colors.white, Pieces.bishop);

    // Black kingside castling should be removed
    try expectEqual(Castling.no_legal, state.castling_rights & Castling.black_kingside);
    // Black queenside should remain
    try expectEqual(Castling.black_queenside, state.castling_rights & Castling.black_queenside);
}

test "capturing white rook removes white castling" {
    const fen = "r3k2r/pppppppp/8/8/8/7b/PPPPPPPP/R3K2R b KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Bishop on h3 captures h1 rook
    _ = state.makeMove(game.Move{ .start = Squares.h3, .end = Squares.h1 }, Colors.black, Pieces.bishop);

    // White kingside castling should be removed
    try expectEqual(Castling.no_legal, state.castling_rights & Castling.white_kingside);
    // White queenside should remain
    try expectEqual(Castling.white_queenside, state.castling_rights & Castling.white_queenside);
}

// Castling execution test
test "castling moves rook correctly" {
    const fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
    var state = try State.fromFen(fen);

    // Castle kingside (O-O)
    _ = state.makeMove(game.Move{ .start = Squares.e1, .end = Squares.g1 }, Colors.white, Pieces.king);

    // King should be on g1
    try expect(state.pieceAt(Squares.g1) == Pieces.king);
    try expect(state.colorAt(Squares.g1) == Colors.white);

    // Rook should be on f1
    try expect(state.pieceAt(Squares.f1) == Pieces.rook);
    try expect(state.colorAt(Squares.f1) == Colors.white);

    // h1 should be empty
    try expect(state.pieceAt(Squares.h1) == null);
}

// En passant square setting tests
test "en passant square set after double push" {
    var state = State.defaultPosition();

    // e2-e4 should set en passant square to e3
    _ = state.makeMove(game.Move{ .start = Squares.e2, .end = Squares.e4 }, Colors.white, Pieces.pawn);

    try expectEqual(Squares.e3, state.en_passant.?);
}

test "en passant square set for black" {
    var state = State.defaultPosition();

    // White move first
    _ = state.makeMove(game.Move{ .start = Squares.e2, .end = Squares.e3 }, Colors.white, Pieces.pawn);

    // d7-d5 should set en passant square to d6
    _ = state.makeMove(game.Move{ .start = Squares.d7, .end = Squares.d5 }, Colors.black, Pieces.pawn);

    try expectEqual(Squares.d6, state.en_passant.?);
}

// En passant square clearing tests
test "en passant cleared after non-pawn move" {
    var state = State.defaultPosition();

    // e2-e4 sets en passant
    _ = state.makeMove(game.Move{ .start = Squares.e2, .end = Squares.e4 }, Colors.white, Pieces.pawn);
    try expect(state.en_passant != null);

    // Knight move should clear en passant
    _ = state.makeMove(game.Move{ .start = Squares.g8, .end = Squares.f6 }, Colors.black, Pieces.knight);

    try expectEqual(@as(?Squares.Square, null), state.en_passant);
}

test "en passant cleared after single push" {
    var state = State.defaultPosition();

    // e2-e4 sets en passant
    _ = state.makeMove(game.Move{ .start = Squares.e2, .end = Squares.e4 }, Colors.white, Pieces.pawn);
    try expect(state.en_passant != null);

    // a7-a6 (single push) should clear en passant
    _ = state.makeMove(game.Move{ .start = Squares.a7, .end = Squares.a6 }, Colors.black, Pieces.pawn);

    try expectEqual(@as(?Squares.Square, null), state.en_passant);
}

// En passant capture tests
test "en passant capture removes captured pawn" {
    const fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
    var state = try State.fromFen(fen);

    // Verify d5 pawn exists before capture
    try expect(state.pieceAt(Squares.d5) == Pieces.pawn);
    try expect(state.colorAt(Squares.d5) == Colors.black);

    // e5xd6 en passant
    _ = state.makeMove(game.Move{ .start = Squares.e5, .end = Squares.d6 }, Colors.white, Pieces.pawn);

    // d5 pawn should be removed
    try expect(state.pieceAt(Squares.d5) == null);

    // White pawn should be on d6
    try expect(state.pieceAt(Squares.d6) == Pieces.pawn);
    try expect(state.colorAt(Squares.d6) == Colors.white);
}

test "black en passant capture" {
    const fen = "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3";
    var state = try State.fromFen(fen);

    // Verify d4 pawn exists before capture
    try expect(state.pieceAt(Squares.d4) == Pieces.pawn);
    try expect(state.colorAt(Squares.d4) == Colors.white);

    // e4xd3 en passant
    _ = state.makeMove(game.Move{ .start = Squares.e4, .end = Squares.d3 }, Colors.black, Pieces.pawn);

    // d4 pawn should be removed
    try expect(state.pieceAt(Squares.d4) == null);

    // Black pawn should be on d3
    try expect(state.pieceAt(Squares.d3) == Pieces.pawn);
    try expect(state.colorAt(Squares.d3) == Colors.black);
}

test "c state roundtrip" {
    const default_state = State.defaultPosition();
    var c_state: ffi.CState = undefined;
    default_state.toCState(&c_state);
    const and_back = State.initCState(&c_state);

    for (0..2) |c| {
        try expectEqual(default_state.colors[c], and_back.colors[c]);
    }

    for (0..6) |p| {
        try expectEqual(default_state.pieces[p], and_back.pieces[p]);
    }

    try expectEqual(default_state.en_passant, and_back.en_passant);
    try expectEqual(default_state.castling_rights, and_back.castling_rights);
    try expectEqual(default_state.to_move, and_back.to_move);
    try expectEqual(default_state.halfmove_clock, and_back.halfmove_clock);
    try expectEqual(default_state.fullmove_clock, and_back.fullmove_clock);
    try expectEqual(default_state.zobrist_hash, and_back.zobrist_hash);
    try expectEqual(default_state.mailbox, and_back.mailbox);
    try expectEqual(default_state.all_pieces, and_back.all_pieces);
}
