// WASM interface for the Chez chess engine
// Uses global state pattern - single game instance, no dynamic allocation needed

const engine = @import("chez.zig").engine;

// Global game state
var game_state: engine.State = undefined;
var move_list: engine.movegen.MoveList = undefined;
var undo_info: [2]?engine.State.UndoInfo = @splat(null);
var last_moves: [2]?engine.Move = @splat(null);
var last_pieces: [2]?engine.piece.Piece = @splat(null);

// Initialize a new game at starting position
export fn wasm_init_default() void {
    game_state = engine.State.defaultPosition();
}

// Initialize from FEN string
// Returns true on success, false on invalid FEN
export fn wasm_init_fen(ptr: [*]const u8, len: usize) bool {
    const fen = ptr[0..len];
    game_state = engine.State.fromFen(fen) catch return false;
    return true;
}

// Get piece at square (0-63)
// Returns piece type (0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king)
// Returns 255 if square is empty
export fn wasm_piece_at(square: u8) u8 {
    if (square > 63) return 255;
    const sq: engine.square.Square = @intCast(square);
    if (game_state.mailbox[sq]) |piece| {
        return piece;
    }
    return 255;
}

// Get color at square (0-63)
// Returns 0 for white, 1 for black, 255 if empty
export fn wasm_color_at(square: u8) u8 {
    if (square > 63) return 255;
    const sq: engine.square.Square = @intCast(square);
    if (game_state.colorAt(sq)) |color| {
        return color;
    }
    return 255;
}

// Get current side to move (0=white, 1=black)
export fn wasm_to_move() u8 {
    return game_state.to_move;
}

// Generate legal moves for current position
// Returns the number of legal moves
export fn wasm_generate_moves() u8 {
    move_list = engine.movegen.legalMoves(&game_state, game_state.to_move);
    return move_list.len;
}

// Get move at index from last generated move list
// Returns packed int: (start << 16) | (end << 8) | promo
// promo is 0 if no promotion, otherwise piece type (1=knight, 2=bishop, 3=rook, 4=queen)
export fn wasm_get_move(index: u8) u32 {
    if (index >= move_list.len) return 0;
    const m = move_list.moves[index];
    const promo: u8 = if (m.is_promotion) m.promotion_piece else 0;
    return (@as(u32, m.start) << 16) | (@as(u32, m.end) << 8) | @as(u32, promo);
}

// Make a move on the board
// Returns true if move was legal and applied, false otherwise
export fn wasm_make_move(start: u8, end: u8, promo: u8, is_player: bool) i8 {
    if (start > 63 or end > 63) return -1;

    const start_sq: engine.square.Square = @intCast(start);
    const end_sq: engine.square.Square = @intCast(end);

    // Get the piece at the start square
    const piece = game_state.mailbox[start_sq] orelse return -2;
    const color = game_state.colorAt(start_sq) orelse return -3;

    // Verify it's this player's turn
    if (color != game_state.to_move) return -4;

    // Construct the move
    const is_promotion = promo != 0;
    const promotion_piece: engine.piece.Piece = if (is_promotion) undefined else @intCast(promo);
    const move = engine.Move{
        .start = start_sq,
        .end = end_sq,
        .promotion_piece = promotion_piece,
        .is_promotion = is_promotion,
    };

    // Verify move is legal
    const legal_moves = engine.movegen.legalMoves(&game_state, game_state.to_move);
    var is_legal = false;
    for (0..legal_moves.len) |i| {
        if (legal_moves.moves[i].eql(move)) {
            is_legal = true;
            break;
        }
    }
    if (!is_legal) return -5;

    // Apply the move
    const u = game_state.makeMove(move, color, piece);
    const idx = @intFromBool(~is_player);

    undo_info[idx] = u;
    last_moves[idx] = move;
    last_pieces[idx] = piece;

    return 1;
}

// Get game result
// Returns: 0=ongoing, 1=white wins, 2=black wins, 3=draw
export fn wasm_game_result() i32 {
    const result = engine.search.isGameOver(&game_state) orelse return 0;
    return switch (result) {
        .checkmate => |winner| if (winner == engine.Colors.white) 1 else 2,
        .stalemate, .fiftyMoveRule, .threefoldRepetition, .insufficientMaterial => 3,
    };
}

export fn wasm_unmake_move() i8 {
    const u_engine = undo_info[1] orelse return -1;
    const u_player = undo_info[0] orelse return -2;
    const mv_engine = last_moves[1] orelse return -3;
    const mv_player = last_moves[0] orelse return -4;
    const p_engine = last_pieces[1] orelse return -5;
    const p_player = last_pieces[0] orelse return -6;

    game_state.unmakeMove(mv_engine, ~game_state.to_move, p_engine, u_engine);
    game_state.unmakeMove(mv_player, ~game_state.to_move, p_player, u_player);

    @memset(&undo_info, null);
    @memset(&last_moves, null);
    @memset(&last_pieces, null);

    return 0;
}

// Get best move from engine
// Returns packed int: (start << 16) | (end << 8) | promo
// Returns 0 if no legal moves
export fn wasm_get_best_move(depth: u8) u32 {
    const result = engine.search.searchSingleThreaded(&game_state, depth) catch return 0;
    if (result) |r| {
        const m = r.move;
        const promo: u8 = if (m.is_promotion) m.promotion_piece else 0;
        return (@as(u32, m.start) << 16) | (@as(u32, m.end) << 8) | @as(u32, promo);
    }
    return 0;
}

// Get fullmove clock (move number)
export fn wasm_fullmove_clock() u16 {
    return game_state.fullmove_clock;
}

// Get halfmove clock (for 50-move rule)
export fn wasm_halfmove_clock() u16 {
    return game_state.halfmove_clock;
}

// Check if king is in check
export fn wasm_in_check() u8 {
    if (game_state.in_check) |color| {
        return color;
    }
    return 255;
}

// Get en passant square (255 if none)
export fn wasm_en_passant() u8 {
    if (game_state.en_passant) |sq| {
        return sq;
    }
    return 255;
}

// Get castling rights as bit flags
// bit 0: white kingside, bit 1: white queenside
// bit 2: black queenside, bit 3: black kingside
export fn wasm_castling_rights() u8 {
    return game_state.castling_rights;
}
