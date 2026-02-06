pub const Piece = u3;

pub const pawn: Piece = 0;
pub const knight: Piece = 1;
pub const bishop: Piece = 2;
pub const rook: Piece = 3;
pub const queen: Piece = 4;
pub const king: Piece = 5;

pub const piece_repr_symbol = [6][]const u8{ "󰡙", "󰡘", "󰡜", "󰡛", "󰡚", "󰡗" };

pub fn pieceName(p: Piece) []const u8 {
    return switch (p) {
        pawn => "pawn",
        knight => "knight",
        bishop => "bishop",
        rook => "rook",
        queen => "queen",
        king => "king",
        else => unreachable,
    };
}

pub fn pieceLetter(p: Piece) u8 {
    return switch (p) {
        pawn => 'p',
        knight => 'n',
        bishop => 'b',
        rook => 'r',
        queen => 'q',
        king => 'k',
        else => unreachable,
    };
}

pub const SliderDirections = [4][2]i2;
pub const rook_directions: SliderDirections = .{
    .{ 0, 1 },
    .{ 1, 0 },
    .{ 0, -1 },
    .{ -1, 0 },
};
pub const bishop_directions: SliderDirections = .{
    .{ 1, 1 },
    .{ 1, -1 },
    .{ -1, 1 },
    .{ -1, -1 },
};
