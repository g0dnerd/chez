// Pure C glue to the vendored Fathom Syzygy prober. Imports NO engine types, so
// it has no module-identity coupling: the uci/selfplay executables install
// &probeRaw into engine.tablebase.raw_probe_fn from their main(). Compiled and
// linked into uci + selfplay only (see build.zig addFathom); the engine module
// and WASM target never reference this file and stay C-free.

const c = @cImport({
    @cInclude("tbprobe.h");
});

// Largest supported man-count, mirrored from Fathom's TB_LARGEST after init().
pub var largest: u32 = 0;

// Initialize the tablebases from a directory path. Returns true only if tables
// were actually found (TB_LARGEST >= 3). Not thread-safe: call before spawning
// workers / while no search is running.
pub fn init(path: [*:0]const u8) bool {
    const ok = c.tb_init(path);
    largest = @intCast(c.TB_LARGEST);
    return ok and largest >= 3;
}

pub fn free() void {
    c.tb_free();
}

// Raw WDL probe matching engine.tablebase.raw_probe_fn's primitive signature.
// rule50 and castling are passed as 0: callers only probe positions with no
// castling rights, and rule50==0 keeps WDL 50-move-correct (cursed/blessed are
// then surfaced and collapsed to draws by the caller).
pub fn probeRaw(
    white: u64,
    black: u64,
    kings: u64,
    queens: u64,
    rooks: u64,
    bishops: u64,
    knights: u64,
    pawns: u64,
    ep: u32,
    turn: bool,
) u32 {
    return @intCast(c.tb_probe_wdl(
        white,
        black,
        kings,
        queens,
        rooks,
        bishops,
        knights,
        pawns,
        0, // rule50
        0, // castling
        ep,
        turn,
    ));
}
