const std = @import("std");
const builtin = @import("builtin");

const is_freestanding = builtin.target.os.tag == .freestanding;

// Monotonic nanoseconds since an unspecified epoch, used only for elapsed-time
// measurement (durations are differences, so the epoch is irrelevant).
//
// On freestanding targets (the wasm web build) there is no OS clock and no
// std.Io backend compiles, so this returns 0. That path searches to a fixed
// depth with no time control and reads no elapsed time, so 0 is sufficient.
pub fn nowNanos() i96 {
    if (comptime is_freestanding) {
        return 0;
    } else {
        // init_single_threaded never spawns a worker pool; it just exposes the
        // host clock.
        var threaded: std.Io.Threaded = .init_single_threaded;
        return std.Io.Clock.awake.now(threaded.io()).nanoseconds;
    }
}
