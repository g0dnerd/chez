const std = @import("std");
const builtin = @import("builtin");

const is_freestanding = builtin.target.os.tag == .freestanding;

// On the wasm web build there is no OS clock, so the host supplies one. JS
// provides env.chez_now_ms returning absolute milliseconds (performance.timeOrigin
// + performance.now()), which is comparable across Web Worker instances — each
// worker's performance.now() has its own origin, but the absolute value does
// not, so threads measuring elapsed time against the same start agree.
extern "env" fn chez_now_ms() f64;

// Monotonic nanoseconds since an unspecified epoch, used only for elapsed-time
// measurement (durations are differences, so the epoch is irrelevant).
//
// On freestanding targets (the wasm web build) there is no OS clock and no
// std.Io backend compiles, so the host clock import is used instead.
pub fn nowNanos() i96 {
    if (comptime is_freestanding) {
        return @intFromFloat(chez_now_ms() * std.time.ns_per_ms);
    } else {
        // init_single_threaded never spawns a worker pool; it just exposes the
        // host clock.
        var threaded: std.Io.Threaded = .init_single_threaded;
        return std.Io.Clock.awake.now(threaded.io()).nanoseconds;
    }
}
