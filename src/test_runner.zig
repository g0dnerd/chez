const builtin = @import("builtin");
const std = @import("std");

pub fn main() !void {
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    var stdout_buffer: [512]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    for (builtin.test_functions) |t| {
        // Check for and skip empty `refAllDecls` tests
        if (std.mem.containsAtLeast(u8, t.name, 1, "_0")) continue;

        t.func() catch |err| {
            try stdout.print("{s} fail: {}\n", .{ t.name, err });
            try stdout.flush();
            continue;
        };
        try stdout.print("{s} passed\n", .{t.name});
        try stdout.flush();
    }
}
