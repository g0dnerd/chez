const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    // const optimize = b.standardOptimizeOption(.{});

    const precompute = b.addExecutable(.{
        .name = "precompute",
        .root_module = b.createModule(.{ .root_source_file = b.path("src/precompute.zig"), .target = target, .optimize = .ReleaseFast }),
    });

    b.installArtifact(precompute);

    const tui = b.addExecutable(.{
        .name = "tui",
        .root_module = b.createModule(.{ .root_source_file = b.path("src/main.zig"), .target = target, .optimize = .ReleaseFast }),
    });

    b.installArtifact(tui);
}
