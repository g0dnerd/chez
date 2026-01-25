const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});

    const kore_dep = b.dependency("kore", .{});
    const kore = kore_dep.module("kore");

    const precompute = b.addExecutable(.{
        .name = "precompute",
        .root_module = b.createModule(.{ .root_source_file = b.path("src/precompute.zig"), .target = target, .optimize = .ReleaseFast }),
    });

    b.installArtifact(precompute);

    const tui = b.addExecutable(.{ .name = "tui", .root_module = b.createModule(.{ .root_source_file = b.path("src/main.zig"), .target = target, .optimize = .ReleaseFast }) });
    tui.root_module.addImport("kore", kore);
    b.installArtifact(tui);

    const test_step = b.step("test", "Run unit tests");
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{ .root_source_file = b.path("src/search.zig"), .target = target }),
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);
}
