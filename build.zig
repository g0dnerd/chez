const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});

    const kore_dep = b.dependency("kore", .{});
    const kore = kore_dep.module("kore");

    const chez_mod = b.addModule("chez", .{
        .root_source_file = b.path("src/chez.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });

    const precompute = b.addExecutable(.{
        .name = "precompute",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/precompute.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });

    const tui = b.addExecutable(.{
        .name = "tui",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tui.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    tui.root_module.addImport("kore", kore);

    const libchez = b.addLibrary(.{
        .name = "chez",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ffi.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
        .version = .{ .major = 0, .minor = 0, .patch = 1 },
    });

    const test_step = b.step("test", "Run unit tests");
    const unit_tests = b.addTest(.{
        .name = "chez_tests",
        .root_module = chez_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);

    b.installArtifact(libchez);
    b.installArtifact(precompute);
    b.installArtifact(tui);
}
