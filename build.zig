const std = @import("std");

pub fn build(b: *std.Build) !void {
    const gb10 = b.option(bool, "gb10", "Force build for NVIDIA Blackwell GB10 chip") orelse false;
    const portable = b.option(bool, "portable", "Build without CPU-specific optimizations") orelse false;
    const target = if (portable)
        b.standardTargetOptions(.{})
    else if (gb10)
        b.resolveTargetQuery(.{
            .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.gb10 },
        })
    else
        b.resolveTargetQuery(.{
            .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 },
        });

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

    const puzzle_mod = b.addModule("puzzles", .{
        .root_source_file = b.path("tests/puzzles.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    puzzle_mod.addImport("chez", chez_mod);
    const puzzle_test_step = b.step("test-puzzle", "Run puzzle tests");
    const puzzle_test_filter = [_][]const u8{"puzzles"};
    const puzzle_tests = b.addTest(.{
        .name = "puzzle_tests",
        .test_runner = .{
            .path = b.path("tests/test_runner.zig"),
            .mode = .simple,
        },
        .root_module = puzzle_mod,
        .filters = &puzzle_test_filter,
    });
    const run_puzzle_tests = b.addRunArtifact(puzzle_tests);
    puzzle_test_step.dependOn(&run_puzzle_tests.step);

    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    const bench_step = b.step("bench", "Run search benchmark");
    const run_bench = b.addRunArtifact(bench);
    if (b.args) |args| {
        run_bench.addArgs(args);
    }
    bench_step.dependOn(&run_bench.step);

    const uci = b.addExecutable(.{
        .name = "uci",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/uci.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });

    b.installArtifact(libchez);
    b.installArtifact(precompute);
    b.installArtifact(tui);
    b.installArtifact(bench);
    b.installArtifact(uci);

    // WASM build for web interface
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const wasm = b.addExecutable(.{
        .name = "chez",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wasm.zig"),
            .target = wasm_target,
            .optimize = .ReleaseSmall,
        }),
    });
    wasm.entry = .disabled;
    wasm.rdynamic = true;

    const wasm_step = b.step("wasm", "Build WASM module");
    const wasm_install = b.addInstallArtifact(wasm, .{
        .dest_dir = .{ .override = .{ .custom = "web" } },
    });
    wasm_step.dependOn(&wasm_install.step);

    // HTTP server for web interface
    const server = b.addExecutable(.{
        .name = "server",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/server.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    b.installArtifact(server);
}
