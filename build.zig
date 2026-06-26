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
        .imports = &.{.{ .name = "kore", .module = kore }},
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
    addFathom(b, tui.root_module);

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
    bench.root_module.addImport("kore", kore);
    const bench_step = b.step("bench", "Run search benchmark");
    const run_bench = b.addRunArtifact(bench);
    if (b.args) |args| {
        run_bench.addArgs(args);
    }
    bench_step.dependOn(&run_bench.step);

    const tune_exe = b.addExecutable(.{
        .name = "tune",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tune.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "chez", .module = chez_mod }},
        }),
    });
    tune_exe.root_module.addImport("kore", kore);
    const tune_step = b.step("tune", "Run Texel SPSA tuner");
    const run_tune = b.addRunArtifact(tune_exe);
    if (b.args) |args| run_tune.addArgs(args);
    tune_step.dependOn(&run_tune.step);
    b.installArtifact(tune_exe);

    const tune_search_exe = b.addExecutable(.{
        .name = "tune-search",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tune_search.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "chez", .module = chez_mod }},
        }),
    });
    tune_search_exe.root_module.addImport("kore", kore);
    const tune_search_step = b.step("tune-search", "Run in-engine SPSA search-param tuner");
    const run_tune_search = b.addRunArtifact(tune_search_exe);
    if (b.args) |args| run_tune_search.addArgs(args);
    tune_search_step.dependOn(&run_tune_search.step);
    b.installArtifact(tune_search_exe);

    const nnue_inspect = b.addExecutable(.{
        .name = "nnue-inspect",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/nnue_inspect.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "chez", .module = chez_mod }},
        }),
    });
    nnue_inspect.root_module.addImport("kore", kore);
    const nnue_inspect_step = b.step("nnue-inspect", "Inspect/debug a trained .nnue net");
    const run_nnue_inspect = b.addRunArtifact(nnue_inspect);
    if (b.args) |args| run_nnue_inspect.addArgs(args);
    nnue_inspect_step.dependOn(&run_nnue_inspect.step);
    b.installArtifact(nnue_inspect);

    const quiet_filter = b.addExecutable(.{
        .name = "quiet-filter",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/quiet_filter.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "chez", .module = chez_mod }},
        }),
    });
    quiet_filter.root_module.addImport("kore", kore);

    const uci = b.addExecutable(.{
        .name = "uci",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/uci.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    uci.root_module.addImport("kore", kore);
    // Syzygy WDL probing via vendored Fathom (uci + selfplay only; engine + WASM
    // stay C-free). tbprobe.c #includes tbchess.c, so only tbprobe.c is listed.
    addFathom(b, uci.root_module);

    const selfplay = b.addExecutable(.{
        .name = "selfplay",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/selfplay.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "chez", .module = chez_mod },
                .{ .name = "kore", .module = kore },
            },
        }),
    });
    addFathom(b, selfplay.root_module);

    // Temp (v14-selfplay-v11labeler branch): build ONLY selfplay, so the
    // format-v4 nnue.zig (which lacks num_output_buckets/outputBucket) doesn't
    // have to satisfy the trainer/inspector binaries.
    const selfplay_only_step = b.step("selfplay-only", "Build only the selfplay binary");
    selfplay_only_step.dependOn(&b.addInstallArtifact(selfplay, .{}).step);

    // Temp (v14-selfplay-v11labeler branch): pilot dataset stats tool.
    const dataset_stats = b.addExecutable(.{
        .name = "dataset-stats",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/dataset_stats.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "chez", .module = chez_mod },
                .{ .name = "kore", .module = kore },
            },
        }),
    });
    const dataset_stats_step = b.step("dataset-stats", "Build the pilot dataset-stats tool");
    dataset_stats_step.dependOn(&b.addInstallArtifact(dataset_stats, .{}).step);

    const train_nnue = b.addExecutable(.{
        .name = "train_nnue",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/train_nnue.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "chez", .module = chez_mod },
                .{ .name = "kore", .module = kore },
            },
        }),
    });
    const train_nnue_step = b.step("train_nnue", "Build the NNUE trainer binary");
    train_nnue_step.dependOn(&b.addInstallArtifact(train_nnue, .{}).step);

    const test_step = b.step("test", "Run unit tests");
    const test_filters: []const []const u8 = b.option(
        []const []const u8,
        "test_filter",
        "Skip tests that do not match any of the specified filters",
    ) orelse &.{};
    const unit_tests = b.addTest(.{
        .name = "chez_tests",
        .root_module = chez_mod,
        .filters = test_filters,
    });
    const selfplay_unit_tests = b.addTest(.{
        .name = "selfplay_tests",
        .root_module = selfplay.root_module,
        .filters = test_filters,
    });
    selfplay_unit_tests.root_module.addImport("kore", kore);

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const run_selfplay_unit_tests = b.addRunArtifact(selfplay_unit_tests);
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_selfplay_unit_tests.step);

    b.installArtifact(libchez);
    b.installArtifact(precompute);
    b.installArtifact(tui);
    b.installArtifact(bench);
    b.installArtifact(uci);
    b.installArtifact(quiet_filter);
    b.installArtifact(selfplay);
    b.installArtifact(train_nnue);

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
    // wasm uses only kore's CPU inference ops; a CPU-only kore instance drops the
    // OpenCL/libc link that freestanding can't satisfy.
    const kore_wasm = b.dependency("kore", .{ .no_gpu = true }).module("kore");
    wasm.root_module.addImport("kore", kore_wasm);
    wasm.entry = .disabled;
    wasm.rdynamic = true;

    const wasm_step = b.step("wasm", "Build WASM module");
    const wasm_install = b.addInstallArtifact(wasm, .{
        .dest_dir = .{ .override = .{ .custom = "web" } },
    });
    wasm_step.dependOn(&wasm_install.step);

    // Optionally stage a .nnue net next to the wasm so the browser engine evals
    // with NNUE (web/app.js fetches "chez.nnue"). Opt-in to avoid copying ~42MB
    // on every build, e.g. `zig build wasm -Dnnue_web=data/net_v13_screlu.nnue`.
    if (b.option([]const u8, "nnue_web", "Path to a .nnue net to install as web/chez.nnue")) |net_path| {
        const install_net = b.addInstallFileWithDir(b.path(net_path), .{ .custom = "web" }, "chez.nnue");
        wasm_step.dependOn(&install_net.step);
    }

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

// Attach the vendored Fathom Syzygy prober (C + libc) to a module. Used only by
// the uci and selfplay executables; the engine module and WASM target never call
// this so they stay free of any C/libc dependency.
fn addFathom(b: *std.Build, mod: *std.Build.Module) void {
    mod.addIncludePath(b.path("vendor/fathom"));
    mod.addCSourceFile(.{
        .file = b.path("vendor/fathom/tbprobe.c"),
        .flags = &.{ "-O3", "-std=gnu11" },
    });
    mod.link_libc = true;
}
