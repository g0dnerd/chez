// Texel SPSA tuner.
//
// Usage:
//   zig build tune -- [options]
//
//   --dataset <path>          Quiet-labeled EPD file (required)
//   --output <path>           Output params.zig path [src/engine/params.zig]
//   --max-positions <n>       Position count limit [5000000]
//   --threads <n>             Worker threads [8]
//   --iterations <n>          SPSA iterations [500000]
//   --batch-size <n>          Positions per SPSA step [16384]
//   --a <f>                   SPSA step-size numerator [10.0]
//   --big-a <f>               SPSA stability constant [100.0]
//   --alpha <f>               Step-size decay exponent [0.602]
//   --c <f>                   Perturbation size [1.0]
//   --gamma <f>               Perturbation decay exponent [0.101]
//   --k <f>                   Skip K-tuning, use this K directly
//   --k-only                  Tune K, print result, exit
//   --checkpoint-interval <n> Write params every N iters [1000]
//   --calibrate-n <n>         Calibration iterations before SPSA [50]
//   --skip-calibrate          Skip the calibration output pass
//   --skip-k-tune             Skip K-tuning (use K=1.0 unless --k supplied)
//   --no-early-stop           Disable early stopping (run all iterations)

const std = @import("std");
const kore = @import("kore");
const chez = @import("chez");

const params_mod = chez.engine.params;
const PARAM_COUNT = params_mod.PARAM_COUNT;
const Params = params_mod.Params;
const ParamsF64 = params_mod.ParamsF64;
const dataset = @import("tuner/dataset.zig");
const mse = @import("tuner/mse.zig");
const spsa_mod = @import("tuner/spsa.zig");
const codegen = @import("tuner/codegen.zig");

const Args = struct {
    dataset: []const u8,
    output_path: ?[]const u8,
    max_positions: ?usize,
    threads: ?usize,
    iterations: ?usize,
    batch_size: ?usize,
    a: ?f64,
    big_a: ?f64,
    alpha: ?f64,
    c: ?f64,
    gamma: ?f64,
    k: ?f64,
    k_only: ?bool,
    checkpoint_interval: ?usize,
    calibrate_n: ?usize,
    skip_calibrate: ?bool,
    skip_k_tune: ?bool,
    no_early_stop: ?bool,
};

pub fn main(init: std.process.Init.Minimal) !void {
    const allocator = std.heap.page_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{ .environ = .empty });
    const io = threaded.io();

    const arg_parser = try kore.args.declarative.Parser(Args);

    var args_iter = if (@import("builtin").os.tag == .windows)
        try init.args.iterateAllocator(std.heap.page_allocator)
    else
        init.args.iterate();
    const args = try arg_parser.parse(&args_iter);

    const output_path: []const u8 = args.output_path orelse "src/engine/params.zig";
    const max_positions: usize = args.max_positions orelse 5_000_000;
    const calibrate_n: usize = args.calibrate_n orelse 50;
    var cfg = spsa_mod.SpsaConfig{};

    if (args.a) |a| cfg.a = a;
    if (args.big_a) |big_a| cfg.big_a = big_a;
    if (args.alpha) |alpha| cfg.alpha = alpha;
    if (args.c) |c| cfg.c = c;
    if (args.gamma) |gamma| cfg.gamma = gamma;
    if (args.iterations) |iterations| cfg.iterations = iterations;
    if (args.batch_size) |batch_size| cfg.batch_size = batch_size;
    if (args.checkpoint_interval) |checkpoint_interval| cfg.checkpoint_interval = checkpoint_interval;
    if (args.threads) |num_threads| cfg.num_threads = num_threads;
    if (args.no_early_stop != null) cfg.early_stop = false;

    const dataset_path = args.dataset;

    // Load dataset
    std.debug.print("tune: loading dataset from {s}\n", .{dataset_path});
    const positions = try dataset.load(io, allocator, dataset_path, max_positions);
    std.debug.print("tune: {d} positions loaded\n", .{positions.len});
    if (positions.len == 0) {
        std.debug.print("tune: no positions loaded: check dataset path and format\n", .{});
        return error.EmptyDataset;
    }

    // Initialise the float parameter vector from default_params
    var floats: [PARAM_COUNT]f64 = undefined;
    params_mod.default_params.toFloats(&floats);

    // K-tuning (ternary search over K in [0.5, 3.0])
    const k: f64 = if (args.k) |k| blk: {
        std.debug.print("tune: using supplied K = {d:.4}\n", .{k});
        break :blk k;
    } else if (args.skip_k_tune) |_| blk: {
        std.debug.print("tune: skipping K-tune, using K = 1.0\n", .{});
        break :blk 1.0;
    } else blk: {
        std.debug.print("tune: tuning K...\n", .{});
        const tuned_k = try tuneK(positions, &floats, cfg.num_threads, allocator);
        std.debug.print("tune: K = {d:.4}\n", .{tuned_k});
        break :blk tuned_k;
    };

    if (args.k_only) |_| {
        std.debug.print("tune: --k-only done. K = {d:.4}\n", .{k});
        return;
    }

    // Calibration pass: estimate and auto-apply the `a` parameter
    if (args.skip_calibrate == null) {
        const suggested_a = try runCalibration(positions, &floats, k, cfg, calibrate_n, allocator);
        if (args.a == null) {
            cfg.a = suggested_a;
            std.debug.print("tune: auto-applied calibrated a = {d:.1}\n", .{suggested_a});
        }
    }

    // SPSA optimisation
    try spsa_mod.run(io, positions, &floats, k, cfg, output_path, allocator);

    // Write final output
    const final_params = Params.fromFloats(&floats);
    try codegen.write(io, allocator, &final_params, output_path);
    std.debug.print("tune: final params written to {s}\n", .{output_path});
}

// K-tuning: ternary search over [0.5, 3.0]
// K normalises the engine's internal unit scale to the [0,1] WDL sigmoid. It
// is tuned once before SPSA and frozen for the entire optimisation run.
fn tuneK(
    positions: []const dataset.Position,
    floats: []const f64,
    num_threads: usize,
    allocator: std.mem.Allocator,
) !f64 {
    const initial_params = ParamsF64.fromFloats(floats);

    var lo: f64 = 0.5;
    var hi: f64 = 3.0;

    // 50 iterations of ternary search ≈ 100 full-dataset MSE evaluations.
    for (0..50) |_| {
        const m1 = lo + (hi - lo) / 3.0;
        const m2 = hi - (hi - lo) / 3.0;

        const mse1 = try mse.compute(positions, &initial_params, m1, num_threads, allocator);
        const mse2 = try mse.compute(positions, &initial_params, m2, num_threads, allocator);

        if (mse1 < mse2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }

    return (lo + hi) / 2.0;
}

// Calibration pass: estimate a good `a` value
// Runs `calibrate_n` SPSA iterations on the current float vector without
// updating it, collecting |g_hat| samples. Returns a suggested `a` value for a
// desired first-step size of 2.0 float units (a conservative start that avoids
// over-shooting in early iterations while remaining responsive).
fn runCalibration(
    positions: []const dataset.Position,
    floats: []f64,
    k: f64,
    cfg: spsa_mod.SpsaConfig,
    calibrate_n: usize,
    allocator: std.mem.Allocator,
) !f64 {
    std.debug.print("Calibration ({d} iters):\n", .{calibrate_n});

    // Temporary buffers for the calibration run.
    const batch_indices = try allocator.alloc(usize, cfg.batch_size);
    defer allocator.free(batch_indices);
    const delta = try allocator.alloc(f64, PARAM_COUNT);
    defer allocator.free(delta);
    const floats_plus = try allocator.alloc(f64, PARAM_COUNT);
    defer allocator.free(floats_plus);
    const floats_minus = try allocator.alloc(f64, PARAM_COUNT);
    defer allocator.free(floats_minus);

    const seed: u64 = 0xdeadbeef_cafeface; // fixed seed for reproducibility
    var rng = std.Random.DefaultPrng.init(seed);

    var sum_abs_g_hat: f64 = 0.0;

    for (1..calibrate_n + 1) |t| {
        const tf: f64 = @floatFromInt(t);
        const c_t = cfg.c / std.math.pow(f64, tf, cfg.gamma);

        for (batch_indices) |*idx| {
            idx.* = rng.random().intRangeLessThan(usize, 0, positions.len);
        }
        for (delta) |*d| {
            d.* = if (rng.random().boolean()) 1.0 else -1.0;
        }
        @memcpy(floats_plus, floats);
        @memcpy(floats_minus, floats);
        for (0..PARAM_COUNT) |i| {
            const pert = c_t * spsa_mod.c_scales[i] * delta[i];
            floats_plus[i] += pert;
            floats_minus[i] -= pert;
        }

        const params_plus = ParamsF64.fromFloats(floats_plus);
        const params_minus = ParamsF64.fromFloats(floats_minus);

        const mse_plus = try mse.computeIndexed(positions, batch_indices, &params_plus, k, cfg.num_threads, allocator);
        const mse_minus = try mse.computeIndexed(positions, batch_indices, &params_minus, k, cfg.num_threads, allocator);

        const g_hat = (mse_plus - mse_minus) / (2.0 * c_t);
        sum_abs_g_hat += @abs(g_hat);
    }

    const avg_abs_g_hat = sum_abs_g_hat / @as(f64, @floatFromInt(calibrate_n));

    // a_1 = a / (1 + A)^alpha (step size at t=1)
    const a_1 = cfg.a / std.math.pow(f64, 1.0 + cfg.big_a, cfg.alpha);
    const current_avg_step = a_1 * avg_abs_g_hat;

    // suggested_a = desired_step * (A+1)^alpha / avg_g_hat
    const desired_step: f64 = 2.0;
    const suggested_a = if (avg_abs_g_hat > 0.0)
        desired_step * std.math.pow(f64, cfg.big_a + 1.0, cfg.alpha) / avg_abs_g_hat
    else
        cfg.a;

    std.debug.print(
        \\  avg |g_hat|    = {d:.6}
        \\  current a      = {d:.1}  ->  avg step = {d:.5}   (a_1 * avg_g_hat, where a_1 = a/(A+1)^alpha)
        \\  suggested a    = {d:.1}  for avg step = {d:.1}
        \\  (auto-applied unless --a is set; use --skip-calibrate to suppress)
        \\
    , .{ avg_abs_g_hat, cfg.a, current_avg_step, suggested_a, desired_step });

    return suggested_a;
}

fn cliError(msg: []const u8) error{CliError} {
    std.debug.print("tune: {s}\n", .{msg});
    return error.CliError;
}
