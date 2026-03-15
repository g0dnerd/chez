// src/tuner/spsa.zig
//
// SPSA (Simultaneous Perturbation Stochastic Approximation) optimiser for
// Texel tuning. Operates entirely in float space ([]f64) to avoid i16
// quantisation noise — see pitfall #1 in the tuning plan.
//
// Key design choices:
//   - Perturbations are applied to the raw float vector; ParamsF64 is
//     reconstructed from it on each MSE call. Params.fromFloats (which rounds
//     to i16) is only called at checkpoint writes.
//   - Frozen indices are snapped back after every gradient step, not just at
//     checkpoint writes (pitfall #2).
//   - Batch MSE uses index-based access to avoid copying Position structs.

const std = @import("std");
const chez = @import("chez");
const params_mod = chez.engine.params;
const PARAM_COUNT = params_mod.PARAM_COUNT;
const Params = params_mod.Params;
const ParamsF64 = params_mod.ParamsF64;

const dataset = @import("dataset.zig");
const Position = dataset.Position;
const mse = @import("mse.zig");
const codegen = @import("codegen.zig");

// ==============================================================================
// SPSA configuration
// ==============================================================================

pub const SpsaConfig = struct {
    a: f64 = 10.0,
    big_a: f64 = 100.0,
    alpha: f64 = 0.602,
    c: f64 = 1.0,
    gamma: f64 = 0.101,
    iterations: usize = 500_000,
    batch_size: usize = 16_384,
    checkpoint_interval: usize = 1_000,
    num_threads: usize = 8,
};

// ==============================================================================
// Frozen parameter indices
// ==============================================================================
//
// These indices (into the flat f64 parameter vector) must stay at their default
// values throughout tuning. They are snapped back after every gradient step.
//
// Flat layout (from params.zig):
//   piece_values [6]Score     → indices   0..11
//   passed_pawn_bonus [8]Score → indices  12..27
//   mobility_bonus [4][28]Score → indices 28..251
//   17 named scalars           → indices 252..285
//   king_proximity_passer      → index   286
//   pst [6][64]Score           → indices 287..1054
//     pst[0] (pawns) sq 0..7   → indices 287..302
//     pst[0] (pawns) sq 56..63 → indices 399..414

const FrozenEntry = struct { index: usize, value: f64 };

// Computed at comptime so the freeze loop has no branches.
const frozen_entries: [38]FrozenEntry = blk: {
    // 2 (king pv) + 2 (ppb rank0) + 2 (ppb rank7) + 16 (pst rank0) + 16 (pst rank7) = 38
    var entries: [38]FrozenEntry = undefined;
    var i: usize = 0;

    // King piece value (MG and EG frozen at 20000).
    entries[i] = .{ .index = 10, .value = 20000.0 };
    i += 1;
    entries[i] = .{ .index = 11, .value = 20000.0 };
    i += 1;

    // Passed pawn bonus rank 0 (MG and EG frozen at 0).
    entries[i] = .{ .index = 12, .value = 0.0 };
    i += 1;
    entries[i] = .{ .index = 13, .value = 0.0 };
    i += 1;

    // Passed pawn bonus rank 7 (MG and EG frozen at 0).
    entries[i] = .{ .index = 26, .value = 0.0 };
    i += 1;
    entries[i] = .{ .index = 27, .value = 0.0 };
    i += 1;

    // Pawn PST rank 0 (squares 0..7, 8 squares × 2 floats = 16 entries).
    // Base index for pst[0][0] is 287.
    for (0..8) |sq| {
        entries[i] = .{ .index = 287 + sq * 2, .value = 0.0 };
        i += 1;
        entries[i] = .{ .index = 287 + sq * 2 + 1, .value = 0.0 };
        i += 1;
    }

    // Pawn PST rank 7 (squares 56..63, 8 squares × 2 floats = 16 entries).
    // pst[0][56] = index 287 + 56*2 = 399.
    for (0..8) |sq| {
        entries[i] = .{ .index = 399 + sq * 2, .value = 0.0 };
        i += 1;
        entries[i] = .{ .index = 399 + sq * 2 + 1, .value = 0.0 };
        i += 1;
    }

    break :blk entries;
};

// Snap all frozen indices back to their default values after a gradient step.
fn restoreFrozen(floats: []f64) void {
    for (frozen_entries) |fe| {
        floats[fe.index] = fe.value;
    }
}

// ==============================================================================
// SPSA run
// ==============================================================================

pub fn run(
    io: std.Io,
    positions: []const Position,
    floats: []f64, // modified in-place; caller owns and reads back final values
    k: f64,
    cfg: SpsaConfig,
    output_path: []const u8,
    allocator: std.mem.Allocator,
) !void {
    // Pre-allocate batch indices once; re-fill each iteration.
    const batch_indices = try allocator.alloc(usize, cfg.batch_size);
    defer allocator.free(batch_indices);

    // Delta vector (Rademacher ±1). Allocated once, re-used each iteration.
    const delta = try allocator.alloc(f64, PARAM_COUNT);
    defer allocator.free(delta);

    // Perturbed float vectors for params_plus and params_minus.
    const floats_plus = try allocator.alloc(f64, PARAM_COUNT);
    defer allocator.free(floats_plus);
    const floats_minus = try allocator.alloc(f64, PARAM_COUNT);
    defer allocator.free(floats_minus);

    // Seed the RNG from system entropy.
    var seed: u64 = undefined;
    std.Io.random(io, std.mem.asBytes(&seed));
    var rng = std.Random.DefaultPrng.init(seed);

    std.debug.print("SPSA: starting {d} iterations, batch={d}, threads={d}\n", .{
        cfg.iterations,
        cfg.batch_size,
        cfg.num_threads,
    });

    for (1..cfg.iterations + 1) |t| {
        const tf: f64 = @floatFromInt(t);

        // Step-size schedule: a_t = a / (t + A)^alpha
        const a_t = cfg.a / std.math.pow(f64, tf + cfg.big_a, cfg.alpha);

        // Perturbation size schedule: c_t = c / t^gamma
        const c_t = cfg.c / std.math.pow(f64, tf, cfg.gamma);

        // Sample a random batch (with replacement).
        for (batch_indices) |*idx| {
            idx.* = rng.random().intRangeLessThan(usize, 0, positions.len);
        }

        // Generate a Rademacher ±1 delta vector.
        for (delta) |*d| {
            d.* = if (rng.random().boolean()) 1.0 else -1.0;
        }

        // Build params_plus and params_minus by applying ±c_t * delta[i].
        @memcpy(floats_plus, floats);
        @memcpy(floats_minus, floats);
        for (0..PARAM_COUNT) |i| {
            floats_plus[i] += c_t * delta[i];
            floats_minus[i] -= c_t * delta[i];
        }

        // Evaluate both perturbations on the batch. ParamsF64 is constructed
        // from the float vector — no i16 rounding at this stage.
        const params_plus = ParamsF64.fromFloats(floats_plus);
        const params_minus = ParamsF64.fromFloats(floats_minus);

        const mse_plus = try mse.computeIndexed(positions, batch_indices, &params_plus, k, cfg.num_threads, allocator);
        const mse_minus = try mse.computeIndexed(positions, batch_indices, &params_minus, k, cfg.num_threads, allocator);

        // SPSA gradient estimate: g_hat[i] = (MSE+ - MSE-) / (2 * c_t * delta[i]).
        // For ±1 delta, 1/delta[i] == delta[i], so:
        //   floats[i] -= a_t * g_hat * delta[i]
        //              = a_t * (MSE+ - MSE-) / (2 * c_t) * delta[i]
        const g_hat_scalar = (mse_plus - mse_minus) / (2.0 * c_t);
        for (0..PARAM_COUNT) |i| {
            floats[i] -= a_t * g_hat_scalar * delta[i];
        }

        // Snap frozen indices back. This must happen every iteration, not just
        // at checkpoints, so that the next batch sees correct frozen values.
        restoreFrozen(floats);

        // Periodic full-dataset MSE logging every 100 iterations.
        if (t % 100 == 0) {
            const current_params = ParamsF64.fromFloats(floats);
            const full_mse = try mse.compute(positions, &current_params, k, cfg.num_threads, allocator);
            std.debug.print("iter {d:>7}  full_mse={d:.6}\n", .{ t, full_mse });
        }

        // Checkpoint: round to i16 and write params file.
        // Rounding only happens here and at final output, never in the hot loop.
        if (t % cfg.checkpoint_interval == 0) {
            const p = Params.fromFloats(floats);
            codegen.write(io, allocator, &p, output_path) catch |err| {
                std.debug.print("SPSA: checkpoint write failed: {s}\n", .{@errorName(err)});
            };
            std.debug.print("SPSA: checkpoint written at iter {d}\n", .{t});
        }
    }

    std.debug.print("SPSA: done.\n", .{});
}
