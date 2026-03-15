// src/tuner/mse.zig
//
// Parallel MSE computation for Texel tuning.
//
// MSE = (1/N) * sum_i (sigmoid(eval_i, K) - result_i)^2
//
// All evaluations use evaluateWithParamsF64 so that gradient signal is never
// lost to i16 rounding — see pitfall #1 in the tuning plan.

const std = @import("std");
const chez = @import("chez");
const evaluation = chez.engine.evaluation;
const params_mod = chez.engine.params;
const ParamsF64 = params_mod.ParamsF64;

const dataset = @import("dataset.zig");
const Position = dataset.Position;

// ==============================================================================
// Sigmoid
// ==============================================================================

// sigma(eval / K) in [0, 1], where K normalises centipawn units to the WDL
// scale. Division by 400 is the conventional centipawn normaliser.
inline fn sigmoid(eval: f64, k: f64) f64 {
    return 1.0 / (1.0 + @exp(-k * eval / 400.0));
}

// ==============================================================================
// Thread context
// ==============================================================================

const ThreadCtx = struct {
    // Slice of positions this thread is responsible for.
    positions: []const Position,
    params_f64: *const ParamsF64,
    k: f64,
    partial_sum: f64,
};

fn computeThread(ctx: *ThreadCtx) void {
    var sum: f64 = 0.0;
    for (ctx.positions) |*pos| {
        const eval = evaluation.evaluateWithParamsF64(&pos.state, ctx.params_f64);
        const predicted = sigmoid(eval, ctx.k);
        const diff = predicted - @as(f64, pos.result);
        sum += diff * diff;
    }
    ctx.partial_sum = sum;
}

// ==============================================================================
// Full-dataset MSE (used for K-tuning and periodic logging every 100 iters)
// ==============================================================================

pub fn compute(
    positions: []const Position,
    params_f64: *const ParamsF64,
    k: f64,
    num_threads: usize,
    allocator: std.mem.Allocator,
) !f64 {
    if (positions.len == 0) return 0.0;

    const actual_threads = @min(num_threads, positions.len);
    const contexts = try allocator.alloc(ThreadCtx, actual_threads);
    defer allocator.free(contexts);
    const threads = try allocator.alloc(std.Thread, actual_threads - 1);
    defer allocator.free(threads);

    // Distribute positions in contiguous chunks to maximise cache locality.
    // The last chunk absorbs any remainder so no positions are skipped.
    const chunk = positions.len / actual_threads;
    for (0..actual_threads) |i| {
        const start = i * chunk;
        const end = if (i + 1 == actual_threads) positions.len else start + chunk;
        contexts[i] = ThreadCtx{
            .positions = positions[start..end],
            .params_f64 = params_f64,
            .k = k,
            .partial_sum = 0.0,
        };
    }

    // Spawn worker threads for all but the last context.
    for (0..actual_threads - 1) |i| {
        threads[i] = try std.Thread.spawn(.{}, computeThread, .{&contexts[i]});
    }
    // Main thread handles the last context.
    computeThread(&contexts[actual_threads - 1]);

    for (threads) |t| t.join();

    var total: f64 = 0.0;
    for (contexts) |*ctx| total += ctx.partial_sum;
    return total / @as(f64, @floatFromInt(positions.len));
}

// ==============================================================================
// Batch (indexed) MSE — used inside the SPSA hot loop
// ==============================================================================
//
// Takes a slice of indices into the full positions array. Avoids copying
// Position structs (which contain a full State) for each batch.

const IndexedThreadCtx = struct {
    positions: []const Position,
    // Sub-slice of the batch indices assigned to this thread.
    indices: []const usize,
    params_f64: *const ParamsF64,
    k: f64,
    partial_sum: f64,
};

fn computeIndexedThread(ctx: *IndexedThreadCtx) void {
    var sum: f64 = 0.0;
    for (ctx.indices) |idx| {
        const pos = &ctx.positions[idx];
        const eval = evaluation.evaluateWithParamsF64(&pos.state, ctx.params_f64);
        const predicted = sigmoid(eval, ctx.k);
        const diff = predicted - @as(f64, pos.result);
        sum += diff * diff;
    }
    ctx.partial_sum = sum;
}

pub fn computeIndexed(
    positions: []const Position,
    indices: []const usize,
    params_f64: *const ParamsF64,
    k: f64,
    num_threads: usize,
    allocator: std.mem.Allocator,
) !f64 {
    if (indices.len == 0) return 0.0;

    const actual_threads = @min(num_threads, indices.len);
    const contexts = try allocator.alloc(IndexedThreadCtx, actual_threads);
    defer allocator.free(contexts);
    const threads = try allocator.alloc(std.Thread, actual_threads - 1);
    defer allocator.free(threads);

    const chunk = indices.len / actual_threads;
    for (0..actual_threads) |i| {
        const start = i * chunk;
        const end = if (i + 1 == actual_threads) indices.len else start + chunk;
        contexts[i] = IndexedThreadCtx{
            .positions = positions,
            .indices = indices[start..end],
            .params_f64 = params_f64,
            .k = k,
            .partial_sum = 0.0,
        };
    }

    for (0..actual_threads - 1) |i| {
        threads[i] = try std.Thread.spawn(.{}, computeIndexedThread, .{&contexts[i]});
    }
    computeIndexedThread(&contexts[actual_threads - 1]);

    for (threads) |t| t.join();

    var total: f64 = 0.0;
    for (contexts) |*ctx| total += ctx.partial_sum;
    return total / @as(f64, @floatFromInt(indices.len));
}
