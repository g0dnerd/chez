const std = @import("std");
const chez = @import("chez");
const kore = @import("kore");
const ml = kore.ml;
const gpu = ml.gpu;
const Context = gpu.Context;

const nnue = chez.engine.nnue;
const Network = nnue.Network;
const NnueModel = @import("model.zig").NnueModel;

fn quantizeI16(val: f32, scale: f32) i16 {
    const scaled = @round(val * scale);
    return @intFromFloat(std.math.clamp(scaled, -32768.0, 32767.0));
}

fn quantizeI8(val: f32, scale: f32) i8 {
    const scaled = @round(val * scale);
    return @intFromFloat(std.math.clamp(scaled, -128.0, 127.0));
}

fn quantizeI32(val: f32, scale: f32) i32 {
    const scaled = @round(val * scale);
    return @intFromFloat(std.math.clamp(scaled, -2147483648.0, 2147483647.0));
}

// Export float32 model to quantized .nnue file.
// Quantization scheme:
//   FT weights/biases: ×127 → i16
//   Hidden weights: ×64 → i8
//   Hidden biases: ×(127×64) → i32
//   Output weights: ×64 → i16
//   Output bias: ×(127×64) → i32
pub noinline fn exportNnue(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    model: *NnueModel,
    path: []const u8,
) !void {
    var params_list = try model.parameters();
    defer params_list.deinit(allocator);
    const params = params_list.items;
    // Order: ft.weight, ft.bias, fc1.weight, fc1.bias, fc2.weight, fc2.bias, out.weight, out.bias
    std.debug.assert(params.len == 8);

    const net = try allocator.create(Network);
    defer allocator.destroy(net);

    // FT weights: [40960, 512] f32 → [40960][512] i16, scale = 127
    {
        const n = nnue.num_features * nnue.ft_out;
        const buf = try allocator.alloc(f32, n);
        defer allocator.free(buf);
        try params[0].storage.gpu.buffer.download(ctx, buf);
        for (0..nnue.num_features) |i| {
            for (0..nnue.ft_out) |j| {
                net.ft_weights[i][j] = quantizeI16(buf[i * nnue.ft_out + j], 127.0);
            }
        }
    }

    // FT biases: [512] f32 → [512] i16, scale = 127
    {
        const n = nnue.ft_out;
        var buf: [n]f32 = undefined;
        try params[1].storage.gpu.buffer.download(ctx, &buf);
        for (0..n) |i| {
            net.ft_biases[i] = quantizeI16(buf[i], 127.0);
        }
    }

    // FC1 weights: [1024, 32] f32 → [32][1024] i8 (output-major), scale = 64
    {
        const n = nnue.fc1_in * nnue.fc1_out;
        var buf: [n]f32 = undefined;
        try params[2].storage.gpu.buffer.download(ctx, &buf);
        for (0..nnue.fc1_in) |i| {
            for (0..nnue.fc1_out) |j| {
                net.fc1_weights[j][i] = quantizeI8(buf[i * nnue.fc1_out + j], 64.0);
            }
        }
    }

    // FC1 biases: [32] f32 → [32] i32, scale = 127*64
    {
        const n = nnue.fc1_out;
        var buf: [n]f32 = undefined;
        try params[3].storage.gpu.buffer.download(ctx, &buf);
        for (0..n) |i| {
            net.fc1_biases[i] = quantizeI32(buf[i], 127.0 * 64.0);
        }
    }

    // FC2 weights: [32, 32] f32 → [32][32] i8 (output-major), scale = 64
    {
        const n = nnue.fc2_in * nnue.fc2_out;
        var buf: [n]f32 = undefined;
        try params[4].storage.gpu.buffer.download(ctx, &buf);
        for (0..nnue.fc2_in) |i| {
            for (0..nnue.fc2_out) |j| {
                net.fc2_weights[j][i] = quantizeI8(buf[i * nnue.fc2_out + j], 64.0);
            }
        }
    }

    // FC2 biases: [32] f32 → [32] i32, scale = 127*64
    {
        const n = nnue.fc2_out;
        var buf: [n]f32 = undefined;
        try params[5].storage.gpu.buffer.download(ctx, &buf);
        for (0..n) |i| {
            net.fc2_biases[i] = quantizeI32(buf[i], 127.0 * 64.0);
        }
    }

    // Output weights: [32] f32 → [32] i16, scale = 64
    {
        const n = nnue.fc2_out;
        var buf: [n]f32 = undefined;
        try params[6].storage.gpu.buffer.download(ctx, &buf);
        for (0..n) |i| {
            net.output_weights[i] = quantizeI16(buf[i], 64.0);
        }
    }

    // Output bias: [1] f32 → i32, scale = 127*64
    {
        var buf: [1]f32 = undefined;
        try params[7].storage.gpu.buffer.download(ctx, &buf);
        net.output_bias = quantizeI32(buf[0], 127.0 * 64.0);
    }

    // Write to file
    var single_threaded: std.Io.Threaded = .init_single_threaded;
    const io = single_threaded.io();
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var writer_buf: [4096]u8 = undefined;
    var w = file.writer(io, &writer_buf);
    var writer = &w.interface;

    try net.writeToWriter(writer);
    try writer.flush();
}
