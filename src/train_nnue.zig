const builtin = @import("builtin");
const std = @import("std");
const kore = @import("kore");
const ml = kore.ml;
const gpu = ml.gpu;

const model_mod = @import("trainer/model.zig");
const NnueModel = model_mod.NnueModel;
const dataloader_mod = @import("trainer/dataloader.zig");
const DataLoader = dataloader_mod.DataLoader;
const nnue_export = @import("trainer/export.zig");

const Args = struct {
    data: []const u8,
    epochs: ?u32,
    batch_size: ?usize,
    lr: ?f32,
    lambda: ?f32,
    checkpoint: ?[]const u8,
    @"export": ?[]const u8,
    checkpoint_interval: ?u32,
};

const default_epochs: u32 = 100;
const default_batch_size: usize = 16384;
const default_lr: f32 = 0.001;
const default_lambda: f32 = 1.0;
const default_lr_min: f32 = 0.0001;
const default_checkpoint_interval: u32 = 5;
const warmup_epochs: u32 = 1;
const grad_clip_norm: f32 = 1.0;
const sigmoid_scale: f32 = 1.0 / 400.0;

pub fn main(init: std.process.Init.Minimal) !void {
    const allocator = std.heap.page_allocator;
    const arg_parser = try kore.args.declarative.Parser(Args);

    var args_iter = if (builtin.os.tag == .windows)
        try init.args.iterateAllocator(allocator)
    else
        init.args.iterate();
    const args = try arg_parser.parse(&args_iter);

    const epochs = args.epochs orelse default_epochs;
    const batch_size = args.batch_size orelse default_batch_size;
    const lr = args.lr orelse default_lr;
    const lambda = args.lambda orelse default_lambda;
    const export_path = args.@"export" orelse "output.nnue";
    const checkpoint_interval = args.checkpoint_interval orelse default_checkpoint_interval;

    var single_threaded: std.Io.Threaded = .init_single_threaded;
    const io = single_threaded.io();
    var stderr_buf: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buf);
    const stderr: *std.Io.Writer = &stderr_writer.interface;

    try stderr.print("NNUE Trainer\n", .{});
    try stderr.print("  data: {s}\n", .{args.data});
    try stderr.print("  epochs: {d}\n", .{epochs});
    try stderr.print("  batch_size: {d}\n", .{batch_size});
    try stderr.print("  lr: {d:.6}\n", .{lr});
    try stderr.print("  lambda: {d:.2}\n", .{lambda});
    try stderr.flush();

    // GPU setup
    var ctx = try gpu.Context.init();
    defer ctx.deinit();
    var ops = try gpu.ops.init(&ctx);
    defer ops.deinit();
    var graph = ml.Graph.init(allocator, &ctx, &ops);
    defer graph.deinit();

    try stderr.print("GPU initialized\n", .{});
    try stderr.flush();

    // Model
    var model = try NnueModel.init(allocator, &ctx);
    defer model.deinit();

    var params_list = try model.parameters();
    defer params_list.deinit(allocator);
    const params = params_list.items;

    try stderr.print("Model initialized ({d} parameter tensors)\n", .{params.len});
    try stderr.flush();

    // Optimizer
    var adam = try ml.Adam.init(allocator, &ctx, &ops, params, .{ .lr = lr });
    defer adam.deinit();

    // Data loader
    var loader = try DataLoader.init(allocator, &ctx, args.data, batch_size, lambda);
    defer loader.deinit();

    try stderr.print("Data: {d} records ({d} train, {d} val)\n", .{
        loader.total_records,
        loader.train_records,
        loader.val_records,
    });
    try stderr.print("Batches per epoch: {d} train, {d} val\n", .{
        loader.numTrainBatches(),
        loader.numValBatches(),
    });
    try stderr.flush();

    // Loss function
    var loss_fn: ml.MseLoss = .{ .use_sigmoid = true, .sigmoid_scale = sigmoid_scale };

    // Resume from checkpoint
    var start_epoch: u32 = 0;
    var best_val_loss: f32 = std.math.inf(f32);

    if (args.checkpoint) |ckpt_path| {
        const named_params = try model.namedParameters();
        defer allocator.free(named_params);
        const meta = try ml.serialize.load(allocator, &ctx, ckpt_path, named_params, &adam);
        start_epoch = meta.epoch + 1;
        best_val_loss = meta.best_val_loss;
        adam.step_count = meta.adam_step;
        adam.config.lr = meta.learning_rate;
        try stderr.print("Resumed from checkpoint: epoch {d}, val_loss {d:.6}\n", .{ meta.epoch, meta.best_val_loss });
        try stderr.flush();
    }

    // Training loop
    const train_batches = loader.numTrainBatches();
    const val_batches = loader.numValBatches();
    const total_steps = epochs * @as(u32, @intCast(train_batches));
    const warmup_steps = warmup_epochs * @as(u32, @intCast(train_batches));

    for (start_epoch..epochs) |epoch_usize| {
        const epoch: u32 = @intCast(epoch_usize);

        loader.shuffleTraining(epoch);

        var epoch_loss: f64 = 0;
        var batch_count: usize = 0;

        for (0..train_batches) |batch_idx| {
            const global_step = epoch * @as(u32, @intCast(train_batches)) + @as(u32, @intCast(batch_idx));

            // LR schedule: linear warmup then cosine decay
            adam.config.lr = computeLR(global_step, total_steps, warmup_steps, lr, default_lr_min);

            try graph.zeroGrads(params);

            var batch = try loader.getTrainBatch(batch_idx);
            defer batch.release();

            const output = try model.forward(batch, &graph);
            const loss = try loss_fn.forward(output, batch.targets, &graph);

            try graph.backward(output);
            _ = try graph.clipGradNorm(params, grad_clip_norm);
            try adam.step(params);

            graph.reset();

            epoch_loss += loss;
            batch_count += 1;

            if (batch_idx % 100 == 0) {
                try stderr.print("\r  epoch {d}/{d} batch {d}/{d} loss={d:.6} lr={d:.6}", .{
                    epoch + 1,
                    epochs,
                    batch_idx,
                    train_batches,
                    loss,
                    adam.config.lr,
                });
                try stderr.flush();
            }
        }

        const avg_train_loss = epoch_loss / @as(f64, @floatFromInt(batch_count));

        // Validation
        var val_loss: f64 = 0;
        var val_count: usize = 0;
        for (0..val_batches) |batch_idx| {
            var batch = try loader.getValBatch(batch_idx);
            defer batch.release();

            const output = try model.forward(batch, &graph);
            const loss = try loss_fn.forward(output, batch.targets, &graph);
            graph.reset();

            val_loss += loss;
            val_count += 1;
        }
        const avg_val_loss = if (val_count > 0)
            val_loss / @as(f64, @floatFromInt(val_count))
        else
            0;

        if (avg_val_loss < best_val_loss) best_val_loss = @floatCast(avg_val_loss);

        try stderr.print("\repoch {d}/{d} train_loss={d:.6} val_loss={d:.6} lr={d:.6}            \n", .{
            epoch + 1,
            epochs,
            avg_train_loss,
            avg_val_loss,
            adam.config.lr,
        });
        try stderr.flush();

        // Checkpoint
        if ((epoch + 1) % checkpoint_interval == 0 or epoch + 1 == epochs) {
            const named_params = try model.namedParameters();
            defer allocator.free(named_params);

            var ckpt_name_buf: [64]u8 = undefined;
            var ckpt_w = std.Io.Writer.fixed(&ckpt_name_buf);
            try ckpt_w.print("checkpoint_epoch{d}.ktml", .{epoch + 1});
            const ckpt_name = ckpt_w.buffered();

            try ml.serialize.save(allocator, &ctx, ckpt_name, named_params, &adam, .{
                .epoch = epoch,
                .step = adam.step_count,
                .learning_rate = adam.config.lr,
                .best_val_loss = @floatCast(best_val_loss),
                .adam_step = adam.step_count,
            });
            try stderr.print("  saved {s}\n", .{ckpt_name});
            try stderr.flush();
        }
    }

    // Export quantized .nnue
    try nnue_export.exportNnue(allocator, &ctx, &model, export_path);
    try stderr.print("Exported {s}\n", .{export_path});
    try stderr.flush();
}

fn computeLR(step: u32, total_steps: u32, warmup_steps: u32, lr_max: f32, lr_min: f32) f32 {
    if (step < warmup_steps) {
        // Linear warmup
        return lr_min + (lr_max - lr_min) * @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(warmup_steps));
    }
    // Cosine decay
    const decay_steps = total_steps - warmup_steps;
    const progress = @as(f32, @floatFromInt(step - warmup_steps)) / @as(f32, @floatFromInt(decay_steps));
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + @cos(std.math.pi * progress));
}
