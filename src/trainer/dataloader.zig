const std = @import("std");
const chez = @import("chez");
const kore = @import("kore");
const ml = kore.ml;
const gpu = ml.gpu;
const Buffer = gpu.Buffer;
const Context = gpu.Context;

const engine = chez.engine;
const nnue = engine.nnue;
const serde = @import("../selfplay/serde.zig");
const Batch = @import("model.zig").Batch;

const record_size = 35; // 32 position + 2 score + 1 WDL
const max_active = nnue.max_active_features; // 30

fn uploadU32(ctx: *const Context, data: []const u32) Context.Error!Buffer {
    const cl = Context.cl;
    var err: cl.cl_int = undefined;
    const byte_size = data.len * @sizeOf(u32);
    const mem = cl.clCreateBuffer(
        ctx.context,
        cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
        byte_size,
        @ptrCast(@constCast(data.ptr)),
        &err,
    );
    try Context.check(err);
    return .{ .mem = mem, .len = data.len };
}

// Per-worker slice of a batch. Each worker fills a contiguous range of batch
// positions [start, end); the output buffers are written at disjoint offsets
// (base = b * max_active), so no synchronization is needed between workers.
const PrepareCtx = struct {
    loader: *DataLoader,
    record_indices: []const u32,
    start: usize,
    end: usize,
    err: ?anyerror = null,
};

pub const DataLoader = struct {
    data: []const u8,
    data_len: usize,
    total_records: usize,
    train_records: usize,
    val_start: usize,
    val_records: usize,
    batch_size: usize,
    lambda: f32,
    // Score->target sigmoid steepness (target = sigmoid(score * sigmoid_k)).
    // Must be calibrated to the dataset's score scale, not hardcoded.
    sigmoid_k: f32,
    ctx: *const Context,
    allocator: std.mem.Allocator,

    // Shuffled training record indices
    indices: []u32,

    // Reusable CPU staging buffers
    stm_idx_buf: []u32,
    opp_idx_buf: []u32,
    stm_na_buf: []u32,
    opp_na_buf: []u32,
    target_buf: []f32,

    // Worker pool for parallel feature extraction (CPU is the training
    // bottleneck; the per-record loop is embarrassingly parallel).
    num_threads: usize,
    prepare_threads: []std.Thread,
    prepare_ctxs: []PrepareCtx,

    file: std.Io.File,

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *const Context,
        path: []const u8,
        batch_size: usize,
        lambda: f32,
        num_threads: usize,
        sigmoid_k: f32,
    ) !DataLoader {
        var single_threaded: std.Io.Threaded = .init_single_threaded;
        const io = single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        const stat = try file.stat(io);
        const len: usize = @intCast(stat.size);

        if (len < record_size or len % record_size != 0)
            return error.InvalidDataFile;

        const total_records = len / record_size;
        const train_records = total_records * 98 / 100;
        const val_start = train_records;
        const val_records = total_records - train_records;

        const mapped = try std.posix.mmap(
            null,
            len,
            std.os.linux.PROT{ .READ = true },
            .{ .TYPE = .SHARED },
            file.handle,
            0,
        );

        const indices = try allocator.alloc(u32, train_records);
        for (0..train_records) |i| {
            indices[i] = @intCast(i);
        }

        const stm_idx_buf = try allocator.alloc(u32, batch_size * max_active);
        const opp_idx_buf = try allocator.alloc(u32, batch_size * max_active);
        const stm_na_buf = try allocator.alloc(u32, batch_size);
        const opp_na_buf = try allocator.alloc(u32, batch_size);
        const target_buf = try allocator.alloc(f32, batch_size);

        const resolved_threads = @max(1, num_threads);
        const prepare_threads = try allocator.alloc(std.Thread, resolved_threads);
        const prepare_ctxs = try allocator.alloc(PrepareCtx, resolved_threads);

        // Trigger the lazy Zobrist key init (via computeHash in decodePosition)
        // on the main thread so worker threads only ever read the immutable keys.
        if (total_records > 0) {
            var warm_buf: [32]u8 = undefined;
            @memcpy(&warm_buf, mapped[0..32]);
            _ = serde.decodePosition(&warm_buf) catch {};
        }

        return .{
            .data = mapped,
            .total_records = total_records,
            .data_len = len,
            .train_records = train_records,
            .val_start = val_start,
            .val_records = val_records,
            .batch_size = batch_size,
            .lambda = lambda,
            .sigmoid_k = sigmoid_k,
            .ctx = ctx,
            .allocator = allocator,
            .indices = indices,
            .stm_idx_buf = stm_idx_buf,
            .opp_idx_buf = opp_idx_buf,
            .stm_na_buf = stm_na_buf,
            .opp_na_buf = opp_na_buf,
            .target_buf = target_buf,
            .num_threads = resolved_threads,
            .prepare_threads = prepare_threads,
            .prepare_ctxs = prepare_ctxs,
            .file = file,
        };
    }

    pub fn shuffleTraining(self: *DataLoader, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();

        var i: usize = self.train_records;
        while (i > 1) {
            i -= 1;
            const j = rng.uintLessThan(usize, i + 1);
            const tmp = self.indices[i];
            self.indices[i] = self.indices[j];
            self.indices[j] = tmp;
        }
    }

    pub fn numTrainBatches(self: *const DataLoader) usize {
        return self.train_records / self.batch_size;
    }

    pub fn numValBatches(self: *const DataLoader) usize {
        return self.val_records / self.batch_size;
    }

    pub fn getTrainBatch(self: *DataLoader, batch_idx: usize) !Batch {
        const start = batch_idx * self.batch_size;
        const actual_size = @min(self.batch_size, self.train_records - start);
        return self.prepareBatch(self.indices[start..][0..actual_size]);
    }

    pub fn getValBatch(self: *DataLoader, batch_idx: usize) !Batch {
        const start = self.val_start + batch_idx * self.batch_size;
        const actual_size = @min(self.batch_size, self.total_records - start);

        // Validation uses sequential indices
        var seq_indices = try self.allocator.alloc(u32, actual_size);
        defer self.allocator.free(seq_indices);
        for (0..actual_size) |i| seq_indices[i] = @intCast(start + i);

        return self.prepareBatch(seq_indices);
    }

    fn prepareBatch(self: *DataLoader, record_indices: []const u32) !Batch {
        const bs = record_indices.len;
        @memset(self.stm_idx_buf[0 .. bs * max_active], 0);
        @memset(self.opp_idx_buf[0 .. bs * max_active], 0);

        // Fan the per-record feature extraction out across worker threads. Each
        // worker owns a contiguous range of batch positions; writes are disjoint
        // (indexed by position), so no locking is needed.
        const nthreads = @max(1, @min(self.num_threads, bs));
        if (nthreads == 1) {
            var ctx = PrepareCtx{ .loader = self, .record_indices = record_indices, .start = 0, .end = bs };
            prepareWorker(&ctx);
            if (ctx.err) |e| return e;
        } else {
            const chunk = bs / nthreads;
            for (0..nthreads) |i| {
                const start = i * chunk;
                const end = if (i + 1 == nthreads) bs else start + chunk;
                self.prepare_ctxs[i] = .{ .loader = self, .record_indices = record_indices, .start = start, .end = end };
            }
            // Workers handle all but the last chunk; the main thread runs the last.
            for (0..nthreads - 1) |i| {
                self.prepare_threads[i] = try std.Thread.spawn(.{}, prepareWorker, .{&self.prepare_ctxs[i]});
            }
            prepareWorker(&self.prepare_ctxs[nthreads - 1]);
            for (0..nthreads - 1) |i| self.prepare_threads[i].join();
            for (0..nthreads) |i| if (self.prepare_ctxs[i].err) |e| return e;
        }

        return .{
            .stm_indices = try uploadU32(self.ctx, self.stm_idx_buf[0 .. bs * max_active]),
            .stm_num_active = try uploadU32(self.ctx, self.stm_na_buf[0..bs]),
            .opp_indices = try uploadU32(self.ctx, self.opp_idx_buf[0 .. bs * max_active]),
            .opp_num_active = try uploadU32(self.ctx, self.opp_na_buf[0..bs]),
            .targets = try Buffer.upload(self.ctx, self.target_buf[0..bs]),
            .size = @intCast(bs),
        };
    }

    // Fills batch positions [ctx.start, ctx.end) of the shared staging buffers.
    // Runs on the main thread and on spawned workers; not a method so it can be
    // passed to std.Thread.spawn.
    fn prepareWorker(ctx: *PrepareCtx) void {
        const self = ctx.loader;
        for (ctx.start..ctx.end) |b| {
            const rec_idx = ctx.record_indices[b];
            const offset = @as(usize, rec_idx) * record_size;
            var pos_buf: [32]u8 = undefined;
            @memcpy(&pos_buf, self.data[offset..][0..32]);

            const state = serde.decodePosition(&pos_buf) catch |e| {
                ctx.err = e;
                return;
            };
            const stm = state.to_move;
            const opp: engine.Color = @intCast(~@as(u1, @intCast(stm)));

            // Feature extraction
            const stm_features = nnue.activeFeatures(&state, stm);
            const opp_features = nnue.activeFeatures(&state, opp);

            const base = b * max_active;
            for (0..stm_features.len) |i| {
                self.stm_idx_buf[base + i] = stm_features.features[i];
            }
            self.stm_na_buf[b] = stm_features.len;

            for (0..opp_features.len) |i| {
                self.opp_idx_buf[base + i] = opp_features.features[i];
            }
            self.opp_na_buf[b] = opp_features.len;

            // Target: blended label
            const score_bytes: *const [2]u8 = @ptrCast(self.data[offset + 32 ..][0..2]);
            const score: f32 = @floatFromInt(std.mem.readInt(i16, score_bytes, .little));
            const wdl_byte = self.data[offset + 34];

            // WDL from STM perspective: 0=white wins, 1=black wins, 2=draw
            const wdl_raw: f32 = switch (wdl_byte) {
                0 => 1.0, // white wins
                1 => 0.0, // black wins
                2 => 0.5, // draw
                else => 0.5,
            };
            // Flip if STM is black (score is already STM perspective)
            const wdl_value = if (stm == engine.Colors.black) 1.0 - wdl_raw else wdl_raw;

            const score_sigmoid = sigmoid(score * self.sigmoid_k);
            self.target_buf[b] = self.lambda * score_sigmoid + (1.0 - self.lambda) * wdl_value;
        }
    }

    pub fn deinit(self: *DataLoader) void {
        var single_threaded: std.Io.Threaded = .init_single_threaded;
        const io = single_threaded.io();

        std.posix.munmap(@alignCast(self.data));
        self.file.close(io);
        self.allocator.free(self.indices);
        self.allocator.free(self.stm_idx_buf);
        self.allocator.free(self.opp_idx_buf);
        self.allocator.free(self.stm_na_buf);
        self.allocator.free(self.opp_na_buf);
        self.allocator.free(self.target_buf);
        self.allocator.free(self.prepare_threads);
        self.allocator.free(self.prepare_ctxs);
    }
};

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}
