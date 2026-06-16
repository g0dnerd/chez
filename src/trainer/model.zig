const std = @import("std");
const chez = @import("chez");
const kore = @import("kore");
const ml = kore.ml;
const gpu = ml.gpu;
const Buffer = gpu.Buffer;
const Context = gpu.Context;
const Graph = ml.Graph;
const Tensor = ml.Tensor;
const Layer = ml.Layer;

const nnue = chez.engine.nnue;

pub const Batch = struct {
    stm_indices: Buffer,
    stm_num_active: Buffer,
    opp_indices: Buffer,
    opp_num_active: Buffer,
    targets: Buffer,
    size: u32,

    pub fn release(self: *Batch) void {
        self.stm_indices.release();
        self.stm_num_active.release();
        self.opp_indices.release();
        self.opp_num_active.release();
        self.targets.release();
    }
};

pub const NnueModel = struct {
    ft: ml.SparseLinear,
    crelu: ml.ClippedReLU,
    dense: ml.Sequential,
    dummy: *Tensor,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, ctx: *const Context) !NnueModel {
        const ft = try ml.SparseLinear.init(
            allocator,
            ctx,
            nnue.num_features, // 40960
            nnue.ft_out, // 256
            nnue.max_active_features, // 30
            nnue.max_active_features, // expected_active for Kaiming
            42,
        );

        const crelu = ml.ClippedReLU.init(1.0);

        const dense = try ml.Sequential.init(allocator, &.{
            Layer.linear(try ml.Linear.init(allocator, ctx, nnue.fc1_in, nnue.fc1_out, 123)), // 512→32
            Layer.clippedRelu(1.0),
            Layer.linear(try ml.Linear.init(allocator, ctx, nnue.fc2_in, nnue.fc2_out, 456)), // 32→32
            Layer.clippedRelu(1.0),
            Layer.linear(try ml.Linear.init(allocator, ctx, nnue.fc2_out, 1, 789)), // 32→1
        });

        // Dummy tensor for SparseLinear (ignores input)
        const dummy_buf = try Buffer.alloc(ctx, 1);
        const dummy = try allocator.create(Tensor);
        dummy.* = .{
            .shape = ml.Shape.init(&.{1}),
            .storage = .{ .gpu = .{ .buffer = dummy_buf } },
            .allocator = allocator,
        };

        return .{
            .ft = ft,
            .crelu = crelu,
            .dense = dense,
            .dummy = dummy,
            .allocator = allocator,
        };
    }

    pub fn forward(self: *NnueModel, batch: Batch, graph: *Graph) !*Tensor {
        // STM perspective
        self.ft.setIndices(batch.stm_indices, batch.stm_num_active, batch.size);
        const stm_acc = try self.ft.forward(self.dummy, graph);

        // OTM perspective (same weights, different indices)
        self.ft.setIndices(batch.opp_indices, batch.opp_num_active, batch.size);
        const opp_acc = try self.ft.forward(self.dummy, graph);

        // ClippedReLU on each accumulator
        const stm_relu = try self.crelu.forward(stm_acc, graph);
        const opp_relu = try self.crelu.forward(opp_acc, graph);

        // Concat: [batch, 256] ++ [batch, 256] → [batch, 512]
        const combined = try graph.concat(stm_relu, opp_relu);

        // Dense: 512→32→32→1
        return self.dense.forward(combined, graph);
    }

    pub fn parameters(self: *NnueModel) !std.ArrayList(*Tensor) {
        var list = std.ArrayList(*Tensor).empty;
        const ft_params = self.ft.parameters();
        try list.appendSlice(self.allocator, &ft_params);
        var dense_params = try self.dense.parameters();
        defer dense_params.deinit(self.allocator);
        try list.appendSlice(self.allocator, dense_params.items);
        return list;
    }

    pub fn namedParameters(self: *NnueModel) ![]ml.serialize.NamedParam {
        const names = [_][]const u8{
            "ft.weight",  "ft.bias",
            "fc1.weight", "fc1.bias",
            "fc2.weight", "fc2.bias",
            "out.weight", "out.bias",
        };

        var params_list = try self.parameters();
        defer params_list.deinit(self.allocator);
        std.debug.assert(params_list.items.len == names.len);

        const result = try self.allocator.alloc(ml.serialize.NamedParam, names.len);
        for (0..names.len) |i| {
            result[i] = .{ .name = names[i], .tensor = params_list.items[i] };
        }
        return result;
    }

    pub fn deinit(self: *NnueModel) void {
        self.ft.deinit();
        self.crelu.deinit();
        self.dense.deinit();
        self.dummy.deinit();
        self.allocator.destroy(self.dummy);
    }
};
