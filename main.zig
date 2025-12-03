const std = @import("std");

// 8 bytes is the size of a safetensors json header (u64)
const HEADER_SIZE_BUF_SIZE = 8;
const SAFETENSORS_PATH = "/Users/reese/code/cur_project/mxfp4-dequantizer/gpt-oss-20b/original/model.safetensors";

// hardcode specific member accesses for gpt-oss
const LAYER_I = "attn.qkv.weight";
const LAYER_II = "attn.out.weight";

// Tensors in a Safetensors file have metadata like a JSON key (name), shape, dtype (for us MXFP4), byte position
// We create instances of this struct to describe each respective tensor.
pub const TensorMetadata = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    shape: []u64,
    dtype: []u8,
    offset_start: u64,
    offset_end: u64,

    // allows struct.function() oop style
    const Self = @This();

    // note readonly fields
    pub fn init(allocator: std.mem.Allocator, name: []const u8, shape: []u64, dtype: []const u8, offset_start: u64, offset_end: u64) !Self {
        return Self{
            // copy all arrays
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .shape = try allocator.dupe(u64, shape),
            .dtype = try allocator.dupe(u8, dtype),
            // copy ints
            .offset_start = offset_start,
            .offset_end = offset_end,
        };
    }

    pub fn deinit(self: Self) void {
        // free arrays
        self.allocator.free(self.name);
        self.allocator.free(self.shape);
        self.allocator.free(self.dtype);
    }

    // getters for locations in tensor data
    pub fn get_tensor_start(self: Self) u64 {
        return self.offset_start;
    }

    pub fn get_tensor_end(self: Self) u64 {
        return self.offset_end;
    }

    // TODO: may need getter for number of elements

    pub fn print(self: TensorMetadata) void {
        std.debug.print("{s}\n dtype:{s}\n", .{ self.name, self.dtype });

        std.debug.print("    shape: ", .{});
        for (self.shape) |el| {
            std.debug.print("{d}", .{el});
        }

        std.debug.print("\n", .{});
        std.debug.print("    data offsets: ", .{});
        std.debug.print("{d} {d} ", .{ self.offset_start, self.offset_end });

        std.debug.print("\n\n", .{});
    }
};

// Layers represent a semantic view of a Transformer block / neural network layer in the model
// pub const Layer = struct {
//     q_weight: ?TensorMetadata, // query projection
//     k_weight: ?TensorMetadata, // key projection
//     v_weight: ?TensorMetadata, // value projection
//     o_weight: ?TensorMetadata, // output projection
//     // from MLP (feed forward) block in a Transformer
//     mlp_fc1_weight: ?TensorMetadata,
//     mlp_fc2_weight: ?TensorMetadata,
// }

pub const TensorsList = struct {
    allocator: std.mem.Allocator,
    tensors_metadata: std.ArrayList(TensorMetadata),
    header_size: u64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, header_size: u64) Self {
        return Self{
            .allocator = allocator,
            .tensors_metadata = std.ArrayList(TensorMetadata).init(allocator),
            .header_size = header_size,
        };
    }

    pub fn deinit(self: TensorsList) void {
        defer {
            for (self.tensors_metadata.items) |item| item.deinit();
            self.tensors_metadata.deinit();
        }
    }
};

// TODO: need these right now:
// function get_safetensors_content - get header + raw tensor data (weights)
// read file

// parse JSON UTF-8 string, return tensors_list
pub fn get_safetensors_content(filepath: []const u8, allocator: std.mem.Allocator) !TensorsList {
    var file = try std.fs.openFileAbsolute(filepath, .{});
    defer file.close();

    // we create a buffer to read our JSON bytes into
    var header_size_buf: [HEADER_SIZE_BUF_SIZE]u8 = undefined;

    const bytes_read = try file.read(&header_size_buf);
    if (bytes_read != HEADER_SIZE_BUF_SIZE) {
        std.debug.panic("Error - Expected bytes: {}. Actual bytes: {}", .{ HEADER_SIZE_BUF_SIZE, bytes_read });
    }

    // take N as per safetensors spec
    const header_size_N = std.mem.readInt(u64, &header_size_buf, .little);
    std.debug.print("Header size: {d} bytes.\n", .{header_size_N});

    // read header -so create another buffer
    const header_buf = try allocator.alloc(u8, header_size_N);
    defer allocator.free(header_buf);

    // CHECK BYTES READ and only save if it succeeds
    _ = try file.read(header_buf);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_buf, .{});
    defer parsed.deinit();
    var iter = parsed.value.object.iterator();

    var tensors_list = TensorsList.init(allocator, header_size_N);

    // we iterate through all JSON entries
    while (iter.next()) |entry| {
        // Zig has a hash map entry API which exposes pointers for key value pairs
        // we dereference these values to receive strings
        const name = entry.key_ptr.*;
        const val = entry.value_ptr.*;

        const dtype = val.object.get("dtype").?.string;

        // we receive it as a raw untyped JSON array and can't work with it properly in this form0
        const raw_shape = val.object.get("shape").?.array;

        // thus allocate with correct types
        const shape = try allocator.alloc(u64, raw_shape.items.len);
        defer allocator.free(shape);

        // populate typed shapes array
        for (raw_shape.items, 0..) |el, idx| {
            switch (el) {
                .integer => |num| {
                    shape[idx] = @intCast(num);
                },
                else => {},
            }
        }

        // get offsets
        const raw_offsets = val.object.get("data_offsets").?.array;
        const offset_start: u64 = @intCast(raw_offsets.items[0].integer);
        const offset_end: u64 = @intCast(raw_offsets.items[1].integer);

        const cur_tensor = try TensorMetadata.init(
            allocator,
            name,
            shape,
            dtype,
            offset_start,
            offset_end,
        );

        try tensors_list.tensors_metadata.append(cur_tensor);
    }

    return tensors_list;
}

// function block_decoder - process individual block

// function tensor_split - turn tensor into individual blocks

// function select tensor layer - how are we choosing the layer

// main
pub fn main() !void {
    var file = try std.fs.openFileAbsolute(SAFETENSORS_PATH, .{});
    defer file.close();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer switch (gpa.deinit()) {
        .leak => std.debug.panic("Leaked", .{}),
        .ok => {},
    };

    const allocator = gpa.allocator();

    // ok now we want to get the weights not just the header info. so need to use the offset to access
    const tensors_list = try get_safetensors_content(SAFETENSORS_PATH, allocator);
    defer tensors_list.deinit();

    // print each TensorMetadata to ensure it works
    for (tensors_list.tensors_metadata.items) |layer_spec| layer_spec.print();
}
