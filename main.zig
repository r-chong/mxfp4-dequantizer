const std = @import("std");

// immutable global constants

// how large our safetensors json header is - we got 8 bytes (u64) as per safetensors spec
const HEADER_SIZE_BYTES = 8;
const SAFETENSORS_PATH = "/Users/reese/code/cur_project/mxfp4-dequantizer/gpt-oss-20b/original/model.safetensors";

// -----------------------------
// hardcoded specifics for gpt-oss weight formats. If you are using non GPT-OSS models then you should use different values.
// -----------------------------

// layer member accesses
const LAYER_I = "attn.qkv.weight";
const LAYER_II = "attn.out.weight";
const TENSOR_PLACEHOLDER = "block.0.mlp.mlp1_weight";

// GPT-OSS stores its scaling weights in separate tensors. if scaling values are stored WITH a block, then ensure this is kept in consideration.
const MXFP4_BLOCK_SIZE: usize = 32;
const MXFP4_VALUES_PER_BYTE: usize = 2;

// Tensors in a Safetensors file have metadata like a JSON key (name), shape, dtype (for us MXFP4), byte position
// We create instances of this struct to describe each respective tensor.
pub const TensorMetadata = struct {
    allocator: std.mem.Allocator,
    name: []u8,     // e.g. "block.0.mlp.mlp1_weight.blocks"
    shape: []u64,   // e.g. [4096, 4096] (rows, cols)
    dtype: []u8,    // e.g. "FP4"
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

// TensorLists are NOT layers. There is only ONE TensorList per file.
pub const TensorList = struct {
    allocator: std.mem.Allocator,
    tensors_metadata: std.ArrayList(TensorMetadata),
    /// TensorList.header_size is the JSON header size for this file. These bytes + 8 is how much we skip to get to our first tensor.
    header_size: u64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, header_size: u64) Self {
        return Self{
            .allocator = allocator,
            .tensors_metadata = std.ArrayList(TensorMetadata).init(allocator),
            .header_size = header_size,
        };
    }

    pub fn deinit(self: TensorList) void {
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
// TODO: turn this into a map. currently using a list approach as we're doing everything on the fly
pub fn get_safetensors_content(filepath: []const u8, allocator: std.mem.Allocator) !TensorList {
    var file = try std.fs.openFileAbsolute(filepath, .{});
    defer file.close();

    // we create a buffer to read our JSON bytes into
    var header_size_buf: [HEADER_SIZE_BYTES]u8 = undefined;

    const bytes_read = try file.read(&header_size_buf);
    if (bytes_read != HEADER_SIZE_BYTES) {
        std.debug.panic("Error - Expected bytes: {}. Actual bytes: {}", .{ HEADER_SIZE_BYTES, bytes_read });
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

    var tensors_list = TensorList.init(allocator, header_size_N);

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

// debug: check we can access bytes of an individual tensor
// this will be repurposed to run for EVERY tensor
// however I am hardcoding it for now, to run for one.
pub fn retrieve_tensor_raw_bytes(header_size: u64, tensor_list: TensorList, allocator: std.mem.Allocator) void {
    const base = "block.0.mlp.mlp1_weight";
    
    // should pass in existing reader
    var file = std.fs.openFileAbsolute(SAFETENSORS_PATH, .{});
    defer file.close();

    // TEMPORARY: 
    // traverse TensorList for name == base.blocks
    // this is our block_tensor_meta
    // traverse TensorList for name == base.scales
    // this is our scale_tensor_meta

    // our memory: | N | File Header | Tensor Data |
    // so our memory calculation becomes HEADER_SIZE_BYTES + header_size
    const start_offset = HEADER_SIZE_BYTES + header_size;

    // in bytes
    const block_tensor_size = block_tensor_meta.offset_end_blocks - block_tensor_meta.offset_start_blocks;
    const scale_tensor_size = scale_tensor_meta.offset_end_blocks - scale_tensor_meta.offset_start_scales;

    // BLOCKS TENSOR
    const num_values = block_tensor_size * 2;
    // from shape: rows x cols
    const expected_values_size = block_tensor_meta.shape[0] * block_tensor_meta.shape[1];

    try file.seekTo(block_tensor_meta.offset_start + start_offset);
    const blocks_buf = try allocator.alloc(u8, expected_values_size);
    defer allocator.free(blocks_buf);

    const blocks_bytes_read = try file.read(blocks_buf);
    if (blocks_bytes_read != expected_values_size) {
        std.debug.panic("Error in Block Tensor - Expected bytes: {}. Actual bytes: {}\n", .{ expected_values_size, blocks_bytes_read });
    }

    // SCALES TENSOR
    // from shape: rows x cols
    const num_scales = scale_tensor_meta.shape[0] * scale_tensor_meta.shape[1];

    try file.seekTo(scale_tensor_meta.offset_start + start_offset);
    const scales_buf = try allocator.alloc(u8, expected_values_size);
    defer allocator.free(scales_buf);

    const scales_bytes_read = try file.read(scales_buf);
    if (scales_bytes_read != scale_tensor_size) {
        std.debug.panic("Error in Scales Tensor - Expected bytes: {}. Actual bytes: {}\n", .{ scale_tensor_size, scales_bytes_read });
    }

    // COMBINING BLOCKS TENSOR, SCALES TENSOR

    const block_size = num_values / num_scales;
    const num_blocks = (num_values + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;

    // for b ... num_scales
    // in between each b, there are block_size values
    // and every v / 2
    // keep track of parity

    // assuming 2 FP4 values per byte (since 1 byte = 8 bits and we're talking about 4 bit numbers)
    // const fp4_bytes = (num_blocks - 1) / 2;
    // const scale_bytes = num_blocks * MXFP4_BLOCK_SIZE;

    // then inside a block,loop through MXFP4_TOTAL_BYTES_PER_BLOCK bytes. ensure that this does not leak

    // print as we go
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
    // const tensors_list = try get_safetensors_content(SAFETENSORS_PATH, allocator);
    // defer tensors_list.deinit();

    // // print each TensorMetadata to ensure it works
    // for (tensors_list.tensors_metadata.items) |layer_spec| layer_spec.print();

    // ok now we want to access one specific tensor and get its fp4 values
    const retrieve_tensor_raw_bytes();
}
