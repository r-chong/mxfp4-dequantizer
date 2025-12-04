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
    name: []u8, // e.g. "block.0.mlp.mlp1_weight.blocks"
    shape: []u64, // e.g. [4096, 4096] (rows, cols)
    dtype: []u8, // e.g. "FP4"
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
            // for each TensorMetadata
            for (self.tensors_metadata.items) |item| {
                item.deinit();
            }
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

        // populate typed shapes array in u64 array
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

        // no allocations necessary as we allocator dupe inside TensorMetadata init

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

// TEMPORARY: traverse TensorList for name == target
fn getTensorByName(tensor_list: TensorList, target: []const u8) ?*const TensorMetadata {
    for (tensor_list.tensors_metadata.items) |*tensor_metadata| {
        std.debug.print("name: {s}\n", .{tensor_metadata.name});
        if (std.mem.eql(u8, tensor_metadata.name, target)) {
            return tensor_metadata;
        }
    } else {
        return null;
    }
}

// debug: check we can access bytes of an individual tensor
// this will be repurposed to run for EVERY tensor
// however I am hardcoding it for now, to run for one.
pub fn retrieve_tensor_raw_bytes(header_size: u64, tensor_list: TensorList, allocator: std.mem.Allocator) !void {
    const base = "block.22.mlp.mlp1_weight";

    // TEMPORARY:
    // this is our block_tensor_meta
    const block_name = base ++ ".blocks";
    // traverse TensorList for name == base.scales
    // this is our scale_tensor_meta
    const scale_name = base ++ ".scales";

    // TEMPORARY: traverse TensorList for name == target
    const block_tensor_meta = getTensorByName(tensor_list, block_name) orelse {
        std.debug.panic("Could not find tensor {s} in safetensors header.\n", .{block_name});
    };
    const scale_tensor_meta = getTensorByName(tensor_list, scale_name) orelse {
        std.debug.panic("Could not find tensor {s} in safetensors header.\n", .{scale_name});
    };

    var file = try std.fs.openFileAbsolute(SAFETENSORS_PATH, .{});
    defer file.close();

    // our memory: | N | File Header | Tensor Data |
    // so our memory calculation becomes HEADER_SIZE_BYTES + header_size
    const start_offset = HEADER_SIZE_BYTES + header_size;

    // in bytes
    const block_tensor_size = block_tensor_meta.offset_end - block_tensor_meta.offset_start;

    // BLOCKS TENSOR
    const num_values = block_tensor_size * 2;
    const num_blocks = (num_values + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;

    try file.seekTo(block_tensor_meta.offset_start + start_offset);
    const blocks_buf = try allocator.alloc(u8, block_tensor_size);
    defer allocator.free(blocks_buf);

    const blocks_bytes_read = try file.read(blocks_buf);
    if (blocks_bytes_read != block_tensor_size) {
        std.debug.panic("Error in Block Tensor - Expected bytes: {}. Actual bytes: {}\n", .{ block_tensor_size, blocks_bytes_read });
    }

    // SCALES TENSOR
    const scale_tensor_size = scale_tensor_meta.offset_end - scale_tensor_meta.offset_start;
    // from shape: rows x cols
    const num_scales = scale_tensor_meta.shape[0] * scale_tensor_meta.shape[1];

    try file.seekTo(scale_tensor_meta.offset_start + start_offset);
    const scales_buf = try allocator.alloc(u8, scale_tensor_size);
    defer allocator.free(scales_buf);

    const scales_bytes_read = try file.read(scales_buf);
    if (scales_bytes_read != scale_tensor_size) {
        std.debug.panic("Error in Scales Tensor - Expected bytes: {}. Actual bytes: {}\n", .{ scale_tensor_size, scales_bytes_read });
    }

    // COMBINING BLOCKS TENSOR, SCALES TENSOR

    const values_per_scale: usize = num_values / num_scales;
    std.debug.print(
        "num_values={d}, num_scales={d}, values_per_scale={d}\n",
        .{ num_values, num_scales, values_per_scale },
    );

    std.debug.print("num_values={d}, num_blocks={d}, num_scales={d}\n", .{ num_values, num_blocks, num_scales });

    // one block per scale
    for (0..num_scales) |scale_idx| {
        // const scale = scales_buf[scale_idx];

        const block_start = scale_idx * values_per_scale;
        const block_end = @min(block_start + values_per_scale, num_values);

        var total_idx = block_start;
        while (total_idx < block_end) : (total_idx += 1) {
            const byte_idx = total_idx / 2;
            const byte = blocks_buf[byte_idx];

            const fp4_raw: u4 = if ((total_idx & 1) == 0)
                @intCast(byte & 0x0f) // low nibble
            else
                @intCast((byte >> 4) & 0x0f); // high nibble

            // only print a tiny sample so this doesn't explode
            if (scale_idx == 0 and total_idx < block_start + 8) {
                std.debug.print(
                    "scale={d}, total_idx={d}, byte_idx={d}, fp4_raw={d}\n",
                    .{ scale_idx, total_idx, byte_idx, fp4_raw },
                );
            }

            // comment out for all 530M lines
            // std.debug.print("total index: {d}, fp4 value: {d}\n", .{ total_idx, fp4_raw });
        }
    }

    // std.debug.print("block size: {d}, num_blocks: {d}\n", .{ block_size, num_blocks });

    // var block_idx: usize = 0;

    // // loop through block
    // while (block_idx < num_scales) : (block_idx += 1) {
    //     // const scale = scales_buf[block_idx];

    //     // nested loop for all values in block. value_idx resets every time whereas total idx keeps track of global values (not resetting every 32 values)
    //     var value_idx: usize = 0;
    //     while (value_idx < block_size) : (value_idx += 1) {
    //         const total_idx = block_idx * block_size + value_idx;
    //         // since 4 bits, divide by 2 since 2 4bits in 8 bytes
    //         const byte_idx = total_idx / 2;
    //         const byte = blocks_buf[byte_idx];

    //         // use bit comparison to check if even (ends in 0) or odd (ends in 1)
    //         const fp4_raw: u4 = if ((value_idx & 1) == 0)
    //             @intCast(byte & 0x0f) // bit mask - take only lower nibble
    //         else
    //             @intCast((byte >> 4) & 0x0f); // bit mask - take only lower nibble

    //         std.debug.print("total index: {d}, fp4 value: {d}\n", .{ total_idx, fp4_raw });
    //     }
    // }

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
    const tensor_list = try get_safetensors_content(SAFETENSORS_PATH, allocator);
    defer tensor_list.deinit();

    // // print each TensorMetadata to ensure it works
    for (tensor_list.tensors_metadata.items) |layer_spec| {
        layer_spec.print();
    }

    const this_tensor_bytes = retrieve_tensor_raw_bytes(tensor_list.header_size, tensor_list, allocator);
    std.debug.print("this tensor bytes: {any} ", .{this_tensor_bytes});

    // ok now we want to access one specific tensor and get its fp4 values
    // const retrieve_tensor_raw_bytes();
}
