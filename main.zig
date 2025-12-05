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

    pub fn deinit(self: *Self) void {
        // free arrays
        self.allocator.free(self.name);
        self.allocator.free(self.shape);
        self.allocator.free(self.dtype);
    }

    pub fn print(self: *const Self) void {
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

pub const QuantizedTensor = struct {
    // everything previously stored inside retrieve_quantized_values() is put into this struct
    allocator: std.mem.Allocator,
    shape: []u64,
    blocks_buf: []u8,
    scales_buf: []u8,
    num_values: u64,
    num_scales: u64,
    values_per_scale: u64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, shape: []const u64, blocks_buf: []u8, scales_buf: []u8, num_values: u64, num_scales: u64, values_per_scale: u64) !Self {
        return Self{
            .allocator = allocator,
            .shape = try allocator.dupe(u64, shape),
            .blocks_buf = blocks_buf,
            .scales_buf = scales_buf,
            .num_values = num_values,
            .num_scales = num_scales,
            .values_per_scale = values_per_scale,
        };
    }

    pub fn deinit(self: *Self) void {
        // free arrays
        self.allocator.free(self.shape);
        self.allocator.free(self.blocks_buf);
        self.allocator.free(self.scales_buf);
    }

    // Take a logical/global element index and return the raw FP4 nibble (0–15).
    pub fn decode_raw(self: *const Self, idx: usize) u4 {
        std.debug.assert(idx < self.num_values);

        const byte_idx = idx / 2;
        const byte = self.blocks_buf[byte_idx];

        const nibble: u4 = if ((idx & 1) == 0)
            @intCast(byte & 0x0f) // low nibble
        else
            @intCast((byte >> 4) & 0x0f); // high nibble

        return nibble;
    }

    pub fn scale_for(self: *const Self, idx: usize) u8 {
        const scale_idx = idx / self.values_per_scale;
        std.debug.assert(scale_idx < self.num_scales);
        return self.scales_buf[scale_idx];
    }

    // // take nibble and turn FP4 ->
    // pub fn decode(self: *const Self, idx: usize) f32 {
    //     const nibble = self.decode_raw(idx);
    //     // TODO: implement your MXFP4 → f32 mapping here
    //     // e.g. sign/exponent/mantissa logic
    //     return fp4ToFloat(nibble);
    // }

    // pub fn dequantize(self: *const Self, idx: usize) f32 {
    //     const val = self.decode(idx);
    //     const scale = self.scale_for(idx);

    //     const s = scaleByteToFloat(scale);
    //     return val * s;
    // }

    // // Returns something that can stream dequantized values
    // pub fn reader(self);
};

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

    pub fn deinit(self: *Self) void {
        defer {
            // for each TensorMetadata
            for (self.tensors_metadata.items) |*item| {
                item.deinit();
            }
            self.tensors_metadata.deinit();
        }
    }
};

const UnifiedMetadata = struct {
    block: *const TensorMetadata,
    scale: *const TensorMetadata,
};

const LoadedBuffers = struct {
    blocks_buf: []u8,
    scales_buf: []u8,
    num_values: usize,
    num_scales: usize,
    values_per_scale: usize,
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

// walk fp4 values
// one block per scale
pub fn print_block(buffers: LoadedBuffers) void {
    for (0..buffers.num_scales) |scale_idx| {
        const block_start = scale_idx * buffers.values_per_scale;
        const block_end = @min(block_start + buffers.values_per_scale, buffers.num_values);

        var total_idx = block_start;
        while (total_idx < block_end) : (total_idx += 1) {
            const byte_idx = total_idx / 2;
            const byte = buffers.blocks_buf[byte_idx];

            const nibble: u4 = if ((total_idx & 1) == 0)
                @intCast(byte & 0x0f) // low nibble
            else
                @intCast((byte >> 4) & 0x0f); // high nibble

            // TODO: move to different function
            // only print a tiny sample so this doesn't explode
            if (scale_idx == 0 and total_idx < block_start + 8) {
                std.debug.print(
                    "scale={d}, total_idx={d}, byte_idx={d}, fp4_raw={d}\n",
                    .{ scale_idx, total_idx, byte_idx, nibble },
                );
            }

            // comment out for all 530M lines
            // std.debug.print("tot/l index: {d}, fp4 value: {d}\n", .{ total_idx, fp4_raw });
        }
    }
}

// given JSON key, gets metadata for both blocks tensor and scales tensor
pub fn getBlockAndScaleMetadata(comptime base: []const u8, tensor_list: TensorList) !UnifiedMetadata {
    const block_name = base ++ ".blocks";
    const scale_name = base ++ ".scales";

    // TEMPORARY: traverse TensorList for name == target
    const block_tensor_meta = getTensorByName(tensor_list, block_name) orelse {
        std.debug.panic("Could not find tensor {s} in safetensors header.\n", .{block_name});
    };
    const scale_tensor_meta = getTensorByName(tensor_list, scale_name) orelse {
        std.debug.panic("Could not find tensor {s} in safetensors header.\n", .{scale_name});
    };

    return UnifiedMetadata{
        .block = block_tensor_meta,
        .scale = scale_tensor_meta,
    };
}

fn loadBlocksAndScales(
    metadata: UnifiedMetadata,
    header_size: u64,
    allocator: std.mem.Allocator,
) !LoadedBuffers {
    // loadBlocksAndScales
    var file = try std.fs.openFileAbsolute(SAFETENSORS_PATH, .{});
    defer file.close();

    // our memory: | N | File Header | Tensor Data |
    // so our memory calculation becomes HEADER_SIZE_BYTES + header_size
    const start_offset = HEADER_SIZE_BYTES + header_size;

    // BLOCKS TENSOR
    const block_tensor_size_u64 = metadata.block.offset_end - metadata.block.offset_start;
    const block_tensor_size: usize = @intCast(block_tensor_size_u64);
    const num_values = block_tensor_size * 2;
    // const num_blocks = (num_values + MXFP4_BLOCK_SIZE - 1) / MXFP4_BLOCK_SIZE;

    // no defer (on success) as we pass to loaded_buffers
    try file.seekTo(metadata.block.offset_start + start_offset);
    const blocks_buf = try allocator.alloc(u8, block_tensor_size);
    errdefer allocator.free(blocks_buf);

    const blocks_bytes_read = try file.read(blocks_buf);
    if (blocks_bytes_read != block_tensor_size) {
        std.debug.panic("Error in Block Tensor - Expected bytes: {}. Actual bytes: {}\n", .{ block_tensor_size, blocks_bytes_read });
    }

    // SCALES TENSOR
    const scale_tensor_size_u64 = metadata.scale.offset_end - metadata.scale.offset_start;
    const scale_tensor_size: usize = @intCast(scale_tensor_size_u64);
    // from shape: rows x cols

    try file.seekTo(metadata.scale.offset_start + start_offset);
    const scales_buf = try allocator.alloc(u8, scale_tensor_size);
    errdefer allocator.free(scales_buf);

    const scales_bytes_read = try file.read(scales_buf);
    if (scales_bytes_read != scale_tensor_size) {
        std.debug.panic("Error in Scales Tensor - Expected bytes: {}. Actual bytes: {}\n", .{ scale_tensor_size, scales_bytes_read });
    }

    // COMBINING BLOCKS TENSOR, SCALES TENSOR

    const num_scales = metadata.scale.shape[0] * metadata.scale.shape[1];
    const values_per_scale: usize = num_values / num_scales;

    return LoadedBuffers{
        .blocks_buf = blocks_buf,
        .scales_buf = scales_buf,
        .num_values = num_values,
        .num_scales = num_scales,
        .values_per_scale = values_per_scale,
    };
}

// debug: check we can access bytes of an individual tensor
// this will be repurposed to run for EVERY tensor
// however I am hardcoding it for now, to run for one.
pub fn retrieve_quantized_values(header_size: u64, tensor_list: TensorList, allocator: std.mem.Allocator) !QuantizedTensor {
    const base = "block.22.mlp.mlp1_weight";

    const metadata = try getBlockAndScaleMetadata(base, tensor_list);
    const loaded = try loadBlocksAndScales(metadata, header_size, allocator);

    const quantized_tensor = try QuantizedTensor.init(
        allocator,
        metadata.block.shape,
        loaded.blocks_buf,
        loaded.scales_buf,
        loaded.num_values,
        loaded.num_scales,
        loaded.values_per_scale,
    );

    return quantized_tensor;
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
    var tensor_list = try get_safetensors_content(SAFETENSORS_PATH, allocator);
    defer tensor_list.deinit();

    // // print each TensorMetadata to ensure it works
    for (tensor_list.tensors_metadata.items) |layer_spec| {
        layer_spec.print();
    }

    var qt = try retrieve_quantized_values(tensor_list.header_size, tensor_list, allocator);
    defer qt.deinit();
    // ok now we want to access one specific tensor and get its fp4 values
    // const retrieve_tensor_raw_bytes();
}
