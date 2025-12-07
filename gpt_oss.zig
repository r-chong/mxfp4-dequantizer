const std = @import("std");
const safetensors = @import("safetensors.zig");
const quant = @import("quantized_tensor.zig");
const layer = @import("layer.zig");

pub const UnifiedMetadata = struct {
    block: *const safetensors.TensorMetadata,
    scale: *const safetensors.TensorMetadata,
};

fn get_blocks_name_str(allocator: std.mem.Allocator, base_name: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "{s}.blocks", .{base_name});
}

fn get_scales_name_str(allocator: std.mem.Allocator, base_name: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "{s}.scales", .{base_name});
}

const LoadedBuffers = struct {
    blocks_buf: []u8,
    scales_buf: []u8,
    num_values: usize,
    num_scales: usize,
    values_per_scale: usize,
};

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
pub fn get_block_and_scale_metadata(allocator: std.mem.Allocator, base_name: []const u8, tensor_list: safetensors.TensorList) !UnifiedMetadata {
    const block_name = try get_blocks_name_str(allocator, base_name);
    defer allocator.free(block_name);

    const scale_name = try get_scales_name_str(allocator, base_name);
    defer allocator.free(scale_name);

    // TEMPORARY: traverse TensorList for name == target
    const block_tensor_meta = tensor_list.get_tensor_by_name(block_name) orelse {
        std.debug.panic("Could not find tensor {s} in safetensors header.\n", .{block_name});
    };
    const scale_tensor_meta = tensor_list.get_tensor_by_name(scale_name) orelse {
        std.debug.panic("Could not find tensor {s} in safetensors header.\n", .{scale_name});
    };

    return UnifiedMetadata{
        .block = block_tensor_meta,
        .scale = scale_tensor_meta,
    };
}

fn load_blocks_and_scales(
    metadata: UnifiedMetadata,
    header_size: u64,
    allocator: std.mem.Allocator,
    file: *std.fs.File,
) !LoadedBuffers {
    // load_blocks_and_scales
    // our memory: | N | File Header | Tensor Data |
    // so our memory calculation becomes HEADER_SIZE_BYTES + header_size
    const start_offset = safetensors.HEADER_SIZE_BYTES + header_size;

    // BLOCKS TENSOR
    const block_tensor_size_u64 = metadata.block.offset_end - metadata.block.offset_start;
    const block_tensor_size: usize = @intCast(block_tensor_size_u64);
    const num_values = block_tensor_size * 2;

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
pub fn retrieve_quantized_values_for_tensor(allocator: std.mem.Allocator, header_size: u64, tensor_list: safetensors.TensorList, base_name: []const u8, file: *std.fs.File) !quant.QuantizedTensor {
    const metadata = try get_block_and_scale_metadata(allocator, base_name, tensor_list);
    const loaded = try load_blocks_and_scales(metadata, header_size, allocator, file);

    const quantized_tensor = try quant.QuantizedTensor.init(
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

// master function - loads a QuantizedTensor object on which you can call .dequantize_ostream()
pub fn load_quantized_tensor(allocator: std.mem.Allocator, lyr: layer.Layer, tensor_list: safetensors.TensorList, file: *std.fs.File) !quant.QuantizedTensor {
    // build JSON key
    const base_name = try layer.makeBaseName(allocator, lyr);
    defer allocator.free(base_name);

    // get QuantizedTensor
    const quantized_tensor = try retrieve_quantized_values_for_tensor(allocator, tensor_list.header_size, tensor_list, base_name, file);
    // do NOT deinit quantized_tensor - caller should free

    return quantized_tensor;
}
