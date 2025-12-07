const std = @import("std");

// immutable global constants

// how large our safetensors json header is - we got 8 bytes (u64) as per safetensors spec
const HEADER_SIZE_BYTES = 8;
const SAFETENSORS_PATH = "/Users/reese/code/cur_project/mxfp4-dequantizer/gpt-oss-20b/original/model.safetensors";

// -----------------------------
// hardcoded specifics for gpt-oss weight formats. If you are using non GPT-OSS models then you should use different values.
// -----------------------------

// layer member accesses
// Extracted from GPT-OSS files: model.safetensors and dtypes.json
const LayerKind = enum {
    AttnNormScale,
    AttnQkvWeight,
    AttnQkvBias,
    AttnOutWeight,
    AttnOutBias,
    AttnSinks,

    MlpNormScale,
    MlpGateWeight,
    MlpGateBias,

    Mlp1WeightQuant, // base name for .blocks/.scales
    Mlp1Bias,
    Mlp2WeightQuant, // base name for .blocks/.scales
    Mlp2Bias,
};

const Layer = struct {
    block_idx: usize,
    kind: LayerKind,
};

fn kind_to_str(kind: LayerKind) []const u8 {
    return switch (kind) {
        .AttnNormScale => "attn.norm.scale",
        .AttnQkvWeight => "attn.qkv.weight",
        .AttnQkvBias => "attn.qkv.bias",
        .AttnOutWeight => "attn.out.weight",
        .AttnOutBias => "attn.out.bias",
        .AttnSinks => "attn.sinks",

        .MlpNormScale => "mlp.norm.scale",
        .MlpGateWeight => "mlp.gate.weight",
        .MlpGateBias => "mlp.gate.bias",

        .Mlp1WeightQuant => "mlp.mlp1_weight",
        .Mlp1Bias => "mlp.mlp1_bias",
        .Mlp2WeightQuant => "mlp.mlp2_weight",
        .Mlp2Bias => "mlp.mlp2_bias",
    };
}

fn get_blocks_name_str(allocator: std.mem.Allocator, base_name: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "{s}.blocks", .{base_name});
}

fn get_scales_name_str(allocator: std.mem.Allocator, base_name: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "{s}.scales", .{base_name});
}

// base name is the JSON key
fn make_base_name(allocator: std.mem.Allocator, layer: Layer) ![]const u8 {
    return try std.fmt.allocPrint(allocator, "block.{d}.{s}", .{ layer.block_idx, kind_to_str(layer.kind) });
}

const TENSOR_PLACEHOLDER = "block.0.mlp.mlp1_weight";

// -----------------------------
// MXFP4 format
// -----------------------------

// GPT-OSS stores its scaling weights in separate tensors. if scaling values are stored WITH a block, then ensure this is kept in consideration.
const MXFP4_BLOCK_SIZE: usize = 32;
const MXFP4_VALUES_PER_BYTE: usize = 2;

fn fp4_to_float(n: u4) f32 {
    // first bit is sign, next two bits exponent, last bit mantissa
    // bit layout: [s e e m]
    const s: u1 = @intCast((n >> 3) & 0x1);
    const e: u2 = @intCast((n >> 1) & 0x3);
    const m: u1 = @intCast(n & 0x1);

    const sign: f32 = if (s == 1) -1.0 else 1.0;

    if (e == 0) {
        // subnormal: exponent = -1, frac = 0.0 or 0.5
        const frac: f32 = if (m == 0) 0.0 else 0.5;
        const scale: f32 = 0.5; // 2^-1
        return sign * frac * scale;
    }

    // normalized: exponent_i = e - 1 ∈ {0,1,2}
    const frac: f32 = if (m == 0) 1.0 else 1.5;
    const scale: f32 = switch (e) {
        1 => 1.0, // 2^0
        2 => 2.0, // 2^1
        3 => 4.0, // 2^2
        else => 1.0, // shouldn't happen
    };

    return sign * frac * scale;
}

// assemble a 16 bit integer from two 8-bit chunks (bytes) - 16-bit not to be confused with bf16
// note: This does NOT mean "scale a byte to a float". It means: return the byte-format representation of our scale (i.e., the scale's bytes).
fn scale_byte_to_float(ptr: []const u8, idx: usize) f32 {
    const raw: u8 = ptr[idx];

    // Build a float32 from E8M0 exponents
    const bits: u32 = @as(u32, raw) << 23;
    const scale: f32 = @bitCast(bits);

    return scale;
}

// function tensor_split - turn tensor into individual blocks

// function select tensor layer - how are we choosing the layer

// -----------------------------
// Safetensors
// -----------------------------

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

const UnifiedMetadata = struct {
    block: *const TensorMetadata,
    scale: *const TensorMetadata,
};

pub const QuantizedTensor = struct {
    // everything previously stored inside retrieve_quantized_values() is put into this struct
    // note: we keep all counts we use as indices in comparisons, as usize
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
            .num_values = @intCast(num_values),
            .num_scales = @intCast(num_scales),
            .values_per_scale = @intCast(values_per_scale),
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

        // integer division allows us to store a low nibble and high nibble per index
        const byte_idx = idx / 2;
        const byte = self.blocks_buf[byte_idx];

        const nibble: u4 = if ((idx & 1) == 0)
            @intCast(byte & 0x0f) // low nibble
        else
            @intCast((byte >> 4) & 0x0f); // high nibble

        return nibble;
    }

    // returns the scale index from the scale tensor
    pub fn scale_idx_for(self: *const Self, idx: usize) usize {
        const scale_idx = idx / self.values_per_scale;
        std.debug.assert(scale_idx < self.num_scales);
        return scale_idx;
    }

    // decode an individual nibble
    pub fn decode(self: *const Self, idx: usize) f32 {
        const nibble = self.decode_raw(idx);
        return fp4_to_float(nibble);
    }

    // take decoded nibble and scale, turn scale into f32, and multiply by value to get dequantized nibble
    // TODO: replace with SIMD
    pub fn dequantize_nibble(self: *const Self, nibble_idx: usize) f32 {
        const val = self.decode(nibble_idx);
        const scale_idx = self.scale_idx_for(nibble_idx);

        const scale = scale_byte_to_float(self.scales_buf, scale_idx);
        return val * scale;
    }

    pub fn dequantize_block(self: *const Self, scale_idx: usize, out: []f32) usize {
        std.debug.assert(out.len == self.values_per_scale);
        const block_start = scale_idx * self.values_per_scale;

        var i: usize = 0;
        while (i < self.values_per_scale) : (i += 1) {
            out[i] = self.dequantize_nibble(block_start + i);
        }
        return i;
    }

    pub fn dequantize_ostream(self: *const Self, start_idx: usize, out: []f32) usize {
        if (start_idx >= self.num_values) return 0;

        var idx: usize = start_idx;
        var written: usize = 0;

        const max_idx = self.num_values;
        const cap = out.len;

        while (idx < max_idx and written < cap) : (idx += 1) {
            out[written] = self.dequantize_nibble(idx);
            written += 1;
        }

        return written;
    }
    // stream dequantized values
    // pub fn read(self);
};

const TensorReader = struct {
    quantized_tensor: *QuantizedTensor,
    cursor: usize,

    const Self = @This();

    pub fn init(quantized_tensor: *QuantizedTensor) Self {
        return Self{
            .quantized_tensor = quantized_tensor,
            .cursor = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        // dont free the quantized tensor here!!!
        self.* = undefined;
    }

    // read <=out.len values into out, returning # vals written'
    pub fn read(self: *Self, out: []f32) usize {
        if (out.len == 0) return 0;

        const written = self.quantized_tensor.dequantize_ostream(self.cursor, out);
        self.cursor += written;
        return written;
    }

    pub fn reset(self: *Self) void {
        self.cursor = 0;
    }
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

    // traverse TensorList for name == target
    fn get_tensor_by_name(self: *const Self, target: []const u8) ?*const TensorMetadata {
        for (self.tensors_metadata.items) |*tensor_metadata| {
            if (std.mem.eql(u8, tensor_metadata.name, target)) {
                return tensor_metadata;
            }
        } else {
            return null;
        }
    }

    fn get_num_blocks(self: *const Self) usize {
        var max_num_blocks: u64 = 0;
        for (self.tensors_metadata.items) |tensor_metadata| {
            const name = tensor_metadata.name;

            if (!std.mem.startsWith(u8, name, "block.")) continue;

            if (std.mem.indexOfScalarPos(u8, name, 6, '.')) |pos| {
                const idx_slice = name[6..pos];

                const block_idx = std.fmt.parseInt(usize, idx_slice, 10) catch continue;

                // Track highest seen block index
                if (block_idx + 1 > max_num_blocks) {
                    max_num_blocks = block_idx + 1;
                }
            }
        }

        return max_num_blocks;
    }
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
pub fn get_safetensors_content(allocator: std.mem.Allocator, file: *std.fs.File) !TensorList {
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
pub fn get_block_and_scale_metadata(allocator: std.mem.Allocator, base_name: []const u8, tensor_list: TensorList) !UnifiedMetadata {
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
) !LoadedBuffers {
    // load_blocks_and_scales
    // TODO: pass file pointer
    var file = try std.fs.openFileAbsolute(SAFETENSORS_PATH, .{});
    defer file.close();

    // our memory: | N | File Header | Tensor Data |
    // so our memory calculation becomes HEADER_SIZE_BYTES + header_size
    const start_offset = HEADER_SIZE_BYTES + header_size;

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
pub fn retrieve_quantized_values_for_tensor(allocator: std.mem.Allocator, header_size: u64, tensor_list: TensorList, base_name: []const u8) !QuantizedTensor {
    const metadata = try get_block_and_scale_metadata(allocator, base_name, tensor_list);
    const loaded = try load_blocks_and_scales(metadata, header_size, allocator);

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

// master function - loads a QuantizedTensor object on which you can call .dequantize_ostream()
pub fn load_quantized_tensor(allocator: std.mem.Allocator, layer: Layer, tensor_list: TensorList) !QuantizedTensor {
    // build JSON key
    const base_name = try make_base_name(allocator, layer);
    defer allocator.free(base_name);

    // get QuantizedTensor
    const quantized_tensor = try retrieve_quantized_values_for_tensor(allocator, tensor_list.header_size, tensor_list, base_name);
    // do NOT deinit quantized_tensor - caller should free

    return quantized_tensor;
}

// these are the only weights that are quantized
const QUANT_LAYER_KINDS = [_]LayerKind{
    .Mlp1WeightQuant,
    .Mlp2WeightQuant,
};

fn model_driver(allocator: std.mem.Allocator, tensor_list: TensorList) !void {
    const sample_len: usize = 16;
    var sample = try allocator.alloc(f32, sample_len);
    defer allocator.free(sample);

    // iterate over blocks
    for (0..tensor_list.get_num_blocks()) |block_idx| {
        for (QUANT_LAYER_KINDS) |kind| {
            const layer: Layer = .{ .block_idx = block_idx, .kind = kind };

            std.debug.print("=== block.{d}.{s} ===\n", .{ block_idx, kind_to_str(kind) });

            var quantized_tensor = load_quantized_tensor(allocator, layer, tensor_list) catch |err| {
                // If a particular layer/kind combo doesn't exist, just skip it.
                std.debug.print("  skipping (could not load): {any}\n", .{err});
                continue;
            };
            defer quantized_tensor.deinit();

            var reader = TensorReader.init(&quantized_tensor);
            defer reader.deinit();

            const written = reader.read(sample);
            std.debug.print("=== {s} ===\n", .{"block.22.mlp.mlp1_weight"});
            for (sample[0..written], 0..) |v, i| {
                std.debug.print("  [{d}] = {d}\n", .{ i, v });
            }
        }
    }
}

// main
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer switch (gpa.deinit()) {
        .leak => std.debug.panic("Leaked", .{}),
        .ok => {},
    };

    const allocator = gpa.allocator();

    var file = try std.fs.openFileAbsolute(SAFETENSORS_PATH, .{});
    defer file.close();

    // DEMO ITEMS:
    // choose layer via for loop
    // const layer: Layer = .{ .block_idx = 22, .kind = LayerKind.Mlp1WeightQuant };

    // parse header
    var tensor_list = try get_safetensors_content(allocator, &file);
    defer tensor_list.deinit();

    // MODEL DRIVER:
    try model_driver(allocator, tensor_list);
}
