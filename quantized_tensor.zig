const std = @import("std");
const mxfp4 = @import("mxfp4.zig");

// SIMD
const Vec4 = @Vector(4, f32);

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
        return mxfp4.fp4_to_float(nibble);
    }

    // take decoded nibble and scale, turn scale into f32, and multiply by value to get dequantized nibble
    // TODO: SIMD
    pub fn dequantize_nibble(self: *const Self, nibble_idx: usize) f32 {
        const val = self.decode(nibble_idx);
        const scale_idx = self.scale_idx_for(nibble_idx);

        const scale = mxfp4.scale_byte_to_float(self.scales_buf, scale_idx);
        return val * scale;
    }

    // TODO: SIMD
    pub fn dequantize_ostream(self: *const Self, start_idx: usize, out: []f32) usize {
        if (start_idx >= self.num_values) return 0;

        var idx: usize = start_idx;
        var written: usize = 0;

        const max_idx = self.num_values;
        const cap = out.len;

        // add 4 to index as we traverse by Vec4
        while (idx + 4 < max_idx and written + 4 < cap) : (idx += 1) {
            const scale_idx0 = self.scale_idx_for(idx);

            out[written] = self.dequantize_nibble(idx);
            written += 1;

            const scale = mxfp4.scale_byte_to_float(self.scales_buf, scale_idx0);
            // apply scale (at scale)
            const scale_vec: Vec4 = @splat(scale);

            // j is the index 0 1 2 3 of the 4 operations that have been parallelized by inline while
            var vals: Vec4 = undefined;
            var j: usize = 0;
            while (j < 4) : (j += 1) {
                vals[j] = self.decode(idx + j);
            }

            // vector mult!
            const res: Vec4 = vals * scale_vec;

            // write 4 results into out[written..written+4]
            const buf_slice = out[written..][0..4];
            const buf_bytes = std.mem.sliceAsBytes(buf_slice);
            const res_bytes = std.mem.asBytes(&res);
            // overwrite the dst slice from left to right
            std.mem.copyForwards(u8, buf_bytes, res_bytes);

            // traverse forward 4
            idx += 4;
            written += 4;
        }

        return written;
    }
    // stream dequantized values
    // pub fn read(self);
};

pub const TensorReader = struct {
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
    pub fn read(self: *Self, buf: []u8) usize {
        if (buf.len == 0) return 0;

        const aligned_buf: []align(@alignOf(f32)) u8 = @alignCast(buf);
        const out = std.mem.bytesAsSlice(f32, aligned_buf);
        const written = self.quantized_tensor.dequantize_ostream(self.cursor, out);
        self.cursor += written;

        return written * @sizeOf(f32);
    }

    pub fn reset(self: *Self) void {
        self.cursor = 0;
    }
};
