const std = @import("std");

// GPT-OSS stores its scaling weights in separate tensors. if scaling values are stored WITH a block, then ensure this is kept in consideration.
pub const MXFP4_BLOCK_SIZE: usize = 32;
pub const MXFP4_VALUES_PER_BYTE: usize = 2;

pub fn fp4_to_float(n: u4) f32 {
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
pub fn scale_byte_to_float(ptr: []const u8, idx: usize) f32 {
    const raw: u8 = ptr[idx];

    // Build a float32 from E8M0 exponents
    const bits: u32 = @as(u32, raw) << 23;
    const scale: f32 = @bitCast(bits);

    return scale;
}
