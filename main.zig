const std = @import("std");

// 8 bytes is the size of a safetensors json header (u64)
const HEADER_SIZE_BUF_SIZE = 8;
const SAFETENSORS_PATH = "./gpt-oss-20b/original/model.safetensors";

// hardcode specific member accesses for gpt-oss
const LAYER_I = "attn.qkv.weight";
const LAYER_II = "attn.out.weight";

// Tensors in a Safetensors file have metadata like a JSON key (name), shape, dtype (for us MXFP4), byte position
// We create instances of this struct to describe each respective tensor.
pub const LayerMetadata = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    shape: []u64,
    dtype: []u8,
    offset_start: u64,
    offset_end: u64,

    // allows struct.function() oop style
    const Self = @This();

    pub fn init(allocator: std.memAllocator, name: []u8, shape: []u64, dtype: []u8, offset_start: u64, offset_end: u64) !Self {
        return Self{
            // copy all arrays
            .name = try allocator.dupe(u8, name);
            .shape = try allocator.dupe(u64, shape);
            .dtype = try allocator.dupe(u8, dtype);
            // copy ints
            .offset_start = offset_start;
            .offset_end = offset_end;
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

    pub fn print(self: LayerMetadata) void {
        std.debug.print("{s}\n dtype:{s}\n", .{self.name, self.dtype});

        std.debug.print("    shape: ", .{});
        for (self.shape) |el| {
            std.debug.print("{d}", .{el});
        }

        std.debug.print("\n", .{});
        std.debug.print("    data offsets: ", .{});
        std.debug.print("{d} {d} ", .{self.offset_start, self.offset_end});

        std.debug.print("\n\n", .{});
    }
}  

// layers_info

// function get_safetensors_content - get header + raw tensor data (weights)

// function block_decoder - process individual block

// function tensor_split - turn tensor into individual blocks

// function select tensor - how are we choosing the tensor to start with

// main
pub fn main() void {
    var file = try std.fs.openFileAbsolute(SAFETENSORS_PATH, .{});
    defer file.close();
    
}

