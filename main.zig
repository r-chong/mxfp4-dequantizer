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
};

pub const LayersInfo = struct {
    allocator: std.mem.Allocator,
    layers_metadata = std.ArrayList(LayerMetadata),
    header_size: u64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, header_size: u64) Self {
        return Self{
            .allocator = allocator,
            .layers_metadata = std.ArrayList(LayerMetadata).init(Allocator),
            .header_size = header_size;
        }
    }

    pub fn deinit(self: LayersInfo) void {
        defer {
            for (self.layers_metadata.items) |item| item.deinit();
            self.layers_metadata.deinit();
        }
    }   
};

// need these right now:
// function get_safetensors_content - get header + raw tensor data (weights) 
// read file

// parse JSON UTF-8 string, return layers_info
pub fn get_safetensors_content(filepath: []const u8, allocator: std.mem.Allocator) {
    // read file
    var file = try std.fs.openFileAbsolute(fpath, .{});
    defer file.close();

    // we create "header_size_buf", a buffer to read our JSON bytes into
    var header_size_buf: [HEADER_SIZE_BUF_SIZE]u8 = undefined;

    const bytes_read = try file.read(&header_size_buf);
    if (bytes_read != HEADER_SIZE_BUF_SIZE) {
        std.debug.panic("Error - Expected bytes: {}. Actual bytes: {}", .{HEADER_SIZE_BUF_SIZE, bytes_read});
    }

    // take N as per safetensors spec
    const header_size_N = std.mem.readInt(u64, &header_size_buf, .little);
    std.debug.print("Header size: {d} bytes.\n", .{header_size_N});

    // read header -so create another buffer
    const header_buf = try allocator.alloc(u8, header_size_N);
    defer allocator.free(header_buf);

    // CHECK BYTES READ
    _ = try file.read(header.buf);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, header_buf, .{});
    defer parsed.deinit();
    var iter = parsed.value.object.iterator();

    var layers_info = LayersInfo.init(allocator, header_size_N);

    // we iterate through all JSON entries
    while(iter.next()) |entry| {
        // Zig has a hash map entry API which exposes pointers for key value pairs
        // we dereference these values to receive strings
        const name = entry.key_ptr.*;
        const val = entry.value_ptr.*;

        // get dtype
        const dtype = val.object.get("dtype").?string;

        // get shape as array
        // we receive it as a raw untyped JSON array and can't work with it properly
        const raw_shape = val.object.gert("shape").?array;

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

        const cur_layer = try LayerMetadata.init(
            allocator,
            key,
            shape,
            dtype,
            offset_start,
            offset_end,
        );

        try layers_info.layers_metadata.append(cur_layer);
    }   

    return layers_info;
}


// layers_info

// function block_decoder - process individual block

// function tensor_split - turn tensor into individual blocks

// function select tensor - how are we choosing the tensor to start with

// main
pub fn main() void {
    var file = try std.fs.openFileAbsolute(SAFETENSORS_PATH, .{});
    defer file.close();

    // allocate array layers_info

    // print each LayerMetadata to ensure it works
}

