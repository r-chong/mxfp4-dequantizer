const std = @import("std");
const layer = @import("layer.zig");
const safetensors = @import("safetensors.zig");
const gpt_oss = @import("gpt_oss.zig");
const quant = @import("quantized_tensor.zig");

const SAFETENSORS_PATH ="";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer switch (gpa.deinit()) {
        .leak => std.debug.panic("Leaked", .{}),
        .ok => {},
    };
    const allocator = gpa.allocator();

    if (SAFETENSORS_PATH.len == 0 or !std.fs.path.isAbsolute(SAFETENSORS_PATH)) {
        std.debug.panic("Please update SAFETENSORS_PATH in main.zig with the absolute path to your safetensors file.\n", .{});
    }

    var file = std.fs.openFileAbsolute(SAFETENSORS_PATH, .{}) catch {
        std.debug.panic("Could not open file at path: {s}\nPlease verify the SAFETENSORS_PATH in main.zig is correct.\n", .{SAFETENSORS_PATH});
    };
    defer file.close();

    var tensor_list = try safetensors.get_safetensors_content(allocator, &file);
    defer tensor_list.deinit();

    const lyr: layer.Layer = .{
        .block_idx = 22,
        .kind = .Mlp1WeightQuant,
    };

    var q = try gpt_oss.load_quantized_tensor(allocator, lyr, tensor_list, &file);
    defer q.deinit();

    var reader = quant.TensorReader.init(&q);
    defer reader.deinit();

    var sample: [16]f32 = undefined;
    const written_bytes = reader.read(std.mem.asBytes(&sample));
    const written_vals = written_bytes / @sizeOf(f32);

    std.debug.print("Written: {d}\n", .{written_bytes});
    for (sample[0..written_vals], 0..) |v, i| {
        std.debug.print("  [{d}] = {d}\n", .{ i, v });
    }
}
