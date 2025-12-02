const std = @import("std");

// fmt is format string
// args is tuple of placeholders. see {s}
pub fn main() void {
    std.debug.print("Hello {s}!\n", .{"World"});
}
