const std = @import("std");

// layer member accesses
// Extracted from GPT-OSS files: model.safetensors and dtypes.json
pub const LayerKind = enum {
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

pub const Layer = struct {
    block_idx: usize,
    kind: LayerKind,
};

pub fn kindToStr(kind: LayerKind) []const u8 {
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

// base name is the JSON key
pub fn makeBaseName(allocator: std.mem.Allocator, layer: Layer) ![]u8 {
    return try std.fmt.allocPrint(allocator, "block.{d}.{s}", .{ layer.block_idx, kindToStr(layer.kind) });
}

// these are the only weights that are quantized
pub const QUANT_LAYER_KINDS = [_]LayerKind{
    .Mlp1WeightQuant,
    .Mlp2WeightQuant,
};
