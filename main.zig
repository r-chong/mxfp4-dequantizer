const std = @import("std");

// 8 bytes is the size of a safetensors json header (u64)
const HEADER_SIZE_BUF_SIZE = 8;
const SAFETENSORS_PATH = "./gpt-oss-20b/original/model.safetensors";

// LayerMetadata struct

// layers_info

// function get_safetensors_content - get header + raw tensor data (weights)

// function block_decoder - process individual block

// function tensor_split - turn tensor into individual blocks

// function select tensor - how are we choosing the tensor to start with
