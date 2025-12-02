# MXFP4 Dequantizer in Zig

A std.io.Reader that takes MXFP4 bytes from a safetensors file and outputs a stream of dequantized float bytes.
When I'm done it should interface like:

```zig
var buf: [N]f32 = undefined;
const n_read = try dequantizer.read(std.mem.asBytes(&buf));
```

# Steps to run:
0. Create venv of choice
1. Download gpt-oss for weights
```
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
pip install gpt-oss
python -m gpt_oss.chat model/
```
2. Interface the dequantizer (WIP)

References:
https://huggingface.co/docs/safetensors/v0.3.2/en/metadata_parsing
https://yobibyte.github.io/safetensors.html
https://huggingface.co/openai/gpt-oss-20b




// TODO: do get_safetensors_content without an allocator
// use a fixed-capacity representation for shapes
// make LayerMetadata not depend on heap-allocated []u64 shapes (use fixed [MAX_DIMS]u64 + rank).