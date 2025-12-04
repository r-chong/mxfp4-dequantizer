# MXFP4 Dequantizer in Zig

A std.io.Reader that takes MXFP4 bytes from a safetensors file and outputs a stream of dequantized float bytes.
When I'm done it should interface like:

```zig
var buf: [N]f32 = undefined;
const n_read = try dequantizer.read(std.mem.asBytes(&buf));
```

Please excuse a lack of formality in my comments and explanations of how things work. Also, feel free to open a PR if you have revisions

# Steps to run:
0. Create venv of choice
1. Download gpt-oss for weights
```
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
pip install gpt-oss
python -m gpt_oss.chat model/
```
2. Interface the dequantizer (WIP)

# Key Terminology

A **tensor** is an array. Ours contain model weights in FP4 format.

A **layer** is a semantic grouping of tensors. It comes from neural network layers.

**Microscaling FP4** (abbrev. MXFP4) is a custom 4-bit quantization format where every TWO FP4 values are packed into one byte, and the number of values depends entirely on the tensor’s shape. This is the quantization format assumed in this program.

**Dequantization** is the reverse of quantization, where we go from low precision -> high precision.

### Quantization
Quantization is a way to "compress" trained AI model weights into smaller file sizes, by representing the higher precision weights in a lower precision data type.

Let's say our higher precision data type is FP16, and our lower precision data type is MXFP4.
Currently, in high-precision, the data is a weight matrix but it is stored linearly. So,

```
W = [
  [a, b, c, d],
  [e, f, g, h],
  [i, j, k, l],
  ...
]
```

is stored as

```
[a, b, c, d, e, f, g, h, i, j, k, l, ...]
```

The rows are side by side instead of on top of each other.

Our matrix is very large, so we split it into "blocks" of length 32 (as per MXFP4).

Our goal is to constrain all values in the block to a smaller length [-a, a], centered around 0. To shrink an interval, one needs to divide by that interval length in order to get into the bounds [-a, a]. Also, instead of -a and a, they're actually called q_min and q_max.

So, for each block, we calculate a scaling value - the value that we divided each of the block values by, so we know what value to multiply by to return to the original weights (or close). We store this value in the block.

Our model weights are stored in tensors alongside a format called Safetensors, which provides JSON information about each tensor's configuration.

# What this program does

In this program, we read the Safetensors JSON header, and then follow pointers to the location of the first tensor. 

There, we decode the lower precision values by multiplying the scaling factor by the FP4 data.

We repeat this process for all tensors.

This process is accelerated using the Single Instruction, Multiple Data (SIMD) technique - which is basically using vectors to do operations at once instead of in sequences.

This program outputs to a stream, the "decompressed" FP4 values in a higher precision form (haven't decided between F32 or BF16 yet).

# References:
- https://huggingface.co/docs/optimum/en/concept_guides/quantization
- https://huggingface.co/docs/safetensors/v0.3.2/en/metadata_parsing
- https://yobibyte.github.io/safetensors.html
- https://huggingface.co/openai/gpt-oss-20b
- https://huggingface.co/blog/RakshitAralimatti/learn-ai-with-me

The above (codebase included) was written without AI.

# TODO:
- get_safetensors_content without an allocator
- use a fixed-capacity representation for shapes
- make LayerMetadata not depend on heap-allocated []u64 shapes (use fixed [MAX_DIMS]u64 + rank).