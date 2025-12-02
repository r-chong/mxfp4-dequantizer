# MXFP4 Dequantizer in Zig

Steps to run:
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
