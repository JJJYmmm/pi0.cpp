# llama.cpp Reuse Plan

This branch now carries `third_party/llama.cpp` as a required submodule. CMake
fails fast if the submodule is missing. The default build links `ggml`, `llama`,
and the public multimodal `mtmd` library into `vlacpp`.

Current verified boundary:

- `ggml`, `ggml-base`, `ggml-cpu`, `llama`, and `mtmd` are visible CMake targets.
- `tools/mtmd/clip.cpp`, `tools/mtmd/clip-graph.h`, and
  `tools/mtmd/models/siglip.cpp` are present in the submodule.
- `vlacpp` links the public `mtmd` target directly; `vlacpp-mtmd-api` verifies
  the default media marker and context params at runtime.
- `src/models/pi0.cpp` directly builds narrow F32 `ggml_mul_mat + ggml_add`
  graphs for single and batched linear layers; the pi0/pi0.5 restricted
  action-head forward path now runs the action horizon through ggml instead of
  a local matmul loop.
- `tools/gguf_writer.py` uses llama.cpp's `gguf-py` `GGUFWriter` instead of a
  local hand-written GGUF serializer.
- Converted GGUF 2D tensor shapes use ggml `ne` ordering (`ne0` is the fastest
  varying/input dimension, `ne1` is the output row count). OpenPI source tensor
  metadata remains `[out, in]`, but the runtime-facing GGUF shape is
  `[in, out]`.

Implication for pi0:

- Use the linked `ggml` and `mtmd` targets for the first C++ graph builder.
- Reuse the mtmd SigLIP/ViT source and public library boundary for the image
  encoder while preserving the existing vlacpp C ABI.
- Keep conversion on the llama.cpp `gguf-py` writer path and add only
  OpenPI-specific tensor naming/metadata logic in this repo.
- Keep OpenPI-specific observation preprocessing, action head wiring, flow
  sampling, and policy comparison in this repo.

Check the boundary with:

```sh
python3 tools/check-llama-components.py \
  --repo . \
  --build-dir build \
  --require-ggml-targets \
  --require-mtmd-target
python3 tools/check-openpi-llama-reuse.py \
  tests/openpi-graph-manifest.json \
  --repo .
```
