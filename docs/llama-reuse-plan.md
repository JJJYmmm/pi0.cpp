# llama.cpp Reuse Plan

This branch now carries `third_party/llama.cpp` as a required submodule. CMake
fails fast if the submodule is missing. The default build links `ggml`, `llama`,
and the public multimodal `mtmd` library into `vlacpp`.

Current verified boundary:

- `ggml`, `ggml-base`, `ggml-cpu`, `llama`, and `mtmd` are visible CMake targets.
- `tools/mtmd/clip.cpp`, `tools/mtmd/clip-graph.h`, and
  `tools/mtmd/models/siglip.cpp` are present in the submodule.

Implication for pi0:

- Use the linked `ggml` and `mtmd` targets for the first C++ graph builder.
- Reuse the mtmd SigLIP/ViT source and public library boundary for the image
  encoder while preserving the existing vlacpp C ABI.
- Keep OpenPI-specific observation preprocessing, action head wiring, flow
  sampling, and policy comparison in this repo.

Check the boundary with:

```sh
python3 tools/check-llama-components.py \
  --repo . \
  --build-dir build \
  --require-ggml-targets \
  --require-mtmd-target
```
