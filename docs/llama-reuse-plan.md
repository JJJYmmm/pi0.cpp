# llama.cpp Reuse Plan

This branch now carries `third_party/llama.cpp` as a real submodule. With the
current default CMake settings, `VLACPP_USE_LLAMA_CPP=ON` builds and links the
core `ggml` and `llama` targets into `vlacpp`.

Current verified boundary:

- `ggml`, `ggml-base`, `ggml-cpu`, and `llama` are visible CMake targets.
- `tools/mtmd/clip.cpp`, `tools/mtmd/clip-graph.h`, and
  `tools/mtmd/models/siglip.cpp` are present in the submodule.
- `mtmd` is not a current build target because vlacpp disables llama.cpp common
  tools by default to keep the runtime build small.

Implication for pi0:

- Use the linked `ggml` target for the first C++ graph builder.
- Reuse the mtmd SigLIP/ViT source as the implementation reference for the image
  encoder. Either wire the `mtmd` target explicitly later, or port the necessary
  graph construction pieces into `src/models/pi0.cpp` while preserving the
  existing vlacpp C ABI.
- Keep OpenPI-specific observation preprocessing, action head wiring, flow
  sampling, and policy comparison in this repo.

Check the boundary with:

```sh
python3 tools/check-llama-components.py \
  --repo . \
  --build-dir build \
  --require-ggml-targets
```
