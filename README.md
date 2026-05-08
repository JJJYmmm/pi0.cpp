# vlacpp

`vlacpp` is a llama.cpp/ggml-style C/C++ runtime scaffold for
vision-language-action models. The v1 target is pi0/pi0.5 inference with a
GGUF-based model format, reusable prefix KV cache, and a small C ABI that can be
wrapped by Python, ROS, or a policy server.

This repository currently contains the runtime skeleton, a deterministic
`mock-pi0` metadata path, and a tiny tensor-backed pi0 GGUF path. The tiny path
validates model loading, preprocessing, model forward, flow sampling, and CLI
behavior before full openpi tensor conversion is wired in.

## Layout

- `include/vlacpp.h`: stable C API.
- `src/core`: context, errors, metadata parsing, preprocessing.
- `src/models`: model registry and pi0/pi0.5 entry point.
- `src/sampling`: flow-matching sampler abstraction.
- `tools/convert-openpi-to-gguf.py`: openpi metadata/conversion entry point.
- `examples/pi0-cli`: command-line smoke inference.
- `third_party/llama.cpp`: intended llama.cpp submodule location.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The tree builds without `third_party/llama.cpp` for early development. Add the
submodule when wiring real ggml graphs:

```sh
git submodule add https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp
git submodule update --init --recursive
cmake -S . -B build -DVLACPP_USE_LLAMA_CPP=ON
```

## Smoke Inference

Create a mock metadata file:

```sh
python3 tools/convert-openpi-to-gguf.py \
  --output /tmp/mock-pi0.json \
  --state-dim 3 \
  --action-dim 2 \
  --action-horizon 4
```

Run the CLI:

```sh
./build/vlacpp-pi0 --model /tmp/mock-pi0.json --state 1,2,3 --prompt "pick up the fork"
```

Create a tiny pi0 GGUF from the checked-in reference checkpoint and compare it
against the Python OpenPI-style reference:

```sh
python3 tools/convert-openpi-to-gguf.py \
  --checkpoint tests/tiny-openpi-checkpoint.json \
  --output /tmp/tiny-openpi.gguf

./build/vlacpp-pi0 \
  --model /tmp/tiny-openpi.gguf \
  --state 1,2,3 \
  --prompt "pick up the fork" \
  --steps 4 \
  --seed 1

python3 tools/compare-openpi-reference.py \
  --checkpoint tests/tiny-openpi-checkpoint.json \
  --model /tmp/tiny-openpi.gguf \
  --steps 4 \
  --seed 1
```

When the official OpenPI package and a real checkpoint are installed, compare
against the OpenPI policy API directly:

```sh
python3 tools/compare-openpi-policy.py \
  --openpi-config pi05_libero \
  --openpi-checkpoint /path/to/openpi/checkpoint \
  --vlacpp-model /path/to/converted.gguf \
  --state 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
  --image-key image \
  --steps 10 \
  --seed 1
```

The OpenPI policy comparison is not registered in default CTest because the
official checkpoints are large and require OpenPI's Python/GPU environment. By
default this harness requires `vlacpp-pi0 --info` to report `full-openpi`; use
`--allow-restricted-vlacpp` only for local harness tests that intentionally
compare a subset model.

The converter also accepts small Hugging Face JSON assets with
`hf://owner/repo/path/to/config.json`; for example, it can download the pi0.5
Libero config and emit a tiny GGUF scaffold with matching action shape:

```sh
python3 tools/convert-openpi-to-gguf.py \
  --checkpoint hf://Tacoin/openpi-pi0.5-libero-onnx/checkpoints/pi05_libero_pytorch/config.json \
  --norm-stats hf://Tacoin/openpi-pi0.5-libero-onnx/assets/physical-intelligence/libero/norm_stats.json \
  --output /tmp/hf-pi05-tiny.gguf \
  --model-type pi05 \
  --state-dim 32 \
  --init-tiny
```

Small `.safetensors` checkpoints are also accepted when they contain
`pi0.velocity.weight` and `pi0.velocity.time_weight`. Metadata can be embedded
as `vlacpp.metadata` in the safetensors header or passed separately with
`--config`.

Inspect safetensors headers without downloading full remote files:

```sh
python3 tools/inspect-safetensors.py \
  hf://maxqualia/openpi-pi0-corkinbox100-1882950e/model.safetensors \
  --contains action \
  --limit 20
```

ModelScope pi0 mirrors can be addressed with `ms://`:

```sh
python3 tools/inspect-safetensors.py \
  ms://lerobot/pi0/model.safetensors \
  --contains action \
  --limit 20
```

The converter also range-reads remote `hf://...safetensors`,
`ms://...safetensors`, and HTTPS headers and required tensor payloads instead of
downloading multi-GB files up front. The same manifest flow also works for local
safetensors files, which keeps the
OpenPI-name-to-vlacpp-name mapping path covered in default tests without
network access. If a safetensors header contains `vlacpp.metadata`,
`map-openpi-tensors.py` preserves it in the manifest so
`convert-openpi-to-gguf.py --tensor-map-manifest` can infer model dimensions
without repeating them on the command line. For OpenPI safetensors without that
metadata, the converter infers `action_dim` from action projection tensors and
`state_dim` from `state_proj.weight` when present; `action_horizon` still has to
come from config or a command-line argument.

When a full local payload is needed, keep it out of git under `ckpts/`:

```sh
python3 tools/download-remote-file.py \
  ms://lerobot/pi0/model.safetensors \
  --output ckpts/lerobot-pi0/model.safetensors
```

Generate a mapping manifest for the real OpenPI state projection and action
head tensors:

```sh
python3 tools/map-openpi-tensors.py \
  hf://maxqualia/openpi-pi0-corkinbox100-1882950e/model.safetensors \
  --family action-expert \
  --require-complete \
  --output /tmp/openpi-action-map.json
```

Convert only the mapped state/action-head tensors into a GGUF shard. This
range-reads the mapped tensors and converts BF16 payloads to F32 GGUF tensors:

```sh
python3 tools/convert-openpi-to-gguf.py \
  --checkpoint hf://maxqualia/openpi-pi0-corkinbox100-1882950e/model.safetensors \
  --tensor-map-manifest /tmp/openpi-action-map.json \
  --output /tmp/openpi-action-head.gguf \
  --model-type pi0 \
  --action-horizon 32
```

The runtime can load that action-head shard and use it as a restricted velocity
path for smoke testing:

```sh
./build/vlacpp-pi0 \
  --model /tmp/openpi-action-head.gguf \
  --state 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
  --prompt "pick up the fork" \
  --steps 1 \
  --seed 1
```

This exercises OpenPI action projection, timestep MLP, action output projection,
and flow sampling, but it still omits the PaliGemma/SigLIP/Gemma transformer
backbone.

Check which runtime subset a model will use:

```sh
./build/vlacpp-pi0 --model /tmp/openpi-action-head.gguf --info
```

Current capability strings include `mock-pi0`, `tiny-velocity`,
`restricted-pi0-state-action-head`, and `restricted-pi05-action-head`.
Use `--require-full` in scripts that must fail unless the model is backed by a
future full OpenPI runtime:

```sh
./build/vlacpp-pi0 --model /tmp/openpi-action-head.gguf --require-full
```

Run the restricted action-head path end to end with one command:

```sh
python3 tools/run-action-head-smoke.py \
  --binary build/vlacpp-pi0 \
  --work-dir /tmp/vlacpp-action-head-smoke
```

For the pi0.5 action/time-head subset:

```sh
python3 tools/run-action-head-smoke.py \
  --family pi05 \
  --binary build/vlacpp-pi0 \
  --work-dir /tmp/vlacpp-pi05-action-head-smoke
```

The pi0.5 smoke uses the Tacoin repo's HF config and norm_stats by default;
override them with `--config` and `--norm-stats` when testing another export.

Default CTest coverage includes local tiny velocity conversion/compare,
safetensors inspect/map/convert/compare, fake OpenPI policy API comparison, and
a small pi0 action-head GGUF inspect/CLI smoke/reference comparison based on
`tests/action-head-checkpoint.json`. It also covers an OpenPI-named local
action-head safetensors fixture through `map-openpi-tensors.py` and
`--tensor-map-manifest`, plus a restricted pi0.5 action-head CLI
smoke/reference comparison based on `tests/pi05-action-head-checkpoint.json`
including an F16 safetensors conversion fixture. It verifies mapped GGUF
metadata and norm_stats with `inspect-gguf.py`, compares the norm-stats
action-head runtime path against the Python reference, runs a local
`run-action-head-smoke.py` map/convert/inspect/CLI chain from OpenPI-named
safetensors, validates tensor inventory reporting, and verifies that
`--require-full` rejects restricted runtime subsets. The pi0.5 path is covered
both from vlacpp-named fixtures and from OpenPI-named safetensors through the
`pi05-action-expert` tensor map.

`map-openpi-tensors.py --include-inventory` adds top-level `family`,
`expected_count`, `mapped_count`, and `coverage` fields, plus group-level
mapped/unmapped counts for planning the remaining full-model tensor map.
Current real HF action-head maps cover 10/777 pi0 tensors and 8/812 pi0.5
tensors; the unmapped tensors are the `paligemma_with_expert` backbone.
Inventory subgroup summaries normalize repeated layer indices so the remaining
backbone work is visible at block granularity, for example pi0 splits into
`paligemma_with_expert.paligemma.model` and
`paligemma_with_expert.gemma_expert.model.layers.*`.
Use `tools/summarize-tensor-map.py` to inspect an inventory manifest or assert
expected coverage in automation.
`--family all` emits an identity mapping for every tensor in the safetensors
header, which is useful for full-checkpoint GGUF conversion experiments; runtime
support still depends on implementing the corresponding graph.
`--family pi0-full` and `--family pi05-full` also include every tensor from the
header, but alias the currently supported action/state head tensors to the
runtime's `vlacpp.openpi.*` names. This lets a full-checkpoint GGUF preserve
backbone tensors for future ggml graph work while still exercising the
implemented restricted action-head path.
Use `tools/summarize-openpi-graph.py` on a full manifest or remote safetensors
source to extract the graph dimensions needed by a llama.cpp/ggml implementation:
action width/dimension, vision tower layers and patch size, PaliGemma language
layers, and action expert layers. The converter writes the inferred values it
can see into `vlacpp.openpi.*` GGUF metadata keys so the eventual C++ graph
builder can consume dimensions without rescanning the original checkpoint.
`vlacpp_model_openpi_graph_info()` and `vlacpp-pi0 --info` expose the parsed
metadata from the loaded model.

See `docs/pi0-infer-audit.md` for the current prompt-to-artifact completion
audit and the remaining gaps before full OpenPI parity.

## Reference Architecture

Implementation should track these upstream designs:

- openpi pi0/pi0.5: image/language prefix prefill, suffix action expert,
  flow-matching denoising, and prefix KV cache reuse.
- StarVLA: keep raw sample boundaries explicit and model components pluggable;
  future model families should register through the same model vtable instead
  of coupling preprocessing, backbone, and action head.
- llama.cpp: follow its converter/graph split where OpenPI-specific code
  parses config and tensor names, then uses GGUF writer conventions and ggml
  graph builders. For the real graph, prefer reusing existing llama.cpp/ggml
  model components such as SigLIP-family multimodal encoders and Gemma-style
  transformer blocks before adding vlacpp-local kernels.

## Next Engineering Steps

1. Refactor `tools/convert-openpi-to-gguf.py` toward the llama.cpp converter
   pattern: OpenPI config/tensor mapping in this repo, GGUF writing through a
   reusable writer module instead of converter-local container emission. The
   current writer is `tools/gguf_writer.py`; it is intentionally small so it can
   later be replaced by, or aligned more closely with, llama.cpp's GGUF writer.
2. Expand openpi tensor collection from the tiny velocity/action-head subsets to
   full pi0/pi0.5 manifests that preserve backbone tensors and runtime aliases.
3. Wire SigLIP/PaliGemma/Gemma execution through ggml/llama.cpp-style graph
   components under `src/models/pi0.cpp`.
4. Add parity tests against full openpi checkpoints for selected intermediate
   tensors and final action chunks.
