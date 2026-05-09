# vlacpp

> Note: this is an experimental repo. It currently only completes pi0 C++
> inference bring-up. On the current LIBERO v1 benchmark, vlacpp measures about
> `0.147-0.149 s` per warm CUDA action chunk and `5.919 s` per warm CPU action
> chunk; it does not beat checkpoint-compiled LeRobot CUDA. This project is for
> learning and experimentation only, not production use.

`vlacpp` is a small C++17 VLA runtime built around llama.cpp/ggml components.
The v1 scope is pi0-style GGUF inference, a stable C ABI, a Python ctypes
wrapper, and one LIBERO benchmark entry point.

## Layout

- `include/vlacpp.h`: public C ABI.
- `src/core`: context lifecycle, GGUF/JSON metadata, preprocessing, errors.
- `src/models`: pi0 VLM prefix, mtmd vision path, action decoder, tokenizer.
- `src/sampling`: flow sampler.
- `python/vlacpp`: Python API wrapping the C ABI.
- `examples/pi0-cli`: minimal C++ CLI.
- `tools/convert-openpi-to-gguf.py`: OpenPI/LeRobot checkpoint conversion.
- `tools/eval-libero-sim-vlacpp-lerobot-env.py`: v1 LIBERO benchmark runner.
- `reports/`: benchmark report and machine-readable summary.

Generated build directories, checkpoints, and downloaded datasets should stay
out of git under `build*/`, `ckpts/`, and `data/`.

## Build

Initialize llama.cpp first:

```sh
git submodule update --init --recursive
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For CUDA, configure the build with the same llama.cpp CUDA options used in the
local environment, then pass the resulting `libvlacpp.so` to the Python runner.

## Convert A Checkpoint

The runtime loads GGUF. Convert a prepared OpenPI/LeRobot pi0 checkpoint with:

```sh
python3 tools/convert-openpi-to-gguf.py \
  --checkpoint ckpts/pepijn-pi0-libero-finetuned-extra2/model.safetensors \
  --config ckpts/pepijn-pi0-libero-finetuned-extra2-v044-local/config.json \
  --output /tmp/vlacpp-pi0-libero-finetuned-v044.gguf \
  --model-type pi0 \
  --action-horizon 50
```

The v1 pi0 path expects the matching sidecars when using the full VLM prefix:

- `/tmp/vlacpp-pi0-libero-finetuned-v044.vision-mtmd.gguf`
- `/tmp/vlacpp-pi0-libero-finetuned-v044.tokenizer.gguf`

## C++ Interface

Use the C ABI from `include/vlacpp.h`. The CLI in `examples/pi0-cli/main.cpp`
is the shortest complete example:

```sh
./build/vlacpp-pi0 \
  --model /tmp/vlacpp-pi0-libero-finetuned-v044.gguf \
  --state 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 \
  --prompt "pick up the fork" \
  --steps 10 \
  --seed 1
```

Programmatic use follows this lifecycle:

1. `vlacpp_load_model`
2. `vlacpp_create_context`
3. `vlacpp_infer_actions`
4. `vlacpp_free_action_chunk`
5. `vlacpp_free_context` / `vlacpp_free_model`

Call `vlacpp_reset_cache` whenever the visual/text prefix changes.

## Python Interface

```python
import numpy as np
import vlacpp

policy = vlacpp.Pi0Policy(
    "/tmp/vlacpp-pi0-libero-finetuned-v044.gguf",
    library_path="build-cuda/libvlacpp.so",
    backend=vlacpp.VLACPP_BACKEND_CUDA,
    seed=1000,
    flow_steps=10,
)

actions = policy.infer(
    state=np.zeros(32, dtype=np.float32),
    images={
        "base_0_rgb": np.zeros((256, 256, 3), dtype=np.uint8),
        "left_wrist_0_rgb": np.zeros((256, 256, 3), dtype=np.uint8),
    },
    prompt="pick up the tomato sauce and place it in the basket\n",
    prompt_tokens=np.array([2, 18075, 908, 573, 32493, 16009, 578, 2040, 665, 575, 573, 12220, 108], dtype=np.int32),
)
policy.close()
```

The Python wrapper returns an `H x action_dim` `float32` NumPy array.

## LIBERO Benchmark

The v1 benchmark runner uses the matching LeRobot v0.4.4 preprocessing,
tokenizer, task prompts, and simulator environment, then calls vlacpp for action
chunks:

```sh
LIBERO_CONFIG_PATH=/tmp/lerobot-v044-libero-config \
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TOKENIZERS_PARALLELISM=false \
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 \
/tmp/lerobot-pi-v044-venv/bin/python \
  tools/eval-libero-sim-vlacpp-lerobot-env.py \
  --vlacpp-model /tmp/vlacpp-pi0-libero-finetuned-v044.gguf \
  --vlacpp-library build-cuda/libvlacpp.so \
  --backend cuda \
  --num-trials-per-task 5 \
  --output /tmp/vlacpp-libero-cuda.json
```

Summarize benchmark evidence with:

```sh
python3 tools/summarize-libero-benchmark.py \
  --vlacpp-cuda /tmp/vlacpp-lerobot-env-alltasks-n5-currentcode-cuda.json /tmp/vlacpp-lerobot-env-alltasks-n5-offset5-currentcode-cuda.json \
  --baseline-cuda /tmp/lerobot-native-controlled-noise-alltasks-n5-compileconfig-current.json /tmp/lerobot-native-controlled-noise-alltasks-n5-offset5-compileconfig-current.json \
  --vlacpp-cpu /tmp/vlacpp-lerobot-env-alltasks-n5-cpu-backend-cache.json \
  --output /tmp/vlacpp-libero-summary.json
```

## Performance Report

The current v1 performance report is
`reports/vlacpp_v1_performance.md`. The detailed bring-up log remains in
`reports/pi0_libero_benchmark.md`, and
`reports/pi0_libero_benchmark_summary.json` contains the final machine-readable
status.
