# LIBERO Simulator Evaluation

This document describes how to evaluate a vlacpp GGUF policy in the LIBERO
simulator. The simulator stack is external to the core runtime and conversion
tools.

## Prerequisites

Set `MODEL_DIR` to a pi0 LIBERO checkpoint directory containing
`model.safetensors` and `config.json`:

```sh
MODEL_DIR=/path/to/pi0-libero
mkdir -p artifacts/eval
```

Create the GGUF model:

```sh
python3 tools/map-openpi-tensors.py "$MODEL_DIR/model.safetensors" \
  --family pi0-full \
  --include-inventory \
  --output artifacts/pi0-libero-map.json

python3 tools/convert-openpi-to-gguf.py \
  --tensor-map-manifest artifacts/pi0-libero-map.json \
  --config "$MODEL_DIR/config.json" \
  --output artifacts/pi0-libero.gguf \
  --model-type pi0 \
  --action-horizon 50 \
  --state-dim 32 \
  --action-dim 32 \
  --image-width 224 \
  --image-height 224

./build/vlacpp-pi0 \
  --model artifacts/pi0-libero.gguf \
  --state "$(python3 -c 'print(",".join(["0"] * 32))')" \
  --prompt "pick up the fork" \
  --steps 10 \
  --seed 1
```

## Environment

Use a LeRobot/LIBERO environment compatible with the checkpoint:

- `lerobot==0.4.4`
- `transformers==4.53.3`
- `torch==2.10.0+cu128`
- `libero==0.1.0`
- `robosuite==1.4.1`
- `mujoco==2.3.7`

Benchmark hardware used in `reports/vlacpp_v1_performance.md`: Intel Xeon
Platinum 8358P and NVIDIA A100-PCIE-40GB.

## Rollout Procedure

1. Load the LeRobot policy config from the original checkpoint directory.
2. Create a `LiberoEnv` suite, usually `libero_object` with task ids `0..9`.
3. Build LeRobot pre/postprocessors on CPU with `compile_model=False`.
4. Build `vlacpp.Pi0Policy` from `artifacts/pi0-libero.gguf`.
5. For each episode, preprocess the observation with LeRobot, pass padded state,
   camera images, prompt text or prompt tokens, and optional fixed noise into
   `policy.infer`.
6. Execute the first `n_action_steps` actions through the LeRobot environment
   postprocessor and simulator.
7. Record success rate and chunk timing. Exclude the first warmup chunk for
   runtime comparisons.

Suggested output fields are:

- `success_rate`
- `episodes`
- `task_suite_name`
- `task_ids`
- `backend`
- `flow_steps`
- `n_action_steps`
- `chunk_infer_time_excluding_prefix_s`
- `chunk_policy_e2e_time_excluding_prefix_s`

Keep large rollout logs and simulator videos outside this repository, for
example under `artifacts/eval/`.
