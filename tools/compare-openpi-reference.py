#!/usr/bin/env python3
"""Compare vlacpp pi0 output against the tiny OpenPI-style reference path."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def resolve_checkpoint(checkpoint: str) -> Path:
    if checkpoint.startswith("hf://"):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("hf:// checkpoints require huggingface_hub") from exc
        repo_and_file = checkpoint[len("hf://") :]
        parts = repo_and_file.split("/", 2)
        if len(parts) != 3:
            raise SystemExit("hf:// checkpoints must look like hf://owner/repo/path/to/checkpoint.json")
        return Path(hf_hub_download(repo_id=f"{parts[0]}/{parts[1]}", filename=parts[2]))
    return Path(checkpoint)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise SystemExit("safetensors checkpoints require the safetensors Python package") from exc

        metadata: dict[str, Any] = {}
        tensors: dict[str, dict[str, Any]] = {}
        with safe_open(path, framework="np") as handle:
            raw_metadata = handle.metadata() or {}
            if "vlacpp.metadata" in raw_metadata:
                metadata = json.loads(raw_metadata["vlacpp.metadata"])
            for name in handle.keys():
                if name not in {"pi0.velocity.weight", "pi0.velocity.time_weight"}:
                    continue
                array = handle.get_tensor(name)
                tensors[name] = {
                    "shape": list(array.shape),
                    "data": array.astype("float32").reshape(-1).tolist(),
                }
        return {"metadata": metadata, "tensors": tensors}
    raise SystemExit("reference comparison expects a tiny JSON or safetensors checkpoint")


def default_tiny_weights(action_dim: int, state_dim: int) -> dict[str, dict[str, Any]]:
    feature_dim = state_dim + 3
    weights: list[float] = []
    for col in range(action_dim):
        phase = (col + 1) / max(1, action_dim)
        row = [0.01 * phase]
        row.extend(0.50 / max(1, state_dim) for _ in range(state_dim))
        row.extend([0.25, 0.25])
        weights.extend(row)
    return {
        "pi0.velocity.weight": {"shape": [action_dim, feature_dim], "data": weights},
        "pi0.velocity.time_weight": {"shape": [action_dim], "data": [0.001] * action_dim},
    }


def cpp_mt19937_normals(seed: int, count: int) -> list[float]:
    code = f"""
#include <iostream>
#include <random>
int main() {{
    std::mt19937 rng({seed}u);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int i = 0; i < {count}; ++i) {{
        if (i) std::cout << " ";
        std::cout << normal(rng);
    }}
}}
"""
    source = Path("/tmp/vlacpp-reference-rng.cpp")
    binary = Path("/tmp/vlacpp-reference-rng")
    source.write_text(code, encoding="utf-8")
    subprocess.run(["c++", "-std=c++17", str(source), "-o", str(binary)], check=True)
    raw = subprocess.check_output([str(binary)], text=True)
    return [float(v) for v in raw.split()]


def reference_actions(checkpoint: dict[str, Any], state: list[float], prompt: str, steps: int, seed: int) -> list[float]:
    metadata = checkpoint.get("metadata", {})
    action_dim = int(metadata.get("action_dim", 2))
    horizon = int(metadata.get("action_horizon", 4))
    state_dim = int(metadata.get("state_dim", len(state)))
    tensors = checkpoint.get("tensors") or default_tiny_weights(action_dim, state_dim)
    weight_tensor = tensors["pi0.velocity.weight"]
    weight = weight_tensor["data"]
    time_weight = tensors["pi0.velocity.time_weight"]["data"]

    image_mean = 127.0 / 255.0 * 2.0 - 1.0
    if "state_mean" in metadata or "state_std" in metadata:
        state_offset = metadata.get("state_mean")
        state_scale = metadata.get("state_std")
        if len(state_offset) != len(state) or len(state_scale) != len(state):
            raise SystemExit("state_mean/state_std must match input state length")
        state = [(value - mean) / scale for value, mean, scale in zip(state, state_offset, state_scale)]
    state_mean = sum(state) / len(state) if state else 0.0
    prompt_signal = (len(prompt) % 97) / 97.0
    full_features = [1.0, *state, image_mean, prompt_signal]
    legacy_features = [1.0, state_mean, image_mean, prompt_signal]
    feature_dim = int(weight_tensor["shape"][1])
    features = full_features if feature_dim == len(full_features) else legacy_features
    x = cpp_mt19937_normals(seed, horizon * action_dim)
    n_steps = max(1, steps)
    time = 1.0
    dt = -1.0 / n_steps
    for _ in range(n_steps):
        for i in range(len(x)):
            col = i % action_dim
            target = sum(weight[col * feature_dim + j] * features[j] for j in range(feature_dim))
            velocity = x[i] - target + time * time_weight[col]
            x[i] += dt * velocity
        time += dt
    if "action_mean" in metadata or "action_std" in metadata:
        action_offset = metadata.get("action_mean")
        action_scale = metadata.get("action_std")
        if len(action_offset) != action_dim or len(action_scale) != action_dim:
            raise SystemExit("action_mean/action_std must match action_dim")
        for i in range(len(x)):
            col = i % action_dim
            x[i] = x[i] * action_scale[col] + action_offset[col]
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/vlacpp-pi0"))
    parser.add_argument("--state", default="1,2,3")
    parser.add_argument("--prompt", default="pick up the fork")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()

    state = [float(v) for v in args.state.split(",") if v]
    checkpoint = load_checkpoint(resolve_checkpoint(args.checkpoint))
    expected = reference_actions(checkpoint, state, args.prompt, args.steps, args.seed)
    actual_json = subprocess.check_output(
        [
            str(args.binary),
            "--model",
            str(args.model),
            "--state",
            args.state,
            "--prompt",
            args.prompt,
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
        ],
        text=True,
    )
    actual = json.loads(actual_json)["actions"]
    max_abs = max(abs(a - b) for a, b in zip(actual, expected))
    if max_abs > args.atol or len(actual) != len(expected):
        raise SystemExit(f"compare failed: max_abs={max_abs:g} len={len(actual)} expected_len={len(expected)}")
    print(json.dumps({"status": "ok", "max_abs": max_abs, "count": len(actual)}, indent=2))


if __name__ == "__main__":
    main()
