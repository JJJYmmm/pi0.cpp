#!/usr/bin/env python3
"""Compare vlacpp actions against an installed official OpenPI policy.

This is intentionally not part of the default CTest suite: official OpenPI
checkpoints are large and the OpenPI runtime has substantial Python/GPU
dependencies. The script is the parity harness for real checkpoints once a
converted GGUF and the corresponding OpenPI checkpoint are available locally.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def openpi_available() -> bool:
    return importlib.util.find_spec("openpi") is not None


def parse_state(text: str) -> np.ndarray:
    return np.asarray([float(v) for v in text.split(",") if v], dtype=np.float32)


def cpp_mt19937_normals(seed: int, count: int) -> np.ndarray:
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
    with tempfile.TemporaryDirectory(prefix="vlacpp-openpi-rng-") as tmp:
        source = Path(tmp) / "rng.cpp"
        binary = Path(tmp) / "rng"
        source.write_text(code, encoding="utf-8")
        subprocess.run(["c++", "-std=c++17", str(source), "-o", str(binary)], check=True)
        raw = subprocess.check_output([str(binary)], text=True)
    return np.asarray([float(v) for v in raw.split()], dtype=np.float32)


def load_observation(args: argparse.Namespace, action_dim: int, horizon: int) -> tuple[dict[str, Any], np.ndarray]:
    if args.observation_npz:
        loaded = np.load(args.observation_npz, allow_pickle=True)
        obs = {key: loaded[key] for key in loaded.files}
        if args.prompt_key in obs:
            obs[args.prompt_key] = str(np.asarray(obs[args.prompt_key]).item())
        else:
            obs[args.prompt_key] = args.prompt
        state = np.asarray(obs[args.state_key], dtype=np.float32)
    else:
        state = parse_state(args.state)
        obs = {
            args.state_key: state,
            args.prompt_key: args.prompt,
        }
        for key in args.image_key:
            obs[key] = np.full((args.image_height, args.image_width, 3), args.image_value, dtype=np.uint8)

    noise = cpp_mt19937_normals(args.seed, horizon * action_dim).reshape(horizon, action_dim)
    return obs, noise


def run_openpi(args: argparse.Namespace, obs: dict[str, Any], noise: np.ndarray) -> np.ndarray:
    try:
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config
    except ImportError as exc:
        raise SystemExit("official OpenPI is not installed; install Physical-Intelligence/openpi first") from exc

    train_config = openpi_config.get_config(args.openpi_config)
    policy = policy_config.create_trained_policy(
        train_config,
        args.openpi_checkpoint,
        sample_kwargs={"num_steps": args.steps},
        default_prompt=args.prompt,
        pytorch_device=args.pytorch_device,
    )
    result = policy.infer(obs, noise=noise)
    return np.asarray(result["actions"], dtype=np.float32)


def run_vlacpp(args: argparse.Namespace) -> np.ndarray:
    output = subprocess.check_output(
        [
            str(args.binary),
            "--model",
            str(args.vlacpp_model),
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
    decoded = json.loads(output)
    return np.asarray(decoded["actions"], dtype=np.float32).reshape(decoded["horizon"], decoded["action_dim"])


def vlacpp_capability(args: argparse.Namespace) -> str:
    output = subprocess.check_output(
        [
            str(args.binary),
            "--model",
            str(args.vlacpp_model),
            "--info",
        ],
        text=True,
    )
    return str(json.loads(output)["capability"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-config", required=True, help="OpenPI training config name, e.g. pi05_libero")
    parser.add_argument("--openpi-checkpoint", required=True, help="OpenPI checkpoint dir or gs:// path")
    parser.add_argument("--vlacpp-model", type=Path, required=True, help="converted vlacpp GGUF model")
    parser.add_argument("--binary", type=Path, default=Path("build/vlacpp-pi0"))
    parser.add_argument("--observation-npz", type=Path)
    parser.add_argument("--state", default="1,2,3")
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--prompt", default="pick up the fork")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--image-key", action="append", default=[])
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-value", type=int, default=127)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--pytorch-device", default=None)
    parser.add_argument(
        "--allow-missing-openpi",
        action="store_true",
        help="emit a skipped JSON result instead of failing when official OpenPI is not installed",
    )
    parser.add_argument(
        "--allow-restricted-vlacpp",
        action="store_true",
        help="allow restricted vlacpp subsets for harness tests; default requires capability=full-openpi",
    )
    args = parser.parse_args()

    if not openpi_available():
        if args.allow_missing_openpi:
            print(json.dumps({"status": "skipped", "reason": "official OpenPI is not installed"}, indent=2))
            return
        raise SystemExit("official OpenPI is not installed; install Physical-Intelligence/openpi first")

    capability = vlacpp_capability(args)
    if capability != "full-openpi" and not args.allow_restricted_vlacpp:
        raise SystemExit(
            f"vlacpp model capability is {capability}, not full-openpi; "
            "pass --allow-restricted-vlacpp only for subset harness tests"
        )

    vlacpp_actions = run_vlacpp(args)
    obs, noise = load_observation(args, vlacpp_actions.shape[1], vlacpp_actions.shape[0])
    openpi_actions = run_openpi(args, obs, noise)

    if openpi_actions.shape != vlacpp_actions.shape:
        raise SystemExit(f"shape mismatch: openpi={openpi_actions.shape} vlacpp={vlacpp_actions.shape}")
    max_abs = float(np.max(np.abs(openpi_actions - vlacpp_actions)))
    if max_abs > args.atol:
        raise SystemExit(f"compare failed: max_abs={max_abs:g} atol={args.atol:g}")
    print(json.dumps({"status": "ok", "max_abs": max_abs, "shape": list(vlacpp_actions.shape)}, indent=2))


if __name__ == "__main__":
    main()
