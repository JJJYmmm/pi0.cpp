#!/usr/bin/env python3
"""Compare restricted pi0.5 action-head CLI output to a reference implementation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path


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
    with tempfile.TemporaryDirectory(prefix="vlacpp-pi05-action-head-rng-") as tmp:
        source = Path(tmp) / "rng.cpp"
        binary = Path(tmp) / "rng"
        source.write_text(code, encoding="utf-8")
        subprocess.run(["c++", "-std=c++17", str(source), "-o", str(binary)], check=True)
        raw = subprocess.check_output([str(binary)], text=True)
    return [float(v) for v in raw.split()]


def swish(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def linear(weight: list[float], bias: list[float], input_values: list[float], out_dim: int, in_dim: int) -> list[float]:
    output = []
    for row in range(out_dim):
        value = bias[row]
        for col in range(in_dim):
            value += weight[row * in_dim + col] * input_values[col]
        output.append(value)
    return output


def posemb_sincos(time: float, width: int) -> list[float]:
    half = width // 2
    result = [0.0] * width
    for i in range(half):
        fraction = 0.0 if half == 1 else i / (half - 1)
        period = 4.0e-3 * ((4.0 / 4.0e-3) ** fraction)
        angle = time / period * 2.0 * math.pi
        result[i] = math.sin(angle)
        result[half + i] = math.cos(angle)
    return result


def velocity(tensors: dict, action: list[float], time: float) -> list[float]:
    in_w = tensors["vlacpp.openpi.action_in_proj.weight"]
    in_b = tensors["vlacpp.openpi.action_in_proj.bias"]
    time_in_w = tensors["vlacpp.openpi.pi05.time_mlp_in.weight"]
    time_in_b = tensors["vlacpp.openpi.pi05.time_mlp_in.bias"]
    time_out_w = tensors["vlacpp.openpi.pi05.time_mlp_out.weight"]
    time_out_b = tensors["vlacpp.openpi.pi05.time_mlp_out.bias"]
    out_w = tensors["vlacpp.openpi.action_out_proj.weight"]
    out_b = tensors["vlacpp.openpi.action_out_proj.bias"]

    width = in_w["shape"][0]
    action_dim = in_w["shape"][1]
    action_token = linear(in_w["data"], in_b["data"], action, width, action_dim)
    time_emb = posemb_sincos(time, width)
    hidden = linear(time_in_w["data"], time_in_b["data"], time_emb, width, width)
    hidden = [swish(value) for value in hidden]
    time_context = linear(time_out_w["data"], time_out_b["data"], hidden, width, width)
    time_context = [swish(value) for value in time_context]
    mixed = [value + time_context[i] for i, value in enumerate(action_token)]
    return linear(out_w["data"], out_b["data"], mixed, out_b["shape"][0], width)


def reference_actions(checkpoint: dict, steps: int, seed: int) -> list[float]:
    metadata = checkpoint["metadata"]
    tensors = checkpoint["tensors"]
    horizon = int(metadata["action_horizon"])
    action_dim = int(metadata["action_dim"])
    x = cpp_mt19937_normals(seed, horizon * action_dim)
    dt = -1.0 / max(1, steps)
    time = 1.0
    for _ in range(max(1, steps)):
        v = [0.0] * len(x)
        for row in range(horizon):
            offset = row * action_dim
            v[offset : offset + action_dim] = velocity(tensors, x[offset : offset + action_dim], time)
        for i, value in enumerate(v):
            x[i] += dt * value
        time += dt
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--state", default="1,-2")
    parser.add_argument("--prompt", default="pick up the fork")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()

    checkpoint = json.loads(Path(args.checkpoint).read_text(encoding="utf-8"))
    expected = reference_actions(checkpoint, args.steps, args.seed)
    output = subprocess.check_output(
        [
            args.binary,
            "--model",
            args.model,
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
    actual = json.loads(output)["actions"]
    max_abs = max(abs(a - b) for a, b in zip(actual, expected))
    if len(actual) != len(expected) or max_abs > args.atol:
        raise SystemExit(f"pi05 action-head compare failed: max_abs={max_abs:g}")
    print(json.dumps({"status": "ok", "max_abs": max_abs, "count": len(actual)}))


if __name__ == "__main__":
    main()
