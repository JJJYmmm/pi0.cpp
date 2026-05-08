#!/usr/bin/env python3
"""Exercise compare-openpi-policy.py against a fake OpenPI package."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: run_fake_openpi_policy_compare.py source_dir binary checkpoint_json model_gguf")

    source_dir = Path(sys.argv[1])
    binary = Path(sys.argv[2])
    checkpoint_json = Path(sys.argv[3])
    model_gguf = Path(sys.argv[4])

    fake_root = Path(os.environ.get("VLACPP_FAKE_OPENPI_DIR", "/tmp/vlacpp-fake-openpi"))
    if fake_root.exists():
        for child in sorted(fake_root.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    write_file(fake_root / "openpi" / "__init__.py", "")
    write_file(fake_root / "openpi" / "policies" / "__init__.py", "")
    write_file(fake_root / "openpi" / "training" / "__init__.py", "")
    write_file(
        fake_root / "openpi" / "training" / "config.py",
        """
def get_config(name):
    return {"name": name}
""",
    )
    write_file(
        fake_root / "openpi" / "policies" / "policy_config.py",
        """
import json
import numpy as np


def create_trained_policy(train_config, checkpoint_dir, *, sample_kwargs=None, default_prompt=None, pytorch_device=None):
    return _FakePolicy(checkpoint_dir, sample_kwargs or {}, default_prompt)


class _FakePolicy:
    def __init__(self, checkpoint_dir, sample_kwargs, default_prompt):
        with open(checkpoint_dir, encoding="utf-8") as handle:
            self._checkpoint = json.load(handle)
        self._sample_kwargs = sample_kwargs
        self._default_prompt = default_prompt

    def infer(self, obs, *, noise=None):
        metadata = self._checkpoint["metadata"]
        tensors = self._checkpoint["tensors"]
        action_dim = int(metadata["action_dim"])
        horizon = int(metadata["action_horizon"])
        steps = max(1, int(self._sample_kwargs.get("num_steps", 10)))
        time_weight = np.asarray(tensors["pi0.velocity.time_weight"]["data"], dtype=np.float32)
        state = np.asarray(obs["state"], dtype=np.float32)
        if "state_mean" in metadata or "state_std" in metadata:
            state = (state - np.asarray(metadata["state_mean"], dtype=np.float32)) / np.asarray(metadata["state_std"], dtype=np.float32)
        prompt = obs.get("prompt", self._default_prompt or "")
        image_mean = np.float32(127.0 / 255.0 * 2.0 - 1.0)
        weight_shape = tensors["pi0.velocity.weight"]["shape"]
        weight = np.asarray(tensors["pi0.velocity.weight"]["data"], dtype=np.float32).reshape(weight_shape)
        full_features = np.asarray([1.0, *state.tolist(), image_mean, (len(prompt) % 97) / 97.0], dtype=np.float32)
        legacy_features = np.asarray([1.0, float(np.mean(state)), image_mean, (len(prompt) % 97) / 97.0], dtype=np.float32)
        features = full_features if weight.shape[1] == full_features.shape[0] else legacy_features
        x = np.asarray(noise, dtype=np.float32).reshape(horizon, action_dim).copy()
        time = 1.0
        dt = -1.0 / steps
        for _ in range(steps):
            for t in range(horizon):
                for col in range(action_dim):
                    target = float(np.dot(weight[col], features))
                    velocity = x[t, col] - target + time * time_weight[col]
                    x[t, col] += dt * velocity
            time += dt
        if "action_mean" in metadata or "action_std" in metadata:
            x = x * np.asarray(metadata["action_std"], dtype=np.float32) + np.asarray(metadata["action_mean"], dtype=np.float32)
        return {"actions": x}
""",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(fake_root) + os.pathsep + env.get("PYTHONPATH", "")
    base_command = [
            sys.executable,
            str(source_dir / "tools" / "compare-openpi-policy.py"),
            "--openpi-config",
            "fake_pi0",
            "--openpi-checkpoint",
            str(checkpoint_json),
            "--vlacpp-model",
            str(model_gguf),
            "--binary",
            str(binary),
            "--state",
            "1,2,3",
            "--prompt",
            "pick up the fork",
            "--steps",
            "4",
            "--seed",
            "1",
    ]

    rejected = subprocess.run(
        base_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    if rejected.returncode == 0:
        raise SystemExit("restricted vlacpp model was accepted without --allow-restricted-vlacpp")
    if "not full-openpi" not in rejected.stderr:
        raise SystemExit(f"unexpected restricted rejection stderr: {rejected.stderr}")

    subprocess.run(
        [*base_command, "--allow-restricted-vlacpp"],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
