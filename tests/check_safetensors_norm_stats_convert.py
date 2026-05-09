#!/usr/bin/env python3
"""Check safetensors normalizer conversion and padding."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        from safetensors.numpy import save_file
    except ImportError:
        return 77

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = args.output_dir / "norm-stats.safetensors"
    gguf = args.output_dir / "norm-stats.gguf"
    save_file(
        {
            "observation.state.mean": np.asarray([0.5, -0.25], dtype=np.float32),
            "observation.state.std": np.asarray([2.0, 4.0], dtype=np.float32),
            "action.mean": np.asarray([0.1], dtype=np.float32),
            "action.std": np.asarray([1.5], dtype=np.float32),
        },
        stats,
    )

    subprocess.check_call(
        [
            sys.executable,
            str(args.repo / "tools" / "convert-openpi-to-gguf.py"),
            "--init-tiny",
            "--state-dim",
            "4",
            "--action-dim",
            "3",
            "--model-type",
            "pi0",
            "--norm-stats",
            str(stats),
            "--output",
            str(gguf),
        ]
    )
    raw = subprocess.check_output(
        [
            sys.executable,
            str(args.repo / "tools" / "inspect-gguf.py"),
            str(gguf),
            "--json",
        ],
        text=True,
    )
    metadata = json.loads(raw)["metadata"]
    expected = {
        "vlacpp.state_mean": [0.5, -0.25, 0.0, 0.0],
        "vlacpp.state_std": [2.0, 4.0, 1.0, 1.0],
        "vlacpp.action_mean": [0.1, 0.0, 0.0],
        "vlacpp.action_std": [1.5, 1.0, 1.0],
    }
    for key, values in expected.items():
        actual = metadata[key]
        if len(actual) != len(values):
            raise SystemExit(f"{key} length mismatch: {len(actual)} != {len(values)}")
        if not np.allclose(np.asarray(actual, dtype=np.float32), np.asarray(values, dtype=np.float32)):
            raise SystemExit(f"{key} mismatch: {actual} != {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
