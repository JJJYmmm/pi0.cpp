#!/usr/bin/env python3
"""Assert selected vlacpp GGUF metadata values."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "vlacpp.model_type": "pi0",
    "vlacpp.state_dim": 2,
    "vlacpp.action_dim": 2,
    "vlacpp.action_horizon": 3,
    "vlacpp.state_mean": [0.5, -0.5],
    "vlacpp.state_std": [2.0, 4.0],
    "vlacpp.action_mean": [0.1, -0.2],
    "vlacpp.action_std": [1.5, 0.25],
    "vlacpp.openpi.action_width": 4,
}


def close_list(actual: list[float], expected: list[float]) -> bool:
    return len(actual) == len(expected) and all(abs(float(a) - b) <= 1e-6 for a, b in zip(actual, expected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    raw = subprocess.check_output(
        [sys.executable, str(args.repo / "tools" / "inspect-gguf.py"), str(args.model), "--json"],
        text=True,
    )
    metadata = json.loads(raw)["metadata"]
    for key, expected in EXPECTED.items():
        actual = metadata.get(key)
        if isinstance(expected, list):
            if not isinstance(actual, list) or not close_list(actual, expected):
                raise SystemExit(f"unexpected metadata {key}: {actual}")
        elif actual != expected:
            raise SystemExit(f"unexpected metadata {key}: {actual}")
    print(json.dumps({"status": "ok", "checked": len(EXPECTED)}))


if __name__ == "__main__":
    main()
