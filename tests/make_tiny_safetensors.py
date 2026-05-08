#!/usr/bin/env python3
"""Create the tiny pi0 safetensors fixture used by CTest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import numpy as np
    from safetensors.numpy import save_file
except ImportError:
    sys.exit(77)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_tiny_safetensors.py output.safetensors")

    metadata = {
        "model_type": "pi0",
        "image_width": 224,
        "image_height": 224,
        "state_dim": 3,
        "action_dim": 2,
        "action_horizon": 4,
        "image_keys": ["base_0_rgb"],
        "state_mean": [0.5, 1.0, -1.0],
        "state_std": [0.5, 2.0, 4.0],
        "action_mean": [0.1, -0.2],
        "action_std": [2.0, 0.5],
    }
    tensors = {
        "pi0.velocity.weight": np.array(
            [[0.01, 0.2, 0.15, 0.15, 0.25, 0.25], [0.02, 0.1, 0.2, 0.15, 0.2, 0.35]],
            dtype=np.float32,
        ),
        "pi0.velocity.time_weight": np.array([0.001, -0.002], dtype=np.float32),
    }

    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={"vlacpp.metadata": json.dumps(metadata)})


if __name__ == "__main__":
    main()
