#!/usr/bin/env python3
"""Create a tiny pi0.5 F16 safetensors fixture used by CTest."""

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
        raise SystemExit("usage: make_pi05_f16_safetensors.py output.safetensors")

    metadata = {
        "model_type": "pi05",
        "image_width": 224,
        "image_height": 224,
        "state_dim": 2,
        "action_dim": 2,
        "action_horizon": 3,
        "image_keys": ["base_0_rgb"],
    }
    tensors = {
        "vlacpp.openpi.action_in_proj.weight": np.asarray(
            [[0.2, -0.1], [0.05, 0.15], [-0.2, 0.1], [0.1, 0.05]],
            dtype=np.float16,
        ),
        "vlacpp.openpi.action_in_proj.bias": np.asarray([0.01, -0.02, 0.03, -0.04], dtype=np.float16),
        "vlacpp.openpi.pi05.time_mlp_in.weight": np.asarray(
            [[0.1, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.1]],
            dtype=np.float16,
        ),
        "vlacpp.openpi.pi05.time_mlp_in.bias": np.asarray([0.0, 0.01, -0.01, 0.02], dtype=np.float16),
        "vlacpp.openpi.pi05.time_mlp_out.weight": np.asarray(
            [[0.2, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 0.2]],
            dtype=np.float16,
        ),
        "vlacpp.openpi.pi05.time_mlp_out.bias": np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float16),
        "vlacpp.openpi.action_out_proj.weight": np.asarray(
            [[0.3, -0.1, 0.2, 0.0], [-0.2, 0.25, 0.0, 0.1]],
            dtype=np.float16,
        ),
        "vlacpp.openpi.action_out_proj.bias": np.asarray([0.01, -0.02], dtype=np.float16),
    }

    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={"vlacpp.metadata": json.dumps(metadata)})


if __name__ == "__main__":
    main()
