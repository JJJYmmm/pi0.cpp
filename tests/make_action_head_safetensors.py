#!/usr/bin/env python3
"""Create a tiny OpenPI-named action-head safetensors fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import numpy as np
    from safetensors.numpy import save_file
except ImportError:
    sys.exit(77)


TARGET_TO_SOURCE = {
    "vlacpp.openpi.state_proj.weight": "state_proj.weight",
    "vlacpp.openpi.state_proj.bias": "state_proj.bias",
    "vlacpp.openpi.action_in_proj.weight": "action_in_proj.weight",
    "vlacpp.openpi.action_in_proj.bias": "action_in_proj.bias",
    "vlacpp.openpi.action_time_mlp_in.weight": "action_time_mlp_in.weight",
    "vlacpp.openpi.action_time_mlp_in.bias": "action_time_mlp_in.bias",
    "vlacpp.openpi.action_time_mlp_out.weight": "action_time_mlp_out.weight",
    "vlacpp.openpi.action_time_mlp_out.bias": "action_time_mlp_out.bias",
    "vlacpp.openpi.action_out_proj.weight": "action_out_proj.weight",
    "vlacpp.openpi.action_out_proj.bias": "action_out_proj.bias",
}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_action_head_safetensors.py checkpoint.json output.safetensors")

    checkpoint = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    tensors = {}
    for target, source in TARGET_TO_SOURCE.items():
        tensor = checkpoint["tensors"][target]
        tensors[source] = np.asarray(tensor["data"], dtype=np.float32).reshape(tensor["shape"])

    output = Path(sys.argv[2])
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={"vlacpp.metadata": json.dumps(checkpoint["metadata"])})


if __name__ == "__main__":
    main()
