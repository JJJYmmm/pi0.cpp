#!/usr/bin/env python3
"""Assert OpenPI vision projector metadata and GGUF ne ordering."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    raw = subprocess.check_output(
        [sys.executable, str(args.repo / "tools" / "inspect-gguf.py"), str(args.model), "--json"],
        text=True,
    )
    decoded = json.loads(raw)
    metadata = decoded["metadata"]
    if metadata.get("vlacpp.openpi.vision_width") != 5:
        raise SystemExit(f"unexpected vision width: {metadata.get('vlacpp.openpi.vision_width')}")
    if metadata.get("vlacpp.openpi.language_width") != 6:
        raise SystemExit(f"unexpected language width: {metadata.get('vlacpp.openpi.language_width')}")

    tensors = {tensor["name"]: tensor for tensor in decoded["tensors"]}
    weight = tensors.get("vlacpp.openpi.vision_projector.weight")
    bias = tensors.get("vlacpp.openpi.vision_projector.bias")
    if weight is None or bias is None:
        raise SystemExit("missing vision projector tensors")
    if weight["shape"] != [5, 6]:
        raise SystemExit(f"unexpected GGUF projector weight shape: {weight['shape']}")
    if bias["shape"] != [6]:
        raise SystemExit(f"unexpected GGUF projector bias shape: {bias['shape']}")

    print(json.dumps({"status": "ok"}))


if __name__ == "__main__":
    main()
