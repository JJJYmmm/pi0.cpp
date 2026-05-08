#!/usr/bin/env python3
"""Assert mapped safetensors conversion rejects truncated local payloads."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    full = args.work_dir / "action-head.safetensors"
    truncated = args.work_dir / "action-head-truncated.safetensors"
    manifest = args.work_dir / "action-head-map.json"
    truncated_manifest = args.work_dir / "action-head-truncated-map.json"
    output = args.work_dir / "truncated.gguf"

    make_result = run(
        [
            sys.executable,
            str(args.repo / "tests" / "make_action_head_safetensors.py"),
            str(args.repo / "tests" / "action-head-checkpoint.json"),
            str(full),
        ]
    )
    if make_result.returncode == 77:
        sys.exit(77)
    if make_result.returncode != 0:
        raise SystemExit(make_result.stderr or make_result.stdout)

    subprocess.run(
        [
            sys.executable,
            str(args.repo / "tools" / "map-openpi-tensors.py"),
            str(full),
            "--family",
            "action-expert",
            "--require-complete",
            "--output",
            str(manifest),
        ],
        check=True,
    )

    shutil.copyfile(full, truncated)
    truncated_size = max(0, full.stat().st_size - 32)
    with truncated.open("r+b") as handle:
        handle.truncate(truncated_size)

    decoded = json.loads(manifest.read_text(encoding="utf-8"))
    decoded["source"] = str(truncated)
    truncated_manifest.write_text(json.dumps(decoded, indent=2) + "\n", encoding="utf-8")

    result = run(
        [
            sys.executable,
            str(args.repo / "tools" / "convert-openpi-to-gguf.py"),
            "--tensor-map-manifest",
            str(truncated_manifest),
            "--output",
            str(output),
        ]
    )
    if result.returncode == 0:
        raise SystemExit("truncated manifest conversion unexpectedly succeeded")
    combined = result.stdout + result.stderr
    if "truncated tensor payload" not in combined:
        raise SystemExit(f"unexpected converter failure:\n{combined}")
    if output.exists():
        raise SystemExit("converter wrote GGUF output for a truncated tensor payload")

    print(json.dumps({"status": "ok"}))


if __name__ == "__main__":
    main()
