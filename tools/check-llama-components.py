#!/usr/bin/env python3
"""Check the llama.cpp components expected by the pi0 graph integration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REQUIRED_PATHS = [
    "ggml/include/ggml.h",
    "src/llama.cpp",
    "tools/mtmd/clip.cpp",
    "tools/mtmd/clip-graph.h",
    "tools/mtmd/models/siglip.cpp",
]


def cmake_targets(build_dir: Path) -> set[str]:
    output = subprocess.check_output(["cmake", "--build", str(build_dir), "--target", "help"], text=True)
    targets = set()
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("... "):
            targets.add(line[4:].strip())
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-ggml-targets", action="store_true")
    args = parser.parse_args()

    llama_dir = args.repo / "third_party" / "llama.cpp"
    missing = [path for path in REQUIRED_PATHS if not (llama_dir / path).exists()]
    targets: set[str] = set()
    if args.build_dir is not None:
        targets = cmake_targets(args.build_dir)

    result = {
        "llama_dir": str(llama_dir),
        "missing_paths": missing,
        "has_siglip_source": not missing,
        "targets": sorted(targets),
        "has_ggml_targets": {"ggml", "ggml-base", "ggml-cpu", "llama"}.issubset(targets) if targets else None,
        "has_mtmd_target": "mtmd" in targets if targets else None,
    }
    if missing:
        raise SystemExit("missing llama.cpp component(s): " + ", ".join(missing))
    if args.require_ggml_targets and not result["has_ggml_targets"]:
        raise SystemExit("CMake build dir does not expose expected ggml/llama targets")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"llama.cpp: {llama_dir}")
        print("siglip source: present")
        if targets:
            print(f"ggml targets: {result['has_ggml_targets']}")
            print(f"mtmd target: {result['has_mtmd_target']}")


if __name__ == "__main__":
    main()
