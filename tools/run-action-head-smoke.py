#!/usr/bin/env python3
"""Run the restricted OpenPI action-head HF -> GGUF -> vlacpp smoke path."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_PI0_SOURCE = "hf://maxqualia/openpi-pi0-corkinbox100-1882950e/model.safetensors"
DEFAULT_PI05_SOURCE = "hf://Tacoin/openpi-pi0.5-libero-onnx/checkpoints/pi05_libero_pytorch/model.safetensors"
DEFAULT_PI05_CONFIG = "hf://Tacoin/openpi-pi0.5-libero-onnx/checkpoints/pi05_libero_pytorch/config.json"
DEFAULT_PI05_NORM_STATS = "hf://Tacoin/openpi-pi0.5-libero-onnx/assets/physical-intelligence/libero/norm_stats.json"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--family", choices=["pi0", "pi05"], default="pi0")
    parser.add_argument("--source")
    parser.add_argument("--config", help="optional OpenPI config JSON, local path or hf:// URI")
    parser.add_argument("--norm-stats", help="optional OpenPI norm_stats JSON, local path or hf:// URI")
    parser.add_argument("--binary", type=Path, default=Path("build/vlacpp-pi0"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="vlacpp-action-head-smoke-")
        work_dir = Path(tmp.name)
    else:
        tmp = None
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        source = args.source or (DEFAULT_PI05_SOURCE if args.family == "pi05" else DEFAULT_PI0_SOURCE)
        config = args.config or (DEFAULT_PI05_CONFIG if args.family == "pi05" else None)
        norm_stats = args.norm_stats or (DEFAULT_PI05_NORM_STATS if args.family == "pi05" else None)
        action_horizon = args.action_horizon if args.action_horizon is not None else (10 if args.family == "pi05" else 32)
        manifest_family = "pi05-action-expert" if args.family == "pi05" else "action-expert"
        model_type = "pi05" if args.family == "pi05" else "pi0"

        manifest = work_dir / "openpi-action-map.json"
        gguf = work_dir / "openpi-action-head.gguf"
        output = work_dir / "actions.json"

        run(
            [
                sys.executable,
                str(args.repo / "tools" / "map-openpi-tensors.py"),
                source,
                "--family",
                manifest_family,
                "--require-complete",
                "--output",
                str(manifest),
            ]
        )
        convert_command = [
                sys.executable,
                str(args.repo / "tools" / "convert-openpi-to-gguf.py"),
                "--checkpoint",
                source,
                "--tensor-map-manifest",
                str(manifest),
                "--output",
                str(gguf),
                "--model-type",
                model_type,
        ]
        if config is not None:
            convert_command.extend(["--config", config])
        if norm_stats is not None:
            convert_command.extend(["--norm-stats", norm_stats])
        if args.action_horizon is not None or config is None:
            convert_command.extend(["--action-horizon", str(action_horizon)])
        run(convert_command)
        inspect_raw = subprocess.check_output(
            [sys.executable, str(args.repo / "tools" / "inspect-gguf.py"), str(gguf), "--json"],
            text=True,
        )
        inspect = json.loads(inspect_raw)
        if inspect["tensor_count"] < 8:
            raise SystemExit(f"expected at least 8 action-head tensors, got {inspect['tensor_count']}")
        metadata = inspect["metadata"]
        state_dim = int(metadata["vlacpp.state_dim"])
        action_dim = int(metadata["vlacpp.action_dim"])
        action_horizon = int(metadata["vlacpp.action_horizon"])

        state = ",".join(["0"] * state_dim)
        action_raw = subprocess.check_output(
            [
                str(args.binary),
                "--model",
                str(gguf),
                "--state",
                state,
                "--prompt",
                "pick up the fork",
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
            ],
            text=True,
        )
        output.write_text(action_raw, encoding="utf-8")
        decoded = json.loads(action_raw)
        actions = decoded["actions"]
        if decoded["horizon"] != action_horizon or decoded["action_dim"] != action_dim:
            raise SystemExit(f"unexpected action shape {decoded['horizon']}x{decoded['action_dim']}")
        if len(actions) != action_horizon * action_dim or not all(math.isfinite(v) for v in actions):
            raise SystemExit("invalid action output")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "family": args.family,
                    "source": source,
                    "work_dir": str(work_dir),
                    "gguf": str(gguf),
                    "tensor_count": inspect["tensor_count"],
                    "action_count": len(actions),
                    "first": actions[:3],
                },
                indent=2,
            )
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
