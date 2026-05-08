#!/usr/bin/env python3
"""Validate vlacpp-pi0 JSON CLI output shape and finite values."""

from __future__ import annotations

import argparse
import json
import math
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--prompt", default="pick up the fork")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--capability")
    args = parser.parse_args()

    if args.capability:
        info = json.loads(
            subprocess.check_output([args.binary, "--model", args.model, "--info"], text=True)
        )
        if info["capability"] != args.capability:
            raise SystemExit(f"unexpected capability: {info['capability']}")

    output = subprocess.check_output(
        [
            args.binary,
            "--model",
            args.model,
            "--state",
            args.state,
            "--prompt",
            args.prompt,
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
        ],
        text=True,
    )
    decoded = json.loads(output)
    actions = decoded["actions"]
    if decoded["horizon"] != args.horizon or decoded["action_dim"] != args.action_dim:
        raise SystemExit(f"unexpected shape: {decoded['horizon']}x{decoded['action_dim']}")
    if len(actions) != args.horizon * args.action_dim:
        raise SystemExit(f"unexpected action count: {len(actions)}")
    if not all(math.isfinite(value) for value in actions):
        raise SystemExit("non-finite action value")
    print(json.dumps({"status": "ok", "count": len(actions), "first": actions[: min(3, len(actions))]}))


if __name__ == "__main__":
    main()
