#!/usr/bin/env python3
"""Audit pi0 inference goal completion against concrete local evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


def command_json(command: list[str]) -> dict[str, Any] | None:
    try:
        raw = subprocess.check_output(command, text=True)
    except Exception:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def git_branch(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    except Exception:
        return None


def openpi_available() -> bool:
    return importlib.util.find_spec("openpi") is not None


def criterion(name: str, ok: bool, evidence: str, missing: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "ok": ok, "evidence": evidence}
    if not ok:
        result["missing"] = missing
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--binary", type=Path, default=Path("build/vlacpp-pi0"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpts/lerobot-pi0/model.safetensors"))
    parser.add_argument("--expected-checkpoint-size", type=int, default=14005623256)
    parser.add_argument("--vlacpp-model", type=Path, help="GGUF model to inspect for full-openpi capability")
    parser.add_argument("--allow-incomplete", action="store_true", help="exit 0 even when requirements are missing")
    args = parser.parse_args()

    checks = []
    branch = git_branch(args.repo)
    checks.append(criterion("git branch initialized", branch == "pi0-infer", f"branch={branch}", "expected pi0-infer"))

    checkpoint_size = args.checkpoint.stat().st_size if args.checkpoint.exists() else None
    checks.append(
        criterion(
            "local checkpoint downloaded",
            checkpoint_size == args.expected_checkpoint_size,
            f"{args.checkpoint} size={checkpoint_size}",
            f"expected {args.expected_checkpoint_size} bytes",
        )
    )

    if args.vlacpp_model is not None:
        info = command_json([str(args.binary), "--model", str(args.vlacpp_model), "--info"])
        capability = info.get("capability") if info else None
        checks.append(
            criterion(
                "vlacpp full-openpi capability",
                capability == "full-openpi",
                f"{args.vlacpp_model} capability={capability}",
                "runtime still reports restricted or unreadable capability",
            )
        )
    else:
        checks.append(
            criterion(
                "vlacpp full-openpi capability",
                False,
                "no --vlacpp-model supplied",
                "inspect a converted full model with --vlacpp-model",
            )
        )

    has_openpi = openpi_available()
    checks.append(
        criterion(
            "official OpenPI installed",
            has_openpi,
            f"openpi_available={has_openpi}",
            "install official OpenPI before real policy parity",
        )
    )

    checks.append(
        criterion(
            "official OpenPI parity run",
            False,
            "no recorded successful compare-openpi-policy.py run for full-openpi",
            "run tools/compare-openpi-policy.py against official OpenPI and a full-openpi GGUF",
        )
    )

    complete = all(check["ok"] for check in checks)
    print(
        json.dumps(
            {
                "status": "complete" if complete else "incomplete",
                "objective": "pi0 end-to-end inference: checkpoint, GGUF, full model forward/sampling, OpenPI parity",
                "checks": checks,
            },
            indent=2,
        )
    )
    if not complete and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
