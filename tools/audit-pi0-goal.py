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


def openpi_available(python: Path | None = None) -> bool:
    if python is None:
        return importlib.util.find_spec("openpi") is not None
    try:
        subprocess.check_call(
            [
                str(python),
                "-c",
                "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('openpi') else 1)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return True


def load_json_file(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
    parser.add_argument("--openpi-python", type=Path, help="Python interpreter for an official OpenPI environment")
    parser.add_argument("--parity-result", type=Path, help="JSON output from compare-openpi-policy.py")
    parser.add_argument("--parity-atol", type=float, default=1e-3)
    parser.add_argument(
        "--allow-restricted-vlacpp",
        action="store_true",
        help="accept a restricted vlacpp capability when a real OpenPI parity result is supplied",
    )
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
        accepted_capability = capability == "full-openpi" or (
            args.allow_restricted_vlacpp and str(capability).startswith("restricted-pi0-mtmd-vlm-action-decoder")
        )
        checks.append(
            criterion(
                "vlacpp OpenPI runtime capability",
                accepted_capability,
                f"{args.vlacpp_model} capability={capability}",
                "runtime still lacks a usable OpenPI pi0 capability",
            )
        )
    else:
        checks.append(
            criterion(
                "vlacpp OpenPI runtime capability",
                False,
                "no --vlacpp-model supplied",
                "inspect a converted full model with --vlacpp-model",
            )
        )

    has_openpi = openpi_available(args.openpi_python)
    checks.append(
        criterion(
            "official OpenPI installed",
            has_openpi,
            f"openpi_available={has_openpi} python={args.openpi_python or 'current'}",
            "install official OpenPI before real policy parity",
        )
    )

    parity = load_json_file(args.parity_result)
    parity_ok = (
        parity is not None
        and parity.get("status") == "ok"
        and float(parity.get("max_abs", float("inf"))) <= args.parity_atol
        and (parity.get("capability") == "full-openpi" or args.allow_restricted_vlacpp)
    )
    checks.append(
        criterion(
            "official OpenPI parity run",
            parity_ok,
            (
                f"{args.parity_result} status={None if parity is None else parity.get('status')} "
                f"max_abs={None if parity is None else parity.get('max_abs')} "
                f"capability={None if parity is None else parity.get('capability')}"
            ),
            "run tools/compare-openpi-policy.py against official OpenPI and a converted GGUF",
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
