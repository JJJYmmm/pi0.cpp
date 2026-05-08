#!/usr/bin/env python3
"""Assert tensor-map inventory coverage for the local action-head fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", default="action-expert")
    parser.add_argument("--total", type=int, default=10)
    parser.add_argument("--mapped", type=int, default=10)
    parser.add_argument("--group", action="append", default=[])
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("family") != args.family:
        raise SystemExit(f"unexpected family: {manifest.get('family')}")
    if manifest.get("expected_count") != args.mapped or manifest.get("mapped_count") != args.mapped:
        raise SystemExit(f"unexpected top-level coverage: {manifest.get('mapped_count')}/{manifest.get('expected_count')}")
    if abs(float(manifest.get("coverage")) - 1.0) > 1e-9:
        raise SystemExit(f"unexpected coverage: {manifest.get('coverage')}")
    inventory = manifest["inventory"]
    if inventory["total_tensor_count"] != args.total:
        raise SystemExit(f"unexpected total_tensor_count: {inventory['total_tensor_count']}")
    if inventory["mapped_tensor_count"] != args.mapped or inventory["unmapped_tensor_count"] != args.total - args.mapped:
        raise SystemExit(
            f"unexpected mapped coverage: {inventory['mapped_tensor_count']}/{inventory['total_tensor_count']}"
        )
    groups = inventory["groups"]
    expected_groups = args.group or ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out", "state_proj"]
    for group in expected_groups:
        if groups.get(group, {}).get("total") != groups.get(group, {}).get("mapped"):
            raise SystemExit(f"group {group} is not fully mapped: {groups.get(group)}")
    print(json.dumps({"status": "ok", "mapped": inventory["mapped_tensor_count"]}))


if __name__ == "__main__":
    main()
