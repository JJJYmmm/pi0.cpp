#!/usr/bin/env python3
"""Assert tensor-map inventory coverage for the local action-head fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    inventory = manifest["inventory"]
    if inventory["total_tensor_count"] != 10:
        raise SystemExit(f"unexpected total_tensor_count: {inventory['total_tensor_count']}")
    if inventory["mapped_tensor_count"] != 10 or inventory["unmapped_tensor_count"] != 0:
        raise SystemExit(
            f"unexpected mapped coverage: {inventory['mapped_tensor_count']}/{inventory['total_tensor_count']}"
        )
    groups = inventory["groups"]
    for group in ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out", "state_proj"]:
        if groups.get(group, {}).get("total") != groups.get(group, {}).get("mapped"):
            raise SystemExit(f"group {group} is not fully mapped: {groups.get(group)}")
    print(json.dumps({"status": "ok", "mapped": inventory["mapped_tensor_count"]}))


if __name__ == "__main__":
    main()
