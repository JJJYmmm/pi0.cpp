#!/usr/bin/env python3
"""Summarize a vlacpp OpenPI tensor-map manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def top_items(items: dict[str, dict[str, int]], limit: int) -> list[dict[str, Any]]:
    rows = [
        {"name": name, "total": stats["total"], "mapped": stats["mapped"], "unmapped": stats["total"] - stats["mapped"]}
        for name, stats in items.items()
    ]
    return sorted(rows, key=lambda row: (-row["total"], row["name"]))[:limit]


def summarize(manifest: dict[str, Any], top: int) -> dict[str, Any]:
    inventory = manifest.get("inventory", {})
    summary = {
        "source": manifest.get("source", ""),
        "family": manifest.get("family", ""),
        "expected_count": int(manifest.get("expected_count", 0)),
        "mapped_count": int(manifest.get("mapped_count", 0)),
        "coverage": float(manifest.get("coverage", 0.0)),
        "missing": manifest.get("missing", []),
    }
    if inventory:
        summary.update(
            {
                "total_tensor_count": int(inventory["total_tensor_count"]),
                "mapped_tensor_count": int(inventory["mapped_tensor_count"]),
                "unmapped_tensor_count": int(inventory["unmapped_tensor_count"]),
                "top_groups": top_items(inventory.get("groups", {}), top),
                "top_subgroups": top_items(inventory.get("subgroups", {}), top),
            }
        )
    return summary


def expect_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise SystemExit(f"unexpected {name}: {actual} != {expected}")


def validate(summary: dict[str, Any], args: argparse.Namespace) -> None:
    if args.expect_family is not None:
        expect_equal("family", summary["family"], args.expect_family)
    if args.expect_expected is not None:
        expect_equal("expected_count", summary["expected_count"], args.expect_expected)
    if args.expect_mapped is not None:
        expect_equal("mapped_count", summary["mapped_count"], args.expect_mapped)
    if args.expect_total is not None:
        expect_equal("total_tensor_count", summary.get("total_tensor_count"), args.expect_total)
    if args.expect_unmapped is not None:
        expect_equal("unmapped_tensor_count", summary.get("unmapped_tensor_count"), args.expect_unmapped)


def print_text(summary: dict[str, Any]) -> None:
    print(f"source: {summary['source']}")
    print(f"family: {summary['family']}")
    print(f"mapped: {summary['mapped_count']}/{summary['expected_count']} ({summary['coverage']:.3f})")
    if "total_tensor_count" not in summary:
        return
    print(
        "inventory: "
        f"{summary['mapped_tensor_count']}/{summary['total_tensor_count']} mapped, "
        f"{summary['unmapped_tensor_count']} unmapped"
    )
    print("groups:")
    for row in summary["top_groups"]:
        print(f"  {row['name']}: {row['mapped']}/{row['total']} mapped, {row['unmapped']} unmapped")
    print("subgroups:")
    for row in summary["top_subgroups"]:
        print(f"  {row['name']}: {row['mapped']}/{row['total']} mapped, {row['unmapped']} unmapped")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--expect-family")
    parser.add_argument("--expect-expected", type=int)
    parser.add_argument("--expect-mapped", type=int)
    parser.add_argument("--expect-total", type=int)
    parser.add_argument("--expect-unmapped", type=int)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = summarize(manifest, args.top)
    validate(summary, args)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_text(summary)


if __name__ == "__main__":
    main()
