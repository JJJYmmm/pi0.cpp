#!/usr/bin/env python3
"""Assert selected metadata values in a converted JSON checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_expected(values: list[str]) -> dict[str, int | str]:
    expected: dict[str, int | str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected KEY=VALUE, got {value}")
        key, raw = value.split("=", 1)
        try:
            expected[key] = int(raw)
        except ValueError:
            expected[key] = raw
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expect", action="append", default=[])
    args = parser.parse_args()

    with args.checkpoint.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    metadata = loaded.get("metadata", loaded)
    expected = parse_expected(args.expect)
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise SystemExit(f"unexpected metadata {key}: expected {expected_value}, got {actual}")
    print(json.dumps({"status": "ok", "checked": len(expected)}))


if __name__ == "__main__":
    main()
