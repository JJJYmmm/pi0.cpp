#!/usr/bin/env python3
"""Assert repository downloads reject concurrent writes to one output path."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    script = args.repo / "tools" / "download-repo-file.py"
    spec = importlib.util.spec_from_file_location("download_repo_file", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    output = args.work_dir / "model.safetensors"
    lock = output.with_name(output.name + ".lock")
    lock.write_text("pid=0\n", encoding="utf-8")
    try:
        try:
            module.download("https://example.invalid/model.safetensors", output, retries=1, chunk_size=1024)
        except SystemExit as exc:
            text = str(exc)
            if "download already in progress" not in text or str(lock) not in text:
                raise SystemExit(f"unexpected lock error: {text}")
        else:
            raise SystemExit("download unexpectedly ignored existing lock")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
