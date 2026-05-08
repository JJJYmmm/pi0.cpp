#!/usr/bin/env python3
"""Download a remote repo file with resume support."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


def resolve_repo_file_url(spec: str) -> str:
    if spec.startswith("hf://"):
        base = "https://huggingface.co"
        ref = "main"
        repo_and_file = spec[len("hf://") :]
    elif spec.startswith("ms://"):
        base = "https://modelscope.cn/models"
        ref = "master"
        repo_and_file = spec[len("ms://") :]
    elif spec.startswith("https://") or spec.startswith("http://"):
        return spec
    else:
        raise SystemExit("source must be https://, http://, hf://owner/repo/path, or ms://owner/repo/path")
    parts = repo_and_file.split("/", 2)
    if len(parts) != 3:
        raise SystemExit("repo paths must look like hf://owner/repo/path or ms://owner/repo/path")
    owner, repo, filename = parts
    return f"{base}/{owner}/{repo}/resolve/{ref}/{quote(filename)}"


def remote_size(url: str) -> int | None:
    request = Request(url, method="HEAD")
    try:
        with urlopen(request, timeout=30) as response:
            value = response.headers.get("content-length")
            return int(value) if value is not None else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="https://, hf://owner/repo/path, or ms://owner/repo/path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    args = parser.parse_args()

    url = resolve_repo_file_url(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = args.output.stat().st_size if args.output.exists() else 0
    headers = {}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = Request(url, headers=headers)
    total = remote_size(url)
    mode = "ab" if existing else "wb"
    with urlopen(request, timeout=60) as response, args.output.open(mode) as handle:
        while True:
            chunk = response.read(args.chunk_size)
            if not chunk:
                break
            handle.write(chunk)
    final = args.output.stat().st_size
    if total is not None and final != total:
        raise SystemExit(f"incomplete download: got {final} bytes, expected {total} bytes")
    print(f"downloaded {final} bytes to {args.output}")


if __name__ == "__main__":
    main()
