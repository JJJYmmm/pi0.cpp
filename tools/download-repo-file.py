#!/usr/bin/env python3
"""Download hf:// or ms:// repository files with simple resume support."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen


def resolve_repo_file_url(spec: str) -> str:
    script = Path(__file__).with_name("inspect-safetensors.py")
    spec_obj = importlib.util.spec_from_file_location("inspect_safetensors", script)
    if spec_obj is None or spec_obj.loader is None:
        raise SystemExit(f"failed to load {script}")
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    return module.resolve_repo_file_url(spec)


def remote_size(url: str) -> int | None:
    req = Request(url, headers={"Range": "bytes=0-0"})
    with urlopen(req, timeout=30) as response:
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        length = response.headers.get("Content-Length")
        return int(length) if length is not None else None


def download(url: str, output: Path, retries: int, chunk_size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)
    existing = output.stat().st_size if output.exists() else 0
    if expected is not None and existing == expected:
        print(f"complete: {output} ({existing} bytes)")
        return
    if expected is not None and existing > expected:
        raise SystemExit(f"local file is larger than remote file: {existing} > {expected}")

    for attempt in range(retries):
        headers = {}
        mode = "wb"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if existing > 0 and response.status == 200:
                    raise RuntimeError("server ignored resume Range request")
                with output.open(mode + "") as handle:
                    shutil.copyfileobj(response, handle, length=chunk_size)
            break
        except Exception:
            if attempt == retries - 1:
                raise
            existing = output.stat().st_size if output.exists() else 0
            time.sleep(1.0 * (attempt + 1))

    final = output.stat().st_size
    if expected is not None and final != expected:
        raise SystemExit(f"incomplete download: {final}/{expected} bytes")
    print(f"downloaded: {output} ({final} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="hf://owner/repo/path, ms://owner/repo/path, or https URL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    args = parser.parse_args()

    url = resolve_repo_file_url(args.source) if "://" in args.source and not args.source.startswith("http") else args.source
    download(url, args.output, args.retries, args.chunk_size)


if __name__ == "__main__":
    main()
