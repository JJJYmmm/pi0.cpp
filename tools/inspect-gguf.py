#!/usr/bin/env python3
"""Inspect the small GGUF subset emitted by vlacpp conversion tools."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


def read_string(data: bytes, offset: int) -> tuple[str, int]:
    size = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    text = data[offset : offset + size].decode("utf-8")
    return text, offset + size


def read_value(data: bytes, offset: int, value_type: int) -> tuple[Any, int]:
    if value_type in {GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32}:
        if value_type == GGUF_TYPE_FLOAT32:
            return struct.unpack_from("<f", data, offset)[0], offset + 4
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    if value_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack_from("<?", data, offset)[0]), offset + 1
    if value_type == GGUF_TYPE_STRING:
        return read_string(data, offset)
    if value_type == GGUF_TYPE_ARRAY:
        elem_type, count = struct.unpack_from("<IQ", data, offset)
        offset += 12
        values = []
        for _ in range(count):
            value, offset = read_value(data, offset, elem_type)
            values.append(value)
        return values, offset
    raise SystemExit(f"unsupported GGUF metadata type: {value_type}")


def inspect(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:4] != b"GGUF":
        raise SystemExit("not a GGUF file")
    version, tensor_count, metadata_count = struct.unpack_from("<IQQ", data, 4)
    offset = 24
    metadata = {}
    for _ in range(metadata_count):
        key, offset = read_string(data, offset)
        value_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        value, offset = read_value(data, offset, value_type)
        metadata[key] = value

    tensors = []
    for _ in range(tensor_count):
        name, offset = read_string(data, offset)
        n_dims = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        shape = list(struct.unpack_from("<" + "Q" * n_dims, data, offset))
        offset += 8 * n_dims
        tensor_type, data_offset = struct.unpack_from("<IQ", data, offset)
        offset += 12
        tensors.append({"name": name, "shape": shape, "type": tensor_type, "data_offset": data_offset})
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "metadata": metadata,
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--contains", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = inspect(args.path)
    tensors = result["tensors"]
    for needle in args.contains:
        tensors = [tensor for tensor in tensors if needle in tensor["name"]]
    result = {**result, "tensor_count": len(tensors), "tensors": tensors}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for tensor in tensors:
            print(f"{tensor['name']}\t{tensor['shape']}\t{tensor['type']}")


if __name__ == "__main__":
    main()
