"""Small GGUF v3 writer used by vlacpp conversion tools."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGML_TYPE_F32 = 0
GGUF_ALIGNMENT = 32


def write_string(buf: bytearray, value: str) -> None:
    data = value.encode("utf-8")
    buf.extend(struct.pack("<Q", len(data)))
    buf.extend(data)


def write_value(buf: bytearray, value: Any) -> None:
    if isinstance(value, str):
        buf.extend(struct.pack("<I", GGUF_TYPE_STRING))
        write_string(buf, value)
    elif isinstance(value, int):
        buf.extend(struct.pack("<I", GGUF_TYPE_INT32))
        buf.extend(struct.pack("<i", value))
    elif isinstance(value, float):
        buf.extend(struct.pack("<I", GGUF_TYPE_FLOAT32))
        buf.extend(struct.pack("<f", value))
    elif isinstance(value, list):
        buf.extend(struct.pack("<I", GGUF_TYPE_ARRAY))
        elem_type = GGUF_TYPE_STRING if value and isinstance(value[0], str) else GGUF_TYPE_FLOAT32
        buf.extend(struct.pack("<IQ", elem_type, len(value)))
        if elem_type == GGUF_TYPE_STRING:
            for item in value:
                write_string(buf, str(item))
        else:
            for item in value:
                buf.extend(struct.pack("<f", float(item)))
    else:
        raise TypeError(f"unsupported metadata value: {value!r}")


def write_gguf(path: Path, metadata: dict[str, Any], tensors: dict[str, dict[str, Any]]) -> None:
    header = bytearray()
    header.extend(b"GGUF")
    header.extend(struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        write_string(header, key)
        write_value(header, value)

    tensor_infos = bytearray()
    tensor_data = bytearray()
    for name, tensor in tensors.items():
        pad = (-len(tensor_data)) % GGUF_ALIGNMENT
        tensor_data.extend(b"\0" * pad)
        offset = len(tensor_data)
        write_string(tensor_infos, name)
        tensor_infos.extend(struct.pack("<I", len(tensor["shape"])))
        for dim in tensor["shape"]:
            tensor_infos.extend(struct.pack("<Q", int(dim)))
        tensor_infos.extend(struct.pack("<IQ", GGML_TYPE_F32, offset))
        tensor_data.extend(struct.pack("<" + "f" * len(tensor["data"]), *tensor["data"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    data_start_pad = (-(len(header) + len(tensor_infos))) % GGUF_ALIGNMENT
    path.write_bytes(bytes(header + tensor_infos + b"\0" * data_start_pad + tensor_data))
