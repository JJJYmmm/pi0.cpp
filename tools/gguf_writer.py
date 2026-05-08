"""GGUF writer adapter backed by llama.cpp's gguf-py module."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


LLAMA_GGUF_PY = Path(__file__).resolve().parents[1] / "third_party" / "llama.cpp" / "gguf-py"
if not LLAMA_GGUF_PY.exists():
    raise ImportError("third_party/llama.cpp/gguf-py is required; initialize the llama.cpp submodule")
sys.path.insert(0, str(LLAMA_GGUF_PY))

import gguf  # noqa: E402


def add_metadata(writer: gguf.GGUFWriter, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        if key == "general.architecture":
            continue
        if isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_int32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)
        else:
            raise TypeError(f"unsupported metadata value for {key}: {value!r}")


def write_gguf(path: Path, metadata: dict[str, Any], tensors: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(path, arch=str(metadata.get("general.architecture", "vlacpp")))
    add_metadata(writer, metadata)
    for name, tensor in tensors.items():
        shape = [int(dim) for dim in tensor["shape"]]
        data = np.asarray(tensor["data"], dtype=np.float32)
        writer.add_tensor(name, data, raw_shape=list(reversed(shape)))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
