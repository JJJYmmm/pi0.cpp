#!/usr/bin/env python3
"""Convert a tiny OpenPI-style checkpoint into vlacpp GGUF.

The production pi0 tensor map is still intentionally narrow, but this writes a
real GGUF v3 container with metadata and F32 tensors consumed by the C++ pi0
runtime. A JSON output path preserves the original metadata-only smoke flow.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from gguf_writer import write_gguf


REQUIRED_TENSORS = {"pi0.velocity.weight", "pi0.velocity.time_weight"}
ACTION_HEAD_TENSORS = {
    "vlacpp.openpi.action_in_proj.weight",
    "vlacpp.openpi.action_in_proj.bias",
    "vlacpp.openpi.action_time_mlp_in.weight",
    "vlacpp.openpi.action_time_mlp_in.bias",
    "vlacpp.openpi.action_time_mlp_out.weight",
    "vlacpp.openpi.action_time_mlp_out.bias",
    "vlacpp.openpi.action_out_proj.weight",
    "vlacpp.openpi.action_out_proj.bias",
}
PI05_ACTION_HEAD_TENSORS = {
    "vlacpp.openpi.action_in_proj.weight",
    "vlacpp.openpi.action_in_proj.bias",
    "vlacpp.openpi.pi05.time_mlp_in.weight",
    "vlacpp.openpi.pi05.time_mlp_in.bias",
    "vlacpp.openpi.pi05.time_mlp_out.weight",
    "vlacpp.openpi.pi05.time_mlp_out.bias",
    "vlacpp.openpi.action_out_proj.weight",
    "vlacpp.openpi.action_out_proj.bias",
}


def default_tiny_weights(action_dim: int, state_dim: int) -> dict[str, dict[str, Any]]:
    feature_dim = state_dim + 3
    weights: list[float] = []
    for col in range(action_dim):
        phase = (col + 1) / max(1, action_dim)
        row = [0.01 * phase]
        row.extend(0.50 / max(1, state_dim) for _ in range(state_dim))
        row.extend([0.25, 0.25])
        weights.extend(row)
    return {
        "pi0.velocity.weight": {
            "shape": [action_dim, feature_dim],
            "data": weights,
        },
        "pi0.velocity.time_weight": {
            "shape": [action_dim],
            "data": [0.001] * action_dim,
        },
    }


def resolve_checkpoint(checkpoint: str | None) -> Path | None:
    if checkpoint is None:
        return None
    if checkpoint.startswith("hf://"):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("hf:// checkpoints require huggingface_hub") from exc
        repo_and_file = checkpoint[len("hf://") :]
        parts = repo_and_file.split("/", 2)
        if len(parts) != 3:
            raise SystemExit("hf:// checkpoints must look like hf://owner/repo/path/to/checkpoint.json")
        return Path(hf_hub_download(repo_id=f"{parts[0]}/{parts[1]}", filename=parts[2]))
    return Path(checkpoint)


def resolve_repo_file_url(spec: str) -> str:
    if spec.startswith("hf://"):
        base = "https://huggingface.co"
        ref = "main"
        repo_and_file = spec[len("hf://") :]
    elif spec.startswith("ms://"):
        base = "https://modelscope.cn/models"
        ref = "master"
        repo_and_file = spec[len("ms://") :]
    else:
        raise SystemExit("remote repo paths must start with hf:// or ms://")
    parts = repo_and_file.split("/", 2)
    if len(parts) != 3:
        raise SystemExit("remote repo paths must look like hf://owner/repo/path or ms://owner/repo/path")
    owner, repo, filename = parts
    return f"{base}/{owner}/{repo}/resolve/{ref}/{quote(filename)}"


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_arg(spec: str | None) -> dict[str, Any]:
    if spec is None:
        return {}
    if spec.startswith("hf://") or spec.startswith("ms://"):
        with urlopen(resolve_repo_file_url(spec), timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    if spec.startswith("https://") or spec.startswith("http://"):
        with urlopen(spec, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    return load_json(Path(spec))


def apply_norm_stats(metadata: dict[str, Any], norm_stats: dict[str, Any]) -> None:
    stats = norm_stats.get("norm_stats", norm_stats)
    if "state" in stats:
        state = stats["state"]
        metadata["state_mean"] = [float(v) for v in state["mean"]]
        metadata["state_std"] = [float(v) for v in state["std"]]
    if "actions" in stats:
        actions = stats["actions"]
        metadata["action_mean"] = [float(v) for v in actions["mean"]]
        metadata["action_std"] = [float(v) for v in actions["std"]]

    if len(metadata["state_mean"]) != int(metadata["state_dim"]) or len(metadata["state_std"]) != int(metadata["state_dim"]):
        raise SystemExit("norm stats state mean/std must match state_dim")
    if len(metadata["action_mean"]) != int(metadata["action_dim"]) or len(metadata["action_std"]) != int(metadata["action_dim"]):
        raise SystemExit("norm stats action mean/std must match action_dim")


def load_safetensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors checkpoints require the safetensors Python package") from exc

    tensors: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    with safe_open(path, framework="np") as handle:
        raw_metadata = handle.metadata() or {}
        if "vlacpp.metadata" in raw_metadata:
            metadata = json.loads(raw_metadata["vlacpp.metadata"])
        allowed = REQUIRED_TENSORS | ACTION_HEAD_TENSORS | PI05_ACTION_HEAD_TENSORS
        for name in handle.keys():
            if name not in allowed:
                continue
            array = handle.get_tensor(name)
            tensors[name] = {
                "shape": list(array.shape),
                "data": array.astype("float32").reshape(-1).tolist(),
            }
    return {"metadata": metadata, "tensors": tensors}


def read_remote_range(url: str, begin: int, end: int) -> bytes:
    request = Request(url, headers={"Range": f"bytes={begin}-{end}"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def read_local_range(path: Path, begin: int, end: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(begin)
        return handle.read(end - begin + 1)


def dtype_nbytes(dtype: str) -> int:
    if dtype == "F32":
        return 4
    if dtype in ("F16", "BF16"):
        return 2
    raise SystemExit(f"unsupported mapped tensor dtype {dtype}; expected F32, F16, or BF16")


def element_count(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def validate_tensor_payload(name: str, dtype: str, shape: list[int], raw: bytes) -> None:
    expected = element_count(shape) * dtype_nbytes(dtype)
    if len(raw) != expected:
        raise SystemExit(f"truncated tensor payload for {name}: expected {expected} bytes, got {len(raw)}")


def bfloat16_to_float32(raw: bytes) -> list[float]:
    values = []
    for (bits,) in struct.iter_unpack("<H", raw):
        values.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    return values


def tensor_payload_to_float32(dtype: str, raw: bytes) -> list[float]:
    if dtype == "F32":
        if len(raw) % 4 != 0:
            raise SystemExit("F32 tensor payload byte count is not divisible by 4")
        return list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
    if dtype == "F16":
        if len(raw) % 2 != 0:
            raise SystemExit("F16 tensor payload byte count is not divisible by 2")
        try:
            import numpy as np
        except ImportError as exc:
            raise SystemExit("F16 safetensors conversion requires numpy") from exc
        return np.frombuffer(raw, dtype="<f2").astype("float32").tolist()
    if dtype == "BF16":
        if len(raw) % 2 != 0:
            raise SystemExit("BF16 tensor payload byte count is not divisible by 2")
        return bfloat16_to_float32(raw)
    dtype_nbytes(dtype)
    raise RuntimeError("unreachable")


def load_remote_safetensors(spec: str) -> dict[str, Any]:
    url = resolve_repo_file_url(spec) if spec.startswith("hf://") or spec.startswith("ms://") else spec
    header_len = struct.unpack("<Q", read_remote_range(url, 0, 7))[0]
    header = json.loads(read_remote_range(url, 8, 8 + header_len - 1).decode("utf-8"))
    raw_metadata = header.get("__metadata__", {})
    metadata = json.loads(raw_metadata.get("vlacpp.metadata", "{}"))
    missing = sorted(REQUIRED_TENSORS - set(header))
    if missing:
        available = sorted(name for name in header if name != "__metadata__")
        preview = ", ".join(available[:20])
        raise SystemExit(
            "remote safetensors checkpoint missing required tensor(s): "
            f"{', '.join(missing)}. First available tensors: {preview}"
        )

    tensors: dict[str, dict[str, Any]] = {}
    data_begin = 8 + header_len
    for name in sorted(REQUIRED_TENSORS):
        meta = header[name]
        if meta["dtype"] != "F32":
            raise SystemExit(f"remote tensor {name} has unsupported dtype {meta['dtype']}; expected F32")
        begin, end = meta["data_offsets"]
        raw = read_remote_range(url, data_begin + begin, data_begin + end - 1)
        validate_tensor_payload(name, meta["dtype"], meta["shape"], raw)
        count = len(raw) // 4
        tensors[name] = {
            "shape": meta["shape"],
            "data": list(struct.unpack("<" + "f" * count, raw)),
        }
    return {"metadata": metadata, "tensors": tensors}


def resolve_manifest_source(manifest_path: Path, source: str) -> Path:
    path = Path(source)
    if path.is_absolute() or path.exists():
        return path
    return manifest_path.parent / path


def load_tensor_map_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    source = manifest["source"]
    if (
        source.startswith("hf://") or
        source.startswith("ms://") or
        source.startswith("https://") or
        source.startswith("http://")
    ):
        url = resolve_repo_file_url(source) if source.startswith("hf://") or source.startswith("ms://") else source
        read_range = lambda begin, end: read_remote_range(url, begin, end)
    else:
        source_path = resolve_manifest_source(manifest_path, source)
        read_range = lambda begin, end: read_local_range(source_path, begin, end)
    header_len = struct.unpack("<Q", read_range(0, 7))[0]
    data_begin = 8 + header_len
    tensors: dict[str, dict[str, Any]] = {}
    for tensor in manifest["tensors"]:
        begin, end = tensor["data_offsets"]
        raw = read_range(data_begin + begin, data_begin + end - 1)
        validate_tensor_payload(tensor["target"], tensor["dtype"], tensor["shape"], raw)
        tensors[tensor["target"]] = {
            "shape": tensor["shape"],
            "data": tensor_payload_to_float32(tensor["dtype"], raw),
        }
    return {"metadata": manifest.get("metadata", {}), "tensors": tensors}


def load_checkpoint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)
    if suffix == ".safetensors":
        return load_safetensors(path)
    raise SystemExit(f"unsupported checkpoint format for {path}; expected .json or .safetensors")


def load_checkpoint_arg(checkpoint: str | None) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    if (
        checkpoint.startswith("hf://") or
        checkpoint.startswith("ms://") or
        checkpoint.startswith("https://") or
        checkpoint.startswith("http://")
    ) and (
        checkpoint.endswith(".safetensors")
    ):
        return load_remote_safetensors(checkpoint)
    return load_checkpoint(resolve_checkpoint(checkpoint))


def infer_metadata_from_tensors(tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}
    if "pi0.velocity.weight" in tensors:
        shape = tensors["pi0.velocity.weight"]["shape"]
        if len(shape) == 2:
            inferred["action_dim"] = int(shape[0])
            if int(shape[1]) != 4:
                inferred["state_dim"] = int(shape[1]) - 3
    action_in = tensors.get("vlacpp.openpi.action_in_proj.weight")
    if action_in is not None and len(action_in["shape"]) == 2:
        inferred["action_dim"] = int(action_in["shape"][1])
    action_out = tensors.get("vlacpp.openpi.action_out_proj.weight")
    if action_out is not None and len(action_out["shape"]) == 2:
        inferred["action_dim"] = int(action_out["shape"][0])
    state_proj = tensors.get("vlacpp.openpi.state_proj.weight")
    if state_proj is not None and len(state_proj["shape"]) == 2:
        inferred["state_dim"] = int(state_proj["shape"][1])
    return inferred


def strip_model_prefix(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def find_shape(tensors: dict[str, dict[str, Any]], suffix: str) -> list[int] | None:
    for name, tensor in tensors.items():
        if strip_model_prefix(name) == suffix:
            return [int(v) for v in tensor["shape"]]
    return None


def layer_count(tensors: dict[str, dict[str, Any]], pattern: str) -> int | None:
    regex = re.compile(pattern)
    indices = set()
    for name in tensors:
        match = regex.match(strip_model_prefix(name))
        if match:
            indices.add(int(match.group(1)))
    if not indices:
        return None
    return len(indices)


def infer_openpi_graph_metadata(tensors: dict[str, dict[str, Any]]) -> dict[str, int]:
    inferred: dict[str, int] = {}
    action_in = find_shape(tensors, "vlacpp.openpi.action_in_proj.weight") or find_shape(tensors, "action_in_proj.weight")
    vision_projector = find_shape(tensors, "vlacpp.openpi.vision_projector.weight")
    patch = find_shape(
        tensors,
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
    )
    language_q = find_shape(
        tensors,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
    )
    language_k = find_shape(
        tensors,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj.weight",
    )
    language_down = find_shape(
        tensors,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight",
    )
    expert_q = find_shape(
        tensors,
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
    )
    expert_k = find_shape(
        tensors,
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.k_proj.weight",
    )
    expert_down = find_shape(
        tensors,
        "paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj.weight",
    )
    if action_in is not None and len(action_in) == 2:
        inferred["openpi_action_width"] = action_in[0]
    if patch is not None and len(patch) == 4:
        inferred["openpi_vision_width"] = patch[0]
        inferred["openpi_vision_patch_height"] = patch[2]
        inferred["openpi_vision_patch_width"] = patch[3]
    if vision_projector is not None and len(vision_projector) == 2:
        inferred.setdefault("openpi_vision_width", vision_projector[1])
        inferred.setdefault("openpi_language_width", vision_projector[0])
    if language_q is not None and len(language_q) == 2:
        inferred["openpi_language_width"] = language_q[1]
        inferred["openpi_language_q_out"] = language_q[0]
    if language_k is not None and len(language_k) == 2:
        inferred["openpi_language_kv_out"] = language_k[0]
    if language_down is not None and len(language_down) == 2:
        inferred["openpi_language_mlp_width"] = language_down[1]
    if expert_q is not None and len(expert_q) == 2:
        inferred["openpi_action_expert_width"] = expert_q[1]
        inferred["openpi_action_expert_q_out"] = expert_q[0]
    if expert_k is not None and len(expert_k) == 2:
        inferred["openpi_action_expert_kv_out"] = expert_k[0]
    if expert_down is not None and len(expert_down) == 2:
        inferred["openpi_action_expert_mlp_width"] = expert_down[1]
    counts = {
        "openpi_vision_layers": layer_count(
            tensors,
            r"paligemma_with_expert\.paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.",
        ),
        "openpi_language_layers": layer_count(
            tensors,
            r"paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)\.",
        ),
        "openpi_action_expert_layers": layer_count(
            tensors,
            r"paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\.",
        ),
    }
    for key, value in counts.items():
        if value is not None:
            inferred[key] = value
    return inferred


def build_metadata(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("metadata", {})
    tensors = checkpoint.get("tensors", {})
    inferred = {**infer_metadata_from_tensors(tensors), **infer_openpi_graph_metadata(tensors)}
    source = {**checkpoint, **inferred, **metadata}
    state_dim = int(source.get("state_dim", args.state_dim))
    action_dim = int(source.get("action_dim", args.action_dim))
    default_model_type = "pi0" if "openpi_action_width" in inferred else "mock-pi0"
    model_type = source.get("model_type", args.model_type)
    if model_type is None:
        model_type = default_model_type
    default_action_horizon = 50 if "openpi_action_expert_layers" in inferred else 32
    action_horizon = source.get("action_horizon", args.action_horizon)
    if action_horizon is None:
        action_horizon = default_action_horizon
    result = {
        "model_type": model_type,
        "image_width": int(source.get("image_width", args.image_width)),
        "image_height": int(source.get("image_height", args.image_height)),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_horizon": int(action_horizon),
        "max_token_len": int(source.get("max_token_len", args.max_token_len)),
        "image_keys": source.get("image_keys", args.image_key),
        "state_mean": source.get("state_mean", [0.0] * state_dim),
        "state_std": source.get("state_std", [1.0] * state_dim),
        "action_mean": source.get("action_mean", [0.0] * action_dim),
        "action_std": source.get("action_std", [1.0] * action_dim),
        "source_checkpoint": args.checkpoint or "",
        "format": "vlacpp-json-metadata-v0",
    }
    for key in inferred:
        if key.startswith("openpi_"):
            result[key] = int(inferred[key])
    return result


def build_tensors(
    metadata: dict[str, Any],
    checkpoint: dict[str, Any],
    init_tiny: bool,
    tensor_map_manifest: Path | None,
) -> dict[str, dict[str, Any]]:
    if tensor_map_manifest is not None:
        return checkpoint.get("tensors", {})

    tensors = checkpoint.get("tensors")
    if not tensors and init_tiny:
        tensors = default_tiny_weights(int(metadata["action_dim"]), int(metadata["state_dim"]))
    if not tensors:
        raise SystemExit(
            "checkpoint does not contain tensors; pass --init-tiny to create a tiny scaffold "
            "from metadata/config only"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        shape = [int(v) for v in tensor["shape"]]
        data = [float(v) for v in tensor["data"]]
        n = 1
        for dim in shape:
            n *= dim
        if n != len(data):
            raise SystemExit(f"tensor {name} shape {shape} expects {n} values, got {len(data)}")
        normalized[name] = {"shape": shape, "data": data}
    has_velocity = REQUIRED_TENSORS.issubset(normalized)
    has_action_head = ACTION_HEAD_TENSORS.issubset(normalized)
    has_pi05_action_head = PI05_ACTION_HEAD_TENSORS.issubset(normalized)
    if not has_velocity and not has_action_head and not has_pi05_action_head:
        missing_velocity = sorted(REQUIRED_TENSORS - set(normalized))
        missing_action_head = sorted(ACTION_HEAD_TENSORS - set(normalized))
        missing_pi05_action_head = sorted(PI05_ACTION_HEAD_TENSORS - set(normalized))
        raise SystemExit(
            "checkpoint must contain tiny velocity tensors or mapped action-head tensors; "
            f"missing velocity: {', '.join(missing_velocity)}; "
            f"missing action-head: {', '.join(missing_action_head)}; "
            f"missing pi05-action-head: {', '.join(missing_pi05_action_head)}"
        )
    action_dim = int(metadata["action_dim"])
    if has_velocity:
        feature_dim = int(metadata["state_dim"]) + 3
        if normalized["pi0.velocity.weight"]["shape"] not in ([action_dim, 4], [action_dim, feature_dim]):
            raise SystemExit("pi0.velocity.weight shape must be [action_dim, 4] or [action_dim, state_dim + 3]")
        if normalized["pi0.velocity.time_weight"]["shape"] != [action_dim]:
            raise SystemExit("pi0.velocity.time_weight shape must be [action_dim]")
    if has_action_head:
        in_weight = normalized["vlacpp.openpi.action_in_proj.weight"]["shape"]
        if len(in_weight) != 2 or in_weight[1] != action_dim:
            raise SystemExit("action_in_proj.weight shape must be [width, action_dim]")
        width = in_weight[0]
        expected = {
            "vlacpp.openpi.action_in_proj.bias": [width],
            "vlacpp.openpi.action_time_mlp_in.weight": [width, 2 * width],
            "vlacpp.openpi.action_time_mlp_in.bias": [width],
            "vlacpp.openpi.action_time_mlp_out.weight": [width, width],
            "vlacpp.openpi.action_time_mlp_out.bias": [width],
            "vlacpp.openpi.action_out_proj.weight": [action_dim, width],
            "vlacpp.openpi.action_out_proj.bias": [action_dim],
        }
        for name, shape in expected.items():
            if normalized[name]["shape"] != shape:
                raise SystemExit(f"{name} shape must be {shape}")
    if has_pi05_action_head:
        in_weight = normalized["vlacpp.openpi.action_in_proj.weight"]["shape"]
        if len(in_weight) != 2 or in_weight[1] != action_dim:
            raise SystemExit("action_in_proj.weight shape must be [width, action_dim]")
        width = in_weight[0]
        expected = {
            "vlacpp.openpi.action_in_proj.bias": [width],
            "vlacpp.openpi.pi05.time_mlp_in.weight": [width, width],
            "vlacpp.openpi.pi05.time_mlp_in.bias": [width],
            "vlacpp.openpi.pi05.time_mlp_out.weight": [width, width],
            "vlacpp.openpi.pi05.time_mlp_out.bias": [width],
            "vlacpp.openpi.action_out_proj.weight": [action_dim, width],
            "vlacpp.openpi.action_out_proj.bias": [action_dim],
        }
        for name, shape in expected.items():
            if normalized[name]["shape"] != shape:
                raise SystemExit(f"{name} shape must be {shape}")
    return normalized


def gguf_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = {
        "general.architecture": metadata["model_type"],
        "vlacpp.model_type": metadata["model_type"],
        "vlacpp.image_width": metadata["image_width"],
        "vlacpp.image_height": metadata["image_height"],
        "vlacpp.state_dim": metadata["state_dim"],
        "vlacpp.action_dim": metadata["action_dim"],
        "vlacpp.action_horizon": metadata["action_horizon"],
        "vlacpp.max_token_len": metadata["max_token_len"],
        "vlacpp.image_keys": metadata["image_keys"],
        "vlacpp.state_mean": metadata["state_mean"],
        "vlacpp.state_std": metadata["state_std"],
        "vlacpp.action_mean": metadata["action_mean"],
        "vlacpp.action_std": metadata["action_std"],
    }
    optional_ints = {
        "vlacpp.openpi.action_width": "openpi_action_width",
        "vlacpp.openpi.vision_width": "openpi_vision_width",
        "vlacpp.openpi.vision_patch_height": "openpi_vision_patch_height",
        "vlacpp.openpi.vision_patch_width": "openpi_vision_patch_width",
        "vlacpp.openpi.vision_layers": "openpi_vision_layers",
        "vlacpp.openpi.language_width": "openpi_language_width",
        "vlacpp.openpi.language_q_out": "openpi_language_q_out",
        "vlacpp.openpi.language_kv_out": "openpi_language_kv_out",
        "vlacpp.openpi.language_mlp_width": "openpi_language_mlp_width",
        "vlacpp.openpi.language_layers": "openpi_language_layers",
        "vlacpp.openpi.action_expert_width": "openpi_action_expert_width",
        "vlacpp.openpi.action_expert_q_out": "openpi_action_expert_q_out",
        "vlacpp.openpi.action_expert_kv_out": "openpi_action_expert_kv_out",
        "vlacpp.openpi.action_expert_mlp_width": "openpi_action_expert_mlp_width",
        "vlacpp.openpi.action_expert_layers": "openpi_action_expert_layers",
    }
    for key, source in optional_ints.items():
        if source in metadata:
            result[key] = int(metadata[source])
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", help="optional JSON metadata/config file, local path, hf:// URI, or ms:// URI")
    parser.add_argument("--norm-stats", help="optional OpenPI norm_stats JSON file, local path, hf:// URI, or ms:// URI")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-format", choices=["auto", "json", "gguf"], default="auto")
    parser.add_argument("--init-tiny", action="store_true", help="create tiny reference tensors when checkpoint has metadata only")
    parser.add_argument("--tensor-map-manifest", type=Path, help="convert tensors listed in a map-openpi-tensors manifest")
    parser.add_argument("--model-type", choices=["mock-pi0", "pi0", "pi05"])
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--max-token-len", type=int, default=250)
    parser.add_argument("--image-key", action="append", default=["base_0_rgb"])
    args = parser.parse_args()

    checkpoint = load_tensor_map_manifest(args.tensor_map_manifest) if args.tensor_map_manifest is not None else load_checkpoint_arg(args.checkpoint)
    config = load_json_arg(args.config)
    if config:
        checkpoint = {**checkpoint, "metadata": {**config, **checkpoint.get("metadata", {})}}
    metadata = build_metadata(args, checkpoint)
    norm_stats = load_json_arg(args.norm_stats)
    if norm_stats:
        apply_norm_stats(metadata, norm_stats)
    output_format = args.output_format
    if output_format == "auto":
        output_format = "json" if args.output.suffix.lower() == ".json" else "gguf"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return

    tensors = build_tensors(metadata, checkpoint, args.init_tiny, args.tensor_map_manifest)
    write_gguf(args.output, gguf_metadata(metadata), tensors)


if __name__ == "__main__":
    main()
