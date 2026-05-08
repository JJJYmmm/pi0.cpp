#!/usr/bin/env python3
"""Summarize OpenPI graph dimensions from a tensor-map manifest or safetensors source."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def inspect_source(source: str) -> list[dict[str, Any]]:
    script = Path(__file__).with_name("inspect-safetensors.py")
    raw = subprocess.check_output(
        [sys.executable, str(script), source, "--json", "--include-metadata", "--limit", "100000"],
        text=True,
    )
    return json.loads(raw)["tensors"]


def load_rows(path_or_source: str) -> list[dict[str, Any]]:
    path = Path(path_or_source)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if "tensors" not in loaded:
            raise SystemExit("input JSON must contain a tensors array")
        return loaded["tensors"]
    return inspect_source(path_or_source)


def strip_model_prefix(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def layer_indices(names: list[str], pattern: str) -> list[int]:
    regex = re.compile(pattern)
    indices = []
    for name in names:
        match = regex.match(strip_model_prefix(name))
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def find_shape(rows: list[dict[str, Any]], suffix: str) -> list[int] | None:
    for row in rows:
        if strip_model_prefix(row["source"] if "source" in row else row["name"]) == suffix:
            return [int(v) for v in row["shape"]]
    return None


def layer_summary(indices: list[int]) -> dict[str, Any]:
    return {
        "count": len(indices),
        "first": indices[0] if indices else None,
        "last": indices[-1] if indices else None,
        "contiguous": indices == list(range(indices[0], indices[-1] + 1)) if indices else True,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = [row["source"] if "source" in row else row["name"] for row in rows]
    action_in = find_shape(rows, "action_in_proj.weight")
    action_out = find_shape(rows, "action_out_proj.weight")
    state_proj = find_shape(rows, "state_proj.weight")
    patch = find_shape(
        rows,
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
    )
    language_q = find_shape(
        rows,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
    )
    language_k = find_shape(
        rows,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj.weight",
    )
    language_down = find_shape(
        rows,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight",
    )
    expert_q = find_shape(
        rows,
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
    )
    expert_k = find_shape(
        rows,
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.k_proj.weight",
    )
    expert_down = find_shape(
        rows,
        "paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj.weight",
    )

    vision_layers = layer_indices(
        names,
        r"paligemma_with_expert\.paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.",
    )
    language_layers = layer_indices(
        names,
        r"paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)\.",
    )
    expert_layers = layer_indices(
        names,
        r"paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\.",
    )

    return {
        "tensor_count": len(rows),
        "modelscope_model_prefix": any(name.startswith("model.") for name in names),
        "action_head": {
            "width": action_in[0] if action_in else None,
            "action_dim": action_in[1] if action_in else (action_out[0] if action_out else None),
            "has_state_proj": state_proj is not None,
        },
        "vision_tower": {
            **layer_summary(vision_layers),
            "width": patch[0] if patch else None,
            "patch": patch[2:] if patch and len(patch) == 4 else None,
        },
        "language_model": {
            **layer_summary(language_layers),
            "width": language_q[1] if language_q else None,
            "q_out": language_q[0] if language_q else None,
            "kv_out": language_k[0] if language_k else None,
            "mlp_width": language_down[1] if language_down else None,
        },
        "action_expert": {
            **layer_summary(expert_layers),
            "width": expert_q[1] if expert_q else None,
            "q_out": expert_q[0] if expert_q else None,
            "kv_out": expert_k[0] if expert_k else None,
            "mlp_width": expert_down[1] if expert_down else None,
        },
    }


def expect(name: str, actual: Any, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise SystemExit(f"unexpected {name}: got {actual}, expected {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="tensor-map manifest JSON, inspect JSON, local safetensors, hf://, ms://, or HTTPS source")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--expect-action-dim", type=int)
    parser.add_argument("--expect-action-width", type=int)
    parser.add_argument("--expect-vision-layers", type=int)
    parser.add_argument("--expect-language-layers", type=int)
    parser.add_argument("--expect-action-expert-layers", type=int)
    args = parser.parse_args()

    summary = summarize(load_rows(args.input))
    expect("action_dim", summary["action_head"]["action_dim"], args.expect_action_dim)
    expect("action_width", summary["action_head"]["width"], args.expect_action_width)
    expect("vision_layers", summary["vision_tower"]["count"], args.expect_vision_layers)
    expect("language_layers", summary["language_model"]["count"], args.expect_language_layers)
    expect("action_expert_layers", summary["action_expert"]["count"], args.expect_action_expert_layers)

    if args.json:
        print(json.dumps(summary, indent=2))
        return
    print(f"tensors: {summary['tensor_count']}")
    print(f"modelscope_model_prefix: {summary['modelscope_model_prefix']}")
    print(
        "action_head: "
        f"width={summary['action_head']['width']} action_dim={summary['action_head']['action_dim']} "
        f"state_proj={summary['action_head']['has_state_proj']}"
    )
    for key in ("vision_tower", "language_model", "action_expert"):
        item = summary[key]
        print(f"{key}: layers={item['count']} first={item['first']} last={item['last']} contiguous={item['contiguous']}")


if __name__ == "__main__":
    main()
