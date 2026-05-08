#!/usr/bin/env python3
"""Check the OpenPI graph parts against the llama.cpp reuse boundary."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_summarizer() -> Any:
    path = Path(__file__).with_name("summarize-openpi-graph.py")
    spec = importlib.util.spec_from_file_location("summarize_openpi_graph", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_file(path: Path, needle: str | None = None) -> None:
    if not path.exists():
        raise SystemExit(f"missing required llama.cpp file: {path}")
    if needle is not None and needle not in path.read_text(encoding="utf-8", errors="ignore"):
        raise SystemExit(f"{path} does not contain required marker: {needle}")


def require_contiguous(name: str, summary: dict[str, Any]) -> None:
    if not summary["contiguous"] or summary["count"] <= 0:
        raise SystemExit(f"{name} layers are not contiguous")


def build_plan(repo: Path, graph_summary: dict[str, Any]) -> dict[str, Any]:
    llama_dir = repo / "third_party" / "llama.cpp"
    require_file(llama_dir / "ggml" / "include" / "ggml.h", "ggml_new_tensor_2d")
    require_file(llama_dir / "gguf-py" / "gguf" / "gguf_writer.py", "class GGUFWriter")
    require_file(llama_dir / "tools" / "mtmd" / "mtmd.h", "mtmd_context_params_default")
    require_file(llama_dir / "tools" / "mtmd" / "models" / "siglip.cpp", "clip_graph_siglip")
    require_file(llama_dir / "tools" / "mtmd" / "clip.cpp", "siglip2 naflex")

    require_contiguous("vision_tower", graph_summary["vision_tower"])
    require_contiguous("language_model", graph_summary["language_model"])
    require_contiguous("action_expert", graph_summary["action_expert"])

    vision = graph_summary["vision_tower"]
    action = graph_summary["action_head"]
    language = graph_summary["language_model"]
    expert = graph_summary["action_expert"]
    if vision["patch"] != [14, 14]:
        raise SystemExit(f"unexpected OpenPI vision patch: {vision['patch']}")
    if action["width"] != expert["width"]:
        raise SystemExit("action head width must match action expert width")
    if language["kv_out"] != expert["kv_out"]:
        raise SystemExit("language and action expert KV widths should stay aligned")

    return {
        "llama_cpp": str(llama_dir),
        "gguf_writer": {
            "decision": "reuse",
            "component": "llama.cpp gguf-py GGUFWriter",
        },
        "vision_tower": {
            "decision": "reuse",
            "component": "mtmd SigLIP/ViT graph",
            "width": vision["width"],
            "layers": vision["count"],
            "patch": vision["patch"],
        },
        "language_model": {
            "decision": "reuse-with-custom-wiring",
            "component": "llama.cpp Gemma graph pieces",
            "width": language["width"],
            "layers": language["count"],
            "kv_out": language["kv_out"],
        },
        "action_expert": {
            "decision": "custom-ggml-graph",
            "component": "OpenPI expert stream and masks",
            "width": expert["width"],
            "layers": expert["count"],
            "kv_out": expert["kv_out"],
        },
        "action_head": {
            "decision": "custom-direct-ggml",
            "component": "OpenPI action/time/state projections and flow sampling",
            "width": action["width"],
            "action_dim": action["action_dim"],
            "has_state_proj": action["has_state_proj"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="OpenPI graph manifest, inspect JSON, safetensors source, hf://, or ms://")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--expect-vision-decision")
    parser.add_argument("--expect-action-expert-decision")
    parser.add_argument("--expect-action-head-decision")
    args = parser.parse_args()

    summarizer = load_summarizer()
    summary = summarizer.summarize(summarizer.load_rows(args.input))
    plan = build_plan(args.repo, summary)

    expectations = {
        "vision_tower": args.expect_vision_decision,
        "action_expert": args.expect_action_expert_decision,
        "action_head": args.expect_action_head_decision,
    }
    for key, expected in expectations.items():
        if expected is not None and plan[key]["decision"] != expected:
            raise SystemExit(f"unexpected {key} decision: got {plan[key]['decision']}, expected {expected}")

    if args.json:
        print(json.dumps(plan, indent=2))
        return
    print(f"vision_tower: {plan['vision_tower']['decision']} {plan['vision_tower']['component']}")
    print(f"language_model: {plan['language_model']['decision']} {plan['language_model']['component']}")
    print(f"action_expert: {plan['action_expert']['decision']} {plan['action_expert']['component']}")
    print(f"action_head: {plan['action_head']['decision']} {plan['action_head']['component']}")


if __name__ == "__main__":
    main()
