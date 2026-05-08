#!/usr/bin/env python3
"""Run a minimal official OpenPI PyTorch smoke against a safetensors checkpoint.

ModelScope pi0 checkpoints wrap OpenPI parameter names in a top-level
``model.`` prefix. Official OpenPI's PyTorch loader expects the unprefixed
module names, so this smoke copies tensors into the official module with that
prefix mapping instead of rewriting the 14 GB checkpoint on disk.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import time


def openpi_available() -> bool:
    return importlib.util.find_spec("openpi") is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--openpi-config", default="pi0_aloha")
    parser.add_argument("--device", default=None, help="torch device, defaults to cuda when available")
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--image-value", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--allow-missing-openpi", action="store_true")
    args = parser.parse_args()

    if not openpi_available():
        if args.allow_missing_openpi:
            print(json.dumps({"status": "skipped", "reason": "official OpenPI is not installed"}, indent=2))
            return
        raise SystemExit("official OpenPI is not installed; run this with the OpenPI Python environment")

    import torch
    from safetensors import safe_open

    from openpi.models import model as openpi_model
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import config as train_config

    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint
    cfg = train_config.get_config(args.openpi_config)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = pi0_pytorch.PI0Pytorch(cfg.model)
    loaded = 0
    skipped: list[str] = []
    started = time.time()
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors_file:
        keys = set(tensors_file.keys())
        metadata = tensors_file.metadata() or {}
        targets = dict(model.named_parameters())
        targets.update(dict(model.named_buffers()))
        for name, target in targets.items():
            source = None
            if name in keys:
                source = name
            elif "model." + name in keys:
                source = "model." + name
            elif "model." + name in metadata and metadata["model." + name] in keys:
                source = metadata["model." + name]

            if source is None:
                skipped.append(name)
                continue

            value = tensors_file.get_tensor(source)
            if tuple(value.shape) != tuple(target.shape):
                raise SystemExit(
                    f"shape mismatch for {name}: expected {tuple(target.shape)} got {tuple(value.shape)} from {source}"
                )
            target.data.copy_(value.to(dtype=target.dtype))
            loaded += 1
            del value
    load_seconds = time.time() - started

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to(device)
    model.eval()

    image = torch.full((1, 224, 224, 3), args.image_value, dtype=torch.uint8, device=device)
    obs = openpi_model.Observation.from_dict(
        {
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": torch.zeros_like(image),
                "right_wrist_0_rgb": torch.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": torch.ones((1,), dtype=torch.bool, device=device),
                "left_wrist_0_rgb": torch.zeros((1,), dtype=torch.bool, device=device),
                "right_wrist_0_rgb": torch.zeros((1,), dtype=torch.bool, device=device),
            },
            "state": torch.zeros((1, cfg.model.action_dim), dtype=torch.float32, device=device),
            "tokenized_prompt": torch.zeros((1, cfg.model.max_token_len), dtype=torch.long, device=device),
            "tokenized_prompt_mask": torch.zeros((1, cfg.model.max_token_len), dtype=torch.bool, device=device),
        }
    )
    noise = torch.zeros((1, cfg.model.action_horizon, cfg.model.action_dim), dtype=torch.float32, device=device)

    sample_started = time.time()
    with torch.no_grad():
        actions = model.sample_actions(device, obs, noise=noise, num_steps=args.num_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    first = actions[0, 0, : min(8, actions.shape[-1])].detach().cpu().to(torch.float32)
    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint": str(checkpoint),
                "openpi_config": args.openpi_config,
                "device": str(device),
                "loaded_tensors": loaded,
                "skipped_tensors": skipped,
                "load_seconds": round(load_seconds, 3),
                "sample_seconds": round(time.time() - sample_started, 3),
                "actions_shape": list(actions.shape),
                "actions_sum": float(actions.detach().cpu().to(torch.float32).sum()),
                "actions_first": [float(v) for v in first],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
