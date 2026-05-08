#!/usr/bin/env python3
"""Benchmark OpenPI PyTorch and vlacpp on deterministic LIBERO-like inputs."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def import_compare_tools(repo: Path) -> Any:
    path = repo / "tools" / "compare-openpi-policy.py"
    spec = importlib.util.spec_from_file_location("compare_openpi_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenPiPytorchPrefixPolicy:
    def __init__(self, *, config_name: str, checkpoint: str, device_name: str | None, n_threads: int) -> None:
        try:
            import torch
            from safetensors import safe_open

            from openpi.models import model as openpi_model
            from openpi.models import tokenizer as openpi_tokenizer
            from openpi.models_pytorch import pi0_pytorch
            from openpi.training import config as train_config
        except ImportError as exc:
            raise SystemExit("official OpenPI PyTorch dependencies are not installed") from exc

        self.torch = torch
        if n_threads > 0:
            torch.set_num_threads(n_threads)
        self.openpi_model = openpi_model
        self.openpi_tokenizer = openpi_tokenizer
        self.checkpoint = Path(checkpoint)
        if self.checkpoint.is_dir():
            self.checkpoint = self.checkpoint / "model.safetensors"
        if not self.checkpoint.exists():
            raise SystemExit(f"OpenPI safetensors checkpoint not found: {self.checkpoint}")

        cfg = train_config.get_config(config_name)
        self.cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
        self.device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = pi0_pytorch.PI0Pytorch(self.cfg.model)
        with safe_open(self.checkpoint, framework="pt", device="cpu") as tensors_file:
            keys = set(tensors_file.keys())
            metadata = tensors_file.metadata() or {}
            targets = dict(self.model.named_parameters())
            targets.update(dict(self.model.named_buffers()))
            for name, target in targets.items():
                source = None
                if name in keys:
                    source = name
                elif "model." + name in keys:
                    source = "model." + name
                elif "model." + name in metadata and metadata["model." + name] in keys:
                    source = metadata["model." + name]
                if source is None:
                    continue
                value = tensors_file.get_tensor(source)
                if tuple(value.shape) != tuple(target.shape):
                    raise SystemExit(
                        f"shape mismatch for {name}: expected {tuple(target.shape)} "
                        f"got {tuple(value.shape)} from {source}"
                    )
                target.data.copy_(value.to(dtype=target.dtype))
                del value

        self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = None

    def infer(self, obs: dict[str, Any], noise: np.ndarray, *, steps: int, image_height: int, image_width: int) -> np.ndarray:
        torch = self.torch
        image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        image_dict = {}
        image_mask = {}
        for key in image_keys:
            if key in obs:
                image = np.asarray(obs[key], dtype=np.uint8)
                image_dict[key] = torch.as_tensor(image, dtype=torch.uint8, device=self.device).unsqueeze(0)
                image_mask[key] = torch.ones((1,), dtype=torch.bool, device=self.device)
            else:
                image_dict[key] = torch.zeros((1, image_height, image_width, 3), dtype=torch.uint8, device=self.device)
                image_mask[key] = torch.zeros((1,), dtype=torch.bool, device=self.device)

        state = np.asarray(obs["state"], dtype=np.float32)
        prompt = str(obs.get("prompt", ""))
        if prompt:
            if self.tokenizer is None:
                self.tokenizer = self.openpi_tokenizer.PaligemmaTokenizer(self.cfg.model.max_token_len)
            tokenized_prompt, tokenized_prompt_mask = self.tokenizer.tokenize(prompt)
        else:
            tokenized_prompt = np.zeros((self.cfg.model.max_token_len,), dtype=np.int64)
            tokenized_prompt_mask = np.zeros((self.cfg.model.max_token_len,), dtype=bool)

        observation = self.openpi_model.Observation.from_dict(
            {
                "image": image_dict,
                "image_mask": image_mask,
                "state": torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0),
                "tokenized_prompt": torch.as_tensor(tokenized_prompt, dtype=torch.long, device=self.device).unsqueeze(0),
                "tokenized_prompt_mask": torch.as_tensor(
                    tokenized_prompt_mask,
                    dtype=torch.bool,
                    device=self.device,
                ).unsqueeze(0),
            }
        )
        noise_tensor = torch.as_tensor(noise, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            actions = self.model.sample_actions(self.device, observation, noise=noise_tensor, num_steps=steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return np.asarray(actions[0].detach().cpu().to(torch.float32), dtype=np.float32)


def make_observations(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.data_seed)
    observations = []
    for index in range(args.samples):
        state = rng.normal(loc=0.0, scale=0.15, size=(args.state_dim,)).astype(np.float32)
        image = np.full((args.image_height, args.image_width, 3), args.image_value + index % 7, dtype=np.uint8)
        image[:, :, 1] = np.clip(image[:, :, 1].astype(np.int16) + index * 3, 0, 255).astype(np.uint8)
        observations.append(
            {
                "state": state,
                "prompt": args.prompt,
                args.image_key: image,
            }
        )
    return observations


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-config", default="pi0_aloha")
    parser.add_argument("--openpi-checkpoint", default="ckpts/lerobot-pi0/model.safetensors")
    parser.add_argument("--vlacpp-model", type=Path, default=Path("/tmp/vlacpp-pi0-full-local.gguf"))
    parser.add_argument("--vlacpp-library", type=Path, default=repo / "build" / "libvlacpp.so")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-seed", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--image-key", default="base_0_rgb")
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-value", type=int, default=127)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="device to use for both OpenPI and vlacpp")
    parser.add_argument("--pytorch-device", default=None, help="override OpenPI device, e.g. cuda:0")
    parser.add_argument("--vlacpp-backend", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for both PyTorch and vlacpp; 0 keeps runtime defaults")
    parser.add_argument("--allow-mixed-device", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(repo / "python"))
    import vlacpp

    compare_tools = import_compare_tools(repo)
    observations = make_observations(args)
    if not observations:
        raise SystemExit("--samples must be positive")

    pytorch_device = args.pytorch_device or args.device
    vlacpp_backend_name = args.vlacpp_backend or args.device
    openpi = OpenPiPytorchPrefixPolicy(
        config_name=args.openpi_config,
        checkpoint=args.openpi_checkpoint,
        device_name=pytorch_device,
        n_threads=args.threads,
    )
    openpi_device_type = openpi.device.type
    if openpi_device_type != vlacpp_backend_name and not args.allow_mixed_device:
        raise SystemExit(
            f"mixed-device benchmark rejected: OpenPI={openpi.device} vlacpp={vlacpp_backend_name}; "
            "pass matching --device or --allow-mixed-device"
        )
    vlacpp_backend = vlacpp.VLACPP_BACKEND_CUDA if vlacpp_backend_name == "cuda" else vlacpp.VLACPP_BACKEND_CPU
    vlacpp_policy = vlacpp.Pi0Policy(
        args.vlacpp_model,
        library_path=args.vlacpp_library,
        backend=vlacpp_backend,
        n_threads=args.threads,
        seed=args.seed,
        flow_steps=args.steps,
    )
    action_horizon = int(openpi.cfg.model.action_horizon)
    action_dim = int(openpi.cfg.model.action_dim)
    noise = compare_tools.cpp_mt19937_normals(args.seed, action_horizon * action_dim).reshape(action_horizon, action_dim)

    for obs in observations[: args.warmup]:
        vlacpp_policy.reset_cache()
        _ = vlacpp_policy.infer(state=obs["state"], images={args.image_key: obs[args.image_key]}, prompt=args.prompt)
        _ = openpi.infer(obs, noise, steps=args.steps, image_height=args.image_height, image_width=args.image_width)

    rows = []
    openpi_times = []
    vlacpp_times = []
    max_abs_values = []
    for index, obs in enumerate(observations):
        start = time.perf_counter()
        openpi_actions = openpi.infer(
            obs,
            noise,
            steps=args.steps,
            image_height=args.image_height,
            image_width=args.image_width,
        )
        openpi_elapsed = time.perf_counter() - start

        vlacpp_policy.reset_cache()
        start = time.perf_counter()
        vlacpp_actions = vlacpp_policy.infer(
            state=obs["state"],
            images={args.image_key: obs[args.image_key]},
            prompt=args.prompt,
        )
        vlacpp_elapsed = time.perf_counter() - start

        if openpi_actions.shape != vlacpp_actions.shape:
            raise SystemExit(f"shape mismatch: openpi={openpi_actions.shape} vlacpp={vlacpp_actions.shape}")
        max_abs = float(np.max(np.abs(openpi_actions - vlacpp_actions)))
        rows.append(
            {
                "sample": index,
                "max_abs": max_abs,
                "openpi_time_s": openpi_elapsed,
                "vlacpp_time_s": vlacpp_elapsed,
            }
        )
        openpi_times.append(openpi_elapsed)
        vlacpp_times.append(vlacpp_elapsed)
        max_abs_values.append(max_abs)

    result = {
        "status": "ok",
        "benchmark": "synthetic-libero-like",
        "samples": args.samples,
        "warmup": args.warmup,
        "steps": args.steps,
        "seed": args.seed,
        "prompt": args.prompt,
        "fair_device_compare": openpi_device_type == vlacpp_backend_name,
        "openpi_device": str(openpi.device),
        "vlacpp_backend": vlacpp_backend_name,
        "threads": args.threads,
        "openpi_num_threads": int(openpi.torch.get_num_threads()),
        "capability": vlacpp_policy.capability,
        "shape": list(vlacpp_actions.shape),
        "max_abs": max(max_abs_values),
        "openpi_time_s": summarize(openpi_times),
        "vlacpp_time_s": summarize(vlacpp_times),
        "speedup_openpi_over_vlacpp": float(np.mean(openpi_times) / np.mean(vlacpp_times)),
        "rows": rows,
    }
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)

    vlacpp_policy.close()


if __name__ == "__main__":
    main()
