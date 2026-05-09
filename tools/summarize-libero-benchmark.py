#!/usr/bin/env python3
"""Summarize final LIBERO benchmark evidence files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def mean_metric(obj: dict[str, Any], key: str) -> float | None:
    value = obj.get(key)
    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
        return float(value["mean"])
    return None


def fmt_range(values: list[float]) -> str:
    return f"{min(values):.3f}-{max(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlacpp-cuda", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-cuda", type=Path, nargs="+", required=True)
    parser.add_argument("--vlacpp-cpu", type=Path, required=True)
    parser.add_argument("--baseline-cpu-success", default="42/50")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vlacpp_cuda = [load(path) for path in args.vlacpp_cuda]
    baseline_cuda = [load(path) for path in args.baseline_cuda]
    vlacpp_cpu = load(args.vlacpp_cpu)

    vlacpp_cuda_successes = sum(int(obj["successes"]) for obj in vlacpp_cuda)
    vlacpp_cuda_episodes = sum(int(obj["episodes"]) for obj in vlacpp_cuda)
    baseline_cuda_successes = sum(int(obj["successes"]) for obj in baseline_cuda)
    baseline_cuda_episodes = sum(int(obj["episodes"]) for obj in baseline_cuda)

    vlacpp_cuda_times = [
        value
        for obj in vlacpp_cuda
        if (value := mean_metric(obj, "chunk_policy_e2e_time_excluding_prefix_s")) is not None
    ]
    baseline_cuda_times = [
        value
        for obj in baseline_cuda
        if (value := mean_metric(obj, "chunk_policy_e2e_time_excluding_prefix_s")) is not None
    ]

    summary = {
        "status": "not_complete",
        "cuda": {
            "vlacpp_success_combined": f"{vlacpp_cuda_successes}/{vlacpp_cuda_episodes}",
            "baseline_success_combined": f"{baseline_cuda_successes}/{baseline_cuda_episodes}",
            "vlacpp_warm_chunk_e2e_s": fmt_range(vlacpp_cuda_times),
            "baseline_warm_chunk_e2e_s": fmt_range(baseline_cuda_times),
            "status": "failed_speed_gate",
        },
        "cpu": {
            "vlacpp_success": f"{int(vlacpp_cpu['successes'])}/{int(vlacpp_cpu['episodes'])}",
            "baseline_success": args.baseline_cpu_success,
            "vlacpp_warm_chunk_e2e_s": mean_metric(vlacpp_cpu, "chunk_policy_e2e_time_excluding_prefix_s"),
            "status": "failed_success_gate",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
