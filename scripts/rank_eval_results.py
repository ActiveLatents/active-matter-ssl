#!/usr/bin/env python3

"""
Rank finished JEPA evaluation runs by a chosen metric.

This reads eval_results.pt files produced by src.evaluate and sorts them by
validation or test performance. It is intended for fast bookkeeping after a
tuning sweep.

Example:
  python scripts/rank_eval_results.py \
    --runs-root /scratch/$USER/active-matter-ssl/runs/tuning_hybrid_stage1 \
    --metric linear_probe.val_mse
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


METRIC_PATHS = {
    "linear_probe.val_mse": ("linear_probe", "val_mse"),
    "linear_probe.test_mse": ("linear_probe", "test_mse"),
    "knn.val_mse": ("knn", "val_mse"),
    "knn.test_mse": ("knn", "test_mse"),
    "linear_probe.val_mse_alpha": ("linear_probe", "val_mse_alpha"),
    "linear_probe.val_mse_zeta": ("linear_probe", "val_mse_zeta"),
    "knn.val_mse_alpha": ("knn", "val_mse_alpha"),
    "knn.val_mse_zeta": ("knn", "val_mse_zeta"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank eval_results.pt files by metric")
    parser.add_argument(
        "--runs-root",
        default="/scratch/$USER/active-matter-ssl/runs/tuning_hybrid_stage1",
        help="Root directory containing run subdirectories",
    )
    parser.add_argument(
        "--metric",
        default="linear_probe.val_mse",
        choices=sorted(METRIC_PATHS),
    )
    return parser.parse_args()


def metric_value(payload: dict, metric: str) -> float:
    outer, inner = METRIC_PATHS[metric]
    return float(payload[outer][inner])


def main() -> None:
    args = parse_args()
    runs_root = Path(os.path.expandvars(args.runs_root))
    ranked = []

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        eval_path = run_dir / "eval_results.pt"
        if not eval_path.exists():
            continue
        payload = torch.load(eval_path, map_location="cpu", weights_only=False)
        ranked.append((metric_value(payload, args.metric), run_dir.name, payload))

    if not ranked:
        raise SystemExit(f"No eval_results.pt files found under {runs_root}")

    ranked.sort(key=lambda item: item[0])

    print(f"Ranking by {args.metric}\n")
    for rank, (score, name, payload) in enumerate(ranked, start=1):
        linear = payload["linear_probe"]
        knn = payload["knn"]
        print(
            f"{rank:>2}. {name:<32s} "
            f"{args.metric}={score:.6f} | "
            f"linear val={linear['val_mse']:.6f} test={linear['test_mse']:.6f} | "
            f"knn val={knn['val_mse']:.6f} test={knn['test_mse']:.6f}"
        )


if __name__ == "__main__":
    main()
