#!/usr/bin/env python3

"""
Submit a small stage-1 hyperparameter sweep for the hierarchical JEPA launcher.

The defaults are chosen for a roughly two-hour single-GPU budget assuming the
full 50-epoch run is about seven hours: three configs, three epochs each.

Run on the cluster login node:

  python scripts/submit_hierarchical_tuning.py --submit
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class SweepConfig:
    name: str
    within_mask_ratio: float
    weight_decay: float
    predictor_dim: int
    predictor_depth: int
    lambda_cross: float
    lambda_action: float


DEFAULT_CONFIGS = [
    SweepConfig(
        name="base",
        within_mask_ratio=0.85,
        weight_decay=0.10,
        predictor_dim=128,
        predictor_depth=2,
        lambda_cross=1.0,
        lambda_action=0.10,
    ),
    SweepConfig(
        name="smallpred",
        within_mask_ratio=0.85,
        weight_decay=0.10,
        predictor_dim=96,
        predictor_depth=2,
        lambda_cross=1.0,
        lambda_action=0.10,
    ),
    SweepConfig(
        name="cross05",
        within_mask_ratio=0.85,
        weight_decay=0.10,
        predictor_dim=128,
        predictor_depth=2,
        lambda_cross=0.5,
        lambda_action=0.10,
    ),
    SweepConfig(
        name="future05",
        within_mask_ratio=0.85,
        weight_decay=0.10,
        predictor_dim=128,
        predictor_depth=2,
        lambda_cross=1.0,
        lambda_action=0.05,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a compact hierarchical JEPA sweep")
    parser.add_argument("--submit", action="store_true", help="Actually submit jobs with sbatch")
    parser.add_argument("--account", default="torch_pr_494_general")
    parser.add_argument("--partition", default="", help="Optional Slurm partition override")
    parser.add_argument("--gres", default="", help="Optional GPU request override, e.g. gpu:h200:1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--run-prefix", default="tune-hier")
    parser.add_argument(
        "--runs-root",
        default="/scratch/$USER/active-matter-ssl/runs/tuning_hierarchical_stage1",
    )
    parser.add_argument(
        "--slurm-script",
        default="scripts/train_ssl_hierarchical.slurm",
    )
    return parser.parse_args()


def build_export_string(args: argparse.Namespace, cfg: SweepConfig) -> str:
    run_name = f"{args.run_prefix}-{cfg.name}"
    checkpoint_dir = os.path.join(args.runs_root, run_name)
    exports = {
        "ALL": None,
        "RUN_NAME": run_name,
        "CHECKPOINT_DIR": checkpoint_dir,
        "EPOCHS": str(args.epochs),
        "VAL_EVERY": str(args.val_every),
        "LOG_EVERY": str(args.log_every),
        "WITHIN_MASK_RATIO": str(cfg.within_mask_ratio),
        "WEIGHT_DECAY": str(cfg.weight_decay),
        "PREDICTOR_DIM": str(cfg.predictor_dim),
        "PREDICTOR_DEPTH": str(cfg.predictor_depth),
        "LAMBDA_CROSS": str(cfg.lambda_cross),
        "LAMBDA_ACTION": str(cfg.lambda_action),
    }
    parts = []
    for key, value in exports.items():
        if value is None:
            parts.append(key)
        else:
            parts.append(f"{key}={value}")
    return ",".join(parts)


def build_command(args: argparse.Namespace, cfg: SweepConfig) -> list[str]:
    cmd = ["sbatch", f"--account={args.account}"]
    if args.partition:
        cmd.append(f"--partition={args.partition}")
    if args.gres:
        cmd.append(f"--gres={args.gres}")
    cmd.append(f"--export={build_export_string(args, cfg)}")
    cmd.append(args.slurm_script)
    return cmd


def main() -> None:
    args = parse_args()
    args.runs_root = os.path.expandvars(args.runs_root)

    for cfg in DEFAULT_CONFIGS:
        cmd = build_command(args, cfg)
        rendered = " ".join(shlex.quote(part) for part in cmd)
        print(rendered)
        if args.submit:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
