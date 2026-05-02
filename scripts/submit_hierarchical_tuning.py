#!/usr/bin/env python3

"""
Submit a full hierarchical JEPA hyperparameter search and optionally chain the
frozen-feature evaluation for each trial.

This searches over all active training loss weights used by the hierarchical
model:
  - lambda_within
  - lambda_cross
  - lambda_future
  - lambda_sigreg
  - lambda_action

It also searches a few high-impact non-loss hyperparameters:
  - within_mask_ratio
  - weight_decay
  - predictor_dim
  - predictor_depth

The search strategy can be either:
  - a small explicit grid
  - randomized sampling from that grid

Examples:
  python scripts/submit_hierarchical_tuning.py
  python scripts/submit_hierarchical_tuning.py --submit
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import shlex
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class SweepConfig:
    name: str
    within_mask_ratio: float
    weight_decay: float
    predictor_dim: int
    predictor_depth: int
    lambda_within: float
    lambda_cross: float
    lambda_future: float
    lambda_sigreg: float
    lambda_action: float
    lambda_latent_kl: float = 0.0
    lambda_vorticity: float = 0.0
    lambda_spectral: float = 0.0


def parse_csv_floats(text: str) -> list[float]:
    normalized = text.replace(":", ",")
    return [float(item) for item in normalized.split(",") if item]


def parse_csv_ints(text: str) -> list[int]:
    normalized = text.replace(":", ",")
    return [int(item) for item in normalized.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a grid or random-search sweep for hierarchical JEPA")
    parser.add_argument("--submit", action="store_true", help="Actually submit jobs with sbatch")
    parser.add_argument("--account", default="torch_pr_494_general")
    parser.add_argument("--partition", default="", help="Optional Slurm partition override")
    parser.add_argument("--gres", default="", help="Optional GPU request override, e.g. gpu:h200:1")
    parser.add_argument("--search-mode", choices=["grid", "random"], default="grid")
    parser.add_argument("--num-trials", type=int, default=8, help="Number of random configs to sample")
    parser.add_argument("--max-configs", type=int, default=0,
                        help="Optional cap on emitted configs in grid mode; 0 means no cap")
    parser.add_argument("--epochs", type=int, default=3, help="Short pretraining horizon per trial")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--probe-epochs", type=int, default=20, help="Short probe horizon for chained eval jobs")
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--run-prefix", default="search-hier")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--runs-root",
        default="/scratch/$USER/active-matter-ssl/runs/search_hierarchical",
        help="Root directory that will contain one subdirectory per sampled trial",
    )
    parser.add_argument("--train-script", default="scripts/train_ssl_hierarchical.slurm")
    parser.add_argument("--eval-script", default="scripts/evaluate.slurm")
    parser.add_argument("--chain-eval", action="store_true", default=True,
                        help="Submit a dependent eval job after each training trial")
    parser.add_argument("--no-chain-eval", dest="chain_eval", action="store_false")

    parser.add_argument("--within-mask-ratios", default="0.75,0.85")
    parser.add_argument("--weight-decays", default="0.05,0.10")
    parser.add_argument("--predictor-dims", default="96,128")
    parser.add_argument("--predictor-depths", default="2,4")
    parser.add_argument("--lambda-within-values", default="0.5,1.0,1.5")
    parser.add_argument("--lambda-cross-values", default="0.25,0.5,1.0")
    parser.add_argument("--lambda-future-values", default="0.5,1.0,1.5")
    parser.add_argument("--lambda-sigreg-values", default="0.1,0.5,1.0")
    parser.add_argument("--lambda-action-values", default="0.05,0.1,0.2")
    return parser.parse_args()


def build_population(args: argparse.Namespace) -> list[SweepConfig]:
    mask_values = parse_csv_floats(args.within_mask_ratios)
    wd_values = parse_csv_floats(args.weight_decays)
    pred_dims = parse_csv_ints(args.predictor_dims)
    pred_depths = parse_csv_ints(args.predictor_depths)
    lambda_within_values = parse_csv_floats(args.lambda_within_values)
    lambda_cross_values = parse_csv_floats(args.lambda_cross_values)
    lambda_future_values = parse_csv_floats(args.lambda_future_values)
    lambda_sigreg_values = parse_csv_floats(args.lambda_sigreg_values)
    lambda_action_values = parse_csv_floats(args.lambda_action_values)

    population = []
    for combo in itertools.product(
        mask_values,
        wd_values,
        pred_dims,
        pred_depths,
        lambda_within_values,
        lambda_cross_values,
        lambda_future_values,
        lambda_sigreg_values,
        lambda_action_values,
    ):
        (
            within_mask_ratio,
            weight_decay,
            predictor_dim,
            predictor_depth,
            lambda_within,
            lambda_cross,
            lambda_future,
            lambda_sigreg,
            lambda_action,
        ) = combo

        name = (
            f"m{str(within_mask_ratio).replace('.', '')}"
            f"_wd{str(weight_decay).replace('.', '')}"
            f"_pd{predictor_dim}"
            f"_dp{predictor_depth}"
            f"_lw{str(lambda_within).replace('.', '')}"
            f"_lc{str(lambda_cross).replace('.', '')}"
            f"_lf{str(lambda_future).replace('.', '')}"
            f"_ls{str(lambda_sigreg).replace('.', '')}"
            f"_la{str(lambda_action).replace('.', '')}"
        )
        population.append(
            SweepConfig(
                name=name,
                within_mask_ratio=within_mask_ratio,
                weight_decay=weight_decay,
                predictor_dim=predictor_dim,
                predictor_depth=predictor_depth,
                lambda_within=lambda_within,
                lambda_cross=lambda_cross,
                lambda_future=lambda_future,
                lambda_sigreg=lambda_sigreg,
                lambda_action=lambda_action,
            )
        )

    return population


def sample_configs(args: argparse.Namespace) -> list[SweepConfig]:
    population = build_population(args)
    if args.search_mode == "grid":
        if args.max_configs > 0:
            return population[: min(args.max_configs, len(population))]
        return population

    rng = random.Random(args.seed)
    if args.num_trials >= len(population):
        return population
    return rng.sample(population, args.num_trials)


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
        "LAMBDA_WITHIN": str(cfg.lambda_within),
        "LAMBDA_CROSS": str(cfg.lambda_cross),
        "LAMBDA_FUTURE": str(cfg.lambda_future),
        "LAMBDA_SIGREG": str(cfg.lambda_sigreg),
        "LAMBDA_ACTION": str(cfg.lambda_action),
        "LAMBDA_LATENT_KL": str(cfg.lambda_latent_kl),
        "LAMBDA_VORTICITY": str(cfg.lambda_vorticity),
        "LAMBDA_SPECTRAL": str(cfg.lambda_spectral),
    }
    parts = []
    for key, value in exports.items():
        if value is None:
            parts.append(key)
        else:
            parts.append(f"{key}={value}")
    return ",".join(parts)


def build_train_command(args: argparse.Namespace, cfg: SweepConfig) -> list[str]:
    cmd = ["sbatch", f"--account={args.account}"]
    if args.partition:
        cmd.append(f"--partition={args.partition}")
    if args.gres:
        cmd.append(f"--gres={args.gres}")
    cmd.append(f"--export={build_export_string(args, cfg)}")
    cmd.append(args.train_script)
    return cmd


def build_eval_command(args: argparse.Namespace, cfg: SweepConfig, dependency_job_id: str | None) -> list[str]:
    run_name = f"{args.run_prefix}-{cfg.name}"
    checkpoint = os.path.join(args.runs_root, run_name, "best.pt")
    exports = ",".join(
        [
            "ALL",
            f"CHECKPOINT={checkpoint}",
            f"RUN_NAME=eval_{run_name}",
            f"PROBE_EPOCHS={args.probe_epochs}",
            f"PROBE_BATCH_SIZE={args.probe_batch_size}",
            f"KNN_K={args.knn_k}",
        ]
    )
    cmd = ["sbatch", f"--account={args.account}"]
    if args.partition:
        cmd.append(f"--partition={args.partition}")
    if args.gres:
        cmd.append(f"--gres={args.gres}")
    if dependency_job_id:
        cmd.append(f"--dependency=afterok:{dependency_job_id}")
    cmd.append(f"--export={exports}")
    cmd.append(args.eval_script)
    return cmd


def parse_job_id(output: str) -> str | None:
    match = re.search(r"Submitted batch job (\d+)", output)
    return match.group(1) if match else None


def write_manifest(runs_root: str, configs: list[SweepConfig]) -> None:
    root = Path(os.path.expandvars(runs_root))
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "search_manifest.tsv"
    headers = [
        "name",
        "within_mask_ratio",
        "weight_decay",
        "predictor_dim",
        "predictor_depth",
        "lambda_within",
        "lambda_cross",
        "lambda_future",
        "lambda_sigreg",
        "lambda_action",
        "lambda_latent_kl",
        "lambda_vorticity",
        "lambda_spectral",
    ]
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(headers) + "\n")
        for cfg in configs:
            row = asdict(cfg)
            handle.write("\t".join(str(row[h]) for h in headers) + "\n")


def main() -> None:
    args = parse_args()
    args.runs_root = os.path.expandvars(args.runs_root)
    configs = sample_configs(args)
    write_manifest(args.runs_root, configs)

    for cfg in configs:
        train_cmd = build_train_command(args, cfg)
        print("TRAIN ", " ".join(shlex.quote(part) for part in train_cmd))

        train_job_id = None
        if args.submit:
            result = subprocess.run(train_cmd, check=True, capture_output=True, text=True)
            print(result.stdout.strip())
            train_job_id = parse_job_id(result.stdout)

        if args.chain_eval:
            eval_cmd = build_eval_command(args, cfg, train_job_id)
            print("EVAL  ", " ".join(shlex.quote(part) for part in eval_cmd))
            if args.submit:
                eval_result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
                print(eval_result.stdout.strip())


if __name__ == "__main__":
    main()
