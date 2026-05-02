#!/usr/bin/env python3

"""
Submit fast evaluation jobs for all finished tuning checkpoints.

This uses the standard frozen-feature evaluation, but with reduced probe
epochs by default so tuning stays cheap.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit fast eval jobs for tuning checkpoints")
    parser.add_argument("--submit", action="store_true", help="Actually submit jobs with sbatch")
    parser.add_argument("--account", default="torch_pr_494_general")
    parser.add_argument("--partition", default="", help="Optional Slurm partition override")
    parser.add_argument("--gres", default="", help="Optional GPU request override, e.g. gpu:l40s:1")
    parser.add_argument(
        "--runs-root",
        default="/scratch/$USER/active-matter-ssl/runs/tuning_hierarchical_stage1",
    )
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--eval-script", default="scripts/evaluate.slurm")
    return parser.parse_args()


def build_command(args: argparse.Namespace, checkpoint: Path) -> list[str]:
    run_name = f"eval_{checkpoint.parent.name}"
    exports = ",".join(
        [
            "ALL",
            f"CHECKPOINT={checkpoint}",
            f"RUN_NAME={run_name}",
            f"PROBE_EPOCHS={args.probe_epochs}",
            f"KNN_K={args.knn_k}",
            f"PROBE_BATCH_SIZE={args.probe_batch_size}",
        ]
    )
    cmd = ["sbatch", f"--account={args.account}"]
    if args.partition:
        cmd.append(f"--partition={args.partition}")
    if args.gres:
        cmd.append(f"--gres={args.gres}")
    cmd.append(f"--export={exports}")
    cmd.append(args.eval_script)
    return cmd


def main() -> None:
    args = parse_args()
    runs_root = Path(os.path.expandvars(args.runs_root))

    checkpoints = sorted(runs_root.glob("*/best.pt"))
    if not checkpoints:
        raise SystemExit(f"No best.pt checkpoints found under {runs_root}")

    for checkpoint in checkpoints:
        cmd = build_command(args, checkpoint)
        rendered = " ".join(shlex.quote(part) for part in cmd)
        print(rendered)
        if args.submit:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
