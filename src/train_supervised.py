"""
Train a simple from-scratch ResNet supervised baseline on normalized [zeta, alpha].

This is intended as an end-to-end regression baseline to compare against
representation-learning methods evaluated with frozen linear probe / kNN.
"""

import argparse
import os
import signal
import sys
import time

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from src.dataset import build_dataloader
from src.models import SupervisedBaseline


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised ResNet baseline")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expandvars("/scratch/$USER/active-matter/data"),
    )
    parser.add_argument("--output_dir", type=str, default="runs/supervised")
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="dl_project_sst")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def compute_label_stats(loader):
    labels = []
    for _, batch_labels in loader:
        labels.append(batch_labels)
    labels = torch.cat(labels, dim=0)
    mean = labels.mean(dim=0)
    std = torch.clamp(labels.std(dim=0), min=1e-8)
    return mean, std


def normalize_labels(labels, mean, std):
    return (labels - mean) / std


def compute_mse(preds, targets):
    mse_per_target = ((preds - targets) ** 2).mean(dim=0)
    mse_total = mse_per_target.mean()
    return mse_total.item(), mse_per_target[0].item(), mse_per_target[1].item()


def run_epoch(model, loader, optimizer, criterion, device, label_mean, label_std, train):
    model.train(train)
    phase = "train" if train else "val"
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=phase, leave=False)
    with torch.set_grad_enabled(train):
        for frames, labels in pbar:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels_norm = normalize_labels(labels, label_mean, label_std)

            preds = model(frames)
            loss = criterion(preds, labels_norm)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if optimizer is not None and hasattr(torch.nn.utils, "clip_grad_norm_") and criterion is not None:
                    if grad_clip := getattr(run_epoch, "_grad_clip", 0.0):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * frames.size(0)
            total_samples += frames.size(0)
            all_preds.append(preds.detach().cpu())
            all_targets.append(labels_norm.detach().cpu())
            pbar.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    mse_total, mse_zeta, mse_alpha = compute_mse(preds, targets)
    return {
        f"{phase}/loss": total_loss / max(total_samples, 1),
        f"{phase}/mse": mse_total,
        f"{phase}/mse_zeta": mse_zeta,
        f"{phase}/mse_alpha": mse_alpha,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_mse, args, label_mean, label_std):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mse": best_val_mse,
            "args": vars(args),
            "label_mean": label_mean.cpu(),
            "label_std": label_std.cpu(),
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")

    train_loader = build_dataloader(
        data_dir=args.data_dir,
        split="train",
        mode="supervised",
        n_frames=args.n_frames,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_dataloader(
        data_dir=args.data_dir,
        split="valid",
        mode="supervised",
        n_frames=args.n_frames,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        prefetch_factor=args.prefetch_factor,
    )

    label_mean, label_std = compute_label_stats(train_loader)
    label_mean = label_mean.to(device)
    label_std = label_std.to(device)
    print(f"Train label mean: {label_mean.tolist()}")
    print(f"Train label std:  {label_std.tolist()}")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                **vars(args),
                "label_mean": label_mean.tolist(),
                "label_std": label_std.tolist(),
            },
        )

    model = SupervisedBaseline(embed_dim=args.embed_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.2f}M")
    if use_wandb:
        wandb.config.update({"total_params": total_params})

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    run_epoch._grad_clip = args.grad_clip

    start_epoch = 1
    best_val_mse = float("inf")
    current_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mse = ckpt.get("best_val_mse", float("inf"))
        current_epoch = ckpt["epoch"]
        print(f"Resumed from epoch {ckpt['epoch']} (best val MSE {best_val_mse:.6f})")

    def _save_and_exit(signum, frame):
        print("\nSIGUSR1 received, saving checkpoint and exiting for requeue.")
        save_checkpoint(
            os.path.join(args.output_dir, "last.pt"),
            model,
            optimizer,
            scheduler,
            current_epoch,
            best_val_mse,
            args,
            label_mean,
            label_std,
        )
        if use_wandb:
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, _save_and_exit)

    log_path = os.path.join(args.output_dir, "log.csv")
    write_mode = "a" if args.resume and os.path.isfile(log_path) else "w"
    with open(log_path, write_mode) as f:
        if write_mode == "w":
            f.write("epoch,train_loss,train_mse,val_loss,val_mse,val_mse_zeta,val_mse_alpha,lr,elapsed_s\n")

    for epoch in range(start_epoch, args.epochs + 1):
        current_epoch = epoch
        start_time = time.time()

        train_metrics = run_epoch(
            model, train_loader, optimizer, criterion, device, label_mean, label_std, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, criterion, device, label_mean, label_std, train=False
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train MSE {train_metrics['train/mse']:.6f} | "
            f"val MSE {val_metrics['val/mse']:.6f} "
            f"(zeta: {val_metrics['val/mse_zeta']:.6f}, alpha: {val_metrics['val/mse_alpha']:.6f}) | "
            f"lr {lr:.2e} | {elapsed:.1f}s"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_metrics['train/loss']:.6f},{train_metrics['train/mse']:.6f},"
                f"{val_metrics['val/loss']:.6f},{val_metrics['val/mse']:.6f},"
                f"{val_metrics['val/mse_zeta']:.6f},{val_metrics['val/mse_alpha']:.6f},"
                f"{lr:.2e},{elapsed:.1f}\n"
            )

        if use_wandb:
            wandb.log(
                {
                    **train_metrics,
                    **val_metrics,
                    "lr": lr,
                    "epoch": epoch,
                },
                step=epoch,
            )

        save_checkpoint(
            os.path.join(args.output_dir, "last.pt"),
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_mse,
            args,
            label_mean,
            label_std,
        )

        if val_metrics["val/mse"] < best_val_mse:
            best_val_mse = val_metrics["val/mse"]
            save_checkpoint(
                os.path.join(args.output_dir, "best.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_mse,
                args,
                label_mean,
                label_std,
            )
            print(f"  -> saved best (val MSE {best_val_mse:.6f})")
            if use_wandb:
                wandb.run.summary["best_val_mse"] = best_val_mse
                wandb.run.summary["best_epoch"] = epoch

    if use_wandb:
        wandb.finish()

    print(f"\nDone. Best val MSE: {best_val_mse:.6f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
