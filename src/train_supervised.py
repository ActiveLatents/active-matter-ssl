"""
Training script for SupervisedBaseline on the ActiveMatter dataset.

Usage:
    python -m src.train_supervised \
        --data_dir /scratch/sk12590/dl_project_data/data \
        --epochs 50 \
        --batch_size 16 \
        --lr 1e-3 \
        --output_dir runs/supervised
"""

import argparse
import os
import signal
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm

from src.dataset import ActiveMatterDataset
from src.supervised_baseline import SupervisedBaseline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="/scratch/sk12590/dl_project_data/data")
    p.add_argument("--output_dir", default="runs/supervised")
    p.add_argument("--n_frames",   type=int,   default=32)
    p.add_argument("--stride",     type=int,   default=1,
                   help="Window stride when building the dataset index")
    p.add_argument("--embed_dim",  type=int,   default=256)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,  default=4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--wandb_project", default="active-matter-ssl")
    p.add_argument("--wandb_run",     default=None,
                   help="W&B run name (auto-generated if omitted)")
    p.add_argument("--no_wandb",  action="store_true",
                   help="Disable W&B logging")
    p.add_argument("--resume", default=None,
                   help="Path to last.pt checkpoint to resume from")
    return p.parse_args()


def make_loader(data_dir, split, n_frames, stride, batch_size, num_workers, shuffle):
    ds = ActiveMatterDataset(
        data_dir=data_dir,
        split=split,
        n_frames=n_frames,
        stride=stride,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool, epoch, n_epochs):
    model.train(train)
    total_loss = 0.0
    n_samples = 0
    phase = "train" if train else "val"

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs} [{phase}]", leave=False)
    with torch.set_grad_enabled(train):
        for frames, labels in pbar:
            frames = frames.to(device, non_blocking=True)   # (B, T, 11, 224, 224)
            labels = labels.to(device, non_blocking=True)   # (B, 2)

            preds = model(frames)
            loss = criterion(preds, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * frames.size(0)
            n_samples  += frames.size(0)
            pbar.set_postfix(loss=f"{total_loss / n_samples:.4f}")

    return total_loss / n_samples


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = make_loader(
        args.data_dir, "train", args.n_frames, args.stride,
        args.batch_size, args.num_workers, shuffle=True,
    )
    val_loader = make_loader(
        args.data_dir, "valid", args.n_frames, args.stride,
        args.batch_size, args.num_workers, shuffle=False,
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SupervisedBaseline(embed_dim=args.embed_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.2f}M")
    if use_wandb:
        wandb.config.update({"total_params": total_params})

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']} (best val {best_val_loss:.4f})")

    # ── Signal handler: save checkpoint and exit on SIGUSR1 (pre-emption) ────
    def _save_and_exit(signum, frame):
        print("\nSIGUSR1 received — saving checkpoint and exiting for requeue.")
        _save_checkpoint("last.pt", epoch=start_epoch - 1)
        if use_wandb:
            wandb.finish()
        sys.exit(0)

    def _save_checkpoint(filename, epoch):
        torch.save({
            "epoch":         epoch,
            "state_dict":    model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args":          vars(args),
        }, os.path.join(args.output_dir, filename))

    signal.signal(signal.SIGUSR1, _save_and_exit)

    # ── Training loop ─────────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, "log.csv")
    write_mode = "a" if args.resume and os.path.isfile(log_path) else "w"
    with open(log_path, write_mode) as f:
        if write_mode == "w":
            f.write("epoch,train_loss,val_loss,lr,elapsed_s\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True,  epoch=epoch, n_epochs=args.epochs)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False, epoch=epoch, n_epochs=args.epochs)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.1f}s"
        )

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{current_lr:.2e},{elapsed:.1f}\n")

        if use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "val/loss":   val_loss,
                "lr":         current_lr,
            }, step=epoch, commit=True)

        # Always save last checkpoint (for resume)
        _save_checkpoint("last.pt", epoch=epoch)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint("best.pt", epoch=epoch)
            print(f"  -> saved best (val {val_loss:.4f})")
            if use_wandb:
                wandb.run.summary["best_val_loss"] = val_loss
                wandb.run.summary["best_epoch"]    = epoch

    if use_wandb:
        wandb.finish()

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
