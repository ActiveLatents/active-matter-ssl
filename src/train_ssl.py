"""
CF-JEPA SSL training script.

Features:
  - Mixed precision (bf16) on A100
  - Wandb experiment tracking
  - Checkpoint/restart for spot instance preemption
  - Cosine LR schedule with warmup
  - Periodic validation loss logging

Usage:
  python -m src.train_ssl --data_dir /scratch/$NETID/dl_project_data/data
"""

import argparse
import os
import signal
import sys
import time

import torch
import torch.nn as nn
import wandb
from torch.amp import GradScaler, autocast

from src.dataset import build_dataloader
from src.ssl_model import CFJEPA


def get_lr(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + __import__("math").cos(__import__("math").pi * progress))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def get_ema_momentum(step, total_steps, base_ema=0.996, max_ema=1.0):
    """Cosine schedule for EMA momentum (typically 0.996 -> 1.0)."""
    progress = step / max(total_steps, 1)
    return max_ema - (max_ema - base_ema) * (1 + __import__("math").cos(__import__("math").pi * progress)) / 2


def save_checkpoint(path, model, optimizer, scaler, epoch, step, best_val_loss, config):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"  Resumed from checkpoint: epoch={ckpt['epoch']}, step={ckpt['step']}")
    return ckpt["epoch"], ckpt["step"], ckpt["best_val_loss"]


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_within = 0.0
    total_cross = 0.0
    n_batches = 0

    for batch in val_loader:
        field_dict = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        with autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = model.forward_ssl(field_dict)

        total_loss += loss.item()
        total_within += loss_dict["loss_within"].item()
        total_cross += loss_dict["loss_cross"].item()
        n_batches += 1

    model.train()

    return {
        "val/loss": total_loss / n_batches,
        "val/loss_within": total_within / n_batches,
        "val/loss_cross": total_cross / n_batches,
    }


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = {
        # Data
        "data_dir": args.data_dir,
        "n_frames": args.n_frames,
        "train_stride": args.train_stride,
        "val_stride": args.val_stride,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        # Model
        "embed_dim": args.embed_dim,
        "encoder_depth": args.encoder_depth,
        "encoder_heads": args.encoder_heads,
        "predictor_dim": args.predictor_dim,
        "predictor_depth": args.predictor_depth,
        "predictor_heads": args.predictor_heads,
        "within_mask_ratio": args.within_mask_ratio,
        # Loss weights
        "lambda_within": args.lambda_within,
        "lambda_cross": args.lambda_cross,
        "lambda_sigreg": args.lambda_sigreg,
        # Ablation
        "use_channel_factored": not args.single_patch_embed,
        # Training
        "epochs": args.epochs,
        "base_lr": args.base_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip": args.grad_clip,
        "base_ema": args.base_ema,
        "max_ema": args.max_ema,
    }

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=config,
            resume="allow",
        )

    print("Loading data...")
    train_loader = build_dataloader(
        data_dir=args.data_dir,
        split="train",
        mode="ssl",
        n_frames=args.n_frames,
        stride=args.train_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    val_loader = build_dataloader(
        data_dir=args.data_dir,
        split="valid",
        mode="ssl",
        n_frames=args.n_frames,
        stride=args.val_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Initializing model...")
    model = CFJEPA(
        embed_dim=args.embed_dim,
        tube_t=2,
        patch_h=16,
        patch_w=16,
        n_frames=args.n_frames,
        spatial_size=256,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        within_mask_ratio=args.within_mask_ratio,
        lambda_within=args.lambda_within,
        lambda_cross=args.lambda_cross,
        lambda_sigreg=args.lambda_sigreg,
        use_channel_factored=not args.single_patch_embed,
    ).to(device)

    model.param_count()

    # Only optimize trainable parameters — target_encoder is frozen (EMA).
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    scaler = GradScaler("cuda", enabled=False)  # bf16 doesn't need loss scaling

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        print("Found checkpoint, resuming...")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            ckpt_path, model, optimizer, scaler, device,
        )
        start_epoch += 1  # start from next epoch

    current_epoch = start_epoch

    def _save_and_exit(signum, frame):
        print("\nSIGUSR1 received — saving checkpoint and exiting for requeue.")
        save_checkpoint(ckpt_path, model, optimizer, scaler,
                        current_epoch, global_step, best_val_loss, config)
        if use_wandb:
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, _save_and_exit)

    print(f"\nStarting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, args.epochs):
        current_epoch = epoch
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            lr = get_lr(global_step, total_steps, warmup_steps, args.base_lr)
            set_lr(optimizer, lr)

            field_dict = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            optimizer.zero_grad()
            with autocast("cuda", dtype=torch.bfloat16):
                loss, loss_dict = model.forward_ssl(field_dict)

            loss.backward()

            if args.grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

            # Step
            optimizer.step()

            ema_momentum = get_ema_momentum(global_step, total_steps, args.base_ema, args.max_ema)
            model.update_target_encoder(ema_momentum)

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/loss_within": loss_dict["loss_within"].item(),
                    "train/loss_cross": loss_dict["loss_cross"].item(),
                    "train/sigreg_total": loss_dict["sigreg_total"].item(),
                    "train/lr": lr,
                    "train/ema_momentum": ema_momentum,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/epoch": epoch,
                    "train/step": global_step,
                }
                if use_wandb:
                    wandb.log(log_dict, step=global_step)

                print(
                    f"  [{epoch+1}/{args.epochs}] "
                    f"step {global_step} | "
                    f"loss {loss.item():.4f} | "
                    f"within {loss_dict['loss_within'].item():.4f} | "
                    f"cross {loss_dict['loss_cross'].item():.4f} | "
                    f"sigreg {loss_dict['sigreg_total'].item():.2f} | "
                    f"lr {lr:.2e} | "
                    f"grad {grad_norm:.2f}"
                )

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{args.epochs} done in {epoch_time:.0f}s | avg loss: {avg_loss:.4f}")

        if (epoch + 1) % args.val_every == 0:
            print("  Running validation...")
            val_metrics = validate(model, val_loader, device)
            if use_wandb:
                wandb.log(val_metrics, step=global_step)
            print(f"  val/loss: {val_metrics['val/loss']:.4f}")

            if val_metrics["val/loss"] < best_val_loss:
                best_val_loss = val_metrics["val/loss"]
                best_path = os.path.join(args.checkpoint_dir, "best.pt")
                save_checkpoint(
                    best_path, model, optimizer, scaler,
                    epoch, global_step, best_val_loss, config,
                )
                print(f"  New best val loss: {best_val_loss:.4f}")

        save_checkpoint(
            ckpt_path, model, optimizer, scaler,
            epoch, global_step, best_val_loss, config,
        )

    encoder_path = os.path.join(args.checkpoint_dir, "encoder_final.pt")
    torch.save({
        "patch_embed_state_dict": model.patch_embed.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "config": config,
    }, encoder_path)
    print(f"\nEncoder weights saved: {encoder_path}")

    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CF-JEPA SSL Training")

    # Data
    parser.add_argument("--data_dir", type=str, default="/scratch/sk12590/dl_project_data/data")
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--train_stride", type=int, default=5)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--encoder_heads", type=int, default=6)
    parser.add_argument("--predictor_dim", type=int, default=192)
    parser.add_argument("--predictor_depth", type=int, default=4)
    parser.add_argument("--predictor_heads", type=int, default=6)
    parser.add_argument("--within_mask_ratio", type=float, default=0.75)
    parser.add_argument("--single_patch_embed", action="store_true",
                        help="Ablation: use one shared 3D conv over all 11 channels "
                             "instead of per-field-group embeddings")

    # Loss weights
    parser.add_argument("--lambda_within", type=float, default=1.0)
    parser.add_argument("--lambda_cross", type=float, default=1.0)
    parser.add_argument("--lambda_sigreg", type=float, default=0.01)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--base_ema", type=float, default=0.996,
                        help="Starting EMA momentum (cosine-scheduled up to max_ema)")
    parser.add_argument("--max_ema", type=float, default=1.0,
                        help="Final EMA momentum")

    # Logging
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--wandb_project", type=str, default="cfjepa-active-matter")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="/scratch/sk12590/active-matter-ssl/runs/ssl")

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train(args)