import argparse
import os
import signal
import sys
import time

import torch
import wandb
from torch.amp import autocast

from src.dataset import build_dataloader
from src.diffusion_model import FutureLatentDiffusion


def get_lr(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + __import__("math").cos(__import__("math").pi * progress))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_ema_momentum(step, total_steps, base_ema=0.996, max_ema=1.0):
    progress = step / max(total_steps, 1)
    return max_ema - (max_ema - base_ema) * (1 + __import__("math").cos(__import__("math").pi * progress)) / 2


def save_checkpoint(path, model, optimizer, epoch, step, best_val_loss, config):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["step"], ckpt["best_val_loss"]


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total = 0.0
    n_batches = 0
    for batch in val_loader:
        field_dict = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model.forward_diffusion(field_dict)
        total += loss.item()
        n_batches += 1
    model.train()
    return {"val/loss": total / max(n_batches, 1)}


def train(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = vars(args).copy()

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=config,
            resume="allow",
        )

    train_loader = build_dataloader(
        data_dir=args.data_dir,
        split="train",
        mode="ssl",
        n_frames=args.n_frames,
        stride=args.train_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_dataloader(
        data_dir=args.data_dir,
        split="valid",
        mode="ssl",
        n_frames=args.n_frames,
        stride=args.val_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model = FutureLatentDiffusion(
        embed_dim=args.embed_dim,
        tube_t=2,
        patch_h=16,
        patch_w=16,
        n_frames=args.n_frames,
        spatial_size=256,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        denoiser_depth=args.denoiser_depth,
        denoiser_heads=args.denoiser_heads,
        num_diffusion_steps=args.num_diffusion_steps,
    ).to(device)

    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    ckpt_path = os.path.join(args.checkpoint_dir, "latest_diffusion.pt")
    if os.path.exists(ckpt_path):
        start_epoch, global_step, best_val_loss = load_checkpoint(
            ckpt_path, model, optimizer, device
        )
        start_epoch += 1

    current_epoch = start_epoch

    def _save_and_exit(signum, frame):
        save_checkpoint(
            ckpt_path, model, optimizer, current_epoch, global_step, best_val_loss, config
        )
        if use_wandb:
            wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGUSR1, _save_and_exit)

    for epoch in range(start_epoch, args.epochs):
        current_epoch = epoch
        epoch_loss = 0.0
        epoch_start = time.time()
        for batch in train_loader:
            lr = get_lr(global_step, total_steps, warmup_steps, args.base_lr)
            set_lr(optimizer, lr)
            field_dict = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16):
                loss, loss_dict = model.forward_diffusion(field_dict)
            loss.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            optimizer.step()

            ema_momentum = get_ema_momentum(global_step, total_steps, args.base_ema, args.max_ema)
            model.update_target_encoder(ema_momentum)

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/loss_noise": loss_dict["loss_noise"].item(),
                    "train/lr": lr,
                    "train/ema_momentum": ema_momentum,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/epoch": epoch,
                    "train/step": global_step,
                }
                if use_wandb:
                    wandb.log(log_dict, step=global_step)

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} done in {time.time() - epoch_start:.0f}s | avg loss: {avg_loss:.4f}")

        if (epoch + 1) % args.val_every == 0:
            val_metrics = validate(model, val_loader, device)
            if use_wandb:
                wandb.log(val_metrics, step=global_step)
            if val_metrics["val/loss"] < best_val_loss:
                best_val_loss = val_metrics["val/loss"]
                save_checkpoint(
                    os.path.join(args.checkpoint_dir, "best_diffusion.pt"),
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    best_val_loss,
                    config,
                )

        save_checkpoint(
            ckpt_path, model, optimizer, epoch, global_step, best_val_loss, config
        )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Future latent diffusion training")
    parser.add_argument("--data_dir", type=str, default="/scratch/zl6113/active-matter/data")
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--train_stride", type=int, default=2)
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--encoder_heads", type=int, default=6)
    parser.add_argument("--denoiser_depth", type=int, default=6)
    parser.add_argument("--denoiser_heads", type=int, default=6)
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--base_lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--base_ema", type=float, default=0.996)
    parser.add_argument("--max_ema", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--wandb_project", type=str, default="dl_project_sst")
    parser.add_argument("--run_name", type=str, default="latent-diffusion-future")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="/scratch/zl6113/active-matter-ssl/runs/diffusion")
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)
