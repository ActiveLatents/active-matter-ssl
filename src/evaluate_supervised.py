"""
Evaluate the supervised baseline on the same valid/test splits used by JEPA.
"""

import argparse
import os

import torch
import wandb

from src.dataset import build_dataloader
from src.models import SupervisedBaseline


def normalize_labels(labels, mean, std):
    return (labels - mean) / std


def compute_mse(preds, targets):
    mse_per_target = ((preds - targets) ** 2).mean(dim=0)
    mse_total = mse_per_target.mean()
    return mse_total.item(), mse_per_target[0].item(), mse_per_target[1].item()


@torch.no_grad()
def evaluate_split(model, loader, device, label_mean, label_std):
    model.eval()
    all_preds = []
    all_targets = []
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_norm = normalize_labels(labels, label_mean, label_std)
        preds = model(frames)
        all_preds.append(preds.cpu())
        all_targets.append(labels_norm.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return compute_mse(preds, targets)


def main():
    parser = argparse.ArgumentParser(description="Evaluate supervised baseline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/scratch/zl6113/active-matter/data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--wandb_project", type=str, default="dl_project_sst")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    label_mean = ckpt["label_mean"].to(device)
    label_std = ckpt["label_std"].to(device)

    model = SupervisedBaseline().to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    loader_kwargs = dict(
        data_dir=args.data_dir,
        mode="supervised",
        n_frames=train_args["n_frames"],
        stride=train_args["stride"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=train_args.get("crop_size"),
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_dataloader(split="valid", **loader_kwargs)
    test_loader = build_dataloader(split="test", **loader_kwargs)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "checkpoint": args.checkpoint,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
            resume="allow",
        )

    val_mse, val_mse_z, val_mse_a = evaluate_split(
        model, val_loader, device, label_mean, label_std
    )
    test_mse, test_mse_z, test_mse_a = evaluate_split(
        model, test_loader, device, label_mean, label_std
    )

    print(f"[Supervised] Valid MSE: {val_mse:.6f} (zeta: {val_mse_z:.6f}, alpha: {val_mse_a:.6f})")
    print(f"[Supervised] Test  MSE: {test_mse:.6f} (zeta: {test_mse_z:.6f}, alpha: {test_mse_a:.6f})")

    results = {
        "checkpoint": args.checkpoint,
        "label_mean": label_mean.cpu().tolist(),
        "label_std": label_std.cpu().tolist(),
        "val_mse": val_mse,
        "val_mse_zeta": val_mse_z,
        "val_mse_alpha": val_mse_a,
        "test_mse": test_mse,
        "test_mse_zeta": test_mse_z,
        "test_mse_alpha": test_mse_a,
    }
    save_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.pt")
    torch.save(results, save_path)
    print(f"Results saved to: {save_path}")

    if use_wandb:
        wandb.log(
            {
                "eval/val_mse": val_mse,
                "eval/val_mse_zeta": val_mse_z,
                "eval/val_mse_alpha": val_mse_a,
                "eval/test_mse": test_mse,
                "eval/test_mse_zeta": test_mse_z,
                "eval/test_mse_alpha": test_mse_a,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
