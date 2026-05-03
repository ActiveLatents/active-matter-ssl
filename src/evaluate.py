"""
Phase 2: Linear probe and kNN regression on frozen CF-JEPA representations.

Pipeline:
  1. Load trained CF-JEPA checkpoint
  2. Extract frozen features for train/valid/test splits
  3. Compute z-score normalization stats from training labels
  4. Train a single linear layer (linear probe) to predict (zeta, alpha)
  5. Evaluate kNN regression on the same features
  6. Report MSE on validation and test sets for both methods

Usage:
  python -m src.evaluate \
      --checkpoint /scratch/$NETID/active-matter-ssl/runs/ssl/best.pt \
      --data_dir /scratch/$NETID/dl_project_data/data
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.amp import autocast

from src.dataset import build_dataloader
from src.ssl_model import CFJEPA


# Feature extraction

@torch.no_grad()
def extract_features(model, loader, device, use_target_encoder=False):
    """
    Run frozen encoder on all samples, collect features and labels.

    Returns:
        features: (N, embed_dim) float32 tensor
        labels:   (N, 2) float32 tensor  [zeta, alpha]
    """
    model.eval()
    all_features = []
    all_labels = []

    for batch in loader:
        field_dict = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]

        with autocast("cuda", dtype=torch.bfloat16):
            features = model.encode(field_dict, use_target_encoder=use_target_encoder)  # (B, embed_dim)

        all_features.append(features.float().cpu())
        all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# Label normalization

def compute_label_stats(labels):
    """Compute per-target mean and std from training labels."""
    mean = labels.mean(dim=0)
    std = labels.std(dim=0)
    std = torch.clamp(std, min=1e-8)  # avoid division by zero
    return mean, std


def normalize_labels(labels, mean, std):
    return (labels - mean) / std


def denormalize_labels(labels, mean, std):
    return labels * std + mean


# Linear probe

def train_linear_probe(
    train_features, train_labels,
    val_features, val_labels,
    embed_dim, lr=1e-3, epochs=100,
    batch_size=256, device="cpu",
    use_wandb=False,
):
    """
    Train a single linear layer to predict z-score normalized labels.

    Returns:
        best model state dict, best val MSE (on normalized targets)
    """
    linear = nn.Linear(embed_dim, 2).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Move to device
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    n_train = train_features.shape[0]
    best_val_mse = float("inf")
    best_state = None

    for epoch in range(epochs):
        linear.train()

        # Shuffle
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            feat = train_features[idx]
            lbl = train_labels[idx]

            pred = linear(feat)
            loss = criterion(pred, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        linear.eval()
        with torch.no_grad():
            val_pred = linear(val_features)
            val_mse = criterion(val_pred, val_labels).item()

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in linear.state_dict().items()}

        if use_wandb:
            wandb.log({
                "probe/train_loss": epoch_loss / n_batches,
                "probe/val_mse": val_mse,
                "probe/epoch": epoch + 1,
            }, step=epoch + 1)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [Linear Probe] Epoch {epoch+1}/{epochs} | "
                  f"train loss: {epoch_loss / n_batches:.6f} | "
                  f"val MSE: {val_mse:.6f}")

    linear.load_state_dict(best_state)
    return linear, best_val_mse


# kNN regression

def knn_regression(train_features, train_labels, query_features, k=20):
    """
    k-Nearest Neighbor regression using L2 distance.

    For each query, find k nearest neighbors in train set and average
    their labels. Performed on CPU to avoid large distance matrix on GPU.

    Args:
        train_features: (N_train, D)
        train_labels:   (N_train, 2)
        query_features: (N_query, D)
        k:              number of neighbors

    Returns:
        predictions: (N_query, 2)
    """
    train_f = torch.nn.functional.normalize(train_features, dim=-1)
    query_f = torch.nn.functional.normalize(query_features, dim=-1)

    chunk_size = 512
    all_preds = []

    for i in range(0, query_f.shape[0], chunk_size):
        chunk = query_f[i : i + chunk_size]

        sim = chunk @ train_f.T  # (chunk, N_train)

        _, topk_idx = sim.topk(k, dim=-1)  # (chunk, k)

        neighbor_labels = train_labels[topk_idx]  # (chunk, k, 2)
        preds = neighbor_labels.mean(dim=1)  # (chunk, 2)
        all_preds.append(preds)

    return torch.cat(all_preds, dim=0)


# Evaluation helpers

def compute_mse(pred, target):
    """Per-target and overall MSE."""
    mse_per_target = ((pred - target) ** 2).mean(dim=0)
    mse_total = mse_per_target.mean()
    return mse_total.item(), mse_per_target[0].item(), mse_per_target[1].item()


def print_results(method, split, mse_total, mse_zeta, mse_alpha):
    print(f"  [{method}] {split} MSE: {mse_total:.6f} "
          f"(zeta: {mse_zeta:.6f}, alpha: {mse_alpha:.6f})")


def evaluate(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            entity=args.wandb_org,
            project=args.wandb_project,
            name=args.run_name,
            config={
                "checkpoint": args.checkpoint,
                "eval_stride": args.eval_stride,
                "batch_size": args.batch_size,
                "probe_lr": args.probe_lr,
                "probe_epochs": args.probe_epochs,
                "probe_batch_size": args.probe_batch_size,
                "knn_k": args.knn_k,
                "use_target_encoder": args.use_target_encoder,
            },
            resume="allow",
        )

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = CFJEPA(
        embed_dim=config["embed_dim"],
        tube_t=2,
        patch_h=16,
        patch_w=16,
        n_frames=config["n_frames"],
        spatial_size=256,
        encoder_depth=config["encoder_depth"],
        encoder_heads=config["encoder_heads"],
        predictor_dim=config["predictor_dim"],
        predictor_depth=config["predictor_depth"],
        predictor_heads=config["predictor_heads"],
        within_mask_ratio=config["within_mask_ratio"],
        use_channel_factored=config.get("use_channel_factored", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("  Model loaded successfully")

    embed_dim = config["embed_dim"]

    print("\nLoading data...")
    loader_kwargs = dict(
        data_dir=args.data_dir,
        mode="ssl",
        n_frames=config["n_frames"],
        stride=args.eval_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_loader = build_dataloader(split="train", **loader_kwargs)
    val_loader = build_dataloader(split="valid", **loader_kwargs)
    test_loader = build_dataloader(split="test", **loader_kwargs)

    print("\nExtracting features...")

    encoder_name = "target_encoder" if args.use_target_encoder else "encoder"
    print(f"  Using: {encoder_name}")

    t0 = time.time()
    train_features, train_labels = extract_features(model, train_loader, device, args.use_target_encoder)
    print(f"  Train: {train_features.shape} in {time.time() - t0:.0f}s")

    t0 = time.time()
    val_features, val_labels = extract_features(model, val_loader, device, args.use_target_encoder)
    print(f"  Valid: {val_features.shape} in {time.time() - t0:.0f}s")

    t0 = time.time()
    test_features, test_labels = extract_features(model, test_loader, device, args.use_target_encoder)
    print(f"  Test:  {test_features.shape} in {time.time() - t0:.0f}s")

    label_mean, label_std = compute_label_stats(train_labels)
    print(f"\nLabel stats (from train): mean={label_mean.tolist()}, std={label_std.tolist()}")

    train_labels_norm = normalize_labels(train_labels, label_mean, label_std)
    val_labels_norm = normalize_labels(val_labels, label_mean, label_std)
    test_labels_norm = normalize_labels(test_labels, label_mean, label_std)

    print("\n" + "=" * 60)
    print("LINEAR PROBE")
    print("=" * 60)

    linear_model, best_val_mse = train_linear_probe(
        train_features, train_labels_norm,
        val_features, val_labels_norm,
        embed_dim=embed_dim,
        lr=args.probe_lr,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        device=device,
        use_wandb=use_wandb,
    )

    linear_model.eval()
    with torch.no_grad():
        val_pred = linear_model(val_features.to(device)).cpu()
        test_pred = linear_model(test_features.to(device)).cpu()

    val_mse, val_mse_z, val_mse_a = compute_mse(val_pred, val_labels_norm)
    test_mse, test_mse_z, test_mse_a = compute_mse(test_pred, test_labels_norm)

    print_results("Linear Probe", "Valid", val_mse, val_mse_z, val_mse_a)
    print_results("Linear Probe", "Test", test_mse, test_mse_z, test_mse_a)

    if use_wandb:
        wandb.log({
            "eval/linear_val_mse": val_mse,
            "eval/linear_val_mse_zeta": val_mse_z,
            "eval/linear_val_mse_alpha": val_mse_a,
            "eval/linear_test_mse": test_mse,
            "eval/linear_test_mse_zeta": test_mse_z,
            "eval/linear_test_mse_alpha": test_mse_a,
        })

    print("\n" + "=" * 60)
    print("kNN REGRESSION")
    print("=" * 60)

    best_k = args.knn_k
    best_knn_val_mse = float("inf")

    # Try multiple k values and pick the best on validation
    k_values = [5, 10, 20, 50] if args.knn_k == 20 else [args.knn_k]

    for k in k_values:
        val_knn_pred = knn_regression(
            train_features, train_labels_norm,
            val_features, k=k,
        )
        mse, mse_z, mse_a = compute_mse(val_knn_pred, val_labels_norm)
        print(f"  k={k:3d} | val MSE: {mse:.6f} (zeta: {mse_z:.6f}, alpha: {mse_a:.6f})")

        if use_wandb:
            wandb.log({
                f"knn/val_mse_k{k}": mse,
                f"knn/val_mse_zeta_k{k}": mse_z,
                f"knn/val_mse_alpha_k{k}": mse_a,
            })

        if mse < best_knn_val_mse:
            best_knn_val_mse = mse
            best_k = k

    print(f"\n  Best k: {best_k}")

    # Final test evaluation with best k
    test_knn_pred = knn_regression(
        train_features, train_labels_norm,
        test_features, k=best_k,
    )
    test_knn_mse, test_knn_mse_z, test_knn_mse_a = compute_mse(
        test_knn_pred, test_labels_norm,
    )
    val_knn_pred = knn_regression(
        train_features, train_labels_norm,
        val_features, k=best_k,
    )
    val_knn_mse, val_knn_mse_z, val_knn_mse_a = compute_mse(
        val_knn_pred, val_labels_norm,
    )

    print_results("kNN", "Valid", val_knn_mse, val_knn_mse_z, val_knn_mse_a)
    print_results("kNN", "Test", test_knn_mse, test_knn_mse_z, test_knn_mse_a)

    if use_wandb:
        wandb.log({
            "eval/knn_best_k": best_k,
            "eval/knn_val_mse": val_knn_mse,
            "eval/knn_val_mse_zeta": val_knn_mse_z,
            "eval/knn_val_mse_alpha": val_knn_mse_a,
            "eval/knn_test_mse": test_knn_mse,
            "eval/knn_test_mse_zeta": test_knn_mse_z,
            "eval/knn_test_mse_alpha": test_knn_mse_a,
        })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Method':<16s} {'Split':<8s} {'MSE':>10s} {'MSE(zeta)':>12s} {'MSE(alpha)':>12s}")
    print(f"  {'-'*58}")
    print(f"  {'Linear Probe':<16s} {'Valid':<8s} {val_mse:>10.6f} {val_mse_z:>12.6f} {val_mse_a:>12.6f}")
    print(f"  {'Linear Probe':<16s} {'Test':<8s} {test_mse:>10.6f} {test_mse_z:>12.6f} {test_mse_a:>12.6f}")
    print(f"  {'kNN (k=' + str(best_k) + ')':<16s} {'Valid':<8s} {val_knn_mse:>10.6f} {val_knn_mse_z:>12.6f} {val_knn_mse_a:>12.6f}")
    print(f"  {'kNN (k=' + str(best_k) + ')':<16s} {'Test':<8s} {test_knn_mse:>10.6f} {test_knn_mse_z:>12.6f} {test_knn_mse_a:>12.6f}")

    results = {
        "checkpoint": args.checkpoint,
        "embed_dim": embed_dim,
        "eval_stride": args.eval_stride,
        "label_mean": label_mean.tolist(),
        "label_std": label_std.tolist(),
        "linear_probe": {
            "val_mse": val_mse, "val_mse_zeta": val_mse_z, "val_mse_alpha": val_mse_a,
            "test_mse": test_mse, "test_mse_zeta": test_mse_z, "test_mse_alpha": test_mse_a,
            "lr": args.probe_lr, "epochs": args.probe_epochs,
        },
        "knn": {
            "best_k": best_k,
            "val_mse": val_knn_mse, "val_mse_zeta": val_knn_mse_z, "val_mse_alpha": val_knn_mse_a,
            "test_mse": test_knn_mse, "test_mse_zeta": test_knn_mse_z, "test_mse_alpha": test_knn_mse_a,
        },
        "train_features_shape": list(train_features.shape),
        "linear_probe_state_dict": linear_model.state_dict(),
    }

    save_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.pt")
    torch.save(results, save_path)
    print(f"\nResults saved to: {save_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CF-JEPA Evaluation: Linear Probe + kNN")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained CF-JEPA checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str,
                        default="/scratch/sk12590/dl_project_data/data")

    # Feature extraction
    parser.add_argument("--eval_stride", type=int, default=16,
                        help="Stride for sliding windows during evaluation "
                             "(16 = non-overlapping, reduces redundancy)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # Linear probe
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--probe_batch_size", type=int, default=256)

    # kNN
    parser.add_argument("--knn_k", type=int, default=20,
                        help="Number of neighbors (if 20, also tries 5,10,50)")

    # Encoder selection
    parser.add_argument("--use_target_encoder", action="store_true",
                        help="Use EMA target encoder for feature extraction instead of online encoder")

    # Logging
    parser.add_argument("--wandb_org", type=str, default=None, help="W&B entity/org name")
    parser.add_argument("--wandb_project", type=str, default="cfjepa-active-matter")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    evaluate(args)