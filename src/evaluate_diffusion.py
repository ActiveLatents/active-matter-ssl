"""
Frozen-feature evaluation for the future latent diffusion baseline.

Pipeline:
  1. Load trained diffusion checkpoint
  2. Extract frozen encoder features for train/valid/test
  3. Z-score labels from the training split
  4. Train a linear probe on frozen features
  5. Evaluate kNN regression on the same features
"""

import argparse
import os
import time

import torch
import wandb

from src.dataset import build_dataloader
from src.diffusion_model import FutureLatentDiffusion
from src.evaluate import (
    compute_label_stats,
    compute_mse,
    extract_features,
    knn_regression,
    normalize_labels,
    print_results,
    train_linear_probe,
    _unwrap_compiled_state_dict,
)


def evaluate(args):
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

    model = FutureLatentDiffusion(
        embed_dim=config["embed_dim"],
        tube_t=2,
        patch_h=16,
        patch_w=16,
        n_frames=config["n_frames"],
        spatial_size=256,
        encoder_depth=config["encoder_depth"],
        encoder_heads=config["encoder_heads"],
        denoiser_depth=config["denoiser_depth"],
        denoiser_heads=config["denoiser_heads"],
        num_diffusion_steps=config["num_diffusion_steps"],
    ).to(device)

    model.load_state_dict(_unwrap_compiled_state_dict(ckpt["model_state_dict"]))
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

    linear_model, _ = train_linear_probe(
        train_features,
        train_labels_norm,
        val_features,
        val_labels_norm,
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
        wandb.log(
            {
                "eval/linear_val_mse": val_mse,
                "eval/linear_val_mse_zeta": val_mse_z,
                "eval/linear_val_mse_alpha": val_mse_a,
                "eval/linear_test_mse": test_mse,
                "eval/linear_test_mse_zeta": test_mse_z,
                "eval/linear_test_mse_alpha": test_mse_a,
            }
        )

    print("\n" + "=" * 60)
    print("kNN REGRESSION")
    print("=" * 60)

    best_k = args.knn_k
    best_knn_val_mse = float("inf")
    k_values = [5, 10, 20, 50] if args.knn_k == 20 else [args.knn_k]

    for k in k_values:
        val_knn_pred = knn_regression(train_features, train_labels_norm, val_features, k=k)
        mse, mse_z, mse_a = compute_mse(val_knn_pred, val_labels_norm)
        print(f"  k={k:3d} | val MSE: {mse:.6f} (zeta: {mse_z:.6f}, alpha: {mse_a:.6f})")
        if use_wandb:
            wandb.log(
                {
                    f"knn/val_mse_k{k}": mse,
                    f"knn/val_mse_zeta_k{k}": mse_z,
                    f"knn/val_mse_alpha_k{k}": mse_a,
                }
            )
        if mse < best_knn_val_mse:
            best_knn_val_mse = mse
            best_k = k

    print(f"\n  Best k: {best_k}")

    test_knn_pred = knn_regression(train_features, train_labels_norm, test_features, k=best_k)
    test_knn_mse, test_knn_mse_z, test_knn_mse_a = compute_mse(test_knn_pred, test_labels_norm)
    val_knn_pred = knn_regression(train_features, train_labels_norm, val_features, k=best_k)
    val_knn_mse, val_knn_mse_z, val_knn_mse_a = compute_mse(val_knn_pred, val_labels_norm)

    print_results("kNN", "Valid", val_knn_mse, val_knn_mse_z, val_knn_mse_a)
    print_results("kNN", "Test", test_knn_mse, test_knn_mse_z, test_knn_mse_a)

    if use_wandb:
        wandb.log(
            {
                "eval/knn_best_k": best_k,
                "eval/knn_val_mse": val_knn_mse,
                "eval/knn_val_mse_zeta": val_knn_mse_z,
                "eval/knn_val_mse_alpha": val_knn_mse_a,
                "eval/knn_test_mse": test_knn_mse,
                "eval/knn_test_mse_zeta": test_knn_mse_z,
                "eval/knn_test_mse_alpha": test_knn_mse_a,
            }
        )

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
            "val_mse": val_mse,
            "val_mse_zeta": val_mse_z,
            "val_mse_alpha": val_mse_a,
            "test_mse": test_mse,
            "test_mse_zeta": test_mse_z,
            "test_mse_alpha": test_mse_a,
            "lr": args.probe_lr,
            "epochs": args.probe_epochs,
        },
        "knn": {
            "best_k": best_k,
            "val_mse": val_knn_mse,
            "val_mse_zeta": val_knn_mse_z,
            "val_mse_alpha": val_knn_mse_a,
            "test_mse": test_knn_mse,
            "test_mse_zeta": test_knn_mse_z,
            "test_mse_alpha": test_knn_mse_a,
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
    parser = argparse.ArgumentParser(description="Diffusion Evaluation: Linear Probe + kNN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained diffusion checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default="/scratch/zl6113/active-matter/data")
    parser.add_argument("--eval_stride", type=int, default=2, help="Stride for sliding windows during evaluation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--knn_k", type=int, default=20, help="Number of neighbors (if 20, also tries 5,10,50)")
    parser.add_argument("--use_target_encoder", action="store_true", help="Use EMA target encoder for feature extraction instead of online encoder")
    parser.add_argument("--wandb_org", type=str, default=None, help="Optional W&B entity/org name")
    parser.add_argument("--wandb_project", type=str, default="dl_project_sst")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()
    evaluate(args)
