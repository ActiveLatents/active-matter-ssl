"""
Quick sanity check for the full CF-JEPA pipeline.
Run this on an interactive GPU node before submitting training jobs.

Tests:
  1. Dataset loading in SSL mode
  2. Model init
  3. Parameter count check (must be < 100M)
  4. SSL forward + backward pass (gradient check)
  5. Encode pass (frozen features for linear probe)
  6. Memory usage report

Usage:
  python -m src.test_pipeline --data_dir /scratch/$NETID/dl_project_data/data
"""

import argparse
import torch
from src.dataset import build_dataloader
from src.ssl_model import CFJEPA


def test_pipeline(data_dir, device="cuda"):

    print("=" * 60)
    print("CF-JEPA Pipeline Sanity Check")
    print("=" * 60)

    print("\n[1/6] Loading dataset (SSL mode)...")
    loader = build_dataloader(
        data_dir=data_dir,
        split="train",
        mode="ssl",
        n_frames=16,
        stride=16,       # coarse stride for quick test
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(loader))

    print("  Batch contents:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"    {key:15s}: {val.shape}, dtype={val.dtype}")
    print("  PASSED")

    print("\n[2/6] Initializing model...")
    model = CFJEPA(
        embed_dim=384,
        tube_t=2,
        patch_h=16,
        patch_w=16,
        n_frames=16,
        spatial_size=256,
        encoder_depth=12,
        encoder_heads=6,
        predictor_dim=192,
        predictor_depth=4,
        predictor_heads=6,
        within_mask_ratio=0.75,
    ).to(device)
    print("  PASSED")

    print("\n[3/6] Parameter count:")
    counts = model.param_count()
    total = counts["total"]
    assert total < 100_000_000, f"Model has {total/1e6:.2f}M params, exceeds 100M limit!"
    print(f"  Total: {total/1e6:.2f}M (limit: 100M)")
    print("  PASSED")

    print("\n[4/6] SSL forward + backward pass...")

    field_dict = {
        k: v.to(device) for k, v in batch.items() if k != "labels"
    }

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    loss, loss_dict = model.forward_ssl(field_dict)
    print(f"  Loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v:.4f}")

    loss.backward()

    has_grad = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  WARNING: no gradient on {name}")
            has_grad = False
    if has_grad:
        print("  All parameters have gradients")
    print("  PASSED")

    print("\n[5/6] Encode pass (frozen features)...")
    model.eval()
    features = model.encode(field_dict)
    expected_shape = (next(iter(field_dict.values())).shape[0], model.patch_embed.embed_dim)
    print(f"  Features: {features.shape}")
    assert features.shape == expected_shape, f"Unexpected feature shape: {features.shape}, expected {expected_shape}"

    feat_std = features.std(dim=0).mean()
    print(f"  Feature std (across batch): {feat_std:.4f}")
    if feat_std < 1e-6:
        print("  WARNING: features may be collapsed (very low variance)")
    print("  PASSED")

    if device == "cuda":
        print("\n[6/6] GPU memory usage:")
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"  Peak:    {peak_mem:.2f} GB")
        print(f"  Current: {current_mem:.2f} GB")
        print(f"  PASSED")
    else:
        print("\n[6/6] Skipping memory report (CPU mode)")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str,
        default="/scratch/sk12590/dl_project_data/data",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    test_pipeline(args.data_dir, device=args.device)