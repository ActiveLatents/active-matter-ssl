# src/model/masking.py

"""
Masking strategies for Channel-Factored JEPA.

Two complementary masking types:
  1. Within-field:  For each field group, randomly mask a fraction of its
                    tokens. The predictor reconstructs them from unmasked
                    tokens of the SAME group.
  2. Cross-field:   Mask ALL tokens of one entire field group. The predictor
                    reconstructs them from ALL tokens of the other groups.

Both produce (visible_indices, masked_indices) pairs that the predictor
uses to know which tokens to predict.
"""

import random

import torch


def within_field_mask(field_indices, mask_ratio=0.75, device="cpu"):
    """
    For each field group, randomly mask `mask_ratio` fraction of its tokens.

    Args:
        field_indices: dict mapping group name to (start_idx, end_idx)
        mask_ratio:    fraction of tokens to mask per group
        device:        torch device

    Returns:
        visible_ids: (N_visible,) -- global token indices that are visible
        masked_ids:  (N_masked,)  -- global token indices that are masked
    """
    all_visible = []
    all_masked = []

    for name, (start, end) in field_indices.items():
        n_tokens = end - start
        n_mask = int(n_tokens * mask_ratio)
        n_keep = n_tokens - n_mask

        # Random permutation within this group's token range
        perm = torch.randperm(n_tokens, device=device)
        group_indices = torch.arange(start, end, device=device)

        keep_ids = group_indices[perm[:n_keep]]
        mask_ids = group_indices[perm[n_keep:]]

        all_visible.append(keep_ids)
        all_masked.append(mask_ids)

    visible_ids = torch.cat(all_visible)
    masked_ids = torch.cat(all_masked)

    # Sort for deterministic ordering
    visible_ids = visible_ids.sort().values
    masked_ids = masked_ids.sort().values

    return visible_ids, masked_ids


def cross_field_mask(field_indices, target_group, device="cpu"):
    """
    Mask ALL tokens of `target_group`. All other groups are fully visible.

    Args:
        field_indices: dict mapping group name to (start_idx, end_idx)
        target_group:  name of the group to mask entirely
        device:        torch device

    Returns:
        visible_ids: (N_visible,) -- token indices from non-target groups
        masked_ids:  (N_masked,)  -- token indices of the target group
    """
    visible_list = []
    masked_list = []

    for name, (start, end) in field_indices.items():
        ids = torch.arange(start, end, device=device)
        if name == target_group:
            masked_list.append(ids)
        else:
            visible_list.append(ids)

    visible_ids = torch.cat(visible_list).sort().values
    masked_ids = torch.cat(masked_list).sort().values

    return visible_ids, masked_ids


def generate_masks(field_indices, within_mask_ratio=0.75, device="cpu"):
    """
    Generate both within-field and cross-field masks for one training step.

    For cross-field masking, one group is randomly selected as the target.

    Args:
        field_indices:      dict mapping group name to (start_idx, end_idx)
        within_mask_ratio:  fraction to mask for within-field
        device:             torch device

    Returns:
        within_masks: dict with keys "visible_ids" and "masked_ids"
        cross_masks:  dict with keys "visible_ids", "masked_ids", "target_group"
    """
    # Within-field: mask a fraction from each group
    w_visible, w_masked = within_field_mask(
        field_indices, mask_ratio=within_mask_ratio, device=device,
    )

    # Cross-field: randomly pick one group to mask entirely
    # Use torch RNG so this is reproducible with torch.manual_seed
    group_names = list(field_indices.keys())
    target_group = group_names[torch.randint(len(group_names), (1,), device=device).item()]
    c_visible, c_masked = cross_field_mask(
        field_indices, target_group=target_group, device=device,
    )

    within_masks = {"visible_ids": w_visible, "masked_ids": w_masked}
    cross_masks = {
        "visible_ids": c_visible,
        "masked_ids": c_masked,
        "target_group": target_group,
    }

    return within_masks, cross_masks


def batch_mask_indices(mask_ids, batch_size):
    """
    Expand single-sample mask indices to a batch.

    Since all samples in a batch use the same mask pattern (standard
    practice in MAE/JEPA), this just repeats the indices.

    Args:
        mask_ids: (N,) token indices
        batch_size: int

    Returns:
        (B, N) batched indices
    """
    return mask_ids.unsqueeze(0).expand(batch_size, -1).contiguous()


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate field indices from ChannelFactoredPatchEmbed
    # Each group has 2048 tokens (8 temporal x 16 x 16 spatial)
    field_indices = {
        "concentration": (0, 2048),
        "velocity":      (2048, 4096),
        "orientation":   (4096, 6144),
        "strain_rate":   (6144, 8192),
    }

    within_masks, cross_masks = generate_masks(
        field_indices, within_mask_ratio=0.75,
    )

    total = 8192
    n_within_vis = within_masks["visible_ids"].shape[0]
    n_within_mask = within_masks["masked_ids"].shape[0]
    n_cross_vis = cross_masks["visible_ids"].shape[0]
    n_cross_mask = cross_masks["masked_ids"].shape[0]

    print(f"Total tokens: {total}")
    print(f"Within-field: {n_within_vis} visible + {n_within_mask} masked "
          f"= {n_within_vis + n_within_mask}")
    print(f"Cross-field:  {n_cross_vis} visible + {n_cross_mask} masked "
          f"= {n_cross_vis + n_cross_mask} (target: {cross_masks['target_group']})")

    # Verify no overlap and full coverage
    w_all = torch.cat([within_masks["visible_ids"], within_masks["masked_ids"]])
    assert w_all.unique().shape[0] == total, "Within-field mask error"

    c_all = torch.cat([cross_masks["visible_ids"], cross_masks["masked_ids"]])
    assert c_all.unique().shape[0] == total, "Cross-field mask error"

    print("All checks passed.")