"""
Masking strategies for Channel-Factored JEPA.

Two complementary masking types:
  1. Within-field:  For each field group, mask a contiguous spatiotemporal
                    block of its tokens (V-JEPA style). The predictor
                    reconstructs them from unmasked tokens of the SAME group.
  2. Cross-field:   Mask ALL tokens of one entire field group. The predictor
                    reconstructs them from ALL tokens of the other groups.

Both produce (visible_indices, masked_indices) pairs that the predictor
uses to know which tokens to predict.

Block masking (vs. random token masking) forces the model to predict across
space and time rather than interpolating from nearby unmasked neighbors.
"""

import torch


def within_field_block_mask(field_indices, grid_shape, mask_ratio=0.75, device="cpu"):
    """
    For each field group, mask a single contiguous spatiotemporal block.

    Block dimensions are sampled so the block covers approximately
    `mask_ratio` of the group's tokens. The temporal and spatial axes
    are scaled independently with slight randomness so the model sees
    varied block shapes across training steps.

    Token layout (per group): flat index = t * n_h * n_w + h * n_w + w,
    matching the (n_t, n_h, n_w) raster order from TubeletEmbedding.

    Args:
        field_indices: dict mapping group name to (start_idx, end_idx)
        grid_shape:    (n_t, n_h, n_w) tubelet grid dimensions
        mask_ratio:    target fraction of tokens to mask per group
        device:        torch device

    Returns:
        visible_ids: (N_visible,) -- global token indices that are visible
        masked_ids:  (N_masked,)  -- global token indices that are masked
    """
    n_t, n_h, n_w = grid_shape
    all_visible = []
    all_masked = []

    for name, (start, end) in field_indices.items():
        t_ratio = torch.empty(1, device=device).uniform_(
            max(0.5, mask_ratio - 0.15), min(1.0, mask_ratio + 0.15)
        ).item()

        # Spatial coverage chosen so t_ratio * h_ratio * w_ratio ≈ mask_ratio
        spatial_ratio = (mask_ratio / max(t_ratio, 1e-6)) ** 0.5
        spatial_ratio = min(spatial_ratio, 1.0)

        bt = max(1, min(n_t, round(n_t * t_ratio)))
        bh = max(1, min(n_h, round(n_h * spatial_ratio)))
        bw = max(1, min(n_w, round(n_w * spatial_ratio)))

        t0 = torch.randint(0, max(1, n_t - bt + 1), (1,), device=device).item()
        h0 = torch.randint(0, max(1, n_h - bh + 1), (1,), device=device).item()
        w0 = torch.randint(0, max(1, n_w - bw + 1), (1,), device=device).item()

        t_in = (torch.arange(n_t, device=device) >= t0) & \
               (torch.arange(n_t, device=device) < t0 + bt)
        h_in = (torch.arange(n_h, device=device) >= h0) & \
               (torch.arange(n_h, device=device) < h0 + bh)
        w_in = (torch.arange(n_w, device=device) >= w0) & \
               (torch.arange(n_w, device=device) < w0 + bw)

        block_mask = (t_in[:, None, None] & h_in[None, :, None] & w_in[None, None, :]).reshape(-1)

        group_indices = torch.arange(start, end, device=device)
        all_visible.append(group_indices[~block_mask])
        all_masked.append(group_indices[block_mask])

    visible_ids = torch.cat(all_visible).sort().values
    masked_ids = torch.cat(all_masked).sort().values

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


def generate_masks(field_indices, grid_shape, within_mask_ratio=0.75, device="cpu"):
    """
    Generate both within-field and cross-field masks for one training step.

    For cross-field masking, one group is randomly selected as the target.

    Args:
        field_indices:      dict mapping group name to (start_idx, end_idx)
        grid_shape:         (n_t, n_h, n_w) tubelet grid dimensions
        within_mask_ratio:  target fraction to mask for within-field block
        device:             torch device

    Returns:
        within_masks: dict with keys "visible_ids" and "masked_ids"
        cross_masks:  dict with keys "visible_ids", "masked_ids", "target_group"
    """
    # Within-field
    w_visible, w_masked = within_field_block_mask(
        field_indices, grid_shape, mask_ratio=within_mask_ratio, device=device,
    )

    # Cross-field
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


if __name__ == "__main__":
    # n_t=8, n_h=16, n_w=16 = 2048 tokens per group
    field_indices = {
        "concentration": (0, 2048),
        "velocity":      (2048, 4096),
        "orientation":   (4096, 6144),
        "strain_rate":   (6144, 8192),
    }
    grid_shape = (8, 16, 16)

    within_masks, cross_masks = generate_masks(
        field_indices, grid_shape, within_mask_ratio=0.75,
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

    w_all = torch.cat([within_masks["visible_ids"], within_masks["masked_ids"]])
    assert w_all.unique().shape[0] == total, "Within-field mask error"

    c_all = torch.cat([cross_masks["visible_ids"], cross_masks["masked_ids"]])
    assert c_all.unique().shape[0] == total, "Cross-field mask error"

    print("All checks passed.")