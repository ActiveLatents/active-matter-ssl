# src/dataset.py

import glob
import os
import re

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


# Channel order for stacking
CHANNEL_NAMES = [
    "concentration",
    "velocity_x", "velocity_y",
    "D_xx", "D_xy", "D_yx", "D_yy",
    "E_xx", "E_xy", "E_yx", "E_yy",
]


def load_channel_stats(stats_path):
    """Load per-channel mean and std from stats.yaml."""
    with open(stats_path, "r") as f:
        stats = yaml.safe_load(f)

    mean = np.array([
        stats["mean"]["concentration"],
        stats["mean"]["velocity"][0],
        stats["mean"]["velocity"][1],
        stats["mean"]["D"][0][0],
        stats["mean"]["D"][0][1],
        stats["mean"]["D"][1][0],
        stats["mean"]["D"][1][1],
        stats["mean"]["E"][0][0],
        stats["mean"]["E"][0][1],
        stats["mean"]["E"][1][0],
        stats["mean"]["E"][1][1],
    ], dtype=np.float32).reshape(11, 1, 1)

    std = np.array([
        stats["std"]["concentration"],
        stats["std"]["velocity"][0],
        stats["std"]["velocity"][1],
        stats["std"]["D"][0][0],
        stats["std"]["D"][0][1],
        stats["std"]["D"][1][0],
        stats["std"]["D"][1][1],
        stats["std"]["E"][0][0],
        stats["std"]["E"][0][1],
        stats["std"]["E"][1][0],
        stats["std"]["E"][1][1],
    ], dtype=np.float32).reshape(11, 1, 1)

    return mean, std


def parse_params_from_filename(filepath):
    """Extract zeta and alpha from the filename."""
    basename = os.path.basename(filepath)
    match = re.search(r"zeta_([\d.]+)_alpha_([-\d.]+)\.hdf5", basename)
    zeta = float(match.group(1))
    alpha = float(match.group(2))
    return zeta, alpha


class ActiveMatterDataset(Dataset):
    """
    Dataset for active_matter simulations.

    Each sample is a 16-frame window extracted from an 81-timestep trajectory.
    Windows are extracted with a configurable stride.
    """

    def __init__(
        self,
        data_dir,
        split="train",
        n_frames=32,
        stride=1,
        normalize=True,
        crop_size=224,
        stats_path=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.n_frames = n_frames
        self.stride = stride
        self.normalize = normalize
        self.crop_size = crop_size

        # Load normalization stats
        if self.normalize:
            if stats_path is None:
                # Default: look for stats.yaml one level above data_dir
                stats_path = os.path.join(os.path.dirname(data_dir), "stats.yaml")
            self.channel_mean, self.channel_std = load_channel_stats(stats_path)

        split_dir = os.path.join(data_dir, split)
        self.file_paths = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))

        # Build an index: list of (file_path, traj_idx, start_frame, zeta, alpha)
        self.samples = []
        for fp in self.file_paths:
            zeta, alpha = parse_params_from_filename(fp)
            with h5py.File(fp, "r") as f:
                n_traj = f["t0_fields"]["concentration"].shape[0]
                n_steps = f["t0_fields"]["concentration"].shape[1]

            for traj_idx in range(n_traj):
                max_start = n_steps - n_frames
                for start in range(0, max_start + 1, stride):
                    self.samples.append((fp, traj_idx, start, zeta, alpha))

        print(f"[{split}] {len(self.samples)} samples from {len(self.file_paths)} files")

    def _center_crop(self, x):
        """Center crop from 256x256 to crop_size x crop_size."""
        _, _, h, w = x.shape
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        return x[:, :, top : top + self.crop_size, left : left + self.crop_size]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, traj_idx, start, zeta, alpha = self.samples[idx]
        end = start + self.n_frames

        with h5py.File(fp, "r") as f:
            conc = f["t0_fields"]["concentration"][traj_idx, start:end]
            vel = f["t1_fields"]["velocity"][traj_idx, start:end]
            D = f["t2_fields"]["D"][traj_idx, start:end]
            E = f["t2_fields"]["E"][traj_idx, start:end]

        # Stack into 11 channels: (16, 11, 256, 256)
        frames = np.stack([
            conc,
            vel[..., 0],
            vel[..., 1],
            D[..., 0, 0],
            D[..., 0, 1],
            D[..., 1, 0],
            D[..., 1, 1],
            E[..., 0, 0],
            E[..., 0, 1],
            E[..., 1, 0],
            E[..., 1, 1],
        ], axis=1).astype(np.float32)

        if self.normalize:
            frames = (frames - self.channel_mean) / (self.channel_std + 1e-8)

        frames = self._center_crop(frames)
        frames = torch.from_numpy(frames)
        labels = torch.tensor([zeta, alpha], dtype=torch.float32)

        return frames, labels


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_dir = "/scratch/sk12590/dl_project_data/data"

    for split in ["train", "valid", "test"]:
        ds = ActiveMatterDataset(data_dir, split=split, stride=1)

    ds = ActiveMatterDataset(data_dir, split="train", stride=1)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch_frames, batch_labels = next(iter(loader))
    print(f"Batch frames: {batch_frames.shape}")
    print(f"Batch labels: {batch_labels.shape}")
    print(f"Frames min={batch_frames.min():.3f}, max={batch_frames.max():.3f}")