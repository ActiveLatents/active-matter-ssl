import glob
import os
import random
import re
import weakref
from collections import OrderedDict

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader


# ── Channel layout ──────────────────────────────────────────────────────────
CHANNEL_NAMES = [
    "concentration",                          # 1 ch  (scalar)
    "velocity_x", "velocity_y",               # 2 ch  (vector)
    "D_xx", "D_xy", "D_yx", "D_yy",          # 4 ch  (orientation tensor)
    "E_xx", "E_xy", "E_yx", "E_yy",          # 4 ch  (strain-rate tensor)
]

# Field group definitions: (name, channel_start, channel_end)
FIELD_GROUPS = {
    "concentration": (0, 1),    # 1 channel
    "velocity":      (1, 3),    # 2 channels
    "orientation":   (3, 7),    # 4 channels
    "strain_rate":   (7, 11),   # 4 channels
}


# ── Normalization helpers ───────────────────────────────────────────────────

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
    Flexible dataset for active_matter simulations.

    Supports two modes controlled by `mode`:
      - "supervised" : returns (frames, labels)
                       frames shape: (T, 11, H, W)
      - "ssl"        : returns a dict with channel-factored tensors + labels
                       keys: concentration, velocity, orientation, strain_rate,
                             labels

    Temporal windowing:
      - Each trajectory has 81 timesteps
      - `n_frames` controls the window length (e.g., 16)
      - `stride` controls spacing between sliding windows
      - All valid windows are extracted deterministically

    Spatial handling:
      - Raw spatial resolution is 256x256
      - `crop_size=None` keeps 256x256 (recommended for training from scratch)
      - `crop_size=224` does a center crop (for compatibility with baselines)
      - `random_crop` applies random spatial crop instead of center crop
    """

    def __init__(
        self,
        data_dir,
        split="train",
        n_frames=16,
        stride=1,
        normalize=True,
        crop_size=None,
        random_crop=False,
        stats_path=None,
        mode="supervised",
        max_open_files=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.n_frames = n_frames
        self.stride = stride
        self.normalize = normalize
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.mode = mode
        self._open_files = None
        self._max_open_files = max_open_files

        assert mode in ("supervised", "ssl"), f"Unknown mode: {mode}"

        # Load normalization stats
        if self.normalize:
            if stats_path is None:
                stats_path = os.path.join(os.path.dirname(data_dir), "stats.yaml")
            self.channel_mean, self.channel_std = load_channel_stats(stats_path)

        # Discover files
        split_dir = os.path.join(data_dir, split)
        self.file_paths = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))

        # Build index: sliding windows over every trajectory
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

        print(
            f"[{split}] {len(self.samples)} samples from {len(self.file_paths)} files "
            f"(mode={mode}, n_frames={n_frames}, stride={stride})"
        )

    def _ensure_file_cache(self):
        if self._open_files is None:
            self._open_files = OrderedDict()
            weakref.finalize(self, self._close_all_files)

    def _close_all_files(self):
        if self._open_files:
            for f in self._open_files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._open_files.clear()

    def _get_file_handle(self, path):
        self._ensure_file_cache()
        if path in self._open_files:
            f = self._open_files.pop(path)
            self._open_files[path] = f
            return f

        while len(self._open_files) >= self._max_open_files:
            _, old_f = self._open_files.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass

        f = h5py.File(path, "r", libver="latest", swmr=True)
        self._open_files[path] = f
        return f

    def _apply_crop(self, x):
        """
        Apply spatial crop if crop_size is set.
        x shape: (T, C, H, W)
        """
        if self.crop_size is None:
            return x

        _, _, h, w = x.shape
        if self.crop_size >= h:
            return x

        if self.random_crop and self.split == "train":
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
        else:
            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2

        return x[:, :, top : top + self.crop_size, left : left + self.crop_size]

    def _split_channels(self, frames):
        """
        Split the 11-channel tensor into field groups.
        frames shape: (T, 11, H, W)
        Returns dict of tensors, each (T, C_group, H, W).
        """
        result = {}
        for name, (start, end) in FIELD_GROUPS.items():
            result[name] = frames[:, start:end]
        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, traj_idx, start, zeta, alpha = self.samples[idx]
        end = start + self.n_frames

        f = self._get_file_handle(fp)
        conc = f["t0_fields"]["concentration"][traj_idx, start:end]
        vel = f["t1_fields"]["velocity"][traj_idx, start:end]
        D = f["t2_fields"]["D"][traj_idx, start:end]
        E = f["t2_fields"]["E"][traj_idx, start:end]

        # Stack into (T, 11, 256, 256)
        frames = np.stack([
            conc,
            vel[..., 0], vel[..., 1],
            D[..., 0, 0], D[..., 0, 1], D[..., 1, 0], D[..., 1, 1],
            E[..., 0, 0], E[..., 0, 1], E[..., 1, 0], E[..., 1, 1],
        ], axis=1).astype(np.float32)

        # Normalize
        if self.normalize:
            frames = (frames - self.channel_mean) / (self.channel_std + 1e-8)

        # Spatial crop
        frames = self._apply_crop(frames)

        # Convert to tensor
        frames = torch.from_numpy(frames)
        labels = torch.tensor([zeta, alpha], dtype=torch.float32)

        # Return based on mode
        if self.mode == "supervised":
            return frames, labels

        elif self.mode == "ssl":
            field_dict = self._split_channels(frames)
            field_dict["labels"] = labels
            return field_dict

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open_files"] = None
        return state


def build_dataloader(
    data_dir,
    split="train",
    mode="supervised",
    n_frames=16,
    stride=1,
    batch_size=4,
    num_workers=4,
    crop_size=None,
    random_crop=False,
    stats_path=None,
    prefetch_factor=2,
):
    """Build a DataLoader with sensible defaults for each mode and split."""

    dataset = ActiveMatterDataset(
        data_dir=data_dir,
        split=split,
        n_frames=n_frames,
        stride=stride,
        normalize=True,
        crop_size=crop_size,
        random_crop=random_crop,
        stats_path=stats_path,
        mode=mode,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**loader_kwargs)

    return loader


# ── Quick sanity check ──────────────────────────────────────────────────────

if __name__ == "__main__":

    data_dir = "/scratch/sk12590/dl_project_data/data"

    print("=" * 60)
    print("SUPERVISED MODE")
    print("=" * 60)
    loader = build_dataloader(
        data_dir, split="train", mode="supervised",
        n_frames=16, stride=16, batch_size=4, num_workers=0,
    )
    frames, labels = next(iter(loader))
    print(f"  frames : {frames.shape}")        # (B, 16, 11, 256, 256)
    print(f"  labels : {labels.shape}")        # (B, 2)
    print(f"  range  : [{frames.min():.3f}, {frames.max():.3f}]")

    print()
    print("=" * 60)
    print("SSL MODE")
    print("=" * 60)
    loader = build_dataloader(
        data_dir, split="train", mode="ssl",
        n_frames=16, batch_size=4, num_workers=0,
    )
    batch = next(iter(loader))
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:15s} : {val.shape}")
        # concentration   : (B, 16, 1, 256, 256)
        # velocity        : (B, 16, 2, 256, 256)
        # orientation     : (B, 16, 4, 256, 256)
        # strain_rate     : (B, 16, 4, 256, 256)
        # labels          : (B, 2)
