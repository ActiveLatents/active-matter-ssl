"""
evaluate.py
===========
Evaluation pipeline for linear probing and kNN regression.

Assumes representations are already extracted and saved as:
  - features:  (N, D) float array
  - labels:    (N, 2) float array  — columns are [alpha, zeta], raw (un-normalised)

Supported input formats: .npy or .pt

Usage
-----
python evaluate.py \
    --train_feats  cache/train_feats.npy \
    --train_labels cache/train_labels.npy \
    --eval_feats   cache/val_feats.npy \
    --eval_labels  cache/val_labels.npy \
    --mode both \
    --knn_k 5 10 20
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 0.  Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 1.  I/O helpers
# ---------------------------------------------------------------------------

def load_array(path: str) -> np.ndarray:
    """Load a .npy or .pt file into a numpy array."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix == ".pt":
        import torch
        return torch.load(p, map_location="cpu").numpy()
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .npy or .pt")


# ---------------------------------------------------------------------------
# 2.  Label normalisation  (z-score, fit on train labels only)
# ---------------------------------------------------------------------------

def fit_label_normalizer(train_labels: np.ndarray) -> StandardScaler:
    """
    Fit a z-score normaliser on training labels only.

    Parameters
    ----------
    train_labels : (N_train, 2)  — columns are [alpha, zeta], raw values

    Returns
    -------
    scaler : fitted StandardScaler (col 0 = alpha, col 1 = zeta)
    """
    scaler = StandardScaler()
    scaler.fit(train_labels)
    logging.info(
        f"Label normaliser fitted — "
        f"alpha: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f} | "
        f"zeta:  mean={scaler.mean_[1]:.4f}, std={scaler.scale_[1]:.4f}"
    )
    return scaler


# ---------------------------------------------------------------------------
# 3.  Evaluation helpers
# ---------------------------------------------------------------------------

def compute_mse(true: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute per-parameter and mean MSE.

    Parameters
    ----------
    true, pred : (N, 2)  — z-scored values

    Returns
    -------
    dict with keys: mse_alpha, mse_zeta, mse_mean
    """
    return {
        "mse_alpha": float(mean_squared_error(true[:, 0], pred[:, 0])),
        "mse_zeta":  float(mean_squared_error(true[:, 1], pred[:, 1])),
        "mse_mean":  float(mean_squared_error(true, pred)),
    }


def log_metrics(tag: str, metrics: dict):
    logging.info(
        f"[{tag}]  alpha={metrics['mse_alpha']:.4f}  "
        f"zeta={metrics['mse_zeta']:.4f}  mean={metrics['mse_mean']:.4f}"
    )


# ---------------------------------------------------------------------------
# 4a.  Linear Probe  (single linear layer = Ridge with alpha→0 ≈ LinearRegression)
# ---------------------------------------------------------------------------

def run_linear_probe(
    train_feats:  np.ndarray,
    train_labels: np.ndarray,
    eval_feats:   np.ndarray,
    eval_labels:  np.ndarray,
    alpha:        float = 1e-3,    # Ridge regularisation; sweep this if needed
) -> dict:
    """
    Fit a single linear layer (Ridge regression) on pre-extracted features.

    Ridge is used instead of plain LinearRegression for numerical stability
    with high-dimensional feature vectors. Setting alpha very small (1e-3)
    approximates an unregularised linear layer, which matches the spirit of
    'single linear layer' in the project spec.

    Parameters
    ----------
    train/eval_feats  : (N, D)  pre-extracted, frozen encoder representations
    train/eval_labels : (N, 2)  z-scored [alpha, zeta]
    alpha             : Ridge regularisation strength

    Returns
    -------
    metrics : dict with mse_alpha, mse_zeta, mse_mean on eval set
    """
    probe = Ridge(alpha=alpha)
    probe.fit(train_feats, train_labels)              # fit jointly on both targets

    preds   = probe.predict(eval_feats)               # (N_eval, 2)
    metrics = compute_mse(eval_labels, preds)
    log_metrics(f"LinearProbe alpha={alpha}", metrics)
    return metrics


# ---------------------------------------------------------------------------
# 4b.  kNN Regression
# ---------------------------------------------------------------------------

def run_knn(
    train_feats:  np.ndarray,
    train_labels: np.ndarray,
    eval_feats:   np.ndarray,
    eval_labels:  np.ndarray,
    k:            int = 10,
    metric:       str = "euclidean",
) -> dict:
    """
    Fit a kNN regressor on pre-extracted features and evaluate.

    Parameters
    ----------
    k      : number of neighbours
    metric : distance metric — 'euclidean' or 'cosine'

    Returns
    -------
    metrics : dict with k, metric, mse_alpha, mse_zeta, mse_mean on eval set
    """
    knn = KNeighborsRegressor(n_neighbors=k, metric=metric, n_jobs=-1)
    knn.fit(train_feats, train_labels)

    preds   = knn.predict(eval_feats)                 # (N_eval, 2)
    metrics = compute_mse(eval_labels, preds)
    metrics.update({"k": k, "metric": metric})
    log_metrics(f"kNN k={k} metric={metric}", metrics)
    return metrics


# ---------------------------------------------------------------------------
# 5.  Entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Linear probe & kNN evaluation")

    # --- input files ---
    p.add_argument("--train_feats",  required=True, help=".npy or .pt  (N_train, D)")
    p.add_argument("--train_labels", required=True, help=".npy or .pt  (N_train, 2) — raw [alpha, zeta]")
    p.add_argument("--eval_feats",   required=True, help=".npy or .pt  (N_eval,  D)")
    p.add_argument("--eval_labels",  required=True, help=".npy or .pt  (N_eval,  2) — raw [alpha, zeta]")

    # --- mode ---
    p.add_argument("--mode", default="both", choices=["linear", "knn", "both"])

    # --- linear probe ---
    p.add_argument("--lp_alpha", type=float, nargs="+", default=[1e-3],
                   help="Ridge regularisation strength(s) to sweep")

    # --- kNN ---
    p.add_argument("--knn_k",      type=int, nargs="+", default=[5, 10, 20],
                   help="k value(s) to sweep")
    p.add_argument("--knn_metric", default="euclidean", choices=["euclidean", "cosine"])

    # --- misc ---
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    # ------------------------------------------------------------------
    # Load pre-extracted features and raw labels
    # ------------------------------------------------------------------
    logging.info("Loading features and labels …")
    train_feats  = load_array(args.train_feats)   # (N_train, D)
    train_labels = load_array(args.train_labels)  # (N_train, 2)
    eval_feats   = load_array(args.eval_feats)    # (N_eval,  D)
    eval_labels  = load_array(args.eval_labels)   # (N_eval,  2)

    logging.info(
        f"train: feats={train_feats.shape}  labels={train_labels.shape} | "
        f"eval:  feats={eval_feats.shape}   labels={eval_labels.shape}"
    )

    # ------------------------------------------------------------------
    # Z-score labels using train statistics only
    # ------------------------------------------------------------------
    scaler       = fit_label_normalizer(train_labels)
    train_labels = scaler.transform(train_labels)  # (N_train, 2) z-scored
    eval_labels  = scaler.transform(eval_labels)   # (N_eval,  2) z-scored

    # ------------------------------------------------------------------
    # Linear Probe
    # ------------------------------------------------------------------
    best_lp = None
    if args.mode in ("linear", "both"):
        logging.info("─" * 50)
        logging.info("LINEAR PROBE")
        lp_results = []
        for alpha in args.lp_alpha:
            m = run_linear_probe(train_feats, train_labels, eval_feats, eval_labels, alpha=alpha)
            lp_results.append(m)

        best_lp = min(lp_results, key=lambda x: x["mse_mean"])
        logging.info(f"Best linear probe: {best_lp}")

    # ------------------------------------------------------------------
    # kNN Regression
    # ------------------------------------------------------------------
    best_knn = None
    if args.mode in ("knn", "both"):
        logging.info("─" * 50)
        logging.info("kNN REGRESSION")
        knn_results = []
        for k in args.knn_k:
            m = run_knn(
                train_feats, train_labels,
                eval_feats,  eval_labels,
                k=k, metric=args.knn_metric,
            )
            knn_results.append(m)

        best_knn = min(knn_results, key=lambda x: x["mse_mean"])
        logging.info(f"Best kNN: {best_knn}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logging.info("=" * 50)
    logging.info("SUMMARY")
    if best_lp is not None:
        logging.info(
            f"  Linear probe (best) — "
            f"alpha={best_lp['mse_alpha']:.4f}  "
            f"zeta={best_lp['mse_zeta']:.4f}  "
            f"mean={best_lp['mse_mean']:.4f}"
        )
    if best_knn is not None:
        logging.info(
            f"  kNN (best k={best_knn['k']}) — "
            f"alpha={best_knn['mse_alpha']:.4f}  "
            f"zeta={best_knn['mse_zeta']:.4f}  "
            f"mean={best_knn['mse_mean']:.4f}"
        )
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
