"""
PC-Reg: Positional Contextual Regression for anomaly detection on MVTec LOCO AD.

Fits a closed-form Ridge regression at each spatial position of the feature grid,
predicting the center patch from its neighbors. Anomaly scores are Mahalanobis
distances of the prediction residuals under the training distribution.

Backbone features (DINOv3 ViT-L/16) must be pre-cached before running this script.

Usage:
    uv run scripts/run_pcreg_loco.py
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import csv
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path

from _repro_utils import repo_root_from, skip_if_missing

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT = repo_root_from(__file__)
CACHE_DIR = REPO_ROOT / "features" / "loco"
OUTPUT_DIR = REPO_ROOT / "results" / "pcreg_loco"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]

# PC-Reg hyperparameters (best config)
RADIUS = 5
LAMBDA = 1.0
AGGREGATION = "p95"
PCA_DIM = 256
SEED = 0

GRID_SIZE = 28
N_PATCHES = GRID_SIZE * GRID_SIZE  # 784

CONFIG_NAME = f"R{RADIUS}_lam{LAMBDA}_p{AGGREGATION[1:]}"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cached_features(category: str) -> dict:
    """Load pre-cached features for a LOCO category."""
    cat_cache = CACHE_DIR / category
    if not cat_cache.exists():
        raise FileNotFoundError(
            f"Feature cache not found for {category}. "
            f"Run the feature extraction script first."
        )
    data = {}
    for split in ["train", "test_good", "test_logical", "test_structural"]:
        feat_path = cat_cache / f"{split}_features.npy"
        if feat_path.exists():
            data[split] = np.load(str(feat_path))
        else:
            raise FileNotFoundError(f"Missing: {feat_path}")
    return data


# =============================================================================
# PC-REG: RIDGE REGRESSION + MAHALANOBIS
# =============================================================================

def build_neighbor_map(grid_size: int, radius: int) -> dict[int, list[int]]:
    """Precompute spatial neighbors for each position in the grid.

    Returns dict: position_index -> list of neighbor indices (excluding self).
    """
    neighbors = {}
    for py in range(grid_size):
        for px in range(grid_size):
            p = py * grid_size + px
            nbrs = []
            for qy in range(max(0, py - radius), min(grid_size, py + radius + 1)):
                for qx in range(max(0, px - radius), min(grid_size, px + radius + 1)):
                    if qy == py and qx == px:
                        continue
                    q = qy * grid_size + qx
                    nbrs.append(q)
            neighbors[p] = nbrs
    return neighbors


def train_pcreg_model(
    train_features: np.ndarray,
    neighbor_map: dict[int, list[int]],
    lam: float,
) -> tuple[dict, dict, dict]:
    """Train Ridge regression + residual distribution for each position.

    Args:
        train_features: (N_train, N_patches, D) - PCA-reduced features
        neighbor_map: position -> list of neighbor positions
        lam: Ridge regularization

    Returns:
        W: dict[int, ndarray(D,D)] - regression weights per position
        mu_r: dict[int, ndarray(D,)] - mean residual per position
        Sigma_r_inv: dict[int, ndarray(D,D)] - inverse covariance per position
    """
    n_train, n_patches, dim = train_features.shape

    W = {}
    mu_r = {}
    Sigma_r_inv = {}

    for p in range(n_patches):
        nbrs = neighbor_map[p]

        # X = mean of neighbor features per train image: (N_train, D)
        X = train_features[:, nbrs, :].mean(axis=1)

        # Y = center feature per train image: (N_train, D)
        Y = train_features[:, p, :]

        # Ridge regression: W_p = (X^T X + lam*I)^{-1} X^T Y
        XtX = X.T @ X  # (D, D)
        XtY = X.T @ Y  # (D, D)
        W_p = np.linalg.solve(XtX + lam * np.eye(dim), XtY)  # (D, D)

        # Compute residuals on train
        Y_pred = X @ W_p  # (N_train, D)
        residuals = Y - Y_pred  # (N_train, D)

        # Mean and covariance of residuals
        mu = residuals.mean(axis=0)  # (D,)

        # Ledoit-Wolf shrinkage for well-conditioned covariance
        try:
            lw = LedoitWolf()
            lw.fit(residuals)
            Sigma_inv = lw.precision_  # Already inverted
        except Exception:
            # Fallback: diagonal + small ridge
            var = residuals.var(axis=0) + 1e-6
            Sigma_inv = np.diag(1.0 / var)

        W[p] = W_p
        mu_r[p] = mu
        Sigma_r_inv[p] = Sigma_inv

    return W, mu_r, Sigma_r_inv


def score_pcreg(
    test_features: np.ndarray,
    neighbor_map: dict[int, list[int]],
    W: dict,
    mu_r: dict,
    Sigma_r_inv: dict,
    agg: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Score test images using PC-Reg model.

    Args:
        test_features: (N_test, N_patches, D)
        neighbor_map, W, mu_r, Sigma_r_inv: trained model
        agg: aggregation method ("max" or "p95")

    Returns:
        image_scores: (N_test,)
        patch_scores: (N_test, N_patches)
    """
    n_test, n_patches, dim = test_features.shape
    patch_scores = np.zeros((n_test, n_patches))

    for p in range(n_patches):
        nbrs = neighbor_map[p]

        # Context: mean of neighbors for all test images
        X = test_features[:, nbrs, :].mean(axis=1)  # (N_test, D)

        # Predict
        Y_pred = X @ W[p]  # (N_test, D)
        Y_real = test_features[:, p, :]  # (N_test, D)

        # Residuals
        residuals = Y_real - Y_pred  # (N_test, D)

        # Mahalanobis distance: sqrt((r - mu)^T Sigma_inv (r - mu))
        centered = residuals - mu_r[p]  # (N_test, D)
        # Vectorized: (N_test, D) @ (D, D) -> (N_test, D), then element-wise * and sum
        mahal_sq = np.sum(centered @ Sigma_r_inv[p] * centered, axis=1)  # (N_test,)
        patch_scores[:, p] = np.sqrt(np.maximum(0, mahal_sq))

    # Aggregate to image score
    if agg == "max":
        image_scores = patch_scores.max(axis=1)
    elif agg == "p95":
        image_scores = np.percentile(patch_scores, 95, axis=1)
    else:
        image_scores = patch_scores.max(axis=1)

    return image_scores, patch_scores


# =============================================================================
# METRICS
# =============================================================================

def compute_auroc_splits(
    scores_good: np.ndarray,
    scores_logical: np.ndarray,
    scores_structural: np.ndarray,
) -> dict:
    """Compute AUROC for combined, logical-only, and structural-only."""
    results = {}

    labels_comb = np.concatenate([
        np.zeros(len(scores_good)),
        np.ones(len(scores_logical)),
        np.ones(len(scores_structural)),
    ])
    scores_comb = np.concatenate([scores_good, scores_logical, scores_structural])
    results["auroc_combined"] = roc_auc_score(labels_comb, scores_comb) if len(np.unique(labels_comb)) > 1 else 0.0

    labels_log = np.concatenate([np.zeros(len(scores_good)), np.ones(len(scores_logical))])
    scores_log = np.concatenate([scores_good, scores_logical])
    results["auroc_logical"] = roc_auc_score(labels_log, scores_log) if len(np.unique(labels_log)) > 1 else 0.0

    labels_str = np.concatenate([np.zeros(len(scores_good)), np.ones(len(scores_structural))])
    scores_str = np.concatenate([scores_good, scores_structural])
    results["auroc_structural"] = roc_auc_score(labels_str, scores_str) if len(np.unique(labels_str)) > 1 else 0.0

    return results


# =============================================================================
# RESUMABILITY
# =============================================================================

FIELDNAMES = [
    "timestamp", "category", "config", "seed",
    "auroc_combined", "auroc_logical", "auroc_structural",
    "n_train", "n_test_good", "n_test_logical", "n_test_structural",
    "inference_time_s",
]


def get_completed() -> set[str]:
    csv_path = OUTPUT_DIR / "results.csv"
    completed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                completed.add(row["category"])
    return completed


def save_result(result: dict) -> None:
    csv_path = OUTPUT_DIR / "results.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def log_error(category: str, error: Exception) -> None:
    log_path = OUTPUT_DIR / "errors.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{datetime.now().isoformat()}] {category}\n")
        f.write(traceback.format_exc())


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PC-Reg: Positional Contextual Regression")
    print(f"  Dataset:  MVTec LOCO AD ({len(CATEGORIES)} categories)")
    print(f"  Config:   R={RADIUS}, lambda={LAMBDA}, agg={AGGREGATION}, PCA={PCA_DIM}")
    print(f"  Output:   {OUTPUT_DIR}")
    print("=" * 70)

    if skip_if_missing(
        required=[CACHE_DIR],
        precomputed=[OUTPUT_DIR / "results.csv"],
        what="cached LOCO features (features/loco)",
        reproduce_hint=(
            "  uv run scripts/extract_features_loco.py\n"
            "  uv run scripts/run_pcreg_loco.py"
        ),
    ):
        return

    completed = get_completed()
    remaining = [c for c in CATEGORIES if c not in completed]
    print(f"[INFO] Completed: {len(completed)}, remaining: {len(remaining)}")

    if not remaining:
        print("[INFO] All categories already completed. Exiting.")
        return

    # Precompute neighbor map
    neighbor_map = build_neighbor_map(GRID_SIZE, RADIUS)

    for category in CATEGORIES:
        if category in completed:
            continue

        print(f"\n{'='*60}")
        print(f"Category: {category}")

        # -- DATA LOADING --
        try:
            data = load_cached_features(category)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            continue

        train_features = data["train"]
        test_good_features = data["test_good"]
        test_logical_features = data["test_logical"]
        test_structural_features = data["test_structural"]

        print(f"  train: {train_features.shape}")
        print(f"  test_good: {test_good_features.shape}")
        print(f"  test_logical: {test_logical_features.shape}")
        print(f"  test_structural: {test_structural_features.shape}")

        # -- PCA REDUCTION --
        n_train_imgs, n_patches, feat_dim = train_features.shape
        all_train_patches = train_features.reshape(-1, feat_dim)
        pca = PCA(n_components=PCA_DIM, random_state=42)
        pca.fit(all_train_patches)

        train_pca = pca.transform(train_features.reshape(-1, feat_dim)).reshape(n_train_imgs, n_patches, PCA_DIM)
        test_good_pca = pca.transform(test_good_features.reshape(-1, feat_dim)).reshape(-1, n_patches, PCA_DIM)
        test_logical_pca = pca.transform(test_logical_features.reshape(-1, feat_dim)).reshape(-1, n_patches, PCA_DIM)
        test_structural_pca = pca.transform(test_structural_features.reshape(-1, feat_dim)).reshape(-1, n_patches, PCA_DIM)

        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA: {feat_dim} -> {PCA_DIM} (variance retained: {var_explained:.3f})")

        try:
            t0 = time.time()

            # -- PC-REG FIT --
            W, mu_r_dict, Sigma_r_inv_dict = train_pcreg_model(
                train_pca, neighbor_map, LAMBDA,
            )

            # -- SCORING --
            scores_good, _ = score_pcreg(
                test_good_pca, neighbor_map, W, mu_r_dict, Sigma_r_inv_dict, AGGREGATION,
            )
            scores_logical, _ = score_pcreg(
                test_logical_pca, neighbor_map, W, mu_r_dict, Sigma_r_inv_dict, AGGREGATION,
            )
            scores_structural, _ = score_pcreg(
                test_structural_pca, neighbor_map, W, mu_r_dict, Sigma_r_inv_dict, AGGREGATION,
            )

            # -- METRICS --
            metrics = compute_auroc_splits(scores_good, scores_logical, scores_structural)
            elapsed = time.time() - t0

            result = {
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "config": CONFIG_NAME,
                "seed": SEED,
                "auroc_combined": f"{metrics['auroc_combined']:.4f}",
                "auroc_logical": f"{metrics['auroc_logical']:.4f}",
                "auroc_structural": f"{metrics['auroc_structural']:.4f}",
                "n_train": train_pca.shape[0],
                "n_test_good": len(scores_good),
                "n_test_logical": len(scores_logical),
                "n_test_structural": len(scores_structural),
                "inference_time_s": f"{elapsed:.1f}",
            }
            save_result(result)

            print(f"  {CONFIG_NAME}: "
                  f"log={metrics['auroc_logical']:.3f} "
                  f"str={metrics['auroc_structural']:.3f} "
                  f"comb={metrics['auroc_combined']:.3f} "
                  f"({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [ERROR] {category}: {e}")
            log_error(category, e)
            continue

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Results: {OUTPUT_DIR / 'results.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
