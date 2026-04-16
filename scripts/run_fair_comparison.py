"""
Fair Comparison: PatchCore, PaDiM, MeanSub vs PC-Reg
=====================================================
All methods use the EXACT same setup:
  - DINOv3-ViT-L/16 features (pre-cached), layer -6
  - PCA d=256, random_state=42
  - P95 image-level aggregation
  - Same train/test splits

Methods:
  1. PatchCore: k-NN memory bank (k=1, L2). Patch-independent.
  2. PaDiM: Per-position Gaussian N(mu_i, Sigma_i). Models p(f_i). No context.
  3. MeanSub: r_i = f_i - mean(neighbors). Simplest contextual baseline.
  4. PC-Reg: Ridge regression + Mahalanobis. Models p(f_i|c_i). Our method.

Datasets: LOCO (5 cats), MVTec AD (15 cats), VisA (12 cats) = 32 categories.

Purpose: Demonstrate that the contribution comes from CONTEXTUAL REGRESSION,
not from the backbone or PCA pipeline.

Usage:
  uv run scripts/run_fair_comparison.py              # all 32 categories
  uv run scripts/run_fair_comparison.py --loco       # LOCO only (5 cats)
  uv run scripts/run_fair_comparison.py --mvtec      # MVTec AD only (15 cats)
  uv run scripts/run_fair_comparison.py --visa       # VisA only (12 cats)
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import sys
import csv
import gc
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
from sklearn.neighbors import NearestNeighbors

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT = repo_root_from(__file__)
OUTPUT_DIR = REPO_ROOT / "results" / "fair_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PCA_DIM = 256
RADIUS = 5
LAMBDA = 1.0

# Dataset definitions: (cache_dir, categories, split_type)
# split_type: "loco" has logical+structural, "binary" has good+anomaly
DATASETS = {}

LOCO_CATS = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]
MVTEC_CATS = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
VISA_CATS = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3",
    "pcb4", "pipe_fryum",
]

for cat in LOCO_CATS:
    DATASETS[f"LOCO/{cat}"] = {
        "cache_dir": REPO_ROOT / "features" / "loco" / cat,
        "split_type": "loco",
        "dataset": "LOCO",
        "category": cat,
    }
for cat in MVTEC_CATS:
    DATASETS[f"MVTec/{cat}"] = {
        "cache_dir": REPO_ROOT / "features" / "mvtec" / cat,
        "split_type": "binary",
        "dataset": "MVTec_AD",
        "category": cat,
    }
for cat in VISA_CATS:
    DATASETS[f"VisA/{cat}"] = {
        "cache_dir": REPO_ROOT / "features" / "visa" / cat,
        "split_type": "binary",
        "dataset": "VisA",
        "category": cat,
    }

# Filter by command-line args
if "--loco" in sys.argv:
    ACTIVE_KEYS = [k for k in DATASETS if DATASETS[k]["dataset"] == "LOCO"]
elif "--mvtec" in sys.argv:
    ACTIVE_KEYS = [k for k in DATASETS if DATASETS[k]["dataset"] == "MVTec_AD"]
elif "--visa" in sys.argv:
    ACTIVE_KEYS = [k for k in DATASETS if DATASETS[k]["dataset"] == "VisA"]
else:
    ACTIVE_KEYS = list(DATASETS.keys())

METHODS = ["patchcore", "padim", "meansub", "pcreg"]


# =============================================================================
# NEIGHBOR MEANS (integral images -- fast)
# =============================================================================

def compute_neighbor_means(features, grid_size, radius):
    """O(1) per-position neighbor mean via summed-area tables."""
    N, _, D = features.shape
    H = W = grid_size
    F = features.reshape(N, H, W, D)

    S = np.zeros((N, H + 1, W + 1, D), dtype=np.float64)
    S[:, 1:, 1:, :] = np.cumsum(np.cumsum(F.astype(np.float64), axis=1), axis=2)

    pos = np.arange(H * W)
    y, x = pos // W, pos % W
    y1 = np.maximum(0, y - radius)
    y2 = np.minimum(H - 1, y + radius) + 1
    x1 = np.maximum(0, x - radius)
    x2 = np.minimum(W - 1, x + radius) + 1

    window_sums = (S[:, y2, x2, :] - S[:, y1, x2, :]
                   - S[:, y2, x1, :] + S[:, y1, x1, :])
    window_sums -= F.reshape(N, H * W, D).astype(np.float64)
    counts = (y2 - y1) * (x2 - x1) - 1

    return (window_sums / counts[None, :, None]).astype(np.float32)


# =============================================================================
# METHOD 1: PATCHCORE
# =============================================================================

def run_patchcore(train_pca, test_pca_dict, grid_size):
    """PatchCore: k-NN memory bank, k=1, L2, P95."""
    n_train, n_patches, dim = train_pca.shape

    # Build memory bank (all train patches)
    memory = train_pca.reshape(-1, dim)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto")
    nn.fit(memory)

    results = {}
    for split_name, test_feat in test_pca_dict.items():
        n_test = test_feat.shape[0]
        flat = test_feat.reshape(-1, dim)
        distances, _ = nn.kneighbors(flat)
        patch_scores = distances[:, 0].reshape(n_test, n_patches)
        results[split_name] = np.percentile(patch_scores, 95, axis=1)

    return results


# =============================================================================
# METHOD 2: PaDiM
# =============================================================================

def run_padim(train_pca, test_pca_dict, grid_size):
    """PaDiM: Per-position Gaussian marginal N(mu_i, Sigma_i), Mahalanobis."""
    n_train, n_patches, dim = train_pca.shape

    # Fit per-position Gaussian
    mu_arr = np.zeros((n_patches, dim), dtype=np.float64)
    Sigma_inv_arr = np.zeros((n_patches, dim, dim), dtype=np.float64)

    for p in range(n_patches):
        Y = train_pca[:, p, :].astype(np.float64)
        mu_arr[p] = Y.mean(axis=0)
        try:
            lw = LedoitWolf()
            lw.fit(Y)
            Sigma_inv_arr[p] = lw.precision_
        except Exception:
            var = Y.var(axis=0) + 1e-6
            Sigma_inv_arr[p] = np.diag(1.0 / var)

    # Score
    results = {}
    for split_name, test_feat in test_pca_dict.items():
        centered = test_feat.astype(np.float64) - mu_arr[np.newaxis, :, :]
        temp = np.einsum('npd,pde->npe', centered, Sigma_inv_arr)
        mahal_sq = np.sum(temp * centered, axis=2)
        patch_scores = np.sqrt(np.maximum(0, mahal_sq))
        results[split_name] = np.percentile(patch_scores, 95, axis=1)

    return results


# =============================================================================
# METHOD 3: MEANSUB (simplest contextual)
# =============================================================================

def run_meansub(train_pca, test_pca_dict, grid_size, radius):
    """MeanSub: r_i = f_i - mean(neighbors), then per-position Mahalanobis."""
    n_train, n_patches, dim = train_pca.shape

    # Compute residuals for training data
    train_nbr_means = compute_neighbor_means(train_pca, grid_size, radius)
    train_residuals = (train_pca - train_nbr_means).astype(np.float64)

    # Fit per-position Gaussian on residuals
    mu_arr = np.zeros((n_patches, dim), dtype=np.float64)
    Sigma_inv_arr = np.zeros((n_patches, dim, dim), dtype=np.float64)

    for p in range(n_patches):
        R = train_residuals[:, p, :]
        mu_arr[p] = R.mean(axis=0)
        try:
            lw = LedoitWolf()
            lw.fit(R)
            Sigma_inv_arr[p] = lw.precision_
        except Exception:
            var = R.var(axis=0) + 1e-6
            Sigma_inv_arr[p] = np.diag(1.0 / var)

    # Score
    results = {}
    for split_name, test_feat in test_pca_dict.items():
        test_nbr_means = compute_neighbor_means(test_feat, grid_size, radius)
        test_residuals = (test_feat - test_nbr_means).astype(np.float64)
        centered = test_residuals - mu_arr[np.newaxis, :, :]
        temp = np.einsum('npd,pde->npe', centered, Sigma_inv_arr)
        mahal_sq = np.sum(temp * centered, axis=2)
        patch_scores = np.sqrt(np.maximum(0, mahal_sq))
        results[split_name] = np.percentile(patch_scores, 95, axis=1)

    return results


# =============================================================================
# METHOD 4: PC-REG (our method)
# =============================================================================

def run_pcreg(train_pca, test_pca_dict, grid_size, radius, lam):
    """PC-Reg: Ridge regression + Mahalanobis residual scoring."""
    n_train, n_patches, dim = train_pca.shape

    train_nbr_means = compute_neighbor_means(train_pca, grid_size, radius)

    I = np.eye(dim, dtype=np.float64)
    W_arr = np.zeros((n_patches, dim, dim), dtype=np.float64)
    mu_arr = np.zeros((n_patches, dim), dtype=np.float64)
    Sigma_inv_arr = np.zeros((n_patches, dim, dim), dtype=np.float64)

    for p in range(n_patches):
        X = train_nbr_means[:, p, :].astype(np.float64)
        Y = train_pca[:, p, :].astype(np.float64)
        XtX = X.T @ X
        XtY = X.T @ Y
        W_p = np.linalg.solve(XtX + lam * I, XtY)
        residuals = Y - X @ W_p
        mu = residuals.mean(axis=0)
        try:
            lw = LedoitWolf()
            lw.fit(residuals)
            Sigma_inv = lw.precision_
        except Exception:
            var = residuals.var(axis=0) + 1e-6
            Sigma_inv = np.diag(1.0 / var)
        W_arr[p] = W_p
        mu_arr[p] = mu
        Sigma_inv_arr[p] = Sigma_inv

    # Score
    results = {}
    for split_name, test_feat in test_pca_dict.items():
        test_nbr_means = compute_neighbor_means(test_feat, grid_size, radius)
        Y_pred = np.einsum('npd,pde->npe',
                           test_nbr_means.astype(np.float64), W_arr)
        centered = (test_feat.astype(np.float64) - Y_pred) - mu_arr[np.newaxis, :, :]
        temp = np.einsum('npd,pde->npe', centered, Sigma_inv_arr)
        mahal_sq = np.sum(temp * centered, axis=2)
        patch_scores = np.sqrt(np.maximum(0, mahal_sq))
        results[split_name] = np.percentile(patch_scores, 95, axis=1)

    return results


# =============================================================================
# AUROC COMPUTATION
# =============================================================================

def compute_aurocs(scores_dict, split_type):
    """Compute AUROC(s) from method output scores."""
    if split_type == "loco":
        s_good = scores_dict["test_good"]
        s_log = scores_dict["test_logical"]
        s_str = scores_dict["test_structural"]

        # Combined
        labels_c = np.concatenate([np.zeros(len(s_good)),
                                   np.ones(len(s_log) + len(s_str))])
        scores_c = np.concatenate([s_good, s_log, s_str])
        auroc_combined = roc_auc_score(labels_c, scores_c)

        # Logical only
        labels_l = np.concatenate([np.zeros(len(s_good)), np.ones(len(s_log))])
        scores_l = np.concatenate([s_good, s_log])
        auroc_logical = roc_auc_score(labels_l, scores_l)

        # Structural only
        labels_s = np.concatenate([np.zeros(len(s_good)), np.ones(len(s_str))])
        scores_s = np.concatenate([s_good, s_str])
        auroc_structural = roc_auc_score(labels_s, scores_s)

        return {
            "auroc_combined": auroc_combined,
            "auroc_logical": auroc_logical,
            "auroc_structural": auroc_structural,
        }
    else:  # binary
        s_good = scores_dict["test_good"]
        s_anom = scores_dict["test_anomaly"]
        labels = np.concatenate([np.zeros(len(s_good)), np.ones(len(s_anom))])
        scores = np.concatenate([s_good, s_anom])
        auroc = roc_auc_score(labels, scores)
        return {
            "auroc_combined": auroc,
            "auroc_logical": -1.0,  # N/A
            "auroc_structural": -1.0,  # N/A
        }


# =============================================================================
# I/O
# =============================================================================

FIELDNAMES = [
    "timestamp", "dataset", "category", "method",
    "auroc_combined", "auroc_logical", "auroc_structural",
    "n_train", "n_test_good", "n_test_anomaly",
    "time_s",
]


def get_completed() -> set[tuple]:
    csv_path = OUTPUT_DIR / "results.csv"
    completed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                completed.add((row["dataset"], row["category"], row["method"]))
    return completed


def save_result(result: dict) -> None:
    csv_path = OUTPUT_DIR / "results.csv"
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
        if not exists:
            w.writeheader()
        w.writerow(result)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Fair Comparison: PatchCore, PaDiM, MeanSub vs PC-Reg")
    print(f"  Config: PCA={PCA_DIM}, R={RADIUS}, lam={LAMBDA}, P95")
    print(f"  Datasets: {len(ACTIVE_KEYS)} categories")
    print(f"  Methods: {METHODS}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    if skip_if_missing(
        required=[REPO_ROOT / "features"],
        precomputed=[OUTPUT_DIR / "results.csv"],
        what="cached features (features/*)",
        reproduce_hint=(
            "  # 1) Place datasets under pc-reg/data\n"
            "  # 2) Cache features:\n"
            "  uv run scripts/extract_features_loco.py\n"
            "  uv run scripts/extract_features_mvtec.py\n"
            "  uv run scripts/extract_features_visa.py\n"
            "  # 3) Run fair comparison:\n"
            "  uv run scripts/run_fair_comparison.py"
        ),
    ):
        return

    completed = get_completed()
    print(f"[INFO] Already completed: {len(completed)} runs")

    total_runs = len(ACTIVE_KEYS) * len(METHODS)
    remaining = sum(1 for key in ACTIVE_KEYS for m in METHODS
                    if (DATASETS[key]["dataset"], DATASETS[key]["category"], m)
                    not in completed)
    print(f"[INFO] Total: {total_runs}, Remaining: {remaining}")

    if remaining == 0:
        print("[INFO] All runs already completed.")
        print_summary()
        return

    GRID_SIZE = 28  # DINOv3-ViT-L/16 at 448

    for key in ACTIVE_KEYS:
        info = DATASETS[key]
        dataset = info["dataset"]
        category = info["category"]
        cache_dir = info["cache_dir"]
        split_type = info["split_type"]

        # Check which methods still need to run for this category
        methods_todo = [m for m in METHODS
                        if (dataset, category, m) not in completed]
        if not methods_todo:
            continue

        print(f"\n{'='*60}")
        print(f"[{dataset}] {category} -- methods: {methods_todo}")

        # Load features
        try:
            train_features = np.load(str(cache_dir / "train_features.npy"))
            test_good_features = np.load(str(cache_dir / "test_good_features.npy"))

            if split_type == "loco":
                test_logical = np.load(str(cache_dir / "test_logical_features.npy"))
                test_structural = np.load(str(cache_dir / "test_structural_features.npy"))
                n_test_anomaly = test_logical.shape[0] + test_structural.shape[0]
            else:
                test_anomaly_features = np.load(str(cache_dir / "test_anomaly_features.npy"))
                n_test_anomaly = test_anomaly_features.shape[0]
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue

        n_train = train_features.shape[0]
        n_test_good = test_good_features.shape[0]
        D = train_features.shape[2]

        print(f"  train: {n_train}, test_good: {n_test_good}, test_anomaly: {n_test_anomaly}")

        # PCA
        pca = PCA(n_components=PCA_DIM, random_state=42)
        pca.fit(train_features.reshape(-1, D))

        train_pca = pca.transform(
            train_features.reshape(-1, D)
        ).reshape(n_train, -1, PCA_DIM).astype(np.float32)

        # Build test splits dict
        if split_type == "loco":
            all_test = np.concatenate([
                test_good_features.reshape(-1, D),
                test_logical.reshape(-1, D),
                test_structural.reshape(-1, D),
            ])
            all_test_pca = pca.transform(all_test).astype(np.float32)
            n_patches = train_pca.shape[1]

            idx = 0
            tg = all_test_pca[idx:idx + n_test_good * n_patches].reshape(
                n_test_good, n_patches, PCA_DIM)
            idx += n_test_good * n_patches
            n_log = test_logical.shape[0]
            tl = all_test_pca[idx:idx + n_log * n_patches].reshape(
                n_log, n_patches, PCA_DIM)
            idx += n_log * n_patches
            n_str = test_structural.shape[0]
            ts = all_test_pca[idx:idx + n_str * n_patches].reshape(
                n_str, n_patches, PCA_DIM)

            test_pca_dict = {
                "test_good": tg,
                "test_logical": tl,
                "test_structural": ts,
            }
        else:
            all_test = np.concatenate([
                test_good_features.reshape(-1, D),
                test_anomaly_features.reshape(-1, D),
            ])
            all_test_pca = pca.transform(all_test).astype(np.float32)
            n_patches = train_pca.shape[1]

            i0 = n_test_good * n_patches
            tg = all_test_pca[:i0].reshape(n_test_good, n_patches, PCA_DIM)
            ta = all_test_pca[i0:].reshape(n_test_anomaly, n_patches, PCA_DIM)

            test_pca_dict = {
                "test_good": tg,
                "test_anomaly": ta,
            }

        # Free raw features
        del train_features, test_good_features
        if split_type == "loco":
            del test_logical, test_structural
        else:
            del test_anomaly_features
        del all_test, all_test_pca
        gc.collect()

        # Run each method
        for method in methods_todo:
            if (dataset, category, method) in completed:
                continue

            print(f"  [{method}]...", end=" ", flush=True)
            t0 = time.time()

            try:
                if method == "patchcore":
                    scores = run_patchcore(train_pca, test_pca_dict, GRID_SIZE)
                elif method == "padim":
                    scores = run_padim(train_pca, test_pca_dict, GRID_SIZE)
                elif method == "meansub":
                    scores = run_meansub(train_pca, test_pca_dict, GRID_SIZE, RADIUS)
                elif method == "pcreg":
                    scores = run_pcreg(train_pca, test_pca_dict, GRID_SIZE, RADIUS, LAMBDA)
                else:
                    raise ValueError(f"Unknown method: {method}")

                elapsed = time.time() - t0
                aurocs = compute_aurocs(scores, split_type)

                result = {
                    "timestamp": datetime.now().isoformat(),
                    "dataset": dataset,
                    "category": category,
                    "method": method,
                    "auroc_combined": f"{aurocs['auroc_combined']:.4f}",
                    "auroc_logical": f"{aurocs['auroc_logical']:.4f}",
                    "auroc_structural": f"{aurocs['auroc_structural']:.4f}",
                    "n_train": n_train,
                    "n_test_good": n_test_good,
                    "n_test_anomaly": n_test_anomaly,
                    "time_s": f"{elapsed:.1f}",
                }
                save_result(result)

                if split_type == "loco":
                    print(f"combined={aurocs['auroc_combined']:.4f} "
                          f"(log={aurocs['auroc_logical']:.4f} "
                          f"str={aurocs['auroc_structural']:.4f}) "
                          f"[{elapsed:.1f}s]")
                else:
                    print(f"AUROC={aurocs['auroc_combined']:.4f} [{elapsed:.1f}s]")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR: {e} [{elapsed:.1f}s]")
                traceback.print_exc()

        # Free PCA data for this category
        del train_pca, test_pca_dict
        gc.collect()

    print_summary()


def print_summary():
    """Print results summary grouped by dataset and method."""
    csv_path = OUTPUT_DIR / "results.csv"
    if not csv_path.exists():
        return

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Load all results
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            rows.append(row)

    # Group by dataset
    for ds in ["LOCO", "MVTec_AD", "VisA"]:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            continue

        print(f"\n--- {ds} ---")
        for method in METHODS:
            m_rows = [r for r in ds_rows if r["method"] == method]
            if not m_rows:
                continue
            aurocs = [float(r["auroc_combined"]) for r in m_rows]
            mean_auroc = np.mean(aurocs) * 100
            n_cats = len(aurocs)

            if ds == "LOCO" and m_rows[0]["auroc_logical"] != "-1.0000":
                log_aurocs = [float(r["auroc_logical"]) for r in m_rows]
                str_aurocs = [float(r["auroc_structural"]) for r in m_rows]
                print(f"  {method:12s}: {mean_auroc:5.1f}% "
                      f"(log={np.mean(log_aurocs)*100:.1f} "
                      f"str={np.mean(str_aurocs)*100:.1f}) "
                      f"[{n_cats} cats]")
            else:
                print(f"  {method:12s}: {mean_auroc:5.1f}% [{n_cats} cats]")

    print(f"\n{'='*70}")
    print(f"Results: {csv_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
