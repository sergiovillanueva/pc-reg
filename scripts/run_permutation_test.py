"""
Synthetic Permutation Test: PC-Reg vs PatchCore
================================================
Demonstrates that PC-Reg detects broken spatial relations while
PatchCore (nearest-neighbor) does not.

Method:
  1. Take test_good images (known normal)
  2. Swap 7x7 blocks in feature space (not pixel space)
  3. Score with both PatchCore and PC-Reg
  4. PatchCore should NOT detect (same patches, different positions)
  5. PC-Reg SHOULD detect (contextual relations are broken)

This is the "smoking gun" figure for the paper.

Usage:
  uv run scripts/run_permutation_test.py
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import csv
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

from _repro_utils import repo_root_from

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT = repo_root_from(__file__)
CACHE_DIR = REPO_ROOT / "features" / "loco"
OUTPUT_DIR = REPO_ROOT / "results" / "permutation_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]

PCA_DIM = 256
GRID_SIZE = 28
N_PATCHES = GRID_SIZE * GRID_SIZE
BLOCK_SIZE = 7  # 7x7 block permutation
SEED = 42

# PC-Reg params
R = 5
LAMBDA = 1.0
AGG = "p95"


# =============================================================================
# BLOCK PERMUTATION
# =============================================================================

def permute_blocks(features: np.ndarray, grid_size: int, block_size: int,
                   n_swaps: int, rng: np.random.Generator) -> np.ndarray:
    """Swap random block pairs in feature maps.

    Args:
        features: (N, grid*grid, D) feature maps
        grid_size: spatial grid dimension (28)
        block_size: block size to swap (7)
        n_swaps: number of block swaps per image
        rng: random generator

    Returns:
        perturbed: (N, grid*grid, D) with swapped blocks
    """
    perturbed = features.copy()
    n_images = features.shape[0]

    # Valid top-left positions for blocks
    max_pos = grid_size - block_size  # 28 - 7 = 21
    valid_positions = [(r, c) for r in range(max_pos + 1)
                       for c in range(max_pos + 1)]

    for img_idx in range(n_images):
        for _ in range(n_swaps):
            # Pick two non-overlapping blocks
            attempts = 0
            while attempts < 100:
                pos1 = valid_positions[rng.integers(len(valid_positions))]
                pos2 = valid_positions[rng.integers(len(valid_positions))]

                # Check non-overlapping
                r1, c1 = pos1
                r2, c2 = pos2
                if (abs(r1 - r2) >= block_size or abs(c1 - c2) >= block_size):
                    break
                attempts += 1

            if attempts >= 100:
                continue  # Skip if can't find non-overlapping

            # Get patch indices for both blocks
            idx1 = []
            idx2 = []
            for dr in range(block_size):
                for dc in range(block_size):
                    idx1.append((r1 + dr) * grid_size + (c1 + dc))
                    idx2.append((r2 + dr) * grid_size + (c2 + dc))

            # Swap
            idx1 = np.array(idx1)
            idx2 = np.array(idx2)
            temp = perturbed[img_idx, idx1, :].copy()
            perturbed[img_idx, idx1, :] = perturbed[img_idx, idx2, :]
            perturbed[img_idx, idx2, :] = temp

    return perturbed


# =============================================================================
# PATCHCORE (NEAREST NEIGHBOR BASELINE)
# =============================================================================

def train_patchcore(train_features: np.ndarray, k: int = 1) -> NearestNeighbors:
    """Train PatchCore: build memory bank from all train patches."""
    # Flatten: (N_train * N_patches, D)
    memory = train_features.reshape(-1, train_features.shape[-1])
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(memory)
    return nn


def score_patchcore(test_features: np.ndarray, nn_model: NearestNeighbors,
                    agg: str = "max") -> np.ndarray:
    """Score test images with PatchCore."""
    n_test, n_patches, dim = test_features.shape
    flat = test_features.reshape(-1, dim)
    distances, _ = nn_model.kneighbors(flat)
    patch_scores = distances[:, 0].reshape(n_test, n_patches)

    if agg == "p95":
        return np.percentile(patch_scores, 95, axis=1)
    return patch_scores.max(axis=1)


# =============================================================================
# PC-REG
# =============================================================================

def build_neighbor_map(grid_size: int, radius: int) -> dict[int, list[int]]:
    neighbors = {}
    for py in range(grid_size):
        for px in range(grid_size):
            p = py * grid_size + px
            nbrs = []
            for qy in range(max(0, py - radius), min(grid_size, py + radius + 1)):
                for qx in range(max(0, px - radius), min(grid_size, px + radius + 1)):
                    if qy == py and qx == px:
                        continue
                    nbrs.append(qy * grid_size + qx)
            neighbors[p] = nbrs
    return neighbors


def train_pcreg(train_features, neighbor_map, lam):
    n_train, n_patches, dim = train_features.shape
    W, mu_r, Sigma_r_inv = {}, {}, {}

    for p in range(n_patches):
        nbrs = neighbor_map[p]
        X = train_features[:, nbrs, :].mean(axis=1)
        Y = train_features[:, p, :]

        XtX = X.T @ X
        XtY = X.T @ Y
        W_p = np.linalg.solve(XtX + lam * np.eye(dim), XtY)

        residuals = Y - X @ W_p
        mu = residuals.mean(axis=0)

        try:
            lw = LedoitWolf()
            lw.fit(residuals)
            Sigma_inv = lw.precision_
        except Exception:
            var = residuals.var(axis=0) + 1e-6
            Sigma_inv = np.diag(1.0 / var)

        W[p] = W_p
        mu_r[p] = mu
        Sigma_r_inv[p] = Sigma_inv

    return W, mu_r, Sigma_r_inv


def score_pcreg(test_features, neighbor_map, W, mu_r, Sigma_r_inv, agg="p95"):
    n_test, n_patches, dim = test_features.shape
    patch_scores = np.zeros((n_test, n_patches))

    for p in range(n_patches):
        nbrs = neighbor_map[p]
        X = test_features[:, nbrs, :].mean(axis=1)
        residuals = test_features[:, p, :] - X @ W[p]
        centered = residuals - mu_r[p]
        mahal_sq = np.sum(centered @ Sigma_r_inv[p] * centered, axis=1)
        patch_scores[:, p] = np.sqrt(np.maximum(0, mahal_sq))

    if agg == "p95":
        return np.percentile(patch_scores, 95, axis=1)
    return patch_scores.max(axis=1)


# =============================================================================
# METRICS AND I/O
# =============================================================================

FIELDNAMES = [
    "timestamp", "category", "method", "n_swaps",
    "auroc_detect_permuted", "mean_score_original", "mean_score_permuted",
    "std_score_original", "std_score_permuted",
    "n_original", "n_permuted", "time_s",
]


def get_completed():
    csv_path = OUTPUT_DIR / "results.csv"
    completed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                completed.add((row["category"], row["method"], int(row["n_swaps"])))
    return completed


def save_result(result):
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
    print("ACID TEST: Feature-Level Block Permutation")
    print(f"  Categories: {CATEGORIES}")
    print(f"  Block size: {BLOCK_SIZE}x{BLOCK_SIZE}")
    print(f"  Swap counts: [1, 2, 3]")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    completed = get_completed()
    neighbor_map = build_neighbor_map(GRID_SIZE, R)
    rng = np.random.default_rng(SEED)

    swap_counts = [1, 2, 3]

    # Reviewer-friendly behavior: if large caches are not present, but the repo
    # ships precomputed numbers, do not crash.
    precomputed_csv = OUTPUT_DIR / "results.csv"
    if not CACHE_DIR.is_dir():
        if precomputed_csv.is_file():
            print(
                "[pc-reg] Cached LOCO features not found; skipping computation.\n"
                f"Precomputed results are available at: {precomputed_csv}\n\n"
                "To reproduce from scratch:\n"
                "  uv run scripts/extract_features_loco.py\n"
                "  uv run scripts/run_permutation_test.py\n",
                flush=True,
            )
            return
        raise SystemExit(
            "[pc-reg] Missing cached features directory: features/loco\n\n"
            "Generate features first:\n"
            "  uv run scripts/extract_features_loco.py\n"
        )

    for category in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Category: {category}")

        # Load features
        cat_cache = CACHE_DIR / category
        if not (cat_cache / "train_features.npy").is_file() or not (cat_cache / "test_good_features.npy").is_file():
            if precomputed_csv.is_file():
                print(
                    f"[pc-reg] Cache missing for '{category}'; skipping. Precomputed results: {precomputed_csv}",
                    flush=True,
                )
                continue
            raise SystemExit(
                f"[pc-reg] Missing cache for '{category}'.\n\n"
                "Run: uv run scripts/extract_features_loco.py\n"
            )
        train_feat = np.load(str(cat_cache / "train_features.npy"))
        test_good_feat = np.load(str(cat_cache / "test_good_features.npy"))

        print(f"  train: {train_feat.shape}")
        print(f"  test_good: {test_good_feat.shape}")

        # PCA
        n_train, n_patches, feat_dim = train_feat.shape
        pca = PCA(n_components=PCA_DIM, random_state=42)
        pca.fit(train_feat.reshape(-1, feat_dim))

        train_pca = pca.transform(train_feat.reshape(-1, feat_dim)).reshape(
            n_train, n_patches, PCA_DIM)
        test_good_pca = pca.transform(test_good_feat.reshape(-1, feat_dim)).reshape(
            -1, n_patches, PCA_DIM)

        n_test = test_good_pca.shape[0]

        # Train both methods
        print("  Training PatchCore...")
        t0 = time.time()
        nn_model = train_patchcore(train_pca)
        print(f"    Done ({time.time()-t0:.1f}s)")

        print("  Training PC-Reg...")
        t0 = time.time()
        W, mu_r, Si = train_pcreg(train_pca, neighbor_map, LAMBDA)
        print(f"    Done ({time.time()-t0:.1f}s)")

        for n_sw in swap_counts:
            for method_name in ["patchcore", "pcreg"]:
                key = (category, method_name, n_sw)
                if key in completed:
                    print(f"  [SKIP] {method_name} swaps={n_sw}")
                    continue

                try:
                    t0 = time.time()

                    # Create permuted versions
                    cat_rng = np.random.default_rng(SEED + hash(category) % 1000 + n_sw)
                    permuted_pca = permute_blocks(
                        test_good_pca, GRID_SIZE, BLOCK_SIZE, n_sw, cat_rng)

                    # Score original and permuted
                    if method_name == "patchcore":
                        scores_orig = score_patchcore(test_good_pca, nn_model, AGG)
                        scores_perm = score_patchcore(permuted_pca, nn_model, AGG)
                    else:
                        scores_orig = score_pcreg(
                            test_good_pca, neighbor_map, W, mu_r, Si, AGG)
                        scores_perm = score_pcreg(
                            permuted_pca, neighbor_map, W, mu_r, Si, AGG)

                    # AUROC: can we distinguish original from permuted?
                    labels = np.concatenate([np.zeros(n_test), np.ones(n_test)])
                    scores = np.concatenate([scores_orig, scores_perm])
                    auroc = roc_auc_score(labels, scores)

                    elapsed = time.time() - t0

                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "category": category,
                        "method": method_name,
                        "n_swaps": n_sw,
                        "auroc_detect_permuted": f"{auroc:.4f}",
                        "mean_score_original": f"{scores_orig.mean():.4f}",
                        "mean_score_permuted": f"{scores_perm.mean():.4f}",
                        "std_score_original": f"{scores_orig.std():.4f}",
                        "std_score_permuted": f"{scores_perm.std():.4f}",
                        "n_original": n_test,
                        "n_permuted": n_test,
                        "time_s": f"{elapsed:.1f}",
                    }
                    save_result(result)

                    sep = scores_perm.mean() / (scores_orig.mean() + 1e-8)
                    print(f"  {method_name} swaps={n_sw}: AUROC={auroc:.3f} "
                          f"orig={scores_orig.mean():.2f}±{scores_orig.std():.2f} "
                          f"perm={scores_perm.mean():.2f}±{scores_perm.std():.2f} "
                          f"ratio={sep:.2f}x ({elapsed:.1f}s)")

                except Exception as e:
                    print(f"  {method_name} swaps={n_sw}: ERROR - {e}")
                    traceback.print_exc()

    print("\n" + "=" * 70)
    print("ACID TEST COMPLETE")
    print(f"Results: {OUTPUT_DIR / 'results.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
