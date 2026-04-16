"""
Pixel-Level Metrics (pixel-AUROC, AUPRO)
=========================================
Computes pixel-level anomaly detection metrics using the 28x28 patch
score maps from PC-Reg, upsampled to original image resolution.

Datasets:
  - MVTec AD (15 categories): all defect types
  - VisA (12 categories): all anomalies
  - LOCO (5 categories): structural only (logical masks are ambiguous)

Metrics:
  - pixel-AUROC: standard pixel-level AUROC
  - AUPRO: Area Under Per-Region Overlap (integration limit FPR=0.3)

Usage:
  uv run scripts/run_pixel_metrics.py            # all categories
  uv run scripts/run_pixel_metrics.py --mvtec    # MVTec AD only
  uv run scripts/run_pixel_metrics.py --visa     # VisA only
  uv run scripts/run_pixel_metrics.py --loco     # LOCO only
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import sys
import csv
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path

from _repro_utils import repo_root_from, skip_if_missing

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from skimage.measure import label as label_connected_components

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT = repo_root_from(__file__)
OUTPUT_DIR = REPO_ROOT / "results" / "pixel_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SIZE = 28
N_PATCHES = GRID_SIZE * GRID_SIZE
PCA_DIM = 256
LAMBDA = 1.0
RADIUS = 5

# Dataset paths
LOCO_DATA = REPO_ROOT / "data" / "mvtec_loco_AD"
MVTEC_DATA = REPO_ROOT / "data" / "mvtec_AD"
VISA_DATA = REPO_ROOT / "data" / "VisA"

# Feature caches
LOCO_CACHE = REPO_ROOT / "features" / "loco"
MVTEC_CACHE = REPO_ROOT / "features" / "mvtec"
VISA_CACHE = REPO_ROOT / "features" / "visa"

# Integration limit for AUPRO (standard is 0.3)
AUPRO_FPR_LIMIT = 0.3

# Evaluation resolution for pixel metrics (resize both score maps and masks)
# 256x256 is standard in literature (PatchCore, PaDiM, EfficientAD)
EVAL_RESOLUTION = 256

# Categories
LOCO_CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
MVTEC_CATS = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
VISA_CATS = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


# =============================================================================
# PC-REG ENGINE
# =============================================================================

def compute_neighbor_means(features, grid_size, radius):
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


def train_pcreg(train_pca, radius):
    n_train, n_patches, dim = train_pca.shape
    train_nbr_means = compute_neighbor_means(train_pca, GRID_SIZE, radius)

    I = np.eye(dim, dtype=np.float64)
    W_arr = np.zeros((n_patches, dim, dim), dtype=np.float64)
    mu_arr = np.zeros((n_patches, dim), dtype=np.float64)
    Sigma_inv_arr = np.zeros((n_patches, dim, dim), dtype=np.float64)

    for p in range(n_patches):
        X = train_nbr_means[:, p, :].astype(np.float64)
        Y = train_pca[:, p, :].astype(np.float64)
        XtX = X.T @ X
        XtY = X.T @ Y
        W_p = np.linalg.solve(XtX + LAMBDA * I, XtY)
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

    return W_arr, mu_arr, Sigma_inv_arr


def compute_patch_scores(test_pca, W_arr, mu_arr, Sigma_inv_arr, radius):
    """Returns (N, 784) Mahalanobis distance scores."""
    test_nbr_means = compute_neighbor_means(test_pca, GRID_SIZE, radius)
    Y_pred = np.einsum('npd,pde->npe',
                       test_nbr_means.astype(np.float64), W_arr)
    centered = (test_pca.astype(np.float64) - Y_pred) - mu_arr[np.newaxis, :, :]
    temp = np.einsum('npd,pde->npe', centered, Sigma_inv_arr)
    mahal_sq = np.sum(temp * centered, axis=2)
    return np.sqrt(np.maximum(0, mahal_sq))


# =============================================================================
# MASK LOADING
# =============================================================================

def load_mvtec_ad_mask(img_path_str):
    """Load MVTec AD ground truth mask from image path.

    Image: data/mvtec_AD/{cat}/test/{defect}/{NNN}.png
    Mask:  data/mvtec_AD/{cat}/ground_truth/{defect}/{NNN}_mask.png
    Values: 0 or 255 -> binarize to 0/1.
    """
    img_path = Path(img_path_str)
    cat_dir = img_path.parent.parent.parent  # data/mvtec_AD/{cat}
    defect_type = img_path.parent.name
    stem = img_path.stem
    mask_path = cat_dir / "ground_truth" / defect_type / f"{stem}_mask.png"

    if not mask_path.exists():
        return None

    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 0).astype(np.uint8)


def load_visa_mask(img_path_str):
    """Load VisA ground truth mask from image path.

    Image: data/VisA/{cat}/Data/Images/Anomaly/{NNN}.JPG
    Mask:  data/VisA/{cat}/Data/Masks/Anomaly/{NNN}.png
    Values: 0 or 1.
    """
    img_path = Path(img_path_str)
    stem = img_path.stem
    cat_dir = img_path.parent.parent.parent.parent  # data/VisA/{cat}
    mask_path = cat_dir / "Data" / "Masks" / "Anomaly" / f"{stem}.png"

    if not mask_path.exists():
        return None

    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 0).astype(np.uint8)


def load_loco_mask(img_path_str, anomaly_type):
    """Load LOCO ground truth mask from image path.

    Image: data/mvtec_loco_AD/{cat}/test/{type}/{NNN}.png
    Masks: data/mvtec_loco_AD/{cat}/ground_truth/{type}/{NNN}/*.png
    Values: 0-255 (variable per category) -> binarize to 0/1.
    Multiple masks per image: merge with np.maximum.
    """
    img_path = Path(img_path_str)
    cat_dir = img_path.parent.parent.parent  # data/mvtec_loco_AD/{cat}
    stem = img_path.stem
    gt_subdir = cat_dir / "ground_truth" / anomaly_type / stem

    if not gt_subdir.exists():
        return None

    mask_files = sorted(gt_subdir.glob("*.png"))
    if not mask_files:
        return None

    combined = None
    for mf in mask_files:
        m = np.array(Image.open(mf).convert("L"))
        if combined is None:
            combined = m
        else:
            combined = np.maximum(combined, m)

    return (combined > 0).astype(np.uint8)


# =============================================================================
# PIXEL METRICS
# =============================================================================

def resize_to_eval(arr, is_mask=False):
    """Resize array to EVAL_RESOLUTION x EVAL_RESOLUTION.

    For score maps: bilinear interpolation (order=1).
    For masks: nearest-neighbor (order=0) to preserve binary values.
    """
    h, w = arr.shape
    if h == EVAL_RESOLUTION and w == EVAL_RESOLUTION:
        return arr
    return zoom(arr, (EVAL_RESOLUTION / h, EVAL_RESOLUTION / w),
                order=0 if is_mask else 1)


def compute_pixel_auroc(score_maps, gt_masks):
    """Compute pixel-level AUROC across all images.

    All arrays should already be at EVAL_RESOLUTION.
    """
    all_scores = np.concatenate([s.ravel() for s in score_maps])
    all_labels = np.concatenate([m.ravel() for m in gt_masks])

    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        return float('nan')

    return roc_auc_score(all_labels, all_scores)


def compute_aupro(score_maps, gt_masks, fpr_limit=0.3, n_thresholds=200):
    """Compute AUPRO (Area Under Per-Region Overlap).

    Optimized: all maps at EVAL_RESOLUTION, precomputed components,
    vectorized FPR computation.
    """
    all_scores = np.concatenate([s.ravel() for s in score_maps])
    all_gt = np.concatenate([m.ravel() for m in gt_masks])

    if all_gt.sum() == 0:
        return float('nan')

    total_normal = int((all_gt == 0).sum())
    if total_normal == 0:
        return float('nan')

    # Thresholds from score percentiles
    thresholds = np.percentile(all_scores, np.linspace(0, 100, n_thresholds))
    thresholds = np.unique(thresholds)[::-1]

    # Precompute connected components
    comp_data = []  # list of (labeled_array, n_components, image_index)
    for i, gt in enumerate(gt_masks):
        if gt.sum() > 0:
            labeled, n_labels = label_connected_components(gt, return_num=True)
            comp_data.append((labeled, n_labels, i))

    # Precompute region masks for each component (as flat boolean arrays)
    regions = []  # list of (flat_region_mask, region_size, image_idx)
    for labeled, n_labels, img_idx in comp_data:
        flat_labeled = labeled.ravel()
        for lid in range(1, n_labels + 1):
            region_flat = (flat_labeled == lid)
            regions.append((region_flat, int(region_flat.sum()), img_idx))

    # Stack all score maps and gt masks as flat arrays for vectorized ops
    n_imgs = len(score_maps)
    R = EVAL_RESOLUTION
    scores_stack = np.stack([s.ravel() for s in score_maps])  # (n_imgs, R*R)
    gt_stack = np.stack([m.ravel() for m in gt_masks])        # (n_imgs, R*R)
    normal_mask_stack = (gt_stack == 0)  # (n_imgs, R*R)

    fprs = []
    pro_values = []

    for thresh in thresholds:
        pred_stack = (scores_stack >= thresh)  # (n_imgs, R*R) bool

        # FPR: sum of FP across all images
        fp_total = int(np.sum(pred_stack & normal_mask_stack))
        fpr = fp_total / total_normal

        # PRO: mean per-region overlap
        if regions:
            overlaps = []
            for region_flat, region_size, img_idx in regions:
                overlap = np.sum(pred_stack[img_idx] & region_flat) / region_size
                overlaps.append(overlap)
            pro = np.mean(overlaps)
        else:
            pro = 0.0

        fprs.append(fpr)
        pro_values.append(pro)

    fprs = np.array(fprs)
    pro_values = np.array(pro_values)

    # Sort by FPR
    sort_idx = np.argsort(fprs)
    fprs = fprs[sort_idx]
    pro_values = pro_values[sort_idx]

    # Remove duplicate FPR values
    unique_mask = np.append(np.diff(fprs) > 0, True)
    fprs = fprs[unique_mask]
    pro_values = pro_values[unique_mask]

    # Integrate up to fpr_limit
    valid = fprs <= fpr_limit
    if valid.sum() < 2:
        return float('nan')

    fprs_valid = fprs[valid]
    pro_valid = pro_values[valid]

    # Add boundary point at fpr_limit
    if fprs_valid[-1] < fpr_limit:
        beyond = np.where(fprs > fpr_limit)[0]
        if len(beyond) > 0:
            idx_b = beyond[0]
            t = (fpr_limit - fprs[idx_b - 1]) / (fprs[idx_b] - fprs[idx_b - 1] + 1e-10)
            pro_at_limit = pro_values[idx_b - 1] + t * (pro_values[idx_b] - pro_values[idx_b - 1])
            fprs_valid = np.append(fprs_valid, fpr_limit)
            pro_valid = np.append(pro_valid, pro_at_limit)

    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    aupro = _trapz(pro_valid, fprs_valid) / fpr_limit

    return float(aupro)


# =============================================================================
# DATASET EVALUATION FUNCTIONS
# =============================================================================

def evaluate_mvtec_ad(category):
    """Evaluate pixel metrics for one MVTec AD category."""
    cache_dir = MVTEC_CACHE / category

    # Load features
    train_feat = np.load(str(cache_dir / "train_features.npy"))
    test_good_feat = np.load(str(cache_dir / "test_good_features.npy"))
    test_anom_feat = np.load(str(cache_dir / "test_anomaly_features.npy"))

    # Load paths
    test_anom_paths = (cache_dir / "test_anomaly_paths.txt").read_text(encoding="utf-8").strip().split("\n")

    n_train = train_feat.shape[0]
    n_good = test_good_feat.shape[0]
    n_anom = test_anom_feat.shape[0]
    _, _, D = train_feat.shape

    # PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(train_feat.reshape(-1, D))

    all_feats = np.concatenate([
        train_feat.reshape(-1, D),
        test_good_feat.reshape(-1, D),
        test_anom_feat.reshape(-1, D),
    ])
    all_pca = pca.transform(all_feats).astype(np.float32)

    idx = 0
    train_pca = all_pca[idx:idx + n_train * N_PATCHES].reshape(n_train, N_PATCHES, PCA_DIM)
    idx += n_train * N_PATCHES
    good_pca = all_pca[idx:idx + n_good * N_PATCHES].reshape(n_good, N_PATCHES, PCA_DIM)
    idx += n_good * N_PATCHES
    anom_pca = all_pca[idx:idx + n_anom * N_PATCHES].reshape(n_anom, N_PATCHES, PCA_DIM)

    del all_feats, all_pca, train_feat, test_good_feat, test_anom_feat

    # Train PC-Reg
    W, mu, Si = train_pcreg(train_pca, RADIUS)

    # Compute patch-level scores
    ps_good = compute_patch_scores(good_pca, W, mu, Si, RADIUS)  # (n_good, 784)
    ps_anom = compute_patch_scores(anom_pca, W, mu, Si, RADIUS)  # (n_anom, 784)

    # Image-level AUROC (sanity check)
    s_good = np.percentile(ps_good, 95, axis=1)
    s_anom = np.percentile(ps_anom, 95, axis=1)
    labels_img = np.concatenate([np.zeros(n_good), np.ones(n_anom)])
    scores_img = np.concatenate([s_good, s_anom])
    auroc_img = roc_auc_score(labels_img, scores_img)

    # Pixel-level evaluation
    # Load masks and upsample score maps for anomaly images only
    score_maps = []
    gt_masks = []
    skipped = 0

    for i, path_str in enumerate(test_anom_paths):
        mask = load_mvtec_ad_mask(path_str)
        if mask is None:
            skipped += 1
            continue

        score_2d = ps_anom[i].reshape(GRID_SIZE, GRID_SIZE)
        score_up = resize_to_eval(score_2d, is_mask=False)
        mask_resized = resize_to_eval(mask, is_mask=True)

        score_maps.append(score_up)
        gt_masks.append(mask_resized)

    # Also include good images (all-zero masks) for FPR computation
    R = EVAL_RESOLUTION
    for i in range(n_good):
        score_2d = ps_good[i].reshape(GRID_SIZE, GRID_SIZE)
        score_up = resize_to_eval(score_2d, is_mask=False)
        score_maps.append(score_up)
        gt_masks.append(np.zeros((R, R), dtype=np.uint8))

    if len(score_maps) == 0:
        return {"auroc_img": auroc_img, "pixel_auroc": float('nan'), "aupro": float('nan')}

    pixel_auroc = compute_pixel_auroc(score_maps, gt_masks)
    aupro = compute_aupro(score_maps, gt_masks, fpr_limit=AUPRO_FPR_LIMIT)

    return {"auroc_img": auroc_img, "pixel_auroc": pixel_auroc, "aupro": aupro}


def evaluate_visa(category):
    """Evaluate pixel metrics for one VisA category."""
    cache_dir = VISA_CACHE / category

    # Load features
    train_feat = np.load(str(cache_dir / "train_features.npy"))
    test_good_feat = np.load(str(cache_dir / "test_good_features.npy"))
    test_anom_feat = np.load(str(cache_dir / "test_anomaly_features.npy"))

    # Load paths for mask mapping
    test_anom_paths = (cache_dir / "test_anomaly_paths.txt").read_text(encoding="utf-8").strip().split("\n")

    n_train = train_feat.shape[0]
    n_good = test_good_feat.shape[0]
    n_anom = test_anom_feat.shape[0]
    _, _, D = train_feat.shape

    # PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(train_feat.reshape(-1, D))

    all_feats = np.concatenate([
        train_feat.reshape(-1, D),
        test_good_feat.reshape(-1, D),
        test_anom_feat.reshape(-1, D),
    ])
    all_pca = pca.transform(all_feats).astype(np.float32)

    idx = 0
    train_pca = all_pca[idx:idx + n_train * N_PATCHES].reshape(n_train, N_PATCHES, PCA_DIM)
    idx += n_train * N_PATCHES
    good_pca = all_pca[idx:idx + n_good * N_PATCHES].reshape(n_good, N_PATCHES, PCA_DIM)
    idx += n_good * N_PATCHES
    anom_pca = all_pca[idx:idx + n_anom * N_PATCHES].reshape(n_anom, N_PATCHES, PCA_DIM)

    del all_feats, all_pca, train_feat, test_good_feat, test_anom_feat

    # Train PC-Reg
    W, mu, Si = train_pcreg(train_pca, RADIUS)

    # Compute patch-level scores
    ps_good = compute_patch_scores(good_pca, W, mu, Si, RADIUS)
    ps_anom = compute_patch_scores(anom_pca, W, mu, Si, RADIUS)

    # Image-level AUROC
    s_good = np.percentile(ps_good, 95, axis=1)
    s_anom = np.percentile(ps_anom, 95, axis=1)
    auroc_img = roc_auc_score(
        np.concatenate([np.zeros(n_good), np.ones(n_anom)]),
        np.concatenate([s_good, s_anom])
    )

    # Pixel-level
    score_maps = []
    gt_masks = []
    skipped = 0

    for i, path_str in enumerate(test_anom_paths):
        mask = load_visa_mask(path_str)
        if mask is None:
            skipped += 1
            continue

        score_2d = ps_anom[i].reshape(GRID_SIZE, GRID_SIZE)
        score_up = resize_to_eval(score_2d, is_mask=False)
        mask_resized = resize_to_eval(mask, is_mask=True)

        score_maps.append(score_up)
        gt_masks.append(mask_resized)

    # Add good images with zero masks
    R = EVAL_RESOLUTION
    for i in range(n_good):
        score_2d = ps_good[i].reshape(GRID_SIZE, GRID_SIZE)
        score_up = resize_to_eval(score_2d, is_mask=False)
        score_maps.append(score_up)
        gt_masks.append(np.zeros((R, R), dtype=np.uint8))

    if len(score_maps) == 0:
        return {"auroc_img": auroc_img, "pixel_auroc": float('nan'), "aupro": float('nan')}

    pixel_auroc = compute_pixel_auroc(score_maps, gt_masks)
    aupro = compute_aupro(score_maps, gt_masks, fpr_limit=AUPRO_FPR_LIMIT)

    return {"auroc_img": auroc_img, "pixel_auroc": pixel_auroc, "aupro": aupro}


def evaluate_loco_structural(category):
    """Evaluate pixel metrics for LOCO structural anomalies only."""
    cache_dir = LOCO_CACHE / category

    # Load features
    train_feat = np.load(str(cache_dir / "train_features.npy"))
    test_good_feat = np.load(str(cache_dir / "test_good_features.npy"))
    test_str_feat = np.load(str(cache_dir / "test_structural_features.npy"))

    # Load paths
    test_str_paths = (cache_dir / "test_structural_paths.txt").read_text(encoding="utf-8").strip().split("\n")

    n_train = train_feat.shape[0]
    n_good = test_good_feat.shape[0]
    n_str = test_str_feat.shape[0]
    _, _, D = train_feat.shape

    # PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(train_feat.reshape(-1, D))

    all_feats = np.concatenate([
        train_feat.reshape(-1, D),
        test_good_feat.reshape(-1, D),
        test_str_feat.reshape(-1, D),
    ])
    all_pca = pca.transform(all_feats).astype(np.float32)

    idx = 0
    train_pca = all_pca[idx:idx + n_train * N_PATCHES].reshape(n_train, N_PATCHES, PCA_DIM)
    idx += n_train * N_PATCHES
    good_pca = all_pca[idx:idx + n_good * N_PATCHES].reshape(n_good, N_PATCHES, PCA_DIM)
    idx += n_good * N_PATCHES
    str_pca = all_pca[idx:idx + n_str * N_PATCHES].reshape(n_str, N_PATCHES, PCA_DIM)

    del all_feats, all_pca, train_feat, test_good_feat, test_str_feat

    # Train PC-Reg
    W, mu, Si = train_pcreg(train_pca, RADIUS)

    # Compute patch-level scores
    ps_good = compute_patch_scores(good_pca, W, mu, Si, RADIUS)
    ps_str = compute_patch_scores(str_pca, W, mu, Si, RADIUS)

    # Image-level AUROC (structural only)
    s_good = np.percentile(ps_good, 95, axis=1)
    s_str = np.percentile(ps_str, 95, axis=1)
    auroc_img = roc_auc_score(
        np.concatenate([np.zeros(n_good), np.ones(n_str)]),
        np.concatenate([s_good, s_str])
    )

    # Pixel-level
    score_maps = []
    gt_masks = []
    skipped = 0

    for i, path_str in enumerate(test_str_paths):
        mask = load_loco_mask(path_str, "structural_anomalies")
        if mask is None:
            skipped += 1
            continue

        score_2d = ps_str[i].reshape(GRID_SIZE, GRID_SIZE)
        score_up = resize_to_eval(score_2d, is_mask=False)
        mask_resized = resize_to_eval(mask, is_mask=True)

        score_maps.append(score_up)
        gt_masks.append(mask_resized)

    # Add good images
    R = EVAL_RESOLUTION
    for i in range(n_good):
        score_2d = ps_good[i].reshape(GRID_SIZE, GRID_SIZE)
        score_up = resize_to_eval(score_2d, is_mask=False)
        score_maps.append(score_up)
        gt_masks.append(np.zeros((R, R), dtype=np.uint8))

    if len(score_maps) == 0:
        return {"auroc_img": auroc_img, "pixel_auroc": float('nan'), "aupro": float('nan')}

    pixel_auroc = compute_pixel_auroc(score_maps, gt_masks)
    aupro = compute_aupro(score_maps, gt_masks, fpr_limit=AUPRO_FPR_LIMIT)

    return {"auroc_img": auroc_img, "pixel_auroc": pixel_auroc, "aupro": aupro}


# =============================================================================
# I/O
# =============================================================================

FIELDNAMES = [
    "timestamp", "dataset", "category",
    "auroc_image", "pixel_auroc", "aupro",
]

def save_result(result):
    csv_path = OUTPUT_DIR / "results.csv"
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
        if not exists:
            w.writeheader()
        w.writerow(result)


def get_completed():
    csv_path = OUTPUT_DIR / "results.csv"
    completed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                completed.add((row["dataset"], row["category"]))
    return completed


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Determine which datasets to run
    run_mvtec = "--mvtec" in sys.argv or (not any(x in sys.argv for x in ["--mvtec", "--visa", "--loco"]))
    run_visa = "--visa" in sys.argv or (not any(x in sys.argv for x in ["--mvtec", "--visa", "--loco"]))
    run_loco = "--loco" in sys.argv or (not any(x in sys.argv for x in ["--mvtec", "--visa", "--loco"]))

    tasks = []
    if run_mvtec:
        for c in MVTEC_CATS:
            tasks.append(("MVTec_AD", c))
    if run_visa:
        for c in VISA_CATS:
            tasks.append(("VisA", c))
    if run_loco:
        for c in LOCO_CATS:
            tasks.append(("LOCO_str", c))

    print("=" * 70)
    print("Pixel-Level Metrics (pixel-AUROC, AUPRO)")
    print(f"  Tasks: {len(tasks)} categories")
    print(f"  AUPRO FPR limit: {AUPRO_FPR_LIMIT}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Reviewer-friendly guard: pixel metrics require BOTH cached features and datasets (masks).
    required = []
    if run_mvtec:
        required += [MVTEC_CACHE, MVTEC_DATA]
    if run_visa:
        required += [VISA_CACHE, VISA_DATA]
    if run_loco:
        required += [LOCO_CACHE, LOCO_DATA]

    if skip_if_missing(
        required=required,
        precomputed=[OUTPUT_DIR / "results.csv"],
        what="pixel-metrics prerequisites (features + data)",
        reproduce_hint=(
            "  # 1) Place datasets under pc-reg/data\n"
            "  # 2) Cache features (example):\n"
            "  uv run scripts/extract_features_mvtec.py\n"
            "  uv run scripts/extract_features_visa.py\n"
            "  uv run scripts/extract_features_loco.py\n"
            "  # 3) Run pixel metrics:\n"
            "  uv run scripts/run_pixel_metrics.py"
        ),
    ):
        return

    completed = get_completed()
    print(f"[INFO] Already completed: {len(completed)} categories")

    for dataset, category in tasks:
        key = (dataset, category)
        if key in completed:
            print(f"\n[SKIP] {dataset}/{category}: already done")
            continue

        print(f"\n{'='*60}")
        print(f"  {dataset} / {category}")
        t0 = time.time()

        try:
            if dataset == "MVTec_AD":
                result = evaluate_mvtec_ad(category)
            elif dataset == "VisA":
                result = evaluate_visa(category)
            elif dataset == "LOCO_str":
                result = evaluate_loco_structural(category)
            else:
                print(f"  [ERROR] Unknown dataset: {dataset}")
                continue

            elapsed = time.time() - t0

            row = {
                "timestamp": datetime.now().isoformat(),
                "dataset": dataset,
                "category": category,
                "auroc_image": f"{result['auroc_img']:.4f}",
                "pixel_auroc": f"{result['pixel_auroc']:.4f}",
                "aupro": f"{result['aupro']:.4f}",
            }
            save_result(row)

            print(f"  img-AUROC={result['auroc_img']:.4f}  "
                  f"pxl-AUROC={result['pixel_auroc']:.4f}  "
                  f"AUPRO={result['aupro']:.4f}  [{elapsed:.1f}s]")

        except FileNotFoundError as e:
            print(f"  [pc-reg] Missing artifact(s): {e}")
            continue
        except Exception as e:
            # Keep output reviewer-friendly: no stack traces by default.
            print(f"  [ERROR] {type(e).__name__}: {e}")
            continue

    # Summary
    csv_path = OUTPUT_DIR / "results.csv"
    if csv_path.exists():
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        rows = []
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                rows.append(row)

        for ds in ["MVTec_AD", "VisA", "LOCO_str"]:
            ds_rows = [r for r in rows if r["dataset"] == ds]
            if not ds_rows:
                continue

            img_aurocs = [float(r["auroc_image"]) for r in ds_rows]
            pxl_aurocs = [float(r["pixel_auroc"]) for r in ds_rows if r["pixel_auroc"] != "nan"]
            aupros = [float(r["aupro"]) for r in ds_rows if r["aupro"] != "nan"]

            print(f"\n  {ds} ({len(ds_rows)} cats):")
            print(f"    image-AUROC: {np.mean(img_aurocs)*100:.1f}%")
            if pxl_aurocs:
                print(f"    pixel-AUROC: {np.mean(pxl_aurocs)*100:.1f}%")
            if aupros:
                print(f"    AUPRO:       {np.mean(aupros)*100:.1f}%")


if __name__ == "__main__":
    main()
