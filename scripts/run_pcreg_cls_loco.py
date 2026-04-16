"""
PC-Reg with CLS-Attention Gating on MVTec LOCO AD
===================================================
Uses DINOv3 CLS-to-patch attention (layer -6) to weight neighbor
contributions in Ridge regression, instead of uniform averaging.

Phase 1: Extract and cache attention maps (GPU, one-time)
Phase 2: Run PC-Reg experiments (CPU)

Configs:
  uniform_R5:  Baseline PC-Reg (uniform neighbor mean)
  cls_perimg_R5: Per-image CLS-to-patch attention weighting

Usage:
  uv run scripts/run_pcreg_cls_loco.py
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT = repo_root_from(__file__)
FEATURE_CACHE = REPO_ROOT / "features" / "loco"
ATTN_CACHE = REPO_ROOT / "features" / "loco_attention"
OUTPUT_DIR = REPO_ROOT / "results" / "pcreg_cls_loco"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ATTN_CACHE.mkdir(parents=True, exist_ok=True)

RESOLUTION = 448
LAYER_IDX = -6
ATTN_BATCH_SIZE = 2  # Small batch for attention extraction (memory-safe)

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]

PCA_DIM = 256
GRID_SIZE = 28
N_PATCHES = GRID_SIZE * GRID_SIZE  # 784
LAMBDA = 1.0
AGG = "p95"

CONFIGS = {
    "uniform_R5": {"weighting": "uniform", "R": 5},
    "cls_perimg_R5": {"weighting": "cls_perimg", "R": 5},
}


# =============================================================================
# PHASE 1: ATTENTION EXTRACTION AND CACHING
# =============================================================================

def needs_attention_cache(category: str) -> bool:
    return not (ATTN_CACHE / category / "DONE").exists()


def extract_and_cache_attention(categories: list[str]) -> None:
    """Extract DINOv3 layer-6 attention maps and cache to disk (float16)."""
    cats_needed = [c for c in categories if needs_attention_cache(c)]
    if not cats_needed:
        print("[ATTN] All attention caches exist. Skipping extraction.")
        return

    print(f"[ATTN] Extracting attention for: {cats_needed}")

    import torch
    from PIL import Image
    from transformers import AutoModel
    from torchvision import transforms

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    print(f"[ATTN] Loading DINOv3 on {device} (eager attention)...")
    model = AutoModel.from_pretrained(
        model_name, attn_implementation="eager"
    ).to(device).eval()

    num_register = getattr(model.config, "num_register_tokens", 4)
    start_idx = 1 + num_register  # Skip CLS + registers

    transform = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    splits = ["train", "test_good", "test_logical", "test_structural"]

    for category in cats_needed:
        print(f"\n[ATTN] Category: {category}")
        cat_feat = FEATURE_CACHE / category
        cat_attn = ATTN_CACHE / category
        cat_attn.mkdir(parents=True, exist_ok=True)

        for split in splits:
            paths_file = cat_feat / f"{split}_paths.txt"
            if not paths_file.exists():
                print(f"  [WARN] {paths_file} not found, skipping")
                continue

            with open(paths_file, "r", encoding="utf-8") as f:
                image_paths = [line.strip() for line in f if line.strip()]

            n_images = len(image_paths)
            all_cls_attn = []
            t0 = time.time()

            for i in range(0, n_images, ATTN_BATCH_SIZE):
                batch_paths = image_paths[i:i + ATTN_BATCH_SIZE]
                images = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
                batch = torch.stack(images).to(device)

                with torch.no_grad():
                    # NOTE: no autocast — eager attention overflows in fp16
                    outputs = model(batch, output_attentions=True, output_hidden_states=False)

                # Layer -6 attention: (batch, heads, seq_len, seq_len)
                attn = outputs.attentions[LAYER_IDX]
                attn_avg = attn.float().mean(dim=1)  # (batch, seq_len, seq_len)

                # CLS-to-patch: (batch, 784)
                cls_attn = attn_avg[:, 0, start_idx:start_idx + N_PATCHES]

                all_cls_attn.append(cls_attn.cpu().numpy().astype(np.float16))

                del batch, outputs, attn, attn_avg, cls_attn
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            ca = np.concatenate(all_cls_attn, axis=0)
            np.save(str(cat_attn / f"{split}_cls_attn.npy"), ca)
            elapsed = time.time() - t0
            print(f"  {split}: cls={ca.shape}, {elapsed:.1f}s")

            del all_cls_attn, ca
            gc.collect()

        (cat_attn / "DONE").write_text(f"extracted {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
        print(f"  [OK] {category} attention cached")

    del model
    gc.collect()
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    print("[ATTN] Extraction complete. Model unloaded.\n")


# =============================================================================
# PHASE 2: DATA LOADING
# =============================================================================

def load_features(category: str) -> dict:
    cat = FEATURE_CACHE / category
    return {s: np.load(str(cat / f"{s}_features.npy"))
            for s in ["train", "test_good", "test_logical", "test_structural"]}


def load_attention(category: str) -> dict:
    cat = ATTN_CACHE / category
    data = {}
    for s in ["train", "test_good", "test_logical", "test_structural"]:
        data[f"{s}_cls"] = np.load(str(cat / f"{s}_cls_attn.npy")).astype(np.float32)
    return data


# =============================================================================
# PC-REG WITH ATTENTION GATING
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


def compute_context(features, p, nbrs, weighting, cls_attn=None):
    """Compute context vector(s) for position p across all images.

    Args:
        features: (N, 784, D)
        p: center position
        nbrs: neighbor indices
        weighting: "uniform" or "cls_perimg"
        cls_attn: (N, 784) CLS-to-patch attention weights

    Returns:
        X: (N, D)
    """
    nbr_feats = features[:, nbrs, :]  # (N, |nbrs|, D)

    if weighting == "uniform":
        return nbr_feats.mean(axis=1)

    elif weighting == "cls_perimg":
        w = cls_attn[:, nbrs]  # (N, |nbrs|)
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-8)
        return (w[:, :, None] * nbr_feats).sum(axis=1)

    else:
        raise ValueError(f"Unknown weighting: {weighting}")


def train_pcreg(train_features, neighbor_map, lam, weighting, cls_attn=None):
    """Train Ridge regression + residual distribution per position."""
    n_train, n_patches, dim = train_features.shape
    W, mu_r, Sigma_r_inv = {}, {}, {}

    for p in range(n_patches):
        nbrs = neighbor_map[p]
        X = compute_context(train_features, p, nbrs, weighting, cls_attn=cls_attn)
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


def score_pcreg(test_features, neighbor_map, W, mu_r, Sigma_r_inv,
                weighting, agg="p95", cls_attn=None):
    """Score test images."""
    n_test, n_patches, dim = test_features.shape
    patch_scores = np.zeros((n_test, n_patches))

    for p in range(n_patches):
        nbrs = neighbor_map[p]
        X = compute_context(test_features, p, nbrs, weighting, cls_attn=cls_attn)

        residuals = test_features[:, p, :] - X @ W[p]
        centered = residuals - mu_r[p]
        mahal_sq = np.sum(centered @ Sigma_r_inv[p] * centered, axis=1)
        patch_scores[:, p] = np.sqrt(np.maximum(0, mahal_sq))

    if agg == "p95":
        image_scores = np.percentile(patch_scores, 95, axis=1)
    else:
        image_scores = patch_scores.max(axis=1)
    return image_scores, patch_scores


# =============================================================================
# METRICS AND I/O
# =============================================================================

def compute_auroc_splits(scores_good, scores_logical, scores_structural):
    results = {}
    labels_c = np.concatenate([np.zeros(len(scores_good)),
                               np.ones(len(scores_logical)),
                               np.ones(len(scores_structural))])
    scores_c = np.concatenate([scores_good, scores_logical, scores_structural])
    results["auroc_combined"] = roc_auc_score(labels_c, scores_c)

    labels_l = np.concatenate([np.zeros(len(scores_good)), np.ones(len(scores_logical))])
    scores_l = np.concatenate([scores_good, scores_logical])
    results["auroc_logical"] = roc_auc_score(labels_l, scores_l)

    labels_s = np.concatenate([np.zeros(len(scores_good)), np.ones(len(scores_structural))])
    scores_s = np.concatenate([scores_good, scores_structural])
    results["auroc_structural"] = roc_auc_score(labels_s, scores_s)
    return results


FIELDNAMES = [
    "timestamp", "category", "config", "seed",
    "auroc_combined", "auroc_logical", "auroc_structural",
    "n_train", "n_test_good", "n_test_logical", "n_test_structural",
    "time_s",
]


def get_completed():
    csv_path = OUTPUT_DIR / "results.csv"
    completed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                completed.add((row["category"], row["config"], int(row["seed"])))
    return completed


def save_result(result):
    csv_path = OUTPUT_DIR / "results.csv"
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
        if not exists:
            w.writeheader()
        w.writerow(result)


def log_error(category, config, seed, error):
    with open(OUTPUT_DIR / "errors.log", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{datetime.now().isoformat()}] {category} | {config} | seed={seed}\n")
        f.write(traceback.format_exc())


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PC-Reg with CLS-Attention Gating on MVTec LOCO AD")
    print(f"  Categories: {CATEGORIES}")
    print(f"  Configs: {list(CONFIGS.keys())}")
    print(f"  PCA_DIM={PCA_DIM}, Lambda={LAMBDA}, Agg={AGG}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Reviewer-friendly guard: allow running without large caches/datasets.
    if skip_if_missing(
        required=[FEATURE_CACHE],
        precomputed=[OUTPUT_DIR / "results.csv"],
        what="cached LOCO features (features/loco)",
        reproduce_hint=(
            "  uv run scripts/extract_features_loco.py\n"
            "  uv run scripts/run_pcreg_cls_loco.py"
        ),
    ):
        return

    # Phase 1: Extract attention if needed
    extract_and_cache_attention(CATEGORIES)

    # Phase 2: Run experiments
    completed = get_completed()
    total = len(CATEGORIES) * len(CONFIGS)
    remaining = total - len(completed)
    print(f"[INFO] Completed: {len(completed)}, remaining: {remaining}")

    if remaining == 0:
        print("[INFO] All experiments done.")
        return

    # Precompute neighbor map
    R = 5
    neighbor_map = build_neighbor_map(GRID_SIZE, R)

    for category in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Category: {category}")

        # Load features
        try:
            feat = load_features(category)
        except FileNotFoundError as e:
            precomputed_csv = OUTPUT_DIR / "results.csv"
            if precomputed_csv.is_file():
                print(
                    f"[pc-reg] Missing cache for '{category}'; skipping. "
                    f"Precomputed results: {precomputed_csv}",
                    flush=True,
                )
                continue
            raise SystemExit(
                f"[pc-reg] Missing cached features for '{category}': {e}\n\n"
                "To reproduce from scratch:\n"
                "  uv run scripts/extract_features_loco.py\n"
                "  uv run scripts/run_pcreg_cls_loco.py\n"
            )
        train_f = feat["train"]
        tg_f = feat["test_good"]
        tl_f = feat["test_logical"]
        ts_f = feat["test_structural"]
        print(f"  Features: train={train_f.shape}, tg={tg_f.shape}, "
              f"tl={tl_f.shape}, ts={ts_f.shape}")

        # Load attention
        try:
            attn = load_attention(category)
        except FileNotFoundError as e:
            precomputed_csv = OUTPUT_DIR / "results.csv"
            if precomputed_csv.is_file():
                print(
                    f"[pc-reg] Missing cached attention for '{category}'; skipping. "
                    f"Precomputed results: {precomputed_csv}",
                    flush=True,
                )
                continue
            raise SystemExit(
                f"[pc-reg] Missing cached attention for '{category}': {e}\n\n"
                "To reproduce from scratch:\n"
                "  uv run scripts/extract_features_loco.py\n"
                "  uv run scripts/run_pcreg_cls_loco.py\n"
            )
        tr_ca = attn["train_cls"]
        tg_ca = attn["test_good_cls"]
        tl_ca = attn["test_logical_cls"]
        ts_ca = attn["test_structural_cls"]
        print(f"  Attention: cls={tr_ca.shape}")

        # PCA
        n_train, n_patches, feat_dim = train_f.shape
        pca = PCA(n_components=PCA_DIM, random_state=42)
        pca.fit(train_f.reshape(-1, feat_dim))
        var_exp = pca.explained_variance_ratio_.sum()

        train_pca = pca.transform(train_f.reshape(-1, feat_dim)).reshape(n_train, n_patches, PCA_DIM)
        tg_pca = pca.transform(tg_f.reshape(-1, feat_dim)).reshape(-1, n_patches, PCA_DIM)
        tl_pca = pca.transform(tl_f.reshape(-1, feat_dim)).reshape(-1, n_patches, PCA_DIM)
        ts_pca = pca.transform(ts_f.reshape(-1, feat_dim)).reshape(-1, n_patches, PCA_DIM)
        print(f"  PCA: {feat_dim} -> {PCA_DIM} (var={var_exp:.3f})")

        del feat, train_f, tg_f, tl_f, ts_f
        gc.collect()

        for cfg_name, params in CONFIGS.items():
            seed = 0
            if (category, cfg_name, seed) in completed:
                print(f"  [SKIP] {cfg_name}")
                continue

            try:
                weighting = params["weighting"]
                print(f"\n  Config: {cfg_name} (w={weighting}, R={R})")
                t0 = time.time()

                # Build train kwargs
                tr_kw = {}
                if weighting == "cls_perimg":
                    tr_kw = {"cls_attn": tr_ca}

                # Train
                W, mu_r, Si = train_pcreg(train_pca, neighbor_map, LAMBDA, weighting, **tr_kw)
                train_time = time.time() - t0
                print(f"    Train: {train_time:.1f}s")

                # Score helper
                def _score(test_pca, sc_attn):
                    kw = {}
                    if weighting == "cls_perimg":
                        kw = {"cls_attn": sc_attn}
                    return score_pcreg(test_pca, neighbor_map, W, mu_r, Si,
                                       weighting, agg=AGG, **kw)

                sg, _ = _score(tg_pca, tg_ca)
                sl, _ = _score(tl_pca, tl_ca)
                ss, _ = _score(ts_pca, ts_ca)

                metrics = compute_auroc_splits(sg, sl, ss)
                elapsed = time.time() - t0

                result = {
                    "timestamp": datetime.now().isoformat(),
                    "category": category,
                    "config": cfg_name,
                    "seed": seed,
                    "auroc_combined": f"{metrics['auroc_combined']:.4f}",
                    "auroc_logical": f"{metrics['auroc_logical']:.4f}",
                    "auroc_structural": f"{metrics['auroc_structural']:.4f}",
                    "n_train": n_train,
                    "n_test_good": len(sg),
                    "n_test_logical": len(sl),
                    "n_test_structural": len(ss),
                    "time_s": f"{elapsed:.1f}",
                }
                save_result(result)

                print(f"    >> log={metrics['auroc_logical']:.3f} "
                      f"str={metrics['auroc_structural']:.3f} "
                      f"comb={metrics['auroc_combined']:.3f} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"    ERROR: {e}")
                log_error(category, cfg_name, seed, e)

        # Free memory
        del attn, tr_ca, tg_ca, tl_ca, ts_ca
        del train_pca, tg_pca, tl_pca, ts_pca
        gc.collect()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results: {OUTPUT_DIR / 'results.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
