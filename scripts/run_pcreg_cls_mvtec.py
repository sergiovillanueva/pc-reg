"""
PC-Reg with CLS-Attention Gating on MVTec AD
=============================================
Validates that CLS-attention gating (cls_perimg) generalizes to MVTec AD
(purely structural anomalies, no logical split).

Phase 1: Extract CLS attention for MVTec AD (GPU, one-time)
Phase 2: Run PC-Reg with uniform and cls_perimg (CPU)

Usage:
  uv run scripts/run_pcreg_cls_mvtec.py
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
FEATURE_CACHE = REPO_ROOT / "features" / "mvtec"
ATTN_CACHE = REPO_ROOT / "features" / "mvtec_attention"
OUTPUT_DIR = REPO_ROOT / "results" / "pcreg_cls_mvtec"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ATTN_CACHE.mkdir(parents=True, exist_ok=True)

DATA_ROOT = REPO_ROOT / "data" / "mvtec_AD"
RESOLUTION = 448
LAYER_IDX = -6
ATTN_BATCH_SIZE = 4

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

PCA_DIM = 256
GRID_SIZE = 28
N_PATCHES = GRID_SIZE * GRID_SIZE  # 784
LAMBDA = 1.0
RADIUS = 5
AGG = "p95"


# =============================================================================
# PHASE 1: CLS ATTENTION EXTRACTION
# =============================================================================

def get_image_paths(directory: Path) -> list[str]:
    return sorted(
        [str(p) for p in directory.rglob("*.png")] +
        [str(p) for p in directory.rglob("*.jpg")]
    )


def get_mvtec_paths(category: str) -> dict:
    cat_dir = DATA_ROOT / category
    test_dir = cat_dir / "test"
    anomaly_paths = []
    for subdir in sorted(test_dir.iterdir()):
        if subdir.is_dir() and subdir.name != "good":
            anomaly_paths.extend(get_image_paths(subdir))
    return {
        "train": get_image_paths(cat_dir / "train" / "good"),
        "test_good": get_image_paths(test_dir / "good"),
        "test_anomaly": anomaly_paths,
    }


def needs_cls_cache(category: str) -> bool:
    return not (ATTN_CACHE / category / "DONE").exists()


def extract_cls_attention(categories: list[str]) -> None:
    """Extract CLS-to-patch attention from DINOv3 layer -6."""
    cats_needed = [c for c in categories if needs_cls_cache(c)]
    if not cats_needed:
        print("[ATTN] All CLS attention caches exist. Skipping.")
        return

    # If data is not available, we cannot extract attention. In reviewer mode
    # we prefer pointing to precomputed results instead of crashing.
    if not DATA_ROOT.is_dir():
        precomputed_csv = OUTPUT_DIR / "results.csv"
        if precomputed_csv.is_file():
            print(
                "[pc-reg] Dataset not found; skipping CLS-attention extraction.\n"
                f"Precomputed results: {precomputed_csv}\n\n"
                "To reproduce from scratch, place datasets under pc-reg/data and run:\n"
                "  uv run scripts/extract_features_mvtec.py\n"
                "  uv run scripts/run_pcreg_cls_mvtec.py\n",
                flush=True,
            )
            return
        raise SystemExit(
            f"[pc-reg] Missing dataset directory: {DATA_ROOT}\n\n"
            "To reproduce from scratch, place datasets under pc-reg/data and run:\n"
            "  uv run scripts/extract_features_mvtec.py\n"
            "  uv run scripts/run_pcreg_cls_mvtec.py\n"
        )

    print(f"[ATTN] Extracting CLS attention for: {cats_needed}")

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
    start_idx = 1 + num_register

    transform = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for category in cats_needed:
        print(f"\n[ATTN] Category: {category}")
        cat_attn = ATTN_CACHE / category
        cat_attn.mkdir(parents=True, exist_ok=True)

        paths_dict = get_mvtec_paths(category)

        for split, image_paths in paths_dict.items():
            if not image_paths:
                print(f"  [WARN] {split}: 0 images, skipping")
                continue

            all_cls_attn = []
            t0 = time.time()

            for i in range(0, len(image_paths), ATTN_BATCH_SIZE):
                batch_paths = image_paths[i:i + ATTN_BATCH_SIZE]
                images = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
                batch = torch.stack(images).to(device)

                with torch.no_grad():
                    outputs = model(batch, output_attentions=True,
                                    output_hidden_states=False)

                attn = outputs.attentions[LAYER_IDX]
                attn_avg = attn.float().mean(dim=1)  # (batch, seq_len, seq_len)
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

        (cat_attn / "DONE").write_text(
            f"extracted {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8"
        )
        print(f"  [OK] {category} CLS attention cached")

    del model
    gc.collect()
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    print("[ATTN] Extraction complete.\n")


# =============================================================================
# PHASE 2: PC-REG WITH UNIFORM AND CLS-GATING
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
    nbr_feats = features[:, nbrs, :]  # (N, |nbrs|, D)
    if weighting == "uniform":
        return nbr_feats.mean(axis=1)
    elif weighting == "cls_perimg":
        w = cls_attn[:, nbrs]  # (N, |nbrs|)
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-8)
        return (w[:, :, None] * nbr_feats).sum(axis=1)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")


def train_pcreg(train_pca, neighbor_map, lam, weighting, cls_attn=None):
    n_train, n_patches, dim = train_pca.shape
    W, mu_r, Sigma_inv = {}, {}, {}

    for p in range(n_patches):
        nbrs = neighbor_map[p]
        X = compute_context(train_pca, p, nbrs, weighting, cls_attn=cls_attn)
        Y = train_pca[:, p, :]

        XtX = X.T @ X
        XtY = X.T @ Y
        W_p = np.linalg.solve(XtX + lam * np.eye(dim), XtY)

        residuals = Y - X @ W_p
        mu = residuals.mean(axis=0)

        try:
            lw = LedoitWolf()
            lw.fit(residuals)
            Si = lw.precision_
        except Exception:
            var = residuals.var(axis=0) + 1e-6
            Si = np.diag(1.0 / var)

        W[p] = W_p
        mu_r[p] = mu
        Sigma_inv[p] = Si

    return W, mu_r, Sigma_inv


def score_pcreg(test_pca, neighbor_map, W, mu_r, Sigma_inv,
                weighting, cls_attn=None):
    n_test, n_patches, dim = test_pca.shape
    patch_scores = np.zeros((n_test, n_patches))

    for p in range(n_patches):
        nbrs = neighbor_map[p]
        X = compute_context(test_pca, p, nbrs, weighting, cls_attn=cls_attn)
        residuals = test_pca[:, p, :] - X @ W[p]
        centered = residuals - mu_r[p]
        mahal_sq = np.sum(centered @ Sigma_inv[p] * centered, axis=1)
        patch_scores[:, p] = np.sqrt(np.maximum(0, mahal_sq))

    return np.percentile(patch_scores, 95, axis=1)


# =============================================================================
# METRICS AND I/O
# =============================================================================

FIELDNAMES = [
    "timestamp", "category", "config",
    "auroc", "n_train", "n_test_good", "n_test_anomaly", "time_s",
]


def get_completed():
    csv_path = OUTPUT_DIR / "results.csv"
    completed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                completed.add((row["category"], row["config"]))
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
    print("PC-Reg with CLS-Attention Gating on MVTec AD")
    print(f"  Categories: {len(CATEGORIES)}")
    print(f"  Configs: uniform_R5, cls_perimg_R5")
    print(f"  PCA_DIM={PCA_DIM}, Lambda={LAMBDA}, R={RADIUS}, Agg={AGG}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    if skip_if_missing(
        required=[FEATURE_CACHE],
        precomputed=[OUTPUT_DIR / "results.csv"],
        what="cached MVTec features (features/mvtec)",
        reproduce_hint=(
            "  uv run scripts/extract_features_mvtec.py\n"
            "  uv run scripts/run_pcreg_cls_mvtec.py"
        ),
    ):
        return

    # Phase 1: Extract CLS attention
    extract_cls_attention(CATEGORIES)

    # Phase 2: Run experiments
    completed = get_completed()
    neighbor_map = build_neighbor_map(GRID_SIZE, RADIUS)

    configs = ["uniform_R5", "cls_perimg_R5"]

    for category in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Category: {category}")

        # Load cached features
        feat_dir = FEATURE_CACHE / category
        try:
            train_f = np.load(str(feat_dir / "train_features.npy"))
            tg_f = np.load(str(feat_dir / "test_good_features.npy"))
            ta_f = np.load(str(feat_dir / "test_anomaly_features.npy"))
        except FileNotFoundError as e:
            precomputed_csv = OUTPUT_DIR / "results.csv"
            if precomputed_csv.is_file():
                print(
                    f"[pc-reg] Missing feature cache for '{category}'; skipping. "
                    f"Precomputed results: {precomputed_csv}",
                    flush=True,
                )
                continue
            raise SystemExit(
                f"[pc-reg] Missing cached features for '{category}': {e}\n\n"
                "To reproduce from scratch:\n"
                "  uv run scripts/extract_features_mvtec.py\n"
                "  uv run scripts/run_pcreg_cls_mvtec.py\n"
            )

        # Load CLS attention
        attn_dir = ATTN_CACHE / category
        try:
            tr_cls = np.load(str(attn_dir / "train_cls_attn.npy")).astype(np.float32)
            tg_cls = np.load(str(attn_dir / "test_good_cls_attn.npy")).astype(np.float32)
            ta_cls = np.load(str(attn_dir / "test_anomaly_cls_attn.npy")).astype(np.float32)
        except FileNotFoundError as e:
            precomputed_csv = OUTPUT_DIR / "results.csv"
            if precomputed_csv.is_file():
                print(
                    f"[pc-reg] Missing CLS-attention cache for '{category}'; skipping. "
                    f"Precomputed results: {precomputed_csv}",
                    flush=True,
                )
                continue
            raise SystemExit(
                f"[pc-reg] Missing cached attention for '{category}': {e}\n\n"
                "To reproduce from scratch:\n"
                "  uv run scripts/extract_features_mvtec.py\n"
                "  uv run scripts/run_pcreg_cls_mvtec.py\n"
            )

        print(f"  Features: train={train_f.shape}, tg={tg_f.shape}, ta={ta_f.shape}")
        print(f"  CLS attn: train={tr_cls.shape}")

        # PCA
        n_train, n_patches, feat_dim = train_f.shape
        pca = PCA(n_components=PCA_DIM, random_state=42)
        pca.fit(train_f.reshape(-1, feat_dim))

        train_pca = pca.transform(train_f.reshape(-1, feat_dim)).reshape(
            n_train, n_patches, PCA_DIM
        )
        tg_pca = pca.transform(tg_f.reshape(-1, feat_dim)).reshape(
            -1, n_patches, PCA_DIM
        )
        ta_pca = pca.transform(ta_f.reshape(-1, feat_dim)).reshape(
            -1, n_patches, PCA_DIM
        )

        del train_f, tg_f, ta_f
        gc.collect()

        for cfg in configs:
            if (category, cfg) in completed:
                print(f"  [SKIP] {cfg}")
                continue

            weighting = "uniform" if "uniform" in cfg else "cls_perimg"
            t0 = time.time()

            try:
                tr_kw = {"cls_attn": tr_cls} if weighting == "cls_perimg" else {}
                W, mu, Si = train_pcreg(train_pca, neighbor_map, LAMBDA,
                                        weighting, **tr_kw)

                tg_kw = {"cls_attn": tg_cls} if weighting == "cls_perimg" else {}
                ta_kw = {"cls_attn": ta_cls} if weighting == "cls_perimg" else {}

                scores_good = score_pcreg(tg_pca, neighbor_map, W, mu, Si,
                                          weighting, **tg_kw)
                scores_anom = score_pcreg(ta_pca, neighbor_map, W, mu, Si,
                                          weighting, **ta_kw)

                labels = np.concatenate([
                    np.zeros(len(scores_good)),
                    np.ones(len(scores_anom)),
                ])
                scores = np.concatenate([scores_good, scores_anom])
                auroc = roc_auc_score(labels, scores)
                elapsed = time.time() - t0

                result = {
                    "timestamp": datetime.now().isoformat(),
                    "category": category,
                    "config": cfg,
                    "auroc": f"{auroc:.4f}",
                    "n_train": n_train,
                    "n_test_good": len(scores_good),
                    "n_test_anomaly": len(scores_anom),
                    "time_s": f"{elapsed:.1f}",
                }
                save_result(result)

                print(f"  {cfg}: AUROC={auroc:.4f} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  {cfg}: ERROR - {e}")
                traceback.print_exc()

        del train_pca, tg_pca, ta_pca, tr_cls, tg_cls, ta_cls
        gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - MVTec AD (15 categories)")
    print(f"{'='*70}")

    csv_path = OUTPUT_DIR / "results.csv"
    if csv_path.exists():
        by_cfg = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                cfg = row["config"]
                if cfg not in by_cfg:
                    by_cfg[cfg] = {}
                by_cfg[cfg][row["category"]] = float(row["auroc"])

        print(f"\n{'Category':<18} {'uniform':>10} {'cls_perimg':>12} {'delta':>8}")
        print("-" * 50)
        for cat in CATEGORIES:
            u = by_cfg.get("uniform_R5", {}).get(cat, 0)
            c = by_cfg.get("cls_perimg_R5", {}).get(cat, 0)
            d = c - u
            sign = "+" if d >= 0 else ""
            print(f"{cat:<18} {u:>10.4f} {c:>12.4f} {sign}{d:>7.4f}")

        print("-" * 50)
        for cfg_name in ["uniform_R5", "cls_perimg_R5"]:
            vals = list(by_cfg.get(cfg_name, {}).values())
            if vals:
                mean = np.mean(vals)
                print(f"  Mean {cfg_name}: {mean:.4f}")

    print(f"\nResults: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
