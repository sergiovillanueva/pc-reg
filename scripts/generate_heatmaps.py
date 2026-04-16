"""
Generate PC-Reg heatmaps for paper figures.

Produces per-image anomaly heatmaps from PC-Reg patch scores (28x28),
upsampled to original image resolution, with overlay on original image.

Generates two variants:
  1. uniform_R5: baseline PC-Reg
  2. cls_perimg_R5: CLS-attention-gated PC-Reg (best method, +3.2pp)

Outputs:
  - Per-category grid figures with selected examples
  - Individual comparison images for paper figures

Usage:
  uv run scripts/generate_heatmaps.py
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from _repro_utils import repo_root_from

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT = repo_root_from(__file__)
CACHE_DIR = REPO_ROOT / "features" / "loco"
ATTN_CACHE_DIR = REPO_ROOT / "features" / "loco_attention"
DATA_ROOT = REPO_ROOT / "data" / "mvtec_loco_AD"
OUTPUT_DIR = REPO_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]

PCA_DIM = 256
GRID_SIZE = 28
N_PATCHES = GRID_SIZE * GRID_SIZE
R = 5
LAM = 1.0

# How many examples per type to include in grid figures
N_EXAMPLES = 4


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cached_features(category: str) -> dict:
    """Load pre-cached DINOv3 features."""
    cat_cache = CACHE_DIR / category
    data = {}
    for split in ["train", "test_good", "test_logical", "test_structural"]:
        data[split] = np.load(str(cat_cache / f"{split}_features.npy"))
    return data


def load_attention_cache(category: str) -> dict:
    """Load cached CLS attention maps."""
    cat_cache = ATTN_CACHE_DIR / category
    data = {}
    for split in ["train", "test_good", "test_logical", "test_structural"]:
        data[split] = np.load(str(cat_cache / f"{split}_cls_attn.npy")).astype(np.float32)
    return data


def get_image_paths(directory: Path) -> list[str]:
    """Get sorted image paths."""
    return sorted(
        [str(p) for p in directory.rglob("*.png")] +
        [str(p) for p in directory.rglob("*.jpg")]
    )


def load_image_paths(category: str) -> dict:
    """Load image paths for a category."""
    cat_dir = DATA_ROOT / category
    return {
        "train": get_image_paths(cat_dir / "train" / "good"),
        "test_good": get_image_paths(cat_dir / "test" / "good"),
        "test_logical": get_image_paths(cat_dir / "test" / "logical_anomalies"),
        "test_structural": get_image_paths(cat_dir / "test" / "structural_anomalies"),
    }


# =============================================================================
# PC-REG MODEL
# =============================================================================

def build_neighbor_map(grid_size: int, radius: int) -> dict[int, list[int]]:
    """Precompute spatial neighbors for each grid position."""
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


def compute_context(features, p, nbrs, cls_attn=None):
    """Compute context vector for position p (uniform or cls-weighted)."""
    nbr_feats = features[:, nbrs, :]  # (N, |nbrs|, D)
    if cls_attn is None:
        return nbr_feats.mean(axis=1)
    else:
        w = cls_attn[:, nbrs]  # (N, |nbrs|)
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-8)
        return (w[:, :, None] * nbr_feats).sum(axis=1)


def train_pcreg(train_features, neighbor_map, lam, cls_attn=None):
    """Train Ridge regression + residual distribution per position."""
    n_train, n_patches, dim = train_features.shape
    W, mu_r, Sigma_r_inv = {}, {}, {}

    for p in range(n_patches):
        nbrs = neighbor_map[p]
        X = compute_context(train_features, p, nbrs, cls_attn=cls_attn)
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
                cls_attn=None):
    """Score test images. Returns image_scores and patch_scores (28x28)."""
    n_test, n_patches, dim = test_features.shape
    patch_scores = np.zeros((n_test, n_patches))

    for p in range(n_patches):
        nbrs = neighbor_map[p]
        X = compute_context(test_features, p, nbrs, cls_attn=cls_attn)

        residuals = test_features[:, p, :] - X @ W[p]
        centered = residuals - mu_r[p]
        mahal_sq = np.sum(centered @ Sigma_r_inv[p] * centered, axis=1)
        patch_scores[:, p] = np.sqrt(np.maximum(0, mahal_sq))

    image_scores = np.percentile(patch_scores, 95, axis=1)
    return image_scores, patch_scores


# =============================================================================
# HEATMAP VISUALIZATION
# =============================================================================

def create_single_figure(orig_img, heatmap_up, gt_mask_path=None,
                         title="", vmin=0, vmax=None, save_path=None):
    """Create a single image: original | heatmap overlay | GT mask."""
    has_gt = gt_mask_path is not None and gt_mask_path.exists()
    ncols = 3 if has_gt else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    # Original
    axes[0].imshow(orig_img)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    # Heatmap overlay
    axes[1].imshow(orig_img)
    im = axes[1].imshow(heatmap_up, cmap="jet", alpha=0.5,
                        vmin=vmin, vmax=vmax)
    axes[1].set_title("PC-Reg Anomaly Map", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # GT mask
    if has_gt:
        gt = Image.open(gt_mask_path).convert("L")
        axes[2].imshow(gt, cmap="gray")
        axes[2].set_title("Ground Truth", fontsize=11)
        axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def create_comparison_figure(orig_img, heatmap_uniform, heatmap_cls,
                             gt_mask_path=None, title="",
                             vmin=0, vmax=None, save_path=None):
    """Create comparison: original | uniform heatmap | cls_perimg heatmap | GT."""
    has_gt = gt_mask_path is not None and gt_mask_path.exists()
    ncols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))

    # Original
    axes[0].imshow(orig_img)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    # Uniform heatmap
    axes[1].imshow(orig_img)
    im1 = axes[1].imshow(heatmap_uniform, cmap="jet", alpha=0.5,
                         vmin=vmin, vmax=vmax)
    axes[1].set_title("PC-Reg (uniform)", fontsize=10)
    axes[1].axis("off")

    # CLS-perimg heatmap
    axes[2].imshow(orig_img)
    im2 = axes[2].imshow(heatmap_cls, cmap="jet", alpha=0.5,
                         vmin=vmin, vmax=vmax)
    axes[2].set_title("PC-Reg (cls-gated)", fontsize=10)
    axes[2].axis("off")

    # GT mask
    if has_gt:
        gt = Image.open(gt_mask_path).convert("L")
        axes[3].imshow(gt, cmap="gray")
        axes[3].set_title("Ground Truth", fontsize=10)
        axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_grid_figure(examples: list[dict], category: str, save_path: Path):
    """Create a large grid figure for paper.

    Each row: [Original | Uniform Heatmap | CLS Heatmap | GT Mask]
    Rows grouped by: normal, logical, structural
    """
    n_rows = len(examples)
    ncols = 4
    fig, axes = plt.subplots(n_rows, ncols, figsize=(18, 4.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    # Compute global vmax for consistent color scale
    all_scores = []
    for ex in examples:
        all_scores.append(ex["heatmap_uniform"].max())
        all_scores.append(ex["heatmap_cls"].max())
    vmax = np.percentile(all_scores, 95) if all_scores else 1.0

    for i, ex in enumerate(examples):
        orig = ex["orig_img"]
        hm_uni = ex["heatmap_uniform"]
        hm_cls = ex["heatmap_cls"]
        gt_path = ex.get("gt_mask_path")
        label = ex["label"]
        score_uni = ex["score_uniform"]
        score_cls = ex["score_cls"]

        # Original
        axes[i, 0].imshow(orig)
        axes[i, 0].set_ylabel(label, fontsize=10, fontweight="bold", rotation=0,
                               labelpad=80, va="center")
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # Uniform heatmap
        axes[i, 1].imshow(orig)
        axes[i, 1].imshow(hm_uni, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        if i == 0:
            axes[i, 1].set_title("PC-Reg (uniform)", fontsize=11, fontweight="bold")
        axes[i, 1].set_xlabel(f"score: {score_uni:.1f}", fontsize=9)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # CLS heatmap
        axes[i, 2].imshow(orig)
        axes[i, 2].imshow(hm_cls, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        if i == 0:
            axes[i, 2].set_title("PC-Reg (cls-gated)", fontsize=11, fontweight="bold")
        axes[i, 2].set_xlabel(f"score: {score_cls:.1f}", fontsize=9)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # GT mask
        if gt_path is not None and gt_path.exists():
            gt = Image.open(gt_path).convert("L")
            axes[i, 3].imshow(gt, cmap="gray")
        else:
            # Normal image: show green "OK" text
            axes[i, 3].text(0.5, 0.5, "No defect", transform=axes[i, 3].transAxes,
                           fontsize=14, ha="center", va="center", color="green",
                           fontweight="bold")
            axes[i, 3].set_facecolor("#f0f0f0")
        if i == 0:
            axes[i, 3].set_title("Ground Truth", fontsize=11, fontweight="bold")
        axes[i, 3].axis("off")

    fig.suptitle(f"PC-Reg Anomaly Heatmaps — {category}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def process_category(category: str):
    """Generate heatmaps for one category."""
    print(f"\n{'='*60}")
    print(f"Category: {category}")

    # Load features
    data = load_cached_features(category)
    attn_data = load_attention_cache(category)
    img_paths = load_image_paths(category)

    train_features = data["train"]
    n_train, n_patches, feat_dim = train_features.shape

    # PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(train_features.reshape(-1, feat_dim))
    print(f"  PCA: {feat_dim} -> {PCA_DIM} (var: {pca.explained_variance_ratio_.sum():.3f})")

    train_pca = pca.transform(train_features.reshape(-1, feat_dim)).reshape(n_train, n_patches, PCA_DIM)

    # Build neighbor map
    neighbor_map = build_neighbor_map(GRID_SIZE, R)

    # Train BOTH models
    print("  Training uniform model...")
    W_uni, mu_uni, Si_uni = train_pcreg(train_pca, neighbor_map, LAM, cls_attn=None)

    print("  Training cls_perimg model...")
    train_cls_attn = attn_data["train"]
    W_cls, mu_cls, Si_cls = train_pcreg(train_pca, neighbor_map, LAM, cls_attn=train_cls_attn)

    # Create output dir for category
    cat_out = OUTPUT_DIR / category
    cat_out.mkdir(parents=True, exist_ok=True)

    # Process each split
    all_examples = []

    for split in ["test_good", "test_logical", "test_structural"]:
        split_features = data[split]
        n_test = split_features.shape[0]
        split_pca = pca.transform(split_features.reshape(-1, feat_dim)).reshape(n_test, n_patches, PCA_DIM)
        split_cls_attn = attn_data[split]

        # Score with both models
        scores_uni, patches_uni = score_pcreg(
            split_pca, neighbor_map, W_uni, mu_uni, Si_uni, cls_attn=None
        )
        scores_cls, patches_cls = score_pcreg(
            split_pca, neighbor_map, W_cls, mu_cls, Si_cls, cls_attn=split_cls_attn
        )

        # Select examples
        if split == "test_good":
            # Select: lowest-score (best normal) + highest-score (near false positive)
            sorted_idx = np.argsort(scores_uni)
            select_low = sorted_idx[:2]  # Best normals
            select_high = sorted_idx[-2:]  # Worst normals (potential FP)
            selected = list(select_low) + list(select_high)
            labels = ["Normal\n(low score)"] * 2 + ["Normal\n(high score)"] * 2
        elif split == "test_logical":
            # Select top N_EXAMPLES by uniform score (best detections)
            sorted_idx = np.argsort(-scores_uni)
            selected = list(sorted_idx[:N_EXAMPLES])
            labels = [f"Logical\n#{i+1}" for i in range(len(selected))]
        else:  # structural
            sorted_idx = np.argsort(-scores_uni)
            selected = list(sorted_idx[:N_EXAMPLES])
            labels = [f"Structural\n#{i+1}" for i in range(len(selected))]

        for j, idx in enumerate(selected):
            orig_path = img_paths[split][idx]
            orig_img = Image.open(orig_path).convert("RGB")
            orig_w, orig_h = orig_img.size

            # Upsample heatmaps
            hm_uni = patches_uni[idx].reshape(GRID_SIZE, GRID_SIZE)
            hm_cls = patches_cls[idx].reshape(GRID_SIZE, GRID_SIZE)

            hm_uni_up = np.array(
                Image.fromarray(hm_uni.astype(np.float32), mode="F").resize(
                    (orig_w, orig_h), Image.BICUBIC
                )
            )
            hm_cls_up = np.array(
                Image.fromarray(hm_cls.astype(np.float32), mode="F").resize(
                    (orig_w, orig_h), Image.BICUBIC
                )
            )

            # GT mask path
            gt_path = None
            if split in ("test_logical", "test_structural"):
                gt_type = "logical_anomalies" if split == "test_logical" else "structural_anomalies"
                gt_dir = DATA_ROOT / category / "ground_truth" / gt_type / f"{idx:03d}"
                masks = sorted(gt_dir.glob("*.png"))
                gt_path = masks[0] if masks else None

            all_examples.append({
                "orig_img": orig_img,
                "heatmap_uniform": hm_uni_up,
                "heatmap_cls": hm_cls_up,
                "gt_mask_path": gt_path,
                "label": labels[j],
                "score_uniform": scores_uni[idx],
                "score_cls": scores_cls[idx],
                "split": split,
                "idx": idx,
            })

            # Also save individual comparison figures for paper
            if split != "test_good" and j < 2:  # Top 2 anomalies per type
                fname = f"{category}_{split.replace('test_', '')}_{idx:03d}.png"
                create_comparison_figure(
                    orig_img, hm_uni_up, hm_cls_up,
                    gt_mask_path=gt_path,
                    title=f"{category} — {split.replace('test_', '')} #{idx:03d}  "
                          f"(uniform: {scores_uni[idx]:.1f}, cls: {scores_cls[idx]:.1f})",
                    vmax=np.percentile(
                        np.concatenate([hm_uni_up.ravel(), hm_cls_up.ravel()]), 98
                    ),
                    save_path=cat_out / fname,
                )

        print(f"  {split}: scored {n_test} images, selected {len(selected)} examples")

    # Create grid figure for the category
    # Select best examples: 1 normal + 3 logical + 3 structural
    grid_examples = []
    for ex in all_examples:
        if ex["split"] == "test_good" and "low score" in ex["label"]:
            grid_examples.append(ex)
            if len([e for e in grid_examples if e["split"] == "test_good"]) >= 1:
                break
    for ex in all_examples:
        if ex["split"] == "test_logical":
            grid_examples.append(ex)
            if len([e for e in grid_examples if e["split"] == "test_logical"]) >= 3:
                break
    for ex in all_examples:
        if ex["split"] == "test_structural":
            grid_examples.append(ex)
            if len([e for e in grid_examples if e["split"] == "test_structural"]) >= 3:
                break

    if grid_examples:
        create_grid_figure(
            grid_examples, category,
            save_path=cat_out / f"{category}_grid.png"
        )

    print(f"  Done: {len(all_examples)} total examples, saved to {cat_out}")
    return all_examples


def create_paper_figure(all_results: dict):
    """Create landscape paper figure: 4 rows x 5 cols.

    Layout (per category column):
      Row 0: Logical -- original image with GT mask in red
      Row 1: Logical -- heatmap overlay (PC-Reg_CLS)
      Row 2: Structural -- original image with GT mask in red
      Row 3: Structural -- heatmap overlay (PC-Reg_CLS)
    """
    n_cats = len(ALL_CATEGORIES)
    n_rows = 4  # img+GT, heatmap, img+GT, heatmap

    # Collect best logical + structural per category (by CLS score)
    selected = {}  # cat -> {"logical": example, "structural": example}
    for cat in ALL_CATEGORIES:
        if cat not in all_results:
            continue
        examples = all_results[cat]
        logical = [e for e in examples if e["split"] == "test_logical"]
        structural = [e for e in examples if e["split"] == "test_structural"]
        selected[cat] = {
            "logical": max(logical, key=lambda e: e["score_cls"]) if logical else None,
            "structural": max(structural, key=lambda e: e["score_cls"]) if structural else None,
        }

    # Compute global vmax across all selected heatmaps for consistent colorscale
    all_hm_vals = []
    for cat in ALL_CATEGORIES:
        if cat not in selected:
            continue
        for atype in ("logical", "structural"):
            ex = selected[cat].get(atype)
            if ex is not None:
                all_hm_vals.append(ex["heatmap_cls"].max())
    vmax = np.percentile(all_hm_vals, 95) if all_hm_vals else 1.0

    # Pretty category names for column headers
    cat_labels = {
        "breakfast_box": "breakfast box",
        "juice_bottle": "juice bottle",
        "pushpins": "pushpins",
        "screw_bag": "screw bag",
        "splicing_connectors": "splicing conn.",
    }

    fig, axes = plt.subplots(n_rows, n_cats, figsize=(3.2 * n_cats, 3.0 * n_rows))

    for col, cat in enumerate(ALL_CATEGORIES):
        if cat not in selected:
            continue

        for row_pair, atype in enumerate(("logical", "structural")):
            ex = selected[cat].get(atype)
            if ex is None:
                for r in range(2):
                    axes[row_pair * 2 + r, col].axis("off")
                continue

            orig_img = ex["orig_img"]
            heatmap = ex["heatmap_cls"]
            gt_path = ex.get("gt_mask_path")
            orig_w, orig_h = orig_img.size

            # --- Row A: original + GT in red ---
            ax_img = axes[row_pair * 2, col]
            ax_img.imshow(orig_img)
            if gt_path is not None and gt_path.exists():
                gt_mask = np.array(Image.open(gt_path).convert("L").resize(
                    (orig_w, orig_h), Image.NEAREST
                ))
                # Create red overlay where GT > 0
                red_overlay = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
                mask_bool = gt_mask > 127
                red_overlay[mask_bool] = [1.0, 0.0, 0.0, 0.35]
                ax_img.imshow(red_overlay)
            ax_img.axis("off")

            # --- Row B: heatmap overlay ---
            ax_hm = axes[row_pair * 2 + 1, col]
            ax_hm.imshow(orig_img)
            ax_hm.imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
            ax_hm.axis("off")

        # Column header (category name)
        axes[0, col].set_title(cat_labels.get(cat, cat), fontsize=11,
                               fontweight="bold", pad=6)

    # Row labels on the left
    for r in range(n_rows):
        if r in (0, 2):  # Image rows -- label spans both rows
            label = "Logical" if r == 0 else "Structural"
            # Place label between the two sub-rows
            mid_y = (axes[r, 0].get_position().y0 + axes[r + 1, 0].get_position().y1) / 2
            fig.text(0.01, mid_y, label, va="center", ha="left",
                     fontsize=12, fontweight="bold", rotation=90,
                     transform=fig.transFigure)

    plt.tight_layout(rect=[0.03, 0, 1, 1])  # Leave space for row labels

    # Save both PNG and PDF
    save_png = OUTPUT_DIR / "paper_figure_heatmaps.png"
    save_pdf = OUTPUT_DIR / "paper_figure_heatmaps.pdf"
    fig.savefig(save_png, dpi=200, bbox_inches="tight")
    fig.savefig(save_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Paper figure saved: {save_png}")
    print(f"  Paper figure saved: {save_pdf}")


def main():
    print("=" * 70)
    print("PC-Reg Heatmap Generator")
    print(f"  Categories: {ALL_CATEGORIES}")
    print(f"  Config: R={R}, lam={LAM}, PCA={PCA_DIM}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    if not CACHE_DIR.is_dir() or not ATTN_CACHE_DIR.is_dir() or not DATA_ROOT.is_dir():
        print(
            "[pc-reg] Skipping heatmap generation: required artifacts not found.\n\n"
            "This script needs:\n"
            "  - LOCO dataset images in data/mvtec_loco_AD/\n"
            "  - cached features in features/loco/\n"
            "  - cached CLS attention in features/loco_attention/\n\n"
            "To generate from scratch:\n"
            "  uv run scripts/extract_features_loco.py\n"
            "  uv run scripts/run_pcreg_cls_loco.py\n"
            "  uv run scripts/generate_heatmaps.py\n",
            flush=True,
        )
        return

    # Fail fast if any per-category cache file is missing.
    required_splits = ["train", "test_good", "test_logical", "test_structural"]
    for category in ALL_CATEGORIES:
        cat_cache = CACHE_DIR / category
        cat_attn = ATTN_CACHE_DIR / category
        for split in required_splits:
            if not (cat_cache / f"{split}_features.npy").is_file() or not (cat_attn / f"{split}_cls_attn.npy").is_file():
                print(
                    f"[pc-reg] Skipping heatmaps: missing cache files for category '{category}'.\n"
                    "Run:\n"
                    "  uv run scripts/extract_features_loco.py\n"
                    "  uv run scripts/run_pcreg_cls_loco.py\n",
                    flush=True,
                )
                return

    all_results = {}
    for category in ALL_CATEGORIES:
        examples = process_category(category)
        all_results[category] = examples

    # Create combined paper figure
    create_paper_figure(all_results)

    print("\n" + "=" * 70)
    print("HEATMAP GENERATION COMPLETE")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
