"""
Generate ACID test diagram for the paper.

Layout: 3 columns x 2 rows
  Top row:    real image | feature map (original) | feature map (permuted)
  Bottom row: (annotation) | score map (original) | score map (permuted)

Usage:
  uv run scripts/generate_acid_figure.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.decomposition import PCA
from PIL import Image

from _repro_utils import repo_root_from

# =============================================================================
# CONFIG
# =============================================================================
REPO_ROOT = repo_root_from(__file__)
OUTPUT_DIR = REPO_ROOT / "figures"
CACHE_DIR = REPO_ROOT / "features" / "loco"
DATA_ROOT = REPO_ROOT / "data" / "mvtec_loco_AD"
CATEGORY = "juice_bottle"
GRID_SIZE = 28
BLOCK_SIZE = 7
PCA_DIM = 256
LAMBDA = 1.0
RADIUS = 5
SEED = 42
IMG_IDX = 0

# Two blocks to swap (chosen for visual clarity -- different image regions)
BLOCK1_POS = (3, 3)
BLOCK2_POS = (18, 18)

# Colors
COLOR_A = '#E63946'  # red
COLOR_B = '#457B9D'  # blue


def compute_neighbor_means(features, grid_size, radius):
    N, _, D = features.shape
    H = W = grid_size
    F = features.reshape(N, H, W, D).astype(np.float64)
    S = np.zeros((N, H + 1, W + 1, D), dtype=np.float64)
    S[:, 1:, 1:, :] = np.cumsum(np.cumsum(F, axis=1), axis=2)
    pos = np.arange(H * W)
    y, x = pos // W, pos % W
    y1 = np.maximum(0, y - radius)
    y2 = np.minimum(H - 1, y + radius) + 1
    x1 = np.maximum(0, x - radius)
    x2 = np.minimum(W - 1, x + radius) + 1
    counts = (y2 - y1) * (x2 - x1) - 1
    ws = S[:, y2, x2, :] - S[:, y1, x2, :] - S[:, y2, x1, :] + S[:, y1, x1, :]
    ws -= F.reshape(N, H * W, D)
    return (ws / counts[None, :, None]).astype(np.float32)


def swap_blocks(features_2d, pos1, pos2, block_size, grid_size):
    out = features_2d.copy()
    r1, c1 = pos1
    r2, c2 = pos2
    idx1, idx2 = [], []
    for dr in range(block_size):
        for dc in range(block_size):
            idx1.append((r1 + dr) * grid_size + (c1 + dc))
            idx2.append((r2 + dr) * grid_size + (c2 + dc))
    idx1, idx2 = np.array(idx1), np.array(idx2)
    temp = out[idx1].copy()
    out[idx1] = out[idx2]
    out[idx2] = temp
    return out


def compute_score_map(feat_pca, train_pca, train_nbr):
    n_patches = feat_pca.shape[0]
    dim = feat_pca.shape[1]
    feat_3d = feat_pca[np.newaxis]
    test_nbr = compute_neighbor_means(feat_3d, GRID_SIZE, RADIUS)
    scores = np.zeros(n_patches)
    I = np.eye(dim, dtype=np.float64)
    for p in range(n_patches):
        X = train_nbr[:, p, :].astype(np.float64)
        Y = train_pca[:, p, :].astype(np.float64)
        W = np.linalg.solve(X.T @ X + LAMBDA * I, X.T @ Y)
        residuals = Y - X @ W
        mu = residuals.mean(axis=0)
        var = residuals.var(axis=0) + 1e-6
        inv_var = 1.0 / var
        x_test = test_nbr[0, p, :].astype(np.float64)
        y_test = feat_pca[p, :].astype(np.float64)
        r_test = y_test - x_test @ W - mu
        scores[p] = np.sqrt(np.sum(r_test ** 2 * inv_var))
    return scores.reshape(GRID_SIZE, GRID_SIZE)


def add_block_rect(ax, pos, block_size, color, linestyle='-', linewidth=2.0, label=None):
    r, c = pos
    rect = mpatches.Rectangle((c - 0.5, r - 0.5), block_size, block_size,
                                linewidth=linewidth, edgecolor=color,
                                facecolor='none', linestyle=linestyle)
    ax.add_patch(rect)
    if label:
        ax.text(c + block_size / 2, r + block_size / 2, label,
                ha='center', va='center', fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=color, edgecolor='none', alpha=0.85))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cat_dir = CACHE_DIR / CATEGORY
    img_dir = DATA_ROOT / CATEGORY / "test" / "good"
    if not (cat_dir / "train_features.npy").is_file() or not (cat_dir / "test_good_features.npy").is_file():
        print(
            "[pc-reg] Skipping figure generation: cached LOCO features not found.\n\n"
            "To generate the ACID figure from scratch:\n"
            "  1) Download MVTec LOCO AD into data/mvtec_loco_AD/\n"
            "  2) uv run scripts/extract_features_loco.py\n"
            "  3) uv run scripts/generate_acid_figure.py\n",
            flush=True,
        )
        return
    if not img_dir.is_dir():
        print(
            "[pc-reg] Skipping figure generation: dataset images not found.\n\n"
            "Expected: data/mvtec_loco_AD/<category>/test/good/*.png\n"
            "See README.md (Datasets section).\n",
            flush=True,
        )
        return

    print("Loading features...", flush=True)
    train_feat = np.load(str(cat_dir / "train_features.npy"))
    test_feat = np.load(str(cat_dir / "test_good_features.npy"))

    # Load real image
    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        print(
            f"[pc-reg] Skipping figure generation: no images found in {img_dir}\n",
            flush=True,
        )
        return
    real_img = np.array(Image.open(img_files[IMG_IDX]).convert("RGB"))

    n_train = train_feat.shape[0]
    D = train_feat.shape[2]

    # PCA
    print("Fitting PCA...", flush=True)
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    pca.fit(train_feat.reshape(-1, D))
    train_pca = pca.transform(train_feat.reshape(-1, D)).astype(np.float32).reshape(n_train, GRID_SIZE**2, PCA_DIM)
    train_nbr = compute_neighbor_means(train_pca, GRID_SIZE, RADIUS)

    test_pca_orig = pca.transform(test_feat[IMG_IDX].reshape(-1, D)).astype(np.float32)
    test_pca_perm = swap_blocks(test_pca_orig, BLOCK1_POS, BLOCK2_POS, BLOCK_SIZE, GRID_SIZE)

    # Feature visualization: use first PC as grayscale intensity
    print("Computing visualizations...", flush=True)
    pc1_orig = test_pca_orig[:, 0].reshape(GRID_SIZE, GRID_SIZE)
    pc1_perm = test_pca_perm[:, 0].reshape(GRID_SIZE, GRID_SIZE)

    # Score maps
    print("Computing score maps...", flush=True)
    scores_orig = compute_score_map(test_pca_orig, train_pca, train_nbr)
    scores_perm = compute_score_map(test_pca_perm, train_pca, train_nbr)

    p95_orig = np.percentile(scores_orig, 95)
    p95_perm = np.percentile(scores_perm, 95)
    ratio = p95_perm / p95_orig

    # ==========================================================================
    # FIGURE
    # ==========================================================================
    print("Generating figure...", flush=True)

    fig = plt.figure(figsize=(11, 6.5))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1],
                  height_ratios=[1, 1], hspace=0.28, wspace=0.12)

    # --- Top row ---

    # (a) Real image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(real_img)
    # Draw approximate block regions on real image (scaled from 28x28 to image size)
    h, w = real_img.shape[:2]
    scale_y, scale_x = h / GRID_SIZE, w / GRID_SIZE
    for pos, color, label in [(BLOCK1_POS, COLOR_A, 'A'), (BLOCK2_POS, COLOR_B, 'B')]:
        r, c = pos
        rect = mpatches.Rectangle((c * scale_x, r * scale_y),
                                   BLOCK_SIZE * scale_x, BLOCK_SIZE * scale_y,
                                   linewidth=2, edgecolor=color, facecolor=color, alpha=0.25)
        ax_img.add_patch(rect)
        rect2 = mpatches.Rectangle((c * scale_x, r * scale_y),
                                    BLOCK_SIZE * scale_x, BLOCK_SIZE * scale_y,
                                    linewidth=2, edgecolor=color, facecolor='none')
        ax_img.add_patch(rect2)
        ax_img.text(c * scale_x + BLOCK_SIZE * scale_x / 2,
                    r * scale_y + BLOCK_SIZE * scale_y / 2,
                    label, ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=color, edgecolor='none', alpha=0.9))
    ax_img.set_title("(a) Input image", fontsize=11, fontweight='bold')
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # (b) Original feature map (1st PC)
    ax_feat_orig = fig.add_subplot(gs[0, 1])
    ax_feat_orig.imshow(pc1_orig, cmap='viridis', interpolation='nearest')
    add_block_rect(ax_feat_orig, BLOCK1_POS, BLOCK_SIZE, COLOR_A, label='A')
    add_block_rect(ax_feat_orig, BLOCK2_POS, BLOCK_SIZE, COLOR_B, label='B')
    ax_feat_orig.set_title("(b) Feature grid (original)", fontsize=11, fontweight='bold')
    ax_feat_orig.set_xticks([])
    ax_feat_orig.set_yticks([])

    # (c) Permuted feature map
    ax_feat_perm = fig.add_subplot(gs[0, 2])
    ax_feat_perm.imshow(pc1_perm, cmap='viridis', interpolation='nearest',
                         vmin=pc1_orig.min(), vmax=pc1_orig.max())
    add_block_rect(ax_feat_perm, BLOCK1_POS, BLOCK_SIZE, COLOR_B, linestyle='--', label='B')
    add_block_rect(ax_feat_perm, BLOCK2_POS, BLOCK_SIZE, COLOR_A, linestyle='--', label='A')
    ax_feat_perm.set_title("(c) Feature grid (A$\\leftrightarrow$B swapped)", fontsize=11, fontweight='bold')
    ax_feat_perm.set_xticks([])
    ax_feat_perm.set_yticks([])

    # Arrow between (b) and (c)
    fig.text(0.62, 0.78, "$\\longleftrightarrow$", ha='center', va='center',
             fontsize=18, fontweight='bold', color='#333333')

    # --- Bottom row ---

    # (d) Annotation panel: explain what happens
    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.axis('off')
    explanation = (
        "Swapping $7{\\times}7$ blocks\n"
        "preserves the multiset\n"
        "of patch features but\n"
        "breaks spatial context.\n\n"
        "PatchCore: same patches\n"
        "$\\Rightarrow$ same score\n"
        "(AUROC $= 0.500$)\n\n"
        "PC-Reg: broken neighbors\n"
        "$\\Rightarrow$ high residual\n"
        "(AUROC $= 1.000$)"
    )
    ax_text.text(0.5, 0.5, explanation, ha='center', va='center',
                 fontsize=10, linespacing=1.4,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8F8F8',
                           edgecolor='#CCCCCC', linewidth=1))

    # (e) Score map original
    ax_score_orig = fig.add_subplot(gs[1, 1])
    vmax_score = scores_perm.max()
    ax_score_orig.imshow(scores_orig, cmap='inferno', interpolation='nearest',
                          vmin=0, vmax=vmax_score)
    add_block_rect(ax_score_orig, BLOCK1_POS, BLOCK_SIZE, 'white', linestyle='--', linewidth=1.5)
    add_block_rect(ax_score_orig, BLOCK2_POS, BLOCK_SIZE, 'white', linestyle='--', linewidth=1.5)
    ax_score_orig.set_title(f"(d) Score map (original, P$_{{95}}$={p95_orig:.0f})",
                             fontsize=11, fontweight='bold')
    ax_score_orig.set_xticks([])
    ax_score_orig.set_yticks([])

    # (f) Score map permuted
    ax_score_perm = fig.add_subplot(gs[1, 2])
    im = ax_score_perm.imshow(scores_perm, cmap='inferno', interpolation='nearest',
                               vmin=0, vmax=vmax_score)
    add_block_rect(ax_score_perm, BLOCK1_POS, BLOCK_SIZE, 'white', linestyle='--', linewidth=1.5)
    add_block_rect(ax_score_perm, BLOCK2_POS, BLOCK_SIZE, 'white', linestyle='--', linewidth=1.5)
    ax_score_perm.set_title(f"(e) Score map (permuted, P$_{{95}}$={p95_perm:.0f})",
                             fontsize=11, fontweight='bold')
    ax_score_perm.set_xticks([])
    ax_score_perm.set_yticks([])

    # Colorbar for score maps
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.38])
    fig.colorbar(im, cax=cbar_ax, label='Mahalanobis distance')

    out_path = OUTPUT_DIR / "acid_test_diagram.pdf"
    fig.savefig(str(out_path), bbox_inches='tight', dpi=200)
    out_png = OUTPUT_DIR / "acid_test_diagram.png"
    fig.savefig(str(out_png), bbox_inches='tight', dpi=200)
    print(f"\n[OK] Saved to {out_path}")
    print(f"[OK] Preview: {out_png}")
    print(f"\nStats: P95 original={p95_orig:.1f}, P95 permuted={p95_perm:.1f}, ratio={ratio:.1f}x")


if __name__ == "__main__":
    main()
