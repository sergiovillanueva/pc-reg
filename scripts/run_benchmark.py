"""
Complexity Benchmark for PC-Reg.

Measures time for each pipeline step separately:
  1. Feature extraction (GPU) -- per image
  2. PCA fitting (CPU) -- one-time
  3. Ridge model fitting (CPU) -- one-time per category
  4. Inference/scoring (CPU) -- per image

Also generates the comparison table for the paper.

Usage:
    uv run scripts/run_benchmark.py
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import time
import warnings
from pathlib import Path

from _repro_utils import repo_root_from, skip_if_missing

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

# =============================================================================
# CONFIG
# =============================================================================

REPO_ROOT = repo_root_from(__file__)
CACHE_DIR = REPO_ROOT / "features" / "loco"
OUTPUT_DIR = REPO_ROOT / "results" / "benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PCA_DIM = 256
GRID_SIZE = 28
N_PATCHES = GRID_SIZE * GRID_SIZE
R = 5
LAM = 1.0

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]


# =============================================================================
# PC-REG FUNCTIONS
# =============================================================================

def build_neighbor_map(grid_size, radius):
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


def score_single_image(features_1img, neighbor_map, W, mu_r, Sigma_r_inv):
    """Score a SINGLE image (1, 784, D) -> image_score, patch_scores."""
    n_patches = features_1img.shape[1]
    dim = features_1img.shape[2]
    patch_scores = np.zeros(n_patches)
    for p in range(n_patches):
        nbrs = neighbor_map[p]
        X = features_1img[0, nbrs, :].mean(axis=0, keepdims=True)  # (1, D)
        Y_pred = X @ W[p]  # (1, D)
        Y_real = features_1img[0, p:p+1, :]  # (1, D)
        residual = Y_real - Y_pred
        centered = residual - mu_r[p]
        mahal_sq = np.sum(centered @ Sigma_r_inv[p] * centered, axis=1)
        patch_scores[p] = np.sqrt(max(0, mahal_sq[0]))
    return np.percentile(patch_scores, 95), patch_scores


def score_batch(test_features, neighbor_map, W, mu_r, Sigma_r_inv):
    """Score a batch of images."""
    n_test, n_patches, dim = test_features.shape
    patch_scores = np.zeros((n_test, n_patches))
    for p in range(n_patches):
        nbrs = neighbor_map[p]
        X = test_features[:, nbrs, :].mean(axis=1)
        residuals = test_features[:, p, :] - X @ W[p]
        centered = residuals - mu_r[p]
        mahal_sq = np.sum(centered @ Sigma_r_inv[p] * centered, axis=1)
        patch_scores[:, p] = np.sqrt(np.maximum(0, mahal_sq))
    image_scores = np.percentile(patch_scores, 95, axis=1)
    return image_scores, patch_scores


# =============================================================================
# FEATURE EXTRACTION BENCHMARK (GPU)
# =============================================================================

def benchmark_feature_extraction():
    """Measure DINOv3 feature extraction time per image."""
    import torch
    from transformers import AutoModel
    from torchvision import transforms
    from PIL import Image

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    print("\n[BENCH] Loading DINOv3-ViT-L/16...")
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get some test images
    data_dir = REPO_ROOT / "data" / "mvtec_loco_AD" / "breakfast_box" / "test" / "good"
    img_paths = sorted([str(p) for p in data_dir.rglob("*.png")])[:20]

    # Warm-up
    with torch.no_grad():
        img = transform(Image.open(img_paths[0]).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
            _ = model(img, output_hidden_states=True)

    # Benchmark: single image
    times_single = []
    for p in img_paths[:10]:
        img = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
                outputs = model(img, output_hidden_states=True)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        t1 = time.perf_counter()
        times_single.append(t1 - t0)

    # Benchmark: batch of 8
    batch_imgs = [transform(Image.open(p).convert("RGB")) for p in img_paths[:8]]
    batch = torch.stack(batch_imgs).to(DEVICE)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
            outputs = model(batch, output_hidden_states=True)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t1 = time.perf_counter()
    time_batch8 = t1 - t0

    del model, batch, outputs
    torch.cuda.empty_cache()

    feat_single_ms = np.mean(times_single) * 1000
    feat_batch_ms = time_batch8 * 1000 / 8

    print(f"  Feature extraction (single): {feat_single_ms:.1f} ms/image")
    print(f"  Feature extraction (batch=8): {feat_batch_ms:.1f} ms/image")

    return {
        "feat_single_ms": feat_single_ms,
        "feat_batch_ms": feat_batch_ms,
    }


# =============================================================================
# CPU BENCHMARK (PCA + Ridge + Scoring)
# =============================================================================

def benchmark_cpu():
    """Measure PCA fitting, Ridge training, and per-image scoring."""
    neighbor_map = build_neighbor_map(GRID_SIZE, R)

    all_results = []

    for category in CATEGORIES:
        print(f"\n[BENCH] {category}")

        # Load features
        cat_cache = CACHE_DIR / category
        train_feat = np.load(str(cat_cache / "train_features.npy"))
        test_feat = np.load(str(cat_cache / "test_good_features.npy"))

        n_train, n_patches, feat_dim = train_feat.shape
        n_test = test_feat.shape[0]

        # 1. PCA fitting
        t0 = time.perf_counter()
        pca = PCA(n_components=PCA_DIM, random_state=42)
        pca.fit(train_feat.reshape(-1, feat_dim))
        pca_fit_time = time.perf_counter() - t0

        # 2. PCA transform (train)
        t0 = time.perf_counter()
        train_pca = pca.transform(train_feat.reshape(-1, feat_dim)).reshape(
            n_train, n_patches, PCA_DIM
        )
        pca_transform_train_time = time.perf_counter() - t0

        # 3. PCA transform (single image)
        t0 = time.perf_counter()
        for i in range(min(10, n_test)):
            _ = pca.transform(test_feat[i].reshape(-1, feat_dim)).reshape(
                1, n_patches, PCA_DIM
            )
        pca_transform_single = (time.perf_counter() - t0) / min(10, n_test)

        # 4. Ridge model fitting
        t0 = time.perf_counter()
        W, mu_r, Sigma_r_inv = train_pcreg(train_pca, neighbor_map, LAM)
        ridge_fit_time = time.perf_counter() - t0

        # 5. Scoring single image
        test_pca = pca.transform(test_feat.reshape(-1, feat_dim)).reshape(
            n_test, n_patches, PCA_DIM
        )
        times_single = []
        for i in range(min(20, n_test)):
            t0 = time.perf_counter()
            _ = score_single_image(
                test_pca[i:i+1], neighbor_map, W, mu_r, Sigma_r_inv
            )
            times_single.append(time.perf_counter() - t0)
        score_single_ms = np.mean(times_single) * 1000

        # 6. Scoring batch
        t0 = time.perf_counter()
        _ = score_batch(test_pca, neighbor_map, W, mu_r, Sigma_r_inv)
        score_batch_time = time.perf_counter() - t0
        score_batch_ms = score_batch_time * 1000 / n_test

        result = {
            "category": category,
            "n_train": n_train,
            "n_test": n_test,
            "pca_fit_s": pca_fit_time,
            "pca_transform_single_ms": pca_transform_single * 1000,
            "ridge_fit_s": ridge_fit_time,
            "score_single_ms": score_single_ms,
            "score_batch_ms": score_batch_ms,
            "total_fit_s": pca_fit_time + pca_transform_train_time + ridge_fit_time,
        }
        all_results.append(result)

        print(f"  n_train={n_train}, n_test={n_test}")
        print(f"  PCA fit: {pca_fit_time:.2f}s")
        print(f"  PCA transform (1 img): {pca_transform_single*1000:.1f}ms")
        print(f"  Ridge fit (784 regressions): {ridge_fit_time:.2f}s")
        print(f"  Score (single img): {score_single_ms:.1f}ms")
        print(f"  Score (batch, per img): {score_batch_ms:.1f}ms")
        print(f"  Total fitting: {result['total_fit_s']:.2f}s")

    return all_results


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def generate_comparison_table(gpu_results, cpu_results):
    """Generate the paper comparison table."""
    # Averages across categories
    avg_pca_fit = np.mean([r["pca_fit_s"] for r in cpu_results])
    avg_ridge_fit = np.mean([r["ridge_fit_s"] for r in cpu_results])
    avg_total_fit = np.mean([r["total_fit_s"] for r in cpu_results])
    avg_score_single = np.mean([r["score_single_ms"] for r in cpu_results])
    avg_score_batch = np.mean([r["score_batch_ms"] for r in cpu_results])
    avg_pca_single = np.mean([r["pca_transform_single_ms"] for r in cpu_results])
    feat_ms = gpu_results["feat_batch_ms"]

    # Total per-image inference: feature extraction + PCA transform + scoring
    total_per_image_ms = feat_ms + avg_pca_single + avg_score_single

    table = f"""
# PC-Reg Complexity Analysis
# ==========================

## Pipeline Timing (measured on i9 + RTX 4070Ti 16GB)

### One-time costs (fitting/training)
| Step | Time | Notes |
|------|------|-------|
| PCA fitting | {avg_pca_fit:.1f}s | {cpu_results[0]['n_train']}~{cpu_results[-1]['n_train']} train images |
| Ridge fitting | {avg_ridge_fit:.1f}s | 784 closed-form regressions + LedoitWolf |
| **Total fitting** | **{avg_total_fit:.1f}s** | **<2 min per category** |

### Per-image costs (inference)
| Step | Time (ms) | Notes |
|------|-----------|-------|
| DINOv3 feature extraction | {feat_ms:.0f}ms | GPU (batch=8) |
| PCA projection | {avg_pca_single:.0f}ms | CPU |
| PC-Reg scoring | {avg_score_single:.0f}ms | CPU (single), {avg_score_batch:.0f}ms (batch) |
| **Total per image** | **{total_per_image_ms:.0f}ms** | **GPU + CPU** |

## Comparison Table (for paper)

| Method | AUROC | Training | Inference | Hardware | VLM? | SAM? | Deterministic? |
|--------|-------|----------|-----------|----------|------|------|----------------|
| **SALAD** | **96.1%** | Hours (GPU) | ~500ms | GPU | No | SAM-HQ | Yes |
| **CSAD** | **95.3%** | Hours (GPU) | ~500ms | GPU | No | G-SAM | Yes |
| EfficientAD | 90.7% | ~30min (GPU) | ~5ms | GPU | No | No | Yes |
| **LogSAD** | **90.2%** | None | ~30s | GPU+API | GPT-4V | SAM-H | No |
| LogicQA | 87.6% (log) | None | ~30s | GPU+API | GPT-4o | No | No |
| **PC-Reg (ours)** | **83.5%** | **{avg_total_fit:.0f}s (CPU)** | **{total_per_image_ms:.0f}ms** | **CPU** | **No** | **No** | **Yes** |
| PatchCore | ~81%* | ~10s (CPU) | ~50ms | CPU | No | No | Yes |
| UniVAD (1-shot) | 71.0% | None | ~2s | GPU | No | G-SAM | Yes |

Notes:
- LogSAD/LogicQA inference includes GPT-4V API call (~$0.02/image)
- EfficientAD timing from paper (student-teacher distillation)
- SALAD/CSAD require GPU training (V100/A100 hours)
- PC-Reg: CPU-only after one-time feature extraction
- *PatchCore 81% from LogSAD paper (possibly DINOv2 features)

## Per-category breakdown

| Category | n_train | PCA fit | Ridge fit | Total fit | Score/img |
|----------|---------|---------|-----------|-----------|-----------|
"""
    for r in cpu_results:
        table += f"| {r['category']} | {r['n_train']} | {r['pca_fit_s']:.1f}s | {r['ridge_fit_s']:.1f}s | {r['total_fit_s']:.1f}s | {r['score_single_ms']:.0f}ms |\n"

    table += f"""
## Key arguments for paper

1. **Fitting cost**: PC-Reg fits in {avg_total_fit:.0f}s on CPU. No GPU needed for training.
   LogSAD needs no fitting but costs ~$0.02/image in API calls.
   SALAD/CSAD need hours of GPU training.

2. **Inference cost**: {total_per_image_ms:.0f}ms per image (dominated by feature extraction).
   LogSAD: ~30s per image (API latency).
   Ratio: PC-Reg is ~{30000/total_per_image_ms:.0f}x faster at inference.

3. **Total cost for LOCO** (5 categories, ~1500 test images):
   PC-Reg: {avg_total_fit*5:.0f}s fitting + {total_per_image_ms*1500/1000:.0f}s inference = ~{(avg_total_fit*5 + total_per_image_ms*1500/1000)/60:.0f} min total
   LogSAD: 0s fitting + {30*1500:.0f}s inference = ~{30*1500/3600:.0f}h + API cost ~${0.02*1500:.0f}

4. **Determinism**: PC-Reg is fully deterministic (Ridge closed-form + PCA seed).
   LogSAD/LogicQA: LLM sampling introduces variance.

5. **Dependencies**: PC-Reg needs 1 backbone (DINOv3).
   LogSAD needs GPT-4V + SAM-H + CLIP + DINOv2 (4 models).
"""

    return table


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PC-Reg Complexity Benchmark")
    print("=" * 70)

    if skip_if_missing(
        required=[CACHE_DIR, REPO_ROOT / "data" / "mvtec_loco_AD"],
        precomputed=[OUTPUT_DIR / "complexity_table.md"],
        what="benchmark prerequisites (cached features + LOCO dataset)",
        reproduce_hint=(
            "  # 1) Place MVTec LOCO AD under pc-reg/data/mvtec_loco_AD\n"
            "  # 2) Cache features:\n"
            "  uv run scripts/extract_features_loco.py\n"
            "  # 3) Run benchmark:\n"
            "  uv run scripts/run_benchmark.py"
        ),
    ):
        return

    # GPU benchmark
    print("\n--- GPU Benchmark (Feature Extraction) ---")
    gpu_results = benchmark_feature_extraction()

    # CPU benchmark
    print("\n--- CPU Benchmark (PCA + Ridge + Scoring) ---")
    cpu_results = benchmark_cpu()

    # Generate table
    table = generate_comparison_table(gpu_results, cpu_results)

    # Save
    table_path = OUTPUT_DIR / "complexity_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"\n\nTable saved to: {table_path}")
    print(table)


if __name__ == "__main__":
    main()
