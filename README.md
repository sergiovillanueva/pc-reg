# PC-Reg: Modeling Spatial Dependencies for Training-Free Anomaly Detection

This repository contains the code and results for reproducing the experiments in our paper.

## Method

PC-Reg detects logical and structural anomalies in industrial images by modeling spatial dependencies between patch features. For each position in a 28x28 feature grid (extracted from DINOv3-ViT-L/16), we fit a closed-form Ridge regression that predicts the center patch from its spatial neighbors. Anomalies are scored via the Mahalanobis distance of the prediction residual.

The method is **training-free**: no gradient-based optimization is required. The entire pipeline (PCA + Ridge regression + Ledoit-Wolf covariance) runs in closed form.

**PC-Reg_CLS** extends this by weighting neighbor contributions using the CLS-to-patch attention from DINOv3 (layer -6), which improves detection on 26 of 32 categories across three benchmarks.

## Results

| Dataset | Categories | PC-Reg | PC-Reg_CLS |
|---------|-----------|--------|------------|
| MVTec LOCO AD | 5 | 80.3% | **83.5%** |
| MVTec AD | 15 | 96.4% | **98.4%** |
| VisA | 12 | 86.1% | **89.6%** |

Image-level AUROC (%), single fixed configuration across all categories.

## Setup

We use [uv](https://docs.astral.sh/uv/) for dependency management. Install it first, then:

```bash
uv sync
```

By default this installs CPU-compatible PyTorch wheels. If you want a specific CUDA build, install PyTorch following the official instructions for your platform.

### Datasets

Download and place the datasets under `data/`:

```
data/
  mvtec_loco_AD/    # MVTec LOCO AD (5 categories)
  mvtec_AD/         # MVTec AD (15 categories)
  VisA/             # VisA (12 categories, with split_csv/1cls.csv)
```

- [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco)
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA](https://github.com/amazon-science/spot-diff)

## Reproducing the experiments

### Step 1: Extract features (GPU)

Extract DINOv3-ViT-L/16 patch features and cache them to disk. This only needs to be done once.

```bash
uv run scripts/extract_features_loco.py
uv run scripts/extract_features_mvtec.py
uv run scripts/extract_features_visa.py
```

Features are saved under `features/`. Requires a GPU with at least 12 GB VRAM.

### Step 2: Run experiments (CPU + GPU for attention)

All experiment scripts load pre-cached features and run on CPU only, except the CLS-gating scripts which extract attention maps on GPU during their first run.

**PC-Reg on LOCO (main ablation):**
```bash
uv run scripts/run_pcreg_loco.py            # PC-Reg on LOCO
```

**PC-Reg_CLS on all three datasets:**
```bash
uv run scripts/run_pcreg_cls_loco.py         # PC-Reg + CLS-gating on LOCO
uv run scripts/run_pcreg_cls_mvtec.py        # PC-Reg + CLS-gating on MVTec AD
uv run scripts/run_pcreg_cls_visa.py         # PC-Reg + CLS-gating on VisA
```

The CLS scripts automatically extract and cache DINOv3 attention maps on the first run (requires GPU). Subsequent runs use cached attention and run on CPU only.

**Fair comparison:**
```bash
uv run scripts/run_fair_comparison.py        # PatchCore, PaDiM, MeanSub, PC-Reg
```

**Permutation test:**
```bash
uv run scripts/run_permutation_test.py       # PatchCore vs PC-Reg on synthetic permutations
```

**Pixel-level metrics:**
```bash
uv run scripts/run_pixel_metrics.py          # pixel-AUROC and AUPRO
```

**Statistical tests:**
```bash
uv run scripts/run_statistical_tests.py      # Wilcoxon signed-rank + Cliff's delta
```

**Complexity benchmark:**
```bash
uv run scripts/run_benchmark.py              # Pipeline timing (GPU + CPU)
```

### Step 3: Generate figures

Note: `generate_heatmaps.py` requires the LOCO attention cache from `run_pcreg_cls_loco.py`.

```bash
uv run scripts/generate_acid_figure.py       # Permutation test diagram
uv run scripts/generate_heatmaps.py          # Anomaly localization heatmaps
```

If datasets/features are not present, the figure scripts print a short message and skip instead of crashing.

## Repository structure

```
pc-reg/
  scripts/
    extract_features_loco.py     Feature extraction for LOCO (GPU, one-time)
    extract_features_mvtec.py    Feature extraction for MVTec AD (GPU, one-time)
    extract_features_visa.py     Feature extraction for VisA (GPU, one-time)
    run_pcreg_loco.py            PC-Reg on MVTec LOCO AD
    run_pcreg_cls_loco.py        PC-Reg + CLS-gating on LOCO
    run_pcreg_cls_mvtec.py       PC-Reg + CLS-gating on MVTec AD
    run_pcreg_cls_visa.py        PC-Reg + CLS-gating on VisA
    run_fair_comparison.py       Controlled comparison (4 methods, 3 datasets)
    run_permutation_test.py      Synthetic permutation test
    run_pixel_metrics.py         Pixel-level AUROC and AUPRO
    run_statistical_tests.py     Wilcoxon + Cliff's delta
    run_benchmark.py             Computational complexity analysis
    generate_acid_figure.py      Permutation test diagram
    generate_heatmaps.py         Anomaly localization heatmaps
  results/                       Pre-computed results (CSV/JSON)
  pyproject.toml                 Dependencies
```

## Pre-computed results

The `results/` directory contains all experimental results as CSV files, so the numbers in the paper can be verified without re-running the experiments. All scripts are resumable: they skip already-completed runs.

## Hardware

Experiments were run on:
- CPU: Intel Core i9
- GPU: NVIDIA RTX 4070 Ti (16 GB)
- RAM: 32 GB

Feature extraction requires GPU. All other experiments (Ridge regression, scoring, ablations) run on CPU in minutes.

## Configuration

All experiments use a single fixed configuration unless stated otherwise:
- Backbone: DINOv3-ViT-L/16, layer -6, resolution 448x448
- PCA: d=256, random_state=42
- Ridge: R=5, lambda=1.0
- Aggregation: 95th percentile
- Covariance: Ledoit-Wolf shrinkage

The pipeline is fully deterministic.
