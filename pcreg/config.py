"""Fixed configuration of the paper (one setting for every category; nothing is tuned per category or dataset)."""
from pathlib import Path

PCA_DIM = 256          # d
RADIUS = 5             # Chebyshev radius of the neighborhood (up to 120 neighbors)
LAMBDA = 1.0           # ridge penalty
PERCENTILE = 95        # image score = P95 over positions
ALPHA = 0.5            # weight of the positional branch in the fusion
LAYER_IDX = -6         # transformer layer for patch tokens and CLS attention
PCA_SEED = 42

BACKBONES = {
    "vitL": {"hf_name": "facebook/dinov3-vitl16-pretrain-lvd1689m", "resolution": 448, "grid": 28, "feat_dim": 1024},
    "vitH": {"hf_name": "facebook/dinov3-vith16plus-pretrain-lvd1689m", "resolution": 512, "grid": 32, "feat_dim": 1280},
}

CATEGORIES = {
    "loco": ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"],
    "mvtec": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
              "tile", "toothbrush", "transistor", "wood", "zipper"],
    "visa": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3",
             "pcb4", "pipe_fryum"],
    "btad": ["01", "02", "03"],
}
SPLITS = {
    "loco": ["train", "test_good", "test_logical", "test_structural"],
    "mvtec": ["train", "test_good", "test_anomaly"],
    "visa": ["train", "test_good", "test_anomaly"],
    "btad": ["train", "test_good", "test_anomaly"],
}
DATA_DIRS = {"loco": "mvtec_loco_AD", "mvtec": "mvtec_AD", "visa": "VisA", "btad": "btad"}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"          # datasets, see README
FEATURE_ROOT = REPO_ROOT / "features"   # cache written by scripts/extract_features.py


def feature_dir(dataset: str, category: str, backbone: str = "vitL") -> Path:
    return FEATURE_ROOT / backbone / dataset / category
