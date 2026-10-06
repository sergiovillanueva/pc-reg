"""Image lists per dataset, category and split (same splits as the paper)."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIRS, DATA_ROOT, SPLITS, feature_dir

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG"}


def _images(folder: Path) -> list[str]:
    return sorted(str(p) for p in folder.rglob("*") if p.is_file() and p.suffix in IMG_EXT)


def _anomaly_dirs(test_dir: Path) -> list[Path]:
    return sorted(d for d in test_dir.iterdir() if d.is_dir() and d.name != "good")


def image_splits(dataset: str, category: str, data_root: Path = DATA_ROOT) -> dict[str, list[str]]:
    root = data_root / DATA_DIRS[dataset]
    if dataset == "loco":
        c = root / category
        return {"train": _images(c / "train" / "good"),
                "test_good": _images(c / "test" / "good"),
                "test_logical": _images(c / "test" / "logical_anomalies"),
                "test_structural": _images(c / "test" / "structural_anomalies")}
    if dataset in ("mvtec", "btad"):
        c = root / category
        return {"train": _images(c / "train" / "good"),
                "test_good": _images(c / "test" / "good"),
                "test_anomaly": [p for d in _anomaly_dirs(c / "test") for p in _images(d)]}
    if dataset == "visa":  # official one-class split, split_csv/1cls.csv
        df = pd.read_csv(root / "split_csv" / "1cls.csv")
        df = df[df.object == category]
        path = lambda r: str(root / r)
        return {"train": [path(r) for r in df[(df.split == "train") & (df.label == "normal")].image],
                "test_good": [path(r) for r in df[(df.split == "test") & (df.label == "normal")].image],
                "test_anomaly": [path(r) for r in df[(df.split == "test") & (df.label == "anomaly")].image]}
    raise ValueError(f"unknown dataset: {dataset}")


def load_category(dataset: str, category: str, backbone: str = "vitL") -> tuple[dict, dict]:
    """Cached patch tokens {split: (N, P, D) float32} and CLS attention {split: (N, P) float32}."""
    d = feature_dir(dataset, category, backbone)
    if not d.exists():
        raise FileNotFoundError(f"{d} not found: run scripts/extract_features.py --dataset {dataset} first")
    feats = {s: np.load(d / f"{s}_features.npy") for s in SPLITS[dataset]}
    attn = {s: np.load(d / f"{s}_cls_attn.npy").astype(np.float32) for s in SPLITS[dataset]}
    return feats, attn
