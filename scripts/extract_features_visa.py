"""
Extract DINOv3 Features for VisA
==================================
Same backbone and settings as the LOCO/MVTec AD extraction scripts.

VisA structure: {category}/Data/Images/{Normal,Anomaly}/
Split defined by: split_csv/1cls.csv

Usage:
  uv run scripts/extract_features_visa.py

Backbone: DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m)
Output: features/visa/{category}/
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import gc
import time
import warnings
from pathlib import Path

from _repro_utils import repo_root_from, require_dir, require_file

warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoModel
from torchvision import transforms

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
REPO_ROOT = repo_root_from(__file__)
DATA_ROOT = REPO_ROOT / "data" / "VisA"
CACHE_DIR = REPO_ROOT / "features" / "visa"

RESOLUTION = 448
LAYER_IDX = -6
BATCH_SIZE = 8

CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3",
    "pcb4", "pipe_fryum",
]


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class DINOv3Extractor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        print(f"[FEAT] Loading DINOv3 on {device}...")
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

        self.patch_size = 16
        self.grid_size = RESOLUTION // self.patch_size  # 28
        self.num_patches = self.grid_size * self.grid_size  # 784
        self.num_register = getattr(self.model.config, "num_register_tokens", 4)
        self.feature_dim = 1024

        self.transform = transforms.Compose([
            transforms.Resize((RESOLUTION, RESOLUTION)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"[FEAT] Grid: {self.grid_size}x{self.grid_size}, "
              f"patches: {self.num_patches}, dim: {self.feature_dim}")

    @torch.no_grad()
    def extract_batch(self, image_paths: list[str],
                      batch_size: int = BATCH_SIZE) -> np.ndarray:
        all_features = []
        start_idx = 1 + self.num_register

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [self.transform(Image.open(p).convert("RGB"))
                      for p in batch_paths]
            batch = torch.stack(images).to(self.device)

            with torch.autocast(device_type="cuda",
                                enabled=self.device.startswith("cuda")):
                outputs = self.model(batch, output_hidden_states=True)

            features = outputs.hidden_states[LAYER_IDX][
                :, start_idx:start_idx + self.num_patches, :]
            all_features.append(features.float().cpu().numpy())

            del batch, outputs, features

        return np.concatenate(all_features, axis=0)


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_visa_split(category: str) -> dict:
    """Load VisA paths using the official 1cls.csv split.

    Returns dict with keys:
        train: list of paths (normal training images)
        test_good: list of paths (normal test images)
        test_anomaly: list of paths (anomalous test images)
    """
    csv_path = DATA_ROOT / "split_csv" / "1cls.csv"
    df = pd.read_csv(csv_path)

    # Filter for this category
    cat_df = df[df["object"] == category]

    data = {"train": [], "test_good": [], "test_anomaly": []}

    for _, row in cat_df.iterrows():
        img_path = str(DATA_ROOT / row["image"])
        split = row["split"]
        label = row["label"]

        if split == "train" and label == "normal":
            data["train"].append(img_path)
        elif split == "test" and label == "normal":
            data["test_good"].append(img_path)
        elif split == "test" and label == "anomaly":
            data["test_anomaly"].append(img_path)

    # Sort for reproducibility
    for k in data:
        data[k] = sorted(data[k])

    return data


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Extract DINOv3 Features for VisA")
    print(f"  Categories: {len(CATEGORIES)} ({CATEGORIES})")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {CACHE_DIR}")
    print("=" * 70)

    require_dir(
        DATA_ROOT,
        hint=(
            "Place VisA under pc-reg/data/VisA (folder name must match).\n"
            "Then re-run: uv run scripts/extract_features_visa.py"
        ),
    )
    require_file(
        DATA_ROOT / "split_csv" / "1cls.csv",
        hint=(
            "Missing official VisA split CSV: pc-reg/data/VisA/split_csv/1cls.csv\n"
            "Then re-run: uv run scripts/extract_features_visa.py"
        ),
    )

    extractor = DINOv3Extractor(DEVICE)

    for category in CATEGORIES:
        cat_cache = CACHE_DIR / category
        cat_cache.mkdir(parents=True, exist_ok=True)

        marker = cat_cache / "DONE"
        if marker.exists():
            print(f"\n[SKIP] {category} already cached")
            continue

        print(f"\n{'='*60}")
        print(f"Category: {category}")

        data = load_visa_split(category)
        print(f"  train: {len(data['train'])} images")
        print(f"  test_good: {len(data['test_good'])} images")
        print(f"  test_anomaly: {len(data['test_anomaly'])} images")

        for split_name, paths in data.items():
            if len(paths) == 0:
                print(f"  [WARN] {split_name}: 0 images, skipping")
                continue

            t0 = time.time()
            features = extractor.extract_batch(paths)
            elapsed = time.time() - t0

            np.save(str(cat_cache / f"{split_name}_features.npy"), features)

            with open(cat_cache / f"{split_name}_paths.txt", "w",
                      encoding="utf-8") as f:
                for p in paths:
                    f.write(p + "\n")

            print(f"  {split_name}: {features.shape} in {elapsed:.1f}s")

        marker.write_text(f"cached at {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [OK] {category} cached")

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
