"""
Extract DINOv3 Features for MVTec AD
======================================
Same backbone and settings as the LOCO extraction script.
MVTec AD structure: train/good + test/good + test/<defect_type>.
All defect types are merged into a single "test_anomaly" split.

Usage:
  uv run scripts/extract_features_mvtec.py

Backbone: DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m)
Output: features/mvtec/{category}/
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import gc
import time
import warnings
from pathlib import Path

from _repro_utils import repo_root_from, require_dir

warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel
from torchvision import transforms

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
REPO_ROOT = repo_root_from(__file__)
DATA_ROOT = REPO_ROOT / "data" / "mvtec_AD"
CACHE_DIR = REPO_ROOT / "features" / "mvtec"

RESOLUTION = 448
LAYER_IDX = -6
BATCH_SIZE = 16

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
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

def get_image_paths(directory: Path) -> list[str]:
    """Get sorted image paths from a directory."""
    paths = sorted(
        [str(p) for p in directory.rglob("*.png")] +
        [str(p) for p in directory.rglob("*.jpg")]
    )
    return paths


def load_mvtec_category(category: str) -> dict:
    """Load all image paths for a MVTec AD category, split by type.

    MVTec AD: train/good + test/good + test/<defect_types>.
    All defect types merged into 'test_anomaly'.
    """
    cat_dir = DATA_ROOT / category

    train_dir = cat_dir / "train" / "good"
    test_dir = cat_dir / "test"

    data = {
        "train": get_image_paths(train_dir),
        "test_good": get_image_paths(test_dir / "good"),
        "test_anomaly": [],
    }

    # Merge all defect subdirectories
    for subdir in sorted(test_dir.iterdir()):
        if subdir.is_dir() and subdir.name != "good":
            data["test_anomaly"].extend(get_image_paths(subdir))

    return data


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Extract DINOv3 Features for MVTec AD")
    print(f"  Categories: {len(CATEGORIES)} ({CATEGORIES})")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {CACHE_DIR}")
    print("=" * 70)

    require_dir(
        DATA_ROOT,
        hint=(
            "Place MVTec AD under pc-reg/data/mvtec_AD (folder name must match).\n"
            "Then re-run: uv run scripts/extract_features_mvtec.py"
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

        data = load_mvtec_category(category)
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
