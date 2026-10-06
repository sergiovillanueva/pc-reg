"""Cache DINOv3 patch tokens and CLS attention (layer -6) for every image of a dataset.

Writes features/<backbone>/<dataset>/<category>/{split}_features.npy (float32), {split}_cls_attn.npy (float16)
and {split}_paths.txt. Resumable: categories with a DONE marker are skipped.

Usage:
  python scripts/extract_features.py --dataset loco            # MVTec LOCO AD, ViT-L/16 at 448 px
  python scripts/extract_features.py --dataset all
  python scripts/extract_features.py --dataset mvtec --backbone vitH   # ViT-H+/16 at 512 px
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcreg.config import CATEGORIES, feature_dir  # noqa: E402
from pcreg.data import image_splits  # noqa: E402
from pcreg.features import FeatureExtractor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CATEGORIES) + ["all"])
    ap.add_argument("--backbone", default="vitL", choices=["vitL", "vitH"])
    args = ap.parse_args()
    ext = FeatureExtractor(args.backbone)
    for ds in (list(CATEGORIES) if args.dataset == "all" else [args.dataset]):
        for cat in CATEGORIES[ds]:
            out = feature_dir(ds, cat, args.backbone)
            if (out / "DONE").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            for split, paths in image_splits(ds, cat).items():
                (out / f"{split}_paths.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
                np.save(out / f"{split}_features.npy", ext.patch_tokens(paths))
                np.save(out / f"{split}_cls_attn.npy", ext.cls_attention(paths))
                print(f"  {ds}/{cat}/{split}: {len(paths)} images", flush=True)
            (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
            print(f"[{ds}] {cat} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
