"""Per-category image AUROC of every detector and variant in the paper, on the same cached features.

Methods (column `method` of the output, as in results/paper/percat.csv):
  patchcore_max, patchcore_p95           PatchCore, exact k = 1, image score max or P95       (Table 1)
  padim_cal, meansub_cal                 per-position Gaussians, standardized P95              (Table 1)
  uniform_cal, cls_cal                   PC-Reg and PC-Reg_CLS, standardized P95               (Tables 1, 2)
  dual_final                             PC-Reg_Dual = 0.5 z(cls_cal) + 0.5 z(quad)            (Tables 1, 2)
  *_raw, quad, gap, dual_paper, dual_cal+gap, dual_raw+quad, dual_cal+gap+quad, dual_uniform    (Table 2)
  dual_a0.2, dual_a0.3, dual_a0.7, dual_maxz                                                     (Table A5)
z-statistics come from the in-sample training scores. Resumable per category.

Usage:
  python scripts/run_main.py                          # ViT-L/16, MVTec LOCO AD, MVTec AD, VisA, BTAD
  python scripts/run_main.py --backbone vitH --datasets loco mvtec visa
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcreg.config import BACKBONES, CATEGORIES, PERCENTILE  # noqa: E402
from pcreg.data import load_category  # noqa: E402
from pcreg.models import (GlobalMeanScore, PCReg, PatchCore, PerPositionGaussian, QuadrantScore, evaluate,  # noqa: E402
                          fit_pca, project, standardized_p95, zscore)


def scores_for_category(ds: str, cat: str, backbone: str, with_baselines: bool) -> dict:
    grid = BACKBONES[backbone]["grid"]
    raw, attn = load_category(ds, cat, backbone)
    pca = fit_pca(raw["train"])
    f = {s: project(pca, a) for s, a in raw.items()}
    del raw
    splits = list(f)
    sc = {}

    def positional(name, maps):
        sc[f"{name}_cal"] = {s: standardized_p95(maps[s], maps["train"]) for s in splits}
        sc[f"{name}_raw"] = {s: np.percentile(maps[s], PERCENTILE, axis=1) for s in splits}

    variants = [("cls", "cls")] + ([("uniform", "uniform")] if with_baselines else [])
    for name, weighting in variants:
        reg = PCReg(grid, weighting).fit(f["train"], attn["train"])
        positional(name, {s: reg.distance_maps(f[s], attn[s]) for s in splits})
    q, g = QuadrantScore(grid).fit(f["train"]), GlobalMeanScore().fit(f["train"])
    sc["quad"] = {s: q.score(f[s]) for s in splits}
    sc["gap"] = {s: g.score(f[s]) for s in splits}
    if with_baselines:
        for name, ms in [("padim", False), ("meansub", True)]:
            m = PerPositionGaussian(grid, mean_subtraction=ms).fit(f["train"])
            positional(name, {s: (m.train_maps if s == "train" else m.distance_maps(f[s])) for s in splits})
        pc = PatchCore().fit(f["train"])
        tests = [s for s in splits if s != "train"]
        maps = {s: pc.distance_maps(f[s]) for s in tests}
        sc["patchcore_max"] = {s: maps[s].max(axis=1) for s in tests}
        sc["patchcore_p95"] = {s: np.percentile(maps[s], PERCENTILE, axis=1) for s in tests}

    tests = [s for s in splits if s != "train"]
    z = {k: {s: zscore(v[s], v["train"]) for s in tests} for k, v in sc.items() if "train" in v}
    out = {k: {s: v[s] for s in tests} for k, v in sc.items()}
    fuse = lambda a, b, w=0.5: {s: w * z[a][s] + (1 - w) * z[b][s] for s in tests}
    out["dual_final"] = fuse("cls_cal", "quad")
    out["dual_paper"] = fuse("cls_raw", "gap")
    out["dual_cal+gap"] = fuse("cls_cal", "gap")
    out["dual_raw+quad"] = fuse("cls_raw", "quad")
    out["dual_cal+gap+quad"] = {s: 0.5 * z["cls_cal"][s] + 0.25 * z["gap"][s] + 0.25 * z["quad"][s] for s in tests}
    if with_baselines:
        out["dual_uniform"] = fuse("uniform_cal", "quad")
    for a in (0.2, 0.3, 0.7):
        out[f"dual_a{a}"] = fuse("cls_cal", "quad", a)
    out["dual_maxz"] = {s: np.maximum(z["cls_cal"][s], z["quad"][s]) for s in tests}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vitL", choices=["vitL", "vitH"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--categories", nargs="+", default=None, help="subset of categories (default: all)")
    ap.add_argument("--out", default=str(ROOT / "results" / "reproduced"))
    args = ap.parse_args()
    datasets = args.datasets or (["loco", "mvtec", "visa", "btad"] if args.backbone == "vitL" else ["loco", "mvtec", "visa"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(datasets) + ("_" + "_".join(args.categories) if args.categories else "")
    csv = out_dir / f"percat_{args.backbone}_{tag}.csv"
    done = set()
    if csv.exists():
        done = set(map(tuple, pd.read_csv(csv, sep=";")[["dataset", "category"]].drop_duplicates().values))
    for ds in datasets:
        for cat in CATEGORIES[ds]:
            if (ds, cat) in done or (args.categories and cat not in args.categories):
                continue
            t0 = time.time()
            res = scores_for_category(ds, cat, args.backbone, with_baselines=args.backbone == "vitL")
            rows = [dict(backbone=args.backbone, dataset=ds, category=cat, method=m, **evaluate(v)) for m, v in res.items()]
            pd.DataFrame(rows).to_csv(csv, sep=";", index=False, mode="a", header=not csv.exists())
            dual = next(r for r in rows if r["method"] == "dual_final")
            print(f"[{ds}] {cat}: PC-Reg_Dual AUROC {100 * dual['auroc']:.1f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
