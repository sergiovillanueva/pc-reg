"""Print the tables of the paper from the per-category result files.

  python scripts/make_tables.py                  # from results/paper (the numbers in the manuscript)
  python scripts/make_tables.py --source reproduced
  python scripts/make_tables.py --compare        # reproduced vs paper, largest difference per method
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAIN = ["patchcore_p95", "patchcore_max", "padim_cal", "meansub_cal", "uniform_cal", "cls_cal", "dual_final"]
NAMES = {"patchcore_p95": "PatchCore (P95)", "patchcore_max": "PatchCore (max)", "padim_cal": "PaDiM",
         "meansub_cal": "MeanSub", "uniform_cal": "PC-Reg", "cls_cal": "PC-Reg_CLS", "dual_final": "PC-Reg_Dual"}


def load(source: str) -> pd.DataFrame:
    files = [RES / "paper" / "percat.csv"] if source == "paper" else sorted((RES / "reproduced").glob("percat_*.csv"))
    if not files:
        sys.exit(f"no per-category results in results/{source}")
    return pd.concat([pd.read_csv(f, sep=";") for f in files]).drop_duplicates(["backbone", "dataset", "category", "method"], keep="last")


def summary(d: pd.DataFrame, backbone: str) -> pd.DataFrame:
    g = d[d.backbone == backbone]
    t = g.pivot_table(index="method", columns="dataset", values="auroc", aggfunc="mean") * 100
    t["all32"] = g[g.dataset != "btad"].groupby("method").auroc.mean() * 100
    lo = g[g.dataset == "loco"].groupby("method")[["logical", "structural"]].mean() * 100
    return t.join(lo).rename(columns={"loco": "LOCO", "logical": "LOCO log", "structural": "LOCO str",
                                      "mvtec": "MVTec AD", "visa": "VisA", "all32": "All 32", "btad": "BTAD"})


def wil(d, a, b, backbone="vitL"):
    g = d[(d.backbone == backbone) & (d.dataset != "btad")].pivot_table(index=["dataset", "category"], columns="method", values="auroc")
    x = g[a] - g[b]
    return f"{a} vs {b}: {100 * x.mean():+.2f} points, {(x > 1e-9).sum()} up / {(x < -1e-9).sum()} down, Wilcoxon p = {wilcoxon(g[a], g[b]).pvalue:.2g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="paper", choices=["paper", "reproduced"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    if args.compare:
        p, r = load("paper"), load("reproduced")
        m = p.merge(r, on=["backbone", "dataset", "category", "method"], suffixes=("_paper", "_new"))
        m["diff"] = (m.auroc_new - m.auroc_paper).abs() * 100
        print(f"{len(m)} (backbone, category, method) pairs compared; largest |difference| in AUROC points:")
        print(m.groupby(["backbone", "method"])["diff"].max().round(3).unstack(0).to_string())
        for bb in m.backbone.unique():
            sp, sr = summary(p, bb), summary(r, bb)
            common = sp.index.intersection(sr.index)
            print(f"\n{bb}: largest difference in the dataset means: {(sp.loc[common] - sr.loc[common]).abs().max().max():.3f} points")
        return
    d = load(args.source)
    cols = ["LOCO", "LOCO log", "LOCO str", "MVTec AD", "VisA", "All 32", "BTAD"]
    s = summary(d, "vitL")
    print("Table 1 (ViT-L/16): image AUROC (%)")
    print(s.loc[[m for m in MAIN if m in s.index], [c for c in cols if c in s]].rename(index=NAMES).round(1).to_string())
    if (d.backbone == "vitH").any():
        sh = summary(d, "vitH")
        print("\nTable 1 (ViT-H+/16)")
        print(sh.loc[["cls_cal", "dual_final"], [c for c in cols if c in sh]].rename(index=NAMES).round(1).to_string())
    comp = ["uniform_raw", "cls_raw", "uniform_cal", "cls_cal", "quad", "gap", "dual_paper", "dual_raw+quad",
            "dual_cal+gap", "dual_uniform", "dual_cal+gap+quad", "dual_final"]
    print("\nTable 2 (components) and Table A5 (fusion weight)")
    print(s.loc[[m for m in comp + ["dual_a0.7", "dual_a0.3", "dual_a0.2", "dual_maxz"] if m in s.index],
                ["LOCO", "LOCO log", "MVTec AD", "VisA", "All 32"]].round(1).to_string())
    print("\nPaired tests over the 32 categories (ViT-L/16)")
    for a, b in [("dual_final", "patchcore_max"), ("dual_final", "patchcore_p95"), ("dual_final", "padim_cal"),
                 ("dual_final", "meansub_cal"), ("uniform_cal", "padim_cal"), ("uniform_cal", "meansub_cal"),
                 ("cls_cal", "uniform_cal"), ("dual_final", "cls_cal")]:
        print("  " + wil(d, a, b))
    print("\nTables A1-A3: per-category AUROC (%), ViT-L/16")
    t = d[(d.backbone == "vitL") & d.method.isin(MAIN)].pivot_table(index=["dataset", "category"], columns="method", values="auroc") * 100
    print(t[[m for m in MAIN if m in t]].rename(columns=NAMES).round(1).to_string())
    fs = RES / args.source / "fewshot.csv"
    if fs.exists():
        f = pd.read_csv(fs, sep=";", dtype={"n_train": str})
        print("\nTable A6: MVTec LOCO AD with N training images (mean over categories and seeds)")
        for col in ("auroc", "logical"):
            print(f"  {col}:")
            print((f.groupby(["method", "n_train"])[col].mean().unstack() * 100)[["10", "25", "50", "100", "200", "full"]].round(1).to_string())
    pt = RES / args.source / "perturbation.csv"
    if pt.exists():
        p = pd.read_csv(pt, sep=";")
        sw = p[p["mode"] == "swap"].groupby("method").auroc.mean() * 100
        tr = p[p["mode"] == "transplant"].groupby(["method", "k"]).auroc.mean().unstack() * 100
        print("\nTable 3: perturbation tests, AUROC (%) original vs perturbed (swap: mean over k)")
        print(pd.concat([sw.rename("swap"), tr.add_prefix("transplant k=")], axis=1).round(1).to_string())


if __name__ == "__main__":
    main()
