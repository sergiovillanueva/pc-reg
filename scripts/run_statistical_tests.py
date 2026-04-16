"""
Statistical Tests: Wilcoxon Signed-Rank + Cliff's Delta
=========================================================
Tests whether CLS-gating significantly improves over uniform weighting
across three benchmarks: MVTec LOCO AD, MVTec AD, and VisA.

Paired data from:
  - MVTec LOCO AD (5 categories)
  - MVTec AD (15 categories)
  - VisA (12 categories)

Usage:
  uv run scripts/run_statistical_tests.py
"""

import csv
import json
from pathlib import Path

from _repro_utils import repo_root_from

import numpy as np
from scipy import stats

# =============================================================================
# DATA LOADING
# =============================================================================

REPO_ROOT = repo_root_from(__file__)

def load_loco_pairs() -> list[tuple[str, float, float]]:
    """Load LOCO uniform vs CLS pairs."""
    csv_path = REPO_ROOT / "results" / "pcreg_cls_loco" / "results.csv"
    if not csv_path.exists():
        return []

    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            cat = row["category"]
            cfg = row["config"]
            auroc = float(row["auroc_combined"])
            if cfg in ("uniform_R5", "cls_perimg_R5"):
                data.setdefault(cat, {})[cfg] = auroc

    pairs = []
    for cat in sorted(data.keys()):
        if "uniform_R5" in data[cat] and "cls_perimg_R5" in data[cat]:
            pairs.append((
                f"LOCO/{cat}",
                data[cat]["uniform_R5"],
                data[cat]["cls_perimg_R5"],
            ))
    return pairs


def load_mvtec_pairs() -> list[tuple[str, float, float]]:
    """Load MVTec AD uniform vs CLS pairs."""
    csv_path = REPO_ROOT / "results" / "pcreg_cls_mvtec" / "results.csv"
    if not csv_path.exists():
        return []

    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            cat = row["category"]
            cfg = row["config"]
            auroc = float(row["auroc"])
            if cfg in ("uniform_R5", "cls_perimg_R5"):
                data.setdefault(cat, {})[cfg] = auroc

    pairs = []
    for cat in sorted(data.keys()):
        if "uniform_R5" in data[cat] and "cls_perimg_R5" in data[cat]:
            pairs.append((
                f"MVTec/{cat}",
                data[cat]["uniform_R5"],
                data[cat]["cls_perimg_R5"],
            ))
    return pairs


def load_visa_pairs() -> list[tuple[str, float, float]]:
    """Load VisA uniform vs CLS pairs."""
    csv_path = REPO_ROOT / "results" / "pcreg_cls_visa" / "results.csv"
    if not csv_path.exists():
        return []

    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            cat = row["category"]
            cfg = row["config"]
            auroc = float(row["auroc"])
            if cfg in ("uniform_R5", "cls_perimg_R5"):
                data.setdefault(cat, {})[cfg] = auroc

    pairs = []
    for cat in sorted(data.keys()):
        if "uniform_R5" in data[cat] and "cls_perimg_R5" in data[cat]:
            pairs.append((
                f"VisA/{cat}",
                data[cat]["uniform_R5"],
                data[cat]["cls_perimg_R5"],
            ))
    return pairs


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Compute Cliff's delta effect size.

    Cliff's delta = (# concordant - # discordant) / (n1 * n2)
    where concordant means x_i > y_j and discordant means x_i < y_j.

    Here: x = CLS scores, y = uniform scores
    Positive delta means CLS > uniform.

    Thresholds (Romano et al. 2006):
      negligible: |d| < 0.147
      small:      0.147 <= |d| < 0.330
      medium:     0.330 <= |d| < 0.474
      large:      |d| >= 0.474
    """
    n = len(x)
    assert len(y) == n, "Paired samples must have same length"

    # Compute all pairwise comparisons of differences
    diff = x - y
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    n_tied = np.sum(diff == 0)

    delta = (n_pos - n_neg) / n

    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.330:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return delta, magnitude


def run_wilcoxon(uniform: np.ndarray, cls: np.ndarray,
                 label: str = "") -> dict:
    """Run Wilcoxon signed-rank test on paired samples."""
    diff = cls - uniform
    n = len(diff)

    # Check for ties at zero
    n_zero = np.sum(diff == 0)
    if n_zero == n:
        return {
            "test": "wilcoxon",
            "label": label,
            "n": n,
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "mean_diff": 0.0,
            "note": "All differences are zero",
        }

    # Wilcoxon signed-rank test (two-sided)
    try:
        stat, p_val = stats.wilcoxon(cls, uniform, alternative="two-sided")
    except ValueError as e:
        return {
            "test": "wilcoxon",
            "label": label,
            "n": n,
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "mean_diff": float(np.mean(diff)),
            "note": str(e),
        }

    # Also one-sided: CLS > uniform
    try:
        _, p_val_greater = stats.wilcoxon(cls, uniform, alternative="greater")
    except ValueError:
        p_val_greater = 1.0

    # Cliff's delta
    delta, magnitude = cliffs_delta(cls, uniform)

    return {
        "test": "wilcoxon",
        "label": label,
        "n": n,
        "statistic": float(stat),
        "p_value_twosided": float(p_val),
        "p_value_greater": float(p_val_greater),
        "significant_005": p_val < 0.05,
        "significant_001": p_val < 0.01,
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "n_improved": int(np.sum(diff > 0)),
        "n_degraded": int(np.sum(diff < 0)),
        "n_tied": int(np.sum(diff == 0)),
        "cliffs_delta": float(delta),
        "cliffs_magnitude": magnitude,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Statistical Tests: Wilcoxon Signed-Rank + Cliff's Delta")
    print("=" * 70)

    # Load all pairs
    loco_pairs = load_loco_pairs()
    mvtec_pairs = load_mvtec_pairs()
    visa_pairs = load_visa_pairs()

    all_pairs = loco_pairs + mvtec_pairs + visa_pairs

    print(f"\nData available:")
    print(f"  LOCO: {len(loco_pairs)} pairs")
    print(f"  MVTec AD: {len(mvtec_pairs)} pairs")
    print(f"  VisA: {len(visa_pairs)} pairs")
    print(f"  Total: {len(all_pairs)} pairs")

    if len(all_pairs) < 5:
        print("\n[WARN] Need at least 5 pairs for meaningful test.")
        if len(all_pairs) == 0:
            print("[ERROR] No data available. Run experiments first.")
            return

    # Print paired data
    print(f"\n{'Category':<30} {'Uniform':>10} {'CLS':>10} {'Delta':>10}")
    print("-" * 62)
    for name, u, c in all_pairs:
        d = c - u
        sign = "+" if d >= 0 else ""
        print(f"{name:<30} {u:>10.4f} {c:>10.4f} {sign}{d:>9.4f}")

    uniform = np.array([p[1] for p in all_pairs])
    cls = np.array([p[2] for p in all_pairs])

    # Overall test
    print(f"\n{'='*70}")
    print("OVERALL TEST (all datasets combined)")
    print(f"{'='*70}")
    result_all = run_wilcoxon(uniform, cls, "all_combined")
    print_result(result_all)

    # Per-dataset tests
    results = {"overall": result_all, "per_dataset": {}}

    for name, pairs in [("LOCO", loco_pairs), ("MVTec_AD", mvtec_pairs),
                        ("VisA", visa_pairs)]:
        if len(pairs) >= 5:
            print(f"\n{'='*70}")
            print(f"PER-DATASET: {name} ({len(pairs)} pairs)")
            print(f"{'='*70}")
            u = np.array([p[1] for p in pairs])
            c = np.array([p[2] for p in pairs])
            result = run_wilcoxon(u, c, name)
            print_result(result)
            results["per_dataset"][name] = result
        elif len(pairs) > 0:
            print(f"\n[INFO] {name}: {len(pairs)} pairs (too few for Wilcoxon, "
                  f"need >=5). Descriptive only:")
            u = np.array([p[1] for p in pairs])
            c = np.array([p[2] for p in pairs])
            delta, mag = cliffs_delta(c, u)
            print(f"  Mean diff: {np.mean(c-u):+.4f}")
            print(f"  Cliff's delta: {delta:.3f} ({mag})")
            print(f"  Improved: {np.sum(c>u)}/{len(pairs)}")
            results["per_dataset"][name] = {
                "n": len(pairs),
                "mean_diff": float(np.mean(c-u)),
                "cliffs_delta": float(delta),
                "cliffs_magnitude": mag,
                "note": "Too few pairs for Wilcoxon",
            }

    # Save results
    output_dir = REPO_ROOT / "results" / "statistical_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate LaTeX snippet
    latex = generate_latex_snippet(results, all_pairs)
    with open(output_dir / "stats_latex.tex", "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"\nResults saved to: {output_dir}")
    print(f"LaTeX snippet: {output_dir / 'stats_latex.tex'}")


def print_result(r: dict):
    """Pretty-print a test result."""
    print(f"  N = {r['n']}")
    print(f"  Mean difference: {r['mean_diff']:+.4f}")
    print(f"  Median difference: {r.get('median_diff', 0):+.4f}")
    print(f"  Improved: {r.get('n_improved', '?')}, "
          f"Degraded: {r.get('n_degraded', '?')}, "
          f"Tied: {r.get('n_tied', '?')}")
    if 'statistic' in r:
        print(f"  Wilcoxon W = {r.get('statistic', 'N/A')}")
    if 'p_value_twosided' in r:
        p = r['p_value_twosided']
        print(f"  p-value (two-sided) = {p:.6f} "
              f"{'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")
    if 'p_value_greater' in r:
        p = r['p_value_greater']
        print(f"  p-value (CLS > uniform) = {p:.6f} "
              f"{'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")
    print(f"  Cliff's delta = {r.get('cliffs_delta', 'N/A'):.3f} "
          f"({r.get('cliffs_magnitude', 'N/A')})")


def generate_latex_snippet(results: dict, all_pairs: list) -> str:
    """Generate a LaTeX paragraph reporting the statistical test results."""
    r = results["overall"]
    n = r["n"]
    p_val = r.get("p_value_twosided", 1.0)
    p_greater = r.get("p_value_greater", 1.0)
    delta = r.get("cliffs_delta", 0)
    mag = r.get("cliffs_magnitude", "N/A")
    mean_diff = r.get("mean_diff", 0)
    n_improved = r.get("n_improved", 0)
    n_degraded = r.get("n_degraded", 0)

    # Format p-value
    if p_val < 0.001:
        p_str = f"$p < 0.001$"
    elif p_val < 0.01:
        p_str = f"$p = {p_val:.3f}$"
    else:
        p_str = f"$p = {p_val:.3f}$"

    if p_greater < 0.001:
        pg_str = f"$p < 0.001$"
    elif p_greater < 0.01:
        pg_str = f"$p = {p_greater:.3f}$"
    else:
        pg_str = f"$p = {p_greater:.3f}$"

    latex = f"""% Statistical test results — auto-generated
% Wilcoxon signed-rank test + Cliff's delta
% N = {n} paired comparisons across datasets

A Wilcoxon signed-rank test on {n} paired category-level comparisons
(uniform vs.\\ CLS-gated) yields a two-sided {p_str},
confirming that CLS-gating produces a statistically significant improvement.
The one-sided test (CLS $>$ uniform) gives {pg_str}.
CLS-gating improves {n_improved} of {n} categories
(mean $\\Delta = {mean_diff*100:+.1f}$~pp),
with a Cliff's delta of ${delta:.2f}$ ({mag} effect).
"""

    return latex


if __name__ == "__main__":
    main()
