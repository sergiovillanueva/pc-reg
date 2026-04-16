"""Small helpers for reviewer-friendly, reproducible scripts.

Goal: fail fast with actionable messages when large artifacts (datasets/features)
are not present, and make paths independent of the current working directory.
"""

from __future__ import annotations

from pathlib import Path


def _fmt_missing(p: Path) -> str:
    kind = "dir" if p.suffix == "" else "path"
    if p.exists():
        return f"- {p} (exists)"
    return f"- {p} (missing)"


def repo_root_from(file: str | Path) -> Path:
    """Return repository root assuming this file lives in <repo>/scripts/."""
    p = Path(file).resolve()
    return p.parents[1]


def require_dir(path: Path, hint: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"[pc-reg] Missing directory: {path}\n\n{hint}\n")


def require_file(path: Path, hint: str) -> None:
    if not path.is_file():
        raise SystemExit(f"[pc-reg] Missing file: {path}\n\n{hint}\n")


def skip_if_missing(
    *,
    required: list[Path],
    precomputed: list[Path] | None = None,
    what: str,
    reproduce_hint: str,
) -> bool:
    """Reviewer-friendly guard for optional large artifacts.

    Returns True if the caller should skip/return early.

    Behavior:
    - If all `required` exist: return False.
    - If something is missing and at least one `precomputed` exists: print a short
      message pointing to the precomputed artifacts; return True.
    - If something is missing and nothing precomputed exists: exit cleanly (no traceback)
      with instructions to reproduce.
    """

    missing = [p for p in required if not p.exists()]
    if not missing:
        return False

    precomputed = precomputed or []
    available = [p for p in precomputed if p.exists()]

    if available:
        joined = "\n".join(f"  - {p}" for p in available)
        print(
            "[pc-reg] Missing required artifacts for this script; skipping computation.\n"
            f"Missing ({what}):\n" + "\n".join(_fmt_missing(p) for p in missing) + "\n\n"
            "Precomputed outputs are available at:\n"
            f"{joined}\n\n"
            "To reproduce from scratch:\n"
            f"{reproduce_hint}\n",
            flush=True,
        )
        return True

    raise SystemExit(
        "[pc-reg] Missing required artifacts for this script.\n"
        f"Missing ({what}):\n" + "\n".join(_fmt_missing(p) for p in missing) + "\n\n"
        "To reproduce from scratch:\n"
        f"{reproduce_hint}\n"
    )
