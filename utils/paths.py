"""
utils/paths.py — Central REPO_ROOT + layout resolution (Phase 6 hygiene).

Replaces ad-hoc `Path(__file__).resolve().parent.parent` chains and scattered
sys.path bootstraps with one resolver. Directory names come from paths.yaml
so a layout rename touches a single file.

Usage:
    from utils.paths import REPO_ROOT, repo_path, ensure_repo_on_path
    triplets = repo_path("processed_triplets")
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml  # type: ignore[import-not-found]
except Exception:
    yaml = None  # type: ignore[assignment]


def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk up from `start` (default: this file) to the repo root.

    The root is the first ancestor containing one of the marker files, else
    the grandparent of utils/ (this file lives at <root>/utils/paths.py).
    Always returns an absolute, symlink-resolved path.
    """
    markers = ("pyproject.toml", ".git", "requirements.in", "paths.yaml")
    here = Path(start).resolve() if start else Path(__file__).resolve()
    for cand in (here, *here.parents):
        try:
            if any((cand / m).exists() for m in markers):
                return cand
        except OSError:
            continue
    return Path(__file__).resolve().parent.parent


REPO_ROOT: Path = _find_repo_root()


@lru_cache(maxsize=1)
def _layout() -> Dict[str, Dict[str, str]]:
    """Parse paths.yaml once; fall back to built-in defaults if unreadable."""
    defaults: Dict[str, Dict[str, str]] = {
        "dirs": {
            "backend": "backend",
            "ml_model": "ML_model",
            "evaluation": "evaluation",
            "scripts": "scripts",
            "utils": "utils",
            "tests": "tests",
            "processed_triplets": "data_preprocessing_pipeline/processed_triplets",
            "dynamic_runs": "data_preprocessing_pipeline/dynamic_runs",
        },
        "files": {},
    }
    if yaml is None:
        return defaults
    try:
        data = yaml.safe_load((REPO_ROOT / "paths.yaml").read_text(encoding="utf-8")) or {}
        merged = {**defaults, **{k: v for k, v in data.items() if isinstance(v, dict)}}
        merged["dirs"] = {**defaults["dirs"], **merged.get("dirs", {})}
        return merged
    except Exception:
        return defaults


def repo_path(key: str) -> Path:
    """Absolute resolved path for a paths.yaml `dirs:`/`files:` key."""
    layout = _layout()
    for section in ("dirs", "files"):
        if key in layout.get(section, {}):
            return (REPO_ROOT / layout[section][key]).resolve()
    raise KeyError(f"Unknown paths.yaml key: {key!r}")


def repo_dir_names() -> List[str]:
    """Sorted paths.yaml `dirs:` keys (for audits/tests)."""
    return sorted(_layout().get("dirs", {}).keys())


def ensure_repo_on_path(*subdirs: str) -> List[str]:
    """Idempotently prepend REPO_ROOT (+ optional subdirs) to sys.path.

    Returns the entries added. Replaces the copy-pasted sys.path bootstrap
    blocks across evaluation/scripts entry points.
    """
    added: List[str] = []
    roots = [REPO_ROOT, *(REPO_ROOT / s for s in subdirs)]
    for root in roots:
        entry = str(root)
        if entry not in sys.path:
            sys.path.insert(0, entry)
            added.append(entry)
    return added
