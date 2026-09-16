"""
evaluation/splits.py — Shared grouped train/test splitting (Phase 6 hygiene).

Dedupes the identical GroupShuffleSplit block previously copy-pasted across
eval_heldout_patch32_end_to_end.py and eval_weighted_registration.py: same
(n_splits=10, test_size=0.25) budget, same both-classes-in-each-side guard,
same sorted group-list outputs, same train/test_groups.json persistence.
One implementation => the published 54/18-style splits cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def grouped_train_test_split(
    groups: Sequence[str],
    y: Sequence[int],
    X: Optional[np.ndarray] = None,
    *,
    seed: int = 42,
    n_splits: int = 10,
    test_size: float = 0.25,
) -> Tuple[List[str], List[str]]:
    """Group-aware split with both-classes-in-each-side guard.

    Returns (train_groups, test_groups) as sorted unique group-id lists.
    X is accepted for API compatibility (GroupShuffleSplit ignores it);
    callers that only have groups/y pass X=None.
    Raises ValueError if no qualifying split exists (previously an
    UnboundLocalError on `selected_split`).
    """
    groups_arr = np.asarray(list(groups), dtype=object)
    y_arr = np.asarray(list(y))
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    probe = groups_arr if X is None else np.asarray(X)
    for tr_i, te_i in splitter.split(probe, y_arr, groups=groups_arr):
        if len(np.unique(y_arr[tr_i])) >= 2 and len(np.unique(y_arr[te_i])) >= 2:
            return (
                sorted(list(set(groups_arr[tr_i]))),
                sorted(list(set(groups_arr[te_i]))),
            )
    raise ValueError(
        f"No GroupShuffleSplit(n_splits={n_splits}, test_size={test_size}) draw "
        "holds both classes on each side; cannot form an honest split."
    )


def persist_group_splits(
    eval_dir: str | Path,
    train_groups: Sequence[str],
    test_groups: Sequence[str],
) -> Tuple[Path, Path]:
    """Write train_groups.json / test_groups.json for auditability (Phase 2.4)."""
    out = Path(eval_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr_path = out / "train_groups.json"
    te_path = out / "test_groups.json"
    tr_path.write_text(json.dumps(list(train_groups), indent=2), encoding="utf-8")
    te_path.write_text(json.dumps(list(test_groups), indent=2), encoding="utf-8")
    return tr_path, te_path
