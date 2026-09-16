"""
test_phase6_hygiene.py — Phase 6 coverage: shared splits, central paths,
canonical GSDs/aliases, JSON-safe sanitize, and single-source dep pins.
"""

import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT))


def test_grouped_split_matches_inline_reference_and_persists():
    from sklearn.model_selection import GroupShuffleSplit
    from splits import grouped_train_test_split, persist_group_splits

    rng = np.random.default_rng(0)
    groups = np.array([f"g{i // 5}" for i in range(100)], dtype=object)
    y = (rng.random(100) < 0.4).astype(np.int64)
    gss = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
    for tr_i, te_i in gss.split(groups, y, groups=groups):
        if len(np.unique(y[tr_i])) >= 2 and len(np.unique(y[te_i])) >= 2:
            ref = (sorted(set(groups[tr_i])), sorted(set(groups[te_i])))
            break
    got = grouped_train_test_split(groups, y, seed=42)
    assert list(got[0]) == list(ref[0]) and list(got[1]) == list(ref[1])

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tr_p, te_p = persist_group_splits(td, got[0], got[1])
        assert tr_p.name == "train_groups.json" and te_p.name == "test_groups.json"
        import json

        assert json.loads(tr_p.read_text()) == list(got[0])

    import pytest

    with pytest.raises(ValueError):
        grouped_train_test_split(["a", "a", "b"], [0, 0, 1], seed=42, n_splits=2)


def test_central_paths_resolve():
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    from utils.paths import REPO_ROOT as RR, repo_path, ensure_repo_on_path, repo_dir_names

    assert RR == REPO_ROOT
    assert repo_path("processed_triplets").name == "processed_triplets"
    assert repo_path("ml_model").name == "ML_model"
    assert set(repo_dir_names()) >= {"backend", "ml_model", "evaluation", "scripts"}
    # Postconditions only: other test modules may already have bootstrapped
    # these entries, so never assume a pristine sys.path here.
    ensure_repo_on_path("scripts")
    assert str(REPO_ROOT / "scripts") in sys.path
    assert str(REPO_ROOT) in sys.path
    n = len(sys.path)
    ensure_repo_on_path("scripts")
    assert len(sys.path) == n, "ensure_repo_on_path must be idempotent"


def test_canonical_gsd_and_aliases():
    from config import SENSOR_GSD_MAP, get_sensor_gsd, IIRS_GSD, LRO_NAC_GSD

    assert IIRS_GSD == 70.0 and LRO_NAC_GSD == 0.9
    assert SENSOR_GSD_MAP["TMC2"] == 5.0 and SENSOR_GSD_MAP["NAC"] == 0.9
    assert get_sensor_gsd("TMC2") == 5.0
    assert get_sensor_gsd("tmc-2") == 5.0
    assert get_sensor_gsd("NAC") == 0.9
    assert get_sensor_gsd("lro nac") == 0.9
    assert get_sensor_gsd("iirs") == 70.0
    assert get_sensor_gsd("NOPE", 1.0) == 1.0


def test_sanitize_maps_nonfinite_to_none():
    from matcher_cfog import sanitize_for_json
    import json

    out = sanitize_for_json({"a": float("inf"), "b": float("nan"), "c": [1.0, 2.0]})
    assert out == {"a": None, "b": None, "c": [1.0, 2.0]}
    json.dumps(out)


def test_metrics_failure_paths_emit_none_not_inf():
    from metrics import verify_transformation_quality

    bad = verify_transformation_quality(np.eye(2), (512, 512))
    assert bad["is_valid"] is False
    assert bad["condition_number"] is None
    import json

    json.dumps(bad)


def test_requirements_single_source():
    """Every == pin in requirements.txt must be chosen in requirements.in."""
    def pins(path):
        out = {}
        for line in Path(path).read_text().splitlines():
            line = line.split("#")[0].strip()
            m = re.match(r"([A-Za-z0-9_.\-\[\]]+)==([^;\s]+)", line)
            if m:
                out[m.group(1).lower().split("[")[0]] = m.group(2)
        return out

    src = pins(REPO_ROOT / "requirements.in")
    txt = pins(REPO_ROOT / "requirements.txt")
    assert txt, "requirements.txt has no pins"
    for name, ver in txt.items():
        assert name in src, f"{name} pinned in .txt but not chosen in .in"
        assert src[name] == ver, f"{name}: .txt has {ver}, .in chooses {src[name]}"
