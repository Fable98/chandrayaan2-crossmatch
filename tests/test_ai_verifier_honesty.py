"""
tests/test_ai_verifier_honesty.py — Step 11: the verifier must not learn from
its own downstream (RANSAC labels are circular supervision).

Contracts:
  * train_ai_verifier fails LOUDLY (exit 2) on single-class label sets.
  * train_ai_verifier fails LOUDLY (exit 2) when no hand labels exist
    (RANSAC-only files are refused without --allow-ransac-labels).
  * A hand-labelled two-class set trains fine and the bundle loads
    (label_source="hand"); a RANSAC-trained bundle is REFUSED at load time.
  * The default verifier (no bundle on disk) is untrained and falls back to
    the documented non-ML baseline.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from train_ai_verifier import main as train_main
from ai_verifier import AIMatchVerifier


def _write_matches(path: Path, records: list[dict]) -> Path:
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _record(conf: float, dx: float = 0.0, **labels) -> dict:
    rec = {
        "confidence": conf, "refinement_dx": dx, "refinement_dy": 0.0,
        "spatial_quality_score": 0.8, "source_x": 1.0, "source_y": 2.0,
        "target_x": 3.0, "target_y": 4.0,
    }
    rec.update(labels)
    return rec


def test_train_fails_loudly_on_single_class(tmp_path):
    only_true = [_record(0.9 - 0.01 * i, human_label=True) for i in range(12)]
    f = _write_matches(tmp_path / "single_matches.json", only_true)
    rc = train_main(["--inputs", str(f), "--output", str(tmp_path / "m.pkl")])
    assert rc == 2, "single-class training must fail loudly (exit 2)"
    assert not (tmp_path / "m.pkl").exists(), "no degenerate model may be saved"


def test_train_refuses_ransac_labels_by_default(tmp_path):
    mixed_ransac = (
        [_record(0.9 - 0.01 * i, is_inlier=True) for i in range(8)]
        + [_record(0.2 + 0.01 * i, is_inlier=False) for i in range(8)]
    )
    f = _write_matches(tmp_path / "ransac_matches.json", mixed_ransac)
    rc = train_main(["--inputs", str(f), "--output", str(tmp_path / "m.pkl")])
    assert rc == 2, "RANSAC-only labels must be refused without --allow-ransac-labels"
    assert not (tmp_path / "m.pkl").exists()


def test_hand_labelled_training_roundtrip_and_ransac_bundle_refused(tmp_path):
    hand = (
        [_record(0.85 + 0.01 * (i % 5), 0.1 * i, human_label=True) for i in range(12)]
        + [_record(0.15 + 0.01 * (i % 5), 2.0 + i, human_label=False) for i in range(12)]
    )
    f = _write_matches(tmp_path / "hand_matches.json", hand)
    out = tmp_path / "hand.pkl"
    assert train_main(["--inputs", str(f), "--output", str(out)]) == 0
    assert out.exists()

    import joblib
    assert joblib.load(out)["label_source"] == "hand"
    verifier = AIMatchVerifier(model_path=out)
    assert verifier.is_trained is True
    kept, rejected = verifier.filter_matches(hand, threshold=0.5)
    assert len(kept) > 0 and len(rejected) > 0

    # A RANSAC-trained bundle (explicit opt-in) must NOT gate matches.
    ransac = (
        [_record(0.85 + 0.01 * (i % 5), 0.1 * i, is_inlier=True) for i in range(12)]
        + [_record(0.15 + 0.01 * (i % 5), 2.0 + i, is_inlier=False) for i in range(12)]
    )
    fr = _write_matches(tmp_path / "r_matches.json", ransac)
    out_r = tmp_path / "ransac.pkl"
    assert train_main(["--inputs", str(fr), "--output", str(out_r),
                       "--allow-ransac-labels"]) == 0
    assert joblib.load(out_r)["label_source"] == "ransac-acknowledged"
    refused = AIMatchVerifier(model_path=out_r)
    assert refused.is_trained is False, "circular bundles must be refused at load"


def test_default_verifier_is_untrained_baseline(tmp_path):
    verifier = AIMatchVerifier(model_path=tmp_path / "does_not_exist.pkl")
    assert verifier.is_trained is False
    recs = [_record(0.95 - 0.1 * i) for i in range(8)]
    kept, rejected = verifier.filter_matches(recs, threshold=0.5)
    # Baseline keeps the top ~75% by construction; it must not nuke everything.
    assert len(kept) >= 4
    assert len(kept) + len(rejected) == len(recs)


def test_bundled_production_model_loads_and_is_trained():
    """Verify that the bundled ai_verifier_model.pkl loads successfully as a trained model."""
    prod_path = REPO_ROOT / "ML_model/ai_verifier_model.pkl"
    if not prod_path.exists():
        pytest.skip("Production model ai_verifier_model.pkl not found.")
    verifier = AIMatchVerifier(model_path=prod_path)
    assert verifier.is_trained is True
    assert verifier.model is not None
    assert hasattr(verifier.model, "predict_proba")

    # Verify predictions on sample matches
    sample_matches = [
        _record(0.85, dx=0.1, dy=0.1),
        _record(0.12, dx=4.5, dy=3.8),
    ]
    confidences = verifier.predict_confidence(sample_matches)
    assert len(confidences) == 2
    # High-confidence, small-residual match should have higher probability than low-conf, large-residual match
    assert confidences[0] > confidences[1]

