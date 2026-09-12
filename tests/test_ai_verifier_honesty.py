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

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

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


def test_unpulled_lfs_pointer_is_detected_not_cryptic(tmp_path, caplog):
    """An unpulled Git LFS pointer must be refused with an actionable message.

    Regression: joblib fails on pointer text with a bare ``KeyError: 118``,
    which used to surface as a cryptic warning while silently disabling the
    trained verifier.
    """
    from ai_verifier import looks_like_git_lfs_pointer

    stub = tmp_path / "stub.pkl"
    stub.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:7543962117de4d46e343ff4646c83df9c2d8f95ae93873ffc3d33\n"
        b"size 3222749\n"
    )
    assert looks_like_git_lfs_pointer(stub) is True
    assert looks_like_git_lfs_pointer(REPO_ROOT / "ML_model" / "ai_verifier.py") is False
    with caplog.at_level("WARNING", logger="ML_model.ai_verifier"):
        verifier = AIMatchVerifier(model_path=stub)
    assert verifier.is_trained is False
    assert "Git LFS" in caplog.text and "git lfs pull" in caplog.text


def test_bundled_production_model_loads_and_is_trained():
    """Verify that the bundled ai_verifier_model.pkl loads successfully as a trained model."""
    prod_path = REPO_ROOT / "ML_model/ai_verifier_model.pkl"
    if not prod_path.exists():
        pytest.skip("Production model ai_verifier_model.pkl not found.")
    with open(prod_path, "rb") as _f:
        if _f.read(30).startswith(b"version https://git-lfs"):
            pytest.skip("Production model ai_verifier_model.pkl is an unpulled Git LFS pointer.")
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


def test_train_skips_rows_missing_live_features(tmp_path):
    """Dumps without refinement_dx/dy + spatial_quality_score must not train.

    Defaulting those fields is how a genuine real inlier became statistically
    identical to the 91% failed-correspondence majority class.
    """
    from train_ai_verifier import extract_feature_row, build_dataset

    complete = _record(0.4, dx=0.2, human_label=True)
    incomplete = {
        "confidence": 0.4, "source_x": 1.0, "source_y": 2.0,
        "target_x": 3.0, "target_y": 4.0, "human_label": True,
    }
    try:
        extract_feature_row(incomplete, require_live_features=True)
        assert False, "missing live features must raise"
    except KeyError:
        pass
    row = extract_feature_row(complete, require_live_features=True)
    assert len(row) == 4

    mixed = []
    for i in range(8):
        rec = _record(0.4 + 0.01 * i, dx=0.2, human_label=True)
        rec["source_x"] = float(i)
        mixed.append(rec)
    mixed.extend(
        {**incomplete, "human_label": False, "source_x": float(20 + i)} for i in range(8)
    )
    f = _write_matches(tmp_path / "mixed_matches.json", mixed)
    X, y, stats = build_dataset([f], allow_ransac=False)
    assert stats["skipped_missing_features"] >= 8
    assert len(y) == 8
    assert set(y.tolist()) == {1}


def test_gt_generator_emits_live_features_and_cross_sensor_domain(tmp_path):
    from generate_ground_truth_matches import generate_matches_for_image

    img = np.zeros((160, 160), dtype=np.uint8)
    rng_img = np.random.RandomState(0)
    for _ in range(18):
        cv2.circle(
            img,
            (int(rng_img.randint(20, 140)), int(rng_img.randint(20, 140))),
            int(rng_img.randint(6, 16)),
            int(rng_img.randint(90, 240)),
            -1,
        )
    path = tmp_path / "tile.png"
    cv2.imwrite(str(path), img)
    recs = generate_matches_for_image(
        path, np.random.default_rng(42), domains=("same_sensor", "cross_sensor"),
    )
    assert recs, "GT generator produced no labeled rows"
    for r in recs:
        assert "spatial_quality_score" in r
        assert "refinement_dx" in r and "refinement_dy" in r
        assert r["label_source"] == "hand"
        assert r["domain"] in ("same_sensor", "cross_sensor")
    assert any(r["human_label"] for r in recs)
    assert any(not r["human_label"] for r in recs)
    assert any(r.get("domain") == "cross_sensor" for r in recs)


def test_live_matcher_records_populate_spatial_quality_and_refinement(tmp_path):
    from matcher_cfog import match_images_cfog

    h, w = 256, 256
    img1 = np.zeros((h, w), dtype=np.uint8)
    np.random.seed(42)
    for _ in range(25):
        cx = np.random.randint(30, w - 30)
        cy = np.random.randint(30, h - 30)
        rad = np.random.randint(10, 25)
        val = int(np.random.randint(100, 240))
        cv2.circle(img1, (cx, cy), rad, val, -1)
    img1 = cv2.GaussianBlur(img1, (5, 5), 1.0)
    img2 = cv2.warpAffine(img1, np.float32([[1, 0, 6], [0, 1, -4]]), (w, h))
    p1, p2 = tmp_path / "s.png", tmp_path / "r.png"
    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)

    res = match_images_cfog(
        p1, p2, output_dir=tmp_path / "out",
        explicit_gsd1=1.0, explicit_gsd2=1.0, grid_size=6,
    )
    matches_path = tmp_path / "out" / "matches.json"
    if not matches_path.exists():
        # Some layouts nest matches.json; search.
        found = list((tmp_path / "out").rglob("matches.json"))
        assert found, f"no matches.json written; status={res.get('status')}"
        matches_path = found[0]
    recs = json.loads(matches_path.read_text(encoding="utf-8"))
    if isinstance(recs, dict):
        recs = recs.get("matches") or recs.get("all_matches") or []
    assert recs, "matcher wrote no correspondence records"
    for r in recs:
        assert "spatial_quality_score" in r, r.keys()
        assert "refinement_dx" in r and "refinement_dy" in r
        assert 0.0 <= float(r["spatial_quality_score"]) <= 1.0
        assert "ai_inlier_prob" in r

