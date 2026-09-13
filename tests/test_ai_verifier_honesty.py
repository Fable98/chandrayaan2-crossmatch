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
from ai_verifier import FEATURE_NAMES as VERIFIER_FEATURES
from train_ai_verifier import FEATURE_NAMES as TRAIN_FEATURES


def test_feature_contract_stays_in_sync():
    assert TRAIN_FEATURES == VERIFIER_FEATURES
    assert TRAIN_FEATURES[:9] == [
        "confidence", "refinement_dx", "refinement_dy", "spatial_quality_score",
        "cfog_distance", "pc_energy_src", "pc_energy_tgt", "nn_ratio", "scale_diff",
    ]
    assert TRAIN_FEATURES[9:] == [
        "disp_consistency", "pairwise_dist_ratio",
    ]


def _write_matches(path: Path, records: list[dict]) -> Path:
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _record(conf: float, dx: float = 0.0, **labels) -> dict:
    rec = {
        "confidence": conf, "refinement_dx": dx, "refinement_dy": 0.0,
        "spatial_quality_score": 0.8, "source_x": 1.0, "source_y": 2.0,
        "target_x": 3.0, "target_y": 4.0,
        "cfog_distance": 0.15 if conf >= 0.5 else 1.2,
        "pc_energy_src": 0.7, "pc_energy_tgt": 0.65,
        "nn_ratio": 0.35 if conf >= 0.5 else 0.92,
        "scale_diff": 0.2 if conf >= 0.5 else 2.5,
        "source_image": "test_src.png",
        "target_image": "test_tgt.png",
        "region": "region_001",
        "domain": "same_sensor",
        "multimodal_pair": False,
    }
    rec.update(labels)
    return rec


def test_train_fails_loudly_on_single_class(tmp_path):
    only_true = [_record(0.9 - 0.01 * i, ground_truth_label=True) for i in range(12)]
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


def test_synthetic_geometry_training_roundtrip_and_ransac_bundle_refused(tmp_path):
    synthetic = (
        [_record(0.85 + 0.01 * (i % 5), 0.1 * i, ground_truth_label=True,
                 label_source="synthetic_geometry", source_image=f"s_{i//4}.png", target_image=f"t_{i//4}.png") for i in range(12)]
        + [_record(0.15 + 0.01 * (i % 5), 2.0 + i, ground_truth_label=False,
                   label_source="synthetic_geometry", source_image=f"s_{i//4}.png", target_image=f"t_{i//4}.png") for i in range(12)]
    )
    f = _write_matches(tmp_path / "synthetic_matches.json", synthetic)
    out = tmp_path / "synthetic.pkl"
    assert train_main(["--inputs", str(f), "--output", str(out)]) == 0
    assert out.exists()

    import joblib
    bundle = joblib.load(out)
    assert bundle["label_source"] == "synthetic_geometry"
    assert bundle["n_samples"] == 24
    assert bundle["n_train"] + bundle["n_test"] == 24
    verifier = AIMatchVerifier(model_path=out)
    assert verifier.is_trained is True
    kept, rejected = verifier.filter_matches(synthetic, threshold=0.5)
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


def test_legacy_hand_labels_backward_compatibility(tmp_path):
    hand = (
        [_record(0.85 + 0.01 * (i % 5), 0.1 * i, human_label=True, label_source="hand",
                 source_image=f"s_{i//4}.png", target_image=f"t_{i//4}.png") for i in range(12)]
        + [_record(0.15 + 0.01 * (i % 5), 2.0 + i, human_label=False, label_source="hand",
                   source_image=f"s_{i//4}.png", target_image=f"t_{i//4}.png") for i in range(12)]
    )
    f = _write_matches(tmp_path / "hand_matches.json", hand)
    out = tmp_path / "hand.pkl"
    assert train_main(["--inputs", str(f), "--output", str(out), "--label-key", "human_label"]) == 0
    assert out.exists()

    import joblib
    bundle = joblib.load(out)
    assert bundle["label_source"] == "hand"
    verifier = AIMatchVerifier(model_path=out)
    assert verifier.is_trained is True


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
    assert len(row) == len(TRAIN_FEATURES)

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
        for k in ("cfog_distance", "pc_energy_src", "pc_energy_tgt", "nn_ratio", "scale_diff"):
            assert k in r, r.keys()
        assert r["label_source"] == "synthetic_geometry"
        assert "ground_truth_label" in r
        assert "human_label" not in r, "Generator must not produce misleading human_label"
        assert "source_image" in r and r["source_image"] == "tile.png"
        assert "target_image" in r and r["target_image"].startswith("tile_")
        assert "region" in r
        assert "domain" in r and r["domain"] in ("same_sensor", "cross_sensor")
        assert "multimodal_pair" in r
    assert any(r["ground_truth_label"] for r in recs)
    assert any(not r["ground_truth_label"] for r in recs)
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
        for k in ("cfog_distance", "pc_energy_src", "pc_energy_tgt", "nn_ratio", "scale_diff"):
            assert k in r, r.keys()


def test_hard_pre_ransac_veto_stays_experimental():
    """Default path must weight RANSAC, not hard-filter, unless experimental_stack."""
    src = (REPO_ROOT / "ML_model/matcher_cfog.py").read_text(encoding="utf-8")
    assert "if experimental_stack and verifier.is_trained:" in src
    assert "verifier.filter_matches(refinement_records, threshold=0.5)" in src
    assert "weighting RANSAC with" in src


def test_descriptor_features_are_finite_measurements():
    from matcher_cfog import compute_descriptor_match_features, DESCRIPTOR_FEATURE_KEYS

    img = np.zeros((64, 64), dtype=np.float32)
    img[20:40, 20:40] = 1.0
    img[8:16, 8:16] = 0.4
    feats = compute_descriptor_match_features(img, img, 30.0, 30.0, 12.0, 12.0)
    assert tuple(feats) == DESCRIPTOR_FEATURE_KEYS
    for v in feats.values():
        assert np.isfinite(v)


def test_grouped_evaluation_prevents_leakage(tmp_path):
    """Grouped evaluation must partition whole groups without train/test overlap."""
    import joblib

    records = []
    # 6 distinct image pairs, each with balanced classes
    for p_idx in range(6):
        src_name = f"src_{p_idx}.png"
        tgt_name = f"tgt_{p_idx}.png"
        reg_name = f"region_{p_idx:03d}"
        dom = "cross_sensor" if p_idx % 2 == 0 else "same_sensor"
        for i in range(8):
            records.append(_record(
                0.85 + 0.005 * i,
                dx=0.1 + 0.01 * i,
                source_x=float(p_idx * 100 + i * 4),
                source_y=float(p_idx * 100 + i * 4 + 1),
                target_x=float(p_idx * 100 + i * 4 + 10),
                target_y=float(p_idx * 100 + i * 4 + 11),
                ground_truth_label=True,
                label_source="synthetic_geometry",
                source_image=src_name,
                target_image=tgt_name,
                region=reg_name,
                domain=dom,
            ))
            records.append(_record(
                0.15 + 0.005 * i,
                dx=3.0 + 0.01 * i,
                source_x=float(p_idx * 100 + i * 4 + 2),
                source_y=float(p_idx * 100 + i * 4 + 3),
                target_x=float(p_idx * 100 + i * 4 + 20),
                target_y=float(p_idx * 100 + i * 4 + 21),
                ground_truth_label=False,
                label_source="synthetic_geometry",
                source_image=src_name,
                target_image=tgt_name,
                region=reg_name,
                domain=dom,
            ))

    f = _write_matches(tmp_path / "grouped_matches.json", records)
    out = tmp_path / "grouped_model.pkl"
    rc = train_main(["--inputs", str(f), "--output", str(out), "--split", "grouped"])
    assert rc == 0
    assert out.exists()

    bundle = joblib.load(out)
    assert bundle["split_strategy"] == "grouped"
    assert bundle["n_samples"] == len(records)
    assert bundle["n_train"] + bundle["n_test"] == len(records)
    assert bundle["n_train"] > 0 and bundle["n_test"] > 0
    assert bundle["group_key"] == "image_pair"

    metrics = bundle["evaluation_metrics"]
    assert "overall" in metrics
    assert "cross_sensor" in metrics
    assert "same_sensor" in metrics
    assert metrics["overall"]["n_samples"] == bundle["n_test"]


def test_grouped_evaluation_fails_loudly_on_insufficient_groups(tmp_path):
    """Grouped split must refuse to silently revert to random if groups < 2."""
    records = []
    # Single image pair only
    for i in range(12):
        records.append(_record(
            0.85 + 0.005 * i,
            source_x=float(i * 2),
            source_y=float(i * 2 + 1),
            target_x=float(i * 2 + 10),
            target_y=float(i * 2 + 11),
            ground_truth_label=True, label_source="synthetic_geometry",
            source_image="solo_src.png", target_image="solo_tgt.png",
        ))
        records.append(_record(
            0.15 + 0.005 * i,
            source_x=float(i * 2 + 50),
            source_y=float(i * 2 + 51),
            target_x=float(i * 2 + 60),
            target_y=float(i * 2 + 61),
            ground_truth_label=False, label_source="synthetic_geometry",
            source_image="solo_src.png", target_image="solo_tgt.png",
        ))

    f = _write_matches(tmp_path / "single_group.json", records)
    out = tmp_path / "model.pkl"
    rc = train_main(["--inputs", str(f), "--output", str(out), "--split", "grouped"])
    assert rc == 2, "grouped split on single group must fail with exit code 2"
    assert not out.exists(), "no model may be saved on grouped split failure"


def test_evaluation_zero_leakage_and_metadata(tmp_path):
    """Test model is fitted strictly on training rows and predictions evaluate test only."""
    from train_ai_verifier import train, build_dataset

    records = []
    for g_idx in range(4):
        for i in range(6):
            records.append(_record(
                0.8 + 0.01 * i,
                source_x=float(g_idx * 100 + i * 2),
                source_y=float(g_idx * 100 + i * 2 + 1),
                target_x=float(g_idx * 100 + i * 2 + 10),
                target_y=float(g_idx * 100 + i * 2 + 11),
                ground_truth_label=True,
                source_image=f"s_{g_idx}.png", target_image=f"t_{g_idx}.png",
                region=f"reg_{g_idx}", domain="cross_sensor" if g_idx < 2 else "same_sensor",
            ))
            records.append(_record(
                0.2 + 0.01 * i,
                source_x=float(g_idx * 100 + i * 2 + 20),
                source_y=float(g_idx * 100 + i * 2 + 21),
                target_x=float(g_idx * 100 + i * 2 + 30),
                target_y=float(g_idx * 100 + i * 2 + 31),
                ground_truth_label=False,
                source_image=f"s_{g_idx}.png", target_image=f"t_{g_idx}.png",
                region=f"reg_{g_idx}", domain="cross_sensor" if g_idx < 2 else "same_sensor",
            ))

    f = _write_matches(tmp_path / "leak_test.json", records)
    X, y, stats = build_dataset([f])
    groups = stats["groups"]
    domains = stats["domains"]

    clf, eval_results, (X_tr, y_tr, X_te, y_te) = train(
        X, y, groups=groups, domains=domains, split_strategy="grouped", test_size=0.25
    )

    assert len(X_tr) + len(X_te) == len(X)
    assert len(X_tr) > 0 and len(X_te) > 0
    assert eval_results["overall"]["n_samples"] == len(X_te)
