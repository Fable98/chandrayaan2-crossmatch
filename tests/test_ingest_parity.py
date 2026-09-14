"""test_ingest_parity.py — ingest gives the full data-region bundle.

Covers the pipeline change "ingest also gives out the images and matching
linked cursor crossgrid matching and everything we do with the data regions":

- ingest result triplets carry images / matches_url / footprint_url /
  registered (blend + checkerboard cross-grid + quiver) parity assets.
- only true OHRC+TMC-2+IIRS triplets reach the results table (no LRO rows).
- Stage 5 never persists <4 correspondences and never warps with an
  identity fallback (fail-closed like Quality Gates 1-3).
- the backend serves honest fallback metrics for ingested regions from
  their own registration_output/<id>/metrics.json.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "data_preprocessing_pipeline" / "scripts"))

from routers.ingest import _enrich_ingest_triplet, _is_true_triplet  # noqa: E402


def _triplet(region_id="region_auto_001"):
    return {
        "region_id": region_id,
        "ohrc_product_id": "ch2_ohr_test",
        "tmc2_product_id": "ch2_tmc_test",
        "iirs_product_id": "ch2_iir_test",
        "overlap_triplet_pct": 95.0,
    }


def test_enrich_carries_data_region_bundle():
    out = _enrich_ingest_triplet(_triplet())
    assert out["images"] == {
        "ohrc": "/images/ohrc/region_auto_001",
        "tmc": "/images/tmc/region_auto_001",
        "iirs": "/images/iirs/region_auto_001",
        "dem": "/images/dem/region_auto_001",
    }
    assert out["matches_url"] == "/triplets/region_auto_001/matches"
    assert out["footprint_url"] == "/triplets/region_auto_001/footprint"
    reg = out["registered"]
    assert reg["warped"].endswith("registered_ohrc.png")
    assert reg["blend"].endswith("blend_overlay.png")
    # cross-grid continuity QA view
    assert reg["checkerboard"].endswith("checkerboard_qa.png")
    assert reg["quiver"].endswith("displacement_quiver.png")


def test_true_triplet_filter_excludes_lro_rows():
    assert _is_true_triplet(_triplet()) is True
    lro_row = {"region_id": "region_001", "reference_type": "external_LRO_NAC",
               "ohrc_product_id": "ch2_ohr_test"}
    assert _is_true_triplet(lro_row) is False
    assert _is_true_triplet({}) is False


def test_no_identity_fallback_without_correspondences(tmp_path):
    """Registration QA must fail closed when <4 matches exist."""
    import ingest_and_prepare as iap

    creds = tmp_path / "region_auto_009"
    creds.mkdir()
    import numpy as np
    import cv2

    cv2.imwrite(str(creds / "ohrc_512.png"), np.zeros((32, 32), np.uint8))
    cv2.imwrite(str(creds / "tmc_512.png"), np.zeros((32, 32), np.uint8))
    try:
        iap._write_registration_products(
            region_id="region_auto_009",
            reg_dir=creds,
            reg_root=tmp_path / "reg",
            match_res={"status": "failed", "matches": [], "homography": None,
                       "metrics": {}},
            manifest={},
        )
    except RuntimeError as exc:
        assert "insufficient_correspondences" in str(exc)
    else:
        raise AssertionError("expected fail-closed RuntimeError, got products")


def test_loader_fallback_metrics_from_ingest_products(tmp_path, monkeypatch):
    """Backend serves ingest metrics from registration_output, not nulls."""
    from data import loader

    bounds = {"west_lon": 0.0, "east_lon": 1.0, "south_lat": 0.0, "north_lat": 1.0}
    reg_dir = Path(loader.REPO_ROOT) / "registration_output" / "region_auto_777"
    reg_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reg_dir / "metrics.json"
    metrics_path.write_text(json.dumps({
        "region_id": "region_auto_777",
        "inlier_count": 6,
        "fit_rmse_px": 1.25,
        "inlier_ratio": 0.2,
        "spatial_coverage": 0.06,
        "spatial_uniformity": 0.5,
    }), encoding="utf-8")
    try:
        m = loader._metrics_from_registration_products("region_auto_777", bounds, 30)
    finally:
        metrics_path.unlink(missing_ok=True)
        try:
            reg_dir.rmdir()
        except OSError:
            pass
    assert m is not None
    assert m["num_inliers"] == 6
    assert m["fit_rmse_px"] == 1.25
    assert m["absolute_rmse_m"] is not None
    assert m["method"].startswith("CFOG")
