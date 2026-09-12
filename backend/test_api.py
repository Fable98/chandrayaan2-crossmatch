"""
test_api.py — Automated tests for the SIH26166 backend (Shared-Bbox Architecture + 6 Regions + DEM).

Run with:  pytest test_api.py -v
No server needs to be running — TestClient spins the app in-process.
"""

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-only-jwt-secret-that-is-long-enough-for-the-32-char-minimum-0123456789",
)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("AUTH_RATE_LIMIT", "1000/minute")

BACKEND_DIR = str(Path(__file__).resolve().parent)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if "config" in sys.modules and not hasattr(sys.modules["config"], "settings"):
    sys.modules.pop("config", None)
if "data" in sys.modules and not hasattr(sys.modules["data"], "loader"):
    sys.modules.pop("data", None)

from fastapi.testclient import TestClient
from main import app
from geo import (
    pixel_to_latlon_from_bounds,
    pixel_to_latlon_from_bounds_batch,
)

client = TestClient(app)


_CACHED_AUTH_HEADERS: dict | None = None


def _auth_headers():
    """Register a throwaway user once and reuse its Bearer header (Step 12)."""
    global _CACHED_AUTH_HEADERS
    if _CACHED_AUTH_HEADERS is not None:
        return _CACHED_AUTH_HEADERS
    import uuid

    email = f"api-test-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post(
        "/auth/register",
        json={"name": "API Test", "email": email, "password": "correct-horse-123"},
    )
    assert r.status_code == 201, r.text[:200]
    _CACHED_AUTH_HEADERS = {"Authorization": f"Bearer {r.json()['access_token']}"}
    return _CACHED_AUTH_HEADERS

VALID_ID = "region_001"
INVALID_ID = "nonexistent_region"

BOUNDS_KEYS = {"west_lon", "east_lon", "south_lat", "north_lat"}


def _ensure_loaded():
    """Ensure data is loaded (lifespan may not have fired yet in TestClient)."""
    client.get("/refresh", headers=_auth_headers())


# ---------------------------------------------------------------------------
# Health & refresh
# ---------------------------------------------------------------------------

def test_root_endpoint():
    _ensure_loaded()
    r_get = client.get("/")
    assert r_get.status_code == 200
    assert r_get.json()["status"] == "online"
    assert r_get.json()["triplets_loaded"] >= 6

    r_head = client.head("/")
    assert r_head.status_code == 200


def test_health():
    _ensure_loaded()
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["triplets_loaded"] >= 6

    r_head = client.head("/health")
    assert r_head.status_code == 200


def test_refresh():
    # Step 12: /refresh requires auth.
    assert client.get("/refresh").status_code == 401
    r = client.get("/refresh", headers=_auth_headers())
    assert r.status_code == 200
    assert r.json()["status"] == "refreshed"
    assert r.json()["triplets_loaded"] >= 6


# ---------------------------------------------------------------------------
# Triplets list & Multi-region tests
# ---------------------------------------------------------------------------

def test_triplets_list():
    _ensure_loaded()
    r = client.get("/triplets")
    assert r.status_code == 200
    data = r.json()
    assert "triplets" in data
    assert isinstance(data["triplets"], list)
    assert len(data["triplets"]) >= 6


def test_triplets_list_has_required_fields():
    _ensure_loaded()
    r = client.get("/triplets")
    triplet = r.json()["triplets"][0]
    assert "id" in triplet
    assert "bounds" in triplet
    assert set(triplet["bounds"].keys()) == BOUNDS_KEYS
    assert "sensors" in triplet
    assert len(triplet["sensors"]) >= 3
    assert "dem_available" in triplet


def test_all_six_real_regions_loaded():
    """
    Verify all 6 validated regions (region_001 through region_006)
    are loaded simultaneously into memory without ID collisions or data bleeding.
    """
    _ensure_loaded()
    r = client.get("/triplets")
    assert r.status_code == 200
    loaded_ids = {t["id"] for t in r.json()["triplets"]}
    expected_ids = {"region_001", "region_002", "region_003", "region_004", "region_005", "region_006"}
    assert expected_ids.issubset(loaded_ids), f"Missing regions: {expected_ids - loaded_ids}"

    # Verify each region has valid, non-degenerate bounds
    for t in r.json()["triplets"]:
        if t["id"] in expected_ids:
            b = t["bounds"]
            assert b["east_lon"] > b["west_lon"], f"{t['id']} east_lon <= west_lon"
            assert b["north_lat"] > b["south_lat"], f"{t['id']} north_lat <= south_lat"


# ---------------------------------------------------------------------------
# Triplet detail & Antimeridian Crossing
# ---------------------------------------------------------------------------

def test_triplet_detail():
    _ensure_loaded()
    r = client.get(f"/triplets/{VALID_ID}")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == VALID_ID
    assert "bounds" in data
    assert set(data["bounds"].keys()) == BOUNDS_KEYS
    assert len(data["sensors"]) >= 3

    sensor_names = {s["sensor"] for s in data["sensors"]}
    assert {"ohrc", "tmc", "iirs"}.issubset(sensor_names)


def test_triplet_detail_404():
    r = client.get(f"/triplets/{INVALID_ID}")
    assert r.status_code == 404


def test_antimeridian_crossing_regions():
    """
    The current processed data uses the standard 0–360° lunar longitude convention
    and does not straddle the 180° meridian for the real region_005 / region_006 values.
    """
    _ensure_loaded()
    for reg_id in ("region_005", "region_006"):
        r = client.get(f"/triplets/{reg_id}")
        assert r.status_code == 200
        data = r.json()
        b = data["bounds"]
        assert 0.0 <= b["west_lon"] <= 360.0
        assert 0.0 <= b["east_lon"] <= 360.0
        assert b["east_lon"] > b["west_lon"]

        # Ensure affine transform maps center pixel cleanly
        lat_c, lon_c = pixel_to_latlon_from_bounds(256.0, 256.0, b, 512, 512)
        assert b["south_lat"] <= lat_c <= b["north_lat"]
        assert b["west_lon"] <= lon_c <= b["east_lon"]


# ---------------------------------------------------------------------------
# DEM (Digital Elevation Model) tests
# ---------------------------------------------------------------------------

def test_dem_metadata_in_triplet_response():
    """
    Verify DEM fields (dem_available: bool, dem_url: str) appear in GET /triplets/{id}.
    """
    _ensure_loaded()
    r = client.get(f"/triplets/{VALID_ID}")
    assert r.status_code == 200
    data = r.json()
    assert data["dem_available"] is True
    assert data["dem_url"].startswith("/images/dem/")

    # Confirm DEM is registered in sensors list
    sensor_names = {s["sensor"] for s in data["sensors"]}
    assert "dem" in sensor_names


def test_static_image_dem():
    """Verify DEM image is served via GET /images/dem/dem_512.png."""
    r = client.get("/images/dem/dem_512.png")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_all_six_regions_dem_resolve():
    """
    Sanity check: confirm DEM images resolve for all 6 regions via their dem_url.
    """
    _ensure_loaded()
    for i in range(1, 7):
        r_id = f"region_{i:03d}"
        r = client.get(f"/triplets/{r_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["dem_available"] is True
        assert data["dem_url"] is not None

        # Hit the dem_url endpoint to confirm image resolves
        img_r = client.get(data["dem_url"])
        assert img_r.status_code == 200, f"Failed to fetch DEM for {r_id} at {data['dem_url']}"
        assert img_r.headers["content-type"].startswith("image/")


# ---------------------------------------------------------------------------
# Footprint & IIRS overlay (Shared-Bbox Invariants)
# ---------------------------------------------------------------------------

def test_footprint_returns_shared_bounds():
    """
    Ensure /triplets/{id}/footprint returns the shared TripletBounds.
    """
    _ensure_loaded()
    r = client.get(f"/triplets/{VALID_ID}/footprint")
    assert r.status_code == 200
    data = r.json()
    assert data["triplet_id"] == VALID_ID
    assert "bounds" in data
    assert set(data["bounds"].keys()) == BOUNDS_KEYS
    for k in BOUNDS_KEYS:
        assert isinstance(data["bounds"][k], (int, float))


def test_footprint_and_iirs_overlay_have_identical_bounds():
    """
    INVARIANT GUARD: OHRC, TMC-2, IIRS, and DEM share one identical bounding box
    by design in the real pipeline.
    Verify /triplets/{id}, /triplets/{id}/footprint, and /triplets/{id}/iirs-overlay
    all return identical bounds.
    """
    _ensure_loaded()
    r_triplet = client.get(f"/triplets/{VALID_ID}")
    r_footprint = client.get(f"/triplets/{VALID_ID}/footprint")
    r_overlay = client.get(f"/triplets/{VALID_ID}/iirs-overlay")

    assert r_triplet.status_code == 200
    assert r_footprint.status_code == 200
    assert r_overlay.status_code == 200

    triplet_bounds = r_triplet.json()["bounds"]
    footprint_bounds = r_footprint.json()["bounds"]
    overlay_bounds = r_overlay.json()["bounds"]

    assert footprint_bounds == triplet_bounds
    assert overlay_bounds == triplet_bounds


def test_footprint_404():
    r = client.get(f"/triplets/{INVALID_ID}/footprint")
    assert r.status_code == 404


def test_iirs_overlay_bounds_present():
    _ensure_loaded()
    r = client.get(f"/triplets/{VALID_ID}/iirs-overlay")
    assert r.status_code == 200
    data = r.json()
    assert data["triplet_id"] == VALID_ID
    assert data["image_url"].startswith("/images/iirs/")
    assert "bounds" in data
    assert set(data["bounds"].keys()) == BOUNDS_KEYS


def test_iirs_overlay_has_opacity_hint():
    _ensure_loaded()
    r = client.get(f"/triplets/{VALID_ID}/iirs-overlay")
    data = r.json()
    assert 0.0 <= data["opacity_hint"] <= 1.0


def test_iirs_overlay_404():
    r = client.get(f"/triplets/{INVALID_ID}/iirs-overlay")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Matches — enriched with shared-bbox geo coordinates
# ---------------------------------------------------------------------------

def test_matches_has_homography():
    _ensure_loaded()
    r = client.get(f"/triplets/{VALID_ID}/matches")
    assert r.status_code == 200
    data = r.json()
    assert data["triplet_id"] == VALID_ID
    if data["homography"] is not None:
        assert len(data["homography"]) == 3
        assert all(len(row) == 3 for row in data["homography"])


def test_matches_has_points_with_pixel_and_latlon():
    """
    Verify each match point has both pixel coords (ohrc_px, tmc_px)
    and geographic coords (ohrc_latlon, tmc_latlon), plus confidence.
    """
    _ensure_loaded()
    r = client.get(f"/triplets/{VALID_ID}/matches")
    data = r.json()
    assert data["num_matches"] == len(data["matches"])
    assert data["num_matches"] >= 1

    for m in data["matches"]:
        assert len(m["ohrc_px"]) == 2, "ohrc_px should be (x, y)"
        assert len(m["tmc_px"]) == 2, "tmc_px should be (x, y)"
        assert len(m["ohrc_latlon"]) == 2, "ohrc_latlon should be (lat, lon)"
        assert len(m["tmc_latlon"]) == 2, "tmc_latlon should be (lat, lon)"
        assert 0.0 <= m["confidence"] <= 1.0


def test_match_latlon_within_footprint():
    """
    Verify all match lat/lon values fall within the triplet's shared bounds.
    """
    _ensure_loaded()
    triplet_r = client.get(f"/triplets/{VALID_ID}")
    bounds = triplet_r.json()["bounds"]
    min_lat, max_lat = bounds["south_lat"], bounds["north_lat"]
    min_lon, max_lon = bounds["west_lon"], bounds["east_lon"]

    eps = 0.01

    r = client.get(f"/triplets/{VALID_ID}/matches")
    for m in r.json()["matches"]:
        ohrc_lat, ohrc_lon = m["ohrc_latlon"]
        assert min_lat - eps <= ohrc_lat <= max_lat + eps
        assert min_lon - eps <= ohrc_lon <= max_lon + eps

        tmc_lat, tmc_lon = m["tmc_latlon"]
        assert min_lat - eps <= tmc_lat <= max_lat + eps
        assert min_lon - eps <= tmc_lon <= max_lon + eps


def test_missing_matches_returns_empty_not_error():
    _ensure_loaded()
    r = client.get(f"/triplets/region_004/matches")
    assert r.status_code == 200
    data = r.json()
    assert "matches" in data
    assert isinstance(data["matches"], list)


def test_matches_404_for_unknown_triplet():
    r = client.get(f"/triplets/{INVALID_ID}/matches")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Direct Geo Conversion Unit Tests (Shared-Bbox & 0-360 Longitude)
# ---------------------------------------------------------------------------

def test_pixel_to_latlon_sanity():
    """
    Verify pixel_to_latlon_from_bounds maps center pixel (256, 256)
    to the geographic centroid of the bounding box.
    """
    bounds = {
        "west_lon": 30.100,
        "east_lon": 30.260,
        "south_lat": -89.950,
        "north_lat": -89.900,
    }

    lat, lon = pixel_to_latlon_from_bounds(256.0, 256.0, bounds, 512, 512)

    mid_lat = (-89.950 + -89.900) / 2.0
    mid_lon = (30.100 + 30.260) / 2.0

    assert abs(lat - mid_lat) < 1e-6
    assert abs(lon - mid_lon) < 1e-6


def test_pixel_to_latlon_corners_map_correctly():
    """
    Verify (0, 0) maps to (north_lat, west_lon) and
    (512, 512) maps to (south_lat, east_lon).
    """
    bounds = {
        "west_lon": 30.100,
        "east_lon": 30.260,
        "south_lat": -89.950,
        "north_lat": -89.900,
    }

    lat_tl, lon_tl = pixel_to_latlon_from_bounds(0, 0, bounds, 512, 512)
    assert abs(lat_tl - bounds["north_lat"]) < 1e-6
    assert abs(lon_tl - bounds["west_lon"]) < 1e-6

    lat_br, lon_br = pixel_to_latlon_from_bounds(512, 512, bounds, 512, 512)
    assert abs(lat_br - bounds["south_lat"]) < 1e-6
    assert abs(lon_br - bounds["east_lon"]) < 1e-6


def test_0_360_longitude_bounds_and_conversion():
    """
    Verify geo transformation using real-world 0–360° longitude convention
    values confirmed from actual data pipeline output:
      west_lon: 336.484646
      east_lon: 336.589455
      south_lat: -3.416904
      north_lat: -2.576048
    """
    real_bounds = {
        "west_lon": 336.484646,
        "east_lon": 336.589455,
        "south_lat": -3.416904,
        "north_lat": -2.576048,
    }

    # Top-left (0, 0)
    lat_tl, lon_tl = pixel_to_latlon_from_bounds(0.0, 0.0, real_bounds, 512, 512)
    assert abs(lat_tl - (-2.576048)) < 1e-6
    assert abs(lon_tl - 336.484646) < 1e-6

    # Bottom-right (512, 512)
    lat_br, lon_br = pixel_to_latlon_from_bounds(512.0, 512.0, real_bounds, 512, 512)
    assert abs(lat_br - (-3.416904)) < 1e-6
    assert abs(lon_br - 336.589455) < 1e-6

    # Batch test
    pts = [(0.0, 0.0), (256.0, 256.0), (512.0, 512.0)]
    batch_res = pixel_to_latlon_from_bounds_batch(pts, real_bounds, 512, 512)
    assert len(batch_res) == 3
    assert abs(batch_res[0][0] - (-2.576048)) < 1e-6
    assert abs(batch_res[0][1] - 336.484646) < 1e-6
    assert abs(batch_res[2][0] - (-3.416904)) < 1e-6
    assert abs(batch_res[2][1] - 336.589455) < 1e-6


# ---------------------------------------------------------------------------
# Static images
# ---------------------------------------------------------------------------

def test_static_image_ohrc():
    r = client.get("/images/ohrc/ohrc_512.png")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_static_image_tmc():
    r = client.get("/images/tmc/tmc_512.png")
    assert r.status_code == 200


def test_static_image_iirs():
    r = client.get("/images/iirs/iirs_overlay.png")
    assert r.status_code == 200


def test_static_image_404():
    r = client.get("/images/ohrc/nonexistent.png")
    assert r.status_code == 404


def test_images_serving_by_region_id():
    """Verify LinkedCursorPanel image requests like /images/ohrc/region_001 resolve."""
    for reg_id in ("region_001", "region_002", "region_003", "region_004", "region_005", "region_006"):
        r_ohrc = client.get(f"/images/ohrc/{reg_id}")
        assert r_ohrc.status_code == 200, f"Failed for /images/ohrc/{reg_id}"
        assert r_ohrc.headers["content-type"].startswith("image/")

        r_tmc = client.get(f"/images/tmc/{reg_id}")
        assert r_tmc.status_code == 200, f"Failed for /images/tmc/{reg_id}"
        assert r_tmc.headers["content-type"].startswith("image/")


def test_triplet_top_level_product_ids():
    """Verify top-level product IDs are present for MetaBar."""
    _ensure_loaded()
    r = client.get("/triplets/region_001")
    assert r.status_code == 200
    data = r.json()
    assert "ohrc_product_id" in data
    assert "tmc2_product_id" in data
    assert "iirs_product_id" in data


def test_matches_canonical_schema():
    """Verify /triplets/{id}/matches conforms strictly to MatchesResponse schema with matches array."""
    _ensure_loaded()
    r = client.get("/triplets/region_001/matches")
    assert r.status_code == 200
    data = r.json()
    assert "triplet_id" in data
    assert "num_matches" in data
    assert "homography" in data
    assert "matches" in data
    assert "points" not in data
    assert isinstance(data["matches"], list)


def test_matches_contains_evaluation_metrics():
    """Verify /triplets/{id}/matches contains computed evaluation metrics (RMSE, inliers, coverage)."""
    _ensure_loaded()
    r = client.get("/triplets/region_001/matches")
    assert r.status_code == 200
    data = r.json()
    assert "metrics" in data
    metrics = data["metrics"]
    if metrics is not None:
        assert "num_inliers" in metrics
        assert "rmse_px" in metrics
        assert "combined_coverage_score" in metrics
        assert "sub_pixel_accurate" in metrics
        assert metrics["num_inliers"] == data["num_matches"]


def test_registered_image_serving():
    """Verify registered output products are served via /images/registered/{region_id}."""
    r = client.get("/images/registered/region_001")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


# ---------------------------------------------------------------------------
# Ingest pipeline endpoints
# ---------------------------------------------------------------------------

def test_ingest_jobs_endpoint():
    """Verify /api/ingest/jobs returns a list of ingest jobs."""
    r = client.get("/api/ingest/jobs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_ingest_status_not_found():
    """Verify /api/ingest/status/{job_id} returns 404 for invalid job."""
    r = client.get("/api/ingest/status/nonexistent_job")
    assert r.status_code == 404


def test_ingest_results_not_found():
    """Verify /api/ingest/results/{job_id} returns 404 for invalid job."""
    r = client.get("/api/ingest/results/nonexistent_job")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Metric completeness: values computed upstream must survive to the API
# (regression: absolute_rmse_m / composite were stripped by the schema)
# ---------------------------------------------------------------------------

def test_lro_metrics_survive_response_model(monkeypatch):
    """Computed values must not be stripped by the response schema.

    Hermetic: injects a loader entry directly, so this holds on CI checkouts
    where registration_output/ (local-only artifacts) does not exist.
    """
    from data import loader as loader_mod

    monkeypatch.setattr(loader_mod, "_matches", {
        "region_001_lro_nac": {
            "triplet_id": "region_001_lro_nac",
            "matches": [],
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "metrics": {
                "num_inliers": 6,
                "num_raw_matches": 32,
                "inlier_ratio": 0.1875,
                "rmse_px": 0.6333,
                "fit_rmse_px": 0.6333,
                "absolute_rmse_m": 0.5788,
                "absolute_rmse_m_provenance": "pipeline_metrics_json",
                "validation_rmse_px": None,
                "validation_status": "insufficient_points_for_holdout",
                "ssim": 0.1441,
                "psnr": 15.0171,
                "nmi": 0.0029,
                "composite_quality_score": 0.3115,
                "metric_notes": {"validation_rmse_px": "held-out needs ≥8 inliers (have 6)"},
            },
        }
    })
    r = client.get("/triplets/region_001_lro_nac/matches")
    assert r.status_code == 200
    m = r.json()["metrics"]
    assert m is not None
    assert m["fit_rmse_px"] == pytest.approx(0.6333)
    assert m["absolute_rmse_m"] == pytest.approx(0.5788)
    assert m["absolute_rmse_m_provenance"] == "pipeline_metrics_json"
    assert m["composite_quality_score"] == pytest.approx(0.3115)
    assert m["ssim"] == pytest.approx(0.1441)
    assert m["metric_notes"]["validation_rmse_px"] == "held-out needs ≥8 inliers (have 6)"


def test_lro_assembly_from_disk_artifacts(tmp_path, monkeypatch):
    """End-to-end loader assembly from a synthetic artifact tree.

    Mirrors registration_output/lro_nac/<id>/ + lro_nac_real/<id>/ with
    generated PNGs (no repo artifacts needed): pipeline numbers must pass
    through exactly, photometrics must compute, honest nulls must be noted.
    """
    import cv2
    import numpy as np
    from data import loader as loader_mod

    reg_dir = tmp_path / "registration_output" / "lro_nac" / "region_001"
    reg_dir.mkdir(parents=True)
    (reg_dir / "metrics.json").write_text(json.dumps({
        "fit_rmse_px": 0.6333,
        "validation_rmse_px": None,
        "validation_status": "insufficient_points_for_holdout",
        "absolute_rmse_m": 0.5788,
        "inlier_count": 6,
        "match_count": 32,
        "inlier_ratio": 0.1875,
        "mean_reprojection_error_px": 0.5,
        "median_reprojection_error_px": 0.45,
        "max_reprojection_error_px": 1.2,
        "fraction_below_1px": 0.9,
        "spatial_coverage": 0.06,
        "spatial_uniformity": 0.0183,
        "composite_quality_score": 0.3115,
        "lk_refinement": {"debug_points": []},
    }))
    (reg_dir / "ohrc_to_nac_homography.json").write_text(json.dumps(
        {"homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}))
    rng = np.random.RandomState(0)
    cv2.imwrite(
        str(reg_dir / "registered_source.png"),
        (rng.rand(64, 64) * 255).astype(np.uint8),
    )
    ref_dir = tmp_path / "data_preprocessing_pipeline" / "lro_nac_real" / "region_001"
    ref_dir.mkdir(parents=True)
    cv2.imwrite(
        str(ref_dir / "lro_nac_reference_512.png"),
        (rng.rand(64, 64) * 255).astype(np.uint8),
    )

    monkeypatch.setattr(loader_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(loader_mod, "_BENCHMARK_SUMMARY", None)
    monkeypatch.setattr(loader_mod, "_BENCHMARK_LOADED", True)
    loader_mod.load_all()

    entry = loader_mod.get_matches("region_001_lro_nac")
    assert entry is not None
    m = entry["metrics"]
    assert m["absolute_rmse_m"] == pytest.approx(0.5788)
    assert m["composite_quality_score"] == pytest.approx(0.3115)
    assert isinstance(m["ssim"], float)
    assert isinstance(m["psnr"], float)
    assert isinstance(m["nmi"], float)
    assert "8 inliers" in m["metric_notes"]["validation_rmse_px"]
    assert "dots unavailable" in m["metric_notes"]["matches"]
    # NOTE: monkeypatched REPO_ROOT/_BENCHMARK_* revert automatically; later
    # tests re-trigger load_all via /refresh with real paths.


def test_regular_metrics_from_benchmark_summary(monkeypatch):
    """Planar absolute + feature-only composite derive correctly.

    Hermetic unit test over a synthetic summary row (the committed summary
    is local-only and absent on CI checkouts).
    """
    from data import loader as loader_mod

    monkeypatch.setattr(loader_mod, "_BENCHMARK_SUMMARY", {"regions": [{
        "region_id": "region_001",
        "status": "success",
        "fit_rmse_px": 1.2715,
        "inlier_count": 7,
        "match_count": 41,
        "inlier_ratio": 0.1707,
        "spatial_coverage": 0.4375,
        "spatial_uniformity": 0.3113,
        "validation_rmse_px": None,
        "validation_status": "insufficient_points_for_holdout",
    }]})
    monkeypatch.setattr(loader_mod, "_BENCHMARK_LOADED", True)
    bounds = {"west_lon": 336.484646, "east_lon": 336.589455,
              "south_lat": -3.374861, "north_lat": -3.248733}
    m = loader_mod._metrics_from_benchmark_summary("region_001", bounds, 7)
    assert m is not None
    assert m["fit_rmse_px"] == pytest.approx(1.2715)
    assert m["num_inliers"] == 7 and m["num_raw_matches"] == 41
    assert m["inlier_ratio"] == pytest.approx(0.1707)
    expected_gsd = loader_mod._footprint_gsd_m(bounds)
    assert m["absolute_rmse_m"] == pytest.approx(1.2715 * expected_gsd, rel=1e-3)
    assert m["absolute_rmse_m_provenance"] == "planar_footprint_gsd_no_dem"
    assert 0.0 <= m["composite_quality_score"] <= 1.0
    assert m["metric_notes"]["validation_rmse_px"].startswith("held-out")


# ---------------------------------------------------------------------------
# LRO reference imagery resolves to real CDR tiles (not 404, not OHRC)
# ---------------------------------------------------------------------------

def test_static_image_lro_real_cdr():
    """GET /images/lro_nac/region_001 serves the real CDR reference tile."""
    _ensure_loaded()
    r = client.get("/images/lro_nac/region_001")
    assert r.status_code == 200, r.text[:200]
    assert r.headers["content-type"].startswith("image/")
    assert len(r.content) > 10000, "tile payload suspiciously small"


# ---------------------------------------------------------------------------
# Step 14: ingest upload auth gate + job lifecycle + concurrency
# ---------------------------------------------------------------------------

def _ingest_auth_headers():
    """Step 12/13: ingest upload requires a Bearer token."""
    return _auth_headers()


def test_ingest_upload_requires_auth():
    """POST /api/ingest/upload without a token is refused before disk writes."""
    r = client.post(
        "/api/ingest/upload",
        files={"files": ("data.zip", b"PK\x03\x04", "application/zip")},
    )
    assert r.status_code == 401


def test_ingest_upload_rejects_non_zip():
    """Non-zip uploads are rejected with 415 (extension gate before write)."""
    r = client.post(
        "/api/ingest/upload",
        headers=_ingest_auth_headers(),
        files={"files": ("evil.php", b"<?php", "application/x-php")},
    )
    assert r.status_code == 415


def test_ingest_upload_starts_job_and_tracks_lifecycle():
    """A .zip upload creates a job; status/results endpoints track it."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("note.txt", "no PDS4 products here")
    r = client.post(
        "/api/ingest/upload",
        headers=_ingest_auth_headers(),
        files={"files": ("empty.zip", buf.getvalue(), "application/zip")},
    )
    assert r.status_code == 200, r.text[:300]
    job_id = r.json()["job_id"]
    assert job_id

    # Status is immediately pollable (pending/running/completed/failed).
    s = client.get(f"/api/ingest/status/{job_id}")
    assert s.status_code == 200
    assert s.json()["job_id"] == job_id
    assert s.json()["status"] in ("pending", "running", "completed", "failed")

    # Results: 409 while running, or the terminal payload once done.
    res = client.get(f"/api/ingest/results/{job_id}")
    assert res.status_code in (200, 409)

    # Jobs index includes the new job.
    jobs = client.get("/api/ingest/jobs").json()
    assert any(j["job_id"] == job_id for j in jobs)

    # Self-cleaning: remove the scratch upload dir for this job.
    import shutil as _shutil

    from pathlib import Path as _Path

    _uproot = _Path("data_preprocessing_pipeline") / ".uploads"
    if not _uproot.is_dir():
        _uproot = _Path(__file__).resolve().parent.parent / "data_preprocessing_pipeline" / ".uploads"
    _shutil.rmtree(_uproot / job_id, ignore_errors=True)


def test_registration_bundle_adjust_concurrent_load():
    """N parallel bundle-adjust calls all succeed with distinct results.

    Exercises the async job/compute path under concurrency (Step 14): the
    scipy solve runs in a threadpool, so parallel requests must not block
    each other or corrupt shared state.
    """
    import concurrent.futures

    import numpy as np

    def _one(seed: int) -> dict:
        rng = np.random.RandomState(seed)
        src = (rng.rand(12, 2) * 100).tolist()
        dst = [[x + 5.0, y - 3.0] for x, y in src]
        body = {
            "constraints": [
                {
                    "img_id_src": "A",
                    "img_id_ref": "B",
                    "src_pts": src,
                    "ref_pts": dst,
                    "initial_matrix": [[1, 0, 5], [0, 1, -3], [0, 0, 1]],
                },
                {
                    "img_id_src": "B",
                    "img_id_ref": "C",
                    "src_pts": dst,
                    "ref_pts": [[x - 2.0, y + 7.0] for x, y in dst],
                    "initial_matrix": [[1, 0, -2], [0, 1, 7], [0, 0, 1]],
                },
            ],
            "robust_loss": "huber",
            "max_iterations": 50,
        }
        r = client.post("/api/registration/bundle-adjust", json=body)
        assert r.status_code == 200, r.text[:300]
        return r.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_one, range(8)))
    assert len(results) == 8
    for res in results:
        assert res.get("status") in ("success", "converged_with_warnings")
        assert "optimized_matrices" in res or "matrices" in res


def test_job_manager_thread_safety_all_backends(tmp_path):
    """Concurrent create/update/log/get stays consistent (memory + sqlite)."""
    import concurrent.futures

    from job_store import MemoryJobStore, DbJobStore

    stores = [MemoryJobStore()]
    try:
        stores.append(DbJobStore(f"sqlite:///{tmp_path}/jobs.db"))
    except Exception as exc:
        raise AssertionError(f"sqlite job store unavailable: {exc}")

    for store in stores:
        def _worker(i: int) -> None:
            jid = f"job-{i}"
            store.create(jid, {"status": "pending", "logs": []})
            for k in range(5):
                store.update(jid, {"progress": float(k)})
                store.append_log(jid, f"line-{k}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_worker, range(16)))
        for i in range(16):
            job = store.get(f"job-{i}")
            assert job is not None
            assert job["progress"] == 4.0
            assert len(job["logs"]) == 5


def test_get_triplet_lro_candidates_endpoint(monkeypatch):
    """GET /triplets/{id}/lro-candidates returns candidate list."""
    candidates_sample = [
        {
            "product_id": "M1417670274LC",
            "label_url": "https://example.com/test.lbl",
            "download_urls": ["https://example.com/test.img"],
            "footprint_bounds": {"west_lon": 336.3, "east_lon": 336.7, "south_lat": -3.6, "north_lat": -3.2},
            "incidence_angle_deg": 5.82,
            "overlap_score": 0.95,
            "ranking_score": 0.95,
        }
    ]
    monkeypatch.setattr(
        "data.loader.get_triplet",
        lambda tid: {
            "id": tid,
            "bounds": {"west_lon": 336.48, "east_lon": 336.58, "south_lat": -3.51, "north_lat": -3.42},
            "ohrc_incidence_angle_deg": 5.82,
        },
    )
    monkeypatch.setattr("lro_ode_client.search_lro_nac_overlap", lambda bounds, **kw: candidates_sample)
    monkeypatch.setattr("lro_ode_client.rank_candidates", lambda c, b, **kw: candidates_sample)

    r = client.get("/triplets/region_001/lro-candidates")
    assert r.status_code == 200
    body = r.json()
    assert body["triplet_id"] == "region_001"
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["product_id"] == "M1417670274LC"




