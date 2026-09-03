"""
test_api.py — Automated tests for the SIH26166 backend (Shared-Bbox Architecture + 6 Regions + DEM).

Run with:  pytest test_api.py -v
No server needs to be running — TestClient spins the app in-process.
"""

from fastapi.testclient import TestClient
from main import app
from geo import (
    pixel_to_latlon_from_bounds,
    pixel_to_latlon_from_bounds_batch,
)

client = TestClient(app)

VALID_ID = "region_001"
INVALID_ID = "nonexistent_region"

BOUNDS_KEYS = {"west_lon", "east_lon", "south_lat", "north_lat"}


def _ensure_loaded():
    """Ensure data is loaded (lifespan may not have fired yet in TestClient)."""
    client.get("/refresh")


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
    r = client.get("/refresh")
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


