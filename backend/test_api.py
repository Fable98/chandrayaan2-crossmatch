"""
test_api.py — Automated tests for the SIH26166 backend.

Run with:  pytest test_api.py -v
No server needs to be running — TestClient spins the app in-process.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

VALID_ID = "region_001"
INVALID_ID = "nonexistent_region"

CORNER_KEYS = {"top_left", "top_right", "bottom_right", "bottom_left"}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health():
    # Ensure data is loaded (lifespan may not have fired yet in TestClient)
    client.get("/refresh")
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["triplets_loaded"] >= 1


def test_refresh():
    r = client.get("/refresh")
    assert r.status_code == 200
    assert r.json()["status"] == "refreshed"
    assert r.json()["triplets_loaded"] >= 1


# ---------------------------------------------------------------------------
# Triplets list
# ---------------------------------------------------------------------------

def test_triplets_list():
    r = client.get("/triplets")
    assert r.status_code == 200
    data = r.json()
    # Response is wrapped: {"triplets": [...]}
    assert "triplets" in data
    assert isinstance(data["triplets"], list)
    assert len(data["triplets"]) >= 1


def test_triplets_list_has_required_fields():
    r = client.get("/triplets")
    triplet = r.json()["triplets"][0]
    assert "id" in triplet
    assert "sensors" in triplet
    assert "intersection_footprint" in triplet
    assert len(triplet["sensors"]) == 3


# ---------------------------------------------------------------------------
# Triplet detail
# ---------------------------------------------------------------------------

def test_triplet_detail():
    r = client.get(f"/triplets/{VALID_ID}")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == VALID_ID
    assert len(data["sensors"]) == 3

    sensor_names = {s["sensor"] for s in data["sensors"]}
    assert sensor_names == {"ohrc", "tmc", "iirs"}


def test_triplet_detail_404():
    r = client.get(f"/triplets/{INVALID_ID}")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Footprint — the bbox-vs-quad regression guard
# ---------------------------------------------------------------------------

def test_footprint_has_four_named_corners_not_bbox():
    """
    REGRESSION GUARD: ensures footprints are 4-corner quads (top_left,
    top_right, bottom_right, bottom_left), NOT axis-aligned bounding boxes.
    A previous min/max bbox implementation caused visible misalignment on
    the Leaflet map. This test must never be removed.
    """
    r = client.get(f"/triplets/{VALID_ID}/footprint")
    assert r.status_code == 200
    data = r.json()

    # Each sensor + intersection must have exactly 4 named corners
    for key in ("ohrc", "tmc", "iirs", "intersection"):
        footprint = data[key]
        assert set(footprint.keys()) == CORNER_KEYS, (
            f"{key} footprint has keys {set(footprint.keys())}, "
            f"expected named corners {CORNER_KEYS} — NOT a bbox"
        )
        # Each corner must be a {lat, lon} object
        for corner_name in CORNER_KEYS:
            corner = footprint[corner_name]
            assert "lat" in corner and "lon" in corner, (
                f"{key}.{corner_name} missing lat/lon"
            )


def test_footprint_corners_are_not_axis_aligned():
    """
    Extra safety: for OHRC/TMC, if the quad is truly rotated, not all
    lats or lons will be identical across corners. This catches accidental
    degenerate data where someone replaced a real quad with a bbox.

    NOTE: may need to be relaxed if some real quads happen to be axis-aligned,
    but for our current lunar south-pole data the slight rotation is expected.
    """
    r = client.get(f"/triplets/{VALID_ID}/footprint")
    data = r.json()

    for sensor in ("ohrc", "tmc"):
        fp = data[sensor]
        lons = [fp[c]["lon"] for c in CORNER_KEYS]
        lats = [fp[c]["lat"] for c in CORNER_KEYS]
        # A true quad should have at least 3 distinct coordinate values
        # (a bbox would have exactly 2 distinct lats and 2 distinct lons)
        assert len(set(lons)) >= 3 or len(set(lats)) >= 3, (
            f"{sensor} footprint looks axis-aligned — possible bbox regression"
        )


def test_footprint_404():
    r = client.get(f"/triplets/{INVALID_ID}/footprint")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

def test_matches_has_homography():
    r = client.get(f"/triplets/{VALID_ID}/matches")
    assert r.status_code == 200
    data = r.json()
    assert data["triplet_id"] == VALID_ID
    # 3×3 homography matrix
    assert len(data["homography"]) == 3
    assert all(len(row) == 3 for row in data["homography"])


def test_matches_has_points_with_confidence():
    r = client.get(f"/triplets/{VALID_ID}/matches")
    data = r.json()
    assert data["num_matches"] == len(data["matches"])
    assert data["num_matches"] >= 1

    for m in data["matches"]:
        assert len(m["ohrc_pixel"]) == 2
        assert len(m["tmc_pixel"]) == 2
        assert 0.0 <= m["confidence"] <= 1.0


def test_matches_404():
    r = client.get(f"/triplets/{INVALID_ID}/matches")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# IIRS overlay
# ---------------------------------------------------------------------------

def test_iirs_overlay_bounds_present():
    r = client.get(f"/triplets/{VALID_ID}/iirs-overlay")
    assert r.status_code == 200
    data = r.json()
    assert data["triplet_id"] == VALID_ID
    assert data["image_url"].startswith("/images/iirs/")

    b = data["bounds"]
    assert b["west_lon"] < b["east_lon"]
    assert b["south_lat"] < b["north_lat"]


def test_iirs_overlay_has_opacity_hint():
    r = client.get(f"/triplets/{VALID_ID}/iirs-overlay")
    data = r.json()
    assert 0.0 <= data["opacity_hint"] <= 1.0


def test_iirs_overlay_404():
    r = client.get(f"/triplets/{INVALID_ID}/iirs-overlay")
    assert r.status_code == 404


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
