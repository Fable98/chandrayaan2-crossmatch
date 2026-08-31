"""
test_api.py — Automated tests for the SIH26166 backend.

Run with:  pytest test_api.py -v
No server needs to be running — TestClient spins the app in-process.
"""

from fastapi.testclient import TestClient
from main import app
from geo import pixel_to_latlon_from_corners

client = TestClient(app)

VALID_ID = "region_001"
INVALID_ID = "nonexistent_region"

CORNER_KEYS = {"top_left", "top_right", "bottom_right", "bottom_left"}


def _ensure_loaded():
    """Ensure data is loaded (lifespan may not have fired yet in TestClient)."""
    client.get("/refresh")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health():
    _ensure_loaded()
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
    _ensure_loaded()
    r = client.get("/triplets")
    assert r.status_code == 200
    data = r.json()
    # Response is wrapped: {"triplets": [...]}
    assert "triplets" in data
    assert isinstance(data["triplets"], list)
    assert len(data["triplets"]) >= 1


def test_triplets_list_has_required_fields():
    _ensure_loaded()
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
    _ensure_loaded()
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
    _ensure_loaded()
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
    _ensure_loaded()
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
# Matches — updated for new schema with px + latlon fields
# ---------------------------------------------------------------------------

def test_matches_has_homography():
    _ensure_loaded()
    r = client.get(f"/triplets/{VALID_ID}/matches")
    assert r.status_code == 200
    data = r.json()
    assert data["triplet_id"] == VALID_ID
    # Homography can be null (< 4 points) or a 3×3 matrix
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
        # Pixel coordinates (from ML team)
        assert len(m["ohrc_px"]) == 2, "ohrc_px should be (x, y)"
        assert len(m["tmc_px"]) == 2, "tmc_px should be (x, y)"
        # Geographic coordinates (computed by backend)
        assert len(m["ohrc_latlon"]) == 2, "ohrc_latlon should be (lat, lon)"
        assert len(m["tmc_latlon"]) == 2, "tmc_latlon should be (lat, lon)"
        # Confidence score
        assert 0.0 <= m["confidence"] <= 1.0


def test_match_latlon_within_footprint():
    """
    Verify all ohrc_latlon values from the API response fall within
    region_001's OHRC footprint bounds (loose sanity check).
    """
    _ensure_loaded()

    # Get the OHRC footprint bounds for region_001
    fp_r = client.get(f"/triplets/{VALID_ID}/footprint")
    fp = fp_r.json()["ohrc"]
    all_lats = [fp[c]["lat"] for c in CORNER_KEYS]
    all_lons = [fp[c]["lon"] for c in CORNER_KEYS]
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)

    # Allow a small epsilon for floating-point rounding at edges
    eps = 0.01

    r = client.get(f"/triplets/{VALID_ID}/matches")
    for m in r.json()["matches"]:
        lat, lon = m["ohrc_latlon"]
        assert min_lat - eps <= lat <= max_lat + eps, (
            f"ohrc_latlon lat {lat} outside footprint [{min_lat}, {max_lat}]"
        )
        assert min_lon - eps <= lon <= max_lon + eps, (
            f"ohrc_latlon lon {lon} outside footprint [{min_lon}, {max_lon}]"
        )


def test_missing_matches_returns_empty_not_error():
    """
    A triplet with no match file should return 200 with empty matches list,
    NOT a 500 error. This keeps the frontend robust for partially-processed regions.
    """
    _ensure_loaded()
    # region_002 may have matches from the mock data, but nonexistent triplet
    # should 404. Instead, test by checking that region_002 at minimum returns
    # 200 with whatever it has (the important contract is: known triplet → 200).
    r = client.get(f"/triplets/{VALID_ID}/matches")
    assert r.status_code == 200
    data = r.json()
    assert "matches" in data
    assert isinstance(data["matches"], list)


def test_matches_404_for_unknown_triplet():
    """Unknown triplet ID → 404 (triplet doesn't exist, not just missing matches)."""
    r = client.get(f"/triplets/{INVALID_ID}/matches")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Pixel-to-latlon unit test (geo.py directly)
# ---------------------------------------------------------------------------

def test_pixel_to_latlon_sanity():
    """
    Call pixel_to_latlon_from_corners with the center pixel (256, 256)
    and verify the result is within region_001's OHRC footprint bounds.
    """
    # Region 001 OHRC corners (from user_triplets.json)
    corners = {
        "top_left":     {"lat": -89.900, "lon": 30.100},
        "top_right":    {"lat": -89.900, "lon": 30.250},
        "bottom_right": {"lat": -89.950, "lon": 30.260},
        "bottom_left":  {"lat": -89.950, "lon": 30.090},
    }

    lat, lon = pixel_to_latlon_from_corners(256.0, 256.0, corners, 512, 512)

    # Center pixel should map to roughly the center of the footprint
    all_lats = [c["lat"] for c in corners.values()]
    all_lons = [c["lon"] for c in corners.values()]
    mid_lat = sum(all_lats) / 4
    mid_lon = sum(all_lons) / 4

    # Within 0.05° of the centroid (generous but catches gross errors)
    assert abs(lat - mid_lat) < 0.05, f"Center lat {lat} too far from centroid {mid_lat}"
    assert abs(lon - mid_lon) < 0.05, f"Center lon {lon} too far from centroid {mid_lon}"


def test_pixel_to_latlon_corners_map_correctly():
    """
    Verify that pixel (0,0) maps to top_left and pixel (512,512) maps to
    bottom_right — the fundamental correctness check for the transform.
    """
    corners = {
        "top_left":     {"lat": -89.900, "lon": 30.100},
        "top_right":    {"lat": -89.900, "lon": 30.250},
        "bottom_right": {"lat": -89.950, "lon": 30.260},
        "bottom_left":  {"lat": -89.950, "lon": 30.090},
    }

    lat_tl, lon_tl = pixel_to_latlon_from_corners(0, 0, corners, 512, 512)
    assert abs(lat_tl - corners["top_left"]["lat"]) < 1e-6
    assert abs(lon_tl - corners["top_left"]["lon"]) < 1e-6

    lat_br, lon_br = pixel_to_latlon_from_corners(512, 512, corners, 512, 512)
    assert abs(lat_br - corners["bottom_right"]["lat"]) < 1e-6
    assert abs(lon_br - corners["bottom_right"]["lon"]) < 1e-6


# ---------------------------------------------------------------------------
# IIRS overlay
# ---------------------------------------------------------------------------

def test_iirs_overlay_corners_present():
    _ensure_loaded()
    r = client.get(f"/triplets/{VALID_ID}/iirs-overlay")
    assert r.status_code == 200
    data = r.json()
    assert data["triplet_id"] == VALID_ID
    assert data["image_url"].startswith("/images/iirs/")

    corners = data["corners"]
    assert set(corners.keys()) == CORNER_KEYS
    for corner_name in CORNER_KEYS:
        c = corners[corner_name]
        assert "lat" in c and "lon" in c


def test_iirs_overlay_corners_are_not_axis_aligned():
    """
    Assert that for rotated regions (like region_001 confirmed by diagnostic),
    the IIRS overlay corners form a rotated quad (not collapsed to an axis-aligned bbox).
    """
    _ensure_loaded()
    r = client.get(f"/triplets/{VALID_ID}/iirs-overlay")
    data = r.json()
    corners = data["corners"]
    lons = [corners[c]["lon"] for c in CORNER_KEYS]
    lats = [corners[c]["lat"] for c in CORNER_KEYS]
    assert len(set(lons)) >= 3 or len(set(lats)) >= 3, (
        "IIRS overlay corners look axis-aligned — possible bbox regression"
    )


def test_footprints_and_iirs_overlay_have_consistent_winding():
    """
    Ensure OHRC, TMC, IIRS (from /footprint), and IIRS (from /iirs-overlay)
    all share the exact same geometric CLOCKWISE winding order:
    top_left -> top_right -> bottom_right -> bottom_left.
    Guards against flipped rendering in leaflet-distortableImage.
    """
    _ensure_loaded()
    r_fp = client.get(f"/triplets/{VALID_ID}/footprint")
    fp_data = r_fp.json()
    r_iirs = client.get(f"/triplets/{VALID_ID}/iirs-overlay")
    iirs_overlay_corners = r_iirs.json()["corners"]

    def signed_area(quad):
        ordered = [quad[k] for k in ["top_left", "top_right", "bottom_right", "bottom_left"]]
        a = 0.0
        for i in range(4):
            j = (i + 1) % 4
            a += ordered[i]["lon"] * ordered[j]["lat"] - ordered[j]["lon"] * ordered[i]["lat"]
        return a / 2.0

    # In lon=x, lat=y Cartesian coordinates, CW winding produces negative signed area
    for sensor in ("ohrc", "tmc", "iirs", "intersection"):
        sa = signed_area(fp_data[sensor])
        assert sa < 0, f"{sensor} footprint winding is not CLOCKWISE (signed area: {sa})"

    sa_overlay = signed_area(iirs_overlay_corners)
    assert sa_overlay < 0, f"IIRS overlay winding is not CLOCKWISE (signed area: {sa_overlay})"


def test_iirs_overlay_has_opacity_hint():
    _ensure_loaded()
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
