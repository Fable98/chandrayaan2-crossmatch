"""
Unit tests for polygon-based lunar footprint overlap calculation.

Verifies:
1. Antimeridian crossing handling without polygon tearing (wrapping at +-180 deg).
2. Polar stereographic projection for high-latitude / polar footprints (lat >= 65 deg).
3. Disjoint footprints yielding 0 area and 0% overlap.
4. Partial overlap calculating correct intersection area and percentages.
5. Nested footprints (e.g. narrow OHRC inside wide TMC-2 strip).
6. PDS4 label parsing for exact polygon vertices, corner points, and bounding box fallback.
7. Overlap confidence categorization (polygon_exact, polygon_approximate, bbox_only, unavailable).
8. Geospatial accuracy against lunar datum (R = 1,737,400 m).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.ingestion.footprint_geometry import (
    MOON_RADIUS_M,
    OverlapConfidence,
    FootprintGeometry,
    FootprintOverlapResult,
    compute_footprint_overlap,
    get_lunar_eqc_crs,
    get_lunar_north_polar_stere_crs,
    get_lunar_south_polar_stere_crs,
    parse_footprint_from_pds4,
    select_lunar_projection,
    unwrap_antimeridian_longitudes,
)


def test_antimeridian_crossing():
    """Test footprints crossing +-180 deg longitude (antimeridian)."""
    # Footprint 1: [179 deg to -179 deg] -> spans 2 degrees across antimeridian
    # [179, -179] expressed as +179 to +181 deg
    fp1_coords = [
        (179.0, 10.0),
        (-179.0, 10.0),
        (-179.0, 12.0),
        (179.0, 12.0),
        (179.0, 10.0),
    ]
    # Footprint 2: [179.5 deg to -178.5 deg]
    fp2_coords = [
        (179.5, 10.0),
        (-178.5, 10.0),
        (-178.5, 12.0),
        (179.5, 12.0),
        (179.5, 10.0),
    ]

    # Test unwrap utility directly
    unwrapped1 = unwrap_antimeridian_longitudes(fp1_coords, ref_lon=180.0)
    for lon, lat in unwrapped1:
        assert 178.0 <= lon <= 182.0, f"Expected lon unwrapped near 180, got {lon}"

    res = compute_footprint_overlap(
        fp1_coords,
        fp2_coords,
        confidence1=OverlapConfidence.POLYGON_EXACT,
        confidence2=OverlapConfidence.POLYGON_EXACT,
    )

    assert res.confidence == OverlapConfidence.POLYGON_EXACT
    assert res.intersection_area_km2 > 0.0
    # Overlap should be ~75% of fp1 (1.5 deg out of 2 deg width)
    assert 70.0 <= res.overlap_pct1 <= 80.0
    assert 70.0 <= res.overlap_pct2 <= 80.0


def test_polar_footprints_south_and_north():
    """Test footprints located in high-latitude polar regions using polar stereographic."""
    # South Pole swath (Shackleton crater vicinity: -89.0 to -87.0 deg lat)
    sp_coords1 = [
        (0.0, -89.0),
        (30.0, -89.0),
        (30.0, -87.0),
        (0.0, -87.0),
        (0.0, -89.0),
    ]
    # Nested south pole swath with internal margin (-88.5 to -87.5 deg lat, 5.0 to 25.0 deg lon)
    sp_coords2 = [
        (5.0, -88.5),
        (25.0, -88.5),
        (25.0, -87.5),
        (5.0, -87.5),
        (5.0, -88.5),
    ]

    # Verify projection selection chooses South Polar Stereographic
    proj_crs, proj_name = select_lunar_projection(sp_coords1)
    assert "stere" in proj_crs
    assert "lat_0=-90" in proj_crs
    assert proj_name == "south_polar_stereographic"

    res = compute_footprint_overlap(sp_coords1, sp_coords2)
    assert res.confidence == OverlapConfidence.POLYGON_EXACT
    assert res.intersection_area_km2 > 0.0
    # sp_coords2 is entirely within sp_coords1
    assert pytest.approx(res.overlap_pct2, rel=1e-2) == 100.0
    assert 0.0 < res.overlap_pct1 < 100.0

    # North Pole footprint (lat +85 deg)
    np_coords = [(0.0, 85.0), (10.0, 85.0), (10.0, 86.0), (0.0, 86.0), (0.0, 85.0)]
    proj_np, proj_np_name = select_lunar_projection(np_coords)
    assert "stere" in proj_np
    assert "lat_0=90" in proj_np
    assert proj_np_name == "north_polar_stereographic"


def test_disjoint_footprints():
    """Test completely disjoint footprints return zero overlap."""
    # Footprint 1: Mare Tranquillitatis [20 to 25 lon, 5 to 10 lat]
    fp1 = [(20.0, 5.0), (25.0, 5.0), (25.0, 10.0), (20.0, 10.0), (20.0, 5.0)]
    # Footprint 2: Oceanus Procellarum [-60 to -55 lon, 5 to 10 lat]
    fp2 = [(-60.0, 5.0), (-55.0, 5.0), (-55.0, 10.0), (-60.0, 10.0), (-60.0, 5.0)]

    res = compute_footprint_overlap(fp1, fp2)
    assert res.intersection_area_km2 == 0.0
    assert res.overlap_pct1 == 0.0
    assert res.overlap_pct2 == 0.0
    assert res.iou == 0.0


def test_partial_overlap():
    """Test two swaths overlapping by approximately 50%."""
    # Footprint 1: lon 10 to 12, lat 0 to 2
    fp1 = [(10.0, 0.0), (12.0, 0.0), (12.0, 2.0), (10.0, 2.0), (10.0, 0.0)]
    # Footprint 2: lon 11 to 13, lat 0 to 2 (shifted by 1 degree east)
    fp2 = [(11.0, 0.0), (13.0, 0.0), (13.0, 2.0), (11.0, 2.0), (11.0, 0.0)]

    res = compute_footprint_overlap(fp1, fp2)
    assert res.confidence == OverlapConfidence.POLYGON_EXACT
    assert res.intersection_area_km2 > 0.0
    # Overlap is exactly 50% of each polygon
    assert pytest.approx(res.overlap_pct1, rel=1e-2) == 50.0
    assert pytest.approx(res.overlap_pct2, rel=1e-2) == 50.0
    # IoU for two equal squares shifted 50% is 1/3 (33.33%)
    assert pytest.approx(res.iou, rel=1e-2) == 1.0 / 3.0


def test_nested_footprints():
    """Test nested footprint (e.g. narrow OHRC strip inside wide TMC-2 swath)."""
    # Outer swath (TMC-2): lon 10 to 12, lat 0 to 2 (area ~ 4 deg^2)
    tmc2 = [(10.0, 0.0), (12.0, 0.0), (12.0, 2.0), (10.0, 2.0), (10.0, 0.0)]
    # Inner strip (OHRC): lon 10.5 to 11.0, lat 0.5 to 1.5 (area ~ 0.5 deg^2, completely inside)
    ohrc = [(10.5, 0.5), (11.0, 0.5), (11.0, 1.5), (10.5, 1.5), (10.5, 0.5)]

    res = compute_footprint_overlap(tmc2, ohrc)
    # OHRC (footprint 2) is 100% inside TMC-2
    assert pytest.approx(res.overlap_pct2, rel=1e-2) == 100.0
    # TMC-2 overlap fraction is OHRC area / TMC2 area (approx 0.5 / 4 = 12.5%)
    assert pytest.approx(res.overlap_pct1, rel=5e-2) == 12.5
    assert res.area1_km2 > res.area2_km2


def test_pds4_vertex_parsing_exact_bounding_polygon():
    """Test parsing exact polygon vertices from PDS4 XML with geom:Bounding_Polygon."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
                           xmlns:geom="http://pds.nasa.gov/pds4/geom/v1">
      <Identification_Area>
        <logical_identifier>urn:isro:ch2:tmc:ch2_tmc_ncn_20200101</logical_identifier>
      </Identification_Area>
      <geom:Geometry>
        <geom:Bounding_Polygon>
          <geom:pixel_latitude>10.0</geom:pixel_latitude>
          <geom:pixel_longitude>20.0</geom:pixel_longitude>
          <geom:pixel_latitude>10.0</geom:pixel_latitude>
          <geom:pixel_longitude>21.0</geom:pixel_longitude>
          <geom:pixel_latitude>11.0</geom:pixel_latitude>
          <geom:pixel_longitude>21.0</geom:pixel_longitude>
          <geom:pixel_latitude>11.0</geom:pixel_latitude>
          <geom:pixel_longitude>20.0</geom:pixel_longitude>
        </geom:Bounding_Polygon>
      </geom:Geometry>
    </Product_Observational>
    """
    fp_geom = parse_footprint_from_pds4(xml_content)
    assert fp_geom.confidence == OverlapConfidence.POLYGON_EXACT
    assert fp_geom.vertices is not None
    assert len(fp_geom.vertices) == 5  # 4 vertices + closed polygon
    assert fp_geom.wkt is not None and fp_geom.wkt.startswith("POLYGON")
    assert fp_geom.bbox == {"west_lon": 20.0, "east_lon": 21.0, "south_lat": 10.0, "north_lat": 11.0}


def test_pds4_vertex_parsing_corner_points():
    """Test parsing polygon from corner coordinate tags."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
      <Identification_Area>
        <logical_identifier>urn:isro:ch2:ohrc:ch2_ohr_ncn_20200101</logical_identifier>
      </Identification_Area>
      <Corner_Point>
        <corner_latitude>15.0</corner_latitude>
        <corner_longitude>30.0</corner_longitude>
      </Corner_Point>
      <Corner_Point>
        <corner_latitude>15.0</corner_latitude>
        <corner_longitude>30.2</corner_longitude>
      </Corner_Point>
      <Corner_Point>
        <corner_latitude>15.8</corner_latitude>
        <corner_longitude>30.2</corner_longitude>
      </Corner_Point>
      <Corner_Point>
        <corner_latitude>15.8</corner_latitude>
        <corner_longitude>30.0</corner_longitude>
      </Corner_Point>
    </Product_Observational>
    """
    fp_geom = parse_footprint_from_pds4(xml_content)
    assert fp_geom.confidence == OverlapConfidence.POLYGON_EXACT
    assert len(fp_geom.vertices) == 5
    assert fp_geom.bbox == {"west_lon": 30.0, "east_lon": 30.2, "south_lat": 15.0, "north_lat": 15.8}


def test_pds4_bounding_box_fallback():
    """Test fallback to bbox when polygon vertices are absent."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
      <Identification_Area>
        <logical_identifier>urn:isro:ch2:iirs:ch2_iir_ncn_20200101</logical_identifier>
      </Identification_Area>
      <Bounding_Coordinates>
        <west_bounding_coordinate>40.0</west_bounding_coordinate>
        <east_bounding_coordinate>42.0</east_bounding_coordinate>
        <south_bounding_coordinate>-5.0</south_bounding_coordinate>
        <north_bounding_coordinate>-3.0</north_bounding_coordinate>
      </Bounding_Coordinates>
    </Product_Observational>
    """
    fp_geom = parse_footprint_from_pds4(xml_content)
    assert fp_geom.confidence == OverlapConfidence.BBOX_ONLY
    assert fp_geom.bbox == {"west_lon": 40.0, "east_lon": 42.0, "south_lat": -5.0, "north_lat": -3.0}
    assert fp_geom.vertices is not None
    assert len(fp_geom.vertices) == 5


def test_confidence_propagation():
    """Test confidence degradation rules when combining polygons of different fidelities."""
    fp = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]

    # Exact + Exact -> POLYGON_EXACT
    r1 = compute_footprint_overlap(fp, fp, confidence1=OverlapConfidence.POLYGON_EXACT, confidence2=OverlapConfidence.POLYGON_EXACT)
    assert r1.confidence == OverlapConfidence.POLYGON_EXACT

    # Exact + Approximate -> POLYGON_APPROXIMATE
    r2 = compute_footprint_overlap(fp, fp, confidence1=OverlapConfidence.POLYGON_EXACT, confidence2=OverlapConfidence.POLYGON_APPROXIMATE)
    assert r2.confidence == OverlapConfidence.POLYGON_APPROXIMATE

    # Exact + Bbox -> BBOX_ONLY
    r3 = compute_footprint_overlap(fp, fp, confidence1=OverlapConfidence.POLYGON_EXACT, confidence2=OverlapConfidence.BBOX_ONLY)
    assert r3.confidence == OverlapConfidence.BBOX_ONLY

    # Missing -> UNAVAILABLE
    r4 = compute_footprint_overlap([], fp)
    assert r4.confidence == OverlapConfidence.UNAVAILABLE


def test_lunar_equatorial_area_metric_accuracy():
    """
    Test area calculation against theoretical IAU Moon sphere.
    Radius = 1,737.4 km.
    Theoretical area of 1 deg x 1 deg patch from lat 0 to 1 deg and lon 0 to 1 deg:
    A = R^2 * Delta_lambda * (sin(phi_2) - sin(phi_1))
      = (1737.4)^2 * (pi/180) * (sin(1 deg) - sin(0 deg))
      = 3,018,558.76 * 0.01745329 * 0.0174524 = ~919.46 km^2
    """
    fp = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    res = compute_footprint_overlap(fp, fp)

    r_km = MOON_RADIUS_M / 1000.0
    d_lambda_rad = math.radians(1.0)
    d_sin_phi = math.sin(math.radians(1.0)) - math.sin(0.0)
    theoretical_area_km2 = (r_km ** 2) * d_lambda_rad * d_sin_phi

    assert pytest.approx(res.area1_km2, rel=0.01) == theoretical_area_km2
