"""
tests/test_lro_projection_audit.py — Verification of map projection handling in data/ingestion/lro_basemap.py.

Requirements Verified:
1. Detects projection from product metadata rather than latitude alone.
2. Supports equirectangular, polar stereographic, and local projected products.
3. For latitude beyond ±60°, requires explicit projection metadata or returns REVIEW.
4. Never assumes a simple cylindrical projection for polar data.
5. Preserves coordinate reference information in LROProductMetadata.
6. Antimeridian crossing and near-polar coordinate transformation tests.
7. Validates that crop coordinates are transformed before array indexing.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.ingestion.lro_basemap import (
    MOON_RADIUS_M,
    LROProductMetadata,
    parse_lro_pds4_label,
    create_mock_pds4_product,
    crop_lro_basemap,
)


def test_detect_projection_from_metadata_not_latitude():
    """Requirement 1 & 2: Projection is detected from XML metadata, not inferred from latitude."""
    # Equatorial latitude with explicit Transverse Mercator (local projected)
    xml_eq = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
                           xmlns:cart="http://pds.nasa.gov/pds4/cart/v1">
      <Identification_Area>
        <logical_identifier>urn:nasa:pds:lro:wac_tmerc</logical_identifier>
      </Identification_Area>
      <cart:Cartography>
        <cart:Spatial_Domain>
          <cart:Bounding_Coordinates>
            <cart:west_bounding_coordinate>10.0</cart:west_bounding_coordinate>
            <cart:east_bounding_coordinate>15.0</cart:east_bounding_coordinate>
            <cart:north_bounding_coordinate>5.0</cart:north_bounding_coordinate>
            <cart:south_bounding_coordinate>0.0</cart:south_bounding_coordinate>
          </cart:Bounding_Coordinates>
        </cart:Spatial_Domain>
        <cart:Spatial_Reference_Information>
          <cart:Map_Projection>
            <cart:map_projection_name>Transverse Mercator</cart:map_projection_name>
            <cart:Transverse_Mercator>
              <cart:latitude_of_projection_origin>2.5</cart:latitude_of_projection_origin>
              <cart:longitude_of_central_meridian>12.5</cart:longitude_of_central_meridian>
            </cart:Transverse_Mercator>
          </cart:Map_Projection>
        </cart:Spatial_Reference_Information>
      </cart:Cartography>
    </Product_Observational>
    """
    meta = parse_lro_pds4_label(xml_eq)
    # Even though latitude is 0-5 deg (equatorial), it must be detected as local_projected / transverse mercator!
    assert meta["projection_name"] == "local_projected"
    assert "tmerc" in meta["proj4_crs"]
    assert meta["projection_status"] == "VALID"
    assert meta["projection_params"]["center_lat"] == 2.5
    assert meta["projection_params"]["center_lon"] == 12.5


def test_support_equirectangular_projection():
    """Requirement 2: Full support for Equirectangular projection with standard parallel."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
                           xmlns:cart="http://pds.nasa.gov/pds4/cart/v1">
      <Identification_Area>
        <logical_identifier>urn:nasa:pds:lro:wac_eqc</logical_identifier>
      </Identification_Area>
      <cart:Cartography>
        <cart:Spatial_Domain>
          <cart:Bounding_Coordinates>
            <cart:west_bounding_coordinate>-30.0</cart:west_bounding_coordinate>
            <cart:east_bounding_coordinate>-20.0</cart:east_bounding_coordinate>
            <cart:north_bounding_coordinate>20.0</cart:north_bounding_coordinate>
            <cart:south_bounding_coordinate>10.0</cart:south_bounding_coordinate>
          </cart:Bounding_Coordinates>
        </cart:Spatial_Domain>
        <cart:Spatial_Reference_Information>
          <cart:Map_Projection>
            <cart:map_projection_name>Equirectangular</cart:map_projection_name>
            <cart:Equirectangular>
              <cart:standard_parallel_1>15.0</cart:standard_parallel_1>
              <cart:longitude_of_central_meridian>-25.0</cart:longitude_of_central_meridian>
            </cart:Equirectangular>
          </cart:Map_Projection>
        </cart:Spatial_Reference_Information>
      </cart:Cartography>
    </Product_Observational>
    """
    meta = parse_lro_pds4_label(xml_content)
    assert meta["projection_name"] == "equirectangular"
    assert "eqc" in meta["proj4_crs"]
    assert "lat_ts=15.0" in meta["proj4_crs"]
    assert "lon_0=-25.0" in meta["proj4_crs"]
    assert meta["projection_status"] == "VALID"


def test_support_polar_stereographic_projection():
    """Requirement 2: Full support for Polar Stereographic projection."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
                           xmlns:cart="http://pds.nasa.gov/pds4/cart/v1">
      <Identification_Area>
        <logical_identifier>urn:nasa:pds:lro:wac_south_pole</logical_identifier>
      </Identification_Area>
      <cart:Cartography>
        <cart:Spatial_Domain>
          <cart:Bounding_Coordinates>
            <cart:west_bounding_coordinate>-180.0</cart:west_bounding_coordinate>
            <cart:east_bounding_coordinate>180.0</cart:east_bounding_coordinate>
            <cart:north_bounding_coordinate>-85.0</cart:north_bounding_coordinate>
            <cart:south_bounding_coordinate>-90.0</cart:south_bounding_coordinate>
          </cart:Bounding_Coordinates>
        </cart:Spatial_Domain>
        <cart:Spatial_Reference_Information>
          <cart:Map_Projection>
            <cart:map_projection_name>Polar Stereographic</cart:map_projection_name>
            <cart:Polar_Stereographic>
              <cart:latitude_of_projection_origin>-90.0</cart:latitude_of_projection_origin>
              <cart:longitude_of_central_meridian>0.0</cart:longitude_of_central_meridian>
              <cart:scale_factor_at_projection_origin>1.0</cart:scale_factor_at_projection_origin>
            </cart:Polar_Stereographic>
          </cart:Map_Projection>
        </cart:Spatial_Reference_Information>
      </cart:Cartography>
    </Product_Observational>
    """
    meta = parse_lro_pds4_label(xml_content)
    assert meta["projection_name"] == "polar_stereographic"
    assert "stere" in meta["proj4_crs"]
    assert "lat_0=-90.0" in meta["proj4_crs"]
    assert meta["projection_status"] == "VALID"


def test_polar_beyond_60_requires_explicit_projection_or_returns_review():
    """Requirement 3 & 4: For latitude beyond ±60°, requires explicit projection metadata or returns REVIEW."""
    # South Polar product (lat -85° to -70°) WITHOUT projection tags
    xml_polar_missing = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
                           xmlns:cart="http://pds.nasa.gov/pds4/cart/v1">
      <Identification_Area>
        <logical_identifier>urn:nasa:pds:lro:wac_polar_unspecified</logical_identifier>
      </Identification_Area>
      <cart:Cartography>
        <cart:Spatial_Domain>
          <cart:Bounding_Coordinates>
            <cart:west_bounding_coordinate>0.0</cart:west_bounding_coordinate>
            <cart:east_bounding_coordinate>45.0</cart:east_bounding_coordinate>
            <cart:north_bounding_coordinate>-70.0</cart:north_bounding_coordinate>
            <cart:south_bounding_coordinate>-85.0</cart:south_bounding_coordinate>
          </cart:Bounding_Coordinates>
        </cart:Spatial_Domain>
        <cart:Spatial_Reference_Information>
          <cart:pixel_resolution>100.0</cart:pixel_resolution>
        </cart:Spatial_Reference_Information>
      </cart:Cartography>
    </Product_Observational>
    """
    meta = parse_lro_pds4_label(xml_polar_missing)

    # Must return REVIEW
    assert meta["projection_status"] == "REVIEW"
    assert meta["projection_name"] == "unknown"
    assert meta["proj4_crs"] is None
    # Must NEVER assume simple cylindrical for polar data!
    assert "eqc" not in str(meta.get("proj4_crs"))
    assert "Polar data" in meta["review_reason"]

    # North Polar product (lat +65° to +80°) WITHOUT projection tags
    xml_north_missing = """<?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
                           xmlns:cart="http://pds.nasa.gov/pds4/cart/v1">
      <Identification_Area>
        <logical_identifier>urn:nasa:pds:lro:wac_north_unspecified</logical_identifier>
      </Identification_Area>
      <cart:Cartography>
        <cart:Spatial_Domain>
          <cart:Bounding_Coordinates>
            <cart:west_bounding_coordinate>10.0</cart:west_bounding_coordinate>
            <cart:east_bounding_coordinate>20.0</cart:east_bounding_coordinate>
            <cart:north_bounding_coordinate>80.0</cart:north_bounding_coordinate>
            <cart:south_bounding_coordinate>65.0</cart:south_bounding_coordinate>
          </cart:Bounding_Coordinates>
        </cart:Spatial_Domain>
      </cart:Cartography>
    </Product_Observational>
    """
    meta_north = parse_lro_pds4_label(xml_north_missing)
    assert meta_north["projection_status"] == "REVIEW"
    assert meta_north["projection_name"] == "unknown"
    assert meta_north["proj4_crs"] is None


def test_preserve_crs_in_lro_product_metadata():
    """Requirement 5: Preserves coordinate reference information in LROProductMetadata."""
    meta_obj = LROProductMetadata(
        product_id="lro_nac_test",
        sensor="LRO_NAC",
        min_lat=-80.0,
        max_lat=-75.0,
        min_lon=30.0,
        max_lon=35.0,
        gsd_m=0.5,
        width=1024,
        height=2048,
        projection_name="polar_stereographic",
        proj4_crs=f"+proj=stere +lat_0=-90 +lon_0=0 +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs +type=crs",
        projection_params={"pole_lat": -90.0, "center_lon": 0.0},
        projection_status="VALID",
    )

    d = meta_obj.to_dict()
    assert d["projection_name"] == "polar_stereographic"
    assert "stere" in d["proj4_crs"]
    assert d["projection_params"]["pole_lat"] == -90.0
    assert d["projection_status"] == "VALID"
    assert d["review_reason"] is None


def test_antimeridian_crossing_crop():
    """Requirement 6 & 7: Antimeridian crossing handles continuous coordinate transformation."""
    # Product spanning antimeridian: lon 170° to 190° (-170°), lat 10° to 20°
    h, w = 200, 200
    mock_raster = np.ones((h, w), dtype=np.float32)

    meta = {
        "product_id": "antimeridian_mosaic",
        "min_lat": 10.0,
        "max_lat": 20.0,
        "min_lon": 170.0,
        "max_lon": -170.0,  # 190 deg
        "width": w,
        "height": h,
        "projection_name": "equirectangular",
        "proj4_crs": f"+proj=eqc +lat_ts=15.0 +lon_0=180.0 +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs +type=crs",
        "projection_status": "VALID",
    }

    # Request crop spanning across 180°: lon 175° to -175° (185°)
    crop_bbox = (12.0, 18.0, 175.0, -175.0)
    cropped, crop_meta = crop_lro_basemap(mock_raster, meta, crop_bbox)

    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0
    assert crop_meta["crop_transform"]["width"] == cropped.shape[1]
    assert crop_meta["crop_transform"]["height"] == cropped.shape[0]


def test_polar_stereographic_crop_transforms_before_indexing():
    """Requirement 7: Validates that crop coordinates are transformed before array indexing."""
    # Polar stereographic raster centered on South Pole (-90°)
    h, w = 500, 500
    mock_raster = np.arange(h * w, dtype=np.float32).reshape((h, w))

    meta = {
        "product_id": "lro_south_pole_basemap",
        "min_lat": -90.0,
        "max_lat": -80.0,
        "min_lon": -180.0,
        "max_lon": 180.0,
        "width": w,
        "height": h,
        "projection_name": "polar_stereographic",
        "proj4_crs": f"+proj=stere +lat_0=-90.0 +lon_0=0.0 +k=1.0 +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs +type=crs",
        "projection_status": "VALID",
    }

    # Crop near South Pole Shackleton area: lat -89° to -85°, lon 0° to 30°
    crop_bbox = (-89.0, -85.0, 0.0, 30.0)
    cropped, crop_meta = crop_lro_basemap(mock_raster, meta, crop_bbox)

    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0
    assert crop_meta["crop_transform"]["col_off"] >= 0
    assert crop_meta["crop_transform"]["row_off"] >= 0
    assert crop_meta["dimensions"]["width"] == cropped.shape[1]
    assert crop_meta["dimensions"]["height"] == cropped.shape[0]


def test_crop_refuses_polar_data_under_review():
    """Requirement 3 & 7: Cropping polar data under REVIEW raises ValueError unless explicitly overridden."""
    mock_raster = np.zeros((100, 100), dtype=np.float32)
    meta = {
        "product_id": "unverified_polar_lro",
        "min_lat": -85.0,
        "max_lat": -70.0,
        "min_lon": 0.0,
        "max_lon": 20.0,
        "width": 100,
        "height": 100,
        "projection_status": "REVIEW",
        "review_reason": "Polar data beyond ±60° lacks explicit projection metadata.",
    }

    with pytest.raises(ValueError, match="REVIEW"):
        crop_lro_basemap(mock_raster, meta, (-80.0, -75.0, 5.0, 15.0))


def test_crop_rejects_completely_disjoint_window():
    """Requirement 7: Out-of-bounds crops are validated and rejected before indexing."""
    mock_raster = np.zeros((100, 100), dtype=np.float32)
    meta = {
        "product_id": "equatorial_lro",
        "min_lat": 0.0,
        "max_lat": 10.0,
        "min_lon": 0.0,
        "max_lon": 10.0,
        "width": 100,
        "height": 100,
        "projection_name": "equirectangular",
        "proj4_crs": f"+proj=eqc +lat_ts=0.0 +lon_0=0.0 +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs +type=crs",
        "projection_status": "VALID",
    }

    # Request window completely outside product (lat 30° to 40°)
    with pytest.raises(ValueError, match="disjoint"):
        crop_lro_basemap(mock_raster, meta, (30.0, 40.0, 0.0, 10.0))
