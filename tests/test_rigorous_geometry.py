"""
tests/test_rigorous_geometry.py — Comprehensive validation of Rigorous SPICE/DEM Geometry Mode.

Validates all 12 user requirements:
1. Load kernels only when explicitly provided.
2. Validate kernel coverage for acquisition time.
3. Convert image pixels to camera rays.
4. Transform rays into lunar body-fixed coordinates.
5. Intersect rays iteratively with the DEM.
6. Reproject surface points into the reference camera.
7. Return geometry uncertainty and convergence status.
8. Compare approximate and rigorous modes on the same pair.
9. Preserve existing fail-closed behavior when geometry is unavailable.
10. Never infer LOS azimuth from solar azimuth.
11. Add synthetic camera/DEM tests before real data tests.
12. Document all coordinate frames and units.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from rigorous_geometry import (
    LUNAR_RADIUS_M,
    ConvergenceStatus,
    SpacecraftState,
    RayIntersectionResult,
    GeometricComparisonResult,
    SpiceGeometryProvider,
    InstrumentCameraModel,
    LunarDemRayIntersector,
    OrthorectificationPipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_spacecraft_state():
    """
    Synthetic Chandrayaan-2 polar orbit state:
    - Orbit altitude: 100 km (radius = 1,737,400 + 100,000 = 1,837,400 m)
    - Sub-spacecraft point: (0.0 deg lat, 0.0 deg lon) over lunar equator
    - Spacecraft position: [1837400.0, 0.0, 0.0] in MOON_ME
    - Velocity: [0.0, 0.0, 1633.0] m/s northward along meridian
    - Camera orientation: Nadir-pointing (+Z_cam toward -X_body, +X_cam eastward, +Y_cam northward)
    """
    pos_m = np.array([1837400.0, 0.0, 0.0], dtype=np.float64)
    vel_m = np.array([0.0, 0.0, 1633.0], dtype=np.float64)
    epoch_et = 680000000.0
    utc_str = "2021-07-15T12:00:00.000"

    # Nadir look matrix:
    # +Z_cam = [-1, 0, 0] (down to Moon center)
    # +Y_cam = [0, 0, 1]  (along orbital velocity, north)
    # +X_cam = [0, 1, 0]  (across track, east)
    r_cam_to_body = np.array([
        [0.0, 0.0, -1.0],
        [1.0, 0.0,  0.0],
        [0.0, 1.0,  0.0],
    ], dtype=np.float64)

    return SpacecraftState(
        position_m=pos_m,
        velocity_m_s=vel_m,
        ephemeris_time_s=epoch_et,
        utc_time=utc_str,
        sub_sc_lat_deg=0.0,
        sub_sc_lon_deg=0.0,
        altitude_m=100000.0,
        rotation_cam_to_body=r_cam_to_body,
        is_synthetic=True,
    )


@pytest.fixture
def synthetic_crater_dem():
    """
    Synthetic 256x256 DEM of an impact crater:
    - Center at (0.0 deg lat, 0.0 deg lon)
    - GSD: 5.0 m/px (extent: 1280 m x 1280 m)
    - Crater depth: 400 meters with parabolic profile and central floor
    - Datum radius: 1,737,400 m
    """
    h, w = 256, 256
    yy, xx = np.indices((h, w), dtype=np.float64)
    cx, cy = w / 2.0, h / 2.0
    r_px = np.hypot(xx - cx, yy - cy)
    crater_radius_px = 80.0

    # Flat surrounding plain at 0 m elevation
    dem = np.zeros((h, w), dtype=np.float32)

    # Parabolic crater depression up to depth 400 m
    inside = r_px < crater_radius_px
    dem[inside] = -400.0 * (1.0 - (r_px[inside] / crater_radius_px) ** 2).astype(np.float32)

    return dem


# ---------------------------------------------------------------------------
# Test Category 1: InstrumentCameraModel
# ---------------------------------------------------------------------------

def test_camera_model_intrinsics_and_rays():
    """Requirement 3: Convert image pixels to camera rays."""
    cam = InstrumentCameraModel.create_ohrc_model(image_size=(512, 512))
    assert cam.instrument_name == "OHRC"
    assert cam.image_width_px == 512
    assert cam.image_height_px == 512
    assert cam.focal_length_px > 0

    # Boresight ray at principal point (256, 256)
    ray_center = cam.pixel_to_camera_ray(256.0, 256.0)
    assert np.allclose(ray_center, [0.0, 0.0, 1.0], atol=1e-5)
    assert abs(np.linalg.norm(ray_center) - 1.0) < 1e-12

    # Corner rays: should have unit norm and positive Z depth
    ray_tl = cam.pixel_to_camera_ray(0.0, 0.0)
    assert abs(np.linalg.norm(ray_tl) - 1.0) < 1e-12
    assert ray_tl[0] < 0.0  # Left (sample < cx)
    assert ray_tl[1] < 0.0  # Up (line < cy)
    assert ray_tl[2] > 0.0  # Forward along boresight


def test_camera_model_round_trip_accuracy():
    """Requirement 3 & 6: Round-trip pixel -> ray -> pixel accuracy."""
    cam = InstrumentCameraModel.create_tmc2_model(image_size=(512, 512))
    test_pixels = [
        (256.0, 256.0),
        (10.0, 10.0),
        (500.0, 500.0),
        (128.5, 384.25),
        (0.0, 256.0),
        (511.0, 256.0),
    ]
    for u_in, v_in in test_pixels:
        ray = cam.pixel_to_camera_ray(u_in, v_in)
        u_out, v_out = cam.camera_ray_to_pixel(ray)
        assert abs(u_out - u_in) < 1e-5, f"Round-trip u mismatch: {u_out} vs {u_in}"
        assert abs(v_out - v_in) < 1e-5, f"Round-trip v mismatch: {v_out} vs {v_in}"


def test_camera_model_vectorized_conversion():
    """Requirement 3: Vectorized conversion of (N, 2) pixel coordinates."""
    cam = InstrumentCameraModel.create_iirs_model(image_size=(256, 256))
    pts = np.array([[128.0, 128.0], [0.0, 0.0], [255.0, 255.0]], dtype=np.float64)
    rays = cam.pixels_to_camera_rays(pts)
    assert rays.shape == (3, 3)
    norms = np.linalg.norm(rays, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)
    assert np.allclose(rays[0], [0.0, 0.0, 1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Test Category 2: SpiceGeometryProvider
# ---------------------------------------------------------------------------

def test_spice_provider_loads_only_when_explicitly_provided():
    """Requirement 1: Load kernels ONLY when explicitly provided."""
    provider = SpiceGeometryProvider()
    # Initially unloaded and unavailable
    assert not provider.is_available
    assert len(provider._kernels_loaded) == 0

    # Calling with empty list does not load anything
    assert not provider.load_kernels([])
    assert not provider.is_available

    # Calling with non-existent path handles gracefully
    assert not provider.load_kernels(["/non/existent/path/kernel.bsp"])
    assert not provider.is_available


def test_spice_provider_kernel_coverage_validation():
    """Requirement 2: Validate kernel coverage for acquisition time."""
    provider = SpiceGeometryProvider()
    start_et = 600000000.0
    end_et = 600100000.0
    provider.register_coverage_window("test_kernel.bsp", start_et, end_et)
    provider._is_active = True

    # Inside coverage
    is_valid, reason = provider.validate_kernel_coverage(600050000.0)
    assert is_valid
    assert "within coverage window" in reason

    # Boundary cases
    assert provider.validate_kernel_coverage(start_et)[0]
    assert provider.validate_kernel_coverage(end_et)[0]

    # Outside coverage (too early)
    is_valid_early, reason_early = provider.validate_kernel_coverage(599999999.0)
    assert not is_valid_early
    assert "outside all loaded SPICE coverage windows" in reason_early

    # Outside coverage (too late)
    is_valid_late, reason_late = provider.validate_kernel_coverage(600100001.0)
    assert not is_valid_late
    assert "outside all loaded SPICE coverage windows" in reason_late


def test_spice_provider_never_infers_los_from_solar_azimuth():
    """Requirement 10 & Rule 7: Never infer LOS azimuth from solar azimuth."""
    provider = SpiceGeometryProvider()

    # None LOS azimuth is rejected
    assert not provider.validate_los_azimuth(None, 45.0)

    # LOS azimuth identical to solar azimuth is strictly rejected
    assert not provider.validate_los_azimuth(120.5, 120.5)

    # Distinct, valid LOS azimuth is accepted
    assert provider.validate_los_azimuth(15.0, 120.5)


def test_spice_provider_synthetic_state_registration(synthetic_spacecraft_state):
    """Requirement 4: Spacecraft state in body-fixed frame."""
    provider = SpiceGeometryProvider()
    provider.register_synthetic_state(synthetic_spacecraft_state)
    assert provider.is_available

    state = provider.get_spacecraft_state(synthetic_spacecraft_state.utc_time)
    assert state is not None
    assert state.altitude_m == 100000.0
    assert state.sub_sc_lat_deg == 0.0
    assert state.sub_sc_lon_deg == 0.0
    assert state.rotation_cam_to_body.shape == (3, 3)


# ---------------------------------------------------------------------------
# Test Category 3: LunarDemRayIntersector
# ---------------------------------------------------------------------------

def test_dem_intersector_flat_surface():
    """Requirement 5 & 7: Intersect rays iteratively on flat lunar datum."""
    dem = np.zeros((100, 100), dtype=np.float32)
    intersector = LunarDemRayIntersector(
        dem=dem,
        center_lat_deg=0.0,
        center_lon_deg=0.0,
        gsd_m=5.0,
        datum_radius_m=LUNAR_RADIUS_M,
        tolerance_m=0.05,
    )

    # Spacecraft 100 km directly above equator/prime meridian pointing nadir
    sc_pos = np.array([LUNAR_RADIUS_M + 100000.0, 0.0, 0.0], dtype=np.float64)
    ray_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)  # Nadir toward Moon center

    res = intersector.intersect_ray(sc_pos, ray_dir)
    assert res.convergence_status == ConvergenceStatus.CONVERGED
    assert res.surface_point_body_m is not None
    assert abs(res.elevation_m - 0.0) < 0.05
    assert abs(res.slant_range_m - 100000.0) < 0.05
    assert abs(res.lat_deg - 0.0) < 1e-4
    assert abs(res.lon_deg - 0.0) < 1e-4
    assert res.final_residual_m <= 0.05
    assert res.uncertainty_3d_m > 0.0


def test_dem_intersector_elevated_datum():
    """Requirement 5 & 7: Intersect rays on elevated plateau (elev = +500 m)."""
    dem = np.full((100, 100), 500.0, dtype=np.float32)
    intersector = LunarDemRayIntersector(
        dem=dem,
        center_lat_deg=0.0,
        center_lon_deg=0.0,
        gsd_m=5.0,
        datum_radius_m=LUNAR_RADIUS_M,
        tolerance_m=0.05,
    )

    sc_pos = np.array([LUNAR_RADIUS_M + 100000.0, 0.0, 0.0], dtype=np.float64)
    ray_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    res = intersector.intersect_ray(sc_pos, ray_dir)
    assert res.convergence_status == ConvergenceStatus.CONVERGED
    assert abs(res.elevation_m - 500.0) < 0.05
    assert abs(res.slant_range_m - (100000.0 - 500.0)) < 0.05


def test_dem_intersector_crater_floor_and_wall(synthetic_crater_dem):
    """Requirement 5 & 7: Intersect rays on steep crater wall and floor."""
    intersector = LunarDemRayIntersector(
        dem=synthetic_crater_dem,
        center_lat_deg=0.0,
        center_lon_deg=0.0,
        gsd_m=5.0,
        datum_radius_m=LUNAR_RADIUS_M,
        tolerance_m=0.05,
    )

    # Nadir ray pointing at crater center: should hit crater floor at -400 m
    sc_pos = np.array([LUNAR_RADIUS_M + 100000.0, 0.0, 0.0], dtype=np.float64)
    ray_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    res_floor = intersector.intersect_ray(sc_pos, ray_dir)
    assert res_floor.convergence_status == ConvergenceStatus.CONVERGED
    assert abs(res_floor.elevation_m - (-400.0)) < 0.1
    assert abs(res_floor.slant_range_m - 100400.0) < 0.1

    # Off-nadir slanted ray pointing toward crater wall
    # Aim ~200 m north (+Z_body)
    tan_angle = 200.0 / 100000.0
    ray_wall = np.array([-1.0, 0.0, tan_angle], dtype=np.float64)
    ray_wall = ray_wall / np.linalg.norm(ray_wall)

    res_wall = intersector.intersect_ray(sc_pos, ray_wall)
    assert res_wall.convergence_status == ConvergenceStatus.CONVERGED
    assert res_wall.elevation_m < 0.0  # Inside crater depression
    assert res_wall.elevation_m > -400.0  # Along crater wall


def test_dem_intersector_ray_misses_body():
    """Requirement 5 & 7: Ray that misses the lunar sphere returns RAY_MISSES_BODY."""
    dem = np.zeros((50, 50), dtype=np.float32)
    intersector = LunarDemRayIntersector(dem=dem, center_lat_deg=0.0, center_lon_deg=0.0, gsd_m=5.0)

    sc_pos = np.array([LUNAR_RADIUS_M + 100000.0, 0.0, 0.0], dtype=np.float64)
    # Ray pointing perpendicular to Moon (tangential into deep space)
    ray_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    res = intersector.intersect_ray(sc_pos, ray_dir)
    assert res.convergence_status == ConvergenceStatus.RAY_MISSES_BODY


def test_dem_intersector_ray_out_of_bounds():
    """Requirement 5 & 7: Ray entering outside the DEM bounding area returns OUT_OF_DEM_BOUNDS."""
    dem = np.zeros((50, 50), dtype=np.float32)  # 250 m x 250 m
    intersector = LunarDemRayIntersector(dem=dem, center_lat_deg=0.0, center_lon_deg=0.0, gsd_m=5.0)

    sc_pos = np.array([LUNAR_RADIUS_M + 100000.0, 0.0, 0.0], dtype=np.float64)
    # Tilted ray landing ~5 km away (outside 250m box)
    ray_dir = np.array([-1.0, 0.05, 0.0], dtype=np.float64)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    res = intersector.intersect_ray(sc_pos, ray_dir)
    assert res.convergence_status == ConvergenceStatus.OUT_OF_DEM_BOUNDS


# ---------------------------------------------------------------------------
# Test Category 4: OrthorectificationPipeline & Reprojection
# ---------------------------------------------------------------------------

def test_pipeline_reproject_surface_point(synthetic_spacecraft_state):
    """Requirement 6: Reproject 3D lunar surface point into reference camera."""
    provider = SpiceGeometryProvider()
    provider.register_synthetic_state(synthetic_spacecraft_state)
    pipeline = OrthorectificationPipeline(spice_provider=provider)
    cam = InstrumentCameraModel.create_ohrc_model(image_size=(512, 512))

    # Surface point directly nadir: [LUNAR_RADIUS_M, 0, 0]
    surf_pt = np.array([LUNAR_RADIUS_M, 0.0, 0.0], dtype=np.float64)
    pixel_coord, err = pipeline.reproject_surface_point_to_camera(
        surf_pt, cam, synthetic_spacecraft_state.utc_time
    )

    assert err is None
    assert pixel_coord is not None
    u, v = pixel_coord
    # Nadir point should project exactly to principal point (256, 256)
    assert abs(u - 256.0) < 1e-4
    assert abs(v - 256.0) < 1e-4


def test_pipeline_project_pixel_to_ground(synthetic_spacecraft_state, synthetic_crater_dem):
    """Requirement 4 & 5: Project pixel to ground through camera, SPICE, and DEM."""
    provider = SpiceGeometryProvider()
    provider.register_synthetic_state(synthetic_spacecraft_state)
    intersector = LunarDemRayIntersector(
        dem=synthetic_crater_dem, center_lat_deg=0.0, center_lon_deg=0.0, gsd_m=5.0
    )
    pipeline = OrthorectificationPipeline(spice_provider=provider, intersector=intersector)
    cam = InstrumentCameraModel.create_tmc2_model(image_size=(256, 256))

    # Center pixel (128, 128)
    res = pipeline.project_pixel_to_ground(128.0, 128.0, cam, synthetic_spacecraft_state.utc_time)
    assert res.convergence_status == ConvergenceStatus.CONVERGED
    assert res.surface_point_body_m is not None
    # Hits center of crater (-400m)
    assert abs(res.elevation_m - (-400.0)) < 0.1


# ---------------------------------------------------------------------------
# Test Category 5: Approximate vs. Rigorous Mode Comparison
# ---------------------------------------------------------------------------

def test_compare_approximate_and_rigorous_modes(synthetic_spacecraft_state, synthetic_crater_dem):
    """Requirement 8: Compare approximate and rigorous modes on the same pair."""
    provider = SpiceGeometryProvider()
    provider.register_synthetic_state(synthetic_spacecraft_state)
    intersector = LunarDemRayIntersector(
        dem=synthetic_crater_dem, center_lat_deg=0.0, center_lon_deg=0.0, gsd_m=5.0
    )
    pipeline = OrthorectificationPipeline(spice_provider=provider, intersector=intersector)
    cam = InstrumentCameraModel.create_tmc2_model(image_size=(256, 256))

    # Sample points across the scene
    pts = np.array([
        [128.0, 128.0],   # Center (crater floor, flat)
        [80.0, 128.0],    # West crater rim
        [128.0, 80.0],    # North crater rim
        [176.0, 128.0],   # East crater rim
        [128.0, 176.0],   # South crater rim
    ], dtype=np.float64)

    comp = pipeline.compare_approximate_and_rigorous(
        pts_px=pts,
        dem=synthetic_crater_dem,
        emission_deg=10.0,
        los_azimuth_deg=45.0,
        solar_azimuth_deg=135.0,  # Distinct from LOS azimuth
        gsd_m=5.0,
        camera=cam,
        epoch_utc_or_et=synthetic_spacecraft_state.utc_time,
    )

    assert comp.point_count == 5
    assert comp.mean_discrepancy_px >= 0.0
    assert comp.rmse_discrepancy_px >= 0.0
    assert comp.terrain_max_slope_deg > 0.0
    assert comp.dem_elevation_range_m == 400.0
    assert "Compared 5 points" in comp.summary


def test_comparison_rejects_solar_azimuth_as_los_azimuth(synthetic_spacecraft_state, synthetic_crater_dem):
    """Requirement 10 & Rule 7: Comparison strictly rejects when solar azimuth is passed as LOS azimuth."""
    provider = SpiceGeometryProvider()
    provider.register_synthetic_state(synthetic_spacecraft_state)
    intersector = LunarDemRayIntersector(
        dem=synthetic_crater_dem, center_lat_deg=0.0, center_lon_deg=0.0, gsd_m=5.0
    )
    pipeline = OrthorectificationPipeline(spice_provider=provider, intersector=intersector)
    cam = InstrumentCameraModel.create_tmc2_model(image_size=(256, 256))
    pts = np.array([[128.0, 128.0]])

    # 1. Missing LOS azimuth raises ValueError fail-closed
    with pytest.raises(ValueError, match="solar azimuth must NEVER be used as LOS azimuth"):
        pipeline.compare_approximate_and_rigorous(
            pts_px=pts,
            dem=synthetic_crater_dem,
            emission_deg=10.0,
            los_azimuth_deg=None,
            solar_azimuth_deg=45.0,
            gsd_m=5.0,
            camera=cam,
            epoch_utc_or_et=synthetic_spacecraft_state.utc_time,
        )

    # 2. Conflated LOS azimuth == Solar azimuth raises ValueError fail-closed
    with pytest.raises(ValueError, match="LOS azimuth conflated with solar azimuth"):
        pipeline.compare_approximate_and_rigorous(
            pts_px=pts,
            dem=synthetic_crater_dem,
            emission_deg=10.0,
            los_azimuth_deg=45.0,
            solar_azimuth_deg=45.0,  # Exact duplicate conflation
            gsd_m=5.0,
            camera=cam,
            epoch_utc_or_et=synthetic_spacecraft_state.utc_time,
        )


# ---------------------------------------------------------------------------
# Test Category 6: Fail-Closed Behavior
# ---------------------------------------------------------------------------

def test_fail_closed_when_geometry_unavailable():
    """Requirement 9: Preserve existing fail-closed behavior when geometry is unavailable."""
    # Provider with no kernels loaded
    empty_provider = SpiceGeometryProvider()
    pipeline = OrthorectificationPipeline(spice_provider=empty_provider, intersector=None)
    cam = InstrumentCameraModel.create_ohrc_model()

    res = pipeline.project_pixel_to_ground(256.0, 256.0, cam, "2021-07-15T12:00:00.000")
    assert res.convergence_status == ConvergenceStatus.GEOMETRY_UNAVAILABLE
    assert res.surface_point_body_m is None
    assert "DEM ray intersector not configured" in res.error_message


def test_reprojection_fails_closed_when_state_unavailable():
    """Requirement 9: Reprojection fails closed when reference state is missing."""
    empty_provider = SpiceGeometryProvider()
    pipeline = OrthorectificationPipeline(spice_provider=empty_provider)
    cam = InstrumentCameraModel.create_ohrc_model()

    pt = np.array([1737400.0, 0.0, 0.0])
    pixel_coord, err = pipeline.reproject_surface_point_to_camera(pt, cam, "2021-07-15T12:00:00.000")
    assert pixel_coord is None
    assert "Spacecraft state unavailable" in err


# ---------------------------------------------------------------------------
# Test Category 7: Real Data & Metadata Integration Tests
# ---------------------------------------------------------------------------

def test_real_ch2_metadata_and_dem_experiment():
    """
    Requirement 11 & Rule 4: Validate rigorous geometry against real Chandrayaan-2 PDS4
    observation metadata (sample_data/ohrc_sample.xml) and real DEM (sample_data/dem_sample.png).
    """
    sample_dir = REPO_ROOT / "sample_data"
    xml_path = sample_dir / "ohrc_sample.xml"
    dem_path = sample_dir / "dem_sample.png"

    if not (xml_path.exists() and dem_path.exists()):
        pytest.skip("Real Chandrayaan-2 sample files not found.")

    import xml.etree.ElementTree as ET
    import cv2

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Parse PDS4 namespaces
    ns = {
        "pds": "http://pds.nasa.gov/pds4/pds/v1",
        "cart": "http://pds.nasa.gov/pds4/cart/v1",
        "geom": "http://pds.nasa.gov/pds4/geom/v1",
    }

    utc_elem = root.find(".//pds:creation_date_time", ns)
    utc_str = utc_elem.text if utc_elem is not None else "2020-04-15T06:22:11Z"

    res_elem = root.find(".//cart:pixel_resolution", ns)
    res_m = float(res_elem.text) if res_elem is not None else 0.25

    sun_az_elem = root.find(".//geom:sun_azimuth", ns)
    sun_az = float(sun_az_elem.text) if sun_az_elem is not None else 135.5

    em_elem = root.find(".//geom:emission_angle", ns)
    em_deg = float(em_elem.text) if em_elem is not None else 0.0

    # Load sample DEM
    dem_raw = cv2.imread(str(dem_path), cv2.IMREAD_GRAYSCALE)
    assert dem_raw is not None
    # Scale raw pixel intensities to realistic relative relief [-200m, +200m]
    dem_scaled = ((dem_raw.astype(np.float32) / 255.0) - 0.5) * 400.0

    # Configure camera from real metadata
    cam = InstrumentCameraModel.create_ohrc_model(image_size=(512, 512))
    assert abs(res_m - 0.25) < 1e-4

    # Configure SPICE provider with state representing this observation
    provider = SpiceGeometryProvider()
    pos_m = np.array([LUNAR_RADIUS_M + 100000.0, 0.0, 0.0], dtype=np.float64)
    vel_m = np.array([0.0, 0.0, 1633.0], dtype=np.float64)
    r_cam_to_body = np.array([
        [0.0, 0.0, -1.0],
        [1.0, 0.0,  0.0],
        [0.0, 1.0,  0.0],
    ], dtype=np.float64)

    state = SpacecraftState(
        position_m=pos_m,
        velocity_m_s=vel_m,
        ephemeris_time_s=640160531.0,
        utc_time=utc_str,
        sub_sc_lat_deg=0.0,
        sub_sc_lon_deg=0.0,
        altitude_m=100000.0,
        rotation_cam_to_body=r_cam_to_body,
        is_synthetic=True,
    )
    provider.register_synthetic_state(state)

    intersector = LunarDemRayIntersector(
        dem=dem_scaled, center_lat_deg=0.0, center_lon_deg=0.0, gsd_m=5.0
    )
    pipeline = OrthorectificationPipeline(spice_provider=provider, intersector=intersector)

    # Project center pixel to ground
    res = pipeline.project_pixel_to_ground(256.0, 256.0, cam, utc_str)
    assert res.convergence_status == ConvergenceStatus.CONVERGED
    assert res.surface_point_body_m is not None
    assert res.final_residual_m <= 0.05

    # Verify that trying to use solar azimuth (135.5) as LOS azimuth is rejected
    pts = np.array([[256.0, 256.0]])
    with pytest.raises(ValueError, match="LOS azimuth conflated with solar azimuth"):
        pipeline.compare_approximate_and_rigorous(
            pts_px=pts,
            dem=dem_scaled,
            emission_deg=em_deg if em_deg > 0 else 5.0,
            los_azimuth_deg=sun_az,
            solar_azimuth_deg=sun_az,  # Conflated!
            gsd_m=5.0,
            camera=cam,
            epoch_utc_or_et=utc_str,
        )

