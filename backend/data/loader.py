"""
loader.py — Loads user_triplets.json, processed_triplets, and match files into memory at startup.

The backend reads manifest data and ML match output, performs pixel-to-geo
conversion once at load time using shared TripletBounds (via geo.py), detects
DEM (Digital Elevation Model) availability, and caches everything in memory.
It NEVER writes to the data directory or re-runs any ML computation.
"""

import json
import logging
import os
import sys
from pathlib import Path

from geo import (
    pixel_to_latlon_from_bounds_batch,
)

logger = logging.getLogger("backend.data.loader")

# ---------------------------------------------------------------------------
# Data directories — override via env vars
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(REPO_ROOT / "ML_model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ML_model"))

try:
    from metrics import compute_canonical_metrics
except ImportError:
    compute_canonical_metrics = None


def _first_existing(*paths: str | Path) -> str:
    """Return the first existing path, or the first candidate if none exist."""
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return str(candidate)
    return str(Path(paths[0]))


DATA_DIR: str = os.environ.get(
    "DATA_DIR",
    _first_existing(
        REPO_ROOT / "data_preprocessing_pipeline",
        REPO_ROOT / "processed_user",
    ),
)

PROCESSED_TRIPLETS_DIR: str = os.environ.get(
    "PROCESSED_TRIPLETS_DIR",
    _first_existing(
        REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets",
        REPO_ROOT / "processed_triplets",
    ),
)

# ML team's output directory — searched as a secondary source for match files
ML_OUTPUT_DIR: str = os.environ.get(
    "ML_OUTPUT_DIR",
    str(REPO_ROOT / "ML_model"),
)

# Image size used by the ML pipeline (matcher.py resizes to this)
IMAGE_SIZE: int = 512


# ---------------------------------------------------------------------------
# In-memory stores (populated by load_all, read by routers)
# ---------------------------------------------------------------------------

_triplets: dict[str, dict] = {}       # keyed by triplet id
_triplet_list: list[dict] = []        # ordered list for GET /triplets
_matches: dict[str, dict] = {}        # keyed by triplet id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_ml_matches(
    raw_matches: list[dict],
    bounds: dict,
    homography: list[list[float]] | None = None,
) -> tuple[list[dict], list[list[float]] | None]:
    """
    Transform raw ML match output into the backend's MatchPoint shape.

    Input format (from ML team):
        [{image1_x, image1_y, image2_x, image2_y, confidence}, ...]

    Output:
        - List of MatchPoint-shaped dicts (with ohrc_px, tmc_px,
          ohrc_latlon, tmc_latlon, confidence)
        - Serialized 3×3 homography matrix loaded from transform.json
    """
    if not raw_matches:
        return [], None

    # Collect pixel pairs for batch geo conversion
    ohrc_pixels = [
        (float(m.get("image1_x", m.get("source_x"))), float(m.get("image1_y", m.get("source_y"))))
        for m in raw_matches
    ]
    tmc_pixels = [
        (float(m.get("image2_x", m.get("target_x"))), float(m.get("image2_y", m.get("target_y"))))
        for m in raw_matches
    ]

    # Convert pixels to lat/lon using the shared affine transform from bounds
    ohrc_latlons = pixel_to_latlon_from_bounds_batch(
        ohrc_pixels, bounds, IMAGE_SIZE, IMAGE_SIZE,
    )
    tmc_latlons = pixel_to_latlon_from_bounds_batch(
        tmc_pixels, bounds, IMAGE_SIZE, IMAGE_SIZE,
    )

    # Build the MatchPoint-shaped dicts
    points = []
    for i, m in enumerate(raw_matches):
        points.append({
            "ohrc_px": (float(m.get("image1_x", m.get("source_x"))), float(m.get("image1_y", m.get("source_y")))),
            "tmc_px": (float(m.get("image2_x", m.get("target_x"))), float(m.get("image2_y", m.get("target_y")))),
            "ohrc_latlon": ohrc_latlons[i],
            "tmc_latlon": tmc_latlons[i],
            "confidence": float(m.get("confidence", 1.0)),
        })

    return points, homography


def _footprint_gsd_m(bounds: dict, image_px: int = 512) -> float | None:
    """Planar meters-per-pixel from shared-bounds footprint width.

    Used ONLY for the honest planar absolute-RMSE fallback (no DEM): the
    512 common grid spans [west_lon, east_lon] on a lunar sphere.
    Returns None when bounds are missing/deenerate.
    """
    try:
        import math

        w_lon = float(bounds["west_lon"])
        e_lon = float(bounds["east_lon"])
        mid_lat = float(bounds["south_lat"] + bounds["north_lat"]) / 2.0
        width_m = abs(e_lon - w_lon) * math.pi / 180.0 * 1737400.0 * abs(math.cos(math.radians(mid_lat)))
        gsd = width_m / max(int(image_px), 1)
        return float(gsd) if gsd > 0 and gsd == gsd else None
    except Exception:
        return None


def _lro_overlap_photometrics(reg_id: str) -> dict:
    """SSIM/PSNR/NMI over the valid overlap of committed LRO rasters.

    Compares registration_output/lro_nac/{id}/registered_source.png (warped
    OHRC) against the real-CDR reference tile with the shared overlap-mask
    math from ML_model/metrics.py. Returns Nones (never raises) when assets
    are absent — the UI then shows the documented reason, not a number.
    """
    out: dict = {"ssim": None, "psnr": None, "nmi": None, "overlap_frac": None}
    try:
        import cv2  # noqa: F401  (already a backend dependency)

        from metrics import (
            calculate_overlap_mask,
            calculate_ssim_over_overlap,
            calculate_psnr_over_overlap,
            calculate_normalized_mutual_information,
        )

        warped_path = os.path.join(REPO_ROOT, "registration_output", "lro_nac", reg_id, "registered_source.png")
        ref_path = os.path.join(
            REPO_ROOT, "data_preprocessing_pipeline", "lro_nac_real", reg_id, "lro_nac_reference_512.png"
        )
        if not (os.path.isfile(warped_path) and os.path.isfile(ref_path)):
            return out
        warped = cv2.imread(warped_path, cv2.IMREAD_UNCHANGED)
        ref = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
        if warped is None or ref is None:
            return out
        mask = calculate_overlap_mask(warped, ref)
        out["overlap_frac"] = round(float(mask.mean()), 4) if mask.size else 0.0
        if int(mask.sum()) < 16:
            return out
        out["ssim"] = calculate_ssim_over_overlap(warped, ref, mask=mask)
        out["psnr"] = calculate_psnr_over_overlap(warped, ref, mask=mask)
        out["nmi"] = calculate_normalized_mutual_information(warped, ref, mask=mask)
    except Exception as exc:
        logger.warning("LRO photometric metrics unavailable for %s: %s", reg_id, exc)
    return out


_BENCHMARK_SUMMARY: dict | None = None
_BENCHMARK_LOADED = False


def _benchmark_row(triplet_id: str) -> dict | None:
    """Committed pipeline benchmark row for this region, if published."""
    global _BENCHMARK_SUMMARY, _BENCHMARK_LOADED
    if not _BENCHMARK_LOADED:
        _BENCHMARK_LOADED = True
        try:
            with open(
                os.path.join(
                    REPO_ROOT, "benchmarks", "registration_benchmark_output",
                    "registration_benchmark_summary.json",
                ),
                "r",
            ) as f:
                _BENCHMARK_SUMMARY = json.load(f)
        except Exception as exc:
            logger.warning("Benchmark summary unavailable: %s", exc)
            _BENCHMARK_SUMMARY = None
    try:
        for row in (_BENCHMARK_SUMMARY or {}).get("regions", []):
            if row.get("region_id") == triplet_id:
                return row
    except Exception:
        pass
    return None


def _metrics_from_benchmark_summary(triplet_id: str, bounds: dict, n_served: int) -> dict | None:
    """Metrics for regions whose pipeline numbers are published.

    Counts/fit/coverage/uniformity/validation come straight from the committed
    summary. Planar absolute RMSE = fit × footprint GSD (no DEM available to
    this loader — provenance-labeled, never shown as DEM-corrected).
    Composite = feature-only formula over the published numbers (no rasters).
    """
    row = _benchmark_row(triplet_id)
    if not row or row.get("status") != "success":
        return None
    fit = row.get("fit_rmse_px")
    n_inl = int(row.get("inlier_count", 0) or 0)
    planar_abs_m = None
    gsd_m = _footprint_gsd_m(bounds) if bounds else None
    if fit is not None and gsd_m is not None:
        planar_abs_m = round(float(fit) * gsd_m, 4)
    composite = None
    try:
        from metrics import calculate_composite_quality_score

        composite = calculate_composite_quality_score(
            inlier_ratio=float(row.get("inlier_ratio", 0.0) or 0.0),
            fit_rmse_px=float(fit) if fit is not None else None,
            spatial_uniformity=float(row.get("spatial_uniformity", 0.0) or 0.0),
        ).get("composite_quality_score")
    except Exception as exc:
        logger.warning("Composite score unavailable for %s: %s", triplet_id, exc)
    notes: dict = {}
    if row.get("validation_rmse_px") is None:
        notes["validation_rmse_px"] = f"held-out needs ≥8 inliers (have {n_inl})"
    if planar_abs_m is None:
        notes["absolute_rmse_m"] = "no footprint GSD available"
    for key in ("ssim", "psnr", "nmi"):
        notes[key] = "no registered overlap rasters on disk for this region"
    return {
        "num_inliers": n_inl,
        "num_raw_matches": int(row.get("match_count", n_served) or n_served),
        "inlier_ratio": float(row.get("inlier_ratio", 0.0) or 0.0),
        "rmse_px": float(fit) if fit is not None else 0.0,
        "fit_rmse_px": fit,
        "absolute_rmse_m": planar_abs_m,
        "absolute_rmse_m_provenance": "planar_footprint_gsd_no_dem" if planar_abs_m is not None else None,
        "validation_rmse_px": row.get("validation_rmse_px"),
        "validation_status": row.get("validation_status"),
        "sub_pixel_accurate": bool(fit is not None and row.get("validation_rmse_px") is not None
                                   and fit < 1.0 and row["validation_rmse_px"] < 1.0),
        "source_coverage_ratio": float(row.get("spatial_coverage", 0.0) or 0.0),
        "destination_coverage_ratio": float(row.get("spatial_coverage", 0.0) or 0.0),
        "combined_coverage_score": float(row.get("spatial_coverage", 0.0) or 0.0),
        "uniformity_score": float(row.get("spatial_uniformity", 0.0) or 0.0),
        "composite_quality_score": composite,
        "method": "CFOG + Phase Congruency (benchmark summary)",
        "metric_notes": notes or None,
    }


def _load_match_file(filepath: str) -> list[dict]:
    """
    Load a match file, handling both the ML team's bare-list format
    and the old wrapper format for backwards compatibility.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # ML team format: bare JSON array
    if isinstance(data, list):
        return data

    # Legacy wrapper format: {triplet_id, homography, matches: [...]}
    if isinstance(data, dict) and "matches" in data:
        return data["matches"]

    return []


def _normalize_triplet(data: dict, default_id: str | None = None, region_dir: str | None = None) -> dict:
    """Ensure triplet has an id, bounds, sensors list, and DEM availability."""
    triplet = dict(data)
    if "id" not in triplet:
        triplet["id"] = default_id or triplet.get("region_id") or triplet.get("ohrc_product_id") or "triplet_01"

    # Ignore placeholder/demo triplets that don't contain a real shared bounding box.
    if not triplet.get("bounds"):
        if triplet.get("region_id") is None and triplet.get("ohrc_product_id") is None:
            triplet["bounds"] = {
                "west_lon": 0.0,
                "east_lon": 0.0,
                "south_lat": 0.0,
                "north_lat": 0.0,
            }

    # If a triplet has no usable bounds, treat it as non-production/demo data.
    if not isinstance(triplet.get("bounds"), dict):
        triplet["bounds"] = None

    # Build sensors list if not present
    if "sensors" not in triplet:
        sensors = []
        if "ohrc_product_id" in triplet:
            sensors.append({
                "sensor": "ohrc",
                "gsd_m": triplet.get("ohrc_gsd_m", 0.25),
                "sun_elevation_deg": triplet.get("ohrc_sun_elevation_deg"),
                "sun_azimuth_deg": triplet.get("ohrc_sun_azimuth_deg"),
                "incidence_angle_deg": triplet.get("ohrc_incidence_deg"),
            })
        if "tmc2_product_id" in triplet or "tmc_product_id" in triplet:
            sensors.append({
                "sensor": "tmc",
                "gsd_m": triplet.get("tmc2_gsd_m", 5.0),
                "sun_elevation_deg": triplet.get("tmc2_sun_elevation_deg"),
                "sun_azimuth_deg": triplet.get("tmc2_sun_azimuth_deg"),
                "incidence_angle_deg": triplet.get("tmc2_incidence_deg"),
            })
        if "iirs_product_id" in triplet:
            sensors.append({
                "sensor": "iirs",
                "tile_id": "iirs_overlay.png",
                "gsd_m": triplet.get("iirs_gsd_m", 80.0),
                "sun_elevation_deg": triplet.get("iirs_sun_elevation_deg"),
                "sun_azimuth_deg": triplet.get("iirs_sun_azimuth_deg"),
                "incidence_angle_deg": triplet.get("iirs_incidence_deg"),
            })
        triplet["sensors"] = sensors

    # Check DEM (Digital Elevation Model) presence
    has_dem = False
    if region_dir and os.path.isdir(region_dir):
        if os.path.isfile(os.path.join(region_dir, "dem_512.png")) or os.path.isfile(os.path.join(region_dir, "dem_overlay.png")):
            has_dem = True

    # Fallback check in central images/dem
    dem_img_path = os.path.join(DATA_DIR, "images", "dem", "dem_512.png")
    if os.path.isfile(dem_img_path) or triplet.get("dem_available"):
        has_dem = True

    triplet["dem_available"] = has_dem
    if has_dem:
        reg_dem_filename = f"{triplet['id']}_dem_512.png"
        if os.path.isfile(os.path.join(DATA_DIR, "images", "dem", reg_dem_filename)):
            triplet["dem_url"] = f"/images/dem/{reg_dem_filename}"
        else:
            triplet["dem_url"] = "/images/dem/dem_512.png"
    else:
        triplet["dem_url"] = None

    if has_dem and not any(s.get("sensor") == "dem" for s in triplet.get("sensors", [])):
        triplet["sensors"].append({
            "sensor": "dem",
            "gsd_m": 5.0,
            "sun_elevation_deg": None,
            "sun_azimuth_deg": None,
            "incidence_angle_deg": None,
        })

    # Check LRO NAC (NASA Lunar Reconnaissance Orbiter Narrow Angle Camera) reference availability
    has_lro = False
    lro_dir = os.path.join(REPO_ROOT, "data_preprocessing_pipeline", "lro_nac_pairs", triplet["id"])
    if os.path.isdir(lro_dir) or triplet.get("lro_nac_available"):
        has_lro = True
        lro_manifest_file = os.path.join(lro_dir, "manifest.json")
        if os.path.isfile(lro_manifest_file):
            try:
                with open(lro_manifest_file, "r") as mf:
                    lro_mf = json.load(mf)
                triplet.setdefault("lro_nac_product_id", lro_mf.get("lro_nac_product_id"))
                triplet.setdefault("lro_nac_gsd_m", lro_mf.get("lro_nac_native_gsd_m", 0.914))
                triplet.setdefault("lro_nac_sun_azimuth_deg", lro_mf.get("lro_nac_sun_azimuth_deg"))
            except Exception:
                pass

    triplet["lro_nac_available"] = has_lro
    if has_lro and not any(s.get("sensor") == "lro_nac" for s in triplet.get("sensors", [])):
        triplet["sensors"].append({
            "sensor": "lro_nac",
            "gsd_m": triplet.get("lro_nac_gsd_m", 0.914),
            "sun_elevation_deg": None,
            "sun_azimuth_deg": triplet.get("lro_nac_sun_azimuth_deg"),
            "incidence_angle_deg": None,
        })

    return triplet


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_all() -> None:
    """
    Read user_triplets.json, processed_triplets, manifest.json, or batch triplet
    directories into memory. Called once at startup and again on GET /refresh.

    For each triplet, match points are enriched with lat/lon coordinates
    computed from the shared TripletBounds (see geo.py).
    """
    global _triplets, _triplet_list, _matches

    triplets_raw = []

    # 1. Check processed_triplets directory for the 6 real validated regions
    if os.path.isdir(PROCESSED_TRIPLETS_DIR):
        for entry in sorted(os.listdir(PROCESSED_TRIPLETS_DIR)):
            sub_dir = os.path.join(PROCESSED_TRIPLETS_DIR, entry)
            if os.path.isdir(sub_dir):
                sub_manifest = os.path.join(sub_dir, "manifest.json")
                if os.path.isfile(sub_manifest):
                    with open(sub_manifest, "r") as f:
                        sub_data = json.load(f)
                    if isinstance(sub_data, dict):
                        triplets_raw.append(_normalize_triplet(sub_data, default_id=entry, region_dir=sub_dir))

    # 2. Check user_triplets.json in DATA_DIR
    manifest_path = os.path.join(DATA_DIR, "user_triplets.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if isinstance(manifest, dict):
            for t in manifest.get("triplets", []):
                triplets_raw.append(_normalize_triplet(t))
        elif isinstance(manifest, list):
            for t in manifest:
                triplets_raw.append(_normalize_triplet(t))

    # 3. Check standalone manifest.json in DATA_DIR
    standalone_manifest = os.path.join(DATA_DIR, "manifest.json")
    if os.path.isfile(standalone_manifest):
        with open(standalone_manifest, "r") as f:
            m_data = json.load(f)
        if isinstance(m_data, dict):
            triplets_raw.append(_normalize_triplet(m_data))

    # 4. Check subdirectories in DATA_DIR for manifest.json
    if os.path.isdir(DATA_DIR):
        for entry in os.listdir(DATA_DIR):
            sub_dir = os.path.join(DATA_DIR, entry)
            if os.path.isdir(sub_dir) and sub_dir != PROCESSED_TRIPLETS_DIR:
                sub_manifest = os.path.join(sub_dir, "manifest.json")
                if os.path.isfile(sub_manifest):
                    with open(sub_manifest, "r") as f:
                        sub_data = json.load(f)
                    if isinstance(sub_data, dict):
                        triplets_raw.append(_normalize_triplet(sub_data, default_id=entry, region_dir=sub_dir))

    # Build lookup dict and ordered list (deduplicating by id), only keeping
    # production triplets that carry a valid shared bounding box.
    _triplets = {}
    _triplet_list = []
    for t in triplets_raw:
        tid = t.get("id")
        bounds = t.get("bounds")
        if not tid or not isinstance(bounds, dict):
            continue
        if tid not in _triplets:
            _triplets[tid] = t
            _triplet_list.append(t)

    # -------------------------------------------------------------------
    # Load match and transform files from two sources:
    #   1. processed_user/matches/ — one file per triplet
    #   2. ML_model/matches.json / transform.json (mapped to region_001)
    # -------------------------------------------------------------------
    _matches = {}
    _transforms = {}

    # Source 1: processed_user/matches/ — one file per triplet
    matches_dir = os.path.join(DATA_DIR, "matches")
    if os.path.isdir(matches_dir):
        for filename in os.listdir(matches_dir):
            if filename.endswith("_matches.json"):
                filepath = os.path.join(matches_dir, filename)
                triplet_id = filename.replace("_matches.json", "")
                raw = _load_match_file(filepath)
                _matches[triplet_id] = raw
                tx_file = os.path.join(matches_dir, f"{triplet_id}_transform.json")
                if os.path.isfile(tx_file):
                    try:
                        with open(tx_file, "r") as f:
                            _transforms[triplet_id] = json.load(f).get("matrix")
                    except Exception:
                        pass

    # Source 2 (fallback only): ML_model/matches.json — legacy seed mapped to
    # region_001. Must NEVER overwrite the fresh DATA_DIR seed above: a stale
    # 7-entry legacy file once shadowed the current 6-entry seed and broke the
    # num_inliers == num_matches contract. Fill only missing ids.
    ml_matches_path = os.path.join(ML_OUTPUT_DIR, "matches.json")
    if os.path.isfile(ml_matches_path):
        raw = _load_match_file(ml_matches_path)
        if raw and "region_001" not in _matches:
            _matches["region_001"] = raw
    ml_transform_path = os.path.join(ML_OUTPUT_DIR, "transform.json")
    if os.path.isfile(ml_transform_path):
        try:
            with open(ml_transform_path, "r") as f:
                _transforms["region_001"] = json.load(f).get("matrix")
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Enrich all match data with lat/lon + loaded homography
    # -------------------------------------------------------------------
    enriched: dict[str, dict] = {}
    for triplet_id, raw_points in _matches.items():
        triplet = _triplets.get(triplet_id)
        if triplet is None:
            continue

        bounds = triplet.get("bounds")
        if bounds is None:
            enriched[triplet_id] = {
                "triplet_id": triplet_id,
                "matches": [],
                "homography": None,
            }
            continue

        loaded_homography = _transforms.get(triplet_id)
        if loaded_homography is None:
            region_dir = triplet.get("region_dir")
            if region_dir:
                candidate_tx = os.path.join(region_dir, "transform.json")
                if os.path.isfile(candidate_tx):
                    try:
                        with open(candidate_tx, "r") as f:
                            loaded_homography = json.load(f).get("matrix")
                    except Exception:
                        pass

        points, homography = _parse_ml_matches(raw_points, bounds, homography=loaded_homography)

        # Primary metrics source: the pipeline's own committed benchmark
        # summary (no refit, no contradictions with published tables).
        # Derived fields (planar absolute, feature-only composite) are computed
        # from those numbers with documented provenance; anything unavailable
        # stays null with a reason in metric_notes.
        metrics_data = _metrics_from_benchmark_summary(triplet_id, bounds, len(raw_points))

        enriched[triplet_id] = {
            "triplet_id": triplet_id,
            "matches": points,
            "homography": homography,
            "metrics": metrics_data,
        }

    # Load LRO NAC matches for regions with registration output
    lro_reg_dir = os.path.join(REPO_ROOT, "registration_output", "lro_nac")
    if os.path.isdir(lro_reg_dir):
        for reg_id in sorted(os.listdir(lro_reg_dir)):
            reg_path = os.path.join(lro_reg_dir, reg_id)
            if not os.path.isdir(reg_path):
                continue
            metrics_path = os.path.join(reg_path, "metrics.json")
            homog_path = os.path.join(reg_path, "ohrc_to_nac_homography.json")
            if os.path.isfile(metrics_path):
                try:
                    with open(metrics_path, "r") as f:
                        m_json = json.load(f)
                    homog_mat = None
                    if os.path.isfile(homog_path):
                        with open(homog_path, "r") as f:
                            h_json = json.load(f)
                            homog_mat = h_json.get("homography_matrix") or h_json.get("matrix")

                    debug_pts = m_json.get("lk_refinement", {}).get("debug_points", [])
                    raw_lro_matches = []
                    for pt in debug_pts:
                        if pt.get("passed", True):
                            raw_lro_matches.append({
                                "image1_x": pt["src_pt"][0],
                                "image1_y": pt["src_pt"][1],
                                "image2_x": pt["dst_pt"][0],
                                "image2_y": pt["dst_pt"][1],
                                "confidence": max(0.1, 1.0 - float(pt.get("fb_err", 0.0))),
                            })

                    bounds = (_triplets.get(reg_id) or {}).get("bounds")
                    if bounds:
                        pts, derived_h = _parse_ml_matches(raw_lro_matches, bounds) if raw_lro_matches else ([], None)
                        # Photometric overlap metrics from the committed rasters
                        # (warped OHRC vs real-CDR reference). Missing assets →
                        # honest nulls, never synthesized.
                        photo = _lro_overlap_photometrics(reg_id)
                        n_lro = int(m_json.get("inlier_count", len(raw_lro_matches)))
                        lro_notes: dict = {}
                        if not raw_lro_matches and n_lro > 0:
                            lro_notes["matches"] = (
                                f"pipeline run reported {n_lro} inliers but persisted no "
                                "correspondence points; dots unavailable"
                            )
                        if m_json.get("validation_rmse_px") is None:
                            lro_notes["validation_rmse_px"] = f"held-out needs ≥8 inliers (have {n_lro})"
                        if photo["ssim"] is None:
                            lro_notes["ssim"] = lro_notes["psnr"] = lro_notes["nmi"] = \
                                "overlap rasters unavailable for this region"
                        metrics_data = {
                            "num_inliers": n_lro,
                            "num_raw_matches": m_json.get("match_count", len(raw_lro_matches)),
                            "inlier_ratio": m_json.get("inlier_ratio", 1.0),
                            "rmse_px": m_json.get("fit_rmse_px") or 0.0,
                            "fit_rmse_px": m_json.get("fit_rmse_px"),
                            "absolute_rmse_m": m_json.get("absolute_rmse_m"),
                            "absolute_rmse_m_provenance": "pipeline_metrics_json" if m_json.get("absolute_rmse_m") is not None else None,
                            "validation_rmse_px": m_json.get("validation_rmse_px"),
                            "validation_status": m_json.get("validation_status", "evaluated"),
                            "mean_reprojection_error_px": m_json.get("mean_reprojection_error_px", 0.0),
                            "median_reprojection_error_px": m_json.get("median_reprojection_error_px", 0.0),
                            "max_reprojection_error_px": m_json.get("max_reprojection_error_px", 0.0),
                            "sub_pixel_accurate": (m_json.get("fit_rmse_px") or 1.0) < 0.5,
                            "fraction_below_1px": m_json.get("fraction_below_1px", 1.0),
                            "source_coverage_ratio": m_json.get("spatial_coverage", 1.0),
                            "destination_coverage_ratio": m_json.get("spatial_coverage", 1.0),
                            "combined_coverage_score": m_json.get("spatial_coverage", 1.0),
                            "uniformity_score": m_json.get("spatial_uniformity", 0.9),
                            "ssim": photo["ssim"],
                            "psnr": photo["psnr"],
                            "nmi": photo["nmi"],
                            "composite_quality_score": m_json.get("composite_quality_score"),
                            "method": "OHRC-to-LRO-NAC Phase Correlation / LK",
                            "metric_notes": lro_notes or None,
                        }
                        enriched[f"{reg_id}_lro_nac"] = {
                            "triplet_id": f"{reg_id}_lro_nac",
                            "matches": pts,
                            "homography": homog_mat or derived_h,
                            "metrics": metrics_data,
                        }
                except Exception:
                    pass

    _matches = enriched

    logger.info(
        "Loaded %d triplet(s) and %d match file(s)",
        len(_triplets),
        len(_matches),
    )


# ---------------------------------------------------------------------------
# Accessors (used by routers — never modify data)
# ---------------------------------------------------------------------------

def get_triplets() -> list[dict]:
    """Return the full ordered list of triplet dicts."""
    return _triplet_list


def get_triplet(triplet_id: str) -> dict | None:
    """Return a single triplet dict by ID, or None if not found."""
    return _triplets.get(triplet_id)


def get_matches(triplet_id: str) -> dict | None:
    """Return enriched match data for a triplet, or None if not available."""
    return _matches.get(triplet_id)


def triplet_count() -> int:
    """Return the number of loaded triplets (for health check)."""
    return len(_triplets)
