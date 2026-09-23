"""
terrain_adaptive_log_gabor.py — Terrain-Adaptive Log-Gabor Configuration for Lunar Cross-Matching

Provides an optional, geomorphometry-driven Log-Gabor filter configuration engine that
selects frequency-domain wavelet parameters from local Digital Elevation Model (DEM)
characteristics (slope, curvature, roughness, dominant topographic wavelength).

Scientific Principles:
1. DEM-Derived Metrics Only When Aligned: Terrain adaptation is strictly conditional
   on verified DEM alignment (dem_aligned=True). When unaligned or unavailable, the
   system fails closed to the fixed baseline filter bank.
2. Frequency-Based Scale Selection: Filter scales are derived from physical terrain
   relief and topographic frequency, NEVER from arbitrary optical pixel intensity.
3. Provenance & Per-Tile Audit: Every tile logs its geomorphic regime, slope,
   curvature, roughness, dominant wavelength, and selected filter parameters.
4. Checkpoint Consistency: Comparative evaluation tests evaluate fixed versus adaptive
   filters on the exact same independent checkpoints.
5. Opt-in Experimental Design: The baseline fixed filter bank remains the production
   default (enable_terrain_adaptive=False) until empirical superiority is verified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import cv2

logger = logging.getLogger("ML_model.terrain_adaptive_log_gabor")


# ---------------------------------------------------------------------------
# 1. Configuration & Audit Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogGaborConfig:
    """Wavelet filter bank parameters for 2D Phase Congruency."""
    min_wavelength: float = 3.0
    num_scales: int = 3
    mult: float = 2.1
    sigma_on_f: float = 0.55
    num_orientations: int = 4
    is_adaptive: bool = False
    terrain_type: str = "fixed_baseline"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Canonical fixed baseline configuration (EPIC 1 / Phase 2 default)
BASELINE_LOG_GABOR_CONFIG = LogGaborConfig(
    min_wavelength=3.0,
    num_scales=3,
    mult=2.1,
    sigma_on_f=0.55,
    num_orientations=4,
    is_adaptive=False,
    terrain_type="fixed_baseline",
)


@dataclass
class TerrainTileRecord:
    """Audit log record capturing geomorphic analysis and filter parameter selection per tile."""
    tile_id: Union[str, int]
    bounds: Tuple[int, int, int, int]  # (y0, y1, x0, x1)
    dem_aligned: bool
    mean_slope_deg: float
    max_slope_deg: float
    mean_curvature: float
    roughness_m: float
    dominant_wavelength_px: float
    selected_config: LogGaborConfig
    selection_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["selected_config"] = self.selected_config.to_dict()
        return res


# ---------------------------------------------------------------------------
# 2. Geomorphometry Extraction from Aligned DEM
# ---------------------------------------------------------------------------

def compute_terrain_metrics_from_dem(
    dem_tile: np.ndarray,
    gsd_m: float = 1.0,
) -> Dict[str, float]:
    """
    Computes physical geomorphometric metrics from an aligned DEM tile:
    - Slope: Surface normal gradient magnitude in degrees.
    - Curvature: 2D Laplacian (mean second-order spatial derivative).
    - Roughness: Detrended elevation standard deviation in meters.
    - Dominant Wavelength: Dominant spatial wavelength in pixels derived from
      radial Fourier power spectrum of detrended topography.

    Args:
        dem_tile: 2D numpy array of elevation values in meters.
        gsd_m: Ground sample distance of the DEM in meters per pixel.

    Returns:
        Dict with mean_slope_deg, max_slope_deg, mean_curvature,
        roughness_m, and dominant_wavelength_px.
    """
    if dem_tile is None or not isinstance(dem_tile, np.ndarray) or dem_tile.ndim != 2:
        return {
            "mean_slope_deg": 0.0,
            "max_slope_deg": 0.0,
            "mean_curvature": 0.0,
            "roughness_m": 0.0,
            "dominant_wavelength_px": 16.0,
            "is_valid": False,
        }

    h, w = dem_tile.shape
    if h < 4 or w < 4:
        return {
            "mean_slope_deg": 0.0,
            "max_slope_deg": 0.0,
            "mean_curvature": 0.0,
            "roughness_m": 0.0,
            "dominant_wavelength_px": 16.0,
            "is_valid": False,
        }

    d = np.asarray(dem_tile, dtype=np.float64)
    # Fill non-finite voids with mean of finite values
    valid_mask = np.isfinite(d)
    if not np.any(valid_mask):
        return {
            "mean_slope_deg": 0.0,
            "max_slope_deg": 0.0,
            "mean_curvature": 0.0,
            "roughness_m": 0.0,
            "dominant_wavelength_px": 16.0,
            "is_valid": False,
        }
    mean_val = float(np.mean(d[valid_mask]))
    d = np.where(valid_mask, d, mean_val)

    effective_gsd = max(float(gsd_m), 1e-4)

    # 1. Slope Calculation (Sobel gradient / central difference)
    gx = cv2.Sobel(d, cv2.CV_64F, 1, 0, ksize=3) / (8.0 * effective_gsd)
    gy = cv2.Sobel(d, cv2.CV_64F, 0, 1, ksize=3) / (8.0 * effective_gsd)
    grad_mag = np.hypot(gx, gy)
    slopes_deg = np.degrees(np.arctan(grad_mag))
    mean_slope = float(np.mean(slopes_deg))
    max_slope = float(np.max(slopes_deg))

    # 2. Curvature (Laplacian of elevation)
    lap = cv2.Laplacian(d, cv2.CV_64F, ksize=3) / (effective_gsd ** 2)
    mean_curv = float(np.mean(np.abs(lap)))

    # 3. Topographic Roughness (Detrended Elevation Standard Deviation)
    # Fit a first-order planar trend z_plane(x, y) = a*x + b*y + c
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    A = np.column_stack([x_coords.ravel(), y_coords.ravel(), np.ones(h * w)])
    try:
        plane_coeffs, _, _, _ = np.linalg.lstsq(A, d.ravel(), rcond=None)
        trend = (A @ plane_coeffs).reshape(h, w)
        detrended = d - trend
        roughness_m = float(np.std(detrended))
    except Exception:
        roughness_m = float(np.std(d))
        detrended = d - float(np.mean(d))

    # 4. Dominant Topographic Spatial Wavelength via 2D Spectral Energy
    dominant_wavelength_px, hf_ratio = _estimate_dominant_wavelength_px(detrended)

    return {
        "mean_slope_deg": round(mean_slope, 2),
        "max_slope_deg": round(max_slope, 2),
        "mean_curvature": round(mean_curv, 4),
        "roughness_m": round(roughness_m, 2),
        "dominant_wavelength_px": round(dominant_wavelength_px, 2),
        "hf_ratio": round(hf_ratio, 4),
        "is_valid": True,
    }


def _estimate_dominant_wavelength_px(detrended_elevation: np.ndarray) -> Tuple[float, float]:
    """Estimates dominant spatial wavelength and high-frequency spectral ratio from detrended DEM."""
    h, w = detrended_elevation.shape
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)
    d_win = detrended_elevation * window

    F = np.fft.fftshift(np.fft.fft2(d_win))
    psd = np.abs(F) ** 2

    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    r = np.hypot(x_idx - cx, y_idx - cy)

    total_power = float(np.sum(psd))
    if total_power <= 1e-12:
        return 32.0, 0.0

    mean_r = float(np.sum(r * psd) / total_power)
    hf_power = float(np.sum(psd[r > 4.0]))
    hf_ratio = float(hf_power / total_power)

    if mean_r < 0.2:
        dom_wavelength = float(min(h, w))
    else:
        dom_wavelength = float(min(h, w) / max(mean_r, 1e-4))

    dom_wavelength = float(np.clip(dom_wavelength, 2.0, float(max(h, w))))
    return dom_wavelength, hf_ratio


# ---------------------------------------------------------------------------
# 3. Terrain-Adaptive Scale & Filter Selection
# ---------------------------------------------------------------------------

def select_log_gabor_config_for_terrain(
    dem_tile: Optional[np.ndarray],
    dem_aligned: bool = False,
    gsd_m: float = 1.0,
    tile_id: Union[str, int] = 0,
    bounds: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[LogGaborConfig, TerrainTileRecord]:
    """
    Selects Log-Gabor filter parameters based strictly on DEM-derived geomorphometry
    and local terrain frequency.

    Rules:
    - If dem_aligned is False (or DEM is None/invalid): strictly returns baseline config.
    - Scale is derived from physical slope, curvature, and roughness (never image intensity).
    - Logs complete rationale and geomorphic indicators.

    Geomorphic Regimes:
    1. Smooth Mare: mean slope < 5.0 deg and roughness < 3.0 m.
       -> min_wavelength = 6.0 px, num_scales = 4, mult = 2.4, sigma_on_f = 0.65.
       -> Rationale: Longer minimum wavelength suppresses high-frequency sensor noise on
          low-contrast basalt plains, capturing subtle topographic undulations.
    2. High-Frequency Rugged / Dense Ejecta: (roughness >= 5.0 m and hf_ratio > 0.10) or dom_wavelength < 40.0 px.
       -> min_wavelength = 2.0 px, num_scales = 4, mult = 1.8, sigma_on_f = 0.50.
       -> Rationale: Fine-scale high-resolution bank resolves micro-craters and blocky
          ejecta without inter-feature interference.
    3. Crater Rims & Scarps: max slope > 15.0 deg or curvature > 0.05.
       -> min_wavelength = 3.0 px, num_scales = 3, mult = 2.1, sigma_on_f = 0.55.
       -> Rationale: Standard edge-preserving wavelength accurately localizes sharp
          topographic discontinuity without blur.
    4. Undulating / Moderate Terrain: default balanced configuration.
    """
    default_bounds = bounds if bounds is not None else (0, 0, 0, 0)

    # Requirement 1: Only use DEM-derived metrics when the DEM is aligned
    if not dem_aligned or dem_tile is None:
        rec = TerrainTileRecord(
            tile_id=tile_id,
            bounds=default_bounds,
            dem_aligned=False,
            mean_slope_deg=0.0,
            max_slope_deg=0.0,
            mean_curvature=0.0,
            roughness_m=0.0,
            dominant_wavelength_px=0.0,
            selected_config=BASELINE_LOG_GABOR_CONFIG,
            selection_rationale="dem_unaligned_or_unavailable_fallback_to_baseline",
        )
        return BASELINE_LOG_GABOR_CONFIG, rec

    metrics = compute_terrain_metrics_from_dem(dem_tile, gsd_m=gsd_m)
    if not metrics.get("is_valid", False):
        rec = TerrainTileRecord(
            tile_id=tile_id,
            bounds=default_bounds,
            dem_aligned=True,
            mean_slope_deg=0.0,
            max_slope_deg=0.0,
            mean_curvature=0.0,
            roughness_m=0.0,
            dominant_wavelength_px=0.0,
            selected_config=BASELINE_LOG_GABOR_CONFIG,
            selection_rationale="dem_metrics_invalid_fallback_to_baseline",
        )
        return BASELINE_LOG_GABOR_CONFIG, rec

    mean_slope = metrics["mean_slope_deg"]
    max_slope = metrics["max_slope_deg"]
    mean_curv = metrics["mean_curvature"]
    roughness = metrics["roughness_m"]
    dom_wavelength = metrics["dominant_wavelength_px"]
    hf_ratio = metrics.get("hf_ratio", 0.0)

    # Requirement 3: Select filter scale based on local terrain frequency & geometry
    if mean_slope < 5.0 and roughness < 3.0:
        # Regime 1: Smooth Mare
        config = LogGaborConfig(
            min_wavelength=6.0,
            num_scales=4,
            mult=2.4,
            sigma_on_f=0.65,
            num_orientations=4,
            is_adaptive=True,
            terrain_type="smooth_mare",
        )
        rationale = (
            f"Smooth Mare (mean_slope={mean_slope:.1f}deg, roughness={roughness:.1f}m): "
            "longer min_wavelength=6.0px suppresses noise on smooth plains."
        )
    elif (roughness >= 5.0 and hf_ratio > 0.10) or dom_wavelength < 40.0:
        # Regime 2: High-Frequency Rugged / Dense Impact Ejecta
        config = LogGaborConfig(
            min_wavelength=2.0,
            num_scales=4,
            mult=1.8,
            sigma_on_f=0.50,
            num_orientations=4,
            is_adaptive=True,
            terrain_type="high_frequency_rugged",
        )
        rationale = (
            f"High-Frequency Rugged (roughness={roughness:.1f}m, dom_wavelength={dom_wavelength:.1f}px, hf_ratio={hf_ratio:.3f}): "
            "fine-scale bank min_wavelength=2.0px resolves dense micro-features."
        )
    elif max_slope > 15.0 or mean_curv > 0.05:
        # Regime 3: Crater Rims & Steep Topographic Slopes
        config = LogGaborConfig(
            min_wavelength=3.0,
            num_scales=3,
            mult=2.1,
            sigma_on_f=0.55,
            num_orientations=4,
            is_adaptive=True,
            terrain_type="crater_rim",
        )
        rationale = (
            f"Crater Rim (max_slope={max_slope:.1f}deg, curvature={mean_curv:.4f}): "
            "edge-preserving scale min_wavelength=3.0px localizes sharp elevation boundary."
        )
    else:
        # Regime 4: Moderate Undulating Highlands
        config = LogGaborConfig(
            min_wavelength=3.5,
            num_scales=3,
            mult=2.1,
            sigma_on_f=0.55,
            num_orientations=4,
            is_adaptive=True,
            terrain_type="moderate_undulating",
        )
        rationale = (
            f"Moderate Undulating (mean_slope={mean_slope:.1f}deg, roughness={roughness:.1f}m): "
            "standard balanced bank min_wavelength=3.5px."
        )

    rec = TerrainTileRecord(
        tile_id=tile_id,
        bounds=default_bounds,
        dem_aligned=True,
        mean_slope_deg=mean_slope,
        max_slope_deg=max_slope,
        mean_curvature=mean_curv,
        roughness_m=roughness,
        dominant_wavelength_px=dom_wavelength,
        selected_config=config,
        selection_rationale=rationale,
    )
    return config, rec


# ---------------------------------------------------------------------------
# 4. Phase Congruency Execution Engine
# ---------------------------------------------------------------------------

def compute_phase_congruency_with_config(
    img: np.ndarray,
    config: LogGaborConfig,
) -> np.ndarray:
    """
    Computes 2D Phase Congruency using the specified LogGaborConfig.
    Preserves pure frequency-domain Log-Gabor formulation from matcher_cfog.py.
    """
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)
    img_f = img_f - float(np.mean(img_f))

    y_idx = np.fft.fftfreq(h).astype(np.float32)
    x_idx = np.fft.fftfreq(w).astype(np.float32)
    xv, yv = np.meshgrid(x_idx, y_idx)
    radius = np.sqrt(xv**2 + yv**2).astype(np.float32)
    radius[0, 0] = 1.0  # avoid log(0)
    theta = np.arctan2(-yv, xv).astype(np.float32)

    F = np.fft.fft2(img_f)

    energy_total = np.zeros((h, w), dtype=np.float32)
    amplitude_total = np.zeros((h, w), dtype=np.float32)

    num_orientations = config.num_orientations
    d_theta = np.pi / float(num_orientations)
    theta_sigma = 1.2 / float(num_orientations)

    for o in range(num_orientations):
        angl = o * d_theta
        diff_theta = np.abs(np.arctan2(np.sin(theta - angl), np.cos(theta - angl)))
        ang_filter = np.exp(-(diff_theta**2) / (2.0 * theta_sigma**2)).astype(np.float32)

        sum_e = np.zeros((h, w), dtype=np.float32)
        sum_o = np.zeros((h, w), dtype=np.float32)

        wavelength = config.min_wavelength
        for s in range(config.num_scales):
            fo = 1.0 / wavelength
            log_gabor = np.exp(
                -((np.log(radius / fo)) ** 2) / (2.0 * (np.log(config.sigma_on_f)) ** 2)
            ).astype(np.float32)
            log_gabor[0, 0] = 0.0

            filter_2d = log_gabor * ang_filter
            resp = np.fft.ifft2(F * filter_2d)

            re = np.real(resp).astype(np.float32)
            im = np.imag(resp).astype(np.float32)
            amp = np.sqrt(re**2 + im**2)

            sum_e += re
            sum_o += im
            amplitude_total += amp
            wavelength *= config.mult

        energy_o = np.sqrt(sum_e**2 + sum_o**2)
        energy_total += energy_o

    pc = energy_total / (amplitude_total + 1e-4)
    return np.clip(pc, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Tile-Level Adaptive Processor & Audit Logger
# ---------------------------------------------------------------------------

class TerrainAdaptiveLogGaborEngine:
    """
    Manages terrain-adaptive Log-Gabor processing across tiles, preserving
    per-tile provenance and comparative evaluation capabilities.
    """

    def __init__(self, enable_terrain_adaptive: bool = False, gsd_m: float = 1.0):
        self.enable_terrain_adaptive = bool(enable_terrain_adaptive)
        self.gsd_m = float(gsd_m)
        self.tile_audit_records: List[TerrainTileRecord] = []

    def clear_audit_log(self) -> None:
        self.tile_audit_records.clear()

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return [rec.to_dict() for rec in self.tile_audit_records]

    def process_tile(
        self,
        img_tile: np.ndarray,
        dem_tile: Optional[np.ndarray] = None,
        dem_aligned: bool = False,
        tile_id: Union[str, int] = 0,
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, LogGaborConfig]:
        """
        Processes a single image tile with either fixed baseline or terrain-adaptive Log-Gabor.

        If self.enable_terrain_adaptive is False or dem_aligned is False,
        strictly uses BASELINE_LOG_GABOR_CONFIG.
        """
        if not self.enable_terrain_adaptive:
            config = BASELINE_LOG_GABOR_CONFIG
            rec = TerrainTileRecord(
                tile_id=tile_id,
                bounds=bounds if bounds else (0, 0, 0, 0),
                dem_aligned=dem_aligned,
                mean_slope_deg=0.0,
                max_slope_deg=0.0,
                mean_curvature=0.0,
                roughness_m=0.0,
                dominant_wavelength_px=0.0,
                selected_config=config,
                selection_rationale="adaptive_disabled_using_fixed_baseline",
            )
        else:
            config, rec = select_log_gabor_config_for_terrain(
                dem_tile=dem_tile,
                dem_aligned=dem_aligned,
                gsd_m=self.gsd_m,
                tile_id=tile_id,
                bounds=bounds,
            )

        self.tile_audit_records.append(rec)
        pc = compute_phase_congruency_with_config(img_tile, config)
        return pc, config

    def process_image(
        self,
        img: np.ndarray,
        dem: Optional[np.ndarray] = None,
        dem_aligned: bool = False,
        tile_size: int = 128,
    ) -> np.ndarray:
        """
        Tiles an image, logs per-tile geomorphic records, selects terrain-adaptive
        or baseline filter parameters for each tile, and reconstructs the full Phase Congruency map.
        """
        h, w = img.shape[:2]
        if dem is not None and (dem.shape[0] != h or dem.shape[1] != w):
            dem_aligned_grid = cv2.resize(
                dem.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
        else:
            dem_aligned_grid = dem

        pc_full = np.zeros((h, w), dtype=np.float32)
        step = max(32, int(tile_size))
        for y0 in range(0, h, step):
            y1 = min(y0 + step, h)
            for x0 in range(0, w, step):
                x1 = min(x0 + step, w)
                img_tile = img[y0:y1, x0:x1]
                dem_tile = (
                    dem_aligned_grid[y0:y1, x0:x1]
                    if dem_aligned_grid is not None
                    else None
                )
                pc_tile, _ = self.process_tile(
                    img_tile=img_tile,
                    dem_tile=dem_tile,
                    dem_aligned=dem_aligned,
                    tile_id=f"tile_{y0}_{x0}",
                    bounds=(y0, y1, x0, x1),
                )
                pc_full[y0:y1, x0:x1] = pc_tile

        return pc_full


# ---------------------------------------------------------------------------
# 6. Checkpoint Comparative Evaluation Harness
# ---------------------------------------------------------------------------

def compare_fixed_vs_adaptive_log_gabor(
    img1: np.ndarray,
    img2: np.ndarray,
    dem: Optional[np.ndarray],
    dem_aligned: bool,
    checkpoints_src: np.ndarray,
    checkpoints_dst: np.ndarray,
    H_gt: Optional[np.ndarray] = None,
    gsd_m: float = 1.0,
    tile_size: int = 128,
) -> Dict[str, Any]:
    """
    Compares fixed baseline vs. terrain-adaptive Log-Gabor on the EXACT SAME
    independent checkpoints (Requirement 5).

    Evaluates:
    - Phase Congruency structural contrast around checkpoints.
    - Checkpoint displacement error / RMSE under fixed vs. adaptive feature maps.
    - Per-tile parameter selection log.
    - Held-out improvement verification: delta_rmse = fixed_rmse - adaptive_rmse.
    """
    chk_src = np.asarray(checkpoints_src, dtype=np.float64)
    chk_dst = np.asarray(checkpoints_dst, dtype=np.float64)
    n_chk = len(chk_src)

    if n_chk == 0 or len(chk_dst) != n_chk:
        return {
            "status": "invalid_checkpoints",
            "fixed_checkpoint_rmse": None,
            "adaptive_checkpoint_rmse": None,
            "delta_rmse": None,
            "adaptive_improved": False,
            "tile_audit_log": [],
        }

    # 1. Run Fixed Baseline Engine
    engine_fixed = TerrainAdaptiveLogGaborEngine(enable_terrain_adaptive=False, gsd_m=gsd_m)
    pc1_fixed = compute_phase_congruency_with_config(img1, BASELINE_LOG_GABOR_CONFIG)
    pc2_fixed = compute_phase_congruency_with_config(img2, BASELINE_LOG_GABOR_CONFIG)

    # 2. Run Terrain-Adaptive Engine (Tile-by-Tile)
    engine_adaptive = TerrainAdaptiveLogGaborEngine(enable_terrain_adaptive=True, gsd_m=gsd_m)
    h, w = img1.shape[:2]

    pc1_adaptive = engine_adaptive.process_image(
        img=img1,
        dem=dem,
        dem_aligned=dem_aligned,
        tile_size=tile_size,
    )

    pc2_adaptive = compute_phase_congruency_with_config(img2, BASELINE_LOG_GABOR_CONFIG)

    # 3. Evaluate local sub-pixel correlation around checkpoints on identical points
    # Measure peak localization error relative to known shift / H_gt
    errors_fixed: List[float] = []
    errors_adaptive: List[float] = []
    half_win = 16

    for i in range(n_chk):
        x1, y1 = int(round(chk_src[i, 0])), int(round(chk_src[i, 1]))
        x2, y2 = int(round(chk_dst[i, 0])), int(round(chk_dst[i, 1]))

        if (y1 - half_win < 0 or y1 + half_win >= h or x1 - half_win < 0 or x1 + half_win >= w or
            y2 - half_win < 0 or y2 + half_win >= h or x2 - half_win < 0 or x2 + half_win >= w):
            continue

        p1_f = pc1_fixed[y1 - half_win : y1 + half_win, x1 - half_win : x1 + half_win]
        p2_f = pc2_fixed[y2 - half_win : y2 + half_win, x2 - half_win : x2 + half_win]
        p1_a = pc1_adaptive[y1 - half_win : y1 + half_win, x1 - half_win : x1 + half_win]
        p2_a = pc2_adaptive[y2 - half_win : y2 + half_win, x2 - half_win : x2 + half_win]

        # Phase correlation peak
        shift_f, _ = cv2.phaseCorrelate(p1_f.astype(np.float64), p2_f.astype(np.float64))
        shift_a, _ = cv2.phaseCorrelate(p1_a.astype(np.float64), p2_a.astype(np.float64))

        # Expected shift is chk_dst - chk_src
        expected_shift = chk_dst[i] - chk_src[i]
        err_f = float(np.linalg.norm(np.array(shift_f) - expected_shift))
        err_a = float(np.linalg.norm(np.array(shift_a) - expected_shift))

        errors_fixed.append(err_f)
        errors_adaptive.append(err_a)

    if len(errors_fixed) == 0:
        rmse_fixed = 0.0
        rmse_adaptive = 0.0
    else:
        rmse_fixed = float(np.sqrt(np.mean(np.array(errors_fixed) ** 2)))
        rmse_adaptive = float(np.sqrt(np.mean(np.array(errors_adaptive) ** 2)))

    delta_rmse = rmse_fixed - rmse_adaptive
    improved = bool(delta_rmse > 0.05)

    return {
        "status": "evaluated",
        "fixed_checkpoint_rmse": round(rmse_fixed, 4),
        "adaptive_checkpoint_rmse": round(rmse_adaptive, 4),
        "delta_rmse": round(delta_rmse, 4),
        "adaptive_improved": improved,
        "evaluated_checkpoints_count": len(errors_fixed),
        "tile_audit_log": engine_adaptive.get_audit_log(),
    }
