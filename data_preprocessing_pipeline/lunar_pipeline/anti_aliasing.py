"""
data_preprocessing_pipeline/lunar_pipeline/anti_aliasing.py — Sensor-Aware Anti-Aliasing Module

Provides optional sensor-aware anti-aliasing and prefiltering for lunar image cross-matching
(OHRC, TMC-2, IIRS, LRO NAC/WAC) when downsampling across disparate Ground Sample Distances.

Key Architectural Guarantees:
1. `cv2.INTER_AREA` is preserved as the reproducible default baseline.
2. Optional prefilter kernels tailored by sensor optical / sampling parameters.
3. Optical Transfer Function (OTF) parameters are configurable and provenance-tracked.
4. STRICT COMPLIANCE (Rule 13): Never claim OTF correction when no sensor OTF is provided.
   When synthetic or uncalibrated OTFs are used, `otf_correction_claimed` is strictly False.
5. Three-way comparative evaluation suite:
   - Baseline INTER_AREA
   - Gaussian prefilter + INTER_AREA
   - Configured OTF filter
6. Quantitative evaluation of:
   - Edge preservation (Tenengrad / gradient energy ratio)
   - Aliasing (high-frequency spectral folding energy)
   - Sub-pixel registration error (translation RMSE under known displacement)
7. Synthetic frequency patterns (Siemens star, Zone plate, chirp grating).
8. Physical Disclaimer: Actual sensor OTF values (e.g. from ISRO SAC pre-flight calibration reports)
   are required for physical interpretation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("lunar_pipeline.anti_aliasing")

PHYSICAL_OTF_DISCLAIMER = (
    "ACTUAL SENSOR OTF VALUES REQUIRED: In-flight or pre-flight laboratory calibrated "
    "sensor Optical Transfer Function (OTF/MTF) measurements (e.g. from ISRO SAC or LROC "
    "calibration teams) are strictly required for physically meaningful OTF correction. "
    "Heuristic or synthetic OTF models serve exclusively for algorithmic demonstration and "
    "must never be claimed as physical ground-truth sensor restoration."
)


class AntiAliasingMethod(str, Enum):
    """Supported anti-aliasing methods for image downsampling."""
    INTER_AREA = "INTER_AREA"  # Default reproducible baseline
    GAUSSIAN_PREFILTER = "GAUSSIAN_PREFILTER"  # Nyquist-matched Gaussian low-pass + INTER_AREA
    CONFIGURED_OTF = "CONFIGURED_OTF"  # Frequency-domain OTF shaping / filtering


@dataclass
class SensorOTFConfig:
    """Configurable and provenance-tracked Optical Transfer Function (OTF) parameters."""
    sensor: str = "OHRC"  # "OHRC", "TMC-2", "IIRS", "LRO_NAC", "LRO_WAC", "CUSTOM"
    cutoff_freq: float = 0.45  # Normalized spatial frequency cutoff (0 < f_c <= 0.5 cycles/pixel)
    aperture_shape: str = "gaussian"  # "gaussian", "circular_diffraction", "lorentzian"
    wiener_damping: float = 0.05  # Regularization parameter (epsilon)
    has_measured_otf: bool = False  # True ONLY if real calibration lab measurements are supplied
    calibration_source: Optional[str] = None  # Document reference, report ID, or manifest link
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AntiAliasingResult:
    """Result of downsampling with explicit provenance and scientific honesty tracking."""
    output_raster: np.ndarray
    method: AntiAliasingMethod
    downsample_factor: float
    otf_correction_claimed: bool  # Strictly False if has_measured_otf is False
    provenance: Dict[str, Any]
    disclaimer: str = PHYSICAL_OTF_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "downsample_factor": float(self.downsample_factor),
            "output_shape": list(self.output_raster.shape),
            "otf_correction_claimed": self.otf_correction_claimed,
            "provenance": self.provenance,
            "disclaimer": self.disclaimer,
        }


@dataclass
class ComparisonMetrics:
    """Quantitative comparison metrics across anti-aliasing methods."""
    method: str
    edge_preservation_score: float  # Normalized Tenengrad gradient sharpness
    aliasing_metric: float  # High-frequency spectral energy ratio (lower = less aliasing)
    registration_error_px: float  # Sub-pixel RMSE against known translation
    otf_correction_claimed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Synthetic Frequency-Pattern Generators (Requirement 7)
# ---------------------------------------------------------------------------
def generate_zone_plate(size: int = 256, max_freq: float = 0.5) -> np.ndarray:
    """
    Generates a 2D radial chirp / Fresnel zone plate: I(r) = 0.5 * (1 + cos(alpha * r^2)).
    Sweeps spatial frequencies radially from DC (center) to Nyquist (periphery).
    """
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r_sq = xx ** 2 + yy ** 2
    # alpha scales frequency so that instantaneous frequency at border r=1 equals max_freq * pi
    alpha = max_freq * np.pi * size
    zp = 0.5 * (1.0 + np.cos(alpha * r_sq))
    return np.clip(zp, 0.0, 1.0).astype(np.float32)


def generate_siemens_star(size: int = 256, num_spokes: int = 32) -> np.ndarray:
    """
    Generates a Siemens star test pattern with spoke rays.
    Spatial frequency increases inversely proportional to radius, clearly exposing
    Moiré artifacts, aliasing, and spurious resolution wheels at the center.
    """
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    theta = np.arctan2(yy, xx)  # range [-pi, pi]
    star = 0.5 * (1.0 + np.sign(np.sin(num_spokes * theta)))
    return np.clip(star, 0.0, 1.0).astype(np.float32)


def generate_high_freq_chirp(size: int = 256) -> np.ndarray:
    """Generates a 1D horizontal chirp grating swept linearly in frequency."""
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    # Quadratic phase creates linear frequency sweep from 0 to Nyquist
    phase = np.pi * (size / 2.0) * (x ** 2)
    chirp_1d = 0.5 * (1.0 + np.cos(phase))
    return np.tile(chirp_1d, (size, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Core Anti-Aliasing Downsampling Engine (Requirements 1, 2, 3, 4)
# ---------------------------------------------------------------------------
def _compute_nyquist_sigma(factor: float) -> float:
    """
    Computes Gaussian filter sigma matched to the downsampling factor's Nyquist limit.
    For scale factor s = in_dim / out_dim > 1, the new Nyquist frequency is 0.5 / s.
    The anti-aliasing Gaussian kernel standard deviation is sigma = sqrt(s^2 - 1) * 0.5.
    """
    if factor <= 1.0:
        return 0.0
    return float(np.sqrt(max(0.0, factor ** 2 - 1.0)) * 0.5)


def _construct_2d_otf(shape: Tuple[int, int], config: SensorOTFConfig) -> np.ndarray:
    """
    Constructs a 2D Optical Transfer Function H(u, v) centered at zero frequency.
    """
    h, w = shape
    u = np.fft.fftfreq(w)
    v = np.fft.fftfreq(h)
    uu, vv = np.meshgrid(u, v)
    rho = np.sqrt(uu ** 2 + vv ** 2)

    fc = max(0.05, min(0.5, config.cutoff_freq))

    if config.aperture_shape == "circular_diffraction":
        # Diffraction-limited circular aperture MTF: 2/pi * (arccos(f/fc) - (f/fc)*sqrt(1-(f/fc)^2))
        norm_f = rho / fc
        mtf = np.zeros_like(rho)
        mask = norm_f < 1.0
        nf = norm_f[mask]
        mtf[mask] = (2.0 / np.pi) * (np.arccos(nf) - nf * np.sqrt(np.maximum(0.0, 1.0 - nf ** 2)))
    elif config.aperture_shape == "lorentzian":
        mtf = 1.0 / (1.0 + (rho / fc) ** 2)
    else:  # "gaussian" default
        sigma_f = fc / 2.0
        mtf = np.exp(-0.5 * (rho / sigma_f) ** 2)

    return np.clip(mtf, 0.0, 1.0).astype(np.float32)


def downsample_sensor_aware(
    image: np.ndarray,
    target_shape_or_factor: Union[Tuple[int, int], float],
    method: AntiAliasingMethod = AntiAliasingMethod.INTER_AREA,
    sensor: Optional[str] = None,
    otf_config: Optional[SensorOTFConfig] = None,
) -> AntiAliasingResult:
    """
    Downsamples an image with optional sensor-aware anti-aliasing prefiltering.

    Guarantees:
    - Default method is cv2.INTER_AREA (reproducible baseline).
    - If method is GAUSSIAN_PREFILTER, applies sensor-adapted Nyquist Gaussian prefiltering.
    - If method is CONFIGURED_OTF, applies frequency-domain OTF shaping.
    - If no measured sensor OTF is supplied (has_measured_otf=False), otf_correction_claimed is
      STRICTLY set to False with an explicit provenance disclaimer (Rule 13).
    """
    img_float = image.astype(np.float32)
    orig_h, orig_w = img_float.shape[:2]

    if isinstance(target_shape_or_factor, (int, float)):
        factor = float(target_shape_or_factor)
        target_w = max(1, int(round(orig_w / factor)))
        target_h = max(1, int(round(orig_h / factor)))
    else:
        target_h, target_w = target_shape_or_factor
        factor = float(orig_w) / float(target_w) if target_w > 0 else 1.0

    provenance: Dict[str, Any] = {
        "sensor": sensor or "GENERIC",
        "method": method.value,
        "input_dimensions": [orig_w, orig_h],
        "output_dimensions": [target_w, target_h],
        "downsample_factor": factor,
    }

    # 1. Baseline cv2.INTER_AREA (Requirement 1)
    if method == AntiAliasingMethod.INTER_AREA:
        out = cv2.resize(img_float, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return AntiAliasingResult(
            output_raster=out,
            method=method,
            downsample_factor=factor,
            otf_correction_claimed=False,
            provenance=provenance,
        )

    # 2. Gaussian Prefilter + INTER_AREA (Requirement 2)
    elif method == AntiAliasingMethod.GAUSSIAN_PREFILTER:
        sigma = _compute_nyquist_sigma(factor)
        provenance["gaussian_sigma"] = sigma
        if sigma > 0.1:
            # Kernel size roughly 6 * sigma rounded to odd integer
            ksize = int(np.ceil(sigma * 6.0))
            if ksize % 2 == 0:
                ksize += 1
            filtered = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)
        else:
            filtered = img_float

        out = cv2.resize(filtered, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return AntiAliasingResult(
            output_raster=out,
            method=method,
            downsample_factor=factor,
            otf_correction_claimed=False,
            provenance=provenance,
        )

    # 3. Configured OTF Filter (Requirements 3 & 4)
    elif method == AntiAliasingMethod.CONFIGURED_OTF:
        cfg = otf_config or SensorOTFConfig(sensor=sensor or "GENERIC")
        provenance["otf_config"] = cfg.to_dict()

        # Strict compliance with Rule 13: Never claim OTF correction without sensor OTF!
        if not cfg.has_measured_otf:
            otf_claimed = False
            provenance["provenance_status"] = "UNMEASURED_HEURISTIC_OTF"
            provenance["warning"] = "Synthetic/heuristic OTF used. No actual sensor OTF measurements supplied."
        else:
            otf_claimed = True
            provenance["provenance_status"] = "MEASURED_SENSOR_OTF"
            provenance["calibration_source"] = cfg.calibration_source

        # Frequency domain processing via 2D FFT
        dft = np.fft.fft2(img_float)
        otf = _construct_2d_otf((orig_h, orig_w), cfg)
        # Apply OTF low-pass shaping to attenuate beyond cutoff
        filtered_freq = dft * otf
        filtered = np.real(np.fft.ifft2(filtered_freq)).astype(np.float32)

        out = cv2.resize(filtered, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return AntiAliasingResult(
            output_raster=out,
            method=method,
            downsample_factor=factor,
            otf_correction_claimed=otf_claimed,
            provenance=provenance,
        )

    else:
        raise ValueError(f"Unsupported anti-aliasing method: {method}")


# ---------------------------------------------------------------------------
# Quantitative Metrics: Edge Preservation, Aliasing, Registration Error (Requirement 6)
# ---------------------------------------------------------------------------
def compute_edge_preservation_score(image: np.ndarray) -> float:
    """
    Computes Tenengrad gradient energy density as an indicator of edge sharpness:
    E = (1 / N) * sum( (Sobel_x)^2 + (Sobel_y)^2 ).
    """
    img = np.ascontiguousarray(image, dtype=np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_sq = gx ** 2 + gy ** 2
    return float(np.mean(grad_sq))


def compute_aliasing_metric(downsampled: np.ndarray) -> float:
    """
    Computes the high-frequency spectral energy ratio in the upper octave [0.25, 0.5] cycles/pixel.
    Higher values on frequency test patterns indicate greater aliased energy folded into the band.
    """
    img = np.ascontiguousarray(downsampled, dtype=np.float32)
    h, w = img.shape
    dft = np.fft.fftshift(np.fft.fft2(img))
    mag_sq = np.abs(dft) ** 2

    # Radial frequency distance from DC center
    u = np.linspace(-0.5, 0.5, w)
    v = np.linspace(-0.5, 0.5, h)
    uu, vv = np.meshgrid(u, v)
    rho = np.sqrt(uu ** 2 + vv ** 2)

    total_energy = float(np.sum(mag_sq))
    if total_energy <= 1e-12:
        return 0.0

    # Upper octave mask: frequencies between 0.25 and 0.5
    upper_mask = (rho >= 0.25) & (rho <= 0.5)
    high_freq_energy = float(np.sum(mag_sq[upper_mask]))
    return high_freq_energy / total_energy


def compute_registration_error(
    base_image: np.ndarray,
    method: AntiAliasingMethod,
    downsample_factor: float,
    true_shift: Tuple[float, float] = (1.5, 2.5),
    sensor: Optional[str] = None,
    otf_config: Optional[SensorOTFConfig] = None,
) -> float:
    """
    Evaluates sub-pixel registration accuracy after anti-aliasing downsampling:
    1. Imposes known sub-pixel affine translation (dx, dy) on base_image.
    2. Downsamples both original and shifted image using the specified anti-aliasing method.
    3. Recovers translation via sub-pixel phase correlation.
    4. Computes displacement RMSE in pixel space: sqrt((dx_recovered - dx_true_scaled)^2 + ...).
    """
    dx, dy = true_shift
    h, w = base_image.shape[:2]

    # Apply known affine shift to high-res image
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    shifted_img = cv2.warpAffine(base_image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Downsample both using the candidate method
    res_base = downsample_sensor_aware(base_image, downsample_factor, method, sensor, otf_config)
    res_shifted = downsample_sensor_aware(shifted_img, downsample_factor, method, sensor, otf_config)

    # Expected shift in downsampled domain
    expected_dx = dx / downsample_factor
    expected_dy = dy / downsample_factor

    # Sub-pixel phase correlation estimation
    b1 = res_base.output_raster.astype(np.float32)
    b2 = res_shifted.output_raster.astype(np.float32)
    (recovered_dx, recovered_dy), response = cv2.phaseCorrelate(b1, b2)

    error_px = float(np.sqrt((recovered_dx - expected_dx) ** 2 + (recovered_dy - expected_dy) ** 2))
    return error_px


# ---------------------------------------------------------------------------
# Comprehensive Three-Way Comparison (Requirement 5)
# ---------------------------------------------------------------------------
def compare_anti_aliasing_methods(
    image: np.ndarray,
    downsample_factor: float = 4.0,
    sensor: str = "OHRC",
    otf_config: Optional[SensorOTFConfig] = None,
    true_shift: Tuple[float, float] = (2.0, 1.5),
) -> Dict[str, ComparisonMetrics]:
    """
    Compares the 3 methods required by Requirement 5:
    1. INTER_AREA (baseline)
    2. GAUSSIAN_PREFILTER + INTER_AREA
    3. CONFIGURED_OTF filter

    Computes:
    - Edge preservation score
    - Aliasing metric
    - Registration error
    - Scientific OTF claim status
    """
    methods = [
        AntiAliasingMethod.INTER_AREA,
        AntiAliasingMethod.GAUSSIAN_PREFILTER,
        AntiAliasingMethod.CONFIGURED_OTF,
    ]
    results: Dict[str, ComparisonMetrics] = {}

    for m in methods:
        ds_res = downsample_sensor_aware(
            image,
            target_shape_or_factor=downsample_factor,
            method=m,
            sensor=sensor,
            otf_config=otf_config,
        )
        edge_score = compute_edge_preservation_score(ds_res.output_raster)
        aliasing_score = compute_aliasing_metric(ds_res.output_raster)
        reg_err = compute_registration_error(
            image,
            method=m,
            downsample_factor=downsample_factor,
            true_shift=true_shift,
            sensor=sensor,
            otf_config=otf_config,
        )
        results[m.value] = ComparisonMetrics(
            method=m.value,
            edge_preservation_score=edge_score,
            aliasing_metric=aliasing_score,
            registration_error_px=reg_err,
            otf_correction_claimed=ds_res.otf_correction_claimed,
        )

    return results
