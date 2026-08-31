from __future__ import annotations

import math

import cv2
import numpy as np

EPS = 1e-6


def _mu(angle_deg: float | None, default: float = 1.0) -> float:
    if angle_deg is None:
        return default
    return max(math.cos(math.radians(angle_deg)), EPS)


def lunar_lambert(incidence_deg: float | None, emission_deg: float | None, phase_deg: float | None) -> float:
    """
    McEwen Lunar-Lambert disk function:
      f = (1-L) * 2*μ0/(μ0+μ) + L * μ0
    L decreases with phase (more Lommel-Seeliger at high phase).
    """
    mu0 = _mu(incidence_deg)
    mu = _mu(emission_deg)
    g = 0.0 if phase_deg is None else abs(phase_deg)
    L = math.exp(-g / 60.0)
    lommel = 2.0 * mu0 / (mu0 + mu)
    lambert = mu0
    return (1.0 - L) * lommel + L * lambert


def hapke_disk(
    incidence_deg: float | None,
    emission_deg: float | None,
    phase_deg: float | None,
    w: float = 0.21,
    b: float = 0.21,
    c: float = 0.7,
    B0: float = 0.9,
    h: float = 0.07,
) -> float:
    """
    Simplified Hapke isotropic multiple-scattering disk function (no roughness).
    Enough to flatten sun-angle shading; not a full photometric inversion.
    """
    mu0 = _mu(incidence_deg)
    mu = _mu(emission_deg)
    g = 0.0 if phase_deg is None else math.radians(phase_deg)
    cosg = math.cos(g)
    # two-parameter HG
    p = (1 - b * b) / (1 + 2 * b * cosg + b * b) ** 1.5
    p = (1 - c) * p + c * (1 - b * b) / (1 - 2 * b * cosg + b * b) ** 1.5
    B = B0 / (1 + math.tan(abs(g) / 2.0) / max(h, EPS))
    M = 1.0  # drop H-function coupling for stability
    f = (w / 4.0 / math.pi) * (mu0 / (mu0 + mu)) * ((1 + B) * p + M)
    return max(f, EPS)


def photometric_factor(model: str, incidence: float | None, emission: float | None, phase: float | None) -> float:
    if model in (None, "none", "off"):
        return 1.0
    if model == "hapke":
        return hapke_disk(incidence, emission, phase)
    return lunar_lambert(incidence, emission, phase)


def apply_photometry(arr: np.ndarray, factor: float) -> np.ndarray:
    return (arr / max(factor, EPS)).astype(np.float32)


def shadow_mask(
    arr: np.ndarray,
    percentile: float = 3.0,
    incidence_deg: float | None = None,
    incidence_limit: float = 85.0,
) -> np.ndarray:
    band = arr[0]
    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros(band.shape, dtype=np.uint8)
    thr = np.percentile(band[finite], percentile)
    mask = (band <= thr) | (~finite)
    if incidence_deg is not None and incidence_deg >= incidence_limit:
        mask[:] = True
    return mask.astype(np.uint8)


def _normalize01(band: np.ndarray) -> np.ndarray:
    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros_like(band, dtype=np.float32)
    lo, hi = np.percentile(band[finite], (1, 99))
    if hi <= lo:
        return np.zeros_like(band, dtype=np.float32)
    out = np.clip((band - lo) / (hi - lo), 0, 1)
    return out.astype(np.float32)


def gradient_orientation(band: np.ndarray) -> np.ndarray:
    """2-channel (cos θ, sin θ) of intensity gradient — shading-robust for matching."""
    x = _normalize01(band)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    ang = np.arctan2(gy, gx)
    return np.stack([np.cos(ang), np.sin(ang)], axis=0).astype(np.float32)


def census_transform(band: np.ndarray, radius: int = 2) -> np.ndarray:
    """Census bitstring packed into float32 in [0, 1] for GeoTIFF convenience."""
    x = _normalize01(band)
    pad = np.pad(x, radius, mode="edge")
    h, w = x.shape
    bits = np.zeros((h, w), dtype=np.uint32)
    k = 0
    center = pad[radius : radius + h, radius : radius + w]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            neigh = pad[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
            bits |= (neigh >= center).astype(np.uint32) << k
            k += 1
            if k >= 32:
                break
        if k >= 32:
            break
    max_val = np.float32((1 << k) - 1) if k > 0 else 1.0
    return (bits.astype(np.float32) / max_val)[np.newaxis, ...]


def lbp(band: np.ndarray) -> np.ndarray:
    x = cv2.normalize(_normalize01(band), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    padded = np.pad(x, 1, mode="edge")
    codes = np.zeros_like(x, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neigh = padded[1 + dy : 1 + dy + x.shape[0], 1 + dx : 1 + dx + x.shape[1]]
        codes |= ((neigh >= padded[1:-1, 1:-1]) << i).astype(np.uint8)
    return (codes.astype(np.float32) / 255.0)[np.newaxis, ...]


def phase_congruency_proxy(band: np.ndarray) -> np.ndarray:
    """
    Lightweight phase-congruency proxy: local energy / (sum of amplitudes)
    via a few log-scaled DoG bandpass filters. Not Kovesi's full PC.
    """
    x = _normalize01(band)
    energies = []
    amps = []
    for sigma in (1.0, 2.0, 4.0, 8.0):
        blur = cv2.GaussianBlur(x, (0, 0), sigma)
        bandpass = x - blur
        energies.append(np.abs(bandpass))
        amps.append(np.abs(bandpass))
    num = np.sum(energies, axis=0)
    den = np.sum(amps, axis=0) + EPS
    pc = num / den
    pc = np.clip(pc, 0, 1)
    return pc.astype(np.float32)[np.newaxis, ...]


INVARIANT_FNS = {
    "gradient": lambda a: gradient_orientation(a[0]),
    "census": lambda a: census_transform(a[0]),
    "lbp": lambda a: lbp(a[0]),
    "phase": lambda a: phase_congruency_proxy(a[0]),
}


def build_invariants(arr: np.ndarray, modes: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for mode in modes:
        fn = INVARIANT_FNS.get(mode)
        if fn is None:
            continue
        out[mode] = fn(arr)
    return out
