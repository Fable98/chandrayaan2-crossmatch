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


def census_transform(band: np.ndarray, radius: int = 1) -> np.ndarray:
    """Census bitstring as raw uint32 codes (1, H, W).

    The old build packed 24-bit codes (radius=2) into [0, 1] float32: code
    spacing 1/2^24 sits BELOW float32 eps near 1.0, so adjacent bitstrings
    collided and Hamming structure was destroyed for bright codes. Raw uint32
    survives float32 GeoTIFF storage exactly (all values < 2^24), and the
    default radius=1 (8-bit) keeps descriptors compact. Callers needing the
    legacy float preview can min-max normalize themselves (order-preserving).

    Perf: a Numba-compiled kernel is used when numba is importable (single
    fused pass, no per-neighbour temporary); otherwise the vectorized NumPy
    fallback below runs (bit-identical output).
    """
    x = _normalize01(band)
    _nb = _census_numba(x, int(radius))
    if _nb is not None:
        return _nb[np.newaxis, ...]
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
            bits |= (neigh >= center).astype(np.uint32) << np.uint32(k)
            k += 1
            if k >= 32:
                break
        if k >= 32:
            break
    return bits[np.newaxis, ...]


def _census_numba(x: np.ndarray, radius: int) -> np.ndarray | None:
    """Numba census kernel; None when numba is unavailable or unsuitable."""
    try:
        import numba as _numba
    except Exception:
        return None
    try:
        h, w = int(x.shape[0]), int(x.shape[1])
        nbits = (2 * radius + 1) * (2 * radius + 1) - 1
        if nbits <= 0 or nbits > 32:
            return None
        xc = np.ascontiguousarray(x, dtype=np.float32)

        @_numba.njit(cache=True)
        def _kernel(img: np.ndarray, r: int, out: np.ndarray) -> None:
            hh, ww = img.shape[0], img.shape[1]
            for yy in range(hh):
                for xx in range(ww):
                    c = img[yy, xx]
                    code = np.uint32(0)
                    k = np.uint32(0)
                    for dyy in range(-r, r + 1):
                        y2 = yy + dyy
                        if y2 < 0:
                            y2 = 0
                        elif y2 >= hh:
                            y2 = hh - 1
                        for dxx in range(-r, r + 1):
                            if dxx == 0 and dyy == 0:
                                continue
                            x2 = xx + dxx
                            if x2 < 0:
                                x2 = 0
                            elif x2 >= ww:
                                x2 = ww - 1
                            if img[y2, x2] >= c:
                                code |= np.uint32(1) << k
                            k += np.uint32(1)
                    out[yy, xx] = code

        out = np.zeros((h, w), dtype=np.uint32)
        _kernel(xc, int(radius), out)
        return out
    except Exception:
        return None


def lbp(band: np.ndarray) -> np.ndarray:
    x = cv2.normalize(_normalize01(band), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    padded = np.pad(x, 1, mode="edge")
    codes = np.zeros_like(x, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neigh = padded[1 + dy : 1 + dy + x.shape[0], 1 + dx : 1 + dx + x.shape[1]]
        codes |= ((neigh >= padded[1:-1, 1:-1]) << i).astype(np.uint8)
    return (codes.astype(np.float32) / 255.0)[np.newaxis, ...]


def phase_congruency_proxy(band: np.ndarray, num_scales: int = 3, num_orientations: int = 4) -> np.ndarray:
    """
    Simplified Kovesi phase congruency via an FFT log-Gabor quadrature bank.

    PC_o = |sum_s R_{s,o}| / (sum_s |R_{s,o}| + eps) per orientation, summed
    over orientations: local Fourier energy over total amplitude. This is
    genuine phase alignment (complex even/odd responses), NOT the deleted
    predecessor which fed identical magnitudes into numerator and denominator
    (num == den -> constant ~1.0 everywhere, labeled "phase congruency").

    Simplifications vs full Kovesi: no noise-floor threshold T, no
    frequency-spread weighting W. Documented as a proxy; for the full
    treatment see ML_model/matcher_cfog.compute_phase_congruency.
    """
    x = _normalize01(band).astype(np.float64)
    h, w = x.shape
    F = np.fft.fft2(x)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0  # avoid log(0); DC killed explicitly below
    theta = np.arctan2(fy, fx)

    min_wavelength = 3.0
    mult = 2.0
    sigma_f = 0.55
    sigma_theta = float(np.pi / num_orientations / 1.5)

    energy_total = np.zeros((h, w), dtype=np.float64)
    amp_total = np.zeros((h, w), dtype=np.float64)
    for o in range(num_orientations):
        angle = o * np.pi / num_orientations
        dtheta = np.arctan2(np.sin(theta - angle), np.cos(theta - angle))
        spread = np.exp(-(dtheta * dtheta) / (2.0 * sigma_theta * sigma_theta))
        sum_even = np.zeros((h, w), dtype=np.float64)
        sum_odd = np.zeros((h, w), dtype=np.float64)
        sum_an = np.zeros((h, w), dtype=np.float64)
        for s in range(num_scales):
            fo = 1.0 / (min_wavelength * mult ** s)
            log_gabor = np.exp(
                -(np.log(radius / fo) ** 2) / (2.0 * math.log(sigma_f) ** 2)
            )
            log_gabor[0, 0] = 0.0
            resp = np.fft.ifft2(F * (log_gabor * spread))
            even, odd = resp.real, resp.imag
            an = np.sqrt(even * even + odd * odd)
            sum_even += even
            sum_odd += odd
            sum_an += an
        energy_total += np.sqrt(sum_even * sum_even + sum_odd * sum_odd)
        amp_total += sum_an
    pc = energy_total / (amp_total + EPS)
    return np.clip(pc, 0.0, 1.0).astype(np.float32)[np.newaxis, ...]


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
