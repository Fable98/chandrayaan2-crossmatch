from __future__ import annotations

import numpy as np
import cv2
from rasterio.transform import Affine
from skimage.transform import pyramid_gaussian


def scale_factor(native_gsd: float | None, working_gsd: float) -> float:
    if not native_gsd or native_gsd <= 0:
        return 1.0
    return working_gsd / native_gsd


def resample_to_gsd(
    arr: np.ndarray,
    transform: Affine | None,
    native_gsd: float | None,
    working_gsd: float,
) -> tuple[np.ndarray, Affine | None, float]:
    """
    Resample so matching GSD is shared (TMC up / OHRC down).
    scale > 1 means coarsen (OHRC → working); < 1 means refine (if ever needed).
    """
    sf = scale_factor(native_gsd, working_gsd)
    if abs(sf - 1.0) < 1e-6:
        return arr, transform, 1.0

    _, h, w = arr.shape
    new_h = max(1, int(round(h / sf)))
    new_w = max(1, int(round(w / sf)))
    # AREA interpolation for downsampling (proper pixel-area averaging;
    # skimage.resize's anti_aliasing Gaussian is slower and softer). CUBIC
    # for the rare upscale path.
    interp = cv2.INTER_AREA if (new_h <= h and new_w <= w) else cv2.INTER_CUBIC
    bands = []
    for i in range(arr.shape[0]):
        band = np.ascontiguousarray(arr[i], dtype=np.float32)
        bands.append(cv2.resize(band, (new_w, new_h), interpolation=interp))
    out = np.stack(bands, axis=0).astype(np.float32, copy=False)

    new_transform = None
    if transform is not None:
        new_transform = transform * Affine.scale(w / new_w, h / new_h)
    return out, new_transform, sf


def gaussian_pyramid(arr: np.ndarray, max_layer: int = 5) -> list[np.ndarray]:
    """Per-band Gaussian pyramid; index 0 is native / current working resolution."""
    levels: list[np.ndarray] = []
    band_pyrs = []
    for b in range(arr.shape[0]):
        gen = pyramid_gaussian(arr[b], max_layer=max_layer, downscale=2, channel_axis=None)
        band_pyrs.append([np.asarray(layer, dtype=np.float32) for layer in gen])
    n_levels = min(len(p) for p in band_pyrs)
    for i in range(n_levels):
        levels.append(np.stack([bp[i] for bp in band_pyrs], axis=0))
    return levels
