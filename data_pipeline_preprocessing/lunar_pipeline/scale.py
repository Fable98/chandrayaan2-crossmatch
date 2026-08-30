from __future__ import annotations

import numpy as np
from rasterio.transform import Affine
from skimage.transform import pyramid_gaussian, resize


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
    out = np.stack(
        [resize(arr[i], (new_h, new_w), anti_aliasing=True, preserve_range=True) for i in range(arr.shape[0])],
        axis=0,
    ).astype(np.float32)

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
