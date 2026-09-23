from __future__ import annotations

import numpy as np
import cv2
from rasterio.transform import Affine
from skimage.transform import pyramid_gaussian


def scale_factor(native_gsd: float | None, working_gsd: float) -> float:
    if not native_gsd or native_gsd <= 0:
        return 1.0
    return working_gsd / native_gsd


def compute_effective_gsd(native_gsd_m: float, physical_scale_factor: float) -> float:
    """
    Computes effective raster GSD: effective_gsd = native_gsd / physical_scale_factor.
    physical_scale_factor = derived_pixels / original_pixels (e.g. 0.05 for 20x downsample).
    """
    if physical_scale_factor <= 0:
        raise ValueError(f"physical_scale_factor must be > 0, got {physical_scale_factor}")
    return float(native_gsd_m) / float(physical_scale_factor)


def compute_resampling_factor(original_size: int, derived_size: int) -> float:
    """Returns dimensional ratio of derived raster to original raster."""
    if original_size <= 0 or derived_size <= 0:
        raise ValueError("Sizes must be > 0")
    return float(derived_size) / float(original_size)


def resample_to_gsd(
    arr: np.ndarray,
    transform: Affine | None,
    native_gsd: float | None,
    working_gsd: float,
    anti_aliasing_method: str = "INTER_AREA",
    sensor: str | None = None,
) -> tuple[np.ndarray, Affine | None, float]:
    """
    Resample so matching GSD is shared (TMC up / OHRC down).
    scale > 1 means coarsen (OHRC → working); < 1 means refine (if ever needed).
    Preserves cv2.INTER_AREA as the reproducible default baseline (Requirement 1).
    """
    sf = scale_factor(native_gsd, working_gsd)
    if abs(sf - 1.0) < 1e-6:
        return arr, transform, 1.0

    _, h, w = arr.shape
    new_h = max(1, int(round(h / sf)))
    new_w = max(1, int(round(w / sf)))

    is_downsample = (new_h <= h and new_w <= w)
    bands = []

    if is_downsample and anti_aliasing_method.upper() != "INTER_AREA":
        try:
            from lunar_pipeline.anti_aliasing import downsample_sensor_aware, AntiAliasingMethod
            aa_method = AntiAliasingMethod(anti_aliasing_method.upper())
        except (ImportError, ValueError):
            from data_preprocessing_pipeline.lunar_pipeline.anti_aliasing import (
                downsample_sensor_aware,
                AntiAliasingMethod,
            )
            aa_method = AntiAliasingMethod(anti_aliasing_method.upper())

        for i in range(arr.shape[0]):
            band = np.ascontiguousarray(arr[i], dtype=np.float32)
            res = downsample_sensor_aware(
                band,
                target_shape_or_factor=(new_h, new_w),
                method=aa_method,
                sensor=sensor,
            )
            bands.append(res.output_raster)
    else:
        # Default reproducible INTER_AREA path
        interp = cv2.INTER_AREA if is_downsample else cv2.INTER_CUBIC
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
