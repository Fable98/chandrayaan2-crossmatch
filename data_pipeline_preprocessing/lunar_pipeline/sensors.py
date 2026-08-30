from __future__ import annotations

import numpy as np


def destripe_pushbroom(arr: np.ndarray) -> np.ndarray:
    """
    Column-wise median normalization for TMC/OHRC striping.
    Each column is scaled so its median matches the global median.
    """
    out = arr.copy()
    for b in range(arr.shape[0]):
        band = arr[b]
        col_med = np.nanmedian(band, axis=0)
        global_med = np.nanmedian(col_med)
        if not np.isfinite(global_med) or global_med == 0:
            continue
        scale = np.where(col_med > 0, global_med / np.maximum(col_med, 1e-6), 1.0)
        out[b] = band * scale[np.newaxis, :]
    return out.astype(np.float32)


def iirs_reduce(arr: np.ndarray, mode: str = "pca", n_components: int = 1, band_index: int = 0) -> np.ndarray:
    if arr.shape[0] == 1:
        return arr
    if mode == "band":
        idx = min(max(band_index, 0), arr.shape[0] - 1)
        return arr[idx : idx + 1]
    return _pca_bands(arr, max(1, n_components))


def _pca_bands(arr: np.ndarray, n_components: int) -> np.ndarray:
    bands, h, w = arr.shape
    x = arr.reshape(bands, -1)
    mean = x.mean(axis=1, keepdims=True)
    xc = x - mean
    cov = np.cov(xc)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    k = min(n_components, bands)
    comps = eigvecs[:, order[:k]].T @ xc
    return comps.reshape(k, h, w).astype(np.float32)


def sensor_cleanup(
    arr: np.ndarray,
    sensor: str,
    destripe: bool = True,
    iirs_mode: str = "pca",
    iirs_components: int = 1,
    iirs_band: int = 0,
) -> np.ndarray:
    if sensor == "IIRS":
        arr = iirs_reduce(arr, mode=iirs_mode, n_components=iirs_components, band_index=iirs_band)
    if destripe and sensor in {"TMC", "OHRC", "UNKNOWN"}:
        arr = destripe_pushbroom(arr)
    return arr
