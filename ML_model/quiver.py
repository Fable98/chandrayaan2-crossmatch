"""
ML_model/quiver.py — Displacement-vector quiver QA overlay.

Renders per-inlier reprojection residual vectors (projected source minus
destination) as arrows over the reference frame. Uniform small arrows mean a
rigid, well-conditioned fit; large or spatially coherent arrows flag local
non-planarity (relief) or a strained homography — visible at a glance in a
way RMSE scalars are not.

Headless-safe (Agg backend) for Render free tier. matplotlib is imported
lazily inside the function so module import stays light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def create_displacement_quiver(
    inlier_src: np.ndarray,
    inlier_dst: np.ndarray,
    H: np.ndarray,
    image_shape: Tuple[int, int],
    path: str | Path,
    max_arrows: int = 100,
    title: Optional[str] = None,
) -> Optional[str]:
    """Draw residual-vector quiver plot; returns str(path) or None on failure.

    Args:
        inlier_src: (N, 2) source points used in the fit.
        inlier_dst: (N, 2) destination points used in the fit.
        H: (3, 3) homography mapping source -> destination.
        image_shape: (height, width) of the reference frame for aspect.
        path: output PNG path.
        max_arrows: cap on drawn arrows (spatially strided subsample).
        title: optional override; defaults to an RMSE/median summary.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    try:
        src = np.asarray(inlier_src, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(inlier_dst, dtype=np.float64).reshape(-1, 2)
        Hm = np.asarray(H, dtype=np.float64).reshape(3, 3)
        if len(src) == 0 or len(src) != len(dst):
            return None

        ones = np.ones((len(src), 1))
        proj = (Hm @ np.hstack([src, ones]).T).T
        proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-12)
        resid = proj - dst
        mags = np.linalg.norm(resid, axis=1)
        rmse = float(np.sqrt(np.mean(mags ** 2)))
        med = float(np.median(mags))

        n = len(src)
        if n > max_arrows:
            idx = np.linspace(0, n - 1, max_arrows).astype(int)
            px, py, u, v, m = dst[idx, 0], dst[idx, 1], resid[idx, 0], resid[idx, 1], mags[idx]
        else:
            px, py, u, v, m = dst[:, 0], dst[:, 1], resid[:, 0], resid[:, 1], mags

        h, w = int(image_shape[0]), int(image_shape[1])
        fig, ax = plt.subplots(figsize=(6, 6 * max(h, 1) / max(w, 1)))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        sc = max(float(np.max(mags)), 1e-6)
        ax.quiver(px, py, u, v, m, cmap="coolwarm", clim=(0, max(sc, 1.0)),
                  angles="xy", scale_units="xy", scale=0.2, width=0.004)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_title(title or f"Residual displacement vectors (N={n}, RMSE={rmse:.3f}px, median={med:.3f}px)")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=120, bbox_inches="tight")
        plt.close(fig)
        return str(out)
    except Exception:
        try:
            plt.close("all")  # noqa
        except Exception:
            pass
        return None
