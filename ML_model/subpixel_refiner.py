"""Sub-pixel refinement of coarse integer matches via NCC + 2D parabolic fitting.

Coordinate convention (CRITICAL GUARDRAIL):
    (x, y) == (column, row).
    NumPy images are indexed as img[row, col] == img[y, x], therefore every
    crop MUST be sliced as img[y:y + h, x:x + w]. Never swap these axes.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SubPixelRefiner:
    """Refine coarse integer-pixel correspondences to sub-pixel accuracy.

    Strategy: extract a small reference patch, correlate it against a
    slightly larger search window in the source image with normalized
    cross-correlation (NCC), then fit a separable 1-D parabola along each
    axis through the correlation peak to recover the fractional offset.
    """

    def __init__(self, min_patch_std: float = 5.0, min_confidence: float = 0.1):
        """Initialise the refiner.

        Args:
            min_patch_std: Shadow guardrail. Reference patches with a
                standard deviation below this are treated as blank / deep
                shadow and are not refined.
            min_confidence: Default confidence cut-off used by
                :meth:`refine_batch`.
        """
        self.min_patch_std = float(min_patch_std)
        self.min_confidence = float(min_confidence)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
        """Convert an image to single-channel uint8 for matchTemplate."""
        if img is None or img.size == 0:
            raise ValueError("Empty image passed to SubPixelRefiner.")
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.dtype != np.uint8:
            arr = cv2.normalize(
                arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _clamp_center(pt, half: int, w: int, h: int) -> tuple:
        """Clamp a patch centre so a (2*half+1) window stays in-bounds."""
        x, y = float(pt[0]), float(pt[1])
        x = min(max(x, half), w - 1 - half)
        y = min(max(y, half), h - 1 - half)
        return x, y

    # ------------------------------------------------------------------
    def refine_match(
        self,
        ref_img: np.ndarray,
        src_img: np.ndarray,
        ref_pt: tuple,
        src_pt: tuple,
        patch_size: int = 21,
    ) -> tuple:
        """Refine one coarse match to sub-pixel accuracy.

        Args:
            ref_img: Reference (template) image.
            src_img: Source (search) image.
            ref_pt: Coarse (x=col, y=row) location in ``ref_img``.
            src_pt: Coarse (x=col, y=row) location in ``src_img``.
            patch_size: Odd side length of the reference patch.

        Returns:
            ``(final_x, final_y, confidence)`` where ``(final_x, final_y)``
            is the refined source-image coordinate and ``confidence`` is
            the NCC peak score in [0, 1] (0.0 when refinement is aborted).
        """
        # Enforce an odd patch size so the window has a true centre pixel.
        patch_size = int(patch_size)
        if patch_size % 2 == 0:
            patch_size += 1
        patch_size = max(3, patch_size)
        half = patch_size // 2
        margin = 5  # search window extends 5 px beyond patch on each side
        search_size = patch_size + 2 * margin  # == patch_size + 10
        s_half = search_size // 2

        ref = self._to_gray_uint8(ref_img)
        src = self._to_gray_uint8(src_img)
        h_ref, w_ref = ref.shape[:2]
        h_src, w_src = src.shape[:2]

        if (
            patch_size >= min(h_ref, w_ref)
            or search_size >= min(h_src, w_src)
        ):
            logger.warning("Image smaller than patch/search window; aborting.")
            return float(src_pt[0]), float(src_pt[1]), 0.0

        # --- Extract patches (GUARDRAIL: img[y, x], i.e. img[row, col]) ---
        rcx, rcy = self._clamp_center(ref_pt, half, w_ref, h_ref)
        scx, scy = self._clamp_center(src_pt, s_half, w_src, h_src)

        ref_x0, ref_y0 = int(round(rcx)) - half, int(round(rcy)) - half
        win_x0, win_y0 = int(round(scx)) - s_half, int(round(scy)) - s_half
        # Clamp top-left corners defensively after rounding.
        ref_x0 = min(max(ref_x0, 0), w_ref - patch_size)
        ref_y0 = min(max(ref_y0, 0), h_ref - patch_size)
        win_x0 = min(max(win_x0, 0), w_src - search_size)
        win_y0 = min(max(win_y0, 0), h_src - search_size)

        # NOTE: row index (y) comes first, column index (x) second.
        ref_patch = ref[ref_y0:ref_y0 + patch_size,
                        ref_x0:ref_x0 + patch_size]
        search_win = src[win_y0:win_y0 + search_size,
                         win_x0:win_x0 + search_size]

        # --- Shadow guardrail: blank/shadowed patches carry no signal. ---
        if float(np.std(ref_patch.astype(np.float32))) < self.min_patch_std:
            return float(src_pt[0]), float(src_pt[1]), 0.0

        # --- NCC correlation surface (template vs. search window). ---
        corr = cv2.matchTemplate(search_win, ref_patch, cv2.TM_CCORR_NORMED)
        corr = np.asarray(corr, dtype=np.float32)
        py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
        peak = float(corr[py, px])
        ch, cw = corr.shape  # == (2*margin+1, 2*margin+1) == 11x11

        # Integer peak on the correlation border -> parabola undefined.
        if px == 0 or py == 0 or px == cw - 1 or py == ch - 1:
            fx = float(win_x0 + px + half)
            fy = float(win_y0 + py + half)
            return fx, fy, max(0.0, min(1.0, peak))

        # --- 2D separable parabolic sub-pixel fit. ---
        # For three samples f(-1), f(0), f(+1) of a parabola
        #   f(t) = a*t^2 + b*t + c,
        # the vertex offset from the centre sample is
        #   t* = 0.5 * (f(-1) - f(+1)) / (f(-1) - 2*f(0) + f(+1)).
        # Applied independently along x (same row py) and y (same col px).
        c = float(corr[py, px])
        lx, rx = float(corr[py, px - 1]), float(corr[py, px + 1])
        uy, dy = float(corr[py - 1, px]), float(corr[py + 1, px])
        denom_x = lx - 2.0 * c + rx
        denom_y = uy - 2.0 * c + dy
        dx = 0.5 * (lx - rx) / denom_x if abs(denom_x) > 1e-9 else 0.0
        dy_ = 0.5 * (uy - dy) / denom_y if abs(denom_y) > 1e-9 else 0.0
        # Clamp the fractional correction to half a pixel (parabola vertex
        # must lie between the neighbouring samples to be trustworthy).
        dx = max(-0.5, min(0.5, dx))
        dy_ = max(-0.5, min(0.5, dy_))

        sub_x = float(px) + dx
        sub_y = float(py) + dy_

        # --- Map correlation-peak coords back to source-image pixels. ---
        # corr(px, py) is the top-left corner of the best template placement
        # inside the search window, so the patch centre (our match point) is
        #   final = window_origin + sub_peak + patch_half_width.
        final_x = float(win_x0 + sub_x + half)
        final_y = float(win_y0 + sub_y + half)
        confidence = max(0.0, min(1.0, peak))
        return final_x, final_y, confidence

    # ------------------------------------------------------------------
    def refine_batch(
        self,
        ref_img: np.ndarray,
        src_img: np.ndarray,
        ref_pts,
        src_pts,
        patch_size: int = 21,
        min_confidence: float = None,
    ) -> tuple:
        """Refine many coarse matches, dropping low-confidence results.

        Args:
            ref_img: Reference image.
            src_img: Source image.
            ref_pts: Array of shape (N, 2) with (x, y) coarse locations.
            src_pts: Array of shape (N, 2) with (x, y) coarse locations.
            patch_size: Odd reference-patch side length.
            min_confidence: Override for the instance default cut-off.

        Returns:
            ``(refined_ref, refined_src)`` as ``np.float32`` arrays of
            shape (M, 2), M <= N after confidence filtering.
        """
        thresh = self.min_confidence if min_confidence is None else float(min_confidence)
        ref_pts = np.asarray(ref_pts, dtype=np.float32).reshape(-1, 2)
        src_pts = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
        if ref_pts.shape[0] != src_pts.shape[0] or ref_pts.shape[0] == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )

        kept_ref, kept_src = [], []
        for rp, sp in zip(ref_pts, src_pts):
            try:
                fx, fy, conf = self.refine_match(
                    ref_img, src_img, (float(rp[0]), float(rp[1])),
                    (float(sp[0]), float(sp[1])), patch_size=patch_size,
                )
            except ValueError:
                continue  # e.g. empty image; skip this pair gracefully
            if conf >= thresh:
                kept_ref.append([float(rp[0]), float(rp[1])])
                kept_src.append([float(fx), float(fy)])

        if not kept_ref:
            logger.warning("SubPixelRefiner: 0/%d matches passed confidence %.3f.",
                           len(ref_pts), thresh)
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
            )
        logger.info("SubPixelRefiner: %d/%d matches kept (conf >= %.3f).",
                    len(kept_ref), len(ref_pts), thresh)
        return (
            np.asarray(kept_ref, dtype=np.float32),
            np.asarray(kept_src, dtype=np.float32),
        )
