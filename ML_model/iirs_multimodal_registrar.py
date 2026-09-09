"""ML_model/iirs_multimodal_registrar.py — IIRS <-> OHRC multi-modal co-registration.

Strategy (why this exists):
  * Chandrayaan-2 IIRS is a 256-band hyperspectral cube (~80 m/px) with
    broad mineralogical gradients and almost no sharp edges, so classic
    feature matchers (SIFT/ORB) fail on it.
  * OHRC is panchromatic (~0.3 m/px) with strong topographic shading.
  * We therefore compress IIRS to a single spatial-structure channel via
    PCA (PC1), photometrically normalise both modalities (CLAHE + blur),
    align them with area-based ECC on an image pyramid, then emit a
    uniform grid of DERIVED (composed, not independently measured) tie-points
    warped into OHRC coordinates for overlay/context only.

Provenance honesty:
  * Points from generate_uniform_tie_points() are DERIVED from the estimated
    area-based warp, NOT independently verified correspondences. They must
    never be counted as measured inliers, never drive Fit RMSE, and are valid
    only as a spatial-spectral contextual overlay given the ~275x scale gap.
  * Direct IIRS tie-point extraction at sub-meter precision is unphysical;
    see README Limitations.
"""

Memory guardrails:
  * Hyperspectral cube is reshaped to (H*W, Bands) once; no 256xHxW
    float64 copies are kept alive longer than needed.
  * MemoryError during reshape/PCA is caught, logged, and re-raised so
    the top-level ``register_iirs_to_ohrc`` can return a clean
    ``{"status": "failed", ...}`` dict instead of crashing.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import rasterio
from sklearn.decomposition import PCA

logger = logging.getLogger("ML_model.iirs_multimodal_registrar")


class IIRS_Multimodal_Registrar:
    """Area-based registrar for IIRS (hyperspectral) -> OHRC (panchromatic)."""

    # ------------------------------------------------------------------
    # Method 1: hyperspectral -> PC1 spatial structure
    # ------------------------------------------------------------------
    def load_and_reduce_hyperspectral(self, iirs_path: str) -> np.ndarray:
        """Load IIRS cube and reduce it to a uint8 PC1 structure image.

        Args:
            iirs_path: Path to the IIRS GeoTIFF (bands, H, W).

        Returns:
            PC1 image of shape (H, W), dtype uint8, range 0-255.
        """
        try:
            with rasterio.open(iirs_path) as src:
                cube = src.read()  # (Bands, H, W)
                bands, height, width = cube.shape
            logger.info("IIRS cube: bands=%d H=%d W=%d", bands, height, width)

            # Reshape to (H*W, Bands) for PCA. Use float32 (not float64)
            # to halve the RAM spike on massive scenes.
            try:
                data = np.transpose(cube, (1, 2, 0)).reshape(-1, bands).astype(
                    np.float32, copy=False
                )
            except MemoryError:
                logger.exception("MemoryError reshaping IIRS cube to (H*W, Bands).")
                raise
            finally:
                del cube  # free the (B, H, W) copy ASAP

            # Handle NaNs/Infs: replace with per-band mean of valid pixels.
            finite_mask = np.isfinite(data)
            if not bool(np.all(finite_mask)):
                band_means = np.zeros((data.shape[1],), dtype=np.float32)
                for b in range(data.shape[1]):
                    col = data[:, b]
                    valid = col[np.isfinite(col)]
                    band_means[b] = float(np.mean(valid)) if valid.size else 0.0
                    col[~np.isfinite(col)] = band_means[b]
                    data[:, b] = col
                logger.info("Replaced NaN/Inf pixels with per-band means.")

            # PCA -> PC1.
            try:
                pca = PCA(n_components=1)
                pc1_flat = pca.fit_transform(data)[:, 0]
                var_ratio = float(pca.explained_variance_ratio_[0])
                logger.info("PCA PC1 explained variance ratio: %.4f", var_ratio)
            except MemoryError:
                logger.exception("MemoryError during PCA(n_components=1).")
                raise
            finally:
                del data

            pc1 = pc1_flat.reshape(height, width).astype(np.float32)
            del pc1_flat

            # Normalize PC1 to 0-255 uint8.
            pc1_min, pc1_max = float(pc1.min()), float(pc1.max())
            if pc1_max > pc1_min:
                pc1_norm = (pc1 - pc1_min) / (pc1_max - pc1_min) * 255.0
            else:
                pc1_norm = np.zeros_like(pc1)
            return np.clip(pc1_norm, 0, 255).astype(np.uint8)
        except MemoryError:
            logger.exception("IIRS PCA failed: image too large for RAM.")
            raise
        except Exception:
            logger.exception("Failed to load/reduce hyperspectral: %s", iirs_path)
            raise

    # ------------------------------------------------------------------
    # Method 2: photometric normalisation for ECC
    # ------------------------------------------------------------------
    def preprocess_for_ecc(self, img: np.ndarray) -> np.ndarray:
        """Prepare an image for ECC (float32, contrast-equalised, denoised).

        ECC optimises gradient correlation, so both modalities must have
        comparable local contrast. CLAHE aligns mineral gradients with
        topographic shading; a mild blur removes noise that breaks ECC.
        """
        arr = np.asarray(img)
        # Drop colour channels -> single grayscale plane.
        if arr.ndim == 3:
            if arr.shape[2] in (3, 4):
                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                arr = np.mean(arr, axis=2)
        arr = np.squeeze(arr)

        # CLAHE needs uint8, so normalise to 0-255 first (ECC itself
        # needs float32 afterwards — hence this uint8 -> CLAHE -> float32
        # order, not float32-first).
        arr_f = arr.astype(np.float32)
        a_min, a_max = float(arr_f.min()), float(arr_f.max())
        if a_max > a_min:
            gray8 = ((arr_f - a_min) / (a_max - a_min) * 255.0).astype(np.uint8)
        else:
            gray8 = np.zeros_like(arr_f, dtype=np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalised = clahe.apply(gray8)
        blurred = cv2.GaussianBlur(equalised, (3, 3), 0)

        # ECC requires float32; scale to [0, 1] for stable gradients.
        return (blurred.astype(np.float32) / 255.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Method 3: coarse-to-fine ECC on a Gaussian pyramid
    # ------------------------------------------------------------------
    def align_ecc_pyramid(
        self,
        ref_img: np.ndarray,
        src_img: np.ndarray,
        num_levels: int = 3,
    ) -> np.ndarray:
        """Estimate a 2x3 affine warp (src -> ref) via pyramid ECC.

        Args:
            ref_img: Preprocessed OHRC reference (float32, 2D).
            src_img: Preprocessed IIRS PC1 moving image (float32, 2D).
            num_levels: Number of pyramid levels (>= 1).

        Returns:
            2x3 affine warp matrix (float32). If ECC diverges at some
            level, the best matrix computed up to that point is returned.
        """
        ref = np.asarray(ref_img, dtype=np.float32)
        src = np.asarray(src_img, dtype=np.float32)
        if ref.ndim != 2:
            ref = np.squeeze(ref)
        if src.ndim != 2:
            src = np.squeeze(src)

        # ECC requires identical sizes: resample the moving image to the
        # reference frame before building pyramids.
        if src.shape != ref.shape:
            logger.info(
                "Resizing src %s -> ref %s for ECC.", src.shape, ref.shape
            )
            src = cv2.resize(
                src, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR
            )

        num_levels = max(1, int(num_levels))
        logger.info("ECC pyramid levels: %d", num_levels)

        # Build Gaussian pyramids (index 0 = full resolution).
        ref_pyr = [ref]
        src_pyr = [src]
        for _ in range(1, num_levels):
            ref_pyr.append(cv2.pyrDown(ref_pyr[-1]))
            src_pyr.append(cv2.pyrDown(src_pyr[-1]))

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            500,
            1e-6,
        )
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        last_cc = None

        # Coarse-to-fine: start at the smallest level.
        for lvl in range(num_levels - 1, -1, -1):
            small_ref = ref_pyr[lvl]
            small_src = src_pyr[lvl]
            try:
                cc, warp_matrix = cv2.findTransformECC(
                    small_ref,
                    small_src,
                    warp_matrix,
                    cv2.MOTION_AFFINE,
                    criteria,
                    None,
                    1,
                )
                last_cc = float(cc)
                logger.info("ECC level %d converged (cc=%.6f).", lvl, last_cc)
            except cv2.error as exc:
                # Mandatory guardrail: gradient correlation can hit zero
                # on multi-modal pairs — never crash, keep prior estimate.
                logger.warning("ECC failed to converge at level %d: %s", lvl, exc)
                return warp_matrix

            # Upscale translation for the next (finer) level. The 2x2
            # linear part is scale-invariant; only tx/ty double.
            if lvl != 0:
                warp_matrix[0, 2] *= 2.0
                warp_matrix[1, 2] *= 2.0

        if last_cc is not None:
            logger.info("Final ECC correlation score: %.6f", last_cc)
        return warp_matrix.astype(np.float32)

    # ------------------------------------------------------------------
    # Method 4: uniform DERIVED tie-points (composed overlay only, NOT measured)
    # ------------------------------------------------------------------
    def generate_uniform_tie_points(
        self,
        iirs_shape: tuple,
        warp_matrix: np.ndarray,
        grid_size: int = 10,
        ohrc_shape: tuple | None = None,
    ) -> dict:
        """Warp a uniform IIRS grid into OHRC coordinates (DERIVED overlay).

        Homogeneous math for each grid point (x_s, y_s):
            [x_r]   [w00 w01 w02] [x_s]
            [y_r] = [w10 w11 w12] [y_s]
                                       [1]
        i.e. x_r = w00*x_s + w01*y_s + w02 (same for y_r).
        Vectorised: ref = (W @ src_h.T).T with src_h = [x_s, y_s, 1].

        WARNING: returned points are derived from the area-based warp estimate.
        They are NOT independently measured correspondences. Callers must tag
        them provenance=derived_composed, inlier_count=0 for metrics, and must
        not compute Fit RMSE from them.
        """
        height, width = int(iirs_shape[0]), int(iirs_shape[1])
        grid_size = max(2, int(grid_size))

        xs = np.linspace(0, width - 1, grid_size, dtype=np.float32)
        ys = np.linspace(0, height - 1, grid_size, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        src_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

        warp = np.asarray(warp_matrix, dtype=np.float64).reshape(2, 3)
        ones = np.ones((src_pts.shape[0], 1), dtype=np.float64)
        src_h = np.hstack([src_pts.astype(np.float64), ones])  # (N, 3)
        # Homogeneous warp: [x_r, y_r]^T = W * [x_s, y_s, 1]^T.
        ref_pts = (warp @ src_h.T).T.astype(np.float32)  # (N, 2)

        # Filter points falling outside the OHRC frame (if known).
        if ohrc_shape is not None:
            oh, ow = int(ohrc_shape[0]), int(ohrc_shape[1])
            inside = (
                (ref_pts[:, 0] >= 0)
                & (ref_pts[:, 0] < ow)
                & (ref_pts[:, 1] >= 0)
                & (ref_pts[:, 1] < oh)
            )
            src_pts = src_pts[inside]
            ref_pts = ref_pts[inside]

        return {
            "status": "success",
            "derivation": "derived_composed_overlay",
            "is_measured_correspondence": False,
            "provenance": "ECC area-based warp estimate; not RANSAC-verified inliers",
            "use_restriction": "contextual overlay only; do not use for Fit RMSE or sub-pixel claims",
            "src_pts": src_pts,
            "ref_pts": ref_pts,
            "warp_matrix": np.asarray(warp_matrix, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Main method: end-to-end IIRS -> OHRC registration
    # ------------------------------------------------------------------
    def register_iirs_to_ohrc(self, iirs_path: str, ohrc_img: np.ndarray) -> dict:
        """Full chain: PC1 -> ECC preprocess -> pyramid ECC -> derived overlay points."""
        try:
            pc1 = self.load_and_reduce_hyperspectral(iirs_path)

            if isinstance(ohrc_img, str):
                ohrc_arr = cv2.imread(ohrc_img, cv2.IMREAD_UNCHANGED)
                if ohrc_arr is None:
                    raise FileNotFoundError(f"Could not read OHRC: {ohrc_img}")
            else:
                ohrc_arr = np.asarray(ohrc_img)

            prep_iirs = self.preprocess_for_ecc(pc1)
            prep_ohrc = self.preprocess_for_ecc(ohrc_arr)

            warp_matrix = self.align_ecc_pyramid(prep_ohrc, prep_iirs, num_levels=3)

            result = self.generate_uniform_tie_points(
                pc1.shape,
                warp_matrix,
                grid_size=10,
                ohrc_shape=ohrc_arr.shape[:2],
            )
            logger.info("IIRS->OHRC tie-points: N=%d", len(result["src_pts"]))
            return result
        except Exception as exc:  # never crash the master pipeline
            logger.exception("IIRS->OHRC registration failed: %s", exc)
            return {"status": "failed", "reason": str(exc)}


__all__ = ["IIRS_Multimodal_Registrar"]
