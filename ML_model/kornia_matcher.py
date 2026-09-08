"""AI-based feature matching with Kornia KeyNetAffNetHardNet + AdaLAM + RANSAC.

Memory guardrail: inputs larger than 1 Mpix (H*W > 1024*1024) are downscaled
so the longest edge is 1024 px before hitting the GPU. Detected keypoint
coordinates are then multiplied by (scale_x, scale_y) to map them back onto
the original full-resolution pixel grid.
"""

import logging

import cv2
import numpy as np
import torch
from kornia.feature import KeyNetAffNetHardNet, match_adalam

logger = logging.getLogger(__name__)

_MAX_PIXELS = 1024 * 1024
_MAX_EDGE = 1024
_LOWE_RATIO = 0.85


class KorniaAI_Matcher:
    """Detect, describe (KeyNet+AffNet+HardNet) and robustly match features."""

    def __init__(self, num_features: int = 8000, upright: bool = False):
        """Initialise the detector/descriptor on the best available device.

        Args:
            num_features: Maximum keypoints retained per image.
            upright: If True, disable rotation handling (nadir lunar views).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KeyNetAffNetHardNet(
            num_features=int(num_features), upright=bool(upright), device=self.device
        ).to(self.device).eval()
        logger.info("KorniaAI_Matcher initialised on device: %s", self.device)

    # ------------------------------------------------------------------
    @staticmethod
    def _load_gray(path: str) -> np.ndarray:
        """Load an image as single-channel uint8; raise if unreadable."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if img.size == 0:
            raise ValueError(f"Empty image: {path}")
        return np.ascontiguousarray(img)

    @staticmethod
    def _downscale_if_needed(img: np.ndarray) -> tuple:
        """Downscale images over 1 Mpix; return (img, scale_x, scale_y).

        Scaling math: with factor s = MAX_EDGE / max(H, W) applied to the
        working image, a working-grid keypoint (u, v) maps back via
        x_orig = u / s == u * scale_x, y_orig = v / s == v * scale_y,
        where scale_x = W_orig / W_work and scale_y = H_orig / H_work.
        """
        h, w = img.shape[:2]
        if h * w <= _MAX_PIXELS:
            return img, 1.0, 1.0
        scale = _MAX_EDGE / float(max(h, w))
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_x, scale_y = w / float(new_w), h / float(new_h)
        logger.info("Downscaled %dx%d -> %dx%d (scale_x=%.4f, scale_y=%.4f).",
                    w, h, new_w, new_h, scale_x, scale_y)
        return np.ascontiguousarray(small), scale_x, scale_y

    @torch.no_grad()
    def _extract(self, img_u8: np.ndarray) -> tuple:
        """Run KeyNetAffNetHardNet -> (keypoints[N,2], descs[N,128], lafs)."""
        t = torch.from_numpy(img_u8).to(self.device, dtype=torch.float32) / 255.0
        t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        lafs, _resp, descs = self.model(t)
        # lafs: (1, N, 2, 3); keypoint centre = lafs[..., :2, 2].
        kps = lafs[0, :, :2, 2].detach().cpu().numpy().astype(np.float32)
        des = descs[0].detach().cpu().numpy().astype(np.float32)
        return (kps.reshape(-1, 2), des.reshape(kps.shape[0], -1),
                lafs.detach().to(self.device))

    def _match_descriptors(self, kps1, des1, lafs1, kps2, des2, lafs2, hw1, hw2):
        """AdaLAM first, BFMatcher + Lowe(0.85) fallback. Returns (p1, p2)."""
        empty = (np.zeros((0, 2), dtype=np.float32),) * 2
        if len(kps1) == 0 or len(kps2) == 0:
            return empty
        # Primary: AdaLAM geometric consistency filter on L2-NN matches,
        # using the true AffNet LAFS shapes from the detector.
        try:
            d1 = torch.as_tensor(np.ascontiguousarray(des1),
                                 dtype=torch.float32, device=self.device)
            d2 = torch.as_tensor(np.ascontiguousarray(des2),
                                 dtype=torch.float32, device=self.device)
            idx1, idx2 = None, None
            # NOTE: match_adalam returns (dists[B3,1], idx_pairs[B3,2]),
            # NOT two separate index vectors.
            _dists, idx_pairs = match_adalam(d1, d2, lafs1, lafs2,
                                             hw1=hw1, hw2=hw2)
            idx_pairs = np.asarray(idx_pairs.detach().cpu().numpy(),
                                     dtype=np.intp).reshape(-1, 2)
            # Drop any out-of-range indices before fancy-indexing.
            if len(idx_pairs) > 0:
                idx1, idx2 = idx_pairs[:, 0], idx_pairs[:, 1]
                valid = (idx1 >= 0) & (idx1 < len(kps1)) & \
                        (idx2 >= 0) & (idx2 < len(kps2))
                idx1, idx2 = idx1[valid], idx2[valid]
            else:
                idx1 = np.zeros((0,), dtype=np.intp)
            if idx1.size > 0:
                logger.info("AdaLAM raw matches: %d.", idx1.size)
                return (np.ascontiguousarray(kps1[idx1], dtype=np.float32),
                        np.ascontiguousarray(kps2[idx2], dtype=np.float32))
            logger.info("AdaLAM found 0 matches; trying BFMatcher fallback.")
        except Exception as exc:  # robust: never let AdaLAM kill the pipeline
            logger.warning("AdaLAM failed (%s); trying BFMatcher fallback.", exc)

        # Fallback: brute-force L2 kNN (k=2) + Lowe ratio test at 0.85.
        try:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            pairs = bf.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
            good1, good2 = [], []
            for m_n in pairs:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < _LOWE_RATIO * n.distance:
                    good1.append(kps1[m.queryIdx])
                    good2.append(kps2[m.trainIdx])
            if not good1:
                return empty
            logger.info("BFMatcher+Lowe(%.2f) raw matches: %d.",
                        _LOWE_RATIO, len(good1))
            return (np.asarray(good1, dtype=np.float32),
                    np.asarray(good2, dtype=np.float32))
        except cv2.error as exc:
            logger.warning("BFMatcher fallback failed: %s", exc)
            return empty

    @staticmethod
    def _ransac_filter(p_ref, p_src):
        """Fundamental-matrix RANSAC (Homography fallback). Returns inlier pts."""
        n = len(p_ref)
        empty = (np.zeros((0, 2), dtype=np.float32),) * 2 + (0,)
        if n == 0:
            return empty
        pts_ref = np.asarray(p_ref, dtype=np.float32).reshape(-1, 2)
        pts_src = np.asarray(p_src, dtype=np.float32).reshape(-1, 2)

        mask = None
        if n >= 8:  # minimum for a fundamental matrix
            try:
                _F, mask = cv2.findFundamentalMat(
                    pts_ref, pts_src, cv2.FM_RANSAC, 3.0, 0.99
                )
            except cv2.error as exc:
                # Degenerate geometry (e.g. identical images -> F undefined).
                logger.info("findFundamentalMat failed (%s); trying Homography.", exc)
                mask = None
        if mask is None and n >= 4:  # nadir fallback: planar homography
            try:
                _H, mask = cv2.findHomography(pts_ref, pts_src, cv2.RANSAC, 3.0)
            except cv2.error as exc:
                logger.info("findHomography failed (%s); rejecting all.", exc)
                return empty
        if mask is None:
            logger.info("RANSAC rejected all %d raw matches.", n)
            return empty
        inl = mask.ravel().astype(bool)
        if int(np.count_nonzero(inl)) == 0:
            logger.info("RANSAC found 0 inliers from %d raw matches.", n)
            return empty
        return (np.ascontiguousarray(pts_ref[inl], dtype=np.float32),
                np.ascontiguousarray(pts_src[inl], dtype=np.float32),
                int(np.count_nonzero(inl)))

    # ------------------------------------------------------------------
    def match_images(self, ref_img_path: str, src_img_path: str) -> dict:
        """Match two image files; return inlier correspondences in ORIG pixels.

        Returns:
            On success: ``{"status": "success", "ref_pts": (M,2) float32,
            "src_pts": (M,2) float32, "inliers": M}``.
            On failure: ``{"status": "failed", "reason": str, ...}`` with
            empty (0,2) float32 point arrays — never raises on empty matches.
        """
        try:
            ref_full = self._load_gray(ref_img_path)
            src_full = self._load_gray(src_img_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("match_images: %s", exc)
            return {"status": "failed", "reason": str(exc),
                    "ref_pts": np.zeros((0, 2), dtype=np.float32),
                    "src_pts": np.zeros((0, 2), dtype=np.float32),
                    "inliers": 0}

        ref_work, rx, ry = self._downscale_if_needed(ref_full)
        src_work, sx, sy = self._downscale_if_needed(src_full)

        kps_ref, des_ref, lafs_ref = self._extract(ref_work)
        kps_src, des_src, lafs_src = self._extract(src_work)
        logger.info("KeyNet features: ref=%d src=%d.", len(kps_ref), len(kps_src))
        if len(kps_ref) == 0 or len(kps_src) == 0:
            return {"status": "failed", "reason": "no_features_detected",
                    "ref_pts": np.zeros((0, 2), dtype=np.float32),
                    "src_pts": np.zeros((0, 2), dtype=np.float32),
                    "inliers": 0}

        m_ref, m_src = self._match_descriptors(
            kps_ref, des_ref, lafs_ref, kps_src, des_src, lafs_src,
            hw1=ref_work.shape[:2], hw2=src_work.shape[:2],
        )
        logger.info("Raw putative matches: %d.", len(m_ref))
        if len(m_ref) == 0:
            return {"status": "failed", "reason": "no_raw_matches",
                    "ref_pts": np.zeros((0, 2), dtype=np.float32),
                    "src_pts": np.zeros((0, 2), dtype=np.float32),
                    "inliers": 0}

        # Map working-grid coords back to ORIGINAL resolution BEFORE RANSAC
        # so reprojection thresholds are in true pixel units:
        #   x_orig = x_work * (W_orig / W_work).
        m_ref[:, 0] *= rx
        m_ref[:, 1] *= ry
        m_src[:, 0] *= sx
        m_src[:, 1] *= sy

        f_ref, f_src, n_inl = self._ransac_filter(m_ref, m_src)
        logger.info("RANSAC inliers: %d/%d.", n_inl, len(m_ref))
        if n_inl == 0:
            return {"status": "failed", "reason": "ransac_no_inliers",
                    "ref_pts": np.zeros((0, 2), dtype=np.float32),
                    "src_pts": np.zeros((0, 2), dtype=np.float32),
                    "inliers": 0}

        return {"status": "success",
                "ref_pts": np.ascontiguousarray(f_ref, dtype=np.float32),
                "src_pts": np.ascontiguousarray(f_src, dtype=np.float32),
                "inliers": int(n_inl)}
