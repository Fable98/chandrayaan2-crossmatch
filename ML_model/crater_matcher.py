"""
ML_model/crater_matcher.py — Auxiliary Crater-Based Matcher for Chandrayaan-2.

Photogrammetric role:
    Classical CFOG / Phase-Congruency matching can fail under extreme
    illumination variation or deep shadows where edge energy is lost.
    Lunar impact craters are quasi-circular, sun-angle-resilient structures
    whose centers are stable, approximately scale- and rotation-invariant
    anchor points. This module detects craters with YOLO and converts
    matched crater centers into tie points for a downstream RANSAC
    geometric (homography / affine) estimation stage.

Pipeline:
    1. ``CraterAnchorMatcher.detect_craters`` — YOLO inference with a
       1024-px memory guardrail and rescaling back to native coordinates.
    2. ``CraterAnchorMatcher.match_craters`` — scale/rotation-invariant
       geometric hashing via normalized pairwise distances + Lowe ratio test.
    3. ``CraterAnchorMatcher.get_anchor_points`` — detection + matching +
       median radius-ratio scale-consistency gate.
    4. ``run_crater_fallback`` — drop-in fallback for ``matcher_cfog.py``.

Dependencies: ultralytics (YOLOv8), cv2, numpy, scipy.spatial.distance, logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger("ML_model.crater_matcher")

ImageInput = Union[str, Path, np.ndarray]

_CUSTOM_WEIGHTS_CANDIDATES = (
    Path("models/crater_detector.pt"),
    Path("ML_model/models/crater_detector.pt"),
    Path(__file__).resolve().parent / "models" / "crater_detector.pt",
)

_FALLBACK_WEIGHTS = ("yolov8n.pt", "yolov8s.pt")


class CraterAnchorMatcher:
    """YOLO-based lunar crater detector + geometric anchor matcher.

    Args:
        model_path: Path to a custom-trained lunar crater ``.pt`` file.
        fallback_model: Generic ultralytics checkpoint used when the custom
            weights are missing (e.g. ``"yolov8n.pt"``).
        conf_thresh: Minimum detection confidence kept (default 0.35).
        max_dim: Largest image side (px) fed to YOLO; larger inputs are
            downscaled to bound VRAM (default 1024).
        k_neighbors: Size ``K`` of the local geometric descriptor — the
            number of nearest-neighbour normalized distances stored per
            crater (default 3).
        ratio_thresh: Lowe-style ratio threshold on descriptor distances
            (default 0.8). A candidate pair is accepted only when
            ``best / second_best < ratio_thresh``.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/crater_detector.pt",
        fallback_model: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        max_dim: int = 1024,
        k_neighbors: int = 3,
        ratio_thresh: float = 0.8,
    ) -> None:
        self.conf_thresh = float(conf_thresh)
        self.max_dim = int(max_dim)
        self.k_neighbors = int(k_neighbors)
        self.ratio_thresh = float(ratio_thresh)
        self.model_path = str(model_path)
        self.model = None
        self.model_name: str = "unloaded"
        self._load_model(fallback_model)

    # ------------------------------------------------------------------
    # Model loading (with critical fallback)
    # ------------------------------------------------------------------
    def _load_model(self, fallback_model: str) -> None:
        """Load custom crater weights, falling back to a stock YOLOv8 model.

        Never raises: if even the fallback cannot load (e.g. ``ultralytics``
        missing or offline with no cached weights), ``self.model`` stays
        ``None`` and detection calls return empty arrays gracefully.
        """
        try:
            from ultralytics import YOLO
        except Exception as exc:  # ultralytics not installed
            logger.warning("ultralytics not available (%s). Crater fallback disabled.", exc)
            return

        # 1) Try explicit path + known repo-relative candidate locations.
        candidates = [Path(self.model_path), *_CUSTOM_WEIGHTS_CANDIDATES]
        for cand in candidates:
            try:
                if cand.is_file():
                    self.model = YOLO(str(cand))
                    self.model_name = str(cand)
                    logger.info("Loaded custom crater YOLO model: %s", cand)
                    return
            except Exception as exc:
                logger.warning("Failed to load custom weights '%s': %s", cand, exc)

        logger.warning(
            "Custom crater model '%s' not found. Falling back to generic '%s' shape detector.",
            self.model_path,
            fallback_model,
        )

        # 2) Generic fallback chain: requested fallback, then known stock nets.
        for weights in (fallback_model, *_FALLBACK_WEIGHTS):
            try:
                self.model = YOLO(weights)
                self.model_name = str(weights)
                logger.info("Loaded fallback YOLO model: %s", weights)
                return
            except Exception as exc:
                logger.warning("Fallback YOLO weights '%s' failed: %s", weights, exc)

        logger.error("No YOLO weights could be loaded. Crater detection will return empty.")
        self.model = None

    # ------------------------------------------------------------------
    # Step 1: detection
    # ------------------------------------------------------------------
    def _load_image_bgr(self, image: ImageInput) -> Union[np.ndarray, None]:
        """Read ``image`` (path or ndarray) as a BGR uint8 array or None."""
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:  # grayscale -> BGR for YOLO
                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = cv2.cvtColor(arr[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2BGR)
            return arr
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:  # maybe single-channel file; retry unchanged then convert
            gray = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            if gray is None:
                return None
            if gray.ndim == 2:
                img = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                return None
        return img

    def detect_craters(self, image_path: ImageInput) -> np.ndarray:
        """Detect craters and return ``(N, 3)`` array of ``[cx, cy, r]``.

        - Reads with ``cv2.imread`` (grayscale converted to BGR for YOLO).
        - Downscales so ``max(h, w) <= max_dim`` (default 1024 px) before
          inference to prevent VRAM OOM, then rescales boxes back to the
          original pixel grid.
        - Keeps boxes with ``confidence >= conf_thresh`` (default 0.35).
        - Center ``(cx, cy)`` is the box midpoint; radius ``r`` is the mean
          of half-width and half-height: ``r = (w + h) / 4``.

        Returns:
            ``np.ndarray`` of shape ``(N, 3)`` dtype ``float32``. Returns an
            empty ``(0, 3)`` array when the image is unreadable, the model
            is unavailable, or nothing is detected (smooth highlands).
        """
        empty = np.empty((0, 3), dtype=np.float32)

        img_bgr = self._load_image_bgr(image_path)
        if img_bgr is None:
            logger.warning("detect_craters: could not read image '%s'.", image_path)
            return empty
        if self.model is None:
            logger.warning("detect_craters: no YOLO model loaded; returning empty.")
            return empty

        orig_h, orig_w = img_bgr.shape[:2]

        # Memory guardrail: uniform downscale so largest side == max_dim.
        scale = 1.0
        infer_img = img_bgr
        if max(orig_h, orig_w) > self.max_dim:
            scale = self.max_dim / float(max(orig_h, orig_w))
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            infer_img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug("Resized %dx%d -> %dx%d (scale %.4f) for inference.", orig_w, orig_h, new_w, new_h, scale)

        try:
            results = self.model.predict(infer_img, verbose=False, conf=self.conf_thresh)
        except Exception as exc:
            logger.warning("YOLO inference failed on '%s': %s", image_path, exc)
            return empty

        if not results:
            logger.info("detect_craters: 0 craters detected in '%s'.", image_path)
            return empty

        try:
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                logger.info("detect_craters: 0 craters detected in '%s'.", image_path)
                return empty
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)  # (M, 4) in inference px
            conf = boxes.conf.cpu().numpy().astype(np.float64).ravel()
        except Exception as exc:
            logger.warning("Failed to parse YOLO boxes for '%s': %s", image_path, exc)
            return empty

        # Explicit confidence gate (also enforced via predict(conf=...)).
        keep = conf >= self.conf_thresh
        xyxy, conf = xyxy[keep], conf[keep]
        if len(xyxy) == 0:
            logger.info("detect_craters: 0 craters above conf %.2f in '%s'.", self.conf_thresh, image_path)
            return empty

        # Box -> (cx, cy, r) in inference pixels, then rescale to native grid:
        #   cx = (x1+x2)/2 / scale,  cy = (y1+y2)/2 / scale,
        #   r  = ((x2-x1) + (y2-y1)) / 4 / scale.
        inv_scale = 1.0 / scale if scale != 0 else 1.0
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5 * inv_scale
        cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5 * inv_scale
        r = (w + h) * 0.25 * inv_scale
        valid = r > 0
        out = np.stack([cx[valid], cy[valid], r[valid]], axis=1).astype(np.float32)

        logger.info("detect_craters: %d craters detected in '%s' (model=%s).", len(out), image_path, self.model_name)
        return out.reshape(-1, 3)

    # ------------------------------------------------------------------
    # Step 2: scale & rotation invariant geometric matching
    # ------------------------------------------------------------------
    @staticmethod
    def _geometric_descriptors(craters: np.ndarray, k: int) -> np.ndarray:
        """Build ``(N, K)`` scale/rotation-invariant descriptors.

        For crater ``i`` with center ``c_i`` and radius ``r_i``:

        .. math::
            d_{ij} = \\frac{\\lVert c_i - c_j \\rVert_2}{(r_i + r_j) / 2}

        Dividing the center distance by the mean radius cancels an unknown
        global image scale ``s`` (both numerator and denominator scale by
        ``s``). Sorting the ``K`` smallest ``d_{ij}`` values makes the
        descriptor invariant to in-plane rotation (pairwise distances are
        preserved under rotation) and to crater ordering.
        """
        n = len(craters)
        if n == 0:
            return np.empty((0, k), dtype=np.float64)
        k_eff = max(0, min(k, n - 1))
        if k_eff == 0:
            return np.zeros((n, 0), dtype=np.float64)

        centers = craters[:, :2].astype(np.float64)  # (N, 2)
        radii = craters[:, 2].astype(np.float64)  # (N,)
        # Pairwise center distances (N, N); rotation invariant by construction.
        dist = cdist(centers, centers, metric="euclidean")
        # Mean-radius normalizer (N, N); scale cancels: (s*d)/(s*r) = d/r.
        mean_r = (radii[:, None] + radii[None, :]) * 0.5
        mean_r[mean_r <= 1e-9] = 1e-9
        normed = dist / mean_r
        np.fill_diagonal(normed, np.inf)  # exclude self-distance
        # K nearest (smallest normalized) neighbours, sorted -> order invariant.
        part = np.partition(normed, k_eff - 1, axis=1)[:, :k_eff]
        desc = np.sort(part, axis=1)
        return desc

    def match_craters(
        self,
        craters_A: np.ndarray,
        craters_B: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Match crater sets across images via geometric hashing.

        Each crater is described by its ``K`` smallest radius-normalized
        neighbour distances (see :meth:`_geometric_descriptors`). Matching
        uses Lowe-style ratio logic in descriptor space: for each ``a`` in
        A, let ``d1 <= d2`` be the two smallest Euclidean distances to
        descriptors in B; accept ``(a, b1)`` iff ``d1 / d2 < ratio_thresh``.
        A mutual (B -> A) consistency check removes one-way collisions.

        Args:
            craters_A: ``(NA, 3)`` array ``[cx, cy, r]`` from source image.
            craters_B: ``(NB, 3)`` array ``[cx, cy, r]`` from reference image.

        Returns:
            List of matched index tuples ``[(idx_A, idx_B), ...]``. Empty
            when either set has fewer than 2 craters or no pair passes the
            ratio test. Never raises on empty input.
        """
        craters_A = np.asarray(craters_A, dtype=np.float64).reshape(-1, 3)
        craters_B = np.asarray(craters_B, dtype=np.float64).reshape(-1, 3)
        na, nb = len(craters_A), len(craters_B)
        if na < 2 or nb < 2:
            logger.info("match_craters: too few craters (A=%d, B=%d); need >= 2 each.", na, nb)
            return []

        k_eff = min(self.k_neighbors, na - 1, nb - 1)
        if k_eff <= 0:
            return []

        desc_A = self._geometric_descriptors(craters_A, k_eff)  # (NA, K)
        desc_B = self._geometric_descriptors(craters_B, k_eff)  # (NB, K)

        # Descriptor-space distances (NA, NB).
        dmat = cdist(desc_A, desc_B, metric="euclidean")
        matches: List[Tuple[int, int]] = []

        # Forward pass A -> B with Lowe ratio test.
        fwd: Dict[int, Tuple[int, float]] = {}
        order_b = np.argsort(dmat, axis=1)  # (NA, NB) best-first B indices
        for i in range(na):
            b1 = int(order_b[i, 0])
            d1 = float(dmat[i, b1])
            d2 = float(dmat[i, order_b[i, 1]]) if nb > 1 else float("inf")
            # Lowe logic: accept only if best is *distinctly* better than runner-up.
            # ratio = d1 / (d2 + eps); require ratio < ratio_thresh (default 0.8).
            ratio = d1 / (d2 + 1e-12)
            if ratio < self.ratio_thresh:
                fwd[i] = (b1, ratio)

        if not fwd:
            logger.info("match_craters: 0 pairs passed ratio test (A=%d, B=%d).", na, nb)
            return []

        # Reverse pass B -> A for mutual-consistency (suppresses collisions
        # where two A craters claim the same B crater).
        order_a = np.argsort(dmat, axis=0)  # (NA, NB) best-first A indices
        for i, (b1, ratio) in fwd.items():
            a_best_for_b = int(order_a[0, b1])
            if a_best_for_b == i:
                matches.append((i, b1))
            else:
                # Keep the mutually-best claim only: compare forward ratios.
                other_ratio = None
                if a_best_for_b in fwd and fwd[a_best_for_b][0] == b1:
                    other_ratio = fwd[a_best_for_b][1]
                if other_ratio is None or ratio < other_ratio:
                    # Replace: current claim is stronger or the rival points elsewhere.
                    matches = [(a, b) for (a, b) in matches if b != b1]
                    matches.append((i, b1))

        # De-duplicate defensively and sort for determinism.
        matches = sorted(set(matches))
        logger.info("match_craters: %d geometric matches (A=%d, B=%d, K=%d).", len(matches), na, nb, k_eff)
        return matches

    # ------------------------------------------------------------------
    # Step 3: anchor extraction + scale-consistency gate
    # ------------------------------------------------------------------
    def get_anchor_points(
        self,
        image_path_A: ImageInput,
        image_path_B: ImageInput,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect, match, and scale-gate crater centers into tie points.

        Scale-consistency check: for each tentative match, the radius ratio
        ``s_i = r_A / r_B`` estimates the inter-image scale. True matches
        share one global scale, so matches with
        ``|s_i - median(s)| / median(s) > 0.20`` are discarded as false.

        Returns:
            ``(src_pts, ref_pts)`` — ``(N, 2)`` float32 arrays of matched
            ``(x, y)`` centers in native pixels. Both are ``(0, 2)`` when
            detection or matching yields nothing usable.
        """
        empty = (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32))

        craters_A = self.detect_craters(image_path_A)
        craters_B = self.detect_craters(image_path_B)
        logger.info("Anchor extraction: %d craters (source) vs %d (reference).", len(craters_A), len(craters_B))

        if len(craters_A) == 0 or len(craters_B) == 0:
            logger.warning("get_anchor_points: empty detection set; no anchors.")
            return empty

        pairs = self.match_craters(craters_A, craters_B)
        if not pairs:
            logger.warning("get_anchor_points: no geometric crater matches.")
            return empty

        # Global scale estimate = median of per-match radius ratios.
        ratios = np.array(
            [float(craters_A[i, 2] / max(craters_B[j, 2], 1e-9)) for i, j in pairs],
            dtype=np.float64,
        )
        median_scale = float(np.median(ratios)) if len(ratios) else 1.0
        logger.info("Estimated global scale factor (median r_A/r_B): %.4f from %d raw matches.", median_scale, len(pairs))

        denom = abs(median_scale) if abs(median_scale) > 1e-9 else 1.0
        kept: List[Tuple[int, int]] = [
            (i, j) for (i, j), s in zip(pairs, ratios) if abs(float(s) - median_scale) / denom <= 0.20
        ]
        logger.info("Scale-consistency gate (±20%%): kept %d/%d matches.", len(kept), len(pairs))

        if not kept:
            return empty

        src_pts = np.array([[float(craters_A[i, 0]), float(craters_A[i, 1])] for i, _ in kept], dtype=np.float32)
        ref_pts = np.array([[float(craters_B[j, 0]), float(craters_B[j, 1])] for _, j in kept], dtype=np.float32)
        logger.info("Final anchor points: %d.", len(kept))
        return src_pts, ref_pts


# ----------------------------------------------------------------------
# Step 4: standalone fallback entry point
# ----------------------------------------------------------------------
def run_crater_fallback(
    src_img: ImageInput,
    ref_img: ImageInput,
    **matcher_kwargs: Any,
) -> Dict[str, Any]:
    """Run the crater-based fallback matcher.

    Args:
        src_img: Source image path (or ndarray).
        ref_img: Reference image path (or ndarray).
        **matcher_kwargs: Forwarded to :class:`CraterAnchorMatcher`.

    Returns:
        - On failure (``< 4`` anchors, incl. empty detections):
          ``{"status": "failed", "reason": "insufficient_craters", ...}``.
        - On success:
          ``{"status": "success", "src_pts": (N,2), "ref_pts": (N,2),
          "inlier_count": N, ...}`` where the points are the
          scale-consistent crater centers ready for RANSAC.
    """
    matcher = CraterAnchorMatcher(**matcher_kwargs)
    try:
        src_pts, ref_pts = matcher.get_anchor_points(src_img, ref_img)
    except Exception as exc:  # never let the fallback crash the pipeline
        logger.warning("run_crater_fallback raised: %s", exc)
        return {"status": "failed", "reason": "insufficient_craters", "inlier_count": 0, "error": str(exc)}

    n = int(len(src_pts)) if src_pts is not None else 0
    if n < 4 or ref_pts is None or len(ref_pts) < 4:
        logger.warning("run_crater_fallback: insufficient craters (%d anchors).", n)
        return {
            "status": "failed",
            "reason": "insufficient_craters",
            "inlier_count": int(n),
            "src_pts": np.empty((0, 2), dtype=np.float32),
            "ref_pts": np.empty((0, 2), dtype=np.float32),
        }

    logger.info("run_crater_fallback: success with %d anchors.", n)
    return {
        "status": "success",
        "src_pts": np.asarray(src_pts, dtype=np.float32),
        "ref_pts": np.asarray(ref_pts, dtype=np.float32),
        "inlier_count": int(n),
    }


__all__ = ["CraterAnchorMatcher", "run_crater_fallback"]
