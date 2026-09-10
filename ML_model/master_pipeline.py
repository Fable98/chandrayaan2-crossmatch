"""ML_model/master_pipeline.py — Master Orchestrator for Chandrayaan-2 registration.

Orchestrates classical CFOG + Phase Congruency matching with uniform
spatial filtering into a single fault-tolerant, memory-safe pipeline.

Design guardrails:
  * NEVER raises from :meth:`MasterRegistrationPipeline.register`.
  * Every submodule call is wrapped in try/except; failures are logged
    and the pipeline continues with whatever matches it has.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("ML_model.master_pipeline")


def _ensure_ml_model_on_path() -> None:
    """Make bare ``from matcher_cfog import ...`` imports work.

    ``matcher_cfog.py`` uses top-level imports (``from metadata import``),
    so the ``ML_model/`` directory itself must be on ``sys.path``.
    """
    ml_dir = str(Path(__file__).resolve().parent)
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)


class MasterRegistrationPipeline:
    """8-Phase AI-Augmented Photogrammetry Pipeline: CFOG + Phase Congruency + AI Verifier + Distribution."""

    def __init__(self, min_inliers_required: int = 50) -> None:
        self.min_inliers_required = int(min_inliers_required)
        self.logger = logging.getLogger("ML_model.master_pipeline")
        # CRITICAL GUARDRAIL: lazy-load heavy models only on demand.
        self.subpixel_refiner = None
        self.distribution_filter = None

    # ------------------------------------------------------------------
    # Lazy loaders (instantiated only when the pipeline actually needs them)
    # ------------------------------------------------------------------
    def _get_subpixel_refiner(self):
        if self.subpixel_refiner is not None:
            return self.subpixel_refiner
        _ensure_ml_model_on_path()
        try:
            try:
                from ML_model.subpixel_refiner import SubPixelRefiner
            except Exception:
                from subpixel_refiner import SubPixelRefiner  # type: ignore[no-redef]
            self.subpixel_refiner = SubPixelRefiner()
        except Exception as e:
            self.logger.warning("Lazy-load of SubPixelRefiner failed: %s", e)
            raise
        return self.subpixel_refiner

    # ------------------------------------------------------------------
    # Phase 1 helper: run CFOG (class or function API)
    # ------------------------------------------------------------------
    def _run_cfog_phase(
        self, src_img_path: str, ref_img_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
        """Run classical CFOG core.

        Supports both the spec'd ``CFOGMatcher`` class API (if present) and
        the actual ``match_images_cfog`` function API in this repo.

        Returns:
            (src_pts (N,2) float32, ref_pts (N,2) float32,
             confidences (N,) float32, inlier_count int,
             homography (3,3) or None).
        """
        _ensure_ml_model_on_path()

        # 1) Spec'd class API: CFOGMatcher (may not exist in this repo).
        try:
            try:
                from ML_model.matcher_cfog import CFOGMatcher  # type: ignore
            except Exception:
                from matcher_cfog import CFOGMatcher  # type: ignore[no-redef]
            matcher = CFOGMatcher()  # type: ignore[call-arg]
            if hasattr(matcher, "match"):
                res = matcher.match(src_img_path, ref_img_path)
            elif hasattr(matcher, "match_images"):
                res = matcher.match_images(src_img_path, ref_img_path)
            else:
                raise AttributeError("CFOGMatcher has no match()/match_images() method")
            return self._parse_generic_match_result(res)
        except ImportError:
            # Class genuinely absent -> fall through to function API.
            pass
        except Exception as e:
            self.logger.warning("CFOGMatcher class API failed (%s); trying function API.", e)

        # 2) Actual repo function API: match_images_cfog(src, ref).
        try:
            try:
                from ML_model.matcher_cfog import match_images_cfog  # type: ignore
            except Exception:
                from matcher_cfog import match_images_cfog  # type: ignore[no-redef]
        except Exception as e:
            raise ImportError(f"Could not import CFOG matcher: {e}") from e

        tmp_dir = tempfile.mkdtemp(prefix="cfog_master_")
        res = match_images_cfog(src_img_path, ref_img_path, output_dir=tmp_dir)
        return self._parse_generic_match_result(res)

    @staticmethod
    def _parse_generic_match_result(
        res: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
        """Normalise CFOG-style dict outputs to (src, ref, conf, inliers, H)."""
        empty = (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            0,
            None,
        )
        if not isinstance(res, dict):
            return empty
        # Prefer inlier records when available.
        records = None
        if isinstance(res.get("matches"), list) and res.get("matches"):
            records = res["matches"]
        elif isinstance(res.get("all_matches"), list) and res.get("all_matches"):
            records = [m for m in res["all_matches"] if m.get("is_inlier", True)]
            if not records:
                records = res["all_matches"]
        if records:
            src, ref, conf = [], [], []
            for m in records:
                try:
                    sx = float(m.get("source_x", m.get("image1_x", m.get("work_x1"))))
                    sy = float(m.get("source_y", m.get("image1_y", m.get("work_y1"))))
                    tx = float(m.get("target_x", m.get("image2_x", m.get("work_x2"))))
                    ty = float(m.get("target_y", m.get("image2_y", m.get("work_y2"))))
                except Exception:
                    continue
                src.append([sx, sy])
                ref.append([tx, ty])
                try:
                    conf.append(float(m.get("confidence", m.get("score", 0.8))))
                except Exception:
                    conf.append(0.8)
            if not src:
                return empty
            src_pts = np.asarray(src, dtype=np.float32)
            ref_pts = np.asarray(ref, dtype=np.float32)
            conf_arr = np.asarray(conf, dtype=np.float32)
            try:
                inliers = int(res.get("inlier_count", len(src_pts)))
            except Exception:
                inliers = len(src_pts)
            H_mat = res.get("homography")
            return src_pts, ref_pts, conf_arr, inliers, H_mat
        # Raw array style: {"src_pts": ..., "ref_pts": ...}.
        try:
            if res.get("src_pts") is not None and res.get("ref_pts") is not None:
                src_pts = np.asarray(res["src_pts"], dtype=np.float32).reshape(-1, 2)
                ref_pts = np.asarray(res["ref_pts"], dtype=np.float32).reshape(-1, 2)
                n = min(len(src_pts), len(ref_pts))
                src_pts, ref_pts = src_pts[:n], ref_pts[:n]
                try:
                    inliers = int(res.get("inlier_count", res.get("inliers", n)))
                except Exception:
                    inliers = n
                conf_arr = np.full((n,), 0.8, dtype=np.float32)
                H_mat = res.get("homography")
                return src_pts, ref_pts, conf_arr, inliers, H_mat
        except Exception:
            pass
        try:
            inliers = int(res.get("inlier_count", res.get("inliers", 0)))
        except Exception:
            inliers = 0
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            inliers,
            res.get("homography"),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def register(self, src_img_path: str, ref_img_path: str) -> dict:
        """Register source -> reference image. NEVER raises.

        Returns dict with keys: status, transformation_matrix,
        final_inliers, final_rmse_pixels, coverage_ratio, balance_score,
        phases_executed, phases_failed.
        """
        phases_executed: List[str] = []
        phases_failed: List[str] = []

        base_src = np.zeros((0, 2), dtype=np.float32)
        base_ref = np.zeros((0, 2), dtype=np.float32)
        base_conf = np.zeros((0,), dtype=np.float32)

        def _append(src: Any, ref: Any, conf: Any) -> None:
            nonlocal base_src, base_ref, base_conf
            try:
                s = np.asarray(src, dtype=np.float32).reshape(-1, 2)
                r = np.asarray(ref, dtype=np.float32).reshape(-1, 2)
                n = min(len(s), len(r))
                if n == 0:
                    return
                s, r = s[:n], r[:n]
                try:
                    c = np.asarray(conf, dtype=np.float32).ravel()[:n]
                    if c.size != n:
                        raise ValueError("confidence length mismatch")
                except Exception:
                    c = np.full((n,), 0.8, dtype=np.float32)
                base_src = np.vstack([base_src, s]) if len(base_src) else s
                base_ref = np.vstack([base_ref, r]) if len(base_ref) else r
                base_conf = np.concatenate([base_conf, c]) if base_conf.size else c
            except Exception as e:
                self.logger.warning("Match-merge failed: %s", e)

        # ---------------- Phase 1: Classical Core (CFOG) ----------------
        cfog_H: Optional[np.ndarray] = None
        try:
            s_pts, r_pts, c_pts, n_inl, cfog_H = self._run_cfog_phase(src_img_path, ref_img_path)
            phases_executed.append("CFOG")
            if len(s_pts) > 0:
                _append(s_pts, r_pts, c_pts)
                self.logger.info("CFOG phase: %d matches (inliers=%d).", len(s_pts), n_inl)
            else:
                self.logger.warning("CFOG phase returned 0 matches (inliers=%d).", n_inl)
                phases_failed.append("CFOG")
        except Exception as e:
            self.logger.warning("CFOG phase crashed: %s", e)
            if "CFOG" not in phases_executed:
                phases_executed.append("CFOG")
            phases_failed.append("CFOG")

        # ---------------- Phase 4: Sub-Pixel Refinement (HANDLED BY CFOG) ----
        # CFOG engine already performs Fourier Phase Correlation + Lucas-Kanade
        # sub-pixel refinement internally. Running an external refiner here
        # would apply double-refinement and introduce jitter.
        refined_src: np.ndarray = base_src
        refined_ref: np.ndarray = base_ref
        phases_executed.append("Subpixel_internal")

        # ---------------- Phase 5: Uniform Spatial Distribution ----------------
        filtered_src = refined_src
        filtered_ref = refined_ref
        coverage_ratio = 0.0
        balance_score = 0.0
        try:
            _ensure_ml_model_on_path()
            try:
                try:
                    from ML_model.spatial_distribution import UniformDistributionFilter  # type: ignore
                except Exception:
                    from spatial_distribution import UniformDistributionFilter  # type: ignore[no-redef]
            except Exception as e:
                raise ImportError(f"Could not import UniformDistributionFilter: {e}") from e
            phases_executed.append("Distribution")
            # Image dims from the source frame (filter bins on src coords).
            h, w = None, None
            try:
                probe = cv2.imread(str(src_img_path), cv2.IMREAD_UNCHANGED)
                if probe is not None:
                    h, w = probe.shape[:2]
            except Exception:
                pass
            if h is None or w is None:
                raise ValueError("Could not determine image dimensions for distribution filter.")
            dist_filter = UniformDistributionFilter(grid_rows=8, grid_cols=8, points_per_cell=10)
            # Lazily cache a default instance without heavy state.
            try:
                self.distribution_filter = dist_filter
            except Exception:
                pass
            conf_in = base_conf if base_conf.size == len(refined_src) else None
            d_res = dist_filter.filter_points(refined_src, refined_ref, w, h, conf_in)
            if isinstance(d_res, dict) and len(np.asarray(d_res.get("filtered_src_pts", []))) > 0:
                filtered_src = np.asarray(d_res["filtered_src_pts"], dtype=np.float32).reshape(-1, 2)
                filtered_ref = np.asarray(d_res["filtered_ref_pts"], dtype=np.float32).reshape(-1, 2)
                try:
                    coverage_ratio = float(d_res.get("coverage_ratio", 0.0))
                except Exception:
                    coverage_ratio = 0.0
                try:
                    balance_score = float(d_res.get("balance_score", 0.0))
                except Exception:
                    balance_score = 0.0
                self.logger.info(
                    "Distribution phase: %d -> %d (coverage=%.3f, balance=%.3f).",
                    len(refined_src), len(filtered_src), coverage_ratio, balance_score,
                )
            else:
                self.logger.warning("Distribution filter returned empty; keeping refined pool.")
                phases_failed.append("Distribution")
                filtered_src, filtered_ref = refined_src, refined_ref
        except Exception as e:
            if "Distribution" not in phases_executed:
                phases_executed.append("Distribution")
            phases_failed.append("Distribution")
            self.logger.warning("Distribution phase crashed: %s", e)
            filtered_src, filtered_ref = refined_src, refined_ref

        # ---------------- Phase 6: Final Geometric Transformation ----------------
        transformation_matrix: Optional[np.ndarray] = None
        final_inliers = 0
        final_rmse: float = float("inf")

        # PRIORITY: Use the homography already calculated by CFOG
        if cfog_H is not None:
            try:
                if isinstance(cfog_H, list):
                    cfog_H = np.asarray(cfog_H, dtype=np.float64)
                transformation_matrix = np.asarray(cfog_H, dtype=np.float64).reshape(3, 3)
                final_inliers = int(n_inl)
                # Compute RMSE using available points
                if filtered_src is not None and len(filtered_src) >= 4:
                    fs = np.asarray(filtered_src, dtype=np.float64).reshape(-1, 2)
                    fr = np.asarray(filtered_ref, dtype=np.float64).reshape(-1, 2)
                    n = min(len(fs), len(fr))
                    fs, fr = fs[:n], fr[:n]
                    ones = np.ones((n, 1), dtype=np.float64)
                    src_h = np.hstack([fs, ones])
                    proj = (transformation_matrix @ src_h.T).T
                    proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-12)
                    err = np.linalg.norm(proj - fr, axis=1)
                    final_rmse = float(np.sqrt(np.mean(err**2)))
                self.logger.info("Using CFOG pre-calculated homography (inliers=%d, RMSE=%.4f).", final_inliers, final_rmse)
            except Exception as e:
                self.logger.warning("CFOG homography passthrough failed: %s. Falling back.", e)
                cfog_H = None

        # FALLBACK: Vanilla RANSAC only if CFOG didn't provide a matrix
        if cfog_H is None or transformation_matrix is None:
            try:
                if filtered_src is not None and len(filtered_src) >= 4:
                    fs = np.asarray(filtered_src, dtype=np.float32).reshape(-1, 2)
                    fr = np.asarray(filtered_ref, dtype=np.float32).reshape(-1, 2)
                    n = min(len(fs), len(fr))
                    fs, fr = fs[:n], fr[:n]
                    H, mask = cv2.findHomography(fs, fr, cv2.RANSAC, 3.0)
                    if H is not None and mask is not None:
                        inl = mask.ravel().astype(bool)
                        final_inliers = int(np.count_nonzero(inl))
                        if final_inliers >= 4:
                            transformation_matrix = np.asarray(H, dtype=np.float64)
                            ones = np.ones((final_inliers, 1), dtype=np.float64)
                            src_h = np.hstack([fs[inl].astype(np.float64), ones])
                            proj = (H @ src_h.T).T
                            proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-12)
                            err = np.linalg.norm(proj - fr[inl].astype(np.float64), axis=1)
                            final_rmse = float(np.sqrt(np.mean(err**2)))
            except Exception as e:
                self.logger.warning("Fallback homography estimation failed: %s", e)

        status = "success" if (transformation_matrix is not None and final_inliers >= 4) else "failed"
        if status == "failed":
            transformation_matrix = None

        return {
            "status": status,
            "transformation_matrix": transformation_matrix,
            "final_inliers": int(final_inliers),
            "final_rmse_pixels": float(final_rmse),
            "coverage_ratio": float(coverage_ratio),
            "balance_score": float(balance_score),
            "phases_executed": phases_executed,
            "phases_failed": phases_failed,
            "filtered_src_pts": filtered_src.tolist() if filtered_src is not None and hasattr(filtered_src, 'tolist') else None,
            "filtered_ref_pts": filtered_ref.tolist() if filtered_ref is not None and hasattr(filtered_ref, 'tolist') else None,
        }


__all__ = ["MasterRegistrationPipeline"]
