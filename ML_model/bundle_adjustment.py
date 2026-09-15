"""ML_model/bundle_adjustment.py — Global Bundle Adjustment for Chandrayaan-2.

Takes pairwise registration results (transformation matrices plus matched
points) and jointly optimizes all image poses against a common reference
frame to minimize total reprojection error.

Design guardrails:
    * Only numpy / scipy.optimize / scipy.sparse / cv2 / logging.
    * NEVER raises from public methods on degenerate input.
    * All math uses np.float64 for numerical stability.
    * The reference image frame is FIXED at identity and never optimized.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List

import cv2  # noqa: F401  (required dependency)
import numpy as np
from scipy import sparse  # noqa: F401  (required dependency)
from scipy.optimize import least_squares

logger = logging.getLogger("ML_model.bundle_adjustment")


def _to_affine_2x3(matrix: np.ndarray) -> np.ndarray:
    """Convert a 2x3 affine or 3x3 homography to 2x3 affine (float64)."""
    try:
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape == (2, 3):
            return np.ascontiguousarray(m, dtype=np.float64)
        if m.shape == (3, 3):
            denom = float(m[2, 2]) if abs(float(m[2, 2])) > 1e-12 else 1.0
            return np.ascontiguousarray(m[:2, :] / denom, dtype=np.float64)
        flat = np.asarray(m, dtype=np.float64).ravel()
        if flat.size == 6:
            return np.ascontiguousarray(flat.reshape(2, 3), dtype=np.float64)
        if flat.size == 9:
            tmp = flat.reshape(3, 3)
            denom = float(tmp[2, 2]) if abs(float(tmp[2, 2])) > 1e-12 else 1.0
            return np.ascontiguousarray(tmp[:2, :] / denom, dtype=np.float64)
    except Exception:
        pass
    return np.eye(2, 3, dtype=np.float64)


def _apply_affine(pts: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to (N, 2) points. All float64."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    m = np.asarray(mat, dtype=np.float64).reshape(2, 3)
    ones = np.ones((p.shape[0], 1), dtype=np.float64)
    hom = np.hstack([p, ones])  # (N, 3) homogeneous coordinates, float64
    return np.asarray(hom @ m.T, dtype=np.float64)  # (N, 2) global coords


def _compose_affine_2x3(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affines: apply `inner` first, then `outer`. float64."""
    a = np.asarray(outer, dtype=np.float64).reshape(2, 3)
    b = np.asarray(inner, dtype=np.float64).reshape(2, 3)
    a33 = np.vstack([a, [0.0, 0.0, 1.0]])
    b33 = np.vstack([b, [0.0, 0.0, 1.0]])
    return np.ascontiguousarray((a33 @ b33)[:2, :], dtype=np.float64)


def _invert_affine_2x3(mat: np.ndarray) -> np.ndarray | None:
    """Inverse of a 2x3 affine, or None if singular. float64."""
    try:
        m = np.asarray(mat, dtype=np.float64).reshape(2, 3)
        a, t = m[:, :2], m[:, 2]
        det = float(np.linalg.det(a))
        if abs(det) < 1e-12:
            return None
        a_inv = np.linalg.inv(a)
        return np.ascontiguousarray(
            np.hstack([a_inv, -(a_inv @ t).reshape(2, 1)]), dtype=np.float64
        )
    except Exception:
        return None


def _filter_finite(src_pts: np.ndarray, ref_pts: np.ndarray) -> tuple:
    """Drop rows containing NaN/Inf in either point set. All float64."""
    s = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(ref_pts, dtype=np.float64).reshape(-1, 2)
    n = int(min(s.shape[0], r.shape[0]))
    s = np.asarray(s[:n], dtype=np.float64)
    r = np.asarray(r[:n], dtype=np.float64)
    if n == 0:
        return s, r
    mask_s = np.all(np.isfinite(s), axis=1)
    mask_r = np.all(np.isfinite(r), axis=1)
    mask = np.asarray(mask_s & mask_r, dtype=np.bool_)
    return np.asarray(s[mask], dtype=np.float64), np.asarray(r[mask], dtype=np.float64)


def _rmse_from_residuals(residuals: np.ndarray) -> float:
    """RMSE = sqrt(mean(residual^2)). All float64."""
    r = np.asarray(residuals, dtype=np.float64).ravel()
    if r.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(r * r, dtype=np.float64), dtype=np.float64))


class GlobalBundleAdjuster:
    """Jointly optimize all image poses from pairwise constraints."""

    _VALID_LOSSES = {"linear", "soft_l1", "huber", "cauchy", "arctan"}

    def __init__(self, robust_loss: str = "huber", huber_delta: float = 1.0) -> None:
        """Store robust loss type and Huber delta. Initialize logger."""
        self.robust_loss: str = robust_loss if isinstance(robust_loss, str) else "huber"
        try:
            self.huber_delta: float = float(huber_delta)
            if not np.isfinite(np.asarray(self.huber_delta, dtype=np.float64)):
                self.huber_delta = 1.0
            if self.huber_delta <= 0:
                self.huber_delta = 1.0
        except Exception:
            self.huber_delta = 1.0
        self.constraints: List[dict] = []
        self.logger = logging.getLogger("ML_model.bundle_adjustment")
        self._ordered_ids: List[str] = []

    def add_pairwise_constraint(
        self,
        img_id_src: str,
        img_id_ref: str,
        src_pts: np.ndarray,
        ref_pts: np.ndarray,
        initial_matrix: np.ndarray,
    ) -> None:
        """Store one pairwise edge of the factor graph. NEVER raises."""
        try:
            s = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2)
            r = np.asarray(ref_pts, dtype=np.float64).reshape(-1, 2)
        except Exception as exc:
            self.logger.warning("Skipping constraint: bad point arrays (%s).", exc)
            return
        # Remove NaN/Inf rows up front so degenerate points never enter graph.
        try:
            s, r = _filter_finite(s, r)
        except Exception as exc:
            self.logger.warning("Skipping constraint: finite-filter failed (%s).", exc)
            return
        if s.shape[0] < 4 or r.shape[0] < 4:
            self.logger.warning(
                "Skipping constraint %s->%s: fewer than 4 valid points (%d, %d).",
                img_id_src,
                img_id_ref,
                int(s.shape[0]),
                int(r.shape[0]),
            )
            return
        n = int(min(s.shape[0], r.shape[0]))
        try:
            mat = _to_affine_2x3(np.asarray(initial_matrix, dtype=np.float64))
            mat = np.asarray(mat, dtype=np.float64).reshape(2, 3)
        except Exception as exc:
            self.logger.warning("Skipping constraint %s->%s: bad matrix (%s).", img_id_src, img_id_ref, exc)
            return
        try:
            self.constraints.append(
                {
                    "src": str(img_id_src),
                    "ref": str(img_id_ref),
                    "src_pts": np.ascontiguousarray(np.asarray(s[:n], dtype=np.float64)),
                    "ref_pts": np.ascontiguousarray(np.asarray(r[:n], dtype=np.float64)),
                    "initial_matrix": np.ascontiguousarray(np.asarray(mat, dtype=np.float64)),
                }
            )
        except Exception as exc:
            self.logger.warning("Skipping constraint: store failed (%s).", exc)

    def _reference_id(self) -> str | None:
        """Image appearing most often as img_id_ref (None if no constraints)."""
        if not self.constraints:
            return None
        counts = Counter(np.asarray([c["ref"] for c in self.constraints], dtype=np.str_))
        return str(counts.most_common(1)[0][0])

    def _all_image_ids(self) -> List[str]:
        """All distinct image ids in first-seen order."""
        ids: List[str] = []
        seen = set()
        for c in self.constraints:
            for k in (str(c["src"]), str(c["ref"])):
                if k not in seen:
                    seen.add(k)
                    ids.append(k)
        return ids

    def _build_initial_guess(self) -> dict:
        """Map img_id -> 2x3 affine (float64); reference is identity.

        Poses are composed by BFS from the fixed reference along the
        constraint graph, so multi-hop chains (A->B->ref) initialize in a
        single consistent global frame. The old code copied each edge's
        pairwise matrix verbatim: for edges whose ref endpoint was NOT the
        global reference, that matrix lived in the wrong frame and the
        optimizer started from an inconsistent guess.
        """
        matrices: Dict[str, np.ndarray] = {}
        if not self.constraints:
            return matrices
        reference_id = str(self._reference_id())
        matrices[reference_id] = np.eye(2, 3, dtype=np.float64)
        # Adjacency: neighbor -> list of (edge_matrix_src_to_ref, direction).
        changed = True
        guard = 0
        while changed and guard < len(self._all_image_ids()) + 1:
            changed = False
            guard += 1
            for c in self.constraints:
                s_id, r_id = str(c["src"]), str(c["ref"])
                m = np.asarray(
                    _to_affine_2x3(np.asarray(c["initial_matrix"], dtype=np.float64)),
                    dtype=np.float64,
                )
                if r_id in matrices and s_id not in matrices:
                    # M maps src frame -> ref frame: pose(src) = pose(ref) o M.
                    matrices[s_id] = _compose_affine_2x3(matrices[r_id], m)
                    changed = True
                elif s_id in matrices and r_id not in matrices:
                    # Reverse: pose(ref) = pose(src) o M^-1.
                    m_inv = _invert_affine_2x3(m)
                    matrices[r_id] = (
                        _compose_affine_2x3(matrices[s_id], m_inv)
                        if m_inv is not None
                        else np.eye(2, 3, dtype=np.float64)
                    )
                    changed = True
        for img_id in self._all_image_ids():
            if img_id not in matrices:
                matrices[img_id] = np.eye(2, 3, dtype=np.float64)
        return matrices

    def _compute_residuals(self, matrices: dict) -> np.ndarray:
        """Concatenated per-point 2D mismatch vectors, raveled to (2N,) float64.

        Math per edge:
          1. src_global = M_src @ [src_pts | 1]  (project source pts to world)
          2. ref_global = M_ref @ [ref_pts | 1]  (project reference pts to world)
          3. diff = src_global - ref_global      (2D mismatch in world frame)
        The (N, 2) diffs are raveled to (2N,) for least_squares. The old code
        returned per-point Euclidean NORMS (N,): for linear loss the RMSE is
        identical either way, but norms pre-collapse the 2D structure and
        distort robust-loss (huber/cauchy/...) weighting, which least_squares
        applies per residual element.
        """
        parts: List[np.ndarray] = []
        for c in self.constraints:
            try:
                # Current pose estimates for both endpoints of this edge.
                m_src = np.asarray(matrices[c["src"]], dtype=np.float64).reshape(2, 3)
                m_ref = np.asarray(matrices[c["ref"]], dtype=np.float64).reshape(2, 3)
                # Point sets in local pixel frames, promoted to float64.
                s = np.asarray(c["src_pts"], dtype=np.float64).reshape(-1, 2)
                r = np.asarray(c["ref_pts"], dtype=np.float64).reshape(-1, 2)
                n = int(min(s.shape[0], r.shape[0]))
                if n == 0:
                    continue
                s = np.asarray(s[:n], dtype=np.float64)
                r = np.asarray(r[:n], dtype=np.float64)
                # Project both sets into the shared global frame.
                s_g = _apply_affine(s, m_src)  # (N, 2) float64
                r_g = _apply_affine(r, m_ref)  # (N, 2) float64
                diff = np.asarray(s_g - r_g, dtype=np.float64)  # (N, 2)
                parts.append(np.ascontiguousarray(np.asarray(diff.ravel(), dtype=np.float64)))
            except Exception as exc:
                self.logger.warning("Residual computation skipped for an edge (%s).", exc)
                continue
        if not parts:
            return np.zeros((0,), dtype=np.float64)
        return np.ascontiguousarray(np.concatenate(parts).astype(np.float64), dtype=np.float64)

    def _pack_matrices(self, matrices: dict) -> np.ndarray:
        """Flatten non-reference 2x3 matrices to a 1D float64 vector.

        Order [m00, m01, m02, m10, m11, m12] per image. The reference image
        is FIXED at identity and is NOT included in the vector.
        """
        reference_id = self._reference_id()
        ids = sorted(np.asarray([k for k in matrices.keys() if k != reference_id], dtype=np.str_).tolist())
        self._ordered_ids = [str(i) for i in ids]
        if not self._ordered_ids:
            return np.zeros((0,), dtype=np.float64)
        chunks = [
            np.asarray(np.asarray(matrices[i], dtype=np.float64).reshape(-1), dtype=np.float64)
            for i in self._ordered_ids
        ]
        vec = np.concatenate(chunks)
        return np.ascontiguousarray(np.asarray(vec, dtype=np.float64))

    def _unpack_matrices(self, param_vector: np.ndarray, reference_id: str) -> dict:
        """Reconstruct {img_id: 2x3} from 1D vector; reference is identity."""
        matrices: Dict[str, np.ndarray] = {}
        try:
            p = np.asarray(param_vector, dtype=np.float64).ravel()
            p = np.asarray(p, dtype=np.float64)
        except Exception:
            p = np.zeros((0,), dtype=np.float64)
        ids = list(self._ordered_ids) if self._ordered_ids else []
        if not ids:
            ids = sorted(np.asarray([k for k in self._all_image_ids() if k != reference_id], dtype=np.str_).tolist())
            self._ordered_ids = [str(i) for i in ids]
        for j, img_id in enumerate(ids):
            seg = np.asarray(p[j * 6:(j + 1) * 6], dtype=np.float64)
            if seg.size < 6:
                pad = np.zeros((6 - seg.size,), dtype=np.float64)
                seg = np.concatenate([seg, pad])
                self.logger.warning("Short parameter vector; padding %s with zeros.", img_id)
            matrices[str(img_id)] = np.ascontiguousarray(np.asarray(seg.reshape(2, 3), dtype=np.float64))
        matrices[str(reference_id)] = np.eye(2, 3, dtype=np.float64)
        return matrices

    def _sanitize_constraints(self) -> None:
        """Remove NaN/Inf points in place; drop edges left with <1 points."""
        kept: List[dict] = []
        for c in self.constraints:
            try:
                s, r = _filter_finite(
                    np.asarray(c["src_pts"], dtype=np.float64),
                    np.asarray(c["ref_pts"], dtype=np.float64),
                )
                if s.shape[0] == 0:
                    self.logger.warning("Dropping edge %s->%s: 0 finite points.", c["src"], c["ref"])
                    continue
                c["src_pts"] = np.ascontiguousarray(np.asarray(s, dtype=np.float64))
                c["ref_pts"] = np.ascontiguousarray(np.asarray(r, dtype=np.float64))
                kept.append(c)
            except Exception as exc:
                self.logger.warning("Dropping edge: sanitize failed (%s).", exc)
                continue
        self.constraints = kept

    def optimize(self, max_iterations: int = 200) -> dict:
        """Jointly optimize all poses. NEVER raises.

        Returns success dict with optimized_matrices / reference_image /
        initial_rmse / final_rmse / improvement_percent / num_iterations /
        success_flag, or converged_with_warnings with initial matrices if
        the optimizer diverges, or failed if input is insufficient.
        """
        try:
            # Guardrail: empty graph fails immediately.
            if not self.constraints:
                return {"status": "failed", "reason": "insufficient_constraints"}
            if len(self.constraints) < 2:
                return {"status": "failed", "reason": "insufficient_constraints"}

            # Guardrail: purge NaN/Inf points before optimization.
            self._sanitize_constraints()
            if len(self.constraints) < 2:
                return {"status": "failed", "reason": "insufficient_constraints"}

            initial_matrices = self._build_initial_guess()
            reference_id = self._reference_id()
            if reference_id is None:
                return {"status": "failed", "reason": "insufficient_constraints"}
            reference_id = str(reference_id)

            initial_params = self._pack_matrices(initial_matrices)
            if np.asarray(initial_params, dtype=np.float64).size == 0:
                return {"status": "failed", "reason": "nothing_to_optimize"}

            init_res = self._compute_residuals(initial_matrices)
            initial_rmse = _rmse_from_residuals(init_res)

            loss = self.robust_loss if self.robust_loss in self._VALID_LOSSES else "huber"
            if loss != self.robust_loss:
                self.logger.warning("Unknown robust loss %r; falling back to 'huber'.", self.robust_loss)

            def _fun(p: np.ndarray) -> np.ndarray:
                return self._compute_residuals(self._unpack_matrices(p, reference_id))

            try:
                max_nfev = int(max_iterations)
            except Exception:
                max_nfev = 200
            max_nfev = int(max(max_nfev, 1))

            x0 = np.ascontiguousarray(np.asarray(initial_params, dtype=np.float64))

            result = None
            last_exc: Exception | None = None
            # scipy 'lm' ONLY supports linear loss (robust loss raises
            # ValueError on every call), so select the method upfront: lm iff
            # loss == linear, else trf. One fallback attempt with the other
            # method survives singular-Jacobian failures either way.
            methods = ["lm", "trf"] if loss == "linear" else ["trf", "lm"]
            for _method in methods:
                try:
                    _kwargs: dict = (
                        {"loss": "linear"}
                        if _method == "lm"
                        else {"loss": loss, "f_scale": np.float64(self.huber_delta)}
                    )
                    result = least_squares(
                        fun=_fun,
                        x0=x0,
                        method=_method,
                        xtol=1e-12,
                        ftol=1e-12,
                        max_nfev=max_nfev,
                        **_kwargs,
                    )
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    self.logger.warning(
                        "least_squares(%s) failed (%s); trying fallback.", _method, e
                    )
                    result = None

            # Guardrail: both optimizers failed -> return INITIAL matrices.
            if result is None:
                self.logger.warning(
                    "Returning initial matrices (initial_rmse=%.4f). Reason: %s",
                    float(np.asarray(initial_rmse, dtype=np.float64)),
                    last_exc,
                )
                initial_rmse_f = float(np.asarray(initial_rmse, dtype=np.float64))
                return {
                    "status": "converged_with_warnings",
                    "optimized_matrices": {
                        k: np.asarray(v, dtype=np.float64) for k, v in initial_matrices.items()
                    },
                    "reference_image": reference_id,
                    "initial_rmse": initial_rmse_f,
                    "final_rmse": initial_rmse_f,
                    "improvement_percent": float(np.asarray(0.0, dtype=np.float64)),
                    "num_iterations": 0,
                    "success_flag": False,
                    # Back-compat aliases.
                    "matrices": {
                        k: np.asarray(v, dtype=np.float64) for k, v in initial_matrices.items()
                    },
                    "reference_id": reference_id,
                }

            # Unpack final poses; reference stays exactly identity.
            final_matrices = self._unpack_matrices(
                np.asarray(result.x, dtype=np.float64), reference_id
            )
            final_matrices[reference_id] = np.eye(2, 3, dtype=np.float64)

            fin_res = self._compute_residuals(final_matrices)
            final_rmse = _rmse_from_residuals(fin_res)
            initial_rmse_f = float(np.asarray(initial_rmse, dtype=np.float64))
            final_rmse_f = float(np.asarray(final_rmse, dtype=np.float64))
            if np.isfinite(np.asarray(initial_rmse_f, dtype=np.float64)) and initial_rmse_f > 0:
                improvement = float(
                    np.asarray(
                        (initial_rmse_f - final_rmse_f) / initial_rmse_f * 100.0,
                        dtype=np.float64,
                    )
                )
            else:
                improvement = float(np.asarray(0.0, dtype=np.float64))
            num_iters = int(getattr(result, "nfev", max_nfev))
            success_flag = bool(getattr(result, "success", False))

            self.logger.info(
                "Bundle adjustment: initial_rmse=%.4f final_rmse=%.4f "
                "improvement=%.2f%% iterations=%d success=%s",
                initial_rmse_f,
                final_rmse_f,
                improvement,
                num_iters,
                success_flag,
            )

            optimized = {k: np.asarray(v, dtype=np.float64) for k, v in final_matrices.items()}
            return {
                "status": "success",
                "optimized_matrices": optimized,
                "reference_image": reference_id,
                "initial_rmse": initial_rmse_f,
                "final_rmse": final_rmse_f,
                "improvement_percent": improvement,
                "num_iterations": num_iters,
                "success_flag": success_flag,
                # Back-compat aliases for earlier consumers.
                "matrices": optimized,
                "reference_id": reference_id,
                "initial_cost": float(
                    np.sum(np.asarray(init_res, dtype=np.float64) ** 2, dtype=np.float64)
                ),
                "final_cost": float(
                    np.sum(np.asarray(fin_res, dtype=np.float64) ** 2, dtype=np.float64)
                ),
                "num_constraints": int(len(self.constraints)),
                "num_images": int(len(optimized)),
            }
        except Exception as exc:
            self.logger.warning("Bundle adjustment crashed: %s", exc)
            return {"status": "failed", "reason": "exception: %s" % exc}


__all__ = ["GlobalBundleAdjuster"]
