"""3D photogrammetric validation on the lunar sphere.

Converts matched lat/lon/alt points to Moon-centered Cartesian (float64),
computes Euclidean intersection errors, and formats ISRO-ready reports.

Math convention (spherical Moon, radius MOON_RADIUS_M):
    R = MOON_RADIUS_M + alt
    X = R * cos(lat) * cos(lon)
    Y = R * cos(lat) * sin(lon)
    Z = R * sin(lat)
with lat/lon in radians. All 3D work uses float64: at ~1.7e6 m from the
origin, float32 quantization (~0.1 m per LSB) would inject meter-level
rounding error into RMSE/CE90.
"""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Mean lunar radius in meters (must match lro_validator.MOON_RADIUS_M).
MOON_RADIUS_M = 1737400.0


class PhotogrammetricValidator:
    """Compute 2D reprojection and 3D intersection error metrics."""

    # -- Coordinate conversion -------------------------------------------

    def _latlon_to_cartesian(
        self,
        lat_deg,
        lon_deg,
        alt_m,
    ) -> np.ndarray:
        """Convert lat/lon/alt to Moon-centered Cartesian XYZ (meters).

        Args:
            lat_deg: Latitude(s) in degrees, scalar or array-like.
            lon_deg: Longitude(s) in degrees, scalar or array-like.
            alt_m: Altitude(s) in meters above ``MOON_RADIUS_M``.

        Returns:
            ``np.ndarray`` of shape ``(N, 3)`` with ``dtype=np.float64``.
        """
        lat = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
        lon = np.asarray(lon_deg, dtype=np.float64).reshape(-1)
        alt = np.asarray(alt_m, dtype=np.float64).reshape(-1)

        # Degrees -> radians (float64 throughout).
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Radial distance from the Moon's center.
        radius = np.asarray(MOON_RADIUS_M, dtype=np.float64) + alt

        cos_lat = np.cos(lat_rad)
        xyz = np.zeros((lat.size, 3), dtype=np.float64)
        xyz[:, 0] = radius * cos_lat * np.cos(lon_rad)  # X
        xyz[:, 1] = radius * cos_lat * np.sin(lon_rad)  # Y
        xyz[:, 2] = radius * np.sin(lat_rad)  # Z
        return xyz

    # -- 3D metrics ------------------------------------------------------

    def calculate_3d_intersection_error(
        self,
        pts_src_llas,
        pts_ref_llas,
    ) -> dict:
        """Compute 3D Euclidean error between matched LLA points.

        Args:
            pts_src_llas: ``(N, 3)`` array of ``[lat, lon, alt]`` for the
                source image (alt sampled from the LOLA DEM).
            pts_ref_llas: ``(N, 3)`` array of ``[lat, lon, alt]`` for the
                reference image.

        Returns:
            Dict with ``RMSE_meters``, ``Mean_Error_m``, ``CE90``,
            ``LE90`` (vertical 90th percentile), ``Max_Error_m``,
            ``num_points``, ``num_valid`` and ``num_dropped_nan``.
        """
        src = (np.array([], dtype=np.float64).reshape(0, 3)
               if pts_src_llas is None else
               np.asarray(pts_src_llas, dtype=np.float64))
        ref = (np.array([], dtype=np.float64).reshape(0, 3)
               if pts_ref_llas is None else
               np.asarray(pts_ref_llas, dtype=np.float64))

        if src.size == 0 or ref.size == 0 or src.shape[0] == 0:
            logger.warning("3D error: empty input arrays (0 matches).")
            return {
                "RMSE_meters": 0.0,
                "Mean_Error_m": 0.0,
                "CE90": 0.0,
                "LE90": 0.0,
                "Max_Error_m": 0.0,
                "num_points": 0,
                "num_valid": 0,
                "num_dropped_nan": 0,
            }

        src = src.reshape(-1, 3)
        ref = ref.reshape(-1, 3)
        num_points = int(min(src.shape[0], ref.shape[0]))
        src, ref = src[:num_points], ref[:num_points]

        # Filter points with NaN elevation (or NaN lat/lon) before RMSE.
        valid_mask = np.isfinite(src).all(axis=1) & np.isfinite(ref).all(axis=1)
        num_dropped = int(num_points - np.count_nonzero(valid_mask))
        if num_dropped:
            logger.warning("3D error: dropping %d point(s) with NaN LLA.",
                           num_dropped)
        src_valid = src[valid_mask]
        ref_valid = ref[valid_mask]
        num_valid = int(src_valid.shape[0])

        if num_valid == 0:
            logger.warning("3D error: no valid points after NaN filtering.")
            return {
                "RMSE_meters": 0.0,
                "Mean_Error_m": 0.0,
                "CE90": 0.0,
                "LE90": 0.0,
                "Max_Error_m": 0.0,
                "num_points": num_points,
                "num_valid": 0,
                "num_dropped_nan": num_dropped,
            }

        logger.info("3D error: using %d/%d valid points for RMSE.",
                    num_valid, num_points)

        # Lat/Lon/Alt -> Moon-centered Cartesian XYZ (float64).
        src_xyz = self._latlon_to_cartesian(
            src_valid[:, 0], src_valid[:, 1], src_valid[:, 2])
        ref_xyz = self._latlon_to_cartesian(
            ref_valid[:, 0], ref_valid[:, 1], ref_valid[:, 2])

        # Per-pair 3D Euclidean intersection distance in meters.
        dist = np.linalg.norm(
            np.asarray(src_xyz, dtype=np.float64)
            - np.asarray(ref_xyz, dtype=np.float64),
            axis=1,
        )

        # Vertical (radial) component for LE90.
        least = np.abs(
            np.linalg.norm(src_xyz, axis=1) - np.linalg.norm(ref_xyz, axis=1))

        return {
            "RMSE_meters": float(np.sqrt(np.mean(dist ** 2))),
            "Mean_Error_m": float(np.mean(dist)),
            "CE90": float(np.percentile(dist, 90)),
            "LE90": float(np.percentile(least, 90)),
            "Max_Error_m": float(np.max(dist)),
            "num_points": num_points,
            "num_valid": num_valid,
            "num_dropped_nan": num_dropped,
        }

    # -- 2D metrics ------------------------------------------------------

    def calculate_2d_reprojection_error(
        self,
        src_pts_2d,
        ref_pts_2d,
        transform_matrix,
    ) -> dict:
        """Compute 2D reprojection error of a 3x3 homography/affine.

        Args:
            src_pts_2d: ``(N, 2)`` source pixel coordinates.
            ref_pts_2d: ``(N, 2)`` reference pixel coordinates.
            transform_matrix: ``(3, 3)`` homography/affine matrix.

        Returns:
            Dict with ``RMSE_pixels``, ``Mean_pixels``, ``Max_pixels``
            and ``num_points``.
        """
        src = (np.array([], dtype=np.float64).reshape(0, 2)
               if src_pts_2d is None else
               np.asarray(src_pts_2d, dtype=np.float64).reshape(-1, 2))
        ref = (np.array([], dtype=np.float64).reshape(0, 2)
               if ref_pts_2d is None else
               np.asarray(ref_pts_2d, dtype=np.float64).reshape(-1, 2))

        if src.shape[0] == 0 or ref.shape[0] == 0:
            logger.warning("2D error: empty input arrays (0 matches).")
            return {
                "RMSE_pixels": 0.0,
                "Mean_pixels": 0.0,
                "Max_pixels": 0.0,
                "num_points": 0,
            }

        n_pts = int(min(src.shape[0], ref.shape[0]))
        src, ref = src[:n_pts], ref[:n_pts]

        # Drop non-finite correspondences before projecting.
        valid_mask = np.isfinite(src).all(axis=1) & np.isfinite(ref).all(axis=1)
        src, ref = src[valid_mask], ref[valid_mask]
        if src.shape[0] == 0:
            logger.warning("2D error: no finite points after filtering.")
            return {
                "RMSE_pixels": 0.0,
                "Mean_pixels": 0.0,
                "Max_pixels": 0.0,
                "num_points": 0,
            }

        try:
            mat = np.asarray(transform_matrix, dtype=np.float64).reshape(3, 3)
        except (TypeError, ValueError) as exc:
            logger.error("2D error: invalid transform matrix: %s", exc)
            raise ValueError(
                "transform_matrix must be a 3x3 array-like.") from exc

        # Homogeneous projection: p' = H @ [x, y, 1]^T, then dehomogenize.
        ones = np.ones((src.shape[0], 1), dtype=np.float64)
        src_h = np.hstack([src, ones])  # (N, 3), float64
        proj_h = (mat @ src_h.T).T  # (N, 3)
        w = proj_h[:, 2:3]
        w[~np.isfinite(w)] = np.nan
        w[w == 0.0] = np.nan
        projected = proj_h[:, :2] / w

        finite = np.isfinite(projected).all(axis=1)
        projected, ref_f = projected[finite], ref[finite]
        if projected.shape[0] == 0:
            logger.warning("2D error: projection produced no finite points.")
            return {
                "RMSE_pixels": 0.0,
                "Mean_pixels": 0.0,
                "Max_pixels": 0.0,
                "num_points": 0,
            }

        err = np.linalg.norm(projected - ref_f, axis=1)
        return {
            "RMSE_pixels": float(np.sqrt(np.mean(err ** 2))),
            "Mean_pixels": float(np.mean(err)),
            "Max_pixels": float(np.max(err)),
            "num_points": int(err.shape[0]),
        }

    # -- Reporting -------------------------------------------------------

    def generate_isro_report(self, metrics_3d: dict, metrics_2d: dict) -> str:
        """Format 3D + 2D metrics as an ISRO-ready JSON report string."""
        metrics_3d = dict(metrics_3d or {})
        metrics_2d = dict(metrics_2d or {})

        rmse_m = float(metrics_3d.get("RMSE_meters", 0.0))
        ce90 = float(metrics_3d.get("CE90", 0.0))
        le90 = float(metrics_3d.get("LE90", 0.0))
        rmse_px = float(metrics_2d.get("RMSE_pixels", 0.0))

        # ISRO-style pass/fail gates: sub-pixel fit + meter-level surface fix.
        status_3d = "PASS" if rmse_m <= 5.0 and ce90 <= 10.0 else "REVIEW"
        status_2d = "PASS" if rmse_px <= 1.0 else "REVIEW"
        overall = "PASS" if status_3d == "PASS" and status_2d == "PASS" \
            else "REVIEW"

        report = {
            "mission": "Chandrayaan-2 image registration validation",
            "reference": "LRO LOLA DEM (Moon radius 1737400.0 m, float64)",
            "metrics_3d_meters": {
                "RMSE_meters": rmse_m,
                "Mean_Error_m": float(metrics_3d.get("Mean_Error_m", 0.0)),
                "CE90": ce90,
                "LE90": le90,
                "Max_Error_m": float(metrics_3d.get("Max_Error_m", 0.0)),
                "num_points": int(metrics_3d.get("num_points", 0)),
                "num_valid": int(metrics_3d.get(
                    "num_valid", metrics_3d.get("num_points", 0))),
                "num_dropped_nan": int(metrics_3d.get("num_dropped_nan", 0)),
            },
            "metrics_2d_pixels": {
                "RMSE_pixels": rmse_px,
                "Mean_pixels": float(metrics_2d.get("Mean_pixels", 0.0)),
                "Max_pixels": float(metrics_2d.get("Max_pixels", 0.0)),
                "num_points": int(metrics_2d.get("num_points", 0)),
            },
            "verdict": {
                "status_3d": status_3d,
                "status_2d": status_2d,
                "overall": overall,
                "criteria": ("3D PASS iff RMSE<=5m and CE90<=10m; "
                             "2D PASS iff RMSE<=1px"),
            },
        }
        return json.dumps(report, indent=2)
