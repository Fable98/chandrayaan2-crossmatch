"""Rigorous SPICE geometry extraction for Chandrayaan-2 imagery.

Loads generic + mission-specific SPICE kernels with spiceypy and
computes, for an exact image capture epoch:

* spacecraft state relative to the Moon,
* Sun illumination geometry (elevation / azimuth above the local horizon),
* camera ground footprint corners on the lunar ellipsoid.

Every SPICE call is guarded with ``try/except SpiceyError`` because a
missing kernel, an unloaded frame, or an epoch outside CK/SPK coverage
raises a harsh C-level error that must never kill the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import spiceypy
    from spiceypy.utils.exceptions import SpiceyError
except ImportError:  # pragma: no cover - spiceypy is a hard dependency
    spiceypy = None  # type: ignore[assignment]

    class SpiceyError(Exception):  # type: ignore[no-redef]
        """Fallback so module imports without spiceypy installed."""


logger = logging.getLogger(__name__)

# Target / observer / frame names used throughout.
MOON = "MOON"
MOON_FRAME = "MOON_ME"          # Moon mean-Earth body-fixed frame
SPACECRAFT = "CHANDRAYAAN-2"
SUN = "SUN"

# Lunar radii (km) from the PCK, fallback to IAU mean values when the
# PCK kernel is unavailable.
MOON_RADII_FALLBACK = (1737.4, 1737.4, 1737.4)


class SpiceMetadataExtractor:
    """Load kernels and extract illumination geometry + footprints."""

    def __init__(self, generic_kernels_dir: Path | str) -> None:
        self.generic_kernels_dir = Path(generic_kernels_dir)
        # False until at least the time-conversion (LSK) kernel loads.
        self.is_loaded = False

        if spiceypy is None:
            logger.error("spiceypy is not installed; SPICE calls disabled")
            return

        # Mission-agnostic kernels: leapseconds (time conversion),
        # planetary constants (body radii), and any frame definitions.
        # Missing files are tolerated individually.
        patterns = ("*.tls", "*.lsk", "*.tpc", "*.pck", "*.tf", "*.fk")
        loaded_any = False
        for pattern in patterns:
            for kernel in sorted(self.generic_kernels_dir.glob(pattern)):
                try:
                    spiceypy.furnsh(str(kernel))
                except SpiceyError as exc:
                    logger.error("furnsh failed for generic kernel %s: %s",
                                 kernel, exc)
                    continue
                logger.info("Loaded generic kernel %s", kernel)
                loaded_any = True

        if not loaded_any:
            logger.error("No generic kernels found in %s; "
                         "time conversion will fail", self.generic_kernels_dir)
        self.is_loaded = loaded_any

    # ------------------------------------------------------------------
    def load_mission_kernels(
        self,
        spk_path: Path | str | None = None,
        ck_path: Path | str | None = None,
        ik_path: Path | str | None = None,
        sclk_path: Path | str | None = None,
    ) -> bool:
        """Furnsh mission-specific binary kernels for one orbit.

        Each furnsh is individually guarded; missing kernels are logged
        and skipped. Returns True if at least one kernel loaded.
        """
        if spiceypy is None:
            logger.error("spiceypy unavailable; cannot load mission kernels")
            return False

        loaded = False
        for label, path in (("SPK", spk_path), ("CK", ck_path),
                            ("IK", ik_path), ("SCLK", sclk_path)):
            if path is None:
                continue
            try:
                spiceypy.furnsh(str(path))
            except SpiceyError as exc:
                logger.error("furnsh failed for %s kernel %s: %s",
                             label, path, exc)
                continue
            except Exception as exc:  # e.g. file missing at C level
                logger.error("furnsh error for %s kernel %s: %s",
                             label, path, exc)
                continue
            logger.info("Loaded %s kernel %s", label, path)
            loaded = True

        if loaded:
            self.is_loaded = True
        return loaded

    # ------------------------------------------------------------------
    def _et_from_utc(self, image_utc_time: str) -> float | None:
        """Convert a UTC string to Ephemeris Time (seconds past J2000)."""
        try:
            return float(spiceypy.str2et(image_utc_time))
        except SpiceyError as exc:
            logger.error("str2et failed for %r: %s", image_utc_time, exc)
            return None

    def _moon_radii(self) -> tuple[float, float, float]:
        """Return lunar radii in km, preferring the loaded PCK."""
        try:
            _, radii = spiceypy.bodvrd("MOON", "RADII", 3)
            return (float(radii[0]), float(radii[1]), float(radii[2]))
        except SpiceyError as exc:
            logger.warning("bodvrd(MOON RADII) failed: %s; using fallback",
                           exc)
            return MOON_RADII_FALLBACK

    # ------------------------------------------------------------------
    def extract_geometry(self, image_utc_time: str) -> dict:
        """Compute Sun azimuth/elevation for the sub-spacecraft point.

        Math:
          1. ``spkpos`` gives the spacecraft and Sun positions relative
             to the Moon centre in the MOON_ME body-fixed frame.
          2. ``subpnt`` (near-point ellipsoid intercept) gives the
             sub-spacecraft surface point and the outward surface normal
             (``srfvec``).
          3. Sun elevation = arcsin(dot(sun_dir, normal_hat)): the dot
             product of the unit Sun vector with the unit surface normal
             equals sin(elevation), since elevation is measured from the
             local horizon plane up to the Sun vector.
          4. Sun azimuth = atan2 of the Sun vector's east/north
             components in the local topocentric frame (0 deg = north,
             increasing eastward).

        Returns a dict with sun_elev / sun_az (degrees) plus spacecraft
        state, or {"status": "failed", ...} with Nones on SPICE errors.
        """
        failed = {"status": "failed", "reason": "spice_error",
                  "sun_elev": None, "sun_az": None}

        if spiceypy is None or not self.is_loaded:
            logger.error("SPICE kernels not loaded; cannot extract geometry")
            return {**failed, "reason": "kernels_not_loaded"}

        et = self._et_from_utc(image_utc_time)
        if et is None:
            return {**failed, "reason": "time_conversion_failed"}

        try:
            # Spacecraft and Sun positions w.r.t. Moon centre, body-fixed.
            sc_pos, _lt_sc = spiceypy.spkpos(SPACECRAFT, et, MOON_FRAME,
                                             "NONE", MOON)
            sun_pos, _lt_sun = spiceypy.spkpos(SUN, et, MOON_FRAME,
                                               "NONE", MOON)
            # Sub-spacecraft point + outward normal (srfvec).
            spoint, _trgepc, srfvec = spiceypy.subpnt(
                "Near point: ellipsoid", MOON, et, MOON_FRAME,
                "NONE", SPACECRAFT)
        except SpiceyError as exc:
            # Typical cause: epoch outside SPK/CK coverage.
            logger.error("SPICE position call failed at et=%.3f: %s", et, exc)
            return failed

        sc_pos = np.asarray(sc_pos, dtype=float)
        sun_pos = np.asarray(sun_pos, dtype=float)
        spoint = np.asarray(spoint, dtype=float)
        normal = np.asarray(srfvec, dtype=float)

        n_norm = float(np.linalg.norm(normal))
        if n_norm == 0.0:
            logger.error("Degenerate surface normal at et=%.3f", et)
            return failed
        normal_hat = normal / n_norm

        # Vector from surface point toward the Sun.
        to_sun = sun_pos - spoint
        r = float(np.linalg.norm(to_sun))
        if r == 0.0:
            logger.error("Sun coincides with surface point at et=%.3f", et)
            return failed
        to_sun_hat = to_sun / r

        # sin(elev) = dot(to_sun_hat, normal_hat); elev in [-90, 90].
        cos_zenith = float(np.clip(np.dot(to_sun_hat, normal_hat), -1.0, 1.0))
        elev_deg = float(np.degrees(np.arcsin(cos_zenith)))

        # Local topocentric basis: up = normal, east = up x moon-z,
        # north = east x up (orthonormalised).
        z_axis = np.array([0.0, 0.0, 1.0])
        east = np.cross(z_axis, normal_hat)
        if float(np.linalg.norm(east)) < 1e-12:
            # Sub-spacecraft point near a pole; pick an arbitrary east.
            east = np.array([1.0, 0.0, 0.0])
        east = east / float(np.linalg.norm(east))
        north = np.cross(normal_hat, east)
        north = north / float(np.linalg.norm(north))

        az_rad = float(np.arctan2(np.dot(to_sun_hat, east),
                                  np.dot(to_sun_hat, north)))
        az_deg = (float(np.degrees(az_rad)) + 360.0) % 360.0

        # Surface point as planetodetic lat/lon for logging/downstream use.
        try:
            _radius, lon_rad, lat_rad = spiceypy.reclat(spoint)
            lat_deg = float(np.degrees(lat_rad))
            lon_deg = float(np.degrees(lon_rad))
        except SpiceyError as exc:
            logger.warning("reclat failed at et=%.3f: %s", et, exc)
            lat_deg, lon_deg = None, None

        logger.info("et=%.3f utc=%s subpoint=(%.4f, %.4f) sun_elev=%.3f "
                    "sun_az=%.3f", et, image_utc_time, lat_deg or 0.0,
                    lon_deg or 0.0, elev_deg, az_deg)

        return {
            "status": "success",
            "et": et,
            "utc": image_utc_time,
            "sun_elev": elev_deg,
            "sun_az": az_deg,
            "subpoint_lat": lat_deg,
            "subpoint_lon": lon_deg,
            "sc_pos_moon_fixed_km": sc_pos.tolist(),
            "sun_pos_moon_fixed_km": sun_pos.tolist(),
        }

    # ------------------------------------------------------------------
    def compute_footprint(
        self,
        image_utc_time: str,
        fov_half_angles: tuple[float, float] = (0.5, 0.5),
    ) -> list[tuple]:
        """Project the 4 FOV corners onto the lunar ellipsoid.

        Uses ``sincpt`` (ray-ellipsoid intercept with light-time aware
        geometry when kernels permit), falling back to analytic
        ``surfpt`` math. Corner boresights are built by rotating the
        nadir direction by +/- the half-angles about the body-fixed x/y
        axes -- a small-angle approximation valid for narrow lunar
        mapping cameras (exact IK boresights preferred when available).

        Returns:
            List of (lat_deg, lon_deg) corners, or [] on SPICE errors.
        """
        if spiceypy is None or not self.is_loaded:
            logger.error("SPICE kernels not loaded; cannot compute footprint")
            return []

        et = self._et_from_utc(image_utc_time)
        if et is None:
            return []

        hx_deg, hy_deg = fov_half_angles
        hx = float(np.radians(hx_deg))
        hy = float(np.radians(hy_deg))
        radii = self._moon_radii()
        corners: list[tuple] = []

        # Four corner offsets: (dx, dy) rotations of the nadir ray.
        offsets = [(+hx, +hy), (-hx, +hy), (-hx, -hy), (+hx, -hy)]

        for dx, dy in offsets:
            # Nadir direction in body-fixed frame, tilted by corner angles.
            dvec = np.array([np.tan(dx), np.tan(dy), -1.0])
            dvec = dvec / float(np.linalg.norm(dvec))
            try:
                sc_pos, _ = spiceypy.spkpos(SPACECRAFT, et, MOON_FRAME,
                                            "NONE", MOON)
                spoint, _trgepc, _srfvec, found = spiceypy.sincpt(
                    "Ellipsoid", MOON, et, MOON_FRAME, "NONE",
                    SPACECRAFT, MOON_FRAME, np.asarray(sc_pos, dtype=float),
                    dvec)
                if not found:
                    # Analytic fallback: ray from sc_pos along dvec.
                    spoint, found = spiceypy.surfpt(sc_pos, dvec, *radii)
                if not found:
                    logger.warning("No surface intercept for corner "
                                   "(%.4f, %.4f) at et=%.3f", dx, dy, et)
                    continue
                _radius, lon_rad, lat_rad = spiceypy.reclat(spoint)
                corners.append((float(np.degrees(lat_rad)),
                                float(np.degrees(lon_rad))))
            except SpiceyError as exc:
                logger.error("Footprint SPICE call failed at et=%.3f: %s",
                             et, exc)
                return []

        if corners:
            centre_lat = sum(c[0] for c in corners) / len(corners)
            centre_lon = sum(c[1] for c in corners) / len(corners)
            logger.info("et=%.3f footprint centre=(%.4f, %.4f) corners=%d",
                        et, centre_lat, centre_lon, len(corners))
        return corners

    # ------------------------------------------------------------------
    def unload_all(self) -> None:
        """Unload all kernels (useful between orbits / in tests)."""
        if spiceypy is None:
            return
        try:
            spiceypy.kclear()
        except SpiceyError as exc:
            logger.warning("kclear failed: %s", exc)
        self.is_loaded = False
