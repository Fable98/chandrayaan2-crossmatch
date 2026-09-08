"""LRO LOLA DEM handler for ground-truth elevation sampling.

Loads LOLA DEM rasters via rasterio and samples elevation (meters,
relative to the Moon's mean radius) at a given lat/lon.
"""

import logging

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

# Mean lunar radius in meters (LOLA reference sphere).
MOON_RADIUS_M = 1737400.0


class LRODataHandler:
    """Fetch/load LRO LOLA DEMs and sample elevation at lat/lon."""

    def load_dem(self, dem_path: str) -> dict:
        """Open a LOLA DEM file and return its array + georeferencing.

        Args:
            dem_path: Path to a ``.tif`` (or PDS ``.img`` readable by
                rasterio) DEM file.

        Returns:
            On success ``{"status": "success", "dem_array": ...,
            "transform": ..., "crs": ..., "bounds": ..., "width": ...,
            "height": ..., "nodata": ...}``; on any failure
            ``{"status": "failed"}``. Never raises for corrupt/missing files.
        """
        try:
            src = rasterio.open(dem_path)
        except Exception as exc:  # missing file, corrupt raster, bad driver
            logger.error("Failed to open DEM %s: %s", dem_path, exc)
            return {"status": "failed"}

        try:
            dem_array = src.read(1)  # band 1 = elevation in meters
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            width = src.width
            height = src.height
            nodata = src.nodata
        except Exception as exc:
            logger.error("Failed to read DEM %s: %s", dem_path, exc)
            try:
                src.close()
            except Exception:
                pass
            return {"status": "failed"}

        # Keep the dataset open handle out of the dict (not picklable);
        # callers work with the in-memory array + transform.
        try:
            src.close()
        except Exception:
            pass

        return {
            "status": "success",
            "dem_array": dem_array,
            "transform": transform,
            "crs": crs,
            "bounds": bounds,
            "width": width,
            "height": height,
            "nodata": nodata,
        }

    def get_elevation(self, lat: float, lon: float, dem_data: dict) -> float:
        """Sample DEM elevation (meters) at a lat/lon in degrees.

        Args:
            lat: Latitude in degrees (-90..90).
            lon: Longitude in degrees (-180..180).
            dem_data: Dict returned by :meth:`load_dem`.

        Returns:
            Elevation in meters relative to ``MOON_RADIUS_M``,
            or ``np.nan`` if the point is outside the DEM, the DEM is
            invalid, or the pixel holds nodata.
        """
        if not dem_data or dem_data.get("status") != "success":
            logger.warning("get_elevation called with invalid dem_data.")
            return float(np.nan)

        dem_array = dem_data.get("dem_array")
        transform = dem_data.get("transform")
        bounds = dem_data.get("bounds")
        nodata = dem_data.get("nodata")
        if dem_array is None or transform is None or bounds is None:
            logger.warning("get_elevation: dem_data missing array/transform.")
            return float(np.nan)

        # Guardrail: outside the DEM footprint -> NaN.
        if (lon < bounds.left or lon > bounds.right
                or lat < bounds.bottom or lat > bounds.top):
            logger.warning(
                "Lat/Lon (%.6f, %.6f) outside DEM bounds %s.",
                lat, lon, bounds,
            )
            return float(np.nan)

        try:
            # Affine inverse: (lon, lat) -> fractional (col, row).
            col_f, row_f = ~transform * (float(lon), float(lat))
            col = int(np.floor(col_f))
            row = int(np.floor(row_f))
        except Exception as exc:
            logger.warning("Affine inversion failed for (%.6f, %.6f): %s",
                           lat, lon, exc)
            return float(np.nan)

        height, width = dem_array.shape[0], dem_array.shape[1]
        if row < 0 or row >= height or col < 0 or col >= width:
            logger.warning(
                "Lat/Lon (%.6f, %.6f) maps to pixel (%d, %d) "
                "outside raster %dx%d.",
                lat, lon, row, col, height, width,
            )
            return float(np.nan)

        value = dem_array[row, col]
        try:
            elev = float(value)
        except (TypeError, ValueError):
            return float(np.nan)

        if nodata is not None and elev == float(nodata):
            return float(np.nan)
        if not np.isfinite(elev):
            return float(np.nan)
        return elev
