"""
data/ingestion/footprint_geometry.py — Geospatially Rigorous Lunar Footprint Overlap

Features:
1. PDS4 XML label polygon vertex and corner point parser with bbox fallback.
2. Lunar geospatial coordinate representations on the IAU lunar datum (R = 1737.4 km).
3. Longitude wrapping and antimeridian crossing support (±180° and 0°..360°).
4. Adaptive projection: Polar Stereographic for polar regions (|lat| >= 65°),
   Equirectangular (Cylindrical with true scale at scene latitude) for equatorial/mid-latitudes.
5. Exact polygon intersection area and overlap percentage calculation via Shapely.
6. Preserves bounding-box overlap as fallback when polygon vertices are unavailable.
7. Overlap confidence categorization:
   - polygon_exact
   - polygon_approximate
   - bbox_only
   - unavailable
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pyproj import CRS, Transformer
import shapely
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import transform as shapely_transform
from shapely import make_valid
from shapely.wkt import loads as wkt_loads


# ---------------------------------------------------------------------------
# Lunar Geospatial Constants (IAU Moon 2000 / 2015 Sphere)
# ---------------------------------------------------------------------------
MOON_RADIUS_M = 1737400.0  # 1,737.4 km mean lunar radius
MOON_GEOG_CRS = "+proj=longlat +a=1737400 +b=1737400 +no_defs +type=crs"


class OverlapConfidence(str, Enum):
    """Categorizes the geometric precision of the footprint overlap computation."""
    POLYGON_EXACT = "polygon_exact"
    POLYGON_APPROXIMATE = "polygon_approximate"
    BBOX_ONLY = "bbox_only"
    UNAVAILABLE = "unavailable"


@dataclass
class FootprintGeometry:
    """Geospatial footprint representation for a lunar raster or observation."""
    vertices: Optional[List[Tuple[float, float]]] = None  # [(lon_deg, lat_deg), ...]
    bbox: Optional[Dict[str, float]] = None  # {"west_lon", "east_lon", "south_lat", "north_lat"}
    wkt: Optional[str] = None
    confidence: OverlapConfidence = OverlapConfidence.UNAVAILABLE
    source_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vertices": self.vertices,
            "bbox": self.bbox,
            "wkt": self.wkt,
            "confidence": self.confidence.value,
            "source_label": self.source_label,
        }


@dataclass
class FootprintOverlapResult:
    """Result of geospatial footprint intersection and overlap computation."""
    intersection_area_km2: float
    area1_km2: float
    area2_km2: float
    overlap_pct1: float  # Percentage of Footprint 1 covered by intersection (0..100)
    overlap_pct2: float  # Percentage of Footprint 2 covered by intersection (0..100)
    iou: float  # Intersection over Union (Jaccard Index, 0..1)
    confidence: OverlapConfidence
    intersection_wkt: Optional[str] = None
    projection_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intersection_area_km2": round(self.intersection_area_km2, 4),
            "area1_km2": round(self.area1_km2, 4),
            "area2_km2": round(self.area2_km2, 4),
            "overlap_pct1": round(self.overlap_pct1, 3),
            "overlap_pct2": round(self.overlap_pct2, 3),
            "iou": round(self.iou, 4),
            "confidence": self.confidence.value,
            "intersection_wkt": self.intersection_wkt,
            "projection_used": self.projection_used,
        }


# ---------------------------------------------------------------------------
# Projection Constructors
# ---------------------------------------------------------------------------
def get_lunar_eqc_crs(lat_ts: float = 0.0, lon_0: float = 0.0) -> str:
    """Returns PROJ string for Lunar Equirectangular (Cylindrical) with true scale at lat_ts."""
    return f"+proj=eqc +lat_ts={lat_ts:.4f} +lat_0=0 +lon_0={lon_0:.4f} +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs +type=crs"


def get_lunar_north_polar_stere_crs(lon_0: float = 0.0) -> str:
    """Returns PROJ string for Lunar North Polar Stereographic."""
    return f"+proj=stere +lat_0=90 +lon_0={lon_0:.4f} +k=1 +x_0=0 +y_0=0 +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs +type=crs"


def get_lunar_south_polar_stere_crs(lon_0: float = 0.0) -> str:
    """Returns PROJ string for Lunar South Polar Stereographic."""
    return f"+proj=stere +lat_0=-90 +lon_0={lon_0:.4f} +k=1 +x_0=0 +y_0=0 +a={MOON_RADIUS_M} +b={MOON_RADIUS_M} +units=m +no_defs +type=crs"


def select_lunar_projection(coords: Sequence[Tuple[float, float]]) -> Tuple[str, str]:
    """
    Selects the optimal lunar projection for a set of (lon, lat) coordinates:
    - Polar Stereographic when |mean_lat| >= 65° or any vertex reaches |lat| >= 80°
    - Equirectangular (Cylindrical) with true scale at mean_lat otherwise.

    Returns:
        (proj4_string, projection_type_name)
    """
    if not coords:
        return get_lunar_eqc_crs(0.0, 0.0), "cylindrical_equirectangular"

    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]

    mean_lat = float(np.mean(lats))
    mean_lon = float(np.mean(lons))
    max_abs_lat = max(abs(l) for l in lats)

    if abs(mean_lat) >= 65.0 or max_abs_lat >= 80.0:
        if mean_lat >= 0:
            return get_lunar_north_polar_stere_crs(mean_lon), "north_polar_stereographic"
        else:
            return get_lunar_south_polar_stere_crs(mean_lon), "south_polar_stereographic"
    else:
        return get_lunar_eqc_crs(lat_ts=mean_lat, lon_0=mean_lon), "cylindrical_equirectangular"


# ---------------------------------------------------------------------------
# Longitude Normalization & Antimeridian Unwrapping
# ---------------------------------------------------------------------------
def normalize_lon_180(lon: float) -> float:
    """Normalizes longitude into [-180, 180) degrees."""
    return ((lon + 180.0) % 360.0) - 180.0


def normalize_lon_360(lon: float) -> float:
    """Normalizes longitude into [0, 360) degrees."""
    return lon % 360.0


def unwrap_antimeridian_longitudes(
    coords: Sequence[Tuple[float, float]],
    ref_lon: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """
    Unwraps longitudes across the ±180° antimeridian relative to a reference longitude.
    Ensures adjacent points separated by the antimeridian do not produce a 350°-wide
    Cartesian tear across the planet.
    """
    if not coords:
        return []

    # If no reference longitude is given, anchor to the first coordinate
    if ref_lon is None:
        ref_lon = coords[0][0]

    unwrapped = []
    for lon, lat in coords:
        # Compute shortest angular offset from ref_lon
        delta = ((lon - ref_lon + 180.0) % 360.0) - 180.0
        unwrapped_lon = ref_lon + delta
        unwrapped.append((unwrapped_lon, lat))
    return unwrapped


# ---------------------------------------------------------------------------
# PDS4 XML Footprint Parser
# ---------------------------------------------------------------------------
def _local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def parse_footprint_from_pds4(
    xml_source: Union[str, Path, ET.Element],
) -> FootprintGeometry:
    """
    Parses footprint geometry from a PDS4 XML label (or file path).
    Tries in precedence:
    1. Bounding_Polygon / Polygon elements with discrete Polygon_Vertex points.
    2. Corner coordinates (upper_left, upper_right, lower_right, lower_left).
    3. WKT polygon strings in geometry attributes.
    4. Fallback: Bounding box coordinates (west, east, south, north).
    """
    if isinstance(xml_source, (str, Path)):
        if isinstance(xml_source, str) and ("<" in xml_source and ">" in xml_source):
            try:
                root = ET.fromstring(xml_source)
                source_label = "in_memory_xml"
            except Exception:
                return FootprintGeometry(confidence=OverlapConfidence.UNAVAILABLE, source_label="in_memory_xml")
        else:
            p = Path(xml_source)
            if not p.exists():
                return FootprintGeometry(confidence=OverlapConfidence.UNAVAILABLE, source_label=str(p))
            try:
                tree = ET.parse(p)
                root = tree.getroot()
                source_label = p.name
            except Exception:
                return FootprintGeometry(confidence=OverlapConfidence.UNAVAILABLE, source_label=str(p))
    else:
        root = xml_source
        source_label = "in_memory_element"

    # Step 1: Look for Bounding_Polygon or Polygon vertex lists
    polygon_vertices = _extract_polygon_vertices(root)
    if polygon_vertices and len(polygon_vertices) >= 3:
        # Ensure polygon ring is closed
        if polygon_vertices[0] != polygon_vertices[-1]:
            polygon_vertices.append(polygon_vertices[0])
        poly_wkt = _coords_to_wkt(polygon_vertices)
        bbox = _bbox_from_coords(polygon_vertices)
        return FootprintGeometry(
            vertices=polygon_vertices,
            bbox=bbox,
            wkt=poly_wkt,
            confidence=OverlapConfidence.POLYGON_EXACT,
            source_label=source_label,
        )

    # Step 2: Look for 4 corner coordinates (UL, UR, LR, LL)
    corner_vertices = _extract_corner_points(root)
    if corner_vertices and len(corner_vertices) >= 4:
        if corner_vertices[0] != corner_vertices[-1]:
            corner_vertices.append(corner_vertices[0])
        poly_wkt = _coords_to_wkt(corner_vertices)
        bbox = _bbox_from_coords(corner_vertices)
        return FootprintGeometry(
            vertices=corner_vertices,
            bbox=bbox,
            wkt=poly_wkt,
            confidence=OverlapConfidence.POLYGON_EXACT,
            source_label=source_label,
        )

    # Step 3: Look for embedded WKT
    wkt_str = _extract_wkt(root)
    if wkt_str:
        try:
            geom = wkt_loads(wkt_str)
            if isinstance(geom, Polygon):
                coords = list(geom.exterior.coords)
                bbox = _bbox_from_coords(coords)
                return FootprintGeometry(
                    vertices=coords,
                    bbox=bbox,
                    wkt=wkt_str,
                    confidence=OverlapConfidence.POLYGON_EXACT,
                    source_label=source_label,
                )
        except Exception:
            pass

    # Step 4: Fallback to Bounding Box
    bbox = _extract_bounding_box(root)
    if bbox is not None:
        w, e, s, n = bbox["west_lon"], bbox["east_lon"], bbox["south_lat"], bbox["north_lat"]
        bbox_ring = [(w, s), (e, s), (e, n), (w, n), (w, s)]
        return FootprintGeometry(
            vertices=bbox_ring,
            bbox=bbox,
            wkt=_coords_to_wkt(bbox_ring),
            confidence=OverlapConfidence.BBOX_ONLY,
            source_label=source_label,
        )

    return FootprintGeometry(confidence=OverlapConfidence.UNAVAILABLE, source_label=source_label)


def _extract_polygon_vertices(root: ET.Element) -> List[Tuple[float, float]]:
    """Extracts (lon, lat) vertex list from Bounding_Polygon / Polygon elements."""
    vertices: List[Tuple[float, float]] = []

    for el in root.iter():
        tag = _local_tag(el.tag).lower()
        if tag in ("bounding_polygon", "polygon", "surface_geometry", "footprint_polygon"):
            child_list = list(el)
            has_vertex_containers = any(
                _local_tag(c.tag).lower() in ("polygon_vertex", "coordinate", "point", "vertex")
                for c in child_list
            )
            if has_vertex_containers:
                for child in child_list:
                    c_tag = _local_tag(child.tag).lower()
                    if c_tag in ("polygon_vertex", "coordinate", "point", "vertex"):
                        lat, lon = None, None
                        for pt in child:
                            p_tag = _local_tag(pt.tag).lower()
                            val_str = (pt.text or "").strip().split()
                            if not val_str:
                                continue
                            try:
                                val = float(val_str[0])
                                if "lat" in p_tag:
                                    lat = val
                                elif "lon" in p_tag or "long" in p_tag:
                                    lon = val
                            except ValueError:
                                pass
                        if lat is not None and lon is not None:
                            vertices.append((lon, lat))
            else:
                # Direct alternating lat/lon children (e.g. <pixel_latitude>, <pixel_longitude>)
                curr_lat = None
                for child in child_list:
                    c_tag = _local_tag(child.tag).lower()
                    val_str = (child.text or "").strip().split()
                    if not val_str:
                        continue
                    try:
                        val = float(val_str[0])
                        if "lat" in c_tag:
                            curr_lat = val
                        elif ("lon" in c_tag or "long" in c_tag) and curr_lat is not None:
                            vertices.append((val, curr_lat))
                            curr_lat = None
                    except ValueError:
                        pass
            if len(vertices) >= 3:
                return vertices

    return vertices


def _extract_corner_points(root: ET.Element) -> List[Tuple[float, float]]:
    """Extracts corner points from named tags or Corner_Point elements."""
    # First check for repeated Corner_Point / Corner_Position elements
    corner_pts: List[Tuple[float, float]] = []
    for el in root.iter():
        tag = _local_tag(el.tag).lower()
        if tag in ("corner_point", "corner_position", "point"):
            lat, lon = None, None
            for child in el:
                c_tag = _local_tag(child.tag).lower()
                text = (child.text or "").strip().split()
                if text:
                    try:
                        val = float(text[0])
                        if "lat" in c_tag:
                            lat = val
                        elif "lon" in c_tag or "long" in c_tag:
                            lon = val
                    except ValueError:
                        pass
            if lat is not None and lon is not None:
                corner_pts.append((lon, lat))
    if len(corner_pts) >= 3:
        return corner_pts

    # Fallback to tag map for named corner coordinates
    tag_map: Dict[str, float] = {}
    for el in root.iter():
        tag = _local_tag(el.tag).lower()
        text = (el.text or "").strip().split()
        if text:
            try:
                tag_map[tag] = float(text[0])
            except ValueError:
                pass

    ul_lat = tag_map.get("upper_left_latitude")
    ul_lon = tag_map.get("upper_left_longitude")
    ur_lat = tag_map.get("upper_right_latitude")
    ur_lon = tag_map.get("upper_right_longitude")
    lr_lat = tag_map.get("lower_right_latitude")
    lr_lon = tag_map.get("lower_right_longitude")
    ll_lat = tag_map.get("lower_left_latitude")
    ll_lon = tag_map.get("lower_left_longitude")

    if None not in (ul_lat, ul_lon, ur_lat, ur_lon, lr_lat, lr_lon, ll_lat, ll_lon):
        # Order in standard counter-clockwise polygon loop: UL -> UR -> LR -> LL -> UL
        return [
            (float(ul_lon), float(ul_lat)),
            (float(ur_lon), float(ur_lat)),
            (float(lr_lon), float(lr_lat)),
            (float(ll_lon), float(ll_lat)),
        ]
    return []


def _extract_wkt(root: ET.Element) -> Optional[str]:
    """Finds WKT footprint string in geometry tags."""
    for el in root.iter():
        tag = _local_tag(el.tag).lower()
        if tag in ("polygon_wkt", "wkt", "footprint_wkt"):
            t = (el.text or "").strip()
            if "POLYGON" in t.upper():
                return t
    return None


def _extract_bounding_box(root: ET.Element) -> Optional[Dict[str, float]]:
    """Extracts bounding coordinates if available."""
    tag_map: Dict[str, float] = {}
    for el in root.iter():
        tag = _local_tag(el.tag).lower()
        text = (el.text or "").strip().split()
        if text:
            try:
                tag_map[tag] = float(text[0])
            except ValueError:
                pass

    w = tag_map.get("west_bounding_coordinate", tag_map.get("minimum_longitude"))
    e = tag_map.get("east_bounding_coordinate", tag_map.get("maximum_longitude"))
    s = tag_map.get("south_bounding_coordinate", tag_map.get("minimum_latitude"))
    n = tag_map.get("north_bounding_coordinate", tag_map.get("maximum_latitude"))

    if None not in (w, e, s, n):
        return {
            "west_lon": float(w),
            "east_lon": float(e),
            "south_lat": float(s),
            "north_lat": float(n),
        }
    return None


def _bbox_from_coords(coords: Sequence[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return {
        "west_lon": float(min(lons)),
        "east_lon": float(max(lons)),
        "south_lat": float(min(lats)),
        "north_lat": float(max(lats)),
    }


def _coords_to_wkt(coords: Sequence[Tuple[float, float]]) -> str:
    ring_str = ", ".join(f"{c[0]:.6f} {c[1]:.6f}" for c in coords)
    return f"POLYGON (({ring_str}))"


# ---------------------------------------------------------------------------
# Polygon Projection & Area Intersection
# ---------------------------------------------------------------------------
def _coerce_to_footprint(
    fp: Union[FootprintGeometry, Dict[str, Any], Sequence[Tuple[float, float]], str],
) -> FootprintGeometry:
    """Coerces various footprint input formats into FootprintGeometry."""
    if isinstance(fp, FootprintGeometry):
        return fp

    if isinstance(fp, dict):
        if "vertices" in fp and fp["vertices"]:
            coords = [(float(c[0]), float(c[1])) for c in fp["vertices"]]
            return FootprintGeometry(
                vertices=coords,
                bbox=fp.get("bbox") or _bbox_from_coords(coords),
                wkt=_coords_to_wkt(coords),
                confidence=OverlapConfidence(fp.get("confidence", OverlapConfidence.POLYGON_EXACT)),
            )
        elif all(k in fp for k in ("west_lon", "east_lon", "south_lat", "north_lat")):
            w, e, s, n = float(fp["west_lon"]), float(fp["east_lon"]), float(fp["south_lat"]), float(fp["north_lat"])
            ring = [(w, s), (e, s), (e, n), (w, n), (w, s)]
            return FootprintGeometry(
                vertices=ring,
                bbox={"west_lon": w, "east_lon": e, "south_lat": s, "north_lat": n},
                wkt=_coords_to_wkt(ring),
                confidence=OverlapConfidence.BBOX_ONLY,
            )

    if isinstance(fp, (list, tuple)):
        if not fp:
            return FootprintGeometry(confidence=OverlapConfidence.UNAVAILABLE)
        coords = [(float(c[0]), float(c[1])) for c in fp]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        return FootprintGeometry(
            vertices=coords,
            bbox=_bbox_from_coords(coords),
            wkt=_coords_to_wkt(coords),
            confidence=OverlapConfidence.POLYGON_EXACT,
        )

    if isinstance(fp, str):
        try:
            geom = wkt_loads(fp)
            if isinstance(geom, Polygon):
                coords = list(geom.exterior.coords)
                return FootprintGeometry(
                    vertices=coords,
                    bbox=_bbox_from_coords(coords),
                    wkt=fp,
                    confidence=OverlapConfidence.POLYGON_EXACT,
                )
        except Exception:
            pass

    return FootprintGeometry(confidence=OverlapConfidence.UNAVAILABLE)


def compute_footprint_overlap(
    fp1_input: Union[FootprintGeometry, Dict[str, Any], Sequence[Tuple[float, float]], str],
    fp2_input: Union[FootprintGeometry, Dict[str, Any], Sequence[Tuple[float, float]], str],
    confidence1: Optional[OverlapConfidence] = None,
    confidence2: Optional[OverlapConfidence] = None,
) -> FootprintOverlapResult:
    """
    Computes geospatially rigorous footprint overlap between two lunar observations:
    1. Selects adaptive lunar projection (Polar Stereographic for polar regions,
       Equirectangular with true scale at scene latitude for equatorial/mid-latitude).
    2. Unwraps antimeridian crossings continuous across ±180° / 360°.
    3. Projects both geometries to the shared metric CRS.
    4. Computes exact polygon intersection area in km².
    5. Falls back to bounding-box overlap when polygon vertices are missing.
    6. Returns overlap percentages, IoU, and confidence indicator.
    """
    fp1 = _coerce_to_footprint(fp1_input)
    fp2 = _coerce_to_footprint(fp2_input)

    if confidence1 is not None:
        fp1.confidence = confidence1
    if confidence2 is not None:
        fp2.confidence = confidence2

    # If either footprint is unavailable, return unavailable result
    if fp1.confidence == OverlapConfidence.UNAVAILABLE or fp2.confidence == OverlapConfidence.UNAVAILABLE:
        return FootprintOverlapResult(
            intersection_area_km2=0.0,
            area1_km2=0.0,
            area2_km2=0.0,
            overlap_pct1=0.0,
            overlap_pct2=0.0,
            iou=0.0,
            confidence=OverlapConfidence.UNAVAILABLE,
            intersection_wkt=None,
            projection_used=None,
        )

    if not fp1.vertices or not fp2.vertices:
        return FootprintOverlapResult(
            intersection_area_km2=0.0,
            area1_km2=0.0,
            area2_km2=0.0,
            overlap_pct1=0.0,
            overlap_pct2=0.0,
            iou=0.0,
            confidence=OverlapConfidence.UNAVAILABLE,
        )

    # Determine confidence classification:
    # If either footprint is only a bounding box fallback, the overlap calculation is bbox_only
    if fp1.confidence == OverlapConfidence.BBOX_ONLY or fp2.confidence == OverlapConfidence.BBOX_ONLY:
        overall_conf = OverlapConfidence.BBOX_ONLY
    elif fp1.confidence == OverlapConfidence.POLYGON_APPROXIMATE or fp2.confidence == OverlapConfidence.POLYGON_APPROXIMATE:
        overall_conf = OverlapConfidence.POLYGON_APPROXIMATE
    else:
        overall_conf = OverlapConfidence.POLYGON_EXACT

    # Select joint projection based on all combined coordinates
    all_coords = fp1.vertices + fp2.vertices
    proj4_str, proj_type = select_lunar_projection(all_coords)

    # Unwrap antimeridian crossings relative to the shared mean longitude
    mean_lon = float(np.mean([c[0] for c in all_coords]))
    v1_unwrapped = unwrap_antimeridian_longitudes(fp1.vertices, ref_lon=mean_lon)
    v2_unwrapped = unwrap_antimeridian_longitudes(fp2.vertices, ref_lon=mean_lon)

    # Build geographical polygons
    p1_geog = make_valid(Polygon(v1_unwrapped))
    p2_geog = make_valid(Polygon(v2_unwrapped))

    if p1_geog.is_empty or p2_geog.is_empty:
        return FootprintOverlapResult(
            intersection_area_km2=0.0,
            area1_km2=0.0,
            area2_km2=0.0,
            overlap_pct1=0.0,
            overlap_pct2=0.0,
            iou=0.0,
            confidence=overall_conf,
            projection_used=proj_type,
        )

    # Project to metric lunar CRS
    crs_geog = CRS.from_string(MOON_GEOG_CRS)
    crs_proj = CRS.from_string(proj4_str)
    transformer_fwd = Transformer.from_crs(crs_geog, crs_proj, always_xy=True)
    transformer_inv = Transformer.from_crs(crs_proj, crs_geog, always_xy=True)

    def _project_forward(x, y):
        return transformer_fwd.transform(x, y)

    def _project_inverse(x, y):
        return transformer_inv.transform(x, y)

    p1_proj = make_valid(shapely_transform(_project_forward, p1_geog))
    p2_proj = make_valid(shapely_transform(_project_forward, p2_geog))

    # Calculate individual projected areas in km²
    area1_km2 = float(p1_proj.area) / 1e6
    area2_km2 = float(p2_proj.area) / 1e6

    if area1_km2 <= 0 or area2_km2 <= 0:
        return FootprintOverlapResult(
            intersection_area_km2=0.0,
            area1_km2=max(0.0, area1_km2),
            area2_km2=max(0.0, area2_km2),
            overlap_pct1=0.0,
            overlap_pct2=0.0,
            iou=0.0,
            confidence=overall_conf,
            projection_used=proj_type,
        )

    # Calculate intersection
    inter_proj = make_valid(p1_proj.intersection(p2_proj))
    inter_area_km2 = float(inter_proj.area) / 1e6 if not inter_proj.is_empty else 0.0

    # Overlap percentages
    overlap_pct1 = (inter_area_km2 / area1_km2) * 100.0 if area1_km2 > 0 else 0.0
    overlap_pct2 = (inter_area_km2 / area2_km2) * 100.0 if area2_km2 > 0 else 0.0
    overlap_pct1 = min(100.0, max(0.0, overlap_pct1))
    overlap_pct2 = min(100.0, max(0.0, overlap_pct2))

    # IoU
    union_area = area1_km2 + area2_km2 - inter_area_km2
    iou = (inter_area_km2 / union_area) if union_area > 0 else 0.0

    # Convert intersection polygon back to geographical WKT
    inter_wkt = None
    if not inter_proj.is_empty and inter_area_km2 > 0:
        inter_geog = make_valid(shapely_transform(_project_inverse, inter_proj))
        if not inter_geog.is_empty:
            inter_wkt = inter_geog.wkt

    return FootprintOverlapResult(
        intersection_area_km2=inter_area_km2,
        area1_km2=area1_km2,
        area2_km2=area2_km2,
        overlap_pct1=overlap_pct1,
        overlap_pct2=overlap_pct2,
        iou=iou,
        confidence=overall_conf,
        intersection_wkt=inter_wkt,
        projection_used=proj_type,
    )
