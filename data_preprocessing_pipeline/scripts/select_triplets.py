#!/usr/bin/env python3
"""
Select overlapping Chandrayaan-2 OHRC + TMC-2 + IIRS triplets from PDS4 labels.

Metadata only: reads XML labels (and optionally raster bounds via rasterio).
Does not load pixel arrays.

Example:
  python scripts/select_triplets.py path/to/labels --out triplets.csv --json triplets.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

LOG = logging.getLogger("select_triplets")

# Lunar planetocentric lon/lat, then equirectangular meters for area ratios.
GEO_CRS = "+proj=longlat +a=1737400 +b=1737400 +no_defs +type=crs"
MOON_EQC = "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs"

SENSOR_ALIASES = (
    ("ohrc", "OHRC"),
    ("optical high resolution camera", "OHRC"),
    ("orbiter high resolution camera", "OHRC"),
    ("ohr", "OHRC"),
    ("tmc-2", "TMC-2"),
    ("tmc2", "TMC-2"),
    ("terrain mapping camera-2", "TMC-2"),
    ("terrain mapping camera 2", "TMC-2"),
    ("tmc", "TMC-2"),
    ("iirs", "IIRS"),
    ("iir", "IIRS"),
    ("imaging infrared spectrometer", "IIRS"),
)

def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def infer_sensor(text: str, path_hint: str = "") -> str | None:
    filename = Path(path_hint).name.lower() if path_hint else ""
    for blob in (text.lower(), filename, path_hint.lower()):
        if not blob:
            continue
        for key, name in SENSOR_ALIASES:
            if key in blob:
                return name
    return None


def _el_text(el: ET.Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    t = el.text.strip()
    return t or None


def _first_text(root: ET.Element, names: tuple[str, ...]) -> str | None:
    wanted = {n.lower() for n in names}
    for el in root.iter():
        if _local(el.tag).lower() in wanted:
            t = _el_text(el)
            if t:
                return t
    return None


def _named_floats(root: ET.Element) -> dict[str, float]:
    out: dict[str, float] = {}
    for el in root.iter():
        name = _local(el.tag)
        if el.text is None:
            continue
        parts = el.text.strip().split()
        if not parts:
            continue
        try:
            out[name] = float(parts[0])
        except ValueError:
            continue
    return out


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1]
    if "+" in s[1:]:
        s = s.rsplit("+", 1)[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    LOG.warning("Unparseable acquisition date %r — leaving empty", value)
    return None


def _lonlat_pairs_from_lists(root: ET.Element) -> list[tuple[float, float]]:
    """Collect sibling longitude/latitude pairs in document order (PDS4 corner lists)."""
    lons: list[float] = []
    lats: list[float] = []
    for el in root.iter():
        name = _local(el.tag).lower()
        t = _el_text(el)
        if t is None:
            continue
        try:
            v = float(t.split()[0])
        except ValueError:
            continue
        if name in {"longitude", "long", "lon"} or name.endswith("_longitude"):
            lons.append(v)
        elif name in {"latitude", "lat"} or name.endswith("_latitude"):
            lats.append(v)
    n = min(len(lons), len(lats))
    if n >= 3:
        return list(zip(lons[:n], lats[:n]))
    return []


def _bbox_polygon(floats: dict[str, float]) -> Polygon | None:
    west = floats.get("west_bounding_coordinate", floats.get("minimum_longitude"))
    east = floats.get("east_bounding_coordinate", floats.get("maximum_longitude"))
    south = floats.get("south_bounding_coordinate", floats.get("minimum_latitude"))
    north = floats.get("north_bounding_coordinate", floats.get("maximum_latitude"))
    if None in (west, east, south, north):
        return None
    # Antimeridian: east of west after unwrapping.
    if east < west:
        east += 360.0
    if north < south:
        south, north = north, south
    poly = box(west, south, east, north)
    return poly if poly.area > 0 else None


def _close_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = list(dict.fromkeys(coords))
    if len(unique) < 3:
        return unique
    cx = sum(p[0] for p in unique) / len(unique)
    cy = sum(p[1] for p in unique) / len(unique)
    sorted_p = sorted(unique, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    if sorted_p[0] != sorted_p[-1]:
        sorted_p.append(sorted_p[0])
    return sorted_p


def footprint_polygon(root: ET.Element, floats: dict[str, float]) -> Polygon | None:
    """
    Build a lon/lat polygon from PDS4 corners or bounding coordinates.
    Prefer explicit corners when at least 3 unique points exist.
    """
    pairs = _lonlat_pairs_from_lists(root)
    unique = list(dict.fromkeys(pairs))
    if len(unique) >= 3:
        try:
            poly = Polygon(_close_ring(unique))
            poly = make_valid(poly)
            if poly.is_empty:
                return None
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            if poly.area > 0:
                return poly
        except (ValueError, TypeError) as exc:
            LOG.debug("Corner polygon failed: %s", exc)

    return _bbox_polygon(floats)


def raster_bounds_polygon(xml_path: Path) -> Polygon | None:
    """Optional fallback: GDAL/rasterio geotransform bounds if the label is georeferenced."""
    try:
        import rasterio
    except ImportError:
        return None

    candidates = [xml_path]
    stem = xml_path.with_suffix("")
    for ext in (".img", ".IMG", ".qub", ".QUB", ".tif", ".tiff", ".cub"):
        cand = stem.parent / (stem.name + ext)
        if cand.exists():
            candidates.append(cand)

    for path in candidates:
        try:
            with rasterio.open(path) as src:
                if src.crs is None or src.transform is None:
                    continue
                b = src.bounds
                poly = box(b.left, b.bottom, b.right, b.top)
                if poly.area > 0:
                    LOG.debug("Footprint from raster bounds: %s", path.name)
                    return poly
        except Exception:
            continue
    return None


def parse_label(xml_path: Path) -> dict | None:
    """Parse one PDS4 XML label. Returns None if the product cannot be used."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        LOG.warning("Skip %s: malformed XML (%s)", xml_path, exc)
        return None

    floats = _named_floats(root)
    product_id = _first_text(root, ("logical_identifier", "product_lid", "title")) or xml_path.stem
    instrument = _first_text(root, ("instrument_id", "instrument_name", "naif_instrument_id", "title", "name")) or ""
    sensor = infer_sensor(f"{instrument} {product_id}", str(xml_path))
    if sensor is None:
        LOG.warning("Skip %s: could not infer sensor (OHRC / TMC-2 / IIRS)", xml_path)
        return None

    geom = footprint_polygon(root, floats) or raster_bounds_polygon(xml_path)
    if geom is None or geom.is_empty:
        LOG.warning("Skip %s: missing or empty footprint", xml_path)
        return None

    incidence = floats.get("incidence_angle")
    sun_az = floats.get("solar_azimuth", floats.get("sun_azimuth"))
    sun_el = floats.get("solar_elevation", floats.get("sun_elevation"))
    if sun_el is None and incidence is not None:
        sun_el = 90.0 - incidence

    gsd = (
        floats.get("pixel_resolution_x")
        or floats.get("pixel_resolution")
        or floats.get("horizontal_pixel_scale")
        or floats.get("map_scale")
        or floats.get("pixel_resolution_y")
    )
    date_raw = _first_text(root, ("start_date_time", "start_date_time_utc", "observation_start"))
    acquired = parse_datetime(date_raw)

    return {
        "product_id": product_id.split(":")[-1],
        "sensor": sensor,
        "label_path": str(xml_path),
        "sun_azimuth_deg": sun_az,
        "sun_elevation_deg": sun_el,
        "incidence_deg": incidence,
        "gsd_m": gsd,
        "acquisition_utc": acquired.isoformat() if acquired else date_raw,
        "acquisition_dt": acquired,
        "geometry": geom,
    }


def discover_labels(label_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in ("*.xml", "*.XML"):
        for p in label_dir.rglob(pat):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(p)
    return sorted(out)


def load_catalog(label_dir: Path) -> gpd.GeoDataFrame:
    rows = []
    for path in discover_labels(label_dir):
        rec = parse_label(path)
        if rec is not None:
            rows.append(rec)
    if not rows:
        raise SystemExit(f"No usable PDS4 footprints under {label_dir}")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=GEO_CRS)
    gdf = gdf.drop_duplicates(subset=["product_id", "sensor"], keep="first").reset_index(drop=True)
    LOG.info(
        "Loaded %d products (%s)",
        len(gdf),
        ", ".join(f"{s}={n}" for s, n in gdf["sensor"].value_counts().items()),
    )
    return gdf


def overlap_fraction(anchor, other) -> float:
    """intersection(anchor, other) / area(anchor). Uses projected geometries."""
    if anchor.is_empty or other.is_empty or anchor.area <= 0:
        return 0.0
    inter = anchor.intersection(other)
    if inter.is_empty:
        return 0.0
    return float(inter.area / anchor.area)


def _sjoin_or_strtree(
    anchors: gpd.GeoDataFrame,
    others: gpd.GeoDataFrame,
    containment: float,
) -> pd.DataFrame:
    """
    Candidate pairs where overlap = |A ∩ B| / |A| >= containment.
    Prefers geopandas spatial join; STRtree is the fallback index.
    """
    if anchors.empty or others.empty:
        return pd.DataFrame(columns=["anchor_idx", "other_idx", "overlap"])

    a_m = anchors.to_crs(MOON_EQC)
    b_m = others.to_crs(MOON_EQC)
    pairs: list[tuple[int, int, float]] = []

    try:
        joined = gpd.sjoin(a_m[["geometry"]], b_m[["geometry"]], how="inner", predicate="intersects")
        for a_idx, row in joined.iterrows():
            b_idx = int(row["index_right"])
            frac = overlap_fraction(a_m.geometry.loc[a_idx], b_m.geometry.loc[b_idx])
            if frac >= containment:
                pairs.append((int(a_idx), b_idx, frac))
    except Exception as exc:
        LOG.warning("sjoin failed (%s); using STRtree", exc)
        tree = STRtree(list(b_m.geometry.values))
        b_index = list(b_m.index)
        for a_idx, geom in a_m.geometry.items():
            for hit in tree.query(geom, predicate="intersects"):
                b_idx = b_index[int(hit)]
                frac = overlap_fraction(geom, b_m.geometry.loc[b_idx])
                if frac >= containment:
                    pairs.append((int(a_idx), int(b_idx), frac))

    return pd.DataFrame(pairs, columns=["anchor_idx", "other_idx", "overlap"])


def _days_apart(a: datetime | None, b: datetime | None) -> float | None:
    if a is None or b is None:
        return None
    return abs((a - b).total_seconds()) / 86400.0


def time_ok(ohrc_dt, tmc_dt, iirs_dt, min_gap: float, max_gap: float, require: bool) -> bool:
    dates = [("TMC-2", tmc_dt), ("IIRS", iirs_dt)]
    for name, dt in dates:
        gap = _days_apart(ohrc_dt, dt)
        if gap is None:
            if require:
                return False
            continue
        if gap < min_gap or gap > max_gap:
            return False
    # Also constrain TMC vs IIRS if both dated
    gap_ti = _days_apart(tmc_dt, iirs_dt)
    if gap_ti is not None and (gap_ti < min_gap or gap_ti > max_gap):
        return False
    return True


def build_triplets(
    gdf: gpd.GeoDataFrame,
    containment: float,
    min_gap: float,
    max_gap: float,
    require_dates: bool,
) -> list[dict]:
    ohrc = gdf[gdf["sensor"] == "OHRC"].copy()
    tmc = gdf[gdf["sensor"] == "TMC-2"].copy()
    iirs = gdf[gdf["sensor"] == "IIRS"].copy()
    if ohrc.empty:
        raise SystemExit("No OHRC labels with footprints — nothing to anchor on")
    if tmc.empty or iirs.empty:
        LOG.warning("Need both TMC-2 and IIRS; TMC-2=%d IIRS=%d", len(tmc), len(iirs))

    tmc_hits = _sjoin_or_strtree(ohrc, tmc, containment)
    iirs_hits = _sjoin_or_strtree(ohrc, iirs, containment)
    LOG.info("OHRC–TMC-2 pairs above threshold: %d", len(tmc_hits))
    LOG.info("OHRC–IIRS pairs above threshold: %d", len(iirs_hits))

    tmc_by_ohrc: dict[int, list[tuple[int, float]]] = {}
    for _, r in tmc_hits.iterrows():
        tmc_by_ohrc.setdefault(int(r.anchor_idx), []).append((int(r.other_idx), float(r.overlap)))
    iirs_by_ohrc: dict[int, list[tuple[int, float]]] = {}
    for _, r in iirs_hits.iterrows():
        iirs_by_ohrc.setdefault(int(r.anchor_idx), []).append((int(r.other_idx), float(r.overlap)))

    ohrc_m = ohrc.to_crs(MOON_EQC)
    tmc_m = tmc.to_crs(MOON_EQC)
    iirs_m = iirs.to_crs(MOON_EQC)

    triplets: list[dict] = []
    for a_idx, tmc_list in tmc_by_ohrc.items():
        iirs_list = iirs_by_ohrc.get(a_idx, [])
        if not iirs_list:
            continue
        a_row = ohrc.loc[a_idx]
        a_geom_m = ohrc_m.geometry.loc[a_idx]
        a_geom_ll = ohrc.geometry.loc[a_idx]
        for t_idx, t_ov in tmc_list:
            t_row = tmc.loc[t_idx]
            t_geom_m = tmc_m.geometry.loc[t_idx]
            for i_idx, i_ov in iirs_list:
                i_row = iirs.loc[i_idx]
                if not time_ok(
                    a_row["acquisition_dt"],
                    t_row["acquisition_dt"],
                    i_row["acquisition_dt"],
                    min_gap,
                    max_gap,
                    require_dates,
                ):
                    continue
                i_geom_m = iirs_m.geometry.loc[i_idx]
                triple_inter_m = a_geom_m.intersection(t_geom_m).intersection(i_geom_m)
                if triple_inter_m.is_empty or a_geom_m.area <= 0:
                    continue
                triple_frac = float(triple_inter_m.area / a_geom_m.area)
                if triple_frac < containment:
                    continue
                # Intersection polygon in lon/lat for the manifest WKT
                triple_ll = a_geom_ll.intersection(tmc.geometry.loc[t_idx]).intersection(iirs.geometry.loc[i_idx])
                triple_ll = make_valid(triple_ll)
                if triple_ll.is_empty:
                    continue
                triplets.append(
                    {
                        "ohrc_product_id": a_row["product_id"],
                        "tmc2_product_id": t_row["product_id"],
                        "iirs_product_id": i_row["product_id"],
                        "overlap_ohrc_tmc_pct": round(t_ov * 100.0, 3),
                        "overlap_ohrc_iirs_pct": round(i_ov * 100.0, 3),
                        "overlap_triplet_pct": round(triple_frac * 100.0, 3),
                        "ohrc_sun_azimuth_deg": a_row["sun_azimuth_deg"],
                        "ohrc_sun_elevation_deg": a_row["sun_elevation_deg"],
                        "ohrc_incidence_deg": a_row["incidence_deg"],
                        "tmc2_sun_azimuth_deg": t_row["sun_azimuth_deg"],
                        "tmc2_sun_elevation_deg": t_row["sun_elevation_deg"],
                        "tmc2_incidence_deg": t_row["incidence_deg"],
                        "iirs_sun_azimuth_deg": i_row["sun_azimuth_deg"],
                        "iirs_sun_elevation_deg": i_row["sun_elevation_deg"],
                        "iirs_incidence_deg": i_row["incidence_deg"],
                        "ohrc_gsd_m": a_row["gsd_m"],
                        "tmc2_gsd_m": t_row["gsd_m"],
                        "iirs_gsd_m": i_row["gsd_m"],
                        "ohrc_acquisition_utc": a_row["acquisition_utc"],
                        "tmc2_acquisition_utc": t_row["acquisition_utc"],
                        "iirs_acquisition_utc": i_row["acquisition_utc"],
                        "ohrc_label": a_row["label_path"],
                        "tmc2_label": t_row["label_path"],
                        "iirs_label": i_row["label_path"],
                        "intersection_wkt": triple_ll.wkt,
                        "_ohrc_geom": a_geom_ll,
                        "_ohrc_sun_el": a_row["sun_elevation_deg"],
                        "_score": triple_frac,
                    }
                )
    triplets.sort(key=lambda r: r["_score"], reverse=True)
    LOG.info("Candidate triplets after containment + time filters: %d", len(triplets))
    return triplets


def _sun_el_spread(el_a, el_b) -> float:
    if el_a is None or el_b is None:
        return math.inf
    return abs(float(el_a) - float(el_b))


def dedup_triplets(
    triplets: list[dict],
    dedup_overlap: float,
    min_sun_el_diff: float,
    max_per_region: int,
) -> list[dict]:
    """
    Greedy spatial diversity: OHRC footprints that overlap each other above
    `dedup_overlap` share a region. Keep up to max_per_region triplets per
    region, and only add another in-region triplet if its OHRC sun elevation
    differs from already-kept ones by at least `min_sun_el_diff`.
    """
    if dedup_overlap <= 0 and max_per_region <= 0:
        return triplets

    kept: list[dict] = []
    # Each cluster: list of kept OHRC geoms + sun els
    clusters: list[dict] = []

    for rec in triplets:
        geom = rec["_ohrc_geom"]
        sun_el = rec["_ohrc_sun_el"]
        host = None
        for c in clusters:
            frac = overlap_fraction(
                geom,
                unary_union(c["geoms"]),
            )
            # Compare in lon/lat; ratio is still a containment proxy for near-identical tiles
            if frac >= dedup_overlap:
                host = c
                break
        if host is None:
            clusters.append({"geoms": [geom], "sun": [sun_el], "n": 1})
            kept.append(rec)
            continue
        if max_per_region > 0 and host["n"] >= max_per_region:
            continue
        if min_sun_el_diff > 0:
            if all(_sun_el_spread(sun_el, s) < min_sun_el_diff for s in host["sun"]):
                continue
        host["geoms"].append(geom)
        host["sun"].append(sun_el)
        host["n"] += 1
        kept.append(rec)
    LOG.info("Triplets after spatial/illumination dedup: %d", len(kept))
    return kept


def strip_internal(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find OHRC + TMC-2 + IIRS triplets whose footprints overlap the same lunar ground."
    )
    p.add_argument("label_dir", type=Path, help="Directory of mixed PDS4 XML labels (recursive)")
    p.add_argument("--out", type=Path, default=Path("triplets.csv"), help="Manifest CSV path")
    p.add_argument("--json", dest="json_path", type=Path, default=None, help="Optional JSON manifest")
    p.add_argument(
        "--containment",
        type=float,
        default=0.8,
        help="Min intersection(OHRC, other) / OHRC area, and the three-way intersection ratio (default 0.8)",
    )
    p.add_argument("--min-time-gap-days", type=float, default=0.0, help="Minimum |Δt| between sensors in a triplet")
    p.add_argument("--max-time-gap-days", type=float, default=1e9, help="Maximum |Δt| between sensors in a triplet")
    p.add_argument("--require-dates", action="store_true", help="Drop triplets missing any acquisition date")
    p.add_argument(
        "--min-sun-el-diff",
        type=float,
        default=0.0,
        help="When two OHRCs cover the same region, keep the extra triplet only if sun elevation differs by this many degrees",
    )
    p.add_argument(
        "--dedup-overlap",
        type=float,
        default=0.5,
        help="OHRC-vs-OHRC overlap fraction that counts as the same ground region (0 disables clustering)",
    )
    p.add_argument(
        "--max-per-region",
        type=int,
        default=0,
        help="Cap triplets per overlapping OHRC region (0 = unlimited)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict]:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    gdf = load_catalog(args.label_dir)
    raw = build_triplets(
        gdf,
        containment=args.containment,
        min_gap=args.min_time_gap_days,
        max_gap=args.max_time_gap_days,
        require_dates=args.require_dates,
    )
    selected = dedup_triplets(
        raw,
        dedup_overlap=args.dedup_overlap,
        min_sun_el_diff=args.min_sun_el_diff,
        max_per_region=args.max_per_region,
    )
    rows = [strip_internal(r) for r in selected]
    write_csv(args.out, rows)
    if args.json_path:
        write_json(args.json_path, rows)
    LOG.info("Wrote %d triplets -> %s", len(rows), args.out)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
