from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from lunar_pipeline.models import ImageMetadata, infer_sensor, _local


def _text(el: ET.Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    t = el.text.strip()
    return t or None


def _find_first(root: ET.Element, names: tuple[str, ...]) -> ET.Element | None:
    wanted = {n.lower() for n in names}
    for el in root.iter():
        if _local(el.tag).lower() in wanted:
            return el
    return None


def _find_text(root: ET.Element, names: tuple[str, ...]) -> str | None:
    return _text(_find_first(root, names))


def _find_float(root: ET.Element, names: tuple[str, ...]) -> float | None:
    t = _find_text(root, names)
    if t is None:
        return None
    try:
        return float(t.split()[0])
    except ValueError:
        return None


def _collect_named_floats(root: ET.Element) -> dict[str, float]:
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


def parse_pds4_label(xml_path: Path) -> tuple[ET.Element, ImageMetadata]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    floats = _collect_named_floats(root)

    product_id = _find_text(root, ("logical_identifier", "product_lid", "title")) or xml_path.stem
    instrument = _find_text(
        root,
        ("instrument_id", "instrument_name", "naif_instrument_id", "title", "name"),
    )
    sensor = infer_sensor(f"{instrument or ''} {product_id} {xml_path.name}", str(xml_path))

    lon_w = floats.get("west_bounding_coordinate", floats.get("minimum_longitude"))
    lon_e = floats.get("east_bounding_coordinate", floats.get("maximum_longitude"))
    lat_s = floats.get("south_bounding_coordinate", floats.get("minimum_latitude"))
    lat_n = floats.get("north_bounding_coordinate", floats.get("maximum_latitude"))

    if None in (lon_w, lon_e, lat_s, lat_n):
        lons, lats = [], []
        for el in root.iter():
            tag = _local(el.tag).lower()
            if el.text is None:
                continue
            parts = el.text.strip().split()
            if not parts:
                continue
            try:
                val = float(parts[0])
                if "longitude" in tag or tag in ("long", "lon"):
                    lons.append(val)
                elif "latitude" in tag or tag in ("lat",):
                    lats.append(val)
            except ValueError:
                pass
        if lons and lats:
            lon_w, lon_e = min(lons), max(lons)
            lat_s, lat_n = min(lats), max(lats)

    footprint: dict = {}
    if None not in (lon_w, lon_e, lat_s, lat_n):
        footprint = {
            "west_lon": lon_w,
            "east_lon": lon_e,
            "south_lat": lat_s,
            "north_lat": lat_n,
        }

    incidence = floats.get("incidence_angle", floats.get("solar_incidence"))
    emission = floats.get("emission_angle")
    phase = floats.get("phase_angle")
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

    meta = ImageMetadata(
        product_id=product_id.split(":")[-1],
        sensor=sensor,
        source_path="",
        label_path=str(xml_path),
        acquisition_utc=_find_text(root, ("start_date_time", "start_date_time_utc", "observation_start")),
        gsd_m=gsd,
        sun_azimuth_deg=sun_az,
        sun_elevation_deg=sun_el,
        incidence_deg=incidence,
        emission_deg=emission,
        phase_deg=phase,
        footprint=footprint,
        extra={"parsed_floats": {k: v for k, v in floats.items() if k.lower().endswith("angle") or "resol" in k.lower()}},
    )
    return root, meta


def _array_shape_from_label(root: ET.Element) -> tuple[int, int, int] | None:
    """Return (bands, height, width) if Array_2D/3D is described in the label."""
    axes: dict[str, int] = {}
    for el in root.iter():
        if _local(el.tag) != "Axis_Array":
            continue
        name = None
        elements = None
        for child in el:
            loc = _local(child.tag)
            if loc == "axis_name":
                name = (child.text or "").strip().lower()
            elif loc == "elements":
                try:
                    elements = int(float(child.text.strip()))
                except (TypeError, ValueError, AttributeError):
                    pass
        if name and elements:
            axes[name] = elements
    if "line" in axes and "sample" in axes:
        bands = axes.get("band", axes.get("channel", 1))
        return bands, axes["line"], axes["sample"]
    return None


def _img_path_for_label(xml_path: Path) -> Path | None:
    stem = xml_path.with_suffix("")
    for ext in (".img", ".IMG", ".dat", ".DAT", ".qub", ".QUB"):
        cand = stem.parent / (stem.name + ext)
        if cand.exists():
            return cand
    # PDS4 often uses <file_name> pointing at a sibling
    try:
        root = ET.parse(xml_path).getroot()
        fname = _find_text(root, ("file_name", "md5_checksum"))  # checksum last is unused if not a name
        name_el = _find_first(root, ("file_name",))
        fname = _text(name_el)
        if fname:
            cand = xml_path.parent / fname
            if cand.exists():
                return cand
    except ET.ParseError:
        pass
    return None


def discover_products(input_dir: Path) -> list[Path]:
    """Prefer PDS4 XML labels; also accept GeoTIFF / ISIS cubes already converted."""
    input_dir = Path(input_dir)
    xmls = {p.resolve() for p in list(input_dir.rglob("*.xml")) + list(input_dir.rglob("*.XML"))}
    rasters = []
    for pat in ("*.tif", "*.tiff", "*.TIF", "*.cub", "*.jp2"):
        rasters.extend(input_dir.rglob(pat))
    pds4 = []
    for p in sorted(xmls):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")[:4000].lower()
        except OSError:
            continue
        if "pds" in txt or "product_observational" in txt or "array_2d" in txt:
            if _img_path_for_label(p) is not None:
                pds4.append(p)
    seen = {p.resolve() for p in pds4}
    extras = []
    for p in rasters:
        rp = p.resolve()
        if rp not in seen:
            extras.append(p)
            seen.add(rp)
    return pds4 + extras


def _read_raw_array(xml_path: Path, img_path: Path, root: ET.Element) -> np.ndarray:
    shape = _array_shape_from_label(root)
    dtype = np.float32
    el = _find_first(root, ("data_type", "Element_Array"))
    # nested data_type
    dt_text = _find_text(root, ("data_type",)) or ""
    dt_map = {
        "ieee754msb4": np.dtype(">f4"),
        "ieee754lsb4": np.dtype("<f4"),
        "ieee754msb8": np.dtype(">f8"),
        "unsignedmsb2": np.dtype(">u2"),
        "unsignedlsb2": np.dtype("<u2"),
        "signedmsb2": np.dtype(">i2"),
        "unsignedbyte": np.dtype("u1"),
        "unsignedbit": np.dtype("u1"),
    }
    for key, dt in dt_map.items():
        if key in dt_text.lower().replace("_", ""):
            dtype = dt
            break

    offset = 0
    off_t = _find_text(root, ("offset", "object_length"))
    # offset under File_Area is more reliable
    for el in root.iter():
        if _local(el.tag) == "offset":
            try:
                unit = (el.attrib.get("unit") or "").lower()
                val = int(float(el.text.strip()))
                if unit in ("byte", "bytes", ""):
                    offset = val
                    break
            except (TypeError, ValueError, AttributeError):
                continue

    data = np.fromfile(img_path, dtype=dtype, offset=offset)
    if shape:
        bands, h, w = shape
        n = bands * h * w
        data = data[:n].reshape((bands, h, w) if bands > 1 else (h, w))
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    return data.astype(np.float32)


def open_raster(path: Path) -> tuple[np.ndarray, dict, ImageMetadata]:
    """
    Open a PDS4 label, GeoTIFF, or cube.
    Returns (array[bands,h,w], profile-like dict, metadata).
    """
    path = Path(path)
    profile: dict = {"crs": None, "transform": None, "count": 1}

    if path.suffix.lower() == ".xml":
        root, meta = parse_pds4_label(path)
        img = _img_path_for_label(path)
        opened = False
        arr = None
        # GDAL PDS4 driver typically wants the XML
        for candidate in ([path] + ([img] if img else [])):
            try:
                with rasterio.open(candidate) as src:
                    arr = src.read().astype(np.float32)
                    profile = src.profile.copy()
                    meta.width = src.width
                    meta.height = src.height
                    meta.bands = src.count
                    if meta.gsd_m is None and src.res and src.res[0]:
                        meta.gsd_m = abs(src.res[0])
                    opened = True
                    meta.source_path = str(candidate)
                    break
            except (RasterioIOError, Exception):
                continue
        if not opened:
            if img is None:
                raise FileNotFoundError(f"No raster sidecar for {path}")
            arr = _read_raw_array(path, img, root)
            meta.bands, meta.height, meta.width = arr.shape
            meta.source_path = str(img)
            profile.update({"width": meta.width, "height": meta.height, "count": meta.bands, "dtype": "float32"})
        return arr, profile, meta

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
        meta = ImageMetadata(
            product_id=path.stem,
            sensor=infer_sensor(path.stem, str(path)),
            source_path=str(path),
            width=src.width,
            height=src.height,
            bands=src.count,
            gsd_m=abs(src.res[0]) if src.res and src.res[0] else None,
        )
        sidecar = path.with_suffix(".xml")
        if sidecar.exists():
            _, parsed = parse_pds4_label(sidecar)
            for field in (
                "acquisition_utc",
                "sun_azimuth_deg",
                "sun_elevation_deg",
                "incidence_deg",
                "emission_deg",
                "phase_deg",
                "footprint",
                "gsd_m",
                "sensor",
            ):
                val = getattr(parsed, field)
                if val not in (None, "", {}, []):
                    setattr(meta, field, val)
            meta.label_path = str(sidecar)
            meta.product_id = parsed.product_id
    return arr, profile, meta


def write_sidecar_json(meta: ImageMetadata, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, indent=2)
