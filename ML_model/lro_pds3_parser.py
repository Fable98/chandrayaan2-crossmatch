"""
ML_model/lro_pds3_parser.py — PDS3 Label Parser for LRO NAC Products

Parses PDS3 formatted label files (.LBL) and attached .IMG headers for NASA
Lunar Reconnaissance Orbiter (LRO) Narrow Angle Camera (NAC) products.
Extracts native GSD, spatial bounds, observation geometry (incidence, emission,
sun azimuth), acquisition time, and map projection metadata into canonical
SensorMetadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

try:
    from metadata import SensorMetadata, SENSOR_SPECS
except ImportError:
    from ML_model.metadata import SensorMetadata, SENSOR_SPECS

logger = logging.getLogger("ML_model.lro_pds3_parser")


def parse_pds3_text(text: str) -> Dict[str, Any]:
    """
    Parses raw PDS3 label text into a nested dictionary of key-value pairs.
    Handles objects, single values, strings, numbers with units, and tuples.
    """
    result: Dict[str, Any] = {}
    obj_stack: list[dict] = [result]

    # Strip multi-line comments: /* ... */
    clean_text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    lines = clean_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == "END":
            break

        # Check for OBJECT = ...
        m_obj = re.match(r"^OBJECT\s*=\s*([A-Za-z0-9_]+)", line, re.IGNORECASE)
        if m_obj:
            obj_name = m_obj.group(1).upper()
            new_obj: Dict[str, Any] = {}
            curr = obj_stack[-1]
            curr[obj_name] = new_obj
            obj_stack.append(new_obj)
            continue

        # Check for END_OBJECT = ...
        m_end_obj = re.match(r"^END_OBJECT(?:\s*=\s*([A-Za-z0-9_]+))?", line, re.IGNORECASE)
        if m_end_obj:
            if len(obj_stack) > 1:
                obj_stack.pop()
            continue

        # Check for KEY = VALUE
        m_kv = re.match(r"^([\^A-Za-z0-9_:]+)\s*=\s*(.*)$", line)
        if not m_kv:
            continue

        key = m_kv.group(1).upper()
        raw_val = m_kv.group(2).strip()

        # Check if value spans multiple lines (e.g. unclosed string or tuple)
        while (
            (raw_val.startswith('"') and not raw_val.endswith('"') and len(raw_val) > 1)
            or (raw_val.startswith("(") and not raw_val.endswith(")"))
        ) and i < len(lines):
            raw_val += " " + lines[i].strip()
            i += 1

        # Clean units like <m>, <deg>, <km>, etc.
        val_cleaned = re.sub(r"<[A-Za-z0-9_]+>", "", raw_val).strip()

        # Parse data type
        val: Any = val_cleaned
        if val_cleaned.startswith('"') and val_cleaned.endswith('"'):
            val = val_cleaned[1:-1].strip()
        elif val_cleaned.startswith("(") and val_cleaned.endswith(")"):
            inner = val_cleaned[1:-1].strip()
            val = [p.strip() for p in inner.split(",") if p.strip()]
        else:
            # Try parsing integer or float
            try:
                if "." in val_cleaned or "e" in val_cleaned.lower():
                    val = float(val_cleaned)
                else:
                    val = int(val_cleaned)
            except ValueError:
                val = val_cleaned

        obj_stack[-1][key] = val

    return result


def read_pds3_label(path: str | Path) -> Dict[str, Any]:
    """
    Reads a PDS3 label from a detached .LBL/.lbl file or the first records
    of a .IMG file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDS3 label or image file not found: {p}")

    # If it is an IMG file, read the first ~64KB which contains the attached label
    if p.suffix.lower() == ".img":
        with open(p, "rb") as f:
            header_bytes = f.read(65536)
        text = header_bytes.decode("ascii", errors="ignore")
    else:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return parse_pds3_text(text)


def extract_lro_nac_metadata(
    label_or_img_path: str | Path,
    explicit_gsd: Optional[float] = None,
    explicit_emission: Optional[float] = None,
    explicit_azimuth: Optional[float] = None,
) -> SensorMetadata:
    """
    Extracts canonical SensorMetadata from an LRO NAC PDS3 label (.LBL or .IMG).
    Populates provenance tracking per project photogrammetric standards.
    """
    p = Path(label_or_img_path)
    lbl_dict = read_pds3_label(p)

    provenance: Dict[str, str] = {
        "sensor": "header",
    }

    # Flatten top-level and IMAGE_MAP_PROJECTION dictionary for easy lookup
    flat: Dict[str, Any] = {}
    def _flatten(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten(v)
            else:
                flat[k] = v
    _flatten(lbl_dict)

    # 1. Verify / Identify Instrument
    inst = str(flat.get("INSTRUMENT_ID", flat.get("INSTRUMENT_NAME", "LROC"))).upper()
    sensor_name = "LRO_NAC"

    # 2. GSD Resolution
    gsd_val: Optional[float] = None
    if explicit_gsd is not None and explicit_gsd > 0:
        gsd_val = float(explicit_gsd)
        provenance["gsd_m"] = "request"
    else:
        for k in ("MAP_SCALE", "PIXEL_RESOLUTION", "MAP_RESOLUTION", "RESOLUTION"):
            if k in flat and isinstance(flat[k], (int, float)):
                gsd_val = float(flat[k])
                provenance["gsd_m"] = "header"
                break

    if gsd_val is None:
        # Fallback to nominal LRO NAC GSD from mission specs
        gsd_val = SENSOR_SPECS.get("LRO_NAC", {}).get("gsd_m", 0.5)
        provenance["gsd_m"] = "sensor_spec"

    # 3. Observation & Illumination Geometry
    # Incidence angle
    incidence_val: Optional[float] = None
    for k in ("INCIDENCE_ANGLE", "CENTER_INCIDENCE_ANGLE", "INCIDENCE"):
        if k in flat and isinstance(flat[k], (int, float)):
            incidence_val = float(flat[k])
            provenance["incidence_angle_deg"] = "header"
            break
    if incidence_val is None:
        provenance["incidence_angle_deg"] = "unavailable"

    # Emission angle
    emission_val: Optional[float] = None
    for k in ("EMISSION_ANGLE", "CENTER_EMISSION_ANGLE", "EMISSION"):
        if k in flat and isinstance(flat[k], (int, float)):
            emission_val = float(flat[k])
            provenance["emission_angle_deg"] = "header"
            break
    if emission_val is None:
        if explicit_emission is not None:
            emission_val = float(explicit_emission)
            provenance["emission_angle_deg"] = "request"
        else:
            emission_val = None
            provenance["emission_angle_deg"] = "unavailable"

    # Sun azimuth angle
    azimuth_val: Optional[float] = None
    for k in ("SOLAR_AZIMUTH_ANGLE", "SUB_SOLAR_AZIMUTH", "SUN_AZIMUTH_ANGLE", "AZIMUTH"):
        if k in flat and isinstance(flat[k], (int, float)):
            azimuth_val = float(flat[k])
            provenance["sun_azimuth_deg"] = "header"
            break
    if azimuth_val is None:
        if explicit_azimuth is not None:
            azimuth_val = float(explicit_azimuth)
            provenance["sun_azimuth_deg"] = "request"
        else:
            azimuth_val = None
            provenance["sun_azimuth_deg"] = "unavailable"

    # Phase angle
    phase_val: Optional[float] = None
    for k in ("PHASE_ANGLE", "CENTER_PHASE_ANGLE", "PHASE"):
        if k in flat and isinstance(flat[k], (int, float)):
            phase_val = float(flat[k])
            provenance["phase_angle_deg"] = "header"
            break

    # 4. Acquisition time
    acq_time: Optional[str] = None
    for k in ("START_TIME", "START_DATE_TIME", "IMAGE_TIME"):
        if k in flat and isinstance(flat[k], str):
            acq_time = flat[k]
            provenance["acquisition_time"] = "header"
            break

    # 5. Bounding coordinates: (min_lon, max_lon, min_lat, max_lat)
    bounds: Optional[Tuple[float, float, float, float]] = None
    min_lat = flat.get("MINIMUM_LATITUDE")
    max_lat = flat.get("MAXIMUM_LATITUDE")
    west_lon = flat.get("WESTERNMOST_LONGITUDE")
    east_lon = flat.get("EASTERNMOST_LONGITUDE")

    if all(isinstance(v, (int, float)) for v in (west_lon, east_lon, min_lat, max_lat)):
        bounds = (float(west_lon), float(east_lon), float(min_lat), float(max_lat))
        provenance["bounds"] = "header"

    specs = SENSOR_SPECS.get("LRO_NAC", {})
    wavelength = specs.get("wavelength_range_um", (0.40, 0.75))
    provenance["wavelength_range_um"] = "sensor_spec"

    return SensorMetadata(
        sensor=sensor_name,
        gsd_m=gsd_val,
        wavelength_range_um=wavelength,
        sun_azimuth_deg=azimuth_val,
        sun_elevation_deg=None if incidence_val is None else (90.0 - incidence_val),
        incidence_angle_deg=incidence_val,
        emission_angle_deg=emission_val,
        phase_angle_deg=phase_val,
        acquisition_time=acq_time,
        bounds=bounds,
        provenance=provenance,
    )
