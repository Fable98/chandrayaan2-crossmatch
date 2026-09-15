"""
data/ingestion/pds4_reader.py — Robust PDS4 XML and Legacy VICAR Planetary Label Parser

Reads and parses planetary product labels from Chandrayaan-2 (OHRC, TMC-2, IIRS) and LRO.
Extracts:
1. Image dimensions and bit-depth.
2. Ground Sample Distance (GSD) in meters/pixel.
3. Sun azimuth and elevation (solar illumination metadata).
4. SPICE kernel files and observation geometry references.
"""

from __future__ import annotations

import os
import re
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import cv2

logger = logging.getLogger("data.ingestion.pds4_reader")


@dataclass
class PDS4ProductInfo:
    product_id: str
    sensor: str
    lines: Optional[int]  # height; None when the label states no dimensions
    samples: Optional[int]  # width; None when the label states no dimensions
    bands: int = 1
    bit_depth: int = 8
    data_type: str = "UnsignedByte"
    # Geometry fields are None when the label does not state them — NEVER
    # sensor-typical constants masquerading as measurement (2026-09-15).
    # Consumers must None-check (or call require_geometry) instead of
    # computing physical products from invented GSD/sun values.
    gsd_m: Optional[float] = None
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    incidence_angle_deg: Optional[float] = None
    emission_angle_deg: Optional[float] = None
    phase_angle_deg: Optional[float] = None
    spice_kernels: List[str] = field(default_factory=list)
    image_file: Optional[str] = None
    label_path: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    unknown_fields: List[str] = field(default_factory=list)

    @property
    def geometry_status(self) -> str:
        """'KNOWN' iff ground scale is label-sourced, else 'UNKNOWN_GEOMETRY'."""
        return "KNOWN" if self.gsd_m is not None else "UNKNOWN_GEOMETRY"

    def require_geometry(self, *fields: str) -> None:
        """Fail fast on unknown geometry instead of computing with invented values.

        Raises ValueError("UNKNOWN_GEOMETRY: ...") if any requested field
        (default: gsd_m) is None.
        """
        wanted = fields or ("gsd_m",)
        missing = [f for f in wanted if getattr(self, f, None) is None]
        if missing:
            raise ValueError(
                f"UNKNOWN_GEOMETRY: {self.label_path or self.product_id} states no "
                f"{', '.join(missing)}; refusing to compute physical products "
                f"from defaults."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "sensor": self.sensor,
            "dimensions": {"lines": self.lines, "samples": self.samples, "bands": self.bands},
            "bit_depth": self.bit_depth,
            "data_type": self.data_type,
            "gsd_m": self.gsd_m,
            "illumination": {
                "sun_azimuth_deg": self.sun_azimuth_deg,
                "sun_elevation_deg": self.sun_elevation_deg,
                "incidence_angle_deg": self.incidence_angle_deg,
                "emission_angle_deg": self.emission_angle_deg,
                "phase_angle_deg": self.phase_angle_deg,
            },
            "spice_kernels": self.spice_kernels,
            "image_file": self.image_file,
            "label_path": self.label_path,
            "geometry_status": self.geometry_status,
            "unknown_fields": list(self.unknown_fields),
        }


def parse_vicar_label(text: str) -> Dict[str, str]:
    """Parses VICAR / PDS3 key=value format labels."""
    metadata = {}
    pattern = re.compile(r"([A-Za-z0-9_]+)\s*=\s*('?[^'\n\r,]+'?|\([^)]+\))")
    for match in pattern.finditer(text):
        key = match.group(1).upper()
        val = match.group(2).strip("'\"")
        metadata[key] = val
    return metadata


def parse_pds4_or_vicar_label(label_path: str | Path) -> PDS4ProductInfo:
    """
    Parses a PDS4 XML label or legacy VICAR text header.
    Automatically extracts dimensions, bit-depth, GSD, Sun azimuth/elevation,
    and SPICE kernel references.
    """
    path = Path(label_path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    content = path.read_text(encoding="utf-8", errors="ignore")

    # Check if XML or VICAR
    is_xml = content.strip().startswith("<?xml") or "<Product_" in content

    if is_xml:
        logger.info("Parsing PDS4 XML label: %s", path.name)
        info = _parse_pds4_xml(content, path)
    else:
        logger.info("Parsing legacy VICAR label: %s", path.name)
        info = _parse_vicar_text(content, path)

    logger.info(
        "PDS4 metadata extracted: sensor=%s, size=(%s), GSD=%s, sun_az=%s, sun_el=%s [%s]",
        info.sensor,
        f"{info.samples}x{info.lines}" if info.samples and info.lines else "unknown",
        f"{info.gsd_m:.2f}m" if info.gsd_m is not None else "unknown",
        f"{info.sun_azimuth_deg:.1f} deg" if info.sun_azimuth_deg is not None else "unknown",
        f"{info.sun_elevation_deg:.1f} deg" if info.sun_elevation_deg is not None else "unknown",
        info.geometry_status,
    )
    if info.unknown_fields:
        logger.warning(
            "Label %s omits %s; recorded as unknown (no defaults invented).",
            path.name, ", ".join(info.unknown_fields)
        )
    return info


def _parse_pds4_xml(xml_text: str, path: Path) -> PDS4ProductInfo:
    tree = ET.fromstring(xml_text)

    # Strip XML namespaces for uniform tag access
    for elem in tree.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    product_id = path.stem
    title = tree.findtext(".//title", default=product_id)

    # Infer Sensor
    sensor = "OHRC" if "OHR" in title.upper() or "OHR" in path.name.upper() else (
        "TMC-2" if "TMC" in title.upper() or "TMC" in path.name.upper() else (
            "IIRS" if "IIR" in title.upper() or "IIR" in path.name.upper() else "LRO_WAC"
        )
    )

    # Dimensions & Data Type (None when unstated — never invented 512s).
    lines: Optional[int] = None
    samples: Optional[int] = None
    bands = 1
    data_type = "UnsignedByte"
    bit_depth = 8
    unknown: List[str] = []

    # Extract Axis_Array dimensions
    for elem in tree.iter():
        clean_tag = elem.tag.split("}", 1)[1] if "}" in elem.tag else elem.tag
        if clean_tag.lower() == "axis_array":
            name = ""
            elems = ""
            for child in elem:
                ct = child.tag.split("}", 1)[1] if "}" in child.tag else child.tag
                if ct.lower() == "axis_name":
                    name = child.text.strip() if child.text else ""
                elif ct.lower() == "elements":
                    elems = child.text.strip() if child.text else ""
            if "line" in name.lower() and elems.isdigit():
                lines = int(elems)
            elif "sample" in name.lower() and elems.isdigit():
                samples = int(elems)
            elif "band" in name.lower() and elems.isdigit():
                bands = int(elems)

    # Build dictionary of all clean tags for fast property extraction
    tag_map = {}
    for elem in tree.iter():
        clean_tag = (elem.tag.split("}", 1)[1] if "}" in elem.tag else elem.tag).lower()
        if elem.text:
            tag_map[clean_tag] = elem.text.strip()

    if "lines" in tag_map and tag_map["lines"].isdigit():
        lines = int(tag_map["lines"])
    if "samples" in tag_map and tag_map["samples"].isdigit():
        samples = int(tag_map["samples"])
    if "bands" in tag_map and tag_map["bands"].isdigit():
        bands = int(tag_map["bands"])

    if "data_type" in tag_map:
        data_type = tag_map["data_type"]
        if "16" in data_type or "Half" in data_type:
            bit_depth = 16
        elif "32" in data_type or "Float" in data_type:
            bit_depth = 32
        elif "Byte" in data_type or "8" in data_type:
            bit_depth = 8

    # Ground Sample Distance: label-sourced only. The old sensor-typical
    # fallback (OHRC 0.25 / TMC-2 5.0 / IIRS 70 / LRO 100) invented precision
    # the label never stated; downstream physical math must see None instead.
    gsd_m: Optional[float] = None
    for k in ("pixel_resolution", "map_scale", "spatial_resolution"):
        if k in tag_map:
            try:
                gsd_m = float(tag_map[k])
                break
            except ValueError:
                pass
    if gsd_m is None:
        unknown.append("gsd_m")

    # Sun Azimuth & Elevation (None when unstated — never invented sun).
    sun_az: Optional[float] = None
    sun_el: Optional[float] = None
    inc_ang: Optional[float] = None
    em_ang: Optional[float] = None
    ph_ang: Optional[float] = None

    for k in ("sun_azimuth", "solar_azimuth", "sub_solar_azimuth"):
        if k in tag_map:
            try:
                sun_az = float(tag_map[k])
                break
            except ValueError:
                pass

    for k in ("sun_elevation", "solar_elevation"):
        if k in tag_map:
            try:
                sun_el = float(tag_map[k])
                break
            except ValueError:
                pass

    if "incidence_angle" in tag_map:
        try:
            inc_ang = float(tag_map["incidence_angle"])
            if "sun_elevation" not in tag_map and "solar_elevation" not in tag_map:
                sun_el = max(0.0, 90.0 - inc_ang)
        except ValueError:
            pass

    if "emission_angle" in tag_map:
        try:
            em_ang = float(tag_map["emission_angle"])
        except ValueError:
            pass

    if "phase_angle" in tag_map:
        try:
            ph_ang = float(tag_map["phase_angle"])
        except ValueError:
            pass

    for _name, _val in (
        ("lines", lines), ("samples", samples),
        ("sun_azimuth_deg", sun_az), ("sun_elevation_deg", sun_el),
        ("incidence_angle_deg", inc_ang), ("emission_angle_deg", em_ang),
        ("phase_angle_deg", ph_ang),
    ):
        if _val is None:
            unknown.append(_name)

    # SPICE Kernels
    spice_kernels = []
    for kernel_elem in tree.findall(".//kernel_file_name") + tree.findall(".//spice_kernel_file"):
        if kernel_elem.text:
            spice_kernels.append(kernel_elem.text.strip())

    if not spice_kernels:
        # Check comment or description for SPICE references
        desc = tree.findtext(".//comment", default="") + tree.findtext(".//description", default="")
        bsp_matches = re.findall(r"([a-zA-Z0-9_\-]+\.(?:bsp|ti|tls|tf|tpc|bc))", desc, re.IGNORECASE)
        spice_kernels.extend(bsp_matches)

    # Associated Image File
    img_file = tree.findtext(".//file_name", default=None)
    if img_file is None:
        for ext in [".png", ".tif", ".img", ".raw"]:
            candidate = path.with_suffix(ext)
            if candidate.exists():
                img_file = candidate.name
                break

    return PDS4ProductInfo(
        product_id=product_id,
        sensor=sensor,
        lines=lines,
        samples=samples,
        bands=bands,
        bit_depth=bit_depth,
        data_type=data_type,
        gsd_m=gsd_m,
        sun_azimuth_deg=sun_az,
        sun_elevation_deg=sun_el,
        incidence_angle_deg=inc_ang,
        emission_angle_deg=em_ang,
        phase_angle_deg=ph_ang,
        spice_kernels=list(set(spice_kernels)),
        image_file=img_file,
        label_path=str(path),
        unknown_fields=sorted(set(unknown)),
    )


def _parse_vicar_text(text: str, path: Path) -> PDS4ProductInfo:
    vicar_dict = parse_vicar_label(text)
    unknown: List[str] = []

    def _opt_int(*keys: str, field: str) -> Optional[int]:
        for k in keys:
            if k in vicar_dict:
                try:
                    return int(float(vicar_dict[k]))
                except (ValueError, TypeError):
                    pass
        unknown.append(field)
        return None

    def _opt_float(*keys: str, field: str) -> Optional[float]:
        for k in keys:
            if k in vicar_dict:
                try:
                    return float(vicar_dict[k])
                except (ValueError, TypeError):
                    pass
        unknown.append(field)
        return None

    lines = _opt_int("LINES", "NL", field="lines")
    samples = _opt_int("SAMPLES", "NS", field="samples")
    bands = _opt_int("BANDS", "NB", field="bands") or 1
    data_type = vicar_dict.get("FORMAT", "BYTE")
    bit_depth = 16 if "HALF" in data_type or "INT2" in data_type else (32 if "REAL" in data_type else 8)

    sensor = "TMC-2"
    inst = vicar_dict.get("INSTRUMENT_NAME", vicar_dict.get("INSTRUMENT_ID", ""))
    if "OHR" in inst.upper() or "OHR" in path.name.upper():
        sensor = "OHRC"
    elif "IIR" in inst.upper() or "IIR" in path.name.upper():
        sensor = "IIRS"

    # Label-sourced GSD only; sensor-typical constants are not measurements.
    gsd_m = _opt_float("PIXEL_RESOLUTION", field="gsd_m")

    sun_az = _opt_float("SOLAR_AZIMUTH", "SUN_AZIMUTH", field="sun_azimuth_deg")
    sun_el = _opt_float("SOLAR_ELEVATION", "SUN_ELEVATION", field="sun_elevation_deg")
    inc_ang = _opt_float("INCIDENCE_ANGLE", field="incidence_angle_deg")
    if sun_el is None and inc_ang is not None:
        # Legitimate solar-geometry identity (not an invented default).
        sun_el = max(0.0, 90.0 - inc_ang)
        if "sun_elevation_deg" in unknown:
            unknown.remove("sun_elevation_deg")
    em_ang = _opt_float("EMISSION_ANGLE", field="emission_angle_deg")
    ph_ang = _opt_float("PHASE_ANGLE", field="phase_angle_deg")

    # Look for kernel files
    spice_kernels = []
    for k, v in vicar_dict.items():
        if "KERNEL" in k or "SPICE" in k:
            spice_kernels.append(v)

    return PDS4ProductInfo(
        product_id=path.stem,
        sensor=sensor,
        lines=lines,
        samples=samples,
        bands=bands,
        bit_depth=bit_depth,
        data_type=data_type,
        gsd_m=gsd_m,
        sun_azimuth_deg=sun_az,
        sun_elevation_deg=sun_el,
        incidence_angle_deg=inc_ang,
        emission_angle_deg=em_ang,
        phase_angle_deg=ph_ang,
        spice_kernels=spice_kernels,
        image_file=None,
        label_path=str(path),
        raw_metadata=vicar_dict,
        unknown_fields=sorted(set(unknown)),
    )
