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
    lines: int  # height
    samples: int  # width
    bands: int = 1
    bit_depth: int = 8
    data_type: str = "UnsignedByte"
    gsd_m: float = 5.0
    sun_azimuth_deg: float = 45.0
    sun_elevation_deg: float = 30.0
    incidence_angle_deg: float = 60.0
    emission_angle_deg: float = 0.0
    phase_angle_deg: float = 60.0
    spice_kernels: List[str] = field(default_factory=list)
    image_file: Optional[str] = None
    label_path: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

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
        "PDS4 metadata extracted: sensor=%s, size=(%dx%d), GSD=%.2fm, sun_az=%.1f deg, sun_el=%.1f deg",
        info.sensor, info.samples, info.lines, info.gsd_m, info.sun_azimuth_deg, info.sun_elevation_deg
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

    # Dimensions & Data Type
    lines = 512
    samples = 512
    bands = 1
    data_type = "UnsignedByte"
    bit_depth = 8

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

    # Ground Sample Distance (GSD)
    gsd_m = 0.25 if sensor == "OHRC" else (5.0 if sensor == "TMC-2" else (70.0 if sensor == "IIRS" else 100.0))
    for k in ("pixel_resolution", "map_scale", "spatial_resolution"):
        if k in tag_map:
            try:
                gsd_m = float(tag_map[k])
                break
            except ValueError:
                pass

    # Sun Azimuth & Elevation (Illumination)
    sun_az = 45.0
    sun_el = 30.0
    inc_ang = 60.0
    em_ang = 0.0
    ph_ang = 60.0

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
    )


def _parse_vicar_text(text: str, path: Path) -> PDS4ProductInfo:
    vicar_dict = parse_vicar_label(text)

    lines = int(vicar_dict.get("LINES", vicar_dict.get("NL", 512)))
    samples = int(vicar_dict.get("SAMPLES", vicar_dict.get("NS", 512)))
    bands = int(vicar_dict.get("BANDS", vicar_dict.get("NB", 1)))
    data_type = vicar_dict.get("FORMAT", "BYTE")
    bit_depth = 16 if "HALF" in data_type or "INT2" in data_type else (32 if "REAL" in data_type else 8)

    sensor = "TMC-2"
    inst = vicar_dict.get("INSTRUMENT_NAME", vicar_dict.get("INSTRUMENT_ID", ""))
    if "OHR" in inst.upper() or "OHR" in path.name.upper():
        sensor = "OHRC"
        gsd_m = 0.25
    elif "IIR" in inst.upper() or "IIR" in path.name.upper():
        sensor = "IIRS"
        gsd_m = 70.0
    else:
        sensor = "TMC-2"
        gsd_m = 5.0

    if "PIXEL_RESOLUTION" in vicar_dict:
        try:
            gsd_m = float(vicar_dict["PIXEL_RESOLUTION"])
        except ValueError:
            pass

    sun_az = float(vicar_dict.get("SOLAR_AZIMUTH", vicar_dict.get("SUN_AZIMUTH", 45.0)))
    sun_el = float(vicar_dict.get("SOLAR_ELEVATION", vicar_dict.get("SUN_ELEVATION", 30.0)))
    inc_ang = float(vicar_dict.get("INCIDENCE_ANGLE", 90.0 - sun_el))
    em_ang = float(vicar_dict.get("EMISSION_ANGLE", 0.0))
    ph_ang = float(vicar_dict.get("PHASE_ANGLE", 60.0))

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
    )
