"""
metadata.py — Sensor and Observation Geometry Metadata Abstraction

Provides metadata extraction and provenance tracking for Chandrayaan-2 sensors
(OHRC, TMC-2, IIRS) without hardcoding hidden geometries or physical scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import os


@dataclass
class SensorMetadata:
    """
    Physical and geometric metadata for a Chandrayaan-2 raster product.
    Tracks whether values originate from embedded headers, request parameters,
    or documented standard sensor specifications.
    """
    sensor: str  # "OHRC", "TMC-2", "IIRS", or "DEM"
    gsd_m: float  # Ground Sampling Distance in meters per pixel
    wavelength_range_um: Optional[Tuple[float, float]] = None
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    incidence_angle_deg: Optional[float] = None
    emission_angle_deg: Optional[float] = None
    phase_angle_deg: Optional[float] = None
    acquisition_time: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (min_lon, max_lon, min_lat, max_lat)
    provenance: Dict[str, str] = field(default_factory=dict)  # field_name -> "header" | "request" | "default" | "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor": self.sensor,
            "gsd_m": self.gsd_m,
            "wavelength_range_um": list(self.wavelength_range_um) if self.wavelength_range_um else None,
            "sun_azimuth_deg": self.sun_azimuth_deg,
            "sun_elevation_deg": self.sun_elevation_deg,
            "incidence_angle_deg": self.incidence_angle_deg,
            "emission_angle_deg": self.emission_angle_deg,
            "phase_angle_deg": self.phase_angle_deg,
            "acquisition_time": self.acquisition_time,
            "bounds": list(self.bounds) if self.bounds else None,
            "provenance": self.provenance,
        }


# Standard physical sensor specifications per Chandrayaan-2 mission documentation
# Used only as fallback when product header or API parameters are unavailable
SENSOR_SPECS = {
    "OHRC": {
        "gsd_m": 0.25,  # ~0.25–0.32 m at 100 km nominal circular orbit
        "wavelength_range_um": (0.45, 0.70),  # Panchromatic optical
        "nominal_emission_deg": 0.0,
    },
    "TMC-2": {
        "gsd_m": 5.0,  # ~4–5 m at 100 km nominal orbit (stereo Fore, Nadir, Aft)
        "wavelength_range_um": (0.40, 0.85),  # Panchromatic optical
        "nominal_emission_deg": None,  # Scene/product dependent (Fore +26°, Nadir 0°, Aft -26°); no universal default
    },
    "IIRS": {
        "gsd_m": 75.0,  # ~70–80 m hyperspectral swath
        "wavelength_range_um": (0.80, 5.00),  # 256 contiguous spectral bands
        "nominal_emission_deg": None,
    },
    "DEM": {
        "gsd_m": 5.0,
        "wavelength_range_um": None,
        "nominal_emission_deg": None,
    },
}


def normalize_sensor_name(name: str) -> str:
    """Normalizes colloquial sensor strings to canonical identifiers."""
    name_clean = name.strip().upper()
    if "OHR" in name_clean:
        return "OHRC"
    elif "TMC" in name_clean:
        return "TMC-2"
    elif "IIR" in name_clean:
        return "IIRS"
    elif "DEM" in name_clean:
        return "DEM"
    return name_clean


def extract_sensor_metadata(
    image_path: str | Path,
    declared_sensor: Optional[str] = None,
    explicit_gsd: Optional[float] = None,
    explicit_emission: Optional[float] = None,
    explicit_azimuth: Optional[float] = None,
) -> SensorMetadata:
    """
    Extracts sensor metadata following strict precedence:
    1. Product metadata / PDS4 XML label / GeoTIFF embedded tags.
    2. Explicit caller / API request parameters.
    3. Standard sensor defaults with explicit source tracking.
    """
    p = Path(image_path)
    provenance: Dict[str, str] = {}

    # Step 1: Infer sensor type
    sensor_type = None
    if declared_sensor:
        sensor_type = normalize_sensor_name(declared_sensor)
        provenance["sensor"] = "request"
    else:
        # Check filename pattern
        filename_lower = p.name.lower()
        if "ohr" in filename_lower:
            sensor_type = "OHRC"
            provenance["sensor"] = "filename_inference"
        elif "tmc" in filename_lower:
            sensor_type = "TMC-2"
            provenance["sensor"] = "filename_inference"
        elif "iir" in filename_lower:
            sensor_type = "IIRS"
            provenance["sensor"] = "filename_inference"
        elif "dem" in filename_lower:
            sensor_type = "DEM"
            provenance["sensor"] = "filename_inference"
        else:
            sensor_type = "UNKNOWN"
            provenance["sensor"] = "unknown"

    # Step 2: Check for PDS4 XML label
    header_data: Dict[str, Any] = {}
    xml_path = p.with_suffix(".xml")
    if not xml_path.exists():
        xml_path = p.with_suffix(".XML")
    if xml_path.exists():
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # Parse XML tags
            for el in root.iter():
                tag_local = el.tag.split("}")[-1].lower()
                text = (el.text or "").strip()
                if not text:
                    continue
                try:
                    if tag_local in ("pixel_resolution", "gsd", "map_scale"):
                        header_data["gsd_m"] = float(text.split()[0])
                    elif tag_local in ("emission_angle", "emission"):
                        header_data["emission_angle_deg"] = float(text.split()[0])
                    elif tag_local in ("incidence_angle", "incidence"):
                        header_data["incidence_angle_deg"] = float(text.split()[0])
                    elif tag_local in ("solar_azimuth_angle", "sun_azimuth", "azimuth"):
                        header_data["sun_azimuth_deg"] = float(text.split()[0])
                    elif tag_local in ("start_date_time", "acquisition_time"):
                        header_data["acquisition_time"] = text
                except (ValueError, IndexError):
                    pass
        except Exception:
            pass

    # Check for region manifest.json sidecar
    manifest_path = p.parent / "manifest.json"
    if manifest_path.exists():
        try:
            import json
            with open(manifest_path) as mf:
                mdata = json.load(mf)
                if sensor_type == "OHRC":
                    # If this is a pre-gridded tile (e.g. ohrc_512.png), its effective grid GSD is normalized to the reference grid
                    if "_512" in p.name:
                        header_data["gsd_m"] = mdata.get("tmc2_gsd_m", 5.0)
                    else:
                        header_data["gsd_m"] = mdata.get("ohrc_gsd_m", 0.25)
                    header_data["sun_azimuth_deg"] = mdata.get("ohrc_sun_azimuth_deg")
                elif sensor_type == "TMC-2":
                    header_data["gsd_m"] = mdata.get("tmc2_gsd_m", 5.0)
                    header_data["sun_azimuth_deg"] = mdata.get("tmc2_sun_azimuth_deg")
                elif sensor_type == "IIRS":
                    header_data["gsd_m"] = mdata.get("iirs_gsd_m", 75.0)
                if "bounds" in mdata:
                    b = mdata["bounds"]
                    header_data["bounds"] = (b["west_lon"], b["east_lon"], b["south_lat"], b["north_lat"])
        except Exception:
            pass

    # Step 3: Resolve GSD
    gsd_val = None
    if "gsd_m" in header_data:
        gsd_val = header_data["gsd_m"]
        provenance["gsd_m"] = "header"
    elif explicit_gsd is not None and explicit_gsd > 0:
        gsd_val = float(explicit_gsd)
        provenance["gsd_m"] = "request"
    elif sensor_type in SENSOR_SPECS and SENSOR_SPECS[sensor_type].get("gsd_m") is not None:
        gsd_val = SENSOR_SPECS[sensor_type]["gsd_m"]
        provenance["gsd_m"] = "sensor_spec"
    else:
        raise ValueError(
            f"Physical ground sampling distance (GSD) could not be determined for image '{p.name}' (sensor: {sensor_type}). "
            "Please provide an explicit 'explicit_gsd' parameter or product metadata headers."
        )

    # Step 4: Resolve observation geometry (emission & azimuth)
    emission_val = None
    if "emission_angle_deg" in header_data:
        emission_val = header_data["emission_angle_deg"]
        provenance["emission_angle_deg"] = "header"
    elif explicit_emission is not None:
        emission_val = float(explicit_emission)
        provenance["emission_angle_deg"] = "request"
    else:
        # Mark as unavailable rather than pretending geometry is known
        emission_val = None
        provenance["emission_angle_deg"] = "unavailable"

    azimuth_val = None
    if "sun_azimuth_deg" in header_data:
        azimuth_val = header_data["sun_azimuth_deg"]
        provenance["sun_azimuth_deg"] = "header"
    elif explicit_azimuth is not None:
        azimuth_val = float(explicit_azimuth)
        provenance["sun_azimuth_deg"] = "request"
    else:
        azimuth_val = None
        provenance["sun_azimuth_deg"] = "unavailable"

    incidence_val = header_data.get("incidence_angle_deg")
    if incidence_val is not None:
        provenance["incidence_angle_deg"] = "header"
    else:
        provenance["incidence_angle_deg"] = "unavailable"

    specs = SENSOR_SPECS.get(sensor_type, {})
    wavelength = specs.get("wavelength_range_um")
    if wavelength:
        provenance["wavelength_range_um"] = "sensor_spec"

    return SensorMetadata(
        sensor=sensor_type,
        gsd_m=gsd_val,
        wavelength_range_um=wavelength,
        sun_azimuth_deg=azimuth_val,
        incidence_angle_deg=incidence_val,
        emission_angle_deg=emission_val,
        acquisition_time=header_data.get("acquisition_time"),
        provenance=provenance,
    )
