from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


@dataclass
class ImageMetadata:
    product_id: str
    sensor: str
    source_path: str
    label_path: str | None = None
    acquisition_utc: str | None = None
    gsd_m: float | None = None
    working_gsd_m: float | None = None
    scale_factor: float | None = None
    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    incidence_deg: float | None = None
    emission_deg: float | None = None
    phase_deg: float | None = None
    footprint: dict[str, Any] = field(default_factory=dict)
    width: int | None = None
    height: int | None = None
    bands: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TileRecord:
    tile_id: str
    product_id: str
    sensor: str
    row: int
    col: int
    level: int
    gsd_m: float | None
    working_gsd_m: float | None
    scale_factor: float | None
    sun_azimuth_deg: float | None
    sun_elevation_deg: float | None
    incidence_deg: float | None
    emission_deg: float | None
    phase_deg: float | None
    acquisition_utc: str | None
    footprint: dict[str, Any]
    bbox: list[float]
    crs: str
    files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SENSOR_ALIASES = {
    "ohrc": "OHRC",
    "optical high resolution camera": "OHRC",
    "orbiter high resolution camera": "OHRC",
    "ohr": "OHRC",
    "tmc": "TMC",
    "tmc-2": "TMC",
    "tmc2": "TMC",
    "terrain mapping camera": "TMC",
    "terrain mapping camera-2": "TMC",
    "terrain mapping camera 2": "TMC",
    "iirs": "IIRS",
    "iir": "IIRS",
    "imaging infrared spectrometer": "IIRS",
}


def infer_sensor(text: str | None, path_hint: str = "") -> str:
    blob_text = (text or "").lower()
    blob_path = (path_hint or "").lower()
    filename = Path(blob_path).name.lower() if blob_path else ""

    for blob in (blob_text, filename, blob_path):
        if not blob:
            continue
        if "iirs" in blob or "iir" in blob or "imaging infrared" in blob:
            return "IIRS"
        if "ohrc" in blob or "ohr" in blob or "high resolution" in blob:
            return "OHRC"
        if "tmc" in blob or "terrain mapping" in blob:
            return "TMC"
    return "UNKNOWN"
