from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class IlluminationConfig:
    photometric_model: str = "lunar_lambert"
    shadow_percentile: float = 3.0
    incidence_shadow_deg: float = 85.0
    write_invariant: bool = True
    invariant_modes: list[str] = field(default_factory=lambda: ["census", "gradient", "lbp"])


@dataclass
class SensorConfig:
    destripe: bool = True
    iirs_reduce: str = "pca"
    iirs_pca_components: int = 1
    iirs_band_index: int = 0


@dataclass
class GeorefConfig:
    crs: str = "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs"
    resampling: str = "bilinear"
    try_isis: bool = False


@dataclass
class OutputConfig:
    dtype: str = "float32"
    compress: str = "lzw"
    write_shadow_mask: bool = True
    catalog_json: str = "catalog.json"
    catalog_csv: str = "catalog.csv"


@dataclass
class PipelineConfig:
    working_gsd_m: float = 5.0
    tile_size: int = 1024
    overlap: int = 128
    pyramid_levels: int = 5
    illumination: IlluminationConfig = field(default_factory=IlluminationConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    georef: GeorefConfig = field(default_factory=GeorefConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        illum = IlluminationConfig(**data.get("illumination", {}))
        sensors = SensorConfig(**data.get("sensors", {}))
        georef = GeorefConfig(**data.get("georef", {}))
        output = OutputConfig(**data.get("output", {}))
        top = {k: v for k, v in data.items() if k not in {"illumination", "sensors", "georef", "output"}}
        return cls(illumination=illum, sensors=sensors, georef=georef, output=output, **top)

    @classmethod
    def load(cls, path: Path | None) -> "PipelineConfig":
        if path is None:
            default = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
            path = default if default.exists() else None
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
