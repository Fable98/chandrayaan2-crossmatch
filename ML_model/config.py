"""
ML_model/config.py — Centralized Configuration and Ground Sampling Distance Constants

Single source of truth for:
- Sensor spatial resolutions (GSD in meters/pixel)
- Global random seed for reproducible estimation across pipeline stages
"""

from __future__ import annotations

import os
from typing import Dict

# Ground Sampling Distance (GSD) in meters per pixel
OHRC_GSD: float = 0.25
TMC_GSD: float = 5.0
IIRS_GSD: float = 80.0
LRO_NAC_GSD: float = 0.5

# Canonical map of sensor names to nominal ground sampling distance (meters/pixel)
SENSOR_GSD_MAP: Dict[str, float] = {
    "OHRC": OHRC_GSD,
    "TMC": TMC_GSD,
    "TMC-2": TMC_GSD,
    "IIRS": IIRS_GSD,
    "LRO_NAC": LRO_NAC_GSD,
}

# Global random seed for stochastic sampling (e.g. RANSAC, hold-out splits)
SEED: int = int(os.environ.get("GLOBAL_SEED", 42))


def get_seed() -> int:
    """Return the global reproducible random seed."""
    return SEED


def get_sensor_gsd(sensor_name: str, fallback: float = 1.0) -> float:
    """Return the canonical nominal GSD in meters/pixel for the given sensor name."""
    if not sensor_name:
        return fallback
    clean = str(sensor_name).strip().upper()
    return SENSOR_GSD_MAP.get(clean, fallback)


# Re-export backend Settings if loaded in a unified sys.path environment
try:
    from backend.config import settings, Settings
except Exception:
    pass
