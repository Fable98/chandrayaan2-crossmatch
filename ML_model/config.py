"""
ML_model/config.py — Centralized Configuration and Ground Sampling Distance Constants

Single source of truth for:
- Sensor spatial resolutions (GSD in meters/pixel)
- Global random seed for reproducible estimation across pipeline stages
"""

from __future__ import annotations

import os
from typing import Dict

# Ground Sampling Distance (GSD) in meters per pixel.
# Canonical stack values (Phases 2/4): OHRC 0.25 / TMC-2 5.0 / IIRS 70.0 /
# LRO NAC 0.9. NOTE: IIRS was 80.0 and LRO_NAC 0.5 before Phase 6 — both stale
# (the evaluator, frontend spec table, and audit all standardize on 70/0.9).
OHRC_GSD: float = 0.25
TMC_GSD: float = 5.0
IIRS_GSD: float = 70.0
LRO_NAC_GSD: float = 0.9

# Canonical map of sensor names to nominal ground sampling distance (meters/pixel)
SENSOR_GSD_MAP: Dict[str, float] = {
    "OHRC": OHRC_GSD,
    "TMC": TMC_GSD,
    "TMC-2": TMC_GSD,
    "TMC2": TMC_GSD,  # alias: hyphen-less form seen in filenames/APIs
    "IIRS": IIRS_GSD,
    "LRO_NAC": LRO_NAC_GSD,
    "LRO-NAC": LRO_NAC_GSD,  # alias
    "NAC": LRO_NAC_GSD,  # alias
}

# Global random seed for stochastic sampling (e.g. RANSAC, hold-out splits)
SEED: int = int(os.environ.get("GLOBAL_SEED", 42))


def get_seed() -> int:
    """Return the global reproducible random seed."""
    return SEED


def get_sensor_gsd(sensor_name: str, fallback: float = 1.0) -> float:
    """Return the canonical nominal GSD in meters/pixel for the given sensor name.

    Accepts hyphen/underscore/space/case variants ("tmc2", "TMC-2", "lro nac")
    via the alias keys above plus a separator-stripped second chance.
    """
    if not sensor_name:
        return fallback
    clean = str(sensor_name).strip().upper()
    if clean in SENSOR_GSD_MAP:
        return SENSOR_GSD_MAP[clean]
    squashed = clean.replace("-", "").replace("_", "").replace(" ", "")
    for key, gsd in SENSOR_GSD_MAP.items():
        if key.replace("-", "").replace("_", "") == squashed:
            return gsd
    return fallback


# ===========================================================================
# Empirically Tuned Thresholds (Calibrated via scripts/tune_thresholds.py)
# ===========================================================================
TUNED_RANSAC_REPROJ_THRESH: float = 5.0  # tuned on 2026-09-10, AUC=0.9010
TUNED_NCC_THRESH: float = 0.25  # tuned on 2026-09-10, AUC=0.9010
TUNED_RELAXED_NCC_THRESH: float = 0.20  # tuned on 2026-09-10, AUC=0.9010
TUNED_MI_THRESH: float = 0.08  # tuned on 2026-09-10, AUC=0.9010
TUNED_RELAXED_MI_THRESH: float = 0.03  # tuned on 2026-09-10, AUC=0.9010

# Quality Gate 3 Matrix Conditioning Thresholds
TUNED_GATE3_MAX_COND: float = 1e7  # tuned on 2026-09-10, AUC=0.9010
TUNED_GATE3_MIN_DET: float = 1e-4  # tuned on 2026-09-10, AUC=0.9010
TUNED_GATE3_MAX_SCALE_RATIO: float = 20.0  # tuned on 2026-09-10, AUC=0.9010
TUNED_GATE3_MAX_PROJ: float = 0.05  # tuned on 2026-09-10, AUC=0.9010
TUNED_GATE3_MAX_RMSE: float = 5.0  # tuned on 2026-09-10, AUC=0.9010


# Re-export backend Settings if loaded in a unified sys.path environment
try:
    from backend.config import settings, Settings
except Exception:
    pass


# ===========================================================================
# SIH Compliance: Sun-Angle Invariance (Task 1 — configurable via env / kwargs)
# ---------------------------------------------------------------------------
# The SIH problem statement supports "Sun angle invariant" correspondence via
# Phase 1 adaptive illumination normalization (homomorphic log decomposition +
# morphological shadow suppression). It can be enabled via env var
# SUN_ANGLE_INVARIANCE_ENABLED=1 or per-call enable_illumination_normalization=True.
# ===========================================================================
import os
SUN_ANGLE_INVARIANCE_ENABLED: bool = os.getenv("SUN_ANGLE_INVARIANCE_ENABLED", "0").lower() in ("1", "true")
ADAPTIVE_ILLUMINATION_NORMALIZATION_ENABLED: bool = os.getenv("ADAPTIVE_ILLUMINATION_NORMALIZATION_ENABLED", "0").lower() in ("1", "true")
EXPERIMENTAL_STACK_ENABLED_BY_DEFAULT: bool = os.getenv("EXPERIMENTAL_STACK_ENABLED_BY_DEFAULT", "0").lower() in ("1", "true")
