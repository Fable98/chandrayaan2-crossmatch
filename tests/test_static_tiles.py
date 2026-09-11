"""
tests/test_static_tiles.py — Staged frontend tiles must BE the data they claim.

Regression test for the mislabeled-LRO incident: public/images/lro_nac/*
shipped OHRC-derived pixels under LRO NAC names (NCC 0.73 vs OHRC, ~0 vs
the real reference), so the vault showed "LRO" thumbs identical to OHRC.
Every staged tile asserted here must near-exactly match its pipeline source
(NCC > 0.99, identical shape), and LRO tiles must NOT match OHRC (NCC < 0.5).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC = REPO_ROOT / "lunar-frontend" / "public" / "images"

LRO_REGIONS = ("region_001", "region_003", "region_006")


def _gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"unreadable staged tile: {path}"
    return img.astype(np.float64)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    a = (a - a.mean()).ravel()
    b = (b - b.mean()).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    assert denom > 1e-9, "degenerate tile (zero variance)"
    return float(a.dot(b) / denom)


@pytest.mark.parametrize("region", LRO_REGIONS)
def test_staged_lro_tile_is_the_real_reference(region: str):
    """public/images/lro_nac/<id> must be the real CDR reference tile."""
    for variant in (f"{region}.png", region):
        staged = _gray(PUBLIC / "lro_nac" / variant)
        real = _gray(
            REPO_ROOT / "data_preprocessing_pipeline" / "lro_nac_real"
            / region / "lro_nac_reference_512.png"
        )
        assert staged.shape == real.shape, (
            f"{variant}: shape {staged.shape} != reference {real.shape} "
            "(wrong file staged?)"
        )
        assert _ncc(staged, real) > 0.99, f"{variant}: not the real reference tile"


@pytest.mark.parametrize("region", LRO_REGIONS)
def test_staged_lro_tile_is_not_ohrc(region: str):
    """LRO tiles must not be OHRC pixels under another name."""
    staged = _gray(PUBLIC / "lro_nac" / f"{region}.png")
    ohrc = _gray(
        REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets"
        / region / "ohrc_512.png"
    )
    assert _ncc(staged, ohrc) < 0.5, (
        f"{region}: staged LRO tile matches OHRC (NCC={_ncc(staged, ohrc):.3f}) — mislabeled file"
    )


@pytest.mark.parametrize(
    "sensor,filename",
    [("ohrc", "ohrc_512.png"), ("tmc", "tmc_512.png"),
     ("iirs", "iirs_512.png"), ("dem", "dem_512.png")],
)
def test_staged_sensor_tiles_match_pipeline(sensor: str, filename: str):
    """Spot-check: other staged tiles must match their pipeline sources."""
    staged = _gray(PUBLIC / sensor / "region_001.png")
    source = _gray(
        REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets"
        / "region_001" / filename
    )
    assert staged.shape == source.shape
    assert _ncc(staged, source) > 0.99
