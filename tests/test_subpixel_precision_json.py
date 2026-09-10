"""
tests/test_subpixel_precision_json.py — Guards sub-pixel coordinate precision
through JSON serialization for Chandrayaan-2 crossmatch.

Regression coverage for SIH sub-pixel claim (RMSE < 0.3px):
- Coordinates must never be cast with int()/round()/astype(int) before dump.
- NumPy float32/float64 must become native floats with full precision.
- Only `confidence` may be rounded (2 decimals); coordinates stay exact.
- Output stays human-readable (indent=2) without losing precision.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import (
    SubpixelJSONEncoder,
    dump_matches_json,
    dumps_matches_json,
    make_match_record,
    sanitize_for_json,
)


def test_numpy_scalars_become_native_floats():
    """float32/float64/int32 must serialize as native types, not strings."""
    obj = {
        "x": np.float32(123.4567),
        "y": np.float64(890.1234),
        "n": np.int32(42),
        "arr": np.array([1.5, 2.5], dtype=np.float32),
    }
    clean = sanitize_for_json(obj)
    assert isinstance(clean["x"], float)
    assert isinstance(clean["y"], float)
    assert isinstance(clean["n"], int)
    assert isinstance(clean["arr"], list)
    assert all(isinstance(v, float) for v in clean["arr"])
    # json.dumps must not raise and must round-trip
    s = json.dumps(clean, cls=SubpixelJSONEncoder, indent=2)
    back = json.loads(s)
    assert abs(back["x"] - 123.4567) < 1e-4
    assert abs(back["y"] - 890.1234) < 1e-4


def test_subpixel_roundtrip_precision():
    """Core SIH audit: (123.4567, 890.1234) survives dump/load with err < 1e-4."""
    orig_x, orig_y = 123.4567, 890.1234
    rec = make_match_record(
        np.float32(orig_x),
        np.float64(orig_y),
        np.float32(456.7891),
        np.float64(321.9876),
        confidence=0.87654,
    )
    s = dumps_matches_json([rec], indent=2)
    back = json.loads(s)[0]

    for key, orig in (
        ("source_x", orig_x),
        ("source_y", orig_y),
        ("target_x", 456.7891),
        ("target_y", 321.9876),
        ("image1_x", orig_x),
        ("image1_y", orig_y),
        ("image2_x", 456.7891),
        ("image2_y", 321.9876),
    ):
        assert isinstance(back[key], float), f"{key} must be float, got {type(back[key])}"
        assert abs(back[key] - orig) < 1e-4, (
            f"{key}: |{back[key]} - {orig}| >= 1e-4 (precision lost)"
        )

    # Coordinates must NOT be truncated to int
    assert back["source_x"] != int(orig_x)
    # Confidence MAY be rounded to 2 decimals
    assert back["confidence"] == round(0.87654, 2)


def test_dump_matches_json_file_roundtrip(tmp_path):
    """dump_matches_json writes indent=2 file that reads back precisely."""
    orig = (123.4567, 890.1234)
    rec = make_match_record(orig[0], orig[1], 200.00005, 300.00005, 0.91234)
    path = tmp_path / "matches.json"
    dump_matches_json([rec], path, indent=2)

    text = path.read_text(encoding="utf-8")
    assert '  "source_x"' in text  # human-readable indent=2
    back = json.loads(text)[0]
    assert abs(back["source_x"] - orig[0]) < 1e-4
    assert abs(back["source_y"] - orig[1]) < 1e-4
    assert abs(back["target_x"] - 200.00005) < 1e-4
    assert abs(back["target_y"] - 300.00005) < 1e-4


def test_coordinates_not_rounded_to_2_decimals():
    """4th decimal place must survive (round(x, 2) would destroy it)."""
    rec = make_match_record(10.12345, 20.67895, 30.11115, 40.99995, 0.5)
    back = json.loads(dumps_matches_json([rec]))[0]
    # If coords were round(..., 2), error would be up to 0.005; require < 1e-4
    assert abs(back["source_x"] - 10.12345) < 1e-4
    assert abs(back["source_y"] - 20.67895) < 1e-4
    assert abs(back["target_x"] - 30.11115) < 1e-4
    assert abs(back["target_y"] - 40.99995) < 1e-4
