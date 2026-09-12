"""
tests/test_pds3_image_decoder.py — Unit tests for PDS3 Binary Raster Decoder (Step 3a)
"""

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "ML_model") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from pds3_image_decoder import decode_pds3_img_to_array, _resolve_numpy_dtype


def test_resolve_numpy_dtype():
    """Verify correct numpy dtype resolution for various PDS3 sample types."""
    assert _resolve_numpy_dtype("UNSIGNED_INTEGER", 8) == np.dtype("uint8")
    assert _resolve_numpy_dtype("LSB_INTEGER", 16) == np.dtype("<i2")
    assert _resolve_numpy_dtype("MSB_INTEGER", 16) == np.dtype(">i2")
    assert _resolve_numpy_dtype("LSB_UNSIGNED_INTEGER", 16) == np.dtype("<u2")
    assert _resolve_numpy_dtype("MSB_UNSIGNED_INTEGER", 16) == np.dtype(">u2")
    assert _resolve_numpy_dtype("IEEE_REAL", 32) == np.dtype(">f4")
    assert _resolve_numpy_dtype("PC_REAL", 32) == np.dtype("<f4")


def test_decode_pds3_synthetic_16bit_lsb(tmp_path):
    """Test decoding a synthetic 16-bit LSB (little-endian) binary raster."""
    lines, samples = 16, 32
    known_pattern = np.arange(lines * samples, dtype="<i2").reshape((lines, samples))

    # Write raw binary data
    img_file = tmp_path / "test_lsb.img"
    with open(img_file, "wb") as f:
        f.write(known_pattern.tobytes())

    label = {
        "LINES": lines,
        "LINE_SAMPLES": samples,
        "SAMPLE_BITS": 16,
        "SAMPLE_TYPE": "LSB_INTEGER",
        "^IMAGE": 1,
    }

    decoded = decode_pds3_img_to_array(img_file, label)
    assert decoded.shape == (lines, samples)
    assert decoded.dtype == np.dtype("<i2")
    np.testing.assert_array_equal(decoded, known_pattern)


def test_decode_pds3_synthetic_16bit_msb_with_record_offset(tmp_path):
    """Test decoding 16-bit MSB (big-endian) raster with attached header record offset."""
    lines, samples = 8, 8
    record_bytes = 128  # 1 header record of 128 bytes
    header_junk = b"HEADER_RECORD_PDS3_METADATA_FILLER".ljust(128, b" ")  # Exactly 128 bytes

    known_pixels = (np.arange(lines * samples) * 100).astype(">u2").reshape((lines, samples))

    img_file = tmp_path / "test_attached_msb.img"
    with open(img_file, "wb") as f:
        f.write(header_junk)  # Record 1
        f.write(known_pixels.tobytes())  # Record 2 (^IMAGE = 2)

    label = {
        "RECORD_BYTES": record_bytes,
        "^IMAGE": 2,
        "LINES": lines,
        "LINE_SAMPLES": samples,
        "SAMPLE_BITS": 16,
        "SAMPLE_TYPE": "MSB_UNSIGNED_INTEGER",
    }

    decoded = decode_pds3_img_to_array(img_file, label)
    assert decoded.shape == (lines, samples)
    assert decoded.dtype == np.dtype(">u2")
    np.testing.assert_array_equal(decoded, known_pixels)


def test_decode_pds3_synthetic_8bit_unsigned(tmp_path):
    """Test decoding 8-bit unsigned integer raster."""
    lines, samples = 20, 20
    known_pixels = (np.arange(lines * samples) % 256).astype("uint8").reshape((lines, samples))

    img_file = tmp_path / "test_8bit.img"
    with open(img_file, "wb") as f:
        f.write(known_pixels.tobytes())

    label = {
        "LINES": lines,
        "LINE_SAMPLES": samples,
        "SAMPLE_BITS": 8,
        "SAMPLE_TYPE": "UNSIGNED_INTEGER",
    }

    decoded = decode_pds3_img_to_array(img_file, label)
    assert decoded.shape == (lines, samples)
    assert decoded.dtype == np.dtype("uint8")
    np.testing.assert_array_equal(decoded, known_pixels)


def test_decode_pds3_size_mismatch_raises(tmp_path):
    """File size smaller than required elements must raise ValueError."""
    img_file = tmp_path / "truncated.img"
    img_file.write_bytes(b"too_short")

    label = {
        "LINES": 512,
        "LINE_SAMPLES": 512,
        "SAMPLE_BITS": 16,
        "SAMPLE_TYPE": "LSB_INTEGER",
    }

    with pytest.raises(ValueError, match="smaller than expected data"):
        decode_pds3_img_to_array(img_file, label)
