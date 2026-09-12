"""
ML_model/pds3_image_decoder.py — PDS3 Binary Image Raster Decoder

Decodes PDS3 formatted lunar imagery (.IMG/.DAT) into numpy ndarrays
using label metadata (LINES, LINE_SAMPLES, SAMPLE_BITS, SAMPLE_TYPE, ^IMAGE offset).
Handles endianness (LSB vs MSB), signed vs unsigned representations, and record offsets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

try:
    from lro_pds3_parser import parse_pds3_text, read_pds3_label
except ImportError:
    from ML_model.lro_pds3_parser import parse_pds3_text, read_pds3_label

logger = logging.getLogger("ML_model.pds3_image_decoder")


def _resolve_numpy_dtype(sample_type: str, sample_bits: int) -> np.dtype:
    """
    Maps PDS3 SAMPLE_TYPE and SAMPLE_BITS to an explicit numpy dtype with byte-order.
    """
    st = str(sample_type).upper().strip().strip('"')
    bits = int(sample_bits)

    if bits == 8:
        if any(w in st for w in ("UNSIGNED", "UINT")):
            return np.dtype("uint8")
        if any(w in st for w in ("SIGNED", "INT8")):
            return np.dtype("int8")
        return np.dtype("uint8")

    # Byte order check
    is_little_endian = any(w in st for w in ("LSB", "PC", "VAX", "LITTLE"))
    prefix = "<" if is_little_endian else ">"

    if bits == 16:
        if any(w in st for w in ("UNSIGNED", "UINT")):
            return np.dtype(f"{prefix}u2")
        return np.dtype(f"{prefix}i2")

    if bits == 32:
        if any(w in st for w in ("REAL", "FLOAT", "IEEE")):
            return np.dtype(f"{prefix}f4")
        if any(w in st for w in ("UNSIGNED", "UINT")):
            return np.dtype(f"{prefix}u4")
        return np.dtype(f"{prefix}i4")

    if bits == 64:
        if any(w in st for w in ("REAL", "FLOAT", "IEEE")):
            return np.dtype(f"{prefix}f8")
        return np.dtype(f"{prefix}i8")

    raise ValueError(f"Unsupported PDS3 SAMPLE_TYPE/BITS: {sample_type} with {sample_bits} bits")


def decode_pds3_img_to_array(
    img_path: Union[str, Path],
    label_metadata: Union[Dict[str, Any], str, Path],
) -> np.ndarray:
    """
    Decodes a PDS3 binary raster into a 2D numpy array (LINES, SAMPLES).

    Parameters:
        img_path: Path to the binary image raster (.IMG or .DAT).
        label_metadata: Either parsed label dictionary or path to .LBL file.

    Returns:
        2D numpy array of shape (lines, samples).
    """
    p_img = Path(img_path)
    if not p_img.exists():
        raise FileNotFoundError(f"PDS3 image file not found: {p_img}")

    # Parse label dictionary if path provided
    if isinstance(label_metadata, (str, Path)):
        label_dict = read_pds3_label(label_metadata)
    elif isinstance(label_metadata, dict):
        label_dict = label_metadata
    else:
        raise TypeError(f"Unsupported label_metadata type: {type(label_metadata)}")

    # Flatten label for easy access
    flat: Dict[str, Any] = {}

    def _flatten(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten(v)
            else:
                flat[k] = v

    _flatten(label_dict)

    # 1. Dimensions
    lines = flat.get("LINES")
    samples = flat.get("LINE_SAMPLES", flat.get("SAMPLES"))
    if lines is None or samples is None:
        raise ValueError(f"Missing LINES or LINE_SAMPLES in PDS3 label: lines={lines}, samples={samples}")

    lines = int(lines)
    samples = int(samples)
    if lines <= 0 or samples <= 0:
        raise ValueError(f"Invalid dimensions in PDS3 label: lines={lines}, samples={samples}")

    # 2. Data Type
    sample_bits = flat.get("SAMPLE_BITS", 8)
    sample_type = flat.get("SAMPLE_TYPE", "UNSIGNED_INTEGER")
    dtype = _resolve_numpy_dtype(sample_type, sample_bits)

    # 3. Byte Offset
    record_bytes = flat.get("RECORD_BYTES")
    image_ptr = flat.get("^IMAGE", 1)

    byte_offset = 0
    if isinstance(image_ptr, (int, float)):
        ptr_val = int(image_ptr)
        if record_bytes is not None and isinstance(record_bytes, (int, float)) and ptr_val > 1:
            byte_offset = (ptr_val - 1) * int(record_bytes)
        elif ptr_val > 1 and record_bytes is None:
            byte_offset = ptr_val - 1
    elif isinstance(image_ptr, (list, tuple)) and len(image_ptr) >= 2:
        try:
            ptr_val = int(image_ptr[1])
            if record_bytes is not None and isinstance(record_bytes, (int, float)) and ptr_val > 1:
                byte_offset = (ptr_val - 1) * int(record_bytes)
        except (ValueError, TypeError):
            pass

    file_size = p_img.stat().st_size
    total_elements = lines * samples
    expected_bytes = total_elements * dtype.itemsize

    # Detached file sanity: if offset + expected exceeds file size, but expected fits from 0
    if (byte_offset + expected_bytes > file_size) and (expected_bytes <= file_size):
        logger.debug(
            "PDS3 byte offset %d + %d > file size %d; falling back to offset 0 for detached raster",
            byte_offset,
            expected_bytes,
            file_size,
        )
        byte_offset = 0

    if byte_offset + expected_bytes > file_size:
        raise ValueError(
            f"File {p_img.name} size ({file_size} bytes) smaller than expected data "
            f"({expected_bytes} bytes at offset {byte_offset})"
        )

    # Read binary raster
    with open(p_img, "rb") as f:
        f.seek(byte_offset)
        arr = np.fromfile(f, dtype=dtype, count=total_elements)

    if arr.size != total_elements:
        raise IOError(f"Read {arr.size} elements, expected {total_elements} from {p_img}")

    return arr.reshape((lines, samples))
