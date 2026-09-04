"""
tests/test_spatial_uniformity.py — Validation of Grid-based Non-Maximum Suppression (Grid NMS) and PDS4 Parsing.
"""

from collections import Counter
import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data" / "ingestion"))

from matcher_cfog import apply_grid_nms
try:
    from data.ingestion.pds4_reader import parse_pds4_or_vicar_label, PDS4ProductInfo
except (ImportError, ModuleNotFoundError):
    from pds4_reader import parse_pds4_or_vicar_label, PDS4ProductInfo


def test_grid_nms_enforces_spatial_uniformity_on_1000_clustered_points():
    """
    CORE VALIDATION: Generates 1,000 match points heavily clustered on a crater rim in cell (1, 1),
    passes them through Grid NMS with a 10x10 grid (max_per_cell=5), and asserts:
    1. The clustered cell is capped at exactly 5 points (preventing crater-rim clustering).
    2. The retained points are evenly distributed across all occupied cells without any cell exceeding 5.
    """
    np.random.seed(42)
    image_shape = (1000, 1000)
    grid_dims = (10, 10)
    max_per_cell = 5

    # Generate 1,000 matches:
    # 850 points clustered inside cell (1, 1): x in [100, 199], y in [100, 199]
    # 150 points scattered across other cells
    clustered_matches = []

    # 900 clustered points
    for i in range(900):
        cx = float(np.random.uniform(105, 195))
        cy = float(np.random.uniform(105, 195))
        clustered_matches.append({
            "source_x": cx,
            "source_y": cy,
            "target_x": cx + 1.0,
            "target_y": cy + 1.0,
            "score": float(np.random.uniform(0.5, 0.99)),
        })

    # 150 distributed points across distinct cells
    for gx in range(10):
        for gy in range(10):
            if (gx, gy) == (1, 1):
                continue
            # Place 1 to 3 points in other cells
            for _ in range(np.random.randint(1, 3)):
                px = float(gx * 100 + np.random.uniform(10, 90))
                py = float(gy * 100 + np.random.uniform(10, 90))
                clustered_matches.append({
                    "source_x": px,
                    "source_y": py,
                    "target_x": px + 1.0,
                    "target_y": py + 1.0,
                    "score": float(np.random.uniform(0.4, 0.9)),
                })

    total_input = len(clustered_matches)
    assert total_input >= 1000, f"Expected >= 1000 input matches, got {total_input}"

    # Pass through Grid NMS
    filtered = apply_grid_nms(
        clustered_matches,
        image_shape=image_shape,
        grid_dims=grid_dims,
        max_per_cell=max_per_cell,
    )

    # Calculate per-cell distribution of filtered points
    cell_counts = Counter()
    for m in filtered:
        gx = int(m["source_x"] // 100)
        gy = int(m["source_y"] // 100)
        cell_counts[(gx, gy)] += 1

    print(f"\n[Grid NMS Test] Input: {total_input} points -> Output: {len(filtered)} points")
    print(f"[Grid NMS Test] Points in clustered cell (1, 1): {cell_counts[(1, 1)]} (was 850)")
    print(f"[Grid NMS Test] Number of occupied grid cells: {len(cell_counts)}")

    # 1. Clustered cell must be strictly capped at max_per_cell
    assert cell_counts[(1, 1)] == max_per_cell, (
        f"Clustered cell should have exactly {max_per_cell} points, but had {cell_counts[(1, 1)]}"
    )

    # 2. NO cell may exceed max_per_cell
    for cell, count in cell_counts.items():
        assert count <= max_per_cell, f"Cell {cell} exceeded max_per_cell with {count} points"

    # 3. Points must span multiple distinct cells (uniform distribution)
    assert len(cell_counts) >= 50, (
        f"Expected distribution across >= 50 grid cells, occupied {len(cell_counts)}"
    )


def test_pds4_reader_extracts_metadata(tmp_path):
    """
    Validates that pds4_reader parses PDS4 XML and extracts dimensions, bit-depth,
    GSD, Sun azimuth/elevation, and SPICE references.
    """
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
    <Identification_Area>
        <logical_identifier>urn:isro:ch2:ch2_ohr_ncp_2020</logical_identifier>
        <title>CH2 OHRC Calibrated Swath</title>
    </Identification_Area>
    <Observation_Area>
        <Observing_System_Component>
            <name>OHRC</name>
        </Observing_System_Component>
    </Observation_Area>
    <File_Area_Observational>
        <file_name>ch2_ohr_test.png</file_name>
        <Array_2D_Image>
            <data_type>UnsignedByte</data_type>
            <Axis_Array>
                <axis_name>Line</axis_name>
                <elements>1024</elements>
            </Axis_Array>
            <Axis_Array>
                <axis_name>Sample</axis_name>
                <elements>1024</elements>
            </Axis_Array>
        </Array_2D_Image>
    </File_Area_Observational>
    <cart:Spatial_Reference_Information xmlns:cart="http://pds.nasa.gov/pds4/cart/v1">
        <cart:pixel_resolution>0.25</cart:pixel_resolution>
    </cart:Spatial_Reference_Information>
    <geom:Geometry xmlns:geom="http://pds.nasa.gov/pds4/geom/v1">
        <geom:sun_azimuth>135.5</geom:sun_azimuth>
        <geom:sun_elevation>42.0</geom:sun_elevation>
        <geom:spice_kernel_file>ch2_orb_v01.bsp</geom:spice_kernel_file>
        <geom:spice_kernel_file>moon_pa_de421.bpc</geom:spice_kernel_file>
    </geom:Geometry>
</Product_Observational>"""

    label_file = tmp_path / "test_label.xml"
    label_file.write_text(xml_content, encoding="utf-8")

    info = parse_pds4_or_vicar_label(label_file)

    assert info.sensor == "OHRC"
    assert info.lines == 1024
    assert info.samples == 1024
    assert info.bit_depth == 8
    assert info.gsd_m == 0.25
    assert info.sun_azimuth_deg == 135.5
    assert info.sun_elevation_deg == 42.0
    assert "ch2_orb_v01.bsp" in info.spice_kernels
    assert "moon_pa_de421.bpc" in info.spice_kernels
