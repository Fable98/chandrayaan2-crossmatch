import numpy as np
import pytest
from ML_model.spatial_suppression import (
    apply_grid_density_budgeting,
    extract_match_covariance_properties,
    compute_covariance_aware_confidence,
    compute_cell_covariance_statistics,
)


def test_covariance_properties_extraction_and_classification():
    """
    Verifies that extract_match_covariance_properties accurately extracts semi-axes,
    orientation, and properly classifies points into isotropic, elongated, and uncertain.
    """
    # 1. Isotropic sharp point
    m_iso = {
        "work_x1": 100.0, "work_y1": 100.0,
        "covariance_xy": [[0.04, 0.0], [0.0, 0.04]],  # sigma = 0.2 px
        "score": 0.85,
    }
    p_iso = extract_match_covariance_properties(m_iso)
    assert p_iso["is_isotropic"] is True
    assert p_iso["is_elongated"] is False
    assert p_iso["is_uncertain"] is False
    assert pytest.approx(p_iso["sigma_major"], abs=1e-3) == 0.2
    assert pytest.approx(p_iso["sigma_minor"], abs=1e-3) == 0.2

    # 2. Elongated sharp ridge point (sigma_major = 2.0, sigma_minor = 0.1)
    m_elong = {
        "work_x1": 150.0, "work_y1": 150.0,
        "sigma_major_px": 2.0,
        "sigma_minor_px": 0.1,
        "ellipse_angle_deg": 45.0,
        "score": 0.85,
    }
    p_elong = extract_match_covariance_properties(m_elong)
    assert p_elong["is_elongated"] is True
    assert p_elong["is_isotropic"] is False
    assert p_elong["anisotropy"] >= 15.0

    # 3. Flat / multimodal uncertain point (sigma_major = 3.0, sigma_minor = 2.5)
    m_unc = {
        "work_x1": 200.0, "work_y1": 200.0,
        "sigma_major_px": 3.0,
        "sigma_minor_px": 2.5,
        "score": 0.90,
    }
    p_unc = extract_match_covariance_properties(m_unc)
    assert p_unc["is_uncertain"] is True
    assert p_unc["is_elongated"] is False


def test_penalize_high_uncertainty_and_preserve_useful_elongated_points():
    """
    Requirements 2, 3, 4, 5:
    - Penalize high uncertainty.
    - Do not select only isotropic points.
    - Preserve useful elongated points when their constrained direction is valuable.
    """
    # Candidate 1: Sharp isotropic point (sigma = 0.2)
    c_sharp_iso = {"score": 0.80, "sigma_major_px": 0.2, "sigma_minor_px": 0.2}
    conf_sharp_iso = compute_covariance_aware_confidence(c_sharp_iso)

    # Candidate 2: Sharp elongated point (sigma_major = 2.2, sigma_minor = 0.08) - useful ridge
    c_sharp_elong = {"score": 0.80, "sigma_major_px": 2.2, "sigma_minor_px": 0.08}
    conf_sharp_elong = compute_covariance_aware_confidence(c_sharp_elong)

    # Candidate 3: Blurry isotropic point (sigma = 1.8) - high uncertainty
    c_blurry_iso = {"score": 0.80, "sigma_major_px": 1.8, "sigma_minor_px": 1.8}
    conf_blurry_iso = compute_covariance_aware_confidence(c_blurry_iso)

    # Candidate 4: Blurry multimodal point (sigma_major = 3.5, sigma_minor = 2.8, high correlation score 0.95)
    c_blurry_multi = {"score": 0.95, "sigma_major_px": 3.5, "sigma_minor_px": 2.8}
    conf_blurry_multi = compute_covariance_aware_confidence(c_blurry_multi)

    # Requirement 3: Blurry points are heavily penalized
    assert conf_sharp_iso > conf_blurry_iso
    assert conf_sharp_elong > conf_blurry_iso
    assert conf_sharp_elong > conf_blurry_multi

    # Requirements 4 & 5: The sharp elongated point comfortably beats blurry isotropic points!
    # Useful elongated points are preserved when their constrained direction (0.08 px) is valuable.
    assert conf_sharp_elong > 1.5 * conf_blurry_iso


def test_clustered_synthetic_experiment_preserves_grid_occupancy():
    """
    Requirement 1 & 7: Clustered Synthetic Test.
    Two hotspot cells contain 50 candidate matches each.
    Eight sparse cells contain only 1 candidate match each.
    
    Verifies:
    - Hard grid occupancy constraint: all 10 occupied cells receive representation in Round 1.
    - Hotspots do not monopolize the candidate pool and starve sparse cells.
    """
    image_shape = (500, 500)
    grid_dims = (10, 10)  # Each cell is 50x50 px
    rng = np.random.default_rng(42)

    matches = []

    # 1. Hotspot 1 in cell (1, 1): x in [55, 95], y in [55, 95]
    for _ in range(50):
        matches.append({
            "work_x1": float(rng.uniform(55, 95)),
            "work_y1": float(rng.uniform(55, 95)),
            "score": float(rng.uniform(0.85, 0.98)),
            "sigma_major_px": float(rng.uniform(0.3, 0.6)),
            "sigma_minor_px": float(rng.uniform(0.2, 0.4)),
        })

    # 2. Hotspot 2 in cell (2, 2): x in [105, 145], y in [105, 145]
    for _ in range(50):
        matches.append({
            "work_x1": float(rng.uniform(105, 145)),
            "work_y1": float(rng.uniform(105, 145)),
            "score": float(rng.uniform(0.85, 0.98)),
            "sigma_major_px": float(rng.uniform(0.3, 0.6)),
            "sigma_minor_px": float(rng.uniform(0.2, 0.4)),
        })

    # 3. Eight sparse cells: exactly 1 match each across diverse cells
    sparse_cells = [(0, 0), (3, 7), (5, 5), (6, 2), (8, 8), (9, 1), (1, 9), (7, 4)]
    for gx, gy in sparse_cells:
        matches.append({
            "work_x1": float(gx * 50 + 25),
            "work_y1": float(gy * 50 + 25),
            "score": 0.70,  # Lower score than hotspot points
            "sigma_major_px": 0.5,
            "sigma_minor_px": 0.4,
        })

    # Total occupied cells = 2 (hotspots) + 8 (sparse) = 10 cells
    selected, stats = apply_grid_density_budgeting(
        matches,
        image_shape=image_shape,
        grid_dims=grid_dims,
        max_per_cell=3,
        return_cell_stats=True,
    )

    # Requirement 1: Hard grid occupancy constraint
    # All 10 occupied cells MUST have at least 1 match selected!
    selected_cells = set()
    for m in selected:
        gx = int(m["work_x1"] // 50)
        gy = int(m["work_y1"] // 50)
        selected_cells.add((gx, gy))

    assert len(selected_cells) == 10
    for sc in sparse_cells:
        assert sc in selected_cells, f"Sparse cell {sc} was starved by hotspot clustering!"

    assert stats["global_statistics"]["occupied_cells_count"] == 10


def test_anisotropic_synthetic_experiment_in_single_cell():
    """
    Requirements 4, 5, 7: Anisotropic Synthetic Test.
    Inside a single cell with 6 candidates:
    - 2 sharp isotropic points
    - 2 sharp elongated points (useful edge/ridge constraints)
    - 2 blurry isotropic/multimodal points
    
    Verifies:
    - Both isotropic and elongated points are selected when max_per_cell=3.
    - Blurry/uncertain points are suppressed.
    - Selection is NOT exclusively isotropic.
    """
    image_shape = (500, 500)
    grid_dims = (10, 10)

    # All points inside cell (3, 3): x in [150, 200], y in [150, 200]
    cell_matches = [
        # Candidate 0: Sharp isotropic
        {"work_x1": 160.0, "work_y1": 160.0, "score": 0.88, "sigma_major_px": 0.20, "sigma_minor_px": 0.20, "ellipse_angle_deg": 0.0, "id": "sharp_iso_1"},
        # Candidate 1: Sharp elongated horizontal ridge (very sharp across y, sigma_y = 0.08)
        {"work_x1": 165.0, "work_y1": 165.0, "score": 0.86, "sigma_major_px": 2.20, "sigma_minor_px": 0.08, "ellipse_angle_deg": 0.0, "id": "sharp_elong_h"},
        # Candidate 2: Sharp elongated vertical ridge (very sharp across x, sigma_x = 0.08)
        {"work_x1": 170.0, "work_y1": 170.0, "score": 0.85, "sigma_major_px": 2.10, "sigma_minor_px": 0.08, "ellipse_angle_deg": 90.0, "id": "sharp_elong_v"},
        # Candidate 3: Blurry isotropic
        {"work_x1": 175.0, "work_y1": 175.0, "score": 0.85, "sigma_major_px": 2.00, "sigma_minor_px": 2.00, "ellipse_angle_deg": 0.0, "id": "blurry_iso"},
        # Candidate 4: Blurry multimodal
        {"work_x1": 180.0, "work_y1": 180.0, "score": 0.92, "sigma_major_px": 3.80, "sigma_minor_px": 2.90, "ellipse_angle_deg": 30.0, "id": "blurry_multi"},
    ]

    selected, stats = apply_grid_density_budgeting(
        cell_matches,
        image_shape=image_shape,
        grid_dims=grid_dims,
        max_per_cell=3,
        return_cell_stats=True,
    )

    selected_ids = [m["id"] for m in selected]

    # Must retain 3 matches
    assert len(selected) == 3

    # Requirement 4 & 5: Both isotropic and elongated points are selected
    assert any("iso" in mid for mid in selected_ids)
    assert any("elong" in mid for mid in selected_ids)

    # Blurry points must be rejected despite high correlation score
    assert "blurry_iso" not in selected_ids
    assert "blurry_multi" not in selected_ids

    # Check cell stats report
    cell_stat = stats["per_cell_statistics"]["cell_3_3"]
    assert cell_stat["num_selected"] == 3
    assert cell_stat["counts_by_type"]["elongated"] >= 1
    assert cell_stat["counts_by_type"]["isotropic"] >= 1
    assert cell_stat["counts_by_type"]["uncertain"] == 0


def test_uniform_synthetic_experiment_across_grid():
    """
    Requirements 6 & 7: Uniform Synthetic Test.
    Points distributed uniformly across an 8x8 grid with heterogeneous noise.
    Verifies:
    - Uniform grid occupancy is maintained.
    - Selected-point covariance statistics are correctly reported per cell and globally.
    """
    image_shape = (400, 400)
    grid_dims = (8, 8)  # 50x50 px per cell
    rng = np.random.default_rng(99)

    matches = []
    # Place 3 matches in each of the 64 cells
    for gx in range(8):
        for gy in range(8):
            # Candidate A: sharp isotropic
            matches.append({
                "work_x1": float(gx * 50 + 10),
                "work_y1": float(gy * 50 + 10),
                "score": float(rng.uniform(0.80, 0.90)),
                "sigma_major_px": float(rng.uniform(0.2, 0.4)),
                "sigma_minor_px": float(rng.uniform(0.2, 0.4)),
            })
            # Candidate B: sharp elongated
            matches.append({
                "work_x1": float(gx * 50 + 25),
                "work_y1": float(gy * 50 + 25),
                "score": float(rng.uniform(0.78, 0.88)),
                "sigma_major_px": float(rng.uniform(1.8, 2.5)),
                "sigma_minor_px": float(rng.uniform(0.08, 0.15)),
                "ellipse_angle_deg": float(rng.uniform(0, 180)),
            })
            # Candidate C: blurry/uncertain
            matches.append({
                "work_x1": float(gx * 50 + 40),
                "work_y1": float(gy * 50 + 40),
                "score": float(rng.uniform(0.70, 0.85)),
                "sigma_major_px": float(rng.uniform(2.5, 4.0)),
                "sigma_minor_px": float(rng.uniform(2.0, 3.5)),
            })

    selected, stats = apply_grid_density_budgeting(
        matches,
        image_shape=image_shape,
        grid_dims=grid_dims,
        max_per_cell=2,
        return_cell_stats=True,
    )

    # 1. Uniform occupancy: all 64 cells are occupied
    assert stats["global_statistics"]["occupied_cells_count"] == 64
    assert stats["global_statistics"]["occupancy_rate"] == 1.0
    assert len(selected) == 128  # 2 per cell * 64 cells

    # 2. Both isotropic and elongated points are retained in global statistics
    assert stats["global_statistics"]["total_isotropic"] > 0
    assert stats["global_statistics"]["total_elongated"] > 0
    # Uncertain points should be 0 because each cell has 2 sharp candidates
    assert stats["global_statistics"]["total_uncertain"] == 0

    # 3. Check that per-cell statistics are present for all cells
    assert len(stats["per_cell_statistics"]) == 64
    for gx in range(8):
        for gy in range(8):
            key = f"cell_{gx}_{gy}"
            assert key in stats["per_cell_statistics"]
            cs = stats["per_cell_statistics"][key]
            assert cs["num_selected"] == 2
            assert cs["mean_sigma_minor_px"] < 0.5  # Filtered out blurry points
