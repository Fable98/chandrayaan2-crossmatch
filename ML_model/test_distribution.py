"""Quick verification test for spatial_distribution.py"""
import numpy as np
import sys
sys.path.insert(0, '.')

from ML_model.spatial_distribution import UniformDistributionFilter

def test_basic():
    """Test with points spread across the image"""
    filt = UniformDistributionFilter(grid_rows=4, grid_cols=4, points_per_cell=2)

    # Create 20 fake points spread across a 1000x1000 image
    np.random.seed(42)
    src_pts = np.random.uniform(0, 1000, size=(20, 2)).astype(np.float32)
    ref_pts = src_pts + np.random.uniform(-5, 5, size=(20, 2)).astype(np.float32)
    scores = np.random.uniform(0.5, 1.0, size=(20,)).astype(np.float32)

    result = filt.filter_points(src_pts, ref_pts, 1000, 1000, scores)

    print(f"Status: {result['status']}")
    print(f"Original: {result['original_count']}, Filtered: {result['filtered_count']}")
    print(f"Coverage: {result['coverage_ratio']:.2%}")
    print(f"Balance: {result['balance_score']:.2f}")
    print(f"Grid occupancy:\n{result['grid_occupancy']}")

    assert result['status'] == 'success', "Basic test failed"
    assert result['filtered_count'] <= 4 * 4 * 2, "Too many points returned"
    assert result['filtered_src_pts'].shape[1] == 2, "Wrong shape"
    print("✅ test_basic PASSED\n")

def test_clustered():
    """Test with all points in one corner (worst case)"""
    filt = UniformDistributionFilter(grid_rows=4, grid_cols=4, points_per_cell=3)

    # All points in top-left corner
    src_pts = np.random.uniform(0, 100, size=(15, 2)).astype(np.float32)
    ref_pts = src_pts.copy()

    result = filt.filter_points(src_pts, ref_pts, 1000, 1000)

    print(f"Status: {result['status']}")
    print(f"Original: {result['original_count']}, Filtered: {result['filtered_count']}")
    print(f"Coverage: {result['coverage_ratio']:.2%} (should be low)")

    assert result['status'] == 'success', "Clustered test failed"
    assert result['coverage_ratio'] < 0.5, "Coverage should be low for clustered points"
    print("✅ test_clustered PASSED\n")

def test_empty():
    """Test with empty input"""
    filt = UniformDistributionFilter()
    result = filt.filter_points(
        np.zeros((0, 2)), np.zeros((0, 2)), 1000, 1000
    )
    assert result['status'] == 'empty', "Empty test failed"
    assert result['filtered_count'] == 0
    print("✅ test_empty PASSED\n")

def test_out_of_bounds():
    """Test with points outside image"""
    filt = UniformDistributionFilter(grid_rows=2, grid_cols=2, points_per_cell=5)

    src_pts = np.array([
        [50, 50],      # valid
        [1500, 50],    # x out of bounds
        [50, -10],     # y out of bounds
        [999, 999],    # valid (edge)
    ], dtype=np.float32)
    ref_pts = src_pts.copy()

    result = filt.filter_points(src_pts, ref_pts, 1000, 1000)

    print(f"Original: {result['original_count']}, Filtered: {result['filtered_count']}")
    assert result['filtered_count'] == 2, f"Expected 2 valid points, got {result['filtered_count']}"
    print("✅ test_out_of_bounds PASSED\n")

if __name__ == "__main__":
    print("=" * 50)
    print("Running spatial_distribution.py verification tests")
    print("=" * 50)
    test_basic()
    test_clustered()
    test_empty()
    test_out_of_bounds()
    print("🎉 ALL TESTS PASSED!")
