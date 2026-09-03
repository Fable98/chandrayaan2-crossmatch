"""
data_preprocessing_pipeline/triplet_evaluator.py — Ground-Truth-Independent Triplet Consistency Evaluator

Computes closed-loop cycle consistency error across 3 sensor perspectives (A -> B -> C -> A).
Single-pair RANSAC RMSE is biased because it only evaluates points that fit its own fitted model.
Cycle consistency provides an unbiased, ground-truth-independent measure of multi-sensor geometric fidelity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

# Add project roots
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML_model"))
from matcher_cfog import match_images_cfog
from metrics import compute_triplet_consistency


def evaluate_triplet_consistency(
    image_a_path: str | Path,
    image_b_path: str | Path,
    image_c_path: str | Path,
    dem_path: str | Path | None = None,
    output_dir: str | Path = "triplet_evaluation_output",
    num_test_points: int = 100,
) -> Dict[str, Any]:
    """
    Executes full 3-way circular cross-registration:
    1. A (OHRC) -> B (TMC)
    2. B (TMC) -> C (IIRS)
    3. C (IIRS) -> A (OHRC)
    Computes closed-loop cycle error: A -> B -> C -> A.
    """
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # 1. Match A -> B
    res_AB = match_images_cfog(image_a_path, image_b_path, dem_path=dem_path, output_dir=out_base / "AB")
    # 2. Match B -> C
    res_BC = match_images_cfog(image_b_path, image_c_path, dem_path=dem_path, output_dir=out_base / "BC")
    # 3. Match C -> A
    res_CA = match_images_cfog(image_c_path, image_a_path, dem_path=dem_path, output_dir=out_base / "CA")

    H_AB = np.array(res_AB.get("homography") or np.eye(3))
    H_BC = np.array(res_BC.get("homography") or np.eye(3))
    H_CA = np.array(res_CA.get("homography") or np.eye(3))

    cycle_rmse, cycle_mean = compute_triplet_consistency(
        H_AB, H_BC, H_CA, image_shape=(512, 512), num_test_points=num_test_points
    )

    evaluation_report = {
        "triplet_cycle_rmse_px": float(cycle_rmse),
        "triplet_mean_cycle_error_px": float(cycle_mean),
        "cycle_closed_successfully": bool(cycle_rmse < 1.5),
        "pair_AB_metrics": res_AB.get("metrics"),
        "pair_BC_metrics": res_BC.get("metrics"),
        "pair_CA_metrics": res_CA.get("metrics"),
    }

    report_path = out_base / "triplet_consistency_report.json"
    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=4)

    return evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3-way Triplet Consistency (A -> B -> C -> A).")
    parser.add_argument("img_a", type=str, help="Path to Image A (e.g. OHRC)")
    parser.add_argument("img_b", type=str, help="Path to Image B (e.g. TMC)")
    parser.add_argument("img_c", type=str, help="Path to Image C (e.g. IIRS)")
    parser.add_argument("--dem", type=str, default=None, help="Path to DEM")
    parser.add_argument("--output", type=str, default="triplet_evaluation_output", help="Output directory")

    args = parser.parse_args()
    report = evaluate_triplet_consistency(args.img_a, args.img_b, args.img_c, dem_path=args.dem, output_dir=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
