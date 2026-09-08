"""
demo_registration.py — Deterministic CLI Demonstration Script

Executes primary OHRC <-> TMC-2 registration with DEM relief displacement compensation,
producing a complete reproducible output package with full provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add ML_model to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML_model"))
from matcher_cfog import match_images_cfog


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate Chandrayaan-2 cross-sensor registration."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png",
        help="Path to source image (e.g. OHRC)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data_preprocessing_pipeline/processed_triplets/region_001/tmc_512.png",
        help="Path to reference image (e.g. TMC-2)",
    )
    parser.add_argument(
        "--dem",
        type=str,
        default="data_preprocessing_pipeline/processed_triplets/region_001/dem_512.png",
        help="Path to DEM elevation map",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="registration_output_demo",
        help="Output destination directory",
    )

    args = parser.parse_args()

    print(f"================================================================")
    print(f" Chandrayaan-2 Cross-Sensor Registration Demo")
    print(f" Source:    {args.source}")
    print(f" Reference: {args.reference}")
    print(f" DEM:       {args.dem}")
    print(f" Output:    {args.output}")
    print(f"================================================================\n")

    res = match_images_cfog(
        args.source,
        args.reference,
        dem_path=args.dem if os.path.exists(args.dem) else None,
        output_dir=args.output,
        source_sensor="OHRC",
        reference_sensor="TMC-2",
    )

    status = res.get("status")
    print(f"Registration Status: {status.upper()}")

    if status == "success":
        metrics = res.get("metrics") or {}
        print(f"\nQuantitative Evaluation Telemetry:")
        print(f"  - Matches Found:        {metrics.get('match_count')}")
        print(f"  - Verified Inliers:     {metrics.get('inlier_count')}")
        print(f"  - Inlier Ratio:         {metrics.get('inlier_ratio') * 100:.1f}%")
        print(f"  - In-Sample Fit RMSE:   {metrics.get('fit_rmse_px')} px")
        print(f"  - Validation Status:    {metrics.get('validation_status')}")
        print(f"  - Validation RMSE:      {metrics.get('validation_rmse_px')}")
        print(f"  - Spatial Coverage:     {metrics.get('spatial_coverage') * 100:.1f}%")
        print(f"  - Uniformity Score:     {metrics.get('spatial_uniformity')}")

        outputs = res.get("outputs") or {}
        print(f"\nGenerated Products:")
        for name, path in outputs.items():
            print(f"  [{name:18s}] -> {path}")
    else:
        print(f"\nDiagnostic: {res.get('message')}")

    print(f"\nDemo completed.")


if __name__ == "__main__":
    main()
