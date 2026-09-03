"""
benchmark_registration.py — End-to-End Registration Performance Benchmark

Evaluates the primary OHRC <-> TMC-2 cross-sensor registration pipeline
across available mission regions, reporting real, un-fabricated telemetry:
- Match count and RANSAC inlier count.
- True inlier ratio.
- In-sample Fit RMSE vs. Independent Held-Out Validation RMSE.
- Spatial coverage and uniformity.
- Wall-clock runtime per region.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add ML_model to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML_model"))
from matcher_cfog import match_images_cfog


def run_full_registration_benchmark(
    data_dir: Path | str = "data_preprocessing_pipeline/processed_triplets",
    output_dir: Path | str = "benchmarks/registration_benchmark_output",
) -> Dict[str, Any]:
    """
    Executes registration across all discoverable region directories.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    region_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("region_")])
    if not region_dirs:
        print(f"No region directories found in {data_path}")
        return {"error": "No regions found"}

    results = []
    print(f"Found {len(region_dirs)} test regions. Running cross-sensor registration benchmark...\n")

    for r_dir in region_dirs:
        ohrc_path = r_dir / "ohrc_512.png"
        tmc_path = r_dir / "tmc_512.png"
        dem_path = r_dir / "dem_512.png"

        if not (ohrc_path.exists() and tmc_path.exists()):
            continue

        region_out = out_path / r_dir.name
        start_time = time.time()
        
        reg_result = match_images_cfog(
            ohrc_path,
            tmc_path,
            dem_path=dem_path if dem_path.exists() else None,
            output_dir=region_out,
            source_sensor="OHRC",
            reference_sensor="TMC-2",
        )
        elapsed_sec = round(time.time() - start_time, 2)

        status = reg_result.get("status")
        metrics = reg_result.get("metrics") or {}

        record = {
            "region_id": r_dir.name,
            "status": status,
            "elapsed_seconds": elapsed_sec,
            "match_count": metrics.get("match_count", 0),
            "inlier_count": metrics.get("inlier_count", 0),
            "inlier_ratio": metrics.get("inlier_ratio", 0.0),
            "fit_rmse_px": metrics.get("fit_rmse_px"),
            "validation_rmse_px": metrics.get("validation_rmse_px"),
            "validation_status": metrics.get("validation_status"),
            "spatial_coverage": metrics.get("spatial_coverage", 0.0),
            "spatial_uniformity": metrics.get("spatial_uniformity", 0.0),
        }
        results.append(record)
        print(f"[{r_dir.name}] Status: {status:28s} | Inliers: {record['inlier_count']:2d} | Fit RMSE: {str(record['fit_rmse_px']):6s} | Time: {elapsed_sec:.2f}s")

    report = {
        "benchmark_name": "Chandrayaan-2 Primary OHRC <-> TMC-2 Cross-Sensor Benchmark",
        "total_regions_tested": len(results),
        "successful_registrations": sum(1 for r in results if r["status"] == "success"),
        "average_runtime_seconds": round(float(sum(r["elapsed_seconds"] for r in results) / max(1, len(results))), 2),
        "regions": results,
    }

    report_path = out_path / "registration_benchmark_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nBenchmark finished. Full summary saved to: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run cross-sensor registration benchmark.")
    parser.add_argument("--data-dir", type=str, default="data_preprocessing_pipeline/processed_triplets")
    parser.add_argument("--output-dir", type=str, default="benchmarks/registration_benchmark_output")
    args = parser.parse_args()

    run_full_registration_benchmark(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
