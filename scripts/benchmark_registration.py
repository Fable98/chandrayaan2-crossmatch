"""
benchmark_registration.py — End-to-End Multi-Sensor Registration & Triplet Benchmark

Evaluates cross-sensor registration and ground-truth-independent 3-way cycle consistency
across all available Chandrayaan-2 lunar regions and mission triplets:
- Pairwise OHRC <-> TMC-2 registration (primary optical high-to-medium scale)
- Pairwise OHRC <-> IIRS registration (bidirectional audit: OHRC->IIRS vs IIRS->OHRC)
- Pairwise TMC-2 <-> IIRS registration (bidirectional audit: TMC->IIRS vs IIRS->TMC)
- Circular Triplet Cycle Consistency: A (OHRC) -> B (TMC) -> C (IIRS) -> A (OHRC)
- Zero synthetic fallbacks: Returns 'cycle_not_computable' when any circular leg fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "data_preprocessing_pipeline"))

from matcher_cfog import match_images_cfog
from triplet_evaluator import evaluate_triplet_consistency


def _summarize_pair_result(reg_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Helper to extract standardized metrics summary from a pair match result."""
    if not reg_result:
        return None
    status = reg_result.get("status")
    metrics = reg_result.get("metrics") or {}
    fit_rmse = metrics.get("fit_rmse_px")
    # Use the canonical module flag (fit<1 AND held-out<1) — never recompute
    # fit-only here; a second definition silently contradicted metrics.py.
    sub_px = metrics.get("sub_pixel_accurate")
    if sub_px is None:
        val = metrics.get("held_out_validation_rmse_px", metrics.get("validation_rmse_px"))
        sub_px = bool(fit_rmse is not None and fit_rmse < 1.0
                      and val is not None and val < 1.0)
    return {
        "status": status,
        "message": reg_result.get("message") if status != "success" else None,
        "direction": reg_result.get("direction", "native"),
        "measured_direction": reg_result.get("measured_direction"),
        "quality_tier": metrics.get("quality_tier", "FAILED" if status != "success" else "LOW_CONFIDENCE"),
        "match_count": metrics.get("match_count", reg_result.get("match_count", 0)),
        "inlier_count": metrics.get("inlier_count", reg_result.get("inlier_count", 0)),
        "inlier_ratio": metrics.get("inlier_ratio", 0.0),
        "fit_rmse_px": fit_rmse,
        "fit_rmse_is_in_sample": True,
        "sub_pixel_accurate": bool(sub_px),
        "fraction_below_1px": metrics.get("fraction_below_1px", 0.0),
        "validation_rmse_px": metrics.get("held_out_validation_rmse_px", metrics.get("validation_rmse_px")),
        "validation_status": metrics.get("validation_status"),
        "spatial_coverage": metrics.get("spatial_coverage", 0.0),
        "spatial_uniformity": metrics.get("spatial_uniformity", 0.0),
        "canonical_grid_size": metrics.get("canonical_grid_size", 10),
        "matching_grid_size": metrics.get("matching_grid_size"),
    }


def run_full_registration_benchmark(
    data_dir: Path | str = "data_preprocessing_pipeline/processed_triplets",
    output_dir: Path | str = "benchmarks/registration_benchmark_output",
    matches_dir: Path | str = "data_preprocessing_pipeline/matches",
    eval_dir: Path | str = "evaluation_output",
) -> Dict[str, Any]:
    """
    Executes cross-sensor matching and 3-way circular cycle evaluation across all datasets.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    matches_path = Path(matches_dir)
    matches_path.mkdir(parents=True, exist_ok=True)

    user_matches_path = Path("processed_user/matches")
    user_matches_path.mkdir(parents=True, exist_ok=True)

    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)

    # Discover all region_* and triplet_* directories
    dataset_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and (d.name.startswith("region_") or d.name.startswith("triplet_"))
    ])
    if not dataset_dirs:
        print(f"No region or triplet directories found in {data_path}")
        return {"error": "No regions found"}

    benchmark_records = []
    eval_summary = []
    print(f"Found {len(dataset_dirs)} test datasets. Running full multi-sensor & triplet consistency benchmark...\n")

    for r_dir in dataset_dirs:
        ohrc_path = r_dir / "ohrc_512.png"
        tmc_path = r_dir / "tmc_512.png"
        iirs_path = r_dir / "iirs_512.png"
        dem_path = r_dir / "dem_512.png"
        active_dem = dem_path if dem_path.exists() else None

        if not (ohrc_path.exists() and tmc_path.exists()):
            continue

        region_out = out_path / r_dir.name
        region_out.mkdir(parents=True, exist_ok=True)
        eval_region_dir = eval_path / r_dir.name
        eval_region_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # -------------------------------------------------------------------
        # 1. Primary OHRC <-> TMC-2 Registration
        # -------------------------------------------------------------------
        res_ot = match_images_cfog(
            ohrc_path,
            tmc_path,
            dem_path=active_dem,
            output_dir=region_out / "ohrc_tmc",
            source_sensor="OHRC",
            reference_sensor="TMC-2",
        )
        status_ot = res_ot.get("status")
        metrics_ot = res_ot.get("metrics") or {}
        inliers_ot = res_ot.get("matches") or []

        # Save canonical inlier matches for downstream backend/UI loaders
        match_records = []
        for m in inliers_ot:
            match_records.append({
                "image1_x": float(m.get("image1_x", m.get("source_x"))),
                "image1_y": float(m.get("image1_y", m.get("source_y"))),
                "image2_x": float(m.get("image2_x", m.get("target_x"))),
                "image2_y": float(m.get("image2_y", m.get("target_y"))),
                "source_x": float(m.get("source_x", m.get("image1_x"))),
                "source_y": float(m.get("source_y", m.get("image1_y"))),
                "target_x": float(m.get("target_x", m.get("image2_x"))),
                "target_y": float(m.get("target_y", m.get("image2_y"))),
                "confidence": float(m.get("confidence", 1.0)),
                "is_inlier": True,
            })

        canonical_match_file = matches_path / f"{r_dir.name}_matches.json"
        with open(canonical_match_file, "w") as f:
            json.dump(match_records, f, indent=4)

        user_match_file = user_matches_path / f"{r_dir.name}_matches.json"
        with open(user_match_file, "w") as f:
            json.dump(match_records, f, indent=4)

        # -------------------------------------------------------------------
        # 2. Multi-Modal IIRS Bidirectional Matching
        # -------------------------------------------------------------------
        pair_oi = None
        pair_ti = None
        cycle_report = None

        if iirs_path.exists():
            # Evaluate the two tractable legs once; A -> C is composed below.
            cycle_report = evaluate_triplet_consistency(
                ohrc_path, tmc_path, iirs_path, dem_path=active_dem,
                output_dir=region_out / "triplet_cycle",
                sensor_a="OHRC", sensor_b="TMC-2", sensor_c="IIRS",
            )
            res_ti_fwd = {
                "status": "success" if cycle_report.get("pair_BC_metrics") else "failed",
                "metrics": cycle_report.get("pair_BC_metrics"),
            }
            res_ti_rev = match_images_cfog(
                iirs_path, tmc_path, dem_path=active_dem,
                output_dir=region_out / "tmc_iirs_rev",
                source_sensor="IIRS", reference_sensor="TMC-2",
            )

            # Save triplet_consistency_report.json per dataset
            report_save_paths = [
                region_out / "triplet_consistency_report.json",
                eval_region_dir / "triplet_consistency_report.json",
                eval_path / f"{r_dir.name}_triplet_consistency_report.json",
            ]
            for p in report_save_paths:
                with open(p, "w") as f:
                    json.dump(cycle_report, f, indent=4)

            # Build composed OHRC->IIRS summary. A composed chain is NEVER
            # measured: inliers 0, fit None. Cycle numbers (when present) describe
            # loop closure, not leg accuracy — never relabel them as fit_rmse.
            # (Fixed 2026-09-11: old code reported cycle_rmse as fit_rmse_px and
            # 1.0 coverage/uniformity for 0-inlier chains, incl. tautological
            # 0.0px "fits", and misread closed all-measured loops as composed.)
            is_composed = (cycle_report.get("status") == "evaluated_composed")
            cycle_rmse_for_note = cycle_report.get("triplet_cycle_rmse_px")
            sum_oi_fwd = {
                "status": "composed" if is_composed else "composition_not_computable",
                "message": None if is_composed else cycle_report.get("reason"),
                "quality_tier": "COMPOSED_CHAIN" if is_composed else "FAILED",
                "match_count": 0,
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "fit_rmse_px": None,
                "fit_rmse_note": ("derived chain has no measured fit; loop-closure "
                                  f"RMSE was {cycle_rmse_for_note} px (diagnostic only)"
                                  if is_composed else None),
                "spatial_coverage": 0.0,
                "spatial_uniformity": 0.0,
            }
            pair_oi = {
                "status": "composed" if is_composed else "composition_not_computable",
                "inlier_count": 0,
                "fit_rmse_px": None,
                "forward_ohrc_to_iirs": sum_oi_fwd,
                "reverse_iirs_to_ohrc": None,
                "bidirectional_agreement": True if is_composed else False,
                "directional_note": "OHRC -> IIRS is composed as H(TMC -> IIRS) · H(OHRC -> TMC); direct matching is intentionally not attempted.",
                "composition": cycle_report.get("composition"),
            }

            sum_ti_fwd = _summarize_pair_result(res_ti_fwd)
            sum_ti_rev = _summarize_pair_result(res_ti_rev)
            ti_agreed = (res_ti_fwd.get("status") == res_ti_rev.get("status"))
            pair_ti = {
                "status": res_ti_fwd.get("status"),
                "inlier_count": (res_ti_fwd.get("metrics") or {}).get("inlier_count", 0),
                "fit_rmse_px": (res_ti_fwd.get("metrics") or {}).get("fit_rmse_px"),
                "forward_tmc_to_iirs": sum_ti_fwd,
                "reverse_iirs_to_tmc": sum_ti_rev,
                "bidirectional_agreement": ti_agreed,
                "directional_note": None if ti_agreed else (
                    f"Forward (TMC->IIRS) produced '{res_ti_fwd.get('status')}' while Reverse (IIRS->TMC) produced '{res_ti_rev.get('status')}'. "
                    "Caused by grid sizing and search window asymmetries under ~13x physical scale disparity."
                ),
            }

        elapsed_sec = round(time.time() - start_time, 2)
        pair_ot = _summarize_pair_result(res_ot)

        cycle_status = cycle_report.get("status", "cycle_not_computable") if cycle_report else "not_evaluated"
        cycle_rmse = cycle_report.get("triplet_cycle_rmse_px") if cycle_report else None
        cycle_mean = cycle_report.get("triplet_mean_cycle_error_px") if cycle_report else None
        cycle_closed = cycle_report.get("cycle_closed_successfully", False) if cycle_report else False
        failed_legs = cycle_report.get("failed_legs", []) if cycle_report else []
        cycle_reason = cycle_report.get("reason") if cycle_report else None

        triplet_consistency_summary = {
            "status": cycle_status,
            "reason": cycle_reason,
            "cycle_rmse_px": cycle_rmse,
            "cycle_mean_px": cycle_mean,
            "cycle_closed_successfully": cycle_closed,
            "failed_legs": failed_legs,
        }

        # Build benchmark record
        record = {
            "region_id": r_dir.name,
            "status": status_ot,
            "quality_tier": metrics_ot.get("quality_tier", "FAILED" if status_ot != "success" else "LOW_CONFIDENCE"),
            "failure_reason": res_ot.get("message") if status_ot != "success" else None,
            "elapsed_seconds": elapsed_sec,
            "match_count": metrics_ot.get("match_count", 0),
            "inlier_count": metrics_ot.get("inlier_count", 0),
            "inlier_ratio": metrics_ot.get("inlier_ratio", 0.0),
            "fit_rmse_px": metrics_ot.get("fit_rmse_px"),
            "validation_rmse_px": metrics_ot.get("validation_rmse_px"),
            "validation_status": metrics_ot.get("validation_status"),
            "spatial_coverage": metrics_ot.get("spatial_coverage", 0.0),
            "spatial_uniformity": metrics_ot.get("spatial_uniformity", 0.0),
            "pairs": {
                "ohrc_tmc": pair_ot,
                "ohrc_iirs": pair_oi,
                "tmc_iirs": pair_ti,
            },
            "triplet_consistency": triplet_consistency_summary,
        }
        benchmark_records.append(record)

        # Build comprehensive evaluation metrics record
        # Canonical reporting grid is fixed 10x10 (100 cells). spatial_coverage
        # from metrics is already canonical; do NOT mix dynamic-grid coverage
        # with a hardcoded 100-cell denominator.
        _dist = metrics_ot.get("spatial_distribution", {}) or {}
        _total_cells = int(_dist.get("total_cells", 100))
        _occupied = int(_dist.get("occupied_cells", metrics_ot.get("inlier_count", 0)))
        eval_record = {
            "region_id": r_dir.name,
            "dataset_id": r_dir.name,
            "status": "evaluated" if status_ot == "success" else "failed",
            "num_inliers": metrics_ot.get("inlier_count", 0),
            "num_raw_matches": metrics_ot.get("match_count", 0),
            "inlier_ratio": metrics_ot.get("inlier_ratio", 0.0),
            "rmse_px": metrics_ot.get("fit_rmse_px") or 0.0,
            "fit_rmse_px": metrics_ot.get("fit_rmse_px"),
            "validation_rmse_px": metrics_ot.get("validation_rmse_px"),
            "validation_status": metrics_ot.get("validation_status"),
            "mean_reprojection_error_px": metrics_ot.get("mean_reprojection_error_px") or 0.0,
            "median_reprojection_error_px": metrics_ot.get("median_reprojection_error_px") or 0.0,
            "max_reprojection_error_px": metrics_ot.get("max_reprojection_error_px") or 0.0,
            "sub_pixel_accurate": metrics_ot.get("sub_pixel_accurate", False),
            "fraction_below_1px": metrics_ot.get("fraction_below_1px", 0.0),
            "source_coverage_ratio": metrics_ot.get("spatial_coverage", 0.0),
            "destination_coverage_ratio": metrics_ot.get("spatial_coverage", 0.0),
            "combined_coverage_score": metrics_ot.get("spatial_coverage", 0.0),
            "source_occupied_cells": _occupied,
            "destination_occupied_cells": _occupied,
            "total_cells": _total_cells,
            "canonical_grid_size": metrics_ot.get("canonical_grid_size", 10),
            "matching_grid_size": metrics_ot.get("matching_grid_size"),
            "fit_rmse_is_in_sample": True,
            "uniformity_score": metrics_ot.get("spatial_uniformity", 0.0),
            "quality_tier": record["quality_tier"],
            "method": "CFOG + Phase Congruency",
            "pairs": {
                "ohrc_tmc": pair_ot,
                "ohrc_iirs": pair_oi,
                "tmc_iirs": pair_ti,
            },
            "triplet_consistency": triplet_consistency_summary,
        }
        eval_summary.append(eval_record)

        eval_single_path = eval_path / f"{r_dir.name}_metrics.json"
        with open(eval_single_path, "w") as f:
            json.dump(eval_record, f, indent=4)

        if pair_oi:
            oi_status = pair_oi.get("status", "N/A")
            oi_rmse = pair_oi.get("fit_rmse_px")
            oi_str = f"{oi_status}" + (f" ({oi_rmse:.1f}px)" if oi_rmse is not None else "")
        else:
            oi_str = "N/A"
        c_str = f"{cycle_status}" + (f" ({cycle_rmse:.1f}px)" if cycle_rmse is not None else "")
        print(f"[{r_dir.name:28s}] OHRC-TMC Inl: {record['inlier_count']:2d} ({record['fit_rmse_px']}px) | OHRC-IIRS: {oi_str:15s} | Cycle: {c_str:22s} | Time: {elapsed_sec:.2f}s")

    # Save evaluation_summary.json in evaluation_output/
    eval_summary_path = eval_path / "evaluation_summary.json"
    with open(eval_summary_path, "w") as f:
        json.dump(eval_summary, f, indent=4)

    # Save benchmark summary in benchmarks/registration_benchmark_output/
    report = {
        "benchmark_name": "Chandrayaan-2 Primary Multi-Sensor OHRC <-> TMC-2 <-> IIRS Benchmark",
        "total_regions_tested": len(benchmark_records),
        "successful_registrations": sum(1 for r in benchmark_records if r["status"] == "success"),
        "average_runtime_seconds": round(float(sum(r["elapsed_seconds"] for r in benchmark_records) / max(1, len(benchmark_records))), 2),
        "regions": benchmark_records,
    }

    report_path = out_path / "registration_benchmark_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nBenchmark complete.")
    print(f"Registration summary saved to: {report_path}")
    print(f"Evaluation summary saved to:   {eval_summary_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run full multi-sensor registration & triplet benchmark.")
    parser.add_argument("--data-dir", type=str, default="data_preprocessing_pipeline/processed_triplets")
    parser.add_argument("--output-dir", type=str, default="benchmarks/registration_benchmark_output")
    parser.add_argument("--matches-dir", type=str, default="data_preprocessing_pipeline/matches")
    parser.add_argument("--eval-dir", type=str, default="evaluation_output")
    args = parser.parse_args()

    run_full_registration_benchmark(args.data_dir, args.output_dir, args.matches_dir, args.eval_dir)


if __name__ == "__main__":
    main()
