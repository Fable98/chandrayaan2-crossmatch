"""
evaluation/run_ablation.py — Systematic Ablation Study and Baseline Comparison Runner

Executes 4-way comparative ablation study on planetary cross-sensor imagery:
(a) Pure SIFT (Standard OpenCV SIFT)
(b) Pure LoFTR (Deep dense matcher via Kornia)
(c) Our Pipeline without DEM (Structural CFOG + GSD, planar homography)
(d) Our Full Pipeline (Structural CFOG + GSD + Piecewise DEM Ray-Intersection + Grid NMS)

Outputs Markdown and HTML comparative telemetry tables and triggers automated failure diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "baselines"))

from baselines.sift_matcher import match_sift
from baselines.loftr_matcher import match_loftr
from baselines.ncc_matcher import match_ncc
from matcher_cfog import match_images_cfog
from failure_analysis import analyze_failure_case


def run_ablation_study(
    source_img: str | Path,
    reference_img: str | Path,
    dem_img: Optional[str | Path] = None,
    output_dir: str | Path = "evaluation_output/ablation",
    gsd_m: float = 5.0,
) -> Dict[str, Any]:
    """Runs the 4-way ablation pipeline and generates formatted comparison reports."""
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"\n================================================================================")
    print(f" Chandrayaan-2 Registration Ablation Study & Baseline Benchmark")
    print(f" Source:    {source_img}")
    print(f" Reference: {reference_img}")
    print(f" DEM:       {dem_img}")
    print(f"================================================================================\n")

    results: List[Dict[str, Any]] = []

    # 1. Pure SIFT Baseline
    print("Running Baseline 1: Pure SIFT...")
    dem_arr = cv2.imread(str(dem_img), cv2.IMREAD_UNCHANGED) if dem_img and Path(dem_img).exists() else None
    res_sift = match_sift(source_img, reference_img, gsd_m=gsd_m, dem=dem_arr)
    results.append(res_sift)
    print(f"  -> SIFT: Inliers={res_sift['inlier_count']}, Ratio={res_sift['inlier_ratio']*100:.1f}%, RMSE={res_sift['fit_rmse_px']} px")

    # 2. Pure LoFTR Baseline
    print("Running Baseline 2: Pure LoFTR...")
    res_loftr = match_loftr(source_img, reference_img, gsd_m=gsd_m, dem=dem_arr)
    results.append(res_loftr)
    print(f"  -> LoFTR: Inliers={res_loftr['inlier_count']}, Ratio={res_loftr['inlier_ratio']*100:.1f}%, RMSE={res_loftr['fit_rmse_px']} px")

    # 3. Our Pipeline without DEM (Ablation: No DEM relief correction)
    print("Running Ablation 3: Our Pipeline without DEM...")
    t0 = time.time()
    res_no_dem = match_images_cfog(
        source_img,
        reference_img,
        dem_path=None,
        output_dir=out_base / "no_dem",
        source_sensor="OHRC",
        reference_sensor="TMC-2",
    )
    t_no_dem = time.time() - t0
    m_no_dem = res_no_dem.get("metrics") or {}
    results.append({
        "method": "Our Pipeline (No DEM)",
        "status": res_no_dem.get("status"),
        "match_count": m_no_dem.get("match_count", 0),
        "inlier_count": m_no_dem.get("inlier_count", 0),
        "inlier_ratio": m_no_dem.get("inlier_ratio", 0.0),
        "fit_rmse_px": m_no_dem.get("fit_rmse_px"),
        "absolute_rmse_m": m_no_dem.get("absolute_rmse_m"),
        "runtime_s": round(t_no_dem, 4),
    })
    print(f"  -> No DEM: Inliers={m_no_dem.get('inlier_count')}, Ratio={m_no_dem.get('inlier_ratio', 0)*100:.1f}%, RMSE={m_no_dem.get('fit_rmse_px')} px")

    # 4. Our Full Pipeline (Structural CFOG + GSD + Piecewise DEM Ray-Intersection + Grid NMS)
    print("Running Configuration 4: Our Full Pipeline (Structural + GSD + DEM + Grid NMS)...")
    t0_full = time.time()
    res_full = match_images_cfog(
        source_img,
        reference_img,
        dem_path=dem_img if dem_img and Path(dem_img).exists() else None,
        output_dir=out_base / "full_pipeline",
        source_sensor="OHRC",
        reference_sensor="TMC-2",
    )
    t_full = time.time() - t0_full
    m_full = res_full.get("metrics") or {}
    results.append({
        "method": "Our Full Pipeline (CFOG+DEM+Grid NMS)",
        "status": res_full.get("status"),
        "match_count": m_full.get("match_count", 0),
        "inlier_count": m_full.get("inlier_count", 0),
        "inlier_ratio": m_full.get("inlier_ratio", 0.0),
        "fit_rmse_px": m_full.get("fit_rmse_px"),
        "absolute_rmse_m": m_full.get("absolute_rmse_m"),
        "runtime_s": round(t_full, 4),
    })
    print(f"  -> Full Pipeline: Inliers={m_full.get('inlier_count')}, Ratio={m_full.get('inlier_ratio', 0)*100:.1f}%, RMSE={m_full.get('fit_rmse_px')} px")

    # Automated Failure Analysis on any failed method
    failures = []
    for r in results:
        if r.get("status") != "success" or r.get("inlier_count", 0) < 4:
            diag = analyze_failure_case(
                source_img,
                reference_img,
                failure_reason=r.get("status", "insufficient_inliers"),
                rmse_px=r.get("fit_rmse_px"),
                out_dir=out_base / f"failure_{r['method'].replace(' ', '_').lower()}",
            )
            r["failure_analysis"] = diag
            failures.append({"method": r["method"], "primary_cause": diag["primary_root_cause"]})

    # Generate Markdown and HTML Comparison Tables
    md_table = _format_markdown_table(results)
    html_table = _format_html_table(results)

    # Save outputs
    with open(out_base / "ablation_results.md", "w") as f:
        f.write(md_table)
    with open(out_base / "ablation_results.html", "w") as f:
        f.write(html_table)
    with open(out_base / "ablation_results.json", "w") as f:
        json.dump({"results": results, "failures": failures}, f, indent=4)

    print("\n" + md_table)
    print(f"\nArtifacts saved to: {out_base.resolve()}")

    return {"results": results, "markdown": md_table, "html": html_table, "failures": failures}


def _format_markdown_table(results: List[Dict[str, Any]]) -> str:
    lines = [
        "| Method | Inlier Count | Inlier Ratio | Pixel RMSE | Absolute RMSE (m) | Processing Time |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        m_name = r.get("method", "Unknown")
        inliers = str(r.get("inlier_count", 0))
        ratio = f"{r.get('inlier_ratio', 0.0) * 100:.1f}%"
        rmse_px = f"{r.get('fit_rmse_px'):.3f} px" if r.get("fit_rmse_px") is not None else "N/A"
        rmse_m = f"{r.get('absolute_rmse_m'):.2f} m" if r.get("absolute_rmse_m") is not None else "N/A"
        runtime = f"{r.get('runtime_s', 0.0):.2f} s"
        lines.append(f"| **{m_name}** | {inliers} | {ratio} | {rmse_px} | {rmse_m} | {runtime} |")
    return "\n".join(lines)


def _format_html_table(results: List[Dict[str, Any]]) -> str:
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Ablation Study Results</title>",
        "<style>table { border-collapse: collapse; width: 100%; font-family: sans-serif; }",
        "th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: center; }",
        "th { background-color: #2c3e50; color: white; }",
        "tr:nth-child(even) { background-color: #f2f2f2; }",
        "tr:hover { background-color: #e2f0d9; }</style></head><body>",
        "<h2>Chandrayaan-2 Cross-Sensor Registration: Ablation & Baseline Benchmark</h2>",
        "<table><thead><tr><th>Method</th><th>Inlier Count</th><th>Inlier Ratio</th><th>Pixel RMSE</th><th>Absolute RMSE (m)</th><th>Processing Time</th></tr></thead><tbody>",
    ]
    for r in results:
        m_name = r.get("method", "Unknown")
        inliers = r.get("inlier_count", 0)
        ratio = f"{r.get('inlier_ratio', 0.0) * 100:.1f}%"
        rmse_px = f"{r.get('fit_rmse_px'):.3f} px" if r.get("fit_rmse_px") is not None else "N/A"
        rmse_m = f"{r.get('absolute_rmse_m'):.2f} m" if r.get("absolute_rmse_m") is not None else "N/A"
        runtime = f"{r.get('runtime_s', 0.0):.2f} s"
        html.append(f"<tr><td><strong>{m_name}</strong></td><td>{inliers}</td><td>{ratio}</td><td>{rmse_px}</td><td>{rmse_m}</td><td>{runtime}</td></tr>")
    html.append("</tbody></table></body></html>")
    return "\n".join(html)


def main():
    parser = argparse.ArgumentParser(description="Run Chandrayaan-2 Registration Ablation Benchmark.")
    parser.add_argument(
        "--source",
        type=str,
        default="data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png",
        help="Source image path",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data_preprocessing_pipeline/processed_triplets/region_001/tmc_512.png",
        help="Reference image path",
    )
    parser.add_argument(
        "--dem",
        type=str,
        default="data_preprocessing_pipeline/processed_triplets/region_001/dem_512.png",
        help="DEM elevation path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_output/ablation",
        help="Output directory",
    )
    args = parser.parse_args()

    run_ablation_study(args.source, args.reference, dem_img=args.dem, output_dir=args.output)


if __name__ == "__main__":
    main()
