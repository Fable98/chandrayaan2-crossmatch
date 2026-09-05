"""Build auditable quantitative evidence tables from recorded evaluations.

The script never turns missing inputs into zeros. Optional experiments write to
the requested output directory and identify synthetic or proxy measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ML_model"))


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _metric(record: dict[str, Any], key: str) -> Any:
    return record.get(key)


def _summarize_registration(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not_available", "reason": f"Missing {path}"}
    report = _load(path)
    rows = []
    for region in report.get("regions", []):
        pair = (region.get("pairs") or {}).get("ohrc_tmc") or region
        rows.append({
            "dataset_id": region.get("region_id"),
            "status": pair.get("status"),
            "quality_tier": pair.get("quality_tier"),
            "inlier_count": pair.get("inlier_count"),
            "fit_rmse_px": pair.get("fit_rmse_px"),
            "spatial_coverage": pair.get("spatial_coverage"),
            "sun_angles_available": False,
        })
    return {
        "status": "recorded",
        "source": str(path),
        "dataset_count": len(rows),
        "rows": rows,
        "high_or_accepted": sum(
            row["quality_tier"] in {"HIGH_CONFIDENCE", "ACCEPTED"} for row in rows
        ),
    }


def _summarize_large_aoi(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not_available", "reason": f"Missing {path}"}
    report = _load(path)
    rows = []
    for dataset in report.get("datasets", []):
        for pair_name, pair in dataset.get("pairs", {}).items():
            result = pair.get("large_aoi") or {}
            metric_path = path.parent / dataset["dataset_id"] / pair_name / "metrics.json"
            metric = _load(metric_path) if metric_path.exists() else {}
            rows.append({
                "dataset_id": dataset.get("dataset_id"),
                "pair": pair_name,
                "status": result.get("status"),
                "inlier_count": result.get("inlier_count"),
                "fit_rmse_px": result.get("fit_rmse_px"),
                "quality_tier": result.get("quality_tier"),
                "trustworthy": bool(result.get("trustworthy")),
                "spatial_coverage": metric.get("spatial_coverage"),
                "coverage_relative_to_inlier_count": metric.get("coverage_relative_to_inlier_count"),
            })
    trustworthy = sum(row["trustworthy"] for row in rows)
    return {
        "status": "recorded",
        "source": str(path),
        "dataset_count": len(report.get("datasets", [])),
        "leg_count": len(rows),
        "trustworthy_legs": trustworthy,
        "trustworthy_rate": trustworthy / len(rows) if rows else None,
        "rows": rows,
    }


def _summarize_triplet_manifests(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.glob("*/manifest.json")) if root.exists() else []:
        manifest = _load(path)
        rows.append({
            "dataset_id": manifest.get("region_id", path.parent.name),
            "ohrc_sun_azimuth_deg": manifest.get("ohrc_sun_azimuth_deg"),
            "tmc2_sun_azimuth_deg": manifest.get("tmc2_sun_azimuth_deg"),
            "sun_azimuth_mismatch_deg": manifest.get("sun_azimuth_mismatch_deg"),
            "iirs_aoi_km": manifest.get("aoi_iirs_km"),
        })
    mismatches = [row["sun_azimuth_mismatch_deg"] for row in rows if row["sun_azimuth_mismatch_deg"] is not None]
    return {
        "status": "recorded" if rows else "not_available",
        "triplet_count": len(rows),
        "sun_azimuth_mismatch_range_deg": [min(mismatches), max(mismatches)] if mismatches else None,
        "rows": rows,
        "note": "Manifest diversity is descriptive; it is not a controlled cross-illumination experiment.",
    }


def _summarize_ablation(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "not_run",
            "reason": "Run evaluation/run_ablation.py with the required SIFT, LoFTR, and image inputs.",
        }
    report = _load(path)
    return {"status": "recorded", "source": str(path), **report}


def _summarize_lro(root: Path) -> dict[str, Any]:
    rows = []
    for path in root.rglob("*.json") if root.exists() else []:
        try:
            record = _load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or "absolute_rmse_m" not in record:
            continue
        text = json.dumps(record).lower()
        if "lro" in text or "basemap" in text:
            rows.append({"source": str(path), "absolute_rmse_m": record.get("absolute_rmse_m")})
    if not rows:
        return {
            "status": "not_available",
            "reason": "No JSON result containing absolute_rmse_m and LRO/basemap provenance was found.",
            "rows": [],
        }
    return {"status": "recorded", "rows": rows}


def _cube_ablation(cube_root: Path) -> dict[str, Any]:
    cubes = sorted(list(cube_root.rglob("*.npy")) + list(cube_root.rglob("*.npz"))) if cube_root.exists() else []
    if not cubes:
        return {
            "status": "not_available",
            "reason": "No IIRS hyperspectral .npy/.npz cubes were supplied; PNG projections cannot support this ablation.",
            "rows": [],
        }
    from spectral import enhance_iirs_structural_features

    rows = []
    for path in cubes:
        loaded = np.load(path)
        cube = loaded[loaded.files[0]] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
        if cube.ndim != 3:
            continue
        if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
            cube_hwb = np.transpose(cube, (1, 2, 0))
        else:
            cube_hwb = cube
        mean_map = np.mean(cube_hwb, axis=2).astype(np.float32)
        mean_map -= mean_map.min()
        mean_map /= max(float(mean_map.max()), 1e-8)
        enhanced = enhance_iirs_structural_features(cube_hwb)
        mean_edges = cv2.Sobel(mean_map, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(mean_map, cv2.CV_32F, 0, 1) ** 2
        enhanced_edges = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(enhanced, cv2.CV_32F, 0, 1) ** 2
        rows.append({
            "cube": str(path),
            "bands": int(cube_hwb.shape[2]),
            "mean_std": round(float(np.std(mean_map)), 6),
            "pca_sam_std": round(float(np.std(enhanced)), 6),
            "mean_gradient_energy": round(float(np.mean(mean_edges)), 6),
            "pca_sam_gradient_energy": round(float(np.mean(enhanced_edges)), 6),
            "registration_delta": None,
            "note": "Feature-map proxy; registration improvement requires paired image evaluation.",
        })
    return {"status": "recorded_proxy", "rows": rows}


def _illumination_stress(pair_dir: Path, output_dir: Path) -> dict[str, Any]:
    source = pair_dir / "ohrc_512.png"
    reference = pair_dir / "tmc_512.png"
    if not source.exists() or not reference.exists():
        return {"status": "not_available", "reason": "Expected ohrc_512.png and tmc_512.png."}
    output_dir.mkdir(parents=True, exist_ok=True)
    from matcher_cfog import match_images_cfog

    source_img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if source_img is None:
        return {"status": "not_available", "reason": "Could not decode source image."}
    variants = {
        "baseline": source_img,
        "darkened": np.clip(source_img.astype(np.float32) * 0.55, 0, 255).astype(np.uint8),
        "brightened": np.clip(source_img.astype(np.float32) * 1.35 + 18, 0, 255).astype(np.uint8),
        "contrast_reversed": 255 - source_img,
    }
    rows = []
    for name, image in variants.items():
        variant_path = output_dir / f"{name}_source.png"
        cv2.imwrite(str(variant_path), image)
        result = match_images_cfog(
            variant_path,
            reference,
            output_dir=output_dir / name,
            source_sensor="OHRC",
            reference_sensor="TMC-2",
        )
        metrics = result.get("metrics") or {}
        rows.append({
            "variant": name,
            "status": result.get("status"),
            "quality_tier": metrics.get("quality_tier"),
            "inlier_count": metrics.get("inlier_count"),
            "fit_rmse_px": metrics.get("fit_rmse_px"),
            "spatial_coverage": metrics.get("spatial_coverage"),
        })
    return {
        "status": "controlled_stress_test",
        "source_pair": str(pair_dir),
        "rows": rows,
        "note": "Synthetic photometric perturbations are not a substitute for independent sun-angle acquisitions.",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Quantitative Evidence Report",
        "",
        "This report is generated from recorded artifacts and explicitly marks unavailable experiments.",
        "",
        "## LRO absolute RMSE",
        "",
        "```json",
        json.dumps(report["lro_absolute_rmse_m"], indent=2),
        "```",
        "",
        "## Ablation",
        "",
        "```json",
        json.dumps(report["ablation"], indent=2),
        "```",
        "",
        "## Large-AOI",
        "",
        "| Dataset | Pair | Inliers | RMSE (px) | Coverage | Trustworthy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["large_aoi"].get("rows", []):
        lines.append(
            f"| {row['dataset_id']} | {row['pair']} | {row['inlier_count']} | "
            f"{row['fit_rmse_px']} | {row['spatial_coverage']} | {row['trustworthy']} |"
        )
    lines.extend([
        "",
        "## IIRS PCA/SAM versus mean collapse",
        "",
        "```json",
        json.dumps(report["iirs_pca_sam_vs_mean"], indent=2),
        "```",
        "",
        "## Expanded real-triplet inventory",
        "",
        "```json",
        json.dumps(report["expanded_real_triplets"], indent=2),
        "```",
        "",
        "## Cross-illumination",
        "",
        "```json",
        json.dumps(report["cross_illumination"], indent=2),
        "```",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration-summary", type=Path, default=ROOT / "benchmarks/registration_benchmark_output/registration_benchmark_summary.json")
    parser.add_argument("--large-aoi-summary", type=Path, default=ROOT / "benchmarks/large_iirs_benchmark_output/large_iirs_benchmark_summary.json")
    parser.add_argument("--triplets-root", type=Path, default=ROOT / "data_preprocessing_pipeline/processed_triplets")
    parser.add_argument("--lro-root", type=Path, default=ROOT / "benchmarks")
    parser.add_argument("--ablation-json", type=Path, default=ROOT / "evaluation_output/ablation/ablation_results.json")
    parser.add_argument("--iirs-cubes-root", type=Path, default=ROOT / "data")
    parser.add_argument("--illumination-pair", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_output/quantitative_evidence")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "registration_set": _summarize_registration(args.registration_summary),
        "large_aoi": _summarize_large_aoi(args.large_aoi_summary),
        "lro_absolute_rmse_m": _summarize_lro(args.lro_root),
        "ablation": _summarize_ablation(args.ablation_json),
        "iirs_pca_sam_vs_mean": _cube_ablation(args.iirs_cubes_root),
        "cross_illumination": (
            _illumination_stress(args.illumination_pair, args.output_dir / "illumination")
            if args.illumination_pair
            else {"status": "not_run", "reason": "Pass --illumination-pair to run the controlled stress test."}
        ),
        "expanded_real_triplets": {
            **_summarize_triplet_manifests(args.triplets_root),
            "additional_genuine_triplets_required": True,
            "note": "Supply a directory of additional processed CH2 triplets and rerun this command.",
        },
    }
    (args.output_dir / "quantitative_evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "quantitative_evidence.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "registration_regions": report["registration_set"].get("dataset_count"),
        "large_aoi_trustworthy_legs": report["large_aoi"].get("trustworthy_legs"),
        "lro_status": report["lro_absolute_rmse_m"]["status"],
        "ablation_status": report["ablation"]["status"],
        "iirs_status": report["iirs_pca_sam_vs_mean"]["status"],
        "illumination_status": report["cross_illumination"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()