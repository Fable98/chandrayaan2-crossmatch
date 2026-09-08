"""
Benchmark the expanded ~20 km IIRS crops against the existing 3.8 km results.

The scalar manifest GSD values are the mean of the exact x/y pixel sizes. The
axis-specific values remain in the manifest for auditability; match_images_cfog
currently accepts one scalar GSD per image.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import match_images_cfog


def _result_summary(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not result:
        return {"status": "not_run", "inlier_count": 0, "fit_rmse_px": None}
    metrics = result.get("metrics") or {}
    return {
        "status": result.get("status"),
        "message": result.get("message") if result.get("status") != "success" else None,
        "direction": result.get("direction", "native"),
        "measured_direction": result.get("measured_direction"),
        "inlier_count": metrics.get("inlier_count", result.get("inlier_count", 0)),
        "fit_rmse_px": metrics.get("fit_rmse_px"),
        "match_count": metrics.get("match_count", result.get("match_count", 0)),
        "quality_tier": metrics.get("quality_tier"),
    }


def _is_trustworthy(result: Dict[str, Any]) -> bool:
    return (
        result.get("status") == "success"
        and result.get("inlier_count", 0) >= 4
        and result.get("fit_rmse_px") is not None
        and result["fit_rmse_px"] <= 5.0
    )


def _trusted_dataset_count(report: Dict[str, Any], result_key: str) -> int:
    return sum(
        1
        for dataset in report["datasets"]
        if any(
            _is_trustworthy(values.get(result_key, {}))
            for values in dataset["pairs"].values()
        )
    )


def _existing_pair(eval_dir: Path, dataset: str, pair: str) -> Dict[str, Any]:
    path = eval_dir / f"{dataset}_metrics.json"
    if not path.exists():
        return {"status": "not_recorded", "inlier_count": 0, "fit_rmse_px": None}
    with path.open(encoding="utf-8") as stream:
        record = json.load(stream)
    pair_record = (record.get("pairs") or {}).get(pair) or {}
    forward_key = "forward_ohrc_to_iirs" if pair == "ohrc_iirs" else "forward_tmc_to_iirs"
    forward = pair_record.get(forward_key) or pair_record
    return {
        "status": forward.get("status"),
        "inlier_count": forward.get("inlier_count", 0),
        "fit_rmse_px": forward.get("fit_rmse_px"),
        "quality_tier": forward.get("quality_tier"),
    }


def run_large_benchmark(
    data_dir: Path,
    output_dir: Path,
    eval_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and (d.name.startswith("region_") or d.name.startswith("triplet_"))
    )
    report: Dict[str, Any] = {
        "benchmark_name": "Large-AOI OHRC/TMC-2 <-> IIRS registration",
        "large_aoi_acceptance": "status=success, inlier_count>=4, fit_rmse_px<=5.0",
        "datasets": [],
    }

    for dataset_dir in dataset_dirs:
        manifest_path = dataset_dir / "manifest.json"
        required = [
            dataset_dir / "ohrc_large_512.png",
            dataset_dir / "tmc_large_512.png",
            dataset_dir / "iirs_large_512.png",
        ]
        if not manifest_path.exists() or not all(path.exists() for path in required):
            continue

        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        gsd_ohrc = float(manifest["ohrc_large_effective_gsd_m"])
        gsd_tmc = float(manifest["tmc2_large_effective_gsd_m"])
        gsd_iirs = float(manifest["iirs_large_effective_gsd_m"])
        dataset_output = output_dir / dataset_dir.name
        dataset_output.mkdir(parents=True, exist_ok=True)

        pairs = {
            "ohrc_iirs": {
                "source": dataset_dir / "ohrc_large_512.png",
                "reference": dataset_dir / "iirs_large_512.png",
                "source_sensor": "OHRC",
                "reference_sensor": "IIRS",
                "gsd1": gsd_ohrc,
                "gsd2": gsd_iirs,
            },
            "tmc_iirs": {
                "source": dataset_dir / "tmc_large_512.png",
                "reference": dataset_dir / "iirs_large_512.png",
                "source_sensor": "TMC-2",
                "reference_sensor": "IIRS",
                "gsd1": gsd_tmc,
                "gsd2": gsd_iirs,
            },
        }
        dataset_record: Dict[str, Any] = {
            "dataset_id": dataset_dir.name,
            "effective_gsd_m": {
                "ohrc": gsd_ohrc,
                "tmc2": gsd_tmc,
                "iirs": gsd_iirs,
                "axis_specific": {
                    "ohrc": manifest.get("ohrc_large_effective_gsd_xy_m"),
                    "tmc2": manifest.get("tmc2_large_effective_gsd_xy_m"),
                    "iirs": manifest.get("iirs_large_effective_gsd_xy_m"),
                },
            },
            "pairs": {},
        }

        for pair_name, config in pairs.items():
            started = time.time()
            result = match_images_cfog(
                config["source"],
                config["reference"],
                output_dir=dataset_output / pair_name,
                source_sensor=config["source_sensor"],
                reference_sensor=config["reference_sensor"],
                explicit_gsd1=config["gsd1"],
                explicit_gsd2=config["gsd2"],
            )
            summary = _result_summary(result)
            summary["trustworthy"] = _is_trustworthy(summary)
            summary["elapsed_seconds"] = round(time.time() - started, 2)
            dataset_record["pairs"][pair_name] = {
                "existing_3_8km": _existing_pair(eval_dir, dataset_dir.name, pair_name),
                "large_aoi": summary,
            }
        report["datasets"].append(dataset_record)

    report["dataset_count"] = len(report["datasets"])
    report["trustworthy_counts"] = {
        pair: sum(
            1 for dataset in report["datasets"]
            if dataset["pairs"].get(pair, {}).get("large_aoi", {}).get("trustworthy")
        )
        for pair in ("ohrc_iirs", "tmc_iirs")
    }
    report["trustworthy_total_legs"] = sum(report["trustworthy_counts"].values())
    report["trustworthy_dataset_counts"] = {
        "existing_3_8km": _trusted_dataset_count(report, "existing_3_8km"),
        "large_aoi": _trusted_dataset_count(report, "large_aoi"),
    }
    report["trustworthy_dataset_delta"] = (
        report["trustworthy_dataset_counts"]["large_aoi"]
        - report["trustworthy_dataset_counts"]["existing_3_8km"]
    )
    report_path = output_dir / "large_iirs_benchmark_summary.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print("dataset | pair | existing 3.8km (inliers, rmse) | large AOI (inliers, rmse, status)")
    for dataset in report["datasets"]:
        for pair, values in dataset["pairs"].items():
            old = values["existing_3_8km"]
            new = values["large_aoi"]
            print(
                f"{dataset['dataset_id']} | {pair} | "
                f"{old.get('inlier_count', 0)}, {old.get('fit_rmse_px')} | "
                f"{new.get('inlier_count', 0)}, {new.get('fit_rmse_px')}, {new.get('status')}"
            )
    print(f"Trustworthy large-AOI legs: {report['trustworthy_total_legs']}/16")
    print(
        "Datasets with a trustworthy IIRS leg: "
        f"{report['trustworthy_dataset_counts']['existing_3_8km']} existing 3.8km -> "
        f"{report['trustworthy_dataset_counts']['large_aoi']} large-AOI "
        f"(delta {report['trustworthy_dataset_delta']:+d})"
    )
    print(f"Report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark large-AOI IIRS crops.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_preprocessing_pipeline/processed_triplets"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/large_iirs_benchmark_output"))
    parser.add_argument("--eval-dir", type=Path, default=Path("evaluation_output"))
    args = parser.parse_args()
    run_large_benchmark(args.data_dir, args.output_dir, args.eval_dir)


if __name__ == "__main__":
    main()
