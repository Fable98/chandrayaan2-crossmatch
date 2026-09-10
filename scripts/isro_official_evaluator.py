"""
isro_official_evaluator.py — Official ISRO Evaluation Wrapper
for Chandrayaan-2 Cross-Sensor Registration Pipeline.

Iterates a hidden test directory of `source_<id>.* / reference_<id>.* [/ dem_<id>.*]`
pairs, runs the core CFOG engine per pair, aggregates metrics, and emits:
  - isro_evaluation_summary.json
  - isro_evaluation_summary.md
  - pairwise_results.json

Usage:
    python scripts/isro_official_evaluator.py --input_dir <test_dir> --output_dir <report_dir> [--use_dem True|False]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("isro_official_evaluator")

# Make ML_model importable (matcher_cfog uses intra-package `from metadata import ...`,
# so ML_model itself must be on sys.path — same convention as demo_registration.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ML_MODEL_DIR = _PROJECT_ROOT / "ML_model"
for _p in (str(_ML_MODEL_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from matcher_cfog import match_images_cfog
except Exception as e:  # deferred hard failure with clear message
    logger.error("Failed to import core engine matcher_cfog: %s", e)
    raise

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ---------------------------------------------------------------------------
# STEP 1: Directory parsing — group source_<id> / reference_<id> / dem_<id>
# ---------------------------------------------------------------------------

def _strip_prefix_and_ext(filename: str, prefix: str) -> Optional[str]:
    """Return the `<id>` part of `prefix_<id>.<ext>`, or None if no match."""
    lower = filename.lower()
    pre = prefix.lower() + "_"
    if not lower.startswith(pre):
        return None
    stem = Path(filename).stem  # strips extension
    pair_id = stem[len(pre):]
    if not pair_id:
        return None
    return pair_id


def find_pairs(input_dir: Path, use_dem: bool = True) -> List[Dict[str, Optional[Path]]]:
    """Scan input_dir and group matching source/reference(/dem) triplets.

    Naming convention: source_<id>.*, reference_<id>.*, dem_<id>.*
    Returns a sorted list of {"id", "source", "reference", "dem"} dicts.
    Pairs missing either source or reference are skipped with a warning.
    """
    sources: Dict[str, Path] = {}
    references: Dict[str, Path] = {}
    dems: Dict[str, Path] = {}

    for entry in sorted(input_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in IMAGE_EXTS:
            continue
        sid = _strip_prefix_and_ext(entry.name, "source")
        if sid is not None:
            if sid not in sources:
                sources[sid] = entry
            else:
                logger.warning("Duplicate source id '%s' — keeping %s, ignoring %s", sid, sources[sid], entry)
            continue
        rid = _strip_prefix_and_ext(entry.name, "reference")
        if rid is not None:
            if rid not in references:
                references[rid] = entry
            else:
                logger.warning("Duplicate reference id '%s' — keeping %s, ignoring %s", rid, references[rid], entry)
            continue
        did = _strip_prefix_and_ext(entry.name, "dem")
        if did is not None:
            if did not in dems:
                dems[did] = entry
            else:
                logger.warning("Duplicate dem id '%s' — keeping %s, ignoring %s", did, dems[did], entry)
            continue

    pairs: List[Dict[str, Optional[Path]]] = []
    all_ids = sorted(set(sources) | set(references))
    for pid in all_ids:
        src = sources.get(pid)
        ref = references.get(pid)
        if src is None or ref is None:
            logger.warning(
                "Skipping id '%s': missing %s.",
                pid,
                "source" if src is None else "reference",
            )
            continue
        dem = dems.get(pid) if use_dem else None
        if use_dem and dem is None and pid not in dems:
            logger.warning("DEM missing for pair id '%s' — falling back to dem_path=None.", pid)
        pairs.append({"id": pid, "source": src, "reference": ref, "dem": dem})
    return pairs


# ---------------------------------------------------------------------------
# Helpers: JSON-safe conversion + metric extraction
# ---------------------------------------------------------------------------

def to_json_safe(obj):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _uniformity_of(metrics: Dict) -> Optional[float]:
    for key in ("uniformity_score", "spatial_uniformity"):
        v = _safe_float(metrics.get(key))
        if v is not None:
            return v
    # Fall back to nested spatial_distribution.uniformity_score
    dist = metrics.get("spatial_distribution")
    if isinstance(dist, dict):
        v = _safe_float(dist.get("uniformity_score"))
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# STEP 2 + 3: Iterate engine + aggregate
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    vals: List[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """Calculates bootstrap confidence interval over non-null metric values."""
    if len(vals) < 3:
        return None
    rng = np.random.RandomState(seed)
    arr = np.asarray(vals, dtype=np.float64)
    boot_means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_resamples)]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return {
        "mean": round(float(np.mean(arr)), 4),
        "ci_lower": round(low, 4),
        "ci_upper": round(high, 4),
        "confidence_level": ci,
        "sample_size": len(vals),
    }


def evaluate_all(
    pairs: List[Dict[str, Optional[Path]]],
    output_dir: Path,
    use_dem: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Run match_images_cfog per pair; return (summary, pairwise_results)."""
    pairwise: List[Dict] = []
    fit_rmses: List[float] = []
    held_out_rmses: List[float] = []
    abs_rmses: List[float] = []
    inlier_counts: List[float] = []
    uniformities: List[float] = []

    successful = 0
    failed = 0

    total = len(pairs)
    per_pair_root = output_dir / "per_pair"
    per_pair_root.mkdir(parents=True, exist_ok=True)

    for idx, pair in enumerate(pairs, start=1):
        pid = pair["id"]
        src = pair["source"]
        ref = pair["reference"]
        dem: Optional[Path] = pair.get("dem") if use_dem else None
        if use_dem and dem is not None and not dem.exists():
            logger.warning("Pair %s: DEM %s not found — falling back to dem_path=None.", pid, dem)
            dem = None

        logger.info("Processing pair %d of %d (id=%s)...", idx, total, pid)
        pair_out = per_pair_root / f"pair_{pid}"
        pair_out.mkdir(parents=True, exist_ok=True)

        status = "failed"
        message = ""
        metrics: Optional[Dict] = None
        try:
            result = match_images_cfog(
                str(src),
                str(ref),
                dem_path=str(dem) if dem is not None else None,
                output_dir=str(pair_out),
            )
            status = str(result.get("status", "failed"))
            message = str(result.get("message", ""))
            metrics = result.get("metrics")
        except Exception as exc:  # Null safety: never crash the whole run
            status = "exception"
            message = f"{type(exc).__name__}: {exc}"
            logger.warning("Pair id=%s raised exception: %s\n%s", pid, exc, traceback.format_exc())
            metrics = None

        fit_insample = _safe_float(metrics.get("fit_rmse_insample_px") or metrics.get("fit_rmse_px")) if (status == "success" and isinstance(metrics, dict)) else None
        heldout = _safe_float(metrics.get("held_out_validation_rmse_px") or metrics.get("held_out_rmse_px") or metrics.get("validation_rmse_px")) if (status == "success" and isinstance(metrics, dict)) else None

        record = to_json_safe({
            "id": pid,
            "source": str(src),
            "reference": str(ref),
            "dem": str(dem) if dem is not None else None,
            "status": status,
            "message": message,
            "fit_insample": fit_insample,
            "heldout": heldout,
            "metrics": metrics,
        })
        pairwise.append(record)

        if status == "success" and isinstance(metrics, dict):
            successful += 1
            if fit_insample is not None:
                fit_rmses.append(fit_insample)
            if heldout is not None:
                held_out_rmses.append(heldout)
            abs_rmse = _safe_float(metrics.get("absolute_rmse_m"))
            if abs_rmse is not None:
                abs_rmses.append(abs_rmse)
            inl = _safe_float(metrics.get("inlier_count"))
            if inl is not None:
                inlier_counts.append(inl)
            uni = _uniformity_of(metrics)
            if uni is not None:
                uniformities.append(uni)
        else:
            failed += 1
            logger.warning("Pair id=%s failed (status=%s): %s", pid, status, message)

    def _mean(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    success_rate = round(float(successful / max(1, total)), 4) if total > 0 else 0.0

    summary = to_json_safe({
        "total_pairs_processed": total,
        "successful_registrations": successful,
        "failed_registrations": failed,
        "success_rate": success_rate,
        "average_fit_rmse_insample_px": _mean(fit_rmses),
        "average_fit_rmse_px": _mean(fit_rmses),
        "average_held_out_rmse_px": _mean(held_out_rmses),
        "average_absolute_rmse_m": _mean(abs_rmses),
        "average_inlier_count": _mean(inlier_counts),
        "average_spatial_uniformity": _mean(uniformities),
        "bootstrap_ci_95": {
            "fit_rmse_insample_px": _bootstrap_ci(fit_rmses),
            "held_out_rmse_px": _bootstrap_ci(held_out_rmses),
            "absolute_rmse_m": _bootstrap_ci(abs_rmses),
        },
    })
    return summary, pairwise


# ---------------------------------------------------------------------------
# STEP 4: Deliverables — JSON + Markdown + pairwise manifest
# ---------------------------------------------------------------------------

def write_markdown_report(summary: Dict, output_dir: Path) -> Path:
    md_path = output_dir / "isro_evaluation_summary.md"

    def _fmt(v, ndigits: int = 4) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, float):
            return f"{v:.{ndigits}f}"
        return str(v)

    def _fmt_ci(ci_dict: Optional[Dict]) -> str:
        if not ci_dict:
            return "N/A"
        return f"{ci_dict.get('mean', 'N/A')} [{ci_dict.get('ci_lower', 'N/A')}, {ci_dict.get('ci_upper', 'N/A')}]"

    b_ci = summary.get("bootstrap_ci_95", {}) or {}

    lines = [
        "# ISRO Evaluation Summary — Chandrayaan-2 Cross-Sensor Registration",
        "",
        "*Generated by Chandrayaan-2 Cross-Sensor Registration Pipeline. Absolute RMSE is DEM-corrected.*",
        "",
        "## Aggregated Metrics",
        "",
        "| Metric | Value | 95% Bootstrap CI |",
        "|---|---|---|",
        f"| Total pairs processed | {summary.get('total_pairs_processed', 0)} | — |",
        f"| Successful registrations | {summary.get('successful_registrations', 0)} | — |",
        f"| Failed registrations | {summary.get('failed_registrations', 0)} | — |",
        f"| Success rate | {_fmt(summary.get('success_rate', 0.0) * 100, 2)}% | — |",
        f"| Average fit RMSE (in-sample, px) | {_fmt(summary.get('average_fit_rmse_insample_px') or summary.get('average_fit_rmse_px'))} | {_fmt_ci(b_ci.get('fit_rmse_insample_px'))} |",
        f"| Average held-out RMSE (out-of-sample, px) | {_fmt(summary.get('average_held_out_rmse_px'))} | {_fmt_ci(b_ci.get('held_out_rmse_px'))} |",
        f"| Average absolute RMSE (m, DEM-corrected) | {_fmt(summary.get('average_absolute_rmse_m'))} | {_fmt_ci(b_ci.get('absolute_rmse_m'))} |",
        f"| Average inlier count | {_fmt(summary.get('average_inlier_count'), 2)} | — |",
        f"| Average spatial uniformity | {_fmt(summary.get('average_spatial_uniformity'))} | — |",
        "",
        "## Integrity & Survivorship Notes",
        "",
        "- Failed pairs are explicitly recorded with `fit_insample = null` and `heldout = null` (never excluded to inflate averages).",
        "- `success_rate` tracks the proportion of pairs meeting quality gates ($N \\ge 10$).",
        "- `average_absolute_rmse_m` is averaged over successful pairs where DEM-corrected absolute RMSE is available.",
        "- Per-pair raw metrics are in `pairwise_results.json`; per-pair engine artefacts are under `per_pair/`.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True/False, got '{value}'.")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official ISRO evaluator for the Chandrayaan-2 cross-sensor registration pipeline."
    )
    parser.add_argument("--input_dir", required=True, type=str,
                        help="Path to folder containing source_<id>.* / reference_<id>.* [/ dem_<id>.*] test images.")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Path where evaluation reports will be saved.")
    parser.add_argument("--use_dem", default=True, type=str2bool, nargs="?",
                        const=True, metavar="True|False",
                        help="Look for dem_<id>.* files (default: True). Missing DEMs fall back to dem_path=None.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    use_dem = bool(args.use_dem)

    if not input_dir.is_dir():
        logger.error("Input directory does not exist or is not a directory: %s", input_dir)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(input_dir, use_dem=use_dem)
    logger.info("Found %d evaluable pair(s) in %s (use_dem=%s).", len(pairs), input_dir, use_dem)
    if not pairs:
        logger.warning("No source_<id>/reference_<id> pairs found — writing empty report.")

    summary, pairwise = evaluate_all(pairs, output_dir, use_dem=use_dem)

    json_path = output_dir / "isro_evaluation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest_path = output_dir / "pairwise_results.json"
    manifest_path.write_text(json.dumps(pairwise, indent=2), encoding="utf-8")

    md_path = write_markdown_report(summary, output_dir)

    logger.info("Wrote summary JSON: %s", json_path)
    logger.info("Wrote pairwise manifest: %s", manifest_path)
    logger.info("Wrote Markdown summary: %s", md_path)
    logger.info("Summary: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
