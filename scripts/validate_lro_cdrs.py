#!/usr/bin/env python3
"""
scripts/validate_lro_cdrs.py — Step 14: LRO CDR registry validator.

Counts ONLY real downloaded CDRs toward the 15-CDR benchmark target.
A registry entry claiming real_downloaded_cdr must have, on disk:
  * manifest.json with reference_provenance == "real_downloaded_cdr",
  * the 512px reference PNG, the OHRC source PNG, and the detached .lbl.
Anything else fails LOUDLY (non-zero exit). Pending slots are reported,
never counted, and carry no numbers by construction.

Usage:
    python scripts/validate_lro_cdrs.py [--registry PATH] [--require N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "data_preprocessing_pipeline" / "lro_cdr_registry.json"


def validate_registry(registry_path: Path) -> dict:
    """Validate every real entry on disk. Returns a report dict."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    base = registry_path.parent
    real = registry.get("real_cdrs", [])
    pending = registry.get("pending_acquisition", [])
    target = int(registry.get("target_cdrs", 15))

    validated: list[dict] = []
    failures: list[str] = []
    for entry in real:
        slot = entry.get("slot")
        region = entry.get("region_id")
        manifest_rel = entry.get("manifest")
        problems: list[str] = []
        if entry.get("status") != "real_downloaded_cdr":
            problems.append(f"slot {slot}: status is not real_downloaded_cdr")
        if not entry.get("lro_nac_product_id"):
            problems.append(f"slot {slot}: missing LRO product ID")
        manifest_path = base / str(manifest_rel or "")
        manifest: dict = {}
        if not manifest_path.is_file():
            problems.append(f"slot {slot}: manifest missing at {manifest_rel}")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                problems.append(f"slot {slot}: manifest unreadable ({exc})")
            if manifest.get("reference_provenance") != "real_downloaded_cdr":
                problems.append(
                    f"slot {slot}: manifest provenance is "
                    f"{manifest.get('reference_provenance')!r}, not real_downloaded_cdr"
                )
        region_dir = manifest_path.parent if manifest_path else None
        for asset in (
            manifest.get("lro_nac_reference_image"),
            manifest.get("ohrc_source_image"),
            manifest.get("lro_nac_label"),
        ):
            if not asset:
                problems.append(f"slot {slot}: manifest missing asset path ({asset})")
                continue
            asset_path = REPO_ROOT / str(asset).replace("\\", "/")
            if not asset_path.is_file():
                # Fall back to region-dir-relative resolution.
                alt = (region_dir / Path(str(asset)).name) if region_dir else None
                if alt is None or not alt.is_file():
                    problems.append(f"slot {slot}: asset missing on disk: {asset}")
        # Cross-check the cited numbers against the manifest result block.
        cited = entry.get("result", {}) or {}
        actual = manifest.get("result", {}) if isinstance(manifest, dict) else {}
        for key in ("fit_rmse_px", "inlier_count", "match_count"):
            if key in cited and cited.get(key) != actual.get(key):
                problems.append(
                    f"slot {slot}: cited {key}={cited.get(key)} != manifest {actual.get(key)}"
                )
        if problems:
            failures.extend(problems)
        else:
            validated.append({"slot": slot, "region_id": region, **cited})

    # Pending slots must stay number-free tickets (schema guard).
    for entry in pending:
        if entry.get("lro_nac_product_id") is not None or "result" in entry:
            failures.append(
                f"pending slot {entry.get('slot')}: carries product data — "
                "pending slots must stay number-free"
            )

    return {
        "registry": str(registry_path),
        "target_cdrs": target,
        "n_real_validated": len(validated),
        "n_pending": len(pending),
        "validated": validated,
        "failures": failures,
        "benchmark_ready": len(validated) >= target,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the LRO CDR registry (real CDRs only).")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--require", type=int, default=0,
                        help="Fail unless at least N real CDRs validate.")
    args = parser.parse_args(argv)

    report = validate_registry(Path(args.registry))
    print(json.dumps(report, indent=2))
    if report["failures"]:
        print(f"VALIDATION FAILED: {len(report['failures'])} problem(s).", file=sys.stderr)
        return 2
    if report["n_real_validated"] < args.require:
        print(
            f"INSUFFICIENT REAL CDRS: {report['n_real_validated']}/{report['target_cdrs']} "
            f"validated (required {args.require}). Pending acquisition is not data.",
            file=sys.stderr,
        )
        return 2
    print(f"OK: {report['n_real_validated']}/{report['target_cdrs']} real CDRs validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
