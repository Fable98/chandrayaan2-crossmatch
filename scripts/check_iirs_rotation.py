"""
check_iirs_rotation.py — Diagnostic: are IIRS footprints axis-aligned or rotated?

For each triplet in user_triplets.json, extracts the IIRS 4-corner footprint
and computes:
  1. The axis-aligned bounding box area (min/max of lat/lon)
  2. The actual polygon area (shoelace formula)
  3. The relative difference between the two

Verdict per triplet:
  - AXIS-ALIGNED: bbox area ≈ polygon area (< 1e-6 relative difference)
  - ROTATED: polygon area meaningfully smaller than bbox area
"""

import json
import os
import sys

MANIFEST = os.path.join(
    os.path.dirname(__file__), "..", "processed_user", "user_triplets.json"
)


def shoelace_area(corners: list[dict]) -> float:
    """
    Compute the area of a polygon using the shoelace formula.
    corners: list of {lat, lon} dicts in order (any consistent winding).
    Returns absolute area in (degrees²).
    """
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        # Using (lon, lat) as (x, y) for the 2D area calculation
        area += corners[i]["lon"] * corners[j]["lat"]
        area -= corners[j]["lon"] * corners[i]["lat"]
    return abs(area) / 2.0


def bbox_area(corners: list[dict]) -> float:
    """Compute area of the axis-aligned bounding box of the corners."""
    lats = [c["lat"] for c in corners]
    lons = [c["lon"] for c in corners]
    return (max(lats) - min(lats)) * (max(lons) - min(lons))


def check_axis_alignment(corners: list[dict]) -> dict:
    """
    Check whether 4 corners form an axis-aligned rectangle.
    Returns diagnostic info.
    """
    lats = sorted(set(round(c["lat"], 10) for c in corners))
    lons = sorted(set(round(c["lon"], 10) for c in corners))

    poly_area = shoelace_area(corners)
    bb_area = bbox_area(corners)

    if bb_area == 0:
        rel_diff = 0.0
    else:
        rel_diff = abs(bb_area - poly_area) / bb_area

    # Also check: does each corner's lat appear exactly twice and
    # each lon appear exactly twice? (true for axis-aligned rect)
    lat_counts = {}
    lon_counts = {}
    for c in corners:
        lat_r = round(c["lat"], 10)
        lon_r = round(c["lon"], 10)
        lat_counts[lat_r] = lat_counts.get(lat_r, 0) + 1
        lon_counts[lon_r] = lon_counts.get(lon_r, 0) + 1

    is_lat_paired = all(v == 2 for v in lat_counts.values()) and len(lat_counts) == 2
    is_lon_paired = all(v == 2 for v in lon_counts.values()) and len(lon_counts) == 2

    return {
        "distinct_lats": lats,
        "distinct_lons": lons,
        "lat_paired": is_lat_paired,
        "lon_paired": is_lon_paired,
        "polygon_area": poly_area,
        "bbox_area": bb_area,
        "relative_diff": rel_diff,
        "is_axis_aligned": is_lat_paired and is_lon_paired,
    }


def main():
    with open(MANIFEST, "r") as f:
        manifest = json.load(f)

    triplets = manifest.get("triplets", [])
    any_rotated = False

    for triplet in triplets:
        tid = triplet["id"]
        iirs_entry = None
        for s in triplet["sensors"]:
            if s["sensor"] == "iirs":
                iirs_entry = s
                break

        if iirs_entry is None:
            print(f"\n{'='*60}")
            print(f"Triplet: {tid}")
            print(f"  NO IIRS SENSOR FOUND — skipping")
            continue

        fp = iirs_entry["footprint"]
        corners = [fp["top_left"], fp["top_right"], fp["bottom_right"], fp["bottom_left"]]

        result = check_axis_alignment(corners)

        print(f"\n{'='*60}")
        print(f"Triplet: {tid}")
        print(f"  Raw corners:")
        for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
            c = fp[name]
            print(f"    {name:>14}: lat={c['lat']:>10.4f}  lon={c['lon']:>10.4f}")

        print(f"  Distinct lats: {result['distinct_lats']}")
        print(f"  Distinct lons: {result['distinct_lons']}")
        print(f"  Lat values paired (2 each)? {result['lat_paired']}")
        print(f"  Lon values paired (2 each)? {result['lon_paired']}")
        print(f"  Polygon area:  {result['polygon_area']:.10f} deg²")
        print(f"  BBox area:     {result['bbox_area']:.10f} deg²")
        print(f"  Relative diff: {result['relative_diff']:.10f}")

        if result["is_axis_aligned"]:
            verdict = "AXIS-ALIGNED ✓ (bbox derivation is lossless)"
        else:
            verdict = "ROTATED ✗ (bbox derivation LOSES information)"
            any_rotated = True

        print(f"  VERDICT: {verdict}")

    print(f"\n{'='*60}")
    if any_rotated:
        print("OVERALL: At least one triplet has a ROTATED IIRS footprint.")
        print("         The bbox-from-corners derivation is LOSSY and must be")
        print("         replaced with full 4-corner output.")
    else:
        print("OVERALL: All IIRS footprints are AXIS-ALIGNED.")
        print("         The bbox-from-corners derivation is LOSSLESS and correct.")
    print(f"{'='*60}")

    return 1 if any_rotated else 0


if __name__ == "__main__":
    sys.exit(main())
