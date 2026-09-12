"""
scripts/generate_ground_truth_matches.py — Generate Objective Ground-Truth Match Dataset.

Photogrammetrically sound, non-circular ground-truth generator for training the
AIMatchVerifier RandomForest model.

Domain gap this script exists to close:
  Same-sensor NCC warps give true matches high correlation. Real OHRC↔TMC
  pairs under ~160° sun-azimuth mismatch flip shadow polarity, so NCC/MI
  confidence barely separates true from false (~0.37 vs ~0.36). A model
  trained only on the NCC domain therefore leans on refinement_dx/dy and
  spatial_quality_score — fields that MUST be computed with the same live
  matcher functions, never defaulted.

Methodology:
1. Ingest genuine Chandrayaan-2 lunar crops across regions 001-006.
2. Apply known projective H_gt (attitude drift / scale / translation).
3. Two photometric domains:
     * same_sensor  — mild gain/gamma/noise, NCC (legacy path)
     * cross_sensor — ~160° antisolar gradient polarity + TMC-like blur,
                      scored with the production MI+NCC unified matcher
4. Sub-pixel Fourier phase correlation + live compute_spatial_quality_score.
5. Euclidean label vs H_gt:
     err <= 2.0 px -> human_label True
     err >= 5.0 px -> human_label False
     2–5 px omitted
6. Writes ML_model/ground_truth_matches.json (label_source=hand).
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from config import SEED
from matcher_cfog import (
    build_cfog_descriptor_gallery,
    compute_descriptor_match_features,
    compute_phase_congruency,
    compute_spatial_quality_score,
    detect_salient_keypoints,
    find_best_correspondence_unified,
    last_peak_uniqueness,
    subpixel_phase_correlation,
    suppression_via_square_covering,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_gt")


def create_random_homography(
    w: int,
    h: int,
    rng: np.random.Generator,
    max_angle_deg: float = 8.0,
    scale_range: Tuple[float, float] = (0.92, 1.08),
    max_trans_px: float = 25.0,
    max_persp: float = 0.0003,
) -> np.ndarray:
    """Generate a realistic orbital homography matrix."""
    cx, cy = w / 2.0, h / 2.0
    angle_rad = math.radians(float(rng.uniform(-max_angle_deg, max_angle_deg)))
    scale = float(rng.uniform(scale_range[0], scale_range[1]))
    tx = float(rng.uniform(-max_trans_px, max_trans_px))
    ty = float(rng.uniform(-max_trans_px, max_trans_px))

    cos_a = math.cos(angle_rad) * scale
    sin_a = math.sin(angle_rad) * scale

    A = np.array([
        [cos_a, -sin_a, (1.0 - cos_a) * cx + sin_a * cy + tx],
        [sin_a,  cos_a, -sin_a * cx + (1.0 - cos_a) * cy + ty],
        [0.0,    0.0,   1.0],
    ], dtype=np.float64)
    A[2, 0] = float(rng.uniform(-max_persp, max_persp))
    A[2, 1] = float(rng.uniform(-max_persp, max_persp))
    return A


def apply_photometric_perturbation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild same-sensor photometric variation (gain, bias, contrast, noise)."""
    out = img.astype(np.float32)
    gamma = float(rng.uniform(0.75, 1.35))
    out = np.power(np.clip(out / 255.0, 0.0, 1.0), gamma) * 255.0
    gain = float(rng.uniform(0.85, 1.15))
    bias = float(rng.uniform(-15.0, 15.0))
    out = np.clip(out * gain + bias, 0.0, 255.0)
    noise = rng.normal(0.0, float(rng.uniform(1.0, 4.0)), out.shape).astype(np.float32)
    return np.clip(out + noise, 0.0, 255.0).astype(np.uint8)


def apply_antisolar_cross_sensor(img: np.ndarray, rng: np.random.Generator,
                                 azimuth_delta_deg: float = 160.0) -> np.ndarray:
    """Approximate OHRC↔TMC near-antisolar illumination + coarser GSD.

    Gradient polarity is reversed along a ~160° sun-azimuth offset so NCC
    collapses the way real shadow-reversed pairs do. Area downsample/upsample
    mimics TMC's ~20× coarser sampling without claiming a new sensor.
    """
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    az = math.radians(float(azimuth_delta_deg) + float(rng.uniform(-8.0, 8.0)))
    shade = gx * math.sin(az) + gy * math.cos(az)
    scale = float(np.percentile(np.abs(shade), 98)) + 1e-6
    shade = np.clip(shade / scale, -1.0, 1.0)
    albedo = np.clip(f / 255.0, 0.0, 1.0)
    # Invert albedo (shadow polarity flip) and mix directional shade.
    mixed = np.clip(0.50 * (1.0 - albedo) + 0.50 * (0.5 - 0.45 * shade), 0.0, 1.0)
    h, w = mixed.shape[:2]
    factor = int(rng.integers(6, 11))
    small = cv2.resize(
        mixed, (max(16, w // factor), max(16, h // factor)), interpolation=cv2.INTER_AREA
    )
    mixed = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    mixed = cv2.GaussianBlur(mixed, (0, 0), sigmaX=float(rng.uniform(1.2, 2.0)))
    noise = rng.normal(0.0, float(rng.uniform(0.01, 0.03)), mixed.shape).astype(np.float32)
    return np.clip((mixed + noise) * 255.0, 0.0, 255.0).astype(np.uint8)


def _label_from_error(true_err: float, true_px: float = 2.0, false_px: float = 5.0) -> bool | None:
    if true_err <= true_px:
        return True
    if true_err >= false_px:
        return False
    return None


def generate_matches_for_image(
    image_path: Path,
    rng: np.random.Generator,
    domains: Tuple[str, ...] = ("same_sensor", "cross_sensor"),
    scene_sink: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Process an image under known transforms in one or more photometric domains.

    If ``scene_sink`` is provided, each (pc1, pc2, gallery, domain) transform is
    appended so a backfill can measure descriptor features at stored coordinates
    without relabelling.
    """
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return []

    h, w = img_gray.shape[:2]
    records: List[Dict[str, Any]] = []

    domain_specs = {
        "same_sensor": {"multimodal": False, "n_transforms": 2, "jitter": 8.0, "search_rad": 20, "true_px": 2.0, "false_px": 5.0, "n_forced_false": 4},
        "cross_sensor": {"multimodal": True, "n_transforms": 4, "jitter": 6.0, "search_rad": 28, "true_px": 4.0, "false_px": 8.0, "n_forced_false": 6},
    }

    pc1 = compute_phase_congruency(img_gray, num_orientations=4, num_scales=3)
    kps_raw = detect_salient_keypoints(pc1, max_corners=300, quality_level=0.008)
    kps_ssc = suppression_via_square_covering(kps_raw, num_ret_points=60, tolerance=0.12, cols=w, rows=h)
    half_p = 8

    for domain in domains:
        spec = domain_specs[domain]
        multimodal = bool(spec["multimodal"])
        for _t_idx in range(int(spec["n_transforms"])):
            H_gt = create_random_homography(w, h, rng)
            warped = cv2.warpPerspective(
                img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            if domain == "cross_sensor":
                warped = apply_antisolar_cross_sensor(warped, rng)
            else:
                warped = apply_photometric_perturbation(warped, rng)

            pc2 = compute_phase_congruency(warped, num_orientations=4, num_scales=3)
            kps2_raw = detect_salient_keypoints(pc2, max_corners=200, quality_level=0.01)
            tgt_gallery = build_cfog_descriptor_gallery(pc2, kps2_raw)
            if scene_sink is not None:
                scene_sink.append({
                    "domain": domain, "pc1": pc1, "pc2": pc2, "gallery": tgt_gallery,
                })
            search_rad = int(spec["search_rad"])
            jitter = float(spec["jitter"])

            for item in kps_ssc:
                rec = _match_one_keypoint(
                    kx=float(item[0]), ky=float(item[1]),
                    pc1=pc1, pc2=pc2, tgt_gallery=tgt_gallery, H_gt=H_gt, w=w, h=h,
                    half_p=half_p, search_rad=search_rad, jitter=jitter,
                    multimodal=multimodal, rng=rng, domain=domain,
                    true_px=float(spec["true_px"]), false_px=float(spec["false_px"]),
                )
                if rec is not None:
                    records.append(rec)

            # Explicit random mismatches so the false class is not only
            # "search failed" — measured live features, known-wrong geometry.
            n_false = min(int(spec["n_forced_false"]), len(kps_ssc))
            for item in list(kps_ssc)[:n_false]:
                rec = _random_false_match(
                    kx=float(item[0]), ky=float(item[1]),
                    pc1=pc1, pc2=pc2, tgt_gallery=tgt_gallery, H_gt=H_gt, w=w, h=h,
                    half_p=half_p, multimodal=multimodal, rng=rng, domain=domain,
                    true_px=float(spec["true_px"]), false_px=float(spec["false_px"]),
                )
                if rec is not None:
                    records.append(rec)

    return records


def _match_one_keypoint(
    *,
    kx: float, ky: float, pc1, pc2, tgt_gallery, H_gt, w, h, half_p, search_rad, jitter,
    multimodal: bool, rng: np.random.Generator, domain: str,
    true_px: float, false_px: float,
) -> Dict[str, Any] | None:
    cx, cy = int(round(kx)), int(round(ky))
    if cy < half_p or cy >= h - half_p or cx < half_p or cx >= w - half_p:
        return None
    tmpl = pc1[cy - half_p : cy + half_p, cx - half_p : cx + half_p]
    if float(np.std(tmpl)) < 1e-4:
        return None

    p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
    gt_x, gt_y = float(p_gt[0]), float(p_gt[1])
    search_cx = int(round(gt_x + float(rng.uniform(-jitter, jitter))))
    search_cy = int(round(gt_y + float(rng.uniform(-jitter, jitter))))
    s_min_x, s_max_x = max(0, search_cx - search_rad), min(w, search_cx + search_rad)
    s_min_y, s_max_y = max(0, search_cy - search_rad), min(h, search_cy + search_rad)
    if s_max_x - s_min_x <= tmpl.shape[1] or s_max_y - s_min_y <= tmpl.shape[0]:
        return None
    search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
    if float(np.std(search_region)) < 1e-4:
        return None

    score, loc = find_best_correspondence_unified(search_region, tmpl, multimodal_pair=multimodal)
    peak_uniq = last_peak_uniqueness()
    found_x = float(s_min_x + loc[0] + half_p)
    found_y = float(s_min_y + loc[1] + half_p)
    return _finalize_record(
        kx, ky, found_x, found_y, tmpl, pc1, pc2, tgt_gallery, w, h, half_p,
        gt_x, gt_y, score, peak_uniq, domain, multimodal,
        true_px=true_px, false_px=false_px,
    )


def _random_false_match(
    *,
    kx: float, ky: float, pc1, pc2, tgt_gallery, H_gt, w, h, half_p,
    multimodal: bool, rng: np.random.Generator, domain: str,
    true_px: float, false_px: float,
) -> Dict[str, Any] | None:
    cx, cy = int(round(kx)), int(round(ky))
    if cy < half_p or cy >= h - half_p or cx < half_p or cx >= w - half_p:
        return None
    tmpl = pc1[cy - half_p : cy + half_p, cx - half_p : cx + half_p]
    if float(np.std(tmpl)) < 1e-4:
        return None
    p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
    gt_x, gt_y = float(p_gt[0]), float(p_gt[1])
    # Search far from the true projection so the peak is a wrong correspondence.
    fx = int(rng.integers(half_p + 4, max(half_p + 5, w - half_p - 4)))
    fy = int(rng.integers(half_p + 4, max(half_p + 5, h - half_p - 4)))
    if math.hypot(fx - gt_x, fy - gt_y) < 12.0:
        return None
    rad = 16
    s_min_x, s_max_x = max(0, fx - rad), min(w, fx + rad)
    s_min_y, s_max_y = max(0, fy - rad), min(h, fy + rad)
    if s_max_x - s_min_x <= tmpl.shape[1] or s_max_y - s_min_y <= tmpl.shape[0]:
        return None
    search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
    if float(np.std(search_region)) < 1e-4:
        return None
    score, loc = find_best_correspondence_unified(search_region, tmpl, multimodal_pair=multimodal)
    peak_uniq = last_peak_uniqueness()
    found_x = float(s_min_x + loc[0] + half_p)
    found_y = float(s_min_y + loc[1] + half_p)
    rec = _finalize_record(
        kx, ky, found_x, found_y, tmpl, pc1, pc2, tgt_gallery, w, h, half_p,
        gt_x, gt_y, score, peak_uniq, domain, multimodal,
        true_px=true_px, false_px=false_px,
    )
    if rec is None or rec["human_label"] is True:
        return None
    rec["forced_mismatch"] = True
    return rec


def _finalize_record(
    kx, ky, found_x, found_y, tmpl, pc1, pc2, tgt_gallery, w, h, half_p,
    gt_x, gt_y, score, peak_uniq, domain, multimodal,
    true_px: float = 2.0, false_px: float = 5.0,
) -> Dict[str, Any] | None:
    ref_dx, ref_dy = 0.0, 0.0
    refined = False
    ibx, iby = int(round(found_x)), int(round(found_y))
    if iby >= half_p and iby + half_p <= h and ibx >= half_p and ibx + half_p <= w:
        p_ref = pc2[iby - half_p : iby + half_p, ibx - half_p : ibx + half_p]
        if p_ref.shape == tmpl.shape:
            dx, dy, _peak, valid = subpixel_phase_correlation(tmpl, p_ref)
            if valid and abs(dx) < 2.0 and abs(dy) < 2.0:
                ref_dx, ref_dy = float(dx), float(dy)
                found_x += ref_dx
                found_y += ref_dy
                refined = True

    true_err = math.hypot(found_x - gt_x, found_y - gt_y)
    label = _label_from_error(true_err, true_px=true_px, false_px=false_px)
    if label is None:
        return None

    spatial_quality = compute_spatial_quality_score(
        peak_uniqueness=peak_uniq,
        refinement_dx=ref_dx,
        refinement_dy=ref_dy,
        is_refined=refined,
        x=kx, y=ky, width=w, height=h,
    )
    desc = compute_descriptor_match_features(
        pc1, pc2, float(kx), float(ky), float(found_x), float(found_y),
        tgt_gallery=tgt_gallery,
    )
    return {
        "source_x": float(kx),
        "source_y": float(ky),
        "target_x": float(found_x),
        "target_y": float(found_y),
        "confidence": float(score),
        "refinement_dx": float(ref_dx),
        "refinement_dy": float(ref_dy),
        "spatial_quality_score": float(spatial_quality),
        "is_refined": bool(refined),
        "true_reprojection_error_px": float(round(true_err, 3)),
        "human_label": bool(label),
        "label_source": "hand",
        "domain": domain,
        "multimodal_pair": bool(multimodal),
        **desc,
    }


def main():
    rng = np.random.default_rng(SEED)
    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"

    image_paths = []
    for reg_dir in sorted(triplets_dir.glob("region_*")):
        for name in ("ohrc_512.png", "tmc_512.png"):
            p = reg_dir / name
            if p.exists():
                image_paths.append(p)

    logger.info("Discovered %d lunar test images across regions.", len(image_paths))
    if not image_paths:
        logger.error("No source images found in %s", triplets_dir)
        return 1

    all_records = []
    for p in image_paths:
        logger.info("Extracting ground-truth correspondences from %s...", p.relative_to(REPO_ROOT))
        recs = generate_matches_for_image(p, rng)
        all_records.extend(recs)
        logger.info("  -> Generated %d labeled correspondences so far.", len(all_records))

    true_count = sum(1 for r in all_records if r["human_label"])
    false_count = sum(1 for r in all_records if not r["human_label"])
    xs_true = [r["confidence"] for r in all_records if r.get("domain") == "cross_sensor" and r["human_label"]]
    xs_false = [r["confidence"] for r in all_records if r.get("domain") == "cross_sensor" and not r["human_label"]]
    logger.info("Generation complete: Total=%d, True=%d (%.1f%%), False=%d (%.1f%%)",
                len(all_records), true_count, 100.0 * true_count / max(1, len(all_records)),
                false_count, 100.0 * false_count / max(1, len(all_records)))
    if xs_true and xs_false:
        logger.info(
            "Cross-sensor confidence means: true=%.3f false=%.3f (gap=%.3f)",
            float(np.mean(xs_true)), float(np.mean(xs_false)),
            float(np.mean(xs_true) - np.mean(xs_false)),
        )

    out_file = REPO_ROOT / "ML_model/ground_truth_matches.json"
    out_file.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    logger.info("Saved ground-truth dataset to %s", out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
