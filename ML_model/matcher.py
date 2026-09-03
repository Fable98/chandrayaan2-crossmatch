"""
matcher.py — Pretrained LoFTR Baseline (Alternative / Benchmark Pipeline)

Retained as an alternative baseline for comparative evaluation against the primary
CFOG / Phase Congruency pipeline. Uses pretrained outdoor transformer weights (not lunar-trained).
"""

import cv2
import numpy as np
import json
import os
import math

from metrics import compute_canonical_metrics

def match_images(img_path1, img_path2, output_dir="output"):
    """
    Pretrained LoFTR Baseline Matcher:
    1. Resizes to model-compatible 512x512 with recorded coordinate scaling.
    2. Runs pretrained LoFTR feature matcher (baseline).
    3. Transforms detected coordinates back to native image dimensions.
    4. Evaluates canonical metrics (Fit RMSE, Held-out Validation RMSE, spatial distribution).
    """
    # LAZY IMPORTS: Prevents OOM crash on Render Free Tier (512MB RAM) during startup
    import torch
    import kornia as K
    from kornia.feature import LoFTR
    import gc

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_as_gray_and_color(path):
        try:
            import rasterio
            with rasterio.open(path) as src:
                if src.count > 3:  # Hyperspectral cube (e.g., IIRS)
                    bands = src.read()  # (C, H, W)
                    gray = np.nanmean(bands, axis=0).astype("float32")
                    gray = np.clip(gray, 0, 255).astype("uint8")
                    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    return gray, color
                elif src.count >= 3:
                    rgb = np.dstack([src.read(i) for i in (1, 2, 3)])
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    color = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    return gray, color
                else:
                    gray = src.read(1).astype(np.uint8)
                    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    return gray, color
        except Exception:
            pass
        
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(path)
        return gray, color

    # 1. Read original images (with IIRS hyperspectral support)
    img1_orig, img1_color = load_as_gray_and_color(img_path1)
    img2_orig, img2_color = load_as_gray_and_color(img_path2)
    
    orig_h1, orig_w1 = img1_orig.shape[:2]
    orig_h2, orig_w2 = img2_orig.shape[:2]

    # 2. Prepare for LoFTR (Resize internally for inference only)
    # LoFTR performs best on ~512x512. We resize for the network, then scale coordinates back.
    target_size = 512
    img1_resized = cv2.resize(img1_orig, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2_orig, (target_size, target_size), interpolation=cv2.INTER_AREA)

    t_img1 = K.image_to_tensor(img1_resized, False).float() / 255.0
    t_img2 = K.image_to_tensor(img2_resized, False).float() / 255.0

    t_img1 = t_img1.to(device)
    t_img2 = t_img2.to(device)

    # 3. Match using pretrained LoFTR
    matcher = LoFTR(pretrained='outdoor').to(device)
    matcher.eval()
    
    with torch.no_grad():
        input_dict = {"image0": t_img1, "image1": t_img2}
        correspondences = matcher(input_dict)

    mkpts0_resized = correspondences['keypoints0'].cpu().numpy()
    mkpts1_resized = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()

    del matcher, t_img1, t_img2, input_dict, correspondences
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if len(mkpts0_resized) < 4:
        return {
            "status": "insufficient_correspondences",
            "message": f"Pretrained LoFTR baseline found only {len(mkpts0_resized)} matches (minimum 4 required).",
            "match_count": len(mkpts0_resized),
            "inlier_count": 0,
            "metrics": None,
            "homography": None,
        }

    # 4. Map coordinates back to NATIVE image space
    scale_x1, scale_y1 = orig_w1 / float(target_size), orig_h1 / float(target_size)
    scale_x2, scale_y2 = orig_w2 / float(target_size), orig_h2 / float(target_size)

    mkpts0 = mkpts0_resized.copy()
    mkpts1 = mkpts1_resized.copy()
    
    mkpts0[:, 0] *= scale_x1
    mkpts0[:, 1] *= scale_y1
    mkpts1[:, 0] *= scale_x2
    mkpts1[:, 1] *= scale_y2

    # 5. RANSAC Filtering in original space
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    inliers_idx = np.where(mask.ravel() == 1)[0]
    
    good_mkpts0 = mkpts0[inliers_idx]
    good_mkpts1 = mkpts1[inliers_idx]
    good_conf = confidence[inliers_idx]

    # 6. Grid-Based Spatial Uniformity Filter (Ensures uniform distribution across image)
    grid_size = 10  # 10x10 grid
    cell_w = orig_w1 / grid_size
    cell_h = orig_h1 / grid_size
    max_points_per_cell = 5
    
    grid_dict = {}
    for i in range(len(good_mkpts0)):
        cx = int(good_mkpts0[i][0] / cell_w)
        cy = int(good_mkpts0[i][1] / cell_h)
        cx = min(cx, grid_size - 1)
        cy = min(cy, grid_size - 1)
        
        grid_dict.setdefault((cx, cy), []).append((good_conf[i], i))
        
    uniform_indices = []
    for cell_points in grid_dict.values():
        # Sort by confidence descending, take top N
        cell_points.sort(key=lambda x: x[0], reverse=True)
        uniform_indices.extend([idx for _, idx in cell_points[:max_points_per_cell]])
        
    uniform_mkpts0 = good_mkpts0[uniform_indices]
    uniform_mkpts1 = good_mkpts1[uniform_indices]

    # 7. Sub-Pixel Refinement using Phase Correlation on patches
    patch_size = 32
    half_patch = patch_size // 2
    
    refined_mkpts1 = uniform_mkpts1.copy()
    
    for i in range(len(uniform_mkpts0)):
        x1, y1 = int(uniform_mkpts0[i][0]), int(uniform_mkpts0[i][1])
        x2, y2 = int(uniform_mkpts1[i][0]), int(uniform_mkpts1[i][1])
        
        # Extract patches (ensure within image boundaries)
        y1_min, y1_max = max(0, y1-half_patch), min(orig_h1, y1+half_patch)
        x1_min, x1_max = max(0, x1-half_patch), min(orig_w1, x1+half_patch)
        
        y2_min, y2_max = max(0, y2-half_patch), min(orig_h2, y2+half_patch)
        x2_min, x2_max = max(0, x2-half_patch), min(orig_w2, x2+half_patch)
        
        # Only process if patches are exactly patch_size x patch_size
        if (y1_max - y1_min == patch_size) and (x1_max - x1_min == patch_size) and \
           (y2_max - y2_min == patch_size) and (x2_max - x2_min == patch_size):
            
            patch1 = img1_orig[y1_min:y1_max, x1_min:x1_max].astype(np.float32)
            patch2 = img2_orig[y2_min:y2_max, x2_min:x2_max].astype(np.float32)
            
            # Phase Correlation for sub-pixel shift
            (dx, dy), response = cv2.phaseCorrelate(patch1, patch2)
            
            # Apply sub-pixel shift to image 2 coordinates
            refined_mkpts1[i][0] += dx
            refined_mkpts1[i][1] += dy

    # 8. Compute Final Homography with Refined Sub-Pixel Matches
    H_final, _ = cv2.findHomography(uniform_mkpts0, refined_mkpts1, cv2.RANSAC, 3.0)
    
    # 9. Generate Registered Product (Visual Output)
    # Warp Image 1 to Image 2's perspective
    warped_img1 = cv2.warpPerspective(img1_color, H_final, (orig_w2, orig_h2))
    
    # Create a Checkerboard Blend to visually prove alignment
    block_size = 50
    blended = np.zeros_like(img2_color)
    for y in range(0, orig_h2, block_size):
        for x in range(0, orig_w2, block_size):
            if ((x // block_size) + (y // block_size)) % 2 == 0:
                blended[y:y+block_size, x:x+block_size] = warped_img1[y:y+block_size, x:x+block_size]
            else:
                blended[y:y+block_size, x:x+block_size] = img2_color[y:y+block_size, x:x+block_size]
                
    vis_path = os.path.join(output_dir, "registered_checkerboard.jpg")
    cv2.imwrite(vis_path, blended)
    
    warp_path = os.path.join(output_dir, "warped_source.jpg")
    cv2.imwrite(warp_path, warped_img1)

    # 10. Compute Canonical Master Metrics
    inlier_mask_final = np.ones((len(uniform_mkpts0), 1), dtype=np.uint8)
    metrics = compute_canonical_metrics(
        uniform_mkpts0, refined_mkpts1, inlier_mask_final, H_final, (orig_h2, orig_w2), grid_size
    )
    # Add backward compatibility aliases
    metrics["num_inliers"] = metrics["inlier_count"]
    metrics["rmse_px"] = metrics["fit_rmse_px"]
    metrics["uniformity_score"] = metrics["spatial_uniformity"]

    # Save matches to JSON
    matches_data = []
    for i in range(len(uniform_mkpts0)):
        matches_data.append({
            "source_x": float(uniform_mkpts0[i][0]),
            "source_y": float(uniform_mkpts0[i][1]),
            "target_x": float(refined_mkpts1[i][0]),
            "target_y": float(refined_mkpts1[i][1]),
            "is_inlier": True,
        })
        
    json_path = os.path.join(output_dir, "matches.json")
    with open(json_path, 'w') as f:
        json.dump(matches_data, f, indent=4)

    return {
        "status": "success",
        "method": "loftr_pretrained_baseline",
        "metrics": metrics,
        "homography": H_final.tolist() if H_final is not None else None,
        "matches_path": json_path,
        "visual_path": vis_path,
        "warped_path": warp_path,
    }

if __name__ == "__main__":
    # Test run
    res = match_images("ohrc_512.jpeg", "tmc_512.jpeg", output_dir="test_output")
    print(res)