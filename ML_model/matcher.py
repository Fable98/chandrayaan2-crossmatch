import cv2
import torch
import kornia as K
from kornia.feature import LoFTR
import numpy as np
import json
import os
import math

def match_images(img_path1, img_path2, output_dir="output"):
    """
    Complete pipeline for SIH26166:
    1. Scale-invariant LoFTR matching (mapping back to original resolution).
    2. Grid-based spatial uniformity filter.
    3. Sub-pixel refinement via Phase Correlation.
    4. Homography computation and Registered Product generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Read original images
    img1_orig = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2_orig = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    img1_color = cv2.imread(img_path1)
    img2_color = cv2.imread(img_path2)
    
    orig_h1, orig_w1 = img1_orig.shape
    orig_h2, orig_w2 = img2_orig.shape

    # 2. Prepare for LoFTR (Resize internally for inference only)
    # LoFTR performs best on ~512x512. We resize for the network, then scale coordinates back.
    target_size = 512
    img1_resized = cv2.resize(img1_orig, (target_size, target_size))
    img2_resized = cv2.resize(img2_orig, (target_size, target_size))

    img1_tensor = K.image_to_tensor(img1_resized, keepdim=True).float() / 255.0
    img2_tensor = K.image_to_tensor(img2_resized, keepdim=True).float() / 255.0
    img1_tensor = img1_tensor.unsqueeze(0).to(device)
    img2_tensor = img2_tensor.unsqueeze(0).to(device)

    # Load LoFTR
    matcher = LoFTR(pretrained=None).to(device).eval()
    # Assume loftr_outdoor.ckpt is in the same directory or root
    ckpt_path = 'loftr_outdoor.ckpt'
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), 'loftr_outdoor.ckpt')
        
    state_dict = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in state_dict:
        matcher.load_state_dict(state_dict['state_dict'])
    else:
        matcher.load_state_dict(state_dict)

    # 3. Inference
    with torch.no_grad():
        correspondences = matcher({"image0": img1_tensor, "image1": img2_tensor})

    mkpts0_resized = correspondences['keypoints0'].cpu().numpy()
    mkpts1_resized = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()

    if len(mkpts0_resized) < 4:
        return {"error": "Not enough matches found."}

    # 4. Scale coordinates back to ORIGINAL image space (Crucial for Scale Invariance)
    scale_x1, scale_y1 = orig_w1 / target_size, orig_h1 / target_size
    scale_x2, scale_y2 = orig_w2 / target_size, orig_h2 / target_size

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

    # 10. Calculate Metrics
    # Calculate RMSE based on the final inliers
    projected_pts = cv2.perspectiveTransform(uniform_mkpts0.reshape(-1, 1, 2), H_final)
    projected_pts = projected_pts.reshape(-1, 2)
    errors = np.linalg.norm(projected_pts - refined_mkpts1, axis=1).flatten()
    
    rmse = np.sqrt(np.mean(errors**2))
    inlier_ratio = len(uniform_mkpts0) / max(1, len(mkpts0))
    uniformity_score = len(grid_dict) / float(grid_size * grid_size)
    
    metrics = {
        "num_inliers": len(uniform_mkpts0),
        "rmse_px": float(rmse),
        "inlier_ratio": float(inlier_ratio),
        "uniformity_score": float(uniformity_score),
        "sub_pixel_accurate": bool(rmse < 1.0),
        "fraction_below_1px": float(np.mean(errors < 1.0))
    }

    # Save matches to JSON
    matches_data = []
    for i in range(len(uniform_mkpts0)):
        matches_data.append({
            "image1_x": float(uniform_mkpts0[i][0]),
            "image1_y": float(uniform_mkpts0[i][1]),
            "image2_x": float(refined_mkpts1[i][0]),
            "image2_y": float(refined_mkpts1[i][1])
        })
        
    json_path = os.path.join(output_dir, "matches.json")
    with open(json_path, 'w') as f:
        json.dump(matches_data, f, indent=4)

    return {
        "metrics": metrics,
        "homography": H_final.tolist() if H_final is not None else None,
        "matches_path": json_path,
        "visual_path": vis_path,
        "warped_path": warp_path
    }

if __name__ == "__main__":
    # Test run
    res = match_images("ohrc_512.jpeg", "tmc_512.jpeg", output_dir="test_output")
    print(res)