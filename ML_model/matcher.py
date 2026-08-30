import cv2
import torch
import kornia as K
from kornia.feature import LoFTR
import numpy as np
import json

def match_images(img_path1, img_path2, output_json="matches.json", output_img="match_visual.jpg"):
    # 1. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load the Pre-trained LoFTR model FROM LOCAL FILE
    print("Loading LoFTR model from local file...")
    matcher = LoFTR(pretrained=None).to(device).eval()
    state_dict = torch.load('loftr_outdoor.ckpt', map_location=device)
    if 'state_dict' in state_dict:
        matcher.load_state_dict(state_dict['state_dict'])
    else:
        matcher.load_state_dict(state_dict)

    # 3. Read and preprocess images using OpenCV (Bulletproof method)
    print("Reading images...")
    img1_np = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2_np = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    # Resize if images are too large (LoFTR struggles with huge images on CPU)
    if img1_np.shape[0] > 512 or img1_np.shape[1] > 512:
        img1_np = cv2.resize(img1_np, (512, 512))
    if img2_np.shape[0] > 512 or img2_np.shape[1] > 512:
        img2_np = cv2.resize(img2_np, (512, 512))

    # Convert OpenCV images to PyTorch Tensors (Format required by LoFTR)
    img1 = K.image_to_tensor(img1_np, keepdim=True).float() / 255.0
    img2 = K.image_to_tensor(img2_np, keepdim=True).float() / 255.0
    img1 = img1.unsqueeze(0).to(device) # Add batch dimension
    img2 = img2.unsqueeze(0).to(device)

    # 4. Run the ML Model (Inference)
    print("Matching images... (This might take 10-30 seconds on CPU)")
    with torch.no_grad():
        input_dict = {"image0": img1, "image1": img2}
        correspondences = matcher(input_dict)

    # 5. Extract matched coordinates
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    confidence = correspondences['confidence'].cpu().numpy()

    if len(mkpts0) < 4:
        print("Not enough matches found. Try different images.")
        return

    # 6. RANSAC Filtering (Remove bad matches using geometry)
    print("Filtering matches with RANSAC...")
    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    
    inliers_idx = np.where(mask.ravel() == 1)[0]
    good_mkpts0 = mkpts0[inliers_idx]
    good_mkpts1 = mkpts1[inliers_idx]
    good_conf = confidence[inliers_idx]

    print(f"Found {len(good_mkpts0)} reliable matches!")

    # 7. Save results to JSON
    matches_data = []
    for i in range(len(good_mkpts0)):
        matches_data.append({
            "image1_x": float(good_mkpts0[i][0]),
            "image1_y": float(good_mkpts0[i][1]),
            "image2_x": float(good_mkpts1[i][0]),
            "image2_y": float(good_mkpts1[i][1]),
            "confidence": float(good_conf[i])
        })

    with open(output_json, 'w') as f:
        json.dump(matches_data, f, indent=4)
    print(f"Saved match coordinates to {output_json}")

    # 8. Create a visual image (For your PPT and demo)
    img1_color = cv2.imread(img_path1)
    img2_color = cv2.imread(img_path2)
    
    img1_color = cv2.resize(img1_color, (512, 512))
    img2_color = cv2.resize(img2_color, (512, 512))

    vis_img = np.hstack((img1_color, img2_color))
    offset = img1_color.shape[1]
    
    # HACKATHON PRO-TIP: Sort by confidence and only draw the TOP 20 matches!
    top_n = 20
    if len(good_mkpts0) > top_n:
        top_indices = np.argsort(good_conf)[-top_n:]
        draw_mkpts0 = good_mkpts0[top_indices]
        draw_mkpts1 = good_mkpts1[top_indices]
    else:
        draw_mkpts0 = good_mkpts0
        draw_mkpts1 = good_mkpts1

    for i in range(len(draw_mkpts0)):
        pt1 = (int(draw_mkpts0[i][0]), int(draw_mkpts0[i][1]))
        pt2 = (int(draw_mkpts1[i][0]) + offset, int(draw_mkpts1[i][1]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(vis_img, pt1, pt2, color, 1)
        cv2.circle(vis_img, pt1, 4, color, -1)
        cv2.circle(vis_img, pt2, 4, color, -1)

    cv2.imwrite(output_img, vis_img)
    print(f"Saved visual matching image to {output_img}")


# --- HOW TO RUN IT ---
if __name__ == "__main__":
    # Updated to use the real lunar data from the data team
    image1 = "ohrc_512.jpeg" 
    image2 = "tmc_512.jpeg"
    
    match_images(image1, image2)