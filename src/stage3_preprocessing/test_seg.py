"""
Test improved avocado segmentation:
- Uses adaptive bounding boxes (from color detection)
- Applies Grayworld + CLAHE preprocessing on each ROI
- Uses GrabCut initialized with color support mask
- Applies morphological smoothing + convex hull to eliminate spiky edges
"""
import cv2
import numpy as np
from pathlib import Path
import os
PROJECT_ROOT = Path(r"c:\fluttersrc\Strawberry-RUL-prediction")
OUTPUT_DIR = PROJECT_ROOT / "scratch_segmented_avocado"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Color ranges for avocado candidate detection
AVOCADO_COLOR_RANGES = [
    (np.array([18, 8, 8]), np.array([95, 255, 255])),
    (np.array([0, 15, 12]), np.array([35, 255, 200])),
]
def apply_grayworld(image):
    try:
        b, g, r = cv2.split(image)
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        if b_mean == 0 or g_mean == 0 or r_mean == 0:
            return image
        avg_mean = (b_mean + g_mean + r_mean) / 3.0
        b_sc = np.clip(b * (avg_mean / b_mean), 0, 255).astype(np.uint8)
        g_sc = np.clip(g * (avg_mean / g_mean), 0, 255).astype(np.uint8)
        r_sc = np.clip(r * (avg_mean / r_mean), 0, 255).astype(np.uint8)
        return cv2.merge((b_sc, g_sc, r_sc))
    except Exception:
        return image
def apply_clahe_lab(image, clip_limit=3.0, grid_size=(8, 8)):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l_clahe = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        return image
def fill_holes(mask):
    flood = mask.copy()
    flood_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)
def smooth_mask_border(mask, blur_size=31, threshold=80):
    """Apply Gaussian blur then threshold to smooth jagged borders."""
    # Must use odd kernel sizes for GaussianBlur
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
    return np.where(blurred > threshold, 255, 0).astype(np.uint8)
def smooth_mask_morphological(mask):
    """
    Aggressive morphological smoothing:
    - Close with large ellipse kernel to fill gaps and round out concavities
    - Open to remove tiny spiky protrusions
    """
    # Large close to round out concavities and fill interior gaps
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    # Open to remove tiny spiky protrusions on the outside
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    return mask
def smooth_mask_distance_transform(mask, strength=0.4):
    """
    Super smooth borders using distance transform:
    - Erode the mask -> get interior
    - Distance transform -> smooth falloff from center
    - Threshold at some fraction of max to get smooth boundary
    This produces organically smooth, round borders like the reference image.
    """
    # Distance transform from foreground to background
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Also distance transform from background to foreground  
    dist_bg = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 5)
    # Signed distance: positive inside, negative outside
    signed = dist.astype(np.float32) - dist_bg.astype(np.float32)
    # Threshold at 0 = original boundary; negative threshold = contract; positive = expand
    # We use a small positive threshold to slightly contract (removes thin spiky protrusions)
    contracted = (signed > 1.0).astype(np.uint8) * 255
    # Then expand back with smooth blur
    blurred = cv2.GaussianBlur(contracted.astype(np.float32), (51, 51), 0)
    smooth = np.where(blurred > 60, 255, 0).astype(np.uint8)
    return smooth
def segment_avocado_roi(roi, pad_x=0, pad_y=0):
    """
    Segment a single avocado from its padded ROI.
    
    Strategy:
    1. Grayworld + CLAHE to boost dark green/brown skin contrast
    2. Color candidate mask to find avocado pixels
    3. GrabCut initialized with color mask
    4. Post-processing: largest component + fill holes + smooth border
    
    Returns: (final_mask, success)
    """
    if roi.size == 0:
        return np.zeros(roi.shape[:2], dtype=np.uint8), False
    
    roi_h, roi_w = roi.shape[:2]
    
    # Step 1: Preprocess for contrast
    gw = apply_grayworld(roi)
    preproc = apply_clahe_lab(gw, clip_limit=3.0, grid_size=(8, 8))
    
    # Step 2: Color candidate mask on the preprocessed ROI
    hsv = cv2.cvtColor(preproc, cv2.COLOR_BGR2HSV)
    color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in AVOCADO_COLOR_RANGES:
        color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))
    
    # Remove surface background (white/neutral)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    neutral = (sat <= 40) & (val >= 80)
    color_mask[neutral] = 0
    
    # Also remove dark edges of the padding area (if they look like wall/container edge)
    # Use dark brown/black that wouldn't be avocado
    very_dark = (val < 15)
    color_mask[very_dark] = 0
    
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_sm)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_lg)
    
    # Step 3: Build GrabCut initialization mask from color support
    gc_mask = np.full((roi_h, roi_w), cv2.GC_PR_BGD, dtype=np.uint8)
    
    # Find the largest connected component in color mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, 8)
    if num_labels <= 1:
        # Fallback: use rect-based GrabCut
        margin = max(10, min(roi_w, roi_h) // 12)
        rect = (margin, margin, roi_w - 2*margin, roi_h - 2*margin)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(preproc, gc_mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
    else:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        main_component = (labels == largest_label).astype(np.uint8) * 255
        
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        
        # Dilated region = probable foreground
        probable_fg = cv2.dilate(main_component, kernel_dilate, iterations=2) > 0
        # Eroded region = definite foreground  
        sure_fg = cv2.erode(main_component, kernel_erode, iterations=2) > 0
        # Outer ring = definite background
        outer = cv2.dilate(main_component, kernel_outer, iterations=1)
        sure_bg = (outer == 0)
        
        # Include dark skin pixels likely to be missed by color mask (dark greens near FG)
        dark_skin = (val < 90) & (sat > 8) & probable_fg
        specular = (sat <= 15) & (val >= 40) & probable_fg
        
        gc_mask[probable_fg | dark_skin | specular] = cv2.GC_PR_FGD
        gc_mask[sure_fg] = cv2.GC_FGD
        gc_mask[sure_bg & ~probable_fg] = cv2.GC_BGD
        
        # Force border to background
        border_px = 3
        gc_mask[:border_px, :] = cv2.GC_BGD
        gc_mask[-border_px:, :] = cv2.GC_BGD
        gc_mask[:, :border_px] = cv2.GC_BGD
        gc_mask[:, -border_px:] = cv2.GC_BGD
        
        # Run GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        if not np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)):
            # No foreground hint — fallback to rect
            margin = max(10, min(roi_w, roi_h) // 12)
            rect = (margin, margin, roi_w - 2*margin, roi_h - 2*margin)
            gc_mask = np.full((roi_h, roi_w), cv2.GC_PR_BGD, dtype=np.uint8)
            cv2.grabCut(preproc, gc_mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
        else:
            cv2.grabCut(preproc, gc_mask, None, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_MASK)
    
    # Step 4: Extract binary mask
    binary_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    # Step 5: Post-processing
    # Remove definite background regions (white/neutral surface)
    binary_mask[neutral] = 0
    
    # Keep only largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
    if num_labels <= 1:
        return binary_mask, False
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = (labels == largest_label).astype(np.uint8) * 255
    
    # Check if the result is large enough  
    area = (clean_mask > 0).sum()
    if area < 5000:
        return clean_mask, False
    
    # Fill interior holes
    clean_mask = fill_holes(clean_mask)
    
    # Aggressive morphological smoothing first (closes concavities, removes spikes)
    clean_mask = smooth_mask_morphological(clean_mask)
    
    # Re-extract largest component after morphology (in case closing merged noise)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(clean_mask, 8)
    if num_labels2 > 1:
        largest_label2 = 1 + np.argmax(stats2[1:, cv2.CC_STAT_AREA])
        clean_mask = (labels2 == largest_label2).astype(np.uint8) * 255
    
    # Fill holes again after morphology
    clean_mask = fill_holes(clean_mask)
    
    # Step 1: Distance transform based smoothing (produces smooth organic borders)
    smooth = smooth_mask_distance_transform(clean_mask)
    
    # Step 2: Re-apply fill holes after distance transform
    smooth = fill_holes(smooth)
    
    # Step 3: Final Gaussian blur + threshold to ensure pixel-level smoothness
    smooth = smooth_mask_border(smooth, blur_size=21, threshold=80)
    
    return smooth, True
def compute_grid_index(cX, cY, img_h, img_w):
    """Map center coords to 1-6 grid index (avocado: bottom row = 1-3, top row = 4-6)."""
    if cX < (img_w / 3):
        col = 0
    elif cX < (2 * img_w / 3):
        col = 1
    else:
        col = 2
    row = 0 if cY >= (img_h / 2) else 1
    return (row * 3) + col + 1
def segment_frame(img_path, output_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot read: {img_path}")
        return
    h, w = img.shape[:2]
    base = Path(img_path).stem
    print(f"Processing {base} ({w}x{h})")
    
    # Build full-frame candidate mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in AVOCADO_COLOR_RANGES:
        color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv, lower, upper))
    
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    neutral = (sat <= 40) & (val >= 80)
    color_mask[neutral] = 0
    
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_sm)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_lg)
    
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter valid avocado contours
    min_area = max(8000, int(0.0015 * h * w))
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    print(f"  Found {len(contours)} candidate avocado regions")
    
    segmented = 0
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        fruit_idx = compute_grid_index(cX, cY, h, w)
        
        # Get bounding box with generous padding
        x, y, wb, hb = cv2.boundingRect(cnt)
        pad = 35  # generous padding so fruit is fully inside
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + wb + pad)
        y2 = min(h, y + hb + pad)
        
        roi = img[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        
        print(f"  Fruit {fruit_idx}: bbox=[{x1},{y1},{x2},{y2}] size={roi_w}x{roi_h}")
        
        # Segment this ROI
        final_mask, success = segment_avocado_roi(roi, pad, pad)
        
        if not success:
            print(f"    -> Segmentation produced small/empty mask, skipping")
            continue
        
        # Save transparent crop (the ROI size, not full frame)
        b_ch, g_ch, r_ch = cv2.split(roi)
        roi_transparent = cv2.merge([b_ch, g_ch, r_ch, final_mask])
        out_path = output_dir / f"{base}_avocado_{fruit_idx}.png"
        cv2.imwrite(str(out_path), roi_transparent)
        
        # Verify quality stats
        area_frac = (final_mask > 0).sum() / final_mask.size
        print(f"    -> area_frac={area_frac:.3f}, saved to {out_path.name}")
        segmented += 1
    
    print(f"  Total segmented: {segmented}")
    return segmented
# Test on the first avocado frame
test_img = PROJECT_ROOT / "data/02_processed/cropped_avocado/webcam_2026-06-14_20-30-44.jpg"
count = segment_frame(test_img, OUTPUT_DIR)
print(f"\nDone. Outputs in {OUTPUT_DIR}")