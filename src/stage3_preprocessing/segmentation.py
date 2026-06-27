import argparse
import cv2
import glob
import os
import re
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = PROJECT_ROOT / "src" / "stage3_preprocessing" / "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    configs = json.load(f)

active_dataset = configs["active_dataset"]
dataset_cfg = configs["datasets"][active_dataset]

processed_root = PROJECT_ROOT / configs["processed_dir"]

input_dir = PROJECT_ROOT / dataset_cfg["output_dir"]
output_dir = PROJECT_ROOT / "data" / "02_processed" / dataset_cfg["mask_dir"]

# Define wider color ranges for avocado candidates in HSV color space.
# Dark green avocado skin can be very deep, with some brown/ripening patches.
STRAWBERRY_COLOR_RANGES = [
    (np.array([0, 25, 18]), np.array([25, 255, 255])),    # red/dark red/orange
    (np.array([160, 25, 18]), np.array([180, 255, 255])), # wrapped red
    (np.array([5, 20, 15]), np.array([45, 255, 245])),    # brown/damaged fruit
    (np.array([35, 25, 15]), np.array([100, 255, 245])),  # green calyx/leaves
]
AVOCADO_COLOR_RANGES = [
    (np.array([18, 8, 8]), np.array([95, 255, 255])),      # dark to bright green avocado skin
    (np.array([0, 15, 12]), np.array([35, 255, 200])),    # brown stem / ripening patches
]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
avocado_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grabcut_outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))


def natural_sort_key(path):
    """Sort frame filenames naturally, so frame-2 comes before frame-10."""

    stem = os.path.splitext(os.path.basename(path))[0].lower()
    parts = re.split(r'(\d+)', stem)
    return [int(part) if part.isdigit() else part for part in parts]


def is_cropped_folder(folder_name):
    return re.match(
        r"^cropped_\d{2}-\d{2}-\d{4}$",
        folder_name
    ) is not None

def extract_frame_number(path):
    """Extract the number from names like frame-12_15-11-29.jpg."""

    filename = os.path.basename(path)
    match = re.search(r'frame-(\d+)', filename, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def should_process_image(path, args):
    """Return True if an image is inside the requested segmentation range."""

    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]
    frame_number = extract_frame_number(path)

    if args.only_frame and frame_number is not None and frame_number not in args.only_frame:
        return False
    if args.skip_frame and frame_number in args.skip_frame:
        return False
    if args.start_frame is not None and frame_number is not None and frame_number < args.start_frame:
        return False
    if args.end_frame is not None and frame_number is not None and frame_number > args.end_frame:
        return False
    if args.start_name and stem < args.start_name:
        return False
    if args.end_name and stem > args.end_name:
        return False
    return True


def is_table_background(hsv):
    """Detect the white table surface and the soft grey shadow directly on it."""

    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)
    table = (saturation >= 10) & (saturation <= 25) & (value >= 130) & (value <= 175)
    shadow = (saturation >= 5) & (saturation <= 30) & (value >= 85) & (value <= 125)
    return table | shadow


def is_surface_background(hsv):
    """Detect table, shadow, and neutral gray/white background pixels."""

    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)
    neutral = (saturation <= 35) & (value >= 85)
    return neutral | is_table_background(hsv)


def is_foreign_foreground(hsv):
    """Detect non-avocado colors such as pink tray or hand artifacts."""

    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.int16)
    return (hue >= 140) & (hue <= 179) & (saturation >= 20)


def peel_attached_background(mask, hsv, background_predicate):
    """Remove foreground pixels matching a predicate and connected to the ROI border."""

    height, width = mask.shape
    bg_like = background_predicate(hsv)
    flood = np.zeros((height, width), np.uint8)
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)

    for x in range(width):
        for y in (0, height - 1):
            if mask[y, x] and bg_like[y, x]:
                cv2.floodFill(flood, flood_mask, (x, y), 255)

    for y in range(height):
        for x in (0, width - 1):
            if mask[y, x] and bg_like[y, x]:
                cv2.floodFill(flood, flood_mask, (x, y), 255)

    cleaned = mask.copy()
    cleaned[(flood > 0) & (mask > 0)] = 0
    return cleaned


def create_avocado_candidate_mask(hsv):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in AVOCADO_COLOR_RANGES:
        mask = cv2.inRange(hsv, lower, upper) | mask

    mask[is_surface_background(hsv)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, avocado_close_kernel)
    return mask


def create_candidate_mask(hsv, dataset="avocado"):
    if dataset == "avocado":
        return create_avocado_candidate_mask(hsv)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in STRAWBERRY_COLOR_RANGES:
        mask = cv2.inRange(hsv, lower, upper) | mask

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    object_mask = ((saturation > 35) & (value > 18) & (value < 245)).astype('uint8') * 255
    damaged_mask = ((saturation > 15) & (value > 12) & (value < 110)).astype('uint8') * 255

    mask = cv2.bitwise_or(mask, object_mask)
    mask = cv2.bitwise_or(mask, damaged_mask)

    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def is_valid_strawberry_contour(cnt, img_h, img_w):
    area = cv2.contourArea(cnt)
    min_area = max(300, int(0.0003 * img_h * img_w))
    if area < min_area:
        return False

    x, y, w_box, h_box = cv2.boundingRect(cnt)
    if w_box < 20 or h_box < 20:
        return False

    aspect_ratio = w_box / float(h_box)
    if aspect_ratio < 0.25 or aspect_ratio > 3.0:
        return False

    extent = area / float(w_box * h_box)
    return extent > 0.12


def is_valid_avocado_contour(cnt, img_h, img_w):
    area = cv2.contourArea(cnt)
    min_area = max(8000, int(0.0015 * img_h * img_w))
    if area < min_area:
        return False

    x, y, w_box, h_box = cv2.boundingRect(cnt)
    if w_box < 40 or h_box < 40:
        return False

    aspect_ratio = w_box / float(h_box)
    if aspect_ratio < 0.2 or aspect_ratio > 4.0:
        return False

    extent = area / float(w_box * h_box)
    return extent > 0.08


def fill_holes(binary_mask):
    flood = binary_mask.copy()
    flood_mask = np.zeros((binary_mask.shape[0] + 2, binary_mask.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary_mask, holes)


def largest_component_mask(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype('uint8') * 255


def create_grabcut_mask_avocado(roi, roi_support):
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_h, roi_w = roi.shape[:2]
    grabcut_mask = np.full((roi_h, roi_w), cv2.GC_PR_BGD, dtype=np.uint8)

    main_support = largest_component_mask(roi_support)
    probable_fg = cv2.dilate(main_support, kernel, iterations=3) > 0

    sure_fg = cv2.erode(main_support, small_kernel, iterations=1) > 0
    sure_bg = cv2.dilate(main_support, grabcut_outer_kernel, iterations=1) == 0
    table_bg = is_surface_background(roi_hsv)

    dark_skin = (roi_hsv[:, :, 2] < 95) & (roi_hsv[:, :, 1] > 2) & probable_fg
    specular_skin = (roi_hsv[:, :, 1] <= 12) & (roi_hsv[:, :, 2] >= 45) & probable_fg

    grabcut_mask[probable_fg | dark_skin | specular_skin] = cv2.GC_PR_FGD
    grabcut_mask[sure_fg] = cv2.GC_FGD
    grabcut_mask[sure_bg | (table_bg & ~probable_fg)] = cv2.GC_BGD

    grabcut_mask[0, :] = cv2.GC_BGD
    grabcut_mask[-1, :] = cv2.GC_BGD
    grabcut_mask[:, 0] = cv2.GC_BGD
    grabcut_mask[:, -1] = cv2.GC_BGD

    if not np.any((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)):
        grabcut_mask[main_support > 0] = cv2.GC_PR_FGD

    return grabcut_mask


def refine_avocado_mask(mask_res, roi, _roi_support):
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    refined = mask_res.copy()
    refined[is_surface_background(roi_hsv) | is_foreign_foreground(roi_hsv)] = 0
    refined = peel_attached_background(refined, roi_hsv, is_surface_background)
    refined = peel_attached_background(refined, roi_hsv, is_foreign_foreground)
    refined = fill_holes(refined * 255) // 255
    refined = cv2.erode(refined, small_kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, 8)
    if num_labels <= 1:
        return refined

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype('uint8')


def create_grabcut_mask(roi, roi_support):
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    grabcut_mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)

    probable_fg = cv2.dilate(roi_support, kernel, iterations=2) > 0
    sure_fg = cv2.erode(roi_support, small_kernel, iterations=1) > 0
    sure_bg = cv2.dilate(roi_support, grabcut_outer_kernel, iterations=1) == 0
    white_bg = (roi_hsv[:, :, 1] < 60) & (roi_hsv[:, :, 2] > 90)

    grabcut_mask[probable_fg] = cv2.GC_PR_FGD
    grabcut_mask[sure_fg] = cv2.GC_FGD
    grabcut_mask[sure_bg | white_bg] = cv2.GC_BGD

    grabcut_mask[0, :] = cv2.GC_BGD
    grabcut_mask[-1, :] = cv2.GC_BGD
    grabcut_mask[:, 0] = cv2.GC_BGD
    grabcut_mask[:, -1] = cv2.GC_BGD

    if not np.any((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)):
        grabcut_mask[roi_support > 0] = cv2.GC_PR_FGD

    return grabcut_mask


def refine_foreground_mask(mask_res, color_support, roi):
    color_support = cv2.dilate(color_support, kernel, iterations=1)
    refined = mask_res & (color_support > 0).astype('uint8')

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    background_like = (roi_hsv[:, :, 1] < 70) & (roi_hsv[:, :, 2] > 70)
    refined[background_like] = 0

    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, small_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, 8)
    if num_labels <= 1:
        return refined

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype('uint8')


def apply_mask_to_roi(roi, mask_res):
    b_channel, g_channel, r_channel = cv2.split(roi)
    alpha_channel = mask_res * 255
    return cv2.merge([b_channel, g_channel, r_channel, alpha_channel])





def compute_grid_index(cX, cY, img_h, img_w, dataset="avocado"):
    """Map object center to a 2x3 grid index from 1 to 6."""

    if cX < (img_w / 3):
        col = 0
    elif cX < (2 * img_w / 3):
        col = 1
    else:
        col = 2

    if dataset == "avocado":
        # Bottom row: 1-3, top row: 4-6.
        row = 0 if cY >= (img_h / 2) else 1
    else:
        # Top row: 1-3, bottom row: 4-6.
        row = 0 if cY < (img_h / 2) else 1

    return (row * 3) + col + 1


def apply_grayworld(image):
    try:
        b, g, r = cv2.split(image)
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        if b_mean == 0 or g_mean == 0 or r_mean == 0:
            return image
        avg_mean = (b_mean + g_mean + r_mean) / 3.0
        b_scaled = b * (avg_mean / b_mean)
        g_scaled = g * (avg_mean / g_mean)
        r_scaled = r * (avg_mean / r_mean)
        return np.clip(cv2.merge((b_scaled.astype(np.float32), g_scaled.astype(np.float32), r_scaled.astype(np.float32))), 0, 255).astype(np.uint8)
    except Exception:
        return image


def apply_clahe_lab(image, clip_limit=2.0, grid_size=(5, 5)):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l_clahe = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        return image


AVOCADO_CELLS = {
    1: [200,  450,  750,  1080],
    2: [750,  450,  1300, 1080],
    3: [1300, 450,  1920, 1080],
    4: [0,    0,    750,  650],
    5: [700,  0,    1350, 650],
    6: [1300, 0,    1920, 650]
}


STRAWBERRY_CELLS = {
    1: [150, 100, 480, 450],
    2: [700, 100, 1050, 450],
    3: [1300, 100, 1650, 450],
    4: [80, 550, 420, 950],
    5: [700, 550, 1050, 950],
    6: [1350, 550, 1700, 950]
}


def parse_args():
    parser = argparse.ArgumentParser(description='Segment strawberries from a selected frame range.')
    parser.add_argument('--input-dir', default=input_dir, help='Folder containing cropped input images.')
    parser.add_argument('--output-dir', default=output_dir, help='Folder where segmented PNG files are written.')
    parser.add_argument('--start-frame', type=int, default=None, help='Process frames with frame number >= this value.')
    parser.add_argument('--end-frame', type=int, default=None, help='Process frames with frame number <= this value.')
    parser.add_argument('--only-frame', type=int, nargs='*', default=None, help='Process only these frame numbers, for example --only-frame 3 4 5.')
    parser.add_argument('--skip-frame', type=int, nargs='*', default=None, help='Skip these frame numbers, for example --skip-frame 3.')
    parser.add_argument('--start-name', default=None, help='Process filenames whose stem is >= this value.')
    parser.add_argument('--end-name', default=None, help='Process filenames whose stem is <= this value.')
    parser.add_argument('--keep-existing', action='store_true', help='Do not delete old segmented PNGs for frames being reprocessed.')
    return parser.parse_args()


def main():
    args = parse_args()

    if active_dataset == "strawberry":
        cropped_folders = sorted(
            [
            folder
            for folder in processed_root.iterdir()
            if folder.is_dir()
            and is_cropped_folder(folder.name)
            ]
        )
    else:
        cropped_folders = [Path(args.input_dir)]

    if not cropped_folders:
        print("No cropped folders found.")
        return

    cells_dict = AVOCADO_CELLS if active_dataset == "avocado" else STRAWBERRY_CELLS

    for current_input_dir in cropped_folders:

        if active_dataset == "strawberry":
            date_str = current_input_dir.name.replace("cropped_", "")
            current_output_dir = (processed_root / f"segmented_{date_str}")
            mask_output_dir = (processed_root / f"mask_{date_str}")
        else:
            date_str = active_dataset
            current_output_dir = (processed_root / f"segmented_{active_dataset}")
            mask_output_dir = (processed_root / f"mask_{active_dataset}")

        os.makedirs(current_output_dir, exist_ok=True)
        os.makedirs(mask_output_dir, exist_ok=True)

        temp_args = argparse.Namespace(**vars(args))
        temp_args.input_dir = str(current_input_dir)
        temp_args.output_dir = str(current_output_dir)

        print("\n" + "=" * 60)
        if active_dataset == "strawberry":
            print(f"Processing date: {date_str}")
        else:
            print(f"Processing dataset: {active_dataset}")
        print("=" * 60)

        # Load CSV report
        csv_path = processed_root / f"frame_differencing_results_{date_str}" / f"frame_differencing_report_{date_str}.csv"
        regenerate_mask_map = {}
        if csv_path.exists():
            import csv
            with open(csv_path, "r", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    frame_path = row.get("frame_path", "")
                    regen_str = row.get("regenerate_mask", "true")
                    if frame_path:
                        filename = os.path.basename(frame_path)
                        regenerate_mask_map[filename] = (regen_str.lower() == "true")
            print(f"Loaded frame differencing report from: {csv_path}")
        else:
            print(f"Warning: CSV report not found at {csv_path}. Defaulting to regenerate_mask = True for all frames.")

        image_paths = sorted(
            glob.glob(
                os.path.join(
                    temp_args.input_dir,
                    "*.jpg"
                )
            ),
            key=natural_sort_key
        )
        image_paths = [path for path in image_paths if should_process_image(path, temp_args)]

        if not image_paths:
            print(f'Cannot find any matching images in: {temp_args.input_dir}')
            continue

        print(f'Found {len(image_paths)} images to process.')
        if args.start_frame is not None or args.end_frame is not None or args.only_frame:
            print(f'Frame filter: start={args.start_frame}, end={args.end_frame}, only={args.only_frame}, skip={args.skip_frame}')
        print()

        for idx_frame, img_path in enumerate(image_paths):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            current_filename = os.path.basename(img_path)
            prev_img_path = image_paths[idx_frame - 1] if idx_frame > 0 else None

            if not args.keep_existing:
                object_label = active_dataset
                for old_output_path in glob.glob(os.path.join(temp_args.output_dir, f'{base_name}_{object_label}_*.png')):
                    try:
                        os.remove(old_output_path)
                    except OSError:
                        pass
                for old_mask_path in glob.glob(os.path.join(mask_output_dir, f'{base_name}_{object_label}_*_mask.png')):
                    try:
                        os.remove(old_mask_path)
                    except OSError:
                        pass

            img = cv2.imread(img_path)
            if img is None:
                print(f'Cannot read image: {img_path}')
                continue

            h, w = img.shape[:2]
            print(f'--- Processing image: {base_name} ({w}x{h}) ---')

            # Check if this frame needs mask regeneration
            regen_frame = regenerate_mask_map.get(current_filename, True)
            if idx_frame == 0 or prev_img_path is None:
                regen_frame = True

            print(f'    CSV regenerate_mask decision: {regen_frame}')

            segmented_count = 0

            if regen_frame:
                # Case 1: Chạy Candidate Mask -> GrabCut (segment) -> lưu mask mới
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                thresh = create_candidate_mask(hsv, dataset=active_dataset)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if active_dataset == "avocado":
                    contours = [cnt for cnt in contours if is_valid_avocado_contour(cnt, h, w)]
                else:
                    contours = [cnt for cnt in contours if is_valid_strawberry_contour(cnt, h, w)]
                contours = sorted(contours, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

                for cnt in contours:
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    fruit_idx = compute_grid_index(cX, cY, h, w, dataset=active_dataset)

                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    pad = 20
                    x_start = max(0, x - pad)
                    y_start = max(0, y - pad)
                    x_end = min(w, x + w_box + pad)
                    y_end = min(h, y + h_box + pad)

                    roi = img[y_start:y_end, x_start:x_end]
                    roi_color_support = thresh[y_start:y_end, x_start:x_end]

                    if active_dataset == "avocado":
                        mask = create_grabcut_mask_avocado(roi, roi_color_support)
                        grabcut_iters = 8
                    else:
                        mask = create_grabcut_mask(roi, roi_color_support)
                        grabcut_iters = 5

                    bgdModel = np.zeros((1, 65), np.float64)
                    fgdModel = np.zeros((1, 65), np.float64)

                    cv2.grabCut(roi, mask, None, bgdModel, fgdModel, grabcut_iters, cv2.GC_INIT_WITH_MASK)

                    mask_res = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                    if active_dataset == "avocado":
                        mask_res = refine_avocado_mask(mask_res, roi, roi_color_support)
                    else:
                        mask_res = refine_foreground_mask(mask_res, roi_color_support, roi)

                    if np.count_nonzero(mask_res) < max(200, int(0.0002 * h * w)):
                        continue

                    # Smooth mask border
                    clean_mask = mask_res * 255
                    blurred = cv2.GaussianBlur(clean_mask, (15, 15), 0)
                    final_mask = np.where(blurred > 127, 255, 0).astype(np.uint8)

                    # Save full-size mask
                    full_size_mask = np.zeros((h, w), dtype=np.uint8)
                    full_size_mask[y_start:y_end, x_start:x_end] = final_mask
                    
                    mask_filename = f'{base_name}_{object_label}_{fruit_idx}_mask.png'
                    mask_path = os.path.join(mask_output_dir, mask_filename)
                    cv2.imwrite(mask_path, full_size_mask)

                    # Save transparent fruit crop
                    b_channel, g_channel, r_channel = cv2.split(roi)
                    roi_transparent = cv2.merge([b_channel, g_channel, r_channel, final_mask])
                    
                    output_filename = f'{base_name}_{object_label}_{fruit_idx}.png'
                    output_path = os.path.join(temp_args.output_dir, output_filename)
                    cv2.imwrite(output_path, roi_transparent)

                    segmented_count += 1
            else:
                # Case 2: Chạy Local GrabCut (segment) -> lưu mask mới
                for idx in range(1, 7):
                    x1, y1, x2, y2 = cells_dict[idx]
                    roi = img[y1:y2, x1:x2]

                    # Preprocess for contrast
                    gw = apply_grayworld(roi)
                    preproc = apply_clahe_lab(gw)

                    roi_h, roi_w = roi.shape[:2]
                    mask = np.zeros((roi_h, roi_w), np.uint8)
                    bgdModel = np.zeros((1, 65), np.float64)
                    fgdModel = np.zeros((1, 65), np.float64)

                    # Rect margin inside compartment
                    margin = 25
                    rect = (margin, margin, roi_w - 2 * margin, roi_h - 2 * margin)

                    # Execute local GrabCut
                    cv2.grabCut(preproc, mask, rect, bgdModel, fgdModel, 6, cv2.GC_INIT_WITH_RECT)

                    # Get binary mask
                    binary_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')

                    # Clean mask: Keep largest component
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        clean_mask = np.zeros_like(binary_mask)
                        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                    else:
                        clean_mask = binary_mask.copy()

                    # Morphological clean
                    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_ellipse)
                    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_ellipse)

                    # Blur and threshold for smooth border
                    blurred = cv2.GaussianBlur(clean_mask, (15, 15), 0)
                    final_mask = np.where(blurred > 127, 255, 0).astype(np.uint8)

                    # Save full-size mask
                    full_size_mask = np.zeros((h, w), dtype=np.uint8)
                    full_size_mask[y1:y2, x1:x2] = final_mask
                    
                    mask_filename = f'{base_name}_{object_label}_{idx}_mask.png'
                    mask_path = os.path.join(mask_output_dir, mask_filename)
                    cv2.imwrite(mask_path, full_size_mask)

                    # Save transparent fruit crop
                    b_channel, g_channel, r_channel = cv2.split(roi)
                    roi_transparent = cv2.merge([b_channel, g_channel, r_channel, final_mask])
                    
                    output_filename = f'{base_name}_{object_label}_{idx}.png'
                    output_path = os.path.join(temp_args.output_dir, output_filename)
                    cv2.imwrite(output_path, roi_transparent)

                    segmented_count += 1

            print(f'-> Done {base_name}: processed {segmented_count} fruits <-\n')

        print('=' * 40)
        print(f'Done segmentation for {date_str}')
        print(f'Output folder: {current_output_dir}')


if __name__ == '__main__':
    main()