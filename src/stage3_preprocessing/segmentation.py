import argparse
import cv2
import glob
import os
import re

import numpy as np


# Default input folder containing cropped raw images and output folder for segmented strawberries.
# You can still edit these defaults, or pass --input-dir / --output-dir from the command line.
input_dir = r'C:\Users\THANH CONG\Documents\Strawberry-RUL-prediction\data\01_raw\18-03-2026\cropped'
output_dir = 'segmented_18-03-2026'

# Define wider color ranges for strawberry candidates in HSV color space.
# Old/damaged strawberries can shift from bright red to orange/brown with lower brightness.
STRAWBERRY_COLOR_RANGES = [
    (np.array([0, 25, 18]), np.array([25, 255, 255])),    # red/dark red/orange
    (np.array([160, 25, 18]), np.array([180, 255, 255])), # wrapped red
    (np.array([5, 20, 15]), np.array([45, 255, 245])),    # brown/damaged fruit
    (np.array([35, 25, 15]), np.array([100, 255, 245])),  # green calyx/leaves
]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grabcut_outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))


def natural_sort_key(path):
    """Sort frame filenames naturally, so frame-2 comes before frame-10."""

    stem = os.path.splitext(os.path.basename(path))[0].lower()
    parts = re.split(r'(\d+)', stem)
    return [int(part) if part.isdigit() else part for part in parts]


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

    if args.only_frame and frame_number not in args.only_frame:
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


def create_strawberry_candidate_mask(hsv):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in STRAWBERRY_COLOR_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dark_or_saturated_object = ((saturation > 35) & (value > 18) & (value < 245)).astype('uint8') * 255
    dark_damaged_object = ((saturation > 15) & (value > 12) & (value < 110)).astype('uint8') * 255
    mask = cv2.bitwise_or(mask, dark_or_saturated_object)
    mask = cv2.bitwise_or(mask, dark_damaged_object)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


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
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg')), key=natural_sort_key)
    image_paths = [path for path in image_paths if should_process_image(path, args)]

    if not image_paths:
        print(f'Cannot find any matching images in: {args.input_dir}')
        return

    print(f'Found {len(image_paths)} images to process.')
    if args.start_frame is not None or args.end_frame is not None or args.only_frame:
        print(f'Frame filter: start={args.start_frame}, end={args.end_frame}, only={args.only_frame}, skip={args.skip_frame}')
    print()

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        if not args.keep_existing:
            for old_output_path in glob.glob(os.path.join(args.output_dir, f'{base_name}_strawberry_*.png')):
                os.remove(old_output_path)

        img = cv2.imread(img_path)
        if img is None:
            print(f'Cannot read image: {img_path}')
            continue

        h, w = img.shape[:2]
        print(f'--- Processing image, pls wait: {base_name} ({w}x{h}) ---')

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        thresh = create_strawberry_candidate_mask(hsv)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if is_valid_strawberry_contour(cnt, h, w)]
        contours = sorted(contours, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

        strawberry_idx = 1

        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)

            pad = 20
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(w, x + w_box + pad)
            y_end = min(h, y + h_box + pad)

            roi = img[y_start:y_end, x_start:x_end]
            roi_color_support = thresh[y_start:y_end, x_start:x_end]

            mask = create_grabcut_mask(roi, roi_color_support)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            cv2.grabCut(roi, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

            mask_res = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            mask_res = refine_foreground_mask(mask_res, roi_color_support, roi)
            if np.count_nonzero(mask_res) < max(200, int(0.0002 * h * w)):
                continue

            strawberry_transparent = apply_mask_to_roi(roi, mask_res)

            output_filename = f'{base_name}_strawberry_{strawberry_idx}.png'
            output_path = os.path.join(args.output_dir, output_filename)
            cv2.imwrite(output_path, strawberry_transparent)

            strawberry_idx += 1

        print(f'-> Done {base_name}: segmented {strawberry_idx - 1} strawberries <-\n')

    print('=' * 40)
    print('Done segmentation for selected images')
    print(f'Output folder: {args.output_dir}')


if __name__ == '__main__':
    main()
