import cv2
import numpy as np
import os
import glob

# input folder containing raw images and output folder for segmented strawberries
input_dir = r'C:\Users\THANH CONG\Documents\Strawberry-RUL-prediction\data\01_raw\18-03-2026\cropped' # can edit with others folder
output_dir = 'segmented_18-03-2026'

#define wider color ranges for strawberry candidates in HSV color space
#Old/damaged strawberries can shift from bright red to orange/brown with lower brightness
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

def create_strawberry_candidate_mask(hsv):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in STRAWBERRY_COLOR_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    dark_or_saturated_object = ((saturation > 35) & (value > 18) & (value < 245)).astype("uint8") * 255
    dark_damaged_object = ((saturation > 15) & (value > 12) & (value < 110)).astype("uint8") * 255
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
    white_bg = (roi_hsv[:, :, 1] < 60) & (roi_hsv[:, :, 2] > 90) # edit remove background

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
    refined = mask_res & (color_support > 0).astype("uint8")

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    background_like = (roi_hsv[:, :, 1] < 70) & (roi_hsv[:, :, 2] > 70) # edit remove background
    refined[background_like] = 0

    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, small_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, 8)
    if num_labels <= 1:
        return refined

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype("uint8")


def apply_mask_to_roi(roi, mask_res):
    b_channel, g_channel, r_channel = cv2.split(roi)
    alpha_channel = mask_res * 255
    return cv2.merge([b_channel, g_channel, r_channel, alpha_channel])





def main():
    os.makedirs(output_dir, exist_ok=True)

    #find all images that has .jpg extension in the input directory (you can change this to match your image format)
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))

    if not image_paths:
        print(f"Cannot find any images in: {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process.\n")

    #read each image, segment strawberries, and save results
    for img_path in image_paths:
        #use base name of the image file (without extension) for naming output files
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        for old_output_path in glob.glob(os.path.join(output_dir, f"{base_name}_strawberry_*.png")):
            os.remove(old_output_path)
    
        #read image and check if it was loaded successfully
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image: {img_path}")
            continue
        
        h, w = img.shape[:2]
        print(f"--- Processing image, pls wait: {base_name} ({w}x{h}) ---")

        #convert image to HSV color space and create masks for strawberry-colored regions
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        thresh = create_strawberry_candidate_mask(hsv)

        #find contours of the thresholded image to get bounding boxes for potential strawberries
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if is_valid_strawberry_contour(cnt, h, w)]
        contours = sorted(contours, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

        strawberry_idx = 1

        for cnt in contours:
            #use bounding box to define ROI for GrabCut
            x, y, w_box, h_box = cv2.boundingRect(cnt)
        
            #expand bounding box
            pad = 20  # increase bounding box by 20 pixels on each side
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(w, x + w_box + pad)
            y_end = min(h, y + h_box + pad)
        
            #crop ROI from original image
            roi = img[y_start:y_end, x_start:x_end]
            roi_color_support = thresh[y_start:y_end, x_start:x_end]
        
            #GrabCut => mask => apply mask
            mask = create_grabcut_mask(roi, roi_color_support)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            #segmentation with GrabCut
            cv2.grabCut(roi, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
            #filter mask to keep only foreground (strawberry) pixels
            mask_res = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            mask_res = refine_foreground_mask(mask_res, roi_color_support, roi)
            if np.count_nonzero(mask_res) < max(200, int(0.0002 * h * w)):
                continue
        
            #apply mask and create transparent image with alpha channel
            strawberry_transparent = apply_mask_to_roi(roi, mask_res)
        
            # save images with name format: {base_name}_strawberry_{index}.png
            output_filename = f'{base_name}_strawberry_{strawberry_idx}.png'
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, strawberry_transparent)
        
            strawberry_idx += 1
        
        print(f"-> Done {base_name}: segmented {strawberry_idx - 1} strawberries <-\n")

    print("="*40)
    print("Done segmentation for all images")

if __name__ == "__main__":
    main()