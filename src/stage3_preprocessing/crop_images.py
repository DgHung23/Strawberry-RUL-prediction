import cv2
from pathlib import Path


INPUT_DIR = Path(r"C:\Users\THANH CONG\Documents\Strawberry-RUL-prediction\data\01_raw\18-03-2026")
OUTPUT_DIR = INPUT_DIR / "cropped" # folder name

# the target you wanna crop image
TARGET_WIDTH = 1150
TARGET_HEIGHT = 930
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def center_crop(image, target_width, target_height):
    height, width = image.shape[:2]
    if width < target_width or height < target_height:
        raise ValueError(
            f"Image size {width}x{height} is smaller than target "
            f"{target_width}x{target_height}"
        )

    x_start = (width - target_width) // 2
    y_start = (height - target_height) // 2
    x_end = x_start + target_width
    y_end = y_start + target_height
    return image[y_start:y_end, x_start:x_end]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path for path in sorted(INPUT_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        print(f"No image files found in: {INPUT_DIR}")
        return

    cropped_count = 0
    skipped_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path.name}")
            skipped_count += 1
            continue

        try:
            cropped = center_crop(image, TARGET_WIDTH, TARGET_HEIGHT)
        except ValueError as error:
            print(f"Skip {image_path.name}: {error}")
            skipped_count += 1
            continue

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), cropped)
        cropped_count += 1
        print(f"Cropped {image_path.name} -> {output_path.name}")

    print("=" * 40)
    print(f"Done. Cropped: {cropped_count}, skipped: {skipped_count}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
