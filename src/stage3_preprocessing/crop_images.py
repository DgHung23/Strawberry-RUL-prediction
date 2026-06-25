import cv2
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]


RAW_INPUT_DIR = PROJECT_ROOT / "data" / "01_raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "02_processed"
OUTPUT_DIR = PROCESSED_DIR

# the target you wanna crop image
TARGET_WIDTH = 1116
TARGET_HEIGHT = 930
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def is_date_folder(folder_name):
    return bool(
        re.match(r"^\d{2}-\d{2}-\d{4}$", folder_name)
    )


def is_frames_folder(folder_name):
    return bool(
        re.match(r"^frames_\d{2}-\d{2}-\d{4}$", folder_name)
    )


def collect_input_folders():
    input_by_date = {}

    if RAW_INPUT_DIR.exists():
        for folder in RAW_INPUT_DIR.iterdir():
            if folder.is_dir() and is_date_folder(folder.name):
                input_by_date.setdefault(folder.name, folder)

    if PROCESSED_DIR.exists():
        for folder in PROCESSED_DIR.iterdir():
            if folder.is_dir() and is_frames_folder(folder.name):
                date_str = folder.name.replace("frames_", "")
                input_by_date[date_str] = folder

    return sorted(input_by_date.items())


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

    
    date_folders = collect_input_folders()
    

    if not date_folders:
        print(
            "No frame folders found. Expected "
            f"{PROCESSED_DIR / 'frames_DD-MM-YYYY'} or legacy "
            f"{RAW_INPUT_DIR / 'DD-MM-YYYY'} folders."
        )
        return

    total_cropped = 0
    total_skipped = 0

    
    for date_str, input_dir in date_folders:
   

        
        output_dir = OUTPUT_DIR / f"cropped_{date_str}"

        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 50)
        print(f"Processing folder: {date_str}")
        print(f"Input: {input_dir}")
        print("=" * 50)
        

        image_paths = [
            path
            for path in sorted(input_dir.iterdir())
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_paths:
            print(f"No image files found in: {input_dir}")
            continue

        cropped_count = 0
        skipped_count = 0

        for image_path in image_paths:

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skip unreadable image: {image_path.name}")
                skipped_count += 1
                continue

            try:
                cropped = center_crop(
                    image,
                    TARGET_WIDTH,
                    TARGET_HEIGHT
                )

            except ValueError as error:
                print(f"Skip {image_path.name}: {error}")
                skipped_count += 1
                continue

            output_path = output_dir / image_path.name

            cv2.imwrite(str(output_path), cropped)

            cropped_count += 1

            print(
                f"Cropped {image_path.name} -> {output_path.name}"
            )

        print("-" * 40)
        print(
            f"Folder {date_str}: "
            f"Cropped={cropped_count}, "
            f"Skipped={skipped_count}"
        )

        total_cropped += cropped_count
        total_skipped += skipped_count

    print("\n" + "=" * 50)
    print(
        f"Done. Total cropped: {total_cropped}, "
        f"total skipped: {total_skipped}"
    )
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
