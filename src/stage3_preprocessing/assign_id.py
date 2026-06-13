import os
import re
import shutil

INPUT_DIR = r"C:\Users\THANH CONG\Documents\Strawberry-RUL-prediction\data\02_processed\segmented_18-03-2026"
OUTPUT_DIR = r"C:\Users\THANH CONG\Documents\assigned_18-03-2026_abcxyz"


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    assigned_count = 0

    for filename in os.listdir(INPUT_DIR):

        if not filename.lower().endswith(".png"):
            continue

        match = re.search(
            r"strawberry_(\d+)\.png$",
            filename
        )

        if not match:
            print(f"Skip: {filename}")
            continue

        strawberry_id = match.group(1)

        target_folder = os.path.join(
            OUTPUT_DIR,
            f"strawberry_{strawberry_id}"
        )

        os.makedirs(
            target_folder,
            exist_ok=True
        )

        src_path = os.path.join(
            INPUT_DIR,
            filename
        )

        dst_path = os.path.join(
            target_folder,
            filename
        )

        shutil.copy2(
            src_path,
            dst_path
        )

        assigned_count += 1

    print("=" * 40)
    print(f"Assigned {assigned_count} images")
    print("Done")


if __name__ == "__main__":
    main()