import os
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "02_processed" / "segmented_21-03-2026"
OUTPUT_DIR = PROJECT_ROOT / "data" / "02_processed" / "assigned_21-03-2026"


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    assigned_count = 0

    for filename in os.listdir(INPUT_DIR):

        if not filename.lower().endswith(".png"):
            continue

        match = re.search(r"^(.*?)_strawberry_(\d+)\.png$", filename)

        if not match:
            print(f"Skip: {filename}")
            continue

        prefix = match.group(1)       # Ví dụ: "frame-1_12-26-28"
        strawberry_id = match.group(2) # Ví dụ: "1"

        # Chuyển đổi ID thành dạng 2 chữ số (1 -> 01, 2 -> 02,...)
        formatted_id = f"{int(strawberry_id):02d}"

        # Tạo tên file mới theo định dạng mong muốn: frame-1_12-26-28_F_01.png
        new_filename = f"{prefix}_F{formatted_id}.png"

        # Tạo thư mục phân loại riêng cho từng quả dâu (ví dụ: folder "strawberry_01")
        target_folder = os.path.join(
            OUTPUT_DIR,
            f"F{formatted_id}"
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
            new_filename
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