import os
import re
import pandas as pd
from datetime import datetime

# read dataset
DATASET_ROOT = r"C:\Users\THANH CONG\Documents\Strawberry-RUL-prediction\data\02_processed"

# EOL of strawberry
EOL_TIME = datetime(
    2026, 3, 26,
    8, 0, 0
)

# create label rul
rows = []

for day_folder in os.listdir(DATASET_ROOT):

    day_path = os.path.join(
        DATASET_ROOT,
        day_folder
    )

    if not os.path.isdir(day_path):
        continue

    if not day_folder.startswith("assigned_"):
        continue

    # assigned_18-03-2026 -> 18-03-2026
    date_str = day_folder.replace(
        "assigned_",
        ""
    )

    try:
        current_date = datetime.strptime(
            date_str,
            "%d-%m-%Y"
        )
    except:
        continue

    print(f"Processing {day_folder}")

    # strawberry folders
    for straw_folder in os.listdir(day_path):

        straw_path = os.path.join(
            day_path,
            straw_folder
        )

        if not os.path.isdir(straw_path):
            continue

        # strawberry_1 -> 1
        try:
            strawberry_id = int(
                straw_folder.split("_")[1]
            )
        except:
            continue

    
        # image files
        for filename in sorted(os.listdir(straw_path)):

            if not filename.lower().endswith(".png"):
                continue

            # example: frame-1_12-26-28_strawberry_1.png
            match = re.search(
                r"frame-\d+_(\d+)-(\d+)-(\d+)_strawberry_(\d+)",
                filename
            )

            if not match:
                print("Skip: ", filename)
                continue

            hour = int(match.group(1))
            minute = int(match.group(2))
            second = int(match.group(3))

            current_timestamp = datetime(
                current_date.year,
                current_date.month,
                current_date.day,
                hour,
                minute,
                second
            )

            rul_hours = (
                EOL_TIME - current_timestamp
            ).total_seconds() / 3600

            image_path = os.path.join(
                day_folder,
                straw_folder,
                filename
            ).replace("\\", "/")

            rows.append({
                "image_path": image_path,
                "date": date_str,
                "strawberry_id": strawberry_id,
                "timestamp": current_timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "rul_hours": round(
                    rul_hours,
                    2
                )
            })

# ==================================================
# SAVE CSV
# ==================================================

df = pd.DataFrame(rows)

df = df.sort_values(
    by=[
        "strawberry_id",
        "timestamp"
    ]
)

output_csv = os.path.join(
    DATASET_ROOT,
    "labels.csv"
)

df.to_csv(
    output_csv,
    index=False
)

print(f"\nSaved: {output_csv}")
print(f"Total samples: {len(df)}")
print(df.head())