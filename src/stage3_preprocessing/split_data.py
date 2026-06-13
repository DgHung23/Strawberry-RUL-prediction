import argparse
import csv
import random
import shutil
from pathlib import Path


# Thu muc goc cua project: .../Strawberry-RUL-prediction
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# File labels tong dang co cac cot:
# image_path,date,strawberry_id,timestamp,rul_hours
DEFAULT_LABELS_CSV = PROJECT_ROOT / "data" / "labels.csv"

# Thu muc output mac dinh sau khi split:
# data/03_split/train, data/03_split/val, data/03_split/test
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "03_split"

# Ti le user yeu cau: 4 trai train, 1 trai validation, 1 trai test
DEFAULT_TRAIN_COUNT = 4
DEFAULT_VAL_COUNT = 1
DEFAULT_TEST_COUNT = 1


def read_labels(labels_csv):
    """Doc labels.csv thanh list dict de xu ly bang thu vien csv co san."""
    with labels_csv.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def group_rows_by_strawberry(rows):
    """Gom cac frame/anh theo strawberry_id de split theo trai, khong split lung tung theo anh."""
    groups = {}

    for row in rows:
        strawberry_id = row["strawberry_id"]

        if strawberry_id not in groups:
            groups[strawberry_id] = []

        groups[strawberry_id].append(row)

    return groups


def split_strawberry_ids(strawberry_ids, train_count, val_count, test_count, seed):
    """Shuffle co seed de moi lan chay deu ra cung mot ket qua split."""
    expected_total = train_count + val_count + test_count

    if len(strawberry_ids) != expected_total:
        raise ValueError(
            f"Can dung {expected_total} strawberry_id de split "
            f"({train_count} train, {val_count} val, {test_count} test), "
            f"nhung tim thay {len(strawberry_ids)} id: {strawberry_ids}"
        )

    shuffled_ids = list(strawberry_ids)
    random.Random(seed).shuffle(shuffled_ids)

    train_ids = shuffled_ids[:train_count]
    val_ids = shuffled_ids[train_count:train_count + val_count]
    test_ids = shuffled_ids[train_count + val_count:]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def copy_split_images(rows, split_name, data_dir, output_dir):
    """Copy anh vao folder split va ghi lai labels.csv rieng cho split do."""
    split_dir = output_dir / split_name
    image_output_dir = split_dir / "images"
    labels_output_csv = split_dir / "labels.csv"

    image_output_dir.mkdir(parents=True, exist_ok=True)

    copied_rows = []

    for row in rows:
        # image_path trong labels.csv la duong dan tuong doi tinh tu folder data.
        # Vi du: assigned_18-03-2026/strawberry_1/xxx.png
        relative_image_path = Path(row["image_path"])
        source_image_path = data_dir / relative_image_path

        if not source_image_path.exists():
            raise FileNotFoundError(f"Khong tim thay anh: {source_image_path}")

        # Giu lai cau truc folder assigned_.../strawberry_... trong tung split
        # de sau nay trace nguoc ve ngay chup va id trai dau de hon.
        target_image_path = image_output_dir / relative_image_path
        target_image_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_image_path, target_image_path)

        copied_row = dict(row)
        copied_row["image_path"] = str(
            Path("images") / relative_image_path
        ).replace("\\", "/")
        copied_rows.append(copied_row)

    with labels_output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(copied_rows)

    return len(copied_rows)


def write_split_summary(split_to_ids, split_to_count, output_dir):
    """Ghi file summary de biet split nao gom nhung strawberry_id nao."""
    summary_csv = output_dir / "split_summary.csv"

    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "strawberry_ids", "num_strawberries", "num_images"],
        )
        writer.writeheader()

        for split_name, strawberry_ids in split_to_ids.items():
            writer.writerow({
                "split": split_name,
                "strawberry_ids": " ".join(strawberry_ids),
                "num_strawberries": len(strawberry_ids),
                "num_images": split_to_count[split_name],
            })


def main():
    parser = argparse.ArgumentParser(
        description="Split strawberry dataset theo ti le 4 train, 1 val, 1 test."
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="Duong dan den file labels.csv tong.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thu muc output cho train/val/test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed de shuffle strawberry_id, giup split co the lap lai.",
    )
    args = parser.parse_args()

    labels_csv = args.labels_csv.resolve()
    data_dir = labels_csv.parent
    output_dir = args.output_dir.resolve()

    rows = read_labels(labels_csv)

    if not rows:
        raise ValueError(f"File labels rong: {labels_csv}")

    groups = group_rows_by_strawberry(rows)
    strawberry_ids = sorted(groups.keys(), key=int)

    split_to_ids = split_strawberry_ids(
        strawberry_ids=strawberry_ids,
        train_count=DEFAULT_TRAIN_COUNT,
        val_count=DEFAULT_VAL_COUNT,
        test_count=DEFAULT_TEST_COUNT,
        seed=args.seed,
    )

    split_to_count = {}

    for split_name, ids in split_to_ids.items():
        # Lay tat ca rows cua cac strawberry_id thuoc split hien tai.
        split_rows = []

        for strawberry_id in ids:
            split_rows.extend(groups[strawberry_id])

        # Sap xep lai labels de file output de doc: id trai dau -> thoi gian chup.
        split_rows = sorted(
            split_rows,
            key=lambda row: (int(row["strawberry_id"]), row["timestamp"]),
        )

        split_to_count[split_name] = copy_split_images(
            rows=split_rows,
            split_name=split_name,
            data_dir=data_dir,
            output_dir=output_dir,
        )

    write_split_summary(split_to_ids, split_to_count, output_dir)

    print("Split done")
    print(f"Labels input: {labels_csv}")
    print(f"Output dir: {output_dir}")

    for split_name, ids in split_to_ids.items():
        print(
            f"{split_name}: strawberry_id={ids}, "
            f"images={split_to_count[split_name]}"
        )


if __name__ == "__main__":
    main()
