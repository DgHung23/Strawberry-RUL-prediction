from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# check columns in frame_manifest.csv
FRAME_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "02_processed"
    / "manifests"
    / "frame_manifest.csv"
)

LABELS_CSV = (
    PROJECT_ROOT
    / "data"
    / "02_processed"
    / "manifests"
    / "labels.csv"
)

df = pd.read_csv(FRAME_MANIFEST)
df2 = pd.read_csv(LABELS_CSV)

print("Columns in frame_manifest.csv:")
print(df.columns.tolist())

print("Columns in labels.csv:")
print(df2.columns.tolist())