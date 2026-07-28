import sys
from pathlib import Path

# Ensure stage3_preprocessing directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crop_images import main as crop_images_main
from segmentation import main as segmentation_main
from frame_differencing import main as frame_differencing_main
from assign_id import main as assign_id_main
from eol import main as eol_main
from manifests import main as manifests_main
from consolidate_final import main as consolidate_final_main
from generate_final_labels import generate_final_labels as generate_final_labels_main
from fake_data import generate_fake_env_data as fake_data_main
from generate_metadata import generate_metadata as generate_metadata_main
from split_data import main as split_data_main

def run_step(name, func):

    print(f"\n=== {name} ===")

    try:
        func()
        print(f"[OK] {name}")

    except Exception as e:
        print(f"[FAILED] {name}")
        print(e)
        raise

def main():
    # Step 1: Crop images to focus on strawberries
    run_step("Crop Images", crop_images_main)
    
    # Step 2: Segment strawberries from the background
    run_step("Segmentation", segmentation_main)

    # Step 3: Perform frame differencing and validate masks
    run_step("Frame Differencing", frame_differencing_main)

    # Step 4: Assign unique IDs to each strawberry
    run_step("Assign IDs", assign_id_main)

    # Step 5: Generate end-of-life (EOL) anchors for strawberries
    run_step("Generate EOL Anchors", eol_main)

    # Step 6: Generate manifests for the dataset
    run_step("Generate Manifests", manifests_main)

    # Step 7: Consolidate final dataset and manifest
    run_step("Consolidate Final Dataset", consolidate_final_main)

    # Step 8: Label remaining useful life (RUL), time_gap, and elapsed_time
    run_step("Label RUL and Temporal Features", generate_final_labels_main)

    # Step 9: Generate fake temperature and humidity data
    run_step("Generate Environment Data (fake_data.py)", fake_data_main)

    # Step 10: Generate consolidated metadata.csv
    run_step("Generate Metadata CSV", generate_metadata_main)

    # Step 11: Split data into training, validation, and test sets
    run_step("Split Data", split_data_main)

    print("\n[SUCCESS] Preprocessing pipeline completed successfully!")

if __name__ == "__main__":
    main()
