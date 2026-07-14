from extracting_frames import main as extracting_frames_main
from crop_images import main as crop_images_main
from segmentation import main as segmentation_main
from frame_differencing import main as frame_differencing_main
from assign_id import main as assign_id_main
from eol import main as eol_main
from label_rul import main as label_rul_main
from manifests import main as manifests_main
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
    # Step 1: Extract frames from videos
    # run_step("Extract Frames", extracting_frames_main)

    # Step 1: Crop images to focus on strawberries
    run_step("Crop Images", crop_images_main)
    
    # Step 2: Perform frame differencing after masks exist for QC
    run_step("Frame Differencing", frame_differencing_main)

    # Step 3: Segment strawberries from the background
    run_step("Segmentation", segmentation_main)

    # Step 4: Assign unique IDs to each strawberry
    run_step("Assign IDs", assign_id_main)

    # Step 5: Generate end-of-life (EOL) anchors for strawberries
    run_step("Generate EOL Anchors", eol_main)

    # Step 6: Generate manifests for the dataset
    run_step("Generate Manifests", manifests_main)

    # Step 7: Label remaining useful life (RUL) for each strawberry
    run_step("Label RUL", label_rul_main)

    # Step 8: Split data into training, validation, and test sets
    run_step("Split Data", split_data_main)

    print("Preprocessing pipeline completed successfully")
    
if __name__ == "__main__":
    main()
