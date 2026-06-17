import cv2
import os
from datetime import datetime, timedelta
import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor

cv2.setNumThreads(1)

video_folder = r"C:\Users\THANH CONG\Documents\extract_frame\2026-03-18" # can change to 19-03-2026 or 21-03-2026 bla bla... depending on the date of the video
script_dir = os.path.dirname(os.path.abspath(__file__))
date_part = "18-03-2026" # create folder with the name of the date part to save frames, can change to 19-03-2026 or 21-03-2026 bla bla... depending on the date of the video
sample_minutes = 15 # 5 minutes of sampling, can adjust if want to save more or less frames per second bla bla...

output_folder = os.path.join(script_dir, date_part)
os.makedirs(output_folder, exist_ok=True)

csv_path = os.path.join(output_folder, "all_frames.csv")
save_workers = max(1, min(8, os.cpu_count() or 1)) # use between 1 and 8 threads for saving frames, can adjust based on performance needs and available CPU cores, for example if your CPU has 4 cores, you can set to 4 or 8 for better performance, but if you have only 2 cores, setting to 8 may cause more overhead than benefit, so adjust accordingly
max_pending_writes = save_workers * 4 # limit the number of pending writes to avoid memory issues, can adjust based on available memory and performance needs


def save_frame(filename, frame):
    return cv2.imwrite(filename, frame)

# sort video files by the time part in their names
video_files = sorted(
    [f for f in os.listdir(video_folder) if f.lower().endswith(".mp4")],
    key=lambda x: x.split("_")[1].replace(".mp4", "")
)

# create CSV 
with open(csv_path, "w", newline="") as f, ThreadPoolExecutor(max_workers=save_workers) as executor:
    writer = csv.writer(f)
    writer.writerow(["video_name", "frame_number", "timestamp", "file_directory"])

    pending_writes = deque()
    frame_index = 1

    # loop through videos following the sorted order
    for video_file in video_files:

        print(f"\nProcessing: {video_file}")

        video_path = os.path.join(video_folder, video_file)

        cap = cv2.VideoCapture(video_path)

        # skip video if cannot open
        if not cap.isOpened():
            print("[SKIP] Cannot open video:", video_file)
            continue

        # collect start time from filename
        try:
            time_part = video_file.split("_")[1].replace(".mp4", "")
            start_time = datetime.strptime(time_part, "%H-%M-%S")
        except Exception:
            print("[SKIP] Wrong filename:", video_file)
            cap.release()
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            print("[SKIP] FPS = 0:", video_file)
            cap.release()
            continue

        interval = max(1, int(fps * 60 * sample_minutes)) # 1 frame every 5 minutes

        frame_count = 0

        # extract frames
        while True:

            if frame_count % interval == 0:
                ret, frame = cap.read()
            else:
                ret = cap.grab()
                frame = None

            if not ret:
                break

            if frame_count % interval == 0:

                current_time = start_time + timedelta(seconds=frame_count / fps)
                timestamp = current_time.strftime("%H-%M-%S")

                # KHONG chua ten video
                filename = os.path.join(
                    output_folder,
                    f"frame-{frame_index}_{timestamp}.jpg"
                )

                future = executor.submit(save_frame, filename, frame.copy())
                row = [video_file, frame_index, timestamp, filename]
                pending_writes.append((future, row))
                frame_index += 1

                if len(pending_writes) >= max_pending_writes:
                    done_future, done_row = pending_writes.popleft()
                    if done_future.result():
                        writer.writerow(done_row)
                    else:
                        print("[SKIP] Cannot save frame:", done_row[3])

            frame_count += 1

        cap.release()

    while pending_writes:
        done_future, done_row = pending_writes.popleft()
        if done_future.result():
            writer.writerow(done_row)
        else:
            print("[SKIP] Cannot save frame:", done_row[3])

print("\nDone extracting frames. CSV saved at:", csv_path)
