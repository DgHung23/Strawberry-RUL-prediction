import pandas as pd
from pathlib import Path
import random
import numpy as np

def generate_fake_env_data():
    project_root = Path(__file__).resolve().parents[2]
    final_dir = project_root / "data" / "02_processed" / "final"
    
    # Collect all labels.csv paths
    label_files = list(final_dir.glob("F0*/labels.csv"))
    
    if not label_files:
        print("No labels.csv found in final directory.")
        return
        
    print(f"Found {len(label_files)} label files.")
    
    # Read all DataFrames
    dfs = {file: pd.read_csv(file) for file in label_files}
    
    # Extract unique timestamps across all files
    unique_timestamps = set()
    for df in dfs.values():
        unique_timestamps.update(df["timestamp"].tolist())
        
    print(f"Found {len(unique_timestamps)} unique timestamps across all fruits.")
    
    # Generate random temperature (~22) and humidity (~60) for each unique timestamp
    # We use a dict to store the mapping so that the same timestamp gets the same values
    env_mapping = {}
    for ts in unique_timestamps:
        # random value around 22 for temp, around 60 for humidity
        temp = round(random.uniform(21.0, 23.0), 1)
        humidity = round(random.uniform(55.0, 65.0), 1)
        env_mapping[ts] = {"temp": temp, "humidity": humidity}
        
    # Apply the mapping to each file and save
    for file, df in dfs.items():
        print(f"Updating {file} ...")
        
        # Map values
        df["temperature_c"] = df["timestamp"].map(lambda ts: env_mapping[ts]["temp"])
        df["humidity_pct"] = df["timestamp"].map(lambda ts: env_mapping[ts]["humidity"])
        
        # Save back to CSV
        df.to_csv(file, index=False)
        
    print("Fake temperature and humidity successfully written to all label files!")

if __name__ == "__main__":
    # Ensure reproducible randomness if needed, or leave it truly random
    random.seed(42)
    generate_fake_env_data()
