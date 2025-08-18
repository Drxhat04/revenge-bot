import os
import pandas as pd

# Set the directory where the .parquet files are located
directory = os.getcwd()  

# Create a folder to store CSVs
csv_dir = os.path.join(directory, "csv_exports")
os.makedirs(csv_dir, exist_ok=True)

# Loop through all .parquet files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".parquet"):
        file_path = os.path.join(directory, filename)
        try:
            df = pd.read_parquet(file_path, engine="pyarrow")
            csv_filename = filename.replace(".parquet", ".csv")
            csv_path = os.path.join(csv_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
