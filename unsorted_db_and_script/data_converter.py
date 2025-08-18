"""
Convert HistData ZIPs to a single yearly CSV with UTC and Berlin timestamps.

Expected folder structure:
.
├── histdata_processor.py   <-- this script
└── raw_zip/
    ├── HISTDATA_COM_MT_XAUUSD_M12023.zip
    ├── HISTDATA_COM_MT_XAUUSD_M12024.zip
    ├── HISTDATA_COM_MT_XAUUSD_M1202501.zip
    ...

Install dependencies:
    pip install pandas pytz tqdm
"""

import zipfile, shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import pytz
from tqdm import tqdm

BERLIN = pytz.timezone("Europe/Berlin")

RAW_DIR = Path("raw_zips")          # where ZIPs are stored
TMP_DIR = Path("tmp_extracted")    # temp unzip location
OUT_DIR = Path("clean_csv")        # output location
OUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

def process_year(year: int):
    zip_paths = sorted(RAW_DIR.glob(f"XAUUSD_ASCII_{year}_M*.zip"))
    if not zip_paths:
        print(f"❌ No ZIPs found for {year}. Skipping.")
        return

    all_frames = []
    for zp in tqdm(zip_paths, desc=f"Unzipping {year}"):
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(TMP_DIR)

    csv_files = sorted(TMP_DIR.glob("*.csv"))
    for cf in tqdm(csv_files, desc=f"Parsing {year}"):
        df = pd.read_csv(
            cf,
            sep=",",
            names=["date", "time", "open", "high", "low", "close", "volume"],
            dtype={
                "open": float, "high": float, "low": float,
                "close": float, "volume": float
            },
            engine="python"
        )
        df["raw_datetime"] = df["date"].astype(str) + "," + df["time"].astype(str)
        df["time_utc"] = pd.to_datetime(df["raw_datetime"], format="%Y.%m.%d,%H:%M", utc=True)
        df.drop(columns=["date", "time", "raw_datetime"], inplace=True)
        all_frames.append(df)

    full = pd.concat(all_frames).sort_values("time_utc").reset_index(drop=True)
    full["time_berlin"] = full["time_utc"].dt.tz_convert(BERLIN)
    full = full[["time_utc", "time_berlin", "open", "high", "low", "close", "volume"]]

    out_file = OUT_DIR / f"XAUUSD_M1_{year}.csv"
    full.to_csv(out_file, index=False)
    print(f"✅ Saved {len(full):,d} rows to {out_file}")

    shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir()

if __name__ == "__main__":
    for yr in [2015, 2016, 2017, 2018, 2025, 2026, 2027, 2028, 2029, 2030, 2031]:
        process_year(yr)
