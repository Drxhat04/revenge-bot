# scanner.py

import pandas as pd
from datetime import datetime
from pathlib import Path

from config import CONFIG
from zone_scanner import scan as run_scan, load_m1 as raw_load_m1


def load_today_m1(folder: str = "data") -> pd.DataFrame:
    """
    Loads today's M1 data from CSV and filters for UTC date.
    """
    df = raw_load_m1(folder)
    today = datetime.utcnow().date()
    return df[df.time.dt.date == today]


def get_today_zones() -> pd.DataFrame:
    """
    Runs scan and returns today's signals with entry_mid and zone_width.
    Applies optional post-gap filter.
    """
    m1 = load_today_m1()
    zones = run_scan(m1)

    if zones.empty:
        return zones

    zones["entry_mid"] = (zones.entry_low + zones.entry_high) / 2
    zones["zone_width"] = (zones.entry_high - zones.entry_low).abs()

    # Optional gap filter
    filtered = []
    for _, row in zones.iterrows():
        match = m1[m1.time == row.datetime]
        if match.empty:
            filtered.append(row)
            continue
        prev_idx = m1.index.get_loc(match.index[0]) - 1
        if prev_idx >= 0:
            prev_close = m1.iloc[prev_idx].close
            bar_open = match.open.values[0]
            gap = abs(bar_open - prev_close) / prev_close
            if gap <= CONFIG["gap_filter"]:
                filtered.append(row)
        else:
            filtered.append(row)

    return pd.DataFrame(filtered).sort_values("datetime").reset_index(drop=True)
