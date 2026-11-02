"""
Step 1 - Data collection & reduction (run once, ~5-10 min).

The raw Backblaze Q1-2026 data is 90 daily CSV files (~340k drives/day, ~11 GB),
too large to load in memory. This script reduces it to three small files:

  data/daily_counts.csv   - date, active drives, failures         (time-series part)
  data/model_counts.csv   - per drive model: drive-days, failures (Poisson / NB part)
  data/drive_days.parquet - SMART readings for every failed drive + a 3% random
                            sample of healthy drives               (classification part)

Usage:  .venv/bin/python prepare_data.py
"""
import glob
import hashlib
import os

import pandas as pd
import pyarrow.csv as pv

RAW_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
HEALTHY_SAMPLE_PCT = 3  # % of never-failed drives to keep

# SMART attributes most often linked to drive failure (Backblaze / literature)
SMART = {
    "smart_5_raw":   "reallocated_sectors",
    "smart_187_raw": "reported_uncorrectable",
    "smart_188_raw": "command_timeout",
    "smart_197_raw": "pending_sectors",
    "smart_198_raw": "offline_uncorrectable",
    "smart_199_raw": "crc_errors",
    "smart_9_raw":   "power_on_hours",
    "smart_12_raw":  "power_cycles",
    "smart_193_raw": "load_cycles",
    "smart_194_raw": "temperature_c",
}
BASE = ["date", "serial_number", "model", "capacity_bytes", "failure"]


def read_day(path, cols):
    opts = pv.ConvertOptions(include_columns=cols)
    return pv.read_csv(path, convert_options=opts).to_pandas()


def in_sample(serial):
    return int(hashlib.md5(serial.encode()).hexdigest(), 16) % 100 < HEALTHY_SAMPLE_PCT


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "2026-*.csv")))
    print(f"{len(files)} daily files")

    # ---- Pass 1: daily totals, per-model exposure, list of failed drives ----
    daily, model_parts, failed = [], [], []
    for f in files:
        d = read_day(f, BASE)
        daily.append({"date": d["date"].iloc[0], "drives": len(d),
                      "failures": int(d["failure"].sum())})
        model_parts.append(d.groupby("model")["failure"].agg(["size", "sum"]))
        failed.append(d.loc[d["failure"] == 1, ["serial_number", "date"]])
        print("pass 1", os.path.basename(f), end="\r")

    pd.DataFrame(daily).to_csv(os.path.join(OUT_DIR, "daily_counts.csv"), index=False)

    mc = pd.concat(model_parts).groupby(level=0).sum()
    mc.columns = ["drive_days", "failures"]
    mc.to_csv(os.path.join(OUT_DIR, "model_counts.csv"))

    failed = pd.concat(failed).drop_duplicates("serial_number")
    failed.columns = ["serial_number", "failure_date"]
    failed_set = set(failed["serial_number"])
    print(f"\n{len(failed_set)} drives failed in the quarter")

    # ---- Pass 2: SMART history of failed drives + sample of healthy drives ----
    parts = []
    for f in files:
        d = read_day(f, BASE + list(SMART))
        keep = d["serial_number"].isin(failed_set) | d["serial_number"].map(in_sample)
        parts.append(d[keep])
        print("pass 2", os.path.basename(f), end="\r")

    df = pd.concat(parts, ignore_index=True).rename(columns=SMART)
    df = df.merge(failed, on="serial_number", how="left")
    df.to_parquet(os.path.join(OUT_DIR, "drive_days.parquet"), index=False)
    print(f"\nsaved {len(df):,} drive-day rows for "
          f"{df['serial_number'].nunique():,} drives")


if __name__ == "__main__":
    main()
