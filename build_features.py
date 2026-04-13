"""
build_features.py — Pipeline script for DemandCast feature engineering
=======================================================================
Loads the raw NYC taxi parquet, applies the feature engineering steps
defined in src/features.py, and writes the output feature matrix to
data/features.parquet.

Usage
-----
    python build_features.py

Output
------
    data/features.parquet   — rows: one per zone per hour (NaN rows dropped)
                            — columns: PULocationID, pickup_datetime, demand,
                              hour, day_of_week, month, is_weekend,
                              is_rush_hour, demand_lag_1h, demand_lag_24h,
                              demand_lag_168h
"""

from pathlib import Path
import pandas as pd
import sys

# Allow imports from src/ when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from src.features import (
    clean_data,
    create_temporal_features,
    aggregate_to_hourly_demand,
    add_lag_features,
    FEATURE_COLS,
)

DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "yellow_tripdata_2025-01.parquet"
OUTPUT_PATH = DATA_DIR / "features.parquet"


def main() -> None:
    # 1. Load raw data
    print(f"Loading {RAW_PATH} ...")
    df = pd.read_parquet(RAW_PATH)
    print(f"  Raw shape: {df.shape}")

    # 2. Clean — apply EDA-determined thresholds (trip_distance, fare_amount, passenger_count)
    df = clean_data(df)
    print(f"  After cleaning: {df.shape}")

    # 3. Temporal features — adds pickup_hour (datetime), hour (int), day_of_week,
    #    is_weekend, month, is_rush_hour to the trip-level DataFrame
    df = create_temporal_features(df)
    print(f"  After temporal features: {df.shape}")

    # 4. Aggregate trips → hourly demand per zone
    #    Output columns: [PULocationID, pickup_datetime, demand]
    #    Note: trip-level temporal integer columns are lost during this groupby
    hourly = aggregate_to_hourly_demand(df)
    print(f"  After aggregation: {hourly.shape}")

    # 4b. Re-derive integer temporal features from pickup_datetime
    #     These must be added back to the hourly DataFrame for modeling
    dt = hourly["pickup_datetime"]
    hourly["hour"] = dt.dt.hour
    hourly["day_of_week"] = dt.dt.dayofweek
    hourly["month"] = dt.dt.month
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)
    hourly["is_rush_hour"] = (
        (hourly["hour"].isin([7, 8, 17, 18])) & (hourly["day_of_week"] < 5)
    ).astype(int)

    # 5. Sort by zone then time — required precondition for add_lag_features
    hourly = hourly.sort_values(["PULocationID", "pickup_datetime"]).reset_index(drop=True)

    # 6. Add lag features (per-zone groupby shift — avoids cross-zone bleeding)
    hourly = add_lag_features(hourly, zone_col="PULocationID", target_col="demand")
    print(f"  After lag features: {hourly.shape}")

    # 7. Drop NaN rows from lag initialization (first 168 rows per zone have no lag_168h)
    before = len(hourly)
    hourly = hourly.dropna(
        subset=["demand_lag_1h", "demand_lag_24h", "demand_lag_168h"]
    ).reset_index(drop=True)
    print(f"  Dropped {before - len(hourly)} NaN-lag rows. Final shape: {hourly.shape}")

    # 8. Verify all FEATURE_COLS are present before saving
    missing = [c for c in FEATURE_COLS if c not in hourly.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    # 9. Save
    hourly.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Columns: {hourly.columns.tolist()}")
    print(f"Shape:   {hourly.shape}")


if __name__ == "__main__":
    main()
