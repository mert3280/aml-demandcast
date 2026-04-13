"""
features.py — Feature engineering for DemandCast
=================================================
This module contains feature engineering logic for the NYC taxi demand
forecasting pipeline. It is imported by pipelines/build_features.py and
src/train.py.

Functions
---------
clean_data             Generic cleaning for raw trip-level DataFrame
create_temporal_features Add time-based features from the pickup datetime column
aggregate_to_hourly_demand Aggregate individual trips into hourly demand per zone
add_lag_features        Add lagged demand columns (1h, 24h, 168h) per zone

Constants
---------
FEATURE_COLS            Intentionally left empty for students to populate.
                        Keep this list in sync with train.py and dashboard.py
                        when you finalize feature choices.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Feature column contract (student exercise)
# ---------------------------------------------------------------------------
# IMPORTANT: Keep this list in sync with train.py and app/dashboard.py.
# Changing a name here without updating those files will break prediction.

FEATURE_COLS: list[str] = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "PULocationID",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
]


# ---------------------------------------------------------------------------
# 1. clean_data
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw trip-level rows before feature engineering.

    Use thresholds determined during EDA. The defaults below are reasonable
    starting points — override them if your EDA revealed different
    breakpoints for your data sample.

    Cleaning strategy (student exercise)
    -----------------------------------
    Implement the data cleaning strategies you determined during exploratory
    data analysis (EDA). Do not hard-code specific thresholds in this
    template; instead document and apply the rules you identified (for
    example: outlier detection, sensible missing-value handling, sensor-error
    filters, or domain-specific rules). Justify your choices in the
    accompanying notebook and use the methods you found appropriate for the
    dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trip-level DataFrame loaded from the parquet file.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame. Index is reset so it is contiguous after row drops.

    Examples
    --------
    >>> clean_df = clean_data(df)
    >>> print(f"Rows removed: {len(df) - len(clean_df)}")
    """
    # Thresholds from EDA (Section 4, 02_eda_skeleton.ipynb):
    #   trip_distance: (0, 20] — removes sensor zeros and implausible long trips
    #   fare_amount:   (0, 70] — removes erroneous/refunded fares and extreme outliers
    #   passenger_count: [1, 7] — removes empty-cab records and overloaded trips
    mask = (
        (df["trip_distance"] > 0) & (df["trip_distance"] <= 20) &
        (df["fare_amount"] > 0) & (df["fare_amount"] <= 70) &
        (df["passenger_count"] >= 1) & (df["passenger_count"] <= 7)
    )
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. create_temporal_features
# ---------------------------------------------------------------------------

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from the tpep_pickup_datetime column.

    All features are derived from a single source column so there is no risk
    of data leakage — we are only decomposing information already present at
    prediction time.

    New columns added
    -----------------
    pickup_hour : datetime64
        The pickup datetime floored to the nearest hour.
        Used as the groupby key in aggregate_to_hourly_demand().
    hour : int
        Hour of day (0–23).
    day_of_week : int
        Day of week (0 = Monday, 6 = Sunday). Use dt.dayofweek.
    is_weekend : int
        1 if day_of_week >= 5, else 0.
    month : int
        Month of year (1–12).
    is_rush_hour : int
        1 if (hour is 7, 8 OR hour is 17, 18) AND day_of_week < 5, else 0.
        Morning rush: 7–9am. Evening rush: 5–7pm. Weekdays only.

    Parameters
    ----------
    df : pd.DataFrame
        Trip-level DataFrame. Must contain column tpep_pickup_datetime.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new feature columns appended.

    Examples
    --------
    >>> df = create_temporal_features(df)
    >>> df[['hour', 'day_of_week', 'is_weekend', 'is_rush_hour']].head()
    """
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('H')
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['tpep_pickup_datetime'].dt.month
    df['is_rush_hour'] = (
        ((df['hour'].isin([7, 8, 17, 18])) & (df['day_of_week'] < 5)).astype(int)
    )
    return df


# ---------------------------------------------------------------------------
# 3. aggregate_to_hourly_demand
# ---------------------------------------------------------------------------

def aggregate_to_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual trips into hourly demand counts per pickup zone.

    This function performs the core transformation that converts the raw
    trip-level data (one row per trip) into the modeling target (one row per
    zone per hour, where the value is the number of pickups).

    Input shape  : (n_trips, many columns)  — e.g. 2.5M rows for January 2024
    Output shape : (n_zones × n_hours, 3)   — e.g. ~260 zones × 744 hours

    Output columns
    --------------
    PULocationID : int
        Pickup zone ID (1–265 in NYC TLC data).
    hour : datetime64
        The hour bucket (pickup_hour floored to the nearest hour).
    demand : int
        Number of taxi pickups in this zone during this hour.

    Parameters
    ----------
    df : pd.DataFrame
        Trip-level DataFrame after create_temporal_features() has been called.
        Must contain columns: PULocationID, pickup_hour.

    Returns
    -------
    pd.DataFrame
        Aggregated demand DataFrame with columns [PULocationID, hour, demand].

    Examples
    --------
    >>> hourly = aggregate_to_hourly_demand(df)
    >>> print(hourly.shape)   # expect (n_zones * n_hours, 3)
    >>> hourly.head()
    """
    return df.groupby(['PULocationID', 'pickup_hour']).size().reset_index(name='demand').rename(columns={'pickup_hour': 'hour'})


# ---------------------------------------------------------------------------
# 4. add_lag_features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    zone_col: str = "PULocationID",
    target_col: str = "demand",
) -> pd.DataFrame:
    """Add lagged demand features, computed separately for each zone.

    ⚠️  COMMON BUG WARNING ⚠️
    Lag features MUST be computed per zone using groupby. If you call
    df[target_col].shift(n) without groupby, you will bleed one zone's demand
    into the previous/next zone's lag column. This is a subtle data quality
    bug — the model will train without errors, but the features are wrong.

    Correct pattern:
        df[target_col].shift(n)                          ← WRONG
        df.groupby(zone_col)[target_col].shift(n)        ← CORRECT

    New columns added
    -----------------
    demand_lag_1h : float
        Demand for this zone 1 time-step ago (= 1 hour in the hourly table).
    demand_lag_24h : float
        Demand for this zone 24 time-steps ago (= same hour yesterday).
    demand_lag_168h : float
        Demand for this zone 168 time-steps ago (= same hour last week).

    Note: The first n rows for each zone will be NaN for a lag of n.
    Drop these rows after calling this function, or handle them in your
    training pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly demand DataFrame returned by aggregate_to_hourly_demand().
        Must be sorted by zone and hour before calling this function.
        Must contain columns: zone_col, target_col.
    zone_col : str, optional
        Name of the zone identifier column. Default: 'PULocationID'.
    target_col : str, optional
        Name of the demand column to lag. Default: 'demand'.

    Returns
    -------
    pd.DataFrame
        DataFrame with three new lag columns appended.

    Examples
    --------
    >>> hourly = hourly.sort_values(['PULocationID', 'hour'])
    >>> hourly = add_lag_features(hourly, zone_col='PULocationID', target_col='demand')
    >>> hourly[['PULocationID', 'hour', 'demand', 'demand_lag_1h']].head(10)
    """
    # Use groupby(zone_col) before .shift() to avoid bleeding demand across zones
    df['demand_lag_1h'] = df.groupby(zone_col)[target_col].shift(1)
    df['demand_lag_24h'] = df.groupby(zone_col)[target_col].shift(24)
    df['demand_lag_168h'] = df.groupby(zone_col)[target_col].shift(168)
    return df
