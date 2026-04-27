"""
dashboard.py - DemandCast Streamlit dashboard
=============================================
Loads the Production model from the MLflow Model Registry and serves an
interactive prediction UI for taxi-demand forecasting by zone and time.

Run from project root with the .venv active:
    streamlit run app/dashboard.py
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration - keep MLFLOW_TRACKING_URI in sync with src/tune.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_TRACKING_URI = "sqlite:///" + str(PROJECT_ROOT / "mlflow.db")
MODEL_NAME = "DemandCast"
MODEL_STAGE = "Production"
DATA_PATH = PROJECT_ROOT / "data" / "features.parquet"

# Copied from src/features.py — must match training order exactly
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
# Resource loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")


@st.cache_data
def load_history() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    return df


def lookup_lags(history: pd.DataFrame, zone: int, hour: int, day_of_week: int) -> dict:
    """Look up the most recent historical lag values for this (zone, hour, dow).

    Falls back to zone-level mean demand if no exact match exists.
    """
    match = history[
        (history["PULocationID"] == zone)
        & (history["hour"] == hour)
        & (history["day_of_week"] == day_of_week)
    ]
    if not match.empty:
        last = match.sort_values("pickup_datetime").iloc[-1]
        return {
            "demand_lag_1h": float(last["demand_lag_1h"]) if pd.notna(last["demand_lag_1h"]) else 0.0,
            "demand_lag_24h": float(last["demand_lag_24h"]) if pd.notna(last["demand_lag_24h"]) else 0.0,
            "demand_lag_168h": float(last["demand_lag_168h"]) if pd.notna(last["demand_lag_168h"]) else 0.0,
        }
    zone_data = history[history["PULocationID"] == zone]["demand"]
    fallback = float(zone_data.mean()) if not zone_data.empty else 0.0
    return {"demand_lag_1h": fallback, "demand_lag_24h": fallback, "demand_lag_168h": fallback}


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DemandCast", layout="wide")
st.title("DemandCast - NYC Taxi Demand Predictor")
st.caption(
    f"Live predictions from MLflow Model Registry: **{MODEL_NAME}** "
    f"({MODEL_STAGE}) - trained on January 2025 NYC TLC yellow cab data."
)

model = load_model()
history = load_history()


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

st.sidebar.header("Inputs")

zones = sorted(history["PULocationID"].unique().tolist())
default_zone_idx = zones.index(132) if 132 in zones else 0  # 132 = JFK Airport
zone = st.sidebar.selectbox("Pickup Zone (PULocationID)", zones, index=default_zone_idx)

hour = st.sidebar.slider("Hour of Day", 0, 23, 17)

day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_label = st.sidebar.selectbox("Day of Week", day_names, index=4)
day_of_week = day_names.index(day_label)

is_weekend = st.sidebar.toggle("Weekend", value=(day_of_week >= 5))


# ---------------------------------------------------------------------------
# Feature vector and prediction
# ---------------------------------------------------------------------------

is_rush_hour = int(hour in {7, 8, 17, 18} and day_of_week < 5)
month = int(history["pickup_datetime"].dt.month.mode().iloc[0])
lags = lookup_lags(history, zone, hour, day_of_week)

feature_row = {
    "hour": hour,
    "day_of_week": day_of_week,
    "month": month,
    "is_weekend": int(is_weekend),
    "is_rush_hour": is_rush_hour,
    "PULocationID": zone,
    **lags,
}
features = pd.DataFrame([feature_row])[FEATURE_COLS]
prediction = float(model.predict(features)[0])


# ---------------------------------------------------------------------------
# Main display
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        label=f"Predicted Pickups in Zone {zone}",
        value=f"{prediction:.1f} trips",
        help="Trips per hour for the selected zone, hour, and day of week.",
    )
    st.caption(f"Rush-hour flag: {'yes' if is_rush_hour else 'no'} - Weekend: {'yes' if is_weekend else 'no'}")

with col2:
    st.subheader("How to read this number")
    st.markdown(
        """
        **MAE = 8.2 trips/hour.** On average, the model's predictions for any
        given zone are off by about 8 trips per hour. If the dashboard says
        30 trips, reality is likely between 22 and 38 - close enough to size
        a driver shift, but with a buffer needed for tight windows.

        **R-squared = 0.95.** The model explains roughly 95% of the variation
        in hourly demand across zones, capturing rush-hour spikes, overnight
        lulls, and zone-level differences reliably.

        **Bias = +0.4 trips/hour.** The model leans slightly toward
        over-prediction - a safer failure mode for staffing than under-counting.
        """
    )


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Average Hourly Demand - All Zones")

hourly_avg = (
    history.groupby("hour")["demand"]
    .mean()
    .reset_index()
    .rename(columns={"demand": "Avg Pickups/hour"})
    .set_index("hour")
)
st.bar_chart(hourly_avg, height=320)
st.caption(
    "Mean demand per zone across the training window (Jan 7 - Feb 1, 2025). "
    "Notice the morning-rush ramp at 7-9am and the evening peak at 5-7pm."
)

st.divider()
st.subheader(f"Hourly Demand Profile - Zone {zone}")

zone_hourly = (
    history[history["PULocationID"] == zone]
    .groupby("hour")["demand"]
    .mean()
    .reset_index()
    .rename(columns={"demand": "Avg Pickups/hour"})
    .set_index("hour")
)
st.bar_chart(zone_hourly, height=320)

with st.expander("Feature vector sent to the model"):
    st.dataframe(features.T.rename(columns={0: "value"}))
