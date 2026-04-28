"""
dashboard.py - DemandCast Streamlit dashboard
=============================================
Loads the Production model from the MLflow Model Registry and serves a
weather-style 7-day forecast of taxi demand by pickup zone.

Run from project root with the .venv active:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path so `src` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from zoneinfo import ZoneInfo

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import streamlit as st

from src.features import FEATURE_COLS


# ---------------------------------------------------------------------------
# Configuration - keep MLFLOW_TRACKING_URI in sync with src/tune.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_TRACKING_URI = "sqlite:///" + str(PROJECT_ROOT / "mlflow.db")
MODEL_NAME = "DemandCast"
MODEL_STAGE = "Production"
DATA_PATH = PROJECT_ROOT / "data" / "features.parquet"
FORECAST_DAYS = 7
FORECAST_HOURS = 24 * FORECAST_DAYS
NY_TZ = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Resource loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_and_metrics():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    val_mae = None
    if versions:
        run = client.get_run(versions[0].run_id)
        m = run.data.metrics
        val_mae = m.get("val_mae") or m.get("test_mae")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    return model, val_mae


@st.cache_data
def load_history() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    return df.sort_values(["PULocationID", "pickup_datetime"]).reset_index(drop=True)


def _local_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY_TZ).tz_localize(None)


def _label_for_day(day: pd.Timestamp) -> str:
    return f"{day.strftime('%a %b')} {day.day}"


def _build_seed_series(history: pd.DataFrame, zone: int) -> pd.Series:
    zone_history = history.loc[history["PULocationID"] == zone, ["pickup_datetime", "demand"]].copy()
    if zone_history.empty:
        return pd.Series(dtype=float)

    zone_history["pickup_datetime"] = pd.to_datetime(zone_history["pickup_datetime"])
    zone_history = zone_history.drop_duplicates(subset=["pickup_datetime"], keep="last")
    zone_history = zone_history.sort_values("pickup_datetime").set_index("pickup_datetime")["demand"].astype(float)
    full_index = pd.date_range(zone_history.index.min(), zone_history.index.max(), freq="h")
    return zone_history.reindex(full_index).fillna(0.0)


def build_daily_forecast(model, history: pd.DataFrame, zone: int, val_mae: float | None = None) -> pd.DataFrame:
    """Generate a 7-day forecast by rolling hourly predictions forward."""

    seed_series = _build_seed_series(history, zone)
    global_mean = float(history["demand"].mean()) if not history.empty else 0.0
    zone_mean = float(seed_series.mean()) if not seed_series.empty else global_mean
    fallback = zone_mean if np.isfinite(zone_mean) else global_mean
    if not np.isfinite(fallback):
        fallback = 0.0

    if seed_series.empty:
        seed_index = pd.date_range(_local_now() - pd.Timedelta(hours=167), periods=168, freq="h")
        seed_series = pd.Series([fallback] * 168, index=seed_index)
    elif len(seed_series) < 168:
        pad_length = 168 - len(seed_series)
        pad_start = seed_series.index.min() - pd.Timedelta(hours=pad_length)
        pad_index = pd.date_range(pad_start, periods=pad_length, freq="h")
        pad_series = pd.Series([fallback] * pad_length, index=pad_index)
        seed_series = pd.concat([pad_series, seed_series]).sort_index()
    else:
        seed_series = seed_series.iloc[-168:]

    rolling_history = seed_series.astype(float).tolist()
    forecast_start = (_local_now() + pd.Timedelta(days=1)).normalize()
    forecast_index = pd.date_range(forecast_start, periods=FORECAST_HOURS, freq="h")

    hourly_rows: list[dict[str, float | int | pd.Timestamp]] = []
    for timestamp in forecast_index:
        lag_1h = rolling_history[-1] if rolling_history else fallback
        lag_24h = rolling_history[-24] if len(rolling_history) >= 24 else fallback
        lag_168h = rolling_history[-168] if len(rolling_history) >= 168 else fallback
        hour = int(timestamp.hour)
        day_of_week = int(timestamp.dayofweek)

        feature_row = pd.DataFrame(
            [
                {
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "month": int(timestamp.month),
                    "is_weekend": int(day_of_week >= 5),
                    "is_rush_hour": int(hour in {7, 8, 17, 18} and day_of_week < 5),
                    "PULocationID": zone,
                    "demand_lag_1h": lag_1h,
                    "demand_lag_24h": lag_24h,
                    "demand_lag_168h": lag_168h,
                }
            ]
        )[FEATURE_COLS]

        prediction = float(model.predict(feature_row)[0])
        prediction = max(prediction, 0.0)
        hourly_rows.append({"pickup_datetime": timestamp, "predicted_demand": prediction})
        rolling_history.append(prediction)

    hourly_forecast = pd.DataFrame(hourly_rows)
    hourly_forecast["forecast_date"] = hourly_forecast["pickup_datetime"].dt.normalize()

    daily_forecast = (
        hourly_forecast.groupby("forecast_date", as_index=False)["predicted_demand"]
        .sum()
        .rename(columns={"predicted_demand": "daily_total"})
    )
    daily_forecast["label"] = daily_forecast["forecast_date"].map(_label_for_day)

    # Busy-level: compare predicted daily total to historical avg for same zone + day-of-week
    zone_hist = history[history["PULocationID"] == zone].copy()
    zone_hist["date"] = zone_hist["pickup_datetime"].dt.normalize()
    zone_hist["dow"] = zone_hist["pickup_datetime"].dt.dayofweek
    daily_hist = zone_hist.groupby(["date", "dow"])["demand"].sum().reset_index()
    dow_avg = daily_hist.groupby("dow")["demand"].mean()

    daily_forecast["dow"] = daily_forecast["forecast_date"].dt.dayofweek
    daily_forecast["hist_avg"] = daily_forecast["dow"].map(dow_avg)
    ratio = daily_forecast["daily_total"] / daily_forecast["hist_avg"].replace(0, float("nan"))
    daily_forecast["busy_level"] = "Normal"
    daily_forecast.loc[ratio >= 1.2, "busy_level"] = "Busy"
    daily_forecast.loc[ratio <= 0.8, "busy_level"] = "Quiet"
    daily_forecast = daily_forecast.drop(columns=["dow", "hist_avg"])

    if val_mae is not None:
        daily_mae = val_mae * 24
        daily_forecast["confidence"] = (
            (daily_forecast["daily_total"] / (daily_forecast["daily_total"] + daily_mae))
            .clip(0, 1)
            .mul(100)
            .round(0)
            .astype(int)
        )
    else:
        daily_forecast["confidence"] = None

    return daily_forecast


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DemandCast", layout="wide")
st.title("DemandCast - NYC Taxi Demand Forecast")
st.caption(
    f"Weather-style 7-day daily totals from MLflow Model Registry: **{MODEL_NAME}** "
    f"({MODEL_STAGE})"
)

model, val_mae = load_model_and_metrics()
history = load_history()


# ---------------------------------------------------------------------------
# Sidebar input
# ---------------------------------------------------------------------------

zones = sorted(history["PULocationID"].dropna().unique().tolist())
default_zone_idx = zones.index(132) if 132 in zones else 0  # 132 = JFK Airport
zone = st.sidebar.selectbox("Pickup Zone (PULocationID)", zones, index=default_zone_idx)


# ---------------------------------------------------------------------------
# Forecast display
# ---------------------------------------------------------------------------

daily_forecast = build_daily_forecast(model, history, zone, val_mae=val_mae)

st.subheader(f"Next 7 days for zone {zone}")
st.caption(
    "Only the pickup zone is editable. Each value below is the sum of 24 hourly model predictions, "
    "starting tomorrow in NYC local time."
)

cards_html = """
<style>
.forecast-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 8px;
}
.forecast-card {
    position: relative;
    flex: 1 1 130px;
    min-width: 110px;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 18px 14px;
    text-align: center;
}
.fc-label { font-size: 0.85rem; opacity: 0.6; margin-bottom: 8px; white-space: nowrap; }
.fc-value { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
.fc-unit  { font-size: 0.72rem; opacity: 0.45; margin-top: 4px; }
.fc-busy  { font-size: 0.72rem; font-weight: 600; margin-top: 10px; letter-spacing: 0.03em; }
.fc-conf  {
    position: absolute;
    top: 7px; right: 8px;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 5px;
    border-radius: 99px;
    opacity: 0.85;
    color: #fff;
    line-height: 1.4;
}
</style>
<div class="forecast-grid">
"""
BUSY_STYLE = {
    "Busy":   ("▲ Busy",   "#f97316"),
    "Normal": ("● Normal", "rgba(128,128,128,0.55)"),
    "Quiet":  ("▼ Quiet",  "#60a5fa"),
}
for row in daily_forecast.itertuples(index=False):
    conf = getattr(row, "confidence", None)
    if conf is not None:
        color = "#22c55e" if conf >= 80 else "#f59e0b" if conf >= 60 else "#ef4444"
        conf_html = f'<div class="fc-conf" style="background:{color}">{conf}%</div>'
    else:
        conf_html = ""
    busy_label, busy_color = BUSY_STYLE.get(row.busy_level, BUSY_STYLE["Normal"])
    busy_html = f'<div class="fc-busy" style="color:{busy_color}">{busy_label}</div>'
    cards_html += (
        f'<div class="forecast-card">'
        f'{conf_html}'
        f'<div class="fc-label">{row.label}</div>'
        f'<div class="fc-value">{row.daily_total:,.0f}</div>'
        f'<div class="fc-unit">pickups</div>'
        f'{busy_html}'
        f'</div>'
    )
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)
