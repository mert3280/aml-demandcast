# DemandCast — Presentation Outline

**Project 1 — Applied Machine Learning (Week 5, Day 1)**
**Author:** Ted Roper

This is a preparation tool, not a slide deck. Each section lists the points to hit and the supporting evidence.

---

## 1. Problem

- **What it is:** DemandCast forecasts hourly taxi pickups by NYC zone, using historical TLC trip data.
- **Why ops cares:** Dispatchers decide where to position drivers and how many shifts to run. Without a forecast, allocation is reactive — drivers chase surges instead of being staged for them. A reasonable hourly forecast (within ~10 trips/hour) is enough to plan shifts, set surge thresholds, and reduce passenger wait time.
- **Stakeholder framing:** The audience is an operations manager, not a data scientist. Every number in this deck must answer the question: *what should I do with this?*

## 2. Data & Features

- **Dataset:** NYC TLC Yellow Cab January 2025 — 3.47M trip records across 263 pickup zones.
- **Aggregation:** Trip-level rows collapsed into hourly demand per zone — 40,547 zone-hour rows after lag-feature drops.
- **Features (9 total):** `hour`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour`, `PULocationID`, plus three demand lags: `demand_lag_1h`, `demand_lag_24h`, `demand_lag_168h`.
- **Most important EDA finding:** The 168-hour (1-week) lag is the strongest single predictor of demand. Demand patterns are dominated by weekly seasonality — Thursday 5pm in midtown looks far more like *last* Thursday 5pm than like Wednesday 5pm or Thursday 4pm. This justified including the 168h lag despite its cost in dropped early-period rows.
- **Cleaning rules applied:** Filtered `trip_distance ∈ (0, 20]`, `fare_amount ∈ (0, 70]`, `passenger_count ∈ [1, 7]`. Dropped rows where any lag was NaN (the first week of data).

## 3. Model

- **Models compared (Week 3):** LinearRegression baseline, RandomForest, LightGBM, XGBoost.
- **Validation MAE results:**
  - LinearRegression: 11.89 trips/hour
  - RandomForest (defaults): 8.33
  - LightGBM (defaults): 8.16
  - XGBoost (defaults): **7.97** (best baseline)
- **Tuning (Week 4):** Optuna study with 15 trials over RandomForest hyperparameters. Best trial reduced RF val MAE from 8.33 → 8.23 — a 1.2% improvement.
- **Production model in MLflow Registry:** Tuned RandomForest (n_estimators=450, max_depth=17, max_features=log2). Week 3 RandomForest baseline retained in Staging.
- **Plain-language headline:** *"Our predictions are off by about 8 trips per hour, on average. If the dashboard says 30 drivers needed in a zone, reality is usually between 22 and 38 — close enough to size a shift, but plan a small buffer for surges."*
- **Honest limitation:** Tuning RF gained little. With more time, the right move was tuning XGBoost (already the best baseline) instead.

## 4. Demo

- **Live walkthrough of `streamlit run app/dashboard.py`:**
  1. Show the sidebar — pick a zone (e.g. JFK = 132); that is the only control.
  2. Point out the 7 forecast cards, each showing a daily total for tomorrow through the next 7 days.
  3. Explain that the app rolls predictions forward hour by hour behind the scenes, then sums them into daily totals.
  4. Use one zone change to show how the whole 7-day pattern updates immediately.
  5. Emphasize that the app loads from the MLflow Model Registry, so promoting a new version in MLflow auto-updates the dashboard.
- **What to emphasize:** The dashboard loads from the MLflow Model Registry, so promoting a new version in MLflow auto-updates the app — no code change required.

## 5. Reflection

- **One thing that surprised me:** The dataset effectively has no test set. The TLC January 2025 file contains exactly one stray record dated 2025-02-01, which got pulled into the test split by my date cutoff. The reported `test_mae` is meaningless on a single row. *Lesson:* always inspect split sizes before trusting a held-out metric — a downstream metric that looks "fine" can hide an empty split.
- **One thing I'd do differently:** Tune the strongest baseline (XGBoost), not the second-strongest (RandomForest). The Optuna budget went to the wrong model — 75 model fits on RF only beat the RF baseline by 1.2% and never approached the untuned XGBoost. The right ordering is: pick the best architecture first, then tune it.
