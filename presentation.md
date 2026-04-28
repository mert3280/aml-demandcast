# DemandCast — Final Results

**Project 1 · Applied Machine Learning · Spring 2026**
**Author:** Ted Roper

---

## Final Features

9 features used for training. All are available at prediction time with no leakage.

| Feature | Type | Description |
|---|---|---|
| `hour` | int (0–23) | Hour of day extracted from pickup datetime |
| `day_of_week` | int (0–6) | Day of week (0 = Monday, 6 = Sunday) |
| `month` | int (1–12) | Month of year |
| `is_weekend` | binary | 1 if day_of_week ≥ 5, else 0 |
| `is_rush_hour` | binary | 1 if hour ∈ {7, 8, 17, 18} on a weekday, else 0 |
| `PULocationID` | int | NYC TLC pickup zone identifier (1–265) |
| `demand_lag_1h` | float | Pickups in this zone 1 hour ago |
| `demand_lag_24h` | float | Pickups in this zone 24 hours ago (same hour yesterday) |
| `demand_lag_168h` | float | Pickups in this zone 168 hours ago (same hour last week) |

Lag features are computed per zone to prevent bleeding between zones. Rows with NaN lags (first 168 hours of each zone's history) are dropped before training.

---

## Best Model

**XGBRegressor** — selected by running separate 15-trial Optuna studies for RandomForest, LightGBM, and XGBoost, then choosing the model with the lowest validation MAE.

| Model | CV MAE | Val MAE |
|---|---|---|
| RandomForest | 9.15 | 9.24 |
| LightGBM | 8.58 | 8.58 |
| **XGBoost** | **8.53** | **8.48** |

---

## Best Parameters

Tuned via Optuna TPE sampler (seed=42, 15 trials). Final model retrained on train + validation combined before test evaluation.

| Parameter | Value |
|---|---|
| `n_estimators` | 600 |
| `max_depth` | 9 |
| `learning_rate` | 0.0317 |
| `subsample` | 1.000 |
| `colsample_bytree` | 0.752 |
| `min_child_weight` | 10 |
| `gamma` | 3.195 |
| `reg_alpha` | 0.572 |
| `reg_lambda` | 3.428 |
| `random_state` | 42 |

---

## Evaluation Metrics

**Data split:** Random 70 / 20 / 10 split (random_state=42, reproducible).

| Split | Rows | Percentage |
|---|---|---|
| Train | 28,382 | 70% |
| Validation | 8,110 | 20% |
| Test | 4,055 | 10% |

**Test set metrics** (held out until final evaluation, never used during tuning):

| Metric | Value | Interpretation |
|---|---|---|
| MAE | **8.13 trips/hour** | Average absolute prediction error per zone-hour |
| RMSE | **15.61 trips/hour** | Penalizes large errors more heavily than MAE |
| R² | **0.965** | Model explains 96.5% of variance in hourly demand |

**Cross-validation (5-fold KFold on training data):**

| Metric | Value |
|---|---|
| Mean CV MAE | 8.53 |


### confidence calculation

daily_total = the sum of 24 hourly predictions for that day

 it's daily_total / (daily_total + test_mae × 24), expressed as a percentage.

The model's test MAE is 8.13 pickups/hour. Multiplied by 24 gives a daily error budget of ~195 pickups. Each day's confidence is just how large the predicted volume is relative to that error budget. High-demand zones like JFK (~3,000/day) will always read ~94%; quieter zones will read lower.