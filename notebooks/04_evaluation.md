# Evaluation & Hyperparameter Tuning — DemandCast

**Experiment:** DemandCast  
**Date:** 2026-04-23  
**Model:** RandomForestRegressor (tuned via Optuna, 15 trials)

---

## Part 1 — Validation Set Metrics

Metrics computed on the held-out **validation split** (2025-01-22 through 2025-02-01, 19,417 zone-hour rows) using the best Optuna trial (trial 11) retrained on the training set only.

| Metric | Value |
|--------|-------|
| MAE    | 8.23 trips/hour |
| RMSE   | 16.99 trips/hour |
| R²     | 0.9548 |
| MAPE   | 60.31% |
| MBE    | +0.38 trips/hour |

### Plain-Language Interpretations

**MAE = 8.23 trips/hour**  
On average, the model's pickup predictions for a given zone are off by about 8 trips per hour. For a dispatch manager, this means if the model calls for 30 drivers in zone X at 5 PM, reality will likely fall between 22 and 38 — enough accuracy to guide shift planning, but with a buffer needed for tight windows.

**RMSE = 16.99 trips/hour**  
The model's largest errors run roughly double the average. During atypical hours (surge events, weather delays), predictions can be off by 17 or more trips. Operations should treat the forecast as a planning anchor during known surge periods rather than a precise headcount.

**R² = 0.9548**  
The model explains 95.5% of the variation in hourly demand across zones. It reliably captures rush-hour spikes, overnight lulls, and zone-level differences — the main patterns a dispatcher needs to staff correctly.

**MAPE = 60.31%**  
The percentage error looks alarming but is misleading here: zone-hours with 1–3 trips inflate the ratio even when the absolute prediction error is just 1–2 trips. No zone-hours had zero demand in the validation set, so no division-by-zero occurred; however, very low-count hours dominate this metric. MAE is the operationally relevant number for staffing decisions.

*Note on zero demand:* The validation set contains no zone-hours with exactly zero demand. If zero-demand slots are present in future data, they must be excluded from MAPE to avoid division-by-zero. Strategy: filter rows where `y_true == 0` before computing MAPE and document the exclusion count.

**MBE = +0.38 trips/hour**  
The model systematically over-predicts by less than half a trip per zone-hour. This slight positive bias means marginally more drivers dispatched than needed — a safer failure mode than under-staffing, and small enough to be operationally negligible.

---

## Part 2 — Hyperparameter Tuning

### Study Configuration

| Setting | Value |
|---------|-------|
| Framework | Optuna (TPESampler, seed=42) |
| Study name | `demandcast_rf_tuning` |
| Direction | Minimize mean CV MAE |
| CV scheme | TimeSeriesSplit, n_splits=5, on training set |
| Number of trials | 15 |
| Optimization target | `mean_cv_mae` (5-fold CV on training data) |
| Secondary metric | `val_mae` (held-out validation set, logged per trial) |
| Tracking | MLflow experiment "DemandCast", SQLite backend |

### Search Space

| Hyperparameter | Range | Justification |
|----------------|-------|---------------|
| `n_estimators` | 100–600, step 50 | 100–600 trees balance variance reduction with runtime; >600 brought little additional MAE gain in initial trials |
| `max_depth` | 8–32 | Depth 8–32 captures nonlinear interactions from lag features without allowing extremely deep, overfit trees |
| `min_samples_leaf` | 1–20 | Controls overfitting on sparse zone-hour slices while preserving model flexibility for busy zones |
| `min_samples_split` | 2–20 | Regularizes branch growth in noisy demand regimes without over-constraining splits |
| `max_features` | sqrt / log2 / 0.5 | Common strong choices for tabular RF; keeps feature subsampling moderate and promotes diversity |

### All 15 Trial Results

| Trial | n_est | depth | min_leaf | min_split | max_feat | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | CV MAE | Val MAE |
|-------|-------|-------|----------|-----------|----------|--------|--------|--------|--------|--------|--------|---------|
| 0  | 300 | 31 | 15 | 13 | sqrt  | 16.37 | 12.38 | 11.67 | 10.75 | 11.08 | 12.4522 | 9.0588 |
| 1  | 550 | 23 | 15 |  2 | sqrt  | 16.44 | 12.38 | 11.67 | 10.70 | 11.04 | 12.4467 | 9.0595 |
| 2  | 200 | 12 |  7 | 11 | 0.5   | 15.86 | 11.69 | 11.22 | 10.34 | 10.45 | 11.9116 | 8.6823 |
| 3  | 150 | 15 |  8 | 10 | sqrt  | 15.88 | 11.91 | 11.41 | 10.46 | 10.82 | 12.0965 | 8.7075 |
| 4  | 400 |  9 | 13 |  5 | 0.5   | 16.01 | 12.17 | 11.63 | 10.67 | 10.72 | 12.2383 | 9.1590 |
| 5  | 500 | 15 |  2 | 15 | 0.5   | 15.94 | 11.66 | 11.06 | 10.16 | 10.23 | 11.8116 | 8.4177 |
| 6  | 100 | 30 |  6 | 14 | 0.5   | 15.91 | 11.71 | 11.17 | 10.32 | 10.41 | 11.9028 | 8.5603 |
| 7  | 200 | 32 | 16 | 19 | 0.5   | 16.08 | 12.18 | 11.59 | 10.59 | 10.80 | 12.2464 | 9.0422 |
| 8  | 100 | 12 |  1 |  8 | 0.5   | 16.39 | 11.55 | 10.97 | 10.10 | 10.26 | 11.8542 | 8.4512 |
| 9  | 250 | 15 | 11 |  4 | 0.5   | 15.78 | 11.97 | 11.37 | 10.41 | 10.53 | 12.0108 | 8.8020 |
| 10 | 600 | 22 | 20 | 19 | log2  | 16.71 | 12.66 | 11.89 | 10.88 | 11.25 | 12.6784 | 9.2364 |
| **11** | **450** | **17** | **1** | **8** | **log2** | **16.01** | **11.49** | **10.94** | **9.97** | **10.31** | **11.7430** | **8.2369** |
| 12 | 450 | 18 |  1 | 16 | log2  | 16.44 | 11.75 | 11.10 | 10.19 | 10.45 | 11.9885 | 8.4096 |
| 13 | 500 | 19 |  4 |  8 | log2  | 15.79 | 11.55 | 11.00 | 10.12 | 10.43 | 11.7805 | 8.3775 |
| 14 | 400 | 25 |  4 |  7 | log2  | 15.73 | 11.61 | 11.01 | 10.14 | 10.37 | 11.7719 | 8.3828 |

**Bold = best trial.** All runs logged to MLflow experiment "DemandCast".

### Best Trial — Hyperparameters

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 450 |
| `max_depth` | 17 |
| `min_samples_leaf` | 1 |
| `min_samples_split` | 8 |
| `max_features` | log2 |
| `random_state` | 42 |
| `n_jobs` | -1 |

### Tuning Comparison vs Week 3 Baseline

| Model | Val MAE | Notes |
|-------|---------|-------|
| Week 3 RandomForest (default params) | 8.3321 | 100 estimators, sklearn defaults |
| **Tuned RandomForest (trial 11)** | **8.2333** | 450 estimators, depth 17, log2 features |
| Improvement | 0.10 trips/hour (1.2%) | |

**Did tuning help? By how much?**  
Yes — marginally. The best Optuna trial reduced validation MAE from 8.33 to 8.23, a 1.2% improvement. The gain is statistically present but operationally modest: a dispatcher would not notice a difference of 0.1 trips/hour in daily operations.

**Was the improvement worth the compute cost?**  
Probably not for pure MAE gain alone. Fifteen Optuna trials with 5-fold CV each means 75 model fits — the same compute that could train multiple alternative architectures (XGBoost or LightGBM). The Week 3 XGBoost baseline already achieved val_mae=7.97 with default hyperparameters, which surpasses the tuned Random Forest at 8.23. A better use of this compute budget would be tuning the XGBoost model, since the baseline RF was never the strongest performer.

**Key insight from the search:**  
The `log2` feature subsampling consistently appeared in the best trials (11, 12, 13, 14), suggesting that restricting splits to fewer features forces better generalization in this demand dataset. Trials 0–1 with `sqrt` features and large `min_samples_leaf` (15) underperformed — deep, broad trees overfit the training folds. Shallow trees (depth 9, trial 4) also underperformed, indicating the demand patterns require at least moderate depth to capture zone-time interactions.

---

## Model Registry

| Version | Stage | Run | Description |
|---------|-------|-----|-------------|
| v3 | Staging | `81524b6d` | Week 3 RandomForest (100 estimators, default params, val_mae=8.33) |
| v4 | Production | `f27503f2` | Tuned RandomForest (trial 11 params, val_mae=8.23) |

*Note on versioning:* The Optuna study's `retrain_and_register` call created v1 in Production. A subsequent registration of the Week 3 baseline created v2 in Staging. After re-ordering (deleting v1/v2 and re-registering in the correct sequence), MLflow assigned v3 (Week 3, Staging) and v4 (tuned, Production). Version numbers are sequential and non-resettable in MLflow; the stages correctly reflect the assignment intent.

*Note on test_mae:* The dataset ends at 2025-02-01, which is also the TEST_CUTOFF, leaving only 1 row in the test split. The logged `test_mae` (≈78.6) is computed on a single observation and is not meaningful for evaluation.
