# Model Comparison — DemandCast

**Experiment:** DemandCast  
**Date:** 2026-04-15  
**Models trained:** LinearRegression, RandomForestRegressor, LGBMRegressor, XGBRegressor

---

## MLflow Comparison Screenshot

> **Instructions:** Run `mlflow ui` from the project root, open http://localhost:5000,
> navigate to the **DemandCast** experiment, select all runs, click **Compare**, and
> take a screenshot. Replace this block with the image.

![MLflow run comparison — DemandCast experiment](../mlflow_comparison_screenshot.png)

---

## Validation Metrics Summary

| Model | val_mae | val_rmse | val_r2 |
|-------|---------|----------|--------|
| LinearRegression | 11.89 | 22.11 | 0.923 |
| RandomForestRegressor | 8.33 | 17.27 | 0.953 |
| LGBMRegressor | 8.16 | 16.10 | 0.959 |
| XGBRegressor | **7.97** | **15.85** | **0.961** |

---

## Cross-Validation Results (5-fold TimeSeriesSplit, training set only)

### Per-fold detail

**LinearRegression**
```
  Fold 1/5: MAE = 9.70   (train=3,524 rows, val=3,521 rows)
  Fold 2/5: MAE = 17.71  (train=7,045 rows, val=3,521 rows)
  Fold 3/5: MAE = 15.53  (train=10,566 rows, val=3,521 rows)
  Fold 4/5: MAE = 12.29  (train=14,087 rows, val=3,521 rows)
  Fold 5/5: MAE = 20.27  (train=17,608 rows, val=3,521 rows)
```

**RandomForestRegressor**
```
  Fold 1/5: MAE = 10.38  (train=3,524 rows, val=3,521 rows)
  Fold 2/5: MAE = 17.27  (train=7,045 rows, val=3,521 rows)
  Fold 3/5: MAE = 16.49  (train=10,566 rows, val=3,521 rows)
  Fold 4/5: MAE = 11.37  (train=14,087 rows, val=3,521 rows)
  Fold 5/5: MAE = 16.87  (train=17,608 rows, val=3,521 rows)
```

**LGBMRegressor**
```
  Fold 1/5: MAE = 10.45  (train=3,524 rows, val=3,521 rows)
  Fold 2/5: MAE = 18.54  (train=7,045 rows, val=3,521 rows)
  Fold 3/5: MAE = 17.36  (train=10,566 rows, val=3,521 rows)
  Fold 4/5: MAE = 10.22  (train=14,087 rows, val=3,521 rows)
  Fold 5/5: MAE = 16.78  (train=17,608 rows, val=3,521 rows)
```

**XGBRegressor**
```
  Fold 1/5: MAE = 11.82  (train=3,524 rows, val=3,521 rows)
  Fold 2/5: MAE = 18.54  (train=7,045 rows, val=3,521 rows)
  Fold 3/5: MAE = 18.59  (train=10,566 rows, val=3,521 rows)
  Fold 4/5: MAE = 12.31  (train=14,087 rows, val=3,521 rows)
  Fold 5/5: MAE = 20.05  (train=17,608 rows, val=3,521 rows)
```

### Summary

| Model | Mean MAE | Std MAE |
|-------|----------|---------|
| LinearRegression | 15.10 | 3.76 |
| **RandomForestRegressor** | **14.48** | **2.97** |
| LGBMRegressor | 14.67 | 3.58 |
| XGBRegressor | 16.26 | 3.47 |

---

## Analysis

**Which model performed best and by how much:**
XGBoost achieved the lowest validation MAE of 7.97, edging out LightGBM (8.16) by 2.4% and
outperforming the Linear Regression baseline (11.89) by 33% — confirming that tree-based boosting
captures non-linear demand patterns that a linear model cannot.

**What the gap between training and validation metrics suggests:**
XGBoost posted the best single held-out validation MAE (7.97) but the worst CV mean MAE (16.26),
while RandomForest was the most stable across CV folds (mean 14.48, std 2.97 — lowest of all
models). This divergence suggests XGBoost may be benefiting from the specific demand pattern of
validation week 4 rather than generalizing robustly; RandomForest's lower CV variance indicates
more consistent performance across different time windows in the training period.

**What I would try next given more time:**
I would run Optuna hyperparameter tuning on XGBoost — specifically searching over
`max_depth`, `learning_rate`, `subsample`, and `colsample_bytree` — and add zone-level
demand rolling-mean features (e.g. 7-day rolling average) to give the model smoother
historical context beyond the discrete 1h/24h/168h lags.
