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

## Cross-Validation Results (LGBMRegressor, 5-fold TimeSeriesSplit)

> **Instructions:** Run `python src/cv.py` and copy the output here.

| | Value |
|-|-------|
| Per-fold MAEs | [10.45, 18.54, 17.36, 10.22, 16.78] |
| Mean MAE | 14.67 |
| Std MAE | 3.58 |

---

## Analysis

**Which model performed best and by how much:**
XGBoost achieved the lowest validation MAE of 7.97, edging out LightGBM (8.16) by 2.4% and
outperforming the Linear Regression baseline (11.89) by 33% — confirming that tree-based boosting
captures non-linear demand patterns that a linear model cannot.

**What the gap between training and validation metrics suggests:**
All four models show strong validation R² scores (0.923–0.961), indicating the lag features
carry most of the predictive signal; however, the two boosting models (LightGBM and XGBoost)
converge tightly in validation performance, suggesting diminishing returns from model complexity
and that the remaining error is likely driven by genuine demand volatility rather than underfitting.

**What I would try next given more time:**
I would run Optuna hyperparameter tuning on XGBoost — specifically searching over
`max_depth`, `learning_rate`, `subsample`, and `colsample_bytree` — and add zone-level
demand rolling-mean features (e.g. 7-day rolling average) to give the model smoother
historical context beyond the discrete 1h/24h/168h lags.
