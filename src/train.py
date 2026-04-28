"""
train.py — Model training and MLflow logging for DemandCast
============================================================
This script loads the engineered feature set, applies a temporal train/val/test
split, and trains regression models to predict hourly taxi demand per zone.
Every run is logged to MLflow — parameters, metrics, and the model artifact.

Usage (from project root with .venv active)
-------------------------------------------
    python src/train.py

Before running
--------------
1. MLflow UI must be running:
       mlflow ui
   Then open http://localhost:5000 in your browser.
2. features.parquet must exist in data/:
       python build_features.py produces this file.

Why temporal split — not random split
--------------------------------------
Taxi demand is a time-series: each hour's demand is correlated with the hours
that preceded it (via lag features and seasonal patterns). A random split would
scatter future observations into the training set, allowing the model to "see"
future data during training — this is data leakage. The model would appear to
perform well on validation but fail in deployment, where it must predict
strictly from past data. A temporal split guarantees the model is always trained
on past data and evaluated on strictly future data, matching the deployment
scenario exactly.

    Training set:   random 70% of zone-hour rows (random_state=42)
    Validation set: random 20%
    Test set:       random 10% (sealed until final evaluation)

The test set is off-limits until final evaluation. Do not use it here.

Functions
---------
evaluate          Compute MAE, RMSE, and R² for a set of predictions.
                  Already implemented — use it, don't rewrite it.
train_and_log     Load data, split, train one model, log everything to MLflow.
"""

from pathlib import Path

import mlflow
from mlflow import sklearn as mlflow_sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from typing import Any

from features import FEATURE_COLS  # features.py lives alongside this file in src/


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "sqlite:///" + str(Path(__file__).parent.parent / "mlflow.db")
EXPERIMENT_NAME     = "DemandCast"

DATA_PATH = Path(__file__).parent.parent / "data" / "features.parquet"

RANDOM_STATE = 42  # fixed seed for reproducible 70/20/10 random split

TARGET = "demand"


# ---------------------------------------------------------------------------
# evaluate() — already implemented, use it as-is
# ---------------------------------------------------------------------------

def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R² for a set of predictions.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth demand values.
    y_pred : np.ndarray
        Model predictions, same length as y_true.

    Returns
    -------
    dict[str, float]
        Keys: 'mae', 'rmse', 'r2'. Values are floats rounded to 4 decimal places.

    Examples
    --------
    >>> val_preds = model.predict(X_val)
    >>> metrics = evaluate(y_val, val_preds)
    >>> print(f"Val MAE: {metrics['mae']:.2f}  RMSE: {metrics['rmse']:.2f}  R²: {metrics['r2']:.3f}")
    """
    return {
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4),
        "r2":   round(r2_score(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# train_and_log()
# ---------------------------------------------------------------------------

def train_and_log(
    model: Any,
    run_name: str,
    params: dict,
) -> str:
    """Train one regression model and log everything to MLflow.

    This function handles the full training loop for a single model run:
      1. Load data/features.parquet
      2. Apply temporal train/val/test split
      3. Separate features (X) and target (y) for each split
      4. Fit the model on the training set
      5. Evaluate on the validation set using evaluate()
      6. Log params, val metrics, and model artifact to MLflow
      7. Print a summary line and return the MLflow run ID

    The test set must NOT be touched here. Seal it until final evaluation.

    MLflow logging checklist (every run must include all three)
    ----------------------------------------------------------
    mlflow.log_params(params)             — algorithm name + hyperparameters
    mlflow.log_metrics(val_metrics)       — val_mae, val_rmse, val_r2
    mlflow.sklearn.log_model(model, "model") — the fitted model artifact

    Consistent metric naming matters: the MLflow comparison view matches runs
    by key name. Always use 'val_mae', 'val_rmse', 'val_r2' exactly.

    Parameters
    ----------
    model : sklearn estimator
        An unfitted sklearn-compatible regression model.
    run_name : str
        Human-readable label shown in the MLflow UI, e.g. "linear_regression_baseline".
        Use snake_case.
    params : dict
        Dictionary of parameters to log. Must include at minimum:
            {"model": type(model).__name__, ...hyperparameters...}

    Returns
    -------
    str
        The MLflow run ID (a hex string).

    Raises
    ------
    FileNotFoundError
        If data/features.parquet does not exist. Run build_features.py first.
    """
    # 1. Load data
    df = pd.read_parquet(DATA_PATH)

    # 2. Random 70/20/10 split — reproducible via fixed random_state
    # First carve out 10% test, then split the remaining 90% into 70/20.
    trainval, _test = train_test_split(df, test_size=0.10, random_state=RANDOM_STATE)
    train,    val   = train_test_split(trainval, test_size=2/9, random_state=RANDOM_STATE)
    # test is sealed — not touched here

    # 3. Separate features and target
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val,   y_val   = val[FEATURE_COLS],   val[TARGET]

    # 4–7. MLflow run
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        val_preds   = model.predict(X_val)
        val_metrics = evaluate(y_val, val_preds)

        # Prefix keys with "val_" so the comparison view groups them together
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow_sklearn.log_model(model, "model")

        print(f"[{run_name}] val_mae={val_metrics['mae']:.2f}  "
              f"val_rmse={val_metrics['rmse']:.2f}  val_r2={val_metrics['r2']:.3f}")
        return run.info.run_id


# ---------------------------------------------------------------------------
# Entry point — trains four models and logs each run to MLflow
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Model 1: Linear Regression — always start with the simplest baseline
    train_and_log(
        model=LinearRegression(),
        run_name="linear_regression_baseline",
        params={"model": "LinearRegression"},
    )

    # Model 2: Random Forest — strong ensemble baseline, good with tabular data
    train_and_log(
        model=RandomForestRegressor(n_estimators=100, random_state=42),
        run_name="random_forest_100_estimators",
        params={
            "model":        "RandomForestRegressor",
            "n_estimators": 100,
            "max_depth":    None,
            "random_state": 42,
        },
    )

    # Model 3: LightGBM — chosen because our features include lag variables
    # (demand_lag_1h, demand_lag_24h, demand_lag_168h) that are derived from
    # the target itself, making them highly correlated. LightGBM's leaf-wise
    # growth strategy and built-in L1/L2 regularization make it less prone
    # to overfitting on correlated tabular features than standard boosting.
    train_and_log(
        model=LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        run_name="lightgbm_100_estimators",
        params={
            "model":        "LGBMRegressor",
            "n_estimators": 100,
            "random_state": 42,
        },
    )

    # Model 4: XGBoost — included alongside LightGBM to compare bias-variance
    # tradeoffs. XGBoost typically achieves high accuracy on smaller structured
    # datasets but can overfit more easily than LightGBM when features are
    # correlated. Training both allows a direct comparison on this task.
    train_and_log(
        model=XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        run_name="xgboost_100_estimators",
        params={
            "model":        "XGBRegressor",
            "n_estimators": 100,
            "random_state": 42,
        },
    )
