"""
cv.py — Time-series cross-validation for DemandCast
=====================================================
Implements time_series_cv() using sklearn's TimeSeriesSplit, which preserves
temporal order across folds. Standard k-fold cross-validation must not be used
for time-series data because it randomly assigns observations to folds,
allowing future data to appear in training folds — data leakage that causes
overly optimistic estimates of real-world performance.

TimeSeriesSplit instead expands the training window forward in time, so each
validation fold always lies strictly in the future relative to its training fold.

Usage (from project root with .venv active)
-------------------------------------------
    python src/cv.py

Functions
---------
time_series_cv    Run expanding-window cross-validation on the training set.
                  Returns per-fold MAEs plus mean and std across folds.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from features import FEATURE_COLS  # features.py lives alongside this file in src/

# ---------------------------------------------------------------------------
# Configuration — must match train.py exactly
# ---------------------------------------------------------------------------

DATA_PATH   = Path(__file__).parent.parent / "data" / "features.parquet"
VAL_CUTOFF  = "2025-01-22"   # training set ends here; val/test are sealed for CV
TARGET      = "demand"


# ---------------------------------------------------------------------------
# time_series_cv()
# ---------------------------------------------------------------------------

def time_series_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """Run time-series cross-validation and return per-fold MAEs.

    Uses sklearn's TimeSeriesSplit, which creates expanding training windows
    so that each validation fold is strictly in the future relative to its
    training fold. Standard k-fold must not be used here — shuffling violates
    the temporal dependency structure of demand data.

    The caller is responsible for passing only training-set rows (before
    VAL_CUTOFF). Passing the full dataset would allow the val/test sets to
    leak into CV folds.

    A fresh clone of the model is fit on each fold to prevent state leakage
    between folds (important for stateful estimators like gradient boosters).

    Parameters
    ----------
    model : sklearn-compatible estimator
        An unfitted regression model. Must implement fit() and predict().
    X : pd.DataFrame
        Feature matrix — training rows only, index reset.
    y : pd.Series
        Target vector aligned with X — training rows only.
    n_splits : int, default 5
        Number of CV folds. With n_splits=5, the first fold trains on 1/6
        of the data; each successive fold adds one more block.

    Returns
    -------
    dict with keys:
        'fold_maes' : list[float]   — MAE for each individual fold
        'mean_mae'  : float         — mean MAE across all folds
        'std_mae'   : float         — std dev of MAE across folds
                                      (measure of model stability over time)

    Examples
    --------
    >>> from lightgbm import LGBMRegressor
    >>> results = time_series_cv(LGBMRegressor(n_estimators=100), X_train, y_train)
    >>> print(f"CV MAE: {results['mean_mae']:.2f} ± {results['std_mae']:.2f}")
    """
    tscv      = TimeSeriesSplit(n_splits=n_splits)
    fold_maes  = []
    fold_rmses = []
    fold_r2s   = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        m = clone(model)  # fresh copy — no state carried between folds
        m.fit(X_tr, y_tr)
        preds = m.predict(X_va)

        mae  = mean_absolute_error(y_va, preds)
        rmse = root_mean_squared_error(y_va, preds)
        r2   = r2_score(y_va, preds)
        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)
        print(f"  Fold {fold}/{n_splits}: MAE = {mae:.2f}  RMSE = {rmse:.2f}  R² = {r2:.3f}  "
              f"(train={len(X_tr):,} rows, val={len(X_va):,} rows)")

    results = {
        "fold_maes":  fold_maes,
        "fold_rmses": fold_rmses,
        "fold_r2s":   fold_r2s,
        "mean_mae":   float(np.mean(fold_maes)),
        "std_mae":    float(np.std(fold_maes)),
        "mean_rmse":  float(np.mean(fold_rmses)),
        "std_rmse":   float(np.std(fold_rmses)),
        "mean_r2":    float(np.mean(fold_r2s)),
        "std_r2":     float(np.std(fold_r2s)),
    }
    return results


# ---------------------------------------------------------------------------
# Entry point — runs CV on all models from train.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    # Load features and restrict to training set only — val/test are sealed
    df    = pd.read_parquet(DATA_PATH)
    train = df[df["pickup_datetime"] < VAL_CUTOFF].reset_index(drop=True)

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]

    print(f"5-fold TimeSeriesSplit CV on all models ({len(X_train):,} training rows)\n")
    print("=" * 60)

    models = [
        ("LinearRegression",      LinearRegression()),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("LGBMRegressor",         LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)),
        ("XGBRegressor",          XGBRegressor(n_estimators=100, random_state=42, verbosity=0)),
    ]

    all_results = {}
    for name, model in models:
        print(f"\n{name}")
        results = time_series_cv(model=model, X=X_train, y=y_train, n_splits=5)
        all_results[name] = results
        print(f"  Per-fold MAEs  : {[round(m, 2) for m in results['fold_maes']]}")
        print(f"  Per-fold RMSEs : {[round(m, 2) for m in results['fold_rmses']]}")
        print(f"  Per-fold R²s   : {[round(r, 3) for r in results['fold_r2s']]}")
        print(f"  Mean MAE       : {results['mean_mae']:.2f}  (std {results['std_mae']:.2f})")
        print(f"  Mean RMSE      : {results['mean_rmse']:.2f}  (std {results['std_rmse']:.2f})")
        print(f"  Mean R²        : {results['mean_r2']:.3f}  (std {results['std_r2']:.3f})")

    print("\n" + "=" * 70)
    print("Summary")
    print(f"  {'Model':<26} {'Mean MAE':>10} {'Std MAE':>9} {'Mean RMSE':>11} {'Std RMSE':>10} {'Mean R²':>9}")
    print(f"  {'-'*26} {'-'*10} {'-'*9} {'-'*11} {'-'*10} {'-'*9}")
    for name, r in all_results.items():
        print(f"  {name:<26} {r['mean_mae']:>10.2f} {r['std_mae']:>9.2f} "
              f"{r['mean_rmse']:>11.2f} {r['std_rmse']:>10.2f} {r['mean_r2']:>9.3f}")
