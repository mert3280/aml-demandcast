"""
tune.py - Hyperparameter tuning for DemandCast (Optuna + MLflow)
=================================================================
Runs an Optuna study to tune a RandomForestRegressor on the train/val
split. Each trial is logged to MLflow, then the best hyperparameters are
retrained on train+val and registered in the MLflow Model Registry.

Run from project root with the .venv active:
	python src/tune.py
"""

from pathlib import Path
import datetime

import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
	# Works when running as: python src/tune.py
	from features import FEATURE_COLS
except ImportError:  # pragma: no cover
	# Fallback for module-style execution
	from src.features import FEATURE_COLS


# ---------------------------------------------------------------------------
# Configuration - keep in sync with train.py and cv.py
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "sqlite:///" + str(Path(__file__).parent.parent / "mlflow.db")
EXPERIMENT_NAME = "DemandCast"
MODEL_REGISTRY_NAME = "DemandCast"

DATA_PATH = Path(__file__).parent.parent / "data" / "features.parquet"
TIME_COL = "pickup_datetime"
VAL_CUTOFF = "2025-01-22"
TEST_CUTOFF = "2025-02-01"
TARGET = "demand"

N_TRIALS = 15


def _utc_now() -> datetime.datetime:
	"""Return timezone-aware UTC timestamp."""
	return datetime.datetime.now(datetime.UTC)


def _ensure_hour_is_int(frame: pd.DataFrame) -> pd.DataFrame:
	"""Convert hour column to integer hour-of-day if it is datetime-typed."""
	if "hour" in frame.columns and pd.api.types.is_datetime64_any_dtype(frame["hour"]):
		frame["hour"] = frame["hour"].dt.hour
	return frame


def load_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
	"""Load features.parquet and return train and validation splits.

	Returns
	-------
	X_train, y_train, X_val, y_val
	"""
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Missing feature file at {DATA_PATH}. Run build_features.py first.")

	df = pd.read_parquet(DATA_PATH)

	if TIME_COL not in df.columns:
		raise ValueError(
			f"Expected datetime split column '{TIME_COL}' in {DATA_PATH}, got columns: {df.columns.tolist()}"
		)

	if TARGET not in df.columns:
		raise ValueError(f"Expected target column '{TARGET}' in {DATA_PATH}.")

	df[TIME_COL] = pd.to_datetime(df[TIME_COL])

	val_cutoff = pd.to_datetime(VAL_CUTOFF)
	test_cutoff = pd.to_datetime(TEST_CUTOFF)

	train = df[df[TIME_COL] < val_cutoff].copy()
	val = df[(df[TIME_COL] >= val_cutoff) & (df[TIME_COL] < test_cutoff)].copy()

	if train.empty:
		raise ValueError("Training split is empty. Check VAL_CUTOFF and data timeframe.")
	if val.empty:
		raise ValueError("Validation split is empty. Check VAL_CUTOFF/TEST_CUTOFF and data timeframe.")

	train = _ensure_hour_is_int(train)
	val = _ensure_hour_is_int(val)

	missing_feature_cols = [c for c in FEATURE_COLS if c not in train.columns]
	if missing_feature_cols:
		raise ValueError(f"Missing FEATURE_COLS in train split: {missing_feature_cols}")

	train = train.set_index(TIME_COL)
	val = val.set_index(TIME_COL)

	X_train = train[FEATURE_COLS]
	y_train = train[TARGET]
	X_val = val[FEATURE_COLS]
	y_val = val[TARGET]

	return X_train, y_train, X_val, y_val


def objective(trial: optuna.Trial) -> float:
	"""Optuna objective: suggest hyperparameters, run TimeSeriesSplit CV,
	log per-fold metrics to MLflow, and return mean CV MAE (minimize).
	"""
	params = {
		# 100-600 trees balances variance reduction with runtime; >600 brought little gain in prior runs.
		"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
		# Depth 8-32 captures nonlinear interactions from lag features without allowing extremely deep trees.
		"max_depth": trial.suggest_int("max_depth", 8, 32),
		# Leaf size 1-20 controls overfitting on sparse zone-hour slices while preserving flexibility.
		"min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
		# Split size 2-20 regularizes branch growth in noisy demand regimes.
		"min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
		# sqrt/log2/0.5 are common strong options for tabular RF and keep feature subsampling moderate.
		"max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
		"random_state": 42,
		"n_jobs": -1,
	}

	X_train, y_train, X_val, y_val = load_splits()

	order = np.argsort(X_train.index)
	X_train = X_train.iloc[order]
	y_train = y_train.iloc[order]

	tscv = TimeSeriesSplit(n_splits=5)

	mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
	mlflow.set_experiment(EXPERIMENT_NAME)

	run_name = f"optuna_trial_{trial.number}_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"

	with mlflow.start_run(run_name=run_name) as run:
		mlflow.log_param("logged_at_utc", _utc_now().isoformat())
		mlflow.log_params(params)
		mlflow.log_param("objective", "tscv_train")

		fold_maes: list[float] = []
		for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
			X_tr = X_train.iloc[train_idx]
			y_tr = y_train.iloc[train_idx]
			X_va = X_train.iloc[val_idx]
			y_va = y_train.iloc[val_idx]

			model = RandomForestRegressor(**params)
			model.fit(X_tr, y_tr)
			preds = model.predict(X_va)

			fold_mae = float(mean_absolute_error(y_va, preds))
			fold_maes.append(fold_mae)
			mlflow.log_metric(f"fold_{fold}_mae", fold_mae, step=fold)

		mean_cv_mae = float(np.mean(fold_maes))
		mlflow.log_metric("mean_cv_mae", mean_cv_mae)

		model = RandomForestRegressor(**params)
		model.fit(X_train, y_train)
		val_preds = model.predict(X_val)

		val_mae = float(mean_absolute_error(y_val, val_preds))
		val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
		val_r2 = float(r2_score(y_val, val_preds))
		val_mbe = float(np.mean(val_preds - y_val))
		_nonzero = y_val != 0
		val_mape = float(np.mean(np.abs((y_val[_nonzero] - val_preds[_nonzero]) / y_val[_nonzero])) * 100)

		mlflow.log_metric("val_mae", val_mae)
		mlflow.log_metric("val_rmse", val_rmse)
		mlflow.log_metric("val_r2", val_r2)
		mlflow.log_metric("val_mbe", val_mbe)
		mlflow.log_metric("val_mape", val_mape)
		mlflow.log_param("val_mape_zero_excluded", int((~_nonzero).sum()))

		mlflow_sklearn.log_model(model, "model")

		trial.set_user_attr("val_mae", val_mae)
		trial.set_user_attr("mlflow_run_id", run.info.run_id)

	return mean_cv_mae


def retrain_and_register(best_params: dict, stage: str = "Production") -> None:
	"""Retrain selected hyperparameters on train+val, evaluate on test,
	and register/promote the final model in MLflow.
	"""
	mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
	mlflow.set_experiment(EXPERIMENT_NAME)

	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Missing feature file at {DATA_PATH}. Run build_features.py first.")

	df = pd.read_parquet(DATA_PATH)
	if TIME_COL not in df.columns:
		raise ValueError(
			f"Expected datetime split column '{TIME_COL}' in {DATA_PATH}, got columns: {df.columns.tolist()}"
		)
	if TARGET not in df.columns:
		raise ValueError(f"Expected target column '{TARGET}' in {DATA_PATH}.")

	df[TIME_COL] = pd.to_datetime(df[TIME_COL])
	test_cutoff = pd.to_datetime(TEST_CUTOFF)

	trainval = df[df[TIME_COL] < test_cutoff].copy()
	test = df[df[TIME_COL] >= test_cutoff].copy()

	if trainval.empty:
		raise ValueError("Train+validation split is empty. Check TEST_CUTOFF and data timeframe.")

	trainval = _ensure_hour_is_int(trainval)
	test = _ensure_hour_is_int(test)

	missing_feature_cols = [c for c in FEATURE_COLS if c not in trainval.columns]
	if missing_feature_cols:
		raise ValueError(f"Missing FEATURE_COLS in trainval split: {missing_feature_cols}")

	X_trainval = trainval[FEATURE_COLS]
	y_trainval = trainval[TARGET]

	if test.empty:
		print("Warning: test split is empty; skipping test_mae logging.")
		X_test = None
		y_test = None
	else:
		X_test = test[FEATURE_COLS]
		y_test = test[TARGET]

	model = RandomForestRegressor(**best_params)
	model.fit(X_trainval, y_trainval)

	run_name = f"final_retrain_and_register_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"
	with mlflow.start_run(run_name=run_name) as run:
		mlflow.log_param("logged_at_utc", _utc_now().isoformat())
		mlflow.log_params(best_params)

		if X_test is not None and y_test is not None:
			test_preds = model.predict(X_test)
			test_mae = float(mean_absolute_error(y_test, test_preds))
			test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
			test_r2 = float(r2_score(y_test, test_preds))
			test_mbe = float(np.mean(test_preds - y_test))
			_nonzero = y_test != 0
			test_mape = float(np.mean(np.abs((y_test[_nonzero] - test_preds[_nonzero]) / y_test[_nonzero])) * 100) if _nonzero.any() else float("nan")
			mlflow.log_metric("test_mae", test_mae)
			mlflow.log_metric("test_rmse", test_rmse)
			mlflow.log_metric("test_r2", test_r2)
			mlflow.log_metric("test_mbe", test_mbe)
			mlflow.log_metric("test_mape", test_mape)
			mlflow.log_param("test_mape_zero_excluded", int((~_nonzero).sum()))
			print(f"test_mae={test_mae:.4f}  test_rmse={test_rmse:.4f}  test_r2={test_r2:.4f}")

		mlflow_sklearn.log_model(model, "model")

		registered = mlflow.register_model(
			model_uri=f"runs:/{run.info.run_id}/model",
			name=MODEL_REGISTRY_NAME,
		)

		client = MlflowClient()

		for mv in client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"]):
			if mv.version != registered.version:
				client.transition_model_version_stage(
					name=MODEL_REGISTRY_NAME,
					version=mv.version,
					stage="Staging",
				)

		client.transition_model_version_stage(
			name=MODEL_REGISTRY_NAME,
			version=registered.version,
			stage=stage,
		)

		print(
			f"Registered {MODEL_REGISTRY_NAME} v{registered.version} to {stage} "
			f"(run_id={run.info.run_id})"
		)


def main() -> None:
	"""Run Optuna tuning, then retrain and register best model."""
	sampler = optuna.samplers.TPESampler(seed=42)
	study = optuna.create_study(
		study_name="demandcast_rf_tuning",
		direction="minimize",
		sampler=sampler,
	)

	study.optimize(objective, n_trials=N_TRIALS)

	print("Best trial:")
	print(f"  number={study.best_trial.number}")
	print(f"  mean_cv_mae={study.best_value:.4f}")
	print(f"  val_mae={study.best_trial.user_attrs.get('val_mae')}")
	print(f"  params={study.best_trial.params}")

	best_params = {
		**study.best_trial.params,
		"random_state": 42,
		"n_jobs": -1,
	}
	retrain_and_register(best_params=best_params, stage="Production")


if __name__ == "__main__":
	main()
