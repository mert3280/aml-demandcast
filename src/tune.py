"""
tune.py - Hyperparameter tuning for DemandCast (Optuna + MLflow)
=================================================================
Runs separate Optuna studies for RandomForestRegressor, LGBMRegressor, and
XGBRegressor on the train/val split. Each trial is logged to MLflow. After all
studies complete, the model with the lowest val_mae is retrained on train+val
and registered in the MLflow Model Registry.

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
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

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

DATA_PATH    = Path(__file__).parent.parent / "data" / "features.parquet"
TIME_COL     = "pickup_datetime"
RANDOM_STATE = 42
TARGET       = "demand"

N_TRIALS = 15


# ---------------------------------------------------------------------------
# Per-model hyperparameter spaces and fixed params
# ---------------------------------------------------------------------------

def _suggest_rf(trial: optuna.Trial) -> dict:
	return {
		"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
		"max_depth": trial.suggest_int("max_depth", 8, 32),
		"min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
		"min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
		"max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
	}


def _suggest_lgbm(trial: optuna.Trial) -> dict:
	return {
		"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
		"num_leaves": trial.suggest_int("num_leaves", 16, 128),
		"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
		"min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
		"subsample": trial.suggest_float("subsample", 0.5, 1.0),
		"colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
		"reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
		"reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
	}


def _suggest_xgb(trial: optuna.Trial) -> dict:
	return {
		"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
		"max_depth": trial.suggest_int("max_depth", 3, 12),
		"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
		"subsample": trial.suggest_float("subsample", 0.5, 1.0),
		"colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
		"min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
		"gamma": trial.suggest_float("gamma", 0, 5),
		"reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
		"reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
	}


MODEL_CONFIG: dict[str, dict] = {
	"RandomForest": {
		"cls": RandomForestRegressor,
		"suggest_fn": _suggest_rf,
		"fixed_params": {"random_state": 42, "n_jobs": -1},
	},
	"LightGBM": {
		"cls": LGBMRegressor,
		"suggest_fn": _suggest_lgbm,
		"fixed_params": {"random_state": 42, "n_jobs": -1, "verbose": -1},
	},
	"XGBoost": {
		"cls": XGBRegressor,
		"suggest_fn": _suggest_xgb,
		"fixed_params": {"random_state": 42, "n_jobs": -1, "verbosity": 0},
	},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime.datetime:
	return datetime.datetime.now(datetime.UTC)


def _ensure_hour_is_int(frame: pd.DataFrame) -> pd.DataFrame:
	if "hour" in frame.columns and pd.api.types.is_datetime64_any_dtype(frame["hour"]):
		frame["hour"] = frame["hour"].dt.hour
	return frame


def load_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
	"""Load features.parquet and return (X_train, y_train, X_val, y_val)."""
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Missing feature file at {DATA_PATH}. Run build_features.py first.")

	df = pd.read_parquet(DATA_PATH)

	if TIME_COL not in df.columns:
		raise ValueError(
			f"Expected datetime split column '{TIME_COL}' in {DATA_PATH}, got columns: {df.columns.tolist()}"
		)
	if TARGET not in df.columns:
		raise ValueError(f"Expected target column '{TARGET}' in {DATA_PATH}.")

	trainval, _test = train_test_split(df, test_size=0.10, random_state=RANDOM_STATE)
	train, val      = train_test_split(trainval, test_size=2/9, random_state=RANDOM_STATE)

	train = _ensure_hour_is_int(train)
	val   = _ensure_hour_is_int(val)

	missing = [c for c in FEATURE_COLS if c not in train.columns]
	if missing:
		raise ValueError(f"Missing FEATURE_COLS in train split: {missing}")

	return train[FEATURE_COLS], train[TARGET], val[FEATURE_COLS], val[TARGET]


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def make_objective(model_name: str):
	"""Return an Optuna objective function for the given model type."""
	cfg = MODEL_CONFIG[model_name]
	model_cls = cfg["cls"]
	suggest_fn = cfg["suggest_fn"]
	fixed_params = cfg["fixed_params"]

	def objective(trial: optuna.Trial) -> float:
		tunable = suggest_fn(trial)
		params = {**tunable, **fixed_params}

		X_train, y_train, X_val, y_val = load_splits()

		kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

		mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
		mlflow.set_experiment(EXPERIMENT_NAME)

		run_name = f"optuna_{model_name}_trial_{trial.number}_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"

		with mlflow.start_run(run_name=run_name) as run:
			mlflow.log_param("logged_at_utc", _utc_now().isoformat())
			mlflow.log_param("model", model_name)
			mlflow.log_params(tunable)
			mlflow.log_param("objective", "tscv_train")

			fold_maes: list[float] = []
			for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
				X_tr = X_train.iloc[train_idx]
				y_tr = y_train.iloc[train_idx]
				X_va = X_train.iloc[val_idx]
				y_va = y_train.iloc[val_idx]

				m = model_cls(**params)
				m.fit(X_tr, y_tr)
				preds = m.predict(X_va)

				fold_mae = float(mean_absolute_error(y_va, preds))
				fold_maes.append(fold_mae)
				mlflow.log_metric(f"fold_{fold}_mae", fold_mae, step=fold)

			mean_cv_mae = float(np.mean(fold_maes))
			mlflow.log_metric("mean_cv_mae", mean_cv_mae)

			model = model_cls(**params)
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

	return objective


# ---------------------------------------------------------------------------
# Final retrain + registry
# ---------------------------------------------------------------------------

def retrain_and_register(model_name: str, best_params: dict, stage: str = "Production") -> None:
	"""Retrain on train+val, evaluate on test, and register the model in MLflow."""
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

	trainval, test = train_test_split(df, test_size=0.10, random_state=RANDOM_STATE)

	trainval = _ensure_hour_is_int(trainval)
	test     = _ensure_hour_is_int(test)

	missing = [c for c in FEATURE_COLS if c not in trainval.columns]
	if missing:
		raise ValueError(f"Missing FEATURE_COLS in trainval split: {missing}")

	X_trainval = trainval[FEATURE_COLS]
	y_trainval = trainval[TARGET]
	X_test     = test[FEATURE_COLS]
	y_test     = test[TARGET]

	model_cls = MODEL_CONFIG[model_name]["cls"]
	model = model_cls(**best_params)
	model.fit(X_trainval, y_trainval)

	run_name = f"final_retrain_{model_name}_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"
	with mlflow.start_run(run_name=run_name) as run:
		mlflow.log_param("logged_at_utc", _utc_now().isoformat())
		mlflow.log_param("model", model_name)
		mlflow.log_params(best_params)

		if X_test is not None and y_test is not None:
			test_preds = model.predict(X_test)
			test_mae = float(mean_absolute_error(y_test, test_preds))
			test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
			test_r2 = float(r2_score(y_test, test_preds))
			test_mbe = float(np.mean(test_preds - y_test))
			_nonzero = y_test != 0
			test_mape = (
				float(np.mean(np.abs((y_test[_nonzero] - test_preds[_nonzero]) / y_test[_nonzero])) * 100)
				if _nonzero.any() else float("nan")
			)
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
			f"(model={model_name}, run_id={run.info.run_id})"
		)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
	"""Tune RF, LightGBM, and XGBoost; register the best model across all three."""
	best_overall: dict = {"model_name": None, "val_mae": float("inf"), "params": None}

	for model_name in MODEL_CONFIG:
		print(f"\n{'=' * 60}")
		print(f"Tuning {model_name}  ({N_TRIALS} trials)")
		print("=" * 60)

		study = optuna.create_study(
			study_name=f"demandcast_{model_name.lower()}_tuning",
			direction="minimize",
			sampler=optuna.samplers.TPESampler(seed=42),
		)
		study.optimize(make_objective(model_name), n_trials=N_TRIALS)

		best_val_mae = float(study.best_trial.user_attrs.get("val_mae", float("inf")))
		print(f"\n{model_name} best trial:")
		print(f"  number    = {study.best_trial.number}")
		print(f"  cv_mae    = {study.best_value:.4f}")
		print(f"  val_mae   = {best_val_mae:.4f}")
		print(f"  params    = {study.best_trial.params}")

		if best_val_mae < best_overall["val_mae"]:
			best_overall = {
				"model_name": model_name,
				"val_mae": best_val_mae,
				"params": {
					**study.best_trial.params,
					**MODEL_CONFIG[model_name]["fixed_params"],
				},
			}

	print(f"\n{'=' * 60}")
	print(f"Winner: {best_overall['model_name']}  (val_mae={best_overall['val_mae']:.4f})")
	print("=" * 60)

	retrain_and_register(
		model_name=best_overall["model_name"],
		best_params=best_overall["params"],
		stage="Production",
	)


if __name__ == "__main__":
	main()
