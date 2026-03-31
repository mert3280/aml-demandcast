# Project 1 — DemandCast: Implementation Guide

**Type:** Individual | **Weeks:** 1–5 | **Presentation:** Week 5, Day 1

---

## Before You Start — Tool Setup
Complete these setup sections from the Student Setup Guide in order:

| When | Setup Guide Section | What You're Installing |
|------|---------------------|------------------------|
| Before Week 1 | Before Week 1 — Core Environment (all 8 steps) | Git, Python 3.11, VS Code, Copilot, core packages |
| Before Week 3 | Before Week 3 — MLflow | MLflow |

Do not skip the checklists at the end of each setup section.

## Repo Structure
Create this on GitHub as `aml-demandcast` (your own individual repo):

```
aml-demandcast/
├── data/                    ← .gitignored — raw files stay here locally
├── notebooks/
│   ├── 01_initial_exploration.ipynb
│   └── 02_eda_skeleton.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   ├── cv.py
│   └── tune.py
├── models/                  ← .gitignored
├── app/
│   └── dashboard.py
├── .gitignore
├── requirements.txt
└── README.md
```

.gitignore must include:

- `data/`
- `models/`
- `mlruns/`
- `*.parquet`
- `__pycache__/`
- `.venv/`

Recommended `requirements.txt` starter:

- pandas
- numpy
- scikit-learn
- mlflow
- streamlit
- pyarrow
- optuna

## Dataset

**NYC TLC Yellow Taxi — January 2024**

Download link: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

→ Select Yellow Taxi Trip Records, January 2024 → download the .parquet file

→ Save to `data/yellow_tripdata_2024-01.parquet`

Load it (from inside a notebook in `notebooks/`):

```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path("../data")   # notebooks/ is one level below the project root
df = pd.read_parquet(DATA_DIR / "yellow_tripdata_2024-01.parquet")
print(df.shape, df.columns.tolist())
```

`../data/` means "go up one folder from notebooks/, then into data/". Use `DATA_DIR` for all subsequent file references in the notebook — e.g. `DATA_DIR / "features.parquet"`.

---

## Week 1 — Repo Scaffold + First Exploration

### Step 1 — Create the repo and connect it to GitHub

If you haven't created the local folder yet (start here):

1. Go to github.com → New repository → name it `aml-demandcast` → add a README → `.gitignore`: Python → No license → Create
2. Open GitHub Desktop → File → Clone Repository → find `aml-demandcast` → choose a local path → Clone
3. Create the folder structure shown above inside the cloned folder (empty folders need a `.gitkeep` file to be tracked by git)

If you've already built the local folder (start here):

1. Open GitHub Desktop → File → Add Local Repository → browse to your `aml-demandcast/` folder
2. Click "create a repository" when prompted → name it `aml-demandcast` → Create Repository
3. Click Publish repository in the top bar → Publish

Either way, the end state is the same: a local folder tracked by git, linked to a repo on github.com.

How it works: Your local folder is where all real work happens. GitHub.com is the online backup and share point. `push` sends your commits up; `pull` brings others' changes down. You never edit files directly on GitHub.

### Step 2 — Write the README and first commit

Write a first draft README: what is this project, what data, what you're predicting. Then commit:

```bash
git add .
git commit -m "scaffold: initial project structure"
git push
```

If `git push` fails with "No configured push destination": GitHub Desktop's Publish step in Step 1 was skipped. Open GitHub Desktop → click Publish repository → then run `git push` again.

### Step 3 — Explore the dataset

Open `notebooks/01_initial_exploration.ipynb`. Use Copilot Chat with these prompts:

- "Write code to load a parquet file using pandas and pyarrow, then print the shape, column names, and data types."
- "Which columns in NYC taxi trip data would be useful for predicting hourly demand by pickup zone? Explain why."
- "What data quality issues would you expect in a large taxi trip dataset, and how would you detect them?"

Run the generated code. Add markdown cells documenting what you found — not just outputs, but what they mean.

**Week 1 Deliverables**

- `aml-demandcast` repo on GitHub with full folder structure
- `.gitignore` includes `data/`, `models/`, `mlruns/`, `*.parquet`
- `requirements.txt` committed
- `notebooks/01_initial_exploration.ipynb` pushed to GitHub

---

## Week 2 — EDA + Feature Engineering

### EDA Notebook — `notebooks/02_eda_skeleton.ipynb`

Your instructor will distribute this skeleton. It has 7 sections with `# TODO` blocks. Complete every section. Every `# TODO` needs a markdown cell below it explaining your decision and why.

Sections (high level):

1. Load & Schema — `pd.read_parquet()`, print dtypes, shape, head
2. Target Variable — Group trips → hourly demand per zone: `df.groupby(['PULocationID', pd.Grouper(key='tpep_pickup_datetime', freq='h')]).size()`
3. Missing Values — `df.isnull().sum()` per column → decide: drop or impute (document your threshold)
4. Outliers — Check `trip_distance`, `fare_amount`, `passenger_count` → define valid ranges → filter
5. Temporal Patterns — Plot average demand by hour, day of week, month → write 3 observations
6. Feature Correlation — Run mutual information + Random Forest importance on candidate features
7. Final Feature List — List selected features, how each is computed, one business justification each

Commit message: `eda: complete sections 1-7, feature candidate list finalized`

### Feature Engineering — `src/features.py`

Your instructor will distribute `src/features_skeleton.py`. It has 4 function stubs with complete docstrings. Implement each using Copilot.

How to use Copilot here:

1. Open `features_skeleton.py` in VS Code
2. Position cursor on the `pass` line inside a function
3. Let Copilot read the docstring and suggest the implementation
4. Run it, verify it matches the docstring, fix anything wrong

The 4 functions and key verifications:

- `create_temporal_features(df)` — `is_rush_hour` is 1 during 7–9am and 5–7pm on weekdays only
- `aggregate_to_hourly_demand(df)` — Output has columns: `PULocationID`, `hour`, `demand`
- `add_lag_features(df, zone_col, target_col)` — Lag is computed per zone: `df.groupby(zone_col)[target_col].shift(1)`
- `filter_outliers(df)` — Thresholds match the ranges you decided in Section 4 of your EDA

After all 4 functions work:

```python
from src.features import create_temporal_features, aggregate_to_hourly_demand, add_lag_features, filter_outliers

df = pd.read_parquet("data/yellow_tripdata_2024-01.parquet")
df = filter_outliers(df)
df = create_temporal_features(df)
hourly = aggregate_to_hourly_demand(df)
hourly = add_lag_features(hourly, zone_col='PULocationID', target_col='demand')
hourly.dropna(inplace=True)   # drop rows where lags are undefined (first hours of data)
hourly.to_parquet("data/features.parquet", index=False)
print(f"Feature matrix: {hourly.shape}")
```

Commit message: `features: implement temporal, lag, outlier, and demand aggregation functions`

**Week 2 Deliverables**

- `notebooks/02_eda_skeleton.ipynb` — all 7 sections, markdown reasoning in every section
- `src/features.py` — all 4 functions implemented and tested
- `data/features.parquet` is NOT committed (it's in `.gitignore`) but `src/features.py` is

---

## Week 3 — Model Training + MLflow + Cross-Validation

### Before anything: start MLflow

From your project folder:

```bash
mlflow ui
```

Open http://localhost:5000 — confirm it loads.

### Train/Val/Test Split

Apply temporal splitting — not random splitting (example):

```python
df = pd.read_parquet("data/features.parquet")
df = df.sort_values('tpep_pickup_datetime')   # or your datetime column

# Cutoffs (adjust based on your data range)
val_cutoff  = "2024-01-22"
test_cutoff = "2024-02-01"

train = df[df['datetime'] < val_cutoff]
val   = df[(df['datetime'] >= val_cutoff) & (df['datetime'] < test_cutoff)]
test  = df[df['datetime'] >= test_cutoff]

# Seal the test set — do NOT look at it until final evaluation
FEATURE_COLS = ['hour', 'day_of_week', 'is_weekend', 'month', 'is_rush_hour',
                                'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h']
TARGET = 'demand'

X_train, y_train = train[FEATURE_COLS], train[TARGET]
X_val,   y_val   = val[FEATURE_COLS],   val[TARGET]
X_test,  y_test  = test[FEATURE_COLS],  test[TARGET]
```

### Training Skeleton — `src/train_skeleton.py`

Your instructor will distribute this. It provides MLflow config, an `evaluate()` helper, and a `train_and_log()` stub. Implement the stub and train at least 3 models (Linear Regression baseline required).

Example models:

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# Third model — your choice (e.g., GradientBoostingRegressor or XGBRegressor)
```

Each run must log params, metrics (`val_mae`, `val_rmse`, `val_r2`), and the model artifact (`mlflow.sklearn.log_model`).

Set MLflow tracking URI in scripts:

```python
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DemandCast")
```

### Cross-Validation — `src/cv_skeleton.py`

Implement `time_series_cv()` using `TimeSeriesSplit` and `clone()` inside the fold loop.

Print CV summary:

```python
print(f"CV MAE: {results['mae'].mean():.2f} ± {results['mae'].std():.2f}")
```

### GitHub PR

Push your work on a branch `feature/model-training` and open a PR to `main` with EDA findings, models trained, CV results, and next steps.

**Week 3 Deliverables**

- ≥3 MLflow runs in the DemandCast experiment, each with model artifact logged
- `src/train.py` committed to `feature/model-training` branch
- `src/cv.py` committed, CV results computed and printed
- GitHub PR open with full description

---

## Week 4 — Tuning + Registration + Dashboard

### Hyperparameter Tuning — `src/tune_skeleton.py`

Run an Optuna study (example):

```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print(f"Best val MAE: {study.best_value:.2f}")
print(f"Best params: {study.best_params}")
```

Every trial is logged to MLflow — compare tuned vs. baseline afterwards.

### Register the Best Model

Register a run as a model and transition stages in MLflow (example code provided in class).

Verify `models:/DemandCast/Production` in MLflow UI.

### Streamlit Dashboard — `app/dashboard_skeleton.py`

Implement sidebar inputs, prediction loop (feature vector must match training exactly), display via `st.metric()`, and one chart.

Run:

```bash
streamlit run app/dashboard.py
```

Opens at http://localhost:8501.

### Final Merge

Before Friday night: merge feature branches (`model-training`, `tuning`, `dashboard`) to `main`.

**Week 4 Deliverables**

- Optuna study with ≥15 trials in MLflow
- `models:/DemandCast/Production` is the tuned model; baseline in Staging
- `app/dashboard.py` committed and running at `localhost:8501`

---

## Week 5, Day 1 — Presentation

Format: 7 minutes + 3 minutes Q&A. Use a running app + MLflow for the demo. Follow the outline in class.

---

## Quick Reference — Starting Your Tools

| Tool | Command | URL |
|---|---|---|
| MLflow | `mlflow ui` (from project folder) | http://localhost:5000 |
| Streamlit | `streamlit run app/dashboard.py` | http://localhost:8501 |

## Troubleshooting

- **ImportError: Missing optional dependency 'pyarrow'**

    ```bash
    pip install pyarrow
    ```

- **`mlruns/` folder appeared in my repo**

    Add `mlruns/` to `.gitignore`, then:

    ```bash
    git rm -r --cached mlruns/
    git commit -m "fix: remove mlruns from tracking"
    ```

- **Dashboard prediction throws an error about column names**

    Copy `FEATURE_COLS` from `src/train.py` into `app/dashboard.py` exactly — same names, same order. Wrap your input DataFrame with `[FEATURE_COLS]` to enforce the order.

- **Lag features look wrong (zone values bleeding into each other)**

    Fix `add_lag_features()` to group by zone before shifting:

    ```python
    df['demand_lag_1h'] = df.groupby(zone_col)[target_col].shift(1)
    ```

- **MLflow UI shows no experiments**

    Confirm `mlflow ui` is running in a terminal and you set `mlflow.set_tracking_uri("http://localhost:5000")` in your script before any `mlflow.*` call.