# aml-demandcast

This repository is the first step in building DemandCast — an individual machine learning project to predict hourly taxi demand by pickup zone in New York City. The goal of this assignment is to create a clean, reproducible project environment and develop a working understanding of the NYC taxi dataset that will guide feature engineering and modeling in subsequent weeks.

## Project Goals

- Train models to predict hourly taxi demand per pickup zone.
- Establish a reproducible project structure and environment.
- Perform initial exploratory data analysis (EDA) to inform features.

## Repository Structure

```
aml-demandcast/
├── data/
│   ├── yellow_tripdata_2025-01.parquet       # NYC yellow taxi trips, Jan 2025
│   └── data_dictionary_trip_records_yellow.pdf
├── notebooks/
│   ├── 01_initial_exploration.ipynb          # Initial data inspection and EDA
│   └── 02_eda_skeleton.ipynb                 # Extended EDA with demand patterns
├── src/
│   ├── features_skeleton.py                  # Feature engineering skeleton
│   ├── train.py                              # Model training script
│   ├── tune.py                               # Hyperparameter tuning (Optuna)
│   └── cv.py                                 # Cross-validation utilities
├── project-1-implementation-guide.md         # Instructor-supplied assignment guide
├── requirements.txt
└── README.md
```

## Dataset

**Source:** NYC TLC Yellow Taxi Trip Records — January 2025 (`yellow_tripdata_2025-01.parquet`)

Key fields:

- `tpep_pickup_datetime`, `tpep_dropoff_datetime` — trip timestamps
- `PULocationID`, `DOLocationID` — taxi zone IDs (1–263)
- `passenger_count`, `trip_distance`, `fare_amount`, `payment_type` — trip attributes

See `data/data_dictionary_trip_records_yellow.pdf` for the full field reference.

**Primary target:** hourly pickup count per zone. Key considerations:
- Timezone normalization (data is UTC; NYC is ET)
- Aggregation to consistent hourly bins
- Zones with zero trips in a given hour (sparse demand)

## Setup and Reproducibility

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# bash / conda
conda create -n demandcast python=3.10
conda activate demandcast
pip install -r requirements.txt
```

**Key dependencies:** pandas, numpy, scikit-learn, mlflow, optuna, streamlit, pyarrow, seaborn, matplotlib

## Exploratory Data Analysis (EDA) Checklist

Initial EDA items to complete in `notebooks/`:

- Inspect timeframe covered by the data and gaps.
- Aggregate pickups by hour and zone; visualize daily/weekly/hourly patterns.
- Identify high-demand zones and their stability over time.
- Check for missing or inconsistent zone IDs and timestamps.
- Examine demand seasonality, weekday vs weekend effects, holidays.
- Compute simple baseline: e.g., previous-hour or historical average per zone.

## Suggested Workflow

1. Place raw source files in `data/raw/` (or add pointers if files are large).
2. Perform cleaning and create processed, analysis-ready tables in `data/processed/`.
3. Build EDA notebooks in `notebooks/` to visualize findings.
4. Implement feature engineering scripts and a simple baseline model.
5. Iterate on model improvements and evaluation.

## Status

| Task | Status |
|---|---|
| Environment setup | Done |
| Raw data acquired | Done |
| Initial exploration (`01_initial_exploration.ipynb`) | Done |
| Extended EDA (`02_eda_skeleton.ipynb`) | In progress |
| Feature engineering (`src/features_skeleton.py`) | In progress |
| Baseline model (`src/train.py`) | In progress |
| Hyperparameter tuning (`src/tune.py`) | In progress |

## Author

Ted Roper — Applied Machine Learning, Spring 2026

