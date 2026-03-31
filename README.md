# aml-demandcast

This repository is the first step in building DemandCast — an individual machine learning project to predict hourly taxi demand by pickup zone in New York City. The goal of this assignment is to create a clean, reproducible project environment and develop a working understanding of the NYC taxi dataset that will guide feature engineering and modeling in subsequent weeks.

## Project Goals

- Train models to predict hourly taxi demand per pickup zone.
- Establish a reproducible project structure and environment.
- Perform initial exploratory data analysis (EDA) to inform features.

## Repository Structure

- data/ : raw and processed datasets (store only metadata or pointers to large raw files).
- notebooks/ : exploratory notebooks and experiments.
- project-1-implementation-guide.md : instructor-supplied assignment guide.
- README.md : this document.

Add scripts and dependency files as you develop (e.g., `requirements.txt`, `environment.yml`, or `setup.py`).

## Dataset Overview

We will use the NYC taxi dataset (yellow/green taxi trips). Key fields to expect:

- `pickup_datetime`, `dropoff_datetime` — timestamps.
- `pickup_zone`, `dropoff_zone` — geospatial pickup/dropoff zones (Borough/Zone or taxi zone ID).
- `passenger_count`, `trip_distance`, `fare_amount`, `payment_type` — trip attributes.

Primary target: hourly count of pickups per pickup zone. Important considerations:

- Timezone and daylight savings handling.
- Aggregation to consistent hourly bins.
- Missing or malformed timestamps and zones.

## Setup and Reproducibility

Recommended Python environment (example):

```powershell
# Create virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you prefer conda:

```bash
conda create -n demandcast python=3.10
conda activate demandcast
pip install -r requirements.txt
```

If `requirements.txt` is not present yet, create one with the packages you use (pandas, numpy, matplotlib/seaborn, scikit-learn, geopandas/pyproj if zone geometry is used).

Reproducibility tips:

- Pin package versions in `requirements.txt`.
- Record the dataset source and any preprocessing steps in `data/README.md`.
- Set random seeds when training models.

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

## Next Steps / Deliverables for This Assignment

- Populate `data/` with a small sample or pointers to the full dataset.
- Create an initial EDA notebook in `notebooks/` summarizing: data coverage, hourly demand plots, top zones, and a short list of candidate features.
- Commit `requirements.txt` and document how to reproduce the environment.

## Contact / Author

Repository maintained by the project owner. For questions about setup or content, add an issue or contact the instructor.

---
Update this README as the project evolves. The EDA findings will directly inform feature engineering and model design decisions in subsequent weeks.

