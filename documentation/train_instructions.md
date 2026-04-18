Instructions
Part 1 — Temporal Train/Validation/Test Split
Using the feature matrix produced in Assignment 2 (data/features.parquet), implement a temporal train/validation/test split that respects the time-ordered nature of the data. A random split is not appropriate here — document why in a markdown cell.

Your split should follow this structure:

Training set: Weeks 1–3 of January 2024
Validation set: Week 4 of January 2024
Test set: February 1–7, 2024
Verify the split is correct by checking that the maximum timestamp in the training set is before the validation set start, and the minimum timestamp in the test set is at or after its cutoff. Commit this split code to your repository.

The test set is off-limits until Week 4's final evaluation. Do not use it to evaluate or select models in this assignment.

Part 2 — Model Training & MLflow Logging
You are provided with train_skeleton.py (see attached), which includes a pre-implemented evaluate() helper and a train_and_log() stub with a complete docstring. Implement train_and_log() using the docstring-first pattern and use it to train at least 3 models, logging every run to MLflow under the experiment name "DemandCast".

Your three models must include:

A Linear Regression baseline — always start with the simplest model
A Random Forest Regressor
A model of your choice — justify your selection in a code comment before the training call
For each run, log consistently named parameters and metrics. Every run must include at minimum: model type, feature list, validation MAE, and validation RMSE. Inconsistent metric naming across runs will make comparison unreliable.

Part 3 — Cross-Validation
You are provided with cv_skeleton.py (see attached), which contains a time_series_cv() stub. Implement the function using TimeSeriesSplit — standard k-fold cross-validation is not appropriate for time-series data and must not be used.

Run cross-validation on your best-performing model from Part 2 and report the mean and standard deviation of MAE across folds. In a markdown cell, interpret what the standard deviation tells you about your model's stability.

Part 4 — Model Comparison & PR Summary
After all runs are logged, open the MLflow comparison view, select all runs, and take a screenshot. Save this screenshot and a 3-sentence written analysis to notebooks/03_model_comparison.md. Your analysis should address: which model performed best, what the gap between training and validation metrics suggests, and what you would try next.

Open a GitHub Pull Request from your working branch to main. The PR description is a professional deliverable — write it as you would a summary to a team lead at the end of a sprint. It must include:

A brief summary of your EDA findings (2–3 observations)
Your chosen feature set and the reasoning behind it
Each model trained with its validation MAE
Your cross-validation results and what the variance across folds tells you
One concrete improvement you would make given more time
Learning Outcomes
This assignment addresses the following course learning outcomes:

Apply supervised learning algorithms, feature engineering techniques, and model evaluation methods to real-world datasets — by training multiple models, comparing their performance, and interpreting results in the context of the prediction task.
Build and evaluate end-to-end machine learning pipelines that transform raw data into deployed predictions — by implementing a reproducible training pipeline with proper data splitting and experiment tracking.
Deliverables
Submit the link to your aml-demandcast GitHub repository. At the time of submission, the repository must contain:

Temporal split code committed with a descriptive commit message
src/train.py — completed from skeleton, all runs logged to MLflow
src/cv.py — completed from skeleton, cross-validation results computed
notebooks/03_model_comparison.md — MLflow screenshot and 3-sentence analysis
A GitHub Pull Request open with a full sprint summary PR description as specified above
Referenced Documents
train_skeleton.py — provided training module with evaluate() helper and train_and_log() stub (distributed Week 3, Day 3)
cv_skeleton.py — provided cross-validation module with time_series_cv() stub (distributed Week 3, Day 4)
Using AI Assistance in This Course — expectations for using and documenting AI-generated code