### Instructions
#### Part 1 — Evaluation Metrics
Using your best model from Assignment 3, compute the following metrics on the validation set: MAE, RMSE, R², MAPE, and MBE. Save your results to notebooks/03_evaluation.md.

For each metric, write one sentence interpreting what the number means in plain language for a taxi operations manager — not a data scientist. For example, stating "MAE = 12.3" is not sufficient; explaining what being off by 12.3 trips per hour means for driver scheduling is. You will reuse this plain-language interpretation in your Streamlit dashboard.

Note on MAPE: zone-hours with zero demand will cause division-by-zero errors. Document how you handle these cases and why.

#### Part 2 — Hyperparameter Tuning
You are provided with tune_skeleton.py (see attached), which includes an Optuna study structure with an objective() stub and MLflow integration. Implement the objective() function and run a study with a minimum of 15 trials against your validation set.

Your search space ranges must be justified — add a comment next to each trial.suggest_* call explaining why you chose those bounds. Copying default ranges without reasoning will not meet the requirements.

After the study completes, compare your tuned model's validation MAE against your Week 3 baseline. Log this comparison in notebooks/03_evaluation.md — did tuning help? By how much? Was the improvement worth the compute cost?

Register the best trial as "DemandCast" version 2 in the MLflow Model Registry and promote it to Production. Your Week 3 best model should remain in Staging.

#### Part 3 — Streamlit Dashboard
You will create a new branch and use it to develop your Streamlit dashboard. You are provided with dashboard_skeleton.py (see attached), which includes the model loading block pre-filled. Using the skeleton as your starting point, build a Streamlit dashboard that:

Loads the Production model from the MLflow Model Registry at startup
Accepts user inputs via a sidebar: pickup zone, hour of day, day of week, and weekend toggle
Displays the predicted demand prominently using st.metric()
Includes the plain-language metric interpretation you wrote in Part 1
Shows at least one visualization — a bar chart of average hourly demand by hour of day is a good starting point
The feature vector passed to model.predict() must match your training data exactly — same column names, same order. Copy your FEATURE_COLS list from train.py directly into the dashboard to avoid column mismatch errors.

To run your dashboard locally: streamlit run app/dashboard.py

#### Part 4 — Presentation Outline
Prepare a 5-point outline for your Project 1 presentation, due Week 5 Day 1. The outline should cover:

Problem — what is DemandCast and why does taxi operations care about it?
Data & Features — what dataset did you use, what features did you engineer, and what was your most important EDA finding?
Model — what models did you try, which won, and what is your best validation MAE in plain terms?
Demo — what will you show live in the Streamlit app?
Reflection — one thing that surprised you and one thing you would do differently
This outline is not your slides — it is your preparation tool. Submit it as docs/presentation_outline.md.

### Learning Outcomes
This assignment addresses the following course learning outcomes:

Apply supervised learning algorithms, feature engineering techniques, and model evaluation methods to real-world datasets — by selecting appropriate evaluation metrics, interpreting them in business terms, and improving model performance through systematic tuning.
Build and evaluate end-to-end machine learning pipelines that transform raw data into deployed predictions — by registering a production model and exposing it through a functional user-facing dashboard.
Communicate model results, technical decisions, and system limitations in plain language to non-technical stakeholders — by writing metric interpretations and a presentation outline designed for a non-technical audience.
### Deliverables
Submit the link to your aml-demandcast GitHub repository. At the time of submission, the repository must contain:

1. notebooks/04_evaluation.md — all four metrics computed, plain-language interpretations written, tuning comparison included
2. src/tune.py — completed from skeleton, Optuna study with ≥15 trials, search space ranges justified in comments
3. "DemandCast" version 2 registered and promoted to Production in MLflow Model Registry
4. app/dashboard.py — completed from skeleton, makes a live prediction from the Production model
5. docs/presentation_outline.md — 5-point presentation outline
6. (dont implement yet) All prior branches merged to main

### Referenced Documents
tune_skeleton.py — provided tuning module with Optuna study structure and objective() stub (distributed Week 4, Day 3)
dashboard_skeleton.py — provided Streamlit skeleton with model loading pre-filled (distributed Week 4, Day 4)
Using AI Assistance in This Course — expectations for using and documenting AI-generated code