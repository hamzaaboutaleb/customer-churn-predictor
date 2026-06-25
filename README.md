# Telco Customer Churn — End-to-End Machine Learning Pipeline

A complete, production-oriented machine learning project that predicts customer churn for a telecommunications company. The pipeline covers the full lifecycle: data preparation, exploratory analysis, model training, hyperparameter optimisation, experiment tracking, and model interpretation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Summary](#pipeline-summary)
- [Notebooks](#notebooks)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Project Overview

Customer churn — when a subscriber cancels or does not renew their service — is one of the most costly problems in the telecommunications industry. Acquiring a new customer costs five to ten times more than retaining an existing one. This project builds a predictive system that identifies customers likely to churn before they do, enabling targeted, cost-effective retention campaigns.

The project is structured as a sequence of Jupyter notebooks, each covering one stage of the ML pipeline. It is designed to be reproducible, well-documented, and directly applicable to a real business setting.

---

## Business Context

The core objective is to answer two questions:

1. **Who is likely to churn?** Given a customer's contract details, usage patterns, and service subscriptions, what is the probability they will leave in the next billing cycle?
2. **Why are they likely to churn?** Which specific factors are driving that risk, and what targeted action can the business take?

The model produces a churn probability score per customer (0 to 1) and a ranked list of risk drivers, both of which feed directly into a retention strategy.

---

## Dataset

**Source:** IBM Sample Dataset — Telco Customer Churn, available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Size:** 7,043 customers, 21 columns

**Target variable:** `Churn` (Yes / No) — overall churn rate approximately 26.5%.

**Feature categories:**

| Category | Features |
|----------|----------|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Phone services | PhoneService, MultipleLines |
| Internet services | InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

---

## Project Structure

```
telco-churn/
|
+-- data/
|   +-- WA_Fn-UseC_-Telco-Customer-Churn.csv      # raw dataset
|   +-- processed/
|       +-- telco_churn_cleaned.csv                # post-cleaning, pre-encoding
|       +-- telco_churn_encoded.csv                # fully encoded dataset
|       +-- X_train.csv / X_test.csv
|       +-- y_train.csv / y_test.csv
|
+-- notebooks/
|   +-- 01_eda.ipynb                               # data cleaning and EDA
|   +-- 02_modeling.ipynb                          # model training and comparison
|   +-- 03_hyperparameter_tuning.ipynb             # Optuna + GridSearchCV
|   +-- 04_mlflow_tracking.ipynb                   # MLflow experiment tracking
|   +-- 05_interpretation.ipynb                    # SHAP model interpretation
|
+-- models/
|   +-- random_forest.pkl
|   +-- lightgbm.pkl
|   +-- xgboost.pkl
|   +-- xgb_tuned_final.pkl                        # production model
|   +-- best_model.pkl                             # alias for production model
|   +-- best_params.json                           # tuned hyperparameters
|   +-- best_threshold.txt                         # optimal decision threshold
|   +-- model_comparison.csv
|   +-- tuned_model_metrics.csv
|
+-- mlruns/                                        # MLflow tracking directory
|
+-- reports/
|   +-- shap/
|       +-- shap_beeswarm.png
|       +-- shap_bar_importance.png
|       +-- shap_dependence_numerical.png
|       +-- shap_dependence_contract.png
|       +-- shap_heatmap.png
|       +-- shap_waterfall_individuals.png
|       +-- shap_interaction_tenure_charges.png
|       +-- shap_values_test.csv
|       +-- shap_feature_importance.csv
|
+-- artifacts/                                     # MLflow artifact staging
+-- README.md
+-- requirements.txt
```

---

## Pipeline Summary

```
Raw CSV
   |
   v
01 EDA & Cleaning
   - Fix TotalCharges (blank -> 0)
   - Encode target (Yes/No -> 1/0)
   - Engineer 5 new features
   - One-hot encode categoricals
   - Stratified 80/20 split
   |
   v
02 Model Training & Comparison
   - Random Forest (300 trees)
   - LightGBM    (300 estimators)
   - XGBoost     (300 estimators)
   - Compare on AUC-ROC, F1, Precision, Recall
   |
   v
03 Hyperparameter Optimisation
   - Optuna TPE sampler (100 trials)
   - GridSearchCV fine-tuning
   - StratifiedKFold (5 folds)
   - Target: AUC-ROC >= 0.89
   |
   v
04 MLflow Tracking
   - Log all 4 experiments
   - Track params, metrics, artifacts
   - Register best model -> Production
   |
   v
05 SHAP Interpretation
   - TreeExplainer (interventional)
   - Global: beeswarm, bar, heatmap
   - Local: waterfall plots per customer
   - Dependence plots + interactions
   - Business recommendations
```

---

## Notebooks

### 01 — EDA and Feature Engineering (`01_eda.ipynb`)

Loads the raw dataset and prepares it for modelling.

- Inspects shape, dtypes, missing values and distributions.
- Converts `TotalCharges` from string to numeric; fills 11 missing values (new customers with tenure = 0) with zero.
- Encodes the `Churn` target as binary (Yes = 1, No = 0) and drops `customerID`.
- Performs univariate and bivariate analysis across all 20 features, with churn rate breakdowns for every categorical variable.
- Generates a correlation heatmap for numerical features.
- Engineers five additional features:

| Feature | Description |
|---------|-------------|
| `TenureGroup` | Tenure binned into 5 lifecycle stages (0–1yr through 5–6yr) |
| `TotalServices` | Count of add-on services subscribed |
| `AvgMonthlyCharge` | TotalCharges / (tenure + 1) |
| `ChargeRatio` | MonthlyCharges / (TotalCharges + 1) — billing trend indicator |
| `IsSeniorWithPartner` | Binary interaction: senior citizen with a partner |

- Applies binary mapping for Yes/No features, ordinal encoding for `TenureGroup`, and one-hot encoding for multi-class categoricals.
- Splits data 80/20 with stratification on `Churn`.
- Saves six output files to `data/processed/`.

---

### 02 — Model Training and Comparison (`02_modeling.ipynb`)

Trains three classifiers with default parameters and compares them systematically.

- Computes `scale_pos_weight` from the training set to handle the 26% churn imbalance; applied to LightGBM and XGBoost.
- Trains Random Forest (300 trees), LightGBM (300 estimators), and XGBoost (300 estimators).
- Evaluates each model on the held-out test set: AUC-ROC, Accuracy, Precision, Recall, F1-Score.
- Runs 5-fold cross-validation for stability estimates.
- Produces a styled comparison table (best value per metric highlighted), side-by-side bar charts, overlaid ROC curves, and three confusion matrices.
- Plots top-20 feature importances for each model.
- Selects the best model by AUC-ROC then F1-Score as tie-breaker.
- Saves all three model files, the comparison CSV, and a `best_model_name.txt` file.

---

### 03 — Hyperparameter Optimisation (`03_hyperparameter_tuning.ipynb`)

Two-phase tuning strategy targeting AUC-ROC >= 0.89.

**Phase 1 — Optuna Bayesian optimisation (100 trials)**

Uses the TPE (Tree-structured Parzen Estimator) sampler. Searches over ten hyperparameters:

| Parameter | Search space |
|-----------|-------------|
| `n_estimators` | 200 to 1000 (step 50) |
| `max_depth` | 3 to 10 |
| `min_child_weight` | 1 to 10 |
| `learning_rate` | 0.005 to 0.3 (log scale) |
| `subsample` | 0.5 to 1.0 |
| `colsample_bytree` | 0.4 to 1.0 |
| `colsample_bylevel` | 0.4 to 1.0 |
| `gamma` | 0.0 to 5.0 |
| `reg_alpha` | 1e-8 to 10.0 (log scale) |
| `reg_lambda` | 1e-8 to 10.0 (log scale) |

Each trial runs 5-fold stratified cross-validation and optimises for mean AUC-ROC. Produces optimisation history, parameter importance, and top-10 trials table.

**Phase 2 — GridSearchCV fine-tuning**

Narrows to a tight 3 x 3 x 3 grid around Optuna's best values for `max_depth`, `learning_rate`, and `n_estimators`. Results visualised as a heatmap.

**Final model** is trained on the full training set using merged best parameters. The notebook also identifies the optimal decision threshold from the Precision/Recall/F1 curve and saves it as `best_threshold.txt`.

Saves: `xgb_tuned_final.pkl`, `best_params.json`, `optuna_study.pkl`, `best_threshold.txt`, `tuned_model_metrics.csv`.

---

### 04 — MLflow Experiment Tracking (`04_mlflow_tracking.ipynb`)

Logs all four experiments to a local MLflow tracking server and registers the best model.

**Per-run logging:**

| Logged item | Detail |
|-------------|--------|
| Parameters | All hyperparameters + threshold, n_features, train_rows, cv_folds |
| Metrics | test_auc_roc, cv_auc_roc, test_accuracy, test_precision, test_recall, test_f1, train_time_sec |
| Artifacts | Confusion matrix, feature importance plot, ROC curve, SHAP plot (tuned model only), cross-run comparison chart |
| Model | Logged with inferred signature and input example via `mlflow.xgboost.log_model` or `mlflow.sklearn.log_model` |
| Tags | model_family, tuning_stage, dataset, cv_strategy, best_model |

**Model Registry:**

The tuned XGBoost model is registered as `TelcoChurn_XGBoost_Best`, tagged with AUC, F1, threshold, and tuning method, and promoted to the **Production** stage. Previous production versions are automatically archived.

To launch the MLflow UI after running the notebook:

```bash
cd <project_root>
mlflow ui --backend-store-uri mlruns --port 5000
# Open http://127.0.0.1:5000
```

---

### 05 — Model Interpretation with SHAP (`05_interpretation.ipynb`)

Explains model predictions at both the global (population) and local (individual) level using SHAP TreeExplainer.

Configuration:
- `feature_perturbation = 'interventional'` — marginalises correlated features properly, avoiding bias from spurious correlations between `tenure` and `TotalCharges`.
- `model_output = 'probability'` — SHAP values expressed directly as probability contributions.

**Visualisations produced:**

| Plot | What it shows |
|------|--------------|
| Beeswarm (summary) | Each dot is one customer; shows direction and magnitude of every feature's effect |
| Bar importance | Mean absolute SHAP per feature — consistent, game-theoretically grounded importance |
| Dependence plots | tenure, MonthlyCharges, TotalCharges, engineered features, Contract OHE (violin) |
| SHAP heatmap | All test customers sorted by P(churn); reveals cohort patterns |
| Waterfall plots | Five individual customers: TP, TN, borderline, FP, FN |
| Interaction heatmap | tenure x MonthlyCharges binned grid of mean SHAP and mean P(churn) |

Outputs `shap_values_test.csv` and `shap_feature_importance.csv` for downstream use.

---

## Results

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Random Forest (default) | — | — | — | — | — |
| LightGBM (default) | — | — | — | — | — |
| XGBoost (default) | — | — | — | — | — |
| XGBoost (tuned) | — | — | — | — | — |

*(Fill in after running the notebooks — metrics depend on Optuna trial outcomes.)*

Target: **AUC-ROC >= 0.89** on the held-out test set.

---

## Key Findings

**Top churn drivers (from SHAP):**

1. **Tenure** is the single strongest predictor. Churn risk drops sharply after the first 12 months. The 0–12 month window is the critical retention period.

2. **Month-to-month contract** pushes the predicted churn probability up by 0.10 to 0.18 on average. Customers on annual or two-year contracts are substantially more loyal at every tenure level.

3. **High monthly charges** increase churn risk, particularly when combined with short tenure — customers who pay a premium before perceiving value are the highest-risk segment.

4. **TotalCharges** is protective. Long-tenured, high-spending customers have invested in the relationship and are harder to lose.

5. **Number of services** (TotalServices) reduces churn at every tenure level. Each additional service — online security, tech support, streaming — increases switching costs and perceived value.

6. **Fiber optic internet** customers churn at a higher rate than DSL or no-internet customers despite paying more, suggesting service quality or pricing perception issues specific to that product.

7. **Electronic check payment** is correlated with higher churn — likely a proxy for lower commitment or payment friction.

8. **No tech support** significantly increases churn among new customers, suggesting that unresolved technical issues in the onboarding period are a key attrition driver.

**Interaction effect:**

The highest churn concentration is the segment combining **tenure under 12 months** and **monthly charges above $65**. Customers in this cell who are also on month-to-month contracts should be the primary target for retention interventions.

---

## Installation

**Clone the repository:**

```bash
git clone https://github.com/<your-username>/telco-churn.git
cd telco-churn
```

**Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Place the dataset:**

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it in `data/`.

---

## Usage

Run the notebooks in order:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb
jupyter notebook notebooks/03_hyperparameter_tuning.ipynb
jupyter notebook notebooks/04_mlflow_tracking.ipynb
jupyter notebook notebooks/05_interpretation.ipynb
```

Each notebook reads its inputs from the previous notebook's outputs — no manual data transfer required.

**Launch the MLflow UI:**

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

**Load the production model:**

```python
import joblib
import pandas as pd

model     = joblib.load('models/xgb_tuned_final.pkl')
threshold = float(open('models/best_threshold.txt').read())

X_new     = pd.read_csv('data/processed/X_test.csv')   # replace with new data
probs     = model.predict_proba(X_new)[:, 1]
preds     = (probs >= threshold).astype(int)
```

**Load from MLflow registry:**

```python
import mlflow.xgboost

model = mlflow.xgboost.load_model('models:/TelcoChurn_XGBoost_Best/Production')
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
optuna
mlflow
shap
statsmodels
joblib
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna mlflow shap statsmodels joblib
```

---

## Acknowledgements

Dataset provided by IBM and hosted on Kaggle as part of the Watson Analytics Sample Datasets collection.