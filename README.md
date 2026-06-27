# Tech Companies Price Prediction

Machine-learning pipelines that predict tech-company **acquisition outcomes** from a
multi-file acquisitions dataset — both the acquisition **price** (regression) and the
acquisition **deal-size class** (classification).

## Overview

This project joins four related CSV datasets (acquired companies, acquiring companies,
acquisitions, and founders/board members) into a single feature table, cleans and
encodes it, then trains and compares several scikit-learn / XGBoost models. It was
built as a university machine-learning project and covers the full applied-ML flow:
data cleaning, feature engineering, outlier handling, encoding, imputation, scaling,
and model comparison.

Two independent pipelines share the same approach:

- **Regression** (`src/regression.py`) — predicts the acquisition **`Price`**.
  Compares `GradientBoostingRegressor`, `RandomForestRegressor`, and
  `LinearRegression`, reporting MSE, cross-validated MSE, and R².
- **Classification** (`src/classification.py`) — predicts the **`Deal size class`**.
  Compares `AdaBoostClassifier` and `RandomForestClassifier` (tuned with grid /
  randomized search) and `XGBClassifier`, reporting accuracy and classification
  reports.

## Features

- Joins four CSV sources into one feature table by matching company names / IDs.
- Data cleaning: fixes mistyped years, normalizes country names, drops un-scrapable
  link/image columns, and groups rare categories.
- Feature engineering: derives company age at acquisition and employee-data staleness.
- Multi-valued categorical handling via one-hot expansion of market categories and
  deal terms; label / one-hot encoding for the rest.
- IQR-based outlier handling and log transforms for skewed numeric features.
- KNN imputation for numeric gaps, most-frequent imputation for categoricals.
- Min-max scaling and a train/test split, then side-by-side model comparison.

## Tech stack

- **Python 3.12**
- **pandas**, **numpy**, **scipy** — data wrangling and statistics
- **scikit-learn** — preprocessing, models, model selection, metrics
- **xgboost** — gradient-boosted classifier
- **matplotlib**, **seaborn**, **plotly** — plots and diagnostics

## Project structure

```
.
├── data/
│   ├── regression/        # CSVs used by the regression pipeline
│   └── classification/    # CSVs used by the classification pipeline
├── src/
│   ├── regression.py      # acquisition price prediction
│   └── classification.py  # deal-size-class prediction
├── requirements.txt
├── LICENSE
└── README.md
```

The scripts are written in the Jupyter "percent" cell format (`# %%`), so they can be
run as plain scripts or opened cell-by-cell in an editor / notebook.

## Getting started

Requires Python 3.12+.

```bash
# clone, then from the repo root:
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the scripts **from the repository root** (the data paths are relative to it):

```bash
python src/regression.py        # prints MSE, CV-MSE, and R² per model
python src/classification.py    # prints best params, accuracy, and reports per model
```

Both scripts also produce diagnostic plots (histograms, box plots, confusion
matrices); set a non-interactive backend (e.g. `MPLBACKEND=Agg`) to run headless.

## License

Released under the [MIT License](LICENSE).
