# RandomForestModel

A compact scikit-learn classification pipeline built around a `RandomForestClassifier`, with end-to-end preprocessing and cross-validated evaluation.

## What it does

`model.py` defines a single reusable `RandomForestModel` class that:

1. **Loads a training dataset** from a remote CSV URL (default: the `andvise/DataAnalyticsDatasets` train set).
2. **Splits** features (`X`) from the target (`y`) and label-encodes the target.
3. **Builds a `Pipeline`** comprising:
   - `SimpleImputer(strategy="mean")` — fills missing values
   - `StandardScaler()` — standardizes features
   - `RandomForestClassifier` tuned with `criterion="entropy"`, `max_depth=10`, `max_features="log2"`, `n_estimators=30`
4. **Evaluates** the pipeline with 5-fold cross-validation (`cross_val_score`, `cv=5`) and reports the mean CV score.

## Project structure

```
model.py          # RandomForestModel class + entry point
```

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

## Usage

```bash
python model.py
```

The script pulls the training CSV from the URL at the top of `model.py`, builds the pipeline, and prints the 5-fold cross-validation mean accuracy.

## Customization

- **Use your own dataset** by passing a different URL or local CSV path to `RandomForestModel(train_url=...)`.
- **Tune hyperparameters** in `create_pipeline()` — e.g. adjust `n_estimators`, `max_depth`, or `min_samples_split`.
- **Swap the classifier** — the pipeline structure makes it easy to replace `RandomForestClassifier` with another estimator.

## Notes

This is an academic / data-analytics exercise. The dataset is fetched remotely at runtime, so an internet connection is required on first run. No trained model artifact is persisted; cross-validation runs in memory each time.
