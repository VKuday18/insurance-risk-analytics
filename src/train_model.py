"""
Train two models on the cleaned policyholder data:

1. Regression: predict `charges` (log-target, since EDA showed heavy right skew)
2. Classification: predict `risk_tier` from features that DON'T just hand the
   model its own answer — see leakage note in train_risk_classifier().

Run:
  python src/train_model.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "policyholders_clean.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
TARGET_REG = "charges"
TARGET_CLF = "risk_tier"


def build_preprocessor(categorical_cols):
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first"), categorical_cols)],
        remainder="passthrough",
    )


def train_regression(df: pd.DataFrame) -> dict:
    X = df[FEATURES]
    y_log = np.log1p(df[TARGET_REG])

    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y_log, df[TARGET_REG], test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(["sex", "smoker", "region"])

    # Baseline: plain linear regression on log(charges) — the "if you did
    # nothing clever" reference point.
    baseline = Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression()),
    ])
    baseline.fit(X_train, y_train)
    baseline_pred_log = baseline.predict(X_test)
    baseline_pred = np.expm1(baseline_pred_log)

    # Main model: gradient boosting, which EDA suggested should win because
    # it can find the age x smoker x bmi interactions without being told
    # about them explicitly.
    gbr = Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        )),
    ])
    gbr.fit(X_train, y_train)
    gbr_pred_log = gbr.predict(X_test)
    gbr_pred = np.expm1(gbr_pred_log)

    metrics = {
        "baseline_linear": {
            "mae_dollars": round(mean_absolute_error(y_raw_test, baseline_pred), 2),
            "r2_on_log_target": round(r2_score(y_test, baseline_pred_log), 4),
        },
        "gradient_boosting": {
            "mae_dollars": round(mean_absolute_error(y_raw_test, gbr_pred), 2),
            "r2_on_log_target": round(r2_score(y_test, gbr_pred_log), 4),
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(gbr, MODELS_DIR / "charges_regressor.joblib")

    return metrics


def train_risk_classifier(df: pd.DataFrame) -> dict:
    """
    IMPORTANT — data leakage note:
    `risk_tier` was *deterministically derived* from `smoker` and `bmi` in
    sql/02_clean_and_transform.sql (high = smoker AND bmi>=30, etc). Training
    a classifier on features that include smoker+bmi to predict a label
    computed directly from smoker+bmi is not a real modeling exercise — it's
    guaranteed near-100% accuracy and would be a leakage bug in a real
    engagement, not a result worth reporting.

    Keeping this documented rather than hiding it. To make the exercise
    meaningful, we instead test something a real underwriting team would
    actually want: can the *other* fields (age, sex, children, region) predict
    risk tier on their own, without being told the smoker/bmi shortcut? This
    tells us how much signal exists outside the two dominant fields.
    """
    LEAKY_FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
    NON_LEAKY_FEATURES = ["age", "sex", "children", "region"]

    y = df[TARGET_CLF]
    results = {}

    for label, feature_set in [
        ("full_features_leaky_reference_only", LEAKY_FEATURES),
        ("non_leaky_features", NON_LEAKY_FEATURES),
    ]:
        X = df[feature_set]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        categorical = [c for c in ["sex", "smoker", "region"] if c in feature_set]
        preprocessor = build_preprocessor(categorical)
        clf = Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
        ])
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[label] = classification_report(y_test, preds, output_dict=True, zero_division=0)

        if label == "non_leaky_features":
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, MODELS_DIR / "risk_tier_classifier.joblib")

    return results


def main():
    df = pd.read_csv(DATA_PATH)

    print("Training charges regression models...")
    reg_metrics = train_regression(df)
    print(json.dumps(reg_metrics, indent=2))

    print("\nTraining risk tier classifier...")
    clf_report = train_risk_classifier(df)
    print(json.dumps(clf_report, indent=2))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "model_metrics.json", "w") as f:
        json.dump({
            "regression": reg_metrics,
            "classification": clf_report,
        }, f, indent=2)

    print(f"\nModels saved to {MODELS_DIR}")
    print(f"Metrics saved to {REPORTS_DIR / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
