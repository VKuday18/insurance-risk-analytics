"""
Health Insurance Premium & Risk Predictor — Flask app.

Serves:
  - GET  /                → the prediction form (index.html)
  - POST /api/predict      → JSON API: takes a policyholder profile,
                              returns predicted charges + risk tier

The charges prediction uses the trained gradient boosting regressor
(models/charges_regressor.joblib). The risk tier is computed with the
SAME deterministic rule used in sql/02_clean_and_transform.sql — not the
weak (49.6% accuracy) non-leaky classifier, which was trained specifically
to test whether demographics alone can substitute for smoker/BMI (they
can't). The real, useful risk rule is the deterministic one, and it's the
one the dashboard's "Risk Tier Validation" chart validates.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)

# Load once at startup, not per-request
regressor = joblib.load(MODELS_DIR / "charges_regressor.joblib")

VALID_REGIONS = {"northeast", "northwest", "southeast", "southwest"}
VALID_SEX = {"male", "female"}


def compute_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def compute_risk_tier(smoker: str, bmi: float) -> str:
    """Mirrors sql/02_clean_and_transform.sql exactly."""
    if smoker == "yes" and bmi >= 30.0:
        return "high"
    elif smoker == "yes" or bmi >= 30.0:
        return "elevated"
    else:
        return "standard"


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25.0:
        return "normal"
    elif bmi < 30.0:
        return "overweight"
    else:
        return "obese"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        age = float(data["age"])
        sex = str(data["sex"]).lower()
        children = int(data["children"])
        smoker = str(data["smoker"]).lower()
        region = str(data["region"]).lower()

        # BMI: either given directly, or computed from height/weight
        if data.get("bmi") not in (None, ""):
            bmi = float(data["bmi"])
        else:
            weight_kg = float(data["weight_kg"])
            height_cm = float(data["height_cm"])
            bmi = compute_bmi(weight_kg, height_cm)

    except (KeyError, ValueError, ZeroDivisionError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    # Validation
    errors = []
    if not (0 < age <= 120):
        errors.append("Age must be between 1 and 120.")
    if sex not in VALID_SEX:
        errors.append("Sex must be 'male' or 'female'.")
    if children < 0 or children > 20:
        errors.append("Number of children must be 0 or more.")
    if smoker not in ("yes", "no"):
        errors.append("Smoker must be 'yes' or 'no'.")
    if region not in VALID_REGIONS:
        errors.append(f"Region must be one of {sorted(VALID_REGIONS)}.")
    if not (10 <= bmi <= 70):
        errors.append("BMI must be between 10 and 70 — check height/weight units.")

    if errors:
        return jsonify({"error": " ".join(errors)}), 400

    # Predict charges (model trained on log1p(charges))
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region,
    }])
    pred_log = regressor.predict(input_df)[0]
    predicted_charges = float(np.expm1(pred_log))

    tier = compute_risk_tier(smoker, bmi)
    category = bmi_category(bmi)

    return jsonify({
        "predicted_annual_charges": round(predicted_charges, 2),
        "risk_tier": tier,
        "bmi_used": round(bmi, 1),
        "bmi_category": category,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
