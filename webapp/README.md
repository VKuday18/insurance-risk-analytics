# Prediction Web App

The actual "prediction" half of the Health Insurance Premium & Risk Analytics
Platform — a form-based tool that calls the trained model and returns a
live estimate, rather than the model sitting unused as a `.joblib` file.

## What it does

- Takes age, sex, BMI (or height/weight if BMI isn't known), smoker status,
  number of children, and region.
- Returns a predicted annual charge from the trained gradient boosting
  regressor (`models/charges_regressor.joblib`).
- Returns a risk tier (standard/elevated/high) using the same deterministic
  smoker+BMI rule validated in `sql/02_clean_and_transform.sql` and the
  Tableau dashboard — not the weak, honestly-documented 49.6%-accuracy
  classifier, which exists specifically to show that demographics alone
  can't substitute for smoker/BMI (see main README's leakage section).

## Run locally

```bash
cd webapp
pip install -r requirements.txt
python app.py
```
Visit http://127.0.0.1:5000

## Deploy (free hosting)

Built for Render's free tier:

1. Push this repo to GitHub (already done).
2. On [render.com](https://render.com), create a new **Web Service**, connect
   this GitHub repo.
3. Root directory: `webapp`
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`
6. Deploy — Render gives a public URL like `https://your-app.onrender.com`.

Render's free tier spins down after inactivity and takes ~30–60 seconds to
wake on the first request — normal for a portfolio project, not a bug.
