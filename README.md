# Health Insurance Premium Prediction & Risk Analytics Platform

Predicts individual health insurance charges and segments policyholders into
risk tiers, using demographic/lifestyle data joined against regional CDC
health benchmarks.

## Data sources

- **`data/raw/insurance.csv`** — 1,338-row public dataset (age, sex, BMI,
  children, smoker status, US region, charges). Originally distributed with
  *Machine Learning with R* (Brett Lantz); mirrored widely on Kaggle as the
  "Medical Cost Personal Datasets."
- **`data/raw/regional_health_benchmarks.csv`** — regional adult obesity
  rates, built from CDC's 2023 BRFSS-based Adult Obesity Prevalence release
  (Midwest 36.0%, South 34.7%, West 29.1%, Northeast 28.6%). The insurance
  dataset's four Census sub-regions are mapped to the nearest CDC region;
  `southeast`/`southwest` both fall under CDC's "South" and share that rate
  — documented in the `census_division_note` column rather than hidden.

## Pipeline

```
data/raw/*.csv
      │
      ▼
src/etl_load.py                  → loads raw CSVs into SQLite staging tables
      │
      ▼
sql/01_data_quality_checks.sql   → audits: dupes, nulls, bad categories, range checks
      │
      ▼
sql/02_clean_and_transform.sql   → dedup + feature engineering + benchmark join
      │
      ▼
data/processed/policyholders_clean.csv   → analysis-ready fact table
      │
      ▼
notebooks/  → EDA
src/        → model training
models/     → saved models
dashboards/ → Power BI
```

## Status

- [x] Repo scaffold + raw data sourced
- [ ] ETL + SQL cleaning/transform layer
- [ ] EDA notebook
- [ ] Predictive model (charges regression)
- [ ] Risk tier classification
- [ ] Power BI dashboard
- [ ] Findings report
