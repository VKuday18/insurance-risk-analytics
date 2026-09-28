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
- [x] ETL + SQL cleaning/transform layer
- [x] EDA notebook — see `notebooks/01_eda.ipynb`, figures in `reports/`
- [ ] Predictive model (charges regression)
- [ ] Risk tier classification
- [ ] Power BI dashboard
- [ ] Findings report

## Key EDA findings

- Charges are right-skewed (skew 1.52 → -0.09 after log transform) — informs
  model choice.
- Smokers pay ~3.8x the mean / ~4.7x the median charges of non-smokers —
  single strongest driver.
- Smoking × obesity is a real interaction, not additive: obese smokers pay
  ~2.1x what normal-BMI smokers pay.
- `risk_tier` (built in SQL) cleanly separates charges: `high` tier averages
  ~5.2x the `standard` tier.
- Age correlates with charges (r=0.30) but the relationship is clearer within
  each smoking band than across the whole population — supports using
  interaction terms or a tree-based model over a plain linear one.


## Data quality findings (from `01_data_quality_checks.sql`)

- 1 exact duplicate row (19yo male, BMI 30.59, northwest) — removed.
- No nulls, no out-of-range values, no invalid categories otherwise.
- Final analysis-ready row count: **1,337**.

## Run it yourself

```bash
pip install -r requirements.txt
python src/etl_load.py
python -c "import sqlite3; sqlite3.connect('data/processed/insurance_warehouse.db').executescript(open('sql/02_clean_and_transform.sql').read())"
```

