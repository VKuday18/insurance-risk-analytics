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
dashboards/ → Tableau Public
```

## Status

- [x] Repo scaffold + raw data sourced
- [x] ETL + SQL cleaning/transform layer
- [x] EDA notebook — see `notebooks/01_eda.ipynb`, figures in `reports/`
- [x] Predictive model (charges regression) — `src/train_model.py`
- [x] Risk tier classification — `src/train_model.py`
- [x] Interactive dashboard — built in Tableau Public (see below)
- [ ] Findings report

## Interactive dashboard

**[View live dashboard on Tableau Public →](https://public.tableau.com/app/profile/uday.babu.dharmapuri4235/viz/HealthInsuranceRiskAnalyticsDashboard/HealthInsuranceRiskAnalyticsKeyFindings)**

Three linked views built directly from `data/processed/policyholders_clean.csv`,
recreating and making interactive the core EDA findings:

1. **Charges by Smoker** — the ~3.8x cost multiplier for smokers
2. **BMI × Smoking Interaction** — obesity compounding smoking's cost impact
   across all four BMI categories
3. **Risk Tier Validation** — the composite `risk_tier` field cleanly
   separating standard/elevated/high charge levels (~5.2x spread)

Built in Tableau Public rather than Power BI Desktop, which has no native
macOS version. Tableau Public is a free, browser/desktop-hybrid BI tool
covering the same core skill set (data modeling, aggregation, dashboard
design, publishing) — noted here for transparency about tooling choice.

## Modeling results

**Charges regression** (log-target, 80/20 split, random_state=42):

| Model | MAE ($) | R² (log target) |
|---|---|---|
| Linear regression (baseline) | 3,756 | 0.830 |
| Gradient boosting (final) | **2,246** | **0.878** |

Gradient boosting cuts mean absolute error by ~40% over the linear baseline —
consistent with the EDA finding that age/smoker/BMI interact rather than add.

**Risk tier classification — data leakage note:**

`risk_tier` is deterministically computed from `smoker` + `bmi` in the SQL
transform step. A classifier trained with those two fields included scores
99.6% accuracy — that's leakage, not a real result, and is documented as
such in `src/train_model.py` rather than reported as a finding.

The actually meaningful test: **can the other fields alone (age, sex,
children, region) predict risk tier, without the smoker/BMI shortcut?**
Answer: no — accuracy drops to 49.6%, and the model misses every single
`high`-risk case. This is a legitimate, useful finding: smoking and BMI
aren't just correlated with risk in this dataset, they're carrying nearly
all of the predictive signal. Demographics alone don't substitute for them.

Saved models: `models/charges_regressor.joblib`,
`models/risk_tier_classifier.joblib` (trained on non-leaky features).
Full metrics: `reports/model_metrics.json`.


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

