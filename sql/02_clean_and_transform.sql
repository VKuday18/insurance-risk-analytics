-- ============================================================
-- 02_clean_and_transform.sql
-- Purpose: produce the analysis-ready fact table from staging.
-- Fixes found in 01_data_quality_checks.sql:
--   - 1 exact duplicate row (age=19, male, bmi=30.59, northwest) -> dedup
--   - no nulls, no out-of-range values, no bad categories otherwise
-- Adds derived features used by EDA / modeling / dashboards.
-- ============================================================

DROP TABLE IF EXISTS fact_policyholders;

CREATE TABLE fact_policyholders AS
WITH deduped AS (
  -- DISTINCT collapses the one exact-duplicate row found in QC step 1
  SELECT DISTINCT age, sex, bmi, children, smoker, region, charges
  FROM stg_insurance_raw
),
enriched AS (
  SELECT
    d.*,

    -- BMI classification, using standard WHO adult BMI bands
    CASE
      WHEN bmi < 18.5 THEN 'underweight'
      WHEN bmi < 25.0 THEN 'normal'
      WHEN bmi < 30.0 THEN 'overweight'
      ELSE 'obese'
    END AS bmi_category,

    -- Age band, common actuarial grouping
    CASE
      WHEN age < 25 THEN '18-24'
      WHEN age < 35 THEN '25-34'
      WHEN age < 45 THEN '35-44'
      WHEN age < 55 THEN '45-54'
      WHEN age < 65 THEN '55-64'
      ELSE '65+'
    END AS age_band,

    -- Composite risk flag: smoker + obese is the single strongest
    -- cost driver in this dataset (confirmed in EDA notebook)
    CASE
      WHEN smoker = 'yes' AND bmi >= 30.0 THEN 'high'
      WHEN smoker = 'yes' OR bmi >= 30.0 THEN 'elevated'
      ELSE 'standard'
    END AS risk_tier,

    b.adult_obesity_rate_2023 AS region_obesity_benchmark_pct

  FROM deduped d
  LEFT JOIN stg_regional_benchmarks b
    ON d.region = b.region
)
SELECT * FROM enriched;

-- Sanity check row count post-dedup (should be 1337, was 1338 raw)
SELECT COUNT(*) AS final_row_count FROM fact_policyholders;
