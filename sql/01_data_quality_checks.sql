-- ============================================================
-- 01_data_quality_checks.sql
-- Purpose: audit stg_insurance_raw for the issues a real intake
-- process has to catch before anything downstream trusts the data.
-- Run against: data/processed/insurance_warehouse.db
-- ============================================================

-- 1. Exact duplicate rows (same person, same everything — a re-submitted record)
SELECT age, sex, bmi, children, smoker, region, charges, COUNT(*) AS occurrences
FROM stg_insurance_raw
GROUP BY age, sex, bmi, children, smoker, region, charges
HAVING COUNT(*) > 1;

-- 2. Null / missing values in any required field
SELECT
  SUM(CASE WHEN age      IS NULL THEN 1 ELSE 0 END) AS null_age,
  SUM(CASE WHEN sex      IS NULL THEN 1 ELSE 0 END) AS null_sex,
  SUM(CASE WHEN bmi      IS NULL THEN 1 ELSE 0 END) AS null_bmi,
  SUM(CASE WHEN children IS NULL THEN 1 ELSE 0 END) AS null_children,
  SUM(CASE WHEN smoker   IS NULL THEN 1 ELSE 0 END) AS null_smoker,
  SUM(CASE WHEN region   IS NULL THEN 1 ELSE 0 END) AS null_region,
  SUM(CASE WHEN charges  IS NULL THEN 1 ELSE 0 END) AS null_charges
FROM stg_insurance_raw;

-- 3. Categorical domain check — region should only be the 4 expected values
SELECT DISTINCT region FROM stg_insurance_raw
WHERE region NOT IN ('northeast', 'northwest', 'southeast', 'southwest');

-- 4. Categorical domain check — sex and smoker flags
SELECT DISTINCT sex FROM stg_insurance_raw WHERE sex NOT IN ('male', 'female');
SELECT DISTINCT smoker FROM stg_insurance_raw WHERE smoker NOT IN ('yes', 'no');

-- 5. Range sanity checks — values outside plausible human/policy ranges
SELECT * FROM stg_insurance_raw WHERE age < 0 OR age > 100;
SELECT * FROM stg_insurance_raw WHERE bmi < 10 OR bmi > 70;
SELECT * FROM stg_insurance_raw WHERE children < 0 OR children > 10;
SELECT * FROM stg_insurance_raw WHERE charges <= 0;

-- 6. Outlier scan on charges using IQR (flag, don't drop — outliers here
--    are often legitimate high-severity claims, which is the exact
--    population a risk model needs to see)
WITH stats AS (
  SELECT
    charges,
    NTILE(4) OVER (ORDER BY charges) AS quartile
  FROM stg_insurance_raw
),
q AS (
  SELECT
    MAX(CASE WHEN quartile = 1 THEN charges END) AS q1_max,
    MIN(CASE WHEN quartile = 4 THEN charges END) AS q4_min
  FROM stats
)
SELECT r.*
FROM stg_insurance_raw r, q
WHERE r.charges > (q.q4_min + 1.5 * (q.q4_min - q.q1_max))
ORDER BY r.charges DESC
LIMIT 20;
