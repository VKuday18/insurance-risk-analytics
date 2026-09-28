"""
ETL step 1: load raw source files into a local SQLite warehouse.

Sources:
  - data/raw/insurance.csv                    (individual-level policyholder data)
  - data/raw/regional_health_benchmarks.csv   (CDC-sourced regional risk benchmarks)

Output:
  - data/processed/insurance_warehouse.db

Run:
  python src/etl_load.py
"""
import sqlite3
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DB_PATH = PROCESSED_DIR / "insurance_warehouse.db"


def load_raw_csvs(conn: sqlite3.Connection) -> None:
    """Load raw CSVs verbatim into staging tables. No cleaning here —
    staging tables should always mirror the source exactly, so any
    transformation bug is caught downstream, not hidden at load time."""
    insurance = pd.read_csv(RAW_DIR / "insurance.csv")
    benchmarks = pd.read_csv(RAW_DIR / "regional_health_benchmarks.csv")

    insurance.to_sql("stg_insurance_raw", conn, if_exists="replace", index=False)
    benchmarks.to_sql("stg_regional_benchmarks", conn, if_exists="replace", index=False)

    print(f"Loaded stg_insurance_raw: {len(insurance)} rows")
    print(f"Loaded stg_regional_benchmarks: {len(benchmarks)} rows")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        load_raw_csvs(conn)
        conn.commit()
    finally:
        conn.close()
    print(f"\nWarehouse ready at {DB_PATH}")


if __name__ == "__main__":
    main()
