"""
01_data_prep_and_sql.py
------------------------
Loads the raw lead-level data, writes it into a local SQLite database
(so the project's SQL query is *actually executed*, not just simulated
in pandas), runs sql/02_cohort_analysis.sql, and saves the aggregated
cohort output table used in the case-study report.

Run:
    python 01_data_prep_and_sql.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SQL_DIR = ROOT / "sql"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RAW_CSV = DATA_DIR / "raw_leads.csv"
DB_PATH = DATA_DIR / "vahan.db"


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)

    # normalise column names to match sql/01_schema.sql
    df = df.rename(columns={
        "Uploaded Leads": "uploaded_leads",
        "Attempted": "attempted",
        "Connected": "connected",
        "Attempt per Lead": "attempt_per_lead",
        "tag_filled": "tag_filled",
        "Interested": "interested",
        "OB_after_upload": "ob_after_upload",
        "OB_after_first_attempt": "ob_after_first_attempt",
        "FT_after_upload": "ft_after_upload",
        "FT_after_first_attempt": "ft_after_first_attempt",
        "upload_to_first_attempt_P50 (hrs)": "upload_to_first_attempt_p50_hrs",
        "Attempted %": "attempted_pct",
        "Attempt \u2192 Connected %": "attempt_to_connected_pct",
        "Connect \u2192 Interested %": "connect_to_interested_pct",
        "Interested \u2192 FT_after_first_attempt %": "interested_to_ft_first_attempt_pct",
        "Attempted \u2192 FT_after_upload %": "attempted_to_ft_after_upload_pct",
    })

    # basic cleaning
    df["candidate_phone"] = df["candidate_phone"].astype(str)
    df = df.dropna(subset=["candidate_phone"])
    df = df[df["candidate_phone"] != "nan"]

    return df


def build_sqlite_db(df: pd.DataFrame) -> None:
    conn = sqlite3.connect(DB_PATH)
    schema_sql = (SQL_DIR / "01_schema.sql").read_text().replace(
        "CREATE TABLE IF NOT EXISTS raw_leads", "DROP TABLE IF EXISTS raw_leads;\nCREATE TABLE raw_leads"
    )
    conn.executescript(schema_sql)
    df.to_sql("raw_leads", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def run_cohort_query() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = (SQL_DIR / "02_cohort_analysis.sql").read_text()
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


def main():
    print("Loading raw data...")
    df = load_raw_data()
    print(f"  {len(df):,} rows loaded")

    print("Building SQLite database...")
    build_sqlite_db(df)

    print("Running cohort SQL query...")
    cohort_df = run_cohort_query()
    cohort_df.to_csv(OUT_DIR / "cohort_aggregated_output.csv", index=False)

    print("\nCohort performance (sorted by FT rate per uploaded lead):\n")
    print(cohort_df.to_string(index=False))
    print(f"\nSaved -> {OUT_DIR / 'cohort_aggregated_output.csv'}")


if __name__ == "__main__":
    main()
