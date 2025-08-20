"""
etl_pipeline.py
Consolidated ETL demo with improvements:
- Extract JSON from API, CSV file, and (optionally) SQL
- Validate with Pydantic
- Load into PostgreSQL with logging and error handling
"""

import os
import logging
import requests
import pandas as pd
from pydantic import BaseModel, ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# -------------------------
# CONFIG (use env vars for secrets)
# -------------------------
API_URL = os.getenv("API_URL", "https://jsonplaceholder.typicode.com/posts")
CSV_PATH = os.getenv("CSV_PATH", "sample.csv")
DB_URI = os.getenv("DB_URI", "postgresql://user:password@localhost:5432/etl_demo")
TABLE_NAME = os.getenv("TABLE_NAME", "etl_output")

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------------
# 1. EXTRACT
# -------------------------
def from_api() -> pd.DataFrame:
    try:
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        logger.info(f"Fetched {len(resp.json())} records from API")
        return pd.DataFrame(resp.json())
    except requests.RequestException as e:
        logger.error(f"API extraction failed: {e}")
        return pd.DataFrame()

def from_csv() -> pd.DataFrame:
    try:
        df = pd.read_csv(CSV_PATH)
        logger.info(f"Read {len(df)} rows from CSV")
        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found: {CSV_PATH}")
        return pd.DataFrame()

def from_sql(query="SELECT * FROM sample_table") -> pd.DataFrame:
    try:
        engine = create_engine(DB_URI)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        logger.info(f"Extracted {len(df)} rows from SQL")
        return df
    except SQLAlchemyError as e:
        logger.error(f"SQL extraction failed: {e}")
        return pd.DataFrame()

# -------------------------
# 2. TRANSFORM (validation)
# -------------------------
class Record(BaseModel):
    userId: int
    id: int
    title: str
    body: str

def validate(df: pd.DataFrame) -> pd.DataFrame:
    valid = []
    for _, row in df.iterrows():
        try:
            rec = Record(**row.to_dict())
            valid.append(rec.dict())
        except ValidationError as e:
            logger.warning(f"Validation error for row {row.to_dict()}: {e}")
    logger.info(f"Validated {len(valid)} / {len(df)} rows")
    return pd.DataFrame(valid)

# -------------------------
# 3. LOAD
# -------------------------
def load_to_db(df: pd.DataFrame, table_name=TABLE_NAME):
    if df.empty:
        logger.warning("No data to load into DB")
        return
    try:
        engine = create_engine(DB_URI)
        with engine.begin() as conn:  # transactional
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        logger.info(f"Loaded {len(df)} rows into table '{table_name}'")
    except SQLAlchemyError as e:
        logger.error(f"DB load failed: {e}")

# -------------------------
# MAIN PIPELINE
# -------------------------
def run_pipeline():
    logger.info("Starting ETL pipeline...")

    df_api = from_api()
    df_csv = from_csv()
    # Uncomment if SQL is available
    # df_sql = from_sql()

    df_valid = validate(df_api)

    load_to_db(df_valid)

    logger.info("Pipeline completed.")

if __name__ == "__main__":
    run_pipeline()
