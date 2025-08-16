"""
etl_pipeline.py
Consolidated ETL demo:
- Extract JSON from API, CSV file, and SQL table
- Validate with Pydantic
- Load into PostgreSQL
"""

import requests
import pandas as pd
from pydantic import BaseModel, ValidationError
from sqlalchemy import create_engine

# -------------------------
# CONFIG
# -------------------------
API_URL = "https://jsonplaceholder.typicode.com/posts"
CSV_PATH = "sample.csv"   # put a small CSV file in your repo
DB_URI = "postgresql://user:password@localhost:5432/etl_demo"

# -------------------------
# 1. EXTRACT
# -------------------------
def from_api():
    resp = requests.get(API_URL, timeout=10)
    return pd.DataFrame(resp.json())

def from_csv():
    return pd.read_csv(CSV_PATH)

def from_sql():
    engine = create_engine(DB_URI)
    return pd.read_sql("SELECT * FROM sample_table", engine)

# -------------------------
# 2. TRANSFORM (validation)
# -------------------------
class Record(BaseModel):
    userId: int
    id: int
    title: str
    body: str

def validate(df):
    valid = []
    for _, row in df.iterrows():
        try:
            rec = Record(**row.to_dict())
            valid.append(rec.dict())
        except ValidationError as e:
            print("Validation error:", e)
    return pd.DataFrame(valid)

# -------------------------
# 3. LOAD
# -------------------------
def load_to_db(df, table_name="etl_output"):
    engine = create_engine(DB_URI)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into {table_name}")

# -------------------------
# MAIN PIPELINE
# -------------------------
def run_pipeline():
    print("Extracting data...")
    df_api = from_api()
    df_csv = from_csv()
    # Commenting SQL for demo unless you have a table ready
    # df_sql = from_sql()

    print("Validating API data...")
    df_valid = validate(df_api)

    print("Loading into PostgreSQL...")
    load_to_db(df_valid)

if __name__ == "__main__":
    run_pipeline()
