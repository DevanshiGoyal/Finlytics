# src/data_pipeline.py
# FinSight: Financial Forecasting System
# Step 1: Load, clean, and save the Lending Club dataset for all 4 modules

import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH       = "data/raw/accepted_2007_to_2018Q4.csv"
PROCESSED_PATH = "data/processed/loans_cleaned.csv"

# ── Step 1: Load ──────────────────────────────────────────────────────────────
def load_data(path=RAW_PATH):
    """
    Load the raw Lending Club CSV.
    Uses low_memory=False to avoid mixed-type warnings on large files.
    """
    print(f"Loading data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"Raw shape: {df.shape}")
    return df


# ── Step 2: Filter valid loan_status ─────────────────────────────────────────
def filter_loan_status(df):
    """
    Keep only loans that have a clear outcome:
      - Fully Paid   → target = 0 (no default)
      - Charged Off  → target = 1 (default)

    Remove rows with statuses like 'Current', 'Late', 'In Grace Period'
    because their final outcome is unknown.
    """
    valid_statuses = ["Fully Paid", "Charged Off"]
    df = df[df["loan_status"].isin(valid_statuses)].copy()
    print(f"After filtering loan_status: {df.shape}")
    return df


# ── Step 3: Drop high-null columns ───────────────────────────────────────────
def drop_high_null_columns(df, threshold=0.5):
    """
    Drop any column where more than `threshold` (default 50%) of values
    are missing. These columns are unlikely to add useful signal.
    """
    null_ratio = df.isnull().mean()
    cols_to_drop = null_ratio[null_ratio > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} high-null columns. Shape: {df.shape}")
    return df


# ── Step 4: Fix data types ────────────────────────────────────────────────────
def fix_dtypes(df):
    """
    Fix common Lending Club type issues:
      - int_rate and revol_util are stored as strings like "13.56%" → convert to float
      - issue_d and earliest_cr_line are strings → convert to datetime
      - emp_length is a string like "10+ years" → convert to integer
    """

    # Convert percentage strings to float
    for col in ["int_rate", "revol_util"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert date columns to datetime
    for col in ["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")

    # Convert emp_length to integer
    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"]
            .astype(str)
            .str.extract(r"(\d+)")          # grab the number only
            .astype(float)
        )

    print("Data types fixed.")
    return df


# ── Step 5: Drop rows with critical nulls ────────────────────────────────────
def drop_critical_nulls(df):
    """
    Drop rows missing values in columns that are essential
    across all 4 modules.
    """
    critical_cols = [
        "loan_status",
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "grade",
        "issue_d",
    ]
    # Only drop based on columns that actually exist
    critical_cols = [c for c in critical_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    print(f"Dropped {before - len(df)} rows with critical nulls. Shape: {df.shape}")
    return df


# ── Step 6: Create shared derived columns ────────────────────────────────────
def create_shared_features(df):
    """
    Add columns that will be reused across modules:
      - default_flag: binary target for Module 1
      - issue_year / issue_month: extracted from issue_d, used in Modules 3 & 4
    """
    # Binary default target
    df["default_flag"] = (df["loan_status"] == "Charged Off").astype(int)

    # Time features from issue date
    if "issue_d" in df.columns:
        df["issue_year"]  = df["issue_d"].dt.year
        df["issue_month"] = df["issue_d"].dt.month

    print("Shared features created: default_flag, issue_year, issue_month")
    return df


# ── Step 7: Save processed data ───────────────────────────────────────────────
def save_data(df, path=PROCESSED_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to: {path}")
    print(f"Final shape: {df.shape}")


# ── Master pipeline ───────────────────────────────────────────────────────────
def run_pipeline():
    df = load_data()
    df = filter_loan_status(df)
    df = drop_high_null_columns(df, threshold=0.5)
    df = fix_dtypes(df)
    df = drop_critical_nulls(df)
    df = create_shared_features(df)
    save_data(df)
    return df


if __name__ == "__main__":
    run_pipeline()