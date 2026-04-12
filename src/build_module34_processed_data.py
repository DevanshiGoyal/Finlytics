from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline import (
    create_shared_features,
    drop_critical_nulls,
    drop_high_null_columns,
    filter_loan_status,
    fix_dtypes,
    load_data,
    save_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
DEFAULT_CLEANED = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.csv"
DEFAULT_MONTHLY = PROJECT_ROOT / "data" / "processed" / "monthly_loan_volume.csv"
DEFAULT_GRADE = PROJECT_ROOT / "data" / "processed" / "grade_monthly_demand.csv"


def _require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {', '.join(missing)}")


def _ensure_cleaned_dataset(cleaned_path: Path, raw_path: Path) -> pd.DataFrame:
    if cleaned_path.exists():
        return pd.read_csv(cleaned_path, low_memory=False)

    if not raw_path.exists():
        raise FileNotFoundError(
            "Cannot build processed datasets because neither cleaned nor raw lending data is available. "
            f"Missing: {cleaned_path} and {raw_path}"
        )

    df = load_data(str(raw_path))
    df = filter_loan_status(df)
    df = drop_high_null_columns(df, threshold=0.5)
    df = fix_dtypes(df)
    df = drop_critical_nulls(df)
    df = create_shared_features(df)
    save_data(df, str(cleaned_path))
    return df


def _build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["issue_d", "funded_amnt", "int_rate", "loan_amnt"], "monthly loan volume")

    work = df.copy()
    work["issue_d"] = pd.to_datetime(work["issue_d"], errors="coerce")
    work = work.dropna(subset=["issue_d", "funded_amnt"])
    work["month_start"] = work["issue_d"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        work.groupby("month_start", as_index=False)
        .agg(
            funded_amnt=("funded_amnt", "sum"),
            loan_count=("funded_amnt", "size"),
            avg_int_rate=("int_rate", "mean"),
            avg_loan_amnt=("loan_amnt", "mean"),
        )
        .sort_values("month_start")
        .reset_index(drop=True)
    )
    monthly["funded_amnt_m"] = monthly["funded_amnt"] / 1_000_000
    return monthly


def _build_grade_monthly(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["issue_d", "grade", "funded_amnt", "int_rate"], "grade demand")

    work = df.copy()
    work["issue_d"] = pd.to_datetime(work["issue_d"], errors="coerce")
    work = work.dropna(subset=["issue_d", "grade", "funded_amnt"])
    work["month_start"] = work["issue_d"].dt.to_period("M").dt.to_timestamp()
    work["grade"] = work["grade"].astype(str).str.upper().str.strip()

    grade_monthly = (
        work.groupby(["month_start", "grade"], as_index=False)
        .agg(
            funded_amnt=("funded_amnt", "sum"),
            loan_count=("funded_amnt", "size"),
            avg_int_rate=("int_rate", "mean"),
        )
        .sort_values(["grade", "month_start"])
        .reset_index(drop=True)
    )
    grade_monthly["funded_amnt_m"] = grade_monthly["funded_amnt"] / 1_000_000
    return grade_monthly


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build data/processed/monthly_loan_volume.csv and "
            "data/processed/grade_monthly_demand.csv from notebook-compatible logic."
        )
    )
    parser.add_argument("--raw", default=str(DEFAULT_RAW), help="Path to raw accepted loans CSV")
    parser.add_argument("--cleaned", default=str(DEFAULT_CLEANED), help="Path to cleaned lending CSV")
    parser.add_argument("--monthly-out", default=str(DEFAULT_MONTHLY), help="Output path for monthly loan volume CSV")
    parser.add_argument("--grade-out", default=str(DEFAULT_GRADE), help="Output path for grade monthly demand CSV")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    cleaned_path = Path(args.cleaned)
    monthly_out = Path(args.monthly_out)
    grade_out = Path(args.grade_out)

    cleaned = _ensure_cleaned_dataset(cleaned_path, raw_path)

    monthly = _build_monthly(cleaned)
    grade_monthly = _build_grade_monthly(cleaned)

    monthly_out.parent.mkdir(parents=True, exist_ok=True)
    grade_out.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(monthly_out, index=False)
    grade_monthly.to_csv(grade_out, index=False)

    print(f"Saved monthly data: {monthly_out} ({len(monthly)} rows)")
    print(f"Saved grade data: {grade_out} ({len(grade_monthly)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
