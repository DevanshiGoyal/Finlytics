from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BANK_IN = PROJECT_ROOT / "data" / "external" / "bank_updated.csv"
DEFAULT_BANK_OUT = PROJECT_ROOT / "data" / "processed" / "bank_term_deposit_processed.csv"

DEFAULT_ANOMALY_IN = PROJECT_ROOT / "data" / "external" / "synthetic_deposits.csv"
DEFAULT_ANOMALY_OUT = PROJECT_ROOT / "data" / "processed" / "deposit_anomaly_processed.csv"

BANK_REQUIRED = {
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
}

ANOMALY_REQUIRED = {"amount", "hour", "day_of_week", "frequency"}


def _require_columns(df: pd.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for {context}: {', '.join(missing)}")


def _build_bank_processed(bank_in: Path, bank_out: Path) -> int:
    if not bank_in.exists():
        raise FileNotFoundError(f"Bank deposit source file not found: {bank_in}")

    df = pd.read_csv(bank_in)
    _require_columns(df, BANK_REQUIRED, "bank deposit AI")

    out = df.copy()

    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Standardize category casing used by the training pipeline.
    out["job"] = out["job"].astype(str).str.strip().str.lower()
    out["marital"] = out["marital"].astype(str).str.strip().str.lower()
    out["education"] = out["education"].astype(str).str.strip().str.lower()
    out["default"] = out["default"].astype(str).str.strip().str.lower()
    out["housing"] = out["housing"].astype(str).str.strip().str.lower()
    out["loan"] = out["loan"].astype(str).str.strip().str.lower()
    out["contact"] = out["contact"].astype(str).str.strip().str.lower()
    out["month"] = out["month"].astype(str).str.strip().str.lower()
    out["poutcome"] = out["poutcome"].astype(str).str.strip().str.lower()
    out["y"] = out["y"].astype(str).str.strip().str.lower()

    out = out.dropna(subset=numeric_cols)
    out = out.reset_index(drop=True)

    bank_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(bank_out, index=False)
    return len(out)


def _build_anomaly_processed(anomaly_in: Path, anomaly_out: Path) -> int:
    if not anomaly_in.exists():
        raise FileNotFoundError(f"Anomaly source file not found: {anomaly_in}")

    df = pd.read_csv(anomaly_in)
    _require_columns(df, ANOMALY_REQUIRED, "deposit anomaly detection")

    out = df.copy()
    for col in ["amount", "hour", "day_of_week", "frequency"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["amount", "hour", "day_of_week", "frequency"])
    out["amount"] = out["amount"].clip(lower=1.0)
    out["hour"] = out["hour"].clip(lower=0, upper=23).round().astype(int)
    out["day_of_week"] = out["day_of_week"].clip(lower=0, upper=6).round().astype(int)
    out["frequency"] = out["frequency"].clip(lower=1).round().astype(int)
    out = out.reset_index(drop=True)

    anomaly_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(anomaly_out, index=False)
    return len(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build processed datasets for Bank Deposit AI and Deposit Anomaly Detection "
            "from repository source CSV files."
        )
    )
    parser.add_argument("--bank-in", default=str(DEFAULT_BANK_IN))
    parser.add_argument("--bank-out", default=str(DEFAULT_BANK_OUT))
    parser.add_argument("--anomaly-in", default=str(DEFAULT_ANOMALY_IN))
    parser.add_argument("--anomaly-out", default=str(DEFAULT_ANOMALY_OUT))
    args = parser.parse_args()

    bank_rows = _build_bank_processed(Path(args.bank_in), Path(args.bank_out))
    anomaly_rows = _build_anomaly_processed(Path(args.anomaly_in), Path(args.anomaly_out))

    print(f"Saved bank processed dataset: {args.bank_out} ({bank_rows} rows)")
    print(f"Saved anomaly processed dataset: {args.anomaly_out} ({anomaly_rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
