from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


CATEGORY_MAPS: Dict[str, Dict[str, int]] = {
    "grade": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6},
    "home_ownership": {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3},
    "purpose": {
        "debt_consolidation": 0,
        "credit_card": 1,
        "home_improvement": 2,
        "other": 3,
        "major_purchase": 4,
        "medical": 5,
        "small_business": 6,
    },
}


def _safe_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def encode_known_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in CATEGORY_MAPS.items():
        if col in out.columns:
            if out[col].dtype == object:
                mapped = out[col].astype(str).str.strip().str.upper() if col == "grade" else out[col].astype(str).str.strip()
                if col == "grade":
                    out[col] = mapped.map(mapping).fillna(0).astype(int)
                else:
                    # purpose in training is lowercase labels
                    if col == "purpose":
                        mapped = mapped.str.lower()
                    out[col] = mapped.map(mapping).fillna(0).astype(int)
            else:
                out[col] = _safe_numeric(out[col], 0).astype(int)
    return out


def prepare_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    out = encode_known_categories(df)
    for feature in feature_list:
        if feature not in out.columns:
            out[feature] = 0
    out = out[feature_list].copy()
    for col in out.columns:
        out[col] = _safe_numeric(out[col], 0.0)
    return out


def generate_synthetic_portfolio(n_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    issue_year = rng.integers(2015, 2024, size=n_rows)
    return pd.DataFrame(
        {
            "loan_amnt": rng.integers(1_000, 40_000, size=n_rows),
            "int_rate": rng.uniform(6.0, 28.0, size=n_rows).round(2),
            "installment": rng.uniform(80, 1_200, size=n_rows).round(2),
            "annual_inc": rng.integers(25_000, 220_000, size=n_rows),
            "dti": rng.uniform(0.5, 35.0, size=n_rows).round(2),
            "delinq_2yrs": rng.integers(0, 5, size=n_rows),
            "inq_last_6mths": rng.integers(0, 8, size=n_rows),
            "open_acc": rng.integers(1, 32, size=n_rows),
            "pub_rec": rng.integers(0, 3, size=n_rows),
            "revol_util": rng.uniform(5, 98, size=n_rows).round(2),
            "total_acc": rng.integers(4, 65, size=n_rows),
            "emp_length": rng.integers(0, 11, size=n_rows),
            "grade": rng.choice(list("ABCDEFG"), size=n_rows, p=[0.2, 0.22, 0.2, 0.16, 0.12, 0.07, 0.03]),
            "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], size=n_rows, p=[0.4, 0.1, 0.47, 0.03]),
            "purpose": rng.choice(
                [
                    "debt_consolidation",
                    "credit_card",
                    "home_improvement",
                    "other",
                    "major_purchase",
                    "medical",
                    "small_business",
                ],
                size=n_rows,
                p=[0.42, 0.22, 0.1, 0.12, 0.06, 0.05, 0.03],
            ),
            "issue_year": issue_year,
            "issue_month": rng.integers(1, 13, size=n_rows),
        }
    )


def apply_stress_scenario(df: pd.DataFrame, rate_bps: int, income_drop_pct: float, dti_bump: float) -> pd.DataFrame:
    stressed = df.copy()
    if "int_rate" in stressed.columns:
        stressed["int_rate"] = _safe_numeric(stressed["int_rate"]) + (rate_bps / 100.0)
    if "annual_inc" in stressed.columns:
        stressed["annual_inc"] = _safe_numeric(stressed["annual_inc"]) * (1 - income_drop_pct / 100.0)
    if "dti" in stressed.columns:
        stressed["dti"] = _safe_numeric(stressed["dti"]) + dti_bump
    if "delinq_2yrs" in stressed.columns and dti_bump >= 3:
        stressed["delinq_2yrs"] = (_safe_numeric(stressed["delinq_2yrs"]) + 1).clip(0, 10)
    return stressed


def score_with_models(raw_df: pd.DataFrame, module1: dict, module2: dict) -> pd.DataFrame:
    m1_x = prepare_features(raw_df, module1["features"])
    m2_x = prepare_features(raw_df, module2["features"])

    out = raw_df.copy()
    out["default_probability"] = module1["xgb"].predict_proba(m1_x)[:, 1]
    out["churn_probability"] = module2["xgb"].predict_proba(m2_x)[:, 1]
    out["risk_score"] = 0.6 * out["default_probability"] + 0.4 * out["churn_probability"]
    out["risk_band"] = pd.cut(
        out["risk_score"],
        bins=[-np.inf, 0.30, 0.55, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return out


def drift_report(current_df: pd.DataFrame, baseline_df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in current_df.columns or col not in baseline_df.columns:
            continue
        cur = _safe_numeric(current_df[col])
        base = _safe_numeric(baseline_df[col])
        mean_shift = float(abs(cur.mean() - base.mean()))
        std = float(base.std() + 1e-6)
        z_shift = mean_shift / std
        miss_delta = float(abs(current_df[col].isna().mean() - baseline_df[col].isna().mean()))
        score = min(1.0, (0.7 * min(1.0, z_shift / 3.0)) + (0.3 * min(1.0, miss_delta * 5)))
        rows.append(
            {
                "feature": col,
                "current_mean": round(float(cur.mean()), 4),
                "baseline_mean": round(float(base.mean()), 4),
                "mean_shift_sigma": round(z_shift, 4),
                "drift_score": round(score, 4),
                "status": "High" if score >= 0.5 else "Moderate" if score >= 0.25 else "Low",
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values("drift_score", ascending=False).reset_index(drop=True)


def model_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
    else:
        values = np.ones(len(feature_names), dtype=float)

    values = np.where(values < 0, np.abs(values), values)
    if values.sum() == 0:
        values = np.ones_like(values, dtype=float)
    norm = values / values.sum()
    return (
        pd.DataFrame({"feature": feature_names, "importance": norm})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def local_feature_impact(row: pd.Series, baseline_means: pd.Series, importance_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    impacts = []
    for _, rec in importance_df.iterrows():
        f = rec["feature"]
        w = float(rec["importance"])
        val = float(row.get(f, 0.0))
        base = float(baseline_means.get(f, 0.0))
        impact = (val - base) * w
        impacts.append(
            {
                "feature": f,
                "value": round(val, 4),
                "baseline": round(base, 4),
                "impact": round(impact, 6),
                "abs_impact": abs(impact),
            }
        )

    df = pd.DataFrame(impacts).sort_values("abs_impact", ascending=False).head(top_n).reset_index(drop=True)
    return df.drop(columns=["abs_impact"])


def generate_demo_monthly_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2007-01-01", "2018-12-01", freq="MS")
    trend = np.linspace(50, 800, len(months))
    season = 30 * np.sin(np.arange(len(months)) * 2 * np.pi / 12)
    noise = rng.normal(0, 12, len(months))
    values = np.maximum(10, trend + season + noise)
    return pd.DataFrame({"month_start": months, "funded_amnt_m": np.round(values, 2)})


def generate_demo_grade_data(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    monthly = generate_demo_monthly_data(seed=seed)
    grade_weights = {"A": 0.28, "B": 0.26, "C": 0.22, "D": 0.15, "E": 0.09}
    rows = []
    for grade, w in grade_weights.items():
        vol = monthly["funded_amnt_m"].values * w
        vol = np.maximum(5, vol + rng.normal(0, 4, len(vol)))
        for dt, v in zip(monthly["month_start"], vol):
            rows.append({"month_start": dt, "grade": grade, "funded_amnt_m": round(float(v), 2)})
    return pd.DataFrame(rows)


def generate_executive_report_md(summary: Dict[str, float], drift: pd.DataFrame) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    high_drift = 0 if drift.empty else int((drift["status"] == "High").sum())
    top_drift = "N/A" if drift.empty else ", ".join(drift.head(3)["feature"].tolist())

    return f"""# Finlytics Executive Risk Brief

Generated: {now}

## Portfolio KPI Snapshot

- Average default probability: **{summary.get('avg_default_probability', 0):.2%}**
- Average churn probability: **{summary.get('avg_churn_probability', 0):.2%}**
- High-risk borrower share: **{summary.get('high_risk_share', 0):.2%}**
- Expected defaults (portfolio): **{summary.get('expected_defaults', 0):.1f}**

## Data Drift Status

- High-drift features: **{high_drift}**
- Most shifted features: **{top_drift}**

## Suggested Actions

1. Tighten underwriting thresholds for applicants in the upper risk band.
2. Prioritize retention campaigns for borrowers with elevated churn probability.
3. Review feature engineering for high-drift variables before retraining.

---
"""
