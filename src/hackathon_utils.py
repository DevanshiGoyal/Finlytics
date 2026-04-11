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


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def compute_baseline_forecasts(monthly_df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    ts = monthly_df.sort_values("month_start").reset_index(drop=True)
    vals = pd.to_numeric(ts["funded_amnt_m"], errors="coerce").ffill().bfill().values
    last_value = float(vals[-1])
    ma3 = float(np.mean(vals[-3:])) if len(vals) >= 3 else last_value

    alpha = 0.35
    smooth = last_value
    for v in vals:
        smooth = alpha * float(v) + (1.0 - alpha) * smooth

    future_idx = pd.date_range(ts["month_start"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.DataFrame(
        {
            "month_start": future_idx,
            "naive_last": np.repeat(last_value, horizon),
            "moving_avg_3": np.repeat(ma3, horizon),
            "exp_smoothing": np.repeat(float(smooth), horizon),
        }
    )


def forecast_with_uncertainty(monthly_df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    ts = monthly_df.sort_values("month_start").reset_index(drop=True)
    vals = pd.to_numeric(ts["funded_amnt_m"], errors="coerce").ffill().bfill().values.astype(float)

    lookback = min(12, max(3, len(vals)))
    x = np.arange(lookback, dtype=float)
    y = vals[-lookback:]
    slope = float(np.polyfit(x, y, 1)[0]) if lookback >= 2 else 0.0
    recent_mean = float(np.mean(y))
    global_mean = float(np.mean(vals)) if len(vals) else max(recent_mean, 1.0)
    month_profile = ts.groupby(ts["month_start"].dt.month)["funded_amnt_m"].mean().to_dict()

    residuals = np.diff(vals) if len(vals) > 2 else np.array([0.0])
    noise = float(np.std(residuals))
    if noise < 1.0:
        noise = max(5.0, 0.03 * max(global_mean, 1.0))

    last_date = ts["month_start"].iloc[-1]
    base = float(vals[-1])
    rows = []
    for step in range(1, horizon + 1):
        dt = last_date + pd.offsets.MonthBegin(step)
        month = int(dt.month)
        seasonal = float(month_profile.get(month, global_mean) / max(global_mean, 1e-6))
        trend_component = base + slope * step
        central = max(0.0, trend_component * seasonal)
        spread = noise * np.sqrt(step)
        lower = max(0.0, central - 1.28 * spread)
        upper = max(lower, central + 1.28 * spread)
        rows.append(
            {
                "month_start": dt,
                "forecast_central": round(float(central), 2),
                "forecast_lower": round(float(lower), 2),
                "forecast_upper": round(float(upper), 2),
            }
        )
    return pd.DataFrame(rows)


def baseline_backtest(monthly_df: pd.DataFrame, holdout: int = 4) -> pd.DataFrame:
    ts = monthly_df.sort_values("month_start").reset_index(drop=True)
    holdout = max(1, min(int(holdout), max(1, len(ts) - 4)))

    train = ts.iloc[:-holdout].copy()
    test = ts.iloc[-holdout:].copy()

    base_fc = compute_baseline_forecasts(train, horizon=holdout)
    adv_fc = forecast_with_uncertainty(train, horizon=holdout)

    compare = test[["month_start", "funded_amnt_m"]].merge(base_fc, on="month_start", how="left")
    compare = compare.merge(adv_fc[["month_start", "forecast_central", "forecast_lower", "forecast_upper"]], on="month_start", how="left")

    y_true = compare["funded_amnt_m"].values
    rows = [
        {"model": "Naive (last value)", "mape": round(_mape(y_true, compare["naive_last"].values) * 100, 2)},
        {"model": "Moving Average (3)", "mape": round(_mape(y_true, compare["moving_avg_3"].values) * 100, 2)},
        {"model": "Exp Smoothing", "mape": round(_mape(y_true, compare["exp_smoothing"].values) * 100, 2)},
        {"model": "FinSight Forecast", "mape": round(_mape(y_true, compare["forecast_central"].values) * 100, 2)},
    ]
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def detect_recent_anomalies(monthly_df: pd.DataFrame, window: int = 6, z_limit: float = 1.8) -> pd.DataFrame:
    ts = monthly_df.sort_values("month_start").reset_index(drop=True).copy()
    vals = pd.to_numeric(ts["funded_amnt_m"], errors="coerce").ffill().bfill()
    rolling_mean = vals.rolling(window=window, min_periods=max(3, window // 2)).mean()
    rolling_std = vals.rolling(window=window, min_periods=max(3, window // 2)).std().fillna(0.0)

    upper = rolling_mean + z_limit * rolling_std
    lower = rolling_mean - z_limit * rolling_std
    ts["upper"] = upper
    ts["lower"] = lower
    ts["is_anomaly"] = (vals > upper) | (vals < lower)
    ts["direction"] = np.where(vals > upper, "Spike", np.where(vals < lower, "Dip", "Normal"))

    out = ts[ts["is_anomaly"]].copy()
    if out.empty:
        return out
    out["funded_amnt_m"] = vals.loc[out.index].round(2)
    return out[["month_start", "funded_amnt_m", "lower", "upper", "direction"]].tail(8).reset_index(drop=True)


def scenario_forecast(
    monthly_df: pd.DataFrame,
    horizon: int,
    growth_adjust_pct: float = 0.0,
    remove_recent_outlier: bool = False,
    pattern_mode: str = "baseline",
) -> pd.DataFrame:
    base_df = monthly_df.sort_values("month_start").reset_index(drop=True).copy()
    if remove_recent_outlier and len(base_df) >= 8:
        recent = pd.to_numeric(base_df["funded_amnt_m"], errors="coerce").ffill().bfill().tail(8)
        med = float(recent.median())
        mad = float(np.median(np.abs(recent - med))) + 1e-6
        z = np.abs((recent - med) / mad)
        base_df.loc[recent.index[z > 3.5], "funded_amnt_m"] = med

    out = forecast_with_uncertainty(base_df, horizon=horizon)
    mult = 1.0 + (growth_adjust_pct / 100.0)
    for col in ["forecast_central", "forecast_lower", "forecast_upper"]:
        out[col] = (pd.to_numeric(out[col], errors="coerce") * mult).clip(lower=0.0)

    if pattern_mode == "flat":
        flat = float(out["forecast_central"].iloc[0])
        out["forecast_central"] = flat
        out["forecast_lower"] = (flat * 0.92).clip(lower=0.0)
        out["forecast_upper"] = np.maximum(out["forecast_lower"], flat * 1.08)
    elif pattern_mode == "seasonal_plus":
        if len(out) > 1:
            phase = np.arange(1, len(out) + 1)
            seasonal_boost = 1.0 + 0.04 * np.sin(phase * 2 * np.pi / max(len(out), 2))
            out["forecast_central"] = (out["forecast_central"].values * seasonal_boost).clip(min=0.0)
            out["forecast_lower"] = (out["forecast_lower"].values * seasonal_boost).clip(min=0.0)
            out["forecast_upper"] = (out["forecast_upper"].values * seasonal_boost).clip(min=0.0)

    return out.round(2)


def forecast_confidence_label(backtest_df: pd.DataFrame, forecast_df: pd.DataFrame) -> Tuple[str, float]:
    finsight_row = backtest_df[backtest_df["model"] == "FinSight Forecast"]
    mape = float(finsight_row["mape"].iloc[0]) if not finsight_row.empty else 18.0
    width = pd.to_numeric(forecast_df["forecast_upper"] - forecast_df["forecast_lower"], errors="coerce")
    central = pd.to_numeric(forecast_df["forecast_central"], errors="coerce").replace(0, np.nan)
    rel_band = float((width / central).replace([np.inf, -np.inf], np.nan).fillna(1.0).mean())

    score = max(0.0, min(100.0, 100.0 - (1.6 * mape) - (40.0 * rel_band)))
    if score >= 70:
        label = "High confidence"
    elif score >= 45:
        label = "Watch closely"
    else:
        label = "Unstable"
    return label, round(score, 1)


def rolling_backtest(monthly_df: pd.DataFrame, max_horizon: int = 6) -> pd.DataFrame:
    ts = monthly_df.sort_values("month_start").reset_index(drop=True)
    if len(ts) < 14:
        return pd.DataFrame(columns=["horizon", "mape", "coverage"])

    rows = []
    max_h = max(1, min(int(max_horizon), 6))
    for h in range(1, max_h + 1):
        all_mapes = []
        all_cov = []
        min_train = max(8, len(ts) - 24)
        for split in range(min_train, len(ts) - h + 1):
            train = ts.iloc[:split].copy()
            test = ts.iloc[split : split + h].copy()
            if len(test) < h:
                continue
            fc = forecast_with_uncertainty(train, horizon=h)
            merged = test[["month_start", "funded_amnt_m"]].merge(fc, on="month_start", how="left")
            y = pd.to_numeric(merged["funded_amnt_m"], errors="coerce").values
            pred = pd.to_numeric(merged["forecast_central"], errors="coerce").values
            low = pd.to_numeric(merged["forecast_lower"], errors="coerce").values
            high = pd.to_numeric(merged["forecast_upper"], errors="coerce").values
            if np.isnan(pred).any() or np.isnan(y).any():
                continue
            all_mapes.append(_mape(y, pred) * 100)
            covered = ((y >= low) & (y <= high)).mean() * 100
            all_cov.append(float(covered))

        if all_mapes:
            rows.append({"horizon": h, "mape": round(float(np.mean(all_mapes)), 2), "coverage": round(float(np.mean(all_cov)), 2)})
    return pd.DataFrame(rows)


def forecast_qa_snapshot(monthly_df: pd.DataFrame, rolling_df: pd.DataFrame) -> Dict[str, float]:
    ts = monthly_df.sort_values("month_start").reset_index(drop=True)
    missing_ratio = float(ts["funded_amnt_m"].isna().mean())
    latest_date = pd.to_datetime(ts["month_start"].iloc[-1])
    lag_days = float((pd.Timestamp.now().normalize() - latest_date).days)
    avg_mape = float(rolling_df["mape"].mean()) if not rolling_df.empty else np.nan
    avg_cov = float(rolling_df["coverage"].mean()) if not rolling_df.empty else np.nan
    return {
        "records": int(len(ts)),
        "missing_ratio": round(missing_ratio * 100, 2),
        "data_lag_days": round(lag_days, 1),
        "avg_rolling_mape": round(avg_mape, 2) if not np.isnan(avg_mape) else np.nan,
        "avg_rolling_coverage": round(avg_cov, 2) if not np.isnan(avg_cov) else np.nan,
    }


def anomaly_root_cause_hint(monthly_df: pd.DataFrame, anomaly_row: pd.Series) -> str:
    ts = monthly_df.sort_values("month_start").reset_index(drop=True)
    recent = pd.to_numeric(ts["funded_amnt_m"], errors="coerce").ffill().bfill().tail(6)
    cv = float((recent.std() / max(recent.mean(), 1e-6))) if len(recent) > 1 else 0.0
    month = int(pd.to_datetime(anomaly_row.get("month_start")).month)
    seasonal_avg = ts.groupby(ts["month_start"].dt.month)["funded_amnt_m"].mean().to_dict().get(month, np.nan)
    actual = float(anomaly_row.get("funded_amnt_m", np.nan))
    reasons = []
    if not np.isnan(seasonal_avg):
        if actual > seasonal_avg * 1.12:
            reasons.append("seasonal demand burst")
        elif actual < seasonal_avg * 0.88:
            reasons.append("seasonal slowdown")
    if cv > 0.18:
        reasons.append("high recent volatility")
    direction = str(anomaly_row.get("direction", "")).lower()
    if "spike" in direction:
        reasons.append("possible campaign or one-off surge")
    if "dip" in direction:
        reasons.append("possible operational or demand drop")
    if not reasons:
        reasons.append("unusual deviation from recent baseline")
    return ", ".join(dict.fromkeys(reasons))


def recommend_next_action(confidence_label: str, anomalies: pd.DataFrame, forecast_growth: float) -> str:
    if not anomalies.empty:
        latest = anomalies.iloc[-1]
        return (
            f"Priority: investigate the latest {str(latest['direction']).lower()} signal and validate source systems for "
            f"{pd.to_datetime(latest['month_start']).strftime('%Y-%m')}."
        )
    if confidence_label == "Unstable":
        return "Priority: rely on baseline models for planning and retrain after reviewing drift and data quality."
    if forecast_growth > 0.05:
        return "Priority: prepare capacity for expected growth and monitor conversion quality weekly."
    if forecast_growth < -0.03:
        return "Priority: activate retention and acquisition checks to mitigate expected slowdown."
    return "Priority: maintain current plan and monitor weekly forecast deltas for early movement."


def generate_forecast_brief_md(
    horizon: int,
    forecast_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    confidence_label: str,
    confidence_score: float,
    anomalies: pd.DataFrame,
    qa_snapshot: Dict[str, float],
    action_text: str,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    first = forecast_df.iloc[0]
    last = forecast_df.iloc[-1]
    base_last = float(baseline_df["naive_last"].iloc[-1]) if "naive_last" in baseline_df.columns else float(last["forecast_central"])
    delta_vs_baseline = float(last["forecast_central"] - base_last)
    anomaly_text = "No recent out-of-band spikes/dips detected."
    if not anomalies.empty:
        la = anomalies.iloc[-1]
        anomaly_text = (
            f"Recent {str(la['direction']).lower()} detected on {pd.to_datetime(la['month_start']).strftime('%Y-%m')} "
            f"(actual {float(la['funded_amnt_m']):.2f}, expected band {float(la['lower']):.2f}-{float(la['upper']):.2f})."
        )

    return f"""# FinSight Forecast Brief

Generated: {now}

## Short-Horizon Outlook ({horizon} months)

- First period likely forecast: **{float(first['forecast_central']):.2f}**
- Final period likely forecast: **{float(last['forecast_central']):.2f}**
- Final range: **{float(last['forecast_lower']):.2f}** to **{float(last['forecast_upper']):.2f}**
- Delta vs naive baseline (final period): **{delta_vs_baseline:+.2f}**

## Confidence

- Confidence label: **{confidence_label}**
- Confidence score: **{confidence_score:.1f}/100**

## Early Warning Signals

- {anomaly_text}

## QA Snapshot

- Records: **{qa_snapshot.get('records', 'N/A')}**
- Missing ratio: **{qa_snapshot.get('missing_ratio', 'N/A')}%**
- Data lag: **{qa_snapshot.get('data_lag_days', 'N/A')} days**
- Rolling MAPE: **{qa_snapshot.get('avg_rolling_mape', 'N/A')}%**
- Rolling interval coverage: **{qa_snapshot.get('avg_rolling_coverage', 'N/A')}%**

## Recommended Next Action

{action_text}

---

Generated by FinSight Forecasting QA layer.
"""
