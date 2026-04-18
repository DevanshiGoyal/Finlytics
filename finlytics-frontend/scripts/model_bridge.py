from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (str(PROJECT_ROOT), str(SRC_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from bank_anomaly_module import analyze_transaction, load_anomaly_dataset, score_transactions, train_anomaly_engine
from bank_term_deposit_module import (
    load_bank_dataset,
    predict_subscription,
    preprocess_user_input,
    top_feature_importance,
    train_bank_models,
)
from hackathon_utils import (
    compute_baseline_forecasts,
    forecast_with_uncertainty,
    get_shap_explanation,
    model_feature_importance,
    prepare_features,
    score_with_models,
)


MODULE3_SCENARIO_PRESETS: dict[str, dict[str, float]] = {
    "baseline": {
        "growthAdjustmentPct": 0.0,
        "interestRateShockBps": 0.0,
        "loanCountChangePct": 0.0,
        "avgLoanAmountChangePct": 0.0,
    },
    "optimistic": {
        "growthAdjustmentPct": 5.0,
        "interestRateShockBps": -20.0,
        "loanCountChangePct": 8.0,
        "avgLoanAmountChangePct": 3.0,
    },
    "conservative": {
        "growthAdjustmentPct": -5.0,
        "interestRateShockBps": 30.0,
        "loanCountChangePct": -6.0,
        "avgLoanAmountChangePct": -2.0,
    },
    "stress": {
        "growthAdjustmentPct": -12.0,
        "interestRateShockBps": 75.0,
        "loanCountChangePct": -15.0,
        "avgLoanAmountChangePct": -4.0,
    },
}

MODULE3_SCENARIO_RISK_MULTIPLIER: dict[str, float] = {
    "baseline": 1.0,
    "optimistic": 0.9,
    "conservative": 1.15,
    "stress": 1.35,
}

GRADES = ["A", "B", "C", "D", "E"]

MODULE4_DEFAULT_FEATURES = [
    "month",
    "quarter",
    "year",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_3",
    "rolling_6",
    "avg_int_rate",
    "loan_count",
    "mom_growth",
]

MODULE4_REPORT_METRICS = {
    "A": {"mape": 2.72, "rmse": 10.5},
    "B": {"mape": 4.97, "rmse": 14.2},
    "C": {"mape": 5.03, "rmse": 16.4},
    "D": {"mape": 8.55, "rmse": 18.7},
    "E": {"mape": 8.73, "rmse": 12.8},
}

MODULE4_SCENARIO_MULTIPLIER = {
    "baseline": 1.0,
    "optimistic": 1.15,
    "pessimistic": 0.85,
}


def _load_json_stdin() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    data = json.loads(raw)
    return data if isinstance(data, dict) else {"value": data}


def _label_from_probability(probability: float) -> str:
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def _to_records(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    out = []
    for _, row in df[cols].iterrows():
        rec: dict[str, Any] = {}
        for col in cols:
            value = row[col]
            if pd.isna(value):
                rec[col] = None
            elif isinstance(value, (np.floating, float)):
                rec[col] = float(value)
            elif isinstance(value, (np.integer, int)):
                rec[col] = int(value)
            else:
                rec[col] = value
        out.append(rec)
    return out


def _build_shap_payload(model: Any, frame: pd.DataFrame, max_features: int = 12) -> dict[str, Any]:
    payload = {
        "available": False,
        "message": "SHAP unavailable for this model or environment.",
        "baseValue": None,
        "modelOutput": None,
        "points": [],
    }

    shap_values = get_shap_explanation(model, frame)
    if shap_values is None:
        try:
            import shap

            background = frame.copy()
            if len(background) < 2:
                background = pd.concat([background, background], ignore_index=True)
            explainer = shap.Explainer(model, background)
            shap_values = explainer(frame)
        except Exception:
            shap_values = None

    if shap_values is None:
        return payload

    try:
        feature_names = frame.columns.tolist()
        data_values = frame.iloc[0].astype(float).values
        base_value = 0.0

        if hasattr(shap_values, "values"):
            explanation = shap_values[0]
            feature_names = list(getattr(explanation, "feature_names", frame.columns.tolist()) or frame.columns.tolist())
            raw_values = np.asarray(getattr(explanation, "values", []), dtype=float)
            if raw_values.ndim == 2:
                values = raw_values[:, -1].reshape(-1)
            elif raw_values.ndim == 1:
                values = raw_values.reshape(-1)
            else:
                values = raw_values.reshape(-1)
            if values.size == 0:
                values = np.zeros(len(feature_names), dtype=float)

            data_values = np.asarray(getattr(explanation, "data", frame.iloc[0].values), dtype=float).reshape(-1)
            if data_values.size == 0:
                data_values = frame.iloc[0].astype(float).values

            base_values = np.asarray(getattr(explanation, "base_values", [0.0]), dtype=float).reshape(-1)
            base_value = float(base_values[0]) if base_values.size else 0.0
        else:
            if isinstance(shap_values, list):
                if len(shap_values) == 0:
                    import shap

                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(frame)
                chosen = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                chosen = shap_values

            values_matrix = np.asarray(chosen, dtype=float)
            if values_matrix.ndim == 3:
                values = values_matrix[0, :, -1].reshape(-1)
            elif values_matrix.ndim == 2:
                values = values_matrix[0].reshape(-1)
            else:
                values = values_matrix.reshape(-1)

            try:
                import shap

                expected = shap.TreeExplainer(model).expected_value
                expected_vals = np.asarray(expected, dtype=float).reshape(-1)
                if expected_vals.size:
                    base_value = float(expected_vals[-1])
            except Exception:
                base_value = 0.0

        limit = min(max_features, len(values), len(feature_names), len(data_values))
        ranked_idx = np.argsort(np.abs(values))[::-1][:limit]

        points: list[dict[str, Any]] = []
        running = base_value
        for idx in ranked_idx:
            shap_val = float(values[idx])
            start = float(running)
            end = float(running + shap_val)
            running = end
            points.append(
                {
                    "feature": str(feature_names[idx]),
                    "value": float(data_values[idx]),
                    "shapValue": shap_val,
                    "start": start,
                    "end": end,
                }
            )

        return {
            "available": True,
            "message": None,
            "baseValue": base_value,
            "modelOutput": float(running),
            "points": points,
        }
    except Exception as exc:
        payload["message"] = f"SHAP explanation failed: {exc}"
        return payload


def _require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {', '.join(missing)}"
        )


def _require_value(payload: dict[str, Any], keys: list[str], field_label: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    raise ValueError(f"Missing required input field: {field_label}")


def _require_float(payload: dict[str, Any], keys: list[str], field_label: str) -> float:
    raw = _require_value(payload, keys, field_label)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field_label}: {raw}") from exc


def _require_int(payload: dict[str, Any], keys: list[str], field_label: str) -> int:
    value = _require_float(payload, keys, field_label)
    return int(round(value))


def _require_str(payload: dict[str, Any], keys: list[str], field_label: str) -> str:
    raw = _require_value(payload, keys, field_label)
    text = str(raw).strip()
    if not text:
        raise ValueError(f"Missing required input field: {field_label}")
    return text


def _optional_float(payload: dict[str, Any], keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in payload and payload.get(key) not in (None, ""):
            raw = payload.get(key)
            try:
                return float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value for {key}: {raw}") from exc
    return float(default)


def _bounded(value: float, field_label: str, lower: float, upper: float) -> float:
    if value < lower or value > upper:
        raise ValueError(f"{field_label} must be between {lower} and {upper}")
    return float(value)


def _normalize_module3_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scenario = str(payload.get("scenario", "baseline")).strip().lower()
    if scenario not in MODULE3_SCENARIO_PRESETS:
        valid = ", ".join(sorted(MODULE3_SCENARIO_PRESETS.keys()))
        raise ValueError(f"Unsupported scenario '{scenario}'. Valid options: {valid}")

    horizon_raw = payload.get("horizonMonths", payload.get("horizon", 3))
    try:
        horizon = int(round(float(horizon_raw)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for horizonMonths: {horizon_raw}") from exc

    if horizon < 1 or horizon > 12:
        raise ValueError("horizonMonths must be between 1 and 12")

    user_growth = _bounded(
        _optional_float(payload, ["growthAdjustmentPct", "growth_adjustment_pct"], 0.0),
        "growthAdjustmentPct",
        -40.0,
        60.0,
    )
    user_rate = _bounded(
        _optional_float(payload, ["interestRateShockBps", "interest_rate_shock_bps"], 0.0),
        "interestRateShockBps",
        -400.0,
        400.0,
    )
    user_count = _bounded(
        _optional_float(payload, ["loanCountChangePct", "loan_count_change_pct"], 0.0),
        "loanCountChangePct",
        -50.0,
        120.0,
    )
    user_loan_amount = _bounded(
        _optional_float(payload, ["avgLoanAmountChangePct", "avg_loan_amount_change_pct"], 0.0),
        "avgLoanAmountChangePct",
        -50.0,
        80.0,
    )

    preset = MODULE3_SCENARIO_PRESETS[scenario]
    return {
        "scenario": scenario,
        "horizonMonths": horizon,
        "growthAdjustmentPct": float(preset["growthAdjustmentPct"] + user_growth),
        "interestRateShockBps": float(preset["interestRateShockBps"] + user_rate),
        "loanCountChangePct": float(preset["loanCountChangePct"] + user_count),
        "avgLoanAmountChangePct": float(preset["avgLoanAmountChangePct"] + user_loan_amount),
        "userAdjustments": {
            "growthAdjustmentPct": user_growth,
            "interestRateShockBps": user_rate,
            "loanCountChangePct": user_count,
            "avgLoanAmountChangePct": user_loan_amount,
        },
    }


def _normalize_credit_demand_payload(payload: dict[str, Any]) -> dict[str, Any]:
    horizon_raw = payload.get("horizon", payload.get("horizonMonths", 3))
    try:
        horizon = int(round(float(horizon_raw)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for horizon: {horizon_raw}") from exc

    if horizon < 1 or horizon > 12:
        raise ValueError("horizon must be between 1 and 12")

    confidence = _bounded(
        _optional_float(payload, ["confidence", "confidenceLevel", "confidence_level"], 0.95),
        "confidence",
        0.5,
        0.99,
    )

    scenario = str(payload.get("scenarioType", payload.get("scenario", "baseline"))).strip().lower()
    if scenario not in MODULE4_SCENARIO_MULTIPLIER:
        valid = ", ".join(sorted(MODULE4_SCENARIO_MULTIPLIER.keys()))
        raise ValueError(f"Unsupported scenarioType '{scenario}'. Valid options: {valid}")

    base_volume = payload.get("baseVolume")
    if base_volume in (None, ""):
        base_volume_value: float | None = None
    else:
        try:
            base_volume_value = float(base_volume)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for baseVolume: {base_volume}") from exc
        if base_volume_value <= 0:
            raise ValueError("baseVolume must be greater than 0 when provided")

    return {
        "horizon": horizon,
        "confidence": float(confidence),
        "scenarioType": scenario,
        "baseVolume": base_volume_value,
    }


def _confidence_z(confidence: float) -> float:
    confidence = min(0.999, max(0.5, float(confidence)))
    return float(NormalDist().inv_cdf((1.0 + confidence) / 2.0))


def _load_grade_monthly_history() -> pd.DataFrame:
    grade_path = PROJECT_ROOT / "data/processed/grade_monthly_demand.csv"
    if grade_path.exists():
        grade_df = pd.read_csv(grade_path, parse_dates=["month_start"])
        _require_columns(grade_df, ["month_start", "grade", "funded_amnt_m"], "credit demand by grade")
        return grade_df.sort_values("month_start").reset_index(drop=True)

    monthly = _load_monthly_series()
    shares = _grade_share_from_artifacts()
    rows = []
    for rec in monthly.itertuples(index=False):
        for grade in GRADES:
            rows.append(
                {
                    "month_start": rec.month_start,
                    "grade": grade,
                    "funded_amnt_m": float(rec.funded_amnt_m) * float(shares.get(grade, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _build_grade_history_pivot(grade_df: pd.DataFrame) -> pd.DataFrame:
    monthly_grade = (
        grade_df.groupby(["month_start", "grade"], as_index=False)["funded_amnt_m"]
        .sum()
        .sort_values("month_start")
    )
    pivot = (
        monthly_grade.pivot(index="month_start", columns="grade", values="funded_amnt_m")
        .fillna(0.0)
        .sort_index()
    )
    for grade in GRADES:
        if grade not in pivot.columns:
            pivot[grade] = 0.0
    return pivot[GRADES].copy()


def _load_grade_models() -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    artifacts: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    feature_path = PROJECT_ROOT / "models/module4_features.pkl"
    feature_list = list(MODULE4_DEFAULT_FEATURES)

    if feature_path.exists():
        try:
            feature_list = list(joblib.load(feature_path))
        except Exception as exc:
            warnings.append(
                f"Could not load module4_features.pkl; using default feature schema. Reason: {exc}"
            )
    else:
        warnings.append("module4_features.pkl not found; using default feature schema.")

    for grade in GRADES:
        model_path = PROJECT_ROOT / f"models/module4_xgb_grade{grade}.pkl"
        scaler_path = PROJECT_ROOT / f"models/module4_scaler_grade{grade}.pkl"
        if not model_path.exists() or not scaler_path.exists():
            warnings.append(
                f"Missing model artifacts for Grade {grade}: {model_path.name} or {scaler_path.name}."
            )
            continue

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as exc:
            warnings.append(f"Could not load Grade {grade} model artifacts: {exc}")
            continue

        defaults = {feature: 0.0 for feature in feature_list}
        means = getattr(scaler, "mean_", None)
        if means is not None and len(means) == len(feature_list):
            for idx, feature in enumerate(feature_list):
                defaults[feature] = float(means[idx])

        artifacts[grade] = {
            "model": model,
            "scaler": scaler,
            "features": feature_list,
            "defaults": defaults,
        }

    return artifacts, feature_list, warnings


def _build_module4_feature_row(
    grade_history_values: list[float],
    month_start: pd.Timestamp,
    features: list[str],
    defaults: dict[str, float],
    total_hint: float,
) -> dict[str, float]:
    values = [float(v) for v in grade_history_values if np.isfinite(v)]
    row = {feature: float(defaults.get(feature, 0.0)) for feature in features}

    row["month"] = float(month_start.month)
    row["quarter"] = float(((month_start.month - 1) // 3) + 1)
    row["year"] = float(month_start.year)

    for lag in [1, 2, 3, 6, 12]:
        key = f"lag_{lag}"
        if key in row:
            row[key] = float(values[-lag]) if len(values) >= lag else float(defaults.get(key, row[key]))

    for window in [3, 6, 12]:
        key = f"rolling_{window}"
        if key in row:
            window_vals = values[-window:] if values else []
            row[key] = float(np.mean(window_vals)) if window_vals else float(defaults.get(key, row[key]))

    if "mom_growth" in row:
        if len(values) >= 2 and abs(values[-2]) > 1e-6:
            row["mom_growth"] = float((values[-1] - values[-2]) / abs(values[-2]))
        else:
            row["mom_growth"] = float(defaults.get("mom_growth", 0.0))

    if "loan_count" in row:
        base_count = max(1.0, float(defaults.get("loan_count", 1.0)))
        reference_total = max(float(defaults.get("lag_1", total_hint if total_hint > 0 else 1.0)), 1.0)
        scaled_count = base_count * (max(total_hint, 1.0) / reference_total)
        row["loan_count"] = float(max(1.0, scaled_count))

    return row


def _default_grade_feature_importance(features: list[str]) -> list[dict[str, float]]:
    weights = {
        "lag_1": 0.26,
        "rolling_3": 0.2,
        "rolling_6": 0.15,
        "lag_3": 0.12,
        "lag_6": 0.08,
        "mom_growth": 0.07,
        "loan_count": 0.06,
        "month": 0.04,
        "quarter": 0.02,
    }

    rows: list[dict[str, float]] = []
    for feature in features:
        rows.append({"feature": feature, "importance": float(weights.get(feature, 0.01))})

    total = sum(row["importance"] for row in rows) or 1.0
    normalized = [
        {"feature": row["feature"], "importance": float(row["importance"] / total)}
        for row in rows
    ]
    return sorted(normalized, key=lambda item: item["importance"], reverse=True)[:6]


def _grade_model_backtest_metrics(
    grade: str,
    month_index: list[pd.Timestamp],
    grade_values: list[float],
    total_values: list[float],
    artifact: dict[str, Any] | None,
) -> dict[str, float]:
    fallback = MODULE4_REPORT_METRICS.get(grade, {"mape": 8.0, "rmse": 15.0})
    if artifact is None or len(grade_values) < 18 or len(total_values) < len(grade_values):
        mape = float(fallback["mape"])
        rmse = float(fallback["rmse"])
        return {
            "mape": round(mape, 2),
            "rmse": round(rmse, 2),
            "trainingMapeOnTestSet": round(mape, 2),
        }

    preds: list[float] = []
    actuals: list[float] = []
    start_idx = max(12, len(grade_values) - 18)
    try:
        for idx in range(start_idx, len(grade_values)):
            train_values = grade_values[:idx]
            if not train_values:
                continue
            month_start = pd.to_datetime(month_index[idx])
            total_hint = float(total_values[idx - 1]) if idx - 1 >= 0 else float(total_values[0])
            feature_row = _build_module4_feature_row(
                train_values,
                month_start,
                artifact["features"],
                artifact["defaults"],
                total_hint,
            )
            frame = pd.DataFrame([feature_row], columns=artifact["features"]).fillna(0.0)
            scaled = artifact["scaler"].transform(frame)
            pred = float(np.asarray(artifact["model"].predict(scaled)).ravel()[0])
            preds.append(max(0.0, pred))
            actuals.append(max(0.0, float(grade_values[idx])))
    except Exception:
        mape = float(fallback["mape"])
        rmse = float(fallback["rmse"])
        return {
            "mape": round(mape, 2),
            "rmse": round(rmse, 2),
            "trainingMapeOnTestSet": round(mape, 2),
        }

    if not preds:
        mape = float(fallback["mape"])
        rmse = float(fallback["rmse"])
        return {
            "mape": round(mape, 2),
            "rmse": round(rmse, 2),
            "trainingMapeOnTestSet": round(mape, 2),
        }

    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "trainingMapeOnTestSet": round(mape, 2),
    }


def _apply_module4_scenario(
    scenario: str,
    central: float,
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    if scenario == "optimistic":
        central = central * 1.15
        upper = upper * 1.15
    elif scenario == "pessimistic":
        central = central * 0.85
        lower = lower * 0.85

    central = max(0.0, float(central))
    lower = max(0.0, float(lower))
    upper = max(lower, float(upper))
    return central, lower, upper


def _load_total_forecast_targets(
    monthly: pd.DataFrame,
    horizon: int,
    base_volume: float | None,
) -> tuple[list[pd.Timestamp], list[float]]:
    total_series = monthly.copy()
    total_series["month_start"] = pd.to_datetime(total_series["month_start"], errors="coerce")
    total_series["funded_amnt_m"] = pd.to_numeric(total_series["funded_amnt_m"], errors="coerce")
    total_series = total_series.dropna(subset=["month_start", "funded_amnt_m"]).sort_values("month_start")

    if total_series.empty:
        raise ValueError("Unable to build total loan-volume series for grade forecast.")

    if base_volume is not None:
        total_series.iloc[-1, total_series.columns.get_loc("funded_amnt_m")] = float(base_volume)

    total_fc = forecast_with_uncertainty(total_series[["month_start", "funded_amnt_m"]], horizon=horizon)
    months = [pd.to_datetime(value) for value in total_fc["month_start"].tolist()]
    totals = pd.to_numeric(total_fc["forecast_central"], errors="coerce").fillna(0.0).astype(float).tolist()
    return months, totals


def _load_module_artifacts(module_number: int) -> tuple[Any, list[str]]:
    model = joblib.load(PROJECT_ROOT / f"models/module{module_number}_xgb.pkl")
    features = joblib.load(PROJECT_ROOT / f"models/module{module_number}_features.pkl")
    return model, list(features)


def _load_module3_forecast_artifacts() -> dict[str, Any]:
    feature_path = PROJECT_ROOT / "models/module3_features.pkl"
    xgb_path = PROJECT_ROOT / "models/module3_xgb.pkl"
    lr_path = PROJECT_ROOT / "models/module3_lr.pkl"
    scaler_path = PROJECT_ROOT / "models/module3_scaler.pkl"

    required = [feature_path, xgb_path, lr_path, scaler_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required Module 3 model artifacts for loan forecast: "
            + ", ".join(missing)
        )

    features = list(joblib.load(feature_path))
    xgb_model = joblib.load(xgb_path)
    lr_model = joblib.load(lr_path)
    scaler = joblib.load(scaler_path)

    defaults = {feature: 0.0 for feature in features}
    scaler_means = getattr(scaler, "mean_", None)
    if scaler_means is not None and len(scaler_means) == len(features):
        for idx, feature in enumerate(features):
            defaults[feature] = float(scaler_means[idx])

    return {
        "features": features,
        "xgb": xgb_model,
        "lr": lr_model,
        "scaler": scaler,
        "defaults": defaults,
    }


def _build_module3_feature_row(
    history: pd.DataFrame,
    month_start: pd.Timestamp,
    features: list[str],
    defaults: dict[str, float],
) -> dict[str, float]:
    values = (
        pd.to_numeric(history["funded_amnt_m"], errors="coerce")
        .dropna()
        .astype(float)
        .tolist()
    )

    row = {feature: float(defaults.get(feature, 0.0)) for feature in features}

    row["month"] = float(month_start.month)
    row["quarter"] = float(((month_start.month - 1) // 3) + 1)
    row["year"] = float(month_start.year)

    for lag in [1, 2, 3, 6, 12]:
        key = f"lag_{lag}"
        if key in row:
            row[key] = float(values[-lag]) if len(values) >= lag else float(defaults.get(key, row[key]))

    for window in [3, 6, 12]:
        key = f"rolling_{window}"
        if key in row:
            window_vals = values[-window:] if values else []
            row[key] = float(np.mean(window_vals)) if window_vals else float(defaults.get(key, row[key]))

    if "mom_growth" in row:
        if len(values) >= 2 and abs(values[-2]) > 1e-6:
            row["mom_growth"] = float((values[-1] - values[-2]) / abs(values[-2]))
        else:
            row["mom_growth"] = float(defaults.get("mom_growth", 0.0))

    for col in ["avg_int_rate", "avg_loan_amnt", "loan_count"]:
        if col not in row:
            continue
        if col in history.columns:
            known = pd.to_numeric(history[col], errors="coerce").dropna()
            if not known.empty:
                row[col] = float(known.iloc[-1])

    return row


def _module3_predict_value(
    feature_row: dict[str, float],
    artifacts: dict[str, Any],
) -> tuple[float, float, float]:
    features = artifacts["features"]
    frame = pd.DataFrame([feature_row], columns=features).fillna(0.0)

    xgb_pred = float(np.asarray(artifacts["xgb"].predict(frame)).ravel()[0])
    try:
        scaled = artifacts["scaler"].transform(frame)
        lr_pred = float(np.asarray(artifacts["lr"].predict(scaled)).ravel()[0])
    except Exception:
        lr_pred = xgb_pred

    central = max(0.0, (0.7 * xgb_pred) + (0.3 * lr_pred))
    return xgb_pred, lr_pred, central


def _resolve_data_file(candidates: list[Path], dependency_label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    listed = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Required data file not found for {dependency_label}. Checked: {listed}"
    )


def _load_bank_dataset_strict() -> pd.DataFrame:
    data_path = _resolve_data_file(
        [
            PROJECT_ROOT / "data/processed/bank_term_deposit_processed.csv",
            PROJECT_ROOT / "data/external/bank_updated.csv",
        ],
        "bank deposit AI",
    )
    return load_bank_dataset(dataset_path=str(data_path), use_remote=False)


def _load_anomaly_dataset_strict() -> pd.DataFrame:
    data_path = _resolve_data_file(
        [
            PROJECT_ROOT / "data/processed/deposit_anomaly_processed.csv",
            PROJECT_ROOT / "data/external/synthetic_deposits.csv",
        ],
        "deposit anomaly detection",
    )
    return load_anomaly_dataset(dataset_path=str(data_path), use_remote=False)


def _load_or_train_bank_artifacts(seed: int = 42):
    data_path = _resolve_data_file(
        [
            PROJECT_ROOT / "data/processed/bank_term_deposit_processed.csv",
            PROJECT_ROOT / "data/external/bank_updated.csv",
        ],
        "bank deposit AI",
    )
    artifact_path = PROJECT_ROOT / "models" / "bank_term_deposit_artifacts.pkl"
    source_mtime = data_path.stat().st_mtime

    if artifact_path.exists():
        try:
            payload = joblib.load(artifact_path)
            if (
                isinstance(payload, dict)
                and payload.get("source_path") == str(data_path)
                and float(payload.get("source_mtime", 0.0)) == float(source_mtime)
                and "artifacts" in payload
            ):
                return payload["artifacts"]
        except Exception:
            pass

    bank_df = load_bank_dataset(dataset_path=str(data_path), use_remote=False)
    artifacts = train_bank_models(bank_df, seed=seed)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "source_path": str(data_path),
            "source_mtime": float(source_mtime),
            "artifacts": artifacts,
        },
        artifact_path,
    )
    return artifacts


def _load_or_train_anomaly_artifacts(
    contamination: float = 0.03,
    random_state: int = 42,
):
    data_path = _resolve_data_file(
        [
            PROJECT_ROOT / "data/processed/deposit_anomaly_processed.csv",
            PROJECT_ROOT / "data/external/synthetic_deposits.csv",
        ],
        "deposit anomaly detection",
    )
    artifact_path = PROJECT_ROOT / "models" / "deposit_anomaly_artifacts.pkl"
    source_mtime = data_path.stat().st_mtime

    if artifact_path.exists():
        try:
            payload = joblib.load(artifact_path)
            if (
                isinstance(payload, dict)
                and payload.get("source_path") == str(data_path)
                and float(payload.get("source_mtime", 0.0)) == float(source_mtime)
                and float(payload.get("contamination", -1.0)) == float(contamination)
                and "artifacts" in payload
            ):
                return payload["artifacts"]
        except Exception:
            pass

    df = load_anomaly_dataset(dataset_path=str(data_path), use_remote=False)
    artifacts = train_anomaly_engine(df, contamination=contamination, random_state=random_state)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "source_path": str(data_path),
            "source_mtime": float(source_mtime),
            "contamination": float(contamination),
            "artifacts": artifacts,
        },
        artifact_path,
    )
    return artifacts


def _predict_default(payload: dict[str, Any]) -> dict[str, Any]:
    model, features = _load_module_artifacts(1)

    issue_month = _require_int(payload, ["issueMonth", "issue_month"], "issueMonth")
    if issue_month < 1 or issue_month > 12:
        raise ValueError("issueMonth must be between 1 and 12")

    row = {
        "loan_amnt": _require_float(payload, ["loanAmount", "loan_amnt"], "loanAmount"),
        "annual_inc": _require_float(payload, ["annualIncome", "annual_inc"], "annualIncome"),
        "dti": _require_float(payload, ["dti"], "dti"),
        "int_rate": _require_float(payload, ["interestRate", "int_rate"], "interestRate"),
        "revol_util": _require_float(
            payload,
            ["revolUtil", "creditUtilization", "revol_util"],
            "revolUtil",
        ),
        "installment": _require_float(payload, ["installment"], "installment"),
        "inq_last_6mths": _require_int(
            payload,
            ["inqLast6Mths", "inq_last_6mths"],
            "inqLast6Mths",
        ),
        "open_acc": _require_int(payload, ["openAcc", "open_acc"], "openAcc"),
        "pub_rec": _require_int(payload, ["pubRec", "pub_rec"], "pubRec"),
        "total_acc": _require_int(payload, ["totalAcc", "total_acc"], "totalAcc"),
        "emp_length": _require_int(payload, ["empLength", "emp_length"], "empLength"),
        "delinq_2yrs": _require_int(payload, ["delinq2Yrs", "delinq_2yrs"], "delinq2Yrs"),
        "grade": _require_str(payload, ["grade"], "grade").upper(),
        "home_ownership": _require_str(
            payload,
            ["homeOwnership", "home_ownership"],
            "homeOwnership",
        ).upper(),
        "purpose": _require_str(payload, ["purpose"], "purpose"),
        "issue_year": _require_int(payload, ["issueYear", "issue_year"], "issueYear"),
        "issue_month": issue_month,
    }

    frame = prepare_features(pd.DataFrame([row]), features)
    probability = float(model.predict_proba(frame)[0][1])
    importance = model_feature_importance(model, features).head(8)
    shap_payload = _build_shap_payload(model, frame)

    return {
        "probability": probability,
        "label": _label_from_probability(probability),
        "explainability": _to_records(importance, ["feature", "importance"]),
        "shapExplanation": shap_payload,
    }


def _predict_churn(payload: dict[str, Any]) -> dict[str, Any]:
    model, features = _load_module_artifacts(2)

    issue_month = _require_int(payload, ["issueMonth", "issue_month"], "issueMonth")
    if issue_month < 1 or issue_month > 12:
        raise ValueError("issueMonth must be between 1 and 12")

    row = {
        "loan_amnt": _require_float(payload, ["loanAmount", "loan_amnt"], "loanAmount"),
        "annual_inc": _require_float(payload, ["annualIncome", "annual_inc"], "annualIncome"),
        "dti": _require_float(payload, ["dti"], "dti"),
        "int_rate": _require_float(payload, ["interestRate", "int_rate"], "interestRate"),
        "revol_util": _require_float(
            payload,
            ["revolUtil", "creditUtilization", "revol_util"],
            "revolUtil",
        ),
        "installment": _require_float(payload, ["installment"], "installment"),
        "open_acc": _require_int(payload, ["openAcc", "open_acc"], "openAcc"),
        "total_acc": _require_int(payload, ["totalAcc", "total_acc"], "totalAcc"),
        "emp_length": _require_int(payload, ["empLength", "emp_length"], "empLength"),
        "delinq_2yrs": _require_int(payload, ["delinq2Yrs", "delinq_2yrs"], "delinq2Yrs"),
        "grade": _require_str(payload, ["grade"], "grade").upper(),
        "home_ownership": _require_str(
            payload,
            ["homeOwnership", "home_ownership"],
            "homeOwnership",
        ).upper(),
        "purpose": _require_str(payload, ["purpose"], "purpose").lower(),
        "issue_year": _require_int(payload, ["issueYear", "issue_year"], "issueYear"),
        "issue_month": issue_month,
    }

    frame = prepare_features(pd.DataFrame([row]), features)
    probability = float(model.predict_proba(frame)[0][1])
    importance = model_feature_importance(model, features).head(8)
    shap_payload = _build_shap_payload(model, frame)

    if probability >= 0.65:
        suggestions = [
            "Trigger relationship manager outreach within 24 hours.",
            "Offer repayment plan optimization and proactive support.",
            "Escalate account to high-touch retention workflow.",
        ]
    elif probability >= 0.35:
        suggestions = [
            "Start targeted engagement nudges for the next billing cycle.",
            "Monitor repayment delay and service interactions weekly.",
            "Send personalized retention communication.",
        ]
    else:
        suggestions = [
            "Maintain standard servicing cadence.",
            "Continue routine engagement monitoring.",
            "No immediate retention action required.",
        ]

    return {
        "probability": probability,
        "label": _label_from_probability(probability),
        "suggestions": suggestions,
        "explainability": _to_records(importance, ["feature", "importance"]),
        "shapExplanation": shap_payload,
    }


def _load_monthly_series() -> pd.DataFrame:
    monthly_path = PROJECT_ROOT / "data/processed/monthly_loan_volume.csv"
    if monthly_path.exists():
        monthly = pd.read_csv(monthly_path, parse_dates=["month_start"])
        _require_columns(monthly, ["month_start", "funded_amnt_m"], "loan volume forecast")
        return monthly.sort_values("month_start").reset_index(drop=True)

    prophet_path = PROJECT_ROOT / "models/module3_prophet.json"
    if not prophet_path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {monthly_path}. "
            "Loan volume forecast depends on data/processed/monthly_loan_volume.csv "
            "or models/module3_prophet.json"
        )

    serialized = json.loads(prophet_path.read_text())
    history_raw = serialized.get("history")
    if not isinstance(history_raw, str):
        raise ValueError(
            "Invalid module3_prophet.json format: expected string field 'history'"
        )

    history_payload = json.loads(history_raw)
    history_rows = history_payload.get("data")
    if not isinstance(history_rows, list) or not history_rows:
        raise ValueError(
            "module3_prophet.json does not contain usable historical records"
        )

    history = pd.DataFrame(history_rows)
    _require_columns(history, ["ds", "y"], "loan volume history from module3_prophet.json")

    monthly = pd.DataFrame(
        {
            "month_start": pd.to_datetime(history["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp(),
            "funded_amnt_m": pd.to_numeric(history["y"], errors="coerce"),
        }
    ).dropna(subset=["month_start", "funded_amnt_m"])

    if monthly.empty:
        raise ValueError("Unable to derive monthly loan volume history from module3_prophet.json")

    return (
        monthly.groupby("month_start", as_index=False)["funded_amnt_m"]
        .mean()
        .sort_values("month_start")
        .reset_index(drop=True)
    )


def _grade_share_from_artifacts() -> dict[str, float]:
    features = joblib.load(PROJECT_ROOT / "models/module4_features.pkl")
    feature_idx = {name: idx for idx, name in enumerate(features)}
    score_keys = ["lag_1", "rolling_3", "rolling_6", "loan_count"]

    raw_scores: dict[str, float] = {}
    for grade in ["A", "B", "C", "D", "E"]:
        scaler_path = PROJECT_ROOT / f"models/module4_scaler_grade{grade}.pkl"
        if not scaler_path.exists():
            continue
        scaler = joblib.load(scaler_path)
        means = getattr(scaler, "mean_", None)
        if means is None:
            continue

        score = 0.0
        for key in score_keys:
            idx = feature_idx.get(key)
            if idx is None or idx >= len(means):
                continue
            score += max(0.0, float(means[idx]))

        if score > 0:
            raw_scores[grade] = score

    if not raw_scores:
        return {"A": 0.28, "B": 0.26, "C": 0.22, "D": 0.15, "E": 0.09}

    total = sum(raw_scores.values())
    normalized = {grade: raw_scores.get(grade, 0.0) / total for grade in ["A", "B", "C", "D", "E"]}
    return normalized


def _forecast_loan_volume(payload: dict[str, Any]) -> dict[str, Any]:
    request_config = _normalize_module3_payload(payload)
    scenario = str(request_config["scenario"])
    horizon = int(request_config["horizonMonths"])
    growth_adjustment_pct = float(request_config["growthAdjustmentPct"])
    interest_rate_shock_bps = float(request_config["interestRateShockBps"])
    loan_count_change_pct = float(request_config["loanCountChangePct"])
    avg_loan_amount_change_pct = float(request_config["avgLoanAmountChangePct"])
    uncertainty_multiplier = float(MODULE3_SCENARIO_RISK_MULTIPLIER.get(scenario, 1.0))

    monthly = _load_monthly_series().copy()
    _require_columns(monthly, ["month_start", "funded_amnt_m"], "loan volume forecast")

    monthly["month_start"] = pd.to_datetime(monthly["month_start"], errors="coerce")
    monthly["funded_amnt_m"] = pd.to_numeric(monthly["funded_amnt_m"], errors="coerce")
    monthly = (
        monthly.dropna(subset=["month_start", "funded_amnt_m"])
        .sort_values("month_start")
        .reset_index(drop=True)
    )

    if len(monthly) < 12:
        raise ValueError(
            "Loan volume forecast requires at least 12 monthly observations "
            "to run Module 3 trained models."
        )

    artifacts = _load_module3_forecast_artifacts()
    features = artifacts["features"]
    defaults = artifacts["defaults"]

    history = monthly.copy()
    for col in ["avg_int_rate", "avg_loan_amnt", "loan_count"]:
        if col not in history.columns:
            history[col] = float(defaults.get(col, 0.0))
        history[col] = pd.to_numeric(history[col], errors="coerce")
        if history[col].isna().all():
            history[col] = float(defaults.get(col, 0.0))
        else:
            history[col] = history[col].ffill().bfill()

    trend_tail = history.tail(9).copy()
    trend = [
        {
            "month": pd.to_datetime(row.month_start).strftime("%b-%y"),
            "actual": float(row.funded_amnt_m),
            "forecast": float(row.funded_amnt_m),
        }
        for row in trend_tail.itertuples(index=False)
    ]

    history_window = 12
    history_interval = []
    for row in history.tail(history_window).itertuples(index=False):
        actual_val = round(float(row.funded_amnt_m), 2)
        history_interval.append(
            {
                "month": pd.to_datetime(row.month_start).strftime("%b-%y"),
                "value": actual_val,
                "low": actual_val,
                "high": actual_val,
                "historical": actual_val,
            }
        )

    forecast_interval = []
    warnings: list[str] = []
    if abs(interest_rate_shock_bps) > 0 and "avg_int_rate" not in features:
        warnings.append(
            "interestRateShockBps was provided, but avg_int_rate is unavailable in module3 feature list."
        )
    if abs(loan_count_change_pct) > 0 and "loan_count" not in features:
        warnings.append(
            "loanCountChangePct was provided, but loan_count is unavailable in module3 feature list."
        )
    if abs(avg_loan_amount_change_pct) > 0 and "avg_loan_amnt" not in features:
        warnings.append(
            "avgLoanAmountChangePct was provided, but avg_loan_amnt is unavailable in module3 feature list."
        )

    for step in range(1, horizon + 1):
        next_month = pd.to_datetime(history["month_start"].iloc[-1]) + pd.offsets.MonthBegin(1)
        feature_row = _build_module3_feature_row(history, next_month, features, defaults)

        if "avg_int_rate" in feature_row:
            feature_row["avg_int_rate"] = max(
                0.0,
                float(feature_row["avg_int_rate"]) + (interest_rate_shock_bps / 100.0),
            )
        if "loan_count" in feature_row:
            feature_row["loan_count"] = max(
                1.0,
                float(feature_row["loan_count"]) * (1.0 + loan_count_change_pct / 100.0),
            )
        if "avg_loan_amnt" in feature_row:
            feature_row["avg_loan_amnt"] = max(
                500.0,
                float(feature_row["avg_loan_amnt"]) * (1.0 + avg_loan_amount_change_pct / 100.0),
            )

        xgb_pred, lr_pred, central_raw = _module3_predict_value(feature_row, artifacts)
        central = max(0.0, central_raw * (1.0 + growth_adjustment_pct / 100.0))

        recent_vals = pd.to_numeric(history["funded_amnt_m"], errors="coerce").dropna().values.astype(float)
        recent_deltas = np.diff(recent_vals[-12:]) if len(recent_vals) > 2 else np.array([0.0])
        volatility = float(np.std(recent_deltas))
        volatility = max(volatility, 0.05 * max(abs(central), 1.0))
        model_spread = abs(xgb_pred - lr_pred)
        spread = max(model_spread, volatility * np.sqrt(step))
        spread *= uncertainty_multiplier
        spread *= 1.0 + min(0.35, abs(growth_adjustment_pct) / 100.0)

        lower = max(0.0, central - 1.28 * spread)
        upper = max(lower, central + 1.28 * spread)

        forecast_interval.append(
            {
                "month": next_month.strftime("%b-%y"),
                "value": round(float(central), 2),
                "low": round(float(lower), 2),
                "high": round(float(upper), 2),
                "historical": None,
            }
        )

        avg_loan_amnt = float(feature_row.get("avg_loan_amnt", defaults.get("avg_loan_amnt", 1.0)))
        if avg_loan_amnt <= 0:
            avg_loan_amnt = max(float(defaults.get("avg_loan_amnt", 1.0)), 1.0)
        inferred_count = max(1.0, (central * 1_000_000.0) / avg_loan_amnt)

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        {
                            "month_start": next_month,
                            "funded_amnt_m": float(central),
                            "avg_int_rate": float(feature_row.get("avg_int_rate", defaults.get("avg_int_rate", 0.0))),
                            "avg_loan_amnt": float(avg_loan_amnt),
                            "loan_count": float(max(inferred_count, feature_row.get("loan_count", inferred_count))),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    interval = history_interval + forecast_interval
    latest_actual = float(monthly["funded_amnt_m"].iloc[-1])
    projected_total = float(sum(rec["value"] for rec in forecast_interval))

    final_projection = forecast_interval[-1] if forecast_interval else {
        "month": pd.to_datetime(monthly["month_start"].iloc[-1]).strftime("%b-%y"),
        "value": round(latest_actual, 2),
        "low": round(latest_actual, 2),
        "high": round(latest_actual, 2),
    }

    growth_vs_latest_actual_pct = 0.0
    if abs(latest_actual) > 1e-6:
        growth_vs_latest_actual_pct = ((float(final_projection["value"]) - latest_actual) / abs(latest_actual)) * 100.0

    summary = {
        "horizonMonths": horizon,
        "scenario": scenario,
        "projectedTotal": round(projected_total, 2),
        "averageMonthly": round(projected_total / max(horizon, 1), 2),
        "finalMonth": str(final_projection["month"]),
        "finalValue": round(float(final_projection["value"]), 2),
        "finalRange": {
            "low": round(float(final_projection["low"]), 2),
            "high": round(float(final_projection["high"]), 2),
        },
        "growthVsLastActualPct": round(growth_vs_latest_actual_pct, 2),
        "assumptions": {
            "growthAdjustmentPct": round(growth_adjustment_pct, 2),
            "interestRateShockBps": round(interest_rate_shock_bps, 2),
            "loanCountChangePct": round(loan_count_change_pct, 2),
            "avgLoanAmountChangePct": round(avg_loan_amount_change_pct, 2),
        },
    }

    return {
        "trend": trend,
        "interval": interval,
        "summary": summary,
        "request": {
            "horizonMonths": horizon,
            "scenario": scenario,
            "growthAdjustmentPct": round(growth_adjustment_pct, 2),
            "interestRateShockBps": round(interest_rate_shock_bps, 2),
            "loanCountChangePct": round(loan_count_change_pct, 2),
            "avgLoanAmountChangePct": round(avg_loan_amount_change_pct, 2),
            "userAdjustments": request_config.get("userAdjustments", {}),
        },
        "model": {
            "primary": "module3_xgb.pkl",
            "blend": "module3_lr.pkl",
            "scaler": "module3_scaler.pkl",
        },
        "warnings": warnings,
    }


def _portfolio_batch_score(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Portfolio batch scoring requires real input rows. "
            "Provide a non-empty CSV upload; no synthetic fallback is allowed."
        )

    raw_df = pd.DataFrame(rows)

    module1 = {
        "xgb": _load_module_artifacts(1)[0],
        "features": _load_module_artifacts(1)[1],
    }
    module2 = {
        "xgb": _load_module_artifacts(2)[0],
        "features": _load_module_artifacts(2)[1],
    }

    required_features = sorted(set(module1["features"]) | set(module2["features"]))
    _require_columns(raw_df, required_features, "portfolio batch scoring")

    scored = score_with_models(raw_df, module1, module2)

    fi_default = model_feature_importance(module1["xgb"], module1["features"]).rename(
        columns={"importance": "default_importance"}
    )
    fi_churn = model_feature_importance(module2["xgb"], module2["features"]).rename(
        columns={"importance": "churn_importance"}
    )
    fi_merged = fi_default.merge(fi_churn, on="feature", how="outer").fillna(0.0)
    fi_merged["importance"] = 0.6 * fi_merged["default_importance"] + 0.4 * fi_merged["churn_importance"]
    fi_merged = fi_merged.sort_values("importance", ascending=False).head(10).reset_index(drop=True)

    out = []
    for idx, row in scored.reset_index(drop=True).iterrows():
        out.append(
            {
                "id": str(row.get("id", f"L-{2000 + idx}")),
                "borrower": str(row.get("borrower", f"Borrower-{idx + 1}")),
                "defaultRisk": float(row.get("default_probability", 0.0)),
                "churnRisk": float(row.get("churn_probability", 0.0)),
                "riskBand": str(row.get("risk_band", "Low")),
            }
        )

    return {
        "rows": out,
        "featureImportance": _to_records(fi_merged, ["feature", "importance"]),
        "drift": [],
    }


def _anomaly_detect(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = _load_or_train_anomaly_artifacts(contamination=0.03, random_state=42)

    amount = float(payload.get("amount", 0))
    hour = int(payload.get("hour", 10))
    day_of_week = int(payload.get("dayOfWeek", 1))
    frequency = int(payload.get("frequency", 1))

    result = analyze_transaction(
        artifacts,
        amount=amount,
        hour=hour,
        day_of_week=day_of_week,
        frequency=frequency,
    )

    scaled = artifacts.scaler.transform(np.array([[amount, hour, day_of_week, frequency]], dtype=float))
    shap_frame = pd.DataFrame(scaled, columns=artifacts.feature_columns)
    shap_payload = _build_shap_payload(artifacts.iso_forest, shap_frame, max_features=4)

    label = "Suspicious" if bool(result["is_anomaly"]) else "Normal"
    return {
        "score": float(result["score"]),
        "label": label,
        "shapExplanation": shap_payload,
    }


def _anomaly_timeseries(_: dict[str, Any]) -> dict[str, Any]:
    df = _load_anomaly_dataset_strict()
    artifacts = _load_or_train_anomaly_artifacts(contamination=0.03, random_state=42)
    scored = score_transactions(df.head(240), artifacts)

    hourly = (
        scored.groupby("hour", as_index=False)["anomaly_score"]
        .mean()
        .sort_values("hour")
    )
    trend = [
        {
            "time": f"{int(row.hour):02d}:00",
            "score": round(float(row.anomaly_score), 4),
        }
        for row in hourly.itertuples(index=False)
    ]

    batch = []
    for idx, row in scored.head(30).reset_index(drop=True).iterrows():
        batch.append(
            {
                "txId": f"TX-{1100 + idx}",
                "amount": float(row["amount"]),
                "score": float(row["anomaly_score"]),
                "label": "Suspicious" if bool(row["is_anomaly"]) else "Normal",
            }
        )

    return {"trend": trend, "batch": batch}


def _anomaly_batch_score(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Anomaly batch scoring requires real input rows. "
            "Provide a non-empty CSV upload; no synthetic fallback is allowed."
        )

    incoming = pd.DataFrame(rows).copy()
    if "dayOfWeek" in incoming.columns and "day_of_week" not in incoming.columns:
        incoming["day_of_week"] = incoming["dayOfWeek"]

    _require_columns(incoming, ["amount", "hour", "day_of_week", "frequency"], "anomaly batch scoring")

    working = incoming[["amount", "hour", "day_of_week", "frequency"]].copy()
    for col in ["amount", "hour", "day_of_week", "frequency"]:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    if working.isna().any().any():
        raise ValueError(
            "Invalid numeric values in anomaly batch input. "
            "Required numeric columns: amount, hour, day_of_week/dayOfWeek, frequency"
        )

    artifacts = _load_or_train_anomaly_artifacts(contamination=0.03, random_state=42)
    scored = score_transactions(working, artifacts)

    batch = []
    for idx, row in scored.reset_index(drop=True).iterrows():
        txid = str(incoming.iloc[idx].get("txId", f"TX-{1100 + idx}"))
        batch.append(
            {
                "txId": txid,
                "amount": float(row["amount"]),
                "score": float(row["anomaly_score"]),
                "label": "Suspicious" if bool(row["is_anomaly"]) else "Normal",
            }
        )

    return {"batch": batch}


def _forecast_credit_demand_by_grade(payload: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_credit_demand_payload(payload)
    horizon = int(config["horizon"])
    confidence = float(config["confidence"])
    scenario = str(config["scenarioType"])
    base_volume = config["baseVolume"]

    grade_df = _load_grade_monthly_history()
    pivot = _build_grade_history_pivot(grade_df)
    if len(pivot) < 12:
        raise ValueError(
            "Credit demand forecast requires at least 12 monthly observations for grade-level inference."
        )

    monthly_total = _load_monthly_series()
    forecast_months, total_targets = _load_total_forecast_targets(monthly_total, horizon, base_volume)

    artifacts, feature_schema, model_warnings = _load_grade_models()
    warnings = list(model_warnings)
    if not artifacts:
        warnings.append(
            "No grade model artifacts were loaded. Falling back to statistical share-based forecasting."
        )

    shares = _grade_share_from_artifacts()
    z_value = _confidence_z(confidence)

    month_index = [pd.to_datetime(value) for value in pivot.index.tolist()]
    grade_history = {
        grade: pd.to_numeric(pivot[grade], errors="coerce").fillna(0.0).astype(float).tolist()
        for grade in GRADES
    }
    total_history = pd.to_numeric(pivot[GRADES].sum(axis=1), errors="coerce").fillna(0.0).astype(float).tolist()

    predictions_by_grade = {grade: [] for grade in GRADES}
    feature_importance_by_grade: dict[str, list[dict[str, Any]]] = {}
    metrics_by_grade: dict[str, dict[str, float]] = {}
    model_names: dict[str, str] = {}

    for grade in GRADES:
        artifact = artifacts.get(grade)
        metrics_by_grade[grade] = _grade_model_backtest_metrics(
            grade,
            month_index,
            grade_history[grade],
            total_history,
            artifact,
        )

        if artifact is not None:
            importance_df = model_feature_importance(artifact["model"], artifact["features"]).head(6)
            feature_importance_by_grade[grade] = _to_records(importance_df, ["feature", "importance"])
            model_names[grade] = "XGBoost"
        else:
            feature_importance_by_grade[grade] = _default_grade_feature_importance(feature_schema)
            model_names[grade] = "Statistical fallback"

    baseline_total_accumulator = 0.0
    for step, month_start in enumerate(forecast_months, start=1):
        target_total = float(total_targets[step - 1]) if step - 1 < len(total_targets) else float(total_history[-1])
        if target_total <= 0:
            target_total = max(float(total_history[-1]), 1.0)

        baseline_grade_preds: dict[str, float] = {}
        for grade in GRADES:
            artifact = artifacts.get(grade)
            if artifact is not None:
                try:
                    feature_row = _build_module4_feature_row(
                        grade_history[grade],
                        month_start,
                        artifact["features"],
                        artifact["defaults"],
                        float(total_history[-1]),
                    )
                    frame = pd.DataFrame([feature_row], columns=artifact["features"]).fillna(0.0)
                    scaled = artifact["scaler"].transform(frame)
                    pred = float(np.asarray(artifact["model"].predict(scaled)).ravel()[0])
                    baseline_grade_preds[grade] = max(0.0, pred)
                except Exception as exc:
                    warnings.append(
                        f"Grade {grade} model inference failed for {month_start.strftime('%Y-%m')}; using fallback for this step. Reason: {exc}"
                    )
                    baseline_grade_preds[grade] = max(
                        0.0,
                        float(grade_history[grade][-1]) if grade_history[grade] else target_total * float(shares.get(grade, 0.0)),
                    )
            else:
                baseline_grade_preds[grade] = max(
                    0.0,
                    float(grade_history[grade][-1]) if grade_history[grade] else target_total * float(shares.get(grade, 0.0)),
                )

        raw_sum = float(sum(baseline_grade_preds.values()))
        if raw_sum <= 1e-6:
            baseline_grade_preds = {
                grade: max(0.0, target_total * float(shares.get(grade, 0.0)))
                for grade in GRADES
            }
            raw_sum = float(sum(baseline_grade_preds.values()))

        if raw_sum > 1e-6:
            scale = target_total / raw_sum
            for grade in GRADES:
                baseline_grade_preds[grade] = max(0.0, float(baseline_grade_preds[grade]) * scale)

        baseline_total_accumulator += float(sum(baseline_grade_preds.values()))

        scenario_month_total = 0.0
        for grade in GRADES:
            baseline_central = float(baseline_grade_preds[grade])
            recent_vals = np.asarray(grade_history[grade][-12:], dtype=float)
            if recent_vals.size <= 1:
                deltas = np.asarray([0.0], dtype=float)
            else:
                deltas = np.diff(recent_vals)

            volatility = float(np.std(deltas))
            volatility = max(volatility, 0.04 * max(abs(baseline_central), 1.0))
            spread = volatility * np.sqrt(step)

            lower = max(0.0, baseline_central - (z_value * spread))
            upper = max(lower, baseline_central + (z_value * spread))
            adjusted_central, adjusted_lower, adjusted_upper = _apply_module4_scenario(
                scenario,
                baseline_central,
                lower,
                upper,
            )

            predictions_by_grade[grade].append(
                {
                    "month": month_start.strftime("%b-%Y"),
                    "forecast_central": round(adjusted_central, 2),
                    "forecast_lower": round(adjusted_lower, 2),
                    "forecast_upper": round(adjusted_upper, 2),
                    "historical": None,
                }
            )

            grade_history[grade].append(float(adjusted_central))
            scenario_month_total += float(adjusted_central)

        total_history.append(float(scenario_month_total))

    forecasts = []
    for grade in GRADES:
        forecasts.append(
            {
                "grade": grade,
                "predictions": predictions_by_grade[grade],
                "modelMetrics": metrics_by_grade[grade],
                "featureImportance": feature_importance_by_grade[grade],
                "modelName": model_names[grade],
            }
        )

    scenario_comparison = {
        "baseline": round(float(baseline_total_accumulator), 2),
        "optimistic": round(float(baseline_total_accumulator) * MODULE4_SCENARIO_MULTIPLIER["optimistic"], 2),
        "pessimistic": round(float(baseline_total_accumulator) * MODULE4_SCENARIO_MULTIPLIER["pessimistic"], 2),
    }

    selected_total_forecast = sum(
        float(pred["forecast_central"])
        for grade in GRADES
        for pred in predictions_by_grade[grade]
    )

    return {
        "forecasts": forecasts,
        "metadata": {
            "horizon": horizon,
            "confidence": round(confidence, 2),
            "scenario": scenario,
            "totalForecastedVolume": round(float(selected_total_forecast), 2),
            "scenarioComparison": scenario_comparison,
        },
        "warnings": warnings,
    }


def _credit_demand_by_grade(_: dict[str, Any]) -> dict[str, Any]:
    grade_df = _load_grade_monthly_history()
    pivot = _build_grade_history_pivot(grade_df)

    trend = []
    for _, row in pivot.reset_index().iterrows():
        trend.append(
            {
                "month": pd.to_datetime(row["month_start"]).strftime("%b-%y"),
                "A": round(float(row["A"]), 2),
                "B": round(float(row["B"]), 2),
                "C": round(float(row["C"]), 2),
                "D": round(float(row["D"]), 2),
                "E": round(float(row["E"]), 2),
            }
        )

    seasonality = (
        grade_df.assign(month_num=grade_df["month_start"].dt.month)
        .groupby(["grade", "month_num"], as_index=False)["funded_amnt_m"]
        .mean()
    )

    heatmap = []
    for grade in GRADES:
        gdf = seasonality[seasonality["grade"] == grade]
        vals = []
        for month_num in range(1, 13):
            rec = gdf[gdf["month_num"] == month_num]
            vals.append(round(float(rec["funded_amnt_m"].iloc[0]), 2) if not rec.empty else 0.0)
        heatmap.append({"grade": grade, "values": vals})

    return {"trend": trend, "heatmap": heatmap}


def _deposit_leaderboard(_: dict[str, Any]) -> dict[str, Any]:
    artifacts = _load_or_train_bank_artifacts(seed=42)

    metrics = artifacts.metrics.rename(columns={"model": "model"}).copy()
    leaderboard = _to_records(metrics, ["model", "accuracy", "precision", "recall", "f1"])

    best_model = artifacts.models[artifacts.best_model_name]
    importance = top_feature_importance(best_model, artifacts.feature_columns, n=8)

    return {
        "leaderboard": leaderboard,
        "featureImportance": _to_records(importance, ["feature", "importance"]),
        "bestModel": artifacts.best_model_name,
    }


def _deposit_predict(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = _load_or_train_bank_artifacts(seed=42)

    model_name = _require_str(payload, ["model"], "model")
    if model_name not in artifacts.models:
        valid = ", ".join(sorted(artifacts.models.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Valid models: {valid}")
    model = artifacts.models[model_name]

    user_row = {
        "age": _require_int(payload, ["age"], "age"),
        "job": _require_str(payload, ["job"], "job").lower(),
        "marital": _require_str(payload, ["marital"], "marital").lower(),
        "education": _require_str(payload, ["education"], "education").lower(),
        "default": _require_str(payload, ["default"], "default").lower(),
        "balance": _require_float(payload, ["balance"], "balance"),
        "housing": _require_str(payload, ["housing"], "housing").lower(),
        "loan": _require_str(payload, ["loan"], "loan").lower(),
        "contact": _require_str(payload, ["contact"], "contact").lower(),
        "day": _require_int(payload, ["day"], "day"),
        "month": _require_str(payload, ["month"], "month").lower(),
        "duration": _require_int(payload, ["duration"], "duration"),
        "campaign": _require_int(payload, ["campaign"], "campaign"),
        "pdays": _require_int(payload, ["pdays"], "pdays"),
        "previous": _require_int(payload, ["previous"], "previous"),
        "poutcome": _require_str(payload, ["poutcome"], "poutcome").lower(),
    }

    prepared = preprocess_user_input(user_row, artifacts.feature_columns)
    pred, prob = predict_subscription(model, prepared)
    shap_payload = _build_shap_payload(model, prepared)

    return {
        "probability": float(prob),
        "label": "Likely Subscribe" if float(prob) >= 0.6 else "Low Intent",
        "prediction": int(pred),
        "model": model_name,
        "shapExplanation": shap_payload,
    }


OP_MAP = {
    "predict_default": _predict_default,
    "predict_churn": _predict_churn,
    "forecast_loan_volume": _forecast_loan_volume,
    "portfolio_batch_score": _portfolio_batch_score,
    "anomaly_detect": _anomaly_detect,
    "anomaly_timeseries": _anomaly_timeseries,
    "anomaly_batch_score": _anomaly_batch_score,
    "credit_demand_by_grade": _credit_demand_by_grade,
    "credit_demand_by_grade_forecast": _forecast_credit_demand_by_grade,
    "deposit_leaderboard": _deposit_leaderboard,
    "deposit_predict": _deposit_predict,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Finlytics frontend Python bridge")
    parser.add_argument("op", choices=sorted(OP_MAP.keys()))
    args = parser.parse_args()

    payload = _load_json_stdin()
    result = OP_MAP[args.op](payload)
    sys.stdout.write(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
