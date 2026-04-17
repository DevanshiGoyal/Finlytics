from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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


def _load_module_artifacts(module_number: int) -> tuple[Any, list[str]]:
    model = joblib.load(PROJECT_ROOT / f"models/module{module_number}_xgb.pkl")
    features = joblib.load(PROJECT_ROOT / f"models/module{module_number}_features.pkl")
    return model, list(features)


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


def _forecast_loan_volume(_: dict[str, Any]) -> dict[str, Any]:
    monthly = _load_monthly_series()
    horizon = 3
    fc = forecast_with_uncertainty(monthly, horizon=horizon)
    baseline = compute_baseline_forecasts(monthly, horizon=horizon)

    trend_tail = monthly.tail(9).copy()
    trend = [
        {
            "month": pd.to_datetime(row.month_start).strftime("%b"),
            "actual": float(row.funded_amnt_m),
            "forecast": float(row.funded_amnt_m),
        }
        for row in trend_tail.itertuples(index=False)
    ]

    interval = []
    for row in fc.itertuples(index=False):
        interval.append(
            {
                "month": pd.to_datetime(row.month_start).strftime("%b"),
                "value": float(row.forecast_central),
                "low": float(row.forecast_lower),
                "high": float(row.forecast_upper),
                "historical": None,
                "baseline": float(baseline.loc[baseline["month_start"] == row.month_start, "naive_last"].iloc[0]),
            }
        )

    return {"trend": trend, "interval": interval}


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


def _credit_demand_by_grade(_: dict[str, Any]) -> dict[str, Any]:
    grade_path = PROJECT_ROOT / "data/processed/grade_monthly_demand.csv"
    if grade_path.exists():
        grade_df = pd.read_csv(grade_path, parse_dates=["month_start"])
        _require_columns(grade_df, ["month_start", "grade", "funded_amnt_m"], "credit demand by grade")
    else:
        monthly = _load_monthly_series()
        shares = _grade_share_from_artifacts()
        rows = []
        for rec in monthly.itertuples(index=False):
            for grade in ["A", "B", "C", "D", "E"]:
                rows.append(
                    {
                        "month_start": rec.month_start,
                        "grade": grade,
                        "funded_amnt_m": float(rec.funded_amnt_m) * float(shares[grade]),
                    }
                )
        grade_df = pd.DataFrame(rows)

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

    for g in ["A", "B", "C", "D", "E"]:
        if g not in pivot.columns:
            pivot[g] = 0.0

    trend = []
    for idx, row in pivot[["A", "B", "C", "D", "E"]].reset_index().iterrows():
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
    for grade in ["A", "B", "C", "D", "E"]:
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
