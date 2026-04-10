from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from io import StringIO
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]
DEFAULT_BANK_DATASET_URL = "https://raw.githubusercontent.com/Zeeshan13/Bank-Term-Deposit-Prediction/main/bank_updated.csv"
DEFAULT_BANK_DATASET_CACHE = "data/external/bank_updated.csv"


@dataclass
class BankModelArtifacts:
    models: Dict[str, object]
    metrics: pd.DataFrame
    best_model_name: str
    feature_columns: List[str]
    train_baseline: pd.DataFrame


REQUIRED_BANK_COLUMNS = {
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


def _read_csv_flexible(path_or_url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path_or_url)
    except Exception:
        # Some bank marketing files use semicolon delimiters
        return pd.read_csv(path_or_url, sep=";")


def _validate_bank_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_BANK_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Bank dataset is missing required columns: {sorted(missing)}")
    return df


def load_bank_dataset(
    dataset_path: str | None = None,
    cache_path: str = DEFAULT_BANK_DATASET_CACHE,
    remote_url: str = DEFAULT_BANK_DATASET_URL,
    use_remote: bool = True,
) -> pd.DataFrame:
    """
    Load bank term deposit dataset.

    Priority:
    1) explicit dataset_path
    2) local cache_path
    3) remote_url download (optional)
    4) synthetic fallback
    """
    if dataset_path:
        return _validate_bank_df(_read_csv_flexible(dataset_path))

    if os.path.exists(cache_path):
        return _validate_bank_df(_read_csv_flexible(cache_path))

    if use_remote:
        try:
            df = _validate_bank_df(_read_csv_flexible(remote_url))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False)
            return df
        except Exception:
            pass

    return _validate_bank_df(generate_synthetic_bank_data(n_rows=4200, seed=42))


def generate_synthetic_bank_data(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    jobs = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed"]
    marital = ["single", "married", "divorced"]
    education = ["primary", "secondary", "tertiary", "unknown"]
    yes_no = ["yes", "no"]
    contacts = ["cellular", "telephone", "unknown"]
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    poutcomes = ["success", "failure", "other", "unknown"]

    age = rng.integers(18, 80, n_rows)
    balance = rng.normal(1400, 2300, n_rows).astype(int)
    balance = np.clip(balance, -1500, 20000)
    duration = rng.integers(30, 2400, n_rows)
    campaign = rng.integers(1, 12, n_rows)
    pdays = rng.integers(-1, 500, n_rows)
    previous = rng.integers(0, 12, n_rows)

    df = pd.DataFrame(
        {
            "age": age,
            "job": rng.choice(jobs, n_rows),
            "marital": rng.choice(marital, n_rows, p=[0.33, 0.57, 0.10]),
            "education": rng.choice(education, n_rows, p=[0.16, 0.50, 0.28, 0.06]),
            "default": rng.choice(yes_no, n_rows, p=[0.03, 0.97]),
            "balance": balance,
            "housing": rng.choice(yes_no, n_rows, p=[0.56, 0.44]),
            "loan": rng.choice(yes_no, n_rows, p=[0.16, 0.84]),
            "contact": rng.choice(contacts, n_rows, p=[0.63, 0.16, 0.21]),
            "month": rng.choice(months, n_rows),
            "day": rng.integers(1, 32, n_rows),
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": rng.choice(poutcomes, n_rows, p=[0.13, 0.16, 0.18, 0.53]),
        }
    )

    # Probabilistic label, designed to be learnable but imbalanced
    score = (
        0.0013 * df["duration"]
        + 0.00020 * np.maximum(df["balance"], 0)
        - 0.10 * (df["campaign"] > 5).astype(float)
        - 0.08 * (df["loan"] == "yes").astype(float)
        + 0.35 * (df["poutcome"] == "success").astype(float)
        + 0.10 * (df["contact"] == "cellular").astype(float)
        - 0.12 * (df["month"].isin(["may", "jun", "jul"])).astype(float)
        - 2.5
    )
    prob = 1 / (1 + np.exp(-score))
    y = rng.binomial(1, np.clip(prob, 0.01, 0.95), n_rows)
    df["y"] = np.where(y == 1, "yes", "no")

    return df


def add_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_category"] = pd.cut(
        pd.to_numeric(out["age"], errors="coerce").fillna(0),
        bins=[0, 30, 45, 60, 120],
        labels=["Young", "Adult", "Mid", "Senior"],
        include_lowest=True,
    ).astype(str)

    out["balance_category"] = pd.cut(
        pd.to_numeric(out["balance"], errors="coerce").fillna(0),
        bins=[-5000, 0, 1500, 5000, 100000],
        labels=["Negative", "Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    out["duration_category"] = pd.cut(
        pd.to_numeric(out["duration"], errors="coerce").fillna(0),
        bins=[0, 180, 600, 2000, 10000],
        labels=["Short", "Medium", "Long", "VeryLong"],
        include_lowest=True,
    ).astype(str)

    return out


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    data = add_binned_features(df)
    y = data["y"].map({"yes": 1, "no": 0}).fillna(0).astype(int)

    preferred = ["age_category", "balance_category", "duration_category"]
    candidate_features = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    ]
    cols = preferred + [c for c in candidate_features if c in data.columns]
    X = data[cols].copy()

    # Use one-hot encoding similar to source repo
    X_encoded = pd.get_dummies(X, columns=X.select_dtypes(include=["object", "category"]).columns)
    return X_encoded, y


def train_bank_models(df: pd.DataFrame, seed: int = 42) -> BankModelArtifacts:
    X_encoded, y = _prepare_xy(df)

    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_resample(X_encoded, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=seed, stratify=y_res
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=220, random_state=seed),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["model"])
    baseline = X_encoded.mean(numeric_only=True).to_frame().T

    return BankModelArtifacts(
        models=models,
        metrics=metrics,
        best_model_name=best_model_name,
        feature_columns=X_encoded.columns.tolist(),
        train_baseline=baseline,
    )


def preprocess_user_input(user_row: Dict[str, object], feature_columns: List[str]) -> pd.DataFrame:
    inp = pd.DataFrame([user_row])
    inp = add_binned_features(inp)

    keep_cols = [
        "age_category",
        "balance_category",
        "duration_category",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    ]
    inp = inp[[c for c in keep_cols if c in inp.columns]]
    enc = pd.get_dummies(inp, columns=inp.select_dtypes(include=["object", "category"]).columns)
    aligned = enc.reindex(columns=feature_columns, fill_value=0)
    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0)
    return aligned


def predict_subscription(model, input_features: pd.DataFrame) -> Tuple[int, float]:
    pred = int(model.predict(input_features)[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_features)[0][1])
    else:
        prob = float(pred)
    return pred, prob


def top_feature_importance(model, feature_columns: List[str], n: int = 6) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_).ravel())
    else:
        imp = np.ones(len(feature_columns), dtype=float)

    imp = np.where(np.isnan(imp), 0.0, imp)
    if imp.sum() == 0:
        imp = np.ones_like(imp, dtype=float)

    values = imp / imp.sum()
    out = pd.DataFrame({"feature": feature_columns, "importance": values})
    return out.sort_values("importance", ascending=False).head(n).reset_index(drop=True)


def _fallback_advice(model_name: str, prediction_prob: float, top_features: pd.DataFrame) -> str:
    feature_text = ", ".join(top_features["feature"].head(3).tolist()) if not top_features.empty else "campaign duration and prior outcomes"
    if prediction_prob >= 0.6:
        tone = "high propensity"
        rec = "Prioritize immediate outreach with premium-term offers and a shorter follow-up cycle."
    elif prediction_prob >= 0.4:
        tone = "moderate propensity"
        rec = "Use a two-step campaign: educational message first, then a targeted offer call."
    else:
        tone = "low propensity"
        rec = "Avoid expensive call-center actions; prefer low-cost digital nurture and re-score later."

    return (
        f"Model `{model_name}` indicates **{tone}** to subscribe (probability {prediction_prob:.1%}). "
        f"Primary drivers: {feature_text}. Recommendation: {rec}"
    )


def generate_llm_campaign_advice(
    model_name: str,
    prediction_prob: float,
    top_features: pd.DataFrame,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
) -> str:
    api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    fallback = _fallback_advice(model_name, prediction_prob, top_features)
    if not api_key:
        return fallback

    prompt = (
        "You are a bank campaign strategy assistant. "
        "Given model probability and top features, return 4 concise bullet points: "
        "risk interpretation, messaging strategy, channel, and next-best-action."
    )
    user_payload = {
        "model_name": model_name,
        "subscription_probability": round(prediction_prob, 4),
        "top_features": top_features.to_dict(orient="records"),
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        "temperature": 0.3,
    }

    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError, TimeoutError):
        return fallback
