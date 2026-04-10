from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DEFAULT_ANOMALY_DATASET_URL = "https://raw.githubusercontent.com/Taliaovescoding/bank-anomaly-system/main/synthetic_deposits.csv"
DEFAULT_ANOMALY_CACHE_PATH = "data/external/synthetic_deposits.csv"
FEATURE_COLUMNS = ["amount", "hour", "day_of_week", "frequency"]


@dataclass
class BankAnomalyArtifacts:
    iso_forest: IsolationForest
    autoencoder: MLPRegressor
    scaler: StandardScaler
    feature_columns: List[str]
    contamination: float
    reconstruction_threshold: float
    amount_high_threshold: float
    frequency_high_threshold: float


def generate_synthetic_deposits(records: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(records):
        amount = rng.normal(500, 200)
        hour = int(rng.choice(np.arange(8, 18)))
        day_of_week = int(rng.choice(np.arange(0, 5)))
        frequency = int(rng.integers(1, 3))

        # Inject ~3% anomalies like source repo
        if rng.random() < 0.03:
            chance = rng.random()
            if chance < 0.33:
                amount = rng.uniform(5000, 10000)
            elif chance < 0.66:
                hour = int(rng.choice([1, 2, 3, 4]))
            else:
                frequency = int(rng.integers(10, 20))

        rows.append([max(1.0, amount), hour, day_of_week, frequency])

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Anomaly dataset missing required columns: {missing}")

    out = df.copy()
    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=FEATURE_COLUMNS)

    out["amount"] = out["amount"].clip(lower=1)
    out["hour"] = out["hour"].clip(lower=0, upper=23).astype(int)
    out["day_of_week"] = out["day_of_week"].clip(lower=0, upper=6).astype(int)
    out["frequency"] = out["frequency"].clip(lower=1).astype(int)
    return out.reset_index(drop=True)


def load_anomaly_dataset(
    dataset_path: str | None = None,
    cache_path: str = DEFAULT_ANOMALY_CACHE_PATH,
    remote_url: str = DEFAULT_ANOMALY_DATASET_URL,
    use_remote: bool = True,
) -> pd.DataFrame:
    if dataset_path:
        return _validate_df(pd.read_csv(dataset_path))

    if os.path.exists(cache_path):
        return _validate_df(pd.read_csv(cache_path))

    if use_remote:
        try:
            df = _validate_df(pd.read_csv(remote_url))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False)
            return df
        except Exception:
            pass

    return _validate_df(generate_synthetic_deposits())


def train_anomaly_engine(
    df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
) -> BankAnomalyArtifacts:
    data = _validate_df(df)
    X = data[FEATURE_COLUMNS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=random_state,
    )
    iso_forest.fit(X_scaled)

    autoencoder = MLPRegressor(
        hidden_layer_sizes=(2, 2),
        activation="relu",
        max_iter=500,
        random_state=random_state,
    )
    autoencoder.fit(X_scaled, X_scaled)

    reconstruction = autoencoder.predict(X_scaled)
    reconstruction_errors = np.mean((X_scaled - reconstruction) ** 2, axis=1)
    reconstruction_threshold = float(np.quantile(reconstruction_errors, 0.97))

    amount_high = float(np.quantile(data["amount"], 0.99))
    freq_high = float(np.quantile(data["frequency"], 0.95))

    return BankAnomalyArtifacts(
        iso_forest=iso_forest,
        autoencoder=autoencoder,
        scaler=scaler,
        feature_columns=FEATURE_COLUMNS,
        contamination=contamination,
        reconstruction_threshold=reconstruction_threshold,
        amount_high_threshold=amount_high,
        frequency_high_threshold=freq_high,
    )


def analyze_transaction(
    artifacts: BankAnomalyArtifacts,
    amount: float,
    hour: int,
    day_of_week: int,
    frequency: int,
) -> Dict[str, object]:
    features = np.array([[amount, hour, day_of_week, frequency]], dtype=float)
    scaled = artifacts.scaler.transform(features)

    iso_pred = int(artifacts.iso_forest.predict(scaled)[0])  # -1 anomaly, 1 normal
    iso_decision = float(artifacts.iso_forest.decision_function(scaled)[0])

    reconstructed = artifacts.autoencoder.predict(scaled)
    reconstruction_error = float(np.mean((scaled - reconstructed) ** 2))

    # unified score in [0,1], higher means riskier
    iso_risk = 1.0 / (1.0 + np.exp(6.0 * iso_decision))
    recon_risk = min(1.0, reconstruction_error / max(artifacts.reconstruction_threshold, 1e-9))
    score = float(np.clip(0.6 * iso_risk + 0.4 * recon_risk, 0.0, 1.0))

    reasons: List[str] = []
    if iso_pred == -1:
        reasons.append("IsolationForest outlier")
    if reconstruction_error >= artifacts.reconstruction_threshold:
        reasons.append("High reconstruction error")
    if amount >= artifacts.amount_high_threshold:
        reasons.append("Unusual amount")
    if hour < 6 or hour > 20:
        reasons.append("Unusual hour")
    if frequency >= artifacts.frequency_high_threshold:
        reasons.append("High frequency")

    is_anomaly = bool((iso_pred == -1) or (reconstruction_error >= artifacts.reconstruction_threshold) or (score >= 0.6))
    reason_text = ", ".join(reasons) if reasons else "Patterns consistent with history"

    return {
        "is_anomaly": is_anomaly,
        "score": score,
        "reasons": reason_text,
        "iso_decision": iso_decision,
        "reconstruction_error": reconstruction_error,
    }


def score_transactions(df: pd.DataFrame, artifacts: BankAnomalyArtifacts) -> pd.DataFrame:
    data = _validate_df(df)
    rows = []
    for _, row in data.iterrows():
        result = analyze_transaction(
            artifacts=artifacts,
            amount=float(row["amount"]),
            hour=int(row["hour"]),
            day_of_week=int(row["day_of_week"]),
            frequency=int(row["frequency"]),
        )
        rows.append(result)

    out = data.copy()
    out["anomaly_score"] = [r["score"] for r in rows]
    out["is_anomaly"] = [r["is_anomaly"] for r in rows]
    out["reasons"] = [r["reasons"] for r in rows]
    out["risk_label"] = np.where(out["is_anomaly"], "Flagged", "Normal")
    return out
