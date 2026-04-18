from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation"
KAGGLE_RFM_NOTEBOOK_URL = "https://www.kaggle.com/code/marcpatrickmargallo/bank-customer-segmentation-1m-transactions"
KAGGLE_PERSONA_NOTEBOOK_URL = "https://www.kaggle.com/code/jianlizhou/customer-segmentation-by-rfm-model-and-k-means"
KAGGLE_BEHAVIOR_NOTEBOOK_URL = "https://www.kaggle.com/code/samudra89/01-customer-segmentation-bank-transactions"

DEFAULT_WEALTH_DATASET_PATH = "data/external/bank_customer_segmentation.csv"

# Required by user specification.
REQUIRED_WEALTH_COLUMNS = {
    "CustomerDOB",
    "CustLocation",
    "CustAccountBalance",
}

TRANSACTION_AMOUNT_ALIASES = ["TransactionAmount", "TransactionAmount (INR)"]

# Required for RFM / persona logic used by the referenced notebooks.
RECOMMENDED_TRANSACTION_COLUMNS = {
    "CustomerID",
    "TransactionDate",
}


@dataclass
class WealthPersonaArtifacts:
    cleaned_transactions: pd.DataFrame
    customer_features: pd.DataFrame
    cluster_profiles: pd.DataFrame
    anomaly_benchmarks: pd.DataFrame
    regional_intelligence: pd.DataFrame
    summary: Dict[str, object]


def _validate_wealth_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_WEALTH_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Bank Customer Segmentation dataset is missing required columns: "
            f"{missing}. Expected Kaggle schema from {KAGGLE_DATASET_URL}."
        )

    if not any(col in df.columns for col in TRANSACTION_AMOUNT_ALIASES):
        raise ValueError(
            "Bank Customer Segmentation dataset is missing transaction amount column. "
            "Expected one of: TransactionAmount or TransactionAmount (INR)."
        )

    missing_recommended = sorted(RECOMMENDED_TRANSACTION_COLUMNS - set(df.columns))
    if missing_recommended:
        raise ValueError(
            "Dataset is missing RFM-critical columns: "
            f"{missing_recommended}. For exact notebook logic, include CustomerID and "
            "TransactionDate columns from the Kaggle transaction file."
        )

    return df


def load_wealth_dataset(dataset_path: str | None = None) -> pd.DataFrame:
    """
    Load the exact Kaggle Bank Customer Segmentation transaction dataset.

    This module intentionally does not generate synthetic/mock data.
    Provide the CSV from: https://www.kaggle.com/datasets/shivamb/bank-customer-segmentation
    """
    path = dataset_path or DEFAULT_WEALTH_DATASET_PATH
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            "Wealth-persona dataset not found. Expected CSV path: "
            f"{resolved}. Download the exact Kaggle dataset from {KAGGLE_DATASET_URL} "
            "and place it at this path or pass dataset_path explicitly."
        )

    df = pd.read_csv(resolved, low_memory=False)
    return _validate_wealth_df(df)


def _normalize_location(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN"})
    )


def _safe_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _parse_datetime_mixed(series: pd.Series) -> pd.Series:
    # Optimized for mixed real-world date strings in the Kaggle transaction dataset.
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def _score_quintiles(series: pd.Series, higher_is_better: bool) -> pd.Series:
    # Use ranked values to handle duplicate-heavy distributions efficiently.
    ranked = pd.to_numeric(series, errors="coerce").rank(method="first")
    try:
        score = pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    except ValueError:
        # Fallback when unique values are insufficient.
        pct = ranked.rank(pct=True)
        score = pd.cut(
            pct,
            bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
            labels=[1, 2, 3, 4, 5],
            include_lowest=True,
        ).astype(int)

    if not higher_is_better:
        score = 6 - score
    return score.astype(int)


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize raw transaction rows for scalable 1M+ analytics.
    """
    data = _validate_wealth_df(df).copy()

    # Normalize Kaggle variant naming to internal canonical column names.
    if "TransactionAmount" not in data.columns and "TransactionAmount (INR)" in data.columns:
        data = data.rename(columns={"TransactionAmount (INR)": "TransactionAmount"})

    keep_cols = [
        "CustomerID",
        "CustomerDOB",
        "CustLocation",
        "CustAccountBalance",
        "TransactionDate",
        "TransactionAmount",
    ]
    data = data[keep_cols].copy()

    data["CustomerID"] = data["CustomerID"].astype(str).str.strip()
    data["CustLocationNorm"] = _normalize_location(data["CustLocation"])

    data["CustAccountBalance"] = _safe_numeric(data["CustAccountBalance"], np.nan)
    data["TransactionAmount"] = _safe_numeric(data["TransactionAmount"], np.nan)

    data["TransactionDate"] = _parse_datetime_mixed(data["TransactionDate"])
    data["CustomerDOB"] = _parse_datetime_mixed(data["CustomerDOB"])

    # Remove clearly invalid DOB values found in public versions of the dataset.
    data.loc[data["CustomerDOB"].dt.year < 1900, "CustomerDOB"] = pd.NaT

    # Drop unusable rows for RFM + anomaly scoring.
    data = data.dropna(subset=["CustomerID", "TransactionDate", "TransactionAmount", "CustAccountBalance"])

    # Balance can be negative in real-world banking data; keep those rows.
    # Transaction amount should be non-negative for this segmentation workflow.
    data = data[data["TransactionAmount"] >= 0].copy()

    reference_date = data["TransactionDate"].max()
    data["Age"] = ((reference_date - data["CustomerDOB"]).dt.days / 365.25).astype(float)
    data.loc[(data["Age"] < 18) | (data["Age"] > 100), "Age"] = np.nan

    return data.reset_index(drop=True)


def build_rfm_wealth_layer(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-level RFM and Pareto 'Vital Few' segmentation.
    """
    tx = prepare_transactions(transactions)
    reference_date = tx["TransactionDate"].max()

    grouped = tx.groupby("CustomerID", observed=True)

    customer = grouped.agg(
        LastTransactionDate=("TransactionDate", "max"),
        Frequency=("TransactionAmount", "count"),
        Monetary=("TransactionAmount", "sum"),
        AvgAccountBalance=("CustAccountBalance", "mean"),
        MedianAge=("Age", "median"),
    ).reset_index()

    location_mode = (
        tx.groupby(["CustomerID", "CustLocationNorm"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["CustomerID", "n", "CustLocationNorm"], ascending=[True, False, True])
        .drop_duplicates("CustomerID")
        [["CustomerID", "CustLocationNorm"]]
    )

    customer = customer.merge(location_mode, on="CustomerID", how="left")

    customer["Recency"] = (reference_date - customer["LastTransactionDate"]).dt.days.astype(int)
    customer["MedianAge"] = customer["MedianAge"].fillna(customer["MedianAge"].median())

    customer["R_Score"] = _score_quintiles(customer["Recency"], higher_is_better=False)
    customer["F_Score"] = _score_quintiles(customer["Frequency"], higher_is_better=True)
    customer["M_Score"] = _score_quintiles(customer["Monetary"], higher_is_better=True)

    customer["RFM_Consolidated"] = customer["R_Score"] + customer["F_Score"] + customer["M_Score"]
    customer["RFM_Score"] = (
        customer["R_Score"].astype(str)
        + customer["F_Score"].astype(str)
        + customer["M_Score"].astype(str)
    )

    # Wealth definition for Pareto layer: transaction monetary power + account liquidity.
    customer["WealthValue"] = customer["Monetary"] + customer["AvgAccountBalance"].clip(lower=0)

    threshold = float(customer["WealthValue"].quantile(0.80))
    customer["IsVitalFew"] = customer["WealthValue"] >= threshold

    customer = customer.sort_values("WealthValue", ascending=False).reset_index(drop=True)
    total_wealth = float(customer["WealthValue"].sum()) if len(customer) else 0.0
    if total_wealth > 0:
        customer["WealthSharePct"] = customer["WealthValue"] / total_wealth
        customer["CumulativeWealthSharePct"] = customer["WealthSharePct"].cumsum()
    else:
        customer["WealthSharePct"] = 0.0
        customer["CumulativeWealthSharePct"] = 0.0

    return customer


def _map_cluster_personas(cluster_profile: pd.DataFrame) -> Dict[int, str]:
    clusters = cluster_profile["Cluster"].tolist()
    if len(clusters) != 4:
        return {int(c): f"Segment {int(c)}" for c in clusters}

    remaining = set(int(c) for c in clusters)
    mapping: Dict[int, str] = {}

    low_activity = int(cluster_profile.sort_values(["Frequency", "Monetary"], ascending=[True, True]).iloc[0]["Cluster"])
    mapping[low_activity] = "Low-Activity Accounts"
    remaining.remove(low_activity)

    wealthy_seniors = int(
        cluster_profile[cluster_profile["Cluster"].isin(remaining)]
        .sort_values(["MedianAge", "Monetary", "AvgAccountBalance"], ascending=[False, False, False])
        .iloc[0]["Cluster"]
    )
    mapping[wealthy_seniors] = "Wealthy Seniors"
    remaining.remove(wealthy_seniors)

    frequent_young = int(
        cluster_profile[cluster_profile["Cluster"].isin(remaining)]
        .assign(young_freq_score=lambda x: x["Frequency"] - 0.10 * x["MedianAge"])
        .sort_values("young_freq_score", ascending=False)
        .iloc[0]["Cluster"]
    )
    mapping[frequent_young] = "Frequent Young Transactors"
    remaining.remove(frequent_young)

    mid_tier = int(next(iter(remaining)))
    mapping[mid_tier] = "Mid-Tier Savers"

    return mapping


def build_persona_layer(
    customer_rfm: pd.DataFrame,
    k: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply StandardScaler + KMeans on RFM features and map to personas.
    """
    if k < 2:
        raise ValueError("k must be >= 2 for K-Means clustering")

    customer = customer_rfm.copy()

    rfm_features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    X = scaler.fit_transform(customer[rfm_features])

    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    customer["Cluster"] = model.fit_predict(X)

    profile = (
        customer.groupby("Cluster", as_index=False, observed=True)
        .agg(
            Customers=("CustomerID", "count"),
            MedianAge=("MedianAge", "mean"),
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
            AvgAccountBalance=("AvgAccountBalance", "mean"),
            VitalFewShare=("IsVitalFew", "mean"),
        )
        .sort_values("Cluster")
        .reset_index(drop=True)
    )

    persona_map = _map_cluster_personas(profile)
    customer["Persona"] = customer["Cluster"].map(persona_map).fillna("Other")
    profile["Persona"] = profile["Cluster"].map(persona_map).fillna("Other")

    return customer, profile


def benchmark_behavioral_anomalies(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Behavioral benchmarking inspired by transaction outlier logic:
    compares each transaction against regional and customer historical baselines.
    """
    tx = prepare_transactions(transactions)

    # Regional benchmark statistics.
    regional = tx.groupby("CustLocationNorm", observed=True)["TransactionAmount"].agg(
        loc_median="median",
        loc_q1=lambda s: s.quantile(0.25),
        loc_q3=lambda s: s.quantile(0.75),
        loc_p99=lambda s: s.quantile(0.99),
    )
    regional = regional.reset_index()
    regional["loc_iqr"] = (regional["loc_q3"] - regional["loc_q1"]).replace(0, 1.0)

    # Customer historical benchmark statistics.
    customer_stats = tx.groupby("CustomerID", observed=True)["TransactionAmount"].agg(
        cust_median="median",
        cust_std="std",
        cust_n="count",
    )
    customer_stats = customer_stats.reset_index()
    customer_stats["cust_std"] = customer_stats["cust_std"].fillna(0.0)

    out = tx.merge(regional, on="CustLocationNorm", how="left").merge(customer_stats, on="CustomerID", how="left")

    out["RegionalIqrDeviation"] = (out["TransactionAmount"] - out["loc_median"]).abs() / out["loc_iqr"].replace(0, 1.0)
    out["CustomerZLike"] = (out["TransactionAmount"] - out["cust_median"]).abs() / (
        out["cust_std"].replace(0, np.nan)
    )
    out["CustomerZLike"] = out["CustomerZLike"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["flag_regional_outlier"] = out["TransactionAmount"] > (out["loc_q3"] + 3.0 * out["loc_iqr"])
    out["flag_extreme_high"] = out["TransactionAmount"] >= out["loc_p99"]
    out["flag_customer_spike"] = (out["cust_n"] >= 5) & (
        out["TransactionAmount"] > np.maximum(out["cust_median"] * 3.0, out["cust_median"] + 3.0 * out["cust_std"]) 
    )
    out["flag_balance_stress"] = out["CustAccountBalance"] < 0

    # Weighted anomaly score [0, 1].
    raw = (
        0.40 * out["flag_regional_outlier"].astype(float)
        + 0.25 * out["flag_extreme_high"].astype(float)
        + 0.25 * out["flag_customer_spike"].astype(float)
        + 0.10 * out["flag_balance_stress"].astype(float)
    )
    out["BehavioralAnomalyScore"] = raw.clip(0.0, 1.0)
    out["IsBehavioralAnomaly"] = out["BehavioralAnomalyScore"] >= 0.50

    reasons: List[str] = []
    for row in out.itertuples(index=False):
        row_reasons: List[str] = []
        if row.flag_regional_outlier:
            row_reasons.append("Regional benchmark outlier")
        if row.flag_extreme_high:
            row_reasons.append("Extreme transaction amount (P99+)")
        if row.flag_customer_spike:
            row_reasons.append("Spike vs customer historical behavior")
        if row.flag_balance_stress:
            row_reasons.append("Negative account balance stress")
        reasons.append(", ".join(row_reasons) if row_reasons else "Within expected behavior")

    out["BehavioralReason"] = reasons
    return out.reset_index(drop=True)


def build_regional_intelligence(customer_features: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate account-balance intelligence by normalized location and score
    high-potential investment zones.
    """
    df = customer_features.copy()

    regional = (
        df.groupby("CustLocationNorm", observed=True)
        .agg(
            Customers=("CustomerID", "count"),
            TotalAccountBalance=("AvgAccountBalance", "sum"),
            AvgAccountBalance=("AvgAccountBalance", "mean"),
            MedianAccountBalance=("AvgAccountBalance", "median"),
            TotalTransactionValue=("Monetary", "sum"),
            VitalFewShare=("IsVitalFew", "mean"),
            AvgRfmConsolidated=("RFM_Consolidated", "mean"),
        )
        .reset_index()
        .sort_values("TotalAccountBalance", ascending=False)
        .reset_index(drop=True)
    )

    def _normalize(col: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(col), dtype=float)
        if len(arr) == 0:
            return arr
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-12:
            return np.ones_like(arr)
        return (arr - lo) / (hi - lo)

    regional["PotentialScore"] = (
        0.40 * _normalize(regional["TotalAccountBalance"])
        + 0.30 * _normalize(regional["TotalTransactionValue"])
        + 0.20 * _normalize(regional["VitalFewShare"])
        + 0.10 * _normalize(regional["AvgRfmConsolidated"])
    )

    threshold = float(regional["PotentialScore"].quantile(0.75)) if len(regional) else 0.0
    regional["HighPotentialZone"] = regional["PotentialScore"] >= threshold

    return regional


def run_wealth_persona_pipeline(
    dataset_path: str | None = None,
    k: int = 4,
    random_state: int = 42,
) -> WealthPersonaArtifacts:
    """
    End-to-end pipeline:
    1) Load exact Kaggle transaction dataset
    2) Build RFM + Pareto wealth layer
    3) Build KMeans persona layer
    4) Build behavioral anomaly benchmark
    5) Build regional intelligence
    """
    raw = load_wealth_dataset(dataset_path=dataset_path)
    cleaned = prepare_transactions(raw)

    customer_rfm = build_rfm_wealth_layer(cleaned)
    customer_persona, cluster_profiles = build_persona_layer(
        customer_rfm,
        k=k,
        random_state=random_state,
    )
    anomaly_benchmarks = benchmark_behavioral_anomalies(cleaned)
    regional_intel = build_regional_intelligence(customer_persona)

    summary = {
        "dataset_source": dataset_path or DEFAULT_WEALTH_DATASET_PATH,
        "kaggle_dataset_url": KAGGLE_DATASET_URL,
        "rfm_notebook_url": KAGGLE_RFM_NOTEBOOK_URL,
        "persona_notebook_url": KAGGLE_PERSONA_NOTEBOOK_URL,
        "behavior_notebook_url": KAGGLE_BEHAVIOR_NOTEBOOK_URL,
        "transaction_rows": int(len(cleaned)),
        "customers": int(customer_persona["CustomerID"].nunique()),
        "vital_few_customers": int(customer_persona["IsVitalFew"].sum()),
        "vital_few_share": float(customer_persona["IsVitalFew"].mean()) if len(customer_persona) else 0.0,
        "behavioral_anomalies": int(anomaly_benchmarks["IsBehavioralAnomaly"].sum()),
        "high_potential_zones": int(regional_intel["HighPotentialZone"].sum()),
    }

    return WealthPersonaArtifacts(
        cleaned_transactions=cleaned,
        customer_features=customer_persona,
        cluster_profiles=cluster_profiles,
        anomaly_benchmarks=anomaly_benchmarks,
        regional_intelligence=regional_intel,
        summary=summary,
    )
