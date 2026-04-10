import pandas as pd

from src.bank_anomaly_module import (
    analyze_transaction,
    generate_synthetic_deposits,
    load_anomaly_dataset,
    score_transactions,
    train_anomaly_engine,
)


def test_generate_synthetic_deposits_schema():
    df = generate_synthetic_deposits(records=120, seed=1)
    assert set(["amount", "hour", "day_of_week", "frequency"]).issubset(df.columns)
    assert len(df) == 120


def test_train_and_analyze_transaction_output_shape():
    df = generate_synthetic_deposits(records=300, seed=2)
    artifacts = train_anomaly_engine(df, contamination=0.03, random_state=2)
    result = analyze_transaction(artifacts, amount=700, hour=10, day_of_week=2, frequency=1)

    assert "is_anomaly" in result
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0


def test_load_fallback_and_batch_scoring(tmp_path):
    cache = tmp_path / "missing_synthetic.csv"
    df = load_anomaly_dataset(cache_path=str(cache), use_remote=False)
    artifacts = train_anomaly_engine(df, contamination=0.03, random_state=42)

    batch = pd.DataFrame(
        [
            {"amount": 400, "hour": 11, "day_of_week": 1, "frequency": 1},
            {"amount": 8000, "hour": 2, "day_of_week": 6, "frequency": 12},
        ]
    )
    scored = score_transactions(batch, artifacts)
    assert set(["anomaly_score", "is_anomaly", "reasons", "risk_label"]).issubset(scored.columns)
