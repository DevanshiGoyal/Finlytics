from pathlib import Path

import pytest

from src.wealth_persona_module import load_wealth_dataset, run_wealth_persona_pipeline


def test_load_wealth_dataset_requires_real_file(tmp_path):
    missing = tmp_path / "not_present.csv"
    with pytest.raises(FileNotFoundError):
        load_wealth_dataset(dataset_path=str(missing))


def test_load_wealth_dataset_rejects_non_kaggle_schema():
    bank_marketing_csv = Path("data/external/bank_updated.csv")
    if not bank_marketing_csv.exists():
        pytest.skip("bank_updated.csv not found in workspace")

    with pytest.raises(ValueError, match="missing required columns"):
        load_wealth_dataset(dataset_path=str(bank_marketing_csv))


def test_pipeline_fails_fast_when_dataset_missing(tmp_path):
    missing = tmp_path / "missing_kaggle_transactions.csv"
    with pytest.raises(FileNotFoundError):
        run_wealth_persona_pipeline(dataset_path=str(missing), k=4, random_state=42)


def test_load_wealth_dataset_accepts_kaggle_transaction_amount_alias():
    kaggle_csv = Path("data/external/bank_transactions.csv")
    if not kaggle_csv.exists():
        pytest.skip("bank_transactions.csv not found in workspace")

    df = load_wealth_dataset(dataset_path=str(kaggle_csv))
    assert "TransactionAmount (INR)" in df.columns or "TransactionAmount" in df.columns
