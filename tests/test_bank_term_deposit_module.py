from src.bank_term_deposit_module import (
    add_binned_features,
    generate_synthetic_bank_data,
    load_bank_dataset,
    preprocess_user_input,
    train_bank_models,
)


def test_generate_synthetic_bank_data_has_expected_columns():
    df = generate_synthetic_bank_data(120, seed=2)
    required = {
        "age", "job", "marital", "education", "default", "balance", "housing", "loan",
        "contact", "month", "day", "duration", "campaign", "pdays", "previous", "poutcome", "y",
    }
    assert required.issubset(df.columns)


def test_add_binned_features_creates_categories():
    df = generate_synthetic_bank_data(10, seed=3)
    out = add_binned_features(df)
    assert "age_category" in out.columns
    assert "balance_category" in out.columns
    assert "duration_category" in out.columns


def test_train_and_preprocess_alignment():
    df = generate_synthetic_bank_data(300, seed=4)
    artifacts = train_bank_models(df, seed=4)

    row = {
        "age": 40,
        "balance": 1500,
        "duration": 420,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day": 14,
        "campaign": 2,
        "pdays": 50,
        "previous": 1,
        "poutcome": "unknown",
    }
    aligned = preprocess_user_input(row, artifacts.feature_columns)
    assert aligned.shape[1] == len(artifacts.feature_columns)


def test_load_bank_dataset_fallback_without_remote(tmp_path):
    cache_file = tmp_path / "missing.csv"
    df = load_bank_dataset(cache_path=str(cache_file), use_remote=False)
    required = {
        "age", "job", "marital", "education", "default", "balance", "housing", "loan",
        "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y",
    }
    assert required.issubset(df.columns)
    assert len(df) > 0
