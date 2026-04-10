import pandas as pd

from src.hackathon_utils import (
    apply_stress_scenario,
    drift_report,
    generate_synthetic_portfolio,
    prepare_features,
)


def test_prepare_features_aligns_schema_and_encodes_categories():
    raw = pd.DataFrame(
        [
            {
                "loan_amnt": 12000,
                "int_rate": 14.2,
                "grade": "C",
                "home_ownership": "MORTGAGE",
                "purpose": "credit_card",
            }
        ]
    )
    features = ["loan_amnt", "int_rate", "grade", "home_ownership", "purpose", "dti"]
    out = prepare_features(raw, features)

    assert out.columns.tolist() == features
    assert out.loc[0, "grade"] == 2
    assert out.loc[0, "home_ownership"] == 2
    assert out.loc[0, "purpose"] == 1
    assert out.loc[0, "dti"] == 0


def test_apply_stress_scenario_increases_risk_drivers():
    base = pd.DataFrame([{"int_rate": 10.0, "annual_inc": 100000, "dti": 12.0, "delinq_2yrs": 0}])
    stressed = apply_stress_scenario(base, rate_bps=150, income_drop_pct=10, dti_bump=4)

    assert stressed.loc[0, "int_rate"] == 11.5
    assert stressed.loc[0, "annual_inc"] == 90000
    assert stressed.loc[0, "dti"] == 16.0
    assert stressed.loc[0, "delinq_2yrs"] == 1


def test_drift_report_returns_ordered_scores():
    baseline = generate_synthetic_portfolio(100, seed=1)
    current = baseline.copy()
    current["int_rate"] = current["int_rate"] + 7
    rep = drift_report(current, baseline, ["int_rate", "dti", "loan_amnt"])

    assert not rep.empty
    assert rep.iloc[0]["feature"] == "int_rate"
    assert rep.iloc[0]["drift_score"] >= rep.iloc[-1]["drift_score"]
