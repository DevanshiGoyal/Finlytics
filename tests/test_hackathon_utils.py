import pandas as pd

from src.hackathon_utils import (
    anomaly_root_cause_hint,
    apply_stress_scenario,
    baseline_backtest,
    compute_baseline_forecasts,
    detect_recent_anomalies,
    drift_report,
    forecast_confidence_label,
    forecast_qa_snapshot,
    generate_forecast_brief_md,
    forecast_with_uncertainty,
    generate_synthetic_portfolio,
    prepare_features,
    recommend_next_action,
    rolling_backtest,
    scenario_forecast,
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


def test_short_horizon_forecast_and_baselines_shape():
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    data = pd.DataFrame({"month_start": months, "funded_amnt_m": [100 + i * 2 for i in range(24)]})

    fc = forecast_with_uncertainty(data, horizon=4)
    base = compute_baseline_forecasts(data, horizon=4)

    assert len(fc) == 4
    assert len(base) == 4
    assert set(["forecast_central", "forecast_lower", "forecast_upper"]).issubset(fc.columns)
    assert set(["naive_last", "moving_avg_3", "exp_smoothing"]).issubset(base.columns)


def test_backtest_and_anomaly_signals_run():
    months = pd.date_range("2021-01-01", periods=30, freq="MS")
    vals = [120 + 1.5 * i for i in range(30)]
    vals[-1] = vals[-1] + 35
    data = pd.DataFrame({"month_start": months, "funded_amnt_m": vals})

    bt = baseline_backtest(data, holdout=4)
    assert not bt.empty
    assert bt.iloc[0]["mape"] <= bt.iloc[-1]["mape"]

    anomalies = detect_recent_anomalies(data, window=6, z_limit=1.8)
    assert set(["month_start", "funded_amnt_m", "lower", "upper", "direction"]).issubset(anomalies.columns) or anomalies.empty


def test_scenario_and_confidence_pipeline_outputs():
    months = pd.date_range("2020-01-01", periods=28, freq="MS")
    data = pd.DataFrame({"month_start": months, "funded_amnt_m": [200 + i * 1.8 for i in range(28)]})

    base_fc = forecast_with_uncertainty(data, horizon=4)
    sc_fc = scenario_forecast(data, horizon=4, growth_adjust_pct=10, remove_recent_outlier=True, pattern_mode="seasonal_plus")
    bt = baseline_backtest(data, holdout=4)
    label, score = forecast_confidence_label(bt, sc_fc)

    assert len(base_fc) == 4 and len(sc_fc) == 4
    assert label in {"High confidence", "Watch closely", "Unstable"}
    assert 0 <= score <= 100


def test_rolling_qa_and_forecast_brief_text():
    months = pd.date_range("2019-01-01", periods=32, freq="MS")
    vals = [150 + (i * 2.2) + (5 if i % 12 == 0 else 0) for i in range(32)]
    data = pd.DataFrame({"month_start": months, "funded_amnt_m": vals})

    rolling = rolling_backtest(data, max_horizon=4)
    qa = forecast_qa_snapshot(data, rolling)
    fc = forecast_with_uncertainty(data, horizon=4)
    base = compute_baseline_forecasts(data, horizon=4)
    anomalies = detect_recent_anomalies(data, window=6, z_limit=1.8)
    action = recommend_next_action("Watch closely", anomalies, 0.03)
    brief = generate_forecast_brief_md(4, fc, base, "Watch closely", 55.0, anomalies, qa, action)

    if not anomalies.empty:
        hint = anomaly_root_cause_hint(data, anomalies.iloc[-1])
        assert isinstance(hint, str) and len(hint) > 0

    assert "FinSight Forecast Brief" in brief
    assert qa["records"] == len(data)
