# dashboard/app.py
# Finlytics: Financial Predictive Analytics — Streamlit Dashboard

import os
import sys
import warnings
warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for path_candidate in (PROJECT_ROOT, SRC_DIR):
    if path_candidate not in sys.path:
        sys.path.insert(0, path_candidate)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

from hackathon_utils import (
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
    generate_demo_grade_data,
    generate_demo_monthly_data,
    generate_executive_report_md,
    generate_synthetic_portfolio,
    local_feature_impact,
    model_feature_importance,
    prepare_features,
    recommend_next_action,
    rolling_backtest,
    score_with_models,
    scenario_forecast,
)
from bank_term_deposit_module import (
    generate_llm_campaign_advice,
    generate_synthetic_bank_data,
    load_bank_dataset,
    predict_subscription,
    preprocess_user_input,
    top_feature_importance,
    train_bank_models,
)
from bank_anomaly_module import (
    analyze_transaction,
    load_anomaly_dataset,
    score_transactions,
    train_anomaly_engine,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finlytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 1.2rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(180deg, #eef3f9 0%, #e7edf6 100%);
        border: 1px solid #d2dceb;
        border-radius: 12px;
        padding: 0.32rem;
        gap: 0.28rem;
        overflow-x: auto;
        scrollbar-width: thin;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 9px;
        padding: 0.48rem 0.88rem;
        min-height: 42px;
        color: #263547;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"] p {
        font-size: 1.05rem;
        line-height: 1.2;
        white-space: nowrap;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #0f2e53;
        border-color: #0f2e53;
        color: #f6fbff;
        box-shadow: 0 4px 12px rgba(15, 46, 83, 0.25);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] p {
        color: #f6fbff;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: transparent;
    }

    .finlytics-hero {
        background: linear-gradient(122deg, #0b2545 0%, #133a63 46%, #1c6e8c 100%);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 12px 28px rgba(11, 37, 69, 0.22);
    }

    .finlytics-hero-title {
        margin: 0;
        color: #f7fbff;
        font-size: 2.55rem;
        line-height: 1.2;
        font-weight: 800;
    }

    .finlytics-hero-subtitle {
        margin: 0.6rem 0 0 0;
        color: #d7e7f6;
        font-size: 1.18rem;
        font-style: italic;
        line-height: 1.45;
    }

    .finsight-guide {
        background: linear-gradient(180deg, #f8fbff 0%, #f2f7fc 100%);
        border: 1px solid #d8e2f0;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin: 0.35rem 0 0.9rem 0;
    }

    .finsight-guide strong {
        color: #0f2e53;
    }

    .finsight-chip {
        display: inline-block;
        background: #edf4ff;
        border: 1px solid #cfdff7;
        color: #0f2e53;
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        margin-right: 0.35rem;
        font-size: 0.82rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = PROJECT_ROOT

def p(path):
    return os.path.join(BASE_DIR, path)


def render_tab_guide(user_action: str, user_output: str):
    st.markdown(
        f"""
        <div class="finsight-guide">
            <div><strong>What you do:</strong> {user_action}</div>
            <div><strong>What you get:</strong> {user_output}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_module1_models():
    return {
        'xgb':      joblib.load(p('models/module1_xgb.pkl')),
        'rf':       joblib.load(p('models/module1_rf.pkl')),
        'lr':       joblib.load(p('models/module1_lr.pkl')),
        'scaler':   joblib.load(p('models/module1_scaler.pkl')),
        'features': joblib.load(p('models/module1_features.pkl')),
    }

@st.cache_resource
def load_module2_models():
    return {
        'xgb':      joblib.load(p('models/module2_xgb.pkl')),
        'rf':       joblib.load(p('models/module2_rf.pkl')),
        'lr':       joblib.load(p('models/module2_lr.pkl')),
        'scaler':   joblib.load(p('models/module2_scaler.pkl')),
        'features': joblib.load(p('models/module2_features.pkl')),
    }

@st.cache_resource
def load_module3_models():
    return {
        'xgb':      joblib.load(p('models/module3_xgb.pkl')),
        'lr':       joblib.load(p('models/module3_lr.pkl')),
        'scaler':   joblib.load(p('models/module3_scaler.pkl')),
        'features': joblib.load(p('models/module3_features.pkl')),
    }

@st.cache_data
def load_monthly_data():
    path = p('data/processed/monthly_loan_volume.csv')
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['month_start'])
    return generate_demo_monthly_data()

@st.cache_data
def load_grade_data():
    path = p('data/processed/grade_monthly_demand.csv')
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=['month_start'])
    return generate_demo_grade_data()


@st.cache_resource
def load_bank_term_deposit_defaults():
    raw_df = load_bank_dataset()
    artifacts = train_bank_models(raw_df, seed=42)
    return artifacts, raw_df


@st.cache_resource
def load_bank_anomaly_defaults():
    raw_df = load_anomaly_dataset()
    artifacts = train_anomaly_engine(raw_df, contamination=0.03, random_state=42)
    return artifacts, raw_df

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="finlytics-hero">
        <h1 class="finlytics-hero-title"> Finlytics: Financial Predictive Analytics</h1>
        <p class="finlytics-hero-subtitle">Plan ahead with clear forecasts, uncertainty ranges, and early warning signals.</p>
    </div>
    """,
    unsafe_allow_html=True,)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Default Risk",
    "Return Risk",
    "Volume Forecast",
    "Demand by Grade",
    "Portfolio Intelligence Hub",
    "Bank Deposit AI",
    "Anomaly Watch",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODULE 1: LOAN DEFAULT RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Loan Default Risk")
    st.markdown("Estimate the chance that a borrower may miss repayment.")
    render_tab_guide(
        "Enter borrower and loan details.",
        "A clear risk score with low, medium, or high signal.",
    )

    try:
        m1 = load_module1_models()

        st.subheader("Model Performance")
        st.caption("AUC closer to 1.0 means better separation between safer and riskier borrowers.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Logistic Regression AUC", "0.6597")
        col2.metric("Random Forest AUC", "0.6831")
        col3.metric("XGBoost AUC", "0.7168", delta="Champion ✓")

        st.markdown("---")
        st.subheader("Try it: Predict Default Risk")
        st.markdown("Enter borrower details to get a default probability score.")

        col1, col2, col3 = st.columns(3)

        with col1:
            loan_amnt    = st.number_input("Loan Amount ($)", 1000, 40000, 10000, step=500)
            int_rate     = st.slider("Interest Rate (%)", 5.0, 30.0, 13.0, step=0.5)
            installment  = st.number_input("Monthly Installment ($)", 50, 1500, 300, step=10)
            annual_inc   = st.number_input("Annual Income ($)", 20000, 300000, 65000, step=1000)
            dti          = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0, step=0.5)
            delinq_2yrs  = st.number_input("Delinquencies (2yrs)", 0, 10, 0)

        with col2:
            inq_last_6mths = st.number_input("Credit Inquiries (6mo)", 0, 10, 1)
            open_acc       = st.number_input("Open Credit Lines", 1, 40, 10)
            pub_rec        = st.number_input("Public Records", 0, 5, 0)
            revol_util     = st.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0)
            total_acc      = st.number_input("Total Credit Lines", 1, 80, 25)
            emp_length     = st.slider("Employment Length (years)", 0, 10, 5)

        with col3:
            grade          = st.selectbox("Loan Grade", ['A','B','C','D','E','F','G'])
            home_ownership = st.selectbox("Home Ownership", ['RENT','OWN','MORTGAGE','OTHER'])
            purpose        = st.selectbox("Loan Purpose", [
                'debt_consolidation','credit_card','home_improvement',
                'other','major_purchase','medical','small_business'
            ])
            issue_year  = st.selectbox("Issue Year", list(range(2015, 2024)))
            issue_month = st.selectbox("Issue Month", list(range(1, 13)))

        grade_map    = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
        home_map     = {'RENT':0,'OWN':1,'MORTGAGE':2,'OTHER':3}
        purpose_map  = {
            'debt_consolidation':0,'credit_card':1,'home_improvement':2,
            'other':3,'major_purchase':4,'medical':5,'small_business':6
        }

        input_data = pd.DataFrame([{
            'loan_amnt':       loan_amnt,
            'int_rate':        int_rate,
            'installment':     installment,
            'annual_inc':      annual_inc,
            'dti':             dti,
            'delinq_2yrs':     delinq_2yrs,
            'inq_last_6mths':  inq_last_6mths,
            'open_acc':        open_acc,
            'pub_rec':         pub_rec,
            'revol_util':      revol_util,
            'total_acc':       total_acc,
            'emp_length':      emp_length,
            'grade':           grade_map[grade],
            'home_ownership':  home_map[home_ownership],
            'purpose':         purpose_map[purpose],
            'issue_year':      issue_year,
            'issue_month':     issue_month,
        }])

        if st.button("🔍 Predict Default Risk", key="m1_predict"):
            prob = m1['xgb'].predict_proba(input_data)[0][1]

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                if prob >= 0.5:
                    st.error(f"⚠️ High Default Risk: {prob:.1%}")
                elif prob >= 0.3:
                    st.warning(f"⚡ Medium Default Risk: {prob:.1%}")
                else:
                    st.success(f"✅ Low Default Risk: {prob:.1%}")

            with col2:
                st.metric("Default Probability", f"{prob:.1%}")
                st.metric("Decision Threshold", "0.30")
                st.metric("Prediction", "DEFAULT" if prob >= 0.3 else "REPAY")

    except Exception as e:
        st.error(f"Error loading Module 1 models: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODULE 2: BORROWER CHURN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Borrower Return Risk")
    st.markdown("Estimate whether a borrower is unlikely to come back for the next loan.")
    render_tab_guide(
        "Enter customer profile and repayment behavior factors.",
        "A churn probability plus a simple retained vs churned decision hint.",
    )

    try:
        m2 = load_module2_models()

        st.subheader("Model Performance")
        st.caption("Higher AUC means the model is better at distinguishing likely return vs likely churn.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Logistic Regression AUC", "0.7823")
        col2.metric("Random Forest AUC", "0.7906")
        col3.metric("XGBoost AUC", "0.8139", delta="Champion ✓")

        st.markdown("---")
        st.subheader("Try it: Predict Churn Probability")

        col1, col2, col3 = st.columns(3)

        with col1:
            loan_amnt_c    = st.number_input("Loan Amount ($)", 1000, 40000, 10000, step=500, key="c_loan")
            int_rate_c     = st.slider("Interest Rate (%)", 5.0, 30.0, 13.0, step=0.5, key="c_rate")
            installment_c  = st.number_input("Monthly Installment ($)", 50, 1500, 300, step=10, key="c_inst")
            annual_inc_c   = st.number_input("Annual Income ($)", 20000, 300000, 65000, step=1000, key="c_inc")
            dti_c          = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0, step=0.5, key="c_dti")

        with col2:
            emp_length_c     = st.slider("Employment Length (years)", 0, 10, 5, key="c_emp")
            home_ownership_c = st.selectbox("Home Ownership", ['RENT','OWN','MORTGAGE','OTHER'], key="c_home")
            purpose_c        = st.selectbox("Loan Purpose", [
                'debt_consolidation','credit_card','home_improvement',
                'other','major_purchase','medical','small_business'
            ], key="c_purpose")
            grade_c          = st.selectbox("Loan Grade", ['A','B','C','D','E','F','G'], key="c_grade")

        with col3:
            open_acc_c    = st.number_input("Open Credit Lines", 1, 40, 10, key="c_open")
            total_acc_c   = st.number_input("Total Credit Lines", 1, 80, 25, key="c_total")
            revol_util_c  = st.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0, key="c_revol")
            delinq_2yrs_c = st.number_input("Delinquencies (2yrs)", 0, 10, 0, key="c_delinq")
            issue_year_c  = st.selectbox("Issue Year", list(range(2015, 2024)), key="c_year")
            issue_month_c = st.selectbox("Issue Month", list(range(1, 13)), key="c_month")

        grade_map2   = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
        home_map2    = {'RENT':0,'OWN':1,'MORTGAGE':2,'OTHER':3}
        purpose_map2 = {
            'debt_consolidation':0,'credit_card':1,'home_improvement':2,
            'other':3,'major_purchase':4,'medical':5,'small_business':6
        }

        input_data_c = pd.DataFrame([{
            'loan_amnt':      loan_amnt_c,
            'int_rate':       int_rate_c,
            'installment':    installment_c,
            'annual_inc':     annual_inc_c,
            'dti':            dti_c,
            'emp_length':     emp_length_c,
            'home_ownership': home_map2[home_ownership_c],
            'purpose':        purpose_map2[purpose_c],
            'grade':          grade_map2[grade_c],
            'open_acc':       open_acc_c,
            'total_acc':      total_acc_c,
            'revol_util':     revol_util_c,
            'delinq_2yrs':    delinq_2yrs_c,
            'issue_year':     issue_year_c,
            'issue_month':    issue_month_c,
        }])

        if st.button("🔍 Predict Churn", key="m2_predict"):
            prob_c = m2['xgb'].predict_proba(input_data_c)[0][1]

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                if prob_c >= 0.6:
                    st.error(f"🚨 High Churn Risk: {prob_c:.1%}")
                elif prob_c >= 0.4:
                    st.warning(f"⚡ Medium Churn Risk: {prob_c:.1%}")
                else:
                    st.success(f"✅ Low Churn Risk — Likely to Return: {prob_c:.1%}")

            with col2:
                st.metric("Churn Probability", f"{prob_c:.1%}")
                st.metric("Prediction", "CHURNED" if prob_c >= 0.5 else "RETAINED")

    except Exception as e:
        st.error(f"Error loading Module 2 models: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODULE 3: LOAN VOLUME FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Loan Volume Forecast")
    st.markdown("See what the next months may look like, with honest uncertainty and baseline comparisons.")
    render_tab_guide(
        "Choose forecast horizon and optional what-if scenario settings.",
        "Low, likely, high outcomes, confidence meter, warnings, and recommended actions.",
    )

    st.subheader("Model Performance (Test period: 2015)")
    st.caption("Lower MAPE is better. It means forecast error is smaller.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression MAPE", "1.92%", delta="Champion ✓")
    col2.metric("XGBoost MAPE", "10.30%")
    col3.metric("Prophet MAPE", "29.87%")

    st.markdown("---")

    try:
        monthly = load_monthly_data().copy()
        monthly = monthly.sort_values('month_start').reset_index(drop=True)

        explain_mode = st.radio(
            "Explanation mode",
            ["Business", "Technical"],
            horizontal=True,
            help="Switch between concise decision language and detailed model diagnostics.",
        )

        col_h1, col_h2 = st.columns([2, 3])
        with col_h1:
            horizon = st.slider("Forecast horizon (months)", 1, 6, 4, help="Short-horizon planning window on existing monthly dataset.")
        with col_h2:
            st.caption("Shows low / likely / high outcomes and compares against simple baselines to keep forecasts honest.")

        forecast_df = forecast_with_uncertainty(monthly, horizon=horizon)
        baseline_df = compute_baseline_forecasts(monthly, horizon=horizon)
        merged_fc = forecast_df.merge(baseline_df, on='month_start', how='left')

        st.markdown("---")
        st.subheader("Scenario Sandbox")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            growth_adjust = st.slider("Adjust growth (%)", -20, 20, 0, step=1)
        with sc2:
            remove_outlier = st.checkbox("Remove recent outliers", value=False)
        with sc3:
            pattern_mode = st.selectbox(
                "Pattern mode",
                ["baseline", "flat", "seasonal_plus"],
                format_func=lambda x: {
                    "baseline": "Baseline pattern",
                    "flat": "Flat trend",
                    "seasonal_plus": "Seasonal boost",
                }[x],
            )

        scenario_df = scenario_forecast(
            monthly,
            horizon=horizon,
            growth_adjust_pct=float(growth_adjust),
            remove_recent_outlier=bool(remove_outlier),
            pattern_mode=pattern_mode,
        )
        scenario_merged = scenario_df.merge(baseline_df, on="month_start", how="left")

        st.subheader("Historical + Forecast Range")
        fig, ax = plt.subplots(figsize=(12, 4.6))
        ax.plot(monthly['month_start'], monthly['funded_amnt_m'], color='steelblue', linewidth=1.5, label='Historical')
        ax.plot(merged_fc['month_start'], merged_fc['forecast_central'], color='#0f2e53', linewidth=2.2, label='FinSight forecast')
        ax.plot(scenario_df['month_start'], scenario_df['forecast_central'], color='#d95f02', linewidth=2.0, linestyle='--', label='Scenario forecast')
        ax.fill_between(
            merged_fc['month_start'],
            merged_fc['forecast_lower'],
            merged_fc['forecast_upper'],
            color='#1c6e8c',
            alpha=0.2,
            label='Uncertainty band',
        )
        ax.set_title('Loan Volume Forecast with Uncertainty Band (USD Millions)')
        ax.set_ylabel('Funded Amount (USD M)')
        ax.grid(alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Forecast Comparison Table")
        table_out = merged_fc.copy()
        table_out['month_start'] = table_out['month_start'].dt.strftime('%Y-%m')
        table_out = table_out.rename(columns={
            'month_start': 'Month',
            'forecast_central': 'Likely',
            'forecast_lower': 'Low',
            'forecast_upper': 'High',
            'naive_last': 'Simple baseline (last value)',
            'moving_avg_3': 'Simple baseline (moving average)',
            'exp_smoothing': 'Simple baseline (smoothed)',
        })
        st.dataframe(table_out, use_container_width=True, hide_index=True)

        delta_df = scenario_df[["month_start", "forecast_central"]].merge(
            merged_fc[["month_start", "forecast_central"]],
            on="month_start",
            suffixes=("_scenario", "_baseline"),
        )
        delta_df["delta"] = delta_df["forecast_central_scenario"] - delta_df["forecast_central_baseline"]
        delta_df["month_start"] = delta_df["month_start"].dt.strftime("%Y-%m")
        st.caption("Scenario vs baseline delta")
        st.dataframe(
            delta_df.rename(
                columns={
                    "month_start": "Month",
                    "forecast_central_scenario": "Scenario Likely",
                    "forecast_central_baseline": "Baseline Likely",
                    "delta": "Delta",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("Reliability Check Against Simple Methods")
        backtest_df = baseline_backtest(monthly, holdout=horizon)
        best_row = backtest_df.iloc[0]
        m1, m2 = st.columns(2)
        m1.metric("Most accurate on recent data", best_row['model'])
        m2.metric("Best MAPE", f"{best_row['mape']:.2f}%")
        st.dataframe(backtest_df.rename(columns={'model': 'Model', 'mape': 'MAPE (%)'}), use_container_width=True, hide_index=True)

        conf_label, conf_score = forecast_confidence_label(backtest_df, merged_fc)
        cfm1, cfm2 = st.columns(2)
        cfm1.metric("Forecast confidence", conf_label)
        cfm2.metric("Confidence score", f"{conf_score:.1f}/100")

        st.markdown("---")
        st.subheader("Forecast Quality Over Time")
        rolling_df = rolling_backtest(monthly, max_horizon=6)
        if rolling_df.empty:
            st.info("Not enough history yet to run the quality-over-time check.")
        else:
            rfig, rax = plt.subplots(figsize=(10.5, 3.8))
            rax.plot(rolling_df["horizon"], rolling_df["mape"], marker="o", color="#0f2e53", label="MAPE (%)")
            rax.plot(rolling_df["horizon"], rolling_df["coverage"], marker="s", color="#1c6e8c", label="Coverage (%)")
            rax.set_xlabel("Horizon (months)")
            rax.set_ylabel("Score")
            rax.set_title("Forecast Quality by Horizon")
            rax.grid(alpha=0.3)
            rax.legend()
            st.pyplot(rfig)
            plt.close(rfig)
            st.dataframe(
                rolling_df.rename(columns={"horizon": "Horizon", "mape": "MAPE (%)", "coverage": "Coverage (%)"}),
                use_container_width=True,
                hide_index=True,
            )

        qa_snapshot = forecast_qa_snapshot(monthly, rolling_df)
        qa1, qa2, qa3, qa4 = st.columns(4)
        qa1.metric("Records", f"{qa_snapshot['records']}")
        qa2.metric("Missing ratio", f"{qa_snapshot['missing_ratio']:.2f}%")
        qa3.metric("Data lag", f"{qa_snapshot['data_lag_days']:.0f} days")
        qa4.metric("Avg rolling MAPE", "N/A" if np.isnan(qa_snapshot['avg_rolling_mape']) else f"{qa_snapshot['avg_rolling_mape']:.2f}%")

        st.markdown("---")
        st.subheader("Early Warning Signals")
        anomalies = detect_recent_anomalies(monthly, window=6, z_limit=1.8)
        if anomalies.empty:
            st.success("No major recent spikes or dips outside the normal expected range.")
            root_cause_text = "No material warning signal in recent months."
        else:
            latest = anomalies.iloc[-1]
            st.warning(
                f"Recent {latest['direction'].lower()} detected on {latest['month_start'].strftime('%Y-%m')} "
                f"(actual {latest['funded_amnt_m']:.2f} vs expected range {latest['lower']:.2f}-{latest['upper']:.2f})."
            )
            root_cause_text = anomaly_root_cause_hint(monthly, latest)
            st.caption(f"Likely drivers: {root_cause_text}")
            preview = anomalies.copy()
            preview['month_start'] = preview['month_start'].dt.strftime('%Y-%m')
            st.dataframe(preview.rename(columns={
                'month_start': 'Month',
                'funded_amnt_m': 'Actual',
                'lower': 'Expected Low',
                'upper': 'Expected High',
                'direction': 'Signal',
            }), use_container_width=True, hide_index=True)

        total_growth = (scenario_merged['forecast_central'].iloc[-1] - monthly['funded_amnt_m'].iloc[-1]) / max(1e-6, monthly['funded_amnt_m'].iloc[-1])
        low_bound = (scenario_merged['forecast_lower'].iloc[-1] - monthly['funded_amnt_m'].iloc[-1]) / max(1e-6, monthly['funded_amnt_m'].iloc[-1])
        high_bound = (scenario_merged['forecast_upper'].iloc[-1] - monthly['funded_amnt_m'].iloc[-1]) / max(1e-6, monthly['funded_amnt_m'].iloc[-1])

        action_text = recommend_next_action(conf_label, anomalies, total_growth)
        st.subheader("Recommended Next Action")
        st.info(action_text)
        rolling_coverage_text = "N/A" if np.isnan(qa_snapshot['avg_rolling_coverage']) else f"{qa_snapshot['avg_rolling_coverage']:.2f}%"

        if explain_mode == "Business":
            st.success(
                f"Outlook for next {horizon} months: likely **{total_growth:+.1%}** change, "
                f"with uncertainty range **{low_bound:+.1%} to {high_bound:+.1%}**."
            )
        else:
            st.info(
                f"Technical summary: scenario growth adjust={growth_adjust:+d}%, pattern={pattern_mode}, "
                f"confidence={conf_label} ({conf_score:.1f}/100), rolling coverage="
                f"{rolling_coverage_text}"
            )

        brief_md = generate_forecast_brief_md(
            horizon=horizon,
            forecast_df=scenario_df,
            baseline_df=baseline_df,
            confidence_label=conf_label,
            confidence_score=conf_score,
            anomalies=anomalies,
            qa_snapshot=qa_snapshot,
            action_text=action_text,
        )
        st.download_button(
            "Download forecast brief (Markdown)",
            data=brief_md,
            file_name="finsight_forecast_brief.md",
            mime="text/markdown",
            key="download_forecast_brief",
        )
        st.text_area("Forecast brief preview", brief_md, height=200)

    except Exception as e:
        st.warning(f"Could not build forecast insights: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODULE 4: CREDIT DEMAND BY GRADE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Credit Demand Forecast by Grade")
    st.markdown("Understand which credit segments are growing or slowing down over time.")
    render_tab_guide(
        "Pick one or more grades to visualize.",
        "Trend lines and heatmaps showing where demand is strongest.",
    )

    st.subheader("Model Performance — MAPE (%) by Grade")
    mape_df = pd.DataFrame({
        'Model':             ['Linear Regression', 'Random Forest', 'XGBoost'],
        'Grade A':           [2.72,  33.42, 18.03],
        'Grade B':           [4.97,  26.56, 12.33],
        'Grade C':           [5.03,  20.96, 12.64],
        'Grade D':           [13.10, 14.35,  8.55],
        'Grade E':           [8.73,  20.99, 19.45],
    })
    st.dataframe(mape_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    try:
        grade_monthly = load_grade_data()

        st.subheader("Historical Demand by Grade")

        selected_grades = st.multiselect(
            "Select grades to display:",
            ['A', 'B', 'C', 'D', 'E'],
            default=['A', 'B', 'C']
        )

        if selected_grades:
            colors = {'A': 'steelblue', 'B': 'seagreen', 'C': 'darkorange', 'D': 'crimson', 'E': 'purple'}
            fig, ax = plt.subplots(figsize=(12, 5))

            for grade in selected_grades:
                gdf = grade_monthly[grade_monthly['grade'] == grade]
                ax.plot(gdf['month_start'], gdf['funded_amnt_m'],
                        label=f'Grade {grade}', color=colors[grade], linewidth=1.5)

            ax.set_title('Monthly Loan Demand by Credit Grade (USD Millions)')
            ax.set_ylabel('Funded Amount (USD M)')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("#### Historical Demand Heatmaps by Grade")
            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            heatmap_cols = st.columns(2)

            for idx, grade in enumerate(selected_grades):
                with heatmap_cols[idx % 2]:
                    gdf = grade_monthly[grade_monthly['grade'] == grade].copy()
                    if gdf.empty:
                        st.info(f"No historical demand data available for Grade {grade}.")
                        continue

                    gdf['year'] = pd.to_datetime(gdf['month_start']).dt.year
                    gdf['month'] = pd.to_datetime(gdf['month_start']).dt.month
                    heatmap_df = (
                        gdf.pivot_table(
                            index='year',
                            columns='month',
                            values='funded_amnt_m',
                            aggfunc='mean',
                        )
                        .reindex(columns=range(1, 13))
                        .sort_index()
                    )

                    fig_hm, ax_hm = plt.subplots(figsize=(7.2, 3.6))
                    hm_values = heatmap_df.fillna(0).values
                    im = ax_hm.imshow(hm_values, aspect='auto', cmap='YlGnBu')

                    ax_hm.set_title(f'Grade {grade} Monthly Demand Heatmap', fontsize=11)
                    ax_hm.set_xlabel('Month')
                    ax_hm.set_ylabel('Year')
                    ax_hm.set_xticks(np.arange(12))
                    ax_hm.set_xticklabels(month_labels, fontsize=8)
                    ax_hm.set_yticks(np.arange(len(heatmap_df.index)))
                    ax_hm.set_yticklabels(heatmap_df.index.astype(str), fontsize=8)

                    cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
                    cbar.set_label('Funded Amount (USD M)', fontsize=8)

                    plt.tight_layout()
                    st.pyplot(fig_hm)
                    plt.close(fig_hm)

    except Exception as e:
        st.warning(f"Could not load grade data: {e}")

    st.markdown("---")
    st.info("💡 **Key Insight:** Linear Regression was the champion model for Grade A, B, C, and E. XGBoost only won on Grade D — the highest-risk segment — where demand volatility justified a more complex model.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PORTFOLIO INTELLIGENCE HUB
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Portfolio Intelligence Hub")
    st.markdown("A control center for stress testing, batch scoring, explainability, and decision-ready export.")
    render_tab_guide(
        "Upload or generate a portfolio, then run scenario and risk analysis.",
        "Risk bands, drift alerts, key drivers, and an executive-ready summary.",
    )

    try:
        m1 = load_module1_models()
        m2 = load_module2_models()

        st.subheader("1) Scenario & Stress Testing")
        base_profile = pd.DataFrame([
            {
                'loan_amnt': 12000,
                'int_rate': 13.0,
                'installment': 310,
                'annual_inc': 65000,
                'dti': 15.0,
                'delinq_2yrs': 0,
                'inq_last_6mths': 1,
                'open_acc': 10,
                'pub_rec': 0,
                'revol_util': 48,
                'total_acc': 24,
                'emp_length': 5,
                'grade': 'C',
                'home_ownership': 'MORTGAGE',
                'purpose': 'debt_consolidation',
                'issue_year': 2021,
                'issue_month': 6,
            }
        ])

        s1, s2, s3 = st.columns(3)
        with s1:
            rate_bps = st.slider("Rate shock (bps)", 0, 500, 100, step=25)
        with s2:
            income_drop = st.slider("Income drop (%)", 0, 40, 10, step=1)
        with s3:
            dti_bump = st.slider("DTI increase", 0.0, 15.0, 3.0, step=0.5)

        stressed_profile = apply_stress_scenario(base_profile, rate_bps, income_drop, dti_bump)
        base_m1 = prepare_features(base_profile, m1['features'])
        stress_m1 = prepare_features(stressed_profile, m1['features'])
        base_m2 = prepare_features(base_profile, m2['features'])
        stress_m2 = prepare_features(stressed_profile, m2['features'])

        base_default = m1['xgb'].predict_proba(base_m1)[0][1]
        stress_default = m1['xgb'].predict_proba(stress_m1)[0][1]
        base_churn = m2['xgb'].predict_proba(base_m2)[0][1]
        stress_churn = m2['xgb'].predict_proba(stress_m2)[0][1]

        c1, c2 = st.columns(2)
        c1.metric("Default Probability", f"{stress_default:.1%}", delta=f"{(stress_default - base_default):+.1%}")
        c2.metric("Churn Probability", f"{stress_churn:.1%}", delta=f"{(stress_churn - base_churn):+.1%}")

        st.markdown("---")
        st.subheader("2) Portfolio Risk Scoring")
        st.caption("Upload CSV with borrower columns (or use generated demo data).")

        sample_df = generate_synthetic_portfolio(300)
        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download sample portfolio CSV",
            data=sample_csv,
            file_name="sample_portfolio.csv",
            mime="text/csv",
            key="download_sample_portfolio"
        )

        uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"], key="portfolio_upload")
        portfolio_df = sample_df.copy() if uploaded is None else pd.read_csv(uploaded)
        scored = score_with_models(portfolio_df, m1, m2)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Records scored", f"{len(scored):,}")
        k2.metric("Avg default prob", f"{scored['default_probability'].mean():.1%}")
        k3.metric("Avg churn prob", f"{scored['churn_probability'].mean():.1%}")
        k4.metric("High risk share", f"{(scored['risk_band'].astype(str) == 'High').mean():.1%}")

        st.dataframe(
            scored[[
                'loan_amnt', 'int_rate', 'grade', 'purpose',
                'default_probability', 'churn_probability', 'risk_score', 'risk_band'
            ]].head(30),
            use_container_width=True,
            hide_index=True,
        )

        scored_csv = scored.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download scored portfolio",
            data=scored_csv,
            file_name="scored_portfolio.csv",
            mime="text/csv",
            key="download_scored_portfolio"
        )

        st.markdown("---")
        st.subheader("3) Explainability Center")
        m1_input = prepare_features(portfolio_df, m1['features'])
        importances = model_feature_importance(m1['xgb'], m1['features'])
        baseline_means = m1_input.mean(numeric_only=True)

        row_idx = st.slider("Choose record index", 0, max(0, len(m1_input) - 1), 0, key="explain_row")
        local_imp = local_feature_impact(m1_input.iloc[row_idx], baseline_means, importances, top_n=8)

        e1, e2 = st.columns(2)
        with e1:
            st.caption("Top global risk drivers")
            st.bar_chart(importances.head(10).set_index('feature'))
        with e2:
            st.caption("Top local drivers for selected record")
            st.bar_chart(local_imp.set_index('feature')[['impact']])

        st.dataframe(local_imp, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("4) Drift Monitor")
        baseline = generate_synthetic_portfolio(len(portfolio_df), seed=11)
        drift_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util', 'total_acc']
        current_numeric = prepare_features(portfolio_df, drift_cols)
        baseline_numeric = prepare_features(baseline, drift_cols)
        drift_df = drift_report(current_numeric, baseline_numeric, drift_cols)

        if drift_df.empty:
            st.info("Drift report unavailable: no overlapping numeric features.")
        else:
            high_drift_count = int((drift_df['status'] == 'High').sum())
            if high_drift_count > 0:
                st.warning(f"⚠️ {high_drift_count} feature(s) show high drift. Consider retraining checks.")
            else:
                st.success("✅ No high drift signals detected in monitored features.")
            st.dataframe(drift_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("5) Executive Report Export")
        summary = {
            'avg_default_probability': float(scored['default_probability'].mean()),
            'avg_churn_probability': float(scored['churn_probability'].mean()),
            'high_risk_share': float((scored['risk_band'].astype(str) == 'High').mean()),
            'expected_defaults': float(scored['default_probability'].sum()),
        }
        report_md = generate_executive_report_md(summary, drift_df)
        st.download_button(
            "Download executive risk brief (Markdown)",
            data=report_md,
            file_name="finlytics_executive_brief.md",
            mime="text/markdown",
            key="download_exec_report"
        )
        st.text_area("Report preview", report_md, height=240)

    except Exception as e:
        st.error(f"Portfolio Intelligence Hub could not load: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — BANK TERM DEPOSIT AI (adapted capability)
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Bank Term Deposit Prediction")
    st.markdown("Predict who is likely to subscribe to a term deposit and improve campaign targeting.")
    render_tab_guide(
        "Use the default dataset or upload your own and run predictions.",
        "Subscription probability, best model snapshot, and top influencing factors.",
    )

    try:
        default_artifacts, default_raw_df = load_bank_term_deposit_defaults()
        if "bank_artifacts" not in st.session_state:
            st.session_state["bank_artifacts"] = default_artifacts
            st.session_state["bank_raw_df"] = default_raw_df

        artifacts = st.session_state["bank_artifacts"]
        bank_df = st.session_state["bank_raw_df"]

        st.subheader("Training Data")
        uploaded_bank = st.file_uploader(
            "Upload bank marketing CSV with target column `y` (yes/no) to retrain models",
            type=["csv"],
            key="bank_module_upload",
        )

        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            st.caption(f"Current dataset size: {len(bank_df):,} rows")
        with col_u2:
            if st.button("Use synthetic default", key="bank_reset_default"):
                st.session_state["bank_artifacts"] = default_artifacts
                st.session_state["bank_raw_df"] = default_raw_df
                artifacts = default_artifacts
                bank_df = default_raw_df
                st.success("Reset to synthetic default dataset.")

        if uploaded_bank is not None and st.button("Retrain models on uploaded data", key="bank_retrain_uploaded"):
            up_df = pd.read_csv(uploaded_bank)
            required_cols = {
                'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                'contact', 'month', 'day', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'
            }
            missing = sorted(required_cols - set(up_df.columns))
            if missing:
                st.error(f"Uploaded file is missing required columns: {missing}")
            else:
                st.session_state["bank_artifacts"] = train_bank_models(up_df)
                st.session_state["bank_raw_df"] = up_df
                artifacts = st.session_state["bank_artifacts"]
                bank_df = up_df
                st.success("Models retrained on uploaded dataset.")

        st.markdown("---")
        st.subheader("Model Leaderboard")
        st.caption("These scores show how well each model predicts subscription on validation data.")
        metrics_df = artifacts.metrics.copy()
        display_metrics = metrics_df.copy()
        for col in ["accuracy", "precision", "recall", "f1"]:
            display_metrics[col] = (display_metrics[col] * 100).round(2).astype(str) + "%"
        st.dataframe(display_metrics, use_container_width=True, hide_index=True)
        st.info(f"Champion model: **{artifacts.best_model_name}**")

        st.markdown("---")
        st.subheader("Interactive Prediction")

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 80, 36, key="bank_age")
            balance = st.number_input("Balance", min_value=-5000, max_value=50000, value=1200, step=100, key="bank_balance")
            duration = st.slider("Call duration (sec)", 30, 3000, 360, key="bank_duration")
            day = st.slider("Last contact day", 1, 31, 15, key="bank_day")
            campaign = st.slider("Campaign contacts", 1, 63, 2, key="bank_campaign")

        with c2:
            pdays = st.slider("Days since previous contact", -1, 900, 40, key="bank_pdays")
            previous = st.slider("Previous contacts", 0, 50, 1, key="bank_previous")
            job = st.selectbox("Job", sorted(bank_df['job'].astype(str).unique().tolist()), key="bank_job")
            marital = st.selectbox("Marital", sorted(bank_df['marital'].astype(str).unique().tolist()), key="bank_marital")
            education = st.selectbox("Education", sorted(bank_df['education'].astype(str).unique().tolist()), key="bank_education")

        with c3:
            default = st.selectbox("Credit default", sorted(bank_df['default'].astype(str).unique().tolist()), key="bank_default")
            housing = st.selectbox("Housing loan", sorted(bank_df['housing'].astype(str).unique().tolist()), key="bank_housing")
            loan = st.selectbox("Personal loan", sorted(bank_df['loan'].astype(str).unique().tolist()), key="bank_loan")
            contact = st.selectbox("Contact type", sorted(bank_df['contact'].astype(str).unique().tolist()), key="bank_contact")
            month = st.selectbox("Last contact month", sorted(bank_df['month'].astype(str).unique().tolist()), key="bank_month")
            poutcome = st.selectbox("Previous outcome", sorted(bank_df['poutcome'].astype(str).unique().tolist()), key="bank_poutcome")

        model_choice = st.selectbox("Prediction model", artifacts.metrics["model"].tolist(), index=0, key="bank_model_pick")
        selected_model = artifacts.models[model_choice]

        user_row = {
            'age': age,
            'balance': balance,
            'duration': duration,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'day': day,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome,
        }

        input_features = preprocess_user_input(user_row, artifacts.feature_columns)
        pred, prob = predict_subscription(selected_model, input_features)

        r1, r2 = st.columns(2)
        with r1:
            if pred == 1:
                st.success(f"Likely to subscribe ✅ ({prob:.1%})")
            else:
                st.warning(f"Unlikely to subscribe ⚠️ ({prob:.1%})")
        with r2:
            st.metric("Subscription probability", f"{prob:.1%}")
            st.metric("Model used", model_choice)

        top_features_df = top_feature_importance(selected_model, artifacts.feature_columns, n=8)
        st.caption("Top feature importances for selected model")
        st.bar_chart(top_features_df.set_index("feature"))
        st.dataframe(top_features_df.head(8), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Suggested Next Campaign Step")
        suggestion = generate_llm_campaign_advice(
            model_name=model_choice,
            prediction_prob=prob,
            top_features=top_features_df,
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        st.markdown(suggestion)

    except Exception as e:
        st.error(f"Bank Deposit module could not load: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DEPOSIT ANOMALY DETECTION (Bank Sentinel style)
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Unusual Deposit Detection")
    st.markdown("Catch unusual deposit patterns early so teams can investigate quickly.")
    render_tab_guide(
        "Scan single transactions or upload a batch file.",
        "Anomaly score, reason hints, risk trend, and downloadable flagged results.",
    )
    try:
        default_anomaly_artifacts, default_anomaly_df = load_bank_anomaly_defaults()
        if "anomaly_artifacts" not in st.session_state:
            st.session_state["anomaly_artifacts"] = default_anomaly_artifacts
            st.session_state["anomaly_raw_df"] = default_anomaly_df
            st.session_state["anomaly_history"] = []

        anomaly_artifacts = st.session_state["anomaly_artifacts"]
        anomaly_df = st.session_state["anomaly_raw_df"]

        st.subheader("Model Setup")
        m1, m2, m3 = st.columns(3)
        m1.metric("Training rows", f"{len(anomaly_df):,}")
        m2.metric("Expected anomaly rate", f"{anomaly_artifacts.contamination:.1%}")
        m3.metric("Alert sensitivity", f"{anomaly_artifacts.reconstruction_threshold:.3f}")

        with st.expander("Optional: Retrain detector on your own file"):
            uploaded_anomaly = st.file_uploader(
                "Upload CSV with columns: amount, hour, day_of_week, frequency",
                type=["csv"],
                key="anomaly_upload",
            )
            contam = st.slider("IsolationForest contamination (%)", 1, 10, 3, step=1, key="anomaly_contam")
            if uploaded_anomaly is not None and st.button("Retrain anomaly models", key="anomaly_retrain_btn"):
                up_df = pd.read_csv(uploaded_anomaly)
                required = {"amount", "hour", "day_of_week", "frequency"}
                missing = sorted(required - set(up_df.columns))
                if missing:
                    st.error(f"Uploaded file missing required columns: {missing}")
                else:
                    st.session_state["anomaly_raw_df"] = up_df
                    st.session_state["anomaly_artifacts"] = train_anomaly_engine(
                        up_df,
                        contamination=contam / 100.0,
                        random_state=42,
                    )
                    anomaly_artifacts = st.session_state["anomaly_artifacts"]
                    anomaly_df = up_df
                    st.success("Anomaly engine retrained successfully.")

        st.markdown("---")
        st.subheader("Live Deposit Analysis")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            amount = st.number_input("Deposit Amount ($)", min_value=1.0, value=500.0, step=50.0, key="anom_amount")
        with c2:
            hour = st.slider("Hour of Day", 0, 23, 10, key="anom_hour")
        with c3:
            day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2, key="anom_day")
        with c4:
            frequency = st.slider("24h Deposit Frequency", 1, 20, 1, key="anom_freq")

        if st.button("Execute AI Scan", key="anomaly_scan_btn"):
            result = analyze_transaction(
                artifacts=anomaly_artifacts,
                amount=amount,
                hour=hour,
                day_of_week=day_of_week,
                frequency=frequency,
            )
            status = "Flagged" if result["is_anomaly"] else "Normal"
            row = {
                "amount": amount,
                "hour": hour,
                "day_of_week": day_of_week,
                "frequency": frequency,
                "status": status,
                "score": float(result["score"]),
                "reasons": result["reasons"],
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state["anomaly_history"].append(row)

        history = pd.DataFrame(st.session_state.get("anomaly_history", []))
        if not history.empty:
            latest = history.iloc[-1]
            rr1, rr2 = st.columns(2)
            with rr1:
                if latest["status"] == "Flagged":
                    st.error(f"⚠️ SUSPICIOUS ACTIVITY | Score: {latest['score']:.3f}")
                else:
                    st.success(f"✅ CLEARED | Score: {latest['score']:.3f}")
                st.caption(f"Reasons: {latest['reasons']}")
            with rr2:
                st.metric("Current risk score", f"{latest['score']:.3f}")
                st.metric("Risk level", latest["status"])

            st.line_chart(history.set_index("timestamp")[["score"]])
            st.dataframe(
                history[["timestamp", "amount", "hour", "day_of_week", "frequency", "status", "score", "reasons"]].sort_values("timestamp", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.subheader("Batch Scoring")
        st.caption("Upload transaction batches with columns: amount, hour, day_of_week, frequency")
        batch_file = st.file_uploader("Upload anomaly batch CSV", type=["csv"], key="anomaly_batch_upload")
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            required = {"amount", "hour", "day_of_week", "frequency"}
            missing = sorted(required - set(batch_df.columns))
            if missing:
                st.error(f"Batch file missing required columns: {missing}")
            else:
                scored_batch = score_transactions(batch_df, anomaly_artifacts)
                flagged_share = float(scored_batch["is_anomaly"].mean())
                st.metric("Flagged share", f"{flagged_share:.1%}")
                st.dataframe(scored_batch.head(50), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download scored anomaly batch",
                    data=scored_batch.to_csv(index=False).encode("utf-8"),
                    file_name="scored_anomaly_batch.csv",
                    mime="text/csv",
                    key="download_scored_anomaly_batch",
                )

    except Exception as e:
        st.error(f"Anomaly module could not load: {e}")

