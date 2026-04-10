# dashboard/app.py
# FinSight: Financial Forecasting System — Streamlit Dashboard

import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def p(path):
    return os.path.join(BASE_DIR, path)

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
    return pd.read_csv(p('data/processed/monthly_loan_volume.csv'), parse_dates=['month_start'])

@st.cache_data
def load_grade_data():
    return pd.read_csv(p('data/processed/grade_monthly_demand.csv'), parse_dates=['month_start'])

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 FinSight: Financial Forecasting System")
st.markdown("*End-to-end ML forecasting on 1.3M Lending Club loan records*")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "⚠️ Module 1: Loan Default Risk",
    "👤 Module 2: Borrower Churn",
    "📈 Module 3: Loan Volume Forecast",
    "📊 Module 4: Credit Demand by Grade",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", "1.34M loans")
    col2.metric("Features", "96 columns")
    col3.metric("Modules", "4")
    col4.metric("Models compared", "3 per module")

    st.markdown("---")

    st.subheader("Module Results Summary")
    summary = pd.DataFrame({
        'Module':        ['Loan Default Risk', 'Borrower Churn', 'Loan Volume Forecast', 'Credit Demand (Grade A)'],
        'Task':          ['Binary Classification', 'Binary Classification', 'Time Series Regression', 'Time Series Regression'],
        'Best Model':    ['XGBoost', 'XGBoost', 'Linear Regression', 'Linear Regression'],
        'Best Metric':   ['AUC 0.717', 'AUC 0.814', 'MAPE 1.92%', 'MAPE 2.72%'],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Tech Stack")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("**Data**\n- Pandas\n- NumPy")
    col2.markdown("**ML Models**\n- Scikit-learn\n- XGBoost")
    col3.markdown("**Forecasting**\n- Prophet\n- Lag features")
    col4.markdown("**Explainability**\n- Feature importance\n- Threshold tuning")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODULE 1: LOAN DEFAULT RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("⚠️ Module 1: Loan Default Risk Prediction")
    st.markdown("Predict whether a borrower will default on their loan.")

    try:
        m1 = load_module1_models()

        st.subheader("Model Performance")
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
# TAB 3 — MODULE 2: BORROWER CHURN
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("👤 Module 2: Borrower Churn Prediction")
    st.markdown("Predict whether a borrower will return for a second loan after repayment.")

    try:
        m2 = load_module2_models()

        st.subheader("Model Performance")
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
# TAB 4 — MODULE 3: LOAN VOLUME FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📈 Module 3: Loan Volume Forecast")
    st.markdown("Forecast monthly total loan volume funded by Lending Club.")

    st.subheader("Model Performance (Test period: 2015)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression MAPE", "1.92%", delta="Champion ✓")
    col2.metric("XGBoost MAPE", "10.30%")
    col3.metric("Prophet MAPE", "29.87%")

    st.markdown("---")

    try:
        monthly = load_monthly_data()

        st.subheader("Historical Loan Volume")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(monthly['month_start'], monthly['funded_amnt_m'],
                color='steelblue', linewidth=1.5)
        ax.set_title('Monthly Loan Volume Funded (USD Millions)')
        ax.set_ylabel('Funded Amount (USD M)')
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"Could not load monthly data: {e}")

    st.markdown("---")
    st.subheader("3-Month Forecast (Prophet)")
    forecast_data = pd.DataFrame({
        'Month':            ['2016-01', '2016-02', '2016-03'],
        'Forecast (USD M)': [613.50, 451.70, 431.82],
        'Lower Bound':      [595.98, 435.95, 414.94],
        'Upper Bound':      [631.59, 467.59, 447.87],
    })
    st.dataframe(forecast_data, use_container_width=True, hide_index=True)

    st.info("💡 **Key Insight:** Linear Regression outperformed XGBoost and Prophet during the stable growth period (2008–2015), demonstrating that model complexity should be justified by data complexity.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODULE 4: CREDIT DEMAND BY GRADE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("📊 Module 4: Credit Demand Forecast by Grade")
    st.markdown("Forecast monthly loan demand segmented by credit grade (A–E).")

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

    except Exception as e:
        st.warning(f"Could not load grade data: {e}")

    st.markdown("---")
    st.info("💡 **Key Insight:** Linear Regression was the champion model for Grade A, B, C, and E. XGBoost only won on Grade D — the highest-risk segment — where demand volatility justified a more complex model.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "FinSight Financial Forecasting System | Built with Lending Club Data | "
    "<a href='https://github.com/Adrian0117/finsight-forecasting'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
