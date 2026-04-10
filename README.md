# FinSight: Financial Forecasting System

> End-to-end ML forecasting system built on 1.34M Lending Club loan records — covering loan default risk, borrower churn, loan volume forecasting, and credit demand segmentation by grade.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square)
![Prophet](https://img.shields.io/badge/Forecasting-Prophet-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-grey?style=flat-square)

---

## What This Project Does

Most ML projects stop at training a model. FinSight goes further — it applies four different forecasting techniques to the same financial dataset, compares models rigorously, tunes decision thresholds based on business context, and serves predictions through a live interactive dashboard.

The entire system is built on a single public dataset: [Lending Club Accepted Loans (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

---

## Live Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard has 5 tabs — one project overview and one per module — with live prediction inputs for Modules 1 and 2, interactive charts for Modules 3 and 4, and model comparison tables throughout.

---

## Results

### Module 1 — Loan Default Risk

> "Which borrowers are likely to default?"

Binary classification on `loan_status` → Fully Paid (0) vs Charged Off (1).

| Model               | AUC-ROC    | Charged Off Recall | Charged Off F1 |
| ------------------- | ---------- | ------------------ | -------------- |
| Logistic Regression | 0.6597     | 0.53               | 0.38           |
| Random Forest       | 0.6831     | 0.28               | 0.31           |
| **XGBoost** ✓       | **0.7168** | **0.46**           | **0.42**       |

- Dataset: 1,344,936 loans / 20% default rate
- Class imbalance handled with SMOTE (80/20 → 50/50 after resampling)
- Decision threshold tuned to 0.30 — best F1 for Charged Off class
- Top predictors: `inq_last_6mths`, `pub_rec`, `delinq_2yrs`, `emp_length`, `home_ownership`

### Module 2 — Borrower Churn Prediction

> "Which borrowers won't return after repaying their loan?"

Churn label engineered from repayment behaviour — borrowers who repaid in less than 85% of their loan term were classified as retained; others as churned.

| Model               | AUC-ROC    | Churned Recall | Churned F1 |
| ------------------- | ---------- | -------------- | ---------- |
| Logistic Regression | 0.7823     | 0.76           | 0.59       |
| Random Forest       | 0.7906     | 0.88           | 0.60       |
| **XGBoost** ✓       | **0.8139** | **0.92**       | **0.62**   |

- Dataset: 1,076,448 fully paid loans / 28.5% churn rate
- Class imbalance handled with `scale_pos_weight`
- Top predictors: `issue_year`, `issue_month`, `grade`, `loan_amnt`, `int_rate`

### Module 3 — Loan Volume Forecast

> "How much will be funded next quarter?"

Transaction-level data aggregated into a monthly time series (139 months, 2007–2018). ML models trained with lag features; Prophet trained directly on the time series.

Test period: 2015 (stable growth phase, avoiding the 2016 platform disruption).

| Model                   | RMSE (USD M) | MAPE      |
| ----------------------- | ------------ | --------- |
| **Linear Regression** ✓ | **10.42**    | **1.92%** |
| XGBoost                 | 73.07        | 10.30%    |
| Prophet                 | 207.29       | 29.87%    |

- Linear Regression won because the 2008–2015 trend was approximately linear
- Prophet decomposed trend + yearly seasonality components
- 3-month forecast produced with confidence intervals

### Module 4 — Credit Demand Forecast by Grade

> "What is demand per loan grade next month?"

Separate model trained per grade (A, B, C, D, E) with lag features and rolling averages.

**MAPE (%) by Model and Grade — lower is better:**

| Model                   | Grade A  | Grade B  | Grade C  | Grade D    | Grade E  |
| ----------------------- | -------- | -------- | -------- | ---------- | -------- |
| **Linear Regression** ✓ | **2.72** | **4.97** | **5.03** | 13.10      | **8.73** |
| Random Forest           | 33.42    | 26.56    | 20.96    | 14.35      | 20.99    |
| XGBoost                 | 18.03    | 12.33    | 12.64    | **8.55** ✓ | 19.45    |

- Linear Regression was champion for 4 out of 5 grades
- XGBoost only won on Grade D — highest-risk segment with most volatile demand
- Consistent finding across Modules 3 and 4: simpler models win on stable trends

---

## Key Design Decisions

**Why one dataset for four modules?**
Using a single dataset forces deeper analytical thinking — asking different business questions about the same data, rather than finding a dataset that already fits the question.

**Why different imbalance strategies per module?**
Module 1 used SMOTE because the 80/20 imbalance was severe. Module 2 used `scale_pos_weight` because the 71/29 split was mild enough. Choosing the right tool for the right problem matters more than applying the same technique everywhere.

**Why tune the threshold only for the champion model?**
All three models were compared at threshold 0.5 to determine the champion fairly. The threshold was then tuned separately on XGBoost only — mixing threshold selection into model comparison would make the comparison unfair.

**Why use 2015 as the test period for Modules 3 and 4?**
The 2016–2018 period was affected by Lending Club's executive fraud scandal, causing an abnormal volume collapse. Using this as a test period would evaluate the model on a structural business disruption, not on its actual forecasting ability.

---

## Project Structure

```
finsight-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                              # accepted_2007_to_2018Q4.csv (not tracked)
│   └── processed/                        # cleaned CSVs (not tracked)
│       ├── loans_cleaned.csv
│       ├── monthly_loan_volume.csv
│       └── grade_monthly_demand.csv
├── notebooks/
│   ├── 01_loan_default_risk.ipynb
│   ├── 02_borrower_churn.ipynb
│   ├── 03_loan_volume_forecast.ipynb
│   └── 04_credit_demand_forecast.ipynb
├── src/
│   └── data_pipeline.py
├── models/                               # Saved .pkl and .json models (not tracked)
├── dashboard/
│   └── app.py
└── reports/
    ├── model_comparison.md
    ├── module1_roc_curve.png
    ├── module1_confusion_matrix.png
    ├── module1_shap.png
    ├── module2_roc_curve.png
    ├── module2_confusion_matrix.png
    ├── module2_feature_importance.png
    ├── module3_time_series.png
    ├── module3_forecast.png
    ├── module3_prophet_components.png
    ├── module4_demand_by_grade.png
    ├── module4_forecast_by_grade.png
    └── module4_mape_heatmap.png
```

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Adrian0117/finsight-forecasting.git
cd finsight-forecasting

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Go to https://www.kaggle.com/datasets/wordsforthewise/lending-club
# Download accepted_2007_to_2018Q4.csv and place in data/raw/

# 5. Run the data pipeline
python src/data_pipeline.py

# 6. Run notebooks in order
# Open notebooks/ in VS Code and run 01 → 02 → 03 → 04

# 7. Launch dashboard
streamlit run dashboard/app.py
```

---

## Tech Stack

| Category          | Tools                    |
| ----------------- | ------------------------ |
| Data processing   | Pandas, NumPy            |
| Machine learning  | Scikit-learn, XGBoost    |
| Imbalanced data   | imbalanced-learn (SMOTE) |
| Time series       | Facebook Prophet         |
| Visualisation     | Matplotlib, Seaborn      |
| Dashboard         | Streamlit                |
| Model persistence | Joblib                   |
| Environment       | Python 3.10, venv        |

---

## Dataset

**Source:** [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Download `accepted_2007_to_2018Q4.csv` and place it in `data/raw/`. The rejected loans CSV is not used — it lacks the loan outcome data required for all four modules.

After running `data_pipeline.py`:

- Raw: 2,260,701 rows × 151 columns
- Cleaned: 1,344,936 rows × 96 columns
- Dropped: 58 high-null columns, 374 rows with critical missing values

---

## Feedback & Discussion

This project was built as a portfolio piece to demonstrate end-to-end ML engineering across financial forecasting use cases.

I'm still learning and there's always room for improvement. If you spot anything that could be done better — whether it's modelling decisions, feature engineering, code quality, or business logic — I'd genuinely love to hear from you.

Feel free to open a [GitHub Issue](https://github.com/Adrian0117/finsight-forecasting/issues) or reach out directly:

📧 8acwolf8@gmail.com

---

## License

MIT License — free to use and adapt.
