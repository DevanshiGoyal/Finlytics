# FinSight — Model Comparison Report

> Full technical record of model selection, evaluation, and design decisions across all four forecasting modules.

---

## Module 1 — Loan Default Risk

### Task
Binary classification: predict whether a borrower will default (`loan_status = Charged Off`).

### Dataset
- Total loans after filtering: 1,344,936
- Class distribution: 80.04% Fully Paid / 19.96% Charged Off
- Class imbalance strategy: SMOTE applied to training set only (80/20 → 50/50)

### Model Comparison (Threshold = 0.5, fair comparison)

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.6597 | 0.30 | 0.53 | 0.38 |
| Random Forest | 0.6831 | 0.37 | 0.28 | 0.31 |
| **XGBoost** | **0.7168** | **0.54** | **0.10** | **0.17** |

### Threshold Tuning — XGBoost Only

After selecting XGBoost as the champion model based on AUC-ROC, the decision threshold was tuned separately to optimise recall for the Charged Off class.

| Threshold | Recall | Precision | F1 |
|---|---|---|---|
| 0.50 | 0.10 | 0.54 | 0.17 |
| 0.40 | 0.25 | 0.46 | 0.32 |
| **0.30** | **0.46** | **0.38** | **0.42** ← selected |
| 0.20 | 0.73 | 0.30 | 0.43 |

**Selected threshold: 0.30**
Threshold 0.30 achieved the best F1 score for the Charged Off class. Threshold 0.20 had higher recall (0.73) but precision dropped to 0.30 — meaning 7 out of every 10 flagged borrowers were actually good customers, generating too many false positives for a practical risk system.

### Final XGBoost Performance (Threshold = 0.30)

| Metric | Value |
|---|---|
| AUC-ROC | 0.7168 |
| Charged Off Recall | 0.46 |
| Charged Off Precision | 0.38 |
| Charged Off F1 | 0.42 |
| Accuracy | 0.74 |

### Feature Importance (Top 5)

| Rank | Feature | Business Interpretation |
|---|---|---|
| 1 | `inq_last_6mths` | More recent credit inquiries signal financial stress |
| 2 | `pub_rec` | Public derogatory records are a strong default predictor |
| 3 | `delinq_2yrs` | Past delinquencies predict future default behaviour |
| 4 | `emp_length` | Shorter employment history = less income stability |
| 5 | `home_ownership` | Homeowners have lower default rates than renters |

### Known Limitations
- AUC of 0.717 reflects the natural ceiling of this dataset. Key predictive signals such as actual bank transaction history and spending behaviour are not available in Lending Club's public data.
- `issue_year` was excluded from features to prevent temporal leakage.

---

## Module 2 — Borrower Churn Prediction

### Task
Binary classification: predict whether a borrower will NOT return for a second loan within 12 months of repayment.

### Label Engineering
`member_id` was not available in the processed dataset (dropped during cleaning due to high null rate). Churn was instead proxied by early repayment behaviour:
- Repaid in less than 85% of the loan term → `churn_flag = 0` (retained — strong financial health, likely to return)
- Repaid at 85% or more of the loan term → `churn_flag = 1` (churned)

### Dataset
- Total fully paid loans: 1,076,448
- Class distribution: 71.47% Retained / 28.53% Churned
- Class imbalance strategy: `scale_pos_weight = 2.51` in XGBoost; `class_weight='balanced'` in Logistic Regression and Random Forest

SMOTE was not used here because the 71/29 split is mild enough to be handled with class weighting alone, avoiding the computational cost of oversampling on 1M+ records.

### Model Comparison (Threshold = 0.5)

| Model | AUC-ROC | Churned Recall | Churned Precision | Churned F1 |
|---|---|---|---|---|
| Logistic Regression | 0.7823 | 0.76 | 0.49 | 0.59 |
| Random Forest | 0.7906 | 0.88 | 0.46 | 0.60 |
| **XGBoost** | **0.8139** | **0.92** | **0.47** | **0.62** |

### Champion Model: XGBoost
XGBoost achieved the highest AUC-ROC (0.8139) and the highest Churned Recall (0.92) — meaning 92 out of every 100 true churners were correctly identified. No threshold tuning was required as performance at 0.5 was already strong.

### Feature Importance (Top 5)

| Rank | Feature | Business Interpretation |
|---|---|---|
| 1 | `issue_year` | Borrower behaviour changed significantly across platform years |
| 2 | `issue_month` | Seasonal patterns in borrowing and repayment behaviour |
| 3 | `grade` | Lower-grade borrowers are less likely to return |
| 4 | `loan_amnt` | Larger loans correlate with different return behaviour |
| 5 | `int_rate` | Higher interest rates may discourage repeat borrowing |

### Known Limitations
- `issue_year` dominated feature importance (~48%), suggesting the model captures temporal platform trends rather than individual borrower characteristics. In production, time-based features would be replaced with borrower behavioural signals across loan cycles.
- The churn proxy (early repayment) is an approximation. True churn would require tracking individual borrowers across multiple loans.

---

## Module 3 — Loan Volume Forecast

### Task
Time series regression: forecast total monthly funded loan amount (USD millions).

### Data Preparation
- 1,344,936 transaction records aggregated to 139 monthly observations (2007–2018)
- Lag features engineered: lag 1, 2, 3, 6, 12 months; rolling averages 3, 6, 12 months; month-over-month growth rate
- After lag creation: 127 usable months

### Train / Test Split
Time series requires chronological splitting — random splitting would leak future information into training.

| Set | Period | Size |
|---|---|---|
| Training | 2008-06 → 2014-12 | 115 months |
| Test | 2015-01 → 2015-12 | 12 months |

**Why 2015 as the test period:**
The 2016–2018 period was severely impacted by Lending Club's executive fraud scandal, causing platform-wide volume collapse. Using this as a test period would evaluate models on an unpredictable structural disruption rather than their genuine forecasting ability. 2015 represents the most stable portion of the growth phase and is the most appropriate period for evaluating model accuracy.

### Model Comparison

| Model | RMSE (USD M) | MAPE |
|---|---|---|
| **Linear Regression** ✓ | **10.42** | **1.92%** |
| XGBoost | 73.07 | 10.30% |
| Prophet | 207.29 | 29.87% |

### Champion Model: Linear Regression
Linear Regression outperformed both XGBoost and Prophet. The underlying trend from 2008 to 2015 was approximately linear — a consistent growth trajectory with seasonal fluctuations. In this context, model complexity did not add value and both XGBoost and Prophet overfit to noise in the training data.

This finding is consistent across Modules 3 and 4: **model complexity should be justified by data complexity.**

### Prophet Analysis
Prophet successfully decomposed the time series into a trend component (exponential growth from 2008–2015) and a yearly seasonality component. However, the seasonality showed extreme volatility (±2000%) due to limited training data — only 78 monthly observations before the test period. In production, `seasonality_prior_scale` would be reduced to prevent overfitting.

### 3-Month Forecast (Prophet, post-test)

| Month | Forecast (USD M) | Lower Bound | Upper Bound |
|---|---|---|---|
| 2016-01 | 613.50 | 595.98 | 631.59 |
| 2016-02 | 451.70 | 435.95 | 467.59 |
| 2016-03 | 431.82 | 414.94 | 447.87 |

Prophet predicted continued growth into 2016. The actual volume collapsed due to the Lending Club scandal — an external event no historical model could anticipate.

### Known Limitations
- Only 139 months of data limits the robustness of seasonal decomposition.
- Models cannot anticipate structural business disruptions. In production, external signals such as investor confidence indices or news sentiment scores would be incorporated.

---

## Module 4 — Credit Demand Forecast by Grade

### Task
Multi-output time series regression: forecast monthly loan demand separately for each credit grade (A, B, C, D, E).

### Data Preparation
- Same source dataset aggregated by `grade` and `month_start`
- Separate lag feature sets created per grade
- Same train/test split logic as Module 3: train on pre-2015 data, test on 2015

### Model Comparison — MAPE (%) by Grade

| Model | Grade A | Grade B | Grade C | Grade D | Grade E |
|---|---|---|---|---|---|
| **Linear Regression** | **2.72** ✓ | **4.97** ✓ | **5.03** ✓ | 13.10 | **8.73** ✓ |
| Random Forest | 33.42 | 26.56 | 20.96 | 14.35 | 20.99 |
| **XGBoost** | 18.03 | 12.33 | 12.64 | **8.55** ✓ | 19.45 |

### Champion Models by Grade

| Grade | Champion | MAPE | Reason |
|---|---|---|---|
| A | Linear Regression | 2.72% | Stable, predictable low-risk demand |
| B | Linear Regression | 4.97% | Consistent growth trend |
| C | Linear Regression | 5.03% | Largest grade by volume, stable trend |
| D | XGBoost | 8.55% | Higher-risk segment with more volatile demand |
| E | Linear Regression | 8.73% | Small volume, but trend is linear |

### Key Finding
XGBoost was the champion only for Grade D — the highest-risk borrower segment. Grade D demand showed more non-linear volatility, where XGBoost's ability to model complex interactions provided an advantage. For all other grades, the demand signal was stable enough that a linear model sufficed.

This reinforces the cross-module finding: **the right model depends on the complexity of the signal, not on the sophistication of the algorithm.**

### Known Limitations
- Per-grade models are trained independently, ignoring potential cross-grade correlations (e.g. a shift in Grade C demand may predict a shift in Grade D demand).
- Small sample sizes for Grades D and E in early years reduce the reliability of lag features.

---

## Cross-Module Summary

| Module | Task | Champion | Key Metric |
|---|---|---|---|
| 1 — Loan Default Risk | Binary classification | XGBoost | AUC 0.7168 |
| 2 — Borrower Churn | Binary classification | XGBoost | AUC 0.8139 |
| 3 — Loan Volume Forecast | Time series regression | Linear Regression | MAPE 1.92% |
| 4 — Credit Demand (Grade A) | Time series regression | Linear Regression | MAPE 2.72% |

### Consistent Findings Across Modules

**1. XGBoost dominates classification tasks.**
For Modules 1 and 2, XGBoost outperformed both Logistic Regression and Random Forest on AUC-ROC. The non-linear interactions between financial features (income, debt ratio, credit history) benefit from gradient boosting.

**2. Linear models win on stable time series.**
For Modules 3 and 4, Linear Regression outperformed XGBoost and Prophet during the 2008–2015 stable growth period. When the underlying signal is approximately linear, adding complexity introduces noise rather than signal.

**3. Threshold tuning is a business decision, not a technical one.**
In Module 1, the default threshold of 0.5 produced a Charged Off Recall of only 0.10 — practically useless for a risk system. Lowering to 0.30 improved recall to 0.46. The threshold was chosen based on F1, which balances the cost of missing a defaulter against the cost of falsely flagging a good borrower.

**4. Feature engineering matters as much as model selection.**
Module 2's churn label did not exist in the raw data — it was constructed from repayment timing. Module 3's lag features transformed a regression problem into a time series problem. In both cases, the quality of the input features determined the quality of the output more than the choice of algorithm.

---

## Future Improvements

| Area | Improvement |
|---|---|
| Module 1 | Incorporate FICO score trend over time as a dynamic feature |
| Module 2 | Replace early repayment proxy with actual repeat loan tracking if member_id becomes available |
| Module 3 | Add external macroeconomic features (Fed rate, unemployment rate) as exogenous regressors in Prophet |
| Module 4 | Build a multi-output model that forecasts all grades simultaneously, capturing cross-grade demand correlations |
| All modules | Implement hyperparameter tuning with Optuna for XGBoost across all classification modules |
| Dashboard | Add model confidence intervals to prediction outputs in Modules 1 and 2 |

---

*Last updated: after completion of all four modules and Streamlit dashboard.*
*Contact: jiarongadrian@gmail.com*
