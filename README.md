# Credit-Card-Default-Prediction
## Machine Learning-Based Classification with Feature Engineering
---
## Project Overview

This study investigates whether feature engineering 
can improve the ability of interpretable machine 
learning models to detect credit card defaults. 
Three classifiers — Decision Tree, Random Forest, 
and Logistic Regression — were trained and compared 
under two conditions: using the original raw 
features, and using a set of 14 behavioral features 
derived through domain-aware feature engineering.

The primary evaluation metric is **recall on the 
default class**, reflecting the real-world priority 
of minimizing undetected default risk over 
maximizing overall accuracy.

---

## Dataset

- **Source:** UCI Machine Learning Repository  
- **URL:** https://archive.ics.uci.edu/dataset/350  
- **Size:** 30,000 credit card clients (Taiwan, 
  April–September 2005)  
- **Target variable:** `Is_Default` 
  (1 = default, 0 = no default)  
- **Class distribution:** 77.9% non-default / 
  22.1% default

---

## Methodology

### Data Preprocessing
- Undocumented categorical values in `EDUCATION` 
  and `MARRIAGE` remapped to *others* category
- 99th percentile winsorization applied to 
  `LIMIT_BAL` and all `PAY_AMT` columns
- Negative `BILL_AMT` values retained as valid 
  overpayment balances
- No missing values found — no imputation required

### Feature Engineering (12 derived features)

| Category | Features |
|---|---|
| Aggregate billing & payment | `avg_bill_amt`, `avg_pay_amt`, `payment_ratio`, `payment_consistency` |
| Payment delay metrics | `avg_payment_delay`, `max_delay`, `n_months_delayed`, `payment_delay_trend` |
| Financial gap & debt trend | `bill_pay_gap`, `bill_trend` |
| Credit utilization | `credit_utilization`, `limit_per_age` |
| Demographic grouping | `age_group_code`, `limit_category_code` |

### Models
| Model | Key Parameters |
|---|---|
| Decision Tree | `max_depth=5`, `class_weight='balanced'` |
| Random Forest | `n_estimators=100`, `class_weight='balanced'` |
| Logistic Regression | `max_iter=1000`, `class_weight='balanced'` |

---

## Results

### Approach 1 — Raw Feature Set (Baseline)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 0.77 | 0.48 | 0.55 | 0.52 |
| Random Forest | 0.81 | 0.64 | 0.35 | 0.45 |
| Logistic Regression | 0.67 | 0.36 | 0.64 | 0.46 |

### Approach 2 — Engineered Feature Set

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 0.74 | 0.44 | 0.64 | 0.52 |
| Random Forest | 0.81 | 0.65 | 0.34 | 0.44 |
| Logistic Regression | 0.74 | 0.44 | 0.60 | 0.50 |

### Key Findings
- **Decision Tree** achieved the highest recall 
  on the default class after feature engineering 
  (0.64), making it the most suitable model for 
  the study's primary objective
- **Random Forest** achieved the highest overall 
  accuracy (0.81) in both approaches but 
  consistently missed the most default cases
- **Feature engineering** improved AUC for 
  Logistic Regression from 0.71 to 0.75, bringing 
  all three models to a similar level (0.75–0.76)
- **9 out of 10** top features in Random Forest's 
  importance ranking were engineered variables


  Install dependencies:

```bash
pip install pandas numpy scikit-learn 
            matplotlib seaborn jupyter
```

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/
credit-default-prediction
cd credit-default-prediction
```

2. Download the dataset from UCI and place it 
   in the project root directory

3. Open and run the notebook:
```bash
jupyter notebook Analiz_updated.ipynb
```

Run all cells in order — preprocessing, EDA, 
feature engineering, model training, and 
evaluation are all contained within the single 
notebook file.
