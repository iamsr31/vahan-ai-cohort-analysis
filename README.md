# Vahan — Lead-Source Cohort Performance Case Study

Product Analytics Internship case study: cohort analysis, SQL aggregation,
and a machine-learning model predicting First-Trip (FT) conversion from
Vahan's lead-level calling data.

## Final Deliverable

**`outputs/Vahan_Case_Study_Report.pdf`** — the 7-page case-study report
(Executive Summary, Cohort Analysis, SQL Analysis, ML Approach, Model
Results, Recommendations & Limitations).

## Project Structure

```
project/
├── data/
│   └── raw_leads.csv                  # raw lead-level data (18,197 rows)
├── sql/
│   ├── 01_schema.sql                  # raw_leads table schema
│   └── 02_cohort_analysis.sql         # cohort aggregation query (the SQL deliverable)
├── python/
│   ├── 01_data_prep_and_sql.py        # loads data into SQLite, runs the SQL query
│   ├── 02_ml_model.py                 # trains + evaluates the logistic regression model
│   ├── 03_cohort_chart.py             # builds the cohort performance chart
│   └── 04_generate_report.py          # assembles the final PDF report
├── outputs/
│   ├── Vahan_Case_Study_Report.pdf    # ⭐ final report
│   ├── cohort_aggregated_output.csv   # SQL query output (one row per cohort)
│   ├── feature_importance.csv         # logistic regression coefficients / odds ratios
│   ├── model_metrics.json             # confusion matrix, precision/recall/F1/AUC
│   ├── charts/                        # all PNG charts used in the report
│   └── model/
│       └── logistic_regression_pipeline.pkl   # trained sklearn pipeline (importable)
└── requirements.txt
```

## How to Reproduce

```bash
pip install -r requirements.txt

cd python
python 01_data_prep_and_sql.py   # -> outputs/cohort_aggregated_output.csv
python 02_ml_model.py            # -> outputs/model_metrics.json, model/*.pkl, charts/*
python 03_cohort_chart.py        # -> outputs/charts/cohort_ft_rate.png
python 04_generate_report.py     # -> outputs/Vahan_Case_Study_Report.pdf
```

## Answers to the Three Case-Study Questions

1. **Which 3 cohorts are best, and on what metric?**
   Ranked by **FT conversion rate on uploaded leads** (FT_after_upload / Uploaded_Leads):
   1. *Single Referral > 7 days – 24th Jul* (0.933%)
   2. *Khanna – 2W 26th Jul* (0.906%)
   3. *PreOb-Ob Fees Paid 29th Jul (set 1)* (0.472%)

   See report Page 2 for why this metric (not raw FT count, not FT-per-attempted-lead)
   is the fairest way to compare cohorts.

2. **SQL query + aggregated table** — `sql/02_cohort_analysis.sql`, output in
   `outputs/cohort_aggregated_output.csv` and report Page 3.

3. **ML model of FT drivers + confusion matrix** — `python/02_ml_model.py`
   (logistic regression, class-weighted), results in report Pages 4–6 and
   `outputs/model_metrics.json`.

## Key Modeling Notes

- **Target:** `FT_after_upload` (0.297% positive rate — a rare event).
- **Leakage guard:** `OB_after_upload` / `OB_after_first_attempt` /
  `FT_after_first_attempt` and derived `*_pct` columns were excluded from
  features — onboarding is a near-perfect proxy for FT and would leak the
  target.
- **Imbalance handling:** `class_weight='balanced'` instead of oversampling,
  evaluated with precision/recall/F1/ROC-AUC/PR-AUC rather than accuracy.

Full reasoning, limitations, and recommendations are in the PDF report.
