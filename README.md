# EEF606 — Lab 1: Mobile Money Fraud Detection
**MSc Telecommunications and Network Engineering | University of Buea**
*| Academic Year 2025/2026*

---

## What This Project Does
Applies five machine learning paradigms to detect mobile money fraud on the PaySim dataset (6.3M transactions, 0.13% fraud rate). The goal is not just to build a detector — it is to understand *why* different problem framings require different models.

| Paradigm | Question Asked | Best Model |
|---|---|---|
| Classification | Is this transaction fraud? | Random Forest (F1=0.84) |
| Regression | What is the fraud probability? | Ridge Regression |
| Ranking | Which transactions to investigate first? | Random Forest (NDCG@100=0.84) |
| Anomaly Detection | Does this deviate from normal? | Local Outlier Factor |
| Clustering | What behavioural groups exist? | K-Means (K=3) |
| Ensemble | Can combining models beat any single model? | Soft Voting (F1=0.84) |

---

## Project Structure
```
lab1/
├── PS_20174392719_1491204439457_log.csv  ← Kaggle dataset (download separately)
├── lab1_00_eda_analysis.ipynb            ← Run FIRST — generates paysim_features.csv
├── lab1_01_classification.ipynb
├── lab1_02_regression.ipynb
├── lab1_03_ranking.ipynb
├── lab1_04_anomaly_detection.ipynb
├── lab1_05_clustering.ipynb
├── lab1_06_ensemble.ipynb
├── lab1_app.py                           ← Flask API — trains all models, serves predictions
├── lab1_dashboard_v2.html                ← Open in browser after running lab1_app.py
└── lab1_report_formatted.docx           ← Final submitted report
```

---

## Quick Start

### 1. Install dependencies (first time only)
```bash
pip install flask flask-cors imbalanced-learn
```

### 2. Download the dataset
[PaySim on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
Place `PS_20174392719_1491204439457_log.csv` in the project folder.

### 3. Run the notebooks (for report figures)
Open Anaconda, run notebooks in order starting from `lab1_00_eda_analysis.ipynb`.

### 4. Run the live dashboard
```bash
python lab1_app.py
```
Then open `lab1_dashboard_v2.html` in Chrome or Edge.
The API trains all 8 models on startup (~5 min on 1M rows) then serves live predictions.

---

## Models in the Dashboard
| Model | Type | Speed |
|---|---|---|
| Logistic Regression | Linear | Fast |
| Decision Tree | Tree | Fast |
| Random Forest | Ensemble (Bagging) | Fast |
| Hist Gradient Boosting | Ensemble (Boosting) | Fast |
| K-Nearest Neighbours | Instance-based | Moderate |
| Naive Bayes | Probabilistic | Fast |
| Linear SVC | Kernel (linear) | Fast |
| Isolation Forest | Anomaly Detection | Fast |

Switch between models live in the dashboard. The **Compare All** button runs every model on the same transaction simultaneously and shows an AUC-weighted ensemble verdict.

---

## Dataset Configuration
In `lab1_app.py`, line ~55:
```python
SAMPLE_ROWS = 1_000_000   # change to None for all 6.3M rows (~30 min training)
```

---

## Key Findings
- Fraud is **exclusively** in TRANSFER and CASH_OUT transactions
- The single strongest fraud signal: origin account drained to exactly zero
- Accuracy is useless here (99.87% baseline by predicting nothing is fraud)
- Use **F1, PR-AUC, and Recall** — not accuracy or ROC-AUC alone
- Unsupervised methods confirm fraud does not form its own behavioural cluster — fraudsters deliberately mimic legitimate behaviour

---

## Dependencies
- Python 3.8+
- Anaconda (includes: numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, joblib)
- flask, flask-cors, imbalanced-learn (install separately)

---

## Authors
Tanjong Pride · Jerry Ebai · Bongjoh Thierry
*FE25P040 · FE25P038 · FE25P043*