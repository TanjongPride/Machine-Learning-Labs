# Lab 4  Telecom Customer Churn Prediction

A complete Machine Learning project that predicts whether a telecom customer will churn (leave) or stay, built with Python, trained on real customer data, and deployed as an interactive web application using Streamlit.

---

  
**Tool:** Python, VS Code, Anaconda

---

##  Project Structure

```
TELECOM_CHURN/
│
├── Data/
│   └── customer_churn_data.csv        ← Raw dataset
│
├── notebook/
│   └── 01_eda.ipynb                   ← Main Jupyter notebook (all code)
│
├── models/
│   ├── logistic_regression.pkl        ← Saved Logistic Regression model
│   ├── random_forest.pkl              ← Saved Random Forest model
│   └── xgboost.pkl                    ← Saved XGBoost model
│
├── reports/
│   └── figures/
│       ├── eda_overview.png           ← EDA charts
│       ├── model_comparison.png       ← Confusion matrices
│       ├── feature_importance.png     ← Feature importance charts
│       └── roc_curve.png              ← ROC curves
│
├── app.py                             ← Streamlit web application
└── README.md                          ← This file
```

---

## 🛠️ Environment Setup

### Requirements
- [Anaconda](https://www.anaconda.com/download) — Python distribution that includes most libraries
- [VS Code](https://code.visualstudio.com/) — Code editor
- VS Code Extensions: **Python**, **Jupyter** (installed through VS Code marketplace)

### Python Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Loading and manipulating data |
| `numpy` | Numerical computations |
| `matplotlib` | Creating charts and plots |
| `seaborn` | Beautiful statistical visualizations |
| `scikit-learn` | Machine learning models and evaluation |
| `xgboost` | XGBoost model |
| `imbalanced-learn` | SMOTE for fixing class imbalance |
| `joblib` | Saving and loading trained models |
| `streamlit` | Building the web application |

### Installing Libraries
Open **Anaconda Prompt** and run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter ipykernel
conda install streamlit -c conda-forge
```

---

##  Dataset Description

**File:** `customer_churn_data.csv`  
**Size:** 1,000 customers, 10 columns

| Column | Type | Description |
|---|---|---|
| CustomerID | Integer | Unique customer identifier (dropped during preprocessing) |
| Age | Integer | Customer age in years |
| Gender | Text | Male or Female |
| Tenure | Integer | Number of months as a customer |
| MonthlyCharges | Float | Monthly bill amount in dollars |
| ContractType | Text | Month-to-Month, One-Year, or Two-Year |
| InternetService | Text | DSL or Fiber Optic (had 297 missing values) |
| TotalCharges | Float | Total amount paid since joining |
| TechSupport | Text | Yes or No |
| Churn | Text | **Target variable** — Yes (churned) or No (stayed) |

---

##   Project Walkthrough

### Step 1 — Load and Explore the Data (EDA)

**What was done:**  
Loaded the dataset into Python and explored its structure to understand what we're working with.

**Key findings:**
- Dataset has 1,000 rows and 10 columns
- `InternetService` column had 297 missing values
- 88.3% of customers churned (Yes) vs 11.7% stayed (No) — severe class imbalance

**Code:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/customer_churn_data.csv")
df.head()                        # Preview first 5 rows
df.shape                         # Check dimensions
df.isnull().sum()                # Check missing values
df['Churn'].value_counts()       # Check churn distribution
```

**Key functions:**
- `pd.read_csv()` — reads a CSV file and converts it into a DataFrame (table)
- `df.head()` — shows the first 5 rows of the data
- `df.shape` — returns (rows, columns)
- `df.isnull().sum()` — counts missing values per column
- `value_counts()` — counts how many times each unique value appears

---

### Step 2 — Data Preprocessing

**What was done:**  
Cleaned and transformed the data so ML models can understand it. Models only work with numbers — not text.

**Actions taken:**
1. Dropped `CustomerID` — not useful for prediction
2. Filled 297 missing `InternetService` values with the most common value (mode)
3. Converted all text columns to numbers using Label Encoding

**Code:**
```python
from sklearn.preprocessing import LabelEncoder

df = df.drop(columns=['CustomerID'])
df['InternetService'] = df['InternetService'].fillna(df['InternetService'].mode()[0])

le = LabelEncoder()
text_columns = ['Gender', 'ContractType', 'InternetService', 'TechSupport', 'Churn']
for col in text_columns:
    df[col] = le.fit_transform(df[col])
```

**Key function explanations:**
- `df.drop(columns=[...])` — removes specified columns from the DataFrame
- `fillna(mode()[0])` — fills empty/missing cells with the most frequently occurring value
- `LabelEncoder()` — converts text categories into numbers (e.g. Male→1, Female→0)
- `fit_transform()` — learns the encoding and applies it at the same time

**Encoding result:**
- Gender: Male=1, Female=0
- Churn: Yes=1, No=0
- ContractType: Month-to-Month=0, One-Year=1, Two-Year=2
- InternetService: DSL=0, Fiber Optic=1
- TechSupport: No=0, Yes=1

---

### Step 3 — Fix Class Imbalance with SMOTE

**What we did:**  
Used SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.

**Why this matters:**  
With 88.3% churn vs 11.7% no-churn, a model could cheat by always predicting "Yes" and still be 88% accurate without learning anything useful. SMOTE creates artificial samples of the minority class to balance things out.

**Result after SMOTE:** 883 churned vs 883 stayed (perfectly balanced)

**Code:**
```python
from imblearn.over_sampling import SMOTE

X = df.drop(columns=['Churn'])   # Features (inputs)
y = df['Churn']                   # Target (what we predict)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

**Key function explanations:**
- `X` — all columns except Churn (the inputs to the model)
- `y` — only the Churn column (what we want the model to predict)
- `SMOTE()` — creates synthetic/artificial data points to balance classes
- `random_state=42` — sets a fixed seed so results are reproducible every time

---

### Step 4 — Train/Test Split

**What was done:**  
Split the data into 80% for training and 20% for testing.

 You can't test a model on the same data it learned from — that's like giving students the exam answers during class. We keep 20% hidden so we can fairly evaluate performance.

**Code:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)
```

**Result:** 1,412 rows for training, 354 rows for testing

---

### Step 5 — Train 3 Machine Learning Models

**What was done:**  
Trained three different models and compared their performance.

#### Model 1: Logistic Regression
A statistical model that estimates the probability of a binary outcome (churn or not churn). Simple, fast, and highly interpretable.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
```

#### Model 2: Random Forest
An ensemble of many decision trees. Each tree votes and the majority wins. More powerful than a single decision tree.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

#### Model 3: XGBoost
An advanced gradient boosting algorithm. Builds trees sequentially where each tree corrects the errors of the previous one. Very powerful for structured/tabular data.

```python
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
```

**Key function explanation:**
- `model.fit(X_train, y_train)` — trains the model by learning patterns from the training data

---

### Step 6 — Evaluate Models



**Metrics explained:**
- **Accuracy** — percentage of correct predictions overall
- **Precision** — of all customers predicted to churn, how many actually did?
- **Recall** — of all customers who actually churned, how many did we catch?
- **F1-Score** — balance between Precision and Recall
- **AUC (Area Under Curve)** — overall ability to distinguish churners from non-churners (1.0 = perfect)

**Results:**

| Model | Accuracy | AUC | F1-Score |
|---|---|---|---|
| Logistic Regression | 90.96% | 0.9641 | 0.908 |
| Random Forest | 100%* | 1.0* | 1.0* |
| XGBoost | 100%* | 1.0* | 1.0* |

* Random Forest and XGBoost scored 100% due to **overfitting** — they memorized the small dataset (1,000 rows) instead of learning general patterns. Logistic Regression at 90.96% is the most honest and realistic model.

---

### Step 7 — Feature Importance

**What we did:**  
Identified which factors most strongly influence whether a customer churns.

**Top churn drivers:**

| Rank | Feature | Importance | Business Meaning |
|---|---|---|---|
| 1 | ContractType | 27.6% | Month-to-Month customers leave most |
| 2 | Tenure | 25.4% | New customers churn faster |
| 3 | TechSupport | 24.3% | No tech support = higher churn |
| 4 | MonthlyCharges | 9.7% | Higher bills push customers away |
| 5 | TotalCharges | 8.1% | Related to tenure length |
| 6 | Gender | 2.0% | Barely matters |
| 7 | InternetService | 1.5% | Minor factor |
| 8 | Age | 1.3% | Least important |

---

### Step 8 — Save Models


Saved all three trained models to disk so the web app can load and use them without retraining.

```python
import joblib
import os

os.makedirs("../models", exist_ok=True)
joblib.dump(lr, "../models/logistic_regression.pkl")
joblib.dump(rf, "../models/random_forest.pkl")
joblib.dump(xgb, "../models/xgboost.pkl")
```

**Key function explanations:**
- `joblib.dump()` — saves a Python object (model) to a file
- `.pkl` — stands for "pickle", a format for saving Python objects
- `os.makedirs(..., exist_ok=True)` — creates a folder, won't crash if it already exists

---

### Step 9 — Web Application (Streamlit)

 
An interactive web app where you enter customer details and get an instant churn prediction with probability and business recommendation.

**Features:**
- Input fields for all customer attributes
- Choice of which model to use (Logistic Regression, Random Forest, XGBoost)
- Prediction result with color coding (red = churn, green = stay)
- Churn probability percentage
- Business recommendation

**File:** `app.py`

---

##  How to Launch the Project

### Step 1 — Open VS Code
Open VS Code and navigate to my TELECOM_CHURN folder.

### Step 2 — Run the Jupyter Notebook 
1. Open `notebook/01_eda.ipynb`
2. Click **"Run All"** at the top
3. we need to always  run all cells since  Python forgets everything when VS Code closes



### Step 3 — Launch the Web App
1. Open **Anaconda Prompt** 
2. Navigate to your project folder:

cd "C:\Users\LENOVO T570\Documents\2nd SEMESTER COURSES\EEF 606 DATA DRIVEN\LABS\Telecom_churn" in my case 

3. Run the app:

streamlit run app.py

4. Your browser will open automatically at `http://localhost:8501` tihs is the address as at the time I launched the web app
5. If browser doesn't open, we  manually go to `http://localhost:8501`

### Step 4 — Stop the Web App
Press **Ctrl + C** in the Anaconda Prompt to stop the app.

---

##  Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `NameError: name 'df' is not defined` | Kernel was restarted | Click "Run All" in the notebook |
| `FileNotFoundError: app.py` | Wrong folder in terminal | Navigate to TELECOM_CHURN folder first using `cd` |
| `pip not recognized` | Wrong terminal | Use Anaconda Prompt instead of PowerShell |
| `ModuleNotFoundError` | Library not installed | Run `conda install <library_name> -c conda-forge` |

---

##  Key Findings & Conclusions

1. **Contract type is the strongest predictor of churn** — customers on Month-to-Month contracts are far more likely to leave than those on annual contracts.

2. **New customers are at highest risk** — tenure is the second most important factor. Customers who have been with the company for a short time churn the most.

3. **Tech support matters** — customers without tech support show significantly higher churn rates.

4. **Overfitting warning** — with only 1,000 rows of data, Random Forest and XGBoost overfit and score 100%. In a real production scenario, more data (10,000+ rows) would give more reliable results.

5. **Best model for this dataset: Logistic Regression** — achieving 90.96% accuracy and AUC of 0.9641 without overfitting.

---



---


