"""
lab1_app.py  —  Mobile Money Fraud Detection · Live API
EEF606 · University of Buea

WHAT THIS FILE DOES:
════════════════════
Completely self-contained. Runs in order:
  1. Finds your PaySim CSV (Kaggle 6.3M or synthetic fallback)
  2. Cleans + engineers all features (same logic as notebook 00)
  3. Trains all 8 models (optimised for speed — see notes below)
  4. Starts the web server → lab1_dashboard_v2.html connects here

You do NOT need to run any notebook before running this.
Notebooks are only for generating report figures.

SPEED NOTES (why certain models were swapped):
══════════════════════════════════════════════
Original sklearn GradientBoosting  → swapped for HistGradientBoosting
  Reason: Original is sequential (slow). Hist version uses histogram
  binning + native parallelism — 10-20x faster, equal or better results.

Original SVM (RBF kernel)          → swapped for LinearSVC
  Reason: RBF SVM scales O(n²) — unusable on >100k rows.
  LinearSVC scales O(n) — fast on millions of rows.
  CalibratedClassifierCV wraps it to give probability output
  (needed for risk scores and PR-AUC).

HOW TO RUN:
═══════════
  1. Place this file + lab1_dashboard_v2.html in same folder as CSV
  2. Anaconda Prompt → cd into that folder
  3. pip install flask flask-cors imbalanced-learn   (first time only)
  4. python lab1_app.py
  5. Open lab1_dashboard_v2.html in Chrome/Edge

DATASET:
════════
  Kaggle file : PS_20174392719_1491204439457_log.csv  (6.3M rows)
  SAMPLE_ROWS : Controls how many rows to load.
                1_000_000 rows → ~8-12 min total training
                400_000   rows → ~3-5  min total training
                None           → all 6.3M rows (30-60 min, more accurate)
"""

import os, sys, time, warnings, threading
import numpy as np
import pandas as pd
import joblib

from flask      import Flask, request, jsonify
from flask_cors import CORS

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import RobustScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import LinearSVC
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import (RandomForestClassifier,
                                       HistGradientBoostingClassifier,
                                       IsolationForest)
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.naive_bayes      import GaussianNB
from sklearn.metrics          import (f1_score, roc_auc_score,
                                       average_precision_score)
from imblearn.over_sampling   import SMOTE

warnings.filterwarnings('ignore')

app  = Flask(__name__)
CORS(app)

# ── Globals
MODELS       = {}
SCALER       = None
FEATURE_COLS = []
STATUS       = {'ready': False, 'message': 'Starting…', 'progress': 0}
SEED         = 42

# ── Dataset config
KAGGLE_FILENAME = 'PS_20174392719_1491204439457_log.csv'
SAMPLE_ROWS     = 400_000   # ← change this. None = all 6.3M rows


# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD DATASET
# ═══════════════════════════════════════════════════════════
def find_dataset():
    if os.path.exists(KAGGLE_FILENAME):
        print(f'\n📂 Kaggle file found: {KAGGLE_FILENAME}')
        if SAMPLE_ROWS:
            print(f'   Loading {SAMPLE_ROWS:,} rows  '
                  f'(set SAMPLE_ROWS=None in app.py for all 6.3M)')
            df = pd.read_csv(KAGGLE_FILENAME, nrows=SAMPLE_ROWS)
        else:
            print('   Loading ALL rows…')
            df = pd.read_csv(KAGGLE_FILENAME)
        return df, f'Kaggle PaySim ({len(df):,} rows)'

    for fname in os.listdir('.'):
        if fname.endswith('.csv') and 'paysim' in fname.lower():
            df = pd.read_csv(fname)
            if 'isFraud' in df.columns:
                print(f'\n📂 Found: {fname}')
                return df, f'{fname} ({len(df):,} rows)'

    print('\n⚠️  No PaySim CSV found — using 50k synthetic fallback.')
    print('   Download real data: kaggle.com/datasets/ealaxi/paysim1\n')
    return _generate_synthetic(50_000), 'Synthetic (50k rows)'


def _generate_synthetic(n=50_000, seed=42):
    rng   = np.random.default_rng(seed)
    types = rng.choice(['PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN'],
                        n, p=[0.35,0.25,0.22,0.10,0.08])
    amt   = np.abs(rng.exponential(150_000, n)).round(2)
    oldO  = np.abs(rng.exponential(200_000, n)).round(2)
    newO  = np.maximum(0, oldO - amt * rng.uniform(0.5,1.5,n)).round(2)
    oldD  = np.abs(rng.exponential(80_000,  n)).round(2)
    newD  = (oldD + amt * rng.uniform(0.8,1.2,n)).round(2)
    step  = rng.integers(1, 745, n)
    risky = np.isin(types, ['TRANSFER','CASH_OUT'])
    nf    = int(0.013 * n)
    fi    = rng.choice(np.where(risky)[0], nf, replace=False)
    fraud = np.zeros(n, dtype=int); fraud[fi] = 1
    amt[fi] = np.abs(rng.exponential(500_000, nf)).round(2)
    newO[fi]= 0.0; oldD[fi] = 0.0
    return pd.DataFrame({
        'step':step,'type':types,'amount':amt,
        'nameOrig':['C0']*n,'oldbalanceOrg':oldO,'newbalanceOrig':newO,
        'nameDest':['C1']*n,'oldbalanceDest':oldD,'newbalanceDest':newD,
        'isFraud':fraud,'isFlaggedFraud':np.zeros(n,dtype=int)
    })


# ═══════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# Mirrors lab1_00_eda_analysis.ipynb exactly
# ═══════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['hour_of_day']             = out['step'] % 24
    out['orig_account_drained']    = ((out['newbalanceOrig']==0) &
                                       (out['oldbalanceOrg']>0)).astype(int)
    out['dest_was_empty']          = (out['oldbalanceDest']==0).astype(int)
    out['amount_to_balance_ratio'] = out['amount'] / (out['oldbalanceOrg']+1)
    out['error_balance_orig']      = np.abs(out['oldbalanceOrg'] -
                                             out['newbalanceOrig'] -
                                             out['amount'])
    out['error_balance_dest']      = np.abs(out['oldbalanceDest'] -
                                             out['newbalanceDest'] +
                                             out['amount'])
    out['log_amount']              = np.log1p(out['amount'])
    out['is_high_risk_type']       = out['type'].isin(
                                       ['TRANSFER','CASH_OUT']).astype(int)
    out['drain_and_risky']         = (out['orig_account_drained'] &
                                       out['is_high_risk_type']).astype(int)
    out = pd.get_dummies(out, columns=['type'], drop_first=True, dtype=int)
    drop = ['nameOrig','nameDest','isFlaggedFraud','step']
    return out.drop(columns=[c for c in drop if c in out.columns])


def engineer_single(tx: dict) -> dict:
    """Engineer features for one transaction from the dashboard."""
    amount   = float(tx.get('amount',         0))
    old_orig = float(tx.get('oldbalanceOrg',  0))
    new_orig = float(tx.get('newbalanceOrig', 0))
    old_dest = float(tx.get('oldbalanceDest', 0))
    new_dest = float(tx.get('newbalanceDest', 0))
    step     = int(tx.get('step', 1))
    tx_type  = tx.get('type', 'PAYMENT')
    hour     = int(tx.get('hour_of_day', step % 24))
    return {
        'amount':                  amount,
        'oldbalanceOrg':           old_orig,
        'newbalanceOrig':          new_orig,
        'oldbalanceDest':          old_dest,
        'newbalanceDest':          new_dest,
        'hour_of_day':             hour,
        'orig_account_drained':    int(new_orig==0 and old_orig>0),
        'dest_was_empty':          int(old_dest==0),
        'amount_to_balance_ratio': amount/(old_orig+1),
        'error_balance_orig':      abs(old_orig-new_orig-amount),
        'error_balance_dest':      abs(old_dest-new_dest+amount),
        'log_amount':              np.log1p(amount),
        'is_high_risk_type':       int(tx_type in ['TRANSFER','CASH_OUT']),
        'drain_and_risky':         int((new_orig==0 and old_orig>0) and
                                        tx_type in ['TRANSFER','CASH_OUT']),
        'type_CASH_OUT': int(tx_type=='CASH_OUT'),
        'type_DEBIT':    int(tx_type=='DEBIT'),
        'type_PAYMENT':  int(tx_type=='PAYMENT'),
        'type_TRANSFER': int(tx_type=='TRANSFER'),
    }


def to_array(feat_dict):
    return np.array([feat_dict.get(c, 0.0) for c in FEATURE_COLS],
                    dtype=float).reshape(1, -1)


# ═══════════════════════════════════════════════════════════
# STEP 3 — TRAIN ALL MODELS
# ═══════════════════════════════════════════════════════════
def train_all_models():
    global SCALER, FEATURE_COLS, MODELS, STATUS

    # ── Load
    STATUS['message'] = 'Loading dataset…'
    raw_df, source = find_dataset()

    # ── Engineer
    STATUS['message'] = f'Engineering features on {len(raw_df):,} rows…'
    print(f'\n⚙️  Engineering features…')
    t0   = time.time()
    full = engineer_features(raw_df)
    X    = full.drop(columns=['isFraud'])
    y    = full['isFraud']
    FEATURE_COLS = list(X.columns)
    print(f'   {len(FEATURE_COLS)} features  |  '
          f'{y.sum():,} fraud ({y.mean()*100:.3f}%)  |  '
          f'{time.time()-t0:.1f}s')

    # ── Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    # ── Scale
    SCALER  = RobustScaler()
    X_tr_sc = SCALER.fit_transform(X_train)
    X_te_sc = SCALER.transform(X_test)

    # ── SMOTE
    STATUS['message'] = 'Applying SMOTE…'
    print(f'\n⚙️  SMOTE…', end=' ', flush=True)
    t0 = time.time()
    smote = SMOTE(random_state=SEED, k_neighbors=5)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_sc, y_train)
    print(f'{len(X_tr_sm):,} training samples  ({time.time()-t0:.1f}s)')

    # ── Model definitions
    #
    # KEY CHANGE vs original:
    # Gradient Boosting → HistGradientBoostingClassifier
    #   Uses histogram binning. Natively supports missing values.
    #   Parallelises internally. 10-20x faster than regular GBM.
    #   class_weight param not supported → handled by SMOTE instead.
    #
    # SVM (RBF) → LinearSVC + CalibratedClassifierCV
    #   LinearSVC is O(n) not O(n²). Trains in seconds not hours.
    #   CalibratedClassifier wraps it with 5-fold cross-val to
    #   produce probability outputs (.predict_proba) which the
    #   dashboard needs for risk scores.
    #
    classifiers = [
        ('Logistic Regression',
         LogisticRegression(C=1.0, max_iter=1000,
                             class_weight='balanced', random_state=SEED),
         'logistic'),

        ('Decision Tree',
         DecisionTreeClassifier(max_depth=8,
                                 class_weight='balanced', random_state=SEED),
         'tree'),

        ('Random Forest',
         RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                 n_jobs=-1, random_state=SEED),
         'forest'),

        ('Hist Gradient Boosting',          # replaces slow GradientBoosting
         HistGradientBoostingClassifier(
             max_iter=100, learning_rate=0.1,
             max_leaf_nodes=31, random_state=SEED),
         'gradient'),

        ('K-Nearest Neighbours',
         KNeighborsClassifier(n_neighbors=5, n_jobs=-1, algorithm='ball_tree', leaf_size=40),
         'knn'),

        ('Naive Bayes',
         GaussianNB(),
         'naive'),

        ('Linear SVC',                      # replaces slow RBF SVM
         CalibratedClassifierCV(
             LinearSVC(C=1.0, class_weight='balanced',
                       max_iter=2000, random_state=SEED),
             cv=3, method='sigmoid'),
         'svm'),
    ]

    total = len(classifiers) + 1
    print(f'\n🏋️  Training {total} models on {len(X_tr_sm):,} samples…\n')

    for idx, (name, clf, mtype) in enumerate(classifiers, 1):
        STATUS.update({'message':  f'Training {name} ({idx}/{total})…',
                       'progress': int(idx / total * 90)})
        print(f'  [{idx}/{total}] {name:<30}', end=' ', flush=True)
        t0 = time.time()
        clf.fit(X_tr_sm, y_tr_sm)
        y_pred  = clf.predict(X_te_sc)
        y_proba = clf.predict_proba(X_te_sc)[:, 1]
        m = _metrics(y_test, y_pred, y_proba, X_tr_sm, X_te_sc)
        MODELS[name] = {'model': clf, 'type': mtype,
                        'metrics': m, 'feature_names': FEATURE_COLS}
        print(f'F1={m["f1"]:.4f}  AUC={m["roc_auc"]:.4f}  '
              f'PR={m["pr_auc"]:.4f}  ({time.time()-t0:.1f}s)')

    # ── Isolation Forest — trained on LEGIT rows only
    STATUS.update({'message': f'Training Isolation Forest ({total}/{total})…',
                   'progress': 95})
    print(f'  [{total}/{total}] {"Isolation Forest":<30}', end=' ', flush=True)
    t0 = time.time()
    legit_mask = (y_train.values == 0)
    iso = IsolationForest(n_estimators=200,
                          contamination=float(y.mean()),
                          random_state=SEED, n_jobs=-1)
    iso.fit(X_tr_sc[legit_mask])
    iso_scores = -iso.score_samples(X_te_sc)
    iso_preds  = (iso.predict(X_te_sc) == -1).astype(int)
    m = _metrics(y_test, iso_preds, iso_scores,
                 X_tr_sc[legit_mask], X_te_sc)
    m['note'] = 'Trained on legitimate rows only. Score = anomaly deviation.'
    MODELS['Isolation Forest'] = {'model': iso, 'type': 'isolation',
                                   'metrics': m, 'feature_names': FEATURE_COLS}
    print(f'F1={m["f1"]:.4f}  AUC={m["roc_auc"]:.4f}  '
          f'PR={m["pr_auc"]:.4f}  ({time.time()-t0:.1f}s)')

    # ── Save artefacts
    joblib.dump(SCALER,       'scaler_api.pkl')
    joblib.dump(FEATURE_COLS, 'feature_cols_api.pkl')
    for name, entry in MODELS.items():
        safe = name.replace(' ','_').replace('(','').replace(')','')
        joblib.dump(entry['model'], f'model_{safe}.pkl')

    STATUS.update({
        'ready': True, 'progress': 100,
        'message': f'{len(MODELS)} models trained on {source}',
        'models':  list(MODELS.keys()),
        'features': FEATURE_COLS,
        'dataset': {
            'source':     source,
            'total':      len(full),
            'fraud':      int(y.sum()),
            'fraud_rate': round(float(y.mean()*100), 3),
            'n_features': len(FEATURE_COLS),
            'train_rows': len(X_tr_sm),
            'test_rows':  len(X_te_sc),
        }
    })

    print(f'\n{"="*55}')
    print(f'  ✅  All {len(MODELS)} models ready')
    print(f'  📊  {source}')
    print(f'  🌐  Open lab1_dashboard_v2.html in Chrome/Edge')
    print(f'{"="*55}\n')


def _metrics(y_true, y_pred, y_score, X_train, X_test):
    return {
        'f1':           round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'roc_auc':      round(float(roc_auc_score(y_true, y_score)), 4),
        'pr_auc':       round(float(average_precision_score(y_true, y_score)), 4),
        'train_samples':int(len(X_train)),
        'test_samples': int(len(X_test)),
        'fraud_in_test':int(y_true.sum()),
    }


# ═══════════════════════════════════════════════════════════
# FEATURE CONTRIBUTIONS
# ═══════════════════════════════════════════════════════════
CONTRIB_METHOD = {
    'logistic':  'coef[i] × scaled_value[i]  — exact linear contribution',
    'tree':      'feature_importance[i] × |scaled_value[i]|  — Gini reduction',
    'forest':    'feature_importance[i] × |scaled_value[i]|  — mean Gini reduction across trees',
    'gradient':  'feature_importance[i] × |scaled_value[i]|  — histogram-based gain',
    'svm':       'scaled_value[i] × 2×(prob−0.5)  — linear margin proxy',
    'knn':       'scaled_value[i] × (prob−0.5)  — proximity-weighted deviation',
    'naive':     'scaled_value[i] × (prob−0.5)  — Gaussian likelihood proxy',
    'isolation': 'scaled_value[i]  — raw deviation from normal behaviour',
}

def get_contributions(model_name, feat_dict, fraud_prob):
    entry   = MODELS[model_name]
    model   = entry['model']
    mtype   = entry['type']
    scaled  = SCALER.transform(to_array(feat_dict))[0]
    contribs = {}

    if mtype == 'logistic':
        coef = model.coef_[0]
        for i, col in enumerate(FEATURE_COLS):
            contribs[col] = float(coef[i] * scaled[i])

    elif mtype in ('tree', 'forest', 'gradient'):
        imps = model.feature_importances_
        for i, col in enumerate(FEATURE_COLS):
            contribs[col] = float(imps[i] * abs(scaled[i]) *
                                   (1 if scaled[i] >= 0 else -1))

    elif mtype == 'svm':
        for i, col in enumerate(FEATURE_COLS):
            contribs[col] = float(scaled[i] * (fraud_prob - 0.5) * 2)

    elif mtype in ('knn', 'naive'):
        for i, col in enumerate(FEATURE_COLS):
            contribs[col] = float(scaled[i] * (fraud_prob - 0.5))

    elif mtype == 'isolation':
        for i, col in enumerate(FEATURE_COLS):
            contribs[col] = float(scaled[i])

    top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    return {
        'values': [{'feature': k, 'contribution': round(v, 5)} for k, v in top],
        'method': CONTRIB_METHOD.get(mtype, mtype),
    }


# ═══════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════
@app.route('/status')
def status():
    return jsonify(STATUS)


@app.route('/models')
def list_models():
    if not STATUS['ready']:
        return jsonify({'error': STATUS['message']}), 503
    return jsonify({
        name: {
            'metrics':    entry['metrics'],
            'type':       entry['type'],
            'is_anomaly': entry['type'] == 'isolation',
        }
        for name, entry in MODELS.items()
    })


@app.route('/predict', methods=['POST'])
def predict():
    if not STATUS['ready']:
        return jsonify({'error': 'Models still training…'}), 503

    data       = request.get_json(force=True)
    model_name = data.get('model', 'Random Forest')
    if model_name not in MODELS:
        return jsonify({'error': f'Unknown model: {model_name}',
                        'available': list(MODELS.keys())}), 400

    entry    = MODELS[model_name]
    mtype    = entry['type']
    feat_dict= engineer_single(data)
    X_scaled = SCALER.transform(to_array(feat_dict))

    if mtype == 'isolation':
        raw        = float(-entry['model'].score_samples(X_scaled)[0])
        fraud_prob = float(1 / (1 + np.exp(-10 * (raw - 0.5))))
        is_fraud   = bool(entry['model'].predict(X_scaled)[0] == -1)
    else:
        proba      = entry['model'].predict_proba(X_scaled)[0]
        fraud_prob = float(proba[1])
        is_fraud   = bool(entry['model'].predict(X_scaled)[0] == 1)

    risk = ('CRITICAL' if fraud_prob >= 0.80 else
            'HIGH'     if fraud_prob >= 0.50 else
            'MEDIUM'   if fraud_prob >= 0.25 else 'LOW')

    contrib = get_contributions(model_name, feat_dict, fraud_prob)

    return jsonify({
        'model':               model_name,
        'model_type':          mtype,
        'fraud_prob':          round(fraud_prob, 6),
        'legit_prob':          round(1 - fraud_prob, 6),
        'is_fraud':            is_fraud,
        'risk_level':          risk,
        'contributions':       contrib['values'],
        'contribution_method': contrib['method'],
        'features_used':       feat_dict,
        'metrics':             entry['metrics'],
    })


@app.route('/compare', methods=['POST'])
def compare_all():
    if not STATUS['ready']:
        return jsonify({'error': 'Models still training…'}), 503

    data    = request.get_json(force=True)
    results = {}
    feat_dict = engineer_single(data)
    X_scaled  = SCALER.transform(to_array(feat_dict))

    for name, entry in MODELS.items():
        mtype = entry['type']
        try:
            if mtype == 'isolation':
                raw  = float(-entry['model'].score_samples(X_scaled)[0])
                prob = float(1 / (1 + np.exp(-10 * (raw - 0.5))))
                lbl  = bool(entry['model'].predict(X_scaled)[0] == -1)
            else:
                p    = entry['model'].predict_proba(X_scaled)[0]
                prob = float(p[1])
                lbl  = bool(entry['model'].predict(X_scaled)[0] == 1)
            results[name] = {
                'fraud_prob': round(prob, 4),
                'is_fraud':   lbl,
                'f1':         entry['metrics']['f1'],
                'roc_auc':    entry['metrics']['roc_auc'],
            }
        except Exception as e:
            results[name] = {'error': str(e)}

    # AUC-weighted ensemble
    probs = [v['fraud_prob'] for v in results.values() if 'fraud_prob' in v]
    aucs  = [v['roc_auc']   for v in results.values() if 'roc_auc' in v]
    ws    = sum(aucs)
    ens   = sum(p*a for p,a in zip(probs,aucs))/ws if ws > 0 else 0.5
    results['⚖️ Ensemble (AUC-weighted)'] = {
        'fraud_prob': round(ens, 4), 'is_fraud': ens >= 0.5,
        'f1': None, 'roc_auc': None,
    }
    return jsonify(results)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('\n' + '='*55)
    print('  EEF606 Lab 1 — Fraud Detection API')
    print('  University of Buea')
    print('='*55)
    train_all_models()
    print('  API: http://127.0.0.1:5000\n')
    app.run(host='127.0.0.1', port=5000, debug=False)
