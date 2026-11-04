# -*- coding: utf-8 -*-
"""
Multi-timescale machine learning prediction pipeline for kidney damage in hyperuricemia.
Workflow: Data preprocessing, MICE imputation, Boruta feature selection,
Optuna hyperparameter tuning, model evaluation (ROC, PR, Calibration, DCA, Radar),
and SHAP interpretability.
"""

import os
import re
import warnings
import pickle
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from boruta import BorutaPy

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import shap

warnings.filterwarnings('ignore')

# Set computing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ======================================================================
# Global Matplotlib Style Configurations
# ======================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 30

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Uniform model color palette
colors_dict = {
    'Logistic': '#6F6F6F',
    'LR': '#6F6F6F',
    'RandomForest': '#547BB4',
    'XGBoost': '#C0321A',
    'LightGBM': '#629C35',
    'SVM': '#DD7C4F',
    'NN': '#6C61AF'
}

# Pipeline execution switch:
# False: Full initial run (data filtering, splitting, MICE imputation)
# True: Resume modeling directly using imputed datasets
USE_EXISTING_IMPUTED = False

desktop = r'C:\Users\57808\Desktop\English'

if not USE_EXISTING_IMPUTED:
    # ==================================================================
    # Step 1: Data Partitioning and Quality Control Preprocessing
    # ==================================================================
    input_path = os.path.join(desktop, '0.5-year group.xlsx')
    df_raw = pd.read_excel(input_path, header=None)

    var_types = df_raw.iloc[0, :].tolist()
    var_names = df_raw.iloc[1, :].tolist()
    data = df_raw.iloc[2:, :].copy()
    data.columns = var_names

    id_col = var_names[0]
    y_col = var_names[1]
    data = data.apply(pd.to_numeric, errors='coerce')

    # Stratified 7:3 split to prevent data leakage
    df_train_orig, df_test_orig = train_test_split(
        data, test_size=0.3, random_state=RANDOM_STATE, stratify=data[y_col]
    )
    df_train_orig = df_train_orig.reset_index(drop=True)
    df_test_orig = df_test_orig.reset_index(drop=True)

    # Feature filtering strictly based on training set criteria
    def filter_variables_on_train(df_train, var_types, var_names, id_col, y_col,
                                  missing_threshold=0.8, single_value_threshold=0.8, cv_threshold=0.1):
        n = len(df_train)
        keep_vars = [id_col, y_col]
        for col in var_names[2:]:
            missing_rate = df_train[col].isna().sum() / n
            if missing_rate > missing_threshold:
                continue
            non_missing = df_train[col].dropna()
            if len(non_missing) == 0:
                continue
            value_counts = non_missing.value_counts()
            max_freq = value_counts.max() / len(non_missing)
            if max_freq > single_value_threshold:
                continue
            try:
                mean_val = non_missing.mean()
                std_val = non_missing.std()
                cv = np.inf if mean_val == 0 else std_val / abs(mean_val)
                if cv < cv_threshold:
                    continue
            except Exception:
                pass
            keep_vars.append(col)
        return keep_vars

    keep_vars = filter_variables_on_train(df_train_orig, var_types, var_names, id_col, y_col)
    print(f'\nFeatures retained after training set filtering: {len(keep_vars)} / {len(var_names)}')

    df_train_filtered = df_train_orig[keep_vars].copy()
    df_test_filtered = df_test_orig[keep_vars].copy()

    type_row_orig = [var_types[var_names.index(v)] for v in keep_vars]
    col_name_row_orig = keep_vars

    train_unimputed_path = os.path.join(desktop, '1.train_dataset_unimputed.xlsx')
    test_unimputed_path = os.path.join(desktop, '1.test_dataset_unimputed.xlsx')

    with pd.ExcelWriter(train_unimputed_path, engine='openpyxl') as writer:
        pd.DataFrame([type_row_orig]).to_excel(writer, startrow=0, index=False, header=False)
        pd.DataFrame([col_name_row_orig]).to_excel(writer, startrow=1, index=False, header=False)
        df_train_filtered.to_excel(writer, startrow=2, index=False, header=False)

    with pd.ExcelWriter(test_unimputed_path, engine='openpyxl') as writer:
        pd.DataFrame([type_row_orig]).to_excel(writer, startrow=0, index=False, header=False)
        pd.DataFrame([col_name_row_orig]).to_excel(writer, startrow=1, index=False, header=False)
        df_test_filtered.to_excel(writer, startrow=2, index=False, header=False)

    # Multivariate Iterative Imputation via Random Forest
    id_train_series = df_train_filtered[id_col]
    id_test_series = df_test_filtered[id_col]
    features_train = df_train_filtered.drop(columns=[id_col])
    features_test = df_test_filtered.drop(columns=[id_col])

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        max_iter=10,
        random_state=42,
        initial_strategy='mean',
        verbose=0
    )
    print('\nFitting iterative Random Forest imputer on training set...')
    features_train_imputed = imputer.fit_transform(features_train)
    print('Imputing missing values in testing set without data leakage...')
    features_test_imputed = imputer.transform(features_test)

    df_train_imputed = pd.DataFrame(features_train_imputed, columns=features_train.columns, index=features_train.index)
    df_train_imputed[id_col] = id_train_series.values
    df_train_imputed = df_train_imputed[keep_vars]

    df_test_imputed = pd.DataFrame(features_test_imputed, columns=features_test.columns, index=features_test.index)
    df_test_imputed[id_col] = id_test_series.values
    df_test_imputed = df_test_imputed[keep_vars]

    train_imputed_path = os.path.join(desktop, '2.train_dataset_imputed.xlsx')
    test_imputed_path = os.path.join(desktop, '2.test_dataset_imputed.xlsx')

    with pd.ExcelWriter(train_imputed_path, engine='openpyxl') as writer:
        pd.DataFrame([type_row_orig]).to_excel(writer, startrow=0, index=False, header=False)
        pd.DataFrame([col_name_row_orig]).to_excel(writer, startrow=1, index=False, header=False)
        df_train_imputed.to_excel(writer, startrow=2, index=False, header=False)

    with pd.ExcelWriter(test_imputed_path, engine='openpyxl') as writer:
        pd.DataFrame([type_row_orig]).to_excel(writer, startrow=0, index=False, header=False)
        pd.DataFrame([col_name_row_orig]).to_excel(writer, startrow=1, index=False, header=False)
        df_test_imputed.to_excel(writer, startrow=2, index=False, header=False)

    params_info = "========== Imputation Parameters and Configurations ==========\n"
    params_info += f"Retained variables count (incl. ID & Target): {len(keep_vars)} / {len(var_names)}\n"
    params_info += f"Retained variables list:\n  {', '.join(keep_vars)}\n\n"
    params_info += "Variable types detail:\n"
    for v in keep_vars:
        idx = var_names.index(v)
        params_info += f"  {v}: {var_types[idx]}\n"

    params_info += "\n========== IterativeImputer Configurations ==========\n"
    imputer_configs = imputer.get_params()
    for param_name, param_value in imputer_configs.items():
        if param_name == 'estimator' and param_value is not None:
            params_info += f"  {param_name}: {repr(param_value)}\n"
        else:
            params_info += f"  {param_name}: {param_value}\n"

    params_path = os.path.join(desktop, '1.imputation_parameters.txt')
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write(params_info)
    print(f'Imputation parameters logged to: {params_path}')

else:
    # ==================================================================
    # Step 1 (Resume): Loading Processed and Imputed Datasets
    # ==================================================================
    print('\n=== Resuming pipeline using existing imputed datasets ===')
    train_imputed_path = os.path.join(desktop, '2.train_dataset_imputed.xlsx')
    test_imputed_path = os.path.join(desktop, '2.test_dataset_imputed.xlsx')

    df_train_raw = pd.read_excel(train_imputed_path, header=None)
    train_var_types = df_train_raw.iloc[0, :].tolist()
    train_var_names = df_train_raw.iloc[1, :].tolist()
    df_train_imputed = df_train_raw.iloc[2:, :].copy()
    df_train_imputed.columns = train_var_names
    df_train_imputed = df_train_imputed.apply(pd.to_numeric, errors='coerce')

    df_test_raw = pd.read_excel(test_imputed_path, header=None)
    test_var_names_orig = df_test_raw.iloc[1, :].tolist()
    df_test_imputed_full = df_test_raw.iloc[2:, :].copy()
    df_test_imputed_full.columns = test_var_names_orig
    df_test_imputed_full = df_test_imputed_full.apply(pd.to_numeric, errors='coerce')

    df_test_imputed = df_test_imputed_full[train_var_names].copy()

    keep_vars = train_var_names
    var_types = train_var_types
    var_names = train_var_names
    id_col = keep_vars[0]
    y_col = keep_vars[1]

    print(f' -> Testing cohort synchronized. Features count (incl. ID/Target): {len(keep_vars)}')

# ----------------------------------------------------------------------
# Step 2: Collinearity Screening via Pearson Correlation (Training Set)
# ----------------------------------------------------------------------
corr_data_train = df_train_imputed[keep_vars[1:]]
corr_matrix = corr_data_train.corr(method='pearson')

corr_matrix_path = os.path.join(desktop, '3.pearson_correlation_matrix.xlsx')
corr_matrix.to_excel(corr_matrix_path)
print(f'\nPairwise Pearson correlation matrix saved to: {corr_matrix_path}')

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    out_corr = os.path.join(desktop, '2.high_correlation_pairs.txt')
    with open(out_corr, 'w', encoding='utf-8') as f:
        f.write("=========================================================================\n")
        f.write("             Training Cohort Collinearity Report (|r| > 0.8)             \n")
        f.write("=========================================================================\n\n")
        for pair in high_corr_pairs:
            f.write(f"{pair[0]} and {pair[1]}: r = {pair[2]:.4f}\n")
    print(f'\nHigh correlation feature pairs (|r| > 0.8) detected. Logged to: {out_corr}')
    print(">> Please remove redundant features in '2.train_dataset_imputed.xlsx' and rerun.")
    exit()
else:
    print('\nAll pairwise feature correlations are within the threshold (|r| <= 0.8). Proceeding.')

# ----------------------------------------------------------------------
# Step 3: Boruta Feature Selection and Multi-Seed Stability Assessment
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("  Step 3: Boruta Feature Selection across Multiple Random Seeds")
print("=" * 70)

X_boruta_in = df_train_imputed[keep_vars[2:]].copy()
y_boruta_in = df_train_imputed[y_col].astype(int)

# Extract pre-imputation missing rates from raw training dataset
missing_rate_dict = {}
train_unimputed_file = os.path.join(desktop, '1.train_dataset_unimputed.xlsx')

if os.path.exists(train_unimputed_file):
    df_unimputed_raw = pd.read_excel(train_unimputed_file, header=None)
    unimputed_cols = df_unimputed_raw.iloc[1, :].tolist()
    df_unimputed_data = df_unimputed_raw.iloc[2:, :].copy()
    df_unimputed_data.columns = unimputed_cols
    df_unimputed_data = df_unimputed_data.apply(pd.to_numeric, errors='coerce')
    for col in X_boruta_in.columns:
        if col in df_unimputed_data.columns:
            m_rate = (df_unimputed_data[col].isna().sum() / len(df_unimputed_data)) * 100.0
            missing_rate_dict[col] = round(m_rate, 2)
        else:
            missing_rate_dict[col] = 0.0
else:
    for col in X_boruta_in.columns:
        missing_rate_dict[col] = 0.0

# 10 independent random seeds for stability assessment
N_SEEDS = 10
stability_seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
feature_selection_counts = {col: 0 for col in X_boruta_in.columns}
primary_seed_support = {}

print(f"Executing Boruta across {N_SEEDS} independent random seeds...")
for i, s in enumerate(stability_seeds):
    rf_temp = RandomForestClassifier(n_jobs=-1, random_state=s)
    boruta_temp = BorutaPy(
        estimator=rf_temp,
        n_estimators='auto',
        perc=90,
        alpha=0.05,
        two_step=True,
        max_iter=100,
        random_state=s,
        verbose=0
    )
    boruta_temp.fit(X_boruta_in.values, y_boruta_in.values)

    for col, is_selected in zip(X_boruta_in.columns, boruta_temp.support_):
        if is_selected:
            feature_selection_counts[col] += 1
        if s == RANDOM_STATE:
            primary_seed_support[col] = is_selected

    print(f" -> [Seed {s}] Completed ({i + 1}/{N_SEEDS}) | Confirmed features: {sum(boruta_temp.support_)}")

# Map column abbreviations to descriptive names
mapping_path = os.path.join(desktop, 'Abbreviations of variables.xlsx')
name_map = {}
if os.path.exists(mapping_path):
    mapping_df = pd.read_excel(mapping_path, header=None)
    name_map = dict(zip(mapping_df.iloc[:, 1].astype(str).str.strip(), mapping_df.iloc[:, 0].astype(str).str.strip()))

# Export Supplementary Table S6
stability_records = []
for col in X_boruta_in.columns:
    count = feature_selection_counts[col]
    freq_pct = (count / N_SEEDS) * 100.0
    primary_status = "Confirmed" if primary_seed_support.get(col, False) else "Rejected"
    final_status = "Confirmed (Selected)" if (primary_status == "Confirmed" and freq_pct >= 70.0) else "Rejected"

    stability_records.append({
        'Feature Code': col,
        'Feature Name': name_map.get(str(col).strip(), col),
        'Missing Rate in Train (%)': f"{missing_rate_dict.get(col, 0.0):.2f}%",
        f'Selection Count (/{N_SEEDS})': count,
        'Selection Frequency (%)': f"{freq_pct:.1f}%",
        'Primary Run (Seed=42)': primary_status,
        'Final Status': final_status
    })

df_boruta_stability = pd.DataFrame(stability_records)
df_boruta_stability.sort_values(by=['Selection Frequency (%)', 'Primary Run (Seed=42)'], ascending=False, inplace=True)

s6_path = os.path.join(desktop, 'Supplementary_Table_S6_Boruta_Stability.xlsx')
df_boruta_stability.to_excel(s6_path, index=False)
print(f"\nBoruta stability summary exported to: {s6_path}")

# Primary confirmed feature subset
boruta_selected = [col for col in X_boruta_in.columns if primary_seed_support.get(col, False)]

print('\n========== Confirmed Features and Stability Summary ==========')
for col in boruta_selected:
    f_name = name_map.get(str(col).strip(), col)
    freq = (feature_selection_counts[col] / N_SEEDS) * 100
    print(f"  * [{col}] {f_name:<20} | Missing: {missing_rate_dict.get(col, 0.0):.2f}% | Stability: {freq:.1f}%")

print(f'\nCandidate features evaluated: {X_boruta_in.shape[1]}')
print(f'Confirmed features selected: {len(boruta_selected)}')
print(f'Selected feature list: {boruta_selected}')

if not boruta_selected:
    print('Warning: No features selected by Boruta. Pipeline terminated.')
    exit()

# ----------------------------------------------------------------------
# Step 4: Z-Score Standardization (Fitted on Training Set Only)
# ----------------------------------------------------------------------
continuous_cols_raw = []
for i, col in enumerate(var_names):
    if i < 2:
        continue
    val_type_str = str(var_types[i]).strip().lower()
    if val_type_str in ['continuous', '连续', 'numeric']:
        continuous_cols_raw.append(col)

continuous_cols_selected = [col for col in boruta_selected if col in continuous_cols_raw]
print(f'\nContinuous features undergoing Z-score standardization: {continuous_cols_selected}')

final_cols = [id_col, y_col] + boruta_selected
new_var_types = []
for col in final_cols:
    idx = var_names.index(col)
    new_var_types.append(var_types[idx])

train_selected_unscaled_path = os.path.join(desktop, '4.train_dataset_selected_unscaled.xlsx')
df_train_selected_unscaled = df_train_imputed[final_cols]
with pd.ExcelWriter(train_selected_unscaled_path, engine='openpyxl') as writer:
    pd.DataFrame([new_var_types]).to_excel(writer, startrow=0, index=False, header=False)
    pd.DataFrame([final_cols]).to_excel(writer, startrow=1, index=False, header=False)
    df_train_selected_unscaled.to_excel(writer, startrow=2, index=False, header=False)

scaler = StandardScaler()
df_train_scaled = df_train_imputed.copy()
df_test_scaled = df_test_imputed.copy()

if continuous_cols_selected:
    X_train_scaled_vals = scaler.fit_transform(df_train_imputed[continuous_cols_selected])
    X_test_scaled_vals = scaler.transform(df_test_imputed[continuous_cols_selected])

    for idx, col in enumerate(continuous_cols_selected):
        df_train_scaled[col] = X_train_scaled_vals[:, idx]
        df_test_scaled[col] = X_test_scaled_vals[:, idx]

train_output_path = os.path.join(desktop, '5.train_dataset.xlsx')
test_output_path = os.path.join(desktop, '5.test_dataset.xlsx')

df_train_final_export = df_train_scaled[final_cols]
df_test_final_export = df_test_scaled[final_cols]

df_train_final_export.to_excel(train_output_path, index=False)
df_test_final_export.to_excel(test_output_path, index=False)
print(f'\nFinal preprocessed datasets saved:\n - Training: {train_output_path}\n - Testing: {test_output_path}')

X_train = df_train_final_export[boruta_selected].values
y_train = df_train_final_export[y_col].astype(int).values
X_test = df_test_final_export[boruta_selected].values
y_test = df_test_final_export[y_col].astype(int).values

scale_pos_weight_val = (y_train == 0).sum() / (y_train == 1).sum()

# ----------------------------------------------------------------------
# Step 5: Hyperparameter Optimization via Optuna (5-Fold CV on Training Set)
# ----------------------------------------------------------------------
def objective_logistic(trial):
    C = trial.suggest_loguniform('C', 1e-3, 1e3)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    solver = 'saga' if penalty in ['l1', 'elasticnet'] else 'lbfgs'
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1) if penalty == 'elasticnet' else None
    model = LogisticRegression(
        C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio,
        max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_pred_proba))
    return np.mean(aucs)

def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 5, 100)
    max_depth = trial.suggest_int('max_depth', 2, 12)
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_pred_proba))
    return np.mean(aucs)

def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 6)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.5)
    reg_alpha = trial.suggest_loguniform('alpha', 1e-8, 1.0)
    reg_lambda = trial.suggest_loguniform('lambda', 1e-8, 1.0)
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight_val,
        random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_pred_proba))
    return np.mean(aucs)

def objective_lgbm(trial):
    max_depth = trial.suggest_int('max_depth', 2, 6)
    num_leaves = trial.suggest_int('num_leaves', 4, 2**max_depth)
    reg_alpha = trial.suggest_loguniform('lambda_l1', 1e-8, 10.0)
    reg_lambda = trial.suggest_loguniform('lambda_l2', 1e-8, 10.0)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
    model = LGBMClassifier(
        num_leaves=num_leaves, max_depth=max_depth, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        learning_rate=learning_rate, class_weight='balanced', random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_pred_proba))
    return np.mean(aucs)

def objective_svm(trial):
    C = trial.suggest_loguniform('C', 1e-3, 100.0)
    model = SVC(C=C, probability=True, random_state=RANDOM_STATE, class_weight='balanced', max_iter=10000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_pred_proba))
    return np.mean(aucs)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, n_layers, units, dropout_rate):
        super(SimpleNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, units))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(units, units))
            layers.append(nn.ReLU())
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(units, 1)
        self.hidden = nn.Sequential(*layers)

    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout(x)
        x = self.output(x)
        return x.squeeze()

def objective_nn_pytorch(trial):
    units = trial.suggest_int('hidden_dim', 10, 100)
    n_layers = trial.suggest_int('n_layers', 2, 5)
    lr = trial.suggest_loguniform('learning_rate', 5e-3, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
    dropout_rate = 0.2
    batch_size = 64
    epochs = 40

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = SimpleNN(X_train.shape[1], n_layers, units, dropout_rate).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                preds.append(torch.sigmoid(outputs).cpu().numpy())

        y_pred_proba = np.concatenate(preds)
        aucs.append(roc_auc_score(y_val, y_pred_proba))

    return np.mean(aucs)

def predict_proba_nn(model, X_data):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_data, dtype=torch.float32).to(device)
        logits = model(X_t)
        proba = torch.sigmoid(logits).cpu().numpy()
    return np.column_stack([1 - proba, proba])

def train_nn_model(X_train_f, y_train_f, input_dim, params, epochs=100, batch_size=64):
    model = SimpleNN(input_dim, params['n_layers'], params['hidden_dim'], 0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.tensor(X_train_f, dtype=torch.float32), torch.tensor(y_train_f, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

models_info = {
    'Logistic': (LogisticRegression, objective_logistic),
    'RandomForest': (RandomForestClassifier, objective_rf),
    'XGBoost': (XGBClassifier, objective_xgb),
    'LightGBM': (LGBMClassifier, objective_lgbm),
    'SVM': (SVC, objective_svm),
    'NN': (SimpleNN, objective_nn_pytorch)
}

search_spaces_doc = {
    'Logistic': {
        'C': 'LogUniform[1e-3, 1e3]',
        'penalty': "Categorical['l1', 'l2', 'elasticnet']",
        'l1_ratio': 'Uniform[0, 1]'
    },
    'RandomForest': {
        'n_estimators': 'UniformInt[5, 100]',
        'max_depth': 'UniformInt[2, 12]'
    },
    'XGBoost': {
        'max_depth': 'UniformInt[2, 6]',
        'learning_rate': 'LogUniform[0.01, 0.5]',
        'alpha': 'LogUniform[1e-8, 1.0]',
        'lambda': 'LogUniform[1e-8, 1.0]'
    },
    'LightGBM': {
        'num_leaves': 'UniformInt[4, 64]',
        'max_depth': 'UniformInt[2, 6]',
        'lambda_l1': 'LogUniform[1e-8, 10.0]',
        'lambda_l2': 'LogUniform[1e-8, 10.0]',
        'learning_rate': 'LogUniform[0.01, 0.3]'
    },
    'SVM': {
        'C': 'LogUniform[1e-3, 100.0]'
    },
    'NN': {
        'hidden_dim': 'UniformInt[10, 100]',
        'n_layers': 'UniformInt[2, 5]',
        'learning_rate': 'LogUniform[5e-3, 1e-2]',
        'weight_decay': 'LogUniform[1e-5, 1e-1]'
    }
}

N_TRIALS = 30
best_params = {}
study_objects = {}

for name, (model_class, objective_func) in models_info.items():
    print(f'\n========== Optimizing {name} (Training Set Only) ==========')
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner()
    )
    study.optimize(objective_func, n_trials=N_TRIALS, show_progress_bar=True)
    best_params[name] = study.best_params
    study_objects[name] = study

hp_records = []
for name, params_dict in search_spaces_doc.items():
    for param_name, space_desc in params_dict.items():
        val = best_params[name].get(param_name, 'N/A')
        if isinstance(val, float):
            val = f"{val:.5e}" if val < 1e-3 or val > 1e4 else f"{val:.4f}"
        hp_records.append({
            'Model': name,
            'Parameter': param_name,
            'Search Spaces': space_desc,
            'Best Value': val
        })

df_hp_summary = pd.DataFrame(hp_records)
df_hp_summary.to_excel(os.path.join(desktop, '6.hyperparameters_summary.xlsx'), index=False)
print('\n--- Hyperparameter Optimization Summary ---')
print(df_hp_summary.to_string(index=False))

train_config_records = [
    {'Model': 'NN', 'Parameter': 'Batch Size', 'Value': '64'},
    {'Model': 'NN', 'Parameter': 'Total Epochs', 'Value': '100'}
]
df_train_config = pd.DataFrame(train_config_records)
train_config_path = os.path.join(desktop, '7.training_config_summary.xlsx')
df_train_config.to_excel(train_config_path, index=False)

# ----------------------------------------------------------------------
# Step 6: Final Model Evaluation, Retraining, and Bootstrap 95% CIs
# ----------------------------------------------------------------------
def find_optimal_threshold(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_indices = tpr - fpr
    optimal_idx = np.argmax(youden_indices)
    optimal_threshold = thresholds[optimal_idx]
    return np.clip(optimal_threshold, 1e-4, 1.0 - 1e-4)

def bootstrap_metric_ci(y_true, y_pred_proba, threshold, n_bootstraps=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    bootstrapped_metrics = {
        'AUC': [], 'PR-AUC': [], 'Accuracy': [], 'F1': [], 'Sensitivity': [], 'Specificity': []
    }
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_t_b = y_true[indices]
        y_p_b = y_pred_proba[indices]
        if len(np.unique(y_t_b)) < 2:
            continue
        y_pred_b = (y_p_b >= threshold).astype(int)

        auc_val = roc_auc_score(y_t_b, y_p_b)
        precision, recall, _ = precision_recall_curve(y_t_b, y_p_b)
        pr_auc_val = auc(recall, precision)
        acc = accuracy_score(y_t_b, y_pred_b)
        f1 = f1_score(y_t_b, y_pred_b)
        sens = recall_score(y_t_b, y_pred_b)
        tn, fp, fn, tp = confusion_matrix(y_t_b, y_pred_b).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        bootstrapped_metrics['AUC'].append(auc_val)
        bootstrapped_metrics['PR-AUC'].append(pr_auc_val)
        bootstrapped_metrics['Accuracy'].append(acc)
        bootstrapped_metrics['F1'].append(f1)
        bootstrapped_metrics['Sensitivity'].append(sens)
        bootstrapped_metrics['Specificity'].append(spec)

    ci_results = {}
    for metric, values in bootstrapped_metrics.items():
        if len(values) > 0:
            low = np.percentile(values, 2.5)
            high = np.percentile(values, 97.5)
        else:
            low, high = 0.0, 0.0
        ci_results[metric] = (low, high)
    return ci_results

plot_data = {'Train': {}, 'Test': {}}
results_raw = {}
metrics_records = []
models = {}

for name, (model_class, _) in models_info.items():
    print(f'\nEvaluating {name} via 5-Fold Cross-Validation and final retraining...')
    params = best_params[name].copy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {
        'AUC': [], 'PR-AUC': [], 'Accuracy': [], 'F1': [], 'Sensitivity': [], 'Specificity': [], 'Threshold': []
    }

    tprs = []
    precisions = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    oof_y_true = []
    oof_y_pred_proba = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        if name == 'Logistic':
            penalty = params.get('penalty', 'l2')
            solver = 'saga' if penalty in ['l1', 'elasticnet'] else 'lbfgs'
            l1_ratio = params.get('l1_ratio', None) if penalty == 'elasticnet' else None
            lr_params = {
                'C': params.get('C', 1.0),
                'penalty': penalty,
                'solver': solver,
                'max_iter': 1000,
                'random_state': RANDOM_STATE,
                'class_weight': 'balanced'
            }
            if penalty == 'elasticnet':
                lr_params['l1_ratio'] = l1_ratio
            fold_model = LogisticRegression(**lr_params)
            fold_model.fit(X_tr, y_tr)
            y_pred_proba_val = fold_model.predict_proba(X_val)[:, 1]
            y_pred_proba_tr = fold_model.predict_proba(X_tr)[:, 1]

        elif name == 'RandomForest':
            fold_model = RandomForestClassifier(
                **params, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
            )
            fold_model.fit(X_tr, y_tr)
            y_pred_proba_val = fold_model.predict_proba(X_val)[:, 1]
            y_pred_proba_tr = fold_model.predict_proba(X_tr)[:, 1]

        elif name == 'XGBoost':
            spw_fold = (y_tr == 0).sum() / (y_tr == 1).sum()
            fold_model = XGBClassifier(
                **params, scale_pos_weight=spw_fold, random_state=RANDOM_STATE,
                use_label_encoder=False, eval_metric='auc', n_jobs=-1, early_stopping_rounds=15
            )
            fold_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            y_pred_proba_val = fold_model.predict_proba(X_val)[:, 1]
            y_pred_proba_tr = fold_model.predict_proba(X_tr)[:, 1]

        elif name == 'LightGBM':
            from lightgbm import early_stopping
            fold_model = LGBMClassifier(
                **params, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
            )
            fold_model.fit(
                X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc',
                callbacks=[early_stopping(stopping_rounds=15, verbose=False)]
            )
            y_pred_proba_val = fold_model.predict_proba(X_val)[:, 1]
            y_pred_proba_tr = fold_model.predict_proba(X_tr)[:, 1]

        elif name == 'SVM':
            fold_model = SVC(
                **params, probability=True, random_state=RANDOM_STATE, class_weight='balanced', max_iter=10000
            )
            fold_model.fit(X_tr, y_tr)
            y_pred_proba_val = fold_model.predict_proba(X_val)[:, 1]
            y_pred_proba_tr = fold_model.predict_proba(X_tr)[:, 1]

        elif name == 'NN':
            fold_model = train_nn_model(X_tr, y_tr, X_train.shape[1], params, epochs=100, batch_size=64)
            y_pred_proba_val = predict_proba_nn(fold_model, X_val)[:, 1]
            y_pred_proba_tr = predict_proba_nn(fold_model, X_tr)[:, 1]

        fold_opt_threshold = find_optimal_threshold(y_tr, y_pred_proba_tr)
        fold_y_pred = (y_pred_proba_val >= fold_opt_threshold).astype(int)

        fold_auc = roc_auc_score(y_val, y_pred_proba_val)
        p_val, r_val, _ = precision_recall_curve(y_val, y_pred_proba_val)
        fold_pr_auc = auc(r_val, p_val)

        fold_acc = accuracy_score(y_val, fold_y_pred)
        fold_f1 = f1_score(y_val, fold_y_pred)
        fold_sens = recall_score(y_val, fold_y_pred)
        tn, fp, fn, tp = confusion_matrix(y_val, fold_y_pred).ravel()
        fold_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        fold_metrics['AUC'].append(fold_auc)
        fold_metrics['PR-AUC'].append(fold_pr_auc)
        fold_metrics['Accuracy'].append(fold_acc)
        fold_metrics['F1'].append(fold_f1)
        fold_metrics['Sensitivity'].append(fold_sens)
        fold_metrics['Specificity'].append(fold_spec)
        fold_metrics['Threshold'].append(fold_opt_threshold)

        fpr, tpr, _ = roc_curve(y_val, y_pred_proba_val)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        r_val_sorted, p_val_sorted = r_val[::-1], p_val[::-1]
        interp_precision = np.interp(mean_recall, r_val_sorted, p_val_sorted)
        precisions.append(interp_precision)

        oof_y_true.extend(y_val)
        oof_y_pred_proba.extend(y_pred_proba_val)

    train_summary = {}
    for m in ['AUC', 'PR-AUC', 'Accuracy', 'F1', 'Sensitivity', 'Specificity', 'Threshold']:
        train_summary[m] = (np.mean(fold_metrics[m]), np.std(fold_metrics[m]))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_precision = np.mean(precisions, axis=0)

    plot_data['Train'][name] = {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'mean_recall': mean_recall,
        'mean_precision': mean_precision,
        'oof_y_true': np.array(oof_y_true),
        'oof_y_pred_proba': np.array(oof_y_pred_proba),
        'auc': train_summary['AUC'][0]
    }

    # Final retraining on complete training cohort
    if name == 'Logistic':
        penalty = params.get('penalty', 'l2')
        solver = 'saga' if penalty in ['l1', 'elasticnet'] else 'lbfgs'
        l1_ratio = params.get('l1_ratio', None) if penalty == 'elasticnet' else None
        lr_params = {
            'C': params.get('C', 1.0),
            'penalty': penalty,
            'solver': solver,
            'max_iter': 1000,
            'random_state': RANDOM_STATE,
            'class_weight': 'balanced'
        }
        if penalty == 'elasticnet':
            lr_params['l1_ratio'] = l1_ratio
        model = LogisticRegression(**lr_params)
        model.fit(X_train, y_train)

    elif name == 'RandomForest':
        model = RandomForestClassifier(
            **params, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
        )
        model.fit(X_train, y_train)

    elif name == 'XGBoost':
        model = XGBClassifier(
            **params, scale_pos_weight=scale_pos_weight_val, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric='auc', n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)

    elif name == 'LightGBM':
        model = LGBMClassifier(
            **params, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
        )
        model.fit(X_train, y_train)

    elif name == 'SVM':
        model = SVC(
            **params, probability=True, random_state=RANDOM_STATE, class_weight='balanced', max_iter=10000
        )
        model.fit(X_train, y_train)

    elif name == 'NN':
        model = train_nn_model(X_train, y_train, X_train.shape[1], params, epochs=100, batch_size=64)

    models[name] = model

    if name == 'NN':
        y_pred_proba_train_full = predict_proba_nn(model, X_train)[:, 1]
        y_pred_proba_test = predict_proba_nn(model, X_test)[:, 1]
    else:
        y_pred_proba_train_full = model.predict_proba(X_train)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    opt_threshold = find_optimal_threshold(y_train, y_pred_proba_train_full)
    y_pred_test = (y_pred_proba_test >= opt_threshold).astype(int)

    test_auc = roc_auc_score(y_test, y_pred_proba_test)
    p_test, r_test, _ = precision_recall_curve(y_test, y_pred_proba_test)
    test_pr_auc = auc(r_test, p_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_sens = recall_score(y_test, y_pred_test)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_test).ravel()
    test_spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0

    test_cis = bootstrap_metric_ci(
        y_test, y_pred_proba_test, opt_threshold, n_bootstraps=1000, random_state=RANDOM_STATE
    )

    plot_data['Test'][name] = {
        'y_true': y_test,
        'y_pred_proba': y_pred_proba_test,
        'auc': test_auc
    }

    results_raw[name] = {
        'Train': {
            'AUC': train_summary['AUC'][0],
            'PR-AUC': train_summary['PR-AUC'][0],
            'Accuracy': train_summary['Accuracy'][0],
            'F1': train_summary['F1'][0],
            'Sensitivity': train_summary['Sensitivity'][0],
            'Specificity': train_summary['Specificity'][0]
        },
        'Test': {
            'AUC': test_auc,
            'PR-AUC': test_pr_auc,
            'Accuracy': test_acc,
            'F1': test_f1,
            'Sensitivity': test_sens,
            'Specificity': test_spec
        }
    }

    # 1,000 Bootstrap 95% CIs for 5-fold cross-validation out-of-fold (OOF) predictions
    train_opt_thresh = train_summary['Threshold'][0]
    train_cis = bootstrap_metric_ci(
        np.array(oof_y_true),
        np.array(oof_y_pred_proba),
        train_opt_thresh,
        n_bootstraps=1000,
        random_state=RANDOM_STATE
    )

    display_name = 'LR' if name == 'Logistic' else name

    # Record Training Set Performance (95% CIs)
    metrics_records.append({
        'Model': display_name,
        'Dataset': 'Train',
        'AUC': f"{train_summary['AUC'][0]:.4f} (95% CI: {train_cis['AUC'][0]:.4f}–{train_cis['AUC'][1]:.4f})",
        'PR-AUC': f"{train_summary['PR-AUC'][0]:.4f} (95% CI: {train_cis['PR-AUC'][0]:.4f}–{train_cis['PR-AUC'][1]:.4f})",
        'Accuracy': f"{train_summary['Accuracy'][0]:.4f} (95% CI: {train_cis['Accuracy'][0]:.4f}–{train_cis['Accuracy'][1]:.4f})",
        'F1-Score': f"{train_summary['F1'][0]:.4f} (95% CI: {train_cis['F1'][0]:.4f}–{train_cis['F1'][1]:.4f})",
        'Sensitivity': f"{train_summary['Sensitivity'][0]:.4f} (95% CI: {train_cis['Sensitivity'][0]:.4f}–{train_cis['Sensitivity'][1]:.4f})",
        'Specificity': f"{train_summary['Specificity'][0]:.4f} (95% CI: {train_cis['Specificity'][0]:.4f}–{train_cis['Specificity'][1]:.4f})",
        'Optimal Threshold': f"{train_opt_thresh:.4f}"
    })

    # Record Testing Set Performance (95% CIs)
    metrics_records.append({
        'Model': display_name,
        'Dataset': 'Test',
        'AUC': f"{test_auc:.4f} (95% CI: {test_cis['AUC'][0]:.4f}–{test_cis['AUC'][1]:.4f})",
        'PR-AUC': f"{test_pr_auc:.4f} (95% CI: {test_cis['PR-AUC'][0]:.4f}–{test_cis['PR-AUC'][1]:.4f})",
        'Accuracy': f"{test_acc:.4f} (95% CI: {test_cis['Accuracy'][0]:.4f}–{test_cis['Accuracy'][1]:.4f})",
        'F1-Score': f"{test_f1:.4f} (95% CI: {test_cis['F1'][0]:.4f}–{test_cis['F1'][1]:.4f})",
        'Sensitivity': f"{test_sens:.4f} (95% CI: {test_cis['Sensitivity'][0]:.4f}–{test_cis['Sensitivity'][1]:.4f})",
        'Specificity': f"{test_spec:.4f} (95% CI: {test_cis['Specificity'][0]:.4f}–{test_cis['Specificity'][1]:.4f})",
        'Optimal Threshold': f"{opt_threshold:.4f}"
    })

df_metrics_summary = pd.DataFrame(metrics_records)
metrics_excel_path = os.path.join(desktop, '8.models_metrics_summary.xlsx')

with pd.ExcelWriter(metrics_excel_path, engine='openpyxl') as writer:
    df_metrics_summary.to_excel(writer, sheet_name='Summary_All_95CI', index=False)
    df_metrics_summary[df_metrics_summary['Dataset'] == 'Test'].to_excel(writer, sheet_name='Testing_Set_Table2', index=False)
    df_metrics_summary[df_metrics_summary['Dataset'] == 'Train'].to_excel(writer, sheet_name='Training_Set_Table2', index=False)

print(f'\nModel performance metrics saved to: {metrics_excel_path}')
print('\n--- Performance Summary Preview (Train & Test with 95% CIs) ---')
print(df_metrics_summary[['Model', 'Dataset', 'AUC', 'F1-Score', 'Optimal Threshold']].to_string(index=False))

# ----------------------------------------------------------------------
# Step 7: Graphical Evaluation (ROC, PR, Calibration, DCA, and Radar)
# ----------------------------------------------------------------------
def calculate_net_benefit(y_true, y_pred_proba, thresholds):
    net_benefits = []
    n = len(y_true)
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if t == 1.0:
            nb = 0.0
        else:
            nb = (tp / n) - (fp / n) * (t / (1.0 - t))
        net_benefits.append(nb)
    return np.array(net_benefits)

def plot_all_curves(plot_data, output_dir):
    for dataset in ['Train', 'Test']:
        # 1. ROC Curves
        plt.figure(figsize=(9, 8))
        for name in models_info.keys():
            display_name = 'LR' if name == 'Logistic' else name
            if dataset == 'Train':
                fpr = plot_data['Train'][name]['mean_fpr']
                tpr = plot_data['Train'][name]['mean_tpr']
                auc_val = plot_data['Train'][name]['auc']
            else:
                fpr, tpr, _ = roc_curve(plot_data['Test'][name]['y_true'], plot_data['Test'][name]['y_pred_proba'])
                auc_val = plot_data['Test'][name]['auc']
            plt.plot(fpr, tpr, label=f'{display_name} ({auc_val:.3f})', color=colors_dict[name], lw=3.5)
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlabel('False Positive Rate', fontsize=24, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=24, fontweight='bold')
        plt.title(f'ROC Curves - {dataset}', fontsize=28, fontweight='bold', pad=15)
        plt.legend(loc='lower right', fontsize=20, frameon=True)
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'9.ROC_curves_{dataset}.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, f'9.ROC_curves_{dataset}.pdf'))
        plt.close()

        # 2. PR Curves
        plt.figure(figsize=(9, 8))
        for name in models_info.keys():
            display_name = 'LR' if name == 'Logistic' else name
            if dataset == 'Train':
                recall = plot_data['Train'][name]['mean_recall']
                precision = plot_data['Train'][name]['mean_precision']
            else:
                precision, recall, _ = precision_recall_curve(plot_data['Test'][name]['y_true'], plot_data['Test'][name]['y_pred_proba'])
            plt.plot(recall, precision, label=display_name, color=colors_dict[name], lw=3.5)
        plt.xlabel('Recall', fontsize=24, fontweight='bold')
        plt.ylabel('Precision', fontsize=24, fontweight='bold')
        plt.title(f'PR Curves - {dataset}', fontsize=28, fontweight='bold', pad=15)
        plt.legend(loc='lower left', fontsize=20, frameon=True)
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'9.PR_curves_{dataset}.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, f'9.PR_curves_{dataset}.pdf'))
        plt.close()

        # 3. Calibration Curves
        plt.figure(figsize=(9, 8))
        for name in models_info.keys():
            display_name = 'LR' if name == 'Logistic' else name
            if dataset == 'Train':
                y_true = plot_data['Train'][name]['oof_y_true']
                y_pred_proba = plot_data['Train'][name]['oof_y_pred_proba']
            else:
                y_true = plot_data['Test'][name]['y_true']
                y_pred_proba = plot_data['Test'][name]['y_pred_proba']
            prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', markersize=8, label=display_name, color=colors_dict[name], lw=3.5)
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlabel('Mean Predicted Probability', fontsize=24, fontweight='bold')
        plt.ylabel('Fraction of Positives', fontsize=24, fontweight='bold')
        plt.title(f'Calibration Curves - {dataset}', fontsize=28, fontweight='bold', pad=15)
        plt.legend(loc='upper left', fontsize=20, frameon=True)
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'9.Calibration_curves_{dataset}.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, f'9.Calibration_curves_{dataset}.pdf'))
        plt.close()

        # 4. Decision Curve Analysis (DCA)
        plt.figure(figsize=(9, 8))
        thresholds = np.linspace(0.01, 0.99, 100)

        if dataset == 'Train':
            y_true_all = plot_data['Train']['Logistic']['oof_y_true']
        else:
            y_true_all = plot_data['Test']['Logistic']['y_true']
        prevalence = np.mean(y_true_all)

        for name in models_info.keys():
            display_name = 'LR' if name == 'Logistic' else name
            if dataset == 'Train':
                y_true = plot_data['Train'][name]['oof_y_true']
                y_pred_proba = plot_data['Train'][name]['oof_y_pred_proba']
            else:
                y_true = plot_data['Test'][name]['y_true']
                y_pred_proba = plot_data['Test'][name]['y_pred_proba']
            nb = calculate_net_benefit(y_true, y_pred_proba, thresholds)
            plt.plot(thresholds, nb, label=display_name, color=colors_dict[name], lw=3.5)

        nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
        plt.plot(thresholds, nb_all, 'k--', label='Treat All', alpha=0.7, lw=2.0)
        plt.plot(thresholds, np.zeros_like(thresholds), 'k:', label='Treat None', alpha=0.7, lw=2.0)

        plt.ylim(-0.05, prevalence + 0.1)
        plt.xlabel('Threshold Probability', fontsize=24, fontweight='bold')
        plt.ylabel('Net Benefit', fontsize=24, fontweight='bold')
        plt.title(f'Decision Curve Analysis - {dataset}', fontsize=28, fontweight='bold', pad=15)
        plt.legend(loc='upper right', fontsize=18, frameon=True)
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'9.DCA_{dataset}.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, f'9.DCA_{dataset}.pdf'))
        plt.close()

plot_all_curves(plot_data, desktop)

def plot_radar(results_raw, dataset, output_dir):
    metrics = ['AUC', 'PR-AUC', 'Accuracy', 'F1', 'Sensitivity', 'Specificity']
    model_names = list(results_raw.keys())
    data_list = []
    for name in model_names:
        row = [results_raw[name][dataset][m] for m in metrics]
        data_list.append(row)
    data_arr = np.array(data_list)

    angles = np.linspace(0, 2 * pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    ax.tick_params(labelsize=18)
    ax.set_title(f'Radar Chart - {dataset}', fontsize=28, pad=35, fontweight='bold')

    for i, name in enumerate(model_names):
        display_name = 'LR' if name == 'Logistic' else name
        values = data_arr[i].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=3.5, label=display_name, color=colors_dict[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=22, fontweight='bold')
    ax.set_ylim(0, 1)

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.15, 0.5),
        fontsize=16,
        handlelength=1.2,
        handletextpad=0.5,
        labelspacing=0.4,
        frameon=True,
        facecolor='white',
        edgecolor='#CCCCCC'
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'9.Radar_{dataset}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'9.Radar_{dataset}.pdf'), bbox_inches='tight')
    plt.close()

plot_radar(results_raw, 'Train', desktop)
plot_radar(results_raw, 'Test', desktop)

# ----------------------------------------------------------------------
# Step 8: Model Explainability via SHAP (Hold-Out Testing Set)
# ----------------------------------------------------------------------
best_model_name = max(results_raw.keys(), key=lambda k: results_raw[k]['Test']['AUC'])
print(f"\nBest model based on Test AUC: {best_model_name} (AUC={results_raw[best_model_name]['Test']['AUC']:.4f})")
best_model = models[best_model_name]

print('Computing SHAP values...')
masker = shap.maskers.Independent(X_train, max_samples=X_train.shape[0])

if best_model_name == 'NN':
    def predict_nn(x):
        best_model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            logits = best_model(x_tensor)
            return torch.sigmoid(logits).cpu().numpy()
    explainer = shap.Explainer(predict_nn, masker)
else:
    if best_model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.Explainer(best_model, masker)

shap_values = explainer(X_test)

mapping_path = os.path.join(desktop, 'Abbreviations of variables.xlsx')
if os.path.exists(mapping_path):
    mapping_df = pd.read_excel(mapping_path, header=None)
    name_map = dict(zip(mapping_df.iloc[:, 1].astype(str).str.strip(), mapping_df.iloc[:, 0].astype(str).str.strip()))
    feature_names = [name_map.get(str(col).strip(), str(col)) for col in boruta_selected]
else:
    feature_names = boruta_selected

shap_values.feature_names = feature_names
print(f'Feature names mapped: {feature_names[:5]}...')

shap_values.values = np.round(shap_values.values, 4)
if hasattr(shap_values, 'base_values') and shap_values.base_values is not None:
    shap_values.base_values = np.round(shap_values.base_values, 4)
if isinstance(shap_values.data, np.ndarray):
    try:
        shap_values.data = np.round(shap_values.data.astype(float), 4)
    except Exception:
        pass

# Generate patient-level decision path logs
explanation_file = os.path.join(desktop, '3.shap_prediction_explanation.txt')
print(f'Writing individual prediction explanations to: {explanation_file}...')
with open(explanation_file, 'w', encoding='utf-8') as f_out:
    f_out.write("=========================================================================\n")
    f_out.write("       SHAP Individual Patient Risk Attribution Report (Test Set)       \n")
    f_out.write("=========================================================================\n\n")

    num_samples_to_explain = min(10, len(X_test))
    for idx in range(num_samples_to_explain):
        f_out.write(f"--- Sample #{idx + 1} Decision Attribution ---\n")
        base_val = shap_values.base_values[idx] if hasattr(shap_values, 'base_values') else 0.0
        sample_shap = shap_values.values[idx]
        sample_data = shap_values.data[idx]
        final_output = base_val + np.sum(sample_shap)

        f_out.write(f"  * Baseline Expected Value (Base Value): {base_val:.4f}\n")
        f_out.write(f"  * Final Model Prediction Log-odds (f(x)): {final_output:.4f}\n")

        pos_drivers = []
        neg_drivers = []
        for f_idx, f_name in enumerate(feature_names):
            val = sample_data[f_idx]
            contrib = sample_shap[f_idx]
            if contrib > 0:
                pos_drivers.append((f_name, val, contrib))
            else:
                neg_drivers.append((f_name, val, contrib))

        pos_drivers.sort(key=lambda x: x[2], reverse=True)
        neg_drivers.sort(key=lambda x: x[2])

        f_out.write("  * Top Positive Risk Drivers:\n")
        if pos_drivers:
            for name, val, contrib in pos_drivers[:5]:
                f_out.write(f"    - Feature [{name}] = {val:.4f} (Risk elevation, SHAP: +{contrib:.4f})\n")
        else:
            f_out.write("    - No positive contributors\n")

        f_out.write("  * Top Protective Factors (Negative Drivers):\n")
        if neg_drivers:
            for name, val, contrib in neg_drivers[:5]:
                f_out.write(f"    - Feature [{name}] = {val:.4f} (Risk reduction, SHAP: {contrib:.4f})\n")
        else:
            f_out.write("    - No negative contributors\n")
        f_out.write("\n" + "=" * 70 + "\n\n")

print('Individual risk attribution report exported successfully.')

# Standard SHAP Matplotlib style context
shap_rc_params = {
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False
}

with plt.rc_context(shap_rc_params):
    # Bar plot
    print('Generating SHAP feature importance bar plot...')
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title('SHAP Feature Importance (Bar)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(desktop, '9.shap_bar_importance.png'), dpi=300)
    plt.savefig(os.path.join(desktop, '9.shap_bar_importance.pdf'))
    plt.close()

    # Beeswarm plot
    print('Generating SHAP beeswarm summary plot...')
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(desktop, '9.shap_beeswarm.png'), dpi=300)
    plt.savefig(os.path.join(desktop, '9.shap_beeswarm.pdf'))
    plt.close()

    # Heatmap plot
    print('Generating SHAP population heatmap...')
    shap.plots.heatmap(shap_values, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(desktop, '9.shap_heatmap.png'), dpi=300)
    plt.savefig(os.path.join(desktop, '9.shap_heatmap.pdf'))
    plt.close()

    # Force plots for representative individual cases
    print('Generating SHAP force plots for individual samples...')
    for i in range(10):
        plt.clf()
        shap.plots.force(shap_values[i], matplotlib=True, show=False)
        plt.savefig(os.path.join(desktop, f'9.shap_force_sample_{i + 1}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(desktop, f'9.shap_force_sample_{i + 1}.pdf'), bbox_inches='tight')
        plt.close()

print('Individual force plots exported successfully.')
print('\nExecution completed successfully.')
