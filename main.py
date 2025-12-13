#!/usr/bin/env python
# coding: utf-8

# ===============================
# 0. Reduce TensorFlow log noise
# ===============================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # 0=all, 1=INFO, 2=WARNING+ERROR, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # disable oneDNN message + small numeric diffs

# ===============================
# 1. Importing Libraries & Seeds
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

print("RUNNING UPDATED SCRIPT ✅")

# =====================================
# 2. Dataset Loading and Preprocessing
# =====================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# '?' indicates missing values
data = pd.read_csv(url, names=column_names, na_values='?')

# Drop missing
data.dropna(inplace=True)

# Convert to numeric (important because '?' -> NaN can keep columns as object)
for c in column_names:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data.dropna(inplace=True)

# Binary target: 0 = no disease; 1-4 = disease -> 1
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

print("Data Shape:", data.shape)
print("Class distribution:\n", data['target'].value_counts())
print("\nFirst few rows:")
print(data.head().to_string(index=False))

# ==========================
# 3. Exploratory Data Analysis
# ==========================
print("\nMissing Values:")
print(data.isnull().sum())

print("\nSummary statistics:")
print(data.describe().to_string())

# Correlation matrix
correlation_matrix = data.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

print("\nCorrelation with target variable:")
for col in correlation_matrix.columns:
    if col != 'target':
        print(f"{col}: {correlation_matrix.loc['target', col]:.2f}")

# Boxplots
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
cols = 3
rows = math.ceil(len(numerical_features) / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()
for i, col in enumerate(numerical_features):
    sns.boxplot(x='target', y=col, data=data, ax=axes[i])
    axes[i].set_title(f'{col} vs. Target')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Countplots
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
cols = 3
rows = math.ceil(len(categorical_features) / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()
for i, col in enumerate(categorical_features):
    sns.countplot(x=col, hue='target', data=data, ax=axes[i])
    axes[i].set_title(f'Countplot of {col} vs. Target')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# ==========================================
# 5. Feature/Target Split and Scaling
# ==========================================
X = data.drop('target', axis=1)
y = data['target'].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =================================================
# 6. Helper Function to Calculate Metrics Manually
# =================================================
def calculate_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    tpr = recall
    fpr = 1 - specificity
    tnr = specificity
    fnr = 1 - recall

    tss = tpr - fpr

    exp_acc = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total * total) if total else 0
    hss = (accuracy - exp_acc) / (1 - exp_acc) if (1 - exp_acc) != 0 else 0

    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1,
        'TPR': tpr,
        'FPR': fpr,
        'TNR': tnr,
        'FNR': fnr,
        'TSS': tss,
        'HSS': hss
    }

# =====================================================
# 7. 10-Fold Cross Validation: RF, SVM, LSTM
# =====================================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)

rf_results, svm_results, lstm_results = [], [], []
fold_number = 0

for train_idx, test_idx in kf.split(X_scaled):
    fold_number += 1
    print(f"\n=== Fold {fold_number} ===")

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # LSTM expects 3D: (samples, timesteps, features)
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

    # -------- Random Forest --------
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf, labels=[0, 1]).ravel()
    m_rf = calculate_metrics(tp, tn, fp, fn)
    m_rf['Fold'] = fold_number
    rf_results.append(m_rf)

    # -------- SVM --------
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svm, labels=[0, 1]).ravel()
    m_svm = calculate_metrics(tp, tn, fp, fn)
    m_svm['Fold'] = fold_number
    svm_results.append(m_svm)

    # -------- LSTM --------
    lstm_model = Sequential([
        LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    lstm_model.fit(
        X_train_lstm, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred_proba_lstm = lstm_model.predict(X_test_lstm, verbose=0)
    y_pred_lstm = (y_pred_proba_lstm > 0.5).astype(int).reshape(-1)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lstm, labels=[0, 1]).ravel()
    m_lstm = calculate_metrics(tp, tn, fp, fn)
    m_lstm['Fold'] = fold_number
    lstm_results.append(m_lstm)

    tf.keras.backend.clear_session()

print("\nCross Validation Complete.")

# ==============================================
# 8. Convert Results to DataFrames and Averages
# ==============================================
rf_df = pd.DataFrame(rf_results)
svm_df = pd.DataFrame(svm_results)
lstm_df = pd.DataFrame(lstm_results)

rf_mean   = rf_df.drop(columns=['Fold']).mean(numeric_only=True)
svm_mean  = svm_df.drop(columns=['Fold']).mean(numeric_only=True)
lstm_mean = lstm_df.drop(columns=['Fold']).mean(numeric_only=True)

mean_results_df = pd.DataFrame({
    'Random Forest': rf_mean,
    'SVM': svm_mean,
    'LSTM': lstm_mean
})

print("\n==== Per-Fold Random Forest Metrics ====")
print(rf_df.to_string(index=False))

print("\n==== Per-Fold SVM Metrics ====")
print(svm_df.to_string(index=False))

print("\n==== Per-Fold LSTM Metrics ====")
print(lstm_df.to_string(index=False))

print("\n==== Average Metrics across 10 folds ====")
print(mean_results_df.to_string())

# =====================================
# 9. Visualizing Key Metrics
# =====================================
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'TSS', 'HSS']

mean_results_df.loc[metrics_to_plot].plot(kind='bar', figsize=(10, 6))
plt.title("Comparison of Classification Algorithms (Average over 10 folds)")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()

for metric in metrics_to_plot:
    plt.figure(figsize=(8, 5))
    data_for_box = [rf_df[metric], svm_df[metric], lstm_df[metric]]
    plt.boxplot(data_for_box, labels=['Random Forest', 'SVM', 'LSTM'])
    plt.title(f"{metric} Across 10 Folds")
    plt.ylabel(metric)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
