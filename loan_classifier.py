"""
Loan Status Classification
Logistic Regression vs Decision Tree
Data Mining Assignment - Grand Canyon University
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             classification_report, roc_curve, auc)
import os

os.makedirs('outputs', exist_ok=True)


# LOAD AND PREPROCESS DATA

print("\n  LOAN STATUS CLASSIFIER")
print("  Logistic Regression vs Decision Tree")
print("  ----------\n")

print("  [1] Loading and Preprocessing Data")

df = pd.read_csv('loan_data.csv')
df = df.drop(columns=['Loan_ID'])

print(f"  Dataset shape    : {df.shape}")
print(f"  Approved (Y)     : {(df['Loan_Status'] == 'Y').sum()}")
print(f"  Rejected (N)     : {(df['Loan_Status'] == 'N').sum()}")
print(f"  Missing values   : {df.isnull().sum().sum()}")

# Fill missing values properly
df = df.assign(
    Gender         = df['Gender'].fillna(df['Gender'].mode()[0]),
    Dependents     = df['Dependents'].fillna(df['Dependents'].mode()[0]),
    Self_Employed  = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0]),
    LoanAmount     = df['LoanAmount'].fillna(df['LoanAmount'].median()),
    Loan_Amount_Term = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0]),
    Credit_History = df['Credit_History'].fillna(df['Credit_History'].mode()[0]),
)

print(f"  After cleaning   : {df.isnull().sum().sum()} missing values")

# Encode categorical columns
le = LabelEncoder()
cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area', 'Loan_Status']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])


# PREPARE TRAINING AND TESTING SETS

print("\n  [2] Preparing Training and Testing Sets")

X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler      = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")
print(f"  Features         : {X_train.shape[1]}")


# CLASSIFIER 1 — LOGISTIC REGRESSION

print("\n  [3] Building Classifier 1 — Logistic Regression")

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_std, y_train)
y_pred_lr = lr.predict(X_test_std)
y_prob_lr  = lr.predict_proba(X_test_std)[:, 1]

cm_lr  = confusion_matrix(y_test, y_pred_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)

# Sensitivity and Specificity
tn, fp, fn, tp = cm_lr.ravel()
sens_lr = tp / (tp + fn)   # sensitivity = recall for positive class
spec_lr = tn / (tn + fp)   # specificity = recall for negative class

print(f"  Accuracy         : {acc_lr*100:.1f}%")
print(f"  Sensitivity      : {sens_lr*100:.1f}%")
print(f"  Specificity      : {spec_lr*100:.1f}%")
print(f"\n  Confusion Matrix:")
print(f"    TN={tn}  FP={fp}")
print(f"    FN={fn}  TP={tp}")


# CLASSIFIER 2 — DECISION TREE

print("\n  [4] Building Classifier 2 — Decision Tree")

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_std, y_train)
y_pred_dt = dt.predict(X_test_std)
y_prob_dt  = dt.predict_proba(X_test_std)[:, 1]

cm_dt  = confusion_matrix(y_test, y_pred_dt)
acc_dt = accuracy_score(y_test, y_pred_dt)

tn2, fp2, fn2, tp2 = cm_dt.ravel()
sens_dt = tp2 / (tp2 + fn2)
spec_dt = tn2 / (tn2 + fp2)

print(f"  Accuracy         : {acc_dt*100:.1f}%")
print(f"  Sensitivity      : {sens_dt*100:.1f}%")
print(f"  Specificity      : {spec_dt*100:.1f}%")
print(f"\n  Confusion Matrix:")
print(f"    TN={tn2}  FP={fp2}")
print(f"    FN={fn2}  TP={tp2}")


# PLOT 1 — CONFUSION MATRICES

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, cm, title in zip(axes,
                          [cm_lr, cm_dt],
                          ['Logistic Regression', 'Decision Tree']):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Rejected', 'Approved'])
    ax.set_yticklabels(['Rejected', 'Approved'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix\n{title}')
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14,
                    color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig('outputs/plot1_confusion_matrices.png')
plt.close()
print("\n  Plot saved: plot1_confusion_matrices.png")


# PLOT 2 — ROC CURVES

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
auc_lr = auc(fpr_lr, tpr_lr)
auc_dt = auc(fpr_dt, tpr_dt)

plt.figure(figsize=(7, 5))
plt.plot(fpr_lr, tpr_lr, color='steelblue', linewidth=2,
         label=f'Logistic Regression (AUC = {auc_lr:.3f})')
plt.plot(fpr_dt, tpr_dt, color='tomato', linewidth=2,
         label=f'Decision Tree (AUC = {auc_dt:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/plot2_roc_curves.png')
plt.close()
print("  Plot saved: plot2_roc_curves.png")


# PLOT 3 — METRICS COMPARISON BAR CHART

metrics      = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']
lr_vals      = [acc_lr, sens_lr, spec_lr, auc_lr]
dt_vals      = [acc_dt, sens_dt, spec_dt, auc_dt]
x            = np.arange(len(metrics))
width        = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, lr_vals, width, label='Logistic Regression',
        color='steelblue', edgecolor='white')
plt.bar(x + width/2, dt_vals, width, label='Decision Tree',
        color='tomato', edgecolor='white')
plt.xticks(x, metrics)
plt.ylabel('Score')
plt.title('Classifier Performance Comparison')
plt.legend()
plt.ylim(0, 1.1)
for i, (lv, dv) in enumerate(zip(lr_vals, dt_vals)):
    plt.text(i - width/2, lv + 0.02, f'{lv:.2f}',
             ha='center', fontsize=9)
    plt.text(i + width/2, dv + 0.02, f'{dv:.2f}',
             ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/plot3_metrics_comparison.png')
plt.close()
print("  Plot saved: plot3_metrics_comparison.png")


# FINAL SUMMARY

print("\n  ----------")
print("  FINAL COMPARISON SUMMARY")
print("  ----------")
print(f"  {'Metric':<16} {'Logistic Reg':>14} {'Decision Tree':>14}")
print(f"  {'─'*46}")
print(f"  {'Accuracy':<16} {acc_lr*100:>13.1f}% {acc_dt*100:>13.1f}%")
print(f"  {'Sensitivity':<16} {sens_lr*100:>13.1f}% {sens_dt*100:>13.1f}%")
print(f"  {'Specificity':<16} {spec_lr*100:>13.1f}% {spec_dt*100:>13.1f}%")
print(f"  {'AUC':<16} {auc_lr:>14.3f} {auc_dt:>14.3f}")
print(f"  {'─'*46}")

if auc_lr > auc_dt:
    print(f"\n  Best classifier  : Logistic Regression (AUC={auc_lr:.3f})")
else:
    print(f"\n  Best classifier  : Decision Tree (AUC={auc_dt:.3f})")

print("\n  Done. Plots saved to outputs/ folder.\n")
