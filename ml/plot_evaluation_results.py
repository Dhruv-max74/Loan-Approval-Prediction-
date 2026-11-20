#!/usr/bin/env python3
"""
Jupyter Notebook / Matplotlib & Seaborn Script to generate:
1. Confusion Matrix Heatmaps (YlGnBu colormap with TP, FP, FN, TN labels)
2. Accuracy, Precision, Recall, and F1-score bar charts for:
   - Decision Tree Classifier
   - Logistic Regression
Trained exclusively on public/assets/aistudio/loan_approval_hybrid_dataset_20711486.csv.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATASET_PATH = "public/assets/aistudio/loan_approval_hybrid_dataset_20711486.csv"
OUT_DIR = "public/charts"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load & prepare dataset
df = pd.read_csv(DATASET_PATH)
y = df["Loan_Status"].astype(int)
X = df.drop(columns=["Loan_Status"]).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

NUM_COLS = [
    "Applicant_Income", "Coapplicant_Income", "Credit_Score", "DTI_Ratio",
    "Loan_Amount", "Loan_Tenure", "Existing_Loans", "Previous_Loan_Defaults",
    "Employment_Stability", "Monthly_Expenses", "Savings_Balance",
    "EMI_to_Income_Ratio", "Collateral_Value",
]
CAT_COLS = ["Payment_History", "Loan_Purpose"]

def add_features(frame):
    out = frame.copy()
    out["Total_Income"] = out["Applicant_Income"] + out["Coapplicant_Income"]
    out["Loan_to_Income"] = out["Loan_Amount"] / (out["Total_Income"] + 1.0)
    out["Expense_to_Income"] = out["Monthly_Expenses"] / (out["Total_Income"] + 1.0)
    out["Collateral_Coverage"] = out["Collateral_Value"] / (out["Loan_Amount"] + 1.0)
    out["DTI_Ratio_Sq"] = out["DTI_Ratio"] ** 2
    out["Credit_Score_Sq"] = out["Credit_Score"] ** 2
    return out

X_train = add_features(X_train)
X_test = add_features(X_test)

ENG_NUM_COLS = NUM_COLS + [
    "Total_Income", "Loan_to_Income", "Expense_to_Income",
    "Collateral_Coverage", "DTI_Ratio_Sq", "Credit_Score_Sq",
]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), ENG_NUM_COLS),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first"))]), CAT_COLS),
])

models = {
    "Decision Tree": Pipeline([
        ("prep", preprocessor),
        ("clf", DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_leaf=20, random_state=42)),
    ]),
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=1.0, penalty="l2", max_iter=2000, random_state=42)),
    ]),
}

def plot_confusion_heatmap(cm, title, filename):
    # Layout matching TP / FN on Positive row, FP / TN on Negative row
    tn, fp, fn, tp = cm.ravel()
    matrix = np.array([[tp, fn], [fp, tn]])
    labels = np.array([[f"TP: {tp}", f"FN: {fn}"], [f"FP: {fp}", f"TN: {tn}"]])

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap="YlGnBu",
        xticklabels=["Positive", "Negative"],
        yticklabels=["Positive", "Negative"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(title, fontsize=15, pad=14)
    plt.xlabel("Predicted labels", fontsize=12)
    plt.ylabel("True labels", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=180)
    plt.show()

def plot_metrics_bar(acc, prec, rec, f1, title, filename):
    names = ["Accuracy", "Precision", "Recall", "F1-score"]
    vals = [acc * 100, prec * 100, rec * 100, f1 * 100]
    colors = ["#1d91c0", "#225ea8", "#081d58", "#41b6c4"]

    plt.figure(figsize=(8.5, 5.2))
    bars = plt.bar(names, vals, color=colors, width=0.52)
    plt.ylim(0, 105)
    plt.ylabel("Score (%)", fontsize=12)
    plt.title(title, fontsize=14, pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.2f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=180)
    plt.show()

for idx, (name, pipe) in enumerate(models.items(), start=1):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"\n{idx}. {name} Results")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-score:  {f1*100:.2f}%")

    slug = name.lower().replace(" ", "_")
    plot_confusion_heatmap(cm, f"Confusion Matrix Heatmap — {name}", f"{slug}_confusion_matrix.png")
    plot_metrics_bar(acc, prec, rec, f1, f"{idx}. {name} Results (Accuracy, Precision, Recall, F1-score)", f"{slug}_metrics.png")
