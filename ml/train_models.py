#!/usr/bin/env python3
"""
CreditWise Analytics — Production Scikit-Learn Training Pipeline
Trains Decision Tree Classifier and Logistic Regression on
public/assets/aistudio/loan_approval_hybrid_dataset_20711486.csv,
evaluates stratified 80/20 holdout & 5-fold CV metrics, and serializes model bundles
for Python deployment (joblib .pkl + JSON metadata).
"""

import json
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

DATASET_PATH = os.environ.get(
    "CREDITWISE_DATA_PATH",
    "public/assets/aistudio/loan_approval_hybrid_dataset_20711486.csv",
)
ARTIFACT_DIR = os.environ.get("CREDITWISE_ARTIFACT_DIR", "ml/artifacts")

NUM_COLS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Credit_Score",
    "DTI_Ratio",
    "Loan_Amount",
    "Loan_Tenure",
    "Existing_Loans",
    "Previous_Loan_Defaults",
    "Employment_Stability",
    "Monthly_Expenses",
    "Savings_Balance",
    "EMI_to_Income_Ratio",
    "Collateral_Value",
]

CAT_COLS = [
    "Payment_History",
    "Loan_Purpose",
]


def build_preprocessor() -> ColumnTransformer:
    eng_num_cols = NUM_COLS + [
        "Total_Income",
        "Loan_to_Income",
        "Expense_to_Income",
        "Collateral_Coverage",
        "DTI_Ratio_Sq",
        "Credit_Score_Sq",
    ]
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, eng_num_cols),
            ("cat", cat_pipe, CAT_COLS),
        ]
    )


def evaluate_model(name: str, pipe: Pipeline, X_test, y_test):
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    metrics = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds)), 4),
        "recall": round(float(recall_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "roc_points": len(fpr),
    }
    print(f"\n=== {name} ===")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))
    return metrics


def main():
    print(f"Loading dataset from: {DATASET_PATH}")
    raw_df = pd.read_csv(DATASET_PATH)
    labeled_df = raw_df[raw_df["Loan_Status"].isin([0, 1])].copy()
    print(f"Loaded {len(raw_df)} total rows ({len(labeled_df)} labeled).")

    y = labeled_df["Loan_Status"].astype(int)
    X_raw = labeled_df.drop(columns=["Loan_Status"])

    # Stratified 80/20 split prior to computing any preprocessing statistics
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=42
    )

    train_medians = {col: X_train_raw[col].median() for col in NUM_COLS}

    def apply_fe(df_part: pd.DataFrame) -> pd.DataFrame:
        out = df_part.copy()
        for col in NUM_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(train_medians[col])
        out["Total_Income"] = out["Applicant_Income"] + out["Coapplicant_Income"]
        out["Loan_to_Income"] = out["Loan_Amount"] / (out["Total_Income"] + 1.0)
        out["Expense_to_Income"] = out["Monthly_Expenses"] / (out["Total_Income"] + 1.0)
        out["Collateral_Coverage"] = out["Collateral_Value"] / (out["Loan_Amount"] + 1.0)
        out["DTI_Ratio_Sq"] = out["DTI_Ratio"] ** 2
        out["Credit_Score_Sq"] = out["Credit_Score"] ** 2
        return out

    X_train = apply_fe(X_train_raw)
    X_test = apply_fe(X_test_raw)

    # 1. Decision Tree Pipeline
    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=6,
                    min_samples_leaf=20,
                    random_state=42,
                ),
            ),
        ]
    )
    dt_pipeline.fit(X_train, y_train)
    dt_metrics = evaluate_model("Decision Tree Classifier", dt_pipeline, X_test, y_test)

    # 2. Logistic Regression Pipeline
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model("Logistic Regression", lr_pipeline, X_test, y_test)

    # 3. Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_full = apply_fe(X_raw)
    dt_cv = cross_val_score(dt_pipeline, X_full, y, cv=skf, scoring="accuracy")
    lr_cv = cross_val_score(lr_pipeline, X_full, y, cv=skf, scoring="accuracy")
    print(f"5-Fold CV Accuracy | Decision Tree: {dt_cv.mean():.4f} ± {dt_cv.std():.4f}")
    print(f"5-Fold CV Accuracy | Logistic Regression: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

    feature_names = dt_pipeline.named_steps["preprocessor"].get_feature_names_out()
    tree_rules = export_text(
        dt_pipeline.named_steps["classifier"],
        feature_names=list(feature_names),
    )

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(dt_pipeline, os.path.join(ARTIFACT_DIR, "decision_tree_model.pkl"))
    joblib.dump(lr_pipeline, os.path.join(ARTIFACT_DIR, "logistic_regression_model.pkl"))

    metadata = {
        "dataset": DATASET_PATH,
        "decision_tree": dt_metrics,
        "logistic_regression": lr_metrics,
        "cv_5fold": {
            "decision_tree_mean": round(float(dt_cv.mean()), 4),
            "decision_tree_std": round(float(dt_cv.std()), 4),
            "logistic_regression_mean": round(float(lr_cv.mean()), 4),
            "logistic_regression_std": round(float(lr_cv.std()), 4),
        },
        "decision_tree_rules": tree_rules,
    }
    with open(os.path.join(ARTIFACT_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved .pkl model bundles and metadata to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
