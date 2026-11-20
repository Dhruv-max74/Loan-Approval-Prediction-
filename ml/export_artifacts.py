#!/usr/bin/env python3
"""
Deterministic pure-Python ML training & artifact exporter for CreditWise Analytics.
Trains both CART Decision Tree and L2-Regularized Logistic Regression on
public/assets/aistudio/loan_approval_hybrid_dataset_20711486.csv with zero data leakage
and writes src/data/trainedModels.json for the web dashboard and IEEE paper generator.
"""

import csv
import json
import math
import os
import random
import statistics
from collections import Counter

CSV_PATH = "public/assets/aistudio/loan_approval_hybrid_dataset_20711486.csv"
OUT_PATH = "src/data/trainedModels.json"

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

ENGINEERED_COLS = [
    "Total_Income",
    "Loan_to_Income",
    "Expense_to_Income",
    "Collateral_Coverage",
    "DTI_Ratio_Sq",
    "Credit_Score_Sq",
]


def load_rows():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_target(r):
    val = str(r.get("Loan_Status", "")).strip()
    if val in ("1", "1.0", "Yes", "Approved"):
        return 1
    if val in ("0", "0.0", "No", "Rejected"):
        return 0
    return None


def fit_preprocessor(train_raw):
    medians = {
        c: statistics.median([float(r[c]) for r in train_raw if str(r.get(c, "")).strip()])
        for c in NUM_COLS
    }
    modes = {}
    for c in CAT_COLS:
        vals = [r[c].strip() for r in train_raw if str(r.get(c, "")).strip()]
        modes[c] = Counter(vals).most_common(1)[0][0]
    cat_values = {
        c: sorted(list(set((r[c].strip() or modes[c]) for r in train_raw)))
        for c in CAT_COLS
    }
    feature_names = list(NUM_COLS) + list(ENGINEERED_COLS)
    for c in CAT_COLS:
        for cv in cat_values[c][1:]:
            feature_names.append(f"{c}_{cv}")
    return medians, modes, cat_values, feature_names


def featurize_row(r, medians, modes, cat_values):
    d = {
        c: (float(r[c]) if str(r.get(c, "")).strip() else medians[c])
        for c in NUM_COLS
    }
    total_inc = d["Applicant_Income"] + d["Coapplicant_Income"]
    loan_to_inc = d["Loan_Amount"] / (total_inc + 1.0)
    exp_to_inc = d["Monthly_Expenses"] / (total_inc + 1.0)
    collat_cov = d["Collateral_Value"] / (d["Loan_Amount"] + 1.0)
    dti_sq = d["DTI_Ratio"] ** 2
    cs_sq = d["Credit_Score"] ** 2
    vec = [d[c] for c in NUM_COLS] + [
        total_inc,
        loan_to_inc,
        exp_to_inc,
        collat_cov,
        dti_sq,
        cs_sq,
    ]
    for c in CAT_COLS:
        val = str(r.get(c, "")).strip() or modes[c]
        for cv in cat_values[c][1:]:
            vec.append(1.0 if val == cv else 0.0)
    return vec


def fit_scaler(X):
    d = len(X[0])
    n = len(X)
    means = [sum(row[j] for row in X) / n for j in range(d)]
    stds = []
    for j in range(d):
        m = means[j]
        var = sum((row[j] - m) ** 2 for row in X) / n
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def apply_scaler(X, means, stds):
    return [[(row[j] - means[j]) / stds[j] for j in range(len(row))] for row in X]


def train_logistic_regression(
    X_norm, y, lr=0.22, reg=0.01, epochs=18, momentum=0.82, batch_size=512
):
    d = len(X_norm[0])
    n = len(X_norm)
    w = [0.0] * d
    vw = [0.0] * d
    b = 0.0
    vb = 0.0
    indices = list(range(n))
    rng = random.Random(42)
    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            inv_m = 1.0 / (end - start)
            gw = [0.0] * d
            gb = 0.0
            for pos_i in range(start, end):
                idx = indices[pos_i]
                x = X_norm[idx]
                z = b
                for j in range(d):
                    z += w[j] * x[j]
                if z > 30.0:
                    z = 30.0
                elif z < -30.0:
                    z = -30.0
                err = (1.0 / (1.0 + math.exp(-z))) - y[idx]
                gb += err
                for j in range(d):
                    gw[j] += err * x[j]
            vb = momentum * vb + lr * (gb * inv_m)
            b -= vb
            for j in range(d):
                vw[j] = momentum * vw[j] + lr * (gw[j] * inv_m + reg * w[j])
                w[j] -= vw[j]
    return w, b


def predict_lr_probs(X_norm, w, b):
    probs = []
    d = len(w)
    for x in X_norm:
        z = b
        for j in range(d):
            z += w[j] * x[j]
        if z > 30.0:
            z = 30.0
        elif z < -30.0:
            z = -30.0
        probs.append(1.0 / (1.0 + math.exp(-z)))
    return probs


def gini_from_counts(pos, n):
    if n == 0:
        return 0.0
    p1 = pos / n
    return 1.0 - p1 * p1 - (1.0 - p1) * (1.0 - p1)


def build_tree(X, y, feature_names, depth=0, max_depth=6, min_leaf=20, node_id_counter=None):
    if node_id_counter is None:
        node_id_counter = [0]
    nid = node_id_counter[0]
    node_id_counter[0] += 1

    n = len(y)
    pos = sum(y)
    neg = n - pos
    p1 = pos / n if n else 0.0
    g_node = gini_from_counts(pos, n)

    if depth >= max_depth or n < 2 * min_leaf or pos == 0 or pos == n:
        return {
            "id": nid,
            "leaf": True,
            "depth": depth,
            "samples": n,
            "negCount": neg,
            "posCount": pos,
            "gini": round(g_node, 5),
            "prob": round(p1, 5),
            "pred": 1 if p1 >= 0.5 else 0,
        }

    best_gain = 0.0
    best_feat = None
    best_thresh = None
    num_features = len(X[0])

    for j in range(num_features):
        pairs = sorted(((X[i][j], y[i]) for i in range(n)), key=lambda t: t[0])
        if pairs[0][0] == pairs[-1][0]:
            continue

        step = max(1, n // 25)
        last_checked = -step
        l_pos = 0
        for idx in range(n - 1):
            l_pos += pairs[idx][1]
            l_cnt = idx + 1
            r_cnt = n - l_cnt
            if pairs[idx][0] == pairs[idx + 1][0]:
                continue
            if l_cnt < min_leaf or r_cnt < min_leaf:
                continue
            if idx - last_checked < step:
                continue
            last_checked = idx
            r_pos = pos - l_pos
            g_l = 1.0 - (l_pos / l_cnt) ** 2 - ((l_cnt - l_pos) / l_cnt) ** 2
            g_r = 1.0 - (r_pos / r_cnt) ** 2 - ((r_cnt - r_pos) / r_cnt) ** 2
            gain = g_node - (l_cnt / n) * g_l - (r_cnt / n) * g_r
            if gain > best_gain:
                best_gain = gain
                best_feat = j
                best_thresh = (pairs[idx][0] + pairs[idx + 1][0]) / 2.0

    if best_feat is None:
        return {
            "id": nid,
            "leaf": True,
            "depth": depth,
            "samples": n,
            "negCount": neg,
            "posCount": pos,
            "gini": round(g_node, 5),
            "prob": round(p1, 5),
            "pred": 1 if p1 >= 0.5 else 0,
        }

    idx_l = [i for i in range(n) if X[i][best_feat] <= best_thresh]
    idx_r = [i for i in range(n) if X[i][best_feat] > best_thresh]

    return {
        "id": nid,
        "leaf": False,
        "depth": depth,
        "featureIndex": best_feat,
        "featureName": feature_names[best_feat],
        "threshold": round(best_thresh, 5),
        "gain": round(best_gain, 5),
        "weightedImpurityDecrease": round((n * best_gain), 5),
        "samples": n,
        "negCount": neg,
        "posCount": pos,
        "gini": round(g_node, 5),
        "prob": round(p1, 5),
        "pred": 1 if p1 >= 0.5 else 0,
        "left": build_tree(
            [X[i] for i in idx_l],
            [y[i] for i in idx_l],
            feature_names,
            depth + 1,
            max_depth,
            min_leaf,
            node_id_counter,
        ),
        "right": build_tree(
            [X[i] for i in idx_r],
            [y[i] for i in idx_r],
            feature_names,
            depth + 1,
            max_depth,
            min_leaf,
            node_id_counter,
        ),
    }


def predict_tree_prob(node, x, max_d=None):
    while not node["leaf"] and (max_d is None or node["depth"] < max_d):
        if x[node["featureIndex"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    return node["prob"]


def compute_tree_importances(node, feature_names):
    totals = [0.0] * len(feature_names)

    def walk(nd):
        if nd["leaf"]:
            return
        totals[nd["featureIndex"]] += nd["weightedImpurityDecrease"]
        walk(nd["left"])
        walk(nd["right"])

    walk(node)
    s = sum(totals) or 1.0
    return [round(v / s, 5) for v in totals]


def compute_metrics(probs, y_true):
    preds = [1 if p >= 0.5 else 0 for p in probs]
    tp = sum(1 for p, y in zip(preds, y_true) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, y_true) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, y_true) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, y_true) if p == 0 and y == 1)
    n = len(y_true)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    eps = 1e-9
    logloss = -sum(
        y * math.log(max(eps, min(1 - eps, p))) + (1 - y) * math.log(max(eps, min(1 - eps, 1 - p)))
        for p, y in zip(probs, y_true)
    ) / n
    brier = sum((p - y) ** 2 for p, y in zip(probs, y_true)) / n

    # Fast ROC curve via sorting pairs by score descending
    paired = sorted(zip(probs, y_true), key=lambda t: t[0], reverse=True)
    pos_total = sum(y_true) or 1
    neg_total = (n - sum(y_true)) or 1
    roc_points = [{"threshold": 1.01, "fpr": 0.0, "tpr": 0.0}]
    tp_c = 0
    fp_c = 0
    for idx, (prob, yt) in enumerate(paired):
        if yt == 1:
            tp_c += 1
        else:
            fp_c += 1
        if idx == n - 1 or prob != paired[idx + 1][0]:
            roc_points.append({
                "threshold": round(prob, 4),
                "fpr": round(fp_c / neg_total, 4),
                "tpr": round(tp_c / pos_total, 4),
            })

    roc_sorted = sorted(roc_points, key=lambda pt: (pt["fpr"], pt["tpr"]))
    auc_val = 0.0
    for i in range(1, len(roc_sorted)):
        dx = roc_sorted[i]["fpr"] - roc_sorted[i - 1]["fpr"]
        avg_y = (roc_sorted[i]["tpr"] + roc_sorted[i - 1]["tpr"]) / 2.0
        auc_val += dx * avg_y

    # Downsample ROC curve cleanly to ~100 points for compact JSON
    step = max(1, len(roc_sorted) // 100)
    compact_roc = [roc_sorted[0]]
    for i in range(step, len(roc_sorted) - 1, step):
        pt = roc_sorted[i]
        if pt["fpr"] != compact_roc[-1]["fpr"] or pt["tpr"] != compact_roc[-1]["tpr"]:
            compact_roc.append(pt)
    if roc_sorted[-1] != compact_roc[-1]:
        compact_roc.append(roc_sorted[-1])

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "rocAuc": round(auc_val, 4),
        "logLoss": round(logloss, 4),
        "brierScore": round(brier, 4),
        "confusionMatrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "rocCurve": compact_roc,
    }


def stratified_kfold_cv(labeled_rows, k=5):
    yes_r = [r for r in labeled_rows if parse_target(r) == 1]
    no_r = [r for r in labeled_rows if parse_target(r) == 0]
    random.seed(42)
    random.shuffle(yes_r)
    random.shuffle(no_r)

    lr_accs, dt_accs = [], []
    lr_f1s, dt_f1s = [], []

    for fold in range(k):
        test_fold = yes_r[fold::k] + no_r[fold::k]
        train_fold = [r for i, r in enumerate(yes_r) if i % k != fold] + [
            r for i, r in enumerate(no_r) if i % k != fold
        ]
        medians, modes, cat_values, feature_names = fit_preprocessor(train_fold)
        X_tr = [featurize_row(r, medians, modes, cat_values) for r in train_fold]
        y_tr = [parse_target(r) for r in train_fold]
        X_te = [featurize_row(r, medians, modes, cat_values) for r in test_fold]
        y_te = [parse_target(r) for r in test_fold]

        means, stds = fit_scaler(X_tr)
        w, b = train_logistic_regression(apply_scaler(X_tr, means, stds), y_tr)
        lr_m = compute_metrics(predict_lr_probs(apply_scaler(X_te, means, stds), w, b), y_te)
        lr_accs.append(lr_m["accuracy"])
        lr_f1s.append(lr_m["f1"])

        tree = build_tree(X_tr, y_tr, feature_names, max_depth=6, min_leaf=20)
        dt_m = compute_metrics([predict_tree_prob(tree, x) for x in X_te], y_te)
        dt_accs.append(dt_m["accuracy"])
        dt_f1s.append(dt_m["f1"])

    return {
        "folds": k,
        "logisticRegression": {
            "foldAccuracies": lr_accs,
            "meanAccuracy": round(statistics.mean(lr_accs), 4),
            "stdAccuracy": round(statistics.pstdev(lr_accs), 4),
            "meanF1": round(statistics.mean(lr_f1s), 4),
        },
        "decisionTree": {
            "foldAccuracies": dt_accs,
            "meanAccuracy": round(statistics.mean(dt_accs), 4),
            "stdAccuracy": round(statistics.pstdev(dt_accs), 4),
            "meanF1": round(statistics.mean(dt_f1s), 4),
        },
    }


def build_eda_stats(raw_rows, labeled_rows):
    # 1. Credit Score Bins Trend (hybrid dataset spans 341..889)
    cs_bins = [
        (300, 580, "<580"),
        (580, 640, "580-639"),
        (640, 700, "640-699"),
        (700, 760, "700-759"),
        (760, 900, "760+"),
    ]
    credit_score_trend = []
    for lo, hi, label in cs_bins:
        sub = [
            r for r in labeled_rows
            if r["Credit_Score"].strip() and lo <= float(r["Credit_Score"]) < hi
        ]
        app = sum(1 for r in sub if parse_target(r) == 1)
        rej = len(sub) - app
        credit_score_trend.append({
            "band": label,
            "total": len(sub),
            "approved": app,
            "rejected": rej,
            "approvalRate": round((app / len(sub) * 100) if sub else 0.0, 1),
        })

    # 2. DTI Ratio Trend (hybrid dataset stored in percentage points: 0..62%)
    dti_bins = [
        (0, 5, "0-5%"),
        (5, 10, "5-10%"),
        (10, 20, "10-20%"),
        (20, 30, "20-30%"),
        (30, 70, "30%+"),
    ]
    dti_trend = []
    for lo, hi, label in dti_bins:
        sub = [
            r for r in labeled_rows
            if r["DTI_Ratio"].strip() and lo <= float(r["DTI_Ratio"]) < hi
        ]
        app = sum(1 for r in sub if parse_target(r) == 1)
        rej = len(sub) - app
        dti_trend.append({
            "band": label,
            "total": len(sub),
            "approved": app,
            "rejected": rej,
            "approvalRate": round((app / len(sub) * 100) if sub else 0.0, 1),
        })

    # 3. Total Income Bracket Trend
    inc_bins = [
        (0, 30000, "<30k"),
        (30000, 50000, "30k-50k"),
        (50000, 75000, "50k-75k"),
        (75000, 110000, "75k-110k"),
        (110000, 2000000, "110k+"),
    ]
    income_trend = []
    for lo, hi, label in inc_bins:
        sub = []
        for r in labeled_rows:
            if r["Applicant_Income"].strip() and r["Coapplicant_Income"].strip():
                tot = float(r["Applicant_Income"]) + float(r["Coapplicant_Income"])
                if lo <= tot < hi:
                    sub.append(r)
        app = sum(1 for r in sub if parse_target(r) == 1)
        avg_loan = statistics.mean([float(r["Loan_Amount"]) for r in sub if r["Loan_Amount"].strip()]) if sub else 0
        income_trend.append({
            "bracket": label,
            "total": len(sub),
            "approved": app,
            "rejected": len(sub) - app,
            "approvalRate": round((app / len(sub) * 100) if sub else 0.0, 1),
            "avgLoanAmount": round(avg_loan, 0),
        })

    # 4. Categorical & Discrete Ordinal Breakdowns
    cat_breakdowns = {}
    for col in ["Payment_History", "Loan_Purpose", "Previous_Loan_Defaults", "Existing_Loans"]:
        groups = sorted(set(r[col].strip() for r in labeled_rows if r[col].strip()))
        items = []
        for g in groups:
            sub = [r for r in labeled_rows if r[col].strip() == g]
            app = sum(1 for r in sub if parse_target(r) == 1)
            label = (
                f"{g} defaults" if col == "Previous_Loan_Defaults"
                else (f"{g} loans" if col == "Existing_Loans" else g)
            )
            items.append({
                "category": label,
                "total": len(sub),
                "approved": app,
                "rejected": len(sub) - app,
                "approvalRate": round((app / len(sub) * 100) if sub else 0.0, 1),
            })
        cat_breakdowns[col] = items

    # 5. Feature comparison table (Approved vs Rejected means)
    feature_means = []
    for c in NUM_COLS:
        y_vals = [float(r[c]) for r in labeled_rows if parse_target(r) == 1 and r[c].strip()]
        n_vals = [float(r[c]) for r in labeled_rows if parse_target(r) == 0 and r[c].strip()]
        feature_means.append({
            "feature": c,
            "approvedMean": round(statistics.mean(y_vals), 2),
            "rejectedMean": round(statistics.mean(n_vals), 2),
            "overallMean": round(statistics.mean(y_vals + n_vals), 2),
        })

    return {
        "totalRecords": len(raw_rows),
        "labeledRecords": len(labeled_rows),
        "unlabeledRecords": len(raw_rows) - len(labeled_rows),
        "approvedCount": sum(1 for r in labeled_rows if parse_target(r) == 1),
        "rejectedCount": sum(1 for r in labeled_rows if parse_target(r) == 0),
        "creditScoreTrend": credit_score_trend,
        "dtiTrend": dti_trend,
        "incomeTrend": income_trend,
        "categoricalBreakdowns": cat_breakdowns,
        "featureMeans": feature_means,
    }


def main():
    raw_rows = load_rows()
    labeled_rows = [r for r in raw_rows if parse_target(r) is not None]

    random.seed(42)
    yes_rows = [r for r in labeled_rows if parse_target(r) == 1]
    no_rows = [r for r in labeled_rows if parse_target(r) == 0]
    random.shuffle(yes_rows)
    random.shuffle(no_rows)

    split_y = int(0.8 * len(yes_rows))
    split_n = int(0.8 * len(no_rows))
    train_raw = yes_rows[:split_y] + no_rows[:split_n]
    test_raw = yes_rows[split_y:] + no_rows[split_n:]

    medians, modes, cat_values, feature_names = fit_preprocessor(train_raw)

    X_train = [featurize_row(r, medians, modes, cat_values) for r in train_raw]
    y_train = [parse_target(r) for r in train_raw]
    X_test = [featurize_row(r, medians, modes, cat_values) for r in test_raw]
    y_test = [parse_target(r) for r in test_raw]

    means, stds = fit_scaler(X_train)
    X_train_norm = apply_scaler(X_train, means, stds)
    X_test_norm = apply_scaler(X_test, means, stds)

    # Train Logistic Regression
    lr_weights, lr_bias = train_logistic_regression(X_train_norm, y_train)
    lr_test_probs = predict_lr_probs(X_test_norm, lr_weights, lr_bias)
    lr_train_probs = predict_lr_probs(X_train_norm, lr_weights, lr_bias)
    lr_metrics = compute_metrics(lr_test_probs, y_test)
    lr_train_metrics = compute_metrics(lr_train_probs, y_train)

    # Train full depth-8 tree once and extract depth-6 production tree + depth curve 1..8
    dt_full = build_tree(X_train, y_train, feature_names, max_depth=8, min_leaf=20)

    def prune_to_depth(nd, max_d):
        if nd["leaf"] or nd["depth"] >= max_d:
            return {
                "id": nd["id"],
                "leaf": True,
                "depth": nd["depth"],
                "samples": nd["samples"],
                "negCount": nd["negCount"],
                "posCount": nd["posCount"],
                "gini": nd["gini"],
                "prob": nd["prob"],
                "pred": nd["pred"],
            }
        return {
            **nd,
            "left": prune_to_depth(nd["left"], max_d),
            "right": prune_to_depth(nd["right"], max_d),
        }

    dt_tree = prune_to_depth(dt_full, 6)
    dt_test_probs = [predict_tree_prob(dt_tree, x) for x in X_test]
    dt_train_probs = [predict_tree_prob(dt_tree, x) for x in X_train]
    dt_metrics = compute_metrics(dt_test_probs, y_test)
    dt_train_metrics = compute_metrics(dt_train_probs, y_train)
    dt_importances = compute_tree_importances(dt_tree, feature_names)

    # 5-fold Stratified Cross Validation
    cv_results = stratified_kfold_cv(labeled_rows, k=5)

    # Depth sensitivity curve for Decision Tree (max_depth 1..8) evaluated from single tree
    depth_curve = []
    for md in range(1, 9):
        tr_m = compute_metrics([predict_tree_prob(dt_full, x, md) for x in X_train], y_train)
        te_m = compute_metrics([predict_tree_prob(dt_full, x, md) for x in X_test], y_test)
        depth_curve.append({
            "maxDepth": md,
            "trainAccuracy": round(tr_m["accuracy"] * 100, 2),
            "testAccuracy": round(te_m["accuracy"] * 100, 2),
            "testF1": round(te_m["f1"] * 100, 2),
        })

    # Score all 10,000 rows for Overview & Trends cohort exploration
    scored_records = []
    for idx, r in enumerate(raw_rows):
        vec = featurize_row(r, medians, modes, cat_values)
        vec_norm = [(vec[j] - means[j]) / stds[j] for j in range(len(vec))]
        z = lr_bias + sum(wj * xj for wj, xj in zip(lr_weights, vec_norm))
        z = max(-30.0, min(30.0, z))
        lr_prob = 1.0 / (1.0 + math.exp(-z))
        dt_prob = predict_tree_prob(dt_tree, vec)

        ai = float(r["Applicant_Income"])
        ci = float(r["Coapplicant_Income"])
        target_val = parse_target(r)

        scored_records.append({
            "rowIndex": idx + 1,
            "applicantId": f"HYB-{idx+1:05d}",
            "applicantIncome": round(ai, 0),
            "coapplicantIncome": round(ci, 0),
            "totalIncome": round(ai + ci, 0),
            "creditScore": int(round(float(r["Credit_Score"]))),
            "dtiRatio": round(float(r["DTI_Ratio"]), 2),
            "loanAmount": round(float(r["Loan_Amount"]), 0),
            "loanTenure": int(round(float(r["Loan_Tenure"]))),
            "existingLoans": int(round(float(r["Existing_Loans"]))),
            "previousLoanDefaults": int(round(float(r["Previous_Loan_Defaults"]))),
            "paymentHistory": r["Payment_History"].strip(),
            "employmentStability": round(float(r["Employment_Stability"]), 1),
            "monthlyExpenses": round(float(r["Monthly_Expenses"]), 0),
            "savingsBalance": round(float(r["Savings_Balance"]), 0),
            "emiToIncomeRatio": round(float(r["EMI_to_Income_Ratio"]), 2),
            "loanPurpose": r["Loan_Purpose"].strip(),
            "collateralValue": round(float(r["Collateral_Value"]), 0),
            "actualApproved": "Yes" if target_val == 1 else ("No" if target_val == 0 else "Unlabeled"),
            "dtProb": round(dt_prob, 4),
            "dtPred": "Yes" if dt_prob >= 0.5 else "No",
            "lrProb": round(lr_prob, 4),
            "lrPred": "Yes" if lr_prob >= 0.5 else "No",
        })

    lr_feature_weights = sorted(
        [
            {
                "feature": feature_names[j],
                "coefficient": round(lr_weights[j], 4),
                "absMagnitude": round(abs(lr_weights[j]), 4),
                "oddsRatio": round(math.exp(max(-10.0, min(10.0, lr_weights[j]))), 4),
            }
            for j in range(len(feature_names))
        ],
        key=lambda x: x["absMagnitude"],
        reverse=True,
    )

    dt_feature_importances = sorted(
        [
            {
                "feature": feature_names[j],
                "importance": dt_importances[j],
            }
            for j in range(len(feature_names))
            if dt_importances[j] > 0
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    artifact = {
        "datasetName": "loan_approval_hybrid_dataset_20711486.csv",
        "datasetSummary": build_eda_stats(raw_rows, labeled_rows),
        "splitInfo": {
            "trainCount": len(train_raw),
            "testCount": len(test_raw),
            "splitRatio": "80/20 Stratified",
            "randomState": 42,
        },
        "preprocessing": {
            "numericCols": NUM_COLS,
            "categoricalCols": CAT_COLS,
            "engineeredCols": ENGINEERED_COLS,
            "medians": {k: round(v, 4) for k, v in medians.items()},
            "modes": modes,
            "catValues": cat_values,
            "featureNames": feature_names,
            "scalerMeans": [round(v, 6) for v in means],
            "scalerStds": [round(v, 6) for v in stds],
        },
        "models": {
            "decisionTree": {
                "name": "Decision Tree Classifier",
                "hyperparameters": {
                    "criterion": "gini",
                    "maxDepth": 6,
                    "minSamplesLeaf": 20,
                    "randomState": 42,
                },
                "trainAccuracy": dt_train_metrics["accuracy"],
                "metrics": dt_metrics,
                "featureImportances": dt_feature_importances,
                "tree": dt_tree,
                "depthCurve": depth_curve,
            },
            "logisticRegression": {
                "name": "Logistic Regression",
                "hyperparameters": {
                    "penalty": "L2",
                    "regularizationLambda": 0.01,
                    "learningRate": 0.22,
                    "epochs": 18,
                },
                "trainAccuracy": lr_train_metrics["accuracy"],
                "metrics": lr_metrics,
                "bias": round(lr_bias, 6),
                "weights": [round(w, 6) for w in lr_weights],
                "featureCoefficients": lr_feature_weights,
            },
        },
        "crossValidation": cv_results,
        "applicants": scored_records,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f)

    print("Exported artifacts to", OUT_PATH)
    print("Decision Tree Test Accuracy:", dt_metrics["accuracy"], "F1:", dt_metrics["f1"], "ROC-AUC:", dt_metrics["rocAuc"])
    print("Logistic Regression Test Accuracy:", lr_metrics["accuracy"], "F1:", lr_metrics["f1"], "ROC-AUC:", lr_metrics["rocAuc"])
    print("5-Fold CV DT:", cv_results["decisionTree"]["meanAccuracy"], "LR:", cv_results["logisticRegression"]["meanAccuracy"])


if __name__ == "__main__":
    main()
