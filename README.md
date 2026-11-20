# CreditWise Analytics — Dual-Model Credit Scoring & Risk Intelligence

A dark-mode machine learning credit risk workbench and research repository trained exclusively on the **Hybrid Consumer Loan Approval Dataset** (`10,000` applications, `15` predictors + binary `Loan_Status` target). The system trains and compares two interpretable supervised classifiers—**Decision Tree (CART)** and **L2-Regularized Logistic Regression**—with zero-leakage preprocessing, switchable dual-model inference, interactive trend analysis charts, cashflow & area-based concessional EMI calculators, and an **IEEE two-column conference paper (`.docx`)**.

---

## Repository Structure

```text
├── ml/
│   ├── train_models.py               # Production Scikit-Learn training pipeline (exports .pkl bundles)
│   ├── export_artifacts.py           # Deterministic zero-dependency artifact exporter for web inference
│   ├── generate_ieee_paper.mjs       # Node.js IEEE two-column .docx manuscript generator
│   ├── generate_ieee_paper.py        # Python-docx IEEE manuscript generator
│   └── requirements.txt              # Python ML & paper dependencies
├── public/
│   ├── assets/aistudio/
│   │   └── loan_approval_hybrid_dataset_20711486.csv   # Source 10,000-row hybrid loan dataset
│   └── CreditWise_IEEE_Conference_Paper.docx           # Generated IEEE two-column research paper
├── src/
│   ├── data/trainedModels.json       # Trained Decision Tree, Logistic Regression, ROC, CV, & EDA metrics
│   ├── utils/inference.ts            # Exact Decision Tree path traversal & Logistic Regression log-odds engine
│   ├── components/
│   │   ├── PredictorView.tsx         # Dashboard: Loan Approval Predictor + Concessional EMI Breakdown
│   │   ├── OverviewTrendsView.tsx    # Overview & Trends: 10,000-record portfolio analytics
│   │   └── ModelBenchmarksView.tsx   # Model Benchmarks: ROC, confusion matrices, CV & CART splits
│   ├── App.tsx                       # Main application workspace (3-item left sidebar)
│   └── index.css                     # Tailwind v4 dark theme & typography configuration
├── index.html
├── package.json
└── README.md
```

---

## 1. Python Machine Learning Training Pipeline

### Setup & Train with Scikit-Learn
```bash
python3 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -r ml/requirements.txt

# Train Decision Tree & Logistic Regression models and serialize .pkl bundles
python ml/train_models.py
```

### Pipeline Architecture (`ml/train_models.py`)
1. **Target Definition**: Uses binary `Loan_Status` (`1` = Approved, `0` = Rejected) across all `10,000` records (`6,555` Approved / `3,445` Rejected).
2. **Stratified Holdout Split**: `80/20` stratified split (`8,000` train / `2,000` test, `random_state=42`) performed **prior** to computing any imputation or scaling statistics to prevent data leakage.
3. **Zero-Leakage Preprocessing**:
   - Median imputation + `StandardScaler` for 13 numerical features (`Applicant_Income`, `Coapplicant_Income`, `Credit_Score`, `DTI_Ratio`, `Loan_Amount`, `Loan_Tenure`, `Existing_Loans`, `Previous_Loan_Defaults`, `Employment_Stability`, `Monthly_Expenses`, `Savings_Balance`, `EMI_to_Income_Ratio`, `Collateral_Value`)
   - Most-frequent mode imputation + `OneHotEncoder(drop="first")` for `Payment_History` and `Loan_Purpose`
   - Domain Feature Engineering: `Total_Income`, `Loan_to_Income`, `Expense_to_Income`, `Collateral_Coverage`, `DTI_Ratio_Sq`, `Credit_Score_Sq`
4. **Trained Classifiers**:
   - **Decision Tree Classifier**: CART Gini impurity (`max_depth=6`, `min_samples_leaf=20`)
   - **Logistic Regression**: L2 regularization (`C=1.0`, `max_iter=2000`)
5. **Artifacts Exported**:
   - `ml/artifacts/decision_tree_model.pkl`
   - `ml/artifacts/logistic_regression_model.pkl`
   - `ml/artifacts/model_metadata.json`

To regenerate the web JSON artifact (`src/data/trainedModels.json`) using only Python standard library:
```bash
python3 ml/export_artifacts.py
```

---

## 2. Frontend Dashboard Setup & Development

### Install & Run Locally
```bash
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

### Build for Production
```bash
npm run build
```

### Key Dashboard Views
- **Dashboard (Loan Approval Predictor)**: Full applicant inputs (including `Monthly_Expenses`, `Savings_Balance`, `Previous_Loan_Defaults`, `Payment_History`, `Employment_Stability`, dynamic per-loan existing EMIs, draggable `Credit_Score`, adjustable `DTI_Ratio`, `Loan_Purpose`, and concessional `Property_Area` interest tiers: Rural 5.0%, Semi-Urban 7.0%, Urban 10.5%) with explicit button-triggered prediction, approval probability, approved-only EMI & Annual Interest breakdown, and model audit trails.
- **Overview & Trends**: Multi-chart dark-mode visualizations tracking approval trajectories across Credit Score bands, DTI risk decay, Total Income brackets, Payment History, Loan Purpose, Previous Loan Defaults, and Existing Loans.
- **Model Benchmarks**: Side-by-side Decision Tree vs. Logistic Regression holdout evaluation, ROC curves, confusion matrices, 5-fold stratified CV stability, and CART split rules.

---

## 3. IEEE Two-Column Research Paper (`.docx`)

To regenerate the IEEE conference paper from the latest trained model metrics:
```bash
node ml/generate_ieee_paper.mjs
```
Or using Python (`python-docx`):
```bash
python ml/generate_ieee_paper.py
```
The compiled manuscript is saved to `public/CreditWise_IEEE_Conference_Paper.docx` in US Letter, two-column IEEE Conference layout with Times New Roman typography, Abstract, Index Terms, four empirical IEEE tables, and numbered references.

---

## 4. Jupyter Notebook & Evaluation Chart Images

Run the interactive notebook `ml/CreditWise_Evaluation_Notebook.ipynb` or Python script `ml/plot_evaluation_results.py` (or regenerate crisp PNGs directly from `src/data/trainedModels.json` via `node ml/render_charts.mjs`):
- `public/charts/decision_tree_confusion_matrix.png`
- `public/charts/decision_tree_metrics.png` (Accuracy, Precision, Recall, F1-score)
- `public/charts/logistic_regression_confusion_matrix.png`
- `public/charts/logistic_regression_metrics.png` (Accuracy, Precision, Recall, F1-score)

