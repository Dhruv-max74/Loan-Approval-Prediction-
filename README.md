# 🏦 Loan Approval Prediction

### An Explainable and Comparative Machine Learning Approach for Loan Approval Prediction Using Decision Tree and Logistic Regression

<p align="center">
  <a href="https://loanapproval-prediction.netlify.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Visit%20Application-success?style=for-the-badge&logo=netlify" alt="Live Demo"/>
  </a>
  <img src="https://img.shields.io/badge/Python-Machine%20Learning-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/React-TypeScript-blue?style=for-the-badge&logo=react" alt="React"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Project Status"/>
</p>

## 🌐 Live Application

**Try the application here:**  
👉 https://loanapproval-prediction.netlify.app/

Loan Approval Prediction is a machine learning-based web application designed to predict whether a loan application is likely to be approved or rejected.

The project compares two machine learning algorithms — **Decision Tree** and **Logistic Regression** — to analyze applicant information, generate predictions, and evaluate model performance.

The application also provides an interactive dashboard for exploring loan trends, model benchmarks, and prediction results.

---

## 📌 Project Overview

Loan approval is an important decision-making process in the banking and financial sector. Traditional loan evaluation can involve manual assessment of several applicant details, which may take time and can introduce inconsistencies.

This project explores how machine learning can assist in analyzing applicant data and predicting loan approval outcomes.

The system uses a hybrid dataset containing **10,000 loan application records**, with 15 input features and one target variable (`Loan_Status`).

Two classification algorithms are trained and compared to understand their predictive performance.

### 🎯 Project Objectives

- Build a machine learning system for loan approval prediction.
- Compare Decision Tree and Logistic Regression algorithms.
- Analyze applicant information using multiple features.
- Evaluate model performance using standard classification metrics.
- Provide an interactive and user-friendly dashboard.
- Improve prediction transparency through explainable results.

---

## ✨ Key Features

- **Loan Prediction:** Predict loan approval or rejection using applicant details.
- **Dual ML Models:** Compare Decision Tree and Logistic Regression predictions.
- **Model Benchmarking:** View and compare accuracy, precision, recall, F1-score, and ROC-AUC.
- **Interactive Dashboard:** Explore loan application statistics and trends.
- **Data Analysis:** Analyze approval and rejection patterns.
- **Visual Analytics:** View confusion matrices and model performance charts.
- **EMI Calculator:** Estimate monthly loan installments.
- **IEEE Research Support:** Includes evaluation results and research paper resources.

---

## 🧠 Machine Learning Models

This project implements two supervised machine learning classification algorithms.

### 1. Decision Tree Classifier

Decision Tree is a supervised learning algorithm that makes predictions by splitting data into decision-based branches.

It is useful for understanding how different applicant features influence the final prediction.

### 2. Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of an outcome.

It is used to predict whether a loan application belongs to the approved or rejected class.

### Model Comparison

| Evaluation Metric | Decision Tree | Logistic Regression |
|---|---:|---:|
| Accuracy | 81.20% | **82.80%** |
| Precision | 83.80% | **85.42%** |
| Recall | 88.41% | **88.94%** |
| F1-Score | 86.04% | **87.14%** |
| ROC-AUC | 0.8787 | **0.8965** |

**Observation:** Logistic Regression achieved better results across all five reported evaluation metrics on the test dataset.

---

## 📊 Dataset Information

The project uses a hybrid loan approval dataset created for machine learning experimentation.

| Dataset Detail | Information |
|---|---|
| Total Records | 10,000 |
| Input Features | 15 |
| Target Variable | Loan_Status |
| Approved Applications | 6,555 |
| Rejected Applications | 3,445 |
| Train-Test Split | 80:20 |
| Feature Scaling | StandardScaler |

### Target Variable

- `1` – Loan Approved
- `0` – Loan Rejected

The dataset is included in the project under:

`public/assets/aistudio/loan_approval_hybrid_dataset_20711486.csv`

**Note:** This is a research and educational dataset. It should not be treated as actual banking or financial institution data.

---

## ⚙️ Technology Stack

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS
- Recharts
- Lucide React

### Machine Learning & Data Science
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

### Development Tools
- Visual Studio Code
- Google AI Studio
- Netlify
- Git & GitHub

---

## 🏗️ Project Structure

```text
Loan-Approval-Prediction/
│
├── public/
│   ├── assets/
│   │   └── aistudio/
│   │       └── loan_approval_hybrid_dataset.csv
│   │
│   ├── charts/
│   │   ├── decision_tree_confusion_matrix.png
│   │   ├── logistic_regression_confusion_matrix.png
│   │   ├── decision_tree_metrics.png
│   │   └── logistic_regression_metrics.png
│   │
│   └── CreditWise_IEEE_Conference_Paper.docx
│
├── src/
│   ├── components/
│   │   ├── PredictorView.tsx
│   │   ├── OverviewTrendsView.tsx
│   │   └── ModelBenchmarksView.tsx
│   │
│   ├── data/
│   │   └── trainedModels.json
│   │
│   ├── types/
│   │   └── creditwise.ts
│   │
│   ├── utils/
│   │   └── inference.ts
│   │
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
│
├── ml/
│   ├── train_models.py
│   ├── export_artifacts.py
│   ├── plot_evaluation_results.py
│   ├── requirements.txt
│   └── CreditWise_Evaluation_Notebook.ipynb
│
├── .env.example
├── .gitignore
├── package.json
├── README.md
└── vite.config.ts
```

---

## 🚀 Installation & Setup

Follow these steps to run the project locally.

### Prerequisites

Make sure you have installed:

- Node.js
- npm
- Python 3.x
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Dhruv-max74/Loan-Approval-Prediction-.git
```

### Step 2: Open the Project

```bash
cd Loan-Approval-Prediction-
```

### Step 3: Install Dependencies

```bash
npm install
```

### Step 4: Configure Environment Variables

Create a `.env` file based on `.env.example` and configure the required environment variables if needed.

Never upload real API keys or secrets to GitHub.

### Step 5: Run the Application

```bash
npm run dev
```

Open the local URL displayed in your terminal to access the application.

### Step 6: Build for Production

```bash
npm run build
```

---

## 🔬 Model Training

The machine learning training scripts are available in the `ml/` directory.

To install the Python dependencies:

```bash
pip install -r ml/requirements.txt
```

The training workflow includes:

1. Loading the loan approval dataset.
2. Data preprocessing.
3. Feature preparation and scaling.
4. Stratified train-test splitting (80:20).
5. Training the Decision Tree model.
6. Training the Logistic Regression model.
7. Evaluating model performance.
8. Exporting trained model artifacts.

---

## 📈 Experimental Results

The models were evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

The experimental results show that Logistic Regression performed better than Decision Tree across the reported evaluation metrics.

The project also includes visual comparison charts and confusion matrices for both models.

---

## 🔮 Future Enhancements

- Integrate larger and more diverse real-world datasets.
- Explore Random Forest, XGBoost, and other classification algorithms.
- Add explainable AI techniques such as SHAP.
- Improve model validation and generalization.
- Add secure user authentication.
- Deploy a dedicated backend prediction API.
- Improve fairness and bias evaluation in loan predictions.

---

## ⚠️ Disclaimer

This project is developed for educational, research, and demonstration purposes only.

The predictions generated by this application are based on machine learning models trained using the available experimental dataset. They do not represent actual bank decisions, guaranteed loan approvals, or professional financial advice.

---

## 👨‍💻 Author & Project

**Project:** Loan Approval Prediction  
**Application Name:** CreditWise Analytics  
**Repository:** [Loan-Approval-Prediction](https://github.com/Dhruv-max74/Loan-Approval-Prediction-)  
**Live Demo:** [Open Application](https://loanapproval-prediction.netlify.app/)

---

<p align="center">
  <b>⭐ If you find this project useful, consider giving it a star!</b>
  <br/>
  Built with ❤️ using Machine Learning and Web Technologies.
</p>
