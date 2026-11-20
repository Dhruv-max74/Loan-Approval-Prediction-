import fs from "fs";
import path from "path";
import {
  AlignmentType,
  BorderStyle,
  Document,
  Packer,
  Paragraph,
  SectionType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
  convertInchesToTwip,
} from "docx";

const ARTIFACT_PATH = path.resolve("src/data/trainedModels.json");
const OUT_DOCX_PATH = path.resolve("public/CreditWise_IEEE_Conference_Paper.docx");

function ieeeSectionHeading(title) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 100 },
    children: [
      new TextRun({
        text: title,
        font: "Times New Roman",
        size: 20,
        bold: true,
      }),
    ],
  });
}

function ieeeSubHeading(title) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 140, after: 60 },
    children: [
      new TextRun({
        text: title,
        font: "Times New Roman",
        size: 20,
        italics: true,
      }),
    ],
  });
}

function ieeeBody(text, firstIndent = true) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: firstIndent ? { firstLine: convertInchesToTwip(0.18) } : undefined,
    spacing: { after: 80, line: 250 },
    children: [
      new TextRun({
        text,
        font: "Times New Roman",
        size: 20,
      }),
    ],
  });
}

function ieeeTableCaption(label, title) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 140, after: 60 },
    children: [
      new TextRun({
        text: `${label}\n`,
        font: "Times New Roman",
        size: 16,
        bold: true,
      }),
      new TextRun({
        text: title.toUpperCase(),
        font: "Times New Roman",
        size: 16,
      }),
    ],
  });
}

function buildIeeeTable(headers, rows) {
  const borderSpec = {
    style: BorderStyle.SINGLE,
    size: 4,
    color: "000000",
  };
  const noneBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map(
      (h, idx) =>
        new TableCell({
          borders: {
            top: borderSpec,
            bottom: borderSpec,
            left: noneBorder,
            right: noneBorder,
          },
          margins: { top: 60, bottom: 60, left: 60, right: 60 },
          children: [
            new Paragraph({
              alignment: idx === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
              children: [
                new TextRun({
                  text: h,
                  font: "Times New Roman",
                  size: 16,
                  bold: true,
                }),
              ],
            }),
          ],
        })
    ),
  });

  const dataRows = rows.map(
    (row, rIdx) =>
      new TableRow({
        children: row.map(
          (cell, cIdx) =>
            new TableCell({
              borders: {
                top: noneBorder,
                bottom: rIdx === rows.length - 1 ? borderSpec : noneBorder,
                left: noneBorder,
                right: noneBorder,
              },
              margins: { top: 40, bottom: 40, left: 60, right: 60 },
              children: [
                new Paragraph({
                  alignment: cIdx === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
                  children: [
                    new TextRun({
                      text: String(cell),
                      font: "Times New Roman",
                      size: 16,
                    }),
                  ],
                }),
              ],
            })
        ),
      })
  );

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [headerRow, ...dataRows],
  });
}

export async function generateIeeeDocx() {
  const artifact = JSON.parse(fs.readFileSync(ARTIFACT_PATH, "utf-8"));
  const ds = artifact.datasetSummary;
  const dt = artifact.models.decisionTree;
  const lr = artifact.models.logisticRegression;
  const cv = artifact.crossValidation;

  const dtAcc = (dt.metrics.accuracy * 100).toFixed(2);
  const dtPrec = (dt.metrics.precision * 100).toFixed(2);
  const dtRec = (dt.metrics.recall * 100).toFixed(2);
  const dtF1 = (dt.metrics.f1 * 100).toFixed(2);
  const dtAuc = dt.metrics.rocAuc.toFixed(4);
  const dtCv = (cv.decisionTree.meanAccuracy * 100).toFixed(2);
  const dtCvStd = (cv.decisionTree.stdAccuracy * 100).toFixed(2);

  const lrAcc = (lr.metrics.accuracy * 100).toFixed(2);
  const lrPrec = (lr.metrics.precision * 100).toFixed(2);
  const lrRec = (lr.metrics.recall * 100).toFixed(2);
  const lrF1 = (lr.metrics.f1 * 100).toFixed(2);
  const lrAuc = lr.metrics.rocAuc.toFixed(4);
  const lrCv = (cv.logisticRegression.meanAccuracy * 100).toFixed(2);
  const lrCvStd = (cv.logisticRegression.stdAccuracy * 100).toFixed(2);

  const appRate = ((ds.approvedCount / ds.labeledRecords) * 100).toFixed(2);
  const rejRate = ((ds.rejectedCount / ds.labeledRecords) * 100).toFixed(2);

  const csMean = ds.featureMeans.find((f) => f.feature === "Credit_Score");
  const dtiMean = ds.featureMeans.find((f) => f.feature === "DTI_Ratio");
  const defMean = ds.featureMeans.find((f) => f.feature === "Previous_Loan_Defaults");
  const emiMean = ds.featureMeans.find((f) => f.feature === "EMI_to_Income_Ratio");

  const doc = new Document({
    styles: {
      default: {
        document: {
          run: {
            font: "Times New Roman",
            size: 20,
          },
        },
      },
    },
    sections: [
      {
        properties: {
          page: {
            size: {
              width: convertInchesToTwip(8.5),
              height: convertInchesToTwip(11.0),
            },
            margin: {
              top: convertInchesToTwip(0.75),
              bottom: convertInchesToTwip(1.0),
              left: convertInchesToTwip(0.625),
              right: convertInchesToTwip(0.625),
            },
          },
        },
        children: [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 140 },
            children: [
              new TextRun({
                text: "CreditWise: Comparative Credit Risk Assessment Using Interpretable Decision Tree and Logistic Regression Classifiers",
                font: "Times New Roman",
                size: 44,
                bold: true,
              }),
            ],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 60 },
            children: [
              new TextRun({
                text: "CreditWise Analytics Research Group",
                font: "Times New Roman",
                size: 22,
              }),
            ],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 200 },
            children: [
              new TextRun({
                text: "Department of Computer Science & Financial Engineering\nAutomated Credit Scoring & Risk Visualization Laboratory\nresearch@creditwise-analytics.org",
                font: "Times New Roman",
                size: 18,
                italics: true,
              }),
            ],
          }),
        ],
      },
      {
        properties: {
          type: SectionType.CONTINUOUS,
          column: {
            count: 2,
            space: convertInchesToTwip(0.25),
            equalWidth: true,
          },
          page: {
            size: {
              width: convertInchesToTwip(8.5),
              height: convertInchesToTwip(11.0),
            },
            margin: {
              top: convertInchesToTwip(0.75),
              bottom: convertInchesToTwip(1.0),
              left: convertInchesToTwip(0.625),
              right: convertInchesToTwip(0.625),
            },
          },
        },
        children: [
          new Paragraph({
            alignment: AlignmentType.JUSTIFIED,
            spacing: { after: 100, line: 230 },
            children: [
              new TextRun({
                text: "Abstract—",
                font: "Times New Roman",
                size: 18,
                bold: true,
                italics: true,
              }),
              new TextRun({
                text: `Automated credit underwriting requires predictive models that balance discriminative accuracy with regulatory transparency. Inspired by the CreditWise loan approval analytics framework, this paper presents an empirical comparative study of two interpretable supervised classifiers—a Classification and Regression Tree (CART) Decision Tree and an L2-regularized Logistic Regression model—evaluated on a hybrid consumer loan dataset of ${ds.totalRecords.toLocaleString()} applications (${ds.approvedCount.toLocaleString()} approved, ${ds.rejectedCount.toLocaleString()} rejected across 15 raw financial, credit-bureau, and behavioral predictors). To prevent data leakage, a strictly train-fitted preprocessing pipeline combining median/mode imputation, six domain-engineered financial ratios (Total Income, Loan-to-Income, Expense-to-Income, Collateral Coverage, squared Debt-to-Income, and squared Credit Score), standardization, and one-hot encoding is implemented. On a stratified 80/20 holdout evaluation (${artifact.splitInfo.trainCount.toLocaleString()} training, ${artifact.splitInfo.testCount.toLocaleString()} testing), the depth-constrained Decision Tree (max_depth=${dt.hyperparameters.maxDepth}) achieves ${dtAcc}% accuracy, ${dtF1}% F1-score, and ${dtAuc} ROC-AUC, compared against Logistic Regression (${lrAcc}% accuracy, ${lrF1}% F1-score, ${lrAuc} ROC-AUC). Five-fold stratified cross-validation confirms stability (${dtCv}% ± ${dtCvStd}% vs. ${lrCv}% ± ${lrCvStd}%). Furthermore, we introduce a dark-mode analytical dashboard featuring switchable dual-model inference, interactive trend exploration, concessional area-based EMI scheduling, and real-time decision-path auditing.`,
                font: "Times New Roman",
                size: 18,
                bold: true,
              }),
            ],
          }),
          new Paragraph({
            alignment: AlignmentType.JUSTIFIED,
            spacing: { after: 160, line: 230 },
            children: [
              new TextRun({
                text: "Index Terms—",
                font: "Times New Roman",
                size: 18,
                bold: true,
                italics: true,
              }),
              new TextRun({
                text: "Credit scoring, Decision Tree, Logistic Regression, Debt-to-Income ratio, payment history, machine learning interpretability, loan approval prediction.",
                font: "Times New Roman",
                size: 18,
                bold: true,
              }),
            ],
          }),

          ieeeSectionHeading("I. INTRODUCTION"),
          ieeeBody(
            "Retail banking and fintech lending institutions process thousands of loan applications under strict regulatory mandates—such as the Equal Credit Opportunity Act (ECOA) and Basel III risk governance—that require clear, auditable explanations for credit denials [1]. While deep neural networks and opaque ensemble methods can capture complex interactions, financial compliance teams frequently prefer glass-box models whose predictions map directly to applicant creditworthiness indicators.",
            false
          ),
          ieeeBody(
            "Interactive credit scoring portals such as CreditWise Insights [2] demonstrated how machine learning classifiers paired with web dashboards streamline preliminary loan screening. However, single-model deployments often obscure how non-linear underwriting thresholds (such as strict Credit Score bands, prior loan defaults, and Debt-to-Income ceilings) interact with continuous linear risk factors. Moreover, naive preprocessing applied prior to dataset splitting introduces subtle target leakage that inflates reported validation metrics."
          ),
          ieeeBody(
            "To address these gaps, this study makes three contributions: (1) a leakage-free Python and Scikit-Learn preprocessing and feature-engineering pipeline for multi-attribute hybrid consumer loan records; (2) a rigorous head-to-head benchmark of CART Decision Tree and L2-regularized Logistic Regression classifiers across holdout metrics, ROC curves, and 5-fold stratified cross-validation; and (3) an interactive dark-mode analytical dashboard supporting switchable dual-model scoring, cashflow affordability verification, and transparent decision auditing."
          ),

          ieeeSectionHeading("II. RELATED WORK AND SYSTEM CONTEXT"),
          ieeeBody(
            "Statistical credit scoring originated with discriminant analysis and evolved toward Logistic Regression as the industry benchmark for probability-of-default estimation due to its calibrated log-odds formulation [3]. Conversely, recursive partitioning via CART Decision Trees [4] naturally models piecewise underwriting rules—mirroring manual credit officer checklists where applicants must simultaneously clear minimum credit bureau thresholds, clean repayment histories, and manageable existing debt obligations.",
            false
          ),
          ieeeBody(
            "Prior CreditWise implementations [2] evaluated baseline classification workflows in Streamlit using standard imputation and scaling. Extending this foundation, our system isolates Decision Tree and Logistic Regression as co-equal, switchable analytical lenses, allowing underwriters to inspect both threshold-based rule paths and continuous marginal odds contributions on identical applicant profiles."
          ),

          ieeeSectionHeading("III. DATASET AND PREPROCESSING METHODOLOGY"),
          ieeeSubHeading("A. Dataset Composition and Class Distribution"),
          ieeeBody(
            `The empirical hybrid dataset comprises ${ds.totalRecords.toLocaleString()} consumer loan applications described by 15 applicant financial, behavioral, and collateral attributes (13 numerical, 2 categorical) plus a binary target (Loan_Status: 1 = Approved, 0 = Rejected). Overall class prevalence exhibits ${ds.approvedCount.toLocaleString()} approved applications (${appRate}%) and ${ds.rejectedCount.toLocaleString()} rejected applications (${rejRate}%) with zero missing entries. Table I summarizes key numerical indicators stratified by loan outcome.`,
            false
          ),

          ieeeTableCaption("TABLE I", "Summary of Key Numerical Features by Loan Outcome"),
          buildIeeeTable(
            ["Feature", "Approved Mean", "Rejected Mean", "Overall Mean"],
            ds.featureMeans.slice(0, 8).map((fm) => [
              fm.feature.replace(/_/g, " "),
              fm.approvedMean.toLocaleString(),
              fm.rejectedMean.toLocaleString(),
              fm.overallMean.toLocaleString(),
            ])
          ),

          ieeeSubHeading("B. Zero-Leakage Preprocessing and Feature Engineering"),
          ieeeBody(
            `The ${ds.labeledRecords.toLocaleString()} labeled observations are partitioned using an 80/20 stratified split (random_state=42), yielding ${artifact.splitInfo.trainCount.toLocaleString()} training and ${artifact.splitInfo.testCount.toLocaleString()} holdout test samples. All imputation statistics (medians for 13 numerical columns, modes for Payment_History and Loan_Purpose) and z-score standardization parameters are fitted strictly on the training fold and applied invariantly to validation and live inference inputs.`,
            false
          ),
          ieeeBody(
            "Domain-guided feature engineering constructs six composite risk signals: (1) Total_Income = Applicant_Income + Coapplicant_Income; (2) Loan_to_Income = Loan_Amount / (Total_Income + 1); (3) Expense_to_Income = Monthly_Expenses / (Total_Income + 1); (4) Collateral_Coverage = Collateral_Value / (Loan_Amount + 1); (5) DTI_Ratio_Sq = DTI_Ratio^2 to amplify high-leverage debt penalties; and (6) Credit_Score_Sq = Credit_Score^2 to capture non-linear bureau quality acceleration. Categorical features are one-hot encoded with first-level drop to avoid multicollinearity."
          ),

          ieeeSectionHeading("IV. MACHINE LEARNING ARCHITECTURES"),
          ieeeSubHeading("A. L2-Regularized Logistic Regression"),
          ieeeBody(
            "Logistic Regression models the posterior approval probability P(y=1 | x) via the sigmoid transformation sigma(w^T x + b) = 1 / (1 + exp(-(w^T x + b))), where x denotes the standardized feature vector. Parameters (w, b) minimize the L2-penalized binary cross-entropy objective J(w,b) = -(1/N) sum [y_i log(p_i) + (1-y_i) log(1-p_i)] + (lambda/2)||w||_2^2. Regularization (lambda = 0.01) stabilizes correlated engineered terms while preserving interpretable odds ratios exp(w_j).",
            false
          ),
          ieeeSubHeading("B. CART Decision Tree Classifier"),
          ieeeBody(
            `The CART Decision Tree partitions the unscaled feature space recursively by selecting feature j and threshold tau that maximize Gini impurity reduction Delta G = G(S) - (|S_L|/|S|)G(S_L) - (|S_R|/|S|)G(S_R), where G(S) = 1 - p_1^2 - p_0^2. To prevent overfitting across ${artifact.splitInfo.trainCount.toLocaleString()} training samples while capturing multi-factor underwriting interactions, tree growth is constrained to max_depth = ${dt.hyperparameters.maxDepth} and min_samples_leaf = ${dt.hyperparameters.minSamplesLeaf}. Each leaf outputs empirical approval probability p_1 and binary decision y_hat.`,
            false
          ),

          ieeeSectionHeading("V. EXPERIMENTAL RESULTS AND ANALYSIS"),
          ieeeSubHeading("A. Holdout and Cross-Validation Performance"),
          ieeeBody(
            `Table II reports holdout test performance and 5-fold stratified cross-validation accuracy across both classifiers. The Decision Tree achieves ${dtAcc}% test accuracy and ${dtAuc} ROC-AUC, while Logistic Regression achieves ${lrAcc}% test accuracy and ${lrAuc} ROC-AUC. Table III details the holdout confusion matrices across ${artifact.splitInfo.testCount.toLocaleString()} test applications.`,
            false
          ),

          ieeeTableCaption("TABLE II", "Comparative Classification Performance (80/20 Holdout & 5-Fold CV)"),
          buildIeeeTable(
            ["Metric", "Decision Tree", "Logistic Regression"],
            [
              ["Test Accuracy (%)", `${dtAcc}%`, `${lrAcc}%`],
              ["Precision (%)", `${dtPrec}%`, `${lrPrec}%`],
              ["Recall (%)", `${dtRec}%`, `${lrRec}%`],
              ["F1-Score (%)", `${dtF1}%`, `${lrF1}%`],
              ["ROC-AUC", dtAuc, lrAuc],
              ["Log Loss", dt.metrics.logLoss.toFixed(4), lr.metrics.logLoss.toFixed(4)],
              ["Brier Score", dt.metrics.brierScore.toFixed(4), lr.metrics.brierScore.toFixed(4)],
              ["5-Fold CV Acc.", `${dtCv}% ± ${dtCvStd}%`, `${lrCv}% ± ${lrCvStd}%`],
            ]
          ),

          ieeeTableCaption("TABLE III", `Holdout Test Set Confusion Matrices (n = ${artifact.splitInfo.testCount.toLocaleString()})`),
          buildIeeeTable(
            ["Actual \\ Predicted", "DT: No / Yes", "LR: No / Yes"],
            [
              [
                "Actual: Rejected (0)",
                `${dt.metrics.confusionMatrix.tn} / ${dt.metrics.confusionMatrix.fp}`,
                `${lr.metrics.confusionMatrix.tn} / ${lr.metrics.confusionMatrix.fp}`,
              ],
              [
                "Actual: Approved (1)",
                `${dt.metrics.confusionMatrix.fn} / ${dt.metrics.confusionMatrix.tp}`,
                `${lr.metrics.confusionMatrix.fn} / ${lr.metrics.confusionMatrix.tp}`,
              ],
            ]
          ),

          ieeeSubHeading("B. Feature Importance and Decision Boundaries"),
          ieeeBody(
            `Inspection of Gini impurity reductions and standardized Logistic Regression weights (Table IV) reveals strong structural consensus: Credit_Score, DTI_Ratio, Previous_Loan_Defaults, Existing_Loans, EMI_to_Income_Ratio, and Payment_History dominate underwriting outcomes. In the hybrid dataset, approved applicants average a Credit Score of ${csMean ? csMean.approvedMean.toFixed(1) : "-"} versus ${csMean ? csMean.rejectedMean.toFixed(1) : "-"} for rejected applicants, a DTI Ratio of ${dtiMean ? dtiMean.approvedMean.toFixed(1) : "-"}% versus ${dtiMean ? dtiMean.rejectedMean.toFixed(1) : "-"}%, ${defMean ? defMean.approvedMean.toFixed(2) : "-"} prior defaults versus ${defMean ? defMean.rejectedMean.toFixed(2) : "-"}, and an EMI-to-Income ratio of ${emiMean ? emiMean.approvedMean.toFixed(1) : "-"}% versus ${emiMean ? emiMean.rejectedMean.toFixed(1) : "-"}%.`,
            false
          ),

          ieeeTableCaption("TABLE IV", "Top Discriminative Features Across Models"),
          buildIeeeTable(
            ["Rank", "DT Feature (Gini Imp.)", "LR Feature (Std. Coef.)"],
            [0, 1, 2, 3, 4].map((idx) => {
              const dtItem = dt.featureImportances[idx];
              const lrItem = lr.featureCoefficients[idx];
              return [
                `#${idx + 1}`,
                dtItem ? `${dtItem.feature} (${(dtItem.importance * 100).toFixed(1)}%)` : "-",
                lrItem ? `${lrItem.feature} (${lrItem.coefficient > 0 ? "+" : ""}${lrItem.coefficient.toFixed(2)})` : "-",
              ];
            })
          ),

          ieeeSectionHeading("VI. INTERACTIVE SYSTEM ARCHITECTURE"),
          ieeeBody(
            "To operationalize these models, we built a dark-mode analytical web platform using React, TypeScript, Recharts, and a decoupled Scikit-Learn Python training pipeline (train_models.py). The dashboard provides three focused workflows with switchable Decision Tree and Logistic Regression evaluation: (1) an interactive Loan Approval Predictor combining ML inference with monthly expense, existing-loan EMI, and concessional area interest-tier amortization auditing; (2) an Overview & Trends view filtering 10,000 applicant records across Payment History and Loan Purpose; and (3) a Model Benchmarks view presenting ROC curves, confusion matrices, and CART split hierarchies.",
            false
          ),

          ieeeSectionHeading("VII. CONCLUSION AND FUTURE WORK"),
          ieeeBody(
            `This study evaluated a depth-bounded CART Decision Tree (${dtAcc}% accuracy, ${dtAuc} ROC-AUC) alongside L2-regularized Logistic Regression (${lrAcc}% accuracy, ${lrAuc} ROC-AUC) on the 10,000-record hybrid consumer loan dataset. While CART captures conjunctive threshold rules across credit score, payment history, and prior defaults, Logistic Regression delivers smooth probabilistic calibration across linear leverage factors. Future work will explore cost-sensitive threshold optimization, monotonic gradient-boosted ensembles, and automated covariate drift monitoring.`,
            false
          ),

          ieeeSectionHeading("REFERENCES"),
          ...[
            '[1] Board of Governors of the Federal Reserve System, "Interagency Guidance on Model Risk Management (SR 11-7)," Washington, DC, 2011.',
            '[2] CreditWise Insights, "Machine Learning Loan Approval & Credit Risk Scoring System," Streamlit Cloud Application. [Online]. Available: https://creditwise-insights.streamlit.app/',
            '[3] D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, Applied Logistic Regression, 3rd ed. Hoboken, NJ, USA: Wiley, 2013.',
            '[4] L. Breiman, J. Friedman, R. Olshen, and C. Stone, Classification and Regression Trees. Belmont, CA, USA: Wadsworth, 1984.',
            '[5] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.',
            '[6] L. C. Thomas, D. B. Edelman, and J. N. Crook, Credit Scoring and Its Applications, 2nd ed. Philadelphia, PA, USA: SIAM, 2017.',
            '[7] G. Louppe, "Understanding Random Forests: From Theory to Practice," Ph.D. dissertation, Univ. of Liege, Belgium, 2014.',
            '[8] Basel Committee on Banking Supervision, "Basel III: A global regulatory framework for more resilient banks and banking systems," Bank for International Settlements, 2011.',
          ].map(
            (ref) =>
              new Paragraph({
                alignment: AlignmentType.JUSTIFIED,
                indent: { left: convertInchesToTwip(0.22), hanging: convertInchesToTwip(0.22) },
                spacing: { after: 60, line: 220 },
                children: [
                  new TextRun({
                    text: ref,
                    font: "Times New Roman",
                    size: 16,
                  }),
                ],
              })
          ),
        ],
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.mkdirSync(path.dirname(OUT_DOCX_PATH), { recursive: true });
  fs.writeFileSync(OUT_DOCX_PATH, buffer);
  console.log(`Generated IEEE Conference Paper at ${OUT_DOCX_PATH} (${buffer.length} bytes)`);
}

generateIeeeDocx().catch((err) => {
  console.error("Failed to generate IEEE docx:", err);
  process.exit(1);
});
