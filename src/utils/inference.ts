import trainedModelsRaw from "../data/trainedModels.json";
import type {
  DecisionPathStep,
  DecisionTreeNode,
  InferenceExplanation,
  LoanInputForm,
  LogisticContribution,
  TrainedModelsArtifact,
} from "../types/creditwise";

export const MODEL_ARTIFACT = trainedModelsRaw as TrainedModelsArtifact;

export const DEFAULT_LOAN_INPUT: LoanInputForm = {
  applicantIncome: 52000,
  coapplicantIncome: 14000,
  creditScore: 735,
  dtiRatio: 9.5,
  loanAmount: 900000,
  loanTenure: 60,
  existingLoans: 1,
  previousLoanDefaults: 0,
  paymentHistory: "Excellent",
  employmentStability: 7.5,
  monthlyExpenses: 18500,
  savingsBalance: 165000,
  emiToIncomeRatio: 29.0,
  loanPurpose: "Home",
  collateralValue: 1200000,
  propertyArea: "Urban",
};

export function buildFeatureVector(input: LoanInputForm): number[] {
  const { preprocessing } = MODEL_ARTIFACT;
  const numMap: Record<string, number> = {
    Applicant_Income: input.applicantIncome,
    Coapplicant_Income: input.coapplicantIncome,
    Credit_Score: input.creditScore,
    DTI_Ratio: input.dtiRatio,
    Loan_Amount: input.loanAmount,
    Loan_Tenure: input.loanTenure,
    Existing_Loans: input.existingLoans,
    Previous_Loan_Defaults: input.previousLoanDefaults,
    Employment_Stability: input.employmentStability,
    Monthly_Expenses: input.monthlyExpenses,
    Savings_Balance: input.savingsBalance,
    EMI_to_Income_Ratio: input.emiToIncomeRatio,
    Collateral_Value: input.collateralValue,
  };

  const totalIncome = input.applicantIncome + input.coapplicantIncome;
  const loanToIncome = input.loanAmount / (totalIncome + 1.0);
  const expenseToIncome = input.monthlyExpenses / (totalIncome + 1.0);
  const collateralCoverage = input.collateralValue / (input.loanAmount + 1.0);
  const dtiSq = input.dtiRatio * input.dtiRatio;
  const csSq = input.creditScore * input.creditScore;

  const vec: number[] = preprocessing.numericCols.map(
    (col) => numMap[col] ?? preprocessing.medians[col] ?? 0
  );
  vec.push(
    totalIncome,
    loanToIncome,
    expenseToIncome,
    collateralCoverage,
    dtiSq,
    csSq
  );

  const catMap: Record<string, string> = {
    Payment_History: input.paymentHistory,
    Loan_Purpose: input.loanPurpose,
  };

  for (const col of preprocessing.categoricalCols) {
    const val = catMap[col] || preprocessing.modes[col];
    const allowed = preprocessing.catValues[col] || [];
    for (let i = 1; i < allowed.length; i++) {
      vec.push(val === allowed[i] ? 1.0 : 0.0);
    }
  }
  return vec;
}

export function scoreApplicant(input: LoanInputForm): InferenceExplanation {
  const vec = buildFeatureVector(input);
  const { preprocessing, models } = MODEL_ARTIFACT;

  // 1. Decision Tree traversal with exact branch trace
  const path: DecisionPathStep[] = [];
  let node: DecisionTreeNode = models.decisionTree.tree;

  while (!node.leaf && node.featureIndex !== undefined && node.threshold !== undefined) {
    const actualValue = vec[node.featureIndex];
    const goLeft = actualValue <= node.threshold;
    path.push({
      nodeId: node.id,
      depth: node.depth,
      featureName: node.featureName || preprocessing.featureNames[node.featureIndex],
      threshold: node.threshold,
      actualValue,
      direction: goLeft ? "left" : "right",
      samples: node.samples,
      prob: node.prob,
      gini: node.gini,
    });
    node = goLeft && node.left ? node.left : node.right!;
  }

  // 2. Logistic Regression standardized inference & log-odds attribution
  const { bias, weights } = models.logisticRegression;
  const { scalerMeans, scalerStds, featureNames } = preprocessing;

  let logit = bias;
  const contributions: LogisticContribution[] = [];

  for (let j = 0; j < vec.length; j++) {
    const normVal = (vec[j] - scalerMeans[j]) / (scalerStds[j] || 1.0);
    const contrib = weights[j] * normVal;
    logit += contrib;
    contributions.push({
      feature: featureNames[j],
      rawValue: vec[j],
      normalizedValue: normVal,
      weight: weights[j],
      contribution: contrib,
    });
  }

  const clampedLogit = Math.max(-30, Math.min(30, logit));
  const lrProb = 1.0 / (1.0 + Math.exp(-clampedLogit));

  const topContributions = [...contributions]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 8);

  return {
    decisionTree: {
      probability: node.prob,
      prediction: node.prob >= 0.5 ? "Yes" : "No",
      leafSamples: node.samples,
      leafGini: node.gini,
      path,
    },
    logisticRegression: {
      probability: lrProb,
      prediction: lrProb >= 0.5 ? "Yes" : "No",
      logit,
      bias,
      topContributions,
    },
  };
}

export function formatFeatureLabel(rawName: string): string {
  return rawName.replace(/_/g, " ");
}
