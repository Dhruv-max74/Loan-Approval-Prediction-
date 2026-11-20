export type ModelKey = "decisionTree" | "logisticRegression";

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface RocPoint {
  threshold: number;
  fpr: number;
  tpr: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  rocAuc: number;
  logLoss: number;
  brierScore: number;
  confusionMatrix: ConfusionMatrix;
  rocCurve: RocPoint[];
}

export interface DecisionTreeNode {
  id: number;
  leaf: boolean;
  depth: number;
  featureIndex?: number;
  featureName?: string;
  threshold?: number;
  gain?: number;
  weightedImpurityDecrease?: number;
  samples: number;
  negCount: number;
  posCount: number;
  gini: number;
  prob: number;
  pred: number;
  left?: DecisionTreeNode;
  right?: DecisionTreeNode;
}

export interface ApplicantRecord {
  rowIndex: number;
  applicantId: string;
  applicantIncome: number;
  coapplicantIncome: number;
  totalIncome: number;
  creditScore: number;
  dtiRatio: number;
  loanAmount: number;
  loanTenure: number;
  existingLoans: number;
  previousLoanDefaults: number;
  paymentHistory: string;
  employmentStability: number;
  monthlyExpenses: number;
  savingsBalance: number;
  emiToIncomeRatio: number;
  loanPurpose: string;
  collateralValue: number;
  actualApproved: "Yes" | "No" | "Unlabeled";
  dtProb: number;
  dtPred: "Yes" | "No";
  lrProb: number;
  lrPred: "Yes" | "No";
}

export interface LoanInputForm {
  applicantIncome: number;
  coapplicantIncome: number;
  creditScore: number;
  dtiRatio: number;
  loanAmount: number;
  loanTenure: number;
  existingLoans: number;
  previousLoanDefaults: number;
  paymentHistory: string;
  employmentStability: number;
  monthlyExpenses: number;
  savingsBalance: number;
  emiToIncomeRatio: number;
  loanPurpose: string;
  collateralValue: number;
  propertyArea: "Urban" | "Semiurban" | "Rural";
}

export interface DecisionPathStep {
  nodeId: number;
  depth: number;
  featureName: string;
  threshold: number;
  actualValue: number;
  direction: "left" | "right";
  samples: number;
  prob: number;
  gini: number;
}

export interface LogisticContribution {
  feature: string;
  rawValue: number;
  normalizedValue: number;
  weight: number;
  contribution: number;
}

export interface InferenceExplanation {
  decisionTree: {
    probability: number;
    prediction: "Yes" | "No";
    leafSamples: number;
    leafGini: number;
    path: DecisionPathStep[];
  };
  logisticRegression: {
    probability: number;
    prediction: "Yes" | "No";
    logit: number;
    bias: number;
    topContributions: LogisticContribution[];
  };
}

export interface TrainedModelsArtifact {
  datasetName: string;
  datasetSummary: {
    totalRecords: number;
    labeledRecords: number;
    unlabeledRecords: number;
    approvedCount: number;
    rejectedCount: number;
    creditScoreTrend: Array<{
      band: string;
      total: number;
      approved: number;
      rejected: number;
      approvalRate: number;
    }>;
    dtiTrend: Array<{
      band: string;
      total: number;
      approved: number;
      rejected: number;
      approvalRate: number;
    }>;
    incomeTrend: Array<{
      bracket: string;
      total: number;
      approved: number;
      rejected: number;
      approvalRate: number;
      avgLoanAmount: number;
    }>;
    categoricalBreakdowns: Record<
      string,
      Array<{
        category: string;
        total: number;
        approved: number;
        rejected: number;
        approvalRate: number;
      }>
    >;
    featureMeans: Array<{
      feature: string;
      approvedMean: number;
      rejectedMean: number;
      overallMean: number;
    }>;
  };
  splitInfo: {
    trainCount: number;
    testCount: number;
    splitRatio: string;
    randomState: number;
  };
  preprocessing: {
    numericCols: string[];
    categoricalCols: string[];
    engineeredCols: string[];
    medians: Record<string, number>;
    modes: Record<string, string>;
    catValues: Record<string, string[]>;
    featureNames: string[];
    scalerMeans: number[];
    scalerStds: number[];
  };
  models: {
    decisionTree: {
      name: string;
      hyperparameters: {
        criterion: string;
        maxDepth: number;
        minSamplesLeaf: number;
        randomState: number;
      };
      trainAccuracy: number;
      metrics: ModelMetrics;
      featureImportances: Array<{ feature: string; importance: number }>;
      tree: DecisionTreeNode;
      depthCurve: Array<{
        maxDepth: number;
        trainAccuracy: number;
        testAccuracy: number;
        testF1: number;
      }>;
    };
    logisticRegression: {
      name: string;
      hyperparameters: {
        penalty: string;
        regularizationLambda: number;
        learningRate: number;
        epochs: number;
      };
      trainAccuracy: number;
      metrics: ModelMetrics;
      bias: number;
      weights: number[];
      featureCoefficients: Array<{
        feature: string;
        coefficient: number;
        absMagnitude: number;
        oddsRatio: number;
      }>;
    };
  };
  crossValidation: {
    folds: number;
    logisticRegression: {
      foldAccuracies: number[];
      meanAccuracy: number;
      stdAccuracy: number;
      meanF1: number;
    };
    decisionTree: {
      foldAccuracies: number[];
      meanAccuracy: number;
      stdAccuracy: number;
      meanF1: number;
    };
  };
  applicants: ApplicantRecord[];
}
