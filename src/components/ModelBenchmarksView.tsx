import React, { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Download } from "lucide-react";
import { MODEL_ARTIFACT, formatFeatureLabel } from "../utils/inference";
import type { DecisionTreeNode, ModelKey } from "../types/creditwise";

interface ModelBenchmarksViewProps {
  activeModel: ModelKey;
  onSelectModel: (model: ModelKey) => void;
}

function flattenTopSplits(node: DecisionTreeNode, out: Array<{ depth: number; rule: string; samples: number; prob: number; gini: number }> = []) {
  if (node.leaf || out.length >= 10) return out;
  out.push({
    depth: node.depth,
    rule: `${formatFeatureLabel(node.featureName || "")} ≤ ${node.threshold?.toFixed(2)}`,
    samples: node.samples,
    prob: node.prob,
    gini: node.gini,
  });
  if (node.left) flattenTopSplits(node.left, out);
  if (node.right) flattenTopSplits(node.right, out);
  return out;
}

export const ModelBenchmarksView: React.FC<ModelBenchmarksViewProps> = ({
  activeModel,
  onSelectModel,
}) => {
  const dt = MODEL_ARTIFACT.models.decisionTree;
  const lr = MODEL_ARTIFACT.models.logisticRegression;
  const cv = MODEL_ARTIFACT.crossValidation;

  // Interpolate ROC Curves on unified FPR grid 0.00..1.00 for clean dual-line plotting
  const unifiedRoc = useMemo(() => {
    const grid = Array.from({ length: 21 }, (_, i) => Number((i * 0.05).toFixed(2)));
    const interp = (curve: Array<{ fpr: number; tpr: number }>, targetFpr: number) => {
      let best = 0;
      for (const pt of curve) {
        if (pt.fpr <= targetFpr && pt.tpr > best) best = pt.tpr;
      }
      return Number(best.toFixed(4));
    };
    return grid.map((fpr) => ({
      fpr,
      dtTpr: interp(dt.metrics.rocCurve, fpr),
      lrTpr: interp(lr.metrics.rocCurve, fpr),
      baseline: fpr,
    }));
  }, [dt.metrics.rocCurve, lr.metrics.rocCurve]);

  const cvComparisonData = useMemo(() => {
    return cv.decisionTree.foldAccuracies.map((acc, i) => ({
      fold: `Fold ${i + 1}`,
      decisionTree: Number((acc * 100).toFixed(2)),
      logisticRegression: Number((cv.logisticRegression.foldAccuracies[i] * 100).toFixed(2)),
    }));
  }, [cv]);

  const topSplits = useMemo(() => flattenTopSplits(dt.tree), [dt.tree]);

  const featureRankingData = useMemo(() => {
    if (activeModel === "decisionTree") {
      return dt.featureImportances.slice(0, 8).map((f) => ({
        feature: formatFeatureLabel(f.feature),
        value: Number((f.importance * 100).toFixed(2)),
      }));
    }
    return lr.featureCoefficients.slice(0, 8).map((f) => ({
      feature: formatFeatureLabel(f.feature),
      value: Number(f.coefficient.toFixed(3)),
    }));
  }, [activeModel, dt.featureImportances, lr.featureCoefficients]);

  return (
    <div className="space-y-6">
      {/* Head-to-Head Comparison Table */}
      <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-5">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Model Benchmark & Cross-Validation Summary
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Evaluated on stratified 80/20 holdout split (train n={MODEL_ARTIFACT.splitInfo.trainCount}, test n={MODEL_ARTIFACT.splitInfo.testCount})
            </p>
          </div>
          <div className="flex items-center gap-1 rounded-lg bg-[#0B0F17] p-1 border border-slate-800">
            <button
              onClick={() => onSelectModel("decisionTree")}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                activeModel === "decisionTree"
                  ? "bg-sky-500 text-slate-950 font-semibold"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              Inspect Decision Tree
            </button>
            <button
              onClick={() => onSelectModel("logisticRegression")}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                activeModel === "logisticRegression"
                  ? "bg-sky-500 text-slate-950 font-semibold"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              Inspect Logistic Regression
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3">Classifier</th>
                <th className="py-2.5 px-3 text-right">Test Accuracy</th>
                <th className="py-2.5 px-3 text-right">Precision</th>
                <th className="py-2.5 px-3 text-right">Recall</th>
                <th className="py-2.5 px-3 text-right">F1-Score</th>
                <th className="py-2.5 px-3 text-right">ROC-AUC</th>
                <th className="py-2.5 px-3 text-right">Log Loss</th>
                <th className="py-2.5 px-3 text-right">5-Fold CV Mean ± SD</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
              <tr className={activeModel === "decisionTree" ? "bg-sky-500/10" : ""}>
                <td className="py-3 px-3 font-sans font-semibold text-sky-400">
                  Decision Tree (CART, depth={dt.hyperparameters.maxDepth})
                </td>
                <td className="py-3 px-3 text-right text-slate-100 font-semibold">
                  {(dt.metrics.accuracy * 100).toFixed(2)}%
                </td>
                <td className="py-3 px-3 text-right text-slate-200">
                  {(dt.metrics.precision * 100).toFixed(2)}%
                </td>
                <td className="py-3 px-3 text-right text-emerald-400 font-semibold">
                  {(dt.metrics.recall * 100).toFixed(2)}%
                </td>
                <td className="py-3 px-3 text-right text-slate-100">
                  {(dt.metrics.f1 * 100).toFixed(2)}%
                </td>
                <td className="py-3 px-3 text-right text-sky-400 font-semibold">
                  {dt.metrics.rocAuc.toFixed(4)}
                </td>
                <td className="py-3 px-3 text-right text-slate-300">
                  {dt.metrics.logLoss.toFixed(4)}
                </td>
                <td className="py-3 px-3 text-right text-slate-200">
                  {(cv.decisionTree.meanAccuracy * 100).toFixed(2)}% ±{" "}
                  {(cv.decisionTree.stdAccuracy * 100).toFixed(2)}%
                </td>
              </tr>
              <tr className={activeModel === "logisticRegression" ? "bg-sky-500/10" : ""}>
                <td className="py-3 px-3 font-sans font-semibold text-amber-400">
                  Logistic Regression (L2, λ=0.01)
                </td>
                <td className="py-3 px-3 text-right text-slate-100 font-semibold">
                  {(lr.metrics.accuracy * 100).toFixed(2)}%
                </td>
                <td className="py-3 px-3 text-right text-slate-200">
                  {(lr.metrics.precision * 100).toFixed(2)}%
                </td>
                <td className="py-3 px-3 text-right text-slate-200">
                  {(lr.metrics.recall * 100).toFixed(2)}%
                </td>
                <td className="py-3 px-3 text-right text-slate-100">
                  {(lr.metrics.f1 * 100).toFixed(2)}%
                </td>
                <td className="py-3 px-3 text-right text-amber-400 font-semibold">
                  {lr.metrics.rocAuc.toFixed(4)}
                </td>
                <td className="py-3 px-3 text-right text-slate-300">
                  {lr.metrics.logLoss.toFixed(4)}
                </td>
                <td className="py-3 px-3 text-right text-slate-200">
                  {(cv.logisticRegression.meanAccuracy * 100).toFixed(2)}% ±{" "}
                  {(cv.logisticRegression.stdAccuracy * 100).toFixed(2)}%
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* ROC Curve + Confusion Matrices */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-7 rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="mb-4">
            <h3 className="text-base font-semibold text-slate-100">
              ROC Curve Comparison
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              True Positive Rate vs. False Positive Rate across probability thresholds
            </p>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={unifiedRoc} margin={{ top: 10, right: 16, left: -12, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="fpr" stroke="#94A3B8" fontSize={11} />
                <YAxis stroke="#94A3B8" fontSize={11} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0F172A",
                    borderColor: "#334155",
                    borderRadius: "8px",
                    fontSize: "12px",
                    color: "#F1F5F9",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  type="monotone"
                  dataKey="dtTpr"
                  name={`Decision Tree (AUC ${dt.metrics.rocAuc.toFixed(4)})`}
                  stroke="#38BDF8"
                  strokeWidth={2.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="lrTpr"
                  name={`Logistic Regression (AUC ${lr.metrics.rocAuc.toFixed(4)})`}
                  stroke="#F59E0B"
                  strokeWidth={2.5}
                  dot={false}
                />
                <Line
                  type="linear"
                  dataKey="baseline"
                  name="Random Classifier (0.50)"
                  stroke="#475569"
                  strokeDasharray="4 4"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrices Side-by-Side */}
        <div className="lg:col-span-5 rounded-xl border border-slate-800/80 bg-[#111827] p-5 space-y-5">
          <div>
            <h3 className="text-base font-semibold text-slate-100">
              Holdout Confusion Matrices (n = {MODEL_ARTIFACT.splitInfo.testCount.toLocaleString()})
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Decision Tree FN: {dt.metrics.confusionMatrix.fn} vs. Logistic Regression FN: {lr.metrics.confusionMatrix.fn}
            </p>
          </div>

          {[
            { label: "Decision Tree Classifier", cm: dt.metrics.confusionMatrix, accent: "text-sky-400" },
            { label: "Logistic Regression", cm: lr.metrics.confusionMatrix, accent: "text-amber-400" },
          ].map((item) => (
            <div key={item.label} className="space-y-2">
              <div className={`text-xs font-semibold ${item.accent}`}>{item.label}</div>
              <div className="grid grid-cols-3 gap-2 text-xs text-center font-mono tabular-nums">
                <div className="p-2 text-slate-500 text-left font-sans">Actual \ Pred</div>
                <div className="p-2 text-slate-400 font-sans">Pred: No</div>
                <div className="p-2 text-slate-400 font-sans">Pred: Yes</div>

                <div className="p-2.5 text-left text-slate-400 font-sans">Actual: No</div>
                <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-2.5 text-emerald-300 font-semibold">
                  TN {item.cm.tn}
                </div>
                <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-2.5 text-rose-300">
                  FP {item.cm.fp}
                </div>

                <div className="p-2.5 text-left text-slate-400 font-sans">Actual: Yes</div>
                <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-2.5 text-rose-300">
                  FN {item.cm.fn}
                </div>
                <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-2.5 text-emerald-300 font-semibold">
                  TP {item.cm.tp}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature Attributions + Depth/CV Curves */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-6 rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="mb-4">
            <h3 className="text-base font-semibold text-slate-100">
              {activeModel === "decisionTree"
                ? "Decision Tree Gini Feature Importance (%)"
                : "Logistic Regression Standardized Coefficients"}
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              {activeModel === "decisionTree"
                ? "Normalized total Gini impurity reduction across all decision nodes"
                : "Positive weights raise approval log-odds; negative weights reduce approval log-odds"}
            </p>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={featureRankingData}
                layout="vertical"
                margin={{ top: 4, right: 20, left: 30, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" horizontal={false} />
                <XAxis type="number" stroke="#94A3B8" fontSize={11} />
                <YAxis type="category" dataKey="feature" stroke="#CBD5E1" fontSize={11} width={115} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0F172A",
                    borderColor: "#334155",
                    borderRadius: "8px",
                    fontSize: "12px",
                    color: "#F1F5F9",
                  }}
                />
                <ReferenceLine x={0} stroke="#64748B" />
                <Bar dataKey="value" name={activeModel === "decisionTree" ? "Importance %" : "Weight"} barSize={18} radius={[3, 3, 3, 3]}>
                  {featureRankingData.map((entry) => (
                    <Cell
                      key={entry.feature}
                      fill={
                        activeModel === "decisionTree"
                          ? "#38BDF8"
                          : entry.value >= 0
                          ? "#10B981"
                          : "#F43F5E"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="lg:col-span-6 rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="mb-4">
            <h3 className="text-base font-semibold text-slate-100">
              {activeModel === "decisionTree"
                ? "Decision Tree Max-Depth Tuning Curve (1–8)"
                : "5-Fold Stratified Cross-Validation Stability"}
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              {activeModel === "decisionTree"
                ? `Training vs. Holdout Test Accuracy across tree depths (configured max_depth = ${dt.hyperparameters.maxDepth})`
                : "Per-fold accuracy comparing Decision Tree and Logistic Regression"}
            </p>
          </div>
          <div className="h-72 w-full">
            {activeModel === "decisionTree" ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dt.depthCurve} margin={{ top: 10, right: 16, left: -12, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="maxDepth" stroke="#94A3B8" fontSize={11} />
                  <YAxis stroke="#94A3B8" fontSize={11} domain={["auto", 100]} unit="%" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0F172A",
                      borderColor: "#334155",
                      borderRadius: "8px",
                      fontSize: "12px",
                      color: "#F1F5F9",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <ReferenceLine x={dt.hyperparameters.maxDepth} stroke="#10B981" strokeDasharray="4 4" />
                  <Line type="monotone" dataKey="trainAccuracy" name="Train Accuracy %" stroke="#94A3B8" strokeWidth={2} />
                  <Line type="monotone" dataKey="testAccuracy" name="Test Accuracy %" stroke="#38BDF8" strokeWidth={2.5} />
                  <Line type="monotone" dataKey="testF1" name="Test F1 %" stroke="#F59E0B" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={cvComparisonData} margin={{ top: 10, right: 16, left: -12, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="fold" stroke="#94A3B8" fontSize={11} />
                  <YAxis stroke="#94A3B8" fontSize={11} domain={["auto", 100]} unit="%" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0F172A",
                      borderColor: "#334155",
                      borderRadius: "8px",
                      fontSize: "12px",
                      color: "#F1F5F9",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <Bar dataKey="decisionTree" name="Decision Tree %" fill="#38BDF8" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="logisticRegression" name="Logistic Regression %" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Primary CART Split Rules Reference */}
      <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-5">
        <div className="mb-3">
          <h3 className="text-sm font-semibold text-slate-100">
            Extracted CART Decision Tree Split Hierarchy (Top Nodes)
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Human-readable underwriting thresholds discovered directly from the training fold
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-xs">
          {topSplits.map((sp, i) => (
            <div
              key={`${sp.depth}-${i}`}
              className="flex items-center justify-between rounded-lg border border-slate-800 bg-[#0B0F17] px-3.5 py-2.5"
            >
              <div className="flex items-center gap-2.5">
                <span className="font-mono text-slate-500">D{sp.depth}</span>
                <span className="font-mono text-sky-400">{sp.rule}</span>
              </div>
              <div className="font-mono text-slate-400 tabular-nums">
                n={sp.samples} · gini {sp.gini.toFixed(3)} · p={(sp.prob * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Notebook Chart Exports (Confusion Matrix Heatmaps & Metric Graphs) */}
      <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Notebook Evaluation Figures (PNG Exports)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Publication-ready Confusion Matrix Heatmaps and Accuracy, Precision, Recall &amp; F1-score plots generated from loan_approval_hybrid_dataset_20711486.csv
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            {
              title: "1. Decision Tree — Confusion Matrix Heatmap",
              src: "/charts/decision_tree_confusion_matrix.png",
              file: "decision_tree_confusion_matrix.png",
            },
            {
              title: "1. Decision Tree Results (Accuracy, Precision, Recall, F1)",
              src: "/charts/decision_tree_metrics.png",
              file: "decision_tree_metrics.png",
            },
            {
              title: "2. Logistic Regression — Confusion Matrix Heatmap",
              src: "/charts/logistic_regression_confusion_matrix.png",
              file: "logistic_regression_confusion_matrix.png",
            },
            {
              title: "2. Logistic Regression Results (Accuracy, Precision, Recall, F1)",
              src: "/charts/logistic_regression_metrics.png",
              file: "logistic_regression_metrics.png",
            },
          ].map((chart) => (
            <div
              key={chart.file}
              className="rounded-lg border border-slate-800 bg-[#0B0F17] p-3.5 space-y-3"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-slate-200 truncate">
                  {chart.title}
                </span>
                <a
                  href={chart.src}
                  download={chart.file}
                  className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-[#111827] px-2.5 py-1 text-xs font-medium text-sky-400 hover:border-sky-500/60 hover:text-sky-300 transition-colors whitespace-nowrap shrink-0"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download PNG</span>
                </a>
              </div>
              <div className="overflow-hidden rounded-md border border-slate-800 bg-white">
                <img
                  src={chart.src}
                  alt={chart.title}
                  className="w-full h-auto object-contain"
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
