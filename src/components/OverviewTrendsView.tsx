import React, { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MODEL_ARTIFACT } from "../utils/inference";
import type { ModelKey } from "../types/creditwise";

interface OverviewTrendsViewProps {
  activeModel: ModelKey;
  onSelectModel: (model: ModelKey) => void;
}

type SegmentDim =
  | "Payment_History"
  | "Loan_Purpose"
  | "Previous_Loan_Defaults"
  | "Existing_Loans";

export const OverviewTrendsView: React.FC<OverviewTrendsViewProps> = ({
  activeModel,
}) => {
  const [paymentFilter, setPaymentFilter] = useState<string>("All");
  const [purposeFilter, setPurposeFilter] = useState<string>("All");
  const [incomeChartMode, setIncomeChartMode] = useState<"brackets" | "scatter">("brackets");
  const [breakdownDim, setBreakdownDim] = useState<SegmentDim>("Payment_History");

  const labeledApplicants = useMemo(() => {
    return MODEL_ARTIFACT.applicants.filter((a) => {
      if (a.actualApproved === "Unlabeled") return false;
      if (paymentFilter !== "All" && a.paymentHistory !== paymentFilter) return false;
      if (purposeFilter !== "All" && a.loanPurpose !== purposeFilter) return false;
      return true;
    });
  }, [paymentFilter, purposeFilter]);

  const datasetMeans = useMemo(() => {
    const cs = MODEL_ARTIFACT.datasetSummary.featureMeans.find(
      (f) => f.feature === "Credit_Score"
    );
    const dti = MODEL_ARTIFACT.datasetSummary.featureMeans.find(
      (f) => f.feature === "DTI_Ratio"
    );
    return {
      csApproved: cs ? cs.approvedMean.toFixed(1) : "0",
      csRejected: cs ? cs.rejectedMean.toFixed(1) : "0",
      dtiApproved: dti ? dti.approvedMean.toFixed(1) : "0",
      dtiRejected: dti ? dti.rejectedMean.toFixed(1) : "0",
    };
  }, []);

  const summaryKpis = useMemo(() => {
    const total = labeledApplicants.length;
    const approved = labeledApplicants.filter((a) => a.actualApproved === "Yes").length;
    const avgCs = total
      ? Math.round(labeledApplicants.reduce((s, a) => s + a.creditScore, 0) / total)
      : 0;
    const avgDti = total
      ? labeledApplicants.reduce((s, a) => s + a.dtiRatio, 0) / total
      : 0;
    const activeMetrics = MODEL_ARTIFACT.models[activeModel].metrics;
    return {
      total,
      approved,
      approvalRate: total ? ((approved / total) * 100).toFixed(1) : "0.0",
      avgCs,
      avgDti: avgDti.toFixed(1),
      modelAcc: (activeMetrics.accuracy * 100).toFixed(2),
      modelAuc: activeMetrics.rocAuc.toFixed(4),
    };
  }, [labeledApplicants, activeModel]);

  const creditScoreTrend = useMemo(() => {
    const bands = [
      { label: "<580", min: 300, max: 580 },
      { label: "580–639", min: 580, max: 640 },
      { label: "640–699", min: 640, max: 700 },
      { label: "700–759", min: 700, max: 760 },
      { label: "760+", min: 760, max: 900 },
    ];
    return bands.map((b) => {
      const sub = labeledApplicants.filter(
        (a) => a.creditScore >= b.min && a.creditScore < b.max
      );
      const app = sub.filter((a) => a.actualApproved === "Yes").length;
      const dtPredApp = sub.filter((a) => a.dtPred === "Yes").length;
      const lrPredApp = sub.filter((a) => a.lrPred === "Yes").length;
      return {
        band: b.label,
        total: sub.length,
        approved: app,
        rejected: sub.length - app,
        actualRate: sub.length ? Number(((app / sub.length) * 100).toFixed(1)) : 0,
        dtRate: sub.length ? Number(((dtPredApp / sub.length) * 100).toFixed(1)) : 0,
        lrRate: sub.length ? Number(((lrPredApp / sub.length) * 100).toFixed(1)) : 0,
      };
    });
  }, [labeledApplicants]);

  const dtiTrend = useMemo(() => {
    const bands = [
      { label: "0–5%", min: 0, max: 5 },
      { label: "5–10%", min: 5, max: 10 },
      { label: "10–20%", min: 10, max: 20 },
      { label: "20–30%", min: 20, max: 30 },
      { label: "30%+", min: 30, max: 75 },
    ];
    return bands.map((b) => {
      const sub = labeledApplicants.filter(
        (a) => a.dtiRatio >= b.min && a.dtiRatio < b.max
      );
      const app = sub.filter((a) => a.actualApproved === "Yes").length;
      const avgDtProb = sub.length
        ? (sub.reduce((s, a) => s + a.dtProb, 0) / sub.length) * 100
        : 0;
      const avgLrProb = sub.length
        ? (sub.reduce((s, a) => s + a.lrProb, 0) / sub.length) * 100
        : 0;
      return {
        band: b.label,
        total: sub.length,
        actualRate: sub.length ? Number(((app / sub.length) * 100).toFixed(1)) : 0,
        dtProb: Number(avgDtProb.toFixed(1)),
        lrProb: Number(avgLrProb.toFixed(1)),
      };
    });
  }, [labeledApplicants]);

  const incomeTrend = useMemo(() => {
    const brackets = [
      { label: "<$30k", min: 0, max: 30000 },
      { label: "$30k–$50k", min: 30000, max: 50000 },
      { label: "$50k–$75k", min: 50000, max: 75000 },
      { label: "$75k–$110k", min: 75000, max: 110000 },
      { label: "$110k+", min: 110000, max: 2000000 },
    ];
    return brackets.map((b) => {
      const sub = labeledApplicants.filter(
        (a) => a.totalIncome >= b.min && a.totalIncome < b.max
      );
      const app = sub.filter((a) => a.actualApproved === "Yes").length;
      const avgLoan = sub.length
        ? Math.round(sub.reduce((s, a) => s + a.loanAmount, 0) / sub.length)
        : 0;
      return {
        bracket: b.label,
        total: sub.length,
        approved: app,
        rejected: sub.length - app,
        approvalRate: sub.length ? Number(((app / sub.length) * 100).toFixed(1)) : 0,
        avgLoan,
      };
    });
  }, [labeledApplicants]);

  const scatterData = useMemo(() => {
    const sampled = labeledApplicants.slice(0, 320);
    return {
      approved: sampled
        .filter((a) => a.actualApproved === "Yes")
        .map((a) => ({
          creditScore: a.creditScore,
          dtiPct: Number(a.dtiRatio.toFixed(1)),
          loanAmount: a.loanAmount,
          id: a.applicantId,
        })),
      rejected: sampled
        .filter((a) => a.actualApproved === "No")
        .map((a) => ({
          creditScore: a.creditScore,
          dtiPct: Number(a.dtiRatio.toFixed(1)),
          loanAmount: a.loanAmount,
          id: a.applicantId,
        })),
    };
  }, [labeledApplicants]);

  const categoricalBreakdown = useMemo(() => {
    const groups = new Map<string, { total: number; approved: number }>();
    for (const row of labeledApplicants) {
      let val = "";
      if (breakdownDim === "Payment_History") val = row.paymentHistory;
      else if (breakdownDim === "Loan_Purpose") val = row.loanPurpose;
      else if (breakdownDim === "Previous_Loan_Defaults")
        val = `${row.previousLoanDefaults} ${row.previousLoanDefaults === 1 ? "default" : "defaults"}`;
      else val = `${row.existingLoans} ${row.existingLoans === 1 ? "loan" : "loans"}`;

      const curr = groups.get(val) || { total: 0, approved: 0 };
      curr.total += 1;
      if (row.actualApproved === "Yes") curr.approved += 1;
      groups.set(val, curr);
    }
    return Array.from(groups.entries())
      .map(([category, stats]) => ({
        category,
        total: stats.total,
        approved: stats.approved,
        rejected: stats.total - stats.approved,
        approvalRate: stats.total ? Number(((stats.approved / stats.total) * 100).toFixed(1)) : 0,
      }))
      .sort((a, b) => b.approvalRate - a.approvalRate);
  }, [labeledApplicants, breakdownDim]);

  const activeModelLabel =
    activeModel === "decisionTree" ? "Decision Tree (CART)" : "Logistic Regression (L2)";

  return (
    <div className="space-y-6">
      {/* Cohort Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800/80 bg-[#111827] px-5 py-3.5">
        <div className="flex flex-wrap items-center gap-6">
          <div className="flex items-center gap-2.5">
            <span className="text-xs font-medium text-slate-400">Payment History</span>
            <div className="flex items-center gap-1 rounded-lg bg-[#0B0F17] p-1 border border-slate-800">
              {["All", "Excellent", "Good", "Average", "Poor"].map((opt) => (
                <button
                  key={opt}
                  onClick={() => setPaymentFilter(opt)}
                  className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                    paymentFilter === opt
                      ? "bg-sky-500/20 text-sky-300 border border-sky-500/30"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <span className="text-xs font-medium text-slate-400">Loan Purpose</span>
            <select
              value={purposeFilter}
              onChange={(e) => setPurposeFilter(e.target.value)}
              aria-label="Filter by Loan Purpose"
              className="rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-1.5 text-xs font-medium text-slate-200 focus:border-sky-500 focus:outline-none"
            >
              <option value="All">All Purposes</option>
              <option value="Home">Home</option>
              <option value="Vehicle">Vehicle</option>
              <option value="Personal">Personal</option>
              <option value="Business">Business</option>
              <option value="Education">Education</option>
              <option value="Medical">Medical</option>
            </select>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-400">
          <span>Active Cohort:</span>
          <span className="font-mono font-semibold text-slate-100 tabular-nums">
            {summaryKpis.total.toLocaleString()} /{" "}
            {MODEL_ARTIFACT.datasetSummary.labeledRecords.toLocaleString()} labeled
          </span>
          {(paymentFilter !== "All" || purposeFilter !== "All") && (
            <button
              onClick={() => {
                setPaymentFilter("All");
                setPurposeFilter("All");
              }}
              className="text-sky-400 hover:text-sky-300 underline underline-offset-2 whitespace-nowrap"
            >
              Reset
            </button>
          )}
        </div>
      </div>

      {/* 4-Column KPI Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="text-xs font-medium text-slate-400">Cohort Approval Rate</div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="font-mono text-2xl font-semibold text-slate-100 tabular-nums">
              {summaryKpis.approvalRate}%
            </span>
            <span className="font-mono text-xs text-emerald-400 tabular-nums">
              {summaryKpis.approved.toLocaleString()} approved
            </span>
          </div>
          <div className="mt-2 text-xs text-slate-500">
            Out of {summaryKpis.total.toLocaleString()} screened records
          </div>
        </div>

        <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="text-xs font-medium text-slate-400">Mean Credit Score</div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="font-mono text-2xl font-semibold text-slate-100 tabular-nums">
              {summaryKpis.avgCs}
            </span>
            <span className="font-mono text-xs text-slate-400 tabular-nums">
              Range 341–889
            </span>
          </div>
          <div className="mt-2 text-xs text-slate-500">
            Approved avg: {datasetMeans.csApproved} · Rejected avg: {datasetMeans.csRejected}
          </div>
        </div>

        <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="text-xs font-medium text-slate-400">Mean Debt-to-Income</div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="font-mono text-2xl font-semibold text-slate-100 tabular-nums">
              {summaryKpis.avgDti}%
            </span>
            <span className="font-mono text-xs text-amber-400 tabular-nums">
              Range 0–62%
            </span>
          </div>
          <div className="mt-2 text-xs text-slate-500">
            Approved avg: {datasetMeans.dtiApproved}% · Rejected avg: {datasetMeans.dtiRejected}%
          </div>
        </div>

        <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="text-xs font-medium text-slate-400">{activeModelLabel}</div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="font-mono text-2xl font-semibold text-sky-400 tabular-nums">
              {summaryKpis.modelAcc}%
            </span>
            <span className="font-mono text-xs text-slate-300 tabular-nums">
              AUC {summaryKpis.modelAuc}
            </span>
          </div>
          <div className="mt-2 text-xs text-slate-500">
            Holdout validation (n = {MODEL_ARTIFACT.splitInfo.testCount.toLocaleString()})
          </div>
        </div>
      </div>

      {/* Primary Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Chart 1: Credit Score Approval Trajectory */}
        <div className="lg:col-span-7 rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="flex flex-wrap items-start justify-between gap-2 mb-4">
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                Credit Score Band Approval Trajectory
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Empirical approval volume vs. predicted approval rate overlay ({activeModelLabel})
              </p>
            </div>
            <div className="text-xs text-slate-400 font-mono tabular-nums">
              Inflection ~640–700
            </div>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={creditScoreTrend} margin={{ top: 10, right: 16, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="band" stroke="#94A3B8" fontSize={12} />
                <YAxis yAxisId="left" stroke="#94A3B8" fontSize={12} />
                <YAxis yAxisId="right" orientation="right" stroke="#38BDF8" fontSize={12} unit="%" domain={[0, 100]} />
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
                <Bar yAxisId="left" dataKey="approved" name="Approved" stackId="a" fill="#10B981" radius={[0, 0, 0, 0]} />
                <Bar yAxisId="left" dataKey="rejected" name="Rejected" stackId="a" fill="#334155" radius={[4, 4, 0, 0]} />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="actualRate"
                  name="Actual Approval %"
                  stroke="#F59E0B"
                  strokeWidth={2}
                  dot={{ r: 4, fill: "#F59E0B" }}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey={activeModel === "decisionTree" ? "dtRate" : "lrRate"}
                  name={activeModel === "decisionTree" ? "DT Predicted %" : "LR Predicted %"}
                  stroke="#38BDF8"
                  strokeWidth={2.5}
                  strokeDasharray="4 2"
                  dot={{ r: 4, fill: "#38BDF8" }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: DTI Ratio Risk Decay Curve */}
        <div className="lg:col-span-5 rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="mb-4">
            <h2 className="text-base font-semibold text-slate-100">
              Debt-to-Income (DTI) Risk Decay Curve
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Comparing actual approval rate with Decision Tree & Logistic Regression probabilities
            </p>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={dtiTrend} margin={{ top: 10, right: 12, left: -16, bottom: 0 }}>
                <defs>
                  <linearGradient id="actualDtiGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="band" stroke="#94A3B8" fontSize={12} />
                <YAxis stroke="#94A3B8" fontSize={12} unit="%" domain={[0, 100]} />
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
                <ReferenceLine x="20–30%" stroke="#F43F5E" strokeDasharray="3 3" />
                <Area
                  type="monotone"
                  dataKey="actualRate"
                  name="Actual Rate %"
                  stroke="#10B981"
                  strokeWidth={2}
                  fill="url(#actualDtiGrad)"
                />
                <Line
                  type="monotone"
                  dataKey="dtProb"
                  name="Decision Tree Prob %"
                  stroke="#38BDF8"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="lrProb"
                  name="Logistic Reg Prob %"
                  stroke="#F59E0B"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Secondary Trend & Segmentation Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Chart 3: Income Bracket vs Scatter Explorer */}
        <div className="lg:col-span-6 rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                {incomeChartMode === "brackets"
                  ? "Total Income Bracket & Approval Volume"
                  : "Credit Score vs. DTI Decision Boundary Map"}
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                {incomeChartMode === "brackets"
                  ? "Combined applicant + co-applicant monthly income brackets vs. approval rate"
                  : "Sampled applicants showing risk separation across Credit Score and DTI %"}
              </p>
            </div>
            <div className="flex items-center gap-1 rounded-lg bg-[#0B0F17] p-1 border border-slate-800">
              <button
                onClick={() => setIncomeChartMode("brackets")}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                  incomeChartMode === "brackets"
                    ? "bg-slate-800 text-slate-100"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Income Trend
              </button>
              <button
                onClick={() => setIncomeChartMode("scatter")}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                  incomeChartMode === "scatter"
                    ? "bg-slate-800 text-slate-100"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Boundary Scatter
              </button>
            </div>
          </div>

          <div className="h-72 w-full">
            {incomeChartMode === "brackets" ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={incomeTrend} margin={{ top: 10, right: 12, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="bracket" stroke="#94A3B8" fontSize={12} />
                  <YAxis yAxisId="left" stroke="#94A3B8" fontSize={12} />
                  <YAxis yAxisId="right" orientation="right" stroke="#38BDF8" fontSize={12} unit="%" domain={[0, 100]} />
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
                  <Bar yAxisId="left" dataKey="approved" name="Approved" fill="#10B981" radius={[3, 3, 0, 0]} />
                  <Bar yAxisId="left" dataKey="rejected" name="Rejected" fill="#475569" radius={[3, 3, 0, 0]} />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="approvalRate"
                    name="Approval Rate %"
                    stroke="#38BDF8"
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: "#38BDF8" }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 12, left: -12, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis
                    type="number"
                    dataKey="creditScore"
                    name="Credit Score"
                    domain={[340, 900]}
                    stroke="#94A3B8"
                    fontSize={12}
                  />
                  <YAxis
                    type="number"
                    dataKey="dtiPct"
                    name="DTI %"
                    unit="%"
                    domain={[0, 65]}
                    stroke="#94A3B8"
                    fontSize={12}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    contentStyle={{
                      backgroundColor: "#0F172A",
                      borderColor: "#334155",
                      borderRadius: "8px",
                      fontSize: "12px",
                      color: "#F1F5F9",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <ReferenceLine x={680} stroke="#38BDF8" strokeDasharray="4 4" />
                  <ReferenceLine y={20} stroke="#F59E0B" strokeDasharray="4 4" />
                  <Scatter name="Approved" data={scatterData.approved} fill="#10B981" fillOpacity={0.8} />
                  <Scatter name="Rejected" data={scatterData.rejected} fill="#F43F5E" fillOpacity={0.55} />
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Chart 4: Portfolio Dimension Breakdown */}
        <div className="lg:col-span-6 rounded-xl border border-slate-800/80 bg-[#111827] p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                Segment Approval Breakdown
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Approval rate (%) across credit history, loan purpose, defaults, and existing loans
              </p>
            </div>
            <select
              value={breakdownDim}
              onChange={(e) => setBreakdownDim(e.target.value as SegmentDim)}
              aria-label="Select Breakdown Dimension"
              className="rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-1.5 text-xs font-medium text-slate-200 focus:border-sky-500 focus:outline-none"
            >
              <option value="Payment_History">Payment History</option>
              <option value="Loan_Purpose">Loan Purpose</option>
              <option value="Previous_Loan_Defaults">Previous Loan Defaults</option>
              <option value="Existing_Loans">Existing Loans</option>
            </select>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={categoricalBreakdown}
                layout="vertical"
                margin={{ top: 6, right: 24, left: 20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} unit="%" stroke="#94A3B8" fontSize={12} />
                <YAxis type="category" dataKey="category" stroke="#CBD5E1" fontSize={12} width={96} />
                <Tooltip
                  formatter={(value: unknown, _name: unknown, props: { payload?: { approved: number; total: number } }) => {
                    const p = props?.payload;
                    return [`${value}% (${p?.approved ?? 0}/${p?.total ?? 0} approved)`, "Approval Rate"];
                  }}
                  contentStyle={{
                    backgroundColor: "#0F172A",
                    borderColor: "#334155",
                    borderRadius: "8px",
                    fontSize: "12px",
                    color: "#F1F5F9",
                  }}
                />
                <Bar dataKey="approvalRate" name="Approval Rate %" radius={[0, 4, 4, 0]} barSize={22}>
                  {categoricalBreakdown.map((entry, idx) => (
                    <Cell
                      key={entry.category}
                      fill={idx === 0 ? "#38BDF8" : entry.approvalRate >= 55 ? "#10B981" : "#64748B"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
