import React, { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Play,
  RotateCcw,
  XCircle,
} from "lucide-react";
import {
  DEFAULT_LOAN_INPUT,
  formatFeatureLabel,
  scoreApplicant,
} from "../utils/inference";
import type {
  InferenceExplanation,
  LoanInputForm,
  ModelKey,
} from "../types/creditwise";

interface PredictorViewProps {
  activeModel: ModelKey;
  onSelectModel: (model: ModelKey) => void;
}

interface EvaluatedSnapshot {
  form: LoanInputForm;
  existingLoanEmis: number[];
  explanation: InferenceExplanation;
}

const DEFAULT_EXISTING_EMIS = [4500, 3500, 2500, 2000, 1500];
const LOAN_TENURES = [12, 24, 36, 48, 60, 72, 84, 120, 144, 180, 240, 300, 360];

function getInterestRateByArea(propertyArea: string): { rate: number; label: string } {
  if (propertyArea === "Rural") {
    return { rate: 5.0, label: "5.0% p.a. (Rural Concessional)" };
  }
  if (propertyArea === "Semiurban") {
    return { rate: 7.0, label: "7.0% p.a. (Semi-Urban Concessional)" };
  }
  return { rate: 10.5, label: "10.5% p.a. (Urban Standard)" };
}

function computeAmortization(principal: number, annualRatePct: number, months: number) {
  if (principal <= 0 || months <= 0) {
    return {
      monthlyEmi: 0,
      annualEmi: 0,
      totalRepayment: 0,
      totalExtraInterest: 0,
      annualExtraInterest: 0,
    };
  }
  const r = annualRatePct / 1200;
  const factor = Math.pow(1 + r, months);
  const monthlyEmi = r > 0 ? (principal * r * factor) / (factor - 1) : principal / months;
  const annualEmi = monthlyEmi * Math.min(12, months);
  const totalRepayment = monthlyEmi * months;
  const totalExtraInterest = Math.max(0, totalRepayment - principal);
  const years = months / 12;
  const annualExtraInterest = years > 0 ? totalExtraInterest / years : totalExtraInterest;
  return {
    monthlyEmi,
    annualEmi,
    totalRepayment,
    totalExtraInterest,
    annualExtraInterest,
  };
}

export const PredictorView: React.FC<PredictorViewProps> = ({
  activeModel,
  onSelectModel,
}) => {
  const [form, setForm] = useState<LoanInputForm>({ ...DEFAULT_LOAN_INPUT });
  const [existingLoanEmis, setExistingLoanEmis] = useState<number[]>(DEFAULT_EXISTING_EMIS);
  const [evaluated, setEvaluated] = useState<EvaluatedSnapshot | null>(null);
  const [isDirty, setIsDirty] = useState<boolean>(false);

  // Live computed EMI-to-Income preview based on selected loan, tenure, area rate, and existing loan EMIs
  const liveEmiStats = useMemo(() => {
    const totalIncome = form.applicantIncome + form.coapplicantIncome;
    const interest = getInterestRateByArea(form.propertyArea);
    const amort = computeAmortization(form.loanAmount, interest.rate, form.loanTenure);
    const activeExistingEmis = existingLoanEmis
      .slice(0, form.existingLoans)
      .reduce((s, v) => s + v, 0);
    const emiToIncome =
      totalIncome > 0
        ? ((amort.monthlyEmi + activeExistingEmis) / totalIncome) * 100
        : 100;
    return {
      monthlyEmi: amort.monthlyEmi,
      activeExistingEmis,
      emiToIncome: Number(emiToIncome.toFixed(2)),
    };
  }, [
    form.applicantIncome,
    form.coapplicantIncome,
    form.loanAmount,
    form.loanTenure,
    form.propertyArea,
    form.existingLoans,
    existingLoanEmis,
  ]);

  const updateField = <K extends keyof LoanInputForm>(key: K, value: LoanInputForm[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    if (evaluated) setIsDirty(true);
  };

  const handleExistingEmiChange = (index: number, val: number) => {
    setExistingLoanEmis((prev) => {
      const next = [...prev];
      next[index] = Math.max(0, val);
      return next;
    });
    if (evaluated) setIsDirty(true);
  };

  const handleEvaluate = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const syncedForm: LoanInputForm = {
      ...form,
      emiToIncomeRatio: liveEmiStats.emiToIncome,
    };
    const explanation = scoreApplicant(syncedForm);
    setEvaluated({
      form: syncedForm,
      existingLoanEmis: [...existingLoanEmis],
      explanation,
    });
    setIsDirty(false);
  };

  const handleReset = () => {
    setForm({ ...DEFAULT_LOAN_INPUT });
    setExistingLoanEmis(DEFAULT_EXISTING_EMIS);
    setEvaluated(null);
    setIsDirty(false);
  };

  const activeResult = evaluated
    ? activeModel === "decisionTree"
      ? evaluated.explanation.decisionTree
      : evaluated.explanation.logisticRegression
    : null;

  const secondaryResult = evaluated
    ? activeModel === "decisionTree"
      ? evaluated.explanation.logisticRegression
      : evaluated.explanation.decisionTree
    : null;

  const snapshotMetrics = evaluated
    ? (() => {
        const f = evaluated.form;
        const totalIncome = f.applicantIncome + f.coapplicantIncome;
        const activeExistingEmis = evaluated.existingLoanEmis
          .slice(0, f.existingLoans)
          .reduce((sum, v) => sum + v, 0);

        const interestInfo = getInterestRateByArea(f.propertyArea);
        const amort = computeAmortization(f.loanAmount, interestInfo.rate, f.loanTenure);

        const disposableAfterAll =
          totalIncome - f.monthlyExpenses - activeExistingEmis - amort.monthlyEmi;
        const loanToIncomeRatio = totalIncome > 0 ? f.loanAmount / totalIncome : 0;
        const collateralRatio = f.loanAmount > 0 ? (f.collateralValue / f.loanAmount) * 100 : 0;
        const expenseBurdenRatio =
          totalIncome > 0
            ? ((f.monthlyExpenses + activeExistingEmis + amort.monthlyEmi) / totalIncome) * 100
            : 100;
        const savingsToLoanRatio = f.loanAmount > 0 ? (f.savingsBalance / f.loanAmount) * 100 : 0;

        const isCashflowNegative = disposableAfterAll < 0;
        const approvalPct = activeResult!.probability * 100;
        const finalApproved =
          activeResult!.prediction === "Yes" && !isCashflowNegative;

        return {
          totalIncome,
          monthlyExpenses: f.monthlyExpenses,
          dailyEquivalent: f.monthlyExpenses / 30,
          activeExistingEmis,
          interestRate: interestInfo.rate,
          interestLabel: interestInfo.label,
          monthlyInstallment: amort.monthlyEmi,
          annualEmi: amort.annualEmi,
          annualExtraInterest: amort.annualExtraInterest,
          totalExtraInterest: amort.totalExtraInterest,
          disposableAfterAll,
          loanToIncomeRatio,
          collateralRatio,
          expenseBurdenRatio,
          savingsToLoanRatio,
          isCashflowNegative,
          approvalPct,
          finalApproved,
        };
      })()
    : null;

  const creditScorePercent = ((form.creditScore - 340) / (890 - 340)) * 100;
  const dtiPercent = (form.dtiRatio / 65) * 100;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      {/* Left Column: Applicant Input Form */}
      <form
        onSubmit={handleEvaluate}
        className="lg:col-span-6 rounded-xl border border-slate-800/80 bg-[#111827] p-6 space-y-5"
      >
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Applicant Financial & Credit Profile
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Trained on the 10,000-record hybrid credit dataset · Click Evaluate Loan Decision
            </p>
          </div>
          <button
            type="button"
            onClick={handleReset}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white hover:border-slate-700 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>

        {/* Income & Monthly Expenses */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
          <div>
            <label htmlFor="inp-app-inc" className="block text-xs font-medium text-slate-300 mb-1.5">
              Applicant Income ($/mo)
            </label>
            <input
              id="inp-app-inc"
              type="number"
              min={0}
              max={1000000}
              step={500}
              value={form.applicantIncome}
              onChange={(e) => updateField("applicantIncome", Number(e.target.value) || 0)}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            />
          </div>
          <div>
            <label htmlFor="inp-coapp-inc" className="block text-xs font-medium text-slate-300 mb-1.5">
              Co-applicant Income ($/mo)
            </label>
            <input
              id="inp-coapp-inc"
              type="number"
              min={0}
              max={1000000}
              step={500}
              value={form.coapplicantIncome}
              onChange={(e) => updateField("coapplicantIncome", Number(e.target.value) || 0)}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            />
          </div>
          <div>
            <label htmlFor="inp-monthly-exp" className="block text-xs font-medium text-slate-300 mb-1.5">
              Monthly Expenses ($/mo)
            </label>
            <input
              id="inp-monthly-exp"
              type="number"
              min={0}
              max={500000}
              step={500}
              value={form.monthlyExpenses}
              onChange={(e) => updateField("monthlyExpenses", Number(e.target.value) || 0)}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Loan Amount, Tenure, Savings, Collateral */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
          <div>
            <label htmlFor="inp-loan-amt" className="block text-xs font-medium text-slate-300 mb-1.5">
              Loan Amount ($)
            </label>
            <input
              id="inp-loan-amt"
              type="number"
              min={10000}
              max={20000000}
              step={10000}
              value={form.loanAmount}
              onChange={(e) => updateField("loanAmount", Number(e.target.value) || 0)}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            />
          </div>
          <div>
            <label htmlFor="inp-loan-term" className="block text-xs font-medium text-slate-300 mb-1.5">
              Loan Tenure (Mo)
            </label>
            <select
              id="inp-loan-term"
              value={form.loanTenure}
              onChange={(e) => updateField("loanTenure", Number(e.target.value))}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            >
              {LOAN_TENURES.map((t) => (
                <option key={t} value={t}>
                  {t} mo
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="inp-savings" className="block text-xs font-medium text-slate-300 mb-1.5">
              Savings Balance ($)
            </label>
            <input
              id="inp-savings"
              type="number"
              min={0}
              max={10000000}
              step={5000}
              value={form.savingsBalance}
              onChange={(e) => updateField("savingsBalance", Number(e.target.value) || 0)}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            />
          </div>
          <div>
            <label htmlFor="inp-collateral" className="block text-xs font-medium text-slate-300 mb-1.5">
              Collateral Value ($)
            </label>
            <input
              id="inp-collateral"
              type="number"
              min={0}
              max={25000000}
              step={10000}
              value={form.collateralValue}
              onChange={(e) => updateField("collateralValue", Number(e.target.value) || 0)}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Credit History, Defaults, Existing Loans, Employment Stability */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
          <div>
            <label htmlFor="inp-payment-history" className="block text-xs font-medium text-slate-300 mb-1.5">
              Payment History
            </label>
            <select
              id="inp-payment-history"
              value={form.paymentHistory}
              onChange={(e) => updateField("paymentHistory", e.target.value)}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            >
              <option value="Excellent">Excellent</option>
              <option value="Good">Good</option>
              <option value="Average">Average</option>
              <option value="Poor">Poor</option>
            </select>
          </div>
          <div>
            <label htmlFor="inp-prev-defaults" className="block text-xs font-medium text-slate-300 mb-1.5">
              Prior Defaults
            </label>
            <select
              id="inp-prev-defaults"
              value={form.previousLoanDefaults}
              onChange={(e) => updateField("previousLoanDefaults", Number(e.target.value))}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            >
              {[0, 1, 2, 3].map((d) => (
                <option key={d} value={d}>
                  {d} {d === 1 ? "default" : "defaults"}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="inp-existing-loans" className="block text-xs font-medium text-slate-300 mb-1.5">
              Existing Loans
            </label>
            <select
              id="inp-existing-loans"
              value={form.existingLoans}
              onChange={(e) => updateField("existingLoans", Number(e.target.value))}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            >
              {[0, 1, 2, 3, 4, 5].map((n) => (
                <option key={n} value={n}>
                  {n} {n === 1 ? "loan" : "loans"}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="inp-emp-stability" className="block text-xs font-medium text-slate-300 mb-1.5">
              Job Stability (Yrs)
            </label>
            <input
              id="inp-emp-stability"
              type="number"
              min={0}
              max={40}
              step={0.5}
              value={form.employmentStability}
              onChange={(e) => updateField("employmentStability", Number(e.target.value) || 0)}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 font-mono text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Dynamic Existing Loans Monthly EMI Inputs (shown only when existingLoans >= 1) */}
        {form.existingLoans > 0 && (
          <div className="rounded-xl border border-slate-800/80 bg-[#0B0F17] p-4 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium text-slate-200">
                Existing Loans Monthly EMI ({form.existingLoans}{" "}
                {form.existingLoans === 1 ? "active loan" : "active loans"})
              </span>
              <span className="font-mono text-amber-400 tabular-nums">
                Total: ${liveEmiStats.activeExistingEmis.toLocaleString()}/mo
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {Array.from({ length: form.existingLoans }).map((_, idx) => (
                <div key={idx}>
                  <label
                    htmlFor={`inp-existing-emi-${idx}`}
                    className="block text-[11px] text-slate-400 mb-1"
                  >
                    {idx === 0
                      ? "1st Loan EMI ($/mo)"
                      : idx === 1
                      ? "2nd Loan EMI ($/mo)"
                      : idx === 2
                      ? "3rd Loan EMI ($/mo)"
                      : `${idx + 1}th Loan EMI ($/mo)`}
                  </label>
                  <input
                    id={`inp-existing-emi-${idx}`}
                    type="number"
                    min={0}
                    max={250000}
                    step={250}
                    value={existingLoanEmis[idx] ?? 0}
                    onChange={(e) =>
                      handleExistingEmiChange(idx, Number(e.target.value) || 0)
                    }
                    className="w-full rounded-lg border border-slate-800 bg-[#111827] px-3 py-1.5 font-mono text-xs text-slate-100 focus:border-sky-500 focus:outline-none"
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Selectable Loan Purpose & Property Area (Interest Tier) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
          <div>
            <label htmlFor="inp-purpose" className="block text-xs font-medium text-slate-300 mb-1.5">
              Loan Purpose
            </label>
            <select
              id="inp-purpose"
              value={form.loanPurpose}
              onChange={(e) => updateField("loanPurpose", e.target.value)}
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            >
              <option value="Home">Home Loan</option>
              <option value="Vehicle">Vehicle / Car Loan</option>
              <option value="Business">Business Loan</option>
              <option value="Education">Education Loan</option>
              <option value="Personal">Personal Loan</option>
              <option value="Medical">Medical Loan</option>
            </select>
          </div>

          <div>
            <label htmlFor="inp-area" className="block text-xs font-medium text-slate-300 mb-1.5">
              Property Area (Interest Rate Tier)
            </label>
            <select
              id="inp-area"
              value={form.propertyArea}
              onChange={(e) =>
                updateField("propertyArea", e.target.value as LoanInputForm["propertyArea"])
              }
              className="w-full rounded-lg border border-slate-800 bg-[#0B0F17] px-3 py-2 text-sm text-slate-100 focus:border-sky-500 focus:outline-none"
            >
              <option value="Urban">Urban Area (10.5% p.a.)</option>
              <option value="Semiurban">Semi-Urban Area (7.0% p.a.)</option>
              <option value="Rural">Rural / Village Area (5.0% p.a.)</option>
            </select>
          </div>
        </div>

        {/* Draggable Credit Score Marker & Adjustable DTI Ratio */}
        <div className="space-y-4 rounded-xl border border-slate-800/80 bg-[#0B0F17] p-4">
          <div>
            <div className="flex items-center justify-between text-xs mb-2">
              <label htmlFor="cs-slider" className="font-medium text-slate-200">
                Credit Score Marker (Drag to Adjust)
              </label>
              <span className="font-mono text-sm font-bold text-sky-400 tabular-nums">
                {form.creditScore}
              </span>
            </div>
            <div className="relative pt-1">
              <input
                id="cs-slider"
                type="range"
                min={340}
                max={890}
                step={1}
                value={form.creditScore}
                onChange={(e) => updateField("creditScore", Number(e.target.value))}
                className="w-full h-2 rounded-lg appearance-none cursor-grab active:cursor-grabbing accent-sky-400 bg-slate-800"
              />
              <div className="mt-1.5 flex justify-between text-[11px] font-mono text-slate-500 tabular-nums">
                <span>340 (High Risk)</span>
                <span>640 (Subprime)</span>
                <span>720 (Prime)</span>
                <span>890</span>
              </div>
              <div className="mt-1 h-1 w-full rounded-full bg-slate-800 overflow-hidden">
                <div
                  className={`h-full transition-all duration-150 ${
                    form.creditScore >= 720
                      ? "bg-emerald-500"
                      : form.creditScore >= 640
                      ? "bg-sky-400"
                      : "bg-rose-500"
                  }`}
                  style={{ width: `${Math.max(2, Math.min(100, creditScorePercent))}%` }}
                />
              </div>
            </div>
          </div>

          {/* Adjustable Debt-to-Income Ratio (%) */}
          <div className="pt-2 border-t border-slate-800/80">
            <div className="flex items-center justify-between text-xs mb-2">
              <label htmlFor="dti-slider" className="font-medium text-slate-200">
                Existing Debt-to-Income (DTI) Ratio
              </label>
              <span className="font-mono text-sm font-bold text-amber-400 tabular-nums">
                {form.dtiRatio.toFixed(1)}%
              </span>
            </div>
            <input
              id="dti-slider"
              type="range"
              min={0}
              max={65}
              step={0.5}
              value={form.dtiRatio}
              onChange={(e) => updateField("dtiRatio", Number(e.target.value))}
              className="w-full h-2 rounded-lg appearance-none cursor-grab active:cursor-grabbing accent-amber-400 bg-slate-800"
            />
            <div className="mt-1.5 flex justify-between text-[11px] font-mono text-slate-500 tabular-nums">
              <span>0% (Debt Free)</span>
              <span>15% (Moderate)</span>
              <span>30%+ (Elevated)</span>
              <span>65%</span>
            </div>
            <div className="mt-1 h-1 w-full rounded-full bg-slate-800 overflow-hidden">
              <div
                className={`h-full transition-all duration-150 ${
                  form.dtiRatio <= 12
                    ? "bg-emerald-500"
                    : form.dtiRatio <= 25
                    ? "bg-amber-400"
                    : "bg-rose-500"
                }`}
                style={{ width: `${Math.max(2, Math.min(100, dtiPercent))}%` }}
              />
            </div>
          </div>

          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
            <span>Computed Total EMI-to-Income Ratio</span>
            <span className="font-mono font-semibold text-slate-200 tabular-nums">
              {liveEmiStats.emiToIncome.toFixed(1)}% (${Math.round(liveEmiStats.monthlyEmi).toLocaleString()} new EMI)
            </span>
          </div>
        </div>

        {/* Evaluate Button */}
        <button
          type="submit"
          className="w-full inline-flex items-center justify-center gap-2.5 rounded-xl bg-sky-500 px-6 py-3.5 text-sm font-semibold text-slate-950 hover:bg-sky-400 active:scale-[0.99] transition-all cursor-pointer"
        >
          <Play className="w-4 h-4 fill-current" />
          <span>Evaluate Loan Decision</span>
        </button>
      </form>

      {/* Right Column: Generated Verdict, Approval %, Repayment Breakdown & Audit */}
      <div className="lg:col-span-6 space-y-5">
        {!evaluated || !activeResult || !secondaryResult || !snapshotMetrics ? (
          <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-8 text-center space-y-4">
            <div className="mx-auto w-12 h-12 rounded-xl border border-slate-800 bg-[#0B0F17] flex items-center justify-center text-sky-400">
              <Play className="w-5 h-5" />
            </div>
            <div className="space-y-1.5 max-w-md mx-auto">
              <h3 className="text-base font-semibold text-slate-100">
                Ready to Score Loan Application
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Configure income, monthly expenses, loan tenure, payment history, prior defaults, credit score, and DTI on the left, then click{" "}
                <strong className="text-slate-200">Evaluate Loan Decision</strong> to run the trained model and affordability audit.
              </p>
            </div>
            <button
              type="button"
              onClick={() => handleEvaluate()}
              className="inline-flex items-center gap-2 rounded-lg border border-sky-500/40 bg-sky-500/10 px-4 py-2 text-xs font-semibold text-sky-300 hover:bg-sky-500/20 transition-colors"
            >
              <Play className="w-3.5 h-3.5" />
              <span>Run Initial Evaluation</span>
            </button>
          </div>
        ) : (
          <>
            {isDirty && (
              <div className="flex items-center justify-between gap-3 rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-xs text-amber-200">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
                  <span>Inputs modified. Click Evaluate Loan Decision to refresh prediction.</span>
                </div>
                <button
                  type="button"
                  onClick={() => handleEvaluate()}
                  className="rounded-md bg-amber-400 px-3 py-1 font-semibold text-slate-950 hover:bg-amber-300 transition-colors whitespace-nowrap shrink-0"
                >
                  Re-Evaluate
                </button>
              </div>
            )}

            {/* Primary Verdict + Approval Percentage Card */}
            <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-6 space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <div className="text-xs font-medium text-slate-400">
                    {activeModel === "decisionTree"
                      ? "Decision Tree Classifier Verdict"
                      : "Logistic Regression Classifier Verdict"}
                  </div>
                  <div className="mt-2 flex items-center gap-3">
                    {snapshotMetrics.finalApproved ? (
                      <CheckCircle2 className="w-8 h-8 text-emerald-400 shrink-0" />
                    ) : (
                      <XCircle className="w-8 h-8 text-rose-400 shrink-0" />
                    )}
                    <span
                      className={`font-display text-2xl sm:text-3xl font-bold ${
                        snapshotMetrics.finalApproved ? "text-emerald-400" : "text-rose-400"
                      }`}
                    >
                      {snapshotMetrics.finalApproved ? "Loan Approved" : "Loan Rejected"}
                    </span>
                  </div>
                  {snapshotMetrics.isCashflowNegative && activeResult.prediction === "Yes" && (
                    <p className="mt-1.5 text-xs text-rose-300">
                      Declined by cashflow affordability check: monthly expenses + EMIs exceed total monthly income.
                    </p>
                  )}
                </div>

                <div className="text-right">
                  <div className="text-xs font-medium text-slate-400">
                    Loan Approval Percentage
                  </div>
                  <div className="mt-1 font-mono text-4xl font-bold text-slate-100 tabular-nums">
                    {snapshotMetrics.approvalPct.toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Approval Probability Meter */}
              <div className="space-y-1.5">
                <div className="h-2.5 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      snapshotMetrics.finalApproved ? "bg-emerald-500" : "bg-rose-500"
                    }`}
                    style={{ width: `${Math.max(2, Math.min(100, snapshotMetrics.approvalPct))}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>
                    Alternate Model (
                    {activeModel === "decisionTree" ? "Logistic Regression" : "Decision Tree"}
                    ):{" "}
                    <strong className="font-mono text-slate-200 tabular-nums">
                      {secondaryResult.prediction === "Yes" ? "Approved" : "Rejected"} (
                      {(secondaryResult.probability * 100).toFixed(1)}%)
                    </strong>
                  </span>
                  <button
                    type="button"
                    onClick={() =>
                      onSelectModel(
                        activeModel === "decisionTree"
                          ? "logisticRegression"
                          : "decisionTree"
                      )
                    }
                    className="text-sky-400 hover:text-sky-300 underline underline-offset-2 whitespace-nowrap"
                  >
                    Switch Model
                  </button>
                </div>
              </div>
            </div>

            {/* Approved-Only Loan Repayment & Annual Interest Breakdown */}
            {snapshotMetrics.finalApproved && (
              <div className="rounded-xl border border-emerald-500/30 bg-[#111827] p-5 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
                  <div>
                    <h3 className="text-sm font-semibold text-emerald-400">
                      Approved Loan EMI & Annual Interest Breakdown
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      {evaluated.form.propertyArea === "Rural"
                        ? "Rural village concessional interest rate (5.0% p.a.) applied"
                        : evaluated.form.propertyArea === "Semiurban"
                        ? "Semi-urban town concessional interest rate (7.0% p.a.) applied"
                        : "Standard urban area interest rate (10.5% p.a.) applied"}
                    </p>
                  </div>
                  <span className="font-mono text-xs font-semibold text-sky-400 tabular-nums">
                    {snapshotMetrics.interestLabel}
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="rounded-lg border border-slate-800 bg-[#0B0F17] p-4">
                    <div className="text-xs text-slate-400">Per Month EMI</div>
                    <div className="mt-1.5 font-mono text-2xl font-bold text-slate-100 tabular-nums">
                      ${Math.round(snapshotMetrics.monthlyInstallment).toLocaleString()}
                      <span className="text-xs font-normal text-slate-400">/mo</span>
                    </div>
                    <div className="mt-1 text-[11px] text-slate-500 font-mono tabular-nums">
                      Over {evaluated.form.loanTenure} months
                    </div>
                  </div>

                  <div className="rounded-lg border border-slate-800 bg-[#0B0F17] p-4">
                    <div className="text-xs text-slate-400">Per Year EMI Outflow</div>
                    <div className="mt-1.5 font-mono text-2xl font-bold text-sky-400 tabular-nums">
                      ${Math.round(snapshotMetrics.annualEmi).toLocaleString()}
                      <span className="text-xs font-normal text-slate-400">/yr</span>
                    </div>
                    <div className="mt-1 text-[11px] text-slate-500 font-mono tabular-nums">
                      Principal ${evaluated.form.loanAmount.toLocaleString()}
                    </div>
                  </div>

                  <div className="rounded-lg border border-slate-800 bg-[#0B0F17] p-4">
                    <div className="text-xs text-slate-400">Annual Extra Interest</div>
                    <div className="mt-1.5 font-mono text-2xl font-bold text-amber-400 tabular-nums">
                      +${Math.round(snapshotMetrics.annualExtraInterest).toLocaleString()}
                      <span className="text-xs font-normal text-slate-400">/yr</span>
                    </div>
                    <div className="mt-1 text-[11px] text-slate-500 font-mono tabular-nums">
                      Total extra interest: +${Math.round(snapshotMetrics.totalExtraInterest).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Key Ratios Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
              <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-4">
                <div className="text-[11px] font-medium text-slate-400">DTI Ratio</div>
                <div className="mt-1.5 font-mono text-lg font-bold text-amber-400 tabular-nums">
                  {evaluated.form.dtiRatio.toFixed(1)}%
                </div>
                <div className="mt-0.5 font-mono text-[11px] text-slate-500 tabular-nums">
                  {evaluated.form.previousLoanDefaults} prior defaults
                </div>
              </div>

              <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-4">
                <div className="text-[11px] font-medium text-slate-400">EMI-to-Income</div>
                <div className="mt-1.5 font-mono text-lg font-bold text-sky-400 tabular-nums">
                  {evaluated.form.emiToIncomeRatio.toFixed(1)}%
                </div>
                <div className="mt-0.5 font-mono text-[11px] text-slate-500 tabular-nums">
                  LTI {snapshotMetrics.loanToIncomeRatio.toFixed(1)}x
                </div>
              </div>

              <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-4">
                <div className="text-[11px] font-medium text-slate-400">Expense + EMI</div>
                <div
                  className={`mt-1.5 font-mono text-lg font-bold tabular-nums ${
                    snapshotMetrics.expenseBurdenRatio <= 75
                      ? "text-emerald-400"
                      : "text-rose-400"
                  }`}
                >
                  {snapshotMetrics.expenseBurdenRatio.toFixed(1)}%
                </div>
                <div className="mt-0.5 text-[11px] text-slate-500">
                  Of total income
                </div>
              </div>

              <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-4">
                <div className="text-[11px] font-medium text-slate-400">Collateral Ratio</div>
                <div className="mt-1.5 font-mono text-lg font-bold text-slate-100 tabular-nums">
                  {snapshotMetrics.collateralRatio.toFixed(0)}%
                </div>
                <div className="mt-0.5 text-[11px] text-slate-500">
                  Savings {snapshotMetrics.savingsToLoanRatio.toFixed(0)}%
                </div>
              </div>
            </div>

            {/* Income & Monthly Expenses Affordability Audit */}
            <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-5 space-y-3">
              <h3 className="text-sm font-semibold text-slate-100">
                Income, Monthly Expenses & Repayment Check
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs pt-1">
                <div>
                  <span className="text-slate-400">Monthly Income</span>
                  <div className="mt-1 font-mono font-semibold text-slate-100 tabular-nums">
                    ${snapshotMetrics.totalIncome.toLocaleString()}
                  </div>
                </div>
                <div>
                  <span className="text-slate-400">Monthly Expenses</span>
                  <div className="mt-1 font-mono font-semibold text-slate-200 tabular-nums">
                    ${snapshotMetrics.monthlyExpenses.toLocaleString()}{" "}
                    <span className="text-slate-500 font-normal">
                      (${Math.round(snapshotMetrics.dailyEquivalent)}/d)
                    </span>
                  </div>
                </div>
                <div>
                  <span className="text-slate-400">Existing EMIs</span>
                  <div className="mt-1 font-mono font-semibold text-slate-200 tabular-nums">
                    ${snapshotMetrics.activeExistingEmis.toLocaleString()}/mo
                  </div>
                </div>
                <div>
                  <span className="text-slate-400">New Loan EMI</span>
                  <div className="mt-1 font-mono font-semibold text-slate-200 tabular-nums">
                    ${Math.round(snapshotMetrics.monthlyInstallment).toLocaleString()}/mo
                  </div>
                </div>
                <div>
                  <span className="text-slate-400">Net Disposable</span>
                  <div
                    className={`mt-1 font-mono font-semibold tabular-nums ${
                      snapshotMetrics.disposableAfterAll >= 0
                        ? "text-emerald-400"
                        : "text-rose-400"
                    }`}
                  >
                    ${Math.round(snapshotMetrics.disposableAfterAll).toLocaleString()}/mo
                  </div>
                </div>
              </div>
            </div>

            {/* Model Decision Audit */}
            <div className="rounded-xl border border-slate-800/80 bg-[#111827] p-5">
              {activeModel === "decisionTree" ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-100">
                        Decision Tree Rule Path
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Dataset-trained CART thresholds evaluated for this applicant
                      </p>
                    </div>
                    <span className="font-mono text-slate-400 tabular-nums">
                      Leaf n={evaluated.explanation.decisionTree.leafSamples}
                    </span>
                  </div>

                  <div className="space-y-2">
                    {evaluated.explanation.decisionTree.path.map((step, idx) => (
                      <div
                        key={step.nodeId}
                        className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-800 bg-[#0B0F17] px-3.5 py-2 text-xs"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-slate-500">0{idx + 1}.</span>
                          <span className="font-medium text-slate-200">
                            {formatFeatureLabel(step.featureName)}
                          </span>
                          <span className="font-mono text-sky-400 tabular-nums">
                            {step.direction === "left" ? "≤" : ">"} {step.threshold.toFixed(2)}
                          </span>
                        </div>
                        <div className="flex items-center gap-2.5 font-mono text-slate-400 tabular-nums">
                          <span>Input: {step.actualValue.toFixed(2)}</span>
                          <ArrowRight className="w-3 h-3" />
                          <span className="text-slate-200">{(step.prob * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-100">
                        Logistic Regression Factor Impact
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Top positive and negative log-odds contributors
                      </p>
                    </div>
                    <span className="font-mono text-xs text-slate-400 tabular-nums">
                      z = {evaluated.explanation.logisticRegression.logit.toFixed(2)}
                    </span>
                  </div>
                  <div className="h-56 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={evaluated.explanation.logisticRegression.topContributions.map((c) => ({
                          feature: formatFeatureLabel(c.feature),
                          contribution: Number(c.contribution.toFixed(3)),
                        }))}
                        layout="vertical"
                        margin={{ top: 4, right: 16, left: 24, bottom: 4 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" horizontal={false} />
                        <XAxis type="number" stroke="#94A3B8" fontSize={11} />
                        <YAxis type="category" dataKey="feature" stroke="#CBD5E1" fontSize={11} width={125} />
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
                        <Bar dataKey="contribution" name="Log-Odds" barSize={15} radius={[3, 3, 3, 3]}>
                          {evaluated.explanation.logisticRegression.topContributions.map((entry) => (
                            <Cell
                              key={entry.feature}
                              fill={entry.contribution >= 0 ? "#10B981" : "#F43F5E"}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};
