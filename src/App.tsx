import React, { useState } from "react";
import {
  BarChart3,
  GitCompareArrows,
  LayoutDashboard,
  Menu,
  X,
} from "lucide-react";
import { OverviewTrendsView } from "./components/OverviewTrendsView";
import { PredictorView } from "./components/PredictorView";
import { ModelBenchmarksView } from "./components/ModelBenchmarksView";
import { MODEL_ARTIFACT } from "./utils/inference";
import type { ModelKey } from "./types/creditwise";

type NavTab = "dashboard" | "overview" | "benchmarks";

const NAV_ITEMS: Array<{
  id: NavTab;
  label: string;
  title: string;
  icon: React.ComponentType<{ className?: string }>;
}> = [
  {
    id: "dashboard",
    label: "Dashboard",
    title: "Loan Approval Prediction Dashboard",
    icon: LayoutDashboard,
  },
  {
    id: "overview",
    label: "Overview & Trends",
    title: "Portfolio Trends & Exploratory Analytics",
    icon: BarChart3,
  },
  {
    id: "benchmarks",
    label: "Model Benchmarks",
    title: "Decision Tree vs. Logistic Regression Evaluation",
    icon: GitCompareArrows,
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>("dashboard");
  const [activeModel, setActiveModel] = useState<ModelKey>("decisionTree");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const currentNav = NAV_ITEMS.find((item) => item.id === activeTab) || NAV_ITEMS[0];
  const dtAcc = (MODEL_ARTIFACT.models.decisionTree.metrics.accuracy * 100).toFixed(2);
  const lrAcc = (MODEL_ARTIFACT.models.logisticRegression.metrics.accuracy * 100).toFixed(2);
  const activeMetrics = MODEL_ARTIFACT.models[activeModel].metrics;

  return (
    <div className="min-h-screen bg-[#0B0F17] text-slate-100 flex">
      {/* Mobile Sidebar Backdrop */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Left Sidebar Navigation (Only 3 Clean Items) */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 shrink-0 border-r border-slate-800/80 bg-[#0F141F] flex flex-col transition-transform duration-200 lg:static lg:translate-x-0 ${
          mobileMenuOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="h-16 px-6 flex items-center justify-between border-b border-slate-800/80">
          <button
            onClick={() => {
              setActiveTab("dashboard");
              setMobileMenuOpen(false);
            }}
            className="font-display text-lg font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0 text-left"
          >
            CreditWise
          </button>
          <button
            onClick={() => setMobileMenuOpen(false)}
            aria-label="Close navigation menu"
            className="lg:hidden rounded-lg p-1.5 text-slate-400 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <nav aria-label="Primary Sidebar Navigation" className="flex-1 px-3 py-5 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveTab(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors whitespace-nowrap shrink-0 ${
                  isActive
                    ? "bg-sky-500/15 text-sky-300 border border-sky-500/30"
                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/50 border border-transparent"
                }`}
              >
                <Icon className={`w-4 h-4 shrink-0 ${isActive ? "text-sky-400" : "text-slate-400"}`} />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Quiet Dataset Summary Footer inside Sidebar */}
        <div className="border-t border-slate-800/80 px-5 py-4 text-xs text-slate-400 space-y-1.5">
          <div className="font-medium text-slate-300">Model Summary</div>
          <div className="flex justify-between font-mono tabular-nums">
            <span>Dataset Rows</span>
            <span className="text-slate-200">{MODEL_ARTIFACT.datasetSummary.totalRecords}</span>
          </div>
          <div className="flex justify-between font-mono tabular-nums">
            <span>Holdout Split</span>
            <span className="text-slate-200">
              {MODEL_ARTIFACT.splitInfo.trainCount} / {MODEL_ARTIFACT.splitInfo.testCount}
            </span>
          </div>
          <div className="flex justify-between font-mono tabular-nums">
            <span>Active AUC</span>
            <span className="text-sky-400">{activeMetrics.rocAuc.toFixed(4)}</span>
          </div>
        </div>
      </aside>

      {/* Main Content Shell */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Top Header Bar */}
        <header className="h-16 border-b border-slate-800/80 bg-[#0B0F17]/95 backdrop-blur px-6 flex items-center justify-between gap-8 sticky top-0 z-20">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open navigation menu"
              className="lg:hidden rounded-lg border border-slate-800 p-2 text-slate-300 hover:text-white"
            >
              <Menu className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-2 text-sm truncate">
              <span className="text-slate-500 hidden sm:inline whitespace-nowrap">
                CreditWise Analytics
              </span>
              <span className="text-slate-600 hidden sm:inline" aria-hidden="true">
                /
              </span>
              <span className="font-display font-semibold text-slate-100 truncate">
                {currentNav.label}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={() => setActiveTab(activeTab === "dashboard" ? "benchmarks" : "dashboard")}
              className="rounded-lg border border-slate-800 bg-[#111827] px-3.5 py-1.5 text-xs font-medium text-slate-200 hover:border-sky-500/50 hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              {activeTab === "dashboard" ? "Compare Models" : "Go to Predictor"}
            </button>
          </div>
        </header>

        {/* Main Workspace Viewport */}
        <main className="flex-1 px-6 py-6 max-w-[1440px] w-full mx-auto space-y-6">
          {/* Workspace Header + Switchable Classifier Selector */}
          <div className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-800/80 pb-5">
            <div>
              <h1 className="font-display text-2xl font-bold text-slate-100 text-balance">
                {currentNav.title}
              </h1>
              <div className="mt-1.5 flex flex-wrap items-center gap-2.5 text-xs text-slate-400">
                <span>{MODEL_ARTIFACT.datasetSummary.totalRecords.toLocaleString()} Consumer Loan Records</span>
                <span aria-hidden="true">·</span>
                <span>
                  Active Model:{" "}
                  <strong className="text-slate-200">
                    {activeModel === "decisionTree"
                      ? "Decision Tree (CART)"
                      : "Logistic Regression (L2)"}
                  </strong>
                </span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-slate-400 whitespace-nowrap">
                Model
              </span>
              <div
                role="group"
                aria-label="Global Classifier Switch"
                className="flex items-center gap-1 rounded-lg bg-[#111827] p-1 border border-slate-800"
              >
                <button
                  onClick={() => setActiveModel("decisionTree")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                    activeModel === "decisionTree"
                      ? "bg-sky-500 text-slate-950 font-semibold"
                      : "text-slate-300 hover:text-white"
                  }`}
                >
                  Decision Tree ({dtAcc}%)
                </button>
                <button
                  onClick={() => setActiveModel("logisticRegression")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                    activeModel === "logisticRegression"
                      ? "bg-sky-500 text-slate-950 font-semibold"
                      : "text-slate-300 hover:text-white"
                  }`}
                >
                  Logistic Regression ({lrAcc}%)
                </button>
              </div>
            </div>
          </div>

          {activeTab === "dashboard" && (
            <PredictorView activeModel={activeModel} onSelectModel={setActiveModel} />
          )}
          {activeTab === "overview" && (
            <OverviewTrendsView activeModel={activeModel} onSelectModel={setActiveModel} />
          )}
          {activeTab === "benchmarks" && (
            <ModelBenchmarksView activeModel={activeModel} onSelectModel={setActiveModel} />
          )}
        </main>
      </div>
    </div>
  );
}
