import fs from "fs";
import path from "path";
import { execFileSync } from "child_process";

const ARTIFACT_PATH = path.resolve("src/data/trainedModels.json");
const OUT_DIR = path.resolve("public/charts");
const BRAIN_DIR = path.resolve(
  "/.aistudio/artifacts/brain/cc815a1b-df1d-4b8e-baf6-2af21cb7c0c1"
);

// Official Matplotlib/Seaborn YlGnBu 9-stop palette
const YLGNBU_STOPS = [
  [0.0, [255, 255, 217]],   // #ffffd9
  [0.125, [237, 248, 177]], // #edf8b1
  [0.25, [199, 233, 180]],  // #c7e9b4
  [0.375, [127, 205, 187]], // #7fcdbb
  [0.5, [65, 182, 196]],    // #41b6c4
  [0.625, [29, 145, 192]],  // #1d91c0
  [0.75, [34, 94, 168]],    // #225ea8
  [0.875, [37, 52, 148]],   // #253494
  [1.0, [8, 29, 88]],       // #081d58
];

function sampleYlGnBu(t) {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < YLGNBU_STOPS.length - 1; i++) {
    const [t0, c0] = YLGNBU_STOPS[i];
    const [t1, c1] = YLGNBU_STOPS[i + 1];
    if (clamped >= t0 && clamped <= t1) {
      const f = (clamped - t0) / (t1 - t0);
      const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
      const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
      const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
      const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      return {
        hex: `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`,
        textColor: luminance < 0.54 ? "#FFFFFF" : "#1E293B",
      };
    }
  }
  return { hex: "#081d58", textColor: "#FFFFFF" };
}

function renderConfusionMatrixPng(title, subtitle, cm, outPath) {
  const vals = [cm.tp, cm.fn, cm.fp, cm.tn];
  const minVal = Math.min(...vals);
  const maxVal = Math.max(...vals);
  const span = Math.max(1, maxVal - minVal);

  const width = 1280;
  const height = 1020;
  const plotX = 230;
  const plotY = 140;
  const plotSize = 740;
  const cell = plotSize / 2;
  const cbX = 1030;
  const cbW = 42;
  const cbH = plotSize;

  const cells = [
    { row: 0, col: 0, label: `TP: ${cm.tp}`, val: cm.tp },
    { row: 0, col: 1, label: `FN: ${cm.fn}`, val: cm.fn },
    { row: 1, col: 0, label: `FP: ${cm.fp}`, val: cm.fp },
    { row: 1, col: 1, label: `TN: ${cm.tn}`, val: cm.tn },
  ];

  const args = ["-size", `${width}x${height}`, "xc:white"];

  // Title & Subtitle
  args.push(
    "-font", "DejaVu-Sans-Bold",
    "-pointsize", "30",
    "-fill", "#0F172A",
    "-gravity", "North",
    "-annotate", `+${plotX + plotSize / 2 - width / 2}+36`, title,
    "-font", "DejaVu-Sans",
    "-pointsize", "17",
    "-fill", "#475569",
    "-annotate", `+${plotX + plotSize / 2 - width / 2}+80`, subtitle
  );

  // Switch gravity back to NorthWest for exact coordinates
  args.push("-gravity", "NorthWest");

  // Heatmap Cells + centered text
  for (const c of cells) {
    const norm = (c.val - minVal) / span;
    const colInfo = sampleYlGnBu(norm);
    const x1 = plotX + c.col * cell;
    const y1 = plotY + c.row * cell;
    const x2 = x1 + cell;
    const y2 = y1 + cell;

    args.push(
      "-fill", colInfo.hex,
      "-stroke", "#FFFFFF",
      "-strokewidth", "2",
      "-draw", `rectangle ${x1},${y1} ${x2},${y2}`
    );

    const cx = x1 + cell / 2;
    const cy = y1 + cell / 2;
    args.push(
      "-gravity", "Center",
      "-stroke", "none",
      "-font", "DejaVu-Sans-Bold",
      "-pointsize", "32",
      "-fill", colInfo.textColor,
      "-annotate", `+${Math.round(cx - width / 2)}+${Math.round(cy - height / 2)}`,
      c.label
    );
  }

  // Outer Plot Border + Axis Ticks
  args.push(
    "-gravity", "NorthWest",
    "-fill", "none",
    "-stroke", "#334155",
    "-strokewidth", "2",
    "-draw", `rectangle ${plotX},${plotY} ${plotX + plotSize},${plotY + plotSize}`,
    "-stroke", "#0F172A",
    "-strokewidth", "2",
    "-draw", `line ${plotX - 10},${plotY + cell / 2} ${plotX},${plotY + cell / 2}`,
    "-draw", `line ${plotX - 10},${plotY + cell * 1.5} ${plotX},${plotY + cell * 1.5}`,
    "-draw", `line ${plotX + cell / 2},${plotY + plotSize} ${plotX + cell / 2},${plotY + plotSize + 10}`,
    "-draw", `line ${plotX + cell * 1.5},${plotY + plotSize} ${plotX + cell * 1.5},${plotY + plotSize + 10}`
  );

  // Y-Axis Tick Labels
  args.push(
    "-stroke", "none",
    "-font", "DejaVu-Sans",
    "-pointsize", "22",
    "-fill", "#0F172A",
    "-gravity", "East",
    "-annotate", `+${width - (plotX - 20)}+${Math.round(plotY + cell / 2 - height / 2)}`, "Positive",
    "-annotate", `+${width - (plotX - 20)}+${Math.round(plotY + cell * 1.5 - height / 2)}`, "Negative"
  );

  // Rotated Y-Axis Title ("True labels")
  args.push(
    "-gravity", "Center",
    "-font", "DejaVu-Sans-Bold",
    "-pointsize", "24",
    "-fill", "#0F172A",
    "-annotate", `270x270+${Math.round(68 - width / 2)}+${Math.round(plotY + plotSize / 2 - height / 2)}`,
    "True labels"
  );

  // X-Axis Tick Labels & Axis Title ("Predicted labels")
  args.push(
    "-gravity", "North",
    "-font", "DejaVu-Sans",
    "-pointsize", "22",
    "-fill", "#0F172A",
    "-annotate", `+${Math.round(plotX + cell / 2 - width / 2)}+${plotY + plotSize + 22}`, "Positive",
    "-annotate", `+${Math.round(plotX + cell * 1.5 - width / 2)}+${plotY + plotSize + 22}`, "Negative",
    "-font", "DejaVu-Sans-Bold",
    "-pointsize", "24",
    "-annotate", `+${Math.round(plotX + plotSize / 2 - width / 2)}+${plotY + plotSize + 72}`, "Predicted labels"
  );

  // Vertical Colorbar (180 smooth bands)
  args.push("-gravity", "NorthWest", "-stroke", "none");
  const bands = 180;
  const sliceH = cbH / bands;
  for (let i = 0; i < bands; i++) {
    const t = 1 - i / (bands - 1);
    const { hex } = sampleYlGnBu(t);
    const yTop = Math.floor(plotY + i * sliceH);
    const yBot = Math.ceil(plotY + (i + 1) * sliceH);
    args.push("-fill", hex, "-draw", `rectangle ${cbX},${yTop} ${cbX + cbW},${yBot}`);
  }
  args.push(
    "-fill", "none",
    "-stroke", "#334155",
    "-strokewidth", "1.5",
    "-draw", `rectangle ${cbX},${plotY} ${cbX + cbW},${plotY + cbH}`
  );

  // Colorbar Ticks & Labels
  const step = maxVal > 600 ? 200 : 100;
  const ticks = [];
  const firstTick = Math.ceil(minVal / step) * step;
  for (let v = firstTick; v <= maxVal; v += step) ticks.push(v);
  if (!ticks.includes(maxVal)) ticks.push(maxVal);

  for (const t of ticks) {
    const frac = (t - minVal) / span;
    const ty = Math.round(plotY + cbH - frac * cbH);
    args.push(
      "-gravity", "NorthWest",
      "-stroke", "#0F172A",
      "-strokewidth", "2",
      "-draw", `line ${cbX + cbW},${ty} ${cbX + cbW + 8},${ty}`,
      "-stroke", "none",
      "-gravity", "West",
      "-font", "DejaVu-Sans",
      "-pointsize", "19",
      "-fill", "#0F172A",
      "-annotate", `+${cbX + cbW + 16}+${Math.round(ty - height / 2)}`, String(t)
    );
  }

  args.push(outPath);
  execFileSync("convert", args);
}

function renderMetricsPng(sectionNum, modelTitle, subtitle, metrics, outPath) {
  const width = 1360;
  const height = 920;
  const plotX = 160;
  const plotY = 170;
  const plotW = 1090;
  const plotH = 500;

  const items = [
    { label: "Accuracy", val: metrics.accuracy * 100, color: "#1d91c0" },
    { label: "Precision", val: metrics.precision * 100, color: "#225ea8" },
    { label: "Recall", val: metrics.recall * 100, color: "#081d58" },
    { label: "F1-score", val: metrics.f1 * 100, color: "#41b6c4" },
  ];

  const cm = metrics.confusionMatrix;
  const totalTest = cm.tn + cm.fp + cm.fn + cm.tp;
  const slotW = plotW / items.length;
  const barW = 148;

  const args = ["-size", `${width}x${height}`, "xc:white"];

  // Title & Subtitle
  args.push(
    "-gravity", "NorthWest",
    "-font", "DejaVu-Sans-Bold",
    "-pointsize", "30",
    "-fill", "#0F172A",
    "-annotate", "+76+42", `${sectionNum}. ${modelTitle} Results`,
    "-font", "DejaVu-Sans",
    "-pointsize", "18",
    "-fill", "#475569",
    "-annotate", "+76+88", subtitle
  );

  // Horizontal Gridlines & Y-Axis Labels
  for (const tick of [0, 20, 40, 60, 80, 100]) {
    const y = Math.round(plotY + plotH - (tick / 100) * plotH);
    args.push(
      "-gravity", "NorthWest",
      "-stroke", tick === 0 ? "#334155" : "#E2E8F0",
      "-strokewidth", tick === 0 ? "2" : "1.5",
      "-draw", `line ${plotX},${y} ${plotX + plotW},${y}`,
      "-stroke", "none",
      "-gravity", "East",
      "-font", "DejaVu-Sans",
      "-pointsize", "18",
      "-fill", "#334155",
      "-annotate", `+${width - (plotX - 18)}+${Math.round(y - height / 2)}`, `${tick}%%`
    );
  }

  // Y-Axis spine
  args.push(
    "-gravity", "NorthWest",
    "-stroke", "#334155",
    "-strokewidth", "2",
    "-draw", `line ${plotX},${plotY} ${plotX},${plotY + plotH}`
  );

  // Bars & Value Labels
  items.forEach((item, idx) => {
    const cx = plotX + slotW * idx + slotW / 2;
    const bx1 = Math.round(cx - barW / 2);
    const bx2 = Math.round(cx + barW / 2);
    const bh = (item.val / 100) * plotH;
    const by1 = Math.round(plotY + plotH - bh);
    const by2 = Math.round(plotY + plotH);

    args.push(
      "-gravity", "NorthWest",
      "-stroke", "none",
      "-fill", item.color,
      "-draw", `roundrectangle ${bx1},${by1} ${bx2},${by2} 8,8`,
      "-gravity", "South",
      "-font", "DejaVu-Sans-Bold",
      "-pointsize", "24",
      "-fill", "#0F172A",
      "-annotate", `+${Math.round(cx - width / 2)}+${height - by1 + 12}`, `${item.val.toFixed(2)}%%`,
      "-gravity", "North",
      "-font", "DejaVu-Sans-Bold",
      "-pointsize", "22",
      "-fill", "#0F172A",
      "-annotate", `+${Math.round(cx - width / 2)}+${plotY + plotH + 22}`, item.label
    );
  });

  // Rotated Y-Axis Label
  args.push(
    "-stroke", "none",
    "-gravity", "Center",
    "-font", "DejaVu-Sans-Bold",
    "-pointsize", "21",
    "-fill", "#0F172A",
    "-annotate", `270x270+${Math.round(52 - width / 2)}+${Math.round(plotY + plotH / 2 - height / 2)}`,
    "Score (%%)"
  );

  // Jupyter Notebook Output Summary Box below Chart
  args.push(
    "-gravity", "NorthWest",
    "-fill", "#F8FAFC",
    "-stroke", "#CBD5E1",
    "-strokewidth", "2",
    "-draw", "roundrectangle 76,755 1284,882 10,10",
    "-stroke", "none",
    "-font", "DejaVu-Sans-Mono-Bold",
    "-pointsize", "18",
    "-fill", "#0F172A",
    "-annotate", "+104+775",
    `Evaluation Summary (Holdout Test Split n = ${totalTest.toLocaleString()} | Target: Loan_Status)`,
    "-font", "DejaVu-Sans-Mono",
    "-pointsize", "17",
    "-fill", "#1E293B",
    "-annotate", "+104+812",
    `Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%%  |  Precision: ${(metrics.precision * 100).toFixed(2)}%%  |  Recall: ${(metrics.recall * 100).toFixed(2)}%%  |  F1-score: ${(metrics.f1 * 100).toFixed(2)}%%`,
    "-font", "DejaVu-Sans-Mono",
    "-pointsize", "16",
    "-fill", "#475569",
    "-annotate", "+104+846",
    `ROC-AUC: ${metrics.rocAuc.toFixed(4)}   |   TP: ${cm.tp}   TN: ${cm.tn}   FP: ${cm.fp}   FN: ${cm.fn}`
  );

  args.push(outPath);
  execFileSync("convert", args);
}

function main() {
  const artifact = JSON.parse(fs.readFileSync(ARTIFACT_PATH, "utf-8"));
  const dt = artifact.models.decisionTree;
  const lr = artifact.models.logisticRegression;
  const nTest = artifact.splitInfo.testCount;

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.mkdirSync(BRAIN_DIR, { recursive: true });

  const tasks = [
    () =>
      renderConfusionMatrixPng(
        "Confusion Matrix Heatmap — Decision Tree",
        `Holdout Test Set (n = ${nTest.toLocaleString()}) · loan_approval_hybrid_dataset_20711486.csv`,
        dt.metrics.confusionMatrix,
        path.join(OUT_DIR, "decision_tree_confusion_matrix.png")
      ),
    () =>
      renderConfusionMatrixPng(
        "Confusion Matrix Heatmap — Logistic Regression",
        `Holdout Test Set (n = ${nTest.toLocaleString()}) · loan_approval_hybrid_dataset_20711486.csv`,
        lr.metrics.confusionMatrix,
        path.join(OUT_DIR, "logistic_regression_confusion_matrix.png")
      ),
    () =>
      renderMetricsPng(
        1,
        "Decision Tree",
        `Accuracy, Precision, Recall, F1-score · max_depth=${dt.hyperparameters.maxDepth} · loan_approval_hybrid_dataset_20711486.csv`,
        dt.metrics,
        path.join(OUT_DIR, "decision_tree_metrics.png")
      ),
    () =>
      renderMetricsPng(
        2,
        "Logistic Regression",
        `Accuracy, Precision, Recall, F1-score · L2 Regularized · loan_approval_hybrid_dataset_20711486.csv`,
        lr.metrics,
        path.join(OUT_DIR, "logistic_regression_metrics.png")
      ),
  ];

  tasks.forEach((fn) => fn());

  for (const name of [
    "decision_tree_confusion_matrix.png",
    "logistic_regression_confusion_matrix.png",
    "decision_tree_metrics.png",
    "logistic_regression_metrics.png",
  ]) {
    fs.copyFileSync(path.join(OUT_DIR, name), path.join(BRAIN_DIR, name));
    console.log("Rendered", path.join(OUT_DIR, name));
  }
}

main();
