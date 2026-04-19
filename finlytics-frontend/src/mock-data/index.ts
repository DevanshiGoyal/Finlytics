import type { Route } from "next";

export type MetricRecord = {
  label: string;
  value: string;
  delta: string;
  tone: "success" | "warning" | "danger" | "info";
};

export const overviewMetrics: MetricRecord[] = [
  { label: "Total Portfolio Value", value: "$248.6M", delta: "+3.2% WoW", tone: "success" },
  { label: "Default Risk Distribution", value: "18.4% High", delta: "-1.1%", tone: "success" },
  { label: "Churn Risk", value: "27.9%", delta: "+2.4%", tone: "warning" },
  { label: "Active Loans", value: "32,804", delta: "+682", tone: "info" },
  { label: "Forecast (Next 3M)", value: "$42.1M", delta: "95% CI stable", tone: "info" }
];

export const loanVolumeTrend = [
  { month: "Jan", actual: 11.4, forecast: 11.6 },
  { month: "Feb", actual: 12.1, forecast: 12.0 },
  { month: "Mar", actual: 12.6, forecast: 12.8 },
  { month: "Apr", actual: 12.2, forecast: 12.5 },
  { month: "May", actual: 13.0, forecast: 13.1 },
  { month: "Jun", actual: 13.4, forecast: 13.6 },
  { month: "Jul", actual: 14.1, forecast: 14.0 },
  { month: "Aug", actual: 14.5, forecast: 14.7 },
  { month: "Sep", actual: 15.2, forecast: 15.3 },
  { month: "Oct", actual: 15.0, forecast: 15.2 },
  { month: "Nov", actual: 15.6, forecast: 15.8 },
  { month: "Dec", actual: 16.2, forecast: 16.4 }
];

export const forecastWithInterval = [
  { month: "Sep", value: 15.2, low: 14.7, high: 15.8, historical: 15.2 },
  { month: "Oct", value: 15.0, low: 14.4, high: 15.7, historical: 15.0 },
  { month: "Nov", value: 15.6, low: 15.0, high: 16.2, historical: 15.6 },
  { month: "Dec", value: 16.2, low: 15.6, high: 16.8, historical: 16.2 },
  { month: "Jan+1", value: 16.8, low: 16.0, high: 17.7, historical: null },
  { month: "Feb+1", value: 17.3, low: 16.4, high: 18.1, historical: null },
  { month: "Mar+1", value: 18.0, low: 17.1, high: 18.9, historical: null }
];

export const nextThreeMonths = [
  { month: "Jan+1", expected: "$16.8M", confidence: "91%" },
  { month: "Feb+1", expected: "$17.3M", confidence: "89%" },
  { month: "Mar+1", expected: "$18.0M", confidence: "87%" }
];

export const riskDistribution = [
  { name: "Low", value: 46 },
  { name: "Medium", value: 36 },
  { name: "High", value: 18 }
];

export const gradeDemand = [
  { grade: "A", demand: 1220 },
  { grade: "B", demand: 1080 },
  { grade: "C", demand: 930 },
  { grade: "D", demand: 740 },
  { grade: "E", demand: 520 }
];

export const gradeTrend = [
  { month: "Jan", A: 95, B: 87, C: 74, D: 61, E: 44 },
  { month: "Feb", A: 98, B: 88, C: 76, D: 62, E: 46 },
  { month: "Mar", A: 100, B: 90, C: 78, D: 63, E: 47 },
  { month: "Apr", A: 101, B: 92, C: 79, D: 64, E: 48 },
  { month: "May", A: 105, B: 93, C: 81, D: 66, E: 49 },
  { month: "Jun", A: 108, B: 95, C: 83, D: 68, E: 50 },
  { month: "Jul", A: 110, B: 97, C: 84, D: 69, E: 51 },
  { month: "Aug", A: 114, B: 98, C: 86, D: 71, E: 52 },
  { month: "Sep", A: 117, B: 101, C: 87, D: 73, E: 54 },
  { month: "Oct", A: 118, B: 102, C: 89, D: 74, E: 55 },
  { month: "Nov", A: 121, B: 104, C: 90, D: 75, E: 57 },
  { month: "Dec", A: 124, B: 106, C: 92, D: 77, E: 58 }
];

export const seasonalityHeatmap = [
  { grade: "A", values: [64, 66, 68, 69, 72, 75, 76, 79, 82, 84, 86, 88] },
  { grade: "B", values: [58, 59, 60, 62, 64, 66, 67, 69, 71, 73, 75, 77] },
  { grade: "C", values: [49, 50, 52, 53, 55, 58, 59, 60, 62, 64, 66, 68] },
  { grade: "D", values: [37, 39, 40, 41, 43, 46, 47, 48, 50, 52, 53, 54] },
  { grade: "E", values: [28, 29, 31, 32, 34, 35, 36, 38, 39, 41, 42, 43] }
];

export const defaultFeatureImportance = [
  { feature: "Debt-to-Income Ratio", importance: 0.26 },
  { feature: "Interest Rate", importance: 0.22 },
  { feature: "Credit Utilization", importance: 0.17 },
  { feature: "Income", importance: 0.14 },
  { feature: "Past Delinquencies", importance: 0.11 },
  { feature: "Employment Length", importance: 0.1 }
];

export const churnFeatureImportance = [
  { feature: "Repayment Delay", importance: 0.25 },
  { feature: "Engagement Score", importance: 0.2 },
  { feature: "Service Interactions", importance: 0.17 },
  { feature: "Interest Rate", importance: 0.15 },
  { feature: "Loan Tenure", importance: 0.13 },
  { feature: "Income Volatility", importance: 0.1 }
];

export const retentionSuggestions = [
  "Offer personalized refinance at 0.8% discount for high churn borrowers.",
  "Trigger loyalty campaign for accounts with low engagement and high tenure.",
  "Escalate to relationship manager for churn probability above 70%."
];

export const portfolioRows = [
  { id: "L-2041", borrower: "Neo Capital", defaultRisk: 0.71, churnRisk: 0.64, riskBand: "High" },
  { id: "L-2042", borrower: "Orion Holdings", defaultRisk: 0.32, churnRisk: 0.28, riskBand: "Medium" },
  { id: "L-2043", borrower: "Cedar Ventures", defaultRisk: 0.12, churnRisk: 0.2, riskBand: "Low" },
  { id: "L-2044", borrower: "Summit Lending", defaultRisk: 0.58, churnRisk: 0.49, riskBand: "High" },
  { id: "L-2045", borrower: "Aurora Trust", defaultRisk: 0.27, churnRisk: 0.23, riskBand: "Medium" }
];

export const stressScenarios = [
  { scenario: "Base", portfolioLoss: 2.8, capitalImpact: 1.1 },
  { scenario: "Rate +150bps", portfolioLoss: 4.6, capitalImpact: 2.2 },
  { scenario: "Income -15%", portfolioLoss: 5.2, capitalImpact: 2.6 },
  { scenario: "Combined Shock", portfolioLoss: 7.3, capitalImpact: 3.9 }
];

export const driftMetrics = [
  { feature: "Income", psi: 0.08, status: "Stable" },
  { feature: "DTI", psi: 0.17, status: "Watch" },
  { feature: "Credit Utilization", psi: 0.25, status: "Drift" },
  { feature: "Delinquencies", psi: 0.11, status: "Stable" }
];

export const modelLeaderboard = [
  { model: "Random Forest", accuracy: 0.89, precision: 0.86, recall: 0.84, f1: 0.85 },
  { model: "XGBoost", accuracy: 0.91, precision: 0.88, recall: 0.86, f1: 0.87 },
  { model: "Logistic Regression", accuracy: 0.84, precision: 0.79, recall: 0.77, f1: 0.78 }
];

export const anomalyTrend = [
  { time: "09:00", score: 0.21 },
  { time: "09:10", score: 0.24 },
  { time: "09:20", score: 0.28 },
  { time: "09:30", score: 0.66 },
  { time: "09:40", score: 0.72 },
  { time: "09:50", score: 0.44 },
  { time: "10:00", score: 0.31 }
];

export const anomalyBatchResults = [
  { txId: "TX-1101", amount: 3200, score: 0.18, label: "Normal" },
  { txId: "TX-1102", amount: 19800, score: 0.81, label: "Suspicious" },
  { txId: "TX-1103", amount: 4500, score: 0.34, label: "Normal" },
  { txId: "TX-1104", amount: 25100, score: 0.88, label: "Suspicious" }
];

export const navItems: Array<{ label: string; href: Route }> = [
  { label: "Dashboard Overview", href: "/" },
  { label: "Loan Default Risk", href: "/loan-default-risk" },
  { label: "Borrower Churn", href: "/borrower-churn" },
  { label: "Loan Volume Forecast", href: "/loan-volume-forecast" },
  { label: "Credit Demand by Grade", href: "/credit-demand-by-grade" },
  { label: "Bank Deposit AI", href: "/bank-deposit-ai" },
  { label: "Deposit Anomaly Detection", href: "/deposit-anomaly-detection" },
  { label: "DataLens", href: "/talk-to-data" }
];

export const roles = ["Risk Analyst", "Marketing Team", "Operations Manager"] as const;

export type Role = (typeof roles)[number];
