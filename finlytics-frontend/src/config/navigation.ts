import type { Route } from "next";

export const navItems: Array<{ label: string; href: Route }> = [
  { label: "Dashboard Overview", href: "/" },
  { label: "Loan Default Risk", href: "/loan-default-risk" },
  { label: "Borrower Churn", href: "/borrower-churn" },
  { label: "Loan Volume Forecast", href: "/loan-volume-forecast" },
  { label: "Credit Demand by Grade", href: "/credit-demand-by-grade" },
  { label: "Bank Deposit AI", href: "/bank-deposit-ai" },
  { label: "Deposit Anomaly Detection", href: "/deposit-anomaly-detection" }
];

export const roles = ["Risk Analyst", "Marketing Team", "Operations Manager"] as const;

export type Role = (typeof roles)[number];
