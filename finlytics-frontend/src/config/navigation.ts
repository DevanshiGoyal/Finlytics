import type { Route } from "next";

export const navItems: Array<{ label: string; href: Route }> = [
  { label: "Dashboard Overview", href: "/" },
  { label: "Loan Default Risk", href: "/loan-default-risk" },
  { label: "Borrower Churn", href: "/borrower-churn" },
  { label: "Loan Volume Forecast", href: "/loan-volume-forecast" },
  { label: "Credit Demand by Grade", href: "/credit-demand-by-grade" },
  { label: "Bank Deposit AI", href: "/bank-deposit-ai" },
  { label: "Deposit Anomaly Detection", href: "/deposit-anomaly-detection" },
  { label: "Wealth Persona Intelligence", href: "/wealth-persona" }
];

export const roles = ["Risk Analyst", "Marketing Team", "Operations Manager"] as const;

export type Role = (typeof roles)[number];

export const rolePermissions: Record<Role, string[]> = {
  "Risk Analyst": [
    "Dashboard Overview",
    "Loan Default Risk",
    "Credit Demand by Grade",
    "Deposit Anomaly Detection",
    "Data Insight Studio"
  ],
  "Operations Manager": [
    "Dashboard Overview",
    "Loan Volume Forecast",
    "Deposit Anomaly Detection",
    "Wealth Persona Intelligence",
    "Data Insight Studio"
  ],
  "Marketing Team": [
    "Dashboard Overview",
    "Borrower Churn",
    "Data Insight Studio"
  ]
};

export function getAllowedNavItems(role: Role) {
  return navItems.filter(item => rolePermissions[role].includes(item.label));
}
