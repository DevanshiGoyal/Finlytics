"use client";

import {
  Activity,
  BarChart3,
  BriefcaseBusiness,
  ChevronLeft,
  ChevronRight,
  LayoutDashboard,
  ShieldAlert,
  TrendingUp,
  UserRoundX,
  WalletCards
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import type { ComponentType } from "react";

import { navItems } from "@/config/navigation";
import { cn } from "@/utils/cn";

const iconMap: Record<string, ComponentType<{ className?: string }>> = {
  "Dashboard Overview": LayoutDashboard,
  "Loan Default Risk": ShieldAlert,
  "Borrower Churn": UserRoundX,
  "Loan Volume Forecast": TrendingUp,
  "Credit Demand by Grade": BarChart3,
  "Portfolio Intelligence Hub": BriefcaseBusiness,
  "Bank Deposit AI": WalletCards,
  "Deposit Anomaly Detection": Activity
};

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname();

  return (
    <motion.aside
      animate={{ width: collapsed ? 92 : 280 }}
      transition={{ type: "spring", stiffness: 240, damping: 28 }}
      className="sticky top-0 h-screen border-r border-white/10 bg-slate-950/70 p-4 backdrop-blur-xl"
    >
      <div className="mb-6 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 overflow-hidden">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-r from-blue-500 to-cyan-400 text-slate-950 font-black">
            F
          </div>
          {!collapsed ? (
            <div>
              <p className="font-bold text-white">Finlytics</p>
              <p className="text-xs text-slate-400">Predictive Analytics</p>
            </div>
          ) : null}
        </div>
        <button
          type="button"
          onClick={onToggle}
          className="grid h-8 w-8 place-items-center rounded-lg border border-white/15 bg-slate-900/60 text-slate-200"
          aria-label="Toggle sidebar"
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>

      <nav className="space-y-2">
        {navItems.map((item) => {
          const Icon = iconMap[item.label] || LayoutDashboard;
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "group flex items-center gap-3 rounded-xl border px-3 py-2.5 text-sm transition",
                active
                  ? "border-cyan-400/40 bg-gradient-to-r from-cyan-300/20 to-blue-400/20 text-cyan-100"
                  : "border-transparent text-slate-300 hover:border-white/10 hover:bg-white/5"
              )}
              title={collapsed ? item.label : undefined}
            >
              <Icon className="h-4.5 w-4.5 shrink-0" />
              {!collapsed ? <span className="truncate">{item.label}</span> : null}
            </Link>
          );
        })}
      </nav>

      {!collapsed ? (
        <div className="mt-6 rounded-2xl border border-white/10 bg-slate-900/50 p-3 text-xs text-slate-300">
          <p className="font-semibold text-cyan-200">System Status</p>
          <p className="mt-1 text-slate-400">All modules operational</p>
        </div>
      ) : null}
    </motion.aside>
  );
}
