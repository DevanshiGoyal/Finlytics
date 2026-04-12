"use client";

import { motion } from "framer-motion";

import { cn } from "@/utils/cn";

export interface TabOption {
  value: string;
  label: string;
}

interface TabsProps {
  options: TabOption[];
  value: string;
  onValueChange: (value: string) => void;
  className?: string;
}

export function Tabs({ options, value, onValueChange, className }: TabsProps) {
  return (
    <div className={cn("flex w-full flex-wrap gap-2 rounded-2xl border border-white/10 bg-slate-950/40 p-2", className)}>
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onValueChange(option.value)}
            className={cn(
              "relative rounded-xl px-4 py-2 text-sm font-semibold transition",
              active ? "text-slate-950" : "text-slate-300 hover:bg-white/10"
            )}
          >
            {active ? (
              <motion.span
                layoutId="tab-pill"
                className="absolute inset-0 rounded-xl bg-gradient-to-r from-cyan-300 to-blue-400"
                transition={{ type: "spring", stiffness: 280, damping: 26 }}
              />
            ) : null}
            <span className="relative z-10">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}
