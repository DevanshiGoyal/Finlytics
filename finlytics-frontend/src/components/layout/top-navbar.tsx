"use client";

import { Bell, Moon, Search, Sun, User2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import type { Role } from "@/config/navigation";

interface TopNavbarProps {
  role: Role;
  onRoleChange: (role: Role) => void;
  darkMode: boolean;
  onThemeToggle: () => void;
  onOpenAssistant: () => void;
}

export function TopNavbar({
  role,
  onRoleChange,
  darkMode,
  onThemeToggle,
  onOpenAssistant
}: TopNavbarProps) {
  return (
    <header className="sticky top-0 z-30 mb-6 border-b border-white/10 bg-slate-950/50 backdrop-blur-xl">
      <div className="flex flex-wrap items-center justify-between gap-3 px-2 py-3 md:px-4">
        <div className="flex min-w-[240px] flex-1 items-center gap-2 rounded-xl border border-white/10 bg-slate-900/40 px-3 py-2">
          <Search className="h-4 w-4 text-slate-400" />
          <Input placeholder="Search borrower, loan id, insight..." className="h-7 border-none bg-transparent p-0 text-xs focus-visible:ring-0" />
        </div>

        <div className="flex items-center gap-2">
          <Select
            value={role}
            onChange={(event) => onRoleChange(event.target.value as Role)}
            options={[
              { label: "Risk Analyst", value: "Risk Analyst" },
              { label: "Marketing Team", value: "Marketing Team" },
              { label: "Operations Manager", value: "Operations Manager" }
            ]}
            className="h-9 min-w-[180px]"
          />

          <Button variant="secondary" size="sm" onClick={onThemeToggle}>
            {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            Theme
          </Button>

          <Button variant="secondary" size="sm" onClick={onOpenAssistant}>
            Ask Finlytics
          </Button>

          <Button variant="secondary" size="sm" className="relative">
            <Bell className="h-4 w-4" />
            <span className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full bg-red-400" />
          </Button>

          <div className="hidden items-center gap-2 rounded-xl border border-white/10 bg-slate-900/50 px-3 py-2 sm:flex">
            <User2 className="h-4 w-4 text-cyan-300" />
            <div className="leading-tight">
              <p className="text-xs font-semibold text-slate-100">Dana Lee</p>
              <Badge variant="info" className="px-1.5 py-0 text-[10px]">
                {role}
              </Badge>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
