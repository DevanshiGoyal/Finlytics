"use client";

import { useEffect, useState } from "react";
import { Toaster } from "sonner";

import { AIAssistantPanel } from "@/components/chat/ai-assistant-panel";
import { Sidebar } from "@/components/layout/sidebar";
import { TopNavbar } from "@/components/layout/top-navbar";
import { type Role } from "@/config/navigation";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [role, setRole] = useState<Role>("Risk Analyst");

  useEffect(() => {
    const stored = localStorage.getItem("finlytics-theme");
    const isDark = stored !== "light";
    setDarkMode(isDark);
    document.documentElement.classList.toggle("dark", isDark);
  }, []);

  const toggleTheme = () => {
    setDarkMode((prev) => {
      const next = !prev;
      document.documentElement.classList.toggle("dark", next);
      localStorage.setItem("finlytics-theme", next ? "dark" : "light");
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-fin-gradient text-slate-100">
      <div className="flex min-h-screen">
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((prev) => !prev)} />
        <div className="flex-1 px-3 pb-6 md:px-5">
          <TopNavbar
            role={role}
            onRoleChange={setRole}
            darkMode={darkMode}
            onThemeToggle={toggleTheme}
            onOpenAssistant={() => setAssistantOpen(true)}
          />
          <main className="space-y-6">{children}</main>
        </div>
      </div>
      <AIAssistantPanel open={assistantOpen} onClose={() => setAssistantOpen(false)} />
      <Toaster position="top-right" richColors />
    </div>
  );
}
