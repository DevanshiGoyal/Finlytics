"use client";

import { motion } from "framer-motion";

import { Badge } from "@/components/ui/badge";

interface PageHeaderProps {
  title: string;
  subtitle: string;
  tag?: string;
}

export function PageHeader({ title, subtitle, tag }: PageHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="rounded-2xl border border-white/10 bg-slate-900/40 p-5 backdrop-blur-xl"
    >
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold text-white md:text-3xl">{title}</h1>
        {tag ? <Badge variant="info">{tag}</Badge> : null}
      </div>
      <p className="mt-2 max-w-4xl text-sm text-slate-400 md:text-base">{subtitle}</p>
    </motion.div>
  );
}
