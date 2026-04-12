import { ArrowDownRight, ArrowUpRight, Dot } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardTitle, CardValue } from "@/components/ui/card";

interface MetricCardProps {
  label: string;
  value: string;
  delta: string;
  tone: "success" | "warning" | "danger" | "info";
}

export function MetricCard({ label, value, delta, tone }: MetricCardProps) {
  const positive = delta.includes("+") || delta.toLowerCase().includes("stable");

  return (
    <Card className="space-y-3">
      <CardTitle>{label}</CardTitle>
      <CardValue>{value}</CardValue>
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1 text-xs text-slate-300">
          {positive ? <ArrowUpRight className="h-3.5 w-3.5 text-emerald-300" /> : <ArrowDownRight className="h-3.5 w-3.5 text-red-300" />}
          {delta}
        </span>
        <Badge variant={tone}>
          <Dot className="mr-1 h-3.5 w-3.5" />
          Live
        </Badge>
      </div>
    </Card>
  );
}
