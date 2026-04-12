"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

interface DataPoint {
  month: string;
  value: number;
  low: number;
  high: number;
  historical: number | null;
}

export function ForecastAreaChart({ data }: { data: DataPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={340}>
      <AreaChart data={data}>
        <defs>
          <linearGradient id="ciFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.4} />
            <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="4 4" />
        <XAxis dataKey="month" stroke="#94a3b8" />
        <YAxis stroke="#94a3b8" />
        <Tooltip
          contentStyle={{
            background: "#020617",
            border: "1px solid rgba(148,163,184,0.25)",
            borderRadius: 12,
            color: "#e2e8f0"
          }}
        />
        <Area type="monotone" dataKey="high" stroke="transparent" fill="transparent" />
        <Area type="monotone" dataKey="low" stroke="transparent" fill="transparent" />
        <Area type="monotone" dataKey="high" stroke="none" fill="url(#ciFill)" />
        <Line type="monotone" dataKey="value" stroke="#38bdf8" strokeWidth={2.6} dot={false} />
        <Line type="monotone" dataKey="historical" stroke="#22c55e" strokeWidth={2} strokeDasharray="4 5" dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
