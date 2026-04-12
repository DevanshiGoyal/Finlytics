"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

interface DataPoint {
  time: string;
  score: number;
}

export function AnomalyTrendChart({ data }: { data: DataPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="4 4" />
        <XAxis dataKey="time" stroke="#94a3b8" />
        <YAxis stroke="#94a3b8" domain={[0, 1]} />
        <Tooltip
          contentStyle={{
            background: "#020617",
            border: "1px solid rgba(148,163,184,0.25)",
            borderRadius: 12,
            color: "#e2e8f0"
          }}
        />
        <ReferenceLine y={0.65} stroke="#ef4444" strokeDasharray="5 5" />
        <Line type="monotone" dataKey="score" stroke="#f59e0b" strokeWidth={2.6} dot />
      </LineChart>
    </ResponsiveContainer>
  );
}
