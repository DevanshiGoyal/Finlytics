"use client";

import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend
} from "recharts";

interface DataPoint {
  month: string;
  actual: number;
  forecast: number;
}

export function LoanVolumeLineChart({ data }: { data: DataPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data}>
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
        <Legend />
        <Line type="monotone" dataKey="actual" stroke="#38bdf8" strokeWidth={2.5} dot={false} />
        <Line type="monotone" dataKey="forecast" stroke="#818cf8" strokeWidth={2.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
