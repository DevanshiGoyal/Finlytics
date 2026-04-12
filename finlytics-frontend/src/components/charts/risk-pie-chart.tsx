"use client";

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

const COLORS = ["#22c55e", "#f59e0b", "#ef4444"];

interface PiePoint {
  name: string;
  value: number;
}

export function RiskPieChart({ data }: { data: PiePoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius={70} outerRadius={110} paddingAngle={3}>
          {data.map((entry, index) => (
            <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            background: "#020617",
            border: "1px solid rgba(148,163,184,0.25)",
            borderRadius: 12,
            color: "#e2e8f0"
          }}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
