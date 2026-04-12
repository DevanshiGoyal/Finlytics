"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

interface GradePoint {
  month: string;
  A: number;
  B: number;
  C: number;
  D: number;
  E: number;
}

const COLORS: Record<string, string> = {
  A: "#22d3ee",
  B: "#60a5fa",
  C: "#818cf8",
  D: "#f59e0b",
  E: "#ef4444"
};

export function GradeMultiLineChart({
  data,
  activeGrades
}: {
  data: GradePoint[];
  activeGrades: string[];
}) {
  return (
    <ResponsiveContainer width="100%" height={360}>
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
        {activeGrades.map((grade) => (
          <Line key={grade} type="monotone" dataKey={grade} stroke={COLORS[grade]} strokeWidth={2.4} dot={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
