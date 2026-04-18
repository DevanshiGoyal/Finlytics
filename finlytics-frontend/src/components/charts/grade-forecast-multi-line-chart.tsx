"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ForecastChartPoint {
  month: string;
  [key: string]: number | string;
}

const COLORS: Record<string, string> = {
  A: "#22d3ee",
  B: "#60a5fa",
  C: "#818cf8",
  D: "#f59e0b",
  E: "#ef4444",
};

export function GradeForecastMultiLineChart({
  data,
  activeGrades,
}: {
  data: ForecastChartPoint[];
  activeGrades: string[];
}) {
  return (
    <ResponsiveContainer width="100%" height={380}>
      <ComposedChart data={data}>
        <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="4 4" />
        <XAxis dataKey="month" stroke="#94a3b8" />
        <YAxis stroke="#94a3b8" />
        <Tooltip
          contentStyle={{
            background: "#020617",
            border: "1px solid rgba(148,163,184,0.25)",
            borderRadius: 12,
            color: "#e2e8f0",
          }}
        />
        <Legend />

        {activeGrades.map((grade) => (
          <Area
            key={`${grade}-base`}
            type="monotone"
            dataKey={`${grade}_lower`}
            stackId={`band-${grade}`}
            stroke="transparent"
            fill="transparent"
            isAnimationActive={false}
            legendType="none"
          />
        ))}
        {activeGrades.map((grade) => (
          <Area
            key={`${grade}-band`}
            type="monotone"
            dataKey={`${grade}_range`}
            stackId={`band-${grade}`}
            stroke="transparent"
            fill={COLORS[grade]}
            fillOpacity={0.12}
            isAnimationActive={false}
            name={`Grade ${grade} CI`}
          />
        ))}
        {activeGrades.map((grade) => (
          <Line
            key={`${grade}-line`}
            type="monotone"
            dataKey={`${grade}_central`}
            stroke={COLORS[grade]}
            strokeWidth={2.4}
            dot={false}
            isAnimationActive={false}
            name={`Grade ${grade}`}
          />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
