"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const CHART_COLORS = ["#22d3ee", "#38bdf8", "#a78bfa", "#34d399", "#f59e0b", "#f97316"];

type ChartKind = "line" | "bar" | "area" | "pie";

interface TalkToDataChartProps {
  data: Array<Record<string, unknown>>;
  chartType?: string | null;
  xKey?: string | null;
  yKeys?: string[];
  height?: number;
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return null;
}

function normaliseChartType(value?: string | null): ChartKind {
  const input = (value ?? "").toLowerCase();

  if (input.includes("pie")) {
    return "pie";
  }

  if (input.includes("bar")) {
    return "bar";
  }

  if (input.includes("area")) {
    return "area";
  }

  return "line";
}

function findFallbackYKeys(data: Array<Record<string, unknown>>, xKey: string): string[] {
  const firstRow = data[0];
  if (!firstRow) {
    return [];
  }

  const keys = Object.keys(firstRow).filter((key) => key !== xKey);
  const numericKeys = keys.filter((key) => data.some((row) => toNumber(row[key]) !== null));

  return numericKeys.slice(0, 2);
}

export function TalkToDataChart({
  data,
  chartType,
  xKey,
  yKeys,
  height = 280
}: TalkToDataChartProps) {
  if (!data.length) {
    return (
      <p className="rounded-xl border border-dashed border-white/20 bg-slate-900/25 p-6 text-sm text-slate-400">
        No chartable data available.
      </p>
    );
  }

  const resolvedXKey = xKey && xKey.trim() ? xKey : Object.keys(data[0] ?? {})[0];
  const fallbackYKeys = findFallbackYKeys(data, resolvedXKey ?? "");
  const resolvedYKeys = (yKeys?.length ? yKeys : fallbackYKeys).filter(Boolean);

  if (!resolvedXKey || !resolvedYKeys.length) {
    return (
      <p className="rounded-xl border border-dashed border-white/20 bg-slate-900/25 p-6 text-sm text-slate-400">
        Missing chart axis configuration for this dataset.
      </p>
    );
  }

  const usableYKeys = resolvedYKeys.filter((key) => data.some((row) => toNumber(row[key]) !== null));
  if (!usableYKeys.length) {
    return (
      <p className="rounded-xl border border-dashed border-white/20 bg-slate-900/25 p-6 text-sm text-slate-400">
        No numeric columns were detected for chart rendering.
      </p>
    );
  }

  const chartRows = data.slice(0, 40).map((row, index) => {
    const labelValue = row[resolvedXKey];
    const base: Record<string, string | number> = {
      name: labelValue === undefined || labelValue === null || labelValue === "" ? `Row ${index + 1}` : String(labelValue)
    };

    for (const key of usableYKeys) {
      base[key] = toNumber(row[key]) ?? 0;
    }

    return base;
  });

  const type = normaliseChartType(chartType);

  const commonAxes = (
    <>
      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.18)" />
      <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} minTickGap={18} axisLine={false} tickLine={false} />
      <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} width={56} />
      <Tooltip
        contentStyle={{
          background: "rgba(15, 23, 42, 0.96)",
          border: "1px solid rgba(148, 163, 184, 0.35)",
          borderRadius: 10,
          color: "#f8fafc"
        }}
      />
      {usableYKeys.length > 1 ? <Legend wrapperStyle={{ fontSize: 12 }} /> : null}
    </>
  );

  const primaryYKey = usableYKeys[0];

  return (
    <div className="h-[280px] w-full">
      <ResponsiveContainer width="100%" height={height}>
        {type === "pie" ? (
          <PieChart>
            <Tooltip
              contentStyle={{
                background: "rgba(15, 23, 42, 0.96)",
                border: "1px solid rgba(148, 163, 184, 0.35)",
                borderRadius: 10,
                color: "#f8fafc"
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Pie
              data={chartRows.slice(0, 12).map((row) => ({
                name: String(row.name),
                value: Number(row[primaryYKey])
              }))}
              dataKey="value"
              nameKey="name"
              innerRadius={56}
              outerRadius={100}
              paddingAngle={3}
            >
              {chartRows.slice(0, 12).map((_, index) => (
                <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        ) : type === "bar" ? (
          <BarChart data={chartRows} margin={{ top: 16, right: 16, left: 0, bottom: 8 }}>
            {commonAxes}
            {usableYKeys.map((key, index) => (
              <Bar key={key} dataKey={key} fill={CHART_COLORS[index % CHART_COLORS.length]} radius={[6, 6, 0, 0]} />
            ))}
          </BarChart>
        ) : type === "area" ? (
          <AreaChart data={chartRows} margin={{ top: 16, right: 16, left: 0, bottom: 8 }}>
            {commonAxes}
            {usableYKeys.map((key, index) => (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                stroke={CHART_COLORS[index % CHART_COLORS.length]}
                fill={CHART_COLORS[index % CHART_COLORS.length]}
                fillOpacity={0.25}
                strokeWidth={2}
              />
            ))}
          </AreaChart>
        ) : (
          <LineChart data={chartRows} margin={{ top: 16, right: 16, left: 0, bottom: 8 }}>
            {commonAxes}
            {usableYKeys.map((key, index) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={CHART_COLORS[index % CHART_COLORS.length]}
                strokeWidth={2.5}
                dot={{ r: 3 }}
                activeDot={{ r: 6 }}
              />
            ))}
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
