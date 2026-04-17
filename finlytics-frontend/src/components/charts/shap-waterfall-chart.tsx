"use client";

import { useMemo } from "react";

import type { ShapWaterfallPoint } from "@/services/api";

interface Props {
  points: ShapWaterfallPoint[];
  baseValue: number;
  modelOutput?: number | null;
}

function pct(value: number, min: number, max: number) {
  if (max <= min) {
    return 0;
  }
  return ((value - min) / (max - min)) * 100;
}

export function ShapWaterfallChart({ points, baseValue, modelOutput }: Props) {
  const extents = useMemo(() => {
    const vals = [
      baseValue,
      ...(modelOutput != null ? [modelOutput] : []),
      ...points.flatMap((point) => [point.start, point.end]),
    ];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    return { min, max };
  }, [baseValue, modelOutput, points]);

  const finalOutput =
    modelOutput ?? (points.length ? points[points.length - 1].end : baseValue);

  return (
    <div className="space-y-4">
      <div className="grid gap-2 rounded-xl border border-white/10 bg-slate-950/40 p-3 text-xs text-slate-300 md:grid-cols-3">
        <div>
          <p className="text-slate-400">Base value</p>
          <p className="font-semibold text-cyan-200">{baseValue.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-slate-400">Model output (log-odds)</p>
          <p className="font-semibold text-cyan-200">
            {finalOutput.toFixed(4)}
          </p>
        </div>
        <div>
          <p className="text-slate-400">Displayed features</p>
          <p className="font-semibold text-cyan-200">{points.length}</p>
        </div>
      </div>

      <div className="space-y-2">
        {points.map((point) => {
          const left = pct(
            Math.min(point.start, point.end),
            extents.min,
            extents.max,
          );
          const right = pct(
            Math.max(point.start, point.end),
            extents.min,
            extents.max,
          );
          const width = Math.max(1, right - left);
          const positive = point.shapValue >= 0;

          return (
            <div
              key={`${point.feature}-${point.start}-${point.end}`}
              className="rounded-lg border border-white/10 bg-slate-900/30 p-2"
            >
              <div className="mb-1 flex items-center justify-between gap-2 text-xs">
                <span className="truncate text-slate-200">{point.feature}</span>
                <span
                  className={positive ? "text-emerald-300" : "text-rose-300"}
                >
                  {positive ? "+" : ""}
                  {point.shapValue.toFixed(4)}
                </span>
              </div>
              <div className="relative h-3 rounded bg-slate-800">
                <div
                  className={`absolute h-3 rounded ${positive ? "bg-emerald-400/80" : "bg-rose-400/80"}`}
                  style={{ left: `${left}%`, width: `${width}%` }}
                />
              </div>
              <div className="mt-1 flex justify-between text-[11px] text-slate-400">
                <span>x = {point.value.toFixed(3)}</span>
                <span>
                  {point.start.toFixed(3)} → {point.end.toFixed(3)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
