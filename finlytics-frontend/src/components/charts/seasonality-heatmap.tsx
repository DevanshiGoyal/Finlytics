import { Fragment } from "react";

import { cn } from "@/utils/cn";

interface HeatmapRow {
  grade: string;
  values: number[];
}

const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function toColor(value: number, min: number, max: number) {
  const ratio = (value - min) / Math.max(1, max - min);
  const blue = Math.round(75 + ratio * 140);
  const green = Math.round(95 + ratio * 110);
  return `rgba(56, ${green}, ${blue}, ${0.25 + ratio * 0.55})`;
}

export function SeasonalityHeatmap({ data }: { data: HeatmapRow[] }) {
  const allValues = data.flatMap((row) => row.values);
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);

  return (
    <div className="overflow-x-auto rounded-2xl border border-white/10 bg-slate-950/40 p-4">
      <div className="grid min-w-[680px] grid-cols-[100px_repeat(12,minmax(42px,1fr))] gap-2 text-xs">
        <div />
        {months.map((month) => (
          <div key={month} className="text-center text-slate-400">
            {month}
          </div>
        ))}

        {data.map((row) => (
          <Fragment key={row.grade}>
            <div key={`${row.grade}-label`} className="self-center font-semibold text-slate-200">
              Grade {row.grade}
            </div>
            {row.values.map((value, index) => (
              <div
                key={`${row.grade}-${index}`}
                className={cn(
                  "grid h-9 place-items-center rounded-md border border-white/10 text-[11px] font-medium text-white"
                )}
                style={{ backgroundColor: toColor(value, min, max) }}
              >
                {value}
              </div>
            ))}
          </Fragment>
        ))}
      </div>
    </div>
  );
}
