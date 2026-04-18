"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { CreditDemandForecastRequest, CreditDemandScenario } from "@/services/types";
import { cn } from "@/utils/cn";

interface CreditDemandInputProps {
  value: CreditDemandForecastRequest;
  loading?: boolean;
  onChange: (next: CreditDemandForecastRequest) => void;
  onSubmit: () => void;
  onReset?: () => void;
}

const SCENARIO_OPTIONS: Array<{ value: CreditDemandScenario; label: string; description: string }> = [
  { value: "baseline", label: "Baseline", description: "Model output as-is" },
  { value: "optimistic", label: "Optimistic", description: "+15% uplift on central and upper bounds" },
  { value: "pessimistic", label: "Pessimistic", description: "-15% reduction on central and lower bounds" },
];

function formatConfidence(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function CreditDemandInput({
  value,
  loading = false,
  onChange,
  onSubmit,
  onReset,
}: CreditDemandInputProps) {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-xl border border-white/10 bg-slate-950/35 p-4">
          <div className="flex items-center justify-between gap-2">
            <label className="text-sm font-semibold text-slate-100">Forecast Horizon</label>
            <span className="text-sm text-cyan-200">{value.horizon} month(s)</span>
          </div>
          <input
            type="range"
            min={1}
            max={12}
            step={1}
            value={value.horizon}
            onChange={(event) =>
              onChange({
                ...value,
                horizon: Number(event.target.value),
              })
            }
            className="mt-3 w-full accent-cyan-400"
            disabled={loading}
          />
          <p className="mt-2 text-xs text-slate-400">Planning window for grade-level demand forecast.</p>
        </div>

        <div className="rounded-xl border border-white/10 bg-slate-950/35 p-4">
          <div className="flex items-center justify-between gap-2">
            <label className="text-sm font-semibold text-slate-100">Confidence Level</label>
            <span className="text-sm text-cyan-200">{formatConfidence(value.confidence)}</span>
          </div>
          <input
            type="range"
            min={50}
            max={99}
            step={1}
            value={Math.round(value.confidence * 100)}
            onChange={(event) =>
              onChange({
                ...value,
                confidence: Number(event.target.value) / 100,
              })
            }
            className="mt-3 w-full accent-cyan-400"
            disabled={loading}
          />
          <p className="mt-2 text-xs text-slate-400">Controls width of prediction intervals by grade.</p>
        </div>
      </div>

      <div className="rounded-xl border border-white/10 bg-slate-950/35 p-4">
        <p className="text-sm font-semibold text-slate-100">Scenario Type</p>
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          {SCENARIO_OPTIONS.map((option) => {
            const selected = value.scenarioType === option.value;
            return (
              <label
                key={option.value}
                className={cn(
                  "cursor-pointer rounded-xl border px-3 py-3 transition",
                  selected
                    ? "border-cyan-300/45 bg-cyan-500/10 text-cyan-100"
                    : "border-white/10 bg-slate-900/45 text-slate-300 hover:border-cyan-300/25"
                )}
              >
                <div className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="credit-demand-scenario"
                    value={option.value}
                    checked={selected}
                    onChange={() =>
                      onChange({
                        ...value,
                        scenarioType: option.value,
                      })
                    }
                    disabled={loading}
                    className="accent-cyan-400"
                  />
                  <span className="text-sm font-semibold">{option.label}</span>
                </div>
                <p className="mt-2 text-xs text-slate-400">{option.description}</p>
              </label>
            );
          })}
        </div>
      </div>

      <div className="rounded-xl border border-white/10 bg-slate-950/35 p-4">
        <label className="mb-1 block text-sm font-semibold text-slate-100">Base Loan Volume Override (USD M, optional)</label>
        <Input
          type="number"
          min={1}
          step={0.1}
          placeholder="Leave blank to use model-derived baseline"
          value={value.baseVolume === null ? "" : String(value.baseVolume)}
          onChange={(event) => {
            const raw = event.target.value;
            onChange({
              ...value,
              baseVolume: raw === "" ? null : Number(raw),
            });
          }}
          disabled={loading}
        />
        <p className="mt-2 text-xs text-slate-400">
          Overrides the latest total loan-volume anchor before grade-level model projections.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={onSubmit} disabled={loading}>
          {loading ? "Forecasting..." : "Forecast by Grade"}
        </Button>
        <Button variant="secondary" onClick={onReset} disabled={loading || !onReset}>
          Reset Inputs
        </Button>
      </div>
    </div>
  );
}
