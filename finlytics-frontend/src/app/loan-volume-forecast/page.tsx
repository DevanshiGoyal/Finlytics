"use client";

import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { ForecastAreaChart } from "@/components/charts/forecast-area-chart";
import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  getLoanVolumeForecast,
  predictLoanVolumeForecast,
  type LoanForecastRequest,
  type LoanForecastResponse,
  type LoanForecastScenario,
} from "@/services/api";

type ForecastFormState = {
  horizonMonths: string;
  scenario: LoanForecastScenario;
  growthAdjustmentPct: string;
  interestRateShockBps: string;
  loanCountChangePct: string;
  avgLoanAmountChangePct: string;
};

const DEFAULT_FORM: ForecastFormState = {
  horizonMonths: "3",
  scenario: "baseline",
  growthAdjustmentPct: "0",
  interestRateShockBps: "0",
  loanCountChangePct: "0",
  avgLoanAmountChangePct: "0",
};

const SCENARIO_OPTIONS: Array<{ label: string; value: LoanForecastScenario }> = [
  { label: "Baseline", value: "baseline" },
  { label: "Optimistic", value: "optimistic" },
  { label: "Conservative", value: "conservative" },
  { label: "Stress", value: "stress" },
];

function parseWithRange(
  raw: string,
  fieldLabel: string,
  min: number,
  max: number,
  mustBeInteger = false,
) {
  const numeric = Number(raw);
  if (!Number.isFinite(numeric)) {
    throw new Error(`${fieldLabel} must be numeric`);
  }
  if (mustBeInteger && !Number.isInteger(numeric)) {
    throw new Error(`${fieldLabel} must be a whole number`);
  }
  if (numeric < min || numeric > max) {
    throw new Error(`${fieldLabel} must be between ${min} and ${max}`);
  }
  return numeric;
}

function buildForecastPayload(form: ForecastFormState): LoanForecastRequest {
  const horizonMonths = parseWithRange(form.horizonMonths, "Horizon", 1, 12, true);
  const growthAdjustmentPct = parseWithRange(form.growthAdjustmentPct, "Growth adjustment", -40, 60);
  const interestRateShockBps = parseWithRange(form.interestRateShockBps, "Interest-rate shock", -400, 400);
  const loanCountChangePct = parseWithRange(form.loanCountChangePct, "Loan-count change", -50, 120);
  const avgLoanAmountChangePct = parseWithRange(
    form.avgLoanAmountChangePct,
    "Average-loan-amount change",
    -50,
    80,
  );

  return {
    horizonMonths,
    scenario: form.scenario,
    growthAdjustmentPct,
    interestRateShockBps,
    loanCountChangePct,
    avgLoanAmountChangePct,
  };
}

function pctText(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;
}

function bpsText(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(0)} bps`;
}

export default function LoanVolumeForecastPage() {
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [data, setData] = useState<LoanForecastResponse | null>(null);
  const [form, setForm] = useState<ForecastFormState>(DEFAULT_FORM);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const run = async () => {
      try {
        const response = await getLoanVolumeForecast();
        setData(response);
        setErrorMessage(null);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Could not load forecast stream";
        setErrorMessage(message);
        toast.error(message);
      } finally {
        setLoading(false);
      }
    };
    void run();
  }, []);

  const forecastRows = useMemo(
    () => (data?.interval ?? []).filter((point) => point.historical === null),
    [data],
  );

  const forecastHighlights = forecastRows.slice(0, 3);

  const runPrediction = async () => {
    let payload: LoanForecastRequest;
    try {
      payload = buildForecastPayload(form);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Please review your forecast inputs.";
      setErrorMessage(message);
      toast.error(message);
      return;
    }

    setSubmitting(true);
    setErrorMessage(null);
    try {
      const response = await predictLoanVolumeForecast(payload);
      setData(response);
      toast.success("Forecast updated using trained Module 3 artifacts");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Loan volume prediction failed";
      setErrorMessage(message);
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  const resetToBaseline = async () => {
    setForm(DEFAULT_FORM);
    setSubmitting(true);
    setErrorMessage(null);
    try {
      const response = await getLoanVolumeForecast();
      setData(response);
      toast.success("Baseline forecast reloaded");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not reload baseline forecast";
      setErrorMessage(message);
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Loan Volume Forecast"
        subtitle="Run scenario-aware loan volume predictions from real Module 3 trained artifacts, with configurable planning assumptions."
        tag="Forecast Studio"
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <h3 className="text-lg font-semibold text-white">Forecast Inputs</h3>
          <p className="mt-1 text-sm text-slate-400">
            Tune horizon and scenario parameters, then generate a fresh projection.
          </p>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            <div>
              <label className="mb-1 block text-xs text-slate-400">Forecast Horizon (months)</label>
              <Select
                value={form.horizonMonths}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    horizonMonths: event.target.value,
                  }))
                }
                options={Array.from({ length: 12 }).map((_, idx) => ({
                  label: String(idx + 1),
                  value: String(idx + 1),
                }))}
              />
            </div>

            <div>
              <label className="mb-1 block text-xs text-slate-400">Scenario</label>
              <Select
                value={form.scenario}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    scenario: event.target.value as LoanForecastScenario,
                  }))
                }
                options={SCENARIO_OPTIONS}
              />
            </div>

            <div>
              <label className="mb-1 block text-xs text-slate-400">Growth Adjustment (%)</label>
              <Input
                type="number"
                step="0.5"
                min={-40}
                max={60}
                value={form.growthAdjustmentPct}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    growthAdjustmentPct: event.target.value,
                  }))
                }
              />
            </div>

            <div>
              <label className="mb-1 block text-xs text-slate-400">Interest-Rate Shock (bps)</label>
              <Input
                type="number"
                step="5"
                min={-400}
                max={400}
                value={form.interestRateShockBps}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    interestRateShockBps: event.target.value,
                  }))
                }
              />
            </div>

            <div>
              <label className="mb-1 block text-xs text-slate-400">Loan-Count Change (%)</label>
              <Input
                type="number"
                step="0.5"
                min={-50}
                max={120}
                value={form.loanCountChangePct}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    loanCountChangePct: event.target.value,
                  }))
                }
              />
            </div>

            <div>
              <label className="mb-1 block text-xs text-slate-400">Avg Loan Amount Change (%)</label>
              <Input
                type="number"
                step="0.5"
                min={-50}
                max={80}
                value={form.avgLoanAmountChangePct}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    avgLoanAmountChangePct: event.target.value,
                  }))
                }
              />
            </div>
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Button onClick={runPrediction} disabled={submitting || loading}>
              {submitting ? "Predicting..." : "Predict Loan Volume"}
            </Button>
            <Button variant="secondary" onClick={resetToBaseline} disabled={submitting || loading}>
              Reset Baseline
            </Button>
          </div>

          <p className="mt-3 text-xs text-slate-500">
            Validation ranges: horizon 1-12 months, growth -40% to +60%, rate shock -400 to +400 bps, loan-count -50% to +120%,
            average-loan-amount -50% to +80%.
          </p>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-white">Prediction Summary</h3>
          {submitting ? <Skeleton className="mt-4 h-44" /> : null}
          {!submitting && data?.summary ? (
            <div className="mt-4 space-y-3">
              <div>
                <p className="text-xs text-slate-400">Projected Total</p>
                <p className="text-3xl font-semibold text-cyan-200">{data.summary.projectedTotal.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-xs text-slate-400">Average Monthly</p>
                <p className="text-xl font-semibold text-white">{data.summary.averageMonthly.toLocaleString()}</p>
              </div>
              <Badge variant="info">
                Growth vs last actual: {pctText(data.summary.growthVsLastActualPct)}
              </Badge>
              <p className="text-xs text-slate-400">
                Final month {data.summary.finalMonth}: {data.summary.finalValue.toLocaleString()} (range {data.summary.finalRange.low.toLocaleString()} - {" "}
                {data.summary.finalRange.high.toLocaleString()})
              </p>
            </div>
          ) : null}
          {!submitting && !data?.summary ? (
            <p className="mt-3 text-sm text-slate-400">
              Run a prediction to see horizon totals, growth impact, and final confidence range.
            </p>
          ) : null}
          {errorMessage ? (
            <p className="mt-3 rounded-lg border border-red-400/25 bg-red-500/10 px-3 py-2 text-xs text-red-200">
              {errorMessage}
            </p>
          ) : null}
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {forecastHighlights.map((point) => (
          <Card key={point.month}>
            <p className="text-sm text-slate-400">{point.month}</p>
            <p className="mt-1 text-2xl font-semibold text-cyan-200">{point.value.toLocaleString()}</p>
            <Badge variant="info" className="mt-2">
              CI {point.low.toLocaleString()} - {point.high.toLocaleString()}
            </Badge>
          </Card>
        ))}
        {!loading && !forecastHighlights.length ? (
          <Card className="md:col-span-3">
            <p className="text-sm text-slate-400">No forward projection rows were returned by the forecast endpoint.</p>
          </Card>
        ) : null}
      </div>

      <Card>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-lg font-semibold text-white">Historical + Forecast + Confidence Region</h3>
          <Badge variant="info">Trained Model Projection</Badge>
        </div>
        {loading ? <Skeleton className="h-[340px]" /> : null}
        {!loading && data ? <ForecastAreaChart data={data.interval} /> : null}
        {!loading && !data ? (
          <p className="rounded-xl border border-dashed border-white/20 bg-slate-900/30 p-6 text-sm text-slate-400">
            Forecast dataset unavailable. Verify API or retry.
          </p>
        ) : null}
      </Card>

      <Card>
        <h3 className="text-lg font-semibold text-white">Forecast Output Grid</h3>
        {!forecastRows.length ? (
          <p className="mt-2 text-sm text-slate-400">No forecast rows returned for the selected scenario.</p>
        ) : (
          <div className="mt-3 overflow-x-auto">
            <table className="min-w-full divide-y divide-white/10 text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-slate-400">
                  <th className="py-2 pr-4">Month</th>
                  <th className="py-2 pr-4">Likely</th>
                  <th className="py-2 pr-4">Low</th>
                  <th className="py-2 pr-4">High</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {forecastRows.map((row) => (
                  <tr key={`${row.month}-${row.value}`} className="text-slate-200">
                    <td className="py-2 pr-4">{row.month}</td>
                    <td className="py-2 pr-4">{row.value.toLocaleString()}</td>
                    <td className="py-2 pr-4">{row.low.toLocaleString()}</td>
                    <td className="py-2 pr-4">{row.high.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {data?.summary ? (
          <div className="mt-4 grid gap-2 md:grid-cols-2 xl:grid-cols-4">
            <Badge variant="secondary">Scenario: {data.summary.scenario}</Badge>
            <Badge variant="secondary">Growth adj: {pctText(data.summary.assumptions.growthAdjustmentPct)}</Badge>
            <Badge variant="secondary">Rate shock: {bpsText(data.summary.assumptions.interestRateShockBps)}</Badge>
            <Badge variant="secondary">Loan count adj: {pctText(data.summary.assumptions.loanCountChangePct)}</Badge>
          </div>
        ) : null}

        {data?.warnings?.length ? (
          <div className="mt-4 space-y-2">
            {data.warnings.map((warning) => (
              <p
                key={warning}
                className="rounded-lg border border-amber-300/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-100"
              >
                {warning}
              </p>
            ))}
          </div>
        ) : null}
      </Card>
    </div>
  );
}
