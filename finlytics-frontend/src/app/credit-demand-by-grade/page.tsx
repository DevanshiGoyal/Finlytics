"use client";

import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { FeatureImportanceChart } from "@/components/charts/feature-importance-chart";
import { GradeForecastMultiLineChart } from "@/components/charts/grade-forecast-multi-line-chart";
import { GradeMultiLineChart } from "@/components/charts/grade-multi-line-chart";
import { SeasonalityHeatmap } from "@/components/charts/seasonality-heatmap";
import { EmptyState } from "@/components/empty-state";
import { CreditDemandInput } from "@/components/forms/credit-demand-input";
import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs } from "@/components/ui/tabs";
import {
  forecastCreditDemandByGrade,
  getCreditDemandByGrade,
  type CreditDemandByGradeResponse,
  type CreditDemandForecast,
  type CreditDemandForecastRequest,
  type CreditDemandForecastResponse,
} from "@/services/api";

const ALL_GRADES = ["A", "B", "C", "D", "E"];

const DEFAULT_FORECAST_REQUEST: CreditDemandForecastRequest = {
  horizon: 3,
  confidence: 0.95,
  scenarioType: "baseline",
  baseVolume: null,
};

type ForecastChartRow = {
  month: string;
  [key: string]: string | number;
};

function fmt(value: number) {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function pct(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

export default function CreditDemandByGradePage() {
  const [activeGrades, setActiveGrades] = useState<string[]>(["A", "B", "C"]);
  const [chartData, setChartData] = useState<Array<{ month: string; A: number; B: number; C: number; D: number; E: number }>>([]);
  const [heatmapData, setHeatmapData] = useState<Array<{ grade: string; values: number[] }>>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);

  const [forecastRequest, setForecastRequest] = useState<CreditDemandForecastRequest>(DEFAULT_FORECAST_REQUEST);
  const [forecastData, setForecastData] = useState<CreditDemandForecastResponse | null>(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [selectedFeatureGrade, setSelectedFeatureGrade] = useState("A");

  useEffect(() => {
    const load = async () => {
      try {
        const response = await getCreditDemandByGrade();
        setChartData(response.trend);
        setHeatmapData(response.heatmap);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to load credit-demand dataset";
        toast.error(message);
      } finally {
        setLoadingHistory(false);
      }
    };
    void load();
  }, []);

  const forecastMap = useMemo(() => {
    const map = new Map<string, CreditDemandForecast>();
    for (const item of forecastData?.forecasts ?? []) {
      map.set(item.grade, item);
    }
    return map;
  }, [forecastData]);

  const forecastChartData = useMemo<ForecastChartRow[]>(() => {
    if (!forecastData?.forecasts.length) {
      return [];
    }

    const first = forecastData.forecasts[0];
    return first.predictions.map((point, index) => {
      const row: ForecastChartRow = {
        month: point.month,
      };

      for (const grade of ALL_GRADES) {
        const gradePoint = forecastMap.get(grade)?.predictions[index];
        const central = Number(gradePoint?.forecast_central ?? 0);
        const lower = Number(gradePoint?.forecast_lower ?? 0);
        const upper = Number(gradePoint?.forecast_upper ?? 0);
        row[`${grade}_central`] = central;
        row[`${grade}_lower`] = lower;
        row[`${grade}_upper`] = upper;
        row[`${grade}_range`] = Math.max(0, upper - lower);
      }

      return row;
    });
  }, [forecastData, forecastMap]);

  const forecastGradeCards = useMemo(() => {
    return ALL_GRADES.map((grade) => {
      const item = forecastMap.get(grade);
      const horizonTotal = (item?.predictions ?? []).reduce((acc, point) => acc + point.forecast_central, 0);
      const lastPoint = item?.predictions.at(-1);
      return {
        grade,
        horizonTotal,
        finalRange: lastPoint ? `${fmt(lastPoint.forecast_lower)} - ${fmt(lastPoint.forecast_upper)}` : "N/A",
        mape: item?.modelMetrics?.mape ?? 0,
        modelName: item?.modelName ?? "N/A",
      };
    });
  }, [forecastMap]);

  const scenarioRows = useMemo(() => {
    const comparison = forecastData?.metadata?.scenarioComparison;
    if (!comparison) {
      return [];
    }
    const baseline = comparison.baseline || 0;
    return [
      { scenario: "Baseline", total: comparison.baseline, delta: 0 },
      { scenario: "Optimistic", total: comparison.optimistic, delta: baseline ? ((comparison.optimistic - baseline) / baseline) * 100 : 0 },
      { scenario: "Pessimistic", total: comparison.pessimistic, delta: baseline ? ((comparison.pessimistic - baseline) / baseline) * 100 : 0 },
    ];
  }, [forecastData]);

  const selectedFeatureData = useMemo(() => {
    return forecastMap.get(selectedFeatureGrade)?.featureImportance ?? [];
  }, [forecastMap, selectedFeatureGrade]);

  const insights = useMemo(() => {
    if (!forecastData?.forecasts.length) {
      return [];
    }

    const growthRows = forecastData.forecasts
      .map((item) => {
        const first = item.predictions[0]?.forecast_central ?? 0;
        const last = item.predictions.at(-1)?.forecast_central ?? 0;
        const growth = first > 0 ? ((last - first) / first) * 100 : 0;
        const avgBand = item.predictions.length
          ? item.predictions.reduce((acc, point) => acc + Math.max(0, point.forecast_upper - point.forecast_lower), 0) / item.predictions.length
          : 0;
        return {
          grade: item.grade,
          growth,
          avgBand,
          mape: item.modelMetrics.mape,
        };
      })
      .sort((a, b) => b.growth - a.growth);

    const topGrowth = growthRows[0];
    const topVolatility = [...growthRows].sort((a, b) => b.avgBand - a.avgBand)[0];
    const highestError = [...growthRows].sort((a, b) => b.mape - a.mape)[0];

    return [
      `Grade ${topGrowth.grade} shows the strongest projected growth (${pct(topGrowth.growth)}) across the selected horizon.`,
      `Grade ${topVolatility.grade} has the widest average confidence band (${fmt(topVolatility.avgBand)}), suggesting elevated uncertainty.`,
      `Grade ${highestError.grade} has the highest MAPE (${highestError.mape.toFixed(2)}%), so monitor it more closely in decision reviews.`,
    ];
  }, [forecastData]);

  const validateForecastRequest = (request: CreditDemandForecastRequest) => {
    if (!Number.isFinite(request.horizon) || request.horizon < 1 || request.horizon > 12) {
      throw new Error("Horizon must be between 1 and 12 months.");
    }
    if (!Number.isFinite(request.confidence) || request.confidence < 0.5 || request.confidence > 0.99) {
      throw new Error("Confidence must be between 50% and 99%.");
    }
    if (request.baseVolume !== null && (!Number.isFinite(request.baseVolume) || request.baseVolume <= 0)) {
      throw new Error("Base volume override must be a positive number when provided.");
    }
  };

  const runForecast = async () => {
    try {
      validateForecastRequest(forecastRequest);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Please review forecast inputs.");
      return;
    }

    setForecastLoading(true);
    try {
      const response = await forecastCreditDemandByGrade(forecastRequest);
      setForecastData(response);
      setSelectedFeatureGrade(response.forecasts[0]?.grade ?? "A");
      if (response.warnings?.length) {
        toast.warning("Forecast generated with fallback warnings. Review the warnings panel.");
      } else {
        toast.success("Grade-level forecast generated.");
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Credit demand forecast failed";
      toast.error(message);
    } finally {
      setForecastLoading(false);
    }
  };

  const resetForecastInputs = () => {
    setForecastRequest(DEFAULT_FORECAST_REQUEST);
  };

  const toggleGrade = (grade: string) => {
    setActiveGrades((prev) => {
      if (prev.includes(grade)) {
        return prev.filter((item) => item !== grade);
      }
      return [...prev, grade];
    });
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Credit Demand by Grade"
        subtitle="Track credit demand velocity from Grade A to E and inspect seasonal concentration patterns."
        tag="Segment Demand"
      />

      <Card>
        <div className="mb-3 flex items-center justify-between gap-2">
          <h3 className="text-lg font-semibold text-white">Forecast Controls</h3>
          <Badge variant="info">Model-Based Projection</Badge>
        </div>
        <CreditDemandInput
          value={forecastRequest}
          loading={forecastLoading}
          onChange={setForecastRequest}
          onSubmit={runForecast}
          onReset={resetForecastInputs}
        />
      </Card>

      <Card>
        <div className="mb-3 flex items-center justify-between gap-2">
          <h3 className="text-lg font-semibold text-white">Forecast Results</h3>
          <Badge variant="secondary">
            {forecastData
              ? `${forecastData.metadata.horizon}M | ${Math.round(forecastData.metadata.confidence * 100)}% CI | ${forecastData.metadata.scenario}`
              : "Awaiting Forecast"}
          </Badge>
        </div>

        {forecastLoading ? (
          <Skeleton className="h-[260px]" />
        ) : !forecastData ? (
          <EmptyState title="No forecast generated yet" description="Set inputs and click Forecast by Grade to generate model-based projections." />
        ) : (
          <div className="space-y-5">
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              {forecastGradeCards.map((item) => (
                <div key={item.grade} className="rounded-xl border border-white/10 bg-slate-950/35 p-4">
                  <p className="text-xs uppercase tracking-wide text-slate-400">Grade {item.grade}</p>
                  <p className="mt-2 text-xl font-semibold text-cyan-200">{fmt(item.horizonTotal)}</p>
                  <p className="mt-1 text-xs text-slate-400">Horizon total (USD M)</p>
                  <p className="mt-2 text-xs text-slate-300">Final range: {item.finalRange}</p>
                  <p className="mt-1 text-xs text-slate-300">MAPE: {item.mape.toFixed(2)}%</p>
                  <p className="mt-1 text-xs text-slate-400">Model: {item.modelName}</p>
                </div>
              ))}
            </div>

            <div>
              <div className="mb-3 flex items-center justify-between gap-2">
                <h4 className="text-base font-semibold text-white">Per-Grade Forecast Trend + Confidence Regions</h4>
                <Badge variant="info">Shaded CI per grade</Badge>
              </div>
              {forecastChartData.length ? (
                <GradeForecastMultiLineChart data={forecastChartData} activeGrades={activeGrades} />
              ) : (
                <EmptyState title="No forecast trend rows available" />
              )}
            </div>

            <div>
              <h4 className="mb-3 text-base font-semibold text-white">Model Metrics</h4>
              <div className="overflow-x-auto rounded-xl border border-white/10 bg-slate-950/35">
                <table className="min-w-full text-left text-sm">
                  <thead className="border-b border-white/10 bg-slate-900/65 text-slate-300">
                    <tr>
                      <th className="px-4 py-3">Grade</th>
                      <th className="px-4 py-3">Model</th>
                      <th className="px-4 py-3">MAPE (%)</th>
                      <th className="px-4 py-3">RMSE</th>
                      <th className="px-4 py-3">Training MAPE on Test (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forecastData.forecasts.map((row, index) => (
                      <tr
                        key={row.grade}
                        className={index % 2 ? "border-b border-white/5 bg-slate-950/20" : "border-b border-white/5 bg-slate-900/10"}
                      >
                        <td className="px-4 py-3 text-slate-100">Grade {row.grade}</td>
                        <td className="px-4 py-3 text-slate-300">{row.modelName ?? "N/A"}</td>
                        <td className="px-4 py-3 text-slate-200">{row.modelMetrics.mape.toFixed(2)}</td>
                        <td className="px-4 py-3 text-slate-200">{row.modelMetrics.rmse.toFixed(2)}</td>
                        <td className="px-4 py-3 text-slate-200">{row.modelMetrics.trainingMapeOnTestSet.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h4 className="mb-3 text-base font-semibold text-white">Feature Importance Breakdown</h4>
              <Tabs
                options={ALL_GRADES.map((grade) => ({ value: grade, label: `Grade ${grade}` }))}
                value={selectedFeatureGrade}
                onValueChange={setSelectedFeatureGrade}
              />
              <div className="mt-4 rounded-xl border border-white/10 bg-slate-950/35 p-3">
                {selectedFeatureData.length ? (
                  <FeatureImportanceChart data={selectedFeatureData} />
                ) : (
                  <EmptyState title="No feature importance data for selected grade" />
                )}
              </div>
            </div>

            <div>
              <h4 className="mb-3 text-base font-semibold text-white">Scenario Comparison</h4>
              {!scenarioRows.length ? (
                <EmptyState title="Scenario comparison unavailable" description="Generate a forecast to compare baseline and stress variants." />
              ) : (
                <div className="overflow-x-auto rounded-xl border border-white/10 bg-slate-950/35">
                  <table className="min-w-full text-left text-sm">
                    <thead className="border-b border-white/10 bg-slate-900/65 text-slate-300">
                      <tr>
                        <th className="px-4 py-3">Scenario</th>
                        <th className="px-4 py-3">Total Forecasted Volume (USD M)</th>
                        <th className="px-4 py-3">Delta vs Baseline</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scenarioRows.map((row, index) => (
                        <tr
                          key={row.scenario}
                          className={index % 2 ? "border-b border-white/5 bg-slate-950/20" : "border-b border-white/5 bg-slate-900/10"}
                        >
                          <td className="px-4 py-3 text-slate-100">{row.scenario}</td>
                          <td className="px-4 py-3 text-slate-200">{fmt(row.total)}</td>
                          <td className="px-4 py-3 text-slate-200">{pct(row.delta)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div>
              <h4 className="mb-3 text-base font-semibold text-white">Actionable Insights</h4>
              <div className="space-y-2">
                {insights.map((insight) => (
                  <p key={insight} className="rounded-lg border border-cyan-300/20 bg-cyan-500/8 px-3 py-2 text-sm text-cyan-100">
                    {insight}
                  </p>
                ))}
              </div>
            </div>

            {forecastData.warnings?.length ? (
              <div>
                <h4 className="mb-2 text-base font-semibold text-white">Forecast Warnings</h4>
                <div className="space-y-2">
                  {forecastData.warnings.map((warning) => (
                    <p
                      key={warning}
                      className="rounded-lg border border-amber-300/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-100"
                    >
                      {warning}
                    </p>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        )}
      </Card>

      <Card>
        <div className="flex flex-wrap items-center gap-2">
          <p className="mr-2 text-sm text-slate-300">Toggle grades:</p>
          {ALL_GRADES.map((grade) => {
            const active = activeGrades.includes(grade);
            return (
              <button
                type="button"
                key={grade}
                onClick={() => toggleGrade(grade)}
                className={
                  active
                    ? "rounded-full border border-cyan-300/40 bg-cyan-500/15 px-3 py-1.5 text-xs font-semibold text-cyan-200"
                    : "rounded-full border border-white/10 bg-slate-900/50 px-3 py-1.5 text-xs font-semibold text-slate-300"
                }
              >
                Grade {grade}
              </button>
            );
          })}
          <Badge variant="info" className="ml-auto">
            Historical Multi-Series Trend
          </Badge>
        </div>
      </Card>

      <Card>
        <h3 className="mb-3 text-lg font-semibold text-white">Historical Grade Demand Trend (A-E)</h3>
        {loadingHistory ? (
          <p className="text-sm text-slate-400">Loading live trend data...</p>
        ) : activeGrades.length && chartData.length ? (
          <GradeMultiLineChart data={chartData} activeGrades={activeGrades} />
        ) : (
          <EmptyState title="No grade trend data available" />
        )}
      </Card>

      <Card>
        <h3 className="mb-3 text-lg font-semibold text-white">Seasonality Heatmap</h3>
        {loadingHistory ? (
          <p className="text-sm text-slate-400">Loading seasonality matrix...</p>
        ) : heatmapData.length ? (
          <SeasonalityHeatmap data={heatmapData} />
        ) : (
          <EmptyState title="No seasonality data available" />
        )}
      </Card>
    </div>
  );
}
