"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";

import { ForecastAreaChart } from "@/components/charts/forecast-area-chart";
import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { getLoanVolumeForecast, type LoanForecastResponse } from "@/services/api";

export default function LoanVolumeForecastPage() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<LoanForecastResponse | null>(null);

  useEffect(() => {
    const run = async () => {
      try {
        const response = await getLoanVolumeForecast();
        setData(response);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Could not load forecast stream";
        toast.error(message);
      } finally {
        setLoading(false);
      }
    };
    void run();
  }, []);

  const forecastHighlights = (data?.interval ?? []).filter((point) => point.historical === null).slice(0, 3);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Loan Volume Forecast"
        subtitle="Visualize historical behavior, AI-projected trend, and confidence interval for the next 3-month planning cycle."
        tag="Forecast Studio"
      />

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
          <Badge variant="info">Shaded CI</Badge>
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
        <h3 className="text-lg font-semibold text-white">Analyst Notes</h3>
        <p className="mt-2 text-sm text-slate-400">
          Forecast curve remains upward-sloping while interval width stays moderate, indicating stable growth with manageable uncertainty.
          Recommended action: monitor monthly variance against lower confidence bound to detect early demand softening.
        </p>
      </Card>
    </div>
  );
}
