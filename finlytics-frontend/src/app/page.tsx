"use client";

import { FileText, Sparkle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { GradeBarChart } from "@/components/charts/grade-bar-chart";
import { LoanVolumeLineChart } from "@/components/charts/loan-volume-line-chart";
import { RiskPieChart } from "@/components/charts/risk-pie-chart";
import { PageHeader } from "@/components/layout/page-header";
import { MetricCard } from "@/components/metric-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  getCreditDemandByGrade,
  getDepositLeaderboard,
  getLoanVolumeForecast,
  type CreditDemandByGradeResponse,
  type DepositLeaderboardResponse,
  type LoanForecastResponse
} from "@/services/api";

export default function DashboardPage() {
  const [loading, setLoading] = useState(true);
  const [forecastData, setForecastData] = useState<LoanForecastResponse | null>(null);
  const [creditDemandData, setCreditDemandData] = useState<CreditDemandByGradeResponse | null>(null);
  const [leaderboardData, setLeaderboardData] = useState<DepositLeaderboardResponse | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [forecast, creditDemand, leaderboard] = await Promise.all([
          getLoanVolumeForecast(),
          getCreditDemandByGrade(),
          getDepositLeaderboard()
        ]);
        setForecastData(forecast);
        setCreditDemandData(creditDemand);
        setLeaderboardData(leaderboard);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to load dashboard datasets";
        toast.error(message);
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, []);

  const gradeDemand = useMemo(() => {
    const latest = creditDemandData?.trend?.at(-1);
    if (!latest) {
      return [] as Array<{ grade: string; demand: number }>;
    }
    return ["A", "B", "C", "D", "E"].map((grade) => ({
      grade,
      demand: Number(latest[grade as keyof typeof latest] || 0)
    }));
  }, [creditDemandData]);

  const riskDistribution = useMemo(() => {
    if (!gradeDemand.length) {
      return [] as Array<{ name: string; value: number }>;
    }
    const low = gradeDemand.filter((row) => row.grade === "A" || row.grade === "B").reduce((sum, row) => sum + row.demand, 0);
    const medium = gradeDemand.filter((row) => row.grade === "C").reduce((sum, row) => sum + row.demand, 0);
    const high = gradeDemand.filter((row) => row.grade === "D" || row.grade === "E").reduce((sum, row) => sum + row.demand, 0);
    return [
      { name: "Low Risk", value: low },
      { name: "Medium Risk", value: medium },
      { name: "High Risk", value: high }
    ];
  }, [gradeDemand]);

  const nextThreeMonths = useMemo(() => {
    return (forecastData?.interval ?? []).filter((item) => item.historical === null).slice(0, 3);
  }, [forecastData]);

  const overviewMetrics = useMemo(() => {
    const topModel = leaderboardData?.leaderboard?.[0];
    const latestForecast = forecastData?.interval?.at(-1);
    const demandTotal = gradeDemand.reduce((sum, row) => sum + row.demand, 0);
    const highDemand = riskDistribution.find((item) => item.name === "High Risk")?.value ?? 0;
    const highShare = demandTotal > 0 ? (highDemand / demandTotal) * 100 : 0;

    return [
      {
        label: "Best Deposit Model",
        value: topModel?.model ?? "N/A",
        delta: topModel ? `${(topModel.accuracy * 100).toFixed(1)}% accuracy` : "No leaderboard data",
        tone: "info" as const
      },
      {
        label: "Latest Loan Forecast",
        value: latestForecast ? latestForecast.value.toLocaleString() : "N/A",
        delta: latestForecast ? `CI ${latestForecast.low.toLocaleString()}-${latestForecast.high.toLocaleString()}` : "No forecast data",
        tone: "success" as const
      },
      {
        label: "Current Demand Total",
        value: demandTotal ? demandTotal.toLocaleString() : "N/A",
        delta: gradeDemand.length ? `Across grades A-E` : "No grade-demand data",
        tone: "info" as const
      },
      {
        label: "High-Risk Demand Share",
        value: `${highShare.toFixed(1)}%`,
        delta: demandTotal ? `${highDemand.toLocaleString()} high-risk units` : "No risk distribution data",
        tone: highShare >= 35 ? ("danger" as const) : highShare >= 20 ? ("warning" as const) : ("success" as const)
      },
      {
        label: "Forecast Horizon",
        value: `${nextThreeMonths.length} months`,
        delta: nextThreeMonths.length ? "Future projection rows available" : "No future rows",
        tone: "info" as const
      }
    ];
  }, [forecastData, gradeDemand, leaderboardData, nextThreeMonths, riskDistribution]);

  const highRiskShare = useMemo(() => {
    const total = riskDistribution.reduce((sum, row) => sum + row.value, 0);
    const high = riskDistribution.find((item) => item.name === "High Risk")?.value ?? 0;
    return total > 0 ? (high / total) * 100 : 0;
  }, [riskDistribution]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Dashboard Overview"
        subtitle="Unified AI command center for portfolio health, risk posture, churn exposure, and forward-looking demand signals."
        tag="Executive View"
      />

      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="secondary"
          onClick={() => {
            window.print();
            toast.success("Preparing PDF print preview");
          }}
        >
          <FileText className="h-4 w-4" />
          Export to PDF
        </Button>
        <Badge variant={highRiskShare >= 35 ? "danger" : highRiskShare >= 20 ? "warning" : "success"}>
          High-risk demand share: {highRiskShare.toFixed(1)}%
        </Badge>
        <Badge variant="info">Forecast rows: {nextThreeMonths.length}</Badge>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        {loading
          ? Array.from({ length: 5 }).map((_, index) => <Skeleton key={index} className="h-36" />)
          : overviewMetrics.map((metric) => (
              <MetricCard key={metric.label} {...metric} />
            ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">Loan Volume Trend</h3>
            <Badge variant="info">Historical + Forecast</Badge>
          </div>
          {loading ? <Skeleton className="h-[320px]" /> : null}
          {!loading && forecastData?.trend?.length ? <LoanVolumeLineChart data={forecastData.trend} /> : null}
          {!loading && !forecastData?.trend?.length ? (
            <p className="rounded-xl border border-dashed border-white/20 bg-slate-900/30 p-6 text-sm text-slate-400">
              Loan volume trend unavailable. Verify forecast endpoint dependencies.
            </p>
          ) : null}
        </Card>

        <Card>
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">Risk Distribution</h3>
            <Badge variant="danger">Live Segmentation</Badge>
          </div>
          {loading ? <Skeleton className="h-[320px]" /> : null}
          {!loading && riskDistribution.length ? <RiskPieChart data={riskDistribution} /> : null}
          {!loading && !riskDistribution.length ? (
            <p className="rounded-xl border border-dashed border-white/20 bg-slate-900/30 p-6 text-sm text-slate-400">
              Risk distribution requires credit-demand grade data.
            </p>
          ) : null}
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">Credit Demand by Grade</h3>
            <Badge variant="success">Origination Pulse</Badge>
          </div>
          {loading ? <Skeleton className="h-[320px]" /> : null}
          {!loading && gradeDemand.length ? <GradeBarChart data={gradeDemand} /> : null}
          {!loading && !gradeDemand.length ? (
            <p className="rounded-xl border border-dashed border-white/20 bg-slate-900/30 p-6 text-sm text-slate-400">
              Grade demand data unavailable. Verify grade-demand endpoint dependencies.
            </p>
          ) : null}
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-white">3-Month Forecast Summary</h3>
          <div className="mt-4 space-y-3">
            {nextThreeMonths.map((item) => (
              <div key={item.month} className="rounded-xl border border-white/10 bg-slate-950/45 p-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-slate-400">{item.month}</p>
                  <Badge variant="info">CI {item.low.toLocaleString()} - {item.high.toLocaleString()}</Badge>
                </div>
                <p className="mt-2 text-xl font-semibold text-cyan-200">{item.value.toLocaleString()}</p>
              </div>
            ))}
            {!loading && !nextThreeMonths.length ? (
              <p className="text-sm text-slate-400">No forward-looking forecast rows were returned.</p>
            ) : null}
          </div>
        </Card>
      </div>

      <Card className="border-amber-300/20 bg-amber-500/5">
        <div className="flex items-start gap-3">
          <Sparkle className="mt-0.5 h-5 w-5 text-amber-300" />
          <div>
            <h4 className="text-sm font-semibold text-amber-100">Proactive Alert</h4>
            <p className="text-sm text-amber-200/80">
              High-risk demand share currently reads {highRiskShare.toFixed(1)}% based on live grade segmentation. Review allocation policy
              if the share exceeds committee threshold.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
