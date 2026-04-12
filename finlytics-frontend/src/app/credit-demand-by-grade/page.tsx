"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";

import { GradeMultiLineChart } from "@/components/charts/grade-multi-line-chart";
import { SeasonalityHeatmap } from "@/components/charts/seasonality-heatmap";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { getCreditDemandByGrade } from "@/services/api";

const ALL_GRADES = ["A", "B", "C", "D", "E"];

export default function CreditDemandByGradePage() {
  const [activeGrades, setActiveGrades] = useState<string[]>(["A", "B", "C"]);
  const [chartData, setChartData] = useState<Array<{ month: string; A: number; B: number; C: number; D: number; E: number }>>([]);
  const [heatmapData, setHeatmapData] = useState<Array<{ grade: string; values: number[] }>>([]);
  const [loading, setLoading] = useState(true);

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
        setLoading(false);
      }
    };
    void load();
  }, []);

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
            Multi-Series Trend
          </Badge>
        </div>
      </Card>

      <Card>
        <h3 className="mb-3 text-lg font-semibold text-white">Grade Demand Trend (A-E)</h3>
        {loading ? (
          <p className="text-sm text-slate-400">Loading live trend data...</p>
        ) : activeGrades.length && chartData.length ? (
          <GradeMultiLineChart data={chartData} activeGrades={activeGrades} />
        ) : (
          <EmptyState title="No grade trend data available" />
        )}
      </Card>

      <Card>
        <h3 className="mb-3 text-lg font-semibold text-white">Seasonality Heatmap</h3>
        {loading ? (
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
