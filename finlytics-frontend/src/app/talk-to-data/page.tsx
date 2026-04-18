"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  YAxis,
} from "recharts";
import {
  Bot,
  FileSpreadsheet,
  Loader2,
  MessageSquare,
  Send,
  Sparkles,
  TableProperties,
  User,
  WandSparkles,
} from "lucide-react";
import { toast } from "sonner";

import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { FileUpload } from "@/components/ui/file-upload";
import { DataTable } from "@/components/ui/table";
import {
  getTalkToDataAutoVisualize,
  getTalkToDataCorrelationMatrix,
  getTalkToDataHealth,
  queryTalkToData,
  type TalkToDataAutoVisualizeResponse,
  type TalkToDataCorrelationPoint,
  type TalkToDataHealth,
  type TalkToDataMode,
  type TalkToDataQueryResponse,
  type TalkToDataUploadResponse,
  uploadTalkToDataCSV,
} from "@/services/api";

const MODES: Array<{ value: TalkToDataMode; label: string; description: string }> = [
  {
    value: "raw",
    label: "Raw",
    description: "Runs SQL directly on uploaded data.",
  },
  {
    value: "smart",
    label: "Smart",
    description: "Applies preprocessing before query execution.",
  },
  {
    value: "scalable",
    label: "Scalable",
    description: "Uses Spark-backed preprocessing for larger workloads.",
  },
];

const CHART_COLORS = ["#38bdf8", "#22d3ee", "#34d399", "#f59e0b", "#f472b6", "#a78bfa"];
const CORRELATION_KEYWORDS = [
  "correlation matrix",
  "corr matrix",
  "correlation heatmap",
  "show correlation",
  "plot correlation",
  "heatmap",
];

type ViewMode = "chat" | "visualize";

interface UserMessage {
  id: string;
  role: "user";
  text: string;
}

interface AIMessage extends TalkToDataQueryResponse {
  id: string;
  role: "ai";
  question: string;
}

type ChatMessage = UserMessage | AIMessage;

interface DashboardSource {
  title: string;
  subtitle: string;
  chartType: string;
  chartX: string | null;
  chartY: string[];
  result: Array<Record<string, unknown>>;
}

function messageId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function isAIMessage(message: ChatMessage): message is AIMessage {
  return message.role === "ai";
}

function asksForCorrelation(question: string) {
  const lower = question.toLowerCase();
  return CORRELATION_KEYWORDS.some((keyword) => lower.includes(keyword));
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "null";
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toLocaleString() : "null";
  }

  if (typeof value === "string") {
    return value;
  }

  return JSON.stringify(value);
}

function fallbackHealth(): TalkToDataHealth {
  return {
    missing_pct: 0,
    outliers: 0,
    rows_used: 0,
    confidence: 0,
  };
}

function CorrelationHeatmap({ points }: { points: TalkToDataCorrelationPoint[] }) {
  const labels = useMemo(() => {
    const set = new Set<string>();
    for (const point of points) {
      set.add(point.col_a);
      set.add(point.col_b);
    }
    return Array.from(set);
  }, [points]);

  const lookup = useMemo(() => {
    const map = new Map<string, number | null>();
    for (const point of points) {
      map.set(`${point.col_a}::${point.col_b}`, point.correlation);
    }
    return map;
  }, [points]);

  if (!labels.length) {
    return <EmptyState title="No Correlation Data" description="Ask for a correlation matrix after uploading a dataset." />;
  }

  const bgForValue = (value: number | null) => {
    if (value === null) {
      return "rgba(148, 163, 184, 0.18)";
    }

    const intensity = Math.min(Math.abs(value), 1);
    if (value >= 0) {
      return `rgba(56, 189, 248, ${0.2 + intensity * 0.6})`;
    }
    return `rgba(244, 63, 94, ${0.2 + intensity * 0.6})`;
  };

  return (
    <div className="overflow-auto rounded-2xl border border-white/10 bg-slate-950/35 p-3">
      <table className="min-w-full border-collapse text-xs text-slate-200">
        <thead>
          <tr>
            <th className="sticky left-0 z-10 border-b border-white/10 bg-slate-900/90 px-3 py-2 text-left font-semibold">
              Field
            </th>
            {labels.map((label) => (
              <th key={label} className="border-b border-white/10 px-3 py-2 text-left font-semibold">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {labels.map((rowLabel) => (
            <tr key={rowLabel}>
              <td className="sticky left-0 z-10 border-b border-white/5 bg-slate-900/90 px-3 py-2 font-medium">
                {rowLabel}
              </td>
              {labels.map((colLabel) => {
                const value = lookup.get(`${rowLabel}::${colLabel}`) ?? null;
                return (
                  <td
                    key={`${rowLabel}-${colLabel}`}
                    className="border-b border-white/5 px-3 py-2 text-center"
                    style={{ backgroundColor: bgForValue(value) }}
                  >
                    {value === null ? "-" : value.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TalkToDataChart({
  chartType,
  chartX,
  chartY,
  result,
  title,
}: {
  chartType?: string | null;
  chartX?: string | null;
  chartY?: string[];
  result: Array<Record<string, unknown>>;
  title: string;
}) {
  if (!chartType || !result.length) {
    return <EmptyState title="No Chart Available" description="This answer did not include chart-ready data." />;
  }

  if (chartType === "correlation_matrix") {
    const points = result
      .map((row) => {
        const correlationValue = asNumber(row.correlation);
        if (typeof row.col_a !== "string" || typeof row.col_b !== "string") {
          return null;
        }

        return {
          col_a: row.col_a,
          col_b: row.col_b,
          correlation: correlationValue,
        } satisfies TalkToDataCorrelationPoint;
      })
      .filter((point): point is TalkToDataCorrelationPoint => point !== null);

    return (
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.12em] text-slate-400">{title}</p>
        <CorrelationHeatmap points={points} />
      </div>
    );
  }

  if (!chartX || !chartY?.length) {
    return <EmptyState title="No Chart Available" description="The response does not include chart axes." />;
  }

  const data = result.slice(0, 40).map((row) => {
    const datum: Record<string, string | number> = {
      name: String(row[chartX] ?? ""),
    };

    for (const key of chartY) {
      const numericValue = asNumber(row[key]);
      if (numericValue !== null) {
        datum[key] = numericValue;
      }
    }

    return datum;
  });

  if (!data.length) {
    return <EmptyState title="No Chart Available" description="The selected result cannot be plotted." />;
  }

  return (
    <div className="space-y-2">
      <p className="text-xs uppercase tracking-[0.12em] text-slate-400">{title}</p>
      <div className="h-[280px] w-full rounded-2xl border border-white/10 bg-slate-950/35 p-3">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === "pie" ? (
            <PieChart>
              <Tooltip
                contentStyle={{
                  background: "#0f172a",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: "12px",
                  color: "#e2e8f0",
                }}
              />
              <Legend />
              <Pie
                data={data.slice(0, 8).map((item) => ({
                  name: String(item.name),
                  value: asNumber(item[chartY[0]]) ?? 0,
                }))}
                dataKey="value"
                nameKey="name"
                outerRadius="78%"
                innerRadius="45%"
                paddingAngle={3}
              >
                {data.slice(0, 8).map((item, index) => (
                  <Cell key={`${String(item.name)}-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                ))}
              </Pie>
            </PieChart>
          ) : chartType === "bar" ? (
            <BarChart data={data}>
              <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="4 4" />
              <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: "#0f172a",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: "12px",
                  color: "#e2e8f0",
                }}
              />
              <Legend />
              {chartY.map((key, index) => (
                <Bar key={key} dataKey={key} fill={CHART_COLORS[index % CHART_COLORS.length]} radius={[6, 6, 0, 0]} />
              ))}
            </BarChart>
          ) : chartType === "area" ? (
            <AreaChart data={data}>
              <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="4 4" />
              <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: "#0f172a",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: "12px",
                  color: "#e2e8f0",
                }}
              />
              <Legend />
              {chartY.map((key, index) => (
                <Area
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={CHART_COLORS[index % CHART_COLORS.length]}
                  fill={CHART_COLORS[index % CHART_COLORS.length]}
                  fillOpacity={0.25}
                  strokeWidth={2.5}
                />
              ))}
            </AreaChart>
          ) : (
            <LineChart data={data}>
              <CartesianGrid stroke="rgba(148,163,184,0.2)" strokeDasharray="4 4" />
              <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: "#0f172a",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: "12px",
                  color: "#e2e8f0",
                }}
              />
              <Legend />
              {chartY.map((key, index) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={CHART_COLORS[index % CHART_COLORS.length]}
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              ))}
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ChartPanel({ source, emptyTitle }: { source: DashboardSource | null; emptyTitle: string }) {
  if (!source) {
    return <EmptyState title={emptyTitle} description="Ask a question or refresh overview visuals." />;
  }

  return (
    <Card className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold uppercase tracking-[0.1em] text-slate-300">{source.title}</h3>
        <Badge variant="info">{source.subtitle}</Badge>
      </div>
      <TalkToDataChart
        chartType={source.chartType}
        chartX={source.chartX}
        chartY={source.chartY}
        result={source.result}
        title={source.title}
      />
    </Card>
  );
}

export default function TalkToDataPage() {
  const [dataset, setDataset] = useState<TalkToDataUploadResponse | null>(null);
  const [mode, setMode] = useState<TalkToDataMode>("raw");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("chat");
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isHealthLoading, setIsHealthLoading] = useState(false);
  const [isAutoVisualizeLoading, setIsAutoVisualizeLoading] = useState(false);
  const [dataHealth, setDataHealth] = useState<TalkToDataHealth | null>(null);
  const [autoVisualizeData, setAutoVisualizeData] = useState<TalkToDataAutoVisualizeResponse | null>(null);

  const chatScrollRef = useRef<HTMLDivElement | null>(null);
  const sessionIdRef = useRef<string>(
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `session-${Date.now()}`
  );

  const latestSuccessfulMessage = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (isAIMessage(message) && !message.error && message.result.length > 0) {
        return message;
      }
    }
    return null;
  }, [messages]);

  const queryChartsReady = Boolean(
    latestSuccessfulMessage &&
      latestSuccessfulMessage.chart_type !== "correlation_matrix" &&
      latestSuccessfulMessage.chart_x &&
      latestSuccessfulMessage.chart_y.length > 0 &&
      latestSuccessfulMessage.result.length > 1
  );

  const refreshAutoVisualize = useCallback(async () => {
    if (!dataset) {
      return;
    }

    setIsAutoVisualizeLoading(true);
    try {
      const response = await getTalkToDataAutoVisualize({
        datasetId: dataset.dataset_id,
        mode,
      });
      setAutoVisualizeData(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load visual overview.";
      toast.error(message);
    } finally {
      setIsAutoVisualizeLoading(false);
    }
  }, [dataset, mode]);

  useEffect(() => {
    if (!dataset) {
      return;
    }

    let cancelled = false;
    setIsHealthLoading(true);

    getTalkToDataHealth({
      datasetId: dataset.dataset_id,
      mode,
    })
      .then((health) => {
        if (!cancelled) {
          setDataHealth(health);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setDataHealth((prev) => prev ?? fallbackHealth());
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsHealthLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [dataset, mode]);

  useEffect(() => {
    if (!dataset || viewMode !== "visualize") {
      return;
    }

    void refreshAutoVisualize();
  }, [dataset, mode, refreshAutoVisualize, viewMode]);

  useEffect(() => {
    if (!chatScrollRef.current) {
      return;
    }

    chatScrollRef.current.scrollTo({
      top: chatScrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isQuerying]);

  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      toast.error("Please upload a CSV file.");
      return;
    }

    setIsUploading(true);
    try {
      const response = await uploadTalkToDataCSV(file);
      setDataset(response);
      setMessages([]);
      setQuestion("");
      setAutoVisualizeData(null);
      setViewMode("chat");

      if (response.columns.length > 0) {
        const averageMissing =
          response.columns.reduce((sum, column) => sum + (column.null_pct || 0), 0) /
          response.columns.length;
        const confidence = Math.max(100 - Math.min(averageMissing * 1.5, 40), 0);

        setDataHealth({
          missing_pct: Number(averageMissing.toFixed(2)),
          outliers: 0,
          rows_used: response.row_count,
          confidence: Number(confidence.toFixed(1)),
        });
      } else {
        setDataHealth(fallbackHealth());
      }

      toast.success(`Dataset loaded: ${response.filename}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload failed.";
      toast.error(message);
    } finally {
      setIsUploading(false);
    }
  };

  const askQuestion = async (prefilledQuestion?: string) => {
    if (!dataset) {
      toast.error("Upload a CSV dataset first.");
      return;
    }

    const activeQuestion = (prefilledQuestion ?? question).trim();
    if (!activeQuestion || isQuerying) {
      return;
    }

    if (!prefilledQuestion) {
      setQuestion("");
    }

    const userMessage: UserMessage = {
      id: messageId(),
      role: "user",
      text: activeQuestion,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsQuerying(true);

    const isCorrelationRequest = asksForCorrelation(activeQuestion);

    try {
      let aiMessage: AIMessage;

      if (isCorrelationRequest) {
        const correlation = await getTalkToDataCorrelationMatrix({
          datasetId: dataset.dataset_id,
          method: "pearson",
        });

        aiMessage = {
          id: messageId(),
          role: "ai",
          question: activeQuestion,
          sql: `# pandas\ndf.select_dtypes(include='number').corr(method='${correlation.method}')`,
          result: correlation.data.map((point) => ({
            col_a: point.col_a,
            col_b: point.col_b,
            correlation: point.correlation,
          })),
          columns: ["col_a", "col_b", "correlation"],
          explanation: `Computed a ${correlation.method} correlation matrix across ${correlation.columns.length} numeric field(s).`,
          insights: [
            "Values close to +1 indicate strong positive relationship.",
            "Values close to -1 indicate strong negative relationship.",
            "Values around 0 suggest weak or no linear relationship.",
          ],
          chart_type: "correlation_matrix",
          chart_x: null,
          chart_y: [],
          data_health: dataHealth ?? fallbackHealth(),
          preprocessing_log: ["Computed with pandas correlation matrix endpoint."],
          mode,
          why_analysis: correlation.note || null,
          error: null,
        };
      } else {
        const response = await queryTalkToData({
          datasetId: dataset.dataset_id,
          question: activeQuestion,
          mode,
          sessionId: sessionIdRef.current,
        });

        aiMessage = {
          id: messageId(),
          role: "ai",
          question: activeQuestion,
          ...response,
        };

        if (response.data_health) {
          setDataHealth(response.data_health);
        }
      }

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Query failed.";
      toast.error(message);

      const errorMessage: AIMessage = {
        id: messageId(),
        role: "ai",
        question: activeQuestion,
        sql: "",
        result: [],
        columns: [],
        explanation: "",
        insights: [],
        chart_type: null,
        chart_x: null,
        chart_y: [],
        data_health: dataHealth ?? fallbackHealth(),
        preprocessing_log: [],
        mode,
        why_analysis: null,
        error: message,
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsQuerying(false);
    }
  };

  const datasetSchema = useMemo(
    () =>
      (dataset?.columns || []).map((column) => ({
        name: column.name,
        type: column.type,
        null_pct: column.null_pct,
      })),
    [dataset]
  );

  const trendSource: DashboardSource | null = queryChartsReady && latestSuccessfulMessage
    ? {
        title: "Trend View",
        subtitle: "From last query",
        chartType: latestSuccessfulMessage.chart_type || "line",
        chartX: latestSuccessfulMessage.chart_x || null,
        chartY: latestSuccessfulMessage.chart_y,
        result: latestSuccessfulMessage.result,
      }
    : autoVisualizeData?.trend
      ? {
          title: autoVisualizeData.trend.title,
          subtitle: "Auto generated",
          chartType: autoVisualizeData.trend.chart_type,
          chartX: autoVisualizeData.trend.chart_x,
          chartY: autoVisualizeData.trend.chart_y,
          result: autoVisualizeData.trend.result,
        }
      : null;

  const compositionSource: DashboardSource | null = queryChartsReady && latestSuccessfulMessage
    ? {
        title: "Composition",
        subtitle: "From last query",
        chartType: "pie",
        chartX: latestSuccessfulMessage.chart_x || null,
        chartY: latestSuccessfulMessage.chart_y,
        result: latestSuccessfulMessage.result,
      }
    : autoVisualizeData?.composition
      ? {
          title: autoVisualizeData.composition.title,
          subtitle: "Auto generated",
          chartType: "pie",
          chartX: autoVisualizeData.composition.chart_x,
          chartY: autoVisualizeData.composition.chart_y,
          result: autoVisualizeData.composition.result,
        }
      : null;

  const comparisonSource: DashboardSource | null = queryChartsReady && latestSuccessfulMessage
    ? {
        title: "Comparison",
        subtitle: "From last query",
        chartType: "bar",
        chartX: latestSuccessfulMessage.chart_x || null,
        chartY: latestSuccessfulMessage.chart_y,
        result: latestSuccessfulMessage.result,
      }
    : autoVisualizeData?.comparison
      ? {
          title: autoVisualizeData.comparison.title,
          subtitle: "Auto generated",
          chartType: "bar",
          chartX: autoVisualizeData.comparison.chart_x,
          chartY: autoVisualizeData.comparison.chart_y,
          result: autoVisualizeData.comparison.result,
        }
      : null;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Talk To Data"
        subtitle="Upload a CSV, ask natural-language questions, and inspect SQL, insights, and charts without affecting existing prediction modules."
        tag="Conversational Analytics"
      />

      <div className="grid gap-4 xl:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <Card className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-lg font-semibold text-white">Dataset</h3>
              {isUploading ? <Badge variant="warning">Uploading...</Badge> : null}
            </div>

            <FileUpload onFileSelect={handleUpload} title="Upload CSV for Talk To Data" />

            {dataset ? (
              <div className="space-y-3 rounded-2xl border border-white/10 bg-slate-950/30 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-slate-100">{dataset.filename}</p>
                    <p className="text-xs text-slate-400">Dataset ID: {dataset.dataset_id}</p>
                  </div>
                  <FileSpreadsheet className="h-4 w-4 text-cyan-300" />
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs text-slate-300">
                  <div className="rounded-xl border border-white/10 bg-slate-900/35 p-2">
                    <p className="text-slate-400">Rows</p>
                    <p className="mt-1 text-sm font-semibold text-slate-100">{dataset.row_count.toLocaleString()}</p>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-slate-900/35 p-2">
                    <p className="text-slate-400">Columns</p>
                    <p className="mt-1 text-sm font-semibold text-slate-100">{dataset.columns.length}</p>
                  </div>
                </div>
              </div>
            ) : (
              <EmptyState title="No Dataset Yet" description="Upload a CSV to activate chat, visualizations, and data health checks." />
            )}
          </Card>

          <Card className="space-y-3">
            <h3 className="text-lg font-semibold text-white">Query Mode</h3>
            <div className="grid grid-cols-1 gap-2">
              {MODES.map((modeOption) => (
                <button
                  key={modeOption.value}
                  type="button"
                  onClick={() => setMode(modeOption.value)}
                  className={`rounded-xl border px-3 py-2 text-left transition ${
                    mode === modeOption.value
                      ? "border-cyan-300/45 bg-cyan-500/12 text-cyan-100"
                      : "border-white/15 bg-slate-900/35 text-slate-300 hover:border-white/30"
                  }`}
                >
                  <p className="text-sm font-semibold">{modeOption.label}</p>
                  <p className="mt-0.5 text-xs text-slate-400">{modeOption.description}</p>
                </button>
              ))}
            </div>
          </Card>

          <Card className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-lg font-semibold text-white">Data Health</h3>
              {isHealthLoading ? <Loader2 className="h-4 w-4 animate-spin text-cyan-300" /> : null}
            </div>
            {dataHealth ? (
              <div className="grid grid-cols-2 gap-2 text-xs text-slate-300">
                <div className="rounded-xl border border-white/10 bg-slate-900/35 p-2">
                  <p className="text-slate-400">Missing %</p>
                  <p className="mt-1 text-sm font-semibold text-slate-100">{dataHealth.missing_pct.toFixed(1)}%</p>
                </div>
                <div className="rounded-xl border border-white/10 bg-slate-900/35 p-2">
                  <p className="text-slate-400">Outliers</p>
                  <p className="mt-1 text-sm font-semibold text-slate-100">{dataHealth.outliers}</p>
                </div>
                <div className="rounded-xl border border-white/10 bg-slate-900/35 p-2">
                  <p className="text-slate-400">Rows Used</p>
                  <p className="mt-1 text-sm font-semibold text-slate-100">{dataHealth.rows_used.toLocaleString()}</p>
                </div>
                <div className="rounded-xl border border-white/10 bg-slate-900/35 p-2">
                  <p className="text-slate-400">Confidence</p>
                  <p className="mt-1 text-sm font-semibold text-slate-100">{dataHealth.confidence.toFixed(1)}%</p>
                </div>
              </div>
            ) : (
              <EmptyState title="Awaiting Metrics" description="Upload data to start health analysis." />
            )}
          </Card>

          {dataset?.suggested_questions?.length ? (
            <Card className="space-y-3">
              <h3 className="text-lg font-semibold text-white">Suggested Questions</h3>
              <div className="space-y-2">
                {dataset.suggested_questions.map((suggestedQuestion) => (
                  <Button
                    key={suggestedQuestion}
                    variant="secondary"
                    size="sm"
                    className="w-full justify-start text-left"
                    onClick={() => {
                      void askQuestion(suggestedQuestion);
                    }}
                  >
                    <WandSparkles className="h-3.5 w-3.5" />
                    <span className="line-clamp-2">{suggestedQuestion}</span>
                  </Button>
                ))}
              </div>
            </Card>
          ) : null}
        </div>

        <Card className="space-y-4 p-0">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
            <div className="flex items-center gap-2">
              <Button
                variant={viewMode === "chat" ? "primary" : "secondary"}
                size="sm"
                onClick={() => setViewMode("chat")}
              >
                <MessageSquare className="h-4 w-4" />
                Chat
              </Button>
              <Button
                variant={viewMode === "visualize" ? "primary" : "secondary"}
                size="sm"
                onClick={() => setViewMode("visualize")}
              >
                <Sparkles className="h-4 w-4" />
                Visualize
              </Button>
            </div>

            {viewMode === "visualize" && dataset ? (
              <Button variant="secondary" size="sm" onClick={() => void refreshAutoVisualize()}>
                {isAutoVisualizeLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <WandSparkles className="h-4 w-4" />}
                Refresh Overview
              </Button>
            ) : null}
          </div>

          <div className="px-4 pb-4">
            {!dataset ? (
              <EmptyState
                title="Talk To Data Is Ready"
                description="Upload a CSV from the left panel, then switch between chat and visualization views."
              />
            ) : viewMode === "chat" ? (
              <div className="space-y-3">
                <div
                  ref={chatScrollRef}
                  className="h-[58vh] overflow-y-auto rounded-2xl border border-white/10 bg-slate-950/30 p-3"
                >
                  {!messages.length ? (
                    <EmptyState
                      title="Start Your First Query"
                      description="Ask in plain English, for example: show average deposit by month or show a correlation matrix."
                    />
                  ) : (
                    <div className="space-y-3">
                      {messages.map((message) =>
                        isAIMessage(message) ? (
                          <div key={message.id} className="rounded-2xl border border-white/10 bg-slate-900/45 p-3">
                            <div className="mb-2 flex items-center gap-2 text-xs text-cyan-200">
                              <Bot className="h-3.5 w-3.5" />
                              <span>AI Analyst</span>
                              <Badge variant="neutral" className="ml-auto">
                                {message.mode.toUpperCase()}
                              </Badge>
                            </div>

                            {message.error ? (
                              <div className="rounded-xl border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-200">
                                {message.error}
                              </div>
                            ) : (
                              <div className="space-y-3">
                                {message.sql ? (
                                  <details className="rounded-xl border border-white/10 bg-slate-950/35 p-2 text-xs text-slate-300">
                                    <summary className="cursor-pointer font-semibold text-slate-200">Generated SQL</summary>
                                    <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-[11px] leading-relaxed text-slate-300">
                                      {message.sql}
                                    </pre>
                                  </details>
                                ) : null}

                                {message.explanation ? (
                                  <p className="text-sm text-slate-200">{message.explanation}</p>
                                ) : null}

                                {message.insights.length ? (
                                  <ul className="space-y-1 text-sm text-slate-300">
                                    {message.insights.map((insight) => (
                                      <li key={insight} className="flex gap-2">
                                        <span className="mt-[6px] h-1.5 w-1.5 rounded-full bg-cyan-300" />
                                        <span>{insight}</span>
                                      </li>
                                    ))}
                                  </ul>
                                ) : null}

                                {message.why_analysis ? (
                                  <div className="rounded-xl border border-amber-400/25 bg-amber-500/10 p-3 text-xs text-amber-200">
                                    <p className="font-semibold uppercase tracking-[0.1em]">Why Analysis</p>
                                    <p className="mt-1 text-sm normal-case tracking-normal text-amber-100">
                                      {message.why_analysis}
                                    </p>
                                  </div>
                                ) : null}

                                <TalkToDataChart
                                  chartType={message.chart_type}
                                  chartX={message.chart_x}
                                  chartY={message.chart_y}
                                  result={message.result}
                                  title="Result Chart"
                                />

                                {message.columns.length && message.result.length ? (
                                  <div className="space-y-2">
                                    <p className="text-xs uppercase tracking-[0.12em] text-slate-400">Result Table</p>
                                    <DataTable<Record<string, unknown>>
                                      columns={message.columns.slice(0, 12).map((columnName) => ({
                                        key: columnName,
                                        header: columnName,
                                        render: (value) => formatCellValue(value),
                                      }))}
                                      data={message.result.slice(0, 100)}
                                      emptyText="No rows returned"
                                    />
                                  </div>
                                ) : null}

                                {message.preprocessing_log.length ? (
                                  <details className="rounded-xl border border-white/10 bg-slate-950/35 p-2 text-xs text-slate-300">
                                    <summary className="cursor-pointer font-semibold text-slate-200">
                                      Preprocessing Log ({message.preprocessing_log.length})
                                    </summary>
                                    <ul className="mt-2 space-y-1">
                                      {message.preprocessing_log.map((entry, index) => (
                                        <li key={`${message.id}-${index}`}>{entry}</li>
                                      ))}
                                    </ul>
                                  </details>
                                ) : null}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div key={message.id} className="ml-auto max-w-[85%] rounded-2xl border border-cyan-400/20 bg-cyan-500/10 p-3">
                            <div className="mb-1 flex items-center gap-2 text-xs text-cyan-200">
                              <User className="h-3.5 w-3.5" />
                              <span>You</span>
                            </div>
                            <p className="text-sm text-slate-100">{message.text}</p>
                          </div>
                        )
                      )}

                      {isQuerying ? (
                        <div className="rounded-2xl border border-white/10 bg-slate-900/40 p-3 text-sm text-slate-300">
                          <div className="flex items-center gap-2">
                            <Loader2 className="h-4 w-4 animate-spin text-cyan-300" />
                            Running query and generating insights...
                          </div>
                        </div>
                      ) : null}
                    </div>
                  )}
                </div>

                <div className="rounded-2xl border border-white/10 bg-slate-950/30 p-3">
                  <label htmlFor="talk-to-data-question" className="mb-2 block text-xs uppercase tracking-[0.12em] text-slate-400">
                    Ask Your Dataset
                  </label>
                  <textarea
                    id="talk-to-data-question"
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        void askQuestion();
                      }
                    }}
                    placeholder="Type your question. Example: show average amount by month"
                    className="h-24 w-full resize-none rounded-xl border border-white/20 bg-slate-900/70 p-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-cyan-300/45 focus:outline-none"
                    disabled={isQuerying}
                  />
                  <div className="mt-2 flex items-center justify-between gap-3">
                    <p className="text-xs text-slate-400">Press Enter to send, Shift+Enter for newline.</p>
                    <Button onClick={() => void askQuestion()} disabled={!question.trim() || isQuerying}>
                      <Send className="h-4 w-4" />
                      Run Query
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-lg font-semibold text-white">Visual Insights</h3>
                    <p className="text-sm text-slate-400">
                      {queryChartsReady
                        ? "Showing charts from your latest query result."
                        : "Showing auto-generated dataset overview charts."}
                    </p>
                  </div>
                  <Button variant="secondary" size="sm" onClick={() => void refreshAutoVisualize()}>
                    {isAutoVisualizeLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <TableProperties className="h-4 w-4" />}
                    Refresh
                  </Button>
                </div>

                <div className="grid gap-4 lg:grid-cols-2">
                  <ChartPanel source={trendSource} emptyTitle="Trend View" />
                  <ChartPanel source={compositionSource} emptyTitle="Composition View" />
                  <ChartPanel source={comparisonSource} emptyTitle="Comparison View" />

                  <Card className="space-y-3">
                    <h3 className="text-sm font-semibold uppercase tracking-[0.1em] text-slate-300">Summary</h3>
                    {latestSuccessfulMessage ? (
                      <div className="space-y-2 text-sm text-slate-300">
                        <p>
                          <span className="text-slate-400">Last question:</span> {latestSuccessfulMessage.question}
                        </p>
                        <p>
                          <span className="text-slate-400">Rows returned:</span>{" "}
                          {latestSuccessfulMessage.result.length.toLocaleString()}
                        </p>
                        <p>
                          <span className="text-slate-400">Columns returned:</span> {latestSuccessfulMessage.columns.length}
                        </p>
                      </div>
                    ) : (
                      <EmptyState title="No Query Yet" description="Run one chat query to enrich the summary panel." />
                    )}
                  </Card>
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
