"use client";

import {
  BarChart3,
  Database,
  Loader2,
  MessageSquare,
  RefreshCw,
  Sparkles
} from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { toast } from "sonner";

import { TalkToDataChart } from "@/components/charts/talk-to-data-chart";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { FileUpload } from "@/components/ui/file-upload";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { DataTable } from "@/components/ui/table";
import { Tabs } from "@/components/ui/tabs";
import {
  getTalkToDataAutoVisualize,
  queryTalkToData,
  uploadTalkToDataCSV,
  type TalkToDataAutoVisualizeResponse,
  type TalkToDataChartDataset,
  type TalkToDataMode,
  type TalkToDataQueryResponse,
  type TalkToDataUploadResponse
} from "@/services/api";

const modeOptions: Array<{ label: string; value: TalkToDataMode }> = [
  { label: "Raw SQL", value: "raw" },
  { label: "Smart", value: "smart" },
  { label: "Scalable", value: "scalable" }
];

const viewOptions = [
  { value: "chat", label: "Chat" },
  { value: "visualize", label: "Visualize" }
] as const;

type ViewState = (typeof viewOptions)[number]["value"];

function formatCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "-";
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toLocaleString() : String(value);
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (Array.isArray(value) || typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}

function toTableColumns(columnNames: string[]) {
  return columnNames.map((column) => ({
    key: column as keyof Record<string, unknown>,
    header: column,
    render: (value: unknown) => formatCell(value)
  }));
}

export default function TalkToDataPage() {
  const [dataset, setDataset] = useState<TalkToDataUploadResponse | null>(null);
  const [mode, setMode] = useState<TalkToDataMode>("smart");
  const [view, setView] = useState<ViewState>("chat");
  const [question, setQuestion] = useState("");
  const [lastQuestion, setLastQuestion] = useState("");
  const [queryResult, setQueryResult] = useState<TalkToDataQueryResponse | null>(null);
  const [autoVisualize, setAutoVisualize] = useState<TalkToDataAutoVisualizeResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [autoLoading, setAutoLoading] = useState(false);
  const [autoError, setAutoError] = useState<string | null>(null);

  const loadAutoVisualize = useCallback(
    async (showToastOnError = false) => {
      if (!dataset?.dataset_id) {
        return;
      }

      setAutoLoading(true);
      setAutoError(null);

      try {
        const data = await getTalkToDataAutoVisualize({
          datasetId: dataset.dataset_id,
          mode
        });
        setAutoVisualize(data);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to generate automatic visualizations";
        setAutoError(message);
        if (showToastOnError) {
          toast.error(message);
        }
      } finally {
        setAutoLoading(false);
      }
    },
    [dataset?.dataset_id, mode]
  );

  const handleUpload = useCallback(async (file: File) => {
    setUploading(true);

    try {
      const uploaded = await uploadTalkToDataCSV(file);
      setDataset(uploaded);
      setQueryResult(null);
      setLastQuestion("");
      setAutoVisualize(null);
      setAutoError(null);
      setView("chat");
      setQuestion(uploaded.suggested_questions?.[0] ?? "");
      toast.success(`Loaded ${uploaded.filename} with ${uploaded.row_count.toLocaleString()} rows`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "CSV upload failed";
      toast.error(message);
    } finally {
      setUploading(false);
    }
  }, []);

  const runQuery = useCallback(
    async (overrideQuestion?: string) => {
      if (!dataset?.dataset_id) {
        toast.error("Upload a CSV dataset first.");
        return;
      }

      const finalQuestion = (overrideQuestion ?? question).trim();
      if (!finalQuestion) {
        toast.error("Enter a question before querying.");
        return;
      }

      setQuerying(true);

      try {
        const response = await queryTalkToData({
          datasetId: dataset.dataset_id,
          question: finalQuestion,
          mode
        });

        setQueryResult(response);
        setLastQuestion(finalQuestion);
        setQuestion("");
        toast.success("Query processed successfully");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to process question";
        toast.error(message);
      } finally {
        setQuerying(false);
      }
    },
    [dataset?.dataset_id, mode, question]
  );

  const sampleRows = useMemo(() => dataset?.sample ?? [], [dataset?.sample]);
  const sampleColumnNames = useMemo(() => {
    if (!sampleRows.length) {
      return dataset?.columns.map((column) => column.name) ?? [];
    }

    return Object.keys(sampleRows[0] ?? {});
  }, [dataset?.columns, sampleRows]);

  const sampleColumns = useMemo(() => toTableColumns(sampleColumnNames), [sampleColumnNames]);

  const resultRows = useMemo(() => queryResult?.result ?? [], [queryResult?.result]);
  const resultColumnNames = useMemo(() => {
    if (!queryResult) {
      return [] as string[];
    }

    if (queryResult.columns.length) {
      return queryResult.columns;
    }

    return resultRows.length ? Object.keys(resultRows[0] ?? {}) : [];
  }, [queryResult, resultRows]);

  const resultColumns = useMemo(() => toTableColumns(resultColumnNames), [resultColumnNames]);

  const queryChartReady = Boolean(queryResult?.chart_x && queryResult?.chart_y.length && queryResult.result.length);

  const autoPanels = useMemo(() => {
    const panels: TalkToDataChartDataset[] = [];

    if (autoVisualize?.trend) {
      panels.push(autoVisualize.trend);
    }

    if (autoVisualize?.composition) {
      panels.push(autoVisualize.composition);
    }

    if (autoVisualize?.comparison) {
      panels.push(autoVisualize.comparison);
    }

    return panels;
  }, [autoVisualize]);

  const onSwitchView = (next: string) => {
    const nextView: ViewState = next === "visualize" ? "visualize" : "chat";
    setView(nextView);

    if (nextView === "visualize" && dataset?.dataset_id && !autoVisualize && !autoLoading) {
      void loadAutoVisualize(false);
    }
  };

  const onSuggestedQuestion = (text: string) => {
    setQuestion(text);
    void runQuery(text);
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="DataLens"
        subtitle="Upload a CSV, ask natural-language questions, and switch to visualization mode for instant charting, all from your local NLP2SQL backend."
        tag="NLP2SQL"
      />

      <div className="grid gap-4 xl:grid-cols-4">
        <Card className="space-y-4 xl:col-span-1">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-lg font-semibold text-white">Dataset Control</h3>
            <Badge variant={dataset ? "success" : "warning"}>{dataset ? "Loaded" : "Awaiting"}</Badge>
          </div>

          <FileUpload
            title={uploading ? "Uploading CSV..." : "Upload CSV Dataset"}
            onFileSelect={(file) => {
              void handleUpload(file);
            }}
          />

          {uploading ? (
            <p className="text-xs text-slate-400">Uploading and profiling dataset...</p>
          ) : null}

          <div>
            <label className="mb-1 block text-xs text-slate-400">Query Mode</label>
            <Select
              value={mode}
              onChange={(event) => {
                setMode(event.target.value as TalkToDataMode);
                setAutoVisualize(null);
                setAutoError(null);
              }}
              options={modeOptions}
            />
          </div>

          {dataset ? (
            <div className="rounded-xl border border-white/10 bg-slate-950/35 p-3 text-xs text-slate-300">
              <div className="flex items-center gap-2 text-cyan-200">
                <Database className="h-4 w-4" />
                <span className="font-semibold">{dataset.filename}</span>
              </div>
              <p className="mt-2 text-slate-400">Rows: {dataset.row_count.toLocaleString()}</p>
              <p className="text-slate-400">Columns: {dataset.columns.length}</p>
            </div>
          ) : (
            <EmptyState title="No Dataset Yet" message="Upload CSV data to enable querying." />
          )}

          {dataset?.suggested_questions.length ? (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Suggested Questions</h4>
              <div className="flex flex-wrap gap-2">
                {dataset.suggested_questions.slice(0, 6).map((item) => (
                  <button
                    key={item}
                    type="button"
                    className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-1.5 text-left text-xs text-slate-200 transition hover:border-cyan-300/35 hover:bg-cyan-400/10"
                    onClick={() => onSuggestedQuestion(item)}
                    disabled={querying}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </Card>

        <div className="space-y-4 xl:col-span-3">
          <Card className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <Tabs options={viewOptions.map((option) => ({ ...option }))} value={view} onValueChange={onSwitchView} />
              <Badge variant="info">Local API: 127.0.0.1:8000</Badge>
            </div>

            <div className="flex flex-col gap-2 md:flex-row">
              <Input
                placeholder={dataset ? "Ask a question about your uploaded data" : "Upload CSV to start"}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void runQuery();
                  }
                }}
                disabled={!dataset || querying}
              />
              <Button
                onClick={() => {
                  void runQuery();
                }}
                disabled={!dataset || querying}
              >
                {querying ? <Loader2 className="h-4 w-4 animate-spin" /> : <MessageSquare className="h-4 w-4" />}
                Ask
              </Button>
            </div>
          </Card>

          {view === "chat" ? (
            <>
              <Card className="space-y-4">
                {!queryResult ? (
                  <EmptyState
                    title="Ask Your First Question"
                    message="Query output will appear here with SQL, explanation, and recommendations."
                  />
                ) : (
                  <>
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <h3 className="text-lg font-semibold text-white">Latest Response</h3>
                      <Badge variant="success">Mode: {queryResult.mode}</Badge>
                    </div>

                    <div className="rounded-xl border border-cyan-400/15 bg-cyan-500/5 p-3 text-sm text-cyan-100">
                      <p className="text-xs uppercase tracking-wide text-cyan-300">Question</p>
                      <p className="mt-1">{lastQuestion}</p>
                    </div>

                    <div className="space-y-2">
                      <p className="text-xs uppercase tracking-wide text-slate-400">Generated SQL</p>
                      <pre className="overflow-x-auto rounded-xl border border-white/10 bg-slate-950/60 p-3 text-xs text-cyan-200">
                        {queryResult.sql}
                      </pre>
                    </div>

                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-400">Explanation</p>
                      <p className="mt-1 text-sm text-slate-200">{queryResult.explanation}</p>
                    </div>

                    {queryResult.insights.length ? (
                      <div className="space-y-2">
                        <p className="text-xs uppercase tracking-wide text-slate-400">Insights</p>
                        <div className="flex flex-wrap gap-2">
                          {queryResult.insights.map((insight) => (
                            <Badge key={insight} variant="info" className="normal-case tracking-normal">
                              {insight}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    <div className="grid gap-2 md:grid-cols-4">
                      <Badge variant="secondary" className="justify-center">
                        Confidence: {queryResult.data_health.confidence.toFixed(1)}%
                      </Badge>
                      <Badge variant="secondary" className="justify-center">
                        Missing: {queryResult.data_health.missing_pct.toFixed(2)}%
                      </Badge>
                      <Badge variant="secondary" className="justify-center">
                        Outliers: {queryResult.data_health.outliers}
                      </Badge>
                      <Badge variant="secondary" className="justify-center">
                        Rows Used: {queryResult.data_health.rows_used}
                      </Badge>
                    </div>
                  </>
                )}
              </Card>

              <Card className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="text-lg font-semibold text-white">Query Result Table</h3>
                  <Badge variant="info">{resultRows.length} rows</Badge>
                </div>
                {queryResult && resultRows.length ? (
                  <DataTable<Record<string, unknown>> columns={resultColumns} data={resultRows} />
                ) : (
                  <EmptyState title="No Result Rows" message="Run a question to populate records." />
                )}
              </Card>
            </>
          ) : (
            <>
              <Card className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-cyan-300" />
                    <h3 className="text-lg font-semibold text-white">Visualization Studio</h3>
                  </div>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => {
                      void loadAutoVisualize(true);
                    }}
                    disabled={!dataset || autoLoading}
                  >
                    {autoLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                    Refresh Auto Visualize
                  </Button>
                </div>

                {queryChartReady && queryResult ? (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-amber-300" />
                      <p className="text-sm font-semibold text-slate-100">From Latest Query</p>
                    </div>
                    <TalkToDataChart
                      data={queryResult.result}
                      chartType={queryResult.chart_type}
                      xKey={queryResult.chart_x}
                      yKeys={queryResult.chart_y}
                    />
                  </div>
                ) : (
                  <EmptyState
                    title="No Query Chart Yet"
                    message="Ask a grouped or trend-style question to generate chart metadata."
                  />
                )}
              </Card>

              {autoError ? (
                <Card>
                  <p className="text-sm text-amber-200">Auto visualization warning: {autoError}</p>
                </Card>
              ) : null}

              {autoPanels.length ? (
                autoPanels.map((panel, index) => (
                  <Card key={`${panel.title}-${index}`} className="space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <h4 className="text-base font-semibold text-white">{panel.title}</h4>
                      <Badge variant="info">Auto</Badge>
                    </div>
                    <TalkToDataChart
                      data={panel.result}
                      chartType={panel.chart_type}
                      xKey={panel.chart_x}
                      yKeys={panel.chart_y}
                    />
                  </Card>
                ))
              ) : (
                <Card>
                  <EmptyState title="Auto Charts Not Ready" message="Use refresh to generate dataset-level visual summaries." />
                </Card>
              )}
            </>
          )}

          {dataset ? (
            <Card className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-lg font-semibold text-white">Uploaded Sample Preview</h3>
                <Badge variant="secondary">Top 5 Rows</Badge>
              </div>
              {sampleRows.length ? (
                <DataTable<Record<string, unknown>> columns={sampleColumns} data={sampleRows.slice(0, 5)} />
              ) : (
                <EmptyState title="No Sample Returned" message="Upload endpoint did not return preview rows." />
              )}
            </Card>
          ) : null}
        </div>
      </div>
    </div>
  );
}
