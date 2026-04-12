"use client";

import { Download, FileDown, Upload } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";

import { FeatureImportanceChart } from "@/components/charts/feature-importance-chart";
import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { FileUpload } from "@/components/ui/file-upload";
import { Modal } from "@/components/ui/modal";
import { Tabs } from "@/components/ui/tabs";
import { DataTable } from "@/components/ui/table";
import { batchScorePortfolio } from "@/services/api";

function exportCsv(rows: Array<Record<string, unknown>>, filename: string) {
  const headers = Object.keys(rows[0] || {});
  const csv = [headers.join(","), ...rows.map((row) => headers.map((header) => String(row[header] ?? "")).join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function parseCsv(file: File): Promise<Array<Record<string, string>>> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || "");
      const rows = text.trim().split(/\r?\n/);
      const headers = rows[0]?.split(",") || [];
      const mapped = rows.slice(1).map((line) => {
        const values = line.split(",");
        return headers.reduce<Record<string, string>>((acc, header, index) => {
          acc[header] = values[index] || "";
          return acc;
        }, {});
      });
      resolve(mapped);
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

export default function PortfolioIntelligenceHubPage() {
  const [tab, setTab] = useState("stress-testing");
  const [shock, setShock] = useState(150);
  const [incomeDrop, setIncomeDrop] = useState(12);
  const [batchPreview, setBatchPreview] = useState<Array<Record<string, string>>>([]);
  const [batchRows, setBatchRows] = useState<
    Array<{ id: string; borrower: string; defaultRisk: number; churnRisk: number; riskBand: string }>
  >([]);
  const [featureImportance, setFeatureImportance] = useState<Array<{ feature: string; importance: number }>>([]);
  const [driftMetrics, setDriftMetrics] = useState<Array<{ feature: string; status: string; drift_score?: number }>>([]);
  const [modalOpen, setModalOpen] = useState(false);

  const reportText = useMemo(() => {
    const highRisk = batchRows.filter((row) => row.riskBand === "High").length;
    return `# Finlytics Portfolio Intelligence Report\n\n- Total records scored: ${batchRows.length}\n- High risk exposures: ${highRisk}\n- Stress assumption: +${shock} bps, income drop ${incomeDrop}%\n- Drift watch: ${driftMetrics.filter((item) => item.status !== "Stable").length} features require monitoring\n\nRecommendation: Run risk committee review for high-risk cohort before next allocation cycle.`;
  }, [batchRows, driftMetrics, incomeDrop, shock]);

  const runBatchScoring = async () => {
    try {
      if (!batchPreview.length) {
        toast.error("Upload a portfolio CSV before batch scoring");
        return;
      }
      const response = await batchScorePortfolio({ rows: batchPreview });
      setBatchRows(response.rows);
      setFeatureImportance(response.featureImportance ?? []);
      setDriftMetrics(response.drift ?? []);
      toast.success("Batch scoring complete");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Batch scoring failed";
      toast.error(message);
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Portfolio Intelligence Hub"
        subtitle="Stress testing, batch scoring, explainability, drift monitoring, and executive report generation in one control room."
        tag="Command Center"
      />

      <Tabs
        value={tab}
        onValueChange={setTab}
        options={[
          { value: "stress-testing", label: "Stress Testing" },
          { value: "batch-scoring", label: "Batch Scoring" },
          { value: "explainability", label: "Explainability" },
          { value: "drift-monitoring", label: "Drift Monitoring" },
          { value: "report-generator", label: "Report Generator" }
        ]}
      />

      {tab === "stress-testing" ? (
        <Card className="space-y-4">
          <h3 className="text-lg font-semibold text-white">Stress Testing Simulator</h3>
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <label className="block text-xs text-slate-400">Rate Shock (+bps): {shock}</label>
              <input
                type="range"
                className="mt-2 w-full"
                min={0}
                max={500}
                value={shock}
                onChange={(event) => setShock(Number(event.target.value))}
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400">Income Drop (%): {incomeDrop}</label>
              <input
                type="range"
                className="mt-2 w-full"
                min={0}
                max={40}
                value={incomeDrop}
                onChange={(event) => setIncomeDrop(Number(event.target.value))}
              />
            </div>
          </div>
          <p className="text-sm text-slate-400">
            Stress scenario outputs require a dedicated stress-testing model artifact or notebook pipeline. No synthetic scenario table is generated.
          </p>
        </Card>
      ) : null}

      {tab === "batch-scoring" ? (
        <Card className="space-y-4">
          <h3 className="text-lg font-semibold text-white">Batch Scoring Workflow</h3>
          <FileUpload
            title="Upload portfolio CSV"
            onFileSelect={async (file) => {
              const parsed = await parseCsv(file);
              setBatchPreview(parsed);
              toast.success("CSV loaded for preview");
            }}
          />

          {batchPreview.length ? (
            <div>
              <p className="mb-2 text-sm text-slate-400">CSV Preview ({batchPreview.length} rows)</p>
              <DataTable
                columns={Object.keys(batchPreview[0]).slice(0, 4).map((key) => ({ key, header: key }))}
                data={batchPreview.slice(0, 8)}
              />
            </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => void runBatchScoring()}>
              <Upload className="h-4 w-4" />
              Run Batch Score
            </Button>
            <Button
              variant="secondary"
              onClick={() => {
                exportCsv(batchRows as Array<Record<string, unknown>>, "portfolio_scored.csv");
                toast.success("Scored CSV downloaded");
              }}
            >
              <Download className="h-4 w-4" />
              Download Results
            </Button>
          </div>

          <DataTable
            columns={[
              { key: "id", header: "Loan ID" },
              { key: "borrower", header: "Borrower" },
              {
                key: "defaultRisk",
                header: "Default Risk",
                render: (value) => `${(Number(value) * 100).toFixed(1)}%`
              },
              {
                key: "churnRisk",
                header: "Churn Risk",
                render: (value) => `${(Number(value) * 100).toFixed(1)}%`
              },
              {
                key: "riskBand",
                header: "Risk Band",
                render: (value) => (
                  <Badge
                    variant={value === "High" ? "danger" : value === "Medium" ? "warning" : "success"}
                  >
                    {String(value)}
                  </Badge>
                )
              }
            ]}
            data={batchRows}
          />
        </Card>
      ) : null}

      {tab === "explainability" ? (
        <Card className="space-y-3">
          <h3 className="text-lg font-semibold text-white">Explainability Workspace</h3>
          <p className="text-sm text-slate-400">
            Global feature ranking across scored portfolio. Use this to validate model behavior before policy changes.
          </p>
          {featureImportance.length ? (
            <FeatureImportanceChart data={featureImportance} />
          ) : (
            <p className="text-sm text-slate-400">Run batch scoring to populate explainability metrics.</p>
          )}
        </Card>
      ) : null}

      {tab === "drift-monitoring" ? (
        <Card className="space-y-3">
          <h3 className="text-lg font-semibold text-white">Drift Monitoring</h3>
          {driftMetrics.length ? (
            <DataTable
              columns={[
                { key: "feature", header: "Feature" },
                {
                  key: "drift_score",
                  header: "Drift Score",
                  render: (value) => (value == null ? "-" : Number(value).toFixed(3))
                },
                {
                  key: "status",
                  header: "Status",
                  render: (value) => (
                    <Badge
                      variant={value === "Drift" ? "danger" : value === "Watch" ? "warning" : "success"}
                    >
                      {String(value)}
                    </Badge>
                  )
                }
              ]}
              data={driftMetrics}
            />
          ) : (
            <p className="text-sm text-slate-400">Run batch scoring to generate drift diagnostics.</p>
          )}
        </Card>
      ) : null}

      {tab === "report-generator" ? (
        <Card className="space-y-3">
          <h3 className="text-lg font-semibold text-white">Report Generator</h3>
          <p className="text-sm text-slate-400">Generate an executive summary with key risk and drift highlights.</p>
          <textarea
            value={reportText}
            readOnly
            className="h-56 w-full rounded-xl border border-white/10 bg-slate-950/50 p-3 text-sm text-slate-200"
          />
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={() => setModalOpen(true)}>
              <FileDown className="h-4 w-4" />
              Preview Report
            </Button>
            <Button
              variant="secondary"
              onClick={() => {
                const blob = new Blob([reportText], { type: "text/markdown" });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.setAttribute("download", "finlytics_report.md");
                link.click();
                toast.success("Report exported");
              }}
            >
              Download Markdown
            </Button>
          </div>
        </Card>
      ) : null}

      <Modal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        title="Executive Report Preview"
        description="Ready for risk committee circulation"
      >
        <pre className="max-h-[360px] overflow-auto rounded-xl border border-white/10 bg-slate-900/50 p-3 text-xs text-slate-300">
          {reportText}
        </pre>
      </Modal>
    </div>
  );
}
