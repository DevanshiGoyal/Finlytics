"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";

import { AnomalyTrendChart } from "@/components/charts/anomaly-trend-chart";
import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { FileUpload } from "@/components/ui/file-upload";
import { Input } from "@/components/ui/input";
import { DataTable } from "@/components/ui/table";
import { detectAnomaly, getAnomalyTimeseries, scoreAnomalyBatch } from "@/services/api";

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

export default function DepositAnomalyDetectionPage() {
  const [form, setForm] = useState({ amount: "7500", hour: "10", dayOfWeek: "2", frequency: "3" });
  const [score, setScore] = useState<number | null>(null);
  const [label, setLabel] = useState<string>("");
  const [trend, setTrend] = useState<Array<{ time: string; score: number }>>([]);
  const [batchRows, setBatchRows] = useState<Array<{ txId: string; amount: number; score: number; label: string }>>([]);

  useEffect(() => {
    const load = async () => {
      try {
        const result = await getAnomalyTimeseries();
        setTrend(result.trend);
        setBatchRows(result.batch);
      } catch {
        toast.error("Unable to load anomaly timeline from repository resources");
      }
    };
    void load();
  }, []);

  const runLiveScan = async () => {
    const response = await detectAnomaly({
      amount: Number(form.amount),
      hour: Number(form.hour),
      dayOfWeek: Number(form.dayOfWeek),
      frequency: Number(form.frequency)
    });
    setScore(response.score);
    setLabel(response.label);
    toast.success("Live transaction scored");
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Deposit Anomaly Detection"
        subtitle="Scan transactions in real time, classify suspicious behavior, and process anomaly batches at scale."
        tag="Fraud Monitoring"
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <h3 className="text-lg font-semibold text-white">Live Transaction Input</h3>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <div>
              <label className="mb-1 block text-xs text-slate-400">Amount</label>
              <Input value={form.amount} type="number" onChange={(event) => setForm((prev) => ({ ...prev, amount: event.target.value }))} />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Hour</label>
              <Input value={form.hour} type="number" onChange={(event) => setForm((prev) => ({ ...prev, hour: event.target.value }))} />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Day of Week</label>
              <Input
                value={form.dayOfWeek}
                type="number"
                onChange={(event) => setForm((prev) => ({ ...prev, dayOfWeek: event.target.value }))}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Frequency (24h)</label>
              <Input
                value={form.frequency}
                type="number"
                onChange={(event) => setForm((prev) => ({ ...prev, frequency: event.target.value }))}
              />
            </div>
          </div>
          <div className="mt-4">
            <Button onClick={() => void runLiveScan()}>Run Anomaly Scan</Button>
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-white">Anomaly Output</h3>
          {score !== null ? (
            <div className="mt-4 space-y-3">
              <p className="text-5xl font-bold text-amber-200">{(score * 100).toFixed(1)}%</p>
              <Badge variant={label === "Suspicious" ? "danger" : "success"}>{label}</Badge>
              <p className="text-xs text-slate-400">Threshold for suspicious label: 65%</p>
            </div>
          ) : (
            <p className="mt-3 text-sm text-slate-400">Run live scan to generate anomaly score.</p>
          )}
        </Card>
      </div>

      <Card>
        <h3 className="mb-3 text-lg font-semibold text-white">Anomaly Score Timeline</h3>
        <AnomalyTrendChart data={trend} />
      </Card>

      <Card className="space-y-4">
        <h3 className="text-lg font-semibold text-white">Batch Upload + Results</h3>
        <FileUpload
          title="Upload anomaly batch CSV"
          onFileSelect={async (file) => {
            try {
              const parsed = await parseCsv(file);
              if (!parsed.length) {
                toast.error("Uploaded file has no rows");
                return;
              }
              const result = await scoreAnomalyBatch({ rows: parsed });
              setBatchRows(result.batch);
              toast.success("Batch scored successfully");
            } catch (error) {
              const message = error instanceof Error ? error.message : "Batch scoring failed";
              toast.error(message);
            }
          }}
        />

        <DataTable
          columns={[
            { key: "txId", header: "Transaction ID" },
            { key: "amount", header: "Amount" },
            {
              key: "score",
              header: "Anomaly Score",
              render: (value) => `${(Number(value) * 100).toFixed(1)}%`
            },
            {
              key: "label",
              header: "Label",
              render: (value) => (
                <Badge variant={value === "Suspicious" ? "danger" : "success"}>{String(value)}</Badge>
              )
            }
          ]}
          data={batchRows}
        />
      </Card>
    </div>
  );
}
