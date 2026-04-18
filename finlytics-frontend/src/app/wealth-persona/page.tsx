"use client";

import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { DataTable } from "@/components/ui/table";
import {
  getWealthPersonaInsights,
  type WealthPersonaInsightsResponse,
} from "@/services/api";

function formatNum(value: number) {
  return Number(value || 0).toLocaleString();
}

function formatCurrency(value: number) {
  return `₹${Number(value || 0).toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  })}`;
}

function formatPercent(value: number) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function clampInt(value: number, min: number, max: number, fallback: number) {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, Math.round(value)));
}

export default function WealthPersonaPage() {
  const [loading, setLoading] = useState(true);
  const [k, setK] = useState(4);
  const [topN, setTopN] = useState(15);
  const [data, setData] = useState<WealthPersonaInsightsResponse | null>(null);

  const load = async (nextK: number, nextTopN: number) => {
    const safeK = clampInt(nextK, 2, 10, 4);
    const safeTopN = clampInt(nextTopN, 5, 100, 15);

    if (safeK !== nextK) {
      setK(safeK);
    }
    if (safeTopN !== nextTopN) {
      setTopN(safeTopN);
    }

    setLoading(true);
    try {
      const response = await getWealthPersonaInsights({
        k: safeK,
        topN: safeTopN,
      });
      setData(response);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unable to load wealth persona insights";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load(k, topN);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const summary = data?.summary;
  const personaCounts = useMemo(() => data?.personaCounts ?? [], [data]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Wealth Persona Intelligence"
        subtitle="RFM wealth segmentation, K-Means personas, behavioral anomalies, and regional investment zones powered by 1M+ transactions."
        tag="Segmentation AI"
      />

      <Card>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="mb-1 block text-xs text-slate-400">Clusters (K)</label>
            <input
              className="h-10 w-28 rounded-xl border border-white/20 bg-slate-950/70 px-3 text-sm text-slate-100"
              type="number"
              min={2}
              max={10}
              value={k}
              onChange={(event) =>
                setK(clampInt(Number(event.target.value || 4), 2, 10, 4))
              }
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-slate-400">Top rows</label>
            <input
              className="h-10 w-28 rounded-xl border border-white/20 bg-slate-950/70 px-3 text-sm text-slate-100"
              type="number"
              min={5}
              max={100}
              value={topN}
              onChange={(event) =>
                setTopN(clampInt(Number(event.target.value || 15), 5, 100, 15))
              }
            />
          </div>
          <Button onClick={() => void load(k, topN)} disabled={loading}>
            {loading ? "Refreshing..." : "Refresh Insights"}
          </Button>
          {summary ? <Badge variant="info">Dataset: {summary.datasetPath}</Badge> : null}
        </div>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card>
          <p className="text-xs uppercase text-slate-400">Transactions</p>
          <p className="mt-2 text-2xl font-semibold text-white">
            {summary ? formatNum(summary.transaction_rows) : "-"}
          </p>
        </Card>
        <Card>
          <p className="text-xs uppercase text-slate-400">Customers</p>
          <p className="mt-2 text-2xl font-semibold text-white">
            {summary ? formatNum(summary.customers) : "-"}
          </p>
        </Card>
        <Card>
          <p className="text-xs uppercase text-slate-400">Vital Few</p>
          <p className="mt-2 text-2xl font-semibold text-white">
            {summary ? formatNum(summary.vital_few_customers) : "-"}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            {summary ? formatPercent(summary.vital_few_share) : "-"}
          </p>
        </Card>
        <Card>
          <p className="text-xs uppercase text-slate-400">Behavioral Anomalies</p>
          <p className="mt-2 text-2xl font-semibold text-amber-200">
            {summary ? formatNum(summary.behavioral_anomalies) : "-"}
          </p>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <h3 className="mb-3 text-lg font-semibold text-white">Persona Breakdown</h3>
          <DataTable
            columns={[
              { key: "persona", header: "Persona" },
              {
                key: "customers",
                header: "Customers",
                render: (value) => formatNum(Number(value)),
              },
              {
                key: "medianAge",
                header: "Median Age",
                render: (value) => Number(value).toFixed(1),
              },
              {
                key: "avgFrequency",
                header: "Avg Frequency",
                render: (value) => Number(value).toFixed(2),
              },
              {
                key: "avgMonetary",
                header: "Avg Monetary",
                render: (value) => formatCurrency(Number(value)),
              },
              {
                key: "vitalFewShare",
                header: "Vital Few Share",
                render: (value) => formatPercent(Number(value)),
              },
            ]}
            data={data?.personaBreakdown ?? []}
            emptyText={loading ? "Loading persona breakdown..." : "No persona rows available"}
          />
        </Card>

        <Card>
          <h3 className="mb-3 text-lg font-semibold text-white">Persona Distribution</h3>
          <DataTable
            columns={[
              { key: "persona", header: "Persona" },
              {
                key: "customers",
                header: "Customers",
                render: (value) => formatNum(Number(value)),
              },
            ]}
            data={personaCounts}
            emptyText={loading ? "Loading distribution..." : "No persona distribution available"}
          />
        </Card>
      </div>

      <Card>
        <h3 className="mb-3 text-lg font-semibold text-white">Top Vital Few Customers</h3>
        <DataTable
          columns={[
            { key: "CustomerID", header: "Customer ID" },
            { key: "persona", header: "Persona" },
            { key: "location", header: "Location" },
            { key: "rfmScore", header: "RFM Score" },
            {
              key: "wealthValue",
              header: "Wealth Value",
              render: (value) => formatCurrency(Number(value)),
            },
            {
              key: "monetary",
              header: "Monetary",
              render: (value) => formatCurrency(Number(value)),
            },
          ]}
          data={data?.topVitalFew ?? []}
          emptyText={loading ? "Loading vital few..." : "No vital few customers found"}
        />
      </Card>

      <Card>
        <h3 className="mb-3 text-lg font-semibold text-white">High-Potential Regions</h3>
        <DataTable
          columns={[
            { key: "location", header: "Location" },
            {
              key: "customers",
              header: "Customers",
              render: (value) => formatNum(Number(value)),
            },
            {
              key: "totalAccountBalance",
              header: "Total Balance",
              render: (value) => formatCurrency(Number(value)),
            },
            {
              key: "totalTransactionValue",
              header: "Transaction Value",
              render: (value) => formatCurrency(Number(value)),
            },
            {
              key: "potentialScore",
              header: "Potential Score",
              render: (value) => Number(value).toFixed(3),
            },
            {
              key: "highPotentialZone",
              header: "Zone",
              render: (value) => (
                <Badge variant={value ? "success" : "warning"}>
                  {value ? "High Potential" : "Monitor"}
                </Badge>
              ),
            },
          ]}
          data={data?.topRegions ?? []}
          emptyText={loading ? "Loading regional intelligence..." : "No regional insights available"}
        />
      </Card>

      <Card>
        <h3 className="mb-3 text-lg font-semibold text-white">Top Behavioral Anomalies</h3>
        <DataTable
          columns={[
            { key: "customerId", header: "Customer ID" },
            { key: "transactionDate", header: "Transaction Date" },
            { key: "location", header: "Location" },
            {
              key: "transactionAmount",
              header: "Amount",
              render: (value) => formatCurrency(Number(value)),
            },
            {
              key: "anomalyScore",
              header: "Score",
              render: (value) => Number(value).toFixed(3),
            },
            { key: "reason", header: "Reason" },
          ]}
          data={data?.topBehavioralAnomalies ?? []}
          emptyText={loading ? "Loading anomalies..." : "No anomalies detected in top slice"}
        />
      </Card>
    </div>
  );
}
