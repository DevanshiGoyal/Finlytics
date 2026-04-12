"use client";

import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { FeatureImportanceChart } from "@/components/charts/feature-importance-chart";
import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { DataTable } from "@/components/ui/table";
import { getDepositLeaderboard, predictDeposit } from "@/services/api";

function toPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function BankDepositAIPage() {
  const [leaderboard, setLeaderboard] = useState<Array<{
    model: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  }>>([]);
  const [featureImportance, setFeatureImportance] = useState<Array<{ feature: string; importance: number }>>([]);
  const [model, setModel] = useState("");
  const [form, setForm] = useState({
    age: "36",
    job: "admin.",
    marital: "divorced",
    education: "primary",
    default: "no",
    balance: "1200",
    housing: "no",
    loan: "no",
    contact: "cellular",
    day: "15",
    month: "apr",
    duration: "320",
    campaign: "2",
    pdays: "40",
    previous: "1",
    poutcome: "failure"
  });
  const [probability, setProbability] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    let active = true;
    getDepositLeaderboard()
      .then((data) => {
        if (!active) {
          return;
        }
        if (Array.isArray(data.leaderboard) && data.leaderboard.length > 0) {
          setLeaderboard(data.leaderboard);
          const best = data.bestModel && data.leaderboard.some((item) => item.model === data.bestModel)
            ? data.bestModel
            : data.leaderboard[0].model;
          setModel(best);
        }
        if (Array.isArray(data.featureImportance) && data.featureImportance.length > 0) {
          setFeatureImportance(data.featureImportance);
        }
      })
      .catch(() => {
        toast.error("Unable to load leaderboard from repository models/data");
      });

    return () => {
      active = false;
    };
  }, []);

  const selectedModel = useMemo(() => leaderboard.find((item) => item.model === model), [leaderboard, model]);

  const runPrediction = async () => {
    try {
      if (!model) {
        toast.error("No model available. Ensure leaderboard endpoint is healthy.");
        return;
      }
      setIsPredicting(true);
      const result = await predictDeposit({
        model,
        age: Number(form.age),
        job: form.job,
        marital: form.marital,
        education: form.education,
        default: form.default,
        balance: Number(form.balance),
        housing: form.housing,
        loan: form.loan,
        contact: form.contact,
        day: Number(form.day),
        month: form.month,
        duration: Number(form.duration),
        campaign: Number(form.campaign),
        pdays: Number(form.pdays),
        previous: Number(form.previous),
        poutcome: form.poutcome
      });
      setProbability(result.probability);
      toast.success("Deposit subscription score generated");
    } catch {
      toast.error("Unable to generate score right now");
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Bank Deposit AI"
        subtitle="Evaluate campaign conversion potential with model leaderboard insights and customer-level subscription scoring."
        tag="Campaign Intelligence"
      />

      <Card className="space-y-4">
        <h3 className="text-lg font-semibold text-white">Model Leaderboard</h3>
        <DataTable
          columns={[
            { key: "model", header: "Model" },
            { key: "accuracy", header: "Accuracy", render: (value) => toPercent(Number(value)) },
            { key: "precision", header: "Precision", render: (value) => toPercent(Number(value)) },
            { key: "recall", header: "Recall", render: (value) => toPercent(Number(value)) },
            { key: "f1", header: "F1", render: (value) => toPercent(Number(value)) }
          ]}
          data={leaderboard}
        />
      </Card>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <h3 className="text-lg font-semibold text-white">Prediction Console</h3>
          <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            <div>
              <label className="mb-1 block text-xs text-slate-400">Model Selector</label>
              <Select
                value={model}
                onChange={(event) => setModel(event.target.value)}
                options={leaderboard.map((item) => ({ label: item.model, value: item.model }))}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Age</label>
              <Input type="number" value={form.age} onChange={(event) => setForm((prev) => ({ ...prev, age: event.target.value }))} />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Days since previous contact</label>
              <Input type="number" value={form.pdays} onChange={(event) => setForm((prev) => ({ ...prev, pdays: event.target.value }))} />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Credit default</label>
              <Select
                value={form.default}
                onChange={(event) => setForm((prev) => ({ ...prev, default: event.target.value }))}
                options={[
                  { label: "no", value: "no" },
                  { label: "yes", value: "yes" }
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Balance</label>
              <Input type="number" value={form.balance} onChange={(event) => setForm((prev) => ({ ...prev, balance: event.target.value }))} />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Previous contacts</label>
              <Input type="number" value={form.previous} onChange={(event) => setForm((prev) => ({ ...prev, previous: event.target.value }))} />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Housing loan</label>
              <Select
                value={form.housing}
                onChange={(event) => setForm((prev) => ({ ...prev, housing: event.target.value }))}
                options={[
                  { label: "no", value: "no" },
                  { label: "yes", value: "yes" }
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Call Duration (sec)</label>
              <Input type="number" value={form.duration} onChange={(event) => setForm((prev) => ({ ...prev, duration: event.target.value }))} />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Job</label>
              <Select
                value={form.job}
                onChange={(event) => setForm((prev) => ({ ...prev, job: event.target.value }))}
                options={[
                  { label: "admin.", value: "admin." },
                  { label: "blue-collar", value: "blue-collar" },
                  { label: "entrepreneur", value: "entrepreneur" },
                  { label: "housemaid", value: "housemaid" },
                  { label: "management", value: "management" },
                  { label: "retired", value: "retired" },
                  { label: "self-employed", value: "self-employed" },
                  { label: "services", value: "services" },
                  { label: "student", value: "student" },
                  { label: "technician", value: "technician" },
                  { label: "unemployed", value: "unemployed" },
                  { label: "unknown", value: "unknown" }
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Personal loan</label>
              <Select
                value={form.loan}
                onChange={(event) => setForm((prev) => ({ ...prev, loan: event.target.value }))}
                options={[
                  { label: "no", value: "no" },
                  { label: "yes", value: "yes" }
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Contact type</label>
              <Select
                value={form.contact}
                onChange={(event) => setForm((prev) => ({ ...prev, contact: event.target.value }))}
                options={[
                  { label: "cellular", value: "cellular" },
                  { label: "telephone", value: "telephone" },
                  { label: "unknown", value: "unknown" }
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Last contact day</label>
              <Input type="number" value={form.day} onChange={(event) => setForm((prev) => ({ ...prev, day: event.target.value }))} />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Marital</label>
              <Select
                value={form.marital}
                onChange={(event) => setForm((prev) => ({ ...prev, marital: event.target.value }))}
                options={[
                  { label: "single", value: "single" },
                  { label: "married", value: "married" },
                  { label: "divorced", value: "divorced" }
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Education</label>
              <Select
                value={form.education}
                onChange={(event) => setForm((prev) => ({ ...prev, education: event.target.value }))}
                options={[
                  { label: "primary", value: "primary" },
                  { label: "secondary", value: "secondary" },
                  { label: "tertiary", value: "tertiary" },
                  { label: "unknown", value: "unknown" }
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Last contact month</label>
              <Select
                value={form.month}
                onChange={(event) => setForm((prev) => ({ ...prev, month: event.target.value }))}
                options={[
                  { label: "jan", value: "jan" },
                  { label: "feb", value: "feb" },
                  { label: "mar", value: "mar" },
                  { label: "apr", value: "apr" },
                  { label: "may", value: "may" },
                  { label: "jun", value: "jun" },
                  { label: "jul", value: "jul" },
                  { label: "aug", value: "aug" },
                  { label: "sep", value: "sep" },
                  { label: "oct", value: "oct" },
                  { label: "nov", value: "nov" },
                  { label: "dec", value: "dec" }
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Campaign Contacts</label>
              <Input type="number" value={form.campaign} onChange={(event) => setForm((prev) => ({ ...prev, campaign: event.target.value }))} />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">Previous outcome</label>
              <Select
                value={form.poutcome}
                onChange={(event) => setForm((prev) => ({ ...prev, poutcome: event.target.value }))}
                options={[
                  { label: "failure", value: "failure" },
                  { label: "other", value: "other" },
                  { label: "success", value: "success" },
                  { label: "unknown", value: "unknown" }
                ]}
              />
            </div>
          </div>
          <div className="mt-4">
            <Button onClick={runPrediction} disabled={isPredicting || !model}>
              {isPredicting ? "Scoring..." : "Predict Subscription"}
            </Button>
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-white">Model Output</h3>
          {selectedModel ? <Badge variant="info">Using {selectedModel.model}</Badge> : null}
          {probability !== null ? (
            <div className="mt-4 space-y-3">
              <p className="text-5xl font-bold text-cyan-200">{(probability * 100).toFixed(1)}%</p>
              <Badge variant={probability > 0.6 ? "success" : "warning"}>
                {probability > 0.6 ? "Likely Subscribe" : "Low Intent"}
              </Badge>
            </div>
          ) : (
            <p className="mt-3 text-sm text-slate-400">Run prediction to view subscription probability.</p>
          )}
        </Card>
      </div>

      <Card>
        <h3 className="text-lg font-semibold text-white">Feature Importance</h3>
        <p className="mt-1 text-sm text-slate-400">Drivers behind campaign conversion performance.</p>
        {featureImportance.length ? (
          <FeatureImportanceChart data={featureImportance} />
        ) : (
          <p className="mt-3 text-sm text-slate-400">No live explainability data available.</p>
        )}
      </Card>
    </div>
  );
}
