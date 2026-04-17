"use client";

import { useMemo, useState } from "react";
import { toast } from "sonner";

import { FeatureImportanceChart } from "@/components/charts/feature-importance-chart";
import { ShapWaterfallChart } from "@/components/charts/shap-waterfall-chart";
import { PageHeader } from "@/components/layout/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { predictDefault } from "@/services/api";
import type { ShapExplanationResponse } from "@/services/api";

interface PredictionResult {
  probability: number;
  label: string;
  explainability: Array<{ feature: string; importance: number }>;
  shapExplanation?: ShapExplanationResponse;
}

function labelVariant(label: string) {
  if (label === "High") {
    return "danger" as const;
  }
  if (label === "Medium") {
    return "warning" as const;
  }
  return "success" as const;
}

export default function LoanDefaultRiskPage() {
  const [form, setForm] = useState({
    loanAmount: "10000",
    inqLast6Mths: "1",
    grade: "A",
    interestRate: "13.0",
    openAcc: "10",
    homeOwnership: "RENT",
    installment: "300",
    pubRec: "0",
    purpose: "debt_consolidation",
    annualIncome: "65000",
    revolUtil: "50",
    issueYear: "2015",
    dti: "15",
    totalAcc: "25",
    issueMonth: "1",
    delinq2Yrs: "0",
    empLength: "5",
  });
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const valid = useMemo(() => {
    const numericKeys = [
      "loanAmount",
      "inqLast6Mths",
      "interestRate",
      "openAcc",
      "installment",
      "pubRec",
      "annualIncome",
      "revolUtil",
      "issueYear",
      "dti",
      "totalAcc",
      "issueMonth",
      "delinq2Yrs",
      "empLength",
    ] as const;

    return numericKeys.every((key) => {
      const numeric = Number(form[key]);
      return Number.isFinite(numeric) && numeric >= 0;
    });
  }, [form]);

  const onSubmit = async () => {
    if (!valid) {
      toast.error("Please enter valid borrower values before prediction.");
      return;
    }

    setSubmitting(true);
    try {
      const response = await predictDefault({
        loanAmount: Number(form.loanAmount),
        inqLast6Mths: Number(form.inqLast6Mths),
        grade: form.grade,
        installment: Number(form.installment),
        openAcc: Number(form.openAcc),
        pubRec: Number(form.pubRec),
        purpose: form.purpose,
        homeOwnership: form.homeOwnership,
        annualIncome: Number(form.annualIncome),
        dti: Number(form.dti),
        interestRate: Number(form.interestRate),
        revolUtil: Number(form.revolUtil),
        totalAcc: Number(form.totalAcc),
        delinq2Yrs: Number(form.delinq2Yrs),
        empLength: Number(form.empLength),
        issueYear: Number(form.issueYear),
        issueMonth: Number(form.issueMonth),
      });
      setResult(response);
      toast.success("Default probability generated");
    } catch {
      toast.error("Prediction failed. Please retry.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Loan Default Risk"
        subtitle="Score borrower risk in real time with explainable AI and risk-banded decision support."
        tag="Risk Scoring"
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <h3 className="text-lg font-semibold text-white">Borrower Input</h3>
          <p className="mt-1 text-sm text-slate-400">
            Enter borrower profile values to estimate default probability.
          </p>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Loan Amount ($)
              </label>
              <Input
                value={form.loanAmount}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    loanAmount: event.target.value,
                  }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Credit Inquiries (6mo)
              </label>
              <Input
                value={form.inqLast6Mths}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    inqLast6Mths: event.target.value,
                  }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Loan Grade
              </label>
              <Select
                value={form.grade}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, grade: event.target.value }))
                }
                options={[
                  { label: "A", value: "A" },
                  { label: "B", value: "B" },
                  { label: "C", value: "C" },
                  { label: "D", value: "D" },
                  { label: "E", value: "E" },
                  { label: "F", value: "F" },
                  { label: "G", value: "G" },
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Interest Rate (%)
              </label>
              <Input
                value={form.interestRate}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    interestRate: event.target.value,
                  }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Open Credit Lines
              </label>
              <Input
                value={form.openAcc}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, openAcc: event.target.value }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Home Ownership
              </label>
              <Select
                value={form.homeOwnership}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    homeOwnership: event.target.value,
                  }))
                }
                options={[
                  { label: "RENT", value: "RENT" },
                  { label: "OWN", value: "OWN" },
                  { label: "MORTGAGE", value: "MORTGAGE" },
                  { label: "OTHER", value: "OTHER" },
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Monthly Installment ($)
              </label>
              <Input
                value={form.installment}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    installment: event.target.value,
                  }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Public Records
              </label>
              <Input
                value={form.pubRec}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, pubRec: event.target.value }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Loan Purpose
              </label>
              <Select
                value={form.purpose}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, purpose: event.target.value }))
                }
                options={[
                  { label: "debt_consolidation", value: "debt_consolidation" },
                  { label: "credit_card", value: "credit_card" },
                  { label: "home_improvement", value: "home_improvement" },
                  { label: "major_purchase", value: "major_purchase" },
                  { label: "small_business", value: "small_business" },
                  { label: "other", value: "other" },
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Annual Income ($)
              </label>
              <Input
                value={form.annualIncome}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    annualIncome: event.target.value,
                  }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Revolving Utilization (%)
              </label>
              <Input
                value={form.revolUtil}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    revolUtil: event.target.value,
                  }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Issue Year
              </label>
              <Input
                value={form.issueYear}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    issueYear: event.target.value,
                  }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Debt-to-Income Ratio
              </label>
              <Input
                value={form.dti}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, dti: event.target.value }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Total Credit Lines
              </label>
              <Input
                value={form.totalAcc}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, totalAcc: event.target.value }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Issue Month
              </label>
              <Select
                value={form.issueMonth}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    issueMonth: event.target.value,
                  }))
                }
                options={Array.from({ length: 12 }).map((_, idx) => ({
                  label: String(idx + 1),
                  value: String(idx + 1),
                }))}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Delinquencies (2yrs)
              </label>
              <Input
                value={form.delinq2Yrs}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    delinq2Yrs: event.target.value,
                  }))
                }
                type="number"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Employment Length (years)
              </label>
              <Input
                value={form.empLength}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    empLength: event.target.value,
                  }))
                }
                type="number"
              />
            </div>
          </div>

          <div className="mt-4">
            <Button onClick={onSubmit} disabled={submitting}>
              {submitting ? "Predicting..." : "Predict Default Risk"}
            </Button>
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-white">
            Prediction Output
          </h3>
          {!result && !submitting ? (
            <p className="mt-3 text-sm text-slate-400">
              Run a prediction to see probability, risk label, and confidence
              meter.
            </p>
          ) : null}
          {submitting ? <Skeleton className="mt-4 h-40" /> : null}
          {result ? (
            <div className="mt-4 space-y-4">
              <p className="text-5xl font-bold text-cyan-200">
                {(result.probability * 100).toFixed(1)}%
              </p>
              <Badge variant={labelVariant(result.label)}>
                {result.label} Risk
              </Badge>
              <div>
                <p className="mb-2 text-xs text-slate-400">Risk Gauge</p>
                <Progress value={result.probability * 100} />
              </div>
              <p className="text-xs text-slate-400">
                Decision threshold: 35% (medium), 65% (high)
              </p>
            </div>
          ) : null}
        </Card>
      </div>

      <Card>
        <h3 className="text-lg font-semibold text-white">
          Feature Importance Explanation
        </h3>
        <p className="mt-1 text-sm text-slate-400">
          Relative contribution of key borrower factors to the current risk
          score.
        </p>
        <div className="mt-4">
          {submitting ? <Skeleton className="h-[300px]" /> : null}
          {result ? (
            <FeatureImportanceChart data={result.explainability} />
          ) : null}
          {!result && !submitting ? (
            <p className="rounded-xl border border-dashed border-white/20 bg-slate-900/30 p-6 text-sm text-slate-400">
              No explainability data yet. Run a prediction to populate this
              panel.
            </p>
          ) : null}
        </div>
      </Card>

      <Card>
        <h3 className="text-lg font-semibold text-white">
          SHAP Waterfall Explanation
        </h3>
        <p className="mt-1 text-sm text-slate-400">
          Feature-level additive impact for this prediction. Positive values
          push risk higher; negative values reduce risk.
        </p>
        <div className="mt-4">
          {submitting ? <Skeleton className="h-[320px]" /> : null}
          {result?.shapExplanation?.available &&
          result.shapExplanation.baseValue != null ? (
            <ShapWaterfallChart
              points={result.shapExplanation.points}
              baseValue={result.shapExplanation.baseValue}
              modelOutput={result.shapExplanation.modelOutput}
            />
          ) : null}
          {result && !result.shapExplanation?.available ? (
            <p className="rounded-xl border border-amber-300/20 bg-amber-500/10 p-4 text-sm text-amber-100">
              {result.shapExplanation?.message ||
                "SHAP explanation is unavailable for this environment."}
            </p>
          ) : null}
          {!result && !submitting ? (
            <p className="rounded-xl border border-dashed border-white/20 bg-slate-900/30 p-6 text-sm text-slate-400">
              No SHAP explanation yet. Run a prediction to populate this panel.
            </p>
          ) : null}
        </div>
      </Card>
    </div>
  );
}
