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
import { predictChurn } from "@/services/api";
import type { ShapExplanationResponse } from "@/services/api";

interface ChurnResult {
  probability: number;
  label: string;
  suggestions: string[];
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

export default function BorrowerChurnPage() {
  const [form, setForm] = useState({
    loanAmount: "10000",
    empLength: "5",
    openAcc: "10",
    interestRate: "13",
    homeOwnership: "RENT",
    totalAcc: "25",
    installment: "300",
    purpose: "debt_consolidation",
    revolUtil: "50",
    annualIncome: "65000",
    grade: "A",
    delinq2Yrs: "0",
    dti: "15",
    issueYear: "2015",
    issueMonth: "1",
  });
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<ChurnResult | null>(null);

  const valid = useMemo(() => {
    const numericKeys = [
      "loanAmount",
      "empLength",
      "openAcc",
      "interestRate",
      "totalAcc",
      "installment",
      "revolUtil",
      "annualIncome",
      "delinq2Yrs",
      "dti",
      "issueYear",
      "issueMonth",
    ] as const;

    return numericKeys.every((key) => {
      const numeric = Number(form[key]);
      return Number.isFinite(numeric) && numeric >= 0;
    });
  }, [form]);

  const runPrediction = async () => {
    if (!valid) {
      toast.error("Please complete all fields with valid values.");
      return;
    }

    setSubmitting(true);
    try {
      const response = await predictChurn({
        loanAmount: Number(form.loanAmount),
        empLength: Number(form.empLength),
        openAcc: Number(form.openAcc),
        interestRate: Number(form.interestRate),
        homeOwnership: form.homeOwnership,
        totalAcc: Number(form.totalAcc),
        installment: Number(form.installment),
        purpose: form.purpose,
        revolUtil: Number(form.revolUtil),
        annualIncome: Number(form.annualIncome),
        grade: form.grade,
        delinq2Yrs: Number(form.delinq2Yrs),
        dti: Number(form.dti),
        issueYear: Number(form.issueYear),
        issueMonth: Number(form.issueMonth),
      });
      setResult(response);
      toast.success("Churn probability generated");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Churn prediction failed";
      toast.error(message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="Borrower Churn"
        subtitle="Estimate churn probability and activate retention suggestions before revenue leakage increases."
        tag="Retention Intelligence"
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <h3 className="text-lg font-semibold text-white">
            Customer Profile Inputs
          </h3>
          <p className="mt-1 text-sm text-slate-400">
            Provide the full borrower profile used by your module-2 churn model.
          </p>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Loan Amount ($)
              </label>
              <Input
                type="number"
                value={form.loanAmount}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    loanAmount: event.target.value,
                  }))
                }
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Employment Length (years)
              </label>
              <Input
                type="number"
                value={form.empLength}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    empLength: event.target.value,
                  }))
                }
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Open Credit Lines
              </label>
              <Input
                type="number"
                value={form.openAcc}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, openAcc: event.target.value }))
                }
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Interest Rate (%)
              </label>
              <Input
                type="number"
                value={form.interestRate}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    interestRate: event.target.value,
                  }))
                }
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
                Total Credit Lines
              </label>
              <Input
                type="number"
                value={form.totalAcc}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, totalAcc: event.target.value }))
                }
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Monthly Installment ($)
              </label>
              <Input
                type="number"
                value={form.installment}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    installment: event.target.value,
                  }))
                }
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
                  { label: "other", value: "other" },
                  { label: "major_purchase", value: "major_purchase" },
                  { label: "medical", value: "medical" },
                  { label: "small_business", value: "small_business" },
                ]}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Revolving Utilization (%)
              </label>
              <Input
                type="number"
                value={form.revolUtil}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    revolUtil: event.target.value,
                  }))
                }
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Annual Income ($)
              </label>
              <Input
                type="number"
                value={form.annualIncome}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    annualIncome: event.target.value,
                  }))
                }
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
                Delinquencies (2yrs)
              </label>
              <Input
                type="number"
                value={form.delinq2Yrs}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    delinq2Yrs: event.target.value,
                  }))
                }
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Debt-to-Income Ratio
              </label>
              <Input
                type="number"
                value={form.dti}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, dti: event.target.value }))
                }
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-slate-400">
                Issue Year
              </label>
              <Input
                type="number"
                value={form.issueYear}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    issueYear: event.target.value,
                  }))
                }
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
          </div>

          <div className="mt-4">
            <Button onClick={runPrediction} disabled={submitting}>
              {submitting ? "Scoring..." : "Predict Churn"}
            </Button>
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-white">Churn Output</h3>
          {submitting ? <Skeleton className="mt-4 h-40" /> : null}
          {result ? (
            <div className="mt-4 space-y-4">
              <p className="text-5xl font-bold text-indigo-200">
                {(result.probability * 100).toFixed(1)}%
              </p>
              <Badge variant={labelVariant(result.label)}>
                {result.label} Churn Risk
              </Badge>
              <Progress value={result.probability * 100} />
            </div>
          ) : null}
          {!result && !submitting ? (
            <p className="mt-3 text-sm text-slate-400">
              Run prediction to get churn probability and retention guidance.
            </p>
          ) : null}
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <h3 className="text-lg font-semibold text-white">
            Feature Importance
          </h3>
          <p className="mt-1 text-sm text-slate-400">
            Top drivers influencing current churn estimate.
          </p>
          <div className="mt-4">
            {result ? (
              <FeatureImportanceChart data={result.explainability} />
            ) : (
              <Skeleton className="h-[300px]" />
            )}
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-white">
            Retention Suggestions
          </h3>
          <div className="mt-3 space-y-2">
            {result?.suggestions?.length ? (
              result.suggestions.map((suggestion) => (
                <div
                  key={suggestion}
                  className="rounded-xl border border-white/10 bg-slate-950/35 p-3 text-sm text-slate-300"
                >
                  {suggestion}
                </div>
              ))
            ) : (
              <p className="text-sm text-slate-400">
                Suggestions will appear after prediction.
              </p>
            )}
          </div>
        </Card>
      </div>

      <Card>
        <h3 className="text-lg font-semibold text-white">
          SHAP Waterfall Explanation
        </h3>
        <p className="mt-1 text-sm text-slate-400">
          Feature-level additive impact for this churn prediction.
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
