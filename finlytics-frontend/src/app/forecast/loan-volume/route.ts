import { NextRequest, NextResponse } from "next/server";

import { callPythonBridge } from "@/lib/python-bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type LoanForecastScenario = "baseline" | "optimistic" | "conservative" | "stress";

type ForecastRequestPayload = {
  horizonMonths: number;
  scenario: LoanForecastScenario;
  growthAdjustmentPct: number;
  interestRateShockBps: number;
  loanCountChangePct: number;
  avgLoanAmountChangePct: number;
};

const SCENARIO_OPTIONS = new Set<LoanForecastScenario>([
  "baseline",
  "optimistic",
  "conservative",
  "stress",
]);

function getMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function parseNumber(value: unknown, fieldLabel: string): number {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error(`${fieldLabel} must be a valid number`);
  }
  return numeric;
}

function assertRange(fieldLabel: string, value: number, min: number, max: number) {
  if (value < min || value > max) {
    throw new Error(`${fieldLabel} must be between ${min} and ${max}`);
  }
}

function normalizeForecastPayload(payload: unknown): ForecastRequestPayload {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("Request body must be a JSON object");
  }

  const body = payload as Record<string, unknown>;
  const scenarioRaw = String(body.scenario ?? "baseline").trim().toLowerCase();
  if (!SCENARIO_OPTIONS.has(scenarioRaw as LoanForecastScenario)) {
    throw new Error("scenario must be one of: baseline, optimistic, conservative, stress");
  }

  const horizonMonths = Math.round(parseNumber(body.horizonMonths ?? body.horizon ?? 3, "horizonMonths"));
  assertRange("horizonMonths", horizonMonths, 1, 12);

  const growthAdjustmentPct = parseNumber(body.growthAdjustmentPct ?? 0, "growthAdjustmentPct");
  const interestRateShockBps = parseNumber(body.interestRateShockBps ?? 0, "interestRateShockBps");
  const loanCountChangePct = parseNumber(body.loanCountChangePct ?? 0, "loanCountChangePct");
  const avgLoanAmountChangePct = parseNumber(body.avgLoanAmountChangePct ?? 0, "avgLoanAmountChangePct");

  assertRange("growthAdjustmentPct", growthAdjustmentPct, -40, 60);
  assertRange("interestRateShockBps", interestRateShockBps, -400, 400);
  assertRange("loanCountChangePct", loanCountChangePct, -50, 120);
  assertRange("avgLoanAmountChangePct", avgLoanAmountChangePct, -50, 80);

  return {
    horizonMonths,
    scenario: scenarioRaw as LoanForecastScenario,
    growthAdjustmentPct,
    interestRateShockBps,
    loanCountChangePct,
    avgLoanAmountChangePct,
  };
}

function statusCodeFromError(error: unknown): number {
  const message = getMessage(error, "");
  const validationPattern =
    /must be|missing required|unsupported scenario|request body|invalid|valueerror|json|syntaxerror|unexpected token|unexpected end/i;
  return validationPattern.test(message) ? 400 : 500;
}

export async function GET() {
  try {
    const data = await callPythonBridge("forecast_loan_volume", {});
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: getMessage(error, "forecast request failed") },
      { status: statusCodeFromError(error) }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const rawPayload = (await request.json()) as unknown;
    const payload = normalizeForecastPayload(rawPayload);
    const data = await callPythonBridge("forecast_loan_volume", payload);
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: getMessage(error, "loan volume prediction failed") },
      { status: statusCodeFromError(error) }
    );
  }
}
