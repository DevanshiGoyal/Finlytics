import { NextRequest, NextResponse } from "next/server";

import { callPythonBridge } from "@/lib/python-bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type ScenarioType = "baseline" | "optimistic" | "pessimistic";

type CreditDemandForecastPayload = {
  horizon: number;
  confidence: number;
  scenarioType: ScenarioType;
  baseVolume: number | null;
};

const SCENARIO_TYPES = new Set<ScenarioType>(["baseline", "optimistic", "pessimistic"]);

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

function normalizePayload(payload: unknown): CreditDemandForecastPayload {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("Request body must be a JSON object");
  }

  const body = payload as Record<string, unknown>;
  const scenarioRaw = String(body.scenarioType ?? "baseline").trim().toLowerCase();
  if (!SCENARIO_TYPES.has(scenarioRaw as ScenarioType)) {
    throw new Error("scenarioType must be one of: baseline, optimistic, pessimistic");
  }

  const horizon = Math.round(parseNumber(body.horizon ?? 3, "horizon"));
  assertRange("horizon", horizon, 1, 12);

  const confidence = parseNumber(body.confidence ?? 0.95, "confidence");
  assertRange("confidence", confidence, 0.5, 0.99);

  let baseVolume: number | null = null;
  if (body.baseVolume !== undefined && body.baseVolume !== null && body.baseVolume !== "") {
    baseVolume = parseNumber(body.baseVolume, "baseVolume");
    assertRange("baseVolume", baseVolume, 1, 1000000);
  }

  return {
    horizon,
    confidence,
    scenarioType: scenarioRaw as ScenarioType,
    baseVolume,
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
    const data = await callPythonBridge("credit_demand_by_grade", {});
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: getMessage(error, "credit demand by grade failed") },
      { status: statusCodeFromError(error) }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const rawPayload = (await request.json()) as unknown;
    const payload = normalizePayload(rawPayload);
    const data = await callPythonBridge("credit_demand_by_grade_forecast", payload);
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: getMessage(error, "credit demand by grade forecast failed") },
      { status: statusCodeFromError(error) }
    );
  }
}
