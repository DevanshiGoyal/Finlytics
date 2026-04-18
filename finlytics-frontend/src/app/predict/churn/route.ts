import { NextRequest, NextResponse } from "next/server";

import { callPythonBridge } from "@/lib/python-bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type ChurnExplainabilityPoint = {
  feature: string;
  importance: number;
};

type ChurnBridgeResponse = {
  probability: number;
  label: string;
  suggestions: string[];
  explainability: ChurnExplainabilityPoint[];
  shapExplanation?: unknown;
};

type GeminiPart = {
  text?: string;
};

type GeminiCandidate = {
  content?: {
    parts?: GeminiPart[];
  };
};

type GeminiResponse = {
  candidates?: GeminiCandidate[];
};

type GeminiSuggestionResult = {
  suggestions: string[];
  source: "gemini" | "fallback";
  model?: string;
  error?: string;
};

const MODEL_FALLBACKS = [
  "gemini-flash-latest",
  "gemini-flash-lite-latest",
  "gemini-2.0-flash",
  "gemini-2.0-flash-lite",
  "gemini-2.0-flash-001",
  "gemini-2.0-flash-lite-001",
  "gemini-2.5-flash",
];

const SUGGESTION_LIMIT = 12;

function dedupeModels(models: string[]): string[] {
  const seen = new Set<string>();
  for (const model of models) {
    const normalized = model.trim();
    if (normalized) {
      seen.add(normalized);
    }
  }
  return [...seen];
}

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => String(item ?? "").trim())
    .filter((item) => item.length > 0);
}

function humanizeFeatureName(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function pickFirst(payload: Record<string, unknown>, keys: string[]): unknown {
  for (const key of keys) {
    const value = payload[key];
    if (value !== undefined && value !== null && value !== "") {
      return value;
    }
  }
  return null;
}

function normalizeValue(value: unknown): string | number {
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    return numeric;
  }
  return String(value).trim();
}

function buildCustomerSnapshot(payload: Record<string, unknown>): Record<string, string | number> {
  const fields: Array<{ label: string; keys: string[] }> = [
    { label: "loanAmount", keys: ["loanAmount", "loan_amnt"] },
    { label: "annualIncome", keys: ["annualIncome", "annual_inc"] },
    { label: "dti", keys: ["dti"] },
    { label: "interestRate", keys: ["interestRate", "int_rate"] },
    { label: "revolUtil", keys: ["revolUtil", "creditUtilization", "revol_util"] },
    { label: "installment", keys: ["installment"] },
    { label: "openAcc", keys: ["openAcc", "open_acc"] },
    { label: "totalAcc", keys: ["totalAcc", "total_acc"] },
    { label: "empLength", keys: ["empLength", "emp_length"] },
    { label: "delinq2Yrs", keys: ["delinq2Yrs", "delinq_2yrs"] },
    { label: "grade", keys: ["grade"] },
    { label: "homeOwnership", keys: ["homeOwnership", "home_ownership"] },
    { label: "purpose", keys: ["purpose"] },
    { label: "issueYear", keys: ["issueYear", "issue_year"] },
    { label: "issueMonth", keys: ["issueMonth", "issue_month"] },
  ];

  const snapshot: Record<string, string | number> = {};
  for (const field of fields) {
    const raw = pickFirst(payload, field.keys);
    if (raw !== null) {
      snapshot[field.label] = normalizeValue(raw);
    }
  }

  return snapshot;
}

function buildTopDrivers(explainability: ChurnExplainabilityPoint[]): string[] {
  return explainability.slice(0, 4).map((point) => {
    const pct = Number.isFinite(point.importance) ? (point.importance * 100).toFixed(1) : "0.0";
    return `${humanizeFeatureName(point.feature)} (${pct}%)`;
  });
}

function extractText(payload: GeminiResponse): string {
  const parts = payload.candidates?.[0]?.content?.parts ?? [];
  return parts
    .map((part) => part.text ?? "")
    .join("\n")
    .trim();
}

function parseJsonObject(rawText: string): Record<string, unknown> | null {
  const attempts: string[] = [rawText];

  const fenced = rawText.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    attempts.push(fenced[1]);
  }

  const firstBrace = rawText.indexOf("{");
  const lastBrace = rawText.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    attempts.push(rawText.slice(firstBrace, lastBrace + 1));
  }

  for (const candidate of attempts) {
    try {
      const parsed = JSON.parse(candidate) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      continue;
    }
  }

  return null;
}

function flattenStructuredSuggestions(structured: Record<string, unknown>): string[] {
  const sections: Array<{ title: string; items: string[] }> = [
    {
      title: "Personalized retention strategy",
      items: toStringArray(structured.personalizedRetentionStrategies),
    },
    {
      title: "Risk-specific action",
      items: toStringArray(structured.riskSpecificActions),
    },
    {
      title: "Behavioral insight",
      items: toStringArray(structured.behavioralInsights),
    },
    {
      title: "Preventive step",
      items: toStringArray(structured.preventiveSteps),
    },
  ];

  const output: string[] = [];
  for (const section of sections) {
    for (const item of section.items) {
      output.push(`${section.title}: ${item}`);
    }
  }

  return output.slice(0, SUGGESTION_LIMIT);
}

function buildSystemInstruction(): string {
  return [
    "You are Finlytics Copilot, an AI retention strategist for retail banking.",
    "Produce practical, compliant recommendations to reduce borrower churn.",
    "Focus on actions banks can execute immediately and explain likely churn behavior.",
    "Return concise recommendations with no markdown and no extra commentary.",
  ].join(" ");
}

function buildGeminiPrompt(payload: Record<string, unknown>, churnData: ChurnBridgeResponse): string {
  const snapshot = buildCustomerSnapshot(payload);
  const topDrivers = buildTopDrivers(churnData.explainability);

  return [
    "You are assisting a bank retention team.",
    `Current churn probability: ${churnData.probability.toFixed(4)} (${churnData.label} risk).`,
    `Customer data (JSON): ${JSON.stringify(snapshot)}`,
    `Top model drivers: ${JSON.stringify(topDrivers)}`,
    "Return ONLY valid JSON with this exact schema:",
    '{"personalizedRetentionStrategies":["..."],"riskSpecificActions":["Low risk: ...","Medium risk: ...","High risk: ..."],"behavioralInsights":["..."],"preventiveSteps":["..."]}',
    "Requirements:",
    "1) personalizedRetentionStrategies must be specific to this customer profile.",
    "2) riskSpecificActions must include distinct actions for low, medium, and high churn risk.",
    "3) behavioralInsights must explain why this borrower may churn, using the provided drivers.",
    "4) preventiveSteps must describe proactive bank-level controls to reduce future churn.",
    "5) Keep each item actionable and concise.",
  ].join("\n");
}

function buildFallbackSuggestions(payload: Record<string, unknown>, churnData: ChurnBridgeResponse): string[] {
  const topDrivers = buildTopDrivers(churnData.explainability);
  const topDriverText = topDrivers.length ? topDrivers.join(", ") : "repayment behavior and servicing engagement";

  const profile = buildCustomerSnapshot(payload);
  const grade = String(profile.grade ?? "unknown").toUpperCase();
  const purpose = String(profile.purpose ?? "general").replace(/_/g, " ");
  const probabilityPct = (churnData.probability * 100).toFixed(1);

  const baseSuggestions = toStringArray(churnData.suggestions).slice(0, 2);

  return [
    `Personalized retention strategy: Offer a tailored retention package for grade ${grade} borrowers with ${purpose} needs, including pricing relief or fee waivers tied to on-time payments.`,
    `Personalized retention strategy: ${baseSuggestions[0] ?? "Schedule proactive outreach from a relationship manager with a borrower-specific repayment check-in."}`,
    "Risk-specific action: Low risk -> maintain light-touch engagement and loyalty nudges to preserve healthy behavior.",
    "Risk-specific action: Medium risk -> trigger targeted outreach, flexible payment options, and weekly servicing follow-up.",
    "Risk-specific action: High risk -> escalate to high-touch retention workflow within 24 hours with restructuring support.",
    `Behavioral insight: Estimated churn risk is ${probabilityPct}% (${churnData.label}); strongest model signals are ${topDriverText}.`,
    "Behavioral insight: Signals suggest borrower friction is likely linked to repayment pressure and engagement quality, which can precede attrition.",
    "Preventive step: Deploy early-warning alerts for spikes in DTI, utilization, and delinquency indicators before churn risk crosses medium.",
    "Preventive step: Track retention campaign uplift by risk band and retrain churn thresholds monthly using outcome feedback.",
  ].slice(0, SUGGESTION_LIMIT);
}

async function generateGeminiSuggestions(
  payload: Record<string, unknown>,
  churnData: ChurnBridgeResponse
): Promise<GeminiSuggestionResult> {
  const fallbackSuggestions = buildFallbackSuggestions(payload, churnData);
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    return {
      suggestions: fallbackSuggestions,
      source: "fallback",
      error: "Gemini API key is not configured.",
    };
  }

  const configuredModel = process.env.GEMINI_MODEL || "";
  const modelsToTry = dedupeModels([configuredModel, ...MODEL_FALLBACKS]);
  const prompt = buildGeminiPrompt(payload, churnData);

  let lastError = "Gemini model resolution failed.";

  for (const model of modelsToTry) {
    const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    const geminiResponse = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      cache: "no-store",
      body: JSON.stringify({
        system_instruction: {
          parts: [{ text: buildSystemInstruction() }],
        },
        contents: [
          {
            role: "user",
            parts: [{ text: prompt }],
          },
        ],
        generationConfig: {
          temperature: 0.2,
          topP: 0.9,
          maxOutputTokens: 700,
          thinkingConfig: {
            thinkingBudget: 0,
          },
        },
      }),
    });

    if (geminiResponse.ok) {
      const responsePayload = (await geminiResponse.json()) as GeminiResponse;
      const text = extractText(responsePayload);
      const structured = parseJsonObject(text);
      if (structured) {
        const suggestions = flattenStructuredSuggestions(structured);
        if (suggestions.length >= 4) {
          return {
            suggestions,
            source: "gemini",
            model,
          };
        }
      }

      lastError = "Gemini response was received but did not contain valid structured JSON suggestions.";
      continue;
    }

    const detail = await geminiResponse.text();
    lastError = `Gemini request failed (${geminiResponse.status}) on model '${model}'. ${detail.slice(0, 280)}`;

    const lowerDetail = detail.toLowerCase();
    const shouldTryNextModel =
      geminiResponse.status === 404 ||
      geminiResponse.status === 429 ||
      geminiResponse.status === 503 ||
      lowerDetail.includes("not found") ||
      lowerDetail.includes("not supported") ||
      lowerDetail.includes("quota") ||
      lowerDetail.includes("rate limit") ||
      lowerDetail.includes("resource exhausted");

    if (!shouldTryNextModel) {
      break;
    }
  }

  return {
    suggestions: fallbackSuggestions,
    source: "fallback",
    error: lastError,
  };
}

export async function POST(request: NextRequest) {
  try {
    const rawPayload = (await request.json()) as unknown;
    const payload =
      rawPayload && typeof rawPayload === "object" && !Array.isArray(rawPayload)
        ? (rawPayload as Record<string, unknown>)
        : {};

    const churnData = await callPythonBridge<ChurnBridgeResponse>("predict_churn", payload);
    const suggestionResult = await generateGeminiSuggestions(payload, churnData);

    return NextResponse.json({
      ...churnData,
      suggestions: suggestionResult.suggestions,
      suggestionSource: suggestionResult.source,
      suggestionModel: suggestionResult.model ?? null,
      suggestionError: suggestionResult.error ?? null,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "churn prediction failed" },
      { status: 500 }
    );
  }
}
