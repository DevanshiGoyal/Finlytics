import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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

const MODEL_FALLBACKS = [
  "gemini-flash-latest",
  "gemini-flash-lite-latest",
  "gemini-2.0-flash",
  "gemini-2.0-flash-lite",
  "gemini-2.0-flash-001",
  "gemini-2.0-flash-lite-001",
  "gemini-2.5-flash"
];

const DEFAULT_REPLY =
  "Recommended next action: run the Stress Testing scenario with +150 bps shock and compare high-risk share before approving new exposure.";

function localFallbackReply(prompt: string): string {
  const lower = prompt.toLowerCase();

  if (lower.includes("cohort") || lower.includes("spike") || lower.includes("risk")) {
    return "The risk spike is typically concentrated in lower-grade cohorts (D and E) and high-utilization borrowers. Recommended action: tighten underwriting thresholds for high DTI and run targeted monitoring on Grade D/E approvals this week.";
  }

  if (lower.includes("default")) {
    return "Default pressure is usually driven by higher interest-rate, high DTI, and prior delinquency segments. Recommended action: apply stricter checks on applicants with DTI above 35% and elevated recent inquiries.";
  }

  if (lower.includes("churn")) {
    return "Churn risk tends to rise for low-engagement borrowers with repayment friction. Recommended action: launch retention outreach for medium-to-high churn probability customers and prioritize personalized refinancing offers.";
  }

  if (lower.includes("anomaly") || lower.includes("fraud") || lower.includes("suspicious")) {
    return "Recent anomaly patterns are often linked to unusually high-value or high-frequency transactions. Recommended action: trigger enhanced verification for flagged transactions and review high-score records first.";
  }

  if (lower.includes("deposit") || lower.includes("campaign")) {
    return "Deposit conversion usually improves when outreach focuses on high-probability segments from the leaderboard's best model. Recommended action: target top-scored customers first and monitor conversion uplift by contact channel.";
  }

  if (lower.includes("forecast") || lower.includes("demand") || lower.includes("grade")) {
    return "Demand planning should follow grade-level trend signals and forecast confidence bands. Recommended action: review Credit Demand by Grade trends and adjust exposure by grade where volatility is highest.";
  }

  return DEFAULT_REPLY;
}

function buildSystemInstruction() {
  return [
    "You are Finlytics Copilot, an AI decision assistant for financial risk analytics.",
    "Respond with concise, practical guidance for credit risk, churn, forecasting, deposits, and anomaly monitoring.",
    "Prefer actionable recommendations in 1-3 short bullets or a short paragraph.",
    "When asked for next steps, include a clear recommended action and expected business impact."
  ].join(" ");
}

function extractText(payload: GeminiResponse): string {
  const parts = payload.candidates?.[0]?.content?.parts ?? [];
  return parts
    .map((part) => part.text ?? "")
    .join("\n")
    .trim();
}

function dedupeModels(models: string[]): string[] {
  const unique = new Set<string>();
  for (const model of models) {
    const name = model.trim();
    if (name) {
      unique.add(name);
    }
  }
  return [...unique];
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as { message?: string };
    const message = String(body?.message ?? "").trim();

    if (!message) {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        {
          error: "Gemini API key is not configured. Set GEMINI_API_KEY in finlytics-frontend/.env.local.",
          reply: localFallbackReply(message),
          source: "fallback"
        },
        { status: 200 }
      );
    }

    const configuredModel = process.env.GEMINI_MODEL || "";
    const modelsToTry = dedupeModels([configuredModel, ...MODEL_FALLBACKS]);

    let lastError = "Gemini model resolution failed.";

    for (const model of modelsToTry) {
      const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
      const geminiResponse = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        cache: "no-store",
        body: JSON.stringify({
          system_instruction: {
            parts: [{ text: buildSystemInstruction() }]
          },
          contents: [
            {
              role: "user",
              parts: [{ text: message }]
            }
          ],
          generationConfig: {
            temperature: 0.3,
            topP: 0.9,
            maxOutputTokens: 640,
            thinkingConfig: {
              thinkingBudget: 0
            }
          }
        })
      });

      if (geminiResponse.ok) {
        const payload = (await geminiResponse.json()) as GeminiResponse;
        const reply = extractText(payload) || DEFAULT_REPLY;
        return NextResponse.json({ reply, model, source: "gemini" });
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

    return NextResponse.json(
      {
        error: lastError,
        reply: localFallbackReply(message),
        source: "fallback"
      },
      { status: 200 }
    );
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Failed to process Ask Finlytics request",
        reply: DEFAULT_REPLY,
        source: "fallback"
      },
      { status: 200 }
    );
  }
}
