import type {
  CreditDemandByGradeResponse,
  CreditDemandForecastRequest,
  CreditDemandForecastResponse,
} from "./types";

export type {
  CreditDemandByGradeResponse,
  CreditDemandForecast,
  CreditDemandForecastRequest,
  CreditDemandForecastResponse,
  CreditDemandScenario,
} from "./types";

type JsonBody = Record<string, unknown>;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });

  if (!response.ok) {
    let message = `API request failed: ${response.status}`;
    const raw = await response.text();
    if (raw) {
      try {
        const parsed = JSON.parse(raw) as { error?: string; message?: string };
        message = parsed.error || parsed.message || message;
      } catch {
        message = `${message} - ${raw}`;
      }
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export type FeatureImportance = Array<{ feature: string; importance: number }>;

export interface ShapWaterfallPoint {
  feature: string;
  value: number;
  shapValue: number;
  start: number;
  end: number;
}

export interface ShapExplanationResponse {
  available: boolean;
  message: string | null;
  baseValue: number | null;
  modelOutput: number | null;
  points: ShapWaterfallPoint[];
}

export interface DepositLeaderboardResponse {
  leaderboard: Array<{
    model: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  }>;
  featureImportance: FeatureImportance;
  bestModel: string;
}

export interface DepositPredictionResponse {
  probability: number;
  label: string;
  prediction: number;
  model: string;
  shapExplanation?: ShapExplanationResponse;
}

export interface DefaultPredictionResponse {
  probability: number;
  label: string;
  explainability: FeatureImportance;
  shapExplanation?: ShapExplanationResponse;
}

export interface ChurnPredictionResponse {
  probability: number;
  label: string;
  suggestions: string[];
  explainability: FeatureImportance;
  shapExplanation?: ShapExplanationResponse;
}

export type LoanForecastScenario = "baseline" | "optimistic" | "conservative" | "stress";

export interface LoanForecastRequest {
  horizonMonths: number;
  scenario: LoanForecastScenario;
  growthAdjustmentPct: number;
  interestRateShockBps: number;
  loanCountChangePct: number;
  avgLoanAmountChangePct: number;
}

export interface LoanForecastSummary {
  horizonMonths: number;
  scenario: string;
  projectedTotal: number;
  averageMonthly: number;
  finalMonth: string;
  finalValue: number;
  finalRange: {
    low: number;
    high: number;
  };
  growthVsLastActualPct: number;
  assumptions: {
    growthAdjustmentPct: number;
    interestRateShockBps: number;
    loanCountChangePct: number;
    avgLoanAmountChangePct: number;
  };
}

export interface LoanForecastModelInfo {
  primary: string;
  blend: string;
  scaler: string;
}

export interface LoanForecastResponse {
  trend: Array<{ month: string; actual: number; forecast: number }>;
  interval: Array<{
    month: string;
    value: number;
    low: number;
    high: number;
    historical: number | null;
  }>;
  summary?: LoanForecastSummary;
  request?: LoanForecastRequest & { userAdjustments?: Partial<LoanForecastRequest> };
  model?: LoanForecastModelInfo;
  warnings?: string[];
}

export interface AnomalyDetectResponse {
  score: number;
  label: string;
  shapExplanation?: ShapExplanationResponse;
}

export interface AnomalyTimeseriesResponse {
  trend: Array<{ time: string; score: number }>;
  batch: Array<{ txId: string; amount: number; score: number; label: string }>;
}

export interface AnomalyBatchScoreResponse {
  batch: Array<{ txId: string; amount: number; score: number; label: string }>;
}

export async function getDepositLeaderboard() {
  return request<DepositLeaderboardResponse>("/deposit/leaderboard", {
    method: "GET",
  });
}

export async function predictDeposit(payload: JsonBody) {
  return request<DepositPredictionResponse>("/deposit/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function predictDefault(payload: JsonBody) {
  return request<DefaultPredictionResponse>("/predict/default", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function predictChurn(payload: JsonBody) {
  return request<ChurnPredictionResponse>("/predict/churn", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getLoanVolumeForecast() {
  return request<LoanForecastResponse>("/forecast/loan-volume", {
    method: "GET",
  });
}

export async function predictLoanVolumeForecast(payload: LoanForecastRequest) {
  return request<LoanForecastResponse>("/forecast/loan-volume", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function detectAnomaly(payload: JsonBody) {
  return request<AnomalyDetectResponse>("/anomaly/detect", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getAnomalyTimeseries() {
  return request<AnomalyTimeseriesResponse>("/anomaly/timeseries", {
    method: "GET",
  });
}

export async function scoreAnomalyBatch(payload: JsonBody) {
  return request<AnomalyBatchScoreResponse>("/anomaly/batch-score", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getCreditDemandByGrade() {
  return request<CreditDemandByGradeResponse>("/credit-demand/by-grade", {
    method: "GET",
  });
}

export async function forecastCreditDemandByGrade(payload: CreditDemandForecastRequest) {
  return request<CreditDemandForecastResponse>("/credit-demand/by-grade", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
