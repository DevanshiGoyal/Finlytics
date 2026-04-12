type JsonBody = Record<string, unknown>;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {})
    }
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

export type FeatureImportance = Array<{ feature: string; importance: number }>;

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
}

export interface DefaultPredictionResponse {
  probability: number;
  label: string;
  explainability: FeatureImportance;
}

export interface ChurnPredictionResponse {
  probability: number;
  label: string;
  suggestions: string[];
  explainability: FeatureImportance;
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
}

export interface AnomalyDetectResponse {
  score: number;
  label: string;
}

export interface AnomalyTimeseriesResponse {
  trend: Array<{ time: string; score: number }>;
  batch: Array<{ txId: string; amount: number; score: number; label: string }>;
}

export interface AnomalyBatchScoreResponse {
  batch: Array<{ txId: string; amount: number; score: number; label: string }>;
}

export interface CreditDemandByGradeResponse {
  trend: Array<{ month: string; A: number; B: number; C: number; D: number; E: number }>;
  heatmap: Array<{ grade: string; values: number[] }>;
}

export async function getDepositLeaderboard() {
  return request<DepositLeaderboardResponse>("/deposit/leaderboard", {
    method: "GET"
  });
}

export async function predictDeposit(payload: JsonBody) {
  return request<DepositPredictionResponse>("/deposit/predict", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function predictDefault(payload: JsonBody) {
  return request<DefaultPredictionResponse>("/predict/default", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function predictChurn(payload: JsonBody) {
  return request<ChurnPredictionResponse>("/predict/churn", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function getLoanVolumeForecast() {
  return request<LoanForecastResponse>("/forecast/loan-volume", {
    method: "GET"
  });
}

export async function detectAnomaly(payload: JsonBody) {
  return request<AnomalyDetectResponse>("/anomaly/detect", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function getAnomalyTimeseries() {
  return request<AnomalyTimeseriesResponse>("/anomaly/timeseries", {
    method: "GET"
  });
}

export async function scoreAnomalyBatch(payload: JsonBody) {
  return request<AnomalyBatchScoreResponse>("/anomaly/batch-score", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function getCreditDemandByGrade() {
  return request<CreditDemandByGradeResponse>("/credit-demand/by-grade", {
    method: "GET"
  });
}
